#!/usr/bin/env python3
"""Where does Q1's time go? Cycles per tuple, split top-down, from vtune_q1.sh.

    vtune_breakdown.py RESULTS... [--out DIR] [--top 25]

RESULTS are vtune_q1.sh output directories (any directory with run
subdirectories that hold a meta.env); several machines can be given at once.

Every number is per lineitem tuple, so runs of different length, thread count
or engine are comparable (as in Kersten et al. and Sirin & Ailamaki):

    cycles/tuple = Clockticks / (tuples * reps)       all threads' cycles
    bucket       = cycles/tuple * share of pipeline slots in that bucket

using VTune's top-down (TMA) tree: level 1 (Retiring, Front-End Bound, Bad
Speculation, Back-End Bound) and level 2 (Light / Heavy Operations, Front-End
Latency / Bandwidth, Branch Mispredict / Machine Clears, Memory / Core Bound).
The level-1 shares are renormalised to sum to 1, so the buckets add up to
cycles/tuple exactly and a change between two runs splits into per-bucket
changes that add up to the change in cycles/tuple. Memory Bound's children
(L1/L2/L3/DRAM/Store Bound) are shares of clockticks, not slots, and are
reported as such; they do not add up to Memory Bound.

Runs (display names):
    Typer           baseline build, -e h          Tectorwise   baseline, -e v
    Typer@HEAD      cmake_defaults build, -e h    VW:<config>  HEAD <config>, -e v

Writes to DIR (default ./vtune_breakdown):
    breakdown.md     the report: per machine and thread count, where the time
                     goes, what each ladder step changed, the gap to Typer,
                     per-operator time, DRAM bandwidth; then machine vs machine
    runs.csv         one row per run: time, cycles/tuple, IPC, GHz, buckets
    steps.csv        per-bucket change vs the previous step and vs Tectorwise
    operators.csv    per run and operator class: cycles/tuple and its split
    functions.csv    the top functions per run
    tma_tree.csv     every summary metric VTune reported, long form
    bandwidth.csv    memory-access bandwidth table rows

Function -> operator classes are the OPERATOR_RULES below (first match wins);
edit them if a primitive lands in "other".
"""
import argparse
import csv
import glob
import math
import os
import re
import sys
from collections import OrderedDict, defaultdict

# ------------------------------------------------------------------ taxonomy
L1 = OrderedDict([
    ("retiring", r"^retiring$"),
    ("frontend", r"^front.?end bound$"),
    ("badspec", r"^bad speculation$"),
    ("backend", r"^back.?end bound$"),
])
# level 2: (bucket, parent, pattern on the node name)
L2 = [
    ("ret_light", "retiring", r"^(light operations|general retirement|base)$"),
    ("ret_heavy", "retiring", r"^(heavy operations|microcode sequencer)$"),
    ("fe_latency", "frontend", r"^front.?end latency$"),
    ("fe_bandwidth", "frontend", r"^front.?end bandwidth$"),
    ("bs_branch", "badspec", r"^branch mispredict"),
    ("bs_clears", "badspec", r"^machine clears$"),
    ("be_memory", "backend", r"^memory bound$"),
    ("be_core", "backend", r"^core bound$"),
]
MEM_CHILDREN = OrderedDict([
    ("l1_bound", r"^l1 bound$"), ("l2_bound", r"^l2 bound$"), ("l3_bound", r"^l3 bound$"),
    ("dram_bound", r"^dram bound$"), ("store_bound", r"^store bound$"),
])
LABELS = {
    "retiring": "Retiring", "frontend": "Front-End", "badspec": "Bad Spec", "backend": "Back-End",
    "ret_light": "Ret: light", "ret_heavy": "Ret: heavy", "fe_latency": "FE: latency",
    "fe_bandwidth": "FE: bandwidth", "bs_branch": "BS: branch misp.", "bs_clears": "BS: clears",
    "be_memory": "BE: memory", "be_core": "BE: core",
    "l1_bound": "L1", "l2_bound": "L2", "l3_bound": "L3", "dram_bound": "DRAM", "store_bound": "Store",
}
L2_KEYS = [k for k, _, _ in L2]

# Function -> operator class, first match wins, on the lower-cased function
# name (and module for the last rules). Tectorwise / VW primitives are real
# functions called through pointers, so they show up one by one; Typer's
# pipeline is one fused loop and lands in "typer pipeline".
OPERATOR_RULES = [
    ("typer pipeline", r"q1_hyper|^hyper::|tpch::q1|morsel"),
    ("selection", r"(^|::)sel(sel)?_|select::|simd_sel|selection"),
    ("hashing", r"(^|::)(re)?hash_|hashgroup::hash|crc32|murmur|::hash_t\b|hash<"),
    ("group lookup", r"keys_equal|hashgroup::(lookup|find|keystep|concat|globalfind|run|resolve)|findorcreate|packedkey|concat_"),
    ("aggregation", r"(^|::)aggr_|updategroups|fixedaggr|aggregat"),
    ("projection", r"(^|::)proj_|project::|compound"),
    ("partition/spill", r"partition|spill|scatter|gather|deque"),
    ("scan", r"scan::"),
    ("memcpy/memset", r"memcpy|memmove|memset"),
    ("threading runtime", r"tbb|barrier|workergroup|pthread|sched_yield|futex|spin"),
    ("kernel", r"\[vmlinux\]|vmlinux|\[kernel|^\[unknown"),
]
OTHER = "other"


def classify(func, module=""):
    s = (func or "").lower()
    for name, pat in OPERATOR_RULES:
        if re.search(pat, s):
            return name
    if re.search(r"vmlinux|kernel", (module or "").lower()):
        return "kernel"
    return OTHER


# ------------------------------------------------------------------ parsing
NUM = re.compile(r"^\s*(-?[0-9][0-9,]*\.?[0-9]*(?:e[+-]?\d+)?)\s*(.*)$", re.I)


def number(s):
    """'80,500,000,000' -> 8.05e10; '45.2% of Pipeline Slots' -> (45.2, '%');
    returns (value, unit) or (None, raw)"""
    if s is None:
        return None, ""
    s = s.strip()
    m = NUM.match(s)
    if not m:
        return None, s
    try:
        v = float(m.group(1).replace(",", ""))
    except ValueError:
        return None, s
    unit = m.group(2).strip()
    return v, unit


def norm(name):
    return re.sub(r"\s+", " ", name.strip().lower().replace("(%)", "").replace("%", "")).strip()


def parse_summary_text(path):
    """VTune's text summary -> list of (path tuple, value, unit). Indentation
    (4 spaces per level) gives the tree; '|' lines are advice and skipped.
    The bandwidth table, if any, is returned separately."""
    rows, stack, bw = [], [], []
    try:
        lines = open(path, errors="replace").read().splitlines()
    except OSError:
        return rows, bw
    i = 0
    while i < len(lines):
        line = lines[i]
        if re.match(r"^\s*Bandwidth Domain\s{2,}", line):
            header = re.split(r"\s{2,}", line.strip())
            i += 1
            if i < len(lines) and set(lines[i].strip()) <= set("- "):
                i += 1
            while i < len(lines) and lines[i].strip():
                cells = re.split(r"\s{2,}", lines[i].strip())
                if len(cells) >= 2:
                    bw.append(dict(zip(header, cells)))
                i += 1
            continue
        i += 1
        if not line.strip() or line.strip().startswith("|") or ":" not in line:
            continue
        indent = len(line) - len(line.lstrip(" "))
        name, _, val = line.strip().partition(":")
        if not name or len(name) > 80:
            continue
        depth = indent // 4
        stack = stack[:depth] + [name.strip()]
        v, unit = number(val)
        if v is None and val.strip():
            continue  # text-valued lines (Application Command Line, ...)
        rows.append((tuple(stack), v, unit))
    return rows, bw


def parse_summary_csv(path):
    """Fallback when there is no text summary: rows with a metric name and a
    value; a 'Hierarchy Level' column (if present) gives the depth."""
    rows = []
    try:
        data = list(csv.reader(open(path, errors="replace")))
    except OSError:
        return rows
    if not data:
        return rows
    hdr = [h.strip().lower() for h in data[0]]
    ni = next((i for i, h in enumerate(hdr) if "metric name" in h), None)
    vi = next((i for i, h in enumerate(hdr) if "metric value" in h), None)
    li = next((i for i, h in enumerate(hdr) if "hierarchy" in h), None)
    body = data[1:] if ni is not None else data
    if ni is None:
        ni, vi = 0, 1
    stack = []
    for r in body:
        if len(r) <= max(ni, vi or 0):
            continue
        name = r[ni].strip()
        if not name:
            continue
        depth = 0
        if li is not None:
            try:
                depth = max(0, int(float(r[li])) - 1)
            except ValueError:
                depth = 0
        stack = stack[:depth] + [name]
        v, unit = number(r[vi])
        if v is not None:
            rows.append((tuple(stack), v, unit))
    return rows


def tree_lookup(rows):
    """index by normalized leaf name -> list of (path, value, unit)"""
    idx = defaultdict(list)
    for path, v, u in rows:
        idx[norm(path[-1])].append((path, v, u))
    return idx


def first_metric(idx, *patterns):
    for pat in patterns:
        for k, lst in idx.items():
            if re.search(pat, k):
                return lst[0][1]
    return None


def tma(rows):
    """-> (l1 shares, l2 shares, memory children shares) as fractions"""
    l1, l2, mem = {}, {}, {}
    l1_path = {}
    for path, v, u in rows:
        leaf = norm(path[-1])
        for key, pat in L1.items():
            if key not in l1 and re.search(pat, leaf):
                l1[key] = v / 100.0
                l1_path[key] = path
    for path, v, u in rows:
        leaf = norm(path[-1])
        parents = [norm(p) for p in path[:-1]]
        for key, parent, pat in L2:
            if key in l2 or not re.search(pat, leaf):
                continue
            ppat = L1[parent]
            # under its parent when the tree says so; flat CSVs have no parents
            if not parents or any(re.search(ppat, p) for p in parents):
                l2[key] = v / 100.0
        if any(re.search(r"^memory bound$", p) for p in parents):
            for key, pat in MEM_CHILDREN.items():
                if key not in mem and re.search(pat, leaf):
                    mem[key] = v / 100.0
    return l1, l2, mem


def read_meta(path):
    meta = {}
    for line in open(path, errors="replace"):
        k, _, v = line.rstrip("\n").partition("=")
        if k:
            meta[k] = v
    return meta


def read_csv_dicts(path):
    try:
        with open(path, errors="replace") as f:
            first = f.readline()
            if not first:
                return []
            f.seek(0)
            dialect = "excel-tab" if first.count("\t") > first.count(",") else "excel"
            return list(csv.DictReader(f, dialect=dialect))
    except OSError:
        return []


def col(d, *patterns):
    """value of the first column whose normalized name matches a pattern"""
    for pat in patterns:
        for k, v in d.items():
            if k is not None and re.search(pat, norm(k)):
                return v
    return None


def runtpch_counters(path):
    """the q1 row of run_tpch's CSV as {column: value}; its counter columns
    are already per tuple (PerfEvents::printAll divides by tuples * reps)"""
    hdr, out = None, {}
    try:
        for row in csv.reader(open(path, errors="replace")):
            cells = [c.strip() for c in row]
            if cells and cells[0] == "name":
                hdr = cells
            elif hdr and cells and cells[0].startswith("q1 "):
                for k, v in zip(hdr, cells):
                    x, _ = number(v)
                    if k and x is not None and not math.isnan(x):
                        out[k] = x
                break
    except OSError:
        pass
    return out


def functions(run_dir):
    """per function: cycles, instructions (hw-events), TMA L1/L2 fractions
    (hotspots view of the uarch result), module"""
    out = OrderedDict()
    for d in read_csv_dicts(os.path.join(run_dir, "ua.hwevents.csv")):
        f = col(d, r"^function$", r"^source function$")
        if not f:
            continue
        key = (f, col(d, r"^module$") or "")
        cyc, _ = number(col(d, r"cpu_clk_unhalted\.thread$", r"cpu_clk_unhalted\.thread_p$", r"cpu_clk_unhalted\.core$") or "")
        ins, _ = number(col(d, r"inst_retired\.any$", r"inst_retired\.any_p$") or "")
        out[key] = dict(func=f, module=key[1], cycles=cyc, instr=ins, l1={}, l2={})
    for d in read_csv_dicts(os.path.join(run_dir, "ua.functions.csv")):
        f = col(d, r"^function$", r"^source function$")
        if not f:
            continue
        key = (f, col(d, r"^module$") or "")
        e = out.setdefault(key, dict(func=f, module=key[1], cycles=None, instr=None, l1={}, l2={}))
        if e["cycles"] is None:
            e["cycles"], _ = number(col(d, r"^clockticks$") or "")
        if e["instr"] is None:
            e["instr"], _ = number(col(d, r"^instructions retired$") or "")
        for k, raw in d.items():
            if k is None:
                continue
            parts = [norm(p) for p in k.split(":")]
            v, _ = number(raw or "")
            if v is None:
                continue
            for key1, pat in L1.items():
                if re.search(pat, parts[-1]) and len(parts) == 1:
                    e["l1"][key1] = v / 100.0
            for key2, parent, pat in L2:
                if re.search(pat, parts[-1]) and (len(parts) == 1 or re.search(L1[parent], parts[-2])):
                    e["l2"][key2] = v / 100.0
    return list(out.values())


# ------------------------------------------------------------------ model
def display(meta):
    b, e = meta.get("build", "?"), meta.get("engine", "?")
    if e == "h":
        return "Typer" if b == "baseline" else "Typer@HEAD" if b == "cmake_defaults" else "Typer@" + b
    return "Tectorwise" if b == "baseline" else "VW:" + b


def renorm(shares):
    s = sum(v for v in shares.values() if v is not None)
    return {k: (v / s if s > 0 else None) for k, v in shares.items()}


def load_runs(results):
    runs = []
    for top in results:
        for meta_path in sorted(glob.glob(os.path.join(top, "**", "meta.env"), recursive=True)):
            d = os.path.dirname(meta_path)
            meta = read_meta(meta_path)
            r = dict(dir=d, meta=meta, name=display(meta), machine=meta.get("machine") or meta.get("host", "?"),
                     host=meta.get("host", "?"), cpu=meta.get("cpu_model", ""), engine=meta.get("engine", "?"),
                     build=meta.get("build", "?"), threads=int(meta.get("threads", "0") or 0),
                     idx=int(meta.get("ladder_index", "0") or 0), notes=[])
            tuples, reps = float(meta.get("tuples") or 0), float(meta.get("reps") or 0)
            r["ms"] = float(meta["timing_median_ms"]) if meta.get("timing_median_ms") else None
            r["ladder"] = meta.get("configs", "")
            # run_tpch's own counters from the untraced run (per tuple already)
            r["rt"] = runtpch_counters(os.path.join(d, "timing.csv"))
            text = os.path.join(d, "ua.summary.txt")
            rows, _ = parse_summary_text(text)
            if not rows:
                rows = parse_summary_csv(os.path.join(d, "ua.summary.csv"))
            r["tree"] = rows
            idx = tree_lookup(rows)
            clk = first_metric(idx, r"^clockticks$")
            ins = first_metric(idx, r"^instructions retired$")
            r["mux"] = first_metric(idx, r"^mux reliability$")
            r["ghz"] = first_metric(idx, r"^average cpu frequency$")
            r["elapsed"] = first_metric(idx, r"^elapsed time$")
            r["paused"] = first_metric(idx, r"^paused time$")
            n = tuples * reps
            r["cpt"] = clk / n if clk and n else None
            r["ipt"] = ins / n if ins and n else None
            r["ipc"] = ins / clk if ins and clk else None
            l1, l2, mem = tma(rows)
            r["l1_raw"] = l1
            if len(l1) == 4:
                total = sum(l1.values())
                if abs(total - 1) > 0.05:
                    r["notes"].append("level-1 shares sum to {:.0f}%".format(100 * total))
                r["l1"] = renorm(l1)
            else:
                r["l1"] = {}
                if rows:
                    r["notes"].append("no complete level-1 TMA in the summary")
            # level 2 in cycles: share of all slots, scaled like level 1
            scale = 1.0 / sum(l1.values()) if len(l1) == 4 and sum(l1.values()) > 0 else 1.0
            r["l2"] = {k: v * scale for k, v in l2.items()}
            r["mem"] = mem
            r["cyc"] = {k: r["cpt"] * v for k, v in r["l1"].items()} if r["cpt"] else {}
            r["cyc2"] = {k: r["cpt"] * v for k, v in r["l2"].items()} if r["cpt"] else {}
            for key, ours in (("cycles", r["cpt"]), ("instr.", r["ipt"])):
                theirs = r["rt"].get(key)
                if ours and theirs and abs(ours / theirs - 1) > 0.05:
                    r["notes"].append("VTune {} / tuple {:.1f} vs run_tpch's {:.1f} ({:+.0f}%){}".format(
                        key.rstrip("."), ours, theirs, 100 * (ours / theirs - 1),
                        "; run_tpch's counters can miss threads that existed before they were opened"
                        if r["threads"] > 1 else "; check the measured window and multiplexing"))
            if r["mux"] is not None and r["mux"] < 0.9:
                r["notes"].append("MUX reliability {:.2f}: multiplexed events disagree, raise MIN_SECONDS".format(r["mux"]))
            if not rows:
                r["notes"].append("no uarch-exploration summary")
            # bandwidth (memory-access run)
            _, bw = parse_summary_text(os.path.join(d, "mem.summary.txt"))
            r["bw"] = bw
            # functions -> operators
            fns = functions(d)
            r["functions"] = fns
            ops = defaultdict(lambda: dict(cycles=0.0, l1=defaultdict(float), l2=defaultdict(float), funcs=0))
            for f in fns:
                c = f["cycles"] or 0.0
                o = ops[classify(f["func"], f["module"])]
                o["cycles"] += c
                o["funcs"] += 1
                for k, v in f["l1"].items():
                    o["l1"][k] += c * v
                for k, v in f["l2"].items():
                    o["l2"][k] += c * v
            r["ops"] = ops
            r["fn_cycles"] = sum(o["cycles"] for o in ops.values())
            runs.append(r)
    return runs


def fmt(v, nd=2):
    return "" if v is None else "{:.{}f}".format(v, nd)


def signed(v, nd=2):
    return "" if v is None else "{:+.{}f}".format(v, nd)


# ------------------------------------------------------------------ report
def group_key(r):
    # one ladder per CONFIGS list, so a second ladder (e.g. the older tuned
    # path) run into another result directory keeps its own "previous step"
    return (r["machine"], r["threads"], r["ladder"])


def ordered(rs):
    """Typer, Tectorwise, then the ladder (HEAD Typer after its build)"""
    def k(r):
        if r["build"] == "baseline":
            return (0, 0 if r["engine"] == "h" else 1)
        return (r["idx"], 1 if r["engine"] == "h" else 0)
    return sorted(rs, key=k)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+")
    ap.add_argument("--out", default="vtune_breakdown")
    ap.add_argument("--top", type=int, default=25, help="functions per run in functions.csv (default 25)")
    args = ap.parse_args()
    runs = load_runs(args.results)
    if not runs:
        sys.exit("no run directories (meta.env) under " + " ".join(args.results))
    os.makedirs(args.out, exist_ok=True)
    l1k, l2k, memk = list(L1), L2_KEYS, list(MEM_CHILDREN)

    # runs.csv
    with open(os.path.join(args.out, "runs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["machine", "threads", "run", "build", "engine", "median_ms", "cycles_per_tuple", "instr_per_tuple",
                    "ipc", "ghz", "mux_reliability", "runtpch_cycles_per_tuple", "runtpch_instr_per_tuple"] + ["cyc_" + k for k in l1k] + ["cyc_" + k for k in l2k] +
                   ["pct_clk_" + k for k in memk] + ["notes", "dir"])
        for r in sorted(runs, key=lambda r: (r["machine"], r["threads"], r["idx"], r["engine"])):
            w.writerow([r["machine"], r["threads"], r["name"], r["build"], r["engine"], fmt(r["ms"], 3), fmt(r["cpt"], 3),
                        fmt(r["ipt"], 3), fmt(r["ipc"], 3), fmt(r["ghz"], 3), fmt(r["mux"], 3),
                        fmt(r["rt"].get("cycles"), 3), fmt(r["rt"].get("instr."), 3)] +
                       [fmt(r["cyc"].get(k), 3) for k in l1k] + [fmt(r["cyc2"].get(k), 3) for k in l2k] +
                       [fmt(100 * r["mem"][k], 1) if k in r["mem"] else "" for k in memk] +
                       ["; ".join(r["notes"]), r["dir"]])

    # tma_tree.csv
    with open(os.path.join(args.out, "tma_tree.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["machine", "threads", "run", "metric_path", "value", "unit"])
        for r in runs:
            for path, v, u in r["tree"]:
                w.writerow([r["machine"], r["threads"], r["name"], " / ".join(path), v, u])

    # bandwidth.csv
    with open(os.path.join(args.out, "bandwidth.csv"), "w", newline="") as f:
        w = csv.writer(f)
        cols = sorted({k for r in runs for row in r["bw"] for k in row})
        w.writerow(["machine", "threads", "run"] + cols)
        for r in runs:
            for row in r["bw"]:
                w.writerow([r["machine"], r["threads"], r["name"]] + [row.get(c, "") for c in cols])

    # operators.csv / functions.csv
    opnames = sorted({o for r in runs for o in r["ops"]})
    with open(os.path.join(args.out, "operators.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["machine", "threads", "run", "operator", "functions", "share_of_cycles", "cycles_per_tuple"] +
                   ["share_" + k for k in l1k] + ["share_" + k for k in l2k])
        for r in runs:
            n = (r["cpt"] or 0) / r["fn_cycles"] if r["fn_cycles"] else None  # cycles/tuple per sampled cycle
            for op, o in sorted(r["ops"].items(), key=lambda kv: -kv[1]["cycles"]):
                c = o["cycles"]
                w.writerow([r["machine"], r["threads"], r["name"], op, o["funcs"],
                            fmt(c / r["fn_cycles"], 4) if r["fn_cycles"] else "", fmt(c * n, 3) if n else ""] +
                           [fmt(o["l1"][k] / c, 3) if c and k in o["l1"] else "" for k in l1k] +
                           [fmt(o["l2"][k] / c, 3) if c and k in o["l2"] else "" for k in l2k])
    with open(os.path.join(args.out, "functions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["machine", "threads", "run", "rank", "operator", "share_of_cycles", "cycles_per_tuple", "ipc",
                    "function", "module"] + ["share_" + k for k in l1k])
        for r in runs:
            n = (r["cpt"] or 0) / r["fn_cycles"] if r["fn_cycles"] else None
            fs = sorted(r["functions"], key=lambda x: -(x["cycles"] or 0))[:args.top]
            for i, fn in enumerate(fs, 1):
                c = fn["cycles"] or 0
                w.writerow([r["machine"], r["threads"], r["name"], i, classify(fn["func"], fn["module"]),
                            fmt(c / r["fn_cycles"], 4) if r["fn_cycles"] else "", fmt(c * n, 3) if n else "",
                            fmt(fn["instr"] / c, 2) if fn["instr"] and c else "", fn["func"], fn["module"]] +
                           [fmt(fn["l1"].get(k), 3) for k in l1k])

    # steps.csv + breakdown.md
    groups = defaultdict(list)
    for r in runs:
        groups[group_key(r)].append(r)
    steps = []
    md = ["# Q1 micro-architectural breakdown", "",
          "Cycles per lineitem tuple (all threads), split by VTune's top-down tree. Level-1 shares are renormalised "
          "to 100% so the columns add up to the total; level-2 columns split their level-1 parent. ms is the "
          "untraced median per query. See the header of scripts/vtune_breakdown.py for the method.", ""]
    many_ladders = len({r["ladder"] for r in runs}) > 1
    for (mach, thr, ladder), rs in sorted(groups.items()):
        rs = ordered(rs)
        cpu = next((r["cpu"] for r in rs if r["cpu"]), "")
        md += ["## {} ({}), {} thread{}{}".format(mach, cpu, thr, "" if thr == 1 else "s",
                                               ", ladder: " + ladder if many_ladders else ""), ""]
        # where the time goes
        md += ["### Where the time goes (cycles / tuple)", "",
               "| run | ms | cyc/tuple | instr/tuple | IPC | GHz | " + " | ".join(LABELS[k] for k in l1k) + " | " +
               " | ".join(LABELS[k] for k in l2k) + " |",
               "|" + "---|" * (6 + len(l1k) + len(l2k))]
        for r in rs:
            md.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                r["name"], fmt(r["ms"], 1), fmt(r["cpt"], 1), fmt(r["ipt"], 1), fmt(r["ipc"], 2), fmt(r["ghz"], 2),
                " | ".join(fmt(r["cyc"].get(k), 1) for k in l1k), " | ".join(fmt(r["cyc2"].get(k), 1) for k in l2k)))
        if any(r["mem"] for r in rs):
            md += ["", "Memory Bound's children, % of clockticks (overlapping; not additive):", "",
                   "| run | " + " | ".join(LABELS[k] for k in memk) + " |", "|" + "---|" * (1 + len(memk))]
            for r in rs:
                md.append("| {} | {} |".format(r["name"], " | ".join(fmt(100 * r["mem"][k], 1) if k in r["mem"] else ""
                                                                   for k in memk)))
        # what each step bought
        ref = next((r for r in rs if r["build"] == "baseline" and r["engine"] == "v"), None)
        typer = next((r for r in rs if r["build"] == "baseline" and r["engine"] == "h"), None)
        ladder = [r for r in rs if r["engine"] == "v"]
        if len(ladder) > 1:
            md += ["", "### What each step changed (Δ cycles / tuple; negative = fewer cycles)", "",
                   "vs prev: against the previous rung of the ladder. vs Tectorwise: cumulative. Δms is the untraced time.",
                   "", "| run | vs | Δms | Δcyc | " + " | ".join(LABELS[k] for k in l1k) + " | " +
                   " | ".join(LABELS[k] for k in l2k) + " |", "|" + "---|" * (4 + len(l1k) + len(l2k))]
            for i, r in enumerate(ladder[1:], 1):
                for tag, base in (("prev: " + ladder[i - 1]["name"], ladder[i - 1]), ("Tectorwise", ref)):
                    if base is None or base is r or (tag == "Tectorwise" and base is ladder[i - 1]):
                        continue
                    row = delta(r, base)
                    steps.append(dict(machine=mach, threads=thr, run=r["name"], vs=base["name"], **row))
                    md.append("| {} | {} | {} | {} | {} | {} |".format(
                        r["name"], tag, signed(row["d_ms"], 1), signed(row["d_cpt"], 1),
                        " | ".join(signed(row["d_" + k], 1) for k in l1k), " | ".join(signed(row["d_" + k], 1) for k in l2k)))
        # gap to Typer
        if typer and ladder:
            md += ["", "### Gap to Typer (cycles / tuple above Typer; negative = fewer than Typer)", "",
                   "| run | Δms | Δcyc | " + " | ".join(LABELS[k] for k in l1k) + " | " +
                   " | ".join(LABELS[k] for k in l2k) + " |", "|" + "---|" * (3 + len(l1k) + len(l2k))]
            for r in ladder:
                row = delta(r, typer)
                steps.append(dict(machine=mach, threads=thr, run=r["name"], vs=typer["name"], **row))
                md.append("| {} | {} | {} | {} | {} |".format(
                    r["name"], signed(row["d_ms"], 1), signed(row["d_cpt"], 1),
                    " | ".join(signed(row["d_" + k], 1) for k in l1k), " | ".join(signed(row["d_" + k], 1) for k in l2k)))
        # operators
        ops = sorted({o for r in rs for o in r["ops"]}, key=lambda o: -max(
            (r["ops"][o]["cycles"] / r["fn_cycles"] if r["fn_cycles"] and o in r["ops"] else 0) for r in rs))
        if ops:
            md += ["", "### Per operator (cycles / tuple; dominant level-1 bucket in brackets)", "",
                   "| run | " + " | ".join(ops) + " |", "|" + "---|" * (1 + len(ops))]
            for r in rs:
                n = (r["cpt"] or 0) / r["fn_cycles"] if r["fn_cycles"] else None
                cells = []
                for o in ops:
                    x = r["ops"].get(o)
                    if not x or not x["cycles"] or not n:
                        cells.append("")
                        continue
                    dom = max(x["l1"], key=lambda k: x["l1"][k]) if x["l1"] else None
                    cells.append("{}{}".format(fmt(x["cycles"] * n, 1), " ({})".format(LABELS[dom]) if dom else ""))
                md.append("| {} | {} |".format(r["name"], " | ".join(cells)))
        # bandwidth
        bwr = [r for r in rs if r["bw"]]
        if bwr:
            md += ["", "### DRAM bandwidth (memory-access)", "",
                   "Average over the collected (resumed) time; utilization = average / platform maximum. "
                   "Check the bandwidth timeline in the GUI if the window looks off.", "",
                   "| run | domain | platform max | observed max | average | utilization |", "|---|---|---|---|---|---|"]
            for r in bwr:
                for row in r["bw"]:
                    dom = col(row, r"bandwidth domain") or ""
                    if "dram" not in dom.lower():
                        continue
                    pm, om, avg = col(row, r"platform maximum"), col(row, r"observed maximum"), col(row, r"^average")
                    pv, av = number(pm or "")[0], number(avg or "")[0]
                    util = "{:.0f}%".format(100 * av / pv) if pv and av is not None else ""
                    md.append("| {} | {} | {} | {} | {} | {} |".format(r["name"], dom, pm or "", om or "", avg or "", util))
        notes = [(r["name"], n) for r in rs for n in r["notes"]]
        if notes:
            md += ["", "Notes:", ""] + ["- {}: {}".format(a, b) for a, b in notes]
        md.append("")

    # machine vs machine: same run and thread setting ("all" = one socket each)
    machines = sorted({r["machine"] for r in runs})
    if len(machines) > 1:
        md += ["## Machine vs machine", "",
               "Same run on each machine; one-socket runs are compared as `socket` (core counts differ). "
               "Cells: cycles/tuple, then the level-1 split in %.", ""]
        by = defaultdict(dict)
        for r in runs:
            setting = "1 thread" if r["threads"] == 1 else "socket"
            by[(setting, r["name"], r["idx"], r["engine"], r["ladder"])][r["machine"]] = r
        md += ["| threads | run | " + " | ".join(machines) + " |", "|" + "---|" * (2 + len(machines))]
        for (setting, name, _, _, _), per in sorted(by.items(), key=lambda kv: (kv[0][4], kv[0][0], kv[0][2], kv[0][3])):
            cells = []
            for m in machines:
                r = per.get(m)
                if not r or r["cpt"] is None:
                    cells.append("")
                    continue
                split = "/".join(fmt(100 * r["l1"].get(k, 0), 0) for k in l1k) if r["l1"] else ""
                cells.append("{} ({})".format(fmt(r["cpt"], 1), split))
            md.append("| {} | {} | {} |".format(setting, name, " | ".join(cells)))
        md += ["", "Split order: " + "/".join(LABELS[k] for k in l1k) + ".", ""]

    with open(os.path.join(args.out, "steps.csv"), "w", newline="") as f:
        w = csv.writer(f)
        keys = ["d_ms", "d_cpt", "d_ipt"] + ["d_" + k for k in l1k] + ["d_" + k for k in l2k]
        w.writerow(["machine", "threads", "run", "vs"] + keys)
        for s in steps:
            w.writerow([s["machine"], s["threads"], s["run"], s["vs"]] + [fmt(s.get(k), 3) for k in keys])
    with open(os.path.join(args.out, "breakdown.md"), "w") as f:
        f.write("\n".join(md))
    print("{} runs ({} machines) -> {}".format(len(runs), len(machines), args.out))


def delta(r, base):
    """per-bucket change from base to r, in cycles/tuple; the level-1 deltas
    add up to d_cpt"""
    row = dict(d_ms=(r["ms"] - base["ms"]) if r["ms"] is not None and base["ms"] is not None else None,
               d_cpt=(r["cpt"] - base["cpt"]) if r["cpt"] is not None and base["cpt"] is not None else None,
               d_ipt=(r["ipt"] - base["ipt"]) if r["ipt"] is not None and base["ipt"] is not None else None)
    for k in list(L1):
        a, b = base["cyc"].get(k), r["cyc"].get(k)
        row["d_" + k] = (b - a) if a is not None and b is not None else None
    for k in L2_KEYS:
        a, b = base["cyc2"].get(k), r["cyc2"].get(k)
        row["d_" + k] = (b - a) if a is not None and b is not None else None
    return row


if __name__ == "__main__":
    main()
