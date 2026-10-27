#!/usr/bin/env python3
"""Top-down per operator from scripts/perf_ops.sh results.

usage: perf_ops_report.py <results dir>... [--out DIR] [--min-samples N]

Reads every run directory (meta.env, stat.csv, fold/) under the given
results and writes to --out (default <first dir>/analysis):

  runs.csv       per run: cycles and instructions per lineitem tuple, top-down
                 shares from the record and from perf stat (exact), and the
                 record / stat ratio of every event (sampling perturbation)
  operators.csv  per run and operator: cycles per tuple in every bucket and
                 the shares; the label from scripts/perf_opmaps.py when the
                 operator matches one, else the raw key
  functions.csv  per run and leaf function, same columns
  report.md      the labelled operators of every query, side by side

Top-down from the SPR general-purpose slot events (Intel perfmon formulas;
bad speculation and machine clears are residuals, as there):
  frontend = fe / slots          fe_latency = spc * fe_lat_cyc / slots
  retiring = ret / slots         heavy = heavy / slots
  backend  = be / slots          memory = mem / slots, core = backend - memory
  badspec  = max(1 - frontend - backend - retiring, 0), br_misp = brmisp / slots
Cycles = slots / spc (slots_per_cycle in meta.env). Front-end bound omits the
INT_MISC.UOP_DROPPING correction (one counter short).
Rows with fewer than --min-samples slots samples are marked lowconf=1.
"""
import argparse
import csv
import glob
import importlib.util
import os
import re
import sys
from collections import OrderedDict

ap = argparse.ArgumentParser()
ap.add_argument("results", nargs="+")
ap.add_argument("--out")
ap.add_argument("--min-samples", type=int, default=300)
args = ap.parse_args()
out = args.out or os.path.join(args.results[0], "analysis")
os.makedirs(out, exist_ok=True)

maps = None
mp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "perf_opmaps.py")
if os.path.exists(mp):
    spec = importlib.util.spec_from_file_location("perf_opmaps", mp)
    maps = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(maps)

BUCKETS = ["retiring", "ret_light", "ret_heavy", "frontend", "fe_latency", "fe_bandwidth", "badspec",
           "bs_branch", "bs_clears", "backend", "be_memory", "be_core"]


def read_meta(path):
    m = {}
    for line in open(path):
        k, _, v = line.rstrip("\n").partition("=")
        m[k] = v
    return m


def metrics(ev, spc, norm, slot_period):
    s = ev.get("slots", 0)
    if not s:
        return None
    sh = lambda k: ev.get(k, 0) / s
    fe, ret, be, mem = sh("fe"), sh("ret"), sh("be"), sh("mem")
    bs = max(1 - fe - be - ret, 0.0)
    fe_lat = min(spc * ev.get("fe_lat_cyc", 0) / s, fe)
    misp = min(sh("brmisp"), bs) if bs else 0.0
    shares = dict(retiring=ret, ret_heavy=sh("heavy"), ret_light=max(ret - sh("heavy"), 0), frontend=fe,
                  fe_latency=fe_lat, fe_bandwidth=max(fe - fe_lat, 0), badspec=bs, bs_branch=misp,
                  bs_clears=max(bs - misp, 0), backend=be, be_memory=mem, be_core=max(be - mem, 0))
    cyc = s / spc / norm
    row = OrderedDict(cycles_per_tuple=cyc, instr_per_tuple=ev.get("instructions", 0) / norm,
                      ipc=ev.get("instructions", 0) / (s / spc), slots_samples=int(s / slot_period))
    for b in BUCKETS:
        row["cyc_" + b] = shares[b] * cyc
    for b in BUCKETS:
        row["share_" + b] = shares[b]
    return row


def read_table(path):
    with open(path) as f:
        r = csv.reader(f)
        head = next(r)
        n_ev = sum(1 for h in head if h in ("slots", "fe", "fe_lat_cyc", "ret", "heavy", "be", "mem", "brmisp",
                                              "instructions"))
        keys, evs = head[:-n_ev], head[-n_ev:]
        for row in r:
            yield dict(zip(keys, row[:-n_ev])), {e: float(v) for e, v in zip(evs, row[-n_ev:])}


def label_v(q, key):
    for pat, name in (getattr(maps, "TECTORWISE", {}).get(q, []) if maps else []):
        if re.fullmatch(pat, key):
            return name
    return ""


def label_h(q, k):
    pipe = op = ""
    if maps:
        m = getattr(maps, "TYPER", {}).get(q, {})
        pipe = next((n for pat, n in m.get("pipelines", []) if re.search(pat, k["pipeline_src"])), "")
        txt = f'{k["src"]} [{k["inlined"]}]'
        op = next((n for pat, n in m.get("operators", []) if re.search(pat, txt)), "")
    return pipe, op


runs, ops, funcs = [], [], []
for res in args.results:
    for d in sorted(glob.glob(os.path.join(res, "*_t*"))):
        if not os.path.exists(os.path.join(d, "fold", "totals.csv")):
            continue
        meta = read_meta(os.path.join(d, "meta.env"))
        q, e = int(meta["query"]), meta["engine"]
        spc, scale = float(meta.get("slots_per_cycle", 6)), int(meta.get("period_scale", 1))
        slot_period = 10000019 * scale
        norm = int(meta["tuples"]) * int(meta["reps"])
        name = ("Typer" if e == "h" else "Tectorwise") if meta["build"] == "baseline" else \
            ("Typer@HEAD" if e == "h" else "VW:" + meta["build"])
        base = OrderedDict(machine=meta.get("machine", ""), threads=meta["threads"], query=q, run=name,
                           build=meta["build"], engine=e)
        tot = next(read_table(os.path.join(d, "fold", "totals.csv")))[1]
        row = OrderedDict(base)
        row.update(metrics(tot, spc, norm, slot_period))
        # perf stat: exact totals over stat_reps
        stat = {}
        sp = os.path.join(d, "stat.csv")
        if os.path.exists(sp):
            for line in open(sp):
                f = line.strip().split(",")
                if len(f) > 4 and f[0] and not f[0].startswith("#") and f[0].replace(".", "").isdigit():
                    stat[f[2].split("/")[0]] = float(f[0])
        if stat:
            sm = metrics(stat, spc, int(meta["tuples"]) * int(meta["stat_reps"]), 1)
            row["stat_cycles_per_tuple"] = sm["cycles_per_tuple"]
            row["stat_instr_per_tuple"] = sm["instr_per_tuple"]
            for b in BUCKETS:
                row["stat_share_" + b] = sm["share_" + b]
            for ev in ("slots", "fe", "ret", "be", "mem", "brmisp", "instructions"):
                if stat.get(ev):
                    row[f"rec_over_stat_{ev}"] = (tot.get(ev, 0) / int(meta["reps"])) / (stat[ev] / int(meta["stat_reps"]))
        row["timing_median_ms"] = meta.get("timing_median_ms", "")
        runs.append(row)
        for key, ev in read_table(os.path.join(d, "fold", "operators.csv")):
            m = metrics(ev, spc, norm, slot_period)
            if not m:
                continue
            r = OrderedDict(base)
            if e == "v":
                r.update(pipeline="", label=label_v(q, key["operator"]), key=key["operator"])
            else:
                pipe, op = label_h(q, key)
                k = key["other"] or f'{key["pipeline_line"]}:{key["pipeline_src"]} | {key["line"]}:{key["src"]} [{key["inlined"]}]'
                r.update(pipeline=pipe, label=op, key=k)
            r.update(m)
            r["share_of_cycles"] = m["cycles_per_tuple"] / row["cycles_per_tuple"]
            r["lowconf"] = int(m["slots_samples"] < args.min_samples)
            ops.append(r)
        for key, ev in read_table(os.path.join(d, "fold", "functions.csv")):
            m = metrics(ev, spc, norm, slot_period)
            if m and m["cycles_per_tuple"] >= 0.005 * row["cycles_per_tuple"]:
                r = OrderedDict(base)
                r.update(function=key["function"])
                r.update(m)
                r["lowconf"] = int(m["slots_samples"] < args.min_samples)
                funcs.append(r)

if not runs:
    sys.exit("perf_ops_report: no runs with fold/totals.csv")


def write(name, rows):
    cols = list(OrderedDict((k, 1) for r in rows for k in r))
    with open(os.path.join(out, name), "w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.5g}" if isinstance(v, float) else v) for k, v in r.items()})


write("runs.csv", runs)
write("operators.csv", ops)
write("functions.csv", funcs)

# report.md: per query and run, the operators by label (unlabelled rows by key)
with open(os.path.join(out, "report.md"), "w") as f:
    f.write("# Top-down per operator (perf)\n\ncycles per lineitem tuple; shares of the operator's slots\n")
    for q in sorted({r["query"] for r in runs}):
        f.write(f"\n## Q{q}\n")
        for run in [r for r in runs if r["query"] == q]:
            f.write(f"\n### {run['run']} ({run['machine']}, t{run['threads']}): {run['cycles_per_tuple']:.2f} cycles, "
                    f"{run['instr_per_tuple']:.2f} instructions per tuple"
                    + (f"; perf stat {run['stat_cycles_per_tuple']:.2f} cycles" if "stat_cycles_per_tuple" in run else "")
                    + "\n\n| operator | cycles | % | ret | fe | bs | be:mem | be:core | samples |\n|---|---|---|---|---|---|---|---|---|\n")
            groups = OrderedDict()
            for r in ops:
                if r["query"] == q and r["run"] == run["run"]:
                    k = (r["pipeline"] + ": " if r["pipeline"] else "") + (r["label"] or r["key"])
                    g = groups.setdefault(k, {"c": 0.0, "s": 0, **{b: 0.0 for b in BUCKETS}})
                    g["c"] += r["cycles_per_tuple"]
                    g["s"] += r["slots_samples"]
                    for b in BUCKETS:
                        g[b] += r["cyc_" + b]
            for k, g in sorted(groups.items(), key=lambda kv: -kv[1]["c"]):
                if g["c"] < 0.01 * run["cycles_per_tuple"]:
                    continue
                sh = lambda b: g[b] / g["c"] if g["c"] else 0
                f.write(f"| {k} | {g['c']:.2f} | {g['c'] / run['cycles_per_tuple']:.0%} | {sh('retiring'):.2f} | "
                        f"{sh('frontend'):.2f} | {sh('badspec'):.2f} | {sh('be_memory'):.2f} | {sh('be_core'):.2f} | "
                        f"{g['s']}{' (low)' if g['s'] < args.min_samples else ''} |\n")
print(f"perf_ops_report: {len(runs)} runs, {len(ops)} operator rows -> {out}")
