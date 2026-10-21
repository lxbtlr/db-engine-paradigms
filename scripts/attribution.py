#!/usr/bin/env python3
"""Attribute each VectorWise build option's effect to a mechanism bin.

    attribution.py [--out DIR] [--br-penalty 15] [--min-effect 0.02] RESULTS...

RESULTS are flag_ablation.sh result directories (any directory under which
counters.csv files are found, e.g. results/flag_ablation_<host>_<stamp>/).

Pairs: within one machine and commit, and among configs with the same run
settings (engine, threads, ... in the name), config B is paired with each
closest A whose effective defines (<compiler>_<config>/defines.txt) are a
strict subset of B's. The option is the set of added defines, labelled by
the option that implies the others (read from CMakeLists.txt). An _autovec
twin is paired as AUTOVECTORIZE, for VectorWise and Hyper (_eh) alike.

Per pair, compiler and query (medians over rounds of the per-tuple
counters):
    dlog(cycles) = dlog(instructions) + dlog(CPI)
    bin W  if |dlog instr| >= |dlog CPI|; else the CPI sub-bin:
        M  memory stalls explain it (mem_stall per tuple),
        S  mispredicts x penalty explain it,
        L  otherwise (dependency chains, ports)
    +H     if the clock moved by 3% or more (frequency license)
    ~      if neither cycles nor time moved by --min-effect
Flags: noisy (noise.csv), clock, dubliner-only (dubliner's time change has
the opposite sign of all other machines, or is >10 points off their
median), proxy-M (roquefort's mem_stall is a dispatch-stall proxy),
cross-run (A and B from different result directories).

Writes DIR/attribution_pairs.csv (every cell) and DIR/attribution_summary.md
(one table per option; dubliner-only and noisy cells are kept out of the
majority bin).
"""
import argparse
import csv
import math
import os
import re
import statistics
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_SUFFIX = re.compile(r"^(?P<base>.*?)(?P<av>_autovec)?(?P<x>_x[0-9a-f]{8})?(?P<run>(_e[a-z]|_t\d+|_v\d+|_sf\d+|_cpu\d+|_simd\w+?\d|_pg\w+)*)$")
METRICS = ("median", "cycles", "instr", "GHz", "mem_stall", "br-misses", "LLC-misses", "l1-misses", "IPC")
CLOCK = 0.03          # |dlog GHz| for H / clock
DUBLINER_OFF = 0.10   # points off the other machines' median


def implied_defines():
    """VW_X -> defines its CMake block adds besides itself (VW_GROUP_DISPATCH
    turns on VW_AGGR_FUSED, ...)."""
    try:
        text = open(os.path.join(ROOT, "CMakeLists.txt")).read()
    except OSError:
        return {}
    out = {}
    for m in re.finditer(r"^if\((VW_[A-Z0-9_]+)\)\s*\n(.*?)^endif", text, re.M | re.S):
        for a in re.finditer(r"ADD_DEFINITIONS\(([^)]*)\)", m.group(2)):
            names = re.findall(r"-D(VW_[A-Z0-9_]+)", a.group(1))
            extra = {n for n in names if n != m.group(1)}
            if extra:
                out.setdefault(m.group(1), set()).update(extra)
    return out


def read_defines(path):
    try:
        toks = open(path).read().split()
    except OSError:
        return None
    return frozenset(t[2:] for t in toks if t.startswith("-D"))


def machine_and_commit(run_dir):
    mach, commit = None, None
    try:
        for line in open(os.path.join(run_dir, "machine.txt")):
            if line.startswith("MACHINE:"):
                mach = line.split()[1]
            elif line.startswith("host:") and not mach:
                mach = line.split()[1]
            elif line.startswith("git:"):
                commit = line.split()[1]
    except OSError:
        pass
    return mach or os.path.basename(run_dir), commit


def load(results):
    """-> list of builds: dict(machine, commit, dir, compiler, config, base,
    autovec, run, atoms, metrics{query: {metric: median}}, noisy{query})"""
    builds = []
    for top in results:
        for dirpath, _, files in os.walk(top):
            if "counters.csv" not in files:
                continue
            mach, commit = machine_and_commit(dirpath)
            vals = defaultdict(list)
            for r in csv.DictReader(open(os.path.join(dirpath, "counters.csv"))):
                if r["metric"] in METRICS:
                    try:
                        vals[(r["compiler"], r["config"], r["query"], r["metric"])].append(float(r["value"]))
                    except ValueError:
                        pass
            noisy = set()
            try:
                for r in csv.DictReader(open(os.path.join(dirpath, "noise.csv"))):
                    if float(r["round_spread"] or 0) > 0.05 or float(r["cv"] or 0) > 0.05:
                        noisy.add((r["compiler"], r["config"], r["query"]))
            except (OSError, KeyError, ValueError):
                pass
            per = defaultdict(lambda: defaultdict(dict))
            for (cc, cfg, q, met), v in vals.items():
                per[(cc, cfg)][q][met] = statistics.median(v)
            for (cc, cfg), qs in per.items():
                m = BUILD_SUFFIX.match(cfg)
                hyper = "_eh" in (m.group("run") or "")
                atoms = read_defines(os.path.join(dirpath, "{}_{}".format(cc, cfg), "defines.txt"))
                if atoms is None:  # thread lists: counters say jd_all_t1, the dir jd_all_t1-2-4-...
                    stem = "{}_{}{}{}".format(cc, m.group("base"), m.group("av") or "", m.group("x") or "")
                    for d in sorted(os.listdir(dirpath)):
                        if d == stem or d.startswith(stem + "_"):
                            atoms = read_defines(os.path.join(dirpath, d, "defines.txt"))
                            if atoms is not None:
                                break
                if atoms is None and not hyper:
                    continue
                atoms = set() if hyper else set(atoms)  # VW options do not touch Hyper
                if m.group("av"):
                    atoms.add("AUTOVECTORIZE")
                builds.append(dict(machine=mach, commit=commit, dir=dirpath, compiler=cc, config=cfg,
                                   base=m.group("base"), run=m.group("run") or "", hyper=hyper,
                                   atoms=frozenset(atoms), metrics=qs,
                                   noisy={q for (c2, f2, q) in noisy if c2 == cc and f2 == cfg}))
    return builds


def label(added, removed, implies):
    if removed:  # a value change, e.g. VW_JOIN_BLOOM_BITS=16 -> 8
        return " ".join(sorted("{}→{}".format(r, a) for r, a in zip(sorted(removed), sorted(added))))
    names = {a.split("=")[0] for a in added}
    heads = {n for n in names if implies.get(n, set()) & names}
    covered = set().union(*(implies[h] for h in heads)) if heads else set()
    return " + ".join(sorted(names - covered)) or " + ".join(sorted(names))


def pairs(builds, implies):
    groups = defaultdict(list)
    for b in builds:
        groups[(b["machine"], b["commit"], b["compiler"], b["run"], b["hyper"])].append(b)
    out = []
    for g in groups.values():
        for b in g:
            subs = [a for a in g if a is not b and a["atoms"] < b["atoms"]]
            if not subs:
                continue
            best = max(len(a["atoms"]) for a in subs)
            for a in subs:
                if len(a["atoms"]) != best:
                    continue
                added = b["atoms"] - a["atoms"]
                out.append((a, b, label(added, set(), implies), sorted(added)))
        # value changes: one define with another value (jd_all -> jd_semi:
        # VW_JOIN_SEMI_MAX_BYTES=0 -> 8388608); A is the smaller value
        key = lambda atom: atom.split("=")[0]
        for a in g:
            for b in g:
                ra, rb = a["atoms"] - b["atoms"], b["atoms"] - a["atoms"]
                if len(ra) != 1 or len(rb) != 1:
                    continue
                (x,), (y,) = ra, rb
                if key(x) != key(y):
                    continue
                vx, vy = x.partition("=")[2], y.partition("=")[2]
                try:
                    smaller = float(vx) < float(vy)
                except ValueError:
                    smaller = vx < vy
                if smaller:
                    out.append((a, b, "{} {}→{}".format(key(x), vx, vy), [y]))
    return out


def dl(x, y):
    return math.log(y / x) if x and y and x > 0 and y > 0 else None


def cell(a, b, q, args):
    ma, mb = a["metrics"].get(q), b["metrics"].get(q)
    if not ma or not mb:
        return None
    t, cyc, ins = dl(ma.get("median"), mb.get("median")), dl(ma.get("cycles"), mb.get("cycles")), dl(ma.get("instr"), mb.get("instr"))
    ghz = dl(ma.get("GHz"), mb.get("GHz"))
    if t is None:
        return None
    r = dict(t=t, cyc=cyc, ins=ins, cpi=(cyc - ins) if cyc is not None and ins is not None else None, ghz=ghz,
             ms_a=ma.get("median"), ms_b=mb.get("median"))
    for k in ("mem_stall", "br-misses", "LLC-misses"):
        r["d_" + k] = (mb.get(k) - ma.get(k)) if ma.get(k) is not None and mb.get(k) is not None else None
    if cyc is None or ins is None:
        r["bin"] = "?"
    elif abs(cyc) < args.min_effect and abs(t) < args.min_effect:
        r["bin"] = "~"
    elif abs(ins) >= abs(r["cpi"]):
        r["bin"] = "W"
    else:
        dcyc = mb["cycles"] - ma["cycles"]
        dmem = r["d_mem_stall"] or 0.0
        dbr = (r["d_br-misses"] or 0.0) * args.br_penalty
        if abs(dmem) >= abs(dbr) and abs(dmem) >= 0.3 * abs(dcyc) and dmem * dcyc > 0:
            r["bin"] = "M"
        elif abs(dbr) >= 0.3 * abs(dcyc) and dbr * dcyc > 0:
            r["bin"] = "S"
        else:
            r["bin"] = "L"
    if ghz is not None and abs(ghz) >= CLOCK:
        r["bin"] += "+H"
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+")
    ap.add_argument("--out", default=".")
    ap.add_argument("--br-penalty", type=float, default=15.0, help="cycles per mispredict (default 15)")
    ap.add_argument("--min-effect", type=float, default=0.02, help="|dlog| below which a cell is ~ (default 0.02)")
    ap.add_argument("--safe", metavar="BASE", help="also write safe_report.md: every config against BASE (e.g. join_base)")
    ap.add_argument("--safe-limit", type=float, default=0.03, help="largest allowed slowdown for SAFE (default 0.03)")
    ap.add_argument("--safe-noisy", choices=("fail", "ignore"), default="fail",
                    help="noisy slow cells: fail the config (default) or list them without counting")
    args = ap.parse_args()
    implies = implied_defines()
    builds = load(args.results)
    if not builds:
        sys.exit("no counters.csv with defines.txt found under " + " ".join(args.results))
    rows = []
    for a, b, lab, added in pairs(builds, implies):
        for q in sorted(set(a["metrics"]) & set(b["metrics"]), key=lambda s: int(s[1:]) if s[1:].isdigit() else 0):
            c = cell(a, b, q, args)
            if not c:
                continue
            flags = []
            if q in a["noisy"] or q in b["noisy"]:
                flags.append("noisy")
            if c["ghz"] is not None and abs(c["ghz"]) >= CLOCK:
                flags.append("clock")
            if a["machine"] == "roquefort" and c["bin"].startswith("M"):
                flags.append("proxy-M")
            if a["dir"] != b["dir"]:
                flags.append("cross-run")
            rows.append(dict(option=lab, added=" ".join(added), engine="h" if a["hyper"] else "v",
                             machine=a["machine"], commit=a["commit"], compiler=a["compiler"], run=a["run"],
                             query=q, base=a["config"], config=b["config"], flags=flags, **c))
    # dubliner-only: per (option, engine, compiler, run, query), against the other machines
    key = lambda r: (r["option"], r["engine"], r["compiler"], r["run"], r["query"], r["base"].split("_x")[0], r["config"].split("_x")[0])
    by = defaultdict(list)
    for r in rows:
        by[key(r)].append(r)
    for rs in by.values():
        dub = [r for r in rs if r["machine"] == "dubliner"]
        others = [r["t"] for r in rs if r["machine"] != "dubliner"]
        if not dub or len(others) < 2:
            continue
        med = statistics.median(others)
        for r in dub:
            opposite = all(o * r["t"] < 0 for o in others) and abs(r["t"]) > 0.03
            if opposite or abs(r["t"] - med) > DUBLINER_OFF:
                r["flags"].append("dubliner-only")
    os.makedirs(args.out, exist_ok=True)
    # the split is additive in log space: dlog_* columns are 100 x dlog;
    # time_pct and the summary cells are the plain percent change in time
    pct = lambda v: "" if v is None else "{:+.1f}".format(100 * v)
    chg = lambda v: "" if v is None else "{:+.1f}".format(100 * (math.exp(v) - 1))
    with open(os.path.join(args.out, "attribution_pairs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["option", "added_defines", "engine", "machine", "commit", "compiler", "run_suffix", "query",
                    "base_config", "config", "base_ms", "ms", "time_pct", "dlog_time", "dlog_cycles", "dlog_instr",
                    "dlog_cpi", "dlog_ghz", "d_mem_stall", "d_br_misses", "d_llc_misses", "bin", "flags"])
        for r in rows:
            w.writerow([r["option"], r["added"], r["engine"], r["machine"], r["commit"], r["compiler"], r["run"],
                        r["query"], r["base"], r["config"], r["ms_a"], r["ms_b"], chg(r["t"]), pct(r["t"]), pct(r["cyc"]),
                        pct(r["ins"]), pct(r["cpi"]), pct(r["ghz"]),
                        *("" if r[k] is None else "{:.3f}".format(r[k]) for k in ("d_mem_stall", "d_br-misses", "d_LLC-misses")),
                        r["bin"], " ".join(r["flags"])])
    with open(os.path.join(args.out, "attribution_summary.md"), "w") as f:
        f.write("# Attribution by option\n\nCell: bin and time change in % (B vs A). `!` dubliner-only, `*` noisy, `c` clock moved ≥3%.\n"
                "Majority bin excludes dubliner-only and noisy cells and `~`.\n")
        opts = defaultdict(list)
        for r in rows:
            opts[(r["option"], r["engine"])].append(r)
        for (opt, eng), rs in sorted(opts.items()):
            qs = sorted({r["query"] for r in rs}, key=lambda s: int(s[1:]) if s[1:].isdigit() else 0)
            clean = [r["bin"].split("+")[0] for r in rs if r["bin"] not in ("~", "?")
                     and not {"dubliner-only", "noisy"} & set(r["flags"])]
            maj = max(set(clean), key=clean.count) if clean else "—"
            ctx = sorted({"{} → {}".format(r["base"], r["config"]) for r in rs})
            f.write("\n## {}{}\n\nmajority bin: **{}** ({} cells); pairs: {}\n\n".format(
                opt, " (Hyper)" if eng == "h" else "", maj, len(clean), "; ".join(ctx[:6]) + (" …" if len(ctx) > 6 else "")))
            f.write("| machine | compiler | pair | " + " | ".join(qs) + " |\n|" + "---|" * (3 + len(qs)) + "\n")
            rowk = defaultdict(dict)
            for r in rs:
                rowk[(r["machine"], r["compiler"], r["base"] + "→" + r["config"])][r["query"]] = r
            for (m, cc, pr), cells in sorted(rowk.items()):
                txt = []
                for q in qs:
                    r = cells.get(q)
                    if not r:
                        txt.append("")
                        continue
                    mark = ("!" if "dubliner-only" in r["flags"] else "") + ("*" if "noisy" in r["flags"] else "") + \
                           ("c" if "clock" in r["flags"] else "")
                    txt.append("{} {}{}".format(r["bin"], chg(r["t"]), mark))
                f.write("| {} | {} | {} | {} |\n".format(m, cc, pr, " | ".join(txt)))
    if args.safe:
        safe_report(builds, args)
    print("{} builds, {} cells -> {}".format(len(builds), len(rows), args.out))


def safe_report(builds, args):
    """Every config against args.safe in the same result directory: a config
    is SAFE if no cell is more than --safe-limit slower, except dubliner cells
    whose clock moved (frequency license) or that are dubliner-only."""
    limit = args.safe_limit
    cells = defaultdict(list)  # (config, run) -> [dict]
    for d in {b["dir"] for b in builds}:
        here = [b for b in builds if b["dir"] == d and not b["hyper"]]
        for base in [b for b in here if b["base"] == args.safe]:
            for b in here:
                if b is base or b["compiler"] != base["compiler"] or b["run"] != base["run"]:
                    continue
                for q in set(b["metrics"]) & set(base["metrics"]):
                    tb, ta = b["metrics"][q].get("median"), base["metrics"][q].get("median")
                    if not tb or not ta:
                        continue
                    g = dl(base["metrics"][q].get("GHz"), b["metrics"][q].get("GHz"))
                    cells[(b["base"], b["run"])].append(dict(
                        machine=b["machine"], compiler=b["compiler"], query=q, r=tb / ta,
                        clock=g is not None and abs(g) >= CLOCK,
                        noisy=q in b["noisy"] or q in base["noisy"]))
    out = []
    for (cfg, run), cs in sorted(cells.items()):
        for c in cs:  # dubliner-only against the other machines, same compiler and query
            if c["machine"] != "dubliner":
                continue
            others = [math.log(o["r"]) for o in cs if o["machine"] != "dubliner"
                      and o["compiler"] == c["compiler"] and o["query"] == c["query"]]
            t = math.log(c["r"])
            c["dub_only"] = len(others) >= 2 and ((all(o * t < 0 for o in others) and abs(t) > 0.03)
                                                   or abs(t - statistics.median(others)) > DUBLINER_OFF)
        slow = [c for c in cs if c["r"] > 1 + limit]
        exempt = [c for c in slow if c["machine"] == "dubliner" and (c["clock"] or c.get("dub_only"))]
        # --safe-noisy ignore: noisy cells (rounds spread or cv > 5%) are listed
        # for a recheck but do not fail the config
        noisy = [c for c in slow if c not in exempt and c["noisy"] and args.safe_noisy == "ignore"]
        bad = [c for c in slow if c not in exempt and c not in noisy]
        per = defaultdict(list)
        for c in cs:
            per[c["machine"]].append(math.log(c["r"]))
        geo = math.exp(statistics.mean(math.log(c["r"]) for c in cs))
        out.append((not bad, geo, cfg, run, per, bad, exempt, len(cs), noisy))
    with open(os.path.join(args.out, "safe_report.md"), "w") as f:
        f.write("# Safe configurations vs `{}`\n\nSAFE: no cell more than {:.0f}% slower than `{}` in the same run, "
                "except dubliner cells whose clock moved ≥3% or that are dubliner-only (listed as exempt){}.\n\n".format(
                    args.safe, 100 * limit, args.safe,
                    "; noisy cells are not counted (listed as noisy, to recheck)" if args.safe_noisy == "ignore" else ""))
        f.write("| verdict | config | geomean | " + " | ".join("{} geomean / worst".format(m) for m in
                sorted({m for o in out for m in o[4]})) + " | slower cells |\n")
        machines = sorted({m for o in out for m in o[4]})
        f.write("|" + "---|" * (4 + len(machines)) + "\n")
        for ok, geo, cfg, run, per, bad, exempt, n, noisy in sorted(out, key=lambda o: (not o[0], o[1])):
            cols = []
            for m in machines:
                v = per.get(m)
                cols.append("{:.3f} / {:.3f}".format(math.exp(statistics.mean(v)), math.exp(max(v))) if v else "")
            f.write("| {} | `{}{}` | {:.3f} | {} | {} bad, {} exempt, {} noisy / {} |\n".format(
                "**SAFE**" if ok else "not safe", cfg, run, geo, " | ".join(cols), len(bad), len(exempt), len(noisy), n))
        for ok, geo, cfg, run, per, bad, exempt, n, noisy in sorted(out, key=lambda o: (not o[0], o[1])):
            if bad or exempt or noisy:
                f.write("\n**`{}{}`**\n".format(cfg, run))
                for c in sorted(bad, key=lambda c: -c["r"]):
                    f.write("- {} {} {}: {:+.1f}%{}\n".format(c["machine"], c["compiler"], c["query"], 100 * (c["r"] - 1),
                                                           " (noisy)" if c["noisy"] else ""))
                for c in sorted(noisy, key=lambda c: -c["r"]):
                    f.write("- noisy, not counted: {} {} {}: {:+.1f}%\n".format(c["machine"], c["compiler"], c["query"], 100 * (c["r"] - 1)))
                for c in sorted(exempt, key=lambda c: -c["r"]):
                    f.write("- exempt: {} {} {}: {:+.1f}% ({})\n".format(c["machine"], c["compiler"], c["query"], 100 * (c["r"] - 1),
                                                                        "clock" if c["clock"] else "dubliner-only"))


if __name__ == "__main__":
    main()
