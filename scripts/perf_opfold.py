#!/usr/bin/env python3
"""Fold sampled top-down events (scripts/perf_ops.sh) into operators.

usage: perf_opfold.py <run_tpch> <engine v|h> <query> <out dir> [perf script output, default stdin]

Input: `perf script -F event,period,ip,sym,symoff` of a perf_ops.sh record,
call stacks included (--call-graph fp). Every sample adds its period to the
event's sum in each of the tables below; perf_ops_report.py turns the sums
into top-down shares and cycles per tuple.

  functions.csv  by leaf function (perf's symbol, inlined code included):
                 compare with VTune's hw-events report at -inline-mode off
  operators.csv  -e v (Tectorwise): by the innermost vectorwise::<Op>::next
                 frame on the stack (exclusive attribution), named by the
                 path of operator frames from the root. Each frame gets a
                 phase from where its return address lies in its next():
                 inside the `if (!consumed)` / `if (!cont.consumed)` block
                 (Hashjoin build, HashGroup local, FixedAggr consume) or after
                 it (probe, global, output). Samples with no operator frame:
                 "plan build" (QueryBuilder / getQuery) or "other: <leaf>".
                 -e h (Typer): by pipeline, the line in qN_hyper that starts
                 it (the tbb::parallel_* call: exact), and by the innermost
                 qN.cpp line with the function inlined directly under it
                 (approximate: -O3 line tables). Both carry the source text,
                 so labels can match code instead of line numbers.
  totals.csv     the whole run

Lines come from addr2line -i on the binary's debug info (-g).
"""
import collections
import csv
import os
import re
import subprocess
import sys

binary, engine, query, outdir = sys.argv[1:5]
src = open(sys.argv[5]) if len(sys.argv) > 5 else sys.stdin
QFILE, QFN = f"q{query}.cpp", f"q{query}_hyper"
OP = re.compile(r"^vectorwise::(Select|Project|Scan|Hashjoin|HashGroup|FixedAggr|ResultWriter)::next$")
PHASE_NAMES = {"Hashjoin": ("build", "probe"), "HashGroup": ("local", "global"), "FixedAggr": ("consume", "output")}
FRAME = re.compile(r"^\s*([0-9a-f]+) (.*?)\+0x([0-9a-f]+)$")
KERNEL = 0xffff000000000000

# ------------------------------------------------------------------ binary
text = [(int(f[0], 16), f[2]) for f in
        (l.split(None, 2) for l in subprocess.run(["nm", binary], capture_output=True, text=True).stdout.splitlines())
        if len(f) == 3 and f[1] in ("T", "t")]
main_static = next(a for a, n in text if n == "main")
text_lo, text_hi = min(a for a, _ in text), max(a for a, _ in text) + 4096
base = None


def static(ip):
    """run_tpch's link-time address for a runtime ip, or None (kernel, libs)"""
    if ip >= KERNEL or base is None:
        return None
    a = ip - base
    return a if text_lo <= a < text_hi else None


# ------------------------------------------------------------------ samples
# aggregate by everything the tables need, then resolve lines once per key
acc = collections.defaultdict(collections.Counter)
events = []


def add(ev, period, fr):
    global base
    if base is None:
        for ip, s, off in fr:
            if s == "main":
                base = ip - off - main_static
                break
    leaf = fr[0][1] if fr else "?"
    if engine == "v":
        ops = tuple((s, static(ip) - (1 if i else 0) if static(ip) is not None else None)
                    for i, (ip, s, off) in enumerate(fr) if OP.match(s))[::-1]
        plan = not ops and any("getQuery" in s or "QueryBuilder" in s for _, s, _ in fr)
        key = (leaf, ops, plan)
    else:
        key = (leaf, tuple(a for a in (static(ip) - (1 if i else 0) if static(ip) is not None else None
                                       for i, (ip, s, off) in enumerate(fr)) if a is not None))
    acc[key][ev] += period
    if ev not in events:
        events.append(ev)


pending = []  # samples before the first main frame fixed the load base
cur = None
for line in src:
    if not line.strip():
        if cur:
            (add(*cur) if base is not None or any(s == "main" for _, s, _ in cur[2]) else pending.append(cur))
        cur = None
        continue
    if cur is None:
        p, ev = line.split()[:2]
        cur = (ev.rstrip(":").split("/")[0], int(p), [])
        continue
    m = FRAME.match(line)
    if m:
        cur[2].append((int(m.group(1), 16), m.group(2), int(m.group(3), 16)))
if cur:
    pending.append(cur)
if base is None:
    sys.exit("perf_opfold: no main() frame on any stack; was the record made with --call-graph fp?")
for s in pending:
    add(*s)

# ------------------------------------------------------------------ lines
chains = {}


def resolve(addrs):
    addrs = sorted({a for a in addrs if a is not None and a not in chains})
    for i in range(0, len(addrs), 4000):
        part = addrs[i:i + 4000]
        out = subprocess.run(["addr2line", "-a", "-f", "-i", "-C", "-e", binary] + [hex(a) for a in part],
                             capture_output=True, text=True).stdout.splitlines()
        a, j = None, 0
        while j < len(out):
            if re.fullmatch(r"0x[0-9a-f]+", out[j]):
                a = int(out[j], 16)
                chains[a] = []
                j += 1
                continue
            fn, loc = out[j], out[j + 1] if j + 1 < len(out) else "?:0"
            path, _, ln = loc.rpartition(":")
            ln = re.match(r"\d+", ln)
            chains[a].append((fn, path, int(ln.group()) if ln else 0))  # inner -> outer
            j += 2


sources = {}


def source_line(path, ln):
    if path not in sources:
        try:
            sources[path] = open(path, errors="replace").read().split("\n")
        except OSError:
            sources[path] = []
    lines = sources[path]
    return lines[ln - 1].strip()[:80] if 0 < ln <= len(lines) else ""


blocks = {}


def consumed_block(path, op):
    """(first, last) line of the `if (!consumed)` block in <op>::next()"""
    if (path, op) in blocks:
        return blocks[(path, op)]
    res = None
    if path not in sources:
        source_line(path, 1)
    lines = sources[path]
    start = next((i for i, l in enumerate(lines) if re.match(rf"^size_t {op}::next\(\)", l)), None)
    if start is not None:
        for i in range(start, min(start + 40, len(lines))):
            if re.search(r"if \(!(cont\.)?consumed\) \{", lines[i]):
                # count braces in the first branch of every #if chain only:
                # #ifdef / #else branches often open the same block twice
                depth, pp = 0, []
                for k in range(i, len(lines)):
                    d = re.match(r"\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b", lines[k])
                    if d:
                        if d.group(1).startswith("if"):
                            pp.append(False)
                        elif d.group(1) == "endif":
                            pp and pp.pop()
                        elif pp:
                            pp[-1] = True
                        continue
                    if any(pp):
                        continue
                    code = re.sub(r"//.*|\"(\\.|[^\"])*\"|'(\\.|[^'])*'", "", lines[k])
                    depth += code.count("{") - code.count("}")
                    if depth == 0:
                        res = (i + 1, k + 1)
                        break
                break
    blocks[(path, op)] = res
    return res


def phase(op, a):
    if op not in PHASE_NAMES:
        return ""
    if a is None or not chains.get(a):
        return "?"
    fn, path, ln = chains[a][-1]  # the frame's own function: outermost entry
    blk = consumed_block(path, op)
    if not blk:
        return "?"
    return PHASE_NAMES[op][0] if blk[0] <= ln <= blk[1] else PHASE_NAMES[op][1]


def short(fn):
    fn = re.sub(r"\(.*", "", fn)
    depth, out = 0, ""
    for ch in fn:  # drop template arguments
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif depth == 0:
            out += ch
    return out.split("::")[-1] or fn


# ------------------------------------------------------------------ tables
functions = collections.defaultdict(collections.Counter)
operators = collections.defaultdict(collections.Counter)
total = collections.Counter()
if engine == "v":
    resolve(a for (_, ops, _) in acc for _, a in ops)
else:
    resolve(a for (_, addrs) in acc for a in addrs)
unresolved = collections.Counter()
for key, c in acc.items():
    functions[(key[0],)].update(c)
    total.update(c)
    if engine == "v":
        leaf, ops, plan = key
        if ops:
            path = []
            for s, a in ops:
                op = OP.match(s).group(1)
                ph = phase(op, a)
                if ph == "?":
                    unresolved[op] += c["slots"]
                path.append(op + (f"[{ph}]" if ph else ""))
            operators[(" > ".join(path),)].update(c)
        else:
            operators[("plan build" if plan else "other: " + leaf,)].update(c)
    else:
        leaf, addrs = key
        entries = [e for a in addrs for e in chains.get(a, [])]  # inner -> outer
        pipe = next(((ln, source_line(p, ln)) for fn, p, ln in reversed(entries)
                     if p.endswith("/" + QFILE) and fn.startswith(QFN + "(") and "{lambda" not in fn), None)
        inner = next(((ln, source_line(p, ln), short(entries[k - 1][0]) if k else "")
                      for k, (fn, p, ln) in enumerate(entries) if p.endswith("/" + QFILE)), None)
        if pipe is None:
            operators[("", "", "", "", "", "other: " + leaf)].update(c)
        else:
            operators[(pipe[0], pipe[1]) + (inner if inner else ("", "", "")) + ("",)].update(c)

os.makedirs(outdir, exist_ok=True)
ev_cols = sorted(events, key=lambda e: (e != "slots", e))


def write(name, cols, table):
    with open(os.path.join(outdir, name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols + ev_cols)
        for k, c in sorted(table.items(), key=lambda kv: -kv[1]["slots"]):
            w.writerow(list(k) + [c[e] for e in ev_cols])


write("functions.csv", ["function"], functions)
write("totals.csv", ["scope"], {("run",): total})
if engine == "v":
    write("operators.csv", ["operator"], operators)
else:
    write("operators.csv", ["pipeline_line", "pipeline_src", "line", "src", "inlined", "other"], operators)
share = sum(unresolved.values()) / (total["slots"] or 1)
print(f"perf_opfold: {len(acc)} stack keys, {len(chains)} addresses; phase unknown for {share:.2%} of slots"
      + (f" ({dict(unresolved)})" if unresolved else ""))
