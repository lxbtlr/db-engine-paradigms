#!/usr/bin/env python3
"""
ts.py -- a readable front-end for turbostat.

    ts.py --explain                     glossary of every column turbostat printed
    ts.py --run -- ./run_tpch --sf=10   measure a command, then summarise
    ts.py --run --interval 1 --for 20   just watch for 20s
    ts.py -f capture.txt --cpu 3        read a saved capture, focus on one CPU
    ts.py -f capture.txt --json         emit a record for the benchmark DB
    ts.py --compare typer.txt tw.txt    frequency delta between two captures

turbostat needs the msr module and usually root:
    sudo modprobe msr && sudo turbostat ...
On a SLURM compute node you likely cannot. Check first:
    ls /dev/cpu/0/msr && cat /proc/sys/kernel/perf_event_paranoid
If msr is unavailable, --run falls back to sampling
/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq, which is coarser
(no C-states, no package power, no IPC) but still catches downclocking.
"""

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time

# ---------------------------------------------------------------------
# Column glossary. Grouped by what question the column answers.
# ---------------------------------------------------------------------
GLOSSARY = {
    # --- identity ---
    "Package":  ("identity", "Physical socket number."),
    "Die":      ("identity", "Die within the package (chiplet on AMD, tile on some Intel)."),
    "Node":     ("identity", "NUMA node."),
    "Core":     ("identity", "Physical core id. Two rows share a Core when SMT is on."),
    "CPU":      ("identity", "Logical CPU (hardware thread) — matches /proc/cpuinfo and taskset."),
    "APIC":     ("identity", "Local APIC id. Rarely useful unless debugging topology."),
    "X2APIC":   ("identity", "Extended APIC id."),

    # --- frequency: the ones that matter for benchmarking ---
    "Avg_MHz":  ("frequency",
                 "Average clock over the WHOLE interval, including idle time. "
                 "Equals Busy% x Bzy_MHz / 100. Misleading for benchmarks: an idle "
                 "core looks slow even if it ran at max turbo when it ran."),
    "Busy%":    ("frequency",
                 "Fraction of the interval spent in C0 (actually executing). "
                 "During a saturated benchmark this should be ~100 on the threads "
                 "you pinned. Below ~95 means idle gaps — setup, I/O, or a barrier."),
    "Bzy_MHz":  ("frequency",
                 "*** The number you want. *** Average clock WHILE BUSY. This is the "
                 "frequency your query actually executed at. Compare it against "
                 "TSC_MHz (base) to see turbo or downclocking."),
    "TSC_MHz":  ("frequency",
                 "Rate of the invariant TSC = the CPU's nominal/base clock. It does "
                 "not change with turbo, so it is a stable denominator. "
                 "Bzy_MHz/TSC_MHz > 1 means turbo; < 1 means you are being throttled."),
    "UncMHz":   ("frequency", "Uncache/mesh frequency. Affects L3 and memory latency; "
                              "drops under power limits independently of core clock."),

    # --- efficiency ---
    "IPC":      ("efficiency",
                 "Instructions per cycle, hardware-measured. Cross-check against your "
                 "own perf counters — a large disagreement usually means your counters "
                 "were multiplexed."),
    "IRQ":      ("efficiency", "Interrupts serviced. A spike on a supposedly-isolated "
                               "core means something else is running there."),
    "SMI":      ("efficiency",
                 "System Management Interrupts. Invisible to perf and to your own "
                 "timers, but they stop the core. Any nonzero count during a timed "
                 "region makes that measurement suspect."),

    # --- idle states ---
    "POLL":     ("idle", "Times entered the polling idle loop (busy-waits, no power saving)."),
    "C1": ("idle", "Times entered core C-state C1 (shallowest halt)."),
    "C1E": ("idle", "Times entered C1E (halt + reduced voltage/frequency)."),
    "C3": ("idle", "Times entered C3."),
    "C6": ("idle", "Times entered C6 (core power-gated; wakeup costs microseconds)."),
    "C7": ("idle", "Times entered C7."),
    "C8": ("idle", "Times entered C8."), "C9": ("idle", "Times entered C9."),
    "C10": ("idle", "Times entered C10 (deepest)."),
    "POLL%": ("idle", "Percent of time in the polling idle loop."),
    "C1%": ("idle", "Percent of time in C1."),
    "C1E%": ("idle", "Percent of time in C1E."),
    "C3%": ("idle", "Percent of time in C3."),
    "C6%": ("idle",
            "Percent of time core was power-gated. Nonzero during a benchmark means "
            "cores are sleeping and paying wakeup latency — bad for timing stability."),
    "C7%": ("idle", "Percent of time in C7."),
    "C8%": ("idle", "Percent of time in C8."),
    "C9%": ("idle", "Percent of time in C9."),
    "C10%": ("idle", "Percent of time in C10."),
    "CPU%c1": ("idle", "Core residency in C1 (hardware view, not the kernel's)."),
    "CPU%c3": ("idle", "Core residency in C3."),
    "CPU%c6": ("idle", "Core residency in C6."),
    "CPU%c7": ("idle", "Core residency in C7."),

    # --- thermal / power ---
    "CoreTmp":  ("thermal", "Core temperature in C. Within ~5C of TjMax (usually 100) "
                            "means you are about to be thermally throttled."),
    "PkgTmp":   ("thermal", "Package temperature in C."),
    "CoreThr":  ("thermal", "Core throttle events. Nonzero = the clock was forcibly cut. "
                            "Discard or flag any measurement covering this interval."),
    "PkgWatt":  ("power", "Package power draw. If pinned at a flat value, you are "
                          "hitting RAPL PL1/PL2 and the clock is power-limited, not "
                          "thermally limited."),
    "CorWatt":  ("power", "Core-only power draw."),
    "GFXWatt":  ("power", "Integrated graphics power."),
    "RAMWatt":  ("power", "DRAM power. A rough proxy for memory traffic — useful for "
                          "confirming a query is bandwidth-bound."),
    "PKG_%":    ("power", "Percent of interval clamped by the RAPL package power limit."),
    "RAM_%":    ("power", "Percent of interval clamped by the DRAM power limit."),
    "GFXMHz":   ("power", "Integrated graphics clock."),

    # --- package idle ---
    "Totl%C0": ("pkg", "Percent of time ANY core in the package was busy."),
    "Any%C0":  ("pkg", "Percent of time at least one core was in C0."),
    "GFX%C0":  ("pkg", "Graphics C0 residency."),
    "CPUGFX%": ("pkg", "Both CPU and GFX in C0."),
    "Pkg%pc2": ("pkg", "Package C-state pc2 residency."),
    "Pkg%pc3": ("pkg", "Package C-state pc3 residency."),
    "Pkg%pc6": ("pkg", "Package C-state pc6 residency (whole socket power-gated)."),
    "Pkg%pc7": ("pkg", "Package C-state pc7 residency."),
    "Pkg%pc8": ("pkg", "Package C-state pc8 residency."),
    "Pkg%pc9": ("pkg", "Package C-state pc9 residency."),
    "Pk%pc10": ("pkg", "Package C-state pc10 residency."),
    "SYS%LPI": ("pkg", "System low-power idle residency."),
    "CPU%LPI": ("pkg", "CPU low-power idle residency."),
    "Time_Of_Day_Seconds": ("identity", "Wall clock timestamp of the sample."),
}

GROUP_ORDER = ["frequency", "efficiency", "thermal", "power", "idle", "pkg", "identity"]
GROUP_TITLE = {
    "frequency": "FREQUENCY — what clock your code actually ran at",
    "efficiency": "EFFICIENCY — how well it used those cycles",
    "thermal": "THERMAL — whether the clock got cut",
    "power": "POWER — whether a power cap cut it",
    "idle": "IDLE — C-state residency (should be ~0 during a benchmark)",
    "pkg": "PACKAGE — socket-wide idle",
    "identity": "IDENTITY — which CPU this row is",
}

NUMERIC_RE = re.compile(r"^-?\d+(\.\d+)?$")
SYNTHETIC_TAG = "# source: cpufreq-sysfs (Busy%% and IPC unavailable)"


# ---------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------
def parse_turbostat(text):
    """Return (header, samples) where samples is a list of intervals, each a
    list of row dicts. turbostat reprints the header each interval; the row
    whose Core/CPU is '-' is the package/system summary."""
    samples, cur, header = [], [], None
    synthetic = False
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        if line.startswith("#"):
            synthetic = synthetic or "cpufreq-sysfs" in line
            continue
        fields = line.split()
        looks_like_header = (
            fields[0] in ("Core", "Package", "CPU", "Avg_MHz", "Time_Of_Day_Seconds")
            and not NUMERIC_RE.match(fields[-1])
        )
        if looks_like_header:
            if cur:
                samples.append(cur)
                cur = []
            header = fields
            continue
        if header is None:
            continue
        if len(fields) != len(header):
            # turbostat pads inconsistently when a column is unsupported;
            # zip to the shorter of the two rather than dropping the row.
            pass
        row = {}
        for k, v in zip(header, fields):
            row[k] = float(v) if NUMERIC_RE.match(v) else v
        cur.append(row)
    if cur:
        samples.append(cur)
    if synthetic:
        for smp in samples:
            for r in smp:
                r["_synthetic_busy"] = True
    return header or [], samples


def is_summary(row):
    for k in ("Core", "CPU", "Package"):
        if k in row and row[k] == "-":
            return True
    return False


def select_rows(samples, cpu=None, summary_only=False):
    out = []
    for s in samples:
        for r in s:
            if summary_only and not is_summary(r):
                continue
            if cpu is not None:
                if is_summary(r) or r.get("CPU") != float(cpu):
                    continue
            if cpu is None and not summary_only and is_summary(r):
                continue
            out.append(r)
    return out


def agg(rows, key):
    vals = [r[key] for r in rows if isinstance(r.get(key), float)]
    return vals or None


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------
def fmt(v, nd=1):
    return "—" if v is None else f"{v:,.{nd}f}"


def report(samples, cpu=None, verbose=False):
    if not samples:
        print("no turbostat samples found in input", file=sys.stderr)
        return {}

    scope = f"CPU {cpu}" if cpu is not None else "all CPUs"
    rows = select_rows(samples, cpu=cpu)
    summ = select_rows(samples, summary_only=True)
    if not rows and cpu is not None:
        avail = sorted({int(r["CPU"]) for s in samples for r in s
                        if isinstance(r.get("CPU"), float)})
        print(f"\n  CPU {cpu} is not in this capture. Present: "
              f"{', '.join(map(str, avail)) if avail else '(none — summary rows only)'}",
              file=sys.stderr)
        if not avail:
            rows, scope = summ, "package summary"
        else:
            return {}
    if not rows:
        rows = summ
        scope = "package summary"
    if not rows:
        print("  capture contained no usable rows", file=sys.stderr)
        return {}

    synthetic_busy = any(r.get("_synthetic_busy") for r in rows)
    bzy = agg(rows, "Bzy_MHz")
    avg = agg(rows, "Avg_MHz")
    busy = agg(rows, "Busy%")
    tsc = agg(rows, "TSC_MHz")
    ipc = agg(rows, "IPC")
    coretmp = agg(rows, "CoreTmp")
    pkgwatt = agg(summ, "PkgWatt") or agg(rows, "PkgWatt")
    ramwatt = agg(summ, "RAMWatt") or agg(rows, "RAMWatt")
    smi = agg(rows, "SMI")
    thr = agg(rows, "CoreThr")
    c6 = agg(rows, "C6%") or agg(rows, "CPU%c6")
    unc = agg(rows, "UncMHz")

    base = statistics.median(tsc) if tsc else None
    bzy_mean = statistics.fmean(bzy) if bzy else None
    bzy_min = min(bzy) if bzy else None
    bzy_max = max(bzy) if bzy else None
    ratio = (bzy_mean / base) if (bzy_mean and base) else None

    print(f"\n  turbostat summary — {scope}, {len(samples)} interval(s), {len(rows)} row(s)")
    print("  " + "─" * 66)
    print(f"  Base clock (TSC_MHz)      {fmt(base, 0)} MHz")
    print(f"  Busy clock (Bzy_MHz)      {fmt(bzy_mean, 0)} MHz   "
          f"[min {fmt(bzy_min, 0)} / max {fmt(bzy_max, 0)}]")
    if ratio:
        arrow = "turbo" if ratio > 1.02 else ("BELOW BASE" if ratio < 0.98 else "at base")
        print(f"  Bzy / base                {ratio:.3f}x  ({arrow})")
    if synthetic_busy:
        print("  Busy%                     n/a (cpufreq fallback cannot measure it)")
    else:
        print(f"  Busy%                     {fmt(statistics.fmean(busy) if busy else None)} %")
    print(f"  Avg_MHz (incl. idle)      {fmt(statistics.fmean(avg) if avg else None, 0)} MHz")
    if unc:
        print(f"  Uncore                    {fmt(statistics.fmean(unc), 0)} MHz")
    if ipc:
        print(f"  IPC                       {fmt(statistics.fmean(ipc), 2)}")
    if coretmp:
        print(f"  Core temp                 {fmt(statistics.fmean(coretmp), 0)} C  "
              f"[max {fmt(max(coretmp), 0)}]")
    if pkgwatt:
        print(f"  Package power             {fmt(statistics.fmean(pkgwatt))} W  "
              f"[max {fmt(max(pkgwatt))}]")
    if ramwatt:
        print(f"  DRAM power                {fmt(statistics.fmean(ramwatt))} W")

    # stability across intervals: the thing that silently ruins repeated runs
    if bzy and len(bzy) > 2:
        spread = (max(bzy) - min(bzy)) / max(min(bzy), 1)
        print(f"  Clock spread              {spread*100:.1f} % across samples")

    print("\n  READING")
    for line in diagnose(bzy_mean, bzy_min, base, busy, coretmp, pkgwatt,
                         smi, thr, c6, ratio, synthetic_busy):
        print("   " + line)

    if verbose:
        print("\n  PER-INTERVAL")
        hdr = ["#", "Bzy_MHz", "Busy%", "IPC", "CoreTmp", "PkgWatt"]
        print("   " + "".join(h.rjust(10) for h in hdr))
        for i, s in enumerate(samples):
            sel = [r for r in s if (cpu is None and is_summary(r))
                   or (cpu is not None and r.get("CPU") == float(cpu))]
            if not sel:
                sel = [r for r in s if not is_summary(r)]
            def m(k):
                v = [r[k] for r in sel if isinstance(r.get(k), float)]
                return statistics.fmean(v) if v else None
            vals = [str(i), fmt(m("Bzy_MHz"), 0), fmt(m("Busy%")), fmt(m("IPC"), 2),
                    fmt(m("CoreTmp"), 0), fmt(m("PkgWatt"))]
            print("   " + "".join(v.rjust(10) for v in vals))

    return {
        "base_mhz": base, "bzy_mean_mhz": bzy_mean,
        "bzy_min_mhz": bzy_min, "bzy_max_mhz": bzy_max,
        "bzy_over_base": ratio,
        "busy_pct": None if synthetic_busy else (statistics.fmean(busy) if busy else None),
        "source": "cpufreq-sysfs" if synthetic_busy else "turbostat",
        "avg_mhz": statistics.fmean(avg) if avg else None,
        "ipc": statistics.fmean(ipc) if ipc else None,
        "core_temp_max_c": max(coretmp) if coretmp else None,
        "pkg_watt_mean": statistics.fmean(pkgwatt) if pkgwatt else None,
        "ram_watt_mean": statistics.fmean(ramwatt) if ramwatt else None,
        "smi_total": sum(smi) if smi else None,
        "core_throttle_total": sum(thr) if thr else None,
        "c6_pct_mean": statistics.fmean(c6) if c6 else None,
        "n_samples": len(samples),
    }


def diagnose(bzy, bzy_min, base, busy, temp, watt, smi, thr, c6, ratio,
             synthetic_busy=False):
    """Heuristics aimed at query-engine benchmarking specifically."""
    out = []
    busy_m = None if synthetic_busy else (statistics.fmean(busy) if busy else None)
    temp_m = max(temp) if temp else None

    if ratio is not None and synthetic_busy:
        if ratio < 0.98:
            out.append(
                f"? Clock averaged {ratio:.2f}x base. This is a cpufreq-sysfs capture, "
                "so idle time is indistinguishable from downclocking — the dip may "
                "just be idle cores. Get turbostat (or msr access) before drawing "
                "conclusions about AVX-512 licensing.")
        elif ratio > 1.02:
            out.append(f"  Clock averaged {ratio:.2f}x base (turbo).")
    elif ratio is not None and busy_m is not None:
        if ratio < 0.98 and busy_m > 90:
            out.append(
                f"! Running BELOW base clock ({ratio:.2f}x) while ~fully busy. On Intel "
                "this is the classic signature of AVX-512/AVX2 license-based "
                "downclocking, or a RAPL power cap. If you are comparing a SIMD "
                "engine against a scalar one, they are not running at the same clock "
                "and a raw ms comparison understates the SIMD version's per-cycle work.")
        elif ratio < 0.98:
            out.append(
                f"! Below base clock ({ratio:.2f}x), but Busy% is only {busy_m:.0f}, so "
                "this is probably idle-state entry dragging the average down rather "
                "than a license or power cap. Re-capture with --cpu on a pinned core "
                "before concluding anything about downclocking.")
        elif ratio > 1.02:
            out.append(f"  Turbo active ({ratio:.2f}x base). Fine, but turbo depends on how "
                       "many cores are active — a 1-thread run turbos higher than a "
                       "20-thread run, so single- vs multi-thread scaling numbers are "
                       "inflated unless you normalise by Bzy_MHz.")
        else:
            out.append("  Clock is sitting at base. Most reproducible case.")

    if bzy and bzy_min and bzy_min < 0.9 * bzy:
        out.append(f"! Clock dipped to {bzy_min:,.0f} MHz at some point (mean {bzy:,.0f}). "
                   "Something transient throttled you; check the per-interval table "
                   "with -v and correlate against which query was running.")

    if not synthetic_busy and busy_m is not None and busy_m < 90:
        out.append(f"! Busy% is only {busy_m:.0f}. The measurement window includes idle "
                   "time — data loading, thread barriers, or you are averaging over "
                   "CPUs you never pinned work to. Restrict with --cpu.")

    if c6 and statistics.fmean(c6) > 1:
        out.append(f"! Cores spent {statistics.fmean(c6):.1f}% in C6 (power-gated). Wakeup "
                   "latency will add variance. Consider "
                   "`cpupower idle-set -D 0` or booting with intel_idle.max_cstate=1.")

    if temp_m is not None and temp_m >= 90:
        out.append(f"! Core temp reached {temp_m:.0f} C — close to TjMax. Later "
                   "repetitions in a long sweep will be slower than early ones. "
                   "Check run_ordinal against runtime in your results DB.")

    if thr and sum(thr) > 0:
        out.append(f"!! {sum(thr):.0f} core throttle event(s). Discard or flag any "
                   "measurement overlapping this capture.")

    if smi and sum(smi) > 0:
        out.append(f"!! {sum(smi):.0f} SMI(s) fired. SMIs stall the core invisibly to "
                   "perf and to gettimeofday. Treat affected timings as outliers.")

    if watt:
        w = [x for x in watt if x]
        if len(w) > 3 and (max(w) - min(w)) / max(max(w), 1) < 0.02 and ratio and ratio < 1.0:
            out.append("  Package power is flat while the clock is below base — that is a "
                       "RAPL power limit, not a thermal one. Raising cooling will not help; "
                       "the cap will.")

    if not out:
        out.append("  Nothing alarming. Clock, temperature, and residency all look clean.")
    return out


# ---------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------
def have_turbostat():
    return shutil.which("turbostat") is not None


def msr_available():
    return os.path.exists("/dev/cpu/0/msr")


def run_turbostat(cmd=None, interval=1.0, duration=None, show=None, out_path=None):
    if not have_turbostat():
        raise RuntimeError("turbostat not found (package: linux-tools-common / kernel-tools)")
    argv = ["turbostat", "--quiet", "--interval", str(interval)]
    if show:
        argv += ["--show", show]
    if cmd:
        argv += ["--"] + cmd
    elif duration:
        argv += ["--num_iterations", str(max(1, int(duration / interval)))]
    proc = subprocess.run(argv, capture_output=True, text=True)
    # turbostat writes the table to stderr and the child's output to stdout
    text = proc.stderr if proc.stderr.strip() else proc.stdout
    if out_path:
        with open(out_path, "w") as f:
            f.write(text)
    if not text.strip():
        raise RuntimeError(
            "turbostat produced no output. Usually means no msr access — try "
            "`sudo modprobe msr` and run as root, or use --fallback.")
    return text, proc.stdout


def sysfs_fallback(cmd=None, interval=0.5, duration=10.0, cpu_root="/sys/devices/system/cpu"):
    """No-privilege substitute: sample scaling_cur_freq. Produces a table with
    the same column names so the rest of the tool works unchanged."""
    import glob
    paths = []
    for d in sorted(glob.glob(os.path.join(cpu_root, "cpu[0-9]*"))):
        m = re.search(r"cpu(\d+)$", d)
        f = os.path.join(d, "cpufreq", "scaling_cur_freq")
        if m and os.path.exists(f):
            paths.append((int(m.group(1)), f))
    paths.sort()
    if not paths:
        raise RuntimeError("no cpufreq sysfs interface either; nothing to sample")

    base_khz = None
    for cand in ("cpu0/cpufreq/base_frequency", "cpu0/cpufreq/cpuinfo_max_freq"):
        p = os.path.join(cpu_root, cand)
        if os.path.exists(p):
            base_khz = int(open(p).read().strip())
            break

    proc = subprocess.Popen(cmd) if cmd else None
    deadline = time.time() + duration if not cmd else float("inf")
    # Tag the capture: Busy% below is synthetic (sysfs cannot measure C0
    # residency), and the reporter must not treat it as real.
    lines = [SYNTHETIC_TAG]
    while True:
        if proc is not None and proc.poll() is not None:
            break
        if proc is None and time.time() > deadline:
            break
        lines.append("Core\tCPU\tAvg_MHz\tBusy%\tBzy_MHz\tTSC_MHz")
        mhzs = []
        for cpu, p in paths:
            try:
                mhz = int(open(p).read().strip()) / 1000.0
            except OSError:
                continue
            mhzs.append(mhz)
            base = (base_khz / 1000.0) if base_khz else mhz
            lines.append(f"{cpu}\t{cpu}\t{mhz:.0f}\t100.0\t{mhz:.0f}\t{base:.0f}")
        if mhzs:
            # synthesise the '-' summary row turbostat would print, so
            # package-scope reporting works the same either way
            mean = sum(mhzs) / len(mhzs)
            base = (base_khz / 1000.0) if base_khz else mean
            lines.insert(len(lines) - len(mhzs),
                         f"-\t-\t{mean:.0f}\t100.0\t{mean:.0f}\t{base:.0f}")
        time.sleep(interval)
    if proc is not None:
        proc.wait()
    if len(lines) <= 1:
        raise RuntimeError("captured no samples; the command exited too fast to sample")
    return "\n".join(lines) + "\n", ""


# ---------------------------------------------------------------------
# Explain / compare
# ---------------------------------------------------------------------
def explain(header=None, all_cols=False):
    cols = header if (header and not all_cols) else list(GLOSSARY)
    seen, groups = set(), {}
    for c in cols:
        if c in seen:
            continue
        seen.add(c)
        grp, desc = GLOSSARY.get(c, ("identity", "(not in glossary — check `man turbostat`)"))
        groups.setdefault(grp, []).append((c, desc))
    for g in GROUP_ORDER:
        if g not in groups:
            continue
        print(f"\n  {GROUP_TITLE[g]}")
        print("  " + "─" * 66)
        for name, desc in groups[g]:
            print(f"  {name:<12} {wrap(desc, 12)}")


def wrap(text, indent, width=66):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    lines.append(cur)
    pad = "\n  " + " " * indent
    return pad.join(lines)


def compare(path_a, path_b, cpu=None):
    stats = {}
    for label, path in (("A: " + os.path.basename(path_a), path_a),
                        ("B: " + os.path.basename(path_b), path_b)):
        _, samples = parse_turbostat(open(path).read())
        rows = select_rows(samples, cpu=cpu) or select_rows(samples, summary_only=True)
        def m(k):
            v = agg(rows, k)
            return statistics.fmean(v) if v else None
        stats[label] = {"Bzy_MHz": m("Bzy_MHz"), "Busy%": m("Busy%"),
                        "IPC": m("IPC"), "CoreTmp": m("CoreTmp"),
                        "PkgWatt": m("PkgWatt"), "RAMWatt": m("RAMWatt")}
    (la, sa), (lb, sb) = list(stats.items())
    print(f"\n  {'metric':<10}{la[:22]:>24}{lb[:22]:>24}{'delta':>12}")
    print("  " + "─" * 70)
    for k in ("Bzy_MHz", "Busy%", "IPC", "CoreTmp", "PkgWatt", "RAMWatt"):
        a, b = sa[k], sb[k]
        d = "" if (a is None or b is None or a == 0) else f"{(b - a) / a * 100:+.1f}%"
        print(f"  {k:<10}{fmt(a, 1):>24}{fmt(b, 1):>24}{d:>12}")
    a, b = sa["Bzy_MHz"], sb["Bzy_MHz"]
    if a and b and abs(b - a) / a > 0.03:
        slower, faster = (lb, la) if b < a else (la, lb)
        print(f"\n  ! {slower.split(':')[0]} ran {abs(b-a)/max(a,b)*100:.1f}% slower in CLOCK "
              f"than {faster.split(':')[0]}. Wall-time differences between these two "
              "captures are partly frequency, not algorithmic. Normalise by cycles "
              "(or by Bzy_MHz) before attributing the gap to the engine.")


# ---------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Readable turbostat front-end for benchmark work.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("-f", "--file", help="read a saved turbostat capture")
    p.add_argument("--run", action="store_true", help="capture now")
    p.add_argument("--fallback", action="store_true",
                   help="skip turbostat, sample cpufreq sysfs (no root needed)")
    p.add_argument("--cpu", type=int, help="focus on one logical CPU")
    p.add_argument("--interval", type=float, default=1.0)
    p.add_argument("--for", dest="duration", type=float, default=10.0,
                   help="seconds to capture when no command is given")
    p.add_argument("--show", help="pass through to turbostat --show")
    p.add_argument("-o", "--save", help="write the raw capture here")
    p.add_argument("--explain", action="store_true", help="glossary for the columns seen")
    p.add_argument("--explain-all", action="store_true", help="glossary for every column")
    p.add_argument("--json", action="store_true",
                   help="emit a JSON record (feeds run.mean_freq_mhz etc.)")
    p.add_argument("-v", "--verbose", action="store_true", help="per-interval table")
    p.add_argument("--compare", nargs=2, metavar=("A", "B"))
    p.add_argument("cmd", nargs=argparse.REMAINDER,
                   help="command to measure, after --")
    args = p.parse_args()

    if args.explain_all:
        explain(all_cols=True)
        return
    if args.compare:
        compare(args.compare[0], args.compare[1], args.cpu)
        return

    if args.file:
        text = open(args.file).read()
        child_out = ""
    elif args.run or args.fallback or args.cmd:
        cmd = [c for c in args.cmd if c != "--"] or None
        try:
            if args.fallback or not (have_turbostat() and msr_available()):
                if not args.fallback:
                    print("  (turbostat or /dev/cpu/*/msr unavailable — "
                          "falling back to cpufreq sysfs)", file=sys.stderr)
                text, child_out = sysfs_fallback(cmd, args.interval, args.duration)
                if args.save:
                    open(args.save, "w").write(text)
            else:
                text, child_out = run_turbostat(cmd, args.interval, args.duration,
                                                args.show, args.save)
        except RuntimeError as e:
            print(f"error: {e}", file=sys.stderr)
            sys.exit(1)
        if child_out.strip():
            print(child_out, end="")
    else:
        if sys.stdin.isatty():
            p.print_help()
            return
        text = sys.stdin.read()

    header, samples = parse_turbostat(text)
    stats = report(samples, cpu=args.cpu, verbose=args.verbose)

    if args.explain:
        explain(header)
    if args.json:
        print("\n" + json.dumps(stats, indent=2))
    print()


if __name__ == "__main__":
    main()
