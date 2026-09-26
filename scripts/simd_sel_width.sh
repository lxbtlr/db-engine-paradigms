#!/usr/bin/env bash
# VW_SIMD_SEL_WIDTH evaluation: do the 256-bit (ymm) selection kernels win
# back Q1's ~5% loss by avoiding the AVX-512 frequency license drop, and
# what do they cost on Q6/Q3/Q5/Q18?
#
# For every compiler x config (all on top of the tuned baseline TUNED):
#   base      VW_SIMD_SEL=OFF
#   w512      VW_SIMD_SEL=ON  WIDTH=512 COMPRESS=mem UNROLL=2  (best 512 so far)
#   w256      VW_SIMD_SEL=ON  WIDTH=256 COMPRESS=reg
#   w256_mem  VW_SIMD_SEL=ON  WIDTH=256 COMPRESS=mem
# it
#   1. configures + builds run_tpch, run_selbench, test_all into build_width/<compiler>_<config>
#      and logs the arch flags the TARGET_MACHINE preset resolved to
#   2. checks codegen: the redirected sel_col_val/sel_col_col kernels in
#      Selection.cpp.o use zmm only for w512, ymm vpcompressd for w256*
#   3. runs the unit tests (SimdSel*) and the TPC-H correctness tests (TPCH.*)
#   4. runs run_selbench (sel_col_val/sel_col_col rows, incl. simd256_*) from the w256 build
#   5. times run_tpch -e v on QUERIES, ROUNDS rounds, configs alternating in
#      each round, pinned to one CPU; with perf, also counts cycles, ref-cycles
#      and the Cascade Lake AVX license cycles (core_power.lvl{0,1,2}_turbo_license)
# Output: results/simd_sel_width_<timestamp>/ with timing.csv, speedup.csv,
# perf.csv, license.csv and per-build logs.
#
# Environment (all optional):
#   COMPILERS  "gcc clang"
#   CONFIGS    "base w512 w256 w256_mem"
#   TUNED      "-DVW_GROUP_AGGR=ON -DVW_GROUP_AGGR_SEL=ON -DVW_POS_16=ON -DVW_USE_CRC32=ON -DHUGE_2MB_MALLOC_HUGE=ON"
#   MACHINE    $(hostname -s) if it is a preset (dubliner, roquefort, manchego, burrata, kafir, rpi5), else custom
#   DATADIR    /tank/alexb/swole/          test_all reads $DATADIR/tpch/sf1/
#   TPCH_PATH  /tank/alexb/swole/tpch/sf1  run_tpch -p
#   QUERIES    1,6,3,5,18
#   REPS       30        run_tpch -r per invocation
#   ROUNDS     3         alternating rounds per query
#   SETTLE     2         run_tpch -s
#   CPU        0         CPU for every timed run (single thread)
#   NODE       NUMA node of CPU (from sysfs)
#   NUMA       "numactl --physcpubind=$CPU --membind=$NODE"; taskset -c $CPU
#              without numactl; "" = no pinning
#   PERF       1         0 = skip perf stat counters
#   TEST_THREADS 4       worker threads for test_all
#   JOBS       $(nproc)
#   SKIP_BUILD 0         1 = reuse build dirs
#   TIMEOUT    1800      seconds per step (0 = none)
#   FORCE      0         1 = run even without AVX-512VL (x86); results are meaningless there
set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPILERS=${COMPILERS:-"gcc clang"}
CONFIGS=${CONFIGS:-"base w512 w256 w256_mem"}
TUNED=${TUNED:-"-DVW_GROUP_AGGR=ON -DVW_GROUP_AGGR_SEL=ON -DVW_POS_16=ON -DVW_USE_CRC32=ON -DHUGE_2MB_MALLOC_HUGE=ON"}
# MACHINE defaults to this host's short name when it is a TARGET_MACHINE
# preset, else "custom" (-march=native, CMake topology defaults).
detect_machine() {
  local h; h=$(hostname -s 2>/dev/null || hostname)
  case "$h" in dubliner|roquefort|manchego|burrata|kafir|rpi5) echo "$h" ;; *) echo custom ;; esac
}
MACHINE=${MACHINE:-$(detect_machine)}
DATADIR=${DATADIR:-/tank/alexb/swole/}
TPCH_PATH=${TPCH_PATH:-/tank/alexb/swole/tpch/sf1}
QUERIES=${QUERIES:-1,6,3,5,18}
REPS=${REPS:-30}
ROUNDS=${ROUNDS:-3}
SETTLE=${SETTLE:-2}
CPU=${CPU:-0}
# NUMA node of CPU from sysfs (0 if the kernel exposes none)
cpu_node() {
  local d
  for d in /sys/devices/system/cpu/cpu"$1"/node[0-9]*; do [ -e "$d" ] && { echo "${d##*node}"; return; }; done
  echo 0
}
NODE=${NODE:-$(cpu_node "$CPU")}
# pin with numactl (CPU + its local memory), else taskset (CPU only), else none
if [ -z "${NUMA+x}" ]; then
  if command -v numactl > /dev/null; then NUMA="numactl --physcpubind=$CPU --membind=$NODE"
  elif command -v taskset > /dev/null; then NUMA="taskset -c $CPU"
  else NUMA=""; fi
fi
PERF=${PERF:-1}
TEST_THREADS=${TEST_THREADS:-4}
JOBS=${JOBS:-$(nproc)}
SKIP_BUILD=${SKIP_BUILD:-0}
TIMEOUT=${TIMEOUT:-1800}

# The script compares AVX-512 (zmm) against AVX-512VL (ymm) kernels, so it
# needs an x86 CPU with AVX-512VL (not Zen 3 or ARM).
if ! grep -q -w avx512vl /proc/cpuinfo && [ "${FORCE:-0}" != 1 ]; then
  echo "simd_sel_width.sh: this CPU ($(uname -m)) has no AVX-512VL; nothing to compare (FORCE=1 to run anyway)" >&2
  exit 2
fi
OUT="$ROOT/results/simd_sel_width_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
FAILS=0
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/driver.log"; }
fail() { log "FAIL: $*"; FAILS=$((FAILS + 1)); }
tlimit() { if [ "$TIMEOUT" -gt 0 ]; then timeout --kill-after=30 "$TIMEOUT" "$@"; else "$@"; fi; }

# Every SIMD_SEL option is passed explicitly: CMake caches them.
config_flags() {
  local sel="-DVW_SIMD_SEL_GATHER=scalar -DVW_SIMD_SEL_CHAR=OFF -DVW_SIMD_HASH=OFF -DVW_JOIN_PREFETCH=OFF"
  case "$1" in
    base)     echo "$sel -DVW_SIMD_SEL=OFF -DVW_SIMD_SEL_WIDTH=512 -DVW_SIMD_SEL_COMPRESS=reg -DVW_SIMD_SEL_UNROLL=1" ;;
    w512)     echo "$sel -DVW_SIMD_SEL=ON  -DVW_SIMD_SEL_WIDTH=512 -DVW_SIMD_SEL_COMPRESS=mem -DVW_SIMD_SEL_UNROLL=2" ;;
    w256)     echo "$sel -DVW_SIMD_SEL=ON  -DVW_SIMD_SEL_WIDTH=256 -DVW_SIMD_SEL_COMPRESS=reg -DVW_SIMD_SEL_UNROLL=1" ;;
    w256_mem) echo "$sel -DVW_SIMD_SEL=ON  -DVW_SIMD_SEL_WIDTH=256 -DVW_SIMD_SEL_COMPRESS=mem -DVW_SIMD_SEL_UNROLL=1" ;;
    *) echo "unknown config $1" >&2; exit 2 ;;
  esac
}
bdir() { echo "$ROOT/build_width/$1_$2"; }
# reject unknown config names before building (exit inside $(config_flags)
# only leaves the subshell)
for cfg in $CONFIGS; do (config_flags "$cfg") > /dev/null || { rm -rf "$OUT"; exit 2; }; done

# ---------------------------------------------------------------- machine info
{
  echo "host: $(hostname)"
  echo "MACHINE: $MACHINE  CPU: $CPU  NODE: $NODE  pin: ${NUMA:-none}"
  echo "date: $(date -Is)"
  echo "git:  $(git -C "$ROOT" rev-parse --short HEAD) $(git -C "$ROOT" status --porcelain --untracked-files=no | wc -l) dirty tracked files"
  lscpu | grep -E 'Model name|Socket|Core|Thread|NUMA node\(s\)|MHz' || true
  echo -n "avx512 flags: "; grep -o -w -E 'avx512(f|bw|vl|dq|cd)' /proc/cpuinfo | sort -u | tr '\n' ' '; echo
  echo -n "governor cpu$CPU: "; cat "/sys/devices/system/cpu/cpu$CPU/cpufreq/scaling_governor" 2>/dev/null || echo n/a
  echo -n "turbo (intel_pstate no_turbo): "; cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo n/a
  echo "TUNED: $TUNED"
} | tee "$OUT/machine.txt"
[ "$MACHINE" = custom ] && log "WARNING: host $(hostname -s) is not a TARGET_MACHINE preset; building with MACHINE=custom (-march=native, default topology). Set MACHINE= to override."
grep -q -w avx512vl /proc/cpuinfo || log "WARNING: no avx512vl (FORCE=1); SIMD kernels fall back and the codegen checks will fail"

# perf events: license counters exist on Skylake-SP / Cascade Lake
EVENTS="cycles,ref-cycles,instructions"
if [ "$PERF" = 1 ]; then
  if ! command -v perf > /dev/null; then log "perf not found; counters skipped"; PERF=0
  elif perf stat -e core_power.lvl0_turbo_license true > /dev/null 2>&1; then
    EVENTS="$EVENTS,core_power.lvl0_turbo_license,core_power.lvl1_turbo_license,core_power.lvl2_turbo_license"
  else
    log "core_power.lvl*_turbo_license not available; counting cycles/ref-cycles only"
  fi
  perf stat -e cycles true > /dev/null 2>&1 || { log "perf stat not permitted (perf_event_paranoid?); counters skipped"; PERF=0; }
fi

# ------------------------------------------------------ 1-3. build and check
for comp in $COMPILERS; do
  for cfg in $CONFIGS; do
    tag="${comp}_${cfg}"; B=$(bdir "$comp" "$cfg"); D="$OUT/$tag"; mkdir -p "$D"
    log "===== $tag ====="
    if [ "$SKIP_BUILD" != 1 ]; then
      # shellcheck disable=SC2046,SC2086
      if ! cmake -S "$ROOT" -B "$B" -DCMAKE_BUILD_TYPE=Release -DCOMPILER="$comp" \
            -DTARGET_MACHINE="$MACHINE" -DTARGET_ARCH= -DDATADIR="$DATADIR" \
            $(config_flags "$cfg") $TUNED > "$D/cmake.log" 2>&1; then
        fail "$tag cmake (see $D/cmake.log)"; continue
      fi
      : > "$D/build.log"
      for tgt in run_tpch run_selbench test_all; do
        echo "### target $tgt" >> "$D/build.log"
        cmake --build "$B" -j "$JOBS" --target "$tgt" >> "$D/build.log" 2>&1 || fail "$tag build of $tgt (see $D/build.log)"
      done
      grep -E "not available, using" "$D/build.log" && fail "$tag: ISA fallback pragma fired"
    fi
    # preset arch (d2daceb): -DTARGET_ARCH= means "use the TARGET_MACHINE preset"
    grep -E "TARGET_ARCH=|WIDTH=" "$D/cmake.log" | sed 's/^-- /  /' | tee -a "$OUT/driver.log"

    # codegen: zmm / ymm compress inside the redirected contiguous kernels
    sel_o=$(find "$B" -name 'Selection.cpp.o' | head -1)
    if [ -z "$sel_o" ]; then fail "$tag: Selection.cpp.o not found"; else
      objdump -d --no-show-raw-insn -C "$sel_o" | awk '
        /^[0-9a-f]+ <.*simd::sel_col_(val|col)</ { f = 1; n++; next }
        /^[0-9a-f]+ </ { f = 0 }
        f && /zmm/ { z++ } f && /vpcompressd.*ymm/ { y++ }
        END { printf "%d %d %d\n", n, z, y }' > "$D/codegen.txt"
      read -r nfn nzmm nymm < "$D/codegen.txt"
      log "$tag codegen: kernels=$nfn zmm=$nzmm ymm_vpcompressd=$nymm"
      case "$cfg" in
        base) [ "$nfn" -eq 0 ] || fail "$tag: SIMD kernels present with VW_SIMD_SEL=OFF" ;;
        w512) { [ "$nfn" -gt 0 ] && [ "$nzmm" -gt 0 ]; } || fail "$tag: expected zmm kernels" ;;
        w256*) { [ "$nfn" -gt 0 ] && [ "$nzmm" -eq 0 ] && [ "$nymm" -gt 0 ]; } || fail "$tag: expected ymm-only kernels" ;;
      esac
    fi

    # unit + TPC-H correctness
    if [ -x "$B/test_all" ]; then
      if threads=$TEST_THREADS tlimit "$B/test_all" --gtest_filter='SimdSel*' > "$D/unit.log" 2>&1; then
        log "$tag unit: $(grep -E '^\[  PASSED  \]' "$D/unit.log")"
      else fail "$tag unit tests (see $D/unit.log)"; fi
      if [ -d "$DATADIR/tpch/sf1" ]; then
        if threads=$TEST_THREADS tlimit "$B/test_all" --gtest_filter='TPCH.*' > "$D/tpch_test.log" 2>&1; then
          log "$tag TPC-H: $(grep -E '^\[  PASSED  \]' "$D/tpch_test.log")"
        else fail "$tag TPC-H tests (see $D/tpch_test.log)"; fi
      else log "$tag: no $DATADIR/tpch/sf1, TPC-H tests skipped"; fi
    else fail "$tag: test_all not built"; fi
  done
done

# ---------------------------------------------------------- 4. microbenchmark
# every kernel variant is compiled into run_selbench; the w256 build also has
# -mprefer-vector-width=256 on it, so its simd256_* rows are the real ymm code
for comp in $COMPILERS; do
  B=$(bdir "$comp" w256)
  [ -x "$B/run_selbench" ] || continue
  log "$comp run_selbench (w256 build)"
  # shellcheck disable=SC2086
  if tlimit $NUMA "$B/run_selbench" > "$OUT/selbench_$comp.raw.csv" 2> "$OUT/selbench_$comp.err"; then
    head -1 "$OUT/selbench_$comp.raw.csv" > "$OUT/selbench_$comp.csv"
    grep -E '^(sel_col_val|sel_col_col),(scalar_bf|avx512_old|simd_mem_u2|simd_reg_u1|simd256_reg|simd256_mem),' \
      "$OUT/selbench_$comp.raw.csv" >> "$OUT/selbench_$comp.csv"
  else fail "$comp run_selbench"; fi
done

# ------------------------------------------------------------ 5. end to end
echo "compiler,config,round,query,median_ms,min_ms" > "$OUT/timing.csv"
echo "compiler,config,round,query,event,count" > "$OUT/perf.csv"
if [ ! -d "$TPCH_PATH" ]; then
  log "no TPCH_PATH=$TPCH_PATH, timing skipped"
else
  for q in ${QUERIES//,/ }; do
    for r in $(seq 1 "$ROUNDS"); do
      for comp in $COMPILERS; do
        for cfg in $CONFIGS; do
          B=$(bdir "$comp" "$cfg"); [ -x "$B/run_tpch" ] || continue
          f="$OUT/${comp}_${cfg}/q${q}_r$r"
          perfcmd=()
          # shellcheck disable=SC2054  # "-x," is perf's CSV separator flag
          [ "$PERF" = 1 ] && perfcmd=(perf stat -x, -e "$EVENTS" -o "$f.perf")
          # shellcheck disable=SC2086
          if ! tlimit "${perfcmd[@]}" $NUMA "$B/run_tpch" -p "$TPCH_PATH" -e v -q "$q" -r "$REPS" -t 1 -s "$SETTLE" \
                > "$f.csv" 2> "$f.err"; then
            fail "${comp}_${cfg} q$q round $r (see $f.err)"; continue
          fi
          awk -F, -v c="$comp" -v g="$cfg" -v r="$r" '
            { l = $1; gsub(/^ +| +$/, "", l) }
            l ~ /^q[0-9]+ v/ { m = $2; n = $4; gsub(/ /, "", m); gsub(/ /, "", n); split(l, p, " ");
                              print c "," g "," r "," p[1] "," m "," n }' "$f.csv" >> "$OUT/timing.csv"
          [ -f "$f.perf" ] && awk -F, -v c="$comp" -v g="$cfg" -v r="$r" -v q="q$q" \
            '$1 ~ /^[0-9]+$/ { ev = $3; sub(/:[ukh]+$/, "", ev); print c "," g "," r "," q "," ev "," $1 }' "$f.perf" >> "$OUT/perf.csv"
        done
      done
    done
    log "q$q done"
  done
fi

# ------------------------------------------------------------------ summaries
# speedup = base median / config median (same compiler, query), where median
# is over rounds of the per-invocation medians, so one disturbed invocation
# does not skew it. Headers are printed outside sort so they stay on line 1.
{ echo "compiler,config,query,median_ms,best_ms,speedup_vs_base"
  awk -F, '
    function median(str,   a, n, i, j, t) {
      n = split(str, a, " ")
      for (i = 2; i <= n; i++) { t = a[i] + 0; for (j = i - 1; j >= 1 && a[j] + 0 > t; j--) a[j + 1] = a[j]; a[j + 1] = t }
      return (n % 2) ? a[(n + 1) / 2] : (a[n / 2] + a[n / 2 + 1]) / 2
    }
    NR > 1 { k = $1 "," $2 "," $4; v[k] = v[k] " " $5; if (!(k in mn) || $6 < mn[k]) mn[k] = $6 }
    END { for (k in v) m[k] = median(v[k])
          for (k in m) { split(k, p, ","); b = p[1] ",base," p[3]
            printf "%s,%.2f,%.2f,%s\n", k, m[k], mn[k], (b in m) ? sprintf("%.3f", m[b] / m[k]) : "" } }' \
    "$OUT/timing.csv" | sort -t, -k1,1 -k3,3 -k2,2; } > "$OUT/speedup.csv"

# license.csv: effective clock ratio (cycles/ref-cycles) and share of cycles in
# AVX license level 1 (AVX2 heavy / AVX-512 light) and level 2 (AVX-512 heavy).
# Counts cover the whole run_tpch invocation including data load; the load
# phase is scalar, so lvl1/lvl2 cycles come from the queries.
{ echo "compiler,config,query,cycles_per_refcycle,lvl1_share_pct,lvl2_share_pct"
awk -F, 'NR > 1 { k = $1 "," $2 "," $4; v[k "," $5] += $6; keys[k] = 1 }
  END { for (k in keys) { c = v[k ",cycles"]; rc = v[k ",ref-cycles"]
          l1 = v[k ",core_power.lvl1_turbo_license"]; l2 = v[k ",core_power.lvl2_turbo_license"]
          printf "%s,%s,%s,%s\n", k, rc ? sprintf("%.3f", c / rc) : "",
                 c ? sprintf("%.2f", 100 * l1 / c) : "", c ? sprintf("%.2f", 100 * l2 / c) : "" } }' \
  "$OUT/perf.csv" | sort -t, -k1,1 -k3,3 -k2,2; } > "$OUT/license.csv"

log "speedup vs base (median over rounds of per-invocation medians, t=1):"
column -t -s, "$OUT/speedup.csv" | tee -a "$OUT/driver.log"
if [ "$PERF" = 1 ]; then
  log "clock / AVX license:"
  column -t -s, "$OUT/license.csv" | tee -a "$OUT/driver.log"
fi
log "results: $OUT"
if [ "$FAILS" -gt 0 ]; then log "$FAILS FAILURE(S)"; exit 1; fi
log "ALL CHECKS PASSED"
