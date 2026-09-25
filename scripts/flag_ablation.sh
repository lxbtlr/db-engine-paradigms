#!/usr/bin/env bash
# Per-flag ablation of the Q1-tuned baseline: which of its flags help or hurt
# each query? The tuned set was chosen for Q1 and costs Q18 ~20%, so this
# measures every flag on its own in both directions:
#
#   default                  none of the tuned flags
#   add_<flag>               default + that flag
#   tuned                    all tuned flags
#   drop_<flag>              tuned - that flag
#
# Flags: group_aggr (VW_GROUP_AGGR), group_aggr_sel (VW_GROUP_AGGR_SEL, only
# active with VW_GROUP_AGGR, so add_group_aggr_sel = +both and
# drop_group_aggr = -both), pos16 (VW_POS_16), crc32 (VW_USE_CRC32),
# huge2mb (HUGE_2MB_MALLOC_HUGE). SEL (default: 256-bit SIMD selection) is
# the same in every config.
#
# For every compiler x config it builds run_tpch + test_all into
# build_flags/<compiler>_<config>, runs the TPC-H correctness tests, then
# times run_tpch -e v on QUERIES at 1 thread pinned to one CPU, ROUNDS rounds
# with all builds alternating in each round.
# Output: results/flag_ablation_<timestamp>/
#   timing.csv   one row per run_tpch invocation
#   matrix.csv   mean-of-medians ms per compiler x query x config
#   effects.csv  per flag: add_speedup = default/add_<flag>, drop_speedup =
#                drop_<flag>/tuned (both > 1 means the flag helps that query)
#   best.csv     fastest config per compiler x query
#
# Environment (all optional):
#   COMPILERS  "gcc clang"
#   CONFIGS    all 12 (see config_flags)
#   SEL        "-DVW_SIMD_SEL=ON -DVW_SIMD_SEL_WIDTH=256 -DVW_SIMD_SEL_COMPRESS=reg"
#   MACHINE    dubliner
#   DATADIR    /tank/alexb/swole/          test_all reads $DATADIR/tpch/sf1/
#   TPCH_PATH  /tank/alexb/swole/tpch/sf1  run_tpch -p
#   QUERIES    1,3,6,9,18
#   REPS       30        run_tpch -r per invocation
#   ROUNDS     3
#   SETTLE     2         run_tpch -s
#   CPU        0
#   NUMA       "numactl --physcpubind=$CPU --membind=0"  ("" = none)
#   TEST_THREADS 4       worker threads for test_all
#   JOBS       $(nproc)
#   SKIP_BUILD 0         1 = reuse build dirs
#   TIMEOUT    1800      seconds per step (0 = none)
set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPILERS=${COMPILERS:-"gcc clang"}
ALL_CONFIGS="default add_group_aggr add_group_aggr_sel add_pos16 add_crc32 add_huge2mb
             tuned drop_group_aggr drop_group_aggr_sel drop_pos16 drop_crc32 drop_huge2mb"
CONFIGS=${CONFIGS:-$ALL_CONFIGS}
SEL=${SEL:-"-DVW_SIMD_SEL=ON -DVW_SIMD_SEL_WIDTH=256 -DVW_SIMD_SEL_COMPRESS=reg"}
MACHINE=${MACHINE:-dubliner}
DATADIR=${DATADIR:-/tank/alexb/swole/}
TPCH_PATH=${TPCH_PATH:-/tank/alexb/swole/tpch/sf1}
QUERIES=${QUERIES:-1,3,6,9,18}
REPS=${REPS:-30}
ROUNDS=${ROUNDS:-3}
SETTLE=${SETTLE:-2}
CPU=${CPU:-0}
NUMA=${NUMA-"numactl --physcpubind=$CPU --membind=0"}
TEST_THREADS=${TEST_THREADS:-4}
JOBS=${JOBS:-$(nproc)}
SKIP_BUILD=${SKIP_BUILD:-0}
TIMEOUT=${TIMEOUT:-1800}

OUT="$ROOT/results/flag_ablation_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
FAILS=0
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/driver.log"; }
fail() { log "FAIL: $*"; FAILS=$((FAILS + 1)); }
tlimit() { if [ "$TIMEOUT" -gt 0 ]; then timeout --kill-after=30 "$TIMEOUT" "$@"; else "$@"; fi; }

# flags <group_aggr> <group_aggr_sel> <pos16> <crc32> <huge2mb>
# Every option is passed explicitly (CMake caches them), including the other
# kernel options, so nothing leaks in from an earlier configure.
flags() {
  echo "-DVW_GROUP_AGGR=$1 -DVW_GROUP_AGGR_SEL=$2 -DVW_POS_16=$3 -DVW_USE_CRC32=$4 -DHUGE_2MB_MALLOC_HUGE=$5" \
       "-DVW_SIMD_SEL=OFF -DVW_SIMD_SEL_WIDTH=512 -DVW_SIMD_SEL_COMPRESS=reg -DVW_SIMD_SEL_UNROLL=1" \
       "-DVW_SIMD_SEL_GATHER=scalar -DVW_SIMD_SEL_CHAR=OFF -DVW_SIMD_HASH=OFF -DVW_JOIN_PREFETCH=OFF" \
       "$SEL" # last -D wins, so SEL overrides the SIMD_SEL defaults above
}
config_flags() {
  case "$1" in
    #                         aggr aggr_sel pos16 crc32 huge2mb
    default)             flags OFF  OFF      OFF   OFF   OFF ;;
    add_group_aggr)      flags ON   OFF      OFF   OFF   OFF ;;
    add_group_aggr_sel)  flags ON   ON       OFF   OFF   OFF ;;
    add_pos16)           flags OFF  OFF      ON    OFF   OFF ;;
    add_crc32)           flags OFF  OFF      OFF   ON    OFF ;;
    add_huge2mb)         flags OFF  OFF      OFF   OFF   ON  ;;
    tuned)               flags ON   ON       ON    ON    ON  ;;
    drop_group_aggr)     flags OFF  OFF      ON    ON    ON  ;;
    drop_group_aggr_sel) flags ON   OFF      ON    ON    ON  ;;
    drop_pos16)          flags ON   ON       OFF   ON    ON  ;;
    drop_crc32)          flags ON   ON       ON    OFF   ON  ;;
    drop_huge2mb)        flags ON   ON       ON    ON    OFF ;;
    *) echo "unknown config $1" >&2; exit 2 ;;
  esac
}
bdir() { echo "$ROOT/build_flags/$1_$2"; }

{
  echo "host: $(hostname)"
  echo "date: $(date -Is)"
  echo "git:  $(git -C "$ROOT" rev-parse --short HEAD) $(git -C "$ROOT" status --porcelain --untracked-files=no | wc -l) dirty tracked files"
  lscpu | grep -E 'Model name|Socket|Core|Thread|NUMA node\(s\)|MHz' || true
  echo -n "governor cpu$CPU: "; cat "/sys/devices/system/cpu/cpu$CPU/cpufreq/scaling_governor" 2>/dev/null || echo n/a
  echo "SEL: $SEL"
  echo "QUERIES: $QUERIES  REPS: $REPS  ROUNDS: $ROUNDS  CPU: $CPU  NUMA: $NUMA"
} | tee "$OUT/machine.txt"

# ------------------------------------------------------- build + correctness
for comp in $COMPILERS; do
  for cfg in $CONFIGS; do
    tag="${comp}_${cfg}"; B=$(bdir "$comp" "$cfg"); D="$OUT/$tag"; mkdir -p "$D"
    log "===== $tag ====="
    if [ "$SKIP_BUILD" != 1 ]; then
      # shellcheck disable=SC2046
      if ! cmake -S "$ROOT" -B "$B" -DCMAKE_BUILD_TYPE=Release -DCOMPILER="$comp" \
            -DTARGET_MACHINE="$MACHINE" -DTARGET_ARCH= -DDATADIR="$DATADIR" \
            $(config_flags "$cfg") > "$D/cmake.log" 2>&1; then
        fail "$tag cmake (see $D/cmake.log)"; continue
      fi
      : > "$D/build.log"
      for tgt in run_tpch test_all; do
        echo "### target $tgt" >> "$D/build.log"
        cmake --build "$B" -j "$JOBS" --target "$tgt" >> "$D/build.log" 2>&1 || fail "$tag build of $tgt (see $D/build.log)"
      done
    fi
    # the effective defines, to confirm each config is what it claims
    cat "$B/build.ninja" "$B/CMakeFiles/vectorwise.dir/flags.make" 2>/dev/null | grep -o -E -- '-D(VW_[A-Z0-9_]+|HUGE_2MB_MALLOC_HUGE)(=[^ ]*)?' | sort -u | tr '\n' ' ' > "$D/defines.txt"
    log "$tag defines: $(cat "$D/defines.txt")"
    if [ -x "$B/test_all" ] && [ -d "$DATADIR/tpch/sf1" ]; then
      if threads=$TEST_THREADS tlimit "$B/test_all" --gtest_filter='TPCH.*' > "$D/tpch_test.log" 2>&1; then
        log "$tag TPC-H: $(grep -E '^\[  PASSED  \]' "$D/tpch_test.log")"
      else fail "$tag TPC-H tests (see $D/tpch_test.log)"; fi
    elif [ ! -x "$B/test_all" ]; then fail "$tag: test_all not built"
    else log "$tag: no $DATADIR/tpch/sf1, TPC-H tests skipped"; fi
  done
done

# ------------------------------------------------------------------ timing
echo "compiler,config,round,query,median_ms,min_ms" > "$OUT/timing.csv"
if [ ! -d "$TPCH_PATH" ]; then
  log "no TPCH_PATH=$TPCH_PATH, timing skipped"
else
  for q in ${QUERIES//,/ }; do
    for r in $(seq 1 "$ROUNDS"); do
      for comp in $COMPILERS; do
        for cfg in $CONFIGS; do
          B=$(bdir "$comp" "$cfg"); [ -x "$B/run_tpch" ] || continue
          f="$OUT/${comp}_${cfg}/q${q}_r$r"
          # shellcheck disable=SC2086
          if ! tlimit $NUMA "$B/run_tpch" -p "$TPCH_PATH" -e v -q "$q" -r "$REPS" -t 1 -s "$SETTLE" \
                > "$f.csv" 2> "$f.err"; then
            fail "${comp}_${cfg} q$q round $r (see $f.err)"; continue
          fi
          awk -F, -v c="$comp" -v g="$cfg" -v r="$r" '
            { l = $1; gsub(/^ +| +$/, "", l) }
            l ~ /^q[0-9]+ v/ { m = $2; n = $4; gsub(/ /, "", m); gsub(/ /, "", n); split(l, p, " ");
                              print c "," g "," r "," p[1] "," m "," n }' "$f.csv" >> "$OUT/timing.csv"
        done
      done
    done
    log "q$q done"
  done
fi

# ---------------------------------------------------------------- summaries
# matrix.csv: compiler,query,config,mean_median_ms,best_ms,speedup_vs_default,speedup_vs_tuned
awk -F, 'NR > 1 { k = $1 "," $4 "," $2; s[k] += $5; n[k]++; if (!(k in mn) || $6 < mn[k]) mn[k] = $6 }
  END { for (k in s) m[k] = s[k] / n[k]
        print "compiler,query,config,mean_median_ms,best_ms,speedup_vs_default,speedup_vs_tuned"
        for (k in m) { split(k, p, ","); d = p[1] "," p[2] ",default"; t = p[1] "," p[2] ",tuned"
          printf "%s,%.2f,%.2f,%s,%s\n", k, m[k], mn[k],
                 (d in m) ? sprintf("%.3f", m[d] / m[k]) : "", (t in m) ? sprintf("%.3f", m[t] / m[k]) : "" } }' \
  "$OUT/timing.csv" | sort -t, -k1,1 -k2,2V -k3,3 > "$OUT/matrix.csv"

# effects.csv: add_speedup = default / add_<flag>, drop_speedup = drop_<flag> / tuned
awk -F, 'NR > 1 { m[$1 "," $2 "," $3] = $4; cq[$1 "," $2] = 1 }
  END { split("group_aggr group_aggr_sel pos16 crc32 huge2mb", fl, " ")
        print "compiler,query,flag,add_speedup,drop_speedup"
        for (k in cq) for (i = 1; i <= 5; i++) { f = fl[i]
          d = m[k ",default"]; a = m[k ",add_" f]; t = m[k ",tuned"]; x = m[k ",drop_" f]
          printf "%s,%s,%s,%s\n", k, f, (d && a) ? sprintf("%.3f", d / a) : "", (t && x) ? sprintf("%.3f", x / t) : "" } }' \
  "$OUT/matrix.csv" | sort -t, -k1,1 -k2,2V > "$OUT/effects.csv"

# best.csv: fastest config per compiler x query (by mean of medians)
awk -F, 'NR > 1 { k = $1 "," $2; if (!(k in b) || $4 < b[k]) { b[k] = $4; c[k] = $3 } }
  END { print "compiler,query,best_config,mean_median_ms"; for (k in b) printf "%s,%s,%.2f\n", k, c[k], b[k] }' \
  "$OUT/matrix.csv" | sort -t, -k1,1 -k2,2V > "$OUT/best.csv"

log "per-flag effect (>1 = flag helps; add = vs default, drop = vs tuned):"
column -t -s, "$OUT/effects.csv" | tee -a "$OUT/driver.log"
log "best config per query:"
column -t -s, "$OUT/best.csv" | tee -a "$OUT/driver.log"
log "results: $OUT"
if [ "$FAILS" -gt 0 ]; then log "$FAILS FAILURE(S)"; exit 1; fi
log "ALL CHECKS PASSED"
