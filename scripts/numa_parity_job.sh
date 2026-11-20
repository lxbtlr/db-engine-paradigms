#!/usr/bin/env bash
# Builds run_tpch in NUMA placement configs and runs Typer (-e h) and/or
# Tectorwise (-e v) on this host, for the Typer/Tectorwise NUMA parity work
# (branch numa-typer-parity).
#
# Configs (CONFIGS, space separated):
#   base_spread    no NUMA option, threads spread round-robin over regions
#   base_packed    THREAD_PIN_PACKED
#   shard_spread   NUMA_SHARD
#   shard_packed   NUMA_SHARD + THREAD_PIN_PACKED
# DEBUG=ON adds NUMA_DEBUG to every config (NUMA trace, placement checks;
# not for timing).
#
# Environment (defaults in brackets):
#   COMPILER  gcc | clang                                   [gcc]
#   CONFIGS   see above                                     [shard_spread]
#   ENGINES   h, v or h,v (one run_tpch invocation each)    [h]
#   QUERIES   run_tpch -q                                   [1,3,9,18]
#   THREADS   run_tpch -t, comma list                       [1]
#   SF        data /tank/alexb/swole/tpch/sf<SF>            [10]
#   REPS      run_tpch -r                                   [5]
#   SETTLE    run_tpch -s                                   [10]
#   DEBUG     ON | OFF                                      [OFF]
#   HUGE      ON | OFF, HUGE_2MB_MALLOC_HUGE                [ON]
#   TIMEOUT   seconds per run_tpch invocation               [3600]
#
# Output: results/numa_parity_<host>_<stamp>/
#   machine.txt               host, topology, numa_balancing, git
#   <config>/build.log
#   <config>_e<engine>.log    run_tpch stdout+stderr
#   timing.csv                config,engine,query,threads,rep,ms
#   nt.txt                    NUMA_DEBUG trace lines ([NT] ...), prefixed
#                             with config and engine
set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_HOST=${BENCH_HOST:-$(hostname -s 2>/dev/null || hostname)}
COMPILER=${COMPILER:-gcc}
CONFIGS=${CONFIGS:-shard_spread}
ENGINES=${ENGINES:-h}
QUERIES=${QUERIES:-1,3,9,18}
THREADS=${THREADS:-1}
SF=${SF:-10}
REPS=${REPS:-5}
SETTLE=${SETTLE:-10}
DEBUG=${DEBUG:-OFF}
HUGE=${HUGE:-ON}
TIMEOUT=${TIMEOUT:-3600}
TPCH_PATH=${TPCH_PATH:-/tank/alexb/swole/tpch/sf$SF}
case "$BENCH_HOST" in
  dubliner|roquefort|manchego|burrata|kafir|rpi5) MACHINE=$BENCH_HOST ;;
  *) MACHINE=custom ;;
esac

config_flags() {
  case "$1" in
    base_spread)  echo "" ;;
    base_packed)  echo "-DTHREAD_PIN_PACKED=ON" ;;
    shard_spread) echo "-DNUMA_SHARD=ON" ;;
    shard_packed) echo "-DNUMA_SHARD=ON -DTHREAD_PIN_PACKED=ON" ;;
    *) return 1 ;;
  esac
}
for c in $CONFIGS; do
  config_flags "$c" > /dev/null || { echo "unknown config $c" >&2; exit 2; }
done

OUT="$ROOT/results/numa_parity_${BENCH_HOST}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
{
  echo "host: $(hostname)  MACHINE: $MACHINE  COMPILER: $COMPILER"
  echo "date: $(date -Is)"
  echo "git:  $(git -C "$ROOT" rev-parse --short HEAD)"
  echo "CONFIGS: $CONFIGS  ENGINES: $ENGINES  QUERIES: $QUERIES  THREADS: $THREADS"
  echo "SF: $SF  REPS: $REPS  SETTLE: $SETTLE  DEBUG: $DEBUG  HUGE: $HUGE"
  echo "numa_balancing: $(cat /proc/sys/kernel/numa_balancing 2>/dev/null)"
  lscpu | grep -E 'Model name|^CPU\(s\)|Socket|NUMA node' || true
  grep -E 'HugePages_(Total|Free)' /proc/meminfo || true
} | tee "$OUT/machine.txt"

echo "config,engine,query,threads,rep,ms" > "$OUT/timing.csv"
: > "$OUT/nt.txt"
status=0
for c in $CONFIGS; do
  B="$ROOT/build_flags/$BENCH_HOST/numa_${COMPILER}_${c}_dbg${DEBUG}_huge${HUGE}"
  mkdir -p "$OUT/$c"
  # shellcheck disable=SC2046
  if ! { cmake -S "$ROOT" -B "$B" -DCMAKE_BUILD_TYPE=Release \
           -DCOMPILER="$COMPILER" -DTARGET_MACHINE="$MACHINE" \
           -DNUMA_DEBUG="$DEBUG" -DHUGE_2MB_MALLOC_HUGE="$HUGE" \
           $(config_flags "$c") > "$OUT/$c/build.log" 2>&1 &&
         cmake --build "$B" -j "$(nproc)" --target run_tpch \
           >> "$OUT/$c/build.log" 2>&1; }; then
    echo "build $c failed (see $c/build.log)" >&2
    tail -30 "$OUT/$c/build.log" >&2
    status=1
    continue
  fi
  for e in ${ENGINES//,/ }; do
    log="$OUT/${c}_e$e.log"
    echo "== $c -e $e"
    timeout "$TIMEOUT" "$B/run_tpch" -p "$TPCH_PATH/" -e "$e" -q "$QUERIES" \
      -t "$THREADS" -r "$REPS" -s "$SETTLE" > "$log" 2>&1
    rc=$?
    [ $rc -eq 0 ] || { echo "run_tpch $c -e $e exited $rc" >&2; status=1; }
    grep -m1 'malloc_huge: MAP_HUGETLB failed' "$log" >&2 || true
    # "per-rep q18 h t88 : 12.3 45.6 ..."
    awk -v c="$c" -v e="$e" '/^per-rep /{
        split($0, a, ":"); h = a[1]; sub(/^per-rep /, "", h)
        n = split(h, f, " "); q = f[1]; t = f[n]; sub(/^t/, "", t)
        m = split(a[2], ms, " ")
        for (i = 1; i <= m; i++) printf "%s,%s,%s,%s,%d,%s\n", c, e, q, t, i, ms[i]
      }' "$log" >> "$OUT/timing.csv"
    grep '^\[NT\]' "$log" | sed "s/^/$c e=$e /" >> "$OUT/nt.txt"
  done
done
echo "$(($(wc -l < "$OUT/timing.csv") - 1)) timings, $(wc -l < "$OUT/nt.txt") trace lines in $OUT"
exit $status
