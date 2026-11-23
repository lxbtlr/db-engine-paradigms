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
#   ENGINES   h, v, h,v (one invocation each) or hv (both  [h]
#             engines in one invocation, as run_tpch runs)
#   QUERIES   run_tpch -q                                   [1,3,9,18]
#   THREADS   run_tpch -t, comma list, one run per count    [1]
#   SF        data /tank/alexb/swole/tpch/sf<SF>            [10]
#   REPS      run_tpch -r                                   [5]
#   SETTLE    run_tpch -s                                   [10]
#   DEBUG     ON | OFF                                      [OFF]
#   HUGE      ON | OFF, HUGE_2MB_MALLOC_HUGE                [ON]
#   TIMEOUT   seconds per run_tpch invocation               [3600]
#   DIGEST    ON: one extra untimed run per query and thread
#             count, printing an order-independent result
#             digest (run_tpch env digest=1)                [OFF]
#   BEFORE    commit sha: also build and run every config from
#             that commit's source, configs suffixed _before []
#
# Output: results/numa_parity_<host>_<stamp>/
#   machine.txt               host, topology, memory, numa_balancing, git
#   <config>/build.log
#   <config>_e<engine>_t<threads>.log  run_tpch stdout+stderr
#   timing.csv                config,engine,query,threads,rep,ms
#   nt.txt                    NUMA_DEBUG trace lines ([NT] ...), prefixed
#                             with config and engine
#   results.txt               result digests ([RESULT] ...), prefixed
#                             with config and engine
#   hugepages.csv             2 MB hugetlb pages per node and invocation:
#                             free before the run and the lowest free count
#                             sampled while it ran (used = before - min)
#   failures.txt              failed builds, run_tpch exits, and runs
#                             whose log has a "MAP_HUGETLB failed" line
#                             (malloc_huge / malloc_numa fell back to base
#                             pages); empty when everything ran cleanly
# Exits 1 only when no timing was recorded, so the bridge still copies the
# results of a partly failed job.
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
DIGEST=${DIGEST:-OFF}
BEFORE=${BEFORE:-}
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
  echo "SF: $SF  REPS: $REPS  SETTLE: $SETTLE  DEBUG: $DEBUG  HUGE: $HUGE  DIGEST: $DIGEST"
  echo "numa_balancing: $(cat /proc/sys/kernel/numa_balancing 2>/dev/null)"
  echo "BEFORE: ${BEFORE:-none}"
  free -g | head -2 || true
  lscpu | grep -E 'Model name|^CPU\(s\)|Socket|NUMA node' || true
  grep -E 'HugePages_(Total|Free)' /proc/meminfo || true
} | tee "$OUT/machine.txt"

HP_FILES=(/sys/devices/system/node/node*/hugepages/hugepages-2048kB/free_hugepages)
# free 2 MB hugetlb pages per node, space separated (read builtin: no forks
# while sampling)
hp_free() {
  local f v out=()
  for f in "${HP_FILES[@]}"; do read -r v < "$f" && out+=("$v"); done
  echo "${out[*]}"
}
# hp_sample FILE: until killed, write the per-node minimum of hp_free to FILE
hp_sample() {
  local min cur i
  read -r -a min <<< "$(hp_free)"
  while :; do
    read -r -a cur <<< "$(hp_free)"
    for i in "${!cur[@]}"; do (( cur[i] < min[i] )) && min[i]=${cur[i]}; done
    # rename, so a reader never sees a truncated file when the sampler is
    # killed mid-write
    echo "${min[*]}" > "$1.tmp" && mv -f "$1.tmp" "$1"
    sleep 0.2
  done
}

echo "config,engine,query,threads,rep,ms" > "$OUT/timing.csv"
echo "config,engine,threads,node,free_before,free_min,used_max" > "$OUT/hugepages.csv"
: > "$OUT/nt.txt"
: > "$OUT/results.txt"
: > "$OUT/failures.txt"
fail() { echo "$*" | tee -a "$OUT/failures.txt" >&2; }
[ "$DIGEST" = ON ] && export digest=1

# Sources: the checked-out commit, and with BEFORE=<sha> that commit's source
# too (git archive, simde copied from this checkout), built and run with the
# same harness; its configs get the suffix _before.
SOURCES="after"
if [ -n "$BEFORE" ]; then
  SRC_BEFORE="$ROOT/build_flags/$BENCH_HOST/src_$BEFORE"
  if [ ! -f "$SRC_BEFORE/CMakeLists.txt" ]; then
    rm -rf "$SRC_BEFORE" && mkdir -p "$SRC_BEFORE" &&
      git -C "$ROOT" archive "$BEFORE" | tar -x -C "$SRC_BEFORE" &&
      cp -a "$ROOT/3rdparty/simde/." "$SRC_BEFORE/3rdparty/simde/" ||
      { echo "cannot export $BEFORE" >&2; exit 1; }
  fi
  # the hugetlb fallback warning (Memory.hpp) in the reference build too;
  # Memory.hpp is otherwise unchanged since 98f4a54
  cp "$ROOT/include/common/runtime/Memory.hpp" "$SRC_BEFORE/include/common/runtime/Memory.hpp"
  echo "BEFORE source: Memory.hpp from $(git -C "$ROOT" rev-parse --short HEAD)" >> "$OUT/machine.txt"
  SOURCES="after before"
fi

for src in $SOURCES; do
for c in $CONFIGS; do
  if [ "$src" = before ]; then
    S="$SRC_BEFORE"; name="${c}_before"; tag="_ref$BEFORE"
  else
    S="$ROOT"; name="$c"; tag=""
  fi
  B="$ROOT/build_flags/$BENCH_HOST/numa_${COMPILER}_${c}_dbg${DEBUG}_huge${HUGE}$tag"
  mkdir -p "$OUT/$name"
  # Configured twice: CMakeLists.txt rebuilds CMAKE_CXX_FLAGS_RELEASE from
  # itself, so the first configure of a build dir drops -DNDEBUG.
  # shellcheck disable=SC2046
  if ! { cmake -S "$S" -B "$B" -DCMAKE_BUILD_TYPE=Release \
           -DCOMPILER="$COMPILER" -DTARGET_MACHINE="$MACHINE" \
           -DNUMA_DEBUG="$DEBUG" -DHUGE_2MB_MALLOC_HUGE="$HUGE" \
           $(config_flags "$c") > "$OUT/$name/build.log" 2>&1 &&
         cmake "$B" >> "$OUT/$name/build.log" 2>&1 &&
         cmake --build "$B" -j "$(nproc)" --target run_tpch \
           >> "$OUT/$name/build.log" 2>&1; }; then
    fail "build $name failed (see $name/build.log)"
    tail -30 "$OUT/$name/build.log" >&2
    continue
  fi
  grep -qs -- -DNDEBUG "$B/CMakeFiles/tpch.dir/flags.make" "$B/build.ninja" &&
    echo "$name: NDEBUG" >> "$OUT/machine.txt" ||
    echo "$name: asserts ON (no -DNDEBUG)" | tee -a "$OUT/machine.txt" >&2
  for e in ${ENGINES//,/ }; do
  # hv: both engines in one invocation (no -e; Tectorwise runs first)
  eflag=(-e "$e"); [ "$e" = hv ] && eflag=()
  # one invocation per thread count: run_tpch shards over the regions the
  # largest -t count occupies, so a list would leave smaller packed counts
  # with shards on regions that have no threads
  for t in ${THREADS//,/ }; do
    log="$OUT/${name}_e${e}_t$t.log"
    echo "== $name -e $e -t $t"
    hp_before=$(hp_free)
    hp_min="$OUT/.hp_min"; echo "$hp_before" > "$hp_min"
    hp_sample "$hp_min" & hp_pid=$!
    timeout "$TIMEOUT" "$B/run_tpch" -p "$TPCH_PATH/" "${eflag[@]}" \
      -q "$QUERIES" -t "$t" -r "$REPS" -s "$SETTLE" > "$log" 2>&1
    rc=$?
    kill "$hp_pid" 2> /dev/null; wait "$hp_pid" 2> /dev/null
    read -r -a hb <<< "$hp_before"; read -r -a hm < "$hp_min"
    for i in "${!hb[@]}"; do hm[i]=${hm[i]:-${hb[i]}}; done
    for i in "${!hb[@]}"; do
      echo "$name,$e,$t,$i,${hb[i]},${hm[i]},$((hb[i] - hm[i]))" >> "$OUT/hugepages.csv"
    done
    echo "   hugepages used per node: $(for i in "${!hb[@]}"; do printf '%s ' $((hb[i] - hm[i])); done)"
    [ $rc -eq 0 ] || fail "run_tpch $name -e $e -t $t exited $rc"
    hl=$(grep -m1 "MAP_HUGETLB failed" "$log") &&
      fail "run_tpch $name -e $e -t $t fell back to base pages: $hl"
    # "per-rep q18 h t88 : 12.3 45.6 ..." (engine from the label)
    awk -v c="$name" '/^per-rep /{
        split($0, a, ":"); h = a[1]; sub(/^per-rep /, "", h)
        n = split(h, f, " "); q = f[1]; e = f[2]; t = f[n]; sub(/^t/, "", t)
        m = split(a[2], ms, " ")
        for (i = 1; i <= m; i++) printf "%s,%s,%s,%s,%d,%s\n", c, e, q, t, i, ms[i]
      }' "$log" >> "$OUT/timing.csv"
    grep '^\[NT\]' "$log" | sed "s/^/$name e=$e /" >> "$OUT/nt.txt"
    grep '^\[RESULT\]' "$log" | sed "s/^/$name e=$e /" >> "$OUT/results.txt"
  done
  done
done
done
echo "$(($(wc -l < "$OUT/timing.csv") - 1)) timings, $(wc -l < "$OUT/nt.txt") trace lines, $(wc -l < "$OUT/results.txt") digests in $OUT"
[ -s "$OUT/failures.txt" ] && echo "$(wc -l < "$OUT/failures.txt") failures, see failures.txt" >&2
[ "$(wc -l < "$OUT/timing.csv")" -gt 1 ]
