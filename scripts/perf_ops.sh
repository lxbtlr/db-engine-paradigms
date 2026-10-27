#!/usr/bin/env bash
# Where inside a TPC-H query does the time go? A top-down (TMA) breakdown per
# operator, from Linux perf alone (no VTune), so it runs on any machine with an
# event set below.
#
# perf samples the top-down slot events with frame-pointer call stacks over
# the measured repetitions only (include/profile_scope.hpp, PROF_PERF_CTL:
# run_tpch switches perf on and off through --control fifos), then
# scripts/perf_opfold.py folds the samples:
#   Tectorwise (-e v)  by operator: the innermost vectorwise::<Op>::next on
#                      the stack, named by its path in the plan, with phases
#                      (Hashjoin build/probe, HashGroup local/global)
#   Typer (-e h)       by pipeline (exact) and by line of qN.cpp (approximate)
# and scripts/perf_ops_report.py writes <out>/analysis/.
#
# Why not perf's topdown-* events: they live in PERF_METRICS, which Linux 5.15
# cannot sample (EINVAL for a {slots,topdown-*}:S group). The general-purpose
# slot events count the same terms and sample like any other event; the TMA
# formulas (Intel perfmon, SPR) apply to them unchanged, with bad speculation
# as the residual. They fill exactly 8 counters; a ninth would multiplex.
# Cycles are slots / SLOTS_PER_CYCLE (6 on Golden Cove: TOPDOWN.SLOTS_P was
# 6.00 x cycles on manchego, SMT sibling idle).
#
# Builds as vtune_q1.sh's: baseline (BASE_REF: Typer + Tectorwise) and CONFIGS
# from flag_ablation.sh (cmake_defaults also runs Typer at HEAD), plus -g and
# -DPROF_PERF_CTL. Per run, <out>/<build>_q<query>_e<engine>_t<threads>/:
#   timing.csv    untraced run_tpch output (median ms; EXTERNALPROFILE=1)
#   stat.csv      perf stat of the event set over STAT_SECONDS: exact totals,
#                 and the running fraction of every event (must be 100%)
#   fold/         perf_opfold.py tables (event sums per operator / function)
#   record.log    perf record's stderr; perf.data with KEEP_DATA=1
#   meta.env      build, query, engine, threads, reps, SF, tuples, events, ...
#
#   sbatch --partition=cheese --nodelist=manchego --exclusive \
#          --output=/tank/alexb/slurm-tpch/manchego/perf_ops_%j.out scripts/perf_ops.sh
#
# Environment (all optional):
#   CONFIGS     "cmake_defaults jd_all_pc"   flag_ablation.sh configs, in ladder order
#   HASH_BASE / BASE_REF / BASE_CXX / COMPILER / COMPILER_VERSION / MACHINE /
#   TARGET_ARCH / DATADIR / JOBS            as vtune_q1.sh
#   SF          1
#   TPCH_PATH   /tank/alexb/swole/tpch/sf$SF
#   TUPLES      lineitem rows; known for SF 1, 10, 30, 100
#   QUERIES     "1 3 5 6 9 18"
#   THREADS     1                 counts or all (the physical cores of CPU's node)
#   CPU         0
#   MIN_SECONDS 200               perf record covers at least this long
#   STAT_SECONDS 20               perf stat covers at least this long
#   TIMING_REPS 10 / SETTLE 1 / VECTOR_SIZE 1024
#   EVENTSET    by CPU model      spr (Sapphire / Emerald Rapids)
#   PERIOD_SCALE 5                sampling periods x this; 1 = about 2K
#                                 samples/s per event (manchego's kernel caps
#                                 perf_event_max_sample_rate at 7000)
#   CALL_GRAPH  fp                perf record --call-graph mode: fp (frame pointers;
#                                 a leaf without one, e.g. in libc, hides its
#                                 caller) or lbr (hardware call stack)
#   KEEP_DATA   0                 1 = keep perf.data
#   BUILD_ONLY  0 / SKIP_BUILD 0  as vtune_q1.sh
#   BUILDS_DIR  build_perf/<host> (build_vtune/<host> builds also work: PROF_ITT
#                                 includes the perf control fifos)
#   PERF        perf
#   OUT         results/perf_ops_<host>_<stamp>
set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_HOST=${BENCH_HOST:-$(hostname -s 2>/dev/null || hostname)}
CONFIGS=${CONFIGS:-"cmake_defaults jd_all_pc"}
BASE_REF=${BASE_REF-origin/baseline}
COMPILER=${COMPILER:-gcc}
COMPILER_VERSION=${COMPILER_VERSION:-new}
case "$(hostname -s 2>/dev/null || hostname)" in
  dubliner|roquefort|manchego|burrata|kafir|rpi5) DEF_MACHINE=$(hostname -s) ;;
  *) DEF_MACHINE=custom ;;
esac
MACHINE=${MACHINE:-$DEF_MACHINE}
case "$MACHINE" in
  manchego) DEF_ARCH=sapphirerapids ;; burrata) DEF_ARCH=neoverse-n1 ;;
  kafir) DEF_ARCH=skylake-x ;; rpi5) DEF_ARCH=cortex-a76 ;; *) DEF_ARCH=native ;;
esac
TARGET_ARCH=${TARGET_ARCH:-$DEF_ARCH}
SF=${SF:-1}
TPCH_PATH=${TPCH_PATH:-/tank/alexb/swole/tpch/sf$SF}
QUERIES=${QUERIES:-"1 3 5 6 9 18"}
THREADS=${THREADS:-1}
CPU=${CPU:-0}
MIN_SECONDS=${MIN_SECONDS:-200}
STAT_SECONDS=${STAT_SECONDS:-20}
TIMING_REPS=${TIMING_REPS:-10}
SETTLE=${SETTLE:-1}
VECTOR_SIZE=${VECTOR_SIZE:-1024}
PERIOD_SCALE=${PERIOD_SCALE:-5}
CALL_GRAPH=${CALL_GRAPH:-fp}
KEEP_DATA=${KEEP_DATA:-0}
BUILD_ONLY=${BUILD_ONLY:-0}
SKIP_BUILD=${SKIP_BUILD:-0}
DATADIR=${DATADIR:-/tank/alexb/swole/}
PERF=${PERF:-perf}
NPROC=$(nproc)
JOBS=${JOBS:-$(( NPROC < 32 ? NPROC : 32 ))}
OUT=${OUT:-$ROOT/results/perf_ops_${BENCH_HOST}_$(date +%Y%m%d_%H%M%S)}
BUILDS_DIR=${BUILDS_DIR:-$ROOT/build_perf/$BENCH_HOST}

if [ -z "${TUPLES:-}" ]; then
  case "$SF" in
    1) TUPLES=6001215 ;; 10) TUPLES=59986052 ;; 30) TUPLES=179998372 ;; 100) TUPLES=600037902 ;;
    *) TUPLES=$( [ -f "$TPCH_PATH/lineitem.tbl" ] && wc -l < "$TPCH_PATH/lineitem.tbl" || echo "") ;;
  esac
fi

mkdir -p "$OUT"
FAILS=0
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/driver.log"; }
fail() { log "FAIL: $*"; FAILS=$((FAILS + 1)); }
excerpt() { [ -s "$1" ] && tail -n "${2:-10}" "$1" | sed 's/^/    | /' | tee -a "$OUT/driver.log"; }
die() { log "$*"; exit 2; }

# ------------------------------------------------------------------ events
# event sets: name=event/umask[/cmask] and a base period (about 2K samples/s
# per event at PERIOD_SCALE=1 on a ~2.5 GHz core); perf_opfold.py and
# perf_ops_report.py use the names, so a new CPU only needs a set here
cpu_key() { awk -F': ' '/^cpu family/ { f = $2 } /^model\t/ { m = $2 } END { print f "/" m }' /proc/cpuinfo; }
case "$(cpu_key)" in
  6/143|6/207) DEF_EVENTSET=spr ;;  # Sapphire Rapids, Emerald Rapids
  *) DEF_EVENTSET= ;;
esac
EVENTSET=${EVENTSET:-$DEF_EVENTSET}
case "$EVENTSET" in
  spr)
    # Intel perfmon SPR: TOPDOWN.SLOTS_P, IDQ_BUBBLES.CORE, IDQ_BUBBLES.CYCLES_0_UOPS_DELIV.CORE,
    # UOPS_RETIRED.SLOTS, UOPS_RETIRED.HEAVY, TOPDOWN.BACKEND_BOUND_SLOTS,
    # TOPDOWN.MEMORY_BOUND_SLOTS, TOPDOWN.BR_MISPREDICT_SLOTS (counter 0 only)
    SLOTS_PER_CYCLE=6
    EVDEFS="slots=0xa4/0x01//10000019 fe=0x9c/0x01//2000003 fe_lat_cyc=0x9c/0x01/6/200003
            ret=0xc2/0x02//5000011 heavy=0xc2/0x01//500009 be=0xa4/0x02//5000011
            mem=0xa4/0x10//5000011 brmisp=0xa4/0x08//2000003"
    INST_PERIOD=5000011 ;;
  "") die "no event set for this CPU (family/model $(cpu_key)); add one to perf_ops.sh or set EVENTSET" ;;
  *) die "unknown EVENTSET $EVENTSET" ;;
esac
EV=""
for d in $EVDEFS; do
  IFS=/ read -r ev um cm per <<< "${d#*=}"
  EV+="cpu/event=$ev,umask=$um${cm:+,cmask=$cm},name=${d%%=*},period=$(( per * PERIOD_SCALE ))/,"
done
EV+="instructions/period=$(( INST_PERIOD * PERIOD_SCALE ))/"

# ------------------------------------------------------------------ checks
[ -n "$TUPLES" ] || die "set TUPLES (lineitem rows) for SF=$SF"
case "$TUPLES" in *[!0-9]*) die "TUPLES must be a number" ;; esac
[ -d "/sys/devices/system/cpu/cpu$CPU" ] || die "CPU $CPU does not exist"
command -v numactl > /dev/null || die "numactl is required (pinning and memory binding)"
if [ "$BUILD_ONLY" != 1 ]; then
  command -v "$PERF" > /dev/null || die "perf not found (set PERF=)"
  command -v addr2line > /dev/null || die "addr2line (binutils) is required by perf_opfold.py"
  command -v python3 > /dev/null || die "python3 is required by perf_opfold.py"
  [ -d "$TPCH_PATH" ] || die "no TPC-H data at $TPCH_PATH"
fi

NODE=0
for d in /sys/devices/system/cpu/cpu"$CPU"/node[0-9]*; do [ -e "$d" ] && NODE=${d##*node}; done
NODE_CORES=$(lscpu -p=CPU,CORE,NODE | grep -v '^#' |
             awk -F, -v n="$NODE" '$3 == n && !seen[$2]++ { printf "%s%s", (k++ ? "," : ""), $1 }')
NCORES=$(tr ',' '\n' <<< "$NODE_CORES" | grep -c .)
[ "$NCORES" -gt 0 ] || die "no physical cores found on NUMA node $NODE"

{
  echo "host: $(hostname)"
  echo "MACHINE: $MACHINE  CPU: $CPU  NODE: $NODE ($NCORES physical cores: $NODE_CORES)"
  echo "date: $(date -Is)"
  echo "git:  $(git -C "$ROOT" rev-parse --short HEAD) $(git -C "$ROOT" status --porcelain --untracked-files=no | wc -l) dirty tracked files"
  echo "BASE_REF: ${BASE_REF:-none} ($( [ -n "$BASE_REF" ] && git -C "$ROOT" rev-parse --short "$BASE_REF" 2> /dev/null))"
  lscpu | grep -E 'Model name|Socket|Core|Thread|NUMA node\(s\)|MHz' || true
  echo -n "governor cpu$CPU: "; cat "/sys/devices/system/cpu/cpu$CPU/cpufreq/scaling_governor" 2> /dev/null || echo n/a
  echo -n "perf_event_paranoid: "; cat /proc/sys/kernel/perf_event_paranoid 2> /dev/null || echo n/a
  echo -n "perf_event_max_sample_rate: "; cat /proc/sys/kernel/perf_event_max_sample_rate 2> /dev/null || echo n/a
  echo -n "nmi_watchdog: "; cat /proc/sys/kernel/nmi_watchdog 2> /dev/null || echo n/a
  echo "kernel: $(uname -r)  perf: $("$PERF" --version 2> /dev/null)"
  echo "EVENTSET: $EVENTSET ($(cpu_key))  PERIOD_SCALE: $PERIOD_SCALE  SLOTS_PER_CYCLE: $SLOTS_PER_CYCLE  CALL_GRAPH: $CALL_GRAPH"
  echo "events: $EV"
  echo "COMPILER: $COMPILER $COMPILER_VERSION  TARGET_ARCH: $TARGET_ARCH  SF: $SF ($TPCH_PATH, $TUPLES lineitem rows)  VECTOR_SIZE: $VECTOR_SIZE"
  echo "CONFIGS: $CONFIGS  QUERIES: $QUERIES  THREADS: $THREADS  MIN_SECONDS: $MIN_SECONDS  STAT_SECONDS: $STAT_SECONDS"
} | tee "$OUT/machine.txt"

# ------------------------------------------------------------------ builds
PROF_CXX="-g -DPROF_PERF_CTL"
cmake_common=(-DCMAKE_BUILD_TYPE=Release -DCOMPILER="$COMPILER" -DCOMPILER_VERSION="$COMPILER_VERSION"
              -DTARGET_MACHINE="$MACHINE" -DTARGET_ARCH="$TARGET_ARCH" -DDATADIR="$DATADIR" -DTHREAD_PIN_PACKED=ON
              "-DCMAKE_CXX_FLAGS=$PROF_CXX")

# apply_scope <tree>: give an older checkout the measured-window hook (as vtune_q1.sh)
apply_scope() {
  local f="$1/include/profile.hpp"
  cp "$ROOT/include/profile_scope.hpp" "$1/include/"
  grep -q 'profscope::' "$f" && return 0
  sed -i '0,/^#include /s//#include "profile_scope.hpp"\n#include /' "$f"
  sed -i '0,/^\( *\)startAll();/s//\1profscope::begin(s.c_str());\n\1startAll();/' "$f"
  sed -i '0,/^\( *\)readAll();/s//\1readAll();\n\1profscope::end();/' "$f"
  [ "$(grep -c 'profscope::\(begin\|end\)' "$f")" = 2 ]
}

# build <name> <source tree> <cmake options...>
build() {
  local name=$1 src=$2; shift 2
  local B=$BUILDS_DIR/$name D=$OUT/builds/$name
  mkdir -p "$D"
  if [ "$SKIP_BUILD" != 1 ]; then
    if ! cmake -S "$src" -B "$B" "${cmake_common[@]}" "$@" > "$D/cmake.log" 2>&1; then
      fail "$name cmake (see $D/cmake.log)"; excerpt "$D/cmake.log"; return 1
    fi
    if ! cmake --build "$B" -j "$JOBS" --target run_tpch > "$D/build.log" 2>&1; then
      fail "$name build (see $D/build.log)"; excerpt "$D/build.log"; return 1
    fi
    { echo "cmake ${cmake_common[*]} $*"; echo "commit: $(git -C "$src" rev-parse HEAD)"; } > "$B/buildcmds.txt"
    grep -m1 -o 'TARGET_MACHINE=.*' "$D/cmake.log" > "$B/topology.txt"
    grep -m1 -o 'ARCH_FLAGS=.*' "$D/cmake.log" > "$B/arch.txt"
  fi
  [ -x "$B/run_tpch" ] || { fail "$name: no run_tpch"; return 1; }
  cp "$B/buildcmds.txt" "$B/topology.txt" "$B/arch.txt" "$D/" 2> /dev/null
  cat "$B/build.ninja" "$B/CMakeFiles/vectorwise.dir/flags.make" 2> /dev/null |
    grep -o -E -- '-D(VW_[A-Z0-9_]+|HUGE_2MB_MALLOC_HUGE|THREAD_PIN_PACKED|PROF_ITT|PROF_PERF_CTL)(=[^ ]*)?' | sort -u | tr '\n' ' ' > "$D/defines.txt"
  # the control fifo hook: its environment variable name is in the binary
  [ "$(grep -c -a PERF_CTL_FIFO "$B/run_tpch")" -gt 0 ] || { fail "$name: run_tpch has no perf control hook (PROF_PERF_CTL)"; return 1; }
  log "$name built: $(cat "$D/defines.txt")"
}

mkdir -p "$BUILDS_DIR"
BUILDS=()
if [ -n "$BASE_REF" ]; then
  # the same source worktree as vtune_q1.sh's (source only; builds differ)
  WT="$ROOT/../vtune_base_$(git -C "$ROOT" rev-parse --short "$BASE_REF" 2> /dev/null || echo none)_$BENCH_HOST"
  if [ "$SKIP_BUILD" != 1 ] && [ ! -d "$WT" ]; then
    git -C "$ROOT" worktree add --detach "$WT" "$BASE_REF" > "$OUT/worktree.log" 2>&1 ||
      { fail "worktree for $BASE_REF (see $OUT/worktree.log)"; WT=; }
  fi
  if [ -n "$WT" ] && [ -d "$WT" ]; then
    if [ ! -e "$WT/3rdparty/simde/simde" ] && [ -d "$ROOT/3rdparty/simde/simde" ]; then
      rm -rf "$WT/3rdparty/simde" && ln -s "$ROOT/3rdparty/simde" "$WT/3rdparty/simde"
    fi
    if apply_scope "$WT"; then
      BASE_CXX=${BASE_CXX-"-include cstdint -Wno-changes-meaning"}
      build baseline "$WT" "-DCMAKE_CXX_FLAGS=$PROF_CXX $BASE_CXX" && BUILDS+=(baseline)
    else fail "could not add the profiling hook to $BASE_REF's include/profile.hpp"; fi
  fi
fi
for cfg in $CONFIGS; do
  flags=$(env -i PATH="$PATH" HOME="${HOME:-/tmp}" PRINT_FLAGS="$cfg" ${SEL:+SEL="$SEL"} ${AUTOVEC:+AUTOVEC="$AUTOVEC"} ${HASH_BASE:+HASH_BASE="$HASH_BASE"} \
          bash "$ROOT/scripts/flag_ablation.sh") || { fail "unknown config $cfg"; continue; }
  # shellcheck disable=SC2086
  build "$cfg" "$ROOT" $flags && BUILDS+=("$cfg")
done
[ "${#BUILDS[@]}" -gt 0 ] || die "nothing built"
archs=$(for b in "${BUILDS[@]}"; do cat "$OUT/builds/$b/arch.txt" 2> /dev/null; done | sort -u)
if [ "$(grep -c . <<< "$archs")" -gt 1 ]; then
  log "WARNING: builds differ in ARCH_FLAGS:"; echo "$archs" | sed 's/^/    | /' | tee -a "$OUT/driver.log"
else log "all builds: ${archs:-ARCH_FLAGS unknown}"; fi
if [ "$BUILD_ONLY" = 1 ]; then log "BUILD_ONLY=1: built ${BUILDS[*]}; no runs"; exit $(( FAILS > 0 )); fi

# ------------------------------------------------------------------ runs
for t in $THREADS; do
  case "$t" in all) ;; *[!0-9]*|0) die "THREADS entries must be counts or all" ;; esac
done
for q in $QUERIES; do
  case "$q" in *[!0-9]*|0) die "QUERIES entries must be query numbers" ;; esac
done
nthreads() { [ "$1" = all ] && echo "$NCORES" || echo "$1"; }
pin_for() {
  if [ "$1" = 1 ]; then echo "numactl --physcpubind=$CPU --membind=$NODE"
  else echo "numactl --physcpubind=$NODE_CORES --membind=$NODE"; fi
}
median_of() { awk -F, -v q="q$2 " '{ n = $1; gsub(/^ +| +$/, "", n) } index(n, q) == 1 { v = $2; gsub(/ /, "", v); print v; exit }' "$1"; }
reps_for() { awk -v ms="$1" -v s="$2" 'BEGIN { r = int(s * 1000 / ms) + 1; if (r < 3) r = 3; if (r > 20000) r = 20000; print r }'; }
# perf_ctl <dir> <stat|record> <perf args...>: run perf with run_tpch driving
# its control fifos; events start disabled (-D -1)
perf_ctl() {
  local D=$1 cmd=$2; shift 2
  rm -f "$D/ctl" "$D/ack"; mkfifo "$D/ctl" "$D/ack"
  EXTERNALPROFILE=1 PERF_CTL_FIFO=$D/ctl PERF_ACK_FIFO=$D/ack \
    "$PERF" "$cmd" -D -1 --control "fifo:$D/ctl,$D/ack" "$@"
  local rc=$?
  rm -f "$D/ctl" "$D/ack"
  return $rc
}

# discard the first untraced run of the job (as vtune_q1.sh)
q0=${QUERIES%% *}
# shellcheck disable=SC2046
EXTERNALPROFILE=1 $(pin_for 1) "$BUILDS_DIR/${BUILDS[0]}/run_tpch" -p "$TPCH_PATH/" -e v -q "$q0" -t 1 \
  -v "$VECTOR_SIZE" -s "$SETTLE" -r "$TIMING_REPS" > /dev/null 2>&1 || log "WARNING: warm-up run failed"

idx=0
for b in "${BUILDS[@]}"; do
  bin=$BUILDS_DIR/$b/run_tpch
  engines=v
  { [ "$b" = baseline ] || [ "$b" = cmake_defaults ]; } && engines="h v"
  for e in $engines; do
   for q in $QUERIES; do
    for t in $THREADS; do
      T=$(nthreads "$t")
      PIN=$(pin_for "$T")
      label=${b}_q${q}_e${e}_t$T; D=$OUT/$label; mkdir -p "$D"
      run=(-p "$TPCH_PATH/" -e "$e" -q "$q" -t "$T" -v "$VECTOR_SIZE" -s "$SETTLE")
      log "===== $label ($PIN)"
      # shellcheck disable=SC2086
      if ! EXTERNALPROFILE=1 $PIN "$bin" "${run[@]}" -r "$TIMING_REPS" > "$D/timing.csv" 2> "$D/timing.err"; then
        fail "$label timing (see $D/timing.err)"; excerpt "$D/timing.err" 5; continue
      fi
      ms=$(median_of "$D/timing.csv" "$q")
      [ -n "$ms" ] || { fail "$label: no q$q row in timing.csv"; continue; }
      reps=$(reps_for "$ms" "$MIN_SECONDS"); sreps=$(reps_for "$ms" "$STAT_SECONDS")
      log "$label: median $ms ms; record reps $reps, stat reps $sreps"
      {
        echo "label=$label"; echo "build=$b"; echo "query=$q"; echo "engine=$e"; echo "threads=$T"; echo "threads_req=$t"
        echo "ladder_index=$idx"; echo "configs=$CONFIGS"; echo "reps=$reps"; echo "stat_reps=$sreps"
        echo "timing_reps=$TIMING_REPS"; echo "timing_median_ms=$ms"; echo "sf=$SF"; echo "tuples=$TUPLES"
        echo "vector_size=$VECTOR_SIZE"; echo "machine=$MACHINE"; echo "host=$(hostname -s)"; echo "node=$NODE"; echo "pin=$PIN"
        echo "cpu_model=$(lscpu | sed -n 's/^Model name: *//p' | head -1)"
        echo "eventset=$EVENTSET"; echo "period_scale=$PERIOD_SCALE"; echo "slots_per_cycle=$SLOTS_PER_CYCLE"
        echo "call_graph=$CALL_GRAPH"
        echo "commit=$(git -C "$ROOT" rev-parse --short HEAD)"
        echo "build_commit=$(sed -n 's/^commit: //p' "$OUT/builds/$b/buildcmds.txt" 2> /dev/null | cut -c1-12)"
        echo "defines=$(cat "$OUT/builds/$b/defines.txt" 2> /dev/null)"
      } > "$D/meta.env"

      # exact totals; every event must count all the time (no multiplexing)
      # shellcheck disable=SC2086
      if ! perf_ctl "$D" stat -e "$EV" -x, -o "$D/stat.csv" -- $PIN "$bin" "${run[@]}" -r "$sreps" \
            > "$D/stat_run.out" 2> "$D/stat.err"; then
        fail "$label perf stat (see $D/stat.err)"; excerpt "$D/stat.err" 5; continue
      fi
      mux=$(awk -F, '!/^#/ && NF > 4 && $5 != "" && $5 + 0 < 99.9 { printf "%s(%s%%) ", $3, $5 }' "$D/stat.csv")
      [ -z "$mux" ] || log "WARNING: $label: events multiplexed: $mux"

      # shellcheck disable=SC2086
      if ! perf_ctl "$D" record -e "$EV" --call-graph "$CALL_GRAPH" -o "$D/perf.data" -- $PIN "$bin" "${run[@]}" -r "$reps" \
            > "$D/record_run.out" 2> "$D/record.log"; then
        fail "$label perf record (see $D/record.log)"; excerpt "$D/record.log" 5; continue
      fi
      thr=$("$PERF" report -i "$D/perf.data" --stats 2> /dev/null | awk '/THROTTLE events/ { print $3; exit }')
      [ "${thr:-0}" = 0 ] || log "WARNING: $label: $thr throttle events (raise PERIOD_SCALE)"
      if ! "$PERF" script -i "$D/perf.data" -F event,period,ip,sym,symoff 2> /dev/null |
           python3 "$ROOT/scripts/perf_opfold.py" "$bin" "$e" "$q" "$D/fold" > "$D/fold.log" 2>&1; then
        fail "$label perf_opfold.py (see $D/fold.log)"; excerpt "$D/fold.log" 5; continue
      fi
      smp=$(awk -F, -v p="$((10000019 * PERIOD_SCALE))" 'NR == 2 { printf "%d", $2 / p }' "$D/fold/totals.csv")
      [ "${smp:-0}" -ge 1000 ] || { fail "$label: only ${smp:-0} slots samples"; continue; }
      [ "$KEEP_DATA" = 1 ] || rm -f "$D/perf.data"
      log "$label done: $smp slots samples; $(cat "$D/fold.log")"
    done
   done
  done
  idx=$((idx + 1))
done

# ----------------------------------------------------------------- analysis
if python3 "$ROOT/scripts/perf_ops_report.py" "$OUT" --out "$OUT/analysis" >> "$OUT/driver.log" 2>&1; then
  log "analysis: $OUT/analysis"
else fail "perf_ops_report.py (see $OUT/driver.log)"; fi
log "results: $OUT"
if [ "$FAILS" -gt 0 ]; then log "$FAILS FAILURE(S)"; exit 1; fi
log "ALL STEPS PASSED"
