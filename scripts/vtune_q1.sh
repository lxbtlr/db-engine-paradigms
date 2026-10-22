#!/usr/bin/env bash
# Where does TPC-H Q1's time go, and what does each VectorWise change buy?
#
# A micro-architectural breakdown in the style of Sirin & Ailamaki,
# "Micro-architectural Analysis of OLAP: Limitations and Opportunities"
# (VLDB 2020), with Intel VTune on this host (dubliner = Cascade Lake,
# manchego = Sapphire Rapids). Profiled, each at 1 thread and at every
# physical core of one socket:
#
#   baseline   BASE_REF (default origin/baseline) with CMake defaults: the
#              original Typer (-e h) and Tectorwise (-e v)
#   <config>   HEAD built with flag_ablation.sh's <config> options, -e v; a
#              cumulative ladder (CONFIGS, in order). cmake_defaults also runs
#              -e h, Typer at HEAD, to show any drift from the baseline commit
#
# Every build uses the same COMPILER / COMPILER_VERSION / TARGET_MACHINE, adds
# -g (debug info only: GCC and Clang emit the same code with or without it),
# -DPROF_ITT and libittnotify, so `vtune -start-paused` records only the
# measured repetitions (include/profile_scope.hpp), and THREAD_PIN_PACKED=ON so
# multi-threaded runs fill one socket (cpuOfThread, Concurrency.hpp) instead of
# spreading worker threads round-robin over all of them.
#
# Per run, results/vtune_q1_<host>_<stamp>/<build>_e<engine>_t<threads>/:
#   timing.csv            untraced run_tpch output (median ms; EXTERNALPROFILE=1, no counters)
#   ua.summary.txt/.csv   uarch-exploration summary: clockticks, instructions,
#                         the top-down (TMA) tree
#   ua.functions.csv      uarch-exploration per function (TMA columns)
#   ua.hwevents.csv       raw event counts per function
#   mem.summary.txt/.csv  memory-access summary (DRAM bandwidth), MEM_THREADS runs
#   meta.env              build, engine, threads, reps, SF, tuples, defines, ...
#   r_ua/ r_mem/          VTune result directories (open them in the GUI)
# then scripts/vtune_breakdown.py writes <out>/analysis/ from all of them.
#
# Typical use (one job per machine; the analysis can merge machines later):
#   sbatch --partition=cheese --nodelist=dubliner --exclusive \
#          --output=/tank/alexb/slurm-tpch/dubliner/vtune_q1_%j.out scripts/vtune_q1.sh
#   scripts/vtune_breakdown.py results/vtune_q1_dubliner_* results/vtune_q1_manchego_* --out q1_breakdown
#
# Environment (all optional):
#   CONFIGS     "cmake_defaults default grp_base grp_dispatch grp_kinds
#               proj_compound pc_runheads"   any flag_ablation.sh config names;
#               the order is the ladder the analysis steps through
#   HASH_BASE   default           flag_ablation.sh base for its hash, join and group
#               configs (e.g. add_huge2mb: HUGE_2MB_MALLOC_HUGE=ON); not cmake_defaults
#   BASE_REF    origin/baseline   commit for Typer / Tectorwise; "" = skip
#   BASE_CXX    "-include cstdint -Wno-changes-meaning"  front-end-only flags
#               the old commit needs to compile with current GCC
#   COMPILER    gcc               gcc | clang
#   COMPILER_VERSION new          new | old (CMakeLists.txt)
#   MACHINE     this host if it is a TARGET_MACHINE preset, else custom
#   TARGET_ARCH the MACHINE preset's (manchego: sapphirerapids, dubliner:
#               native); passed explicitly to every build so all get one -march
#   SF          10                (Q1's sum_charge overflows int64 near SF 40;
#                                 see TYPER_Q1_TUNING.md)
#   TPCH_PATH   /tank/alexb/swole/tpch/sf$SF
#   TUPLES      lineitem rows; known for SF 1, 10, 30, 100
#   THREADS     "1 all"           all = the physical cores of CPU's NUMA node
#   CPU         0                 the 1-thread run's CPU; its node is "the socket"
#   MIN_SECONDS 20                VTune reps are chosen so the measured window
#                                 lasts at least this long (Cascade Lake
#                                 multiplexes the TMA events and needs time)
#   TIMING_REPS 10                reps of the untraced timing run
#   SETTLE      1                 run_tpch -s
#   ANALYSES    "ua mem"          ua = uarch-exploration, mem = memory-access
#   MEM_THREADS "all"             thread settings that also get memory-access
#   VECTOR_SIZE 1024
#   VTUNE       vtune             VTune command line tool
#   ITT_DIR     VTune's sdk dir   needs include/ittnotify.h and
#                                 lib64/libittnotify.a (or lib/, bin/); or set
#                                 ITT_INC (dir) and ITT_LIB (the .a) directly
#   VTUNE_UA_OPTS / VTUNE_MEM_OPTS  extra collect options, e.g. "-knob sampling-interval=0.5"
#   VTUNE_KEEP  1                 0 = delete r_ua/ r_mem/ after exporting reports
#   BUILD_ONLY  0                 1 = build (and check pinning), skip all runs
#   SKIP_BUILD  0                 1 = reuse build_vtune/<host>/ binaries
#   DATADIR     /tank/alexb/swole/  CMake DATADIR
#   JOBS        min(nproc, 32)
#   OUT         results/vtune_q1_<host>_<stamp>
set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_HOST=${BENCH_HOST:-$(hostname -s 2>/dev/null || hostname)}
CONFIGS=${CONFIGS:-"cmake_defaults default grp_base grp_dispatch grp_kinds proj_compound pc_runheads"}
BASE_REF=${BASE_REF-origin/baseline}
COMPILER=${COMPILER:-gcc}
COMPILER_VERSION=${COMPILER_VERSION:-new}
case "$(hostname -s 2>/dev/null || hostname)" in
  dubliner|roquefort|manchego|burrata|kafir|rpi5) DEF_MACHINE=$(hostname -s) ;;
  *) DEF_MACHINE=custom ;;
esac
MACHINE=${MACHINE:-$DEF_MACHINE}
# one explicit -march for every build: HEAD reads TARGET_ARCH= as "the
# preset's", but older commits default to native, which on manchego would
# compare -march=native against -march=sapphirerapids. Preset values as in
# CMakeLists.txt.
case "$MACHINE" in
  manchego) DEF_ARCH=sapphirerapids ;; burrata) DEF_ARCH=neoverse-n1 ;;
  kafir) DEF_ARCH=skylake-x ;; rpi5) DEF_ARCH=cortex-a76 ;; *) DEF_ARCH=native ;;
esac
TARGET_ARCH=${TARGET_ARCH:-$DEF_ARCH}
SF=${SF:-10}
TPCH_PATH=${TPCH_PATH:-/tank/alexb/swole/tpch/sf$SF}
THREADS=${THREADS:-"1 all"}
CPU=${CPU:-0}
MIN_SECONDS=${MIN_SECONDS:-20}
TIMING_REPS=${TIMING_REPS:-10}
SETTLE=${SETTLE:-1}
ANALYSES=${ANALYSES:-"ua mem"}
MEM_THREADS=${MEM_THREADS:-all}
VECTOR_SIZE=${VECTOR_SIZE:-1024}
VTUNE=${VTUNE:-vtune}
VTUNE_UA_OPTS=${VTUNE_UA_OPTS:-}
VTUNE_MEM_OPTS=${VTUNE_MEM_OPTS:-}
VTUNE_KEEP=${VTUNE_KEEP:-1}
BUILD_ONLY=${BUILD_ONLY:-0}
SKIP_BUILD=${SKIP_BUILD:-0}
DATADIR=${DATADIR:-/tank/alexb/swole/}
NPROC=$(nproc)
JOBS=${JOBS:-$(( NPROC < 32 ? NPROC : 32 ))}
OUT=${OUT:-$ROOT/results/vtune_q1_${BENCH_HOST}_$(date +%Y%m%d_%H%M%S)}
BUILDS_DIR=$ROOT/build_vtune/$BENCH_HOST

# lineitem rows for the standard scale factors (dbgen)
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

# ------------------------------------------------------------------ checks
[ -n "$TUPLES" ] || die "set TUPLES (lineitem rows) for SF=$SF"
case "$TUPLES" in *[!0-9]*) die "TUPLES must be a number" ;; esac
[ -d "/sys/devices/system/cpu/cpu$CPU" ] || die "CPU $CPU does not exist"
command -v numactl > /dev/null || die "numactl is required (pinning and memory binding)"
if [ "$BUILD_ONLY" != 1 ]; then
  command -v "$VTUNE" > /dev/null || die "VTune not found (set VTUNE=, or source <oneapi>/vtune/latest/env/vars.sh)"
  [ -d "$TPCH_PATH" ] || die "no TPC-H data at $TPCH_PATH"
fi
# ittnotify from VTune's own SDK unless ITT_DIR says otherwise
if [ -z "${ITT_DIR:-}" ] && command -v "$VTUNE" > /dev/null; then
  ITT_DIR=$(cd "$(dirname "$(readlink -f "$(command -v "$VTUNE")")")/../sdk" 2> /dev/null && pwd)
fi
ITT_INC=${ITT_INC:-${ITT_DIR:-}/include}
if [ -z "${ITT_LIB:-}" ]; then
  for l in lib64 lib bin; do [ -f "${ITT_DIR:-}/$l/libittnotify.a" ] && { ITT_LIB=$ITT_DIR/$l/libittnotify.a; break; }; done
fi
[ -f "$ITT_INC/ittnotify.h" ] && [ -n "$ITT_LIB" ] ||
  die "ittnotify not found under ITT_DIR='${ITT_DIR:-}' (needs include/ittnotify.h and lib64/libittnotify.a)"

# the socket: CPU's NUMA node, and one hardware thread per physical core on it
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
  echo "VTune: $("$VTUNE" -version 2> /dev/null | head -1)  ITT: $ITT_LIB"
  echo "COMPILER: $COMPILER $COMPILER_VERSION  TARGET_ARCH: $TARGET_ARCH  SF: $SF ($TPCH_PATH, $TUPLES lineitem rows)  VECTOR_SIZE: $VECTOR_SIZE"
  echo "CONFIGS: $CONFIGS"
  echo "THREADS: $THREADS  ANALYSES: $ANALYSES  MEM_THREADS: $MEM_THREADS  MIN_SECONDS: $MIN_SECONDS"
} | tee "$OUT/machine.txt"
p=$(cat /proc/sys/kernel/perf_event_paranoid 2> /dev/null || echo 0)
[ "$p" -gt 0 ] && log "WARNING: perf_event_paranoid=$p; VTune's uncore (DRAM bandwidth) events need <= 0 or the sampling driver"

# ------------------------------------------------------------------ builds
ITT_CXX="-g -DPROF_ITT -I$ITT_INC"
ITT_LIBS="$ITT_LIB -ldl"
cmake_common=(-DCMAKE_BUILD_TYPE=Release -DCOMPILER="$COMPILER" -DCOMPILER_VERSION="$COMPILER_VERSION"
              -DTARGET_MACHINE="$MACHINE" -DTARGET_ARCH="$TARGET_ARCH" -DDATADIR="$DATADIR" -DTHREAD_PIN_PACKED=ON
              "-DCMAKE_CXX_FLAGS=$ITT_CXX" "-DCMAKE_CXX_STANDARD_LIBRARIES=$ITT_LIBS")

# apply_scope <tree>: give an older checkout the measured-window hook
# (profile_scope.hpp + two calls in timeAndProfile); HEAD already has it
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
    # kept beside the binary, so SKIP_BUILD=1 runs still know them
    { echo "cmake ${cmake_common[*]} $*"; echo "commit: $(git -C "$src" rev-parse HEAD)"; } > "$B/buildcmds.txt"
    grep -m1 -o 'TARGET_MACHINE=.*' "$D/cmake.log" > "$B/topology.txt"
    grep -m1 -o 'ARCH_FLAGS=.*' "$D/cmake.log" > "$B/arch.txt"
  fi
  [ -x "$B/run_tpch" ] || { fail "$name: no run_tpch"; return 1; }
  cp "$B/buildcmds.txt" "$B/topology.txt" "$B/arch.txt" "$D/" 2> /dev/null
  # the effective defines, as flag_ablation.sh records them
  cat "$B/build.ninja" "$B/CMakeFiles/vectorwise.dir/flags.make" 2> /dev/null |
    grep -o -E -- '-D(VW_[A-Z0-9_]+|HUGE_2MB_MALLOC_HUGE|THREAD_PIN_PACKED|PROF_ITT)(=[^ ]*)?' | sort -u | tr '\n' ' ' > "$D/defines.txt"
  # grep -c, not -q: -q exits early, nm gets SIGPIPE and pipefail fails the check
  [ "$(nm "$B/run_tpch" 2> /dev/null | grep -c __itt_resume)" -gt 0 ] || { fail "$name: run_tpch has no ITT symbols"; return 1; }
  log "$name built: $(cat "$D/defines.txt")"
}

# pinned_cpus <threads> <topology line>: the CPUs run_tpch pins worker
# threads 0..T-2 to (cpuOfThread with THREAD_PIN_PACKED); the last worker is
# the main thread and stays wherever numactl puts it
pinned_cpus() {
  local T=$1 topo=$2 S C M contig
  S=$(sed -E 's/.*=> *([0-9]+)S.*/\1/' <<< "$topo"); C=$(sed -E 's/.* ([0-9]+)C .*/\1/' <<< "$topo")
  M=$(sed -E 's/.* ([0-9]+)T .*/\1/' <<< "$topo"); contig=$(sed -E 's/.*contiguous=([A-Za-z]+).*/\1/' <<< "$topo")
  awk -v T="$T" -v S="$S" -v C="$C" -v M="$M" -v contig="$contig" 'BEGIN {
    tps = C * M
    for (i = 0; i < T - 1; i++) {
      c = (contig == "ON") ? i : (i % tps) * S + int(i / tps)
      printf "%s%d", (i ? "," : ""), c } }'
}

# placement_ok <build> <threads>: every pinned worker on NODE, one per core
placement_ok() {
  local topo cpus bad
  topo=$(cat "$OUT/builds/$1/topology.txt" 2> /dev/null)
  [ -n "$topo" ] || { log "WARNING: $1: topology unknown (SKIP_BUILD?); placement not checked"; return 0; }
  cpus=$(pinned_cpus "$2" "$topo")
  [ -z "$cpus" ] && return 0
  bad=$(lscpu -p=CPU,CORE,NODE | grep -v '^#' | awk -F, -v want="$cpus" -v n="$NODE" '
    BEGIN { k = split(want, w, ","); for (i = 1; i <= k; i++) need[w[i]] = 1 }
    ($1 in need) { found[$1] = 1; if ($3 != n) print "cpu" $1 "@node" $3; if (core[$2]++) print "cpu" $1 "-shares-core-" $2 }
    END { for (c in need) if (!(c in found)) print "cpu" c "-missing" }')
  if [ -n "$bad" ]; then
    log "WARNING: $1 t$2: worker pinning ($topo) lands off node $NODE's cores: $(echo $bad); skipping"
    return 1
  fi
}

mkdir -p "$BUILDS_DIR"
BUILDS=()
if [ -n "$BASE_REF" ]; then
  WT="$ROOT/../vtune_base_$(git -C "$ROOT" rev-parse --short "$BASE_REF" 2> /dev/null || echo none)_$BENCH_HOST"
  if [ "$SKIP_BUILD" != 1 ] && [ ! -d "$WT" ]; then
    git -C "$ROOT" worktree add --detach "$WT" "$BASE_REF" > "$OUT/worktree.log" 2>&1 ||
      { fail "worktree for $BASE_REF (see $OUT/worktree.log)"; WT=; }
  fi
  if [ -n "$WT" ] && [ -d "$WT" ]; then
    # the header-only submodule (as q1_ab.sh)
    if [ ! -e "$WT/3rdparty/simde/simde" ] && [ -d "$ROOT/3rdparty/simde/simde" ]; then
      rm -rf "$WT/3rdparty/simde" && ln -s "$ROOT/3rdparty/simde" "$WT/3rdparty/simde"
    fi
    if apply_scope "$WT"; then
      # Older commits don't compile with current GCC as they are: they use
      # uint32_t & co. without <cstdint> (no longer included transitively
      # since GCC 13), and QueryBuilder's Project()/HashGroup()/Expression()
      # members change the meaning of names the class already used, an
      # error by default (-Wchanges-meaning). Both fixes are front-end only:
      # -include makes the header visible, -Wno-changes-meaning accepts the
      # declarations; neither changes generated code. BASE_CXX overrides.
      BASE_CXX=${BASE_CXX-"-include cstdint -Wno-changes-meaning"}
      build baseline "$WT" "-DCMAKE_CXX_FLAGS=$ITT_CXX $BASE_CXX" && BUILDS+=(baseline)
    else fail "could not add the profiling hook to $BASE_REF's include/profile.hpp"; fi
  fi
fi
for cfg in $CONFIGS; do
  # a clean environment: flag_ablation.sh validates its own THREADS, PIN, ...
  # even when it only prints; SEL, AUTOVEC and HASH_BASE (the base the
  # hash, join and group configs build on, e.g. add_huge2mb) change flags
  flags=$(env -i PATH="$PATH" HOME="${HOME:-/tmp}" PRINT_FLAGS="$cfg" ${SEL:+SEL="$SEL"} ${AUTOVEC:+AUTOVEC="$AUTOVEC"} ${HASH_BASE:+HASH_BASE="$HASH_BASE"} \
            bash "$ROOT/scripts/flag_ablation.sh") || { fail "unknown config $cfg"; continue; }
  # shellcheck disable=SC2086
  build "$cfg" "$ROOT" $flags && BUILDS+=("$cfg")
done
[ "${#BUILDS[@]}" -gt 0 ] || die "nothing built"
# every build must have compiled for the same target
archs=$(for b in "${BUILDS[@]}"; do cat "$OUT/builds/$b/arch.txt" 2> /dev/null; done | sort -u)
if [ "$(grep -c . <<< "$archs")" -gt 1 ]; then
  log "WARNING: builds differ in ARCH_FLAGS:"; echo "$archs" | sed 's/^/    | /' | tee -a "$OUT/driver.log"
else log "all builds: ${archs:-ARCH_FLAGS unknown}"; fi

# ------------------------------------------------------------------ runs
for t in $THREADS; do
  case "$t" in all) ;; *[!0-9]*|0) die "THREADS entries must be counts or all" ;; esac
done
nthreads() { [ "$1" = all ] && echo "$NCORES" || echo "$1"; }
pin_for() { # 1 thread: that CPU and its memory; more: the node's cores and memory
  if [ "$1" = 1 ]; then echo "numactl --physcpubind=$CPU --membind=$NODE"
  else echo "numactl --physcpubind=$NODE_CORES --membind=$NODE"; fi
}
median_of() { awk -F, '{ n = $1; gsub(/^ +| +$/, "", n) } n ~ /^q1 / { v = $2; gsub(/ /, "", v); print v; exit }' "$1"; }
has() { case " $1 " in *" $2 "*) return 0 ;; esac; return 1; }

if [ "$BUILD_ONLY" = 1 ]; then
  for b in "${BUILDS[@]}"; do for t in $THREADS; do
    T=$(nthreads "$t"); [ "$T" -gt 1 ] && placement_ok "$b" "$T" && log "$b t$T: placement on node $NODE ok"
  done; done
  log "BUILD_ONLY=1: built ${BUILDS[*]}; no runs"; exit $(( FAILS > 0 ))
fi

idx=0
for b in "${BUILDS[@]}"; do
  bin=$BUILDS_DIR/$b/run_tpch
  engines=v
  { [ "$b" = baseline ] || [ "$b" = cmake_defaults ]; } && engines="h v"
  for e in $engines; do
    for t in $THREADS; do
      T=$(nthreads "$t")
      [ "$T" -gt 1 ] && ! placement_ok "$b" "$T" && continue
      PIN=$(pin_for "$T")
      label=${b}_e${e}_t$T; D=$OUT/$label; mkdir -p "$D"
      run=(-p "$TPCH_PATH/" -e "$e" -q 1 -t "$T" -v "$VECTOR_SIZE" -s "$SETTLE")
      log "===== $label ($PIN)"
      # untraced timing; EXTERNALPROFILE: run_tpch opens no counters of its own,
      # so the timing runs with the PMU as idle as the VTune runs leave it
      # shellcheck disable=SC2086
      if ! EXTERNALPROFILE=1 $PIN "$bin" "${run[@]}" -r "$TIMING_REPS" > "$D/timing.csv" 2> "$D/timing.err"; then
        fail "$label timing (see $D/timing.err)"; excerpt "$D/timing.err" 5; continue
      fi
      ms=$(median_of "$D/timing.csv")
      [ -n "$ms" ] || { fail "$label: no q1 row in timing.csv"; continue; }
      reps=$(awk -v ms="$ms" -v s="$MIN_SECONDS" 'BEGIN { r = int(s * 1000 / ms) + 1; if (r < 3) r = 3; if (r > 5000) r = 5000; print r }')
      log "$label: median $ms ms; VTune reps $reps (>= ${MIN_SECONDS}s measured)"
      {
        echo "label=$label"; echo "build=$b"; echo "engine=$e"; echo "threads=$T"; echo "threads_req=$t"
        echo "ladder_index=$idx"; echo "configs=$CONFIGS"; echo "reps=$reps"; echo "timing_reps=$TIMING_REPS"
        echo "timing_median_ms=$ms"; echo "sf=$SF"; echo "tuples=$TUPLES"; echo "vector_size=$VECTOR_SIZE"
        echo "machine=$MACHINE"; echo "host=$(hostname -s)"; echo "node=$NODE"; echo "pin=$PIN"
        echo "cpu_model=$(lscpu | sed -n 's/^Model name: *//p' | head -1)"
        echo "commit=$(git -C "$ROOT" rev-parse --short HEAD)"
        echo "build_commit=$(sed -n 's/^commit: //p' "$OUT/builds/$b/buildcmds.txt" 2> /dev/null | cut -c1-12)"
        echo "defines=$(cat "$OUT/builds/$b/defines.txt" 2> /dev/null)"
      } > "$D/meta.env"
      for a in $ANALYSES; do
        case "$a" in
          ua)  coll=uarch-exploration; opts=$VTUNE_UA_OPTS ;;
          mem) has "$MEM_THREADS" "$t" || has "$MEM_THREADS" "$T" || continue
               coll=memory-access; opts=$VTUNE_MEM_OPTS ;;
          *) fail "unknown analysis $a (ua, mem)"; continue ;;
        esac
        R=$D/r_$a; rm -rf "$R"
        # EXTERNALPROFILE: run_tpch opens no counters of its own, leaving the
        # PMU to VTune; -start-paused + ITT: only the measured reps are sampled
        # shellcheck disable=SC2086
        if ! EXTERNALPROFILE=1 "$VTUNE" -collect "$coll" -start-paused -data-limit=0 -result-dir "$R" $opts \
              -- $PIN "$bin" "${run[@]}" -r "$reps" > "$D/$a.run.log" 2>&1; then
          fail "$label $coll (see $D/$a.run.log)"; excerpt "$D/$a.run.log" 8; continue
        fi
        "$VTUNE" -report summary -result-dir "$R" > "$D/$a.summary.txt" 2> "$D/$a.report.err"
        "$VTUNE" -report summary -result-dir "$R" -format csv -csv-delimiter comma > "$D/$a.summary.csv" 2>> "$D/$a.report.err"
        if [ "$a" = ua ]; then
          "$VTUNE" -report hotspots -result-dir "$R" -group-by function -format csv -csv-delimiter comma \
            > "$D/ua.functions.csv" 2>> "$D/$a.report.err"
          "$VTUNE" -report hw-events -result-dir "$R" -group-by function -format csv -csv-delimiter comma \
            > "$D/ua.hwevents.csv" 2>> "$D/$a.report.err"
        fi
        [ -s "$D/$a.summary.txt" ] || { fail "$label $coll: empty summary (see $D/$a.report.err)"; continue; }
        [ "$VTUNE_KEEP" = 1 ] || rm -rf "$R"
        log "$label $coll done"
      done
    done
  done
  idx=$((idx + 1))
done

# ----------------------------------------------------------------- analysis
if python3 "$ROOT/scripts/vtune_breakdown.py" "$OUT" --out "$OUT/analysis" >> "$OUT/driver.log" 2>&1; then
  log "analysis: $OUT/analysis (breakdown.md first)"
else fail "vtune_breakdown.py (see $OUT/driver.log)"; fi
log "results: $OUT"
if [ "$FAILS" -gt 0 ]; then log "$FAILS FAILURE(S)"; exit 1; fi
log "ALL STEPS PASSED"
