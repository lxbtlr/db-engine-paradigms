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
# Hash configs: the hash function on top of one base config (HASH_BASE):
#   hash_murmur          default MurMurHash
#   hash_simd            VW_SIMD_HASH (AVX-512 MurMurHash; x86 AVX-512F/DQ)
#   hash_crc32           VW_USE_CRC32
#   hash_crc32_fast      VW_USE_CRC32 + VW_CRC32_FAST
#   hash_crc32_vpclmul   VW_USE_CRC32 + VW_CRC32_FAST + VW_CRC32_VPCLMUL
#                        (needs VPCLMULQDQ: Ice Lake+, Sapphire Rapids, Zen 4)
#
# Join configs: the hash join probe on top of HASH_BASE + VW_USE_CRC32 +
# VW_CRC32_FAST (the recommended hash):
#   join_base            joinAllParallel / joinSelParallel (today)
#   join_twophase        VW_JOIN_TWOPHASE (group prefetch, run_joinbench J2)
#   join_simd            VW_JOIN_SIMD (AVX-512 probes, run_joinbench J4;
#                        x86 AVX-512F)
#   nj_tag               VW_NEW_JOIN, full = tag hit (andrew_pseudocode.md)
#   nj_occ               VW_NEW_JOIN, full = slot occupied
#   join_bloom           VW_JOIN_BLOOM (VW-owned Bloom filter before the probe)
#   nj_bloom             VW_NEW_JOIN + VW_JOIN_BLOOM
#   join_fused           VW_JOIN_BLOOM + VW_JOIN_FUSED_PROBE (probe hashing in the
#                        filter pass)
#   join_semi            VW_JOIN_SEMI (bitmap semi join on plan-marked joins)
#   join_all             one config for every machine: join_fused + join_semi +
#                        VW_GROUP_DISPATCH + VW_GROUP_RUN_HEADS
#   join_dispatch        VW_JOIN_DISPATCH (study/JOIN_DISPATCH_STUDY.md): the semi /
#                        Bloom / fused-probe choice resolved once per join from
#                        plan facts and the build summary; also builds and runs
#                        run_joindispatchbench (joindispatchbench.csv)
#   jd_all               join_dispatch + VW_GROUP_HAVING + VW_GROUP_DISPATCH +
#                        VW_GROUP_RUN_HEADS (compare with join_all_valid).
#                        join_dispatch and jd_all have no semi bitmap
#                        (VW_JOIN_SEMI_MAX_BYTES=0, a structure Typer lacks)
#   jd_semi              jd_all + the semi bitmap (VW_JOIN_SEMI_MAX_BYTES=8 MiB)
#   join_all_valid       join_all without VW_JOIN_SEMI: join_fused +
#                        VW_GROUP_DISPATCH + VW_GROUP_RUN_HEADS
#
# CONFIGS may contain the group name join_valid, which expands to the configs
# AUDIT_TPC_OPTIONS.md finds TPC-valid for every query (no VW_Q9_FIELD_ORDER;
# VW_PROJ_DENSE was removed). The semi-join configs are included: semi joins
# are derived from the declared primary keys (QueryBuilder::uniqueBuild), not
# asserted by the plans:
#   join_base join_twophase join_simd nj_tag nj_occ join_bloom nj_bloom
#   join_fused join_semi join_all join_dispatch jd_all join_all_valid
#   grp_base grp_dispatch grp_kinds grp_runheads grp_having
#
# Group-by configs (study/VW_OPPORTUNITY_STUDY.md), on HASH_BASE + CRC32 + FAST:
#   grp_base             today's HashGroup
#   grp_batch            VW_GROUP_BATCH_CREATE
#   grp_global           VW_GROUP_GLOBAL_DIRECT
#   grp_q18              batch + global + VW_FUSE_HASH
#                        + VW_GROUP_NO_CONCAT + VW_SPILL_WORD_COPY
#   grp_all              grp_q18 + VW_AGGR_FUSED
#   grp_dispatch         VW_GROUP_DISPATCH (plan-resolved paths: batch, global,
#                        fused aggregates, spill copy, hash in the lookup)
#   grp_kinds            grp_dispatch + VW_AGGR_FUSED_KINDS (fused aggregate
#                        pass specialized by aggregate kind)
#   grp_runheads         grp_dispatch + VW_GROUP_RUN_HEADS (branch-free run-head
#                        lookup; AVX-512BW/VL run pass, scalar elsewhere)
#   grp_having           grp_runheads + VW_GROUP_HAVING (TPC-H Q18: HAVING in
#                        the group-by instead of a Select over its output)
# Configs whose ISA this CPU lacks are skipped (their builds would fall back
# and duplicate another config).
#
# For every compiler x config it builds run_tpch (run_ssb with BENCH=ssb) +
# test_all into build_flags/<host>/<compiler>_<config>, runs the TPC-H (SSB)
# correctness tests, then times the runner -e v on QUERIES at 1 thread pinned to one CPU, ROUNDS rounds
# with all builds alternating in each round.
# Output: results/flag_ablation_<timestamp>/
#   timing.csv   one row per run_tpch invocation
#   matrix.csv   mean-of-medians ms per compiler x query x config
#   effects.csv  per flag: add_speedup = default/add_<flag>, drop_speedup =
#                drop_<flag>/tuned (both > 1 means the flag helps that query)
#   best.csv     fastest config per compiler x query
#   hash.csv     hash configs: median ms and speedup vs hash_murmur
#   join.csv     join configs: median ms and speedup vs join_base
#   group.csv    group-by configs: median ms and speedup vs grp_base
#   noise.csv    per config x query: rounds, spread of the round medians, and
#                the median within-invocation stddev/mean (cv)
#   counters.csv every run_tpch column per invocation, long form
#                (compiler,config,round,query,metric,value: IPC, LLC-misses, ...)
#
# Environment (all optional):
#   COMPILERS  "gcc clang"
#   CONFIGS    all 12 flag configs + the 5 hash configs (see config_flags);
#              e.g. CONFIGS="$(echo hash_{murmur,simd,crc32,crc32_fast,crc32_vpclmul})"
#   HASH_BASE  default   flag config the hash configs build on (e.g. tuned,
#                        drop_group_aggr); its crc32 setting is overridden
#   SEL        by CPU: AVX-512VL "-DVW_SIMD_SEL=ON -DVW_SIMD_SEL_WIDTH=256 -DVW_SIMD_SEL_COMPRESS=reg",
#              AVX-512F only "-DVW_SIMD_SEL=ON -DVW_SIMD_SEL_WIDTH=512 -DVW_SIMD_SEL_COMPRESS=reg",
#              otherwise (Zen 3, ARM, ...) "-DVW_SIMD_SEL=OFF"
#   MACHINE    $(hostname -s) if it is a preset (dubliner, roquefort, manchego, burrata, kafir, rpi5), else custom
#   BENCH      tpch      tpch (run_tpch, TPCH.* tests) or ssb (run_ssb, SSB.*
#                        tests); ssb adds _ssb to every config name
#   DATADIR    /tank/alexb/swole/          test_all reads $DATADIR/<BENCH>/sf1/
#   SF         1         scale factor: TPCH_PATH/SSB_PATH default .../<BENCH>/sf$SF
#   TPCH_PATH  /tank/alexb/swole/tpch/sf$SF  run_tpch -p (BENCH=tpch)
#   SSB_PATH   /tank/alexb/swole/ssb/sf$SF   run_ssb -p (BENCH=ssb)
#   QUERIES    tpch: 1,3,6,9,18; ssb: 11,12,13,21,22,23,31,32,33,34,41,42,43
#              (run_ssb also takes 1.1 ... 4.3; unknown ids fail the run)
#   REPS       30        run_tpch -r per invocation
#   ROUNDS     3
#   SETTLE     2         run_tpch -s
#   CPU        0
#   NODE       NUMA node of CPU (from sysfs)
#   NUMA       pinning command; default from PIN:
#              cpu = "numactl --physcpubind=$CPU --membind=$NODE" (taskset -c $CPU without numactl)
#   PIN        cpu if THREADS=1, else none: cpu (NUMA default above; refused
#              with more than one thread), node
#              (numactl --cpunodebind=$NODE --membind=$NODE) or none; ignored if
#              NUMA is set
#   TEST_THREADS 4       worker threads for test_all
#   JOBS       $(nproc)
#   SKIP_BUILD 0         1 = reuse build dirs
#   TIMEOUT    1800      seconds per step (0 = none)
#   AUTOVEC    OFF       ON = -DAUTOVECTORIZE=ON for every build
#   ENGINE     v         run_tpch -e: v (Vectorwise), h (Hyper), b (Hyper Q6, branching)
#   THREADS    1         run_tpch -t: a count or a comma list (1,4,8), all
#                        in one invocation; each count is its own config _t<N>
#   VECTOR_SIZE 1024     run_tpch -v; also the vector size of test_all
#   SIMDhash SIMDjoin SIMDsel SIMDproj  0|1, unset = built-in default:
#              run-time SIMD primitive switches read by run_tpch and test_all
#   VW_FLAGS   ""        extra CMake options on every config, "VW_X=ON VW_Y=8":
#                        VW_* (and HUGE_2MB_MALLOC_HUGE) only; configs are
#                        named _x<hash>, vw_flags.txt maps the hash back
#   CPU        (above)   also part of the name when not 0 (_cpu<N>)
#   TESTS      all       TPCH/SSB tests: all, none, or a query list (1,6 or 11,21)
#   PERF       ""        stat or record: one perf run per config x query after
#                        the timing; <compiler>_<config>/perf_q<N>.txt
#                        (PERF_KEEP=1 keeps perf.data)
#   PERF_GROUP ""        G0..G8, G3a, G3b (+ G1b G2b G4b G5b G6b G8b on
#                        Cascade Lake / Zen 3): run_tpch counts that group
#                        (cycles + instr. + the group, scheduled together,
#                        with a <event>.run column) instead of the default
#                        counters; see PMU_EVENT_GROUPS.md
#   Each of these that differs from its default is appended to the config
#   name: <config>[_autovec][_x<hash>][_e<engine>][_t<threads>][_v<size>]
#   [_sf<SF>][_cpu<N>][_simdhash<0|1>]...[_pg<group>], so results and database entries
#   never collide; summaries compare configs with the same suffix.
#   Build dirs carry _autovec and _x<hash>. clearCaches (needs root) and the
#   canary (-c/-C, CANARY_IN_BENCH builds) are not exposed.
#   SUMMARIZE_ONLY <dir> recompute matrix/effects/best from <dir>/timing.csv and exit
#   RESULTS_PREFIX flag_ablation  results directory name before _<host>_<timestamp>
#   MICROBENCH 1         0 = skip run_joindispatchbench for join_dispatch
#   PRINT_CONFIGS 0      1 = print the expanded CONFIGS and exit
set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# per-host build/results paths: several machines may share this checkout
# over NFS, and must never write the same build tree
BENCH_HOST=${BENCH_HOST:-$(hostname -s 2>/dev/null || hostname)}
COMPILERS=${COMPILERS:-"gcc clang"}
FLAG_CONFIGS="default add_group_aggr add_group_aggr_sel add_pos16 add_crc32 add_huge2mb
             tuned drop_group_aggr drop_group_aggr_sel drop_pos16 drop_crc32 drop_huge2mb"
HASH_CONFIGS="hash_murmur hash_simd hash_crc32 hash_crc32_fast hash_crc32_vpclmul"
JOIN_CONFIGS="join_base join_twophase join_simd nj_tag nj_occ join_bloom nj_bloom join_fused join_semi join_all join_dispatch jd_all jd_semi join_all_valid"
GROUP_CONFIGS="grp_base grp_batch grp_global grp_q18 grp_all grp_dispatch grp_kinds grp_runheads grp_having"
CONFIGS=${CONFIGS:-"$FLAG_CONFIGS $HASH_CONFIGS $JOIN_CONFIGS $GROUP_CONFIGS"}
# config group: the TPC-valid join and group-by configs (see header)
JOIN_VALID_CONFIGS="join_base join_twophase join_simd nj_tag nj_occ join_bloom nj_bloom join_fused join_semi join_all join_dispatch jd_all join_all_valid grp_base grp_dispatch grp_kinds grp_runheads grp_having"
CONFIGS=$(for c in $CONFIGS; do if [ "$c" = join_valid ]; then printf "%s\n" $JOIN_VALID_CONFIGS; else echo "$c"; fi; done | awk '!seen[$0]++' | tr '\n' ' ')
# PRINT_CONFIGS=1: print the expanded config list and exit (single_build.sh)
if [ "${PRINT_CONFIGS:-0}" = 1 ]; then echo $CONFIGS; exit 0; fi
HASH_BASE=${HASH_BASE:-default}
# SIMD selection only where the CPU has the kernels' ISA, so a config never
# claims SIMD selection while silently running the scalar fallback
if [ -z "${SEL:-}" ]; then
  if grep -q -w avx512vl /proc/cpuinfo; then SEL="-DVW_SIMD_SEL=ON -DVW_SIMD_SEL_WIDTH=256 -DVW_SIMD_SEL_COMPRESS=reg"
  elif grep -q -w avx512f /proc/cpuinfo; then SEL="-DVW_SIMD_SEL=ON -DVW_SIMD_SEL_WIDTH=512 -DVW_SIMD_SEL_COMPRESS=reg"
  else SEL="-DVW_SIMD_SEL=OFF"; fi
fi
# MACHINE defaults to this host's short name when it is a TARGET_MACHINE
# preset, else "custom" (-march=native, CMake topology defaults).
detect_machine() {
  local h; h=$(hostname -s 2>/dev/null || hostname)
  case "$h" in dubliner|roquefort|manchego|burrata|kafir|rpi5) echo "$h" ;; *) echo custom ;; esac
}
MACHINE=${MACHINE:-$(detect_machine)}
DATADIR=${DATADIR:-/tank/alexb/swole/}
SF=${SF:-1}
BENCH=${BENCH:-tpch}
case "$BENCH" in
  tpch) RUNNER=run_tpch; BENCH_NAME=TPC-H; TEST_PREFIX=TPCH
        TPCH_PATH=${TPCH_PATH:-/tank/alexb/swole/tpch/sf$SF}; DATA_PATH=$TPCH_PATH
        QUERIES=${QUERIES:-1,3,6,9,18} ;;
  ssb)  RUNNER=run_ssb; BENCH_NAME=SSB; TEST_PREFIX=SSB
        SSB_PATH=${SSB_PATH:-/tank/alexb/swole/ssb/sf$SF}; DATA_PATH=$SSB_PATH
        QUERIES=${QUERIES:-11,12,13,21,22,23,31,32,33,34,41,42,43} ;;
  *) echo "BENCH must be tpch or ssb" >&2; exit 2 ;;
esac
ENGINE=${ENGINE:-v}
THREADS=${THREADS:-1}
VECTOR_SIZE=${VECTOR_SIZE:-1024}
PIN=${PIN:-$([ "$THREADS" = 1 ] && echo cpu || echo none)}
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
# PIN=cpu: numactl (CPU + its local memory), else taskset (CPU only);
# PIN=node: numactl on CPU's NUMA node (cores + memory); PIN=none: no pinning
NUMA_SET=${NUMA+1} # NUMA given explicitly: PIN is ignored
if [ -z "${NUMA+x}" ]; then
  case "$PIN" in
    cpu) if command -v numactl > /dev/null; then NUMA="numactl --physcpubind=$CPU --membind=$NODE"
         elif command -v taskset > /dev/null; then NUMA="taskset -c $CPU"
         else NUMA=""; fi ;;
    node) command -v numactl > /dev/null || { echo "PIN=node needs numactl" >&2; exit 2; }
          NUMA="numactl --cpunodebind=$NODE --membind=$NODE" ;;
    none) NUMA="" ;;
    *) echo "PIN must be cpu, node or none" >&2; exit 2 ;;
  esac
fi
TEST_THREADS=${TEST_THREADS:-4}
JOBS=${JOBS:-$(nproc)}
SKIP_BUILD=${SKIP_BUILD:-0}
TIMEOUT=${TIMEOUT:-1800}
AUTOVEC=${AUTOVEC:-OFF}
case "$AUTOVEC" in ON|OFF) ;; *) echo "AUTOVEC must be ON or OFF" >&2; exit 2 ;; esac
case "$ENGINE" in v|h|b) ;; *) echo "ENGINE must be v, h or b" >&2; exit 2 ;; esac
# THREADS: one count or a comma list (run_tpch -t runs each in one invocation)
case "$THREADS" in ''|*[!0-9,]*|,*|*,|*,,*) echo "THREADS must be a count or a comma list of counts" >&2; exit 2 ;; esac
for t in ${THREADS//,/ }; do
  case "$t" in 0*) echo "THREADS: $t is not a positive count" >&2; exit 2 ;; esac
done
[ -n "$(tr , '\n' <<< "$THREADS" | sort | uniq -d)" ] && { echo "THREADS repeats a count" >&2; exit 2; }
[ "$PIN" = cpu ] && [ "$THREADS" != 1 ] && [ -z "${NUMA_SET:-}" ] &&
  { echo "PIN=cpu pins every thread to CPU $CPU; use PIN=node or none with THREADS=$THREADS" >&2; exit 2; }
for n in VECTOR_SIZE SF; do
  case "${!n}" in ''|*[!0-9]*|0) echo "$n must be a positive integer" >&2; exit 2 ;; esac
done
case "$CPU" in ''|*[!0-9]*) echo "CPU must be a CPU number" >&2; exit 2 ;; esac
[ -d "/sys/devices/system/cpu/cpu$CPU" ] || { echo "CPU $CPU does not exist on $(hostname -s)" >&2; exit 2; }
# TESTS: all (default), none, or a comma list of queries to test
# (TPCH.q<N> or SSB.q<N> by BENCH)
TESTS=${TESTS:-all}
case "$TESTS" in
  all) TEST_FILTER="$TEST_PREFIX.*" ;;
  none) TEST_FILTER= ;;
  *) case "$TESTS" in *[!0-9,]*|,*|*,|*,,*) echo "TESTS must be all, none or a comma list of queries" >&2; exit 2 ;; esac
     TEST_FILTER=$(echo "$TESTS" | sed "s/[0-9]*/$TEST_PREFIX.q&/g; s/,/:/g") ;;
esac
# PERF: profile each config x query once, after the timing: stat (perf stat
# -d) or record (perf record -g, report kept as text; PERF_KEEP=1 keeps
# perf.data too)
PERF=${PERF:-}
case "$PERF" in ''|stat|record) ;; *) echo "PERF must be stat or record" >&2; exit 2 ;; esac
# VW_FLAGS: extra CMake options applied on top of every config, last -D wins
# ("VW_JOIN_BLOOM=ON VW_JOIN_SEMI_MAX_BYTES=4194304"). Configs are then named
# <config>_x<hash> after the sorted option list (vw_flags.txt maps it back).
VW_FLAGS=${VW_FLAGS:-}
VW_FLAGS_D= VW_FLAGS_HASH=
if [ -n "$VW_FLAGS" ]; then
  for kv in $VW_FLAGS; do
    [[ $kv =~ ^(VW_[A-Z0-9_]+|HUGE_2MB_MALLOC_HUGE)=[A-Za-z0-9_.]+$ ]] ||
      { echo "VW_FLAGS: $kv is not OPTION=VALUE with a VW_ option" >&2; exit 2; }
  done
  [ -n "$(tr ' ' '\n' <<< "$VW_FLAGS" | grep . | cut -d= -f1 | sort | uniq -d)" ] &&
    { echo "VW_FLAGS sets an option twice" >&2; exit 2; }
  VW_FLAGS=$(tr ' ' '\n' <<< "$VW_FLAGS" | grep . | sort | tr '\n' ' ' | sed 's/ $//')
  VW_FLAGS_D=$(sed 's/[^ ]*/-D&/g' <<< "$VW_FLAGS")
  VW_FLAGS_HASH=$(printf '%s' "$VW_FLAGS" | sha1sum | cut -c1-8)
fi
# run_tpch and test_all read these from the environment (configFromEnv)
export vectorSize=$VECTOR_SIZE
for n in SIMDhash SIMDjoin SIMDsel SIMDproj; do
  case "${!n:-}" in ''|0|1) ;; *) echo "$n must be 0 or 1" >&2; exit 2 ;; esac
done
# run_tpch reads PERF_GROUP from the environment (profile.hpp)
case "${PERF_GROUP:-}" in ''|G[0-8]|G3a|G3b|G[124568]b) ;; *) echo "PERF_GROUP must be G0..G8, G3a, G3b or G1b, G2b, G4b, G5b, G6b, G8b" >&2; exit 2 ;; esac
[ -n "${PERF_GROUP:-}" ] && export PERF_GROUP
# Every setting that differs from the default is part of the config name
# (results, build dirs, database), so runs never collide:
# <config>[_autovec][_x<VW_FLAGS hash>][_e<engine>][_t<threads>][_v<vector size>]
# [_sf<SF>][_cpu<CPU>][_simd<x>]...
# With THREADS other than 1, each timing row is named with its own count
# (_t<N>) and the per-build directory with all of them (_t1-4-8). The build
# dir only depends on AUTOVEC and VW_FLAGS (the rest are run-time settings).
AV_SUFFIX=$([ "$AUTOVEC" = ON ] && echo _autovec)${VW_FLAGS_HASH:+_x$VW_FLAGS_HASH}
PRE_SUFFIX=$AV_SUFFIX
[ "$ENGINE" = v ] || PRE_SUFFIX+=_e$ENGINE
POST_SUFFIX=
[ "$BENCH" = tpch ] || POST_SUFFIX+=_$BENCH
[ "$VECTOR_SIZE" = 1024 ] || POST_SUFFIX+=_v$VECTOR_SIZE
[ "$SF" = 1 ] || POST_SUFFIX+=_sf$SF
[ "$CPU" = 0 ] || POST_SUFFIX+=_cpu$CPU
for n in SIMDhash SIMDjoin SIMDsel SIMDproj; do
  [ -n "${!n:-}" ] && POST_SUFFIX+=_$(echo "$n" | tr '[:upper:]' '[:lower:]')${!n}
done
[ -n "${PERF_GROUP:-}" ] && POST_SUFFIX+=_pg$(echo "$PERF_GROUP" | tr '[:upper:]' '[:lower:]')
TN=$([ "$THREADS" = 1 ] || echo 1) # 1: name timing rows _t<N>
RUN_SUFFIX=$PRE_SUFFIX${TN:+_t${THREADS//,/-}}$POST_SUFFIX

# matrix.csv: compiler,query,config,median_ms,best_ms,speedup_vs_default,speedup_vs_tuned
# median_ms is the median over rounds of the per-invocation medians, so one
# disturbed invocation (seen on dubliner: 2-3x slower rounds) does not skew it.
# A config name is a known config plus the suffix of its run settings
# (_autovec, _t4, ...; see RUN_SUFFIX). Every comparison pairs configs with
# the same suffix: jd_all_t4 is compared with join_base_t4, never join_base.
# noise.csv: per compiler x query x config, the spread of the round medians
# ((max - min) / median) and the median within-invocation coefficient of
# variation (stddev / mean, from counters.csv); either above 0.05 means the
# median is not trustworthy to a few percent.
# The header is printed outside sort so it stays on line 1.
KNOWN_CONFIGS="$FLAG_CONFIGS $HASH_CONFIGS $JOIN_CONFIGS $GROUP_CONFIGS"
summarize() {
  : > "$OUT/matrix.body"; : > "$OUT/noise.body"; : > "$OUT/effects.body"
  awk -F, -v known="$(echo $KNOWN_CONFIGS)" -v out="$OUT" -v counters="$OUT/counters.csv" '
    function median(str,   a, n, i, j, t) {
      n = split(str, a, " ")
      for (i = 2; i <= n; i++) { t = a[i] + 0; for (j = i - 1; j >= 1 && a[j] + 0 > t; j--) a[j + 1] = a[j]; a[j + 1] = t }
      return (n % 2) ? a[(n + 1) / 2] : (a[n / 2] + a[n / 2 + 1]) / 2
    }
    # base(c): the longest known config that c is, or starts with plus "_";
    # sets SUF to the rest (the run-setting suffix)
    function base(c,   i, k, best) {
      best = ""
      for (i = 1; i <= nk; i++) { k = K[i]
        if ((c == k || substr(c, 1, length(k) + 1) == k "_") && length(k) > length(best)) best = k }
      SUF = best == "" ? "" : substr(c, length(best) + 1)
      return best
    }
    function sp(b, x) { return (b != "" && x + 0 > 0) ? sprintf("%.3f", b / x) : "" }
    # one family table: rows in family order, each vs its base with the same suffix
    function family(file, hdr, list, b,   n, F, k, s, i, c, cmd, rows, nr) {
      n = split(list, F, " ")
      print hdr > file
      nr = 0
      for (k in cqs) { split(k, p, SUBSEP); s = p[3]
        for (i = 1; i <= n; i++) { c = F[i] s
          if ((p[1] "," p[2] "," c) in m)
            rows[++nr] = p[1] "," p[2] "," s "," i "," c "," sprintf("%.2f", m[p[1] "," p[2] "," c]) "," sp(m[p[1] "," p[2] "," b s], m[p[1] "," p[2] "," c]) } }
      close(file)
      # sort on compiler, query, suffix, family order; drop the two sort keys
      cmd = "sort -t, -k1,1 -k2,2V -k3,3 -k4,4n | cut -d, -f1,2,5- >> \"" file "\""
      for (i = 1; i <= nr; i++) print rows[i] | cmd
      close(cmd)
    }
    BEGIN { nk = split(known, K, " ") }
    FILENAME == counters { if (FNR > 1 && ($5 == "stddev" || $5 == "mean")) cv[$1 "," $4 "," $2 "," $3, $5] = $6; next }
    FNR > 1 { k = $1 "," $4 "," $2; v[k] = v[k] " " $5; if (!(k in mn) || $6 < mn[k]) mn[k] = $6
              if (!(k in hi) || $5 > hi[k]) hi[k] = $5; if (!(k in lo) || $5 < lo[k]) lo[k] = $5; nr_[k]++
              if ((k "," $3, "mean") in cv && cv[k "," $3, "mean"] > 0) cvs[k] = cvs[k] " " cv[k "," $3, "stddev"] / cv[k "," $3, "mean"] }
    END {
      for (k in v) { m[k] = median(v[k]); split(k, p, ","); base(p[3]); cqs[p[1], p[2], SUF] = 1; sufs[SUF] = 1 }
      mf = out "/matrix.body"
      for (k in m) { split(k, p, ","); base(p[3])
        d = p[1] "," p[2] ",default" SUF; t = p[1] "," p[2] ",tuned" SUF
        printf "%s,%.2f,%.2f,%s,%s\n", k, m[k], mn[k], (d in m) ? sp(m[d], m[k]) : "", (t in m) ? sp(m[t], m[k]) : "" > mf }
      close(mf)
      nf = out "/noise.body"
      for (k in m) printf "%s,%.2f,%.2f,%d,%.3f,%s\n", k, m[k], mn[k], nr_[k], (m[k] > 0 ? (hi[k] - lo[k]) / m[k] : 0),
                          (k in cvs) ? sprintf("%.3f", median(cvs[k])) : "" > nf
      close(nf)
      # effects.csv: add_speedup = default / add_<flag>, drop_speedup = drop_<flag> / tuned
      ef = out "/effects.body"; nfl = split("group_aggr group_aggr_sel pos16 crc32 huge2mb", fl, " ")
      for (k in cqs) { split(k, p, SUBSEP); s = p[3]; cq = p[1] "," p[2]
        if (!((cq ",default" s) in m) && !((cq ",tuned" s) in m)) continue
        for (i = 1; i <= nfl; i++) { f = fl[i]
          d = m[cq ",default" s]; a = m[cq ",add_" f s]; t = m[cq ",tuned" s]; x = m[cq ",drop_" f s]
          printf "%s,%s,%s,%s\n", cq, f s, (d && a) ? sprintf("%.3f", d / a) : "", (t && x) ? sprintf("%.3f", x / t) : "" > ef } }
      close(ef)
      family(out "/hash.csv", "compiler,query,hash_config,median_ms,speedup_vs_murmur",
             "hash_murmur hash_simd hash_crc32 hash_crc32_fast hash_crc32_vpclmul", "hash_murmur")
      family(out "/join.csv", "compiler,query,join_config,median_ms,speedup_vs_base",
             "join_base join_twophase join_simd nj_tag nj_occ join_bloom nj_bloom join_fused join_semi join_all join_dispatch jd_all jd_semi join_all_valid", "join_base")
      family(out "/group.csv", "compiler,query,group_config,median_ms,speedup_vs_base",
             "grp_base grp_batch grp_global grp_q18 grp_all grp_dispatch grp_kinds grp_runheads grp_having", "grp_base")
    }' $([ -f "$OUT/counters.csv" ] && echo "$OUT/counters.csv") "$OUT/timing.csv"
  { echo "compiler,query,config,median_ms,best_ms,speedup_vs_default,speedup_vs_tuned"
    sort -t, -k1,1 -k2,2V -k3,3 "$OUT/matrix.body"; } > "$OUT/matrix.csv"
  { echo "compiler,query,config,median_ms,best_ms,rounds,round_spread,cv"
    sort -t, -k1,1 -k2,2V -k3,3 "$OUT/noise.body"; } > "$OUT/noise.csv"
  { echo "compiler,query,flag,add_speedup,drop_speedup"
    sort -t, -k1,1 -k2,2V "$OUT/effects.body" 2> /dev/null; } > "$OUT/effects.csv"
  # best.csv: fastest config per compiler x query (by median_ms)
  { echo "compiler,query,best_config,median_ms,vs_default,vs_tuned"
    awk -F, '{ k = $1 "," $2; if (!(k in b) || $4 < b[k]) { b[k] = $4; c[k] = $3; d[k] = $6; t[k] = $7 } }
      END { for (k in b) printf "%s,%s,%.2f,%s,%s\n", k, c[k], b[k], d[k], t[k] }' "$OUT/matrix.body" | sort -t, -k1,1 -k2,2V; } > "$OUT/best.csv"
  rm -f "$OUT/matrix.body" "$OUT/noise.body" "$OUT/effects.body"
}
# SUMMARIZE_ONLY=<results dir>: recompute the summaries from its timing.csv
if [ -n "${SUMMARIZE_ONLY:-}" ]; then OUT=$SUMMARIZE_ONLY; summarize
  column -t -s, "$OUT/effects.csv"; column -t -s, "$OUT/best.csv"; column -t -s, "$OUT/hash.csv"
  column -t -s, "$OUT/join.csv"; column -t -s, "$OUT/group.csv"; exit 0; fi

OUT="$ROOT/results/${RESULTS_PREFIX:-flag_ablation}_${BENCH_HOST}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
FAILS=0
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/driver.log"; }
fail() { log "FAIL: $*"; FAILS=$((FAILS + 1)); }
# excerpt FILE [N]: the last N lines of a failed step's log into driver.log
# and stdout, so the cause survives when only the job's stdout is kept
excerpt() { [ -s "$1" ] && tail -n "${2:-10}" "$1" | sed 's/^/    | /' | tee -a "$OUT/driver.log"; }
tlimit() { if [ "$TIMEOUT" -gt 0 ]; then timeout --kill-after=30 "$TIMEOUT" "$@"; else "$@"; fi; }

# flags <group_aggr> <group_aggr_sel> <pos16> <crc32> <huge2mb>
# Every option is passed explicitly (CMake caches them), including the other
# kernel options, so nothing leaks in from an earlier configure.
flags() {
  echo "-DVW_GROUP_AGGR=$1 -DVW_GROUP_AGGR_SEL=$2 -DVW_POS_16=$3 -DVW_USE_CRC32=$4 -DHUGE_2MB_MALLOC_HUGE=$5" \
       "-DVW_SIMD_SEL=OFF -DVW_SIMD_SEL_WIDTH=512 -DVW_SIMD_SEL_COMPRESS=reg -DVW_SIMD_SEL_UNROLL=1" \
       "-DVW_SIMD_SEL_GATHER=scalar -DVW_SIMD_SEL_CHAR=OFF -DVW_SIMD_HASH=OFF -DVW_JOIN_PREFETCH=OFF" \
       "-DVW_CRC32_FAST=OFF -DVW_CRC32_VPCLMUL=OFF -DVW_JOIN_TWOPHASE=OFF -DVW_JOIN_SIMD=OFF" \
       "-DVW_NEW_JOIN=OFF -DVW_NEW_JOIN_FULL=tag -DVW_JOIN_BLOOM=OFF -DVW_SIMD_HASH_GATHER=insert" \
       "-DVW_GROUP_BATCH_CREATE=OFF -DVW_GROUP_GLOBAL_DIRECT=OFF -DVW_FUSE_HASH=OFF" \
       "-DVW_GROUP_NO_CONCAT=OFF -DVW_SPILL_WORD_COPY=OFF -DVW_AGGR_FUSED=OFF -DVW_AGGR_FUSED_KINDS=OFF" \
       "-DVW_GROUP_DISPATCH=OFF -DVW_GROUP_RUN_HEADS=OFF -DVW_JOIN_FUSED_PROBE=OFF -DVW_JOIN_SEMI=OFF" \
       "-DVW_JOIN_DISPATCH=OFF -DVW_GROUP_HAVING=OFF -DVW_JOIN_SEMI_MAX_BYTES=0" \
       "-DAUTOVECTORIZE=$AUTOVEC" "$SEL" # last -D wins, so SEL overrides the SIMD_SEL defaults above
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
    # hash configs: HASH_BASE, then the hash options (last -D wins)
    hash_murmur)         echo "$(config_flags "$HASH_BASE") -DVW_USE_CRC32=OFF -DVW_SIMD_HASH=OFF" ;;
    hash_simd)           echo "$(config_flags "$HASH_BASE") -DVW_USE_CRC32=OFF -DVW_SIMD_HASH=ON" ;;
    hash_crc32)          echo "$(config_flags "$HASH_BASE") -DVW_USE_CRC32=ON" ;;
    hash_crc32_fast)     echo "$(config_flags "$HASH_BASE") -DVW_USE_CRC32=ON -DVW_CRC32_FAST=ON" ;;
    hash_crc32_vpclmul)  echo "$(config_flags "$HASH_BASE") -DVW_USE_CRC32=ON -DVW_CRC32_FAST=ON -DVW_CRC32_VPCLMUL=ON" ;;
    # join configs: HASH_BASE + CRC32 + FAST, then the probe option
    join_base)           echo "$(config_flags "$HASH_BASE") -DVW_USE_CRC32=ON -DVW_CRC32_FAST=ON" ;;
    join_twophase)       echo "$(config_flags join_base) -DVW_JOIN_TWOPHASE=ON" ;;
    join_simd)           echo "$(config_flags join_base) -DVW_JOIN_SIMD=ON" ;;
    nj_tag)              echo "$(config_flags join_base) -DVW_NEW_JOIN=ON -DVW_NEW_JOIN_FULL=tag" ;;
    nj_occ)              echo "$(config_flags join_base) -DVW_NEW_JOIN=ON -DVW_NEW_JOIN_FULL=occupied" ;;
    join_bloom)          echo "$(config_flags join_base) -DVW_JOIN_BLOOM=ON" ;;
    nj_bloom)            echo "$(config_flags join_base) -DVW_NEW_JOIN=ON -DVW_JOIN_BLOOM=ON" ;;
    join_fused)          echo "$(config_flags join_base) -DVW_JOIN_BLOOM=ON -DVW_JOIN_FUSED_PROBE=ON" ;;
    join_semi)           echo "$(config_flags join_base) -DVW_JOIN_SEMI=ON" ;;
    join_all)            echo "$(config_flags join_fused) -DVW_JOIN_SEMI=ON -DVW_GROUP_DISPATCH=ON -DVW_GROUP_RUN_HEADS=ON" ;;
    join_all_valid)      echo "$(config_flags join_fused) -DVW_GROUP_DISPATCH=ON -DVW_GROUP_RUN_HEADS=ON" ;;
    join_dispatch)       echo "$(config_flags join_base) -DVW_JOIN_DISPATCH=ON" ;;
    jd_all)              echo "$(config_flags join_dispatch) -DVW_GROUP_HAVING=ON -DVW_GROUP_DISPATCH=ON -DVW_GROUP_RUN_HEADS=ON" ;;
    jd_semi)             echo "$(config_flags jd_all) -DVW_JOIN_SEMI_MAX_BYTES=8388608" ;;
    # group-by configs: HASH_BASE + CRC32 + FAST, then the HashGroup options
    grp_base)            echo "$(config_flags "$HASH_BASE") -DVW_USE_CRC32=ON -DVW_CRC32_FAST=ON" ;;
    grp_batch)           echo "$(config_flags grp_base) -DVW_GROUP_BATCH_CREATE=ON" ;;
    grp_global)          echo "$(config_flags grp_base) -DVW_GROUP_GLOBAL_DIRECT=ON" ;;
    grp_q18)             echo "$(config_flags grp_base) -DVW_GROUP_BATCH_CREATE=ON -DVW_GROUP_GLOBAL_DIRECT=ON" \
                              "-DVW_FUSE_HASH=ON -DVW_GROUP_NO_CONCAT=ON -DVW_SPILL_WORD_COPY=ON" ;;
    grp_all)             echo "$(config_flags grp_q18) -DVW_AGGR_FUSED=ON" ;;
    grp_dispatch)        echo "$(config_flags grp_base) -DVW_GROUP_DISPATCH=ON" ;;
    grp_kinds)           echo "$(config_flags grp_dispatch) -DVW_AGGR_FUSED_KINDS=ON" ;;
    grp_runheads)        echo "$(config_flags grp_dispatch) -DVW_GROUP_RUN_HEADS=ON" ;;
    grp_having)          echo "$(config_flags grp_runheads) -DVW_GROUP_HAVING=ON" ;;
    *) echo "unknown config $1" >&2; exit 2 ;;
  esac
}
bdir() { echo "$ROOT/build_flags/$BENCH_HOST/$1_$2$AV_SUFFIX"; }
# reject unknown config names before building (exit inside $(config_flags)
# only leaves the subshell)
for cfg in $CONFIGS; do (config_flags "$cfg") > /dev/null || { rm -rf "$OUT"; exit 2; }; done
case " $FLAG_CONFIGS " in *" $HASH_BASE "*) ;; *)
  echo "HASH_BASE=$HASH_BASE is not a flag config ($(echo $FLAG_CONFIGS))" >&2; rm -rf "$OUT"; exit 2 ;; esac
# drop hash configs this CPU cannot run (their build would fall back to
# scalar and duplicate another config)
cpu_has() { grep -q -w "$1" /proc/cpuinfo; }
kept=""
for cfg in $CONFIGS; do
  case "$cfg" in
    hash_simd) cpu_has avx512f && cpu_has avx512dq || { log "skipping hash_simd: no AVX-512F/DQ on this CPU"; continue; } ;;
    hash_crc32_vpclmul) cpu_has vpclmulqdq && cpu_has avx512vl && cpu_has avx512bw && cpu_has avx512dq ||
                        { log "skipping hash_crc32_vpclmul: no VPCLMULQDQ + AVX-512 on this CPU"; continue; } ;;
    join_simd) cpu_has avx512f || { log "skipping join_simd: no AVX-512F on this CPU"; continue; } ;;
  esac
  kept="$kept $cfg"
done
CONFIGS=$kept

{
  echo "host: $(hostname)"
  echo "MACHINE: $MACHINE  CPU: $CPU  NODE: $NODE  pin: ${NUMA:-none}"
  echo "date: $(date -Is)"
  echo "git:  $(git -C "$ROOT" rev-parse --short HEAD) $(git -C "$ROOT" status --porcelain --untracked-files=no | wc -l) dirty tracked files"
  lscpu | grep -E 'Model name|Socket|Core|Thread|NUMA node\(s\)|MHz' || true
  echo -n "governor cpu$CPU: "; cat "/sys/devices/system/cpu/cpu$CPU/cpufreq/scaling_governor" 2>/dev/null || echo n/a
  echo "SEL: $SEL  (avx512: $(grep -o -w -E 'avx512(f|vl)' /proc/cpuinfo | sort -u | tr '\n' ' '))"
  echo "QUERIES: $QUERIES  REPS: $REPS  ROUNDS: $ROUNDS  AUTOVEC: $AUTOVEC"
  echo "BENCH: $BENCH ($RUNNER)  ENGINE: $ENGINE  THREADS: $THREADS  VECTOR_SIZE: $VECTOR_SIZE  SF: $SF ($DATA_PATH)  PIN: $PIN"
  echo "SIMDhash: ${SIMDhash:-}  SIMDjoin: ${SIMDjoin:-}  SIMDsel: ${SIMDsel:-}  SIMDproj: ${SIMDproj:-}  PERF_GROUP: ${PERF_GROUP:-none}  label suffix: ${RUN_SUFFIX:-none}"
  echo "VW_FLAGS: ${VW_FLAGS:-none}${VW_FLAGS_HASH:+ (x$VW_FLAGS_HASH)}  TESTS: $TESTS  PERF: ${PERF:-none}"
} | tee "$OUT/machine.txt"
[ -n "$VW_FLAGS" ] && echo "x$VW_FLAGS_HASH $VW_FLAGS" > "$OUT/vw_flags.txt"
[ "$MACHINE" = custom ] && log "WARNING: host $(hostname -s) is not a TARGET_MACHINE preset; building with MACHINE=custom (-march=native, default topology). Set MACHINE= to override."

# ------------------------------------------------------- build + correctness
for comp in $COMPILERS; do
  for cfg in $CONFIGS; do
    tag="${comp}_${cfg}$RUN_SUFFIX"; B=$(bdir "$comp" "$cfg"); D="$OUT/$tag"; mkdir -p "$D"
    log "===== $tag ====="
    if [ "$SKIP_BUILD" != 1 ]; then
      # shellcheck disable=SC2046
      if ! cmake -S "$ROOT" -B "$B" -DCMAKE_BUILD_TYPE=Release -DCOMPILER="$comp" \
            -DTARGET_MACHINE="$MACHINE" -DTARGET_ARCH= -DDATADIR="$DATADIR" \
            $(config_flags "$cfg") $VW_FLAGS_D > "$D/cmake.log" 2>&1; then
        fail "$tag cmake (see $D/cmake.log)"; excerpt "$D/cmake.log"; continue
      fi
      : > "$D/build.log"
      for tgt in $RUNNER test_all; do
        echo "### target $tgt" >> "$D/build.log"
        cmake --build "$B" -j "$JOBS" --target "$tgt" >> "$D/build.log" 2>&1 || { fail "$tag build of $tgt (see $D/build.log)"; excerpt "$D/build.log"; }
      done
    fi
    # the effective defines, to confirm each config is what it claims
    cat "$B/build.ninja" "$B/CMakeFiles/vectorwise.dir/flags.make" 2>/dev/null | grep -o -E -- '-D(VW_[A-Z0-9_]+|HUGE_2MB_MALLOC_HUGE)(=[^ ]*)?' | sort -u | tr '\n' ' ' > "$D/defines.txt"
    log "$tag defines: $(cat "$D/defines.txt")"
    if [ -z "$TEST_FILTER" ]; then log "$tag: $BENCH_NAME tests skipped (TESTS=none)"
    elif [ -x "$B/test_all" ] && [ -d "$DATADIR/$BENCH/sf1" ]; then
      if threads=$TEST_THREADS tlimit "$B/test_all" --gtest_filter="$TEST_FILTER" > "$D/${BENCH}_test.log" 2>&1; then
        log "$tag $BENCH_NAME: $(grep -E '^\[  PASSED  \]' "$D/${BENCH}_test.log")"
      else fail "$tag $BENCH_NAME tests (see $D/${BENCH}_test.log)"; excerpt "$D/${BENCH}_test.log" 15; fi
    elif [ ! -x "$B/test_all" ]; then fail "$tag: test_all not built"
    else log "$tag: no $DATADIR/$BENCH/sf1, $BENCH_NAME tests skipped"; fi
    # join_dispatch also runs the microbenchmark behind its rules (pinned,
    # once per compiler; it is the same for every config)
    if [ "$cfg" = join_dispatch ] && [ "${MICROBENCH:-1}" = 1 ]; then
      if [ "$SKIP_BUILD" = 1 ] || cmake --build "$B" -j "$JOBS" --target run_joindispatchbench >> "$D/build.log" 2>&1; then
        # shellcheck disable=SC2086
        if tlimit $NUMA "$B/run_joindispatchbench" > "$D/joindispatchbench.csv" 2> "$D/joindispatchbench.err"; then
          log "$tag run_joindispatchbench: $D/joindispatchbench.csv"
        else fail "$tag run_joindispatchbench (see $D/joindispatchbench.err)"; fi
      else fail "$tag build of run_joindispatchbench (see $D/build.log)"; fi
    fi
  done
done

# ------------------------------------------------------------------ timing
echo "compiler,config,round,query,median_ms,min_ms" > "$OUT/timing.csv"
echo "compiler,config,round,query,metric,value" > "$OUT/counters.csv"
if [ ! -d "$DATA_PATH" ]; then
  log "no data at $DATA_PATH, timing skipped"
else
  for q in ${QUERIES//,/ }; do
    for r in $(seq 1 "$ROUNDS"); do
      for comp in $COMPILERS; do
        for cfg in $CONFIGS; do
          B=$(bdir "$comp" "$cfg"); [ -x "$B/$RUNNER" ] || continue
          f="$OUT/${comp}_${cfg}$RUN_SUFFIX/q${q}_r$r"
          # shellcheck disable=SC2086
          if ! tlimit $NUMA "$B/$RUNNER" -p "$DATA_PATH" -e "$ENGINE" -q "$q" -r "$REPS" -t "$THREADS" -v "$VECTOR_SIZE" -s "$SETTLE" \
                > "$f.csv" 2> "$f.err"; then
            fail "${comp}_${cfg}$RUN_SUFFIX q$q round $r (see $f.err)"
            [ "$r" = 1 ] && excerpt "$f.err" 5
            continue
          fi
          # timing.csv gets median and min; counters.csv every run_tpch column
          # (IPC, LLC-misses, ...) in long form, named as in its header
          awk -F, -v c="$comp" -v g="$cfg$PRE_SUFFIX" -v post="$POST_SUFFIX" -v tn="$TN" \
              -v r="$r" -v e="$ENGINE" -v tf="$OUT/timing.csv" -v cf="$OUT/counters.csv" '
            { l = $1; gsub(/^ +| +$/, "", l) }
            l == "name" { for (i = 2; i <= NF; i++) { h[i] = $i; gsub(/^ +| +$/, "", h[i]); gsub(/[ .]+/, "-", h[i]); sub(/-$/, "", h[i]) } }
            # label "q6 v  t4": p[1] = query, p[3] = thread count
            l ~ "^q[0-9]+ " e { m = $2; n = $4; gsub(/ /, "", m); gsub(/ /, "", n); split(l, p, " ")
                              name = g (tn ? "_" p[3] : "") post
                              print c "," name "," r "," p[1] "," m "," n >> tf
                              for (i = 2; i <= NF; i++) { v = $i; gsub(/ /, "", v)
                                if (h[i] != "" && v != "") print c "," name "," r "," p[1] "," h[i] "," v >> cf } }' "$f.csv"
        done
      done
    done
    log "q$q done"
  done
fi

# ----------------------------------------------------------------- profiling
# after the timing, so profiling never disturbs it: one perf run per config x
# query, same settings, into <compiler>_<config>/perf_q<N>.txt
if [ -n "$PERF" ] && [ -d "$DATA_PATH" ]; then
  if ! command -v perf > /dev/null; then fail "PERF=$PERF: perf is not installed on $(hostname -s)"
  else
    for comp in $COMPILERS; do
      for cfg in $CONFIGS; do
        B=$(bdir "$comp" "$cfg"); [ -x "$B/$RUNNER" ] || continue
        D="$OUT/${comp}_${cfg}$RUN_SUFFIX"
        for q in ${QUERIES//,/ }; do
          run=("$B/$RUNNER" -p "$DATA_PATH" -e "$ENGINE" -q "$q" -r "$REPS" -t "$THREADS" -v "$VECTOR_SIZE" -s "$SETTLE")
          if [ "$PERF" = stat ]; then
            # shellcheck disable=SC2086
            tlimit $NUMA perf stat -d -o "$D/perf_q$q.txt" -- "${run[@]}" > /dev/null 2> "$D/perf_q$q.err" ||
              fail "${comp}_${cfg}$RUN_SUFFIX perf stat q$q (see $D/perf_q$q.err)"
          else
            # shellcheck disable=SC2086
            if tlimit $NUMA perf record -g -o "$D/perf_q$q.data" -- "${run[@]}" > /dev/null 2> "$D/perf_q$q.err"; then
              perf report --stdio --no-children --percent-limit 0.5 -i "$D/perf_q$q.data" > "$D/perf_q$q.txt" 2>> "$D/perf_q$q.err" ||
                fail "${comp}_${cfg}$RUN_SUFFIX perf report q$q (see $D/perf_q$q.err)"
              [ "${PERF_KEEP:-0}" = 1 ] || rm -f "$D/perf_q$q.data"
            else fail "${comp}_${cfg}$RUN_SUFFIX perf record q$q (see $D/perf_q$q.err)"; fi
          fi
        done
      done
    done
    log "perf $PERF: <compiler>_<config>/perf_q<N>.txt"
  fi
fi

# ---------------------------------------------------------------- summaries
summarize

log "per-flag effect (>1 = flag helps; add = vs default, drop = vs tuned):"
column -t -s, "$OUT/effects.csv" | tee -a "$OUT/driver.log"
log "best config per query:"
column -t -s, "$OUT/best.csv" | tee -a "$OUT/driver.log"
if [ "$(wc -l < "$OUT/hash.csv")" -gt 1 ]; then
  log "hash configs (base: $HASH_BASE), speedup vs the default MurMurHash:"
  column -t -s, "$OUT/hash.csv" | tee -a "$OUT/driver.log"
fi
if [ "$(wc -l < "$OUT/group.csv")" -gt 1 ]; then
  log "group-by configs (base: $HASH_BASE + CRC32 + FAST), speedup vs grp_base:"
  column -t -s, "$OUT/group.csv" | tee -a "$OUT/driver.log"
fi
if [ "$(wc -l < "$OUT/join.csv")" -gt 1 ]; then
  log "join configs (base: $HASH_BASE + CRC32 + FAST), speedup vs join_base:"
  column -t -s, "$OUT/join.csv" | tee -a "$OUT/driver.log"
fi
log "results: $OUT"
if [ "$FAILS" -gt 0 ]; then log "$FAILS FAILURE(S)"; exit 1; fi
log "ALL CHECKS PASSED"
