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
#   join_dispatch        VW_JOIN_DISPATCH (JOIN_DISPATCH_STUDY.md): the semi /
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
#   grp_base grp_dispatch grp_runheads grp_having
#
# Group-by configs (VW_OPPORTUNITY_STUDY.md), on HASH_BASE + CRC32 + FAST:
#   grp_base             today's HashGroup
#   grp_batch            VW_GROUP_BATCH_CREATE
#   grp_global           VW_GROUP_GLOBAL_DIRECT
#   grp_q18              batch + global + VW_FUSE_HASH
#                        + VW_GROUP_NO_CONCAT + VW_SPILL_WORD_COPY
#   grp_all              grp_q18 + VW_AGGR_FUSED
#   grp_dispatch         VW_GROUP_DISPATCH (plan-resolved paths: batch, global,
#                        fused aggregates, spill copy, hash in the lookup)
#   grp_runheads         grp_dispatch + VW_GROUP_RUN_HEADS (branch-free run-head
#                        lookup; AVX-512BW/VL run pass, scalar elsewhere)
#   grp_having           grp_runheads + VW_GROUP_HAVING (TPC-H Q18: HAVING in
#                        the group-by instead of a Select over its output)
# Configs whose ISA this CPU lacks are skipped (their builds would fall back
# and duplicate another config).
#
# For every compiler x config it builds run_tpch + test_all into
# build_flags/<host>/<compiler>_<config>, runs the TPC-H correctness tests, then
# times run_tpch -e v on QUERIES at 1 thread pinned to one CPU, ROUNDS rounds
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
#   DATADIR    /tank/alexb/swole/          test_all reads $DATADIR/tpch/sf1/
#   TPCH_PATH  /tank/alexb/swole/tpch/sf1  run_tpch -p
#   QUERIES    1,3,6,9,18
#   REPS       30        run_tpch -r per invocation
#   ROUNDS     3
#   SETTLE     2         run_tpch -s
#   CPU        0
#   NODE       NUMA node of CPU (from sysfs)
#   NUMA       "numactl --physcpubind=$CPU --membind=$NODE"; taskset -c $CPU
#              without numactl; "" = no pinning
#   TEST_THREADS 4       worker threads for test_all
#   JOBS       $(nproc)
#   SKIP_BUILD 0         1 = reuse build dirs
#   TIMEOUT    1800      seconds per step (0 = none)
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
GROUP_CONFIGS="grp_base grp_batch grp_global grp_q18 grp_all grp_dispatch grp_runheads grp_having"
CONFIGS=${CONFIGS:-"$FLAG_CONFIGS $HASH_CONFIGS $JOIN_CONFIGS $GROUP_CONFIGS"}
# config group: the TPC-valid join and group-by configs (see header)
JOIN_VALID_CONFIGS="join_base join_twophase join_simd nj_tag nj_occ join_bloom nj_bloom join_fused join_semi join_all join_dispatch jd_all join_all_valid grp_base grp_dispatch grp_runheads grp_having"
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
TPCH_PATH=${TPCH_PATH:-/tank/alexb/swole/tpch/sf1}
QUERIES=${QUERIES:-1,3,6,9,18}
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
TEST_THREADS=${TEST_THREADS:-4}
JOBS=${JOBS:-$(nproc)}
SKIP_BUILD=${SKIP_BUILD:-0}
TIMEOUT=${TIMEOUT:-1800}

# matrix.csv: compiler,query,config,median_ms,best_ms,speedup_vs_default,speedup_vs_tuned
# median_ms is the median over rounds of the per-invocation medians, so one
# disturbed invocation (seen on dubliner: 2-3x slower rounds) does not skew it.
# The header is printed outside sort so it stays on line 1.
summarize() {
  awk -F, '
    function median(str,   a, n, i, j, t) {
      n = split(str, a, " ")
      for (i = 2; i <= n; i++) { t = a[i] + 0; for (j = i - 1; j >= 1 && a[j] + 0 > t; j--) a[j + 1] = a[j]; a[j + 1] = t }
      return (n % 2) ? a[(n + 1) / 2] : (a[n / 2] + a[n / 2 + 1]) / 2
    }
    NR > 1 { k = $1 "," $4 "," $2; v[k] = v[k] " " $5; if (!(k in mn) || $6 < mn[k]) mn[k] = $6 }
    END { for (k in v) m[k] = median(v[k])
          for (k in m) { split(k, p, ","); d = p[1] "," p[2] ",default"; t = p[1] "," p[2] ",tuned"
            printf "%s,%.2f,%.2f,%s,%s\n", k, m[k], mn[k],
                   (d in m) ? sprintf("%.3f", m[d] / m[k]) : "", (t in m) ? sprintf("%.3f", m[t] / m[k]) : "" } }' \
    "$OUT/timing.csv" | sort -t, -k1,1 -k2,2V -k3,3 > "$OUT/matrix.body"
  { echo "compiler,query,config,median_ms,best_ms,speedup_vs_default,speedup_vs_tuned"; cat "$OUT/matrix.body"; } > "$OUT/matrix.csv"

  # effects.csv: add_speedup = default / add_<flag>, drop_speedup = drop_<flag> / tuned
  { echo "compiler,query,flag,add_speedup,drop_speedup"
    awk -F, '{ m[$1 "," $2 "," $3] = $4; cq[$1 "," $2] = 1 }
      END { split("group_aggr group_aggr_sel pos16 crc32 huge2mb", fl, " ")
            for (k in cq) for (i = 1; i <= 5; i++) { f = fl[i]
              d = m[k ",default"]; a = m[k ",add_" f]; t = m[k ",tuned"]; x = m[k ",drop_" f]
              printf "%s,%s,%s,%s\n", k, f, (d && a) ? sprintf("%.3f", d / a) : "", (t && x) ? sprintf("%.3f", x / t) : "" } }' \
      "$OUT/matrix.body" | sort -t, -k1,1 -k2,2V; } > "$OUT/effects.csv"

  # best.csv: fastest config per compiler x query (by median_ms)
  { echo "compiler,query,best_config,median_ms,vs_default,vs_tuned"
    awk -F, '{ k = $1 "," $2; if (!(k in b) || $4 < b[k]) { b[k] = $4; c[k] = $3; d[k] = $6; t[k] = $7 } }
      END { for (k in b) printf "%s,%s,%.2f,%s,%s\n", k, c[k], b[k], d[k], t[k] }' "$OUT/matrix.body" | sort -t, -k1,1 -k2,2V; } > "$OUT/best.csv"

  # hash.csv: hash configs, speedup vs hash_murmur (the default hash)
  { echo "compiler,query,hash_config,median_ms,speedup_vs_murmur"
    awk -F, '$3 ~ /^hash_/ { m[$1 "," $2 "," $3] = $4; cq[$1 "," $2] = 1 }
      END { n = split("hash_murmur hash_simd hash_crc32 hash_crc32_fast hash_crc32_vpclmul", hc, " ")
            for (k in cq) { b = m[k ",hash_murmur"]
              for (i = 1; i <= n; i++) if ((k "," hc[i]) in m)
                printf "%s,%s,%.2f,%s\n", k, hc[i], m[k "," hc[i]], b ? sprintf("%.3f", b / m[k "," hc[i]]) : "" } }' \
      "$OUT/matrix.body" | sort -t, -k1,1 -k2,2V; } > "$OUT/hash.csv"

  # join.csv: join configs, speedup vs join_base (today's probe, same hash)
  { echo "compiler,query,join_config,median_ms,speedup_vs_base"
    awk -F, '$3 ~ /^(join_|nj_|jd_)/ { m[$1 "," $2 "," $3] = $4; cq[$1 "," $2] = 1 }
      END { n = split("join_base join_twophase join_simd nj_tag nj_occ join_bloom nj_bloom join_fused join_semi join_all join_dispatch jd_all jd_semi join_all_valid", jc, " ")
            for (k in cq) { b = m[k ",join_base"]
              for (i = 1; i <= n; i++) if ((k "," jc[i]) in m)
                printf "%s,%s,%.2f,%s\n", k, jc[i], m[k "," jc[i]], b ? sprintf("%.3f", b / m[k "," jc[i]]) : "" } }' \
      "$OUT/matrix.body" | sort -t, -k1,1 -k2,2V; } > "$OUT/join.csv"

  # group.csv: group-by configs, speedup vs grp_base (today's HashGroup)
  { echo "compiler,query,group_config,median_ms,speedup_vs_base"
    awk -F, '$3 ~ /^grp_/ { m[$1 "," $2 "," $3] = $4; cq[$1 "," $2] = 1 }
      END { n = split("grp_base grp_batch grp_global grp_q18 grp_all grp_dispatch grp_runheads grp_having", gc, " ")
            for (k in cq) { b = m[k ",grp_base"]
              for (i = 1; i <= n; i++) if ((k "," gc[i]) in m)
                printf "%s,%s,%.2f,%s\n", k, gc[i], m[k "," gc[i]], b ? sprintf("%.3f", b / m[k "," gc[i]]) : "" } }' \
      "$OUT/matrix.body" | sort -t, -k1,1 -k2,2V; } > "$OUT/group.csv"
  rm -f "$OUT/matrix.body"
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
       "-DVW_GROUP_NO_CONCAT=OFF -DVW_SPILL_WORD_COPY=OFF -DVW_AGGR_FUSED=OFF" \
       "-DVW_GROUP_DISPATCH=OFF -DVW_GROUP_RUN_HEADS=OFF -DVW_JOIN_FUSED_PROBE=OFF -DVW_JOIN_SEMI=OFF" \
       "-DVW_JOIN_DISPATCH=OFF -DVW_GROUP_HAVING=OFF -DVW_JOIN_SEMI_MAX_BYTES=0" \
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
    grp_runheads)        echo "$(config_flags grp_dispatch) -DVW_GROUP_RUN_HEADS=ON" ;;
    grp_having)          echo "$(config_flags grp_runheads) -DVW_GROUP_HAVING=ON" ;;
    *) echo "unknown config $1" >&2; exit 2 ;;
  esac
}
bdir() { echo "$ROOT/build_flags/$BENCH_HOST/$1_$2"; }
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
  echo "QUERIES: $QUERIES  REPS: $REPS  ROUNDS: $ROUNDS"
} | tee "$OUT/machine.txt"
[ "$MACHINE" = custom ] && log "WARNING: host $(hostname -s) is not a TARGET_MACHINE preset; building with MACHINE=custom (-march=native, default topology). Set MACHINE= to override."

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
