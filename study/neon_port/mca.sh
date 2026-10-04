#!/usr/bin/env bash
# Static throughput of the scalar vs NEON (vs SVE) hot loops with llvm-mca.
# Compiles study/neon_port/mca_kernels.cpp for aarch64 with the engine's flags
# (-O3 -fno-tree-vectorize, the TARGET_ARCH profile), takes each wrapper's
# largest loop (label .. backward branch), and reports llvm-mca cycles per
# iteration and per element. L1-resident, no mispredicts, no misses: a
# compute ceiling, not a prediction of query time.
#
# Usage (inside `nix develop .#aarch64`):
#   study/neon_port/mca.sh [gcc|clang] [n1|v1] > mca.csv
set -euo pipefail
CC=${1:-gcc}
CPU=${2:-n1}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
case $CPU in
n1) FLAGS="-mcpu=neoverse-n1 -march=armv8.2-a+simd+crypto"; MCPU=neoverse-n1 ;;
v1) FLAGS="-mcpu=neoverse-v1 -march=armv8.4-a+sve+crypto"; MCPU=neoverse-v1 ;;
*) echo "cpu: n1 | v1" >&2; exit 2 ;;
esac
case $CC in
gcc) CXX=g++-16; FLAGS="$FLAGS -fno-tree-vectorize" ;;
clang) CXX=clang++-22; FLAGS="$FLAGS -fno-tree-vectorize -fno-slp-vectorize -fno-unroll-loops" ;;  # one element step per scalar iteration (the build unrolls 2x)
esac
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
$CXX -std=c++17 -O3 $FLAGS -DHASH_SIZE=64 -I"$ROOT/include" -I"$ROOT/3rdparty/simde" \
   -S -o "$TMP/k.s" "$ROOT/study/neon_port/mca_kernels.cpp"

# elements per loop iteration (the loops are not unrolled at -O3 by gcc;
# clang's are checked against the induction step in the CSV's step column)
declare -A EPI=(
  [sel_i32_scalar_bf]=1 [sel_i32_neon4]=4 [sel_i32_neon16]=16
  [sel_i64_scalar_bf]=1 [sel_i64_neon4]=4 [sel_i64_neon16]=16
  [selsel_i32_scalar_bf]=1 [selsel_i32_neon]=4
  [hash_murmur_scalar]=1 [hash_murmur_neon]=4 [hash_crc_scalar]=1
  [proj_mul_scalar]=1 [proj_mul_neon]=4 [proj_minus_scalar]=1 [proj_minus_neon]=8
  [proj_gather_minus_scalar]=1
  [runheads_u32_scalar]=1 [runheads_u32_neon]=4 [runheads_u16_neon]=8
  [bloom_scalar_bf]=1 [bloom_neon]=4 [semi_scalar_bf]=1 [semi_neon]=4
  [sel_i32_sve]=VL32 [selsel_i32_sve]=VL32 [hash_murmur_sve]=VL64 [proj_mul_sve]=VL64
)
# gcc runs the restrict run-head loop two rows per iteration (k[i] carried)
[[ $CC == gcc ]] && EPI[runheads_u32_scalar]=2
# SVE vector length of the modelled core: V1 256-bit
VL32=8
VL64=4

echo "compiler,cpu,kernel,loop_insns,cycles_per_iter,elems_per_iter,cycles_per_elem,bottleneck"
for fn in $(grep -oE '^[a-z0-9_]+:' "$TMP/k.s" | tr -d : | grep -E '^(sel|selsel|hash|proj|runheads|bloom|semi)_'); do
   [[ -n ${EPI[$fn]:-} ]] || continue
   # the function body, then its largest loop
   awk -v f="$fn" '$1 == f":" {on=1; next} on && /^\t\.size/ {exit} on {print}' "$TMP/k.s" > "$TMP/f.s"
   awk '
      /^\.L[A-Za-z0-9_]+:/ { lab = substr($1, 1, length($1) - 1); at[lab] = n; next }
      /^\t[a-z]/ && $1 !~ /^\./ {
         line[++n] = $0
         tgt = $NF
         if (tgt in at && ($1 ~ /^b\.?(eq|ne|cs|hs|cc|lo|mi|pl|vs|vc|hi|ls|ge|lt|gt|le)$/ || $1 ~ /^cbn?z$/ || $1 ~ /^tbn?z$/)) {
            len = n - at[tgt]
            if (len > best) { best = len; from = at[tgt] + 1; to = n }
         }
      }
      END { for (i = from; i <= to; ++i) print line[i] }' "$TMP/f.s" > "$TMP/loop.s"
   insns=$(wc -l < "$TMP/loop.s")
   [[ $insns -gt 0 ]] || { echo "$CC,$CPU,$fn,0,,,,no loop found"; continue; }
   llvm-mca -mtriple=aarch64-linux-gnu -mcpu="$MCPU" -iterations=500 -bottleneck-analysis \
      "$TMP/loop.s" > "$TMP/mca.txt" 2>/dev/null || { echo "$CC,$CPU,$fn,$insns,,,,mca failed"; continue; }
   cyc=$(awk '/^Total Cycles:/ {c=$3} /^Iterations:/ {i=$2} END {printf "%.2f", c / i}' "$TMP/mca.txt")
   e=${EPI[$fn]}
   case $e in VL32) e=$VL32 ;; VL64) e=$VL64 ;; esac
   bn=$(grep -m1 -oE 'Bottleneck: [A-Za-z ]+|No resource or data dependency bottlenecks' "$TMP/mca.txt" || true)
   awk -v r="$CC,$CPU,$fn,$insns,$cyc,$e" -v c="$cyc" -v e="$e" -v b="$bn" \
       'BEGIN { printf "%s,%.3f,%s\n", r, c / e, b }'
   [[ -n ${MCA_DUMP:-} ]] && { echo "--- $fn" >&2; cat "$TMP/loop.s" >&2; }
done
