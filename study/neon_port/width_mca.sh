#!/usr/bin/env bash
# llvm-mca's prediction for each run_widthbench loop body, to set beside the
# measured IPC from the target. Usage (inside `nix develop .#aarch64`):
#   study/neon_port/width_mca.sh <run_widthbench> [mcpu, default neoverse-n1]
# The body dump runs the aarch64 binary under qemu.
set -euo pipefail
BIN=$1
MCPU=${2:-neoverse-n1}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
qemu-aarch64 "$BIN" -d | awk -v d="$TMP" '/^### / {f = d "/" $2 ".s"; next} {print > f}'
echo "test,mca_cycles_per_iter,mca_ipc,mca_dispatch_width"
for s in "$TMP"/*.s; do
   llvm-mca -mtriple=aarch64-linux-gnu -mcpu="$MCPU" -iterations=200 "$s" 2>/dev/null |
      awk -v t="$(basename "$s" .s)" '
         /^Iterations:/ {it = $2} /^Instructions:/ {ins = $2} /^Total Cycles:/ {c = $3}
         /^Dispatch Width:/ {w = $3}
         END {printf "%s,%.2f,%.2f,%s\n", t, c / it, ins / c, w}'
done
