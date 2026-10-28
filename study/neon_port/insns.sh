#!/usr/bin/env bash
# Dynamic aarch64 instructions per element for every run_neonbench row,
# under qemu-user with the insncount plugin (study/neon_port/insncount.c).
#
# Each row runs twice with -T 1 and -T 3 (timed trials); everything else in
# the process (data generation, the reference check pass) is the same in both
# runs, so
#   insns_per_elem = (insns(T=3) - insns(T=1)) / (2 * elements per trial)
# with elements per trial from NEONBENCH_TRACE. l1 rows for the vector
# families, the smallest table (32KiB) for the sweep families: instruction
# counts do not depend on the table size, only time does.
#
# Usage: study/neon_port/insns.sh <run_neonbench> <libinsncount.so> [qemu -cpu] > insns.csv
#   EXTRA="-f proj,runheads" limits the l1 rows to some families
#   (inside `nix develop .#aarch64`, which has qemu-aarch64; building the
#   plugin is in study/neon_port/README.md)
set -euo pipefail
BIN=$1
PLUGIN=$2
CPU=${3:-neoverse-n1}
COMMON=(-v 1024 -r 20 -s 1 -p 16384 -t 1 ${EXTRA:-})
Q=(qemu-aarch64 -cpu "$CPU" -plugin "$PLUGIN")

insns() { "${Q[@]}" "$BIN" "${COMMON[@]}" "$@" 2>&1 >/dev/null | awk '/^insns /{print $2}'; }

# row list: one quick traced pass (l1 + sweep)
mapfile -t ROWS < <(NEONBENCH_TRACE=1 "${Q[@]}" "$BIN" "${COMMON[@]}" -T 1 -m l1 2>&1 >/dev/null |
                    awk '/^elems /{print $2, $3, $4, $5, $6, $7}'
                    NEONBENCH_TRACE=1 "${Q[@]}" "$BIN" "${COMMON[@]}" -T 1 -m sweep \
                        -f bloom,semi,dirprobe 2>&1 >/dev/null |
                    awk '/^elems /{print $2, $3, $4, $5, $6, $7}')

echo "cpu,family,impl,type,mode,param,insns_per_elem"
for r in "${ROWS[@]}"; do
   read -r fam impl type mode param elems <<<"$r"
   F=(-f "$fam" -i "$impl" -y "$type" -q "$param" -m "$mode")
   i1=$(insns "${F[@]}" -T 1)
   i3=$(insns "${F[@]}" -T 3)
   awk -v a="$i1" -v b="$i3" -v e="$elems" -v row="$CPU,$fam,$impl,$type,$mode,$param" \
       'BEGIN { printf "%s,%.2f\n", row, (b - a) / (2 * e) }'
done
