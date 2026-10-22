#!/usr/bin/env bash
# Slurm job for scripts/vtune_q1.sh: loads VTune's environment and runs the
# driver from the checkout the job was submitted in. Submit from the repo
# root (Slurm runs a copy of this file from its spool directory, so the
# checkout comes from SLURM_SUBMIT_DIR, not from this file's path):
#
#   cd <checkout> && sbatch --partition=cheese --nodelist=dubliner --exclusive \
#       --output=/tank/alexb/slurm-tpch/dubliner/vtune_q1_%j.out \
#       --export=ALL,SF=1,THREADS=1,ANALYSES=ua scripts/vtune_q1_job.sh
#
# vtune_q1.sh settings (SF, THREADS, CONFIGS, OUT, ...) pass through --export.
#
# Environment:
#   VTUNE_VARS  VTune's env/vars.sh. Default per host: dubliner uses 2025.3
#               (2025.8 does not recognise its Cascade Lake CPU), elsewhere
#               /opt/intel/oneapi/vtune/latest
#   REPO        ${SLURM_SUBMIT_DIR:-$PWD}   the checkout to run
set -u -o pipefail

case "$(hostname -s 2> /dev/null || hostname)" in
  dubliner) DEF_VARS=/home/alexb/intel/oneapi/vtune/2025.3/env/vars.sh ;;
  *) DEF_VARS=/opt/intel/oneapi/vtune/latest/env/vars.sh ;;
esac
VTUNE_VARS=${VTUNE_VARS:-$DEF_VARS}
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}

[ -f "$VTUNE_VARS" ] || { echo "no VTune environment at $VTUNE_VARS (set VTUNE_VARS)" >&2; exit 2; }
[ -f "$REPO/scripts/vtune_q1.sh" ] ||
  { echo "no scripts/vtune_q1.sh under $REPO (submit from the repo root, or set REPO)" >&2; exit 2; }
# vars.sh reads its own arguments; keep this script's out of it
set --
# shellcheck disable=SC1090
source "$VTUNE_VARS" > /dev/null
cd "$REPO" && exec bash scripts/vtune_q1.sh
