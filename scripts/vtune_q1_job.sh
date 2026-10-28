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
#   VTUNE_VARS  VTune's env/vars.sh. Default per host: dubliner and manchego
#               use 2025.3 in ~/intel (2025.8 does not recognise dubliner's
#               Cascade Lake CPU; manchego has no /opt install), elsewhere
#               /opt/intel/oneapi/vtune/latest
#   REPO        ${SLURM_SUBMIT_DIR:-$PWD}   the checkout to run
set -u -o pipefail

case "$(hostname -s 2> /dev/null || hostname)" in
  dubliner|manchego) DEF_VARS=/home/alexb/intel/oneapi/vtune/2025.3/env/vars.sh ;;
  *) DEF_VARS=/opt/intel/oneapi/vtune/latest/env/vars.sh ;;
esac
VTUNE_VARS=${VTUNE_VARS:-$DEF_VARS}
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}

# Slurm takes the job's groups from the controller, which does not know the
# node-local vtune group that owns the sampling driver (/dev/sep5, /dev/pax).
# Without it VTune silently falls back to driverless perf sampling. sg gets
# the group back from this node's /etc/group; the environment carries over.
if [ -e /dev/pax ] && [ ! -r /dev/pax ] && [ -z "${VTUNE_SG:-}" ] &&
   getent group vtune | cut -d: -f4 | tr ',' '\n' | grep -qx "$(id -un)"; then
  export VTUNE_SG=1
  exec sg vtune -c "$(printf '%q ' bash "$0" "$@")"
fi
[ -e /dev/pax ] && [ ! -r /dev/pax ] &&
  echo "WARNING: no access to the VTune sampling driver (/dev/pax); VTune will use driverless perf" >&2

[ -f "$VTUNE_VARS" ] || { echo "no VTune environment at $VTUNE_VARS (set VTUNE_VARS)" >&2; exit 2; }
[ -f "$REPO/scripts/vtune_q1.sh" ] ||
  { echo "no scripts/vtune_q1.sh under $REPO (submit from the repo root, or set REPO)" >&2; exit 2; }
# vars.sh reads its own arguments; keep this script's out of it
set --
# shellcheck disable=SC1090
source "$VTUNE_VARS" > /dev/null
cd "$REPO" && exec bash scripts/vtune_q1.sh
