#!/bin/bash
# tldr_collect.sh — assemble a tldr from a finished run_tpch .out and text it.
# Intended to run as a slurm collector job chained with --dependency=afterany:<runjob>.
#
# Usage: tldr_collect.sh <outfile> <title> [note]
#   <outfile>  path to the run's .out CSV (slurm stdout of run_tpch)
#   <title>    short title line for the tldr (e.g. "TPC-H default_huge sf100 ...")
#   [note]     optional one-line observation appended to the text message
set -euo pipefail

OUT="${1:?need .out file}"
TITLE="${2:?need title}"
NOTE="${3:-}"
[ -s "$OUT" ] || { echo "tldr_collect: $OUT empty/missing" >&2; exit 1; }

TLDR="$(mktemp /tmp/tldr_XXXXXXXX.md)"

# --- Markdown: title + full CSV ---
{
  echo "# $TITLE"
  echo "- Source: $OUT"
  echo "- Median time in ms."
  echo '```'
  cat "$OUT"
  echo '```'
} > "$TLDR"

# --- Text message: group rows by query+engine, thread medians ---
MSG="$(OUT="$OUT" TITLE="$TITLE" NOTE="$NOTE" python3 - <<'PY'
import os, re, sys, csv, io

title = os.environ["TITLE"]; note = os.environ.get("NOTE","")
out = os.environ["OUT"]

pat = re.compile(r'^\s*(q\d+)\s+([hv])\s+t(\d+)\s*$')
groups = {}   # (query,engine) -> ordered list of (thread, median)
order = []
with open(out) as f:
    rows = list(csv.reader(f))
# find header, then data rows
for r in rows:
    if not r or not r[0].strip(): continue
    if r[0].strip().lower().startswith("name"): continue
    if len(r) < 2: continue
    m = pat.match(r[0])
    if not m: continue
    q, e, t = m.group(1), m.group(2), int(m.group(3))
    med = r[1].strip()
    key = (q, e)
    if key not in groups:
        groups[key] = []; order.append(key)
    groups[key].append((t, med))

lines = [title]
for key in order:
    q, e = key
    cells = " | ".join(f"t{t} {med}" for t, med in groups[key])
    lines.append(f"{q} {e}:  {cells}")
if note:
    lines.append(note)
msg = "\n".join(lines)
# hard guard: keep text under the 4000-char limit by trimming tail lines
while len(msg) > 3950 and len(lines) > 3:
    lines.pop(); msg = "\n".join(lines)
msg += "\nFull results attached."
print(msg)
PY
)"

echo "--- generated message (${#MSG} chars) ---"
printf '%s\n' "$MSG"

# --- Send: message first, then markdown attachment ---
/tank/project/text/text-alex2 "$TLDR" "$MSG"
echo "tldr_collect: sent (exit $?)"
rm -f "$TLDR"
