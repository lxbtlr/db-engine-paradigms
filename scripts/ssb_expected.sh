#!/usr/bin/env bash
# Regenerate the SSB gtest reference answers with DuckDB.
#
#   scripts/ssb_expected.sh <data dir> [out dir]
#
# <data dir> holds {lineorder,part,supplier,customer,date}.tbl from
# electrum/ssb-dbgen. [out dir] defaults to src/test. Writes
# q{21..43}_ssb_expected.csv and q1x.txt; the q1x values times 10000 go into
# the EXPECT_EQ lines of src/test/ssb.cpp (printed at the end).
set -euo pipefail
D=$(realpath "${1:?usage: $0 <data dir> [out dir]}")
OUT=$(realpath "${2:-$(dirname "$0")/../src/test}")
SQL=$(dirname "$0")/ssb_expected.sql
DUCKDB=${DUCKDB:-duckdb}

sed -e "s|\$D|$D|g" -e "s|\$OUT|$OUT|g" "$SQL" | "$DUCKDB"
sed -i 's/\r$//' "$OUT"/q*_ssb_expected.csv "$OUT/q1x.txt"
awk -F, '{ printf "%s %d\n", $1, $2 * 10000 }' "$OUT/q1x.txt"
rm -f "$OUT/q1x.txt"
