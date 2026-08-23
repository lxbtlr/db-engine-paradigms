#!/bin/bash
# Build the 4 plain sweep configs (NO interleave_ht) into isolated tmp dirs,
# per build-tpch (auto-named dirs, ephemeral artifacts). Records mapping.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build
# register-build needs the corpus tools on PATH
CORPUS_HOME=${CORPUS_HOME:-/tank/alexb/vldb-db}
export CORPUS_HOME
export PATH="$CORPUS_HOME/bin:$PATH"
MAP=/tank/alexb/swole/db-engines/build_plain_map.txt
: > "$MAP"

build_one() {
  local name="$1"; shift
  local opts="$*"
  local DIR
  DIR=$(mktemp -d "build/tmp.XXXXXXXX")
  ( cd "$DIR"
    cmake ../.. $opts > cmake.log 2>&1
    make -j 22 run_tpch > make.log 2>&1
    {
      echo "# build-tpch provenance ($name)"
      echo "# built:       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
      echo "# cmake opts:  $opts"
      echo "# configure:   cmake ../.. $opts"
      echo "# build:       make -j 22 run_tpch"
      echo "# git HEAD:    $(git rev-parse HEAD 2>/dev/null || echo unknown)"
    } > buildcmds.txt
    rm -rf CMakeCache.txt CMakeFiles/
  )
  local BIN
  BIN=$(realpath "$DIR/run_tpch")
  # Register the build; the returned build_event_id becomes the 3rd map column,
  # making it knowable at submit time for the ingest collector.
  local BE
  BE=$(register-build --binary "$BIN" --config "$name" --buildcmds "$DIR/buildcmds.txt")
  echo "$name $BIN $BE" >> "$MAP"
  echo "OK $name -> $BIN (build_event_id=$BE)"
}

build_one default_huge
build_one default_nohuge -DNO_HUGE_PAGES=ON
build_one shard_huge -DNUMA_SHARD=ON
build_one shard_nohuge -DNUMA_SHARD=ON -DNO_HUGE_PAGES=ON
echo "=== mapping ==="
cat "$MAP"
