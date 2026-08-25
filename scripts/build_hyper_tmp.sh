#!/bin/bash
# Build the 2 HYPER_FLOAT configs into isolated tmp dirs, per build-tpch
# (auto-named dirs, ephemeral artifacts). Records mapping.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build
# register-build needs the corpus tools on PATH
CORPUS_HOME=${CORPUS_HOME:-/tank/alexb/vldb-db}
export CORPUS_HOME
export PATH="$CORPUS_HOME/bin:$PATH"
MAP=/tank/alexb/swole/db-engines/build_hyper_map.txt
: > "$MAP"

build_one() {
  local name="$1"; shift
  local opts="$*"
  local DIR
  DIR=$(mktemp -d "build/tmp.XXXXXXXX")
  ( cd "$DIR"
    cmake ../.. $opts -DHYPER_FLOAT=ON > cmake.log 2>&1
    make -j 22 run_tpch > make.log 2>&1
    {
      echo "# build-tpch provenance ($name)"
      echo "# built:       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
      echo "# cmake opts:  $opts -DHYPER_FLOAT=ON"
      echo "# configure:   cmake ../.. $opts -DHYPER_FLOAT=ON"
      echo "# build:       make -j 22 run_tpch"
      echo "# git HEAD:    $(git rev-parse HEAD 2>/dev/null || echo unknown)"
    } > buildcmds.txt
    rm -rf CMakeCache.txt CMakeFiles/
  )
  local BIN
  BIN=$(realpath "$DIR/run_tpch")
  # Register the build; build_event_id becomes the 3rd map column. HYPER_FLOAT
  # is recorded explicitly.
  local BE
  BE=$(register-build --binary "$BIN" --config "$name" \
                      --define HYPER_FLOAT=ON --buildcmds "$DIR/buildcmds.txt")
  echo "$name $BIN $BE" >> "$MAP"
  echo "OK $name -> $BIN (build_event_id=$BE)"
}

build_one default_huge_hyper
build_one default_nohuge_hyper -DNO_HUGE_PAGES=ON
echo "=== mapping ==="
cat "$MAP"
