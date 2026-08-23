#!/bin/bash
# Build the 5 NUMA configs the user wants to test, at current HEAD.
set -u
cd /tank/alexb/swole/db-engines
HEAD=$(git rev-parse HEAD 2>/dev/null || echo unknown)

CONFIGS=(
  "numa_alloc_debug|-DNUMA_ALLOC=ON -DNUMA_DEBUG=ON"
  "numa_shard_debug|-DNUMA_SHARD=ON -DNUMA_DEBUG=ON"
  "numa_alloc|-DNUMA_ALLOC=ON"
  "numa_shard|-DNUMA_SHARD=ON"
  "baseline|"
)

for entry in "${CONFIGS[@]}"; do
  name="${entry%%|*}"
  opts="${entry#*|}"
  DIR="build/$name"
  echo "===== building $name ====="
  rm -rf "$DIR"
  mkdir -p "$DIR"
  ( cd "$DIR" \
      && cmake ../.. $opts >cmake.log 2>&1 \
      && make -j 22 run_tpch >>cmake.log 2>&1 \
      && cp run_tpch "$name" ) \
      && { echo "# build-tpch provenance"
           echo "# built:       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
           echo "# cmake opts:  ${opts:-(none)}"
           echo "# configure:   cmake ../.. ${opts}"
           echo "# build:       make -j 22 run_tpch"
           echo "# git HEAD:    ${HEAD}"
         } > "$DIR/buildcmds.txt" \
      && echo "OK $name -> build/$name/$name" \
      || { echo "FAIL $name (see build/$name/cmake.log)"; }
done
echo "===== done ====="
