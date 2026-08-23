#!/bin/bash
# Build the 4 huge-page/NUMA-shard sweep configs at current HEAD.
set -u
cd /tank/alexb/swole/db-engines
HEAD=$(git rev-parse HEAD 2>/dev/null || echo unknown)

CONFIGS=(
  "default_huge|"
  "default_nohuge|-DNO_HUGE_PAGES=ON"
  "shard_huge|-DNUMA_SHARD=ON"
  "shard_nohuge|-DNUMA_SHARD=ON -DNO_HUGE_PAGES=ON"
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
