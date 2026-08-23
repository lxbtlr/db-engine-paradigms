#!/bin/bash
# Build all non-conflicting permutations of the huge-page CMake options
# (added in commit 26a5609) on top of the shard (NUMA_SHARD) build.
# Output binaries land in build/shard/ named run_tpch_<opts>.
set -u
cd /tank/alexb/swole/db-engines
BASE="build/shard"
HEAD=$(git rev-parse HEAD 2>/dev/null || echo unknown)

# name|extra_cmake_opts   (all carry -DNUMA_SHARD=ON as the shard base)
PERMS=(
  "shard|"
  "shard_numa1gb|-DHUGE_1GB_MALLOC_NUMA=ON"
  "shard_nohuge|-DNO_HUGE_PAGES=ON"
  "shard_nohuge_numa1gb|-DNO_HUGE_PAGES=ON -DHUGE_1GB_MALLOC_NUMA=ON"
  "shard_huge2mb|-DHUGE_2MB_MALLOC_HUGE=ON"
  "shard_huge2mb_numa1gb|-DHUGE_2MB_MALLOC_HUGE=ON -DHUGE_1GB_MALLOC_NUMA=ON"
  "shard_huge1gb|-DHUGE_1GB_MALLOC_HUGE=ON"
  "shard_huge1gb_numa1gb|-DHUGE_1GB_MALLOC_HUGE=ON -DHUGE_1GB_MALLOC_NUMA=ON"
)

for entry in "${PERMS[@]}"; do
  name="${entry%%|*}"
  huge_opts="${entry#*|}"
  opts="-DNUMA_SHARD=ON ${huge_opts}"
  DIR="$BASE/$name"
  echo "===== building $name ====="
  rm -rf "$DIR"
  mkdir -p "$DIR"
  ( cd "$DIR" \
      && cmake ../../.. $opts >cmake.log 2>&1 \
      && make -j 22 run_tpch >>cmake.log 2>&1 ) \
      && cp "$DIR/run_tpch" "$BASE/run_tpch_$name" \
      && { echo "# build-tpch provenance"
           echo "# built:       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
           echo "# cmake opts:  ${opts}"
           echo "# configure:   cmake ../../.. ${opts}"
           echo "# build:       make -j 22 run_tpch"
           echo "# git HEAD:    ${HEAD}"
         } > "$DIR/buildcmds.txt" \
      && echo "OK $name -> $BASE/run_tpch_$name" \
      || { echo "FAIL $name (see $DIR/cmake.log)"; }
done
echo "===== done ====="
