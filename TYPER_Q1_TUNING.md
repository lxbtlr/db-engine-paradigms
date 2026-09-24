# Typer (HyPer-style) TPC-H Q1 Tuning

## Change ladder

| Step | CMake option | Typer change | Tectorwise counterpart |
|------|-------------|--------------|----------------------|
| 1 | `AUTOVECTORIZE=ON` | `NOVECTORIZE` made empty for both compilers; `-ftree-vectorize` globally | Same flag controls both engines symmetrically |
| 2 | `HYPER_Q1_DIRECT_AGG` | Per-morsel stack accumulators indexed by slot; hash table touched only at morsel end | `VW_GROUP_AGGR` (group-wise aggregation) |
| 3 | `HYPER_Q1_SIMD` | AVX-512 intrinsics: masked date filter, `_mm512_mullo_epi64` for arithmetic, per-slot `_mm512_mask_add_epi64` | `SIMDsel` AVX-512 selection primitives |
| 4 | (skipped) | Narrow `l_discount`/`l_tax`/`l_quantity` to int32 | `VW_POS_16` (narrower pos_t) |
| 5a | `MORSEL_SIZE=N` | Sweep morsel size: {1000, 5000, 10000, 50000, 100000} | Same knob affects both engines |
| 5b | `PGO=generate/use` | Profile-guided optimization build cycle | N/A |

## Benchmark configurations

All builds should include baseline VW tuning flags:
```
-DTARGET_MACHINE=<machine> -DVW_USE_CRC32=ON -DVW_GROUP_AGGR=ON
```

### Build matrix

| Build | Additional CMake flags |
|-------|----------------------|
| B1 (baseline) | (none) |
| B2 (autovec) | `-DAUTOVECTORIZE=ON` |
| B3 (direct-agg) | `-DHYPER_Q1_DIRECT_AGG=ON` |
| B4 (direct-agg + autovec) | `-DHYPER_Q1_DIRECT_AGG=ON -DAUTOVECTORIZE=ON` |
| B5 (SIMD) | `-DHYPER_Q1_DIRECT_AGG=ON -DHYPER_Q1_SIMD=ON` |
| B6 (PGO generate) | Same as B5 + `-DPGO=generate` |
| B7 (PGO use) | Same as B5 + `-DPGO=use` |

### PGO workflow

```bash
# Step 1: build with profiling
cmake .. -DPGO=generate [other flags]
make -j
./run_tpch -q 1 -t 1 -r 3 -s <sf>

# Step 2: merge profiles (Clang only)
llvm-profdata merge -output=pgo-profiles/default.profdata pgo-profiles/

# Step 3: rebuild with profiles
cmake .. -DPGO=use [other flags]
make -j
./run_tpch -q 1 -t 1 -r 5 -s <sf>
```

### Morsel size sweep

Rebuild with `-DMORSEL_SIZE=<N>` for each value in {1000, 5000, 10000, 50000, 100000}.
Test with B3 (scalar direct-agg) and B5 (SIMD) configurations.

## Benchmark results

> Fill in: median of 5 runs, single-threaded and full-threaded.

| Build | Hyper Q1 (ms) | VW Q1 (ms) | Threads | SF | Compiler | CPU |
|-------|--------------|------------|---------|----|---------|----|
| B1 | | | | | | |
| B2 | | | | | | |
| B3 | | | | | | |
| B4 | | | | | | |
| B5 | | | | | | |
| B7 | | | | | | |

## Vectorizer report excerpts

### B1 (baseline, AUTOVECTORIZE=OFF)
`q1_hyper` loop not compiled with vectorization (global `-fno-tree-vectorize`).

### B2 (AUTOVECTORIZE=ON, no DIRECT_AGG)
Expected: "not vectorized" due to hash table lookup (`findOrCreate`) inside loop body.
GCC reason: _(fill in from `-fopt-info-vec-all`)_
Clang reason: _(fill in from `-Rpass-missed=loop-vectorize`)_

### B4 (DIRECT_AGG + AUTOVECTORIZE)
Expected: "not vectorized" due to data-dependent slot indexing and branch.
GCC reason: _(fill in)_
Clang reason: _(fill in)_

### B5 (SIMD)
Hand-vectorized with intrinsics. No compiler auto-vectorization needed.

## Overflow analysis: sum_charge

`sum_charge` accumulates `Numeric<12,6>` (int64_t, scale 6).

Per-row maximum charge:
- `l_extendedprice` max = 99,999.99 → raw 9,999,999 (scale 2)
- `l_discount` min = 0.00 → factor (100 - 0) = 100
- `l_tax` max = 0.08 → factor (100 + 8) = 108
- `disc_price` = 9,999,999 × 100 = 999,999,900 (scale 4)
- `charge` = 999,999,900 × 108 = 107,999,989,200 (scale 6)

Per-row realistic average (from TPC-H data generator):
- avg extendedprice ≈ 38,255 → raw ~3,825,500
- avg discount ≈ 0.05 → factor 95
- avg tax ≈ 0.04 → factor 104
- avg charge ≈ 3,825,500 × 95 × 104 ≈ 37,795,940,000

Accumulation over all qualifying rows (~98% pass date filter):

| SF | Rows passing | Worst-case sum_charge | Realistic sum_charge | int64_t max |
|----|-------------|----------------------|---------------------|-------------|
| 1 | ~5.9M | 6.4 × 10^17 | 2.2 × 10^17 | 9.22 × 10^18 |
| 10 | ~59M | 6.4 × 10^18 | 2.2 × 10^18 | 9.22 × 10^18 |
| 100 | ~588M | 6.4 × 10^19 | 2.2 × 10^19 | 9.22 × 10^18 |

**Overflow occurs at SF ≥ ~40 with realistic data, SF ≥ ~14 in worst case.**

Note: this is sum_charge for ALL groups combined. Per-group sums are smaller (roughly 4x).
Per-group overflow threshold: ~SF 56 (realistic), ~SF 160 (worst case per group).

However, the intermediate products `disc_price` and `charge` computed per tuple never overflow:
max per-tuple `charge` = 1.08 × 10^11, well within int64_t range.

Mitigation options:
1. Use `__int128` accumulators (no SIMD support)
2. Use `double` accumulators (precision loss at high SF)
3. Accept: SF ≤ 10 is the common benchmark range and is safe
