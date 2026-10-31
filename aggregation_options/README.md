# HashGroup aggregation options (branch `aggregation`)

Every CMake option that changes VectorWise's `HashGroup` (GROUP BY) on this
branch: what each changes, what it depends on, and when its code actually
runs. Only options CMake can set are shown. Hashing options (`VW_USE_CRC32`,
`VW_CRC32_*`) and `VW_PROJ_COMPOUND` (a `Project` option) are left out.

| Diagram | Shows |
|---|---|
| `relations.png` | build dependencies, and the run-time fallback order of the aggregate updates |
| `code_map.png` | which stage of the operator each option changes, by phase |
| `decision.png` | how the aggregate update is chosen: at build time, once per operator from the plan, and per vector (with the sticky switches) |

Render: `.claude/skills/writeup/wenv mmdc -i <f>.mmd -o <f>.png -s 2`.

## The options

| Option | Phase / stage | Change | Requires |
|---|---|---|---|
| `VW_GROUP_DISPATCH` | 1 / key step | `keyStepT`: one key loop chosen once per operator (key width, single or packed key, selection); hashes inside the lookup | — |
| `VW_GROUP_BATCH_CREATE` | 1 / group creation | new groups from one block, one `buildScatter` per vector (in `Lookup_T` or `keyStepT`) | — |
| `VW_AGGR_DIRECT` | 1 / aggregate update | with ≤ 8 groups in the vector: one SIMD pass per aggregate, one register accumulator per group, COUNT from the masks | — |
| `VW_AGGR_DIRECT_ISA` (cache, auto) | — | kernel ISA: auto (AVX-512F, else AVX2, else scalar), or force avx512 / avx2 / scalar | (direct) |
| `VW_AGGR_FUSED` | 1 / aggregate update | one per-row pass updates all aggregates of the row's entry | — |
| `VW_AGGR_FUSED_BUCKET` | 1 / aggregate update | ids per touched group, rows counting-sorted by group, per-group register sums | `VW_AGGR_FUSED` |
| `VW_AGGR_FUSED_BUCKET_MAX_GROUPS` (cache, 256) | — | groups per vector above which bucketing switches off for good | (bucket) |
| `VW_GROUP_GLOBAL_DIRECT` | 2 / find or create | one-pass find-or-create on a tagged directory | — |
| `VW_GROUP_HAVING` | 2 / output | HAVING evaluated before the output gather | — |

The flush (stage 4) and the phase 2 merge (stage 6) have no option.

Test-only macros (pass through `CMAKE_CXX_FLAGS`):
`-DVW_AGGR_DIRECT_MAX_GROUPS=<1..8>` lowers direct's group limit, and
`-DVW_AGGR_DIRECT_MIN_AGGRS=<n>` changes how many aggregates direct needs
(default 2), to force and test the fallbacks.

## Dependencies (`relations.png`)

Six options are independent: any subset builds and is correct, with or
without the others. `VW_GROUP_DISPATCH` turns on nothing else; its key loop
batches creation only under `VW_GROUP_BATCH_CREATE`, and its aggregate step
uses whichever update options are compiled in.

The only build dependency left is `VW_AGGR_FUSED_BUCKET` → `VW_AGGR_FUSED`:
the bucket path is a branch at the top of `updateGroupsFused`.

`VW_AGGR_DIRECT` sits beside `VW_AGGR_FUSED`. Both read the operator's
aggregate list, which `QueryBuilder::addValue` records whenever either is
compiled in (`VW_AGGR_LIST`, `Operators.hpp`). The bucket and direct paths
share the per-vector group ids (an 8-byte field at the end of each local
entry, `groupIdOffset`, and `assignGroupIds`), compiled in when either is
on (`VW_AGGR_GROUP_IDS`).

At run time the aggregate updates fall back in one order:

```
direct  ->  bucket  ->  per-row fused  ->  interpreted passes
```

each step only if it is compiled in and applies to the operator (`aggUpdate`
in `Operators.cpp`; with dispatch, `resolvePaths` sets the aggregate step to
it when direct applies, else to the fused pass or the interpreted passes).

## When each path runs (`decision.png`)

Three kinds of decision pick the aggregate update:

1. **Build time:** the options compiled in.
2. **Plan, once per operator** (fixed for the query):
   - direct, fused and bucket need ≥ 2 aggregates, all int64 SUM or
     COUNT(*); otherwise the operator runs one interpreted pass per
     aggregate and none of their code runs;
   - bucket also needs ≤ 4 dense and ≤ 4 selected aggregates.
3. **Per vector, from the data:**
   - the number of groups the vector touches, `t`: direct takes `t ≤ 8`;
     bucket takes `t ≤ 256`; above that the per-row fused pass;
   - direct also needs every selected column to use one selection whose
     span fits the position buffer; otherwise that one vector falls back,
     and the next vector tries direct again.

The group-count switches are **sticky, per thread**: the first vector with
`t > 8` turns direct off for the rest of the query, the first with
`t > 256` turns bucket off. A path never comes back.

The key step and phase 2:

- key step: `keyStepT` with dispatch, else `Concat` + `Hash` + `Lookup_T`;
  new groups batched with `VW_GROUP_BATCH_CREATE` in either.
- phase 2: `globalFindOrCreate` with `VW_GROUP_GLOBAL_DIRECT` if every key
  compares with `memcmp` (int8/16/32/64, Date, hash, `Char<1>`); other keys
  keep `findGroups` + `createMissingGroups`.
- `VW_GROUP_HAVING` acts only in plans that call `having()` (Q18).

## Which TPC-H group-bys reach which path

From the plans in `src/benchmarks/tpch/queries/`:

| Query: group-by | Keys | Aggregates | Direct / fused / bucket | Global direct | HAVING |
|---|---|---|---|---|---|
| Q1 | `l_returnflag`, `l_linestatus` (`Char<1>`) | 4 SUM + COUNT | **yes** (4 groups: direct) | yes | — |
| Q3 | `l_orderkey`, `o_orderdate`, `o_shippriority` | 1 SUM | no (1 aggregate) | yes | — |
| Q5 | `n_name` (`Char<25>`) | 1 SUM | no | no | — |
| Q9 | `n_name` (`Char<25>`), year | 1 SUM | no | no | — |
| Q18: by `l_orderkey` | `l_orderkey` | 1 SUM | no | yes | **yes** |
| Q18: outer | `c_name` (`Char<25>`), custkey, orderkey, date, price | 1 SUM | no | no | — |

So the aggregate-update options change only Q1 among these queries.

**Why direct needs ≥ 2 aggregates.** With 1 (tested 2026-10-31), Q3 and
Q18 entered direct, touched more than 8 groups on their first vector and
fell back, but kept the 8-byte entry field: Q3 22.55 / 22.19 / 22.17 ms and
Q18 66.20 / 65.78 / 65.90 ms in three builds with the minimum at 1, against
21.94 and 64.85 ms with it at 2 (and 21.79 / 64.27 for `grp_dispatch`).

## Ablation configs (`scripts/flag_ablation.sh`)

| Config | Options on top of `grp_base` (CRC32 + FAST) |
|---|---|
| `grp_disp_only` | dispatch |
| `grp_dispatch` | dispatch + batch + global + fused (`GDISP`, the old meaning of dispatch) |
| `grp_bucket` | `grp_dispatch` + bucket |
| `grp_direct` | `grp_dispatch` + direct (falls back to fused) |
| `grp_direct_only` | dispatch + batch + global + direct, no fused (falls back to interpreted) |
| `grp_direct_bucket` | `grp_bucket` + direct (falls back to bucket) |
| `direct_pc` | `grp_direct` + `VW_PROJ_COMPOUND` |

Force the direct kernels' ISA with `VW_FLAGS="VW_AGGR_DIRECT_ISA=avx2"`.

## Measured (Q1)

gcc 16, Ryzen 7 7840U (Zen 4) laptop, locally generated SF1, 1 thread on core
4, 5 interleaved rounds of 25 repetitions with Hyper in the same rounds
(2026-10-31). Run-to-run noise between runs was up to ~1.8 ms.

| Build | Q1 (ms) | vs Hyper (32.7) |
|---|---|---|
| `proj_compound` at `16fbc54` (previous best: dispatch + `FUSED_KINDS` + compound) | 30.7 | 0.94 |
| `grp_dispatch` (per-row fused) | 36.8 | 1.12 |
| `grp_direct` | 33.9 | 1.04 |
| `grp_direct_only` | 33.0 | 1.01 |
| `direct_pc` | 31.9 | 0.97 |
| `grp_bucket` (earlier run) | 41.9 | 1.28 |
| direct with AVX2 / scalar kernels (earlier run, direct then under bucket) | 35.9 / 55.8 | 1.10 / 1.71 |

These Q1 runs had the minimum at 1, which does not change Q1 (5 aggregates).
The scalar direct kernels are for correctness on machines without AVX2 /
AVX-512; there the option should stay off. All other queries were within
noise of `grp_dispatch` with the minimum at 2.

## Tested

All builds pass the TPC-H result tests (`TPCH.*`: Q1, Q3, Q5, Q6, Q9, Q18 at
SF1) and the kernel test `AggrDirect.*` (every group count 1..8, lengths on
and off a SIMD block, rows of no group, `sum`, `count`, `sumCount`), with no
new compiler warnings: `grp_dispatch`, `grp_bucket`, `grp_direct` (minimum
1 and 2), `grp_direct_only`, `grp_direct_bucket`, `direct_pc`, direct with
the avx2 and scalar kernels, and `VW_AGGR_DIRECT` alone without dispatch.
Forced fallbacks (`-DVW_AGGR_DIRECT_MAX_GROUPS=2`) to each next step: direct
→ interpreted (no fused; with and without dispatch), direct → fused, direct
→ bucket, and direct → bucket → fused (bucket max 3); `grp_bucket` with
bucket max 2 for bucket → fused. `perf record` confirmed each build runs
the path the decision diagram predicts.

## Removed on this branch

`VW_SPILL_WORD_COPY` (option and code). `VW_AGGR_FUSED_BUCKET_KEYLOOP`
(option and code, added and removed on this branch): it ran the bucket
path's pass 1 inside the dispatch key loop and made Q1 slower (44.1 ms
against 40.1 for `grp_bucket`, same laptop setup), from register spills
and the id loads waiting on the chain walk. `VW_AGGR_FUSED_DIRECT` was
renamed `VW_AGGR_DIRECT` when it stopped depending on the fused and bucket
options. `VW_GROUP_RUN_HEADS`, `VW_AGGR_FUSED_KINDS`, `VW_FUSE_HASH`,
`VW_GROUP_NO_CONCAT`, `VW_GROUP_AGGR` and `VW_GROUP_AGGR_SEL` lost their
CMake options; their code is still in the sources under its `#ifdef` but is
unreachable through CMake.
