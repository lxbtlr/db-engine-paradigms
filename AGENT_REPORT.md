# Agent B: register-resident key through the dispatch lookup

Commits on top of `45b8007`:

- `e4f06f1` feat(vectorwise): VW_GROUP_DISPATCH_ROWKEY (operator + gtests)
- `e58f4e9` feat(benchmarks): run_concatbench `-L` (concat + hash + probe)
- `5ad3d21` feat(scripts): `fair_concat_rowkey`, `hash_simd_gdisp_rowkey`

## Design

New CMake option **`VW_GROUP_DISPATCH_ROWKEY`** (default OFF, requires
`VW_GROUP_DISPATCH`, configure fails without it).

- `HashGroup::resolvePaths()` keeps the `keyStepT<T, kPacked, Runs>` step it
  chose as `keyStepPacked`. If the mode is `kPacked`, run heads are off, and
  `resolveRowKey()` finds an instantiated shape, `keyStep` becomes
  `keyStepRow<Sel, W...>`. A shape matches on column count, widths,
  contiguous offsets, the sum equal to `totalKeySize` in {1,2,4,8}, no
  padding, and either every column read through a selection vector or every
  column dense. All of these are plan facts.
- Instantiated shapes: `{1,1}` through a selection vector (TPC-H Q1),
  `{4,4}` through a selection vector, and `{4,4}` dense. The two `{4,4}`
  shapes exist only so the gtests reach the path.
- `keyStepRow` does this per row:
  1. `r = sel[i]`.
  2. `key = T(simd_concat::rowKey<W...>(d, r))`. Each column is loaded at
     its exact width and shifted to its offset, the same assembly as the row
     kernels (new inline helper in `SimdConcat.hpp`).
  3. `hash = hashFn.hashKey(key)`, the same call as `keyStepT`.
  4. Walk the chain comparing `entry->hash` and the entry's key as a `T`
     against the register.
  5. On a miss, it does exactly what `keyStepT` does (batch create, or one
     allocation plus `buildScatter` per group). The key is stored into the
     new entry from the register.
- If the key columns of a vector read *different* selection vectors, that
  vector goes through `keyStepPacked`. This pointer check is the same one
  `simd_concat::run` makes for the row kernel.
- The miss branch is marked `[[unlikely]]`. Without it, gcc 16 kept the
  create path in registers and spilled two hit-path values (two stack loads
  per row).

### What reads `packedKeys` (checked)

`packedKeys` appears only in `Concat`/`Concat_T`, `concatSimd`/`concatHash`,
`Hash_T`, `Lookup_T` and `keyStepT` (its run-heads pass 1 and lookup). Under
dispatch, per-vector work is only `keyStep` and `aggStep`. The other steps
do not need it:

- `buildScatter` reads `groupHashes[groupRepresentatives[k]]`, and the key
  columns through `lookup_sel` + `groupRepresentatives` (`QueryBuilder.cpp`
  ~770, 959, 1186).
- The aggregate steps (`aggEvaluate`, `updateGroupsFused`, `aggUpdate`) read
  `htMatches`.
- The spill (`flushAndClear`) reads the entries.

So `keyStepRow` does not write `packedKeys`. It still writes `hashes[i]` and
`groupRepresentatives` for new groups.

## Difference from the current specification

- **Layout:** unchanged. The entry key bytes match `Concat_T`'s packed key,
  because `rowKey` is little-endian and zero-extended to `T`.
- **Hash:** the same `hashFn.hashKey(T)` that `keyStepT` uses, so it is
  bit-identical for Murmur and CRC32 (including `VW_CRC32_FAST`, which does
  not touch the integer path). The dispatch key step never used the
  AVX-512/VPCLMUL hash kernels (it hashes inline), so nothing is lost there.
- **Plan decisions:** come from plan facts only. The only per-vector check
  is the shared selection-vector pointer, which is the existing row kernel's
  check (the pointer is fixed by the plan, not by the data).
- **Fallbacks:** wider, padded, mixed dense/sel and non-instantiated shapes
  keep `keyStepT`. The path is off under `VW_GROUP_RUN_HEADS`, whose pass 1
  compares `packedKeys`.

## Expected work reduction

Only **TPC-H Q1 (`1s1s`)** is affected. Q3 `4s4d4d` (12 bytes, padded to
16, mixed sel/dense), Q9/Q18 (wide), and every SSB shape (wider than 8 bytes
or mixed) keep `keyStepT`. A per-row reduction is possible only for keys of
at most 8 bytes that fit one register; 16-byte keys would need two registers
and a `Key16` path.

Per row, hit path (`objdump -d --no-show-raw-insn -C`, Murmur build; files
`dis-keyStepT-u16.txt` and `dis-keyStepRow-q1.txt` in the agent
directory):

| | before: row kernel + `keyStepT<u16,kPacked>` | after: `keyStepRow<true,1,1>` |
|---|---|---|
| passes over vector | 2 (row kernel, lookup loop) | 1 |
| loads | sel, 2 bytes, key reload for hash, key reload for compare, mask, dir, slot, entry hash, entry key (10) | sel, 2 bytes, mask, dir, slot, entry hash, entry key (8) |
| stores | packed key, match (2) | match (1) |
| instructions (perf, `-L`) | 33.3 (CRC32) / 40.3 (Murmur) | 28.2 / 35.3, about **-5 per row** |

In the whole operator (scratch Q1-shaped group-by: 4 groups, 2 SUMs, a
filter in front, single thread, `perf stat instructions:u`, difference of
13 and 3 repetitions):

| per input row | Murmur | CRC32 |
|---|---|---|
| sel 98%, before → after | 57.4 → 53.4 (-7%) | 50.6 → 45.6 (-9.9%) |
| sel 50%, before → after | 32.0 → 29.9 | 28.5 → 25.9 |

## Measured (laptop, Zen 4; indicative only, machine shared with other agents)

`run_concatbench -L -k q1,4s4s,4d4d,2s2s2s2s -s 98,50 -D 2,8 -N 2097152 -r 31`,
best of 5 processes. Columns are ns per selected row; `row` is today's
`fair_concat` key step; the ratio is row / rowkey.

| shape | mode, sel | hash | cols+probe | row+probe | rowkey | row/rowkey |
|---|---|---|---|---|---|---|
| q1/D2 | l1 98 | crc32 | 1.634 | 1.556 | 1.437 | 1.08 |
| q1/D2 | l1 98 | murmur | 2.228 | 2.157 | 2.142 | 1.01 |
| q1/D2 | stream 98 | crc32 | 1.770 | 1.595 | 1.495 | 1.07 |
| q1/D2 | stream 98 | murmur | 2.248 | 2.150 | 2.233 | 0.96 |
| q1/D2 | stream 50 | crc32 | 2.237 | 2.042 | 1.656 | 1.23 |
| q1/D8 | stream 98 | crc32 | 1.769 | 1.575 | 1.441 | 1.09 |
| q1/D8 | stream 98 | murmur | 2.668 | 2.458 | 2.359 | 1.04 |
| 4d4d/D8 | stream 98 | crc32 | 1.869 | 1.647 | 1.222 | 1.35 |
| 4s4s/D8 | stream 50 | crc32 | 2.840 | 2.415 | 1.845 | 1.31 |
| 4s4s/D2 | l1 98 | crc32 | 1.784 | 1.577 | 1.577 | 1.00 |

The full table is in `summary-L-min5.txt` in the agent directory. Every
variant's matched entries agreed with `cols+probe` (`ok` = 1 in all rows).

- **CRC32 (the fair ladder):** rowkey wins 7-25% on the Q1 shape in the
  microbenchmark.
- **Murmur:** -4% to +8%, close to the noise. Murmur's longer multiply chain
  makes the hash dominate, so saving memory operations matters less.

**Whole operator (cycles), the negative part.** In the scratch Q1-shaped
operator, cycles did **not** drop despite about 10% fewer instructions. With
`perf stat cycles:u`, the quietest of 4 runs gave:

- Murmur: 17.5-18.0 cyc per input row before, 18.1-18.4 after.
- CRC32: 13.5-15.1 before, 14.6-14.8 after.

`perf record` share agrees: row kernel + `keyStepT` = 49.1% vs `keyStepRow`
51.3% (Murmur), and 41.3% vs 40.8% (CRC32). Wall-clock runs varied by ±50%
because other agents were running on the machine, so they are not usable.

My reading: the fused loop puts the selection-index load, the column loads
and the hash on every row's pointer-chasing chain (slot → entry → compare).
About 30 instructions per row then share the reorder window, so fewer rows'
chains overlap. The separate loop's packed-key load is an L1 hit with a
short chain. The cycles attributed in `keyStepRow` sit on the slot load and
the entry compare (`perf annotate`). The microbenchmark's tables are smaller
and its loop has less around it, which fits it showing the gain while the
operator does not. **This is unconfirmed on Intel or Neoverse.**

**Tests:**

| platform | build | `GroupConcat*:HashGroup*` |
|---|---|---|
| x86 | option OFF (defaults) | 13/13 |
| x86 | dispatch + batch create + concat, Murmur and CRC32 | 13/13 |
| x86 | dispatch + rowkey without batch create and without CONCAT_SIMD | 13/13 |
| aarch64 qemu (burrata) | rowkey + CRC32 | 13/13 |
| aarch64 qemu (burrata) | rowkey + Murmur | 13/13 |

`gdb` breakpoints confirmed that the three new tests run `keyStepRow<true,1,1>`,
`<true,4,4>` and `<false,4,4>`. The full `test_all` TPCH/SSB tests fail
locally without data, the same as with the option OFF.

## Risks / open questions

- **Main risk:** no end-to-end gain. The work reduction is real (-5
  instructions per row, -10% operator instructions on Q1), but on Zen 4 it
  did not turn into fewer cycles at the operator level. The cluster has to
  decide it.
- Coverage is Q1 only. Other plans have no key of at most 8 bytes with one
  selection vector or all-dense columns. Extending to 16-byte keys (Q3
  padded) means a two-register `Key16` compare.
- The ht `mask`/`entries` are reloaded from `this` every row. This happens
  in both paths (aliasing through the `matches` stores). Hoisting them into
  locals is an orthogonal fix that applies to `keyStepT` as well. Not done.
- `[[unlikely]]` is a C++20 attribute in a C++17 build. gcc 16 accepts it
  without a warning.
- The build has asserts on (no `-DNDEBUG` on first configure);
  `Hashmap::insert` asserts are on both paths.

**Cluster configs** (added to `scripts/flag_ablation.sh`):

- `fair_concat_rowkey` = `fair_concat` + `-DVW_GROUP_DISPATCH_ROWKEY=ON`.
  Compare against `fair_concat`, Q1, on manchego, dubliner and burrata.
- `hash_simd_gdisp_rowkey` = `hash_simd_gdisp_concat` + the option. Compare
  against `hash_simd_gdisp_concat`.

Check instructions per tuple as well as time: the effect is around the
manchego bimodal-IPC noise level.
