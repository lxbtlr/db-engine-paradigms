# Agent C: shape-generic row kernel (VW_GROUP_CONCAT_ROW_GENERIC)

Base `45b8007`. Commit `b041244` holds the code; this report is in a second commit.

## Design

Two row kernels, each instantiated per **column count** and not per width tuple. Widths, offsets and masks are plan-time tables that are copied into locals before the row loop. The loop has no per-column branch. Both kernels live in `src/vectorwise/primitives/Concat.cpp` behind the existing `Plan` / `build` / `run` interface, as new `Isa::RowMasked` and `Isa::RowGeneric`.

- **RowMasked** (AVX-512BW/VL, `VW_CONCAT_HAVE_MASKED`): the key is built in one xmm.
  - Each column is read by one `vmovdqu8 xmm{k_c}{z}, [value - offset_c]`, where `k_c` holds the column's key bytes. The value lands at its key offset without a shift. Masked-off bytes are neither read nor faulted on, so there is no over-read or under-read.
  - The columns are combined with OR (`vpor` / `vpternlog`), and the key is written by one 8-byte store (K <= 8) or one 16-byte store.
  - Template parameters: `<C, S, Wide, Shift>`.
    - S is the number of selection-vector columns, which the plan orders first. Dense columns get no multiply and selection-vector columns no pointer bump.
    - `Shift`: when every selection-vector width is a power of two, the row offset is computed with `shlx` instead of `imul`.
  - 8 x 9 x 2 x 2 table entries, of which about 176 kernels are instantiated. Covers any column widths 1..16 and K <= 16, padded keys included (padding is written as zero).
- **RowGeneric** (any ISA, opt-in with `_KERNEL=gpr`): the key is built in two 64-bit halves.
  - A column is split at byte 8 into pieces. Each piece is read by the 8 bytes that end with it, then `>> (64-8len) << 8dst`, then OR-ed in.
  - The loads only reach up to 7 bytes *before* a value, never past it. Rows whose loads would start before the column pointer are built byte by byte instead (Plan::selGuard / denseGuard):
    - dense rows are peeled;
    - selection-vector rows branch on `sel[i] < guard` in a cold path that sits outside the inner loop.
  - Template parameters: `<P pieces, Plo, AnySel, AnyDense>`, 135 instantiations.

**Options.**
- `VW_GROUP_CONCAT_ROW_GENERIC` (default OFF; needs `VW_GROUP_CONCAT_SIMD`).
- `VW_GROUP_CONCAT_ROW_GENERIC_KERNEL=auto|gpr`:
  - `auto` uses RowMasked, and only in AVX-512BW/VL builds. Other builds behave as they do today: burrata / NEON and AVX2-only machines are unchanged.
  - `gpr` uses RowGeneric everywhere.

**Plug-in (`HashGroup::planConcat`).** The order is:
1. The instantiated row kernel (Q1), if `VW_GROUP_CONCAT_ROW`.
2. All-dense keys keep the vector shuffle kernels where those build.
3. The generic kernel.
4. As before: the shuffle kernels, else per-column `Concat_T`.

With `VW_GROUP_CONCAT_ROW=OFF` the generic kernel also takes Q1, so the two can be compared. Columns that read different selection vectors fall back to the per-column copy at run time, the same check as Row (this is a pointer-identity check, not a density check).

## Difference from the current specification
- **Layout and hashes:** packedKeys bytes are identical to `Concat_T` (tests compare against `runColumns`). Hashing is unchanged: `Hash_T` still runs as a separate pass, so hashes are bit-identical. `concatHash` still requires `Isa::Row`, so `VW_GROUP_CONCAT_HASH` is not fused with the generic kernels.
- **Stores:** 8 or 16 bytes per row. These overlap the next row's bytes, which are written after. The last row writes up to 15 bytes past n*K, which is within kSlack = 64.
- **Padded keys:** the generic kernels write padding bytes as zero (Row refuses padded keys).
- **Plan facts:** every decision uses only widths, offsets, sel/dense, K and the ISA macro. RowGeneric's `sel[i] < guard` branch is a memory-safety check on the index, not a performance choice. Its outcome does not change which kernel runs, only how that row is copied.
- **Keys over 16 bytes** (Q9, Q18, SSB q3.x/q4.x) are not covered. `build` returns false and they keep `Concat_T` / `VW_GROUP_CONCAT_WIDE`. RowMasked would extend to 32 or 64 bytes with ymm/zmm (`__mmask32/64`), but that overlaps idea E.

## Expected work reduction (per row, hot loop from `objdump -d --no-show-raw-insn -C`, gcc 16 -march=native Zen 4)
| shape | today | RowMasked |
|---|---|---|
| Q1 `1s1s` | Row tmpl: 9 instr (idx, 2 movzx, shl, or, store, loop 3) | 11 instr: idx, 2 shlx, 2 masked loads, vpor, vmovq, ptr/out add, cmp/jne |
| Q3 `4s4d4d` (12, padded 16) | 3 Concat_T passes: 3 idx/ptr walks, 3 loads, 3 stores, ~18 instr | 1 pass: idx load, 1 shlx, 3 masked loads, 2 ptr bumps, 1 vpternlog, 1 x 16B store, loop: 13 instr |
| SSB q2.x `4d9s` (13, padded 16) | WIDE: 2 passes, 4B column load+store; 9B column idx + 2x8B loads + 2 stores | 1 pass: idx, imul (9 not a power of 2), 2 masked loads, vpor, 16B store, loop: 11 instr |
| Q9, Q18, SSB q3.x/q4.x (>16 B) | unchanged | not covered |

Passes over packedKeys go from C (one per column) to 1. Stores go from C per row to 1.

RowGeneric needs 2 shifts, a multiply and a load per piece. With 4 values per piece in registers, gcc spills its shift counts from P = 2 on (stack reloads in the loop). That is what makes it slow.

## Measured (laptop Zen 4, `run_concatbench`, pinned `taskset -c 6`, min of 3 runs x 7 reps, ns/selected row, 98% selected)

| shape | mode | per-column copy | current best | row_masked | row_generic |
|---|---|---|---|---|---|
| q1 1s1s | l1 / stream | 0.49 / 0.53 | row_tmpl 0.42 / 0.41 | 0.51 / 0.50 | 1.25 / 1.23 |
| q3 4s4d4d | l1 / stream | 0.77 / 1.01 | (copy) | **0.56 / 0.84** | 2.11 / 2.35 |
| q3p (+4 pad, 16 B) | l1 / stream | 0.74 / 1.13 | (copy) | **0.59 / 0.87** | 2.10 / 2.31 |
| ssb2 4d9s | l1 / stream | 2.64 / 2.88 | wide_fixed 0.78 / 1.00 | **0.53 / 0.75** | 2.03 / 2.07 |
| ssb2p (+3 pad) | l1 / stream | 2.87 / 3.06 | wide_fixed 0.79 / 1.15 | **0.59 / 0.84** | 2.29 / 2.18 |
| 1s2s4s | l1 / stream | 0.73 / 0.87 | row_tmpl 0.54 / 0.59 | 0.53 / 0.60 | 1.97 / 2.03 |
| 2s2s2s2s | l1 / stream | 0.97 / 1.11 | row_tmpl 0.74 / 0.76 | 0.74 / 0.81 | 2.23 / 2.20 |
| 1s1s1s1s4s | l1 / stream | 1.13 / 1.48 | row_tmpl 0.86 / 0.93 | 1.12 / 1.16 | 2.90 / 2.98 |
| 4s4s | l1 / stream | 0.46 / 0.86 | row_tmpl 0.38 / 0.71 | 0.45 / 0.78 | 1.10 / 1.59 |
| 2d2d | l1 | 0.45 | vbmi 0.20 | 0.44 | 0.57 |
| 1d2d4d | l1 | 0.67 | row_tmpl 0.47 / vbmi 0.50 | 0.45 | 1.15 |
| 8d8d | l1 | 0.47 | (vbmi 0.85 is what HashGroup picks today) | 0.47 | 0.59 |
| 3s4s / 5d2s1s / 7s / 16s | l1 | 2.69 / 3.09 / 2.52 / 2.37 | wide_fixed 16s 0.57 | 0.57 / 0.49 / 0.30 / 0.36 | 1.14 / 1.91 / 0.52 / 1.11 |

Results at 50% selected follow the same pattern (`res_all.csv` in my agent directory).

With `-H` (concat plus HashGroup's hash, stream), the scalar Murmur hash of a 12/16-byte key dominates:
- Q3: 3.36 -> 3.12;
- Q3p: 3.66 -> 3.33;
- ssb2p: 5.30 -> 3.24;
- Q1: row_masked+simdhash 0.68 vs row+simdhash 0.61 vs fused_simd 0.51.

**Summary.**
- RowMasked is within 0..22% of `row_tmpl` on the instantiated-style shapes:
  - equal on 1s2s4s and 2s2s2s2s;
  - +21% on Q1 (l1), +22% stream;
  - +18% on 4s4s;
  - +25..30% on 1s1s1s1s4s.
- It is the fastest kernel for every shape it newly covers:
  - Q3 1.2..1.4x over the copy;
  - SSB q2.x 1.3..1.5x over wide_fixed;
  - odd widths 4..8x over the copy.
- It does **not** replace the Q1 instantiation. Under `auto` Q1 keeps `row_tmpl`, and all-dense keys keep vbmi.
- The guide's goal was met only in part: masked is within ~10% of `row_tmpl` on 3 of 6 shapes, not all.
- **RowGeneric (GPR) is a negative result on x86.** It is 2..3x slower than `row_tmpl` and slower than the per-column copy on 1/2/4/8-byte shapes. It beats the copy only on odd widths, and there `wide_fixed` / RowMasked beat it. It is therefore not used under `auto`.
- The earlier `row_gpr` (0.26x of tmpl) problem is the same: runtime shift counts plus register pressure. The `__restrict__` and local-variable fixes removed the aliasing reloads, but not the spills.

**Tests.** `test_all --gtest_filter='GroupConcat*:HashGroup*'` passes (12 tests) in:
- x86 with `ROW_GENERIC=ON`;
- x86 with `ROW=OFF + ROW_GENERIC + KEY_PAD + GROUP_DISPATCH + BATCH_CREATE + CRC32`;
- x86 with `KERNEL=gpr + KEY_PAD`;
- aarch64 qemu (burrata, `KERNEL=gpr`, `ROW=OFF`, `KEY_PAD`).

New tests:
- `GenericRowKernels`: 150 random shapes per kernel with widths 1..16, mixed sel/dense, padded keys, shared and separate selection vectors, and both end and **front** guard pages.
- `GenericQueryShapes`: Q1, Q3, Q3p, q2.x, q2.x padded, 4d7s, 16s, plus a K = 17 refusal.

Mutation check: zeroing RowGeneric's guards makes `GenericRowKernels` segfault on the front guard page.

## Risks / open questions
- **Sapphire Rapids:** a byte-masked load may be 2 uops (load + p05), and `imul` runs only on p1. The laptop gap to `row_tmpl` (+21% on Q1) may be larger on manchego. This needs the cluster.
- **dubliner** (Cascade Lake) has AVX-512BW/VL, so `auto` picks RowMasked there. Its ops are xmm-only, so there should be no frequency license, but this is unmeasured.
- The HashGroup gtests do not have a Q3-like mixed 12-byte shape. The end-to-end check that Q3 and SSB q2.x results are unchanged is left for the cluster.
- RowGeneric reads up to 7 bytes before a value. That is safe by construction (guards), but ASan / valgrind would flag reads before the column pointer.
- The 8d8d all-dense choice (vbmi 0.85 vs masked 0.47) is the existing shuffle policy. A plan-fact rule "K > 8 all-dense -> masked" could be added; I did not change it.
- Fusing the hash (`concatHash`) is not done for generic plans; Q3's 16-byte key hash dominates anyway.

## Configs to test (`scripts/flag_ablation.sh`)
- `hash_simd_concat_generic` = `hash_simd_concat_wide_pad` + `ROW_GENERIC`. Q3 and SSB q2.x are the targets; compare against `hash_simd_concat_wide_pad`.
- `hash_simd_concat_generic_q1` = the above + `ROW=OFF`. This puts the masked kernel on Q1, to measure the gap to the instantiated kernel end to end.
- `fair_concat_generic` = `fair_concat` + `ROW_GENERIC`, the fair ladder (CRC32).

Run on manchego and dubliner. burrata is unaffected under `auto`.
