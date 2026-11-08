# Agent E: row-wise wide concat with masked moves

Branch base `45b8007` (group_concat). Laptop: Zen 4 (AVX-512BW/VL, VBMI),
gcc 16, Release. All numbers are indicative only: other agents were
building on the same machine part of the time (load average 5-15).

## Design

New option `VW_GROUP_CONCAT_WIDE_ROW` (default OFF, needs
`VW_GROUP_CONCAT_SIMD`), plus `VW_GROUP_CONCAT_WIDE_ROW_FOR` = `sel`
(default) | `all`.

- Kernel (`src/vectorwise/primitives/Concat.cpp`, `wideRowKernel<D, S, V,
  Shared>`): one pass over the rows. Per row: one load of the shared
  selection index (or one per selection-vector column when they read
  different selection vectors). Per column: one byte-masked load
  (`vmovdqu8 {k}{z}`, `_mm{,256,512}_maskz_loadu_epi8`) and one byte-masked
  store (`_mm*_mask_storeu_epi8`) of exactly the value's bytes. No byte
  past the value is read, so there is no over-read and no fault at a page
  end.
  - Templated on the number of dense (D) and selection-vector (S) columns,
    not on the widths. Widths, masks and offsets are plan data copied into
    locals.
  - The plan orders the dense columns first. They are walked by pointers
    that advance by their width; only the selection-vector columns
    multiply (`idx * w`).
  - V (16/32/64) is the narrowest register that holds the widest column.
  - 44 (D, S) combinations x 4 V (0 = fallback) x 2 (Shared) entries, in a
    constexpr table `kWideKernels`.
- Fallback without AVX-512BW (`V = 0`): the same loop with two overlapping
  fixed-size moves per value (copyWideColumn's), chosen per column by a
  switch. Compiled everywhere (tests, run_concatbench), but **not planned**:
  it is slower than the per-column `wide_fixed` (see below).
- API (`SimdConcat.hpp`):
  - `Isa::WideRow`, `buildWideRow(plan, cols, width, offset, sel, keySize,
    forceFallback)`, also reachable as `build(Isa::WideRow, ...)`;
    `wideRowMasked()`.
  - Plan fields `wide`/`wideSep`/`wideBytes`/`order`/`moveCode`/`byteMask`.
  - `run()` dispatches `Isa::WideRow`: Shared kernel if every sel column's
    pointer is the same, else the separate-index kernel.
- Plan hook (`HashGroup::planConcat`, `Operators.cpp`): after the Row
  attempt, before the vector kernels. WideRow is taken when
  `wideRowMasked()` holds (AVX-512BW build) and
  - `FOR=sel`: some column of a width other than 1/2/4/8 is read through
    a selection vector (SSB q2..q4);
  - `FOR=all`: any such column (also TPC-H Q9, Q18).

  It runs from `Concat` and `concatResolved` (`VW_GROUP_DISPATCH`) through
  `concatSimd`. `concatHash` is unaffected: WideRow is not `Isa::Row`, so
  Concat + Hash run as before.

## Difference from the current specification

- packedKeys layout: unchanged and byte-identical. Hashes are untouched
  (computed afterwards by `Hash_T`, same input bytes).
- Padding (`VW_GROUP_KEY_PAD`): pad bytes are never written, so they stay
  zero, as with Concat_T. Tested with 0..3 trailing pad bytes; the test
  checks that untouched bytes keep their sentinel.
- Writes stay within `[0, n*K)`. Unlike the vector kernels, nothing goes
  into kSlack.
- The plan decision uses only plan facts and ISA: key widths, which
  columns are selection-vector columns, AVX-512BW.
- Whether the sel columns share one selection vector is checked per call
  by pointer equality, like the existing Row kernel. Both outcomes run a
  WideRow kernel; there is no switch to another path. In a plan this is
  fixed by which Buffer each key reads.
- Columns over 64 bytes, overlapping columns, or more than 8 columns: build
  fails and the existing path is used.
- Observed while doing this: the guide's shapes use the nominal Char
  lengths. The packed widths are `sizeof(Char<N>)` = N + 1 (Char<25> is 26
  bytes, Char<15> is 16, Char<10> is 11, Char<9> is 10, Char<7> is 8).
  - Real shapes: Q9 `26d4d`, Q18 `26d4d4d8d4s`, q2.x `4d10s`, q3.1
    `4d16s16s`, q3.2-3.4 `4d11s11s`, q4.1 `4d16s`, q4.2 `4d8s16s`, q4.3
    `4d10s11s`.
  - In SSB q3.x / q4.x the two sel key columns read **different**
    selection vectors (`lineorder_date_customer` vs `lineorder_date`,
    `sbb/queries/q31.cpp`). A shared-index-only kernel (like Row) would
    fall back on exactly these shapes, hence the separate-index form.

## Expected work reduction (per row, Zen 4 code from gcc 16)

Instruction counts are from the hot loops (`objdump -d --no-show-raw-insn
-C run_concatbench`).

- **SSB q3.x, `4d16s16s` (separate selection vectors),
  `wideRowKernel<1,2,16,false>`.** 16 instructions per row:
  ```
  vmovdqu8 (%rcx),%xmm0{%k3}{z}          ; dense d_year
  vmovdqu8 %xmm0,(%r8){%k3}
  mov -0x4(%rax),%edx ; imul %r13,%rdx  ; sel idx 1
  vmovdqu8 (%r15,%rdx,1),%xmm0{%k2}{z} ; vmovdqu8 %xmm0,(%r9,%r8,1){%k2}
  mov -0x4(%rsi),%edx ; imul %rbx,%rdx  ; sel idx 2
  vmovdqu8 (%r12,%rdx,1),%xmm0{%k1}{z} ; vmovdqu8 %xmm0,(%rdi,%r8,1){%k1}
  + 4 add, cmp, jne
  ```
  - Before (`wide_fixed`): about 6 instructions for the 4d pass plus 11 per
    16s pass (`copyWideColumn` overlap<16>: idx load, imul, add, 2 x 16 B
    loads, 2 x 16 B stores, add, cmp, jne), so about 28 instructions in 3
    passes over packedKeys. Loads 1 + 3 + 3 = 7, stores 1 + 2 + 2 = 5.
  - After: 16 instructions in 1 pass. Loads 5 (3 values + 2 indexes; 4 if
    shared), 3 masked stores.
  - The overlapping-moves path needs 2 moves for every width that is not a
    power of two (10, 11, 26); the masked move needs 1.
- **SSB q2.x, `4d10s`.** About 17 instructions, loads 1 + 3 = 4, stores 3,
  2 passes. Becomes about 12 instructions, loads 3, stores 2, 1 pass.
- **Q18, `26d4d4d8d4s`, `wideRowKernel<4,1,32,true>`.** 25 instructions per
  row:
  - 5 masked loads + 1 index load, 5 masked stores, 1 imul;
  - 4 pointer bumps reloaded from xmm spills (register pressure: 5 column
    pointers, 5 output offsets, 5 masks);
  - 1 pass instead of 5.
  - Before: about 34 instructions, 7 loads, 6 stores, 5 passes.
- **Q9, `26d4d`.** 2 passes become 1; 3 loads + 3 stores become 2 + 2.
- **Not affected:** Q1 `1s1s`, which keeps the Row kernel, and Q3
  `4s4d4d`, which has no wide column.

## Measured (laptop, run_concatbench, ns per selected row)

`taskset` pinned, `-r 15`, best of two runs. `fixed` = `wide_fixed`
(VW_GROUP_CONCAT_WIDE, the current best for these shapes), `row` =
`wide_row`, `moves` = `wide_row_moves` (the fallback). All variants were
verified against the reference (`ok` = 1 everywhere).

Shared selection vector (default `run_concatbench`):

| shape         | mode   | sel% | scalar | fixed | row  | moves | fixed/row |
|---------------|--------|------|--------|-------|------|-------|-----------|
| 4d10s (q2)    | l1     | 98   | 2.77   | 0.76  | 0.55 | 1.87  | 1.38x |
| 4d10s         | stream | 98   | 2.92   | 1.05  | 1.02 | 1.82  | 1.03x |
| 4d16s (q4.1)  | l1     | 98   | 3.43   | 1.56  | 0.98 | 2.85  | 1.59x |
| 4d16s         | stream | 98   | 2.94   | 1.61  | 1.30 | 2.58  | 1.23x |
| 4d16s16s      | l1     | 98   | 4.62   | 1.53  | 0.93 | 4.12  | 1.65x |
| 4d16s16s      | stream | 98   | 5.46   | 5.55  | 3.09 | 5.75  | 1.79x |
| 4d11s11s      | l1     | 98   | 5.05   | 1.55  | 0.98 | 2.90  | 1.58x |
| 4d11s11s      | stream | 98   | 5.62   | 2.65  | 1.82 | 3.96  | 1.45x |
| 4d8s16s       | l1     | 98   | 3.13   | 1.27  | 0.97 | 4.27  | 1.30x |
| 4d10s11s      | l1     | 98   | 5.56   | 1.48  | 0.95 | 2.86  | 1.56x |
| 26d4d (Q9)    | l1     | 98   | 3.32   | 0.87  | 0.90 | 1.77  | 0.97x |
| 26d4d         | stream | 98   | 3.11   | 1.79  | 1.61 | 2.24  | 1.12x |
| 26d4d4d8d4s   | l1     | 98   | 3.76   | 2.16  | 2.32 | 5.71  | 0.93x |
| 26d4d4d8d4s   | l1     | 50   | 3.36   | 1.76  | 2.14 | 5.63  | 0.82x |
| 26d4d4d8d4s   | stream | 98   | 5.47   | 4.24  | 2.75 | 5.82  | 1.54x |

`-D` (each sel column its own selection vector, as in the real SSB
q3.x / q4.x plans):

| shape     | mode   | sel% | scalar | fixed | row  | moves | fixed/row |
|-----------|--------|------|--------|-------|------|-------|-----------|
| 4d16s16s  | l1     | 98   | 5.33   | 1.52  | 0.90 | 3.91  | 1.68x |
| 4d16s16s  | stream | 98   | 5.72   | 3.28  | 2.14 | 4.61  | 1.54x |
| 4d11s11s  | l1     | 98   | 4.83   | 1.48  | 0.92 | 2.43  | 1.61x |
| 4d11s11s  | stream | 98   | 5.40   | 2.54  | 1.65 | 2.70  | 1.54x |
| 4d8s16s   | l1     | 98   | 2.79   | 1.22  | 0.92 | 3.99  | 1.32x |
| 4d10s11s  | l1     | 98   | 4.83   | 1.46  | 0.94 | 2.41  | 1.55x |
| 4d10s11s  | stream | 98   | 5.60   | 2.43  | 1.59 | 2.58  | 1.53x |

At 50% selected the SSB rows are within a few percent of the 98% ratios.
Full CSVs:
`/tmp/claude-1000/-home-lxbtlr-projects-allocator/f3345f44-08d7-4f85-bd57-235088e5fb55/scratchpad/agents/e-wide-row/final_{shared,shared2,dist,dist2}.csv`.

Summary:

- **SSB keys (a wide column read through a selection vector):** 1.3-1.7x
  over `wide_fixed` in L1, 1.0-1.8x streaming. This holds with shared and
  with separate selection vectors.
- **TPC-H Q9 / Q18 (wide columns dense):** flat to 0.82x in L1, 1.1-1.5x
  only when streaming from memory. The join output buffers HashGroup reads
  are vector-sized and hot, so L1 is the relevant mode. Hence the default
  `FOR=sel` excludes them.
- **Fallback (`wide_row_moves`, the non-AVX-512BW form):** 1.5-3x slower
  than `wide_fixed`, because of a per-column indirect jump through the
  switch's jump table. Hence it is not planned. On burrata (NEON) the
  option changes nothing: a clean negative result for the overlapping-moves
  row form.
- End-to-end effect: concat is only a part of SSB q2..q4 HashGroup time.
  The earlier `wide_fixed` gave about 2% on q3.1/q3.2, so expect at most
  about 1% more. That needs a cluster run.

Tests:

- **x86 (`-DVW_GROUP_CONCAT_SIMD=ON -DVW_GROUP_CONCAT_WIDE=ON
  -DVW_GROUP_CONCAT_WIDE_ROW=ON`, `FOR=sel` and `FOR=all`):**
  `test_all --gtest_filter='GroupConcat*:HashGroup*'` passes 11/11.
  - New `GroupConcat.WideRow`: 200 random shapes with 1..8 columns of
    1..64 bytes, mixed dense and sel, shared and separate selection
    vectors, unsorted and sorted selections, 0..3 pad bytes; columns at
    PROT_NONE guard pages; masked and fallback kernels; output past
    `n*K + kSlack` and pad bytes checked untouched; columns given out of
    offset order; rejected shapes.
- **aarch64 (`TARGET_MACHINE=burrata`, qemu):** 11/11 pass. The fallback
  kernel is exercised there.
- **Default build (all options OFF):** compiles, HashGroup tests pass.
- **New `HashGroupT.DISABLED_wideCompoundKey`** (Char(25) + int32 through
  HashGroup): group keys are correct in every build. With all concat
  options OFF too, the aggregates come out wrong in most runs (all 750 or
  none) and GlobalPool's teardown throws "Memory unmapping failed". This is
  a pre-existing harness or operator problem with Char keys, not chased;
  the test is left disabled with that note. With `FOR=all` it takes the
  WideRow path (3/3 runs had correct aggregates; run-to-run, as in
  baseline).

## Risks / open questions

- **Top risk: masked-load assists on Intel.** On Sapphire Rapids and
  Cascade Lake, a masked load whose masked-off lanes cross into an
  unmapped page takes a microcode assist (about 100+ cycles). It is rare:
  at most once per column end per vector. Masked-store throughput there
  is unmeasured here, and the Zen 4 L1 loss on Q18 suggests 5 masked
  stores per row can cost more than plain stores. manchego and dubliner
  numbers are needed.
- dubliner: only xmm/ymm masked moves for widths up to 32 bytes (all
  TPC-H/SSB keys), so no AVX-512 frequency license issue is expected.
  zmm is used only for columns of 33..64 bytes.
- Q18 register pressure (5 columns): spills of the pointer increments
  to xmm. A `FOR=all` cluster run would show whether Q18 loses end to
  end.
- The shared-vs-separate selection-vector choice is a pointer compare per
  call (as in Row); both choices run WideRow kernels.
- The fallback could be made branch-free (move class per column as a
  template mask), but per-column templates are what this idea avoids;
  not pursued.

## Cluster configs (`scripts/flag_ablation.sh`)

- `hash_simd_concat_wide_row` = `hash_simd_concat_wide` +
  `VW_GROUP_CONCAT_WIDE_ROW=ON`. Compare with `hash_simd_concat_wide` on
  SSB q2.x/q3.x/q4.x, manchego and dubliner.
- `hash_simd_concat_wide_row_all`: `+ _FOR=all`. TPC-H Q9, Q18 vs
  `hash_simd_concat_wide_row`.
- `fair_concat_row` = `fair_concat` + WIDE_ROW (with KEY_PAD: q2.x 14 -> 16
  bytes), and `fair_concat_row_all`.
- burrata: identical to the base configs by construction (no AVX-512BW).
