# Porting the AVX-512 VectorWise kernels to aarch64 (NEON, SVE)

Date: 2026-10-04. Branch `query1` (HEAD `eff34c2`) plus the uncommitted files listed under "Deliverables".

Scope: every kernel behind a VectorWise (and one Typer) CMake option that uses AVX-512 intrinsics. For each one, the study answers three questions:
- What should the aarch64 build use: the SIMDe translation, a hand NEON kernel with the same algorithm, an SVE kernel, or the scalar path?
- What throughput should each option reach on Neoverse N1 (burrata)?
- Is the kernel compute-bound or memory-bound there?

**Evidence.** No ARM hardware was used. The numbers come from three sources:
- dynamic aarch64 instruction counts under qemu;
- llvm-mca hot-loop throughput for Neoverse N1 and V1;
- a native AVX-512 reference run on a Zen 4 laptop.

The one exception is the CRC path: burrata's existing `run_crcbench` output measures it directly (`results/crcbench.out`). All other wall-clock numbers for ARM have to come from a burrata run of the new bench (§7).

## TL;DR

1. **On aarch64 today, every AVX-512 kernel compiles out, and the scalar templates run instead.** There are 14 sites (§1). On burrata these options split two ways:
   - **No effect at all:** `VW_SIMD_SEL`, `VW_SIMD_HASH`, `VW_CRC32_VPCLMUL`, `VW_JOIN_SIMD`.
   - **Only the scalar fallback runs:** `VW_JOIN_BLOOM`, `VW_JOIN_SEMI`, `VW_NEW_JOIN`, `VW_GROUP_RUN_HEADS`, `VW_PROJ_DENSE`.
2. **Translating the kernels with SIMDe is not viable.**
   - **Cost.** On aarch64, the SIMDe translations of the mask, compress and gather kernels run 1.4–4× more instructions than the scalar engine code they would replace. Examples:
     - selection: 17.3 vs 7.0 instructions/element;
     - Bloom probe: 81.6 vs 21.0;
     - new-join first pass: 71.0 vs 19.6;
     - run heads: 16.6–18.9 vs 12.0;
     - MurmurHash: 27.7 vs 13.1.
   - **Causes.** Mask compares, compress stores and gathers all become per-lane scalar loops.
   - **`-ftree-vectorize` barely matters.** `run_neonbench_autovec` changes one SIMDe row (Bloom 81.6 → 74.5). It vectorizes the scalar dense projections instead (7.15 → 3.10).
   - **The exception is plain arithmetic.** The dense projections translate well: 1.96–4.0 vs 7.2 instructions, because the add, sub and per-lane multiply have no masks. A hand NEON loop still beats SIMDe on minus and plus (1.07).
   - **Missing intrinsics.** SIMDe (the 2026-02 snapshot in `3rdparty/simde`) does not provide 15 intrinsics that the engine's kernels use (§3). Each was checked by compiling a one-line use for aarch64.
   - **Unsafe masked gather.** `simde_mm512_mask_i64gather_epi64` loads the masked-off lanes too. `joinNewFirstPass` gathers through null heads, so the translated kernel segfaults (it crashed the clang build of the bench).
3. **Port to NEON, same algorithm** (gain in llvm-mca N1 cycles per element over the scalar path):
   - **selection** (`VW_SIMD_SEL`): 3.01 → 1.75 for int32 (1.7×), 3.01 → 2.0 for int64 (1.5×);
   - **run-head pass** (`VW_GROUP_RUN_HEADS`): 3.76 → 2.0 for 2-byte keys (1.9×), 3.76 → 2.76 for 4-byte keys (1.36×);
   - **dense minus/plus** (`VW_PROJ_DENSE`): 2.51 → 0.88 (2.9×).

   NEON has no compress instruction. The kernels replace it with a 16-entry `TBL` lookup plus a full-register store that may overwrite past the end (§3).
4. **Keep the scalar path on N1:**
   - **MurmurHash (`VW_SIMD_HASH`).** NEON is 10.2 vs 12.0 cycles/key, but scalar CRC32 is 5.0. Use `VW_USE_CRC32`.
   - **The VPCLMUL CRC (`VW_CRC32_VPCLMUL`).** PMULL measured on burrata: 3.11 ns/key, against 1.20–1.41 for `crc32cx`.
   - **Dense multiply.** NEON is 3.26 vs 3.02. NEON has no 64-bit lane multiply, and both forms are bound by multiply throughput.
   - **Gathered-input selection (`selsel`).** 3.27 vs 3.52, within model error.
   - **Bloom and semi bit tests.** The NEON versions are 8.8 vs 7.5 and 5.8 vs 5.0: worse than scalar.
5. **The join kernels need a rework, not a port.**
   - **Where the AVX-512 win comes from.** In the Bloom filter, the semi bitmap probe and the new-join first pass, AVX-512 wins by issuing 8 or 16 independent loads per gather. That is memory-level parallelism, not arithmetic. NEON has no gathers.
   - **The Bloom filter (x86, 128 MiB):** scalar code plus a software prefetch already matches the AVX-512 kernel (7.9 vs 8.2 ns/probe).
   - **The new-join first pass (x86, 16 and 128 MiB):** the AVX-512 version is slower than the scalar 8-lane block (15.5 vs 11.0, and 20.8 vs 15.8).
   - **The rework:** keep these kernels scalar and add prefetch, or group prefetching for the join. Size the filters for N1's L2.
6. **SVE (Neoverse V1/V2) maps almost one-to-one onto AVX-512** (COMPACT, gathers, 64-bit `MUL`). On V1, llvm-mca gives these gains over scalar:
   - gathered selection 1.8×;
   - dense multiply 2×.

   (NEON alone already gives dense minus/plus 2.7× on V1.)

   Two cases where SVE does not help:
   - MurmurHash with SVE stays at scalar speed (2.26 vs 2.62).
   - For contiguous selection, `COMPACT` (0.88 cycles/element) is slower than the NEON TBL kernel (0.64). So contiguous selection should stay NEON even on V1.

   burrata (N1) and rpi5 (Cortex-A76) have no SVE.
7. **Next step:** run `study/neon_port/run_target.sh burrata` (§7). It measures the memory side (sweep and stream modes), which qemu and llvm-mca cannot. Then wire the PORT set following the skeleton in `SimdNeon.hpp`.

## Deliverables

| file | what |
|---|---|
| `src/benchmarks/primitives/neonbench.cpp` | `run_neonbench` (and `run_neonbench_autovec`, the same source without `-fno-tree-vectorize`). 8 kernel families plus roofline rows. Each row is checked against the engine's scalar output. |
| `include/vectorwise/SimdNeon.hpp` | Contains three things: the NEON building blocks (lane mask, TBL compress, 64-bit multiply); the NEON/SVE kernels the bench measures; and the engine-wiring skeleton, a per-site decision table plus a `pick_sel_col_val` picker. Not called by the engine. |
| `CMakeLists.txt` | the two targets |
| `study/neon_port/` | qemu instruction-count plugin and driver, llvm-mca harness and driver, target run script, data (`README.md` lists them) |

All five builds compile without warnings in the new files, and every row passes its check (under qemu for aarch64):
- x86-64 gcc and clang, native AVX-512;
- aarch64 gcc and clang, `TARGET_MACHINE=burrata`;
- aarch64 gcc, `TARGET_ARCH=neoverse-v1` with SVE, run at vector lengths 256 and 512.

## 1. What runs on aarch64 today

Each site's guard requires x86 AVX-512, so an aarch64 build compiles the `#else` (scalar) side:

| site | guard | option | TPC-H |
|---|---|---|---|
| `SimdSelection.hpp:28` sel/selsel/char kernels | `__x86_64__ && __AVX512F__` (+BW, VL) | `VW_SIMD_SEL`, `_GATHER`, `_CHAR` | Q1, Q3, Q5, Q6, Q18 |
| `Selection.cpp:125` old `*_avx512` kernels | `__AVX512F__` | `SIMDsel=1` env | Q1, Q6 |
| `SimdHash.hpp:31` MurmurHash ×8 | `__AVX512F__ && __AVX512DQ__` | `VW_SIMD_HASH` | joins, group-bys |
| `SimdCrc.hpp:31` CRC32 ×8 | + `__VPCLMULQDQ__` | `VW_CRC32_VPCLMUL` | joins, group-bys |
| `Hash.cpp:80`, `Primitives.hpp:1359` hash4/hash8, proj8 | `__AVX512F__` | `SIMDhash` env | — |
| `Projection.cpp:159` dense projections | `__AVX512F__` / `__AVX2__` | `VW_PROJ_DENSE` | Q1 |
| `Operators.cpp:346,658` `joinAllSIMD`/`joinSelSIMD` | `__AVX512F__` | `VW_JOIN_SIMD` | Q3, Q5, Q9 |
| `Operators.cpp:1008` `bloomFilter` | `__AVX512F__ && !VW_POS_16` | `VW_JOIN_BLOOM` | Q3, Q5, Q9 |
| `Operators.cpp:1114` `fusedHashFilter` / `fusedHash8` | same | `VW_JOIN_FUSED_PROBE` | Q3, Q5, Q9 |
| `Operators.cpp:918` `semiProbe` | same | `VW_JOIN_SEMI` | Q3, Q9, Q18 |
| `Operators.cpp:1259` `joinNewFirstPass` | `__AVX512F__ && HASH_SIZE 64 && !VW_POS_16` | `VW_NEW_JOIN` | Q3, Q5, Q9 |
| `Operators.cpp:2027` `runHeadMask16` | `__AVX512BW__ && __AVX512VL__` | `VW_GROUP_RUN_HEADS` | Q1, Q18 |
| `q1.cpp:14` Typer fused Q1 loop | `__AVX512F__ && DQ && VL` | `HYPER_Q1_SIMD` | Q1 (Hyper) |
| `SimdAggr.hpp:32` grouped SUM kernels | `__AVX512F__ && VL && BW` | — (only `run_aggrbench`) | — |

Two more facts about the aarch64 build:
- **`SIMD.hpp` already includes SIMDe's AVX-512 headers on ARM.** This keeps `Vec8u`/`Vec16u` compiling, but no kernel uses them there.
- **`Hash.hpp` gets `_mm_crc32_u64` from SIMDe.** SIMDe lowers it to the native `crc32cx`, so `VW_USE_CRC32` already runs native code on ARM.

**Consequence for the flag ablation on burrata** (`FLAG_ABLATION_STUDY.md`, `VW_OPTIONS.md`). Its `VW_SIMD_SEL` rows were scalar. Its `grp_runheads` Q1 loss is consistent with the scalar run pass: dispatch alone is 78.8 ms, run heads 84.2 ms. Its `join_bloom` and `join_semi` gains come from skipping work, not from SIMD.

## 2. Method

**`run_neonbench`.** One binary per target, with one family per AVX-512 kernel. Each family runs several implementations:

| impl | what it is |
|---|---|
| `scalar*` | the engine's scalar template (called directly, e.g. `sel_col_val_bf`), or its scalar loop copied with a `file:line` citation |
| `avx512_simde` | the engine's AVX-512 kernel, spelled with `simde_` names. On AVX-512 x86 this is the native kernel; on aarch64 it is SIMDe's emulation. Intrinsics SIMDe lacks use local polyfills (§3). |
| `neon*`, `sve` | `SimdNeon.hpp` |
| extra scalar variants | `scalar_bf` (branch-free), `scalar_bf_pf` (+ software prefetch), `scalar_2phase` (group prefetch). These test the memory-stall question. |

The bench has three modes:
- `l1`: one 1024-row vector, cache-resident;
- `stream`: a 64 MiB column, vector by vector, as a Scan feeds it;
- `sweep`: the random-access families against tables of 32 KiB–128 MiB.

There are also roofline rows: 1-thread read bandwidth, dependent-load latency, and independent-load throughput per working-set size. Output is CSV, one row per (family, impl, type, mode, param). Every row's output is checked against the family's first implementation (count plus order-sensitive checksum), and the process exits with status 1 on any mismatch. The checked inputs include sign-extension edge keys and every selectivity from 1% to 99%.

**Dynamic instruction counts (aarch64).**
- Tooling: `study/neon_port/insncount.c` is a QEMU plugin; `insns.sh` runs each row with 1 and with 3 timed trials and divides the difference by the elements processed, which removes setup and checking.
- Counts are exact for the compiled code. They are not time.

**llvm-mca (aarch64).**
- What runs: `mca.sh` compiles `mca_kernels.cpp` with the engine's flags (`-O3 -fno-tree-vectorize`, the N1 or V1 profile), extracts each kernel's hot loop, and runs `llvm-mca -mcpu=neoverse-n1|neoverse-v1`.
- What it assumes: L1-resident data, no mispredicts, no misses.
- **LLVM's N1 model dispatches 3 µops per cycle.** So on N1, these loops are front-end bound and µop count is close to time. That matches the flag study's observation that "hashing cost per instruction is high on N1".
- Calibration against measured burrata: scalar CRC32Hash is 5.0 cycles/key in the model and 1.41 ns/key measured, about 4.2 cycles at ~3 GHz. **The burrata clock was not checked.** If that clock is right, the model is about 20% pessimistic.

**x86 reference.** `run_neonbench` on a Ryzen 7 7840U (Zen 4, native AVX-512, 1 pinned core, laptop power management): `study/neon_port/data/x86_zen4_gcc_full.csv`. This run is used only for the memory-sweep shapes and as the AVX-512 ceiling.

## 3. Intrinsic map: AVX-512 → aarch64

| AVX-512 construct (engine use) | NEON | SVE | SIMDe on aarch64 |
|---|---|---|---|
| `_mm512_cmp_ep{i32,i64}_mask` → `__mmask16` (every kernel) | `vcltq/vcgtq/vceqq…` give all-ones lanes; a GPR bitmask costs AND + `ADDV` + `FMOV` (`mask4`). `mask16` folds 4 compares into one `ADDV`. | predicate from `svcmp*` | per-lane loop building the mask; **`_mm512_mask_cmp_epi32_mask` (masked tails) missing** |
| `_mm512_maskz_compress_epi32` + masked store (`emit`, `compressStore32/64`) | `vqtbl1q_u8` with a 16-entry byte table, full `vst1q` store, advance by a 16-entry popcount table (`compress4`, `emit4`). Writes up to 3 lanes past the last match: the "store-overwrite" contract, safe for all engine output buffers (§6). | `svcompact` + `svst1` under `svwhilelt` | per-lane loop with a branch per lane |
| `_mm512_mask_compressstoreu_*` | same | same | same |
| `_mm512_i32gather_*`, `_mm512_i64gather_epi64` (selsel, bloom, semi, new join) | none: lane loads (`vld1q_lane`) or scalar; compute indices in GPRs from memory, not by extracting lanes | `svld1_gather_*index` | `i64gather`: per-lane `memcpy`. **Missing: `_mm512_i32gather_epi{32,64}`, `_mm512_mask_i32gather_epi{32,64}`, `_mm256_mmask_i32gather_epi32`** |
| `_mm512_mask_i64gather_epi64` (new join, `joinAllSIMD`) | — | predicated gather | **unmasked gather + blend: loads masked-off lanes, faults on null heads** |
| `_mm512_mullo_epi64` (MurmurHash, dense multiply, CRC finalizer) | none: 3 `UMULL`/`UMLAL` + narrows per 2 lanes (`mullo64`) | `svmul_u64` | per-lane scalar multiply |
| `_mm512_sllv/srlv_epi64/32` (bloom, semi, tag test) | `vshlq_u64/u32` with a signed shift vector | `svlsl/svlsr` | native |
| `_mm512_test_epi64_mask` | `vtstq_u64` | `svcmpne(svand)` | native |
| `_mm512_cvtep{i,u}{8,16,32}_epi64` (`widen8`, `SIMD.hpp`) | `vmovl_s32` / `vmovl_high_s32` | `svld1sw` | **all 6 missing** |
| `_mm512_maskz_loadu_*` (masked tails) | none: scalar tail | `svwhilelt` predicate | native |
| `_mm512_mask_cvtepi32_storeu_epi16` (`VW_POS_16`) | `vmovn_u32` + store | `svst1h` | **missing** |
| `_mm_maskz_set1_epi8` (run-head flags) | `vmovn` of the compare, AND 1 | predicate → `svdup` | **missing** |
| `_mm512_reduce_add_epi64` (SimdAggr) | `vaddvq_u64` | `svaddv` | **missing** |
| `_mm512_clmulepi64_epi128` (VPCLMUL CRC) | `vmull_p64` (PMULL), 1 product per instruction | `svpmullb_pair` (SVE2) | native |
| `_mm_crc32_u64` (CRC32Hash) | `__crc32cd` | — | native (`crc32cx`) |
| 512-bit registers | 128-bit: 4 × int32 / 2 × int64 per instruction | 128–2048 bit (V1 256, V2 128) | split into 4 × 128-bit |

**What does not port well.**
- **Compress through a lane mask.** It is an `ADDV` + cross-domain move per vector. That is only cheap when amortized, as `mask16` does over 4 compares.
- **Gathers.** No NEON form.
- **64-bit multiplies.** No NEON lane form.

**What ports well.** Compares, shifts, bitwise ops, and contiguous loads and stores.

## 4. Per kernel

Units:
- `insns` = dynamic aarch64 instructions per element (gcc 16, N1 build);
- `N1` / `V1` = llvm-mca cycles per element of the hot loop;
- `x86` = Zen 4 ns per element, native AVX-512 vs scalar, `l1` mode.

Element sizes are per family. Selection params are selectivity %; the representative row is 50% (counts do not depend on selectivity for the branch-free and SIMD kernels).

| family (bench row) | engine code | scalar insns / N1 / V1 | SIMDe insns | NEON insns / N1 / V1 | SVE insns / V1 | x86 AVX-512 vs scalar | decision |
|---|---|---|---|---|---|---|---|
| `sel` int32 | `SimdSelection.hpp:235` `sel_col_val_v` | 7.04 / 3.01 / 1.02 | 17.30 | 3.57 / 1.75 / 0.66 (`neon16`) | 1.52 / 0.88 | 0.069 vs 0.455 | **PORT** `sel_col_val_x16` |
| `sel` int64 | same | 7.04 / 3.01 / 1.02 | 14.67 | 4.19 / 2.07 / 0.75 | — | 0.108 vs 0.456 | **PORT** |
| `selsel` int32/int64 | `SimdSelection.hpp:594` `selsel_col_val_hwgather` | 9.04 / 3.52 / 1.36 | 18.04 / 16.76 | 6.59 / 3.27 / 1.68 | 1.59 / 0.76 | 0.44 vs 0.50 | **SCALAR** on N1, **SVE** on V1 |
| `hash` dense (Murmur) | `SimdHash.hpp:74,184` | 13.05 / 12.02 / 2.62 | 27.69 (clang 12.90) | 17.05 / 10.23 / 5.08 | 4.55 / 2.26 | 0.21 vs 1.49 | **SCALAR**: use CRC (scalar 10.05 / 5.02 / 2.02) |
| `hash` sel50 (Murmur) | `SimdHash.hpp:204` | 15.05 / – / – | 34.40 | 19.87 | — | — | **SCALAR** |
| CRC32 VPCLMUL | `SimdCrc.hpp` | burrata: 1.41 ns (2 × crc32cx), 1.20 (`VW_CRC32_FAST`) | — | PMULL Barrett 3.11 ns (`run_crcbench`) | — | — | **SCALAR** + `VW_CRC32_FAST` |
| `proj` minus/plus, 98% | `Projection.cpp:197-240` | 7.15 / 2.51 / 1.02 (gather path 8.04 / 3.02) | 1.96–2.98 | 1.07 / 0.88 / 0.38 | — | 0.083 vs 0.353 | **PORT** `proj_dense_minus/plus` |
| `proj` multiplies, 98% | `Projection.cpp:242` | 7.16 / 3.02 / 1.02 | 4.01 | 6.41 / 3.26 / 1.14 | 2.08 / 0.51 | 0.092 vs 0.293 | **SCALAR** on N1, **SVE** on V1 |
| `runheads` uint16 (Q1) | `Operators.cpp:2027,2123` | 12.03 / 3.76 / 1.51 | 18.93 | 4.07 / 2.0 / 0.79 | — | 0.10 vs 0.57 | **PORT** `run_heads<K>` |
| `runheads` uint32 (Q18) | same | 12.03 / 3.76 / 1.51 | 16.60 | 5.80 / 2.76 / 1.06 | — | 0.11 vs 0.57 | **PORT** |
| `bloom` | `Operators.cpp:1008` | 21.04 / 7.51 / 4.02 | 81.59 | 19.07 / 8.76 / 3.52 | — | §5 | **REWORK**: scalar + prefetch |
| `semi` | `Operators.cpp:918` | 16.01 / 5.03 / 2.69 | 28.90 | 14.56 / 5.77 / 3.32 | — | 0.43 vs 0.85 (32 KiB) | **SCALAR** (+ prefetch if > L2) |
| `dirprobe` | `Operators.cpp:1274` (AVX-512), `:1340` (scalar block) | 19.61 (`scalar_block8`) | 70.99 | — | — | §5 | **REWORK**: block-8 + group prefetch |

### Selection (`VW_SIMD_SEL`) — port

**How the kernel maps.** The AVX-512 kernel compares 16 lanes, compresses the lane ids under the mask, and stores the match count. The NEON kernel does the same per 4 lanes:
- compare;
- AND with {1,2,4,8} and `ADDV` to a 4-bit mask;
- `TBL` the lane ids with the mask's table row;
- store all 4 lanes, and advance `found` by `kPop4[mask]`.

**Unrolled variant.** `neon16` amortizes the `ADDV` + `FMOV` (the long-latency part) over 4 compares. In the model this matters for int64 (2.07 vs 2.01, about even) and at V1 widths. The win over scalar is about the same either way: 1.7× (int32) and 1.5× (int64) on N1.

**Selectivity.** The branch-free scalar template is flat across selectivity. The branching template only wins above 90% (`scalar_branch`, x86 rows).

**Stream mode.** At 1.75 cycles/element, int32 selection needs about 4 B / 0.58 ns ≈ 7 GB/s at 3 GHz. It stays compute-bound if a burrata core streams more than that. The `roof` row on burrata answers this. On x86, the AVX-512 kernel's lead over scalar shrinks from cache to a 64 MiB stream: 6.6× → 4.6× (int32) and 4.2× → 2.3× (int64). This matches the direction of `FLAG_ABLATION_STUDY.md`.

**SVE `COMPACT` is slower than the TBL kernel on V1** in the model (0.88 vs 0.64). Keep NEON for contiguous selection even where SVE exists.

`sel_col_col` has the same shape with two loads. `VW_SIMD_SEL_CHAR` (byte compare of `{len, bytes}`) was not benchmarked. It needs `vceqq_u8` + a `vminvq` reduction, with no compress inside the element.

### Gathered-input selection (`selsel`) — scalar on N1

**Measured.** NEON with lane loads is 3.27 vs 3.52 cycles/element (7%, model noise). The cost is the 4 scalar loads per 4 elements, which NEON cannot remove.

**Same conclusion as on x86.** This matches the decision behind `VW_SIMD_SEL_GATHER=scalar` on x86, where `vpgather` does not pay either. On V1, SVE's gather halves it (0.76 vs 1.36).

### Hashing (`VW_SIMD_HASH`, `VW_CRC32_VPCLMUL`) — scalar

**MurmurHash.**
- It is 4 dependent 64-bit multiplies per key. In LLVM's model, N1 issues a 64-bit `MUL` every 3 cycles, which gives 12 cycles/key.
- NEON emulates each 64-bit lane multiply with 3 `UMULL`/`UMLAL`. That reaches 10.2 cycles/key: barely better on N1 and 2× worse on V1.
- Scalar CRC32Hash is 5.0 cycles/key (N1). With `VW_CRC32_FAST` it drops to 1.20 ns measured on burrata.

**VPCLMUL CRC.** The carry-less-multiply CRC loses on burrata. PMULL handles 2 keys per register and measured 3.11 ns/key against 1.20–1.41 for `crc32cx` (`results/crcbench.out`).

**What this means for burrata.** It is the mechanism behind the flag study's burrata result: CRC32 is 13–25% faster than MurmurHash there. Use `VW_USE_CRC32` + `VW_CRC32_FAST` and leave both SIMD hash options x86-only.

### Q1 projections (`VW_PROJ_DENSE`) — port minus/plus, keep multiply scalar

**Today's scalar dense loop barely beats the gather path on N1.** The model gives 2.51 vs 3.02 cycles/element, and both are dispatch-bound.

**NEON minus/plus reaches 0.88.** The loop loads, computes and stores 4 registers (8 rows) per iteration with `ld1`/`st1` ×4 and a 64-bit index.
- **Loop shape mattered more than lane width.** The first version (2 registers per iteration, 32-bit `pos_t` index) ran 3.6 instructions per row at 2.0 cycles. On a dispatch-bound core, loop overhead and the index arithmetic dominate.
- **Same lesson from the SIMDe translation.** Its 512-bit version is effectively an 8-row unroll, and it needed only 1.96 instructions per row.

**Multiply does not port.** The emulated 64-bit multiply is 3.26 vs 3.02. In the model, N1 issues one 64-bit `MUL` every 3 cycles, and the 3 `UMULL`s per 2 lanes are no cheaper. On V1, SVE `MUL` halves it (0.51 vs 1.02).

**Effect on Q1's projection block.** Two minus/plus and two multiplies go from about 11.1 (scalar dense) or 12.1 (gather path) to 7.8 cycles per row, about 30% of that block. Q1's total is dominated by the aggregate update (`updateGroupsFused`, scalar, store-forwarding bound) and run heads.

### HashGroup run heads (`VW_GROUP_RUN_HEADS`) — port

**Algorithm.** Pass 1 compares each key with its predecessor (an unaligned load at `i-1`), compresses the head positions and writes 0/1 flags.

**Result.** NEON is 2.0 cycles/element for 2-byte keys (8 lanes per compare, two TBLs) and 2.76 for 4-byte keys, against 3.76 for the scalar loop. On real queries the gain is bounded by pass 1's share of `keyStepT`.

**Context on burrata.** The scalar pass is why `grp_runheads` lost Q1 against `grp_dispatch` there (84.2 vs 78.8 ms) while winning Q18 (1.37×). Passes 2 and 3 are scalar on every ISA.

## 5. Memory stalls: the join kernels

**Bloom, semi bitmap and new-join first pass are latency-bound.** Each does one random load per probe into a table that, at TPC-H scale, ranges from L1 (Q3 customer bitmap, 19 KB) to beyond L2 (Q3's Bloom filter, 300 KB; the Q3/Q9 directories, 2–4 MB). On x86 (Zen 4, ns per probe):

| table | roof: chase / independent loads | bloom scalar / +prefetch / AVX-512 | semi scalar_bf / +prefetch / AVX-512 | dirprobe block8 / 2-phase / AVX-512 |
|---|---|---|---|---|
| 32 KiB | 1.21 / 0.40 | 1.49 / 1.83 / 1.11 | 0.85 / 1.31 / 0.43 | 2.47 / 2.98 / 2.65 |
| 2 MiB | 10.8 / 0.92 | 2.07 / 1.79 / 1.29 | 1.51 / 1.33 / 0.64 | 3.64 / 4.14 / 4.11 |
| 16 MiB | 47.6 / 3.64 | 8.55 / 4.77 / 5.66 | 5.57 / 4.17 / 2.66 | 11.0 / 10.5 / 15.5 |
| 128 MiB | 101 / 5.84 | 12.9 / 7.91 / 8.19 | 8.67 / 6.87 / 5.03 | 15.8 / 16.2 / 20.8 |

What the table shows:
- **Out of cache, the AVX-512 kernels' edge is memory-level parallelism.** A gather puts 8 or 16 independent misses in flight. Scalar code with a software prefetch 16 probes ahead gets the same for the Bloom filter (7.9 vs 8.2 at 128 MiB), and part of it for the semi bitmap (6.9 vs 5.0).
- **The new-join first pass loses with AVX-512 out of cache.** It gathers the head entry of every `full` lane, even when one block has a single candidate. The scalar block loop only touches what it needs.
- **The branch-free variant of the scalar first pass (`scalar_bf`) is slower everywhere.** It is 1.5–1.8× slower with +73% instructions. Over 90% of probes are not `full`, so the branch predicts well; the branch-free form pays a dependent dummy-entry load for all of them.
- **Group prefetching (`scalar_2phase`) helps only at 16 MiB on Zen 4** (10.5 vs 11.0). On N1, which has a smaller out-of-order window, both prefetch forms should matter more. That is the experiment to run on burrata.

**NEON on these kernels.** NEON can only vectorize the arithmetic between the scalar loads. In the model that arithmetic is no cheaper than scalar on N1 (bloom 8.76 vs 7.51, semi 5.77 vs 5.03), and the vector forms add cross-domain moves.

**The rework, ISA-independent.** The algorithms stay; the inner loops change:
- **`bloomFilter`, `fusedHashFilter`:** scalar branch-free test with `__builtin_prefetch` of the word D probes ahead.
  - `VW_JOIN_FUSED_PROBE`'s benefit is not writing all hashes. That survives on ARM as a scalar loop: `crc32cx` (+ linearity), filter test, write survivors.
  - Size `VW_JOIN_BLOOM_BITS` so that Q3's filter fits N1's L2.
- **`semiProbe`:** scalar. Bitmaps in TPC-H are ≤ 750 KB (Q18), so add the prefetch only above L2.
- **`joinNewFirstPass`:** the existing 8-lane scalar block (already the `#else` path), plus a prefetch pass over the vector's directory slots (`VW_JOIN_TWOPHASE`'s idea).
  - `VW_JOIN_TWOPHASE` measured 0.92–0.94 on burrata Q3/Q5 (`VW_OPTIONS.md`). So gate the prefetch on directory size (`VW_JOIN_TWOPHASE_MIN_SLOTS`) and tune it on burrata's sweep.
- **`joinAllSIMD` / `joinSelSIMD`:** no aarch64 version. `joinAllParallel` / `joinSelParallel` remain.

**Streaming kernels.** These are selection, hashing, projections and run heads. Each reads 2–16 B per element at 1–4 cycles per element. Whether they are compute- or bandwidth-bound at 1 thread depends on burrata's single-core read bandwidth (the `roof,read_*` rows). At 128 cores per socket, the shared memory bandwidth will cap all of them well before the compute ceilings.

## 6. What changes in the engine (skeleton)

`include/vectorwise/SimdNeon.hpp` ends with the per-site decision table and a `pick_sel_col_val` for aarch64. Wiring the PORT set means:

1. **Selection.** `SimdSelection.hpp` keeps its x86 block. Next to it, `#elif defined(VW_HAVE_NEON)` makes `pick_sel_col_val` / `pick_sel_col_col` return the NEON kernels (`sel_col_col` needs a two-column variant of `sel_col_val_x16`). `pick_selsel_*` stays scalar.
   - **Output contract.** The store-overwrite contract writes up to `result[found + 3]`, where `found ≤ i` and `i + 4 ≤ n`. That stays inside every selection buffer of `n` entries. The `_bf` templates already rely on writing `result[found]` unconditionally.
   - **`VW_POS_16`.** Excluded by a `static_assert`. It would need 16-bit stores (`vmovn_u32`).
2. **Run heads.** `Operators.cpp:2027` gains a `VW_HAVE_NEON` branch that calls `neon::run_heads<T>` for 2- and 4-byte keys in modes `kSingle` / `kPacked`. 8-byte keys and `kSingleSel` keep the scalar loop.
3. **Dense projections.** In `Projection.cpp:159`, minus and plus get a NEON loop before the scalar tail (as the ymm loop sits after the zmm loop). Multiply gets an SVE loop under `VW_HAVE_SVE` and stays scalar on NEON-only targets.
4. **Join kernels.** The changes are the scalar reworks from §5. They apply to every ISA, so they are new options or changes to the `#else` paths, measured on burrata first.
5. **SVE set (V1/V2 only, no machine yet).** Gathered selection, dense multiply, and Typer's `HYPER_Q1_SIMD`, whose cost is 64-bit multiplies.
6. **Tests.** `src/test/vectorwise/simd_selection.cpp` and `simd_hash.cpp` are differential tests against the scalar templates. They gate on `VW_HAVE_SIMD_SEL`; an aarch64 build needs the same tests gated on `VW_HAVE_NEON`. They run under `qemu-aarch64 build/test_all`.

## 7. Running it on the target

```bash
study/neon_port/run_target.sh burrata     # gcc + clang, both binaries, pinned to core 0
```

This writes `results/neonbench_burrata_<date>/{gcc,clang}_run_neonbench{,_autovec}.csv` plus `machine.txt`. What to read from it:

1. **`l1` rows.** Do the measured N1 ratios match §4? The model predicts NEON/scalar of 0.58 (sel int32), 0.69 (sel int64), 0.53 (run heads 2-byte), 0.73 (run heads 4-byte) and 0.35 (dense minus). Also check `neon4` vs `neon16`.
2. **`stream` rows against `roof,read_*`.** These show which streaming kernels become bandwidth-bound at 1 thread.
3. **`sweep` rows.** Two questions:
   - Where do `bloom`/`semi` `scalar_bf_pf` and `dirprobe` `scalar_2phase` beat the plain scalar loop on N1?
   - At what table size does that happen, relative to `roof,chase_latency`? This sets the prefetch gates.
4. **`avx512_simde` rows.** These measure what "just compile the AVX-512 code with SIMDe" would cost.

## 8. Open items

- **Instruction counts and models only.** No ARM wall-clock number in this study except `run_crcbench`'s. Every N1 figure in §4 is a model until §7 runs.
- **`VW_SIMD_SEL_CHAR`, `sel_col_col`, `selsel_col_col` and `SimdAggr` have no NEON kernel or bench row yet.**
- **SVE results are from qemu (correctness) and llvm-mca (V1) only.**
- **Multi-thread.** The study is 1 thread. On a 128-core socket, the streaming kernels' compute gains matter only while per-core bandwidth is not the limit.
