# Test 1b — independent (wide) Dd kernel for Q1

Extends Test 1 (`docs/dd_test1.md`) with a second, *independent* kernel shape that
widens the live register set to isolate **register pressure** from instruction
count. The chained kernel (`DdKernel<W>`) folds every `v_k` into a running sum,
so all intermediates have short live ranges (narrow live set). The independent
kernel (`DdKernelIndep<W>`) materializes all `W` `v_k` from the columns before
any is consumed (wide live set), so at large `W` the compiler must spill.

## Branch / scope
- Branch `exp/dd-test1b` (from `exp/dd-test1`). `exp/dd-test1` and `numa-alloc`
  are untouched.
- `DdKernel<W>` (chained) is **byte-for-byte unchanged**.
- No changes to base `q1.cpp`, other queries, `profile.hpp`, CSV columns, or
  Slurm scripts.

## Kernel shape
`DdKernelIndep<W>` (in `include/benchmarks/tpch/DdKernel.hpp`) computes each
`v_k` directly from the columns — `ep * C_k + (ep ^ (disc+k))` with a distinct
`C_k`, **not** from `v_{k-1}` — using template unrolling with named scalar
locals (no `std::array`, no runtime W-loop). An `asm volatile("" : "+r"(v))`
scheduling barrier inside each intermediate prevents GCC from reassociating the
final reduction into a running sum (which would collapse the live set back to
chained width). Arith per intermediate is ~4 ops, matched to the chained kernel.

```cpp
template <int W, int K>
struct DdIndepK;
template <int W>
struct DdIndepK<W, 0> { static inline int64_t gen(int64_t ep, int64_t disc) { ... } };
template <int W, int K>
struct DdIndepK<W, K> {
  static inline int64_t gen(int64_t ep, int64_t disc) {
    int64_t v = DdIndepK<W, K-1>::gen(ep, disc);
    int64_t cur = ep * ddConstant(K) + (ep ^ (disc + K));
    asm volatile("" : "+r"(cur));          // scheduling barrier: keep cur live
    return v + cur;
  }
};
```
`Kernel<W, Shape>` selects `DdKernel<W>` (Chained) or `DdKernelIndep<W>`
(Independent). Shape is a compile-time template parameter, so both instantiations
co-exist (`q1_dd_hyper_impl<W, (dd::Shape)0/1>`).

## CLI
- `-k <chained|independent>` selects the kernel shape, applied to **both**
  Typer (`q1_dd_hyper`) and Tectorwise (`q1_dd_vectorwise`). Default `chained`.
- Dispatched over `DD_WIDTHS = {2,4,...,32}`; unknown W or shape → hard error
  (exit 1).
- Config line now prints `Shape: <chained|independent>`.
- Labels keep the ` tN` thread token and put shape+W ahead:
  `q1dd h ind W16  t1 `, `q1dd v ch W8  t1 `.
- `-e <h|v>` still works; omit `-e` for both engines.

## Verification tooling
`scripts/dd_verify_asm.sh` (gitignored helper, not committed) gains:
- `--shape chained|independent|both` (default `both`).
- `--shape <single> --dump <W>` prints full disassembly of the identified
  per-tuple loop.
- Side-by-side mode (`both`) prints chained vs independent at
  `W ∈ {2,8,12,16,18,20,24,32}` with **delta columns** for arith and spills.
- Same structural per-tuple-loop identification as Task 1 (backward closer
  preceded by `cmp` + span containing kernel `movabs`); chained symbols matched
  as `q1_dd_hyper_impl<W, (dd::Shape)0>`, independent as `(dd::Shape)1`.

## Results (per-tuple loop, sf1, Release, -fno-omit-frame-pointer)

### Side-by-side (arith / spills)
| W | ch inst | ch arith | ind inst | ind arith | arith Δ | ch st/rl | ind st/rl | st Δ | rl Δ | vec |
|---|--------:|---------:|---------:|----------:|--------:|---------:|----------:|-----:|-----:|----:|
| 2 | 90  | 24 | 91  | 24 | +0 | 3/0 | 3/0  | +0 | +0 | 0 |
| 8 | 126 | 47 | 127 | 49 | +2 | 3/0 | 5/0  | +2 | +0 | 0 |
| 12| 149 | 63 | 161 | 65 | +2 | 3/0 | 8/4  | +5 | +4 | 0 |
| 16| 173 | 79 | 197 | 81 | +2 | 3/0 | 12/8 | +9 | +8 | 0 |
| 18| 185 | 87 | 215 | 89 | +2 | 3/0 | 14/10| +11| +10| 0 |
| 20| 197 | 95 | 231 | 97 | +2 | 3/0 | 16/12| +13| +12| 0 |
| 24| 221 | 111| 267 | 113| +2 | 3/0 | 20/16| +17| +16| 0 |
| 32| 269 | 143| 337 | 145| +2 | 3/0 | 28/24| +25| +24| 0 |

Chained numbers match Test 1 exactly (byte-for-byte, criterion 1): inst
90/126/173/185, arith 24/47/79/87, spills flat 3/0 at all W. Independent
spills grow with W; **reloads first go nonzero at W=12** (8 stores / 4 reloads);
at W=32: 28 stores / 24 reloads.

### Arithmetic parity
`arith_ind − arith_ch = +2` for all W ≥ 8 — a **fixed** 2-op overhead, not a
per-intermediate cost, so the per-intermediate difference shrinks with W
(W=8: +0.29/op → W=32: +0.06/op). Within the ≤ ~1 op/intermediate criterion.

### Gates (independent shape)
- arith monotonic in W: `[24,49,81,89,97,105,113,121,145]` → **PASS**.
- loop-closer count constant `[5,5,5,5,5,5,5,5,5]` (no W-trip loop) → **PASS**.
- register class: vec `[(0,0)×9]` all-GP (no xmm/ymm/zmm operands) → clean.
- spill store/reload counts reported as-is (not gated / not tuned).
- **OVERALL: PASS**.

## Correctness (Q1 result unchanged)
Verified with a temporary standalone driver against base `q1_hyper` on sf1: all
five Q1 result columns (sum_qty, sum_base_price, sum_disc_price, sum_charge,
count_order) identical across base, chained W=2/32, independent W=2/32, for both
Typer and Tectorwise engines. (Driver was a local build artifact, not committed.)

## Sample inner-loop disassembly (independent)
`--shape independent --dump 32` → loop `0x15e8fd..0x15eef4`:
- 32 steps of `movabs $Ck,%r; imul %rdx,%r; lea k(%rcx),%r2; xor %rdx,%r2; add %r,%r2`
  for each `v_k`, each `v_k` kept live by the barrier → stack staging
  (`mov %rax,-0x68(%rbp)` … and reloads `-0x60(%rbp),%rax` …) once the live set
  exceeds ~16 GP registers.
- `--shape independent --dump 8` → loop `0x1594dd..0x1596e6`: only the 3 fixed
  Q1 group-key stores (`%sil,-0x41(%rbp)`, `%r8b,-0x42(%rbp)`, `%rdx,-0x40(%rbp)`),
  zero kernel spills — W=8 fits in registers.

## Timing note (sf1, t1)
`q1dd h ind W16` ≈ 133 ms vs `q1dd h ch W16` ≈ 122 ms (hyper: independent
slower, register pressure shows). `q1dd v ind W16` ≈ 338 ms vs `q1dd v ch W16`
≈ 336 ms (Tectorwise: near cost-neutral, as the vectorized engine predicted).
