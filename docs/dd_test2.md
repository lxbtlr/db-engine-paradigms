# Test 2 — the vector-size × width crossing for Q1 dd

Branch: `exp/dd-test2` (branched from `exp/dd-test1`; `exp/dd-test1` is not
modified or merged). Adds the missing *third* axis — the vectorized engine's
vector size `V` — crossed against the dd width `W`.

## Question

Test 1 showed the vectorized engine's cost per tuple grows ~linearly with the
data-dependency width `W`. That cost has two separable components:

- **2a — per-vector dispatch / loop overhead**: the vectorized engine breaks the
  scan into vectors of `V` tuples and materializes each intermediate to memory.
  There is a fixed per-vector, per-intermediate instruction cost (primitive
  dispatch, loop control). Larger `V` amortizes it → marginal
  instructions/intermediate should fall as `V` grows.
- **2b — live intermediate footprint / L1 locality**: the vectorized engine
  materializes `W` intermediates, each `V×8` bytes, so the live footprint is
  `W×V×8`. With a 32 KB L1 the predicted knee is at `V* ≈ 4096/W`
  (V*=2048 at W=2 … 128 at W=32), where the working set leaves L1 and
  L1-misses/tuple jump. A second knee at `V ≈ 131072/W` (1 MB L2) should
  appear if the grid reaches it.

**Falsifiable prediction (2b)**: the cost-minimizing vector size `V*` should
track `4096/W`. **Prediction (2a)**: marginal instructions/intermediate should
shrink with `V` (dispatch amortization).

## Changes vs `exp/dd-test1`

1. **Raise the intermediate-buffer cap** (blocking prereq for W=48/64).
   - `Queries.hpp`: `dd_scratch = 64 + 64` (128), `dd_sink = 64 + 65` (129),
     `dd_sink_out = 64 + 66` (130) — 64 slots of headroom past the 64 W
     intermediates; `ddC[32]` → `ddC[64]` so 64 distinct SSA constants exist.
     No buffer-id collisions (`SharedStateManager` is an unbounded
     `unordered_map` keyed by id — no cap).
   - `q1_dd.cpp`: `DD_WIDTHS` now `{2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,
     48,64}`, `DD_WIDTHS_MAX = 64`, and `q1_dd_hyper` switch gains cases 48 and
     64.
   - `run.cpp`: dd labels now carry V —
     `"q1dd v %s W%d V%d "` / `"q1dd h %s W%d V%d "` (both engines); label()
     buffer widened 64→128 to silence the fortify truncation warning.
2. **Sweep driver** `scripts/sweep_vw.sh` (modeled on `sweep_w.sh`, reuses its
   `-P` plumbing/output format): grid `W∈{2,4,8,16,32,64}` ×
   `V∈{64,128,256,512,1024,2048,4096,8192,16384}` (×9), one binary invocation
   per `(W,V)` with `-w W -v V`, `-k` default chained, `-t`, `-r` (default 5),
   `-P`. Both `W` and `V` are recoverable from the label, which ends in the
   ` tN` thread token. `W` and `V` are printed on the `Config:` line.
3. **Analysis** `scripts/dd_vsweep_report.py`: per `(W,V)` cycles/tuple,
   instr/tuple, IPC, L1/LLC-misses/tuple, mem_stall/tuple; per `W`:
   cost-minimizing `V*`, predicted `4096/W`, L1 plateau-departure V, marginal
   instructions/intermediate (2a amortization, least-squares slope of
   instr-vs-W at fixed V). Plus `scripts/dd_vsweep_plot.py` for the headline
   plots.

`DdKernel.hpp`, the Typer dd path, base `q1.cpp`, other queries,
`profile.hpp`, and the CSV columns are untouched; the default vector size
(1024) for non-dd queries is unchanged. No Test 3.

## Correctness

A throwaway driver (linked against the built libs, not committed) ran base
`q1_hyper`, `q1_dd_hyper<W>` (chained+independent) at W∈{2,64}, and
`q1_dd_vectorwise` at W∈{2,64} × V∈{64,16384}. The 5 real Q1 result columns
(`sum_qty`, `sum_base_price`, `sum_disc_price`, `sum_charge`, `count_order`)
were **byte-identical to base `q1_hyper` at every sampled (W,V)** (all four
R/F/A/N/O groups matched).

## Structural asm gate (unchanged)

`scripts/dd_verify_asm.sh build/dd-test1/run_tpch` → **OVERALL: PASS**, and the
Task 1b numbers are unchanged:

```
  W | ch inst ch arith ind inst ind arith | ch st/rl ind st/rl
  2 |      90       24      91       24    | 3/0   3/0
 12 |     149       63     161       65    | 3/0   8/4   <- reloads begin
 32 |     269      143     337      145    | 3/0   28/24
```
chained spills stay 3/0 (flat); independent spills 3/5/8/12/14/16/20/22/24/26/28
with reloads first at W=12; all-GP; arith monotonic; loop-closer count constant.

## Results (sf1, Release, chained, t=1, reps=5)

Full grid is in the commit under `docs/data/dd_vsweep_grid.txt` (CSV, saved
with a .txt suffix because `*.csv` is gitignored). Headline
tables from `dd_vsweep_report.py`:

### cost-minimizing V (V*) vs predicted 4096/W — vectorwise

```
  W     V*  4096/W    L1@V*  L1depV   L1@dep    L1lo
  2    512  2048.0     2.58     256     1.74    0.73
  4   1024  1024.0     3.92     256     2.39    0.79
  8   1024   512.0     5.34     128     1.85    1.04
 16   1024   256.0     8.15     128     4.93    2.18
 32   1024   128.0    13.39    2048    36.78    9.52
 64   1024    64.0    22.89    2048    70.75   28.71
```

### cycles/tuple vs V (vectorwise), rows = W

```
  W | V=64     V=512    V=1024   V=2048   V=16384
  2 |   70.1     61.8     61.9     62.3     62.9
  4 |   83.7     74.9     74.8     75.5     85.8
  8 |  114.3    100.5     99.5    101.2    101.6
 16 |  172.0    151.7    149.0    152.5    153.3
 32 |  295.8    252.4    248.0    254.3    256.9
 64 |  546.3    464.5    454.7    467.6    469.4
```

### marginal instructions/intermediate (slope of instr vs W at fixed V)

```
  V=   64: 22.72
  V= 1024: 20.82
  V=16384: 20.71
```

## Findings

**2b — V* does NOT track 4096/W (falsified).** Measured `V*` is pinned at
512–1024 for every `W` (V*=512 at W=2, V*=1024 for all W≥4), while the
prediction `4096/W` runs 2048→64. The vectorized engine's cost minimum is
independent of `W` and sits at roughly the default vector size. The predicted
L1 knee never appears as a cost signal: L1-misses/tuple do jump (the knee is
real), but cycles/tuple is flat-to-shallowly-rising across it — at W=64 the
L1-miss knee at V=2048 (70.75 misses/tuple, 3.6× the V*=1024 value) costs only
+2.8% cycles. The OoO core absorbs the L1→L2 traffic; locality never becomes
the bottleneck inside this grid.

**2a — confirmed (modest).** Marginal instructions/intermediate falls with V,
22.72 (V=64) → 20.82 (V=1024, the default) → 20.71 (V=16384): ~2
instructions/intermediate of per-vector dispatch overhead is amortized by
growing V. The effect is small because the vectorized engine already bakes most
dispatch into unrolled per-vector loops.

**No L2 threshold observed.** LLC-misses/tuple is flat ~0.60 across the grid;
only the widest kernel at large V moves it (W=64: 0.60 at V≤1024 → 0.94 at
V=16384, near the predicted 131072/64=2048 knee but shallow and not cost-
relevant).

**Hyper control is flat in V (no wiring bug).** `q1dd h W64` = 136.7/136.0/
135.8/136.7/136.4/135.8/136.0/136.5/135.9 ms across V=64…16384; other W flat to
within a couple ms. Typer has no vector size — its per-W control line is
invariant to `-v`, as required.

## Success criteria

- [x] Clean Release build (`build/dd-test1`); W=48/64 kernels link and run.
- [x] Existing `-w`/`-k` unchanged; asm gate re-verified, Task 1b numbers
      unchanged.
- [x] `q1_dd_hyper` / `q1_dd_vectorwise` real Q1 results identical to base
      `q1_hyper` at every sampled (W,V).
- [x] Typer timing invariant to `-v`.
- [x] W=64 grid runs to completion — no buffer-id collisions / allocation
      failure.
- [x] Report script produces the tables (above), plots emitted.
- [x] Falsifiable prediction recorded and reported plainly: **V* does not track
      4096/W** (null); 2a dispatch amortization confirmed.
