# Hashjoin dispatch (`resolvePaths`) and Q18's Select elision

Date: 2026-10-04. Branch `query1` (HEAD `2b1cbdf`) plus the uncommitted files under "Implementation".

Goal: let the VectorWise `Hashjoin` pick its probe path from facts the query plan fixes, without tuning to any one query. Decide it once per operator, not once per probe vector. And remove TPC-H Q18's `Select` over the group-by output, which the plan does not otherwise need.

Status: implemented behind two new CMake options, `VW_JOIN_DISPATCH` and `VW_GROUP_HAVING` (both OFF by default).
- **Correctness:** the TPC-H tests pass for 7 builds at 1 and 4 threads.
- **Cost rules:** each rule is backed by a new microbenchmark, `run_joindispatchbench`.
- **Measurement:** there are new `flag_ablation.sh` configs for the four machines. Only laptop measurements exist so far.

## TL;DR

1. **What the plan fixes, `next()` re-decided per probe vector.** `Hashjoin::next()` re-checks per vector:
   - whether the probe hash is fused (`fusedReady()`);
   - whether the Bloom filter runs (a skip counter, set when more than half a vector passes);
   - whether the join is `VW_NEW_JOIN` (member-pointer compares).

   Under `VW_JOIN_DISPATCH`, `resolvePaths()` runs once at the end of the build, and each probe vector calls one member-function pointer (`probeStep`).
   - The per-vector decision cost itself is negligible: 2.7 → 1.9 ns per 1024-row vector.
   - The value is removing the one data-dependent switch, the Bloom skip counter, and making each join's path a fixed, traceable fact (`VW_DISPATCH_TRACE=1`).
2. **Facts allowed:** static plan structure, plus a build summary known once after the build (row count, key range). Never per-vector data. Four rules come out of the microbenchmarks:
   - **Semi bitmap vs hash** (plan-marked `semi()`, one int32 key per side). Use the bitmap when it is at most `VW_JOIN_SEMI_MAX_BYTES` (default 8 MiB).
     - Zen 4: the bitmap probes 2–4× faster than the hash path up to 8 MiB (0.5–1.0 vs 1.8–5.2 ns/probe), and 2× slower at 16 MiB (3.6–4.1 vs 1.9–2.3).
     - Today's cap is 2^27 bits = 16 MiB, so the largest allowed bitmaps are on the losing side.
   - **Bloom filter.** Use it when the hash table's directory is at least the L1 data cache (a hardware fact, read from sysfs; `VW_JOIN_BLOOM_MIN_KEYS` rows only if the size is unknown) **and** its build keys are read through a selection vector (a plan fact: the build side is a filtered subset). §4.4 explains why L1 and not L2.
     - Kept always on, the filter wins at 5–50% hit rates (1.16–1.32×) and costs 12–22% at 90–100% hits.
     - Probes into an unfiltered build side are foreign keys that all hit, so the plan fact replaces the per-vector skip counter.
   - **Fused probe hash.** One int32 probe key (a plan fact), unchanged from `VW_JOIN_FUSED_PROBE`.
   - **No Bloom filter on a semi join that takes the bitmap.** Before, Q3 J1 and Q9 J2 built a filter they never read.
3. **Eliding Q18's `Select` belongs in the group-by, as SQL HAVING, not in the join.** The bench measured the options on Q18's group output (1.5M groups, 57 pass):

   | variant | ns per group, cached | ns per group, streamed |
   |---|---|---|
   | today: gather key + sum, `Select`, build | 0.92 | 1.43 |
   | join-side filter on the already-gathered columns | ≈ today | ≈ today |
   | filter on the sum, then gather the key for passing rows only | 0.66 | 0.96 |
   | fused filter over the group entries in place | 0.43 | 0.50 |

   - A join-side filter saves only the `Select` operator call. The saving needs fewer gathers, and the gathers happen inside `HashGroup`.
   - `HashGroupBuilder::having()` therefore filters before gathering. It reuses the engine's selection primitives (the "filter, then gather" row: it saves 0.26–0.47 ns of the 0.49–0.93 ns per group that fusing would).
   - Q18's plan drops its `Select`, and the semi join builds from dense keys.
4. **The machine-dependent choice stays a compile-time choice.** That is the lookup kernel: `joinAllParallel` vs `VW_JOIN_TWOPHASE` / `VW_NEW_JOIN` / `VW_JOIN_SIMD`. The flag ablation shows it depends on the microarchitecture (two-phase wins on x86 and loses on burrata), not on the plan.
   - The dispatch keeps whatever kernel the build compiled, and only decides what plan facts decide.
   - Plan-fact candidates for later: run dedup for probes in clustered-key order, and a bucketized table for small builds (`JOIN_OPERATOR_STUDY.md` §8).

## 1. What was written before

| where | what it says | used here |
|---|---|---|
| `VW_OPTIONS.md` §3, §9 | open task: "join dispatch (`Hashjoin::resolvePaths`); semi join absorbing Q18's `Select`" | the two goals |
| `VW_OPTIONS.md`, `VW_JOIN_SEMI` follow-up | let the semi variant absorb Q18's `Select`: read key and aggregate from the group entries, skip `gatherGroups`, the `Select` pass and the selection buffer | measured (§4); moved to HashGroup |
| `JOIN_OPERATOR_STUDY.md` §5 P1–P6 | two-phase probe, compact filter, entry layout, fused key check + gather, build side, Q9 payload | P1 and P2 shipped as `VW_JOIN_TWOPHASE` and `VW_JOIN_BLOOM`; P4 is the next dispatch candidate (§6) |
| `JOIN_OPERATOR_STUDY.md` §8.3–8.4 | "the decision should be made once per operator: from the plan (probe side is a sorted scan of the join key) or from the first vectors" (run dedup); bucketized table "selected by build size" | the dispatch rule shape: plan fact or build summary, once per operator |
| `NEW_JOIN_FILTER_PLAN.md` | head flags in directory pointer bits | touches `runtime::Hashmap` (shared with Hyper), out of scope (VW-only) |
| `Operators.cpp:1970` `HashGroup::resolvePaths` | resolve once at the first `next()`, then member-function pointers per vector | the pattern copied |

## 2. What `Hashjoin` decides today, and when

| decision | where | when | inputs |
|---|---|---|---|
| lookup kernel (`joinAllParallel`, `joinSelParallel`, `joinAllNew`, `joinAllSIMD`, ...) | `config.cpp:100-121` `joinAll()`/`joinSel()` | plan build, from compile flags and the `SIMDjoin` env var | build + env |
| semi bitmap | `next()` build phase | once per operator | plan (`semi()`, key types), build key range |
| Bloom filter allocated | `next()` build barrier | once | build rows ≥ `VW_JOIN_BLOOM_MIN_KEYS` |
| Bloom filter run this vector | `next()` lookup loop | **per probe vector** | skip counter, set by the previous vector's pass rate (**data**) |
| fused probe hash | `next()` lookup loop | **per probe vector** (`fusedReady()`, constant) | plan |
| new-join hashing mode | `next()` lookup loop | **per probe vector** (pointer compares, constant) | build |

## 3. Design

**Rule.** Every input to a decision is either:
- **a plan fact**, which `QueryBuilder` records while building the operator; or
- **a build summary**, known once the build is done: global build row count, and min/max of the semi key.

Nothing samples the probe stream. The decision runs once per operator (per worker), at the end of the build, after the last barrier. That is the earliest point where the build summary exists, and before the first probe vector.

**Facts and where they come from:**

| fact | source | decides |
|---|---|---|
| `semiJoin` | `HashJoinBuilder::semi()` | semi bitmap eligible |
| one int32 key per side, `keys_equal_int32_t_col`, no build payload | `addBuildKey`/`addProbeKey`/`addBuildValue` (`semiCandidate()`) | semi bitmap eligible |
| build keys read through a selection | `addBuildKey(col, sel, ...)` sets `buildKeysSelected` | Bloom filter wanted |
| one int32 probe key | `addProbeKey` (`fusedReady()`) | fused probe hash |
| join function | builder (`config.cpp`) | fused hashing in `joinNewFirstPass` (mode B) |
| build rows | `shared.found` after the build | Bloom filter at `VW_JOIN_BLOOM_MIN_KEYS` |
| semi key range | `semiRange()` after insert | bitmap size vs `VW_JOIN_SEMI_MAX_BYTES` |

**Resolved paths** (`Operators.cpp:1536` `resolvePaths`):

```
semi candidate and bitmap <= VW_JOIN_SEMI_MAX_BYTES -> semiProbe (no hash, directory, key check, gather)
else, per probe vector, probeStep =
  fused probe key and filter    -> stepFusedBloom   hash in registers + filter, survivors only
  fused probe key and new join  -> stepFusedNew     joinNewFirstPass hashes
  filter                        -> stepHashBloom    probe hash expression, then filter
  otherwise                     -> stepHash         probe hash expression
then the compiled lookup kernel, keyEquality, buildGather (unchanged)
```

**Folded flags.** `VW_JOIN_DISPATCH` compiles in the supporting code of `VW_JOIN_SEMI`, `VW_JOIN_BLOOM` and `VW_JOIN_FUSED_PROBE`, as `VW_GROUP_DISPATCH` does for its parts. Each join uses a part only when its facts say so. It excludes `VW_JOIN_TWOPHASE`, `VW_JOIN_SIMD` and `VW_JOIN_PREFETCH`, as `VW_JOIN_BLOOM` does. It combines with `VW_NEW_JOIN`.

**Portability.** Every path has its scalar version, and no rule depends on the ISA. The AVX-512 loops inside `bloomFilter`, `fusedHashFilter` and `semiProbe` are used where compiled. On burrata the same paths run scalar.

### Pros and cons

| | for | against |
|---|---|---|
| once-per-operator resolution | no per-vector state; each join's path is printable (`VW_DISPATCH_TRACE`) and reproducible; the same pattern as `HashGroup::resolvePaths` | the decision cost it removes is small (2.7 → 1.9 ns per vector) |
| plan facts only, no adaptive filter switch | no fitted threshold on running data; the same query takes the same path every run | `buildKeysSelected` is a proxy: a filtered build side can still be fully hit. Q9 J3 (supplier, 10K build, 43K probes) gets a filter it does not need; the cost there is about 43K × ~1 ns |
| build summary allowed (row count, key range) | bitmap size and filter size need it; no plan fact gives them | runs after a barrier, so it is per operator, not per query plan; the same for every worker |
| semi bitmap capped at 8 MiB | bench: bitmap 2–4× faster up to 8 MiB, 2× slower at 16 MiB (Zen 4) | the crossover depends on the L2/L3 sizes; dubliner, manchego and burrata need the sweep (`joindispatchbench.csv` from the `join_dispatch` config) |
| compile-time lookup kernel | the machine-dependent choice (two-phase, new join, SIMD) stays where the evidence put it | a machine still needs its own build |
| alternative: sample the first vectors (as `JOIN_OPERATOR_STUDY.md` §8.3 allows) | would catch hit rates the plan cannot see | data-dependent; ruled out by the goal |
| alternative: per-machine fixed paths per query | fastest per cell | query-tuned; ruled out by the goal |

## 4. Microbenchmarks (`run_joindispatchbench`)

`src/benchmarks/primitives/joindispatchbench.cpp`, target `run_joindispatchbench`. Each family checks every variant's output against its first variant. The numbers below are from a Zen 4 laptop (Ryzen 7 7840U, gcc 16, 1 pinned core; noisy, absolute values move ±20% run to run). The machines run it through the `join_dispatch` config (§7).

### 4.1 `elide`: Q18's Select over the group output

Group entries as `QueryBuilder` lays them out: 32 B, `l_orderkey` at offset 16, `sum(l_quantity)` at 20. Each variant ends with the passing keys and their build hashes.

| shape (pass rate) | today | lazy_key (`having`) | fused_bf | fused_br |
|---|---|---|---|---|
| q18 (57 of 1.5M), cache | 0.92 | 0.66 | 0.44 | 0.43 |
| q18, stream (48 MB) | 1.43 | 0.96 | 0.51 | 0.50 |
| 10%, stream | 1.54 | 1.32 | 0.92 | 0.92 |
| 50%, stream | 2.01 | 1.88 | 1.45 | 2.96 |
| 90%, stream | 2.49 | 2.34 | 2.08 | 1.71 |

(ns per group row)

- **Where the saving is.** It is in gathering and copying fewer columns per group, not in removing the operator. A join-side filter on the gathered columns does the same gathers.
- **Q18 at SF1.** The q18 cache-to-stream saving is 0.26–0.47 ns per group for `having` and 0.49–0.93 for fused. Over 1.5M groups that is 0.4–0.7 ms and 0.7–1.4 ms, about 1–3% of Q18 on this laptop.
- **The branching fused variant is not data-safe.** It is the fastest at low and high pass rates and 2× slower at 50%. So any fused version should be branch-free (`fused_bf`).

### 4.2 `semi`: bitmap vs hash path for a plan-marked semi join

Each probe loop reproduces the engine's passes:
- bitmap: `semiProbe`'s scalar loop;
- hash: probe hash pass, tagged directory + chain, key-equality pass.

`*_setup` variants include each query execution's structure setup: the bitmap's allocate + zero + set bits, or `Hashmap::setSize` + insert.

| shape | build keys | bitmap | bitmap | bitmap_setup | hash | hash_setup |
|---|---|---|---|---|---|---|
| q18 J1 (orders, sorted probes) | 57 | 703 KiB | 0.63 | 0.46 | 1.71 | 2.16 |
| q3 J1 (customer, random probes) | 30K | 18 KiB | 1.89 | 2.12 | 4.63 | 5.21 |
| q9 J2 (part, sorted, 4 per key) | 10.8K | 24 KiB | 0.74 | 0.60 | 1.99 | 2.48 |
| sweep, random probes | 4096 | 127 KiB | 0.68 | 0.56 | 1.93 | 2.35 |
| | | 1 MiB | 0.81 | 0.69 | 1.84 | 2.27 |
| | | 2 MiB | 0.89 | 0.78 | 1.84 | 2.26 |
| | | 4 MiB | 0.91 | 0.84 | 1.84 | 2.36 |
| | | 8 MiB | 1.00 | 1.00 | 1.89 | 2.33 |
| | | 16 MiB | **4.05** | **3.61** | 1.86 | 2.32 |
| | | 24 MiB | **5.04** | **4.81** | 1.81 | 2.26 |
| | | 32 MiB | **5.49** | **5.39** | 1.81 | 2.27 |

(ns per probe)

- **On Zen 4 the bitmap wins up to 8 MiB.** That includes Q18's 703 KiB bitmap for 57 keys, where a tiny hash table might look like the obvious choice.
- **Hence the default `VW_JOIN_SEMI_MAX_BYTES` = 8 MiB.** The crossover moves with the cache sizes, so this default is to be checked per machine.
- **Open: dubliner's Q18 regression.** The 2026-10-04 flag ablation (`2026-10-04-safe-7`) measured `join_semi` Q18 at 0.84 on dubliner and 1.01–1.02 on the other three. The bench says the bitmap probe should win. Either dubliner's setup cost differs (page faults on the 703 KiB `new[]()` under Slurm), or that run's dubliner baseline (Q18 base 332 ms against 232 ms in the previous ablation) was disturbed. The `join_dispatch` run on dubliner settles it: compare the `semi` rows there.

### 4.3 `bloom`: filter always on vs hash path, by hit rate

The filter is `VW_JOIN_BLOOM`'s: 16 bits per key, 4 bits in one 64-bit word, the scalar `bloomFilter` loop. Only survivors go through the first pass and the key check.

| shape | build keys | probes | hit rate | filter | hash | bloom + hash | ratio |
|---|---|---|---|---|---|---|---|
| q3 J2 | 146K | 2M | 1% | 512 KiB | 2.67 | 2.89 | 0.92 |
| q9 J4 | 43K | 2M | 5.4% | 128 KiB | 3.73 | 3.22 | 1.16 |
| q9 J5 | 325K | 1.5M | 20% | 1 MiB | 6.94 | 6.12 | 1.13 |
| h50 | 100K | 2M | 50% | 256 KiB | 8.88 | 6.74 | 1.32 |
| h90 | 100K | 2M | 90% | 256 KiB | 8.63 | 9.79 | 0.88 |
| fk100 | 150K | 2M | 100% | 512 KiB | 10.15 | 12.36 | 0.82 |

(ns per probe)

- **High hit rates lose.** An always-on filter costs 12–22% at 90–100% hits, which is what the per-vector skip counter guarded against. The plan fact that predicts misses is a filtered build side, so the filter is gated on `buildKeysSelected`.
- **Every TPC-H join with ≥ 4096 build rows and a filtered build has a 1–20% hit rate,** except Q9 J3 (100%, 43K probes; §3).
- **The q3 J2 row (0.92) disagrees with the engine.** End to end, `VW_JOIN_BLOOM` made Q3 17% faster locally and up to 1.41× on manchego. The bench probes random keys, while Q3's lineitem probes are sorted by `l_orderkey`, so the directory loads there are the expensive part. Treat this row as unexplained until the machines' rows are in.

### 4.4 `bloom_size`: where the filter starts to pay, and why the rule uses L1

The filter replaces a directory load. It can only pay once that load misses cache, so the threshold should be a cache size, not a key count. The old `VW_JOIN_BLOOM_MIN_KEYS` = 4096 was chosen to keep Q18's 57-key builds unfiltered.

`bloom_size` probes builds of 256 to 4M keys at a 1% hit rate. It reports each build's directory size and the machine's caches (`runtime::cacheBytes`, read from sysfs). The `hit1_stream32` shape also streams 32 B per probe row through the caches, as a probe pipeline does with its other columns. Zen 4 laptop (L1d 32 KiB, L2 1 MiB), ns per probe, filter vs hash:

| directory | hit1 (isolated) | hit1_stream32 |
|---|---|---|
| 4–64 KiB | filter 13–33% slower | filter 13–21% slower |
| 256 KiB | filter 12–15% slower | tie (5.29 vs 5.38) |
| 1 MiB (= L2) | tie to 1.11× | filter 1.32× |
| 4 MiB+ | filter 1.2–2.6× | filter 1.1–1.6× |

(The two runs differ in absolute speed; the laptop was under varying load.)

- **The isolated crossover is at directory = L2.** A rule built on that ("filter iff directory ≥ L2") lost Q9 by about 13% end to end (63.5–67.5 → 73.3–77.0 ms, 3 rounds). The reason: it switched off Q9 J4's filter (43K keys, 512 KiB directory), which the real query needs.
- **Cache competition moves the crossover down to about L2/4.** Under the competing stream it sits at about 256 KiB: a probe pipeline's other columns occupy L2, so a directory under L2 does not stay resident.
- **Why the rule uses L1.** L1 is the level that does survive streaming, and it is a named hardware fact rather than a fitted fraction of L2. On TPC-H it makes the same decisions as an L2/4 rule for every join except Q9's supplier join (128 KiB directory, 100% hits, 43K probes), where the filter costs little either way. It filters Q3 orders, Q5's 30K and 46K builds, Q9 J4 and J5; tiny builds and the semi bitmaps are left out.
- **What would still have to be fitted.** A threshold between L1 and L2 (such as L2/4) would match the competing-stream sweep more closely, but its factor would come from this sweep.
- **Open question for the machines.** The `join_dispatch` config's `bloom_size` rows on dubliner, manchego and burrata show whether the competing-stream crossover sits near L2/4 there too.

### 4.5 `dispatch`: the per-vector decision itself

| | ns per probe vector |
|---|---|
| today's chain (`fusedReady`, skip counter, pointer compares) | 2.70 |
| one resolved member-function pointer | 1.91 |

That is under 1 ns per 1024 rows. The dispatch is not a speed lever. Its value is §3's: fixed, plan-derived paths.

## 5. Q18: HAVING in the group-by (`VW_GROUP_HAVING`)

**Builder API** (`QueryBuilder.hpp`, `HashGroupBuilder::having`):

```cpp
B& having(DS input, std::unique_ptr<vectorwise::Expression>&& condition, DS selection);
```

- `input` is one of this group-by's output buffers.
- `condition` is any selection `Expression` over it, for example the same `sel_greater_int64_t_col_int64_t_val` Q18's `Select` used. On x86 that includes the SIMD selection kernels.
- `selection` is the condition's output selection buffer.
- Call it after every `addKey`/`addValue` (the builder throws otherwise). At most one call per group-by; put every conjunct in its `Expression`.

**What changes in `HashGroup`'s output phase** (`Operators.cpp:1887`), per output block:
1. Gather `input` for every group: its `GatherOpVal`, moved out of `gatherGroups`.
2. Evaluate `condition` into `selection`. A block with no passing group is skipped.
3. Gather every other output for the passing groups only: `GatherOpValSel` in `Operations.cpp`, typed loops for 1/2/4/8 bytes, `memcpy` otherwise.
4. Compact `input` in place through the selection (the selection is ascending).
5. Return the passing count. Consumers see the passing groups densely.

**Q18** (`q18.cpp:218-262`): the `Select` is gone, and the semi join's build key is `Buffer(l_orderkey)` with `hash_int32_t_col` / `scatter_int32_t_col` instead of the selection forms. In the dispatch trace, that join is now `keysSelected=0`, so it gets no Bloom filter. It never would have, at 57 keys.

**Why not the fused in-place filter.** It is faster (§4.1: 0.43–0.51 vs 0.66–0.96 ns), but it needs a new strided selection primitive family per type and comparator. `having()` reuses the engine's primitives with no new kernel. If the machines show the gap matters, a `sel_strided_*` family behind the same builder call is the follow-up.

## 6. Not folded (yet), and why

- **Lookup kernel:** two-phase, new join and SIMD join are machine-dependent (`VW_OPTIONS.md` §3: two-phase 0.92–0.94 on burrata Q3/Q5, new join wins only on manchego), so they stay compile-time.
- **Run dedup** (`JOIN_OPERATOR_STUDY.md` J1): 1.2–1.7× on probes in clustered-key order, 10–20% loss on random probes. The plan fact is "the probe column is the scan's clustering key" (lineitem by `l_orderkey`). `Relation` has no clustering metadata today, so it would be a builder annotation on the scan. Next candidate.
- **Bucketized table for small builds** (`JOIN_OPERATOR_STUDY.md` J3): 2.1–2.3× on dimension-sized builds. Selected by build rows, a build-summary fact, but it needs a second build structure (VW-only). Candidate after run dedup.
- **Fused key check + payload gather** (P4): a per-join specialized kernel from plan facts (key count/types, payload widths). It touches `keyEquality`/`buildGather`, not the probe step.

## 7. Implementation

| file | change |
|---|---|
| `CMakeLists.txt` | `VW_JOIN_DISPATCH` (defines `VW_JOIN_SEMI`, `VW_JOIN_BLOOM`, `VW_JOIN_FUSED_PROBE`), `VW_JOIN_SEMI_MAX_BYTES` (default 8388608), `VW_GROUP_HAVING`, target `run_joindispatchbench` |
| `include/vectorwise/Operators.hpp` | `Hashjoin`: `buildKeysSelected`, `probeStep`, `resolvePaths`, `step*`. `HashGroup`: `havingInput`, `havingCondition`, `havingSel`, `gatherSizes` |
| `src/vectorwise/Operators.cpp` | `resolvePaths` and the probe steps (`:1536`); `next()`: Bloom allocation rule, deferred filter for semi candidates, semi bitmap cap, `resolvePaths()` at the end of the build, resolved probe loop; `HashGroup::next` output-phase HAVING (`:1887`); `VW_DISPATCH_TRACE` |
| `include/vectorwise/Operations.hpp`, `src/vectorwise/Operations.cpp` | `GatherOpValSel` |
| `include/vectorwise/QueryBuilder.hpp`, `src/vectorwise/QueryBuilder.cpp` | `HashGroupBuilder::having`; `gatherSizes` recorded per output; `buildKeysSelected` set by `addBuildKey(col, sel, ...)` |
| `src/benchmarks/tpch/queries/q18.cpp` | `having()` instead of `Select` under `VW_GROUP_HAVING` |
| `src/benchmarks/primitives/joindispatchbench.cpp` | the microbenchmark (§4) |
| `scripts/flag_ablation.sh` | configs `join_dispatch`, `jd_all`, `grp_having`; `join_dispatch` also builds and runs `run_joindispatchbench` into `<compiler>_join_dispatch/joindispatchbench.csv` |

**Builds off are unchanged.** With both options OFF, every change is under `#ifdef`.

## 8. Verification

- **TPC-H correctness:** `test_all --gtest_filter='TPCH.*'` (Q1, Q3, Q5, Q6, Q9, Q18) passes at 1 and 4 threads for these builds (gcc unless noted), all with CRC32 + FAST:
  - base;
  - `VW_JOIN_DISPATCH`;
  - `VW_GROUP_HAVING`;
  - dispatch + HAVING + group dispatch + run heads;
  - dispatch + `VW_NEW_JOIN`;
  - clang dispatch + HAVING + group dispatch + run heads;
  - dispatch + HAVING with `VW_JOIN_SEMI_MAX_BYTES=1`, which forces the semi joins onto the hash path with a deferred Bloom filter.
- **aarch64:** the dispatch + HAVING + group dispatch + run heads build (`TARGET_MACHINE=burrata`) passes the same 6 TPC-H tests under `qemu-aarch64` at 4 threads, using only the scalar paths. `run_joindispatchbench` passes its checks under qemu.
- **Trace** (`VW_DISPATCH_TRACE=1 run_tpch -q N`, dispatch + HAVING build):

| query | join | build rows | path |
|---|---|---|---|
| Q3 | customer → orders (semi) | 30K | semi bitmap, no filter |
| Q3 | orders → lineitem | 147K | fused hash + Bloom |
| Q5 | region → nation (semi) | 1 | semi bitmap |
| Q5 | nation (5 rows) | 5 | hash |
| Q5 | 30K / 46K builds | | fused hash + Bloom |
| Q5 | supplier (unfiltered) | 10K | hash, no filter |
| Q9 | part → partsupp (semi) | 10.7K | semi bitmap, no filter |
| Q9 | nation → supplier | 25 | hash |
| Q9 | supplier → J2 output | 10K | fused hash + Bloom (100% hits; the proxy's known miss) |
| Q9 | J3 output → lineitem (2 keys) | 43K | hash + Bloom (not fused: 2 keys) |
| Q9 | J4 output → orders | 319K | fused hash + Bloom |
| Q18 | group-by (HAVING) → orders (semi) | 57 | semi bitmap |
| Q18 | customer (unfiltered) | 150K | hash, no filter |
| Q18 | orders → lineitem | 57 | hash (below `MIN_KEYS`) |

## 9. Running it on the machines

```
bridge/laptop/bridge submit --id 2026-10-0X-jdispatch --machines dubliner,manchego,roquefort,burrata \
    --env COMPILERS=gcc --env QUERIES=1,3,5,6,9,18 \
    --env "CONFIGS=join_base join_all join_dispatch jd_all grp_runheads grp_having"
```

(This needs these changes committed and pushed first; the bridge runs a pinned commit.)

What to read:
- **`join.csv`, `join_dispatch` vs `join_all`:** same parts, plan-resolved vs per-vector. Expect within noise except where the filter rule changes a join (Q9 J3 gains a filter; unfiltered builds lose theirs).
- **`jd_all` vs `join_all`:** adds HAVING.
- **`group.csv`, `grp_having` vs `grp_runheads`:** Q18 only.
- **`gcc_join_dispatch/joindispatchbench.csv` per machine:** the semi crossover (set `VW_JOIN_SEMI_MAX_BYTES` below it), the Bloom-by-hit-rate rows, and dubliner's `semi` q18 rows for the 0.84 regression.

## 10. Local end-to-end (Zen 4 laptop)

`scripts/flag_ablation.sh`, gcc, 1 pinned core, 3 interleaved rounds × 15 reps, median ms (`results/flag_ablation_cuttlefish_20261004_125900/`). All six builds pass the TPC-H tests. The qemu tests ran on other cores during part of the timing, so differences under ~3% are noise.

| query | join_base | join_all | join_dispatch | jd_all |
|---|---|---|---|---|
| Q3 | 20.78 | 13.83 (1.50) | 13.96 (1.49) | 13.60 (1.53) |
| Q5 | 23.34 | 18.28 (1.28) | 18.59 (1.26) | 18.60 (1.26) |
| Q9 | 70.99 | 60.80 (1.17) | 59.58 (1.19) | 60.92 (1.17) |
| Q18 | 91.81 | 45.28 (2.03) | 90.47 (1.02) | 44.94 (2.04) |

| query | grp_runheads | grp_having |
|---|---|---|
| Q18 | 46.83 | 46.54 (1.006) |
| Q3 / Q5 / Q9 | 20.41 / 23.26 / 72.18 | 20.47 / 23.47 / 72.34 (unchanged; no HAVING) |

- **`join_dispatch` and `jd_all` match the per-vector versions within noise.** That is expected: the parts are the same; only when they are chosen changed. `join_dispatch`'s Q18 is 1.02 because the Q18 gain is the group-by options, which `jd_all` and `join_all` add.
- **HAVING moves Q18 by +0.6%, below this laptop's noise floor.** The bench predicted 1–3%. The machines decide whether it is measurable; it is small either way, and the fused in-place variant (§5) is the only route to more.

## 11. Open items

- Machine numbers after the run in §9.
- The `semi()` uniqueness assumption is unchanged (`VW_OPTIONS.md`, `VW_JOIN_SEMI`).
- `buildKeysSelected` is a proxy for "probes may miss". A builder annotation (`.probesMayMiss()` / `.fkTotal()`) would be exact, but every plan would have to state it.
- HAVING supports one condition `Expression` per group-by, over one input column.
