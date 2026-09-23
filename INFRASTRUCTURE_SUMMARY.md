# db-engines: Infrastructure & Skills Summary

Captures the full working system and workflow so a frontier model can design
skills that integrate cleanly. Last updated from live state: HEAD `1f01d6a`.

---

## 1. Project

- **Repo**: `/tank/alexb/swole/db-engines` (git). A vectorized in-memory
  database engine used to run TPC-H benchmarks.
- **Purpose of our workflow**: rebuild the engine under different compile-time
  configs, sweep runtime parameters, and collect/compare perf results — all as
  async Slurm jobs, with results auto-reported and published.
- **HEAD history** (newest → oldest): `1f01d6a` (loader: MADV_POPULATE_WRITE) →
  `4732dad` (INTERLEAVE_HT option) → `2bd3dd1` (MADV_COLLAPSE) → `df2fb78`
  (fix query prefix match) → `bcb0948` (multi-thread-per-single-load).

## 2. Environment / Infrastructure

- **Cluster scheduler**: Slurm. Key node: `dubliner` (our benchmark node).
- **Partitions** (`sinfo`): `cheese` (burrata, colbyjack), `compute*`
  (dubliner, ...), `intel`, `r6515`, `arm64`, `firesim`, `phis`. We always
  target **`--partition=cheese --nodelist=dubliner --exclusive`**.
- **Slurm log root**: `/tank/alexb/slurm-tpch/<node>/` — every job's
  `--output`/`--error` `.out`/`.err` lives here, named `<label>_<TS>.out`.
- **TPC-H data**: `/tank/alexb/swole/tpch/<sf>/` — sf1..sf100 plus sf001/sf01,
  dbgen, cached, ref_data. **Data root is `/tank/alexb/swole/tpch`, NOT `/data`**
  (a stale path breaks smoke tests).
- **Public web dir** (network-exposed): `/tank/www/alexb/swole/`, URL base
  `https://cheesemonger.cs.northwestern.edu/alexb/swole/<file>`.
- **Text sender** (Telegram bot "CheeseClusterBot" → chat 833955854, Alex):
  - `/tank/project/text/text-alex <msg>` — message **only**, forces
    `parse_mode=Markdown` (Mangles underscores in URLs — avoid for links).
  - `/tank/project/text/text-alex2 <file> <msg>` — plain-text message + file
    attach; its message path uses **no parse_mode** (safe for URLs).
  - Both embed the same bot token; bot chat id 833955854.
- **Dependency**: oneTBB 2021.5.0 (`libtbb.so.12`, Ubuntu package
  `libtbb-dev 2021.5.0-7ubuntu2`).

## 3. The benchmark binary & CLI

Source: `src/benchmarks/tpch/run.cpp`. Invoked as `run_tpch`:
`-p <tpch dir> [-q query] [-e engine] [-r reps] [-t threads] [-v vSize] [-s settleSeconds]`

- `-p` **required**; `-t` accepts comma-separated thread counts, e.g. `-t 16,64,88`
  (defaults to `hardware_concurrency`).
- **Query/engine selection**: master set `allQueries =
  {1h,1v,3h,3v,5h,5v,6h,6v,9h,9v,18h,18v}` (query number + `h`=huge/`v`=vector
  engine). If `-q` omitted → **all queries, both engines** run. If `-q` given
  with `-e` → exact (q,e) pairs; if `-q` given alone → both engines for those
  queries. **Prefix bug fixed at `df2fb78`**: `queryNum(aq) == qn` extracts the
  leading number from keys like `18h`→`18`, so `-q 1` no longer pulls q18.
- **Single load, many thread counts**: the data is loaded once and all requested
  thread counts are run off that one load (`bcb0948`).
- **Output**: CSV to stdout, one row per (query, engine, thread). Raw header:
  `name, median, mean, min, max, stddev, CPUs, IPC, GHz, Bandwidth, cycles,
  LLC-misses, LLC-misses2, l1-misses, instr., br. misses, all_rd, br. misses,
  stores, loads, mem_stall, task-clock,` (note trailing comma; **duplicate
  `br. misses` column** appears twice). `name` looks like `q1 h  t16` (spaces,
  `t` prefix on threads).

## 4. Build system & configs

- CMake (`CMakeLists.txt`). `find_package(TBB)`; links `benchmark_config pthread
  hyper ${TBB_LIBRARIES} vectorwise common`.
- **Exposed options** (~lines 28–36): `NUMA_ALLOC`, `NUMA_SHARD`, `THREAD_PIN_PACKED`,
  `NUMA_DEBUG`, `NO_HUGE_PAGES`, `HUGE_2MB_MALLOC_HUGE`, `HUGE_1GB_MALLOC_HUGE`,
  `HUGE_1GB_MALLOC_NUMA`, `INTERLEAVE_HT`. `NUMA_ALLOC` + `NUMA_SHARD` are
  mutually exclusive (CMake `FATAL_ERROR`).
- **The 4 canonical configs** we build/benchmark:
  1. `default_huge`   → `cmake ..`
  2. `default_nohuge` → `cmake .. -DNO_HUGE_PAGES=ON`
  3. `shard_huge`     → `cmake .. -DNUMA_SHARD=ON`
  4. `shard_nohuge`   → `cmake .. -DNUMA_SHARD=ON -DNO_HUGE_PAGES=ON`
  - Optionally `-DINTERLEAVE_HT=ON` is added (interleaves hash-table pages across
    NUMA nodes). User explicitly toggles this per batch — never assume it.
- **Build style — "build-tpch" method**: builds go into **auto-named tmp dirs**
  `build/tmp.XXXXXXXX` (e.g. `build/tmp.5sXF1VEG/run_tpch`), ephemeral artifacts
  (`CMakeCache.txt`/`CMakeFiles/` removed after build), with a `buildcmds.txt`
  provenance file (config opts, git HEAD, timestamp) left behind. A **mapping
  file** records config→binary, e.g. `build_interleave_map.txt` /
  `build_plain_map.txt`:
  `default_huge /tank/alexb/swole/db-engines/build/tmp.5sXF1VEG/run_tpch` ...

## 5. Output pipeline & data model

- Each run's Slurm `.out` **is** the CSV (pure CSV — no tldr text mixed in).
- **Combined CSV** (from `combine-tpch`): one merged file with columns
  `build, query, engine, thread, median, mean, min, max, stddev, <perf counters...>`.
  `name` is split into query/engine/thread; the `t` prefix is stripped so threads
  are numeric. Header whitespace stripped, trailing empty field dropped, repeated
  headers skipped. **Source columns preserved verbatim** (incl. duplicate
  `br. misses` — disambiguation is an open decision).

## 6. The skills (in `/home/alexb/.pi/agent/skills/`)

Composition chain: **build → run/sweep → send-tldr (per build) → combine →
make-public (publish + link)**. Most are thin orchestrators over shell scripts.

1. **build-tpch** — builds the binary (incl. tmp-dir style, named builds, cmake
   options). Produces a fresh `run_tpch`. Runs BEFORE running anything.
2. **slurm-run-tpch** — runs one `run_tpch` invocation as an exclusive Slurm job
   on a pinned node/partition (`--nodelist/--partition/--exclusive`), `.out`/`.err`
   under `/tank/alexb/slurm-tpch/<node>/`. Async by default. Single run only.
3. **sweep** — runs the binary repeatedly varying one runtime option across a set
   of values (e.g. threads, sf), calls slurm-run-tpch per value.
4. **sweep-status** — reports on an async sweep (completed/running/pending/failed)
   from a fresh session with no memory of submission (reads Slurm queue + log dirs).
5. **send-tldr** — summarizes a finished run/sweep's numbers and texts them
   (median-based, <4000 chars, **report-only: no causal language**). Backing
   script `scripts/tldr_collect.sh <outfile> <title> [note]` groups medians by
   query+engine per thread; sends via `/tank/project/text/text-alex2`.
6. **combine-tpch** — merges per-build `.out` files into one combined CSV.
   `scripts/combine_tpch.py MANIFEST OUT_CSV`; manifest lines are
   `buildname <path/to/out>` (`#`/blank ignored; exits 1 naming skipped builds if
   missing/empty). New bundled `scripts/combine_publish.sh MANIFEST OUT_CSV
   [PUBLIC_NAME]` = combine + make-public in one step.
7. **make-public** — copies a file into `/tank/www/alexb/swole/`, chmod 644,
   verifies non-empty, sends the public link as a **plain-text** Telegram message
   (no Markdown — because underscores in filenames get eaten by Markdown italic
   parsing). `scripts/make_public.sh SOURCE [DESTNAME]`.
8. **corpus-setup** — one-time corpus infrastructure: applies migrations
   (`python3 migrate.py`), registers the node (`register-machine` on the compute
   node), and registers the data dirs (`register-dataset --content` on first
   registration; async for the big scales so the rest of the DB can be built
   while sf100 hashes).
9. **corpus-ingest** — sibling collector to combine-publish: chains `ingest-tpch`
   as an `afterany` collector that fills `corpus.sqlite` as a side effect of the
   batch. Reads the ingest manifest + each run's `.meta` sidecar for
   `snapshot_id`. `CORPUS_SCRATCH` is node-local, set inside the sbatch script.
10. **corpus-status** — read-only corpus queries (`vocab builds|datasets` plus a
    run-coverage matrix and missing-cell diff). Independent of, but
    complementary to, sweep-status.

**Corpus side effect**: the normal chain now also fills
`/tank/alexb/vldb-db/corpus.sqlite`. build-tpch registers the build
(build_event_id → 3rd map column); the run-job prologue writes a `<out>.meta`
sidecar (snapshot_id via `register-machine`); sweep writes the ingest manifest
at submit time and chains a sibling `ingest-tpch` collector. Headers: `v1`
(pre-patch, 22 cols) vs `v2` (post-patch, 21 cols); pass `--header-version`
explicitly. Timings are **milliseconds** (`--timing-unit ms`).

## 7. The async end-to-end workflow (current standard)

The defining pattern: **everything is async and self-reporting, no babysitting.**

1. **Rebuild the 4 configs** in tmp dirs (build-tpch style) at current HEAD with
   the user's chosen options (interleave_ht on/off). Write config→binary map.
2. **Write the combine manifest at submit time** — deterministic because
   node/label/output paths are pinned up front.
3. **Submit 4 run jobs** (one per config), each `sbatch --parsable
   --partition=cheese --nodelist=dubliner --exclusive`, output
   `/<node>/<label>_<TS>.out`. Collect their job IDs.
4. **Chain a collector** with `--dependency=afterany:<run1>:<run2>:...` running
   `combine_publish.sh MANIFEST OUT_CSV [PUBLIC_NAME]`. `afterany` (not
   `afterok`) so failures still fire and publish what exists.
5. When all runs land, the collector automatically: **combine** → merged CSV →
   **make-public** → copies to web dir + texts the public link. (Optional
   per-build `send-tldr` collectors can be chained `afterany:<that run>` to text
   each build's summary independently.)
6. **Sibling corpus ingest** — in parallel with the combiner, chain a second
   `afterany` collector running `ingest-tpch MANIFEST --timing-unit ms
   --header-version v2` on the same run IDs. It is a *sibling* (never a step
   inside the combiner): a DB failure must never block combine/publish.

**Key invariants for new skills:**
- Async via Slurm + `afterany` dependency chaining; deterministic paths known at
  submit time so collectors need no live state.
- Report numbers only (no causal claims); tldr < 4000 chars; use **median**.
- Links must go out as **plain text** (no Markdown) to survive underscores.
- Data root `/tank/alexb/swole/tpch`; log root `/tank/alexb/slurm-tpch`.
- Always `--nodelist=dubliner --partition=cheese --exclusive`.
- Corpus write path: `CORPUS_SCRATCH` node-local inside sbatch only (never
  `/tank`); `register-machine` in the run-job prologue is the drift check.
- Confirm config-affecting choices (esp. `INTERLEAVE_HT`) with the user before
  committing to long sf100 runs.

## 8. Gotchas / decisions a future model should know

- `/data` does **not** exist — data is under `/tank/alexb/swole/tpch`.
- The loader critical path has churned (MADV_COLLAPSE → MADV_POPULATE_WRITE);
  rebuilds in fresh tmp dirs are the norm after each change.
- `INTERLEAVE_HT` is toggled per request — never baked in.
- Duplicate `br. misses` column in the combined CSV is preserved; disambiguation
  (suffix second occurrence) is proposed but **undecided**.
- `text-alex` (Markdown) vs `text-alex2` (plain) — pick per content type.
- The repo home is `/tank/alexb/swole/db-engines` (was `/home/...`; all skills/tool
  calls now default to the `/tank` home). Build helper scripts live in
  `/tank/alexb/swole/db-engines/scripts/`.
- **Operating convention (authoritative):** all work is done out of the
  `/tank/alexb/swole/db-engines` copy. The `/home/alexb/swole/db-engines` copy is
  the user's private staging area where they develop and stage implementations,
  and is **removed from my view** — never read, run, or modify it. Code flows
  into `/tank` via the shared `origin` remote (`git@github.com:lxbtlr/db-engine-paradigms.git`)
  or the user's explicit sync, never by inspecting `/home`.
- The NUMA sharding smoke test verified 4 regions; `NUMA_ALLOC` variant failed
  NUMA-validation reproducibly (no tldr; heads-up texts sent instead).
- **baseline full-thread sweep (Sep 19):** 4 configs on 4 machines from `baseline`
  branch @ `adae38e` — `baseline` (no flags), `gcc_crc` (-DVW_USE_CRC32=ON),
  `clang` (-DUSE_CLANG=ON), `clang_crc` (both). Threads 4..max step 4; sf100;
  q1,3,6,9,18; 5 reps. Scripts: `build_baseline_sweep.sh`,
  `submit_baseline_sweep_machine.sh`. Builds run CONCURRENT per-machine (build
  dirs are per-machine named; the orchestrator pre-checks-out `baseline` and the
  build script VERIFIES (does not mutate) the shared NFS worktree, so concurrent
  builds are safe).
- **clang-on-roquefort libstdc++ gotcha:** `clang++-18` defaults to the NEWEST
  installed libstdc++ (GCC 16 on roquefort) which FAILS to compile the
  thread-pool code (`Concurrency.hpp` std::vector<std::thread> emplace_back —
  "member access into incomplete type __normal_iterator"). Fix: pin clang to the
  default g++'s libstdc++ via
  `-DCMAKE_CXX_FLAGS=--gcc-install-dir=/usr/lib/gcc/$(gcc -print-multiarch)/$(g++ -dumpversion)`
  for `clang_*` configs. manchego worked un-pinned (only gcc 11/12 present).
  Detect with: `echo '#include <vector>' | clang++-18 -E -x c++ - | grep stl_vector.h`.
- **Build parallelism cap:** clang builds on large nodes (dubliner 176thr,
  burrata 128thr) capped at `-j32` to avoid memory pressure; `build_baseline_sweep.sh`
  uses `JOBS=$(( nproc < 32 ? nproc : 32 ))`.

- **query1 matrix (Sep 20):** 3 compilers (gcc16/gcc9/clang22) x 2 CRC32 x 2 POS16
  x 3 group {none,og,ga} = 36 configs/machine, run `-q1 -s0 -r10 -t1,2,4,8` sf1 both
  engines. gcc9 SKIPPED on burrata (ARM): SIMDe `__builtin_inf16` missing on gcc-9
  (no `__has_builtin` pre-GCC-10). Result: 1056 runs (dubliner 288 + manchego 288 +
  roquefort 288 + burrata 192). Combined CSVs under `/tank/alexb/swole/sweeps/`.
- **query1 branch CHANGED run.cpp PMU column ORDER** (l1-hits before instr.; stores/
  loads/all_rd repositioned). The new order still matches EXISTING registered
  header_versions exactly: dubliner=v3, roquefort=v4, manchego=v4_manchego,
  burrata=v4_burrata. NO new migration needed.
- **header auto-detect AMBIGUITY:** ingest-tpch's auto-detect compares SORTED column
  sets, so versions with the same column SET but different ORDER (v3 vs v4) are
  indistinguishable -> picks arbitrarily, then ingest fails on positional mismatch.
  roquefort auto-picked v3 (wrong) -> 0 ingested. MUST force explicit
  `--header-version` for roquefort (=v4), dubliner (=v3) on query1 output.
- **ingest-tpch ABORTS on first malformed manifest line** (does NOT skip per-line):
  an empty build_event field (e.g. a config absent from build_map) makes the WHOLE
  ingest fail -> 0 runs. Guard with `[ -n "$BE" ] || continue` when building the
  manifest.
- **combine_tpch.py takes 2 args** (MANIFEST OUT_CSV), writes to OUT_CSV directly;
  manifest second field must be the **.out path** (NOT the binary path), one
  `buildname <path>` per line. Invoking with 1 arg + `> out.csv` just dumps usage
  text into the CSV. The q1 matrix finalize scripts had both bugs (wrote binary
  path + 1-arg redirect) -> regenerated correctly from login host.
- **make_public.sh not reachable from compute nodes** (`.pi/agent` path not on nodes
  / only dubliner had it): only dubliner's CSV auto-published; manchego/roquefort/
  burrata CSVs copied to `/tank/www/alexb/swole/` manually from the login host.
- **ingest-tpch is idempotent on the run natural key**
  (task_id, build_event_id, snapshot_id, engine_id, trial). Re-ingesting the same
  config+snapshot+trial yields `0 runs ingested from N file(s)` — NOT an error.
  The q1_tests phase re-ran `-t 1` with `chrt -f 1`, but threads=1 was already in
  the corpus from the q1 matrix (same build_event 166, snapshot 4, trial 0) -> all
  132 re-runs deduped to 0. Scheduling policy (chrt) is NOT part of the key, so it
  cannot be used to distinguish otherwise-identical runs. Combined CSVs still built
  fine (combine is independent of ingest). Verify a config is truly new by checking
  parse_out (via corpus/parse.py) + the natural key before expecting inserts.
- **Run-policy dimension: `scheduling`** (migration 0020). Runtime scheduling
  policy is NOT a build characteristic, so it must NOT be minted as a new
  build_event_id (build_event = spec + compiled binary; chrt is neither).
  `run.scheduling TEXT DEFAULT 'default'` is now part of the run natural key
  (task, build_event, snapshot, engine, trial, scheduling). Ingest with
  `ingest-tpch MANIFEST --scheduling chrt-f1` to record a re-run of the same
  binary under a different policy as a distinct, labeled run. `ingest_batch`
  keeps UNIQUE(out_sha256) (one .out = one run = one scheduling); to re-read a
  file under a new policy, clear its stale ingest_batch row + FK children
  (environment_fact -> run_environment -> run_timeline/run_canary -> ingest_batch)
  first. SQLite table rebuild needs `PRAGMA legacy_alter_table=ON` before the
  RENAME or the 11 views that SELECT FROM run break ("no such table: run").
- **q1_tests phase (2026-09-20)**: 132 single-thread `chrt -f 1` runs, all
  succeeded, ingested as 264 runs under scheduling='chrt-f1' (dubliner v3,
  manchego v4_manchego, roquefort v4, burrata v4_burrata). This was the first
  use of the scheduling dimension.
