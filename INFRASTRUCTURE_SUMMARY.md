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
