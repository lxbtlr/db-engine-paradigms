# NEON port study: scripts and data

Write-up: [`study/NEON_PORT_STUDY.md`](../NEON_PORT_STUDY.md).

| file | what |
|---|---|
| `../../src/benchmarks/primitives/neonbench.cpp` | `run_neonbench` / `run_neonbench_autovec` (CMake targets) |
| `../../include/vectorwise/SimdNeon.hpp` | the NEON / SVE kernels the bench measures, and the engine-wiring skeleton |
| `run_target.sh` | build + pinned run of both binaries, gcc and clang, on a real machine |
| `insncount.c` | QEMU TCG plugin: guest instructions executed |
| `insns.sh` | dynamic aarch64 instructions per element for every bench row (qemu) |
| `mca_kernels.cpp`, `mca.sh` | llvm-mca cycles per element of the scalar / NEON / SVE hot loops (N1, V1 models) |
| `../../src/benchmarks/hardware/widthbench.cpp`, `width_mca.sh` | `run_widthbench` (aarch64): sustained instructions / uops per cycle against llvm-mca's dispatch width |
| `data/insns_{n1_gcc,n1_clang,n1_gcc_autovec,v1_gcc}.csv` | `insns.sh` output (gcc data is the one quoted in the study) |
| `data/mca_{n1,v1}_{gcc,clang}.csv` | `mca.sh` output |
| `data/x86_zen4_gcc_full.csv` | `run_neonbench` (default sizes) on a Ryzen 7 7840U (Zen 4, native AVX-512), 1 pinned core |
| `data/x86_zen4_gcc_autovec_l1.csv` | `run_neonbench_autovec -m l1` on the same core |

`data/*.csv` match the repo's `*.csv` ignore rule; add them with `git add -f`
if they should be committed.

## Reproducing (no ARM hardware needed)

```bash
# cross builds (build dirs anywhere)
nix develop .#aarch64 -c bash -c '
  cmake -S . -B /tmp/arm   -G Ninja -DCMAKE_BUILD_TYPE=Release -DCOMPILER=gcc   -DTARGET_MACHINE=burrata
  cmake -S . -B /tmp/armc  -G Ninja -DCMAKE_BUILD_TYPE=Release -DCOMPILER=clang -DTARGET_MACHINE=burrata
  cmake -S . -B /tmp/armv1 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCOMPILER=gcc -DTARGET_MACHINE=custom \
        -DSOCKETS_COUNT=1 -DCORES_PER_SOCKET=64 -DSMT_PER_CORE=1 -DTARGET_ARCH=neoverse-v1
  for b in /tmp/arm /tmp/armc /tmp/armv1; do cmake --build $b --target run_neonbench run_neonbench_autovec; done'

# correctness under qemu (exit 1 on any MISMATCH row); SVE at two vector lengths
qemu-aarch64 -cpu neoverse-n1 /tmp/arm/run_neonbench -s 1 -p 16384 -t 2 -r 3 > /dev/null
qemu-aarch64 -cpu max,sve-default-vector-length=64 /tmp/armv1/run_neonbench -s 1 -p 16384 -t 2 -r 3 -f sel,selsel,hash,proj > /dev/null

# the instruction-count plugin (needs glib headers; the flake's qemu-user ships no plugins)
QI=$(dirname $(dirname $(readlink -f $(nix develop .#aarch64 -c which qemu-aarch64))))/include
G=$(nix build --no-link --print-out-paths nixpkgs#glib.dev)
GO=$(nix build --no-link --print-out-paths nixpkgs#glib.out)
gcc -O2 -shared -fPIC -I$QI -I$G/include/glib-2.0 -I$GO/lib/glib-2.0/include \
    study/neon_port/insncount.c -o /tmp/libinsncount.so

nix develop .#aarch64 -c study/neon_port/insns.sh /tmp/arm/run_neonbench /tmp/libinsncount.so neoverse-n1 > insns_n1_gcc.csv
nix develop .#aarch64 -c study/neon_port/mca.sh gcc n1 > mca_n1_gcc.csv
```

Timings under qemu mean nothing; only the check column and the instruction
counts are used from it.

## On the target

```bash
study/neon_port/run_target.sh burrata          # results/neonbench_burrata_<date>/
```

Full default sizes: 64 MiB stream columns, 2M probes, tables up to 128 MiB:
12 s per binary on a Zen 4 laptop core.
