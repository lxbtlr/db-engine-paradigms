{
  description = "Tectorwise/Typer engine: GCC 16 + Clang 22 toolchains for x86_64 (native) and aarch64 (cross)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  # Dev shells (build host: x86_64-linux):
  #
  #   nix develop            # = .#x86_64: native x86_64 gcc-16 / clang-22
  #   nix develop .#aarch64  # cross to aarch64-linux, binaries run via qemu-aarch64
  #
  # CMakeLists.txt hardcodes the compiler names gcc-16 / g++-16 / clang-22 /
  # clang++-22 (COMPILER=gcc|clang, COMPILER_VERSION=new). Each shell puts
  # scripts with those names on PATH that exec the matching toolchain, so the
  # usual configure line works unchanged in both shells:
  #
  #   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  #         -DCOMPILER=gcc -DTARGET_MACHINE=dubliner
  #
  # The aarch64 shell also sets CMAKE_TOOLCHAIN_FILE (system name/processor,
  # cross ar/ranlib, qemu as the crosscompiling emulator) and needs an arm
  # TARGET_MACHINE preset (e.g. -DTARGET_MACHINE=burrata) or an arm
  # -DTARGET_ARCH, since "native" means the x86 build host.
  #
  # The cross GCC 16 is not in the binary cache: the first `nix develop
  # .#aarch64` builds it from source.

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
      lib = pkgs.lib;
      cross = pkgs.pkgsCross.aarch64-multiplatform;

      # Script named `name` that runs `target` (absolute path).
      alias = name: target: pkgs.writeShellScriptBin name ''exec ${target} "$@"'';

      # FindTBB.cmake wants one root with include/ and lib/ (TBBROOT).
      tbbRoot = p: p.symlinkJoin {
        name = "tbb-root";
        paths = [ p.tbb_2022.out p.tbb_2022.dev ];
      };

      # --- x86_64 (native) -------------------------------------------------
      gccX86 = pkgs.gcc16;
      # clang uses the GCC 16 libstdc++, same as the gcc build
      clangX86 = pkgs.llvmPackages_22.clang.override { gccForLibs = pkgs.gcc16.cc; };

      # --- aarch64 (cross) -------------------------------------------------
      armPrefix = cross.stdenv.targetPlatform.config + "-"; # aarch64-unknown-linux-gnu-
      gccArm = cross.buildPackages.gcc16;
      clangArm = cross.buildPackages.llvmPackages_22.clang.override {
        gccForLibs = cross.buildPackages.gcc16.cc;
      };
      binutilsArm = cross.buildPackages.binutils;

      armToolchain = pkgs.writeText "aarch64-toolchain.cmake" ''
        set(CMAKE_SYSTEM_NAME Linux)
        set(CMAKE_SYSTEM_PROCESSOR aarch64)
        set(CMAKE_AR ${binutilsArm}/bin/${armPrefix}ar CACHE FILEPATH "")
        set(CMAKE_RANLIB ${binutilsArm}/bin/${armPrefix}ranlib CACHE FILEPATH "")
        set(CMAKE_CROSSCOMPILING_EMULATOR ${pkgs.qemu-user}/bin/qemu-aarch64)
      '';

      # --- shared tools ----------------------------------------------------
      python = pkgs.python3.withPackages (ps: [ ps.questionary ps.rich ]); # configure.py
      common = with pkgs; [
        cmake
        ninja
        gnumake
        git # googletest is fetched at configure time
        python
        llvmPackages_22.llvm # llvm-mca, llvm-objdump, llvm-ar, ...
        gdb
        numactl
        util-linux # column, taskset
        coreutils # timeout
        bc
        gawk
      ];
      # CMake 4 dropped compatibility with cmake_minimum_required < 3.5, which
      # the top level, the googletest download and old googletest all use.
      commonEnv = {
        CMAKE_POLICY_VERSION_MINIMUM = "3.5";
        CMAKE_EXPORT_COMPILE_COMMANDS = "ON";
      };

      x86Shell = pkgs.mkShellNoCC (commonEnv // {
        name = "vw-x86_64";
        packages = common ++ [
          (alias "gcc-16" "${gccX86}/bin/gcc")
          (alias "g++-16" "${gccX86}/bin/g++")
          (alias "clang-22" "${clangX86}/bin/clang")
          (alias "clang++-22" "${clangX86}/bin/clang++")
          pkgs.binutils # objdump, nm, ar
          pkgs.perf
        ];
        TBBROOT = "${tbbRoot pkgs}";
        # FindTBB.cmake only searches TBBROOT/lib/intel64/gcc4.4 and ENV LIBRARY_PATH
        LIBRARY_PATH = "${tbbRoot pkgs}/lib";
        shellHook = ''
          echo "vw x86_64: $(g++-16 --version | head -1) | $(clang++-22 --version | head -1)"
          [ -e 3rdparty/simde/simde ] || echo "  3rdparty/simde missing: git submodule update --init"
        '';
      });

      armShell = pkgs.mkShellNoCC (commonEnv // {
        name = "vw-aarch64";
        packages = common ++ [
          (alias "gcc-16" "${gccArm}/bin/${armPrefix}gcc")
          (alias "g++-16" "${gccArm}/bin/${armPrefix}g++")
          (alias "clang-22" "${clangArm}/bin/${armPrefix}clang")
          (alias "clang++-22" "${clangArm}/bin/${armPrefix}clang++")
          binutilsArm # aarch64-unknown-linux-gnu-objdump, -nm, ...
          pkgs.qemu-user # qemu-aarch64
        ];
        TBBROOT = "${tbbRoot cross}";
        LIBRARY_PATH = "${tbbRoot cross}/lib";
        CMAKE_TOOLCHAIN_FILE = "${armToolchain}";
        shellHook = ''
          echo "vw aarch64 (cross): $(g++-16 --version | head -1) | $(clang++-22 --version | head -1)"
          echo "  run binaries with: qemu-aarch64 ./build/test_all"
          [ -e 3rdparty/simde/simde ] || echo "  3rdparty/simde missing: git submodule update --init"
        '';
      });
    in
    {
      devShells.${system} = {
        default = x86Shell;
        x86_64 = x86Shell;
        aarch64 = armShell;
      };

      # `nix build .#toolchains` realizes both shells' compilers (e.g. to
      # prebuild the cross GCC 16 in the background).
      packages.${system}.toolchains = pkgs.linkFarm "vw-toolchains" [
        { name = "x86_64-gcc"; path = gccX86; }
        { name = "x86_64-clang"; path = clangX86; }
        { name = "aarch64-gcc"; path = gccArm; }
        { name = "aarch64-clang"; path = clangArm; }
        { name = "aarch64-tbb"; path = tbbRoot cross; }
      ];

      formatter.${system} = pkgs.nixfmt;
    };
}
