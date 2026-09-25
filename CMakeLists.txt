cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

if(UNIX AND NOT APPLE)
  set(LINUX true)
else()
  set(LINUX )
endif()

# Compiler selection: COMPILER={gcc,clang} x COMPILER_VERSION={new,old}
#   gcc-new  = gcc-16 / g++-16       gcc-old  = gcc-9 / g++-9
#   clang-new = clang-22 / clang++-22  clang-old = clang-11 / clang++-11
set(COMPILER "gcc" CACHE STRING "Compiler family: gcc or clang")
set(COMPILER_VERSION "new" CACHE STRING "Compiler generation: new or old")

if(COMPILER STREQUAL "clang")
  if(COMPILER_VERSION STREQUAL "old")
    set(CMAKE_C_COMPILER clang-11)
    set(CMAKE_CXX_COMPILER clang++-11)
  else()
    set(CMAKE_C_COMPILER clang-22)
    set(CMAKE_CXX_COMPILER clang++-22)
  endif()
else()
  if(COMPILER_VERSION STREQUAL "old")
    set(CMAKE_C_COMPILER gcc-9)
    set(CMAKE_CXX_COMPILER g++-9)
  else()
    set(CMAKE_C_COMPILER gcc-16)
    set(CMAKE_CXX_COMPILER g++-16)
  endif()
endif()

if(LINUX)
  ADD_DEFINITIONS(-D_LIBCPP_VERSION=1) # tbb clang compatibility
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

OPTION(VECTORWISE_BRANCHING "Use branching vectorwise primitives" OFF)
OPTION(AUTOVECTORIZE "Allow the compiler to autovectorize" OFF)
OPTION(DATADIR "Directory containing testdata" "/tank/alexb/swole/tpch/")
OPTION(AVX512EXPERIMENTS OFF)
OPTION(CANARY_IN_BENCH "Compile the canary into run_tpch (per-cell bracketing, -c/-C). Default OFF: the canary lives in the standalone `canary` binary (job-level prologue/epilogue) and run_tpch has zero canary code." OFF)
OPTION(HARDWARE_BENCHMARKS OFF)
OPTION(INTERPRET_SEPARATE OFF)
OPTION(HASH_SIZE_32 OFF)
OPTION(THREAD_PIN_PACKED "Pack threads per socket (vs spread round-robin)" OFF)

# ── Machine topology ─────────────────────────────────────────────────────
# TARGET_MACHINE selects a preset for SOCKETS_COUNT, CORES_PER_SOCKET,
# SMT_PER_CORE, and THREAD_PIN_PACKED.  Use "custom" to set them manually.
set(TARGET_MACHINE "custom" CACHE STRING
    "Machine topology preset: custom, dubliner, roquefort, manchego, burrata, kafir")
OPTION(CPU_LAYOUT_CONTIGUOUS "Contiguous CPU numbering per node (vs interleaved)" OFF)

set(SOCKETS_COUNT 4 CACHE STRING "Number of sockets/chiplets/memory domains")
set(CORES_PER_SOCKET 22 CACHE STRING "Physical cores per socket/chiplet")
set(SMT_PER_CORE 2 CACHE STRING "Hardware threads per physical core")

if(TARGET_MACHINE STREQUAL "dubliner")
  # 4-socket Xeon, 22 cores/socket, 2-way SMT, interleaved CPU numbering
  set(SOCKETS_COUNT 4)
  set(CORES_PER_SOCKET 22)
  set(SMT_PER_CORE 2)
  set(TARGET_ARCH "native")
elseif(TARGET_MACHINE STREQUAL "roquefort")
  # 1-socket AMD EPYC 7443P (Zen3), 24 cores, 2-way SMT, 1 NUMA node
  set(SOCKETS_COUNT 1)
  set(CORES_PER_SOCKET 24)
  set(SMT_PER_CORE 2)
  set(TARGET_ARCH "native")
elseif(TARGET_MACHINE STREQUAL "manchego")
  # 2-socket Intel Xeon Silver 4509Y (Sapphire Rapids), 8 cores/socket,
  # 2-way SMT, interleaved CPU numbering (even=node0, odd=node1)
  set(SOCKETS_COUNT 2)
  set(CORES_PER_SOCKET 8)
  set(SMT_PER_CORE 2)
  set(TARGET_ARCH "sapphirerapids")
elseif(TARGET_MACHINE STREQUAL "burrata")
  # 1-socket ARM Neoverse-N1, 128 cores, no SMT, 1 NUMA node
  set(SOCKETS_COUNT 1)
  set(CORES_PER_SOCKET 128)
  set(SMT_PER_CORE 1)
  set(TARGET_ARCH "neoverse-n1")
elseif(TARGET_MACHINE STREQUAL "kafir")
  # 8-node Skylake-SP, 24 cores/node, 2-way SMT, contiguous CPU numbering
  set(SOCKETS_COUNT 8)
  set(CORES_PER_SOCKET 24)
  set(SMT_PER_CORE 2)
  set(CPU_LAYOUT_CONTIGUOUS ON)
  set(TARGET_ARCH "skylake-x")
elseif(TARGET_MACHINE STREQUAL "rpi5")
  # Raspberry Pi 5: Broadcom BCM2712 (4x Cortex-A76), ARMv8.2-A, no SMT, 1 NUMA node
  # -march=native is broken on RPi (missing /proc/cpuinfo fields), so we specify explicitly
  set(SOCKETS_COUNT 1)
  set(CORES_PER_SOCKET 4)
  set(SMT_PER_CORE 1)
  set(TARGET_ARCH "cortex-a76")
elseif(NOT TARGET_MACHINE STREQUAL "custom")
  message(FATAL_ERROR "Unknown TARGET_MACHINE: ${TARGET_MACHINE}. Valid: custom, dubliner, roquefort, manchego, burrata, kafir, rpi5")
endif()

IF(CPU_LAYOUT_CONTIGUOUS)
  ADD_DEFINITIONS(-DCPU_LAYOUT_CONTIGUOUS)
ENDIF()
ADD_DEFINITIONS(-DCFG_SOCKETS_COUNT=${SOCKETS_COUNT})
ADD_DEFINITIONS(-DCFG_CORES_PER_SOCKET=${CORES_PER_SOCKET})
ADD_DEFINITIONS(-DCFG_SMT_PER_CORE=${SMT_PER_CORE})
message(STATUS "TARGET_MACHINE=${TARGET_MACHINE}  =>  ${SOCKETS_COUNT}S x ${CORES_PER_SOCKET}C x ${SMT_PER_CORE}T  contiguous=${CPU_LAYOUT_CONTIGUOUS}")
OPTION(NO_HUGE_PAGES "Disable huge pages for malloc_huge" OFF)
OPTION(HUGE_2MB_MALLOC_HUGE "Use 2MB hugetlb pages for malloc_huge" OFF)
set(MORSEL_SIZE 10000 CACHE STRING "Morsel size for parallel scan loops")

OPTION(VW_USE_CRC32 "Set the default hashing method to CRC32hash for VectorWise" OFF)

if (VW_USE_CRC32)
  ADD_DEFINITIONS(-DVW_USE_CRC32=1)
  message(STATUS "Building with CRC32Hash")
else ()
  message(STATUS "Building with MurMurHash")
endif (VW_USE_CRC32)



OPTION(HYPER_Q1_DIRECT_AGG "Typer Q1: direct-mapped aggregation bypassing hash table" OFF)
if(HYPER_Q1_DIRECT_AGG)
  ADD_DEFINITIONS(-DHYPER_Q1_DIRECT_AGG=1)
  message(STATUS "Building with HYPER_Q1_DIRECT_AGG")
endif(HYPER_Q1_DIRECT_AGG)

OPTION(HYPER_Q1_SIMD "Typer Q1: AVX-512 fused scan+aggregate loop (requires HYPER_Q1_DIRECT_AGG)" OFF)
if(HYPER_Q1_SIMD)
  if(NOT HYPER_Q1_DIRECT_AGG)
    message(FATAL_ERROR "HYPER_Q1_SIMD requires HYPER_Q1_DIRECT_AGG")
  endif()
  ADD_DEFINITIONS(-DHYPER_Q1_SIMD=1)
  message(STATUS "Building with HYPER_Q1_SIMD")
endif(HYPER_Q1_SIMD)

OPTION(VW_GROUP_AGGR "Perform group-wise GROUP BY aggregation for VectorWise" OFF)
if(VW_GROUP_AGGR)
  ADD_DEFINITIONS(-DVW_GROUP_AGGR=1)
  message(STATUS "Building with VW_GROUP_AGGR")
endif(VW_GROUP_AGGR)

OPTION(VW_GROUP_AGGR_SEL "Perform selection vector batching for VectorWise" OFF)
if(VW_GROUP_AGGR_SEL)
  ADD_DEFINITIONS(-DVW_GROUP_AGGR_SEL=1)
  message(STATUS "Building with VW_GROUP_AGGR_SEL")
endif(VW_GROUP_AGGR_SEL)

OPTION(VW_POS_16 "Use uint16_t for VectorWise pos_t" OFF)
if(VW_POS_16)
  ADD_DEFINITIONS(-DVW_POS_16=1)
  message(STATUS "Building with VW_POS_16")
endif(VW_POS_16)

OPTION(VW_FUSE_HASH "Perform hash inside Lookup() for VectorWise GROUP BY" OFF)
if(VW_FUSE_HASH)
  ADD_DEFINITIONS(-DVW_FUSE_HASH=1)
  message(STATUS "Building with VW_FUSE_HASH")
endif(VW_FUSE_HASH)

# AVX-512 selection kernels (include/vectorwise/SimdSelection.hpp). The
# existing sel_*/selsel_* primitive names are redirected to the SIMD kernels
# for supported types; without AVX-512F in ARCH_FLAGS they stay scalar.
OPTION(VW_SIMD_SEL "AVX-512 selection kernels for int32/int64/Date (sel/selsel, col_val/col_col)" OFF)
if(VW_SIMD_SEL)
  ADD_DEFINITIONS(-DVW_SIMD_SEL=1)
  message(STATUS "Building with VW_SIMD_SEL")
endif(VW_SIMD_SEL)

# Gathered-input selection (selsel_*) under VW_SIMD_SEL:
#   scalar     : keep the scalar _bf templates (default; on GDS-mitigated
#                Cascade Lake neither SIMD variant beats them)
#   scalarload : scalar loads through inSel -> mask -> compress
#   hwgather   : vpgatherdd/vpgatherdq -> compare -> compress
set(VW_SIMD_SEL_GATHER "scalar" CACHE STRING "VW_SIMD_SEL gathered-input kernels: scalar, scalarload, hwgather")
set_property(CACHE VW_SIMD_SEL_GATHER PROPERTY STRINGS scalar scalarload hwgather)
if(VW_SIMD_SEL_GATHER STREQUAL "scalarload")
  ADD_DEFINITIONS(-DVW_SIMD_SEL_GATHER_SCALARLOAD=1)
elseif(VW_SIMD_SEL_GATHER STREQUAL "hwgather")
  ADD_DEFINITIONS(-DVW_SIMD_SEL_GATHER_HWGATHER=1)
elseif(NOT VW_SIMD_SEL_GATHER STREQUAL "scalar")
  message(FATAL_ERROR "VW_SIMD_SEL_GATHER must be scalar, scalarload or hwgather (got '${VW_SIMD_SEL_GATHER}')")
endif()
# Contiguous selection kernels under VW_SIMD_SEL:
#   VW_SIMD_SEL_COMPRESS reg : register vpcompressd + masked store (default;
#                              safe on Zen 4 where the memory form is slow)
#                        mem : memory-form vpcompressd, like the old kernels
#   VW_SIMD_SEL_UNROLL   1|2 : vectors of 16 per loop iteration
set(VW_SIMD_SEL_COMPRESS "reg" CACHE STRING "VW_SIMD_SEL compress form: reg, mem")
set_property(CACHE VW_SIMD_SEL_COMPRESS PROPERTY STRINGS reg mem)
if(VW_SIMD_SEL_COMPRESS STREQUAL "mem")
  ADD_DEFINITIONS(-DVW_SIMD_SEL_COMPRESS_MEM=1)
elseif(NOT VW_SIMD_SEL_COMPRESS STREQUAL "reg")
  message(FATAL_ERROR "VW_SIMD_SEL_COMPRESS must be reg or mem (got '${VW_SIMD_SEL_COMPRESS}')")
endif()
set(VW_SIMD_SEL_UNROLL "1" CACHE STRING "VW_SIMD_SEL contiguous unroll: 1, 2")
set_property(CACHE VW_SIMD_SEL_UNROLL PROPERTY STRINGS 1 2)
if(VW_SIMD_SEL_UNROLL STREQUAL "2")
  ADD_DEFINITIONS(-DVW_SIMD_SEL_UNROLL2=1)
elseif(NOT VW_SIMD_SEL_UNROLL STREQUAL "1")
  message(FATAL_ERROR "VW_SIMD_SEL_UNROLL must be 1 or 2 (got '${VW_SIMD_SEL_UNROLL}')")
endif()
if(VW_SIMD_SEL)
  message(STATUS "VW_SIMD_SEL_GATHER=${VW_SIMD_SEL_GATHER} COMPRESS=${VW_SIMD_SEL_COMPRESS} UNROLL=${VW_SIMD_SEL_UNROLL}")
endif()

# Software prefetch in the hash join probe loops (joinAllParallel /
# joinSelParallel): directory slot 2*DIST probes ahead, first chain entry
# DIST ahead.
OPTION(VW_JOIN_PREFETCH "Software prefetch in VectorWise hash join probes" OFF)
set(VW_JOIN_PREFETCH_DIST 16 CACHE STRING "VW_JOIN_PREFETCH: prefetch distance in probes")
if(VW_JOIN_PREFETCH)
  ADD_DEFINITIONS(-DVW_JOIN_PREFETCH=1 -DVW_JOIN_PREFETCH_DIST=${VW_JOIN_PREFETCH_DIST})
  message(STATUS "Building with VW_JOIN_PREFETCH (distance ${VW_JOIN_PREFETCH_DIST})")
endif(VW_JOIN_PREFETCH)

# AVX-512 MurmurHash64A kernels (include/vectorwise/SimdHash.hpp) for the
# hash/hash_sel/rehash/rehash_sel primitives and HashGroup's packed-key hash.
# Bit-identical to the scalar hash. Only applies to the default MurMurHash
# (not with VW_USE_CRC32 or HASH_SIZE_32); needs AVX512F + AVX512DQ.
OPTION(VW_SIMD_HASH "AVX-512 MurmurHash64A kernels for VectorWise hashing" OFF)
if(VW_SIMD_HASH)
  if(VW_USE_CRC32 OR HASH_SIZE_32)
    message(WARNING "VW_SIMD_HASH only affects MurMurHash; it is a no-op with VW_USE_CRC32 / HASH_SIZE_32")
  endif()
  ADD_DEFINITIONS(-DVW_SIMD_HASH=1)
  message(STATUS "Building with VW_SIMD_HASH")
endif(VW_SIMD_HASH)

OPTION(VW_SIMD_SEL_CHAR "AVX-512BW Char<N> == constant selection kernel" OFF)
if(VW_SIMD_SEL_CHAR)
  ADD_DEFINITIONS(-DVW_SIMD_SEL_CHAR=1)
  message(STATUS "Building with VW_SIMD_SEL_CHAR")
endif(VW_SIMD_SEL_CHAR)

#set(GROUP_VERSION 0 CACHE STRING "Group Version to use")
#message(STATUS "Building with GROUP_VERSION: ${GROUP_VERSION}")
# add_compile_definitions(GROUP_VERSION=${GROUP_VERSION})

# Compiler flags for the different targets
# TARGET_ARCH selects a machine-specific flag profile.
# "native" (default) auto-detects via -march=native / -mcpu=native.
# Set to a known profile for cross-compilation or reproducible builds.
set(TARGET_ARCH "native" CACHE STRING
    "Target architecture profile: native, skylake-x, icelake, sapphirerapids, neoverse-n1, neoverse-v1, neoverse-v2, ampere-altra, cortex-a76, armv8.2-generic, armv9-generic")

if(TARGET_ARCH STREQUAL "native")
  # Auto-detect: -mcpu=native on ARM, -march=native on x86
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    set(ARCH_FLAGS "-mcpu=native")
  else()
    set(ARCH_FLAGS "-march=native")
  endif()
# --- x86 profiles ---
elseif(TARGET_ARCH STREQUAL "skylake-x")
  set(ARCH_FLAGS "-march=skylake-avx512 -mtune=skylake-avx512")
elseif(TARGET_ARCH STREQUAL "icelake")
  set(ARCH_FLAGS "-march=icelake-server -mtune=icelake-server")
elseif(TARGET_ARCH STREQUAL "sapphirerapids")
  set(ARCH_FLAGS "-march=sapphirerapids -mtune=sapphirerapids")
# --- ARM profiles ---
elseif(TARGET_ARCH STREQUAL "neoverse-n1")
  # AWS Graviton 2, Ampere eMAG2
  set(ARCH_FLAGS "-mcpu=neoverse-n1 -march=armv8.2-a+simd+crypto")
elseif(TARGET_ARCH STREQUAL "neoverse-v1")
  # AWS Graviton 3
  set(ARCH_FLAGS "-mcpu=neoverse-v1 -march=armv8.4-a+sve+crypto")
elseif(TARGET_ARCH STREQUAL "neoverse-v2")
  # AWS Graviton 4, Microsoft Cobalt 100
  set(ARCH_FLAGS "-mcpu=neoverse-v2 -march=armv9-a+sve2+crypto")
elseif(TARGET_ARCH STREQUAL "ampere-altra")
  set(ARCH_FLAGS "-mcpu=neoverse-n1 -march=armv8.2-a+simd+crypto")
elseif(TARGET_ARCH STREQUAL "cortex-a76")
  # Raspberry Pi 5 (BCM2712) — -march=native broken, specify explicitly
  set(ARCH_FLAGS "-mcpu=cortex-a76 -march=armv8.2-a+simd+crypto")
elseif(TARGET_ARCH STREQUAL "armv8.2-generic")
  set(ARCH_FLAGS "-march=armv8.2-a+simd+crypto")
elseif(TARGET_ARCH STREQUAL "armv9-generic")
  set(ARCH_FLAGS "-march=armv9-a+sve2+crypto")
else()
  message(FATAL_ERROR "Unknown TARGET_ARCH: ${TARGET_ARCH}. See CMakeLists.txt for valid profiles.")
endif()

message(STATUS "TARGET_ARCH=${TARGET_ARCH}  =>  ARCH_FLAGS=${ARCH_FLAGS}")

# Compiler-specific warning flags
if(COMPILER STREQUAL "clang")
  #if(USE_CLANG)
  # clang doesn't have -Wno-psabi (GCC-specific ABI warning)
  if(COMPILER_VERSION STREQUAL "old")
    set(WARN_FLAGS "-Wall -Wextra -Wno-unknown-pragmas")
  else()
    set(WARN_FLAGS "-Wall -Wextra -Wno-unknown-pragmas")
  endif()
else()
  if(COMPILER_VERSION STREQUAL "old")
    set(WARN_FLAGS "-Wall -Wextra -Wno-psabi -Wno-unknown-pragmas")
  else()
    set(WARN_FLAGS "-Wchanges-meaning -Wall -Wextra -Wno-psabi -Wno-unknown-pragmas")
  endif()
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 ${ARCH_FLAGS} -std=c++17 -fPIC ${WARN_FLAGS} -fno-omit-frame-pointer ")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fPIC -Wall -Wextra -fno-omit-frame-pointer ${ARCH_FLAGS} -fdiagnostics-color ")

# Debug sanitizer flags (compiler-specific)
if(LINUX)
  if(COMPILER STREQUAL "clang")
    #if(USE_CLANG)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -static-libsan")
  else()
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -static-libasan")
  endif()
else()
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -fsanitize=address")
endif(LINUX)


string(REPLACE "-O2" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
string(REPLACE "-O2" "" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}  -O3")

string(REPLACE "-O2" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O3")
#set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -fopt-info -fopt-info-missed -O3")
string(REPLACE "-O2" "" CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
set(CMAKE_CFLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} -O3")

set(CMAKE_MACOSX_RPATH 1)

IF(VECTORWISE_BRANCHING)
    ADD_DEFINITIONS(-DBRANCHING)
ENDIF(VECTORWISE_BRANCHING)
IF(INTERPRET_SEPARATE)
  ADD_DEFINITIONS(-DINTERPRET_SEPARATE)
ENDIF(INTERPRET_SEPARATE)
IF(HASH_SIZE_32)
  ADD_DEFINITIONS(-DHASH_SIZE=32)
ELSE()
  ADD_DEFINITIONS(-DHASH_SIZE=64)
ENDIF()

IF(THREAD_PIN_PACKED)
  ADD_DEFINITIONS(-DTHREAD_PIN_PACKED)
ENDIF()

IF(NO_HUGE_PAGES)
  ADD_DEFINITIONS(-DNO_HUGE_PAGES)
ENDIF()

IF(HUGE_2MB_MALLOC_HUGE)
  ADD_DEFINITIONS(-DHUGE_2MB_MALLOC_HUGE)
ENDIF()

ADD_DEFINITIONS(-DCFG_MORSEL_SIZE=${MORSEL_SIZE})


IF(AUTOVECTORIZE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftree-vectorize")
ELSE(AUTOVECTORIZE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-tree-vectorize")
	if(COMPILER STREQUAL "clang")

	  if(COMPILER_VERSION STREQUAL "old")
	    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-tree-vectorize -fno-tree-slp-vectorize")
	  else()
	  #if(USE_CLANG)
	    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-tree-vectorize -fno-tree-slp-vectorize")
	  endif(CLANG_OLD)
	endif(CLANG)
ENDIF(AUTOVECTORIZE)

# PGO: profile-guided optimization
# Usage: build with -DPGO=generate, run workload, rebuild with -DPGO=use
set(PGO "OFF" CACHE STRING "Profile-guided optimization: OFF, generate, or use")
if(PGO STREQUAL "generate")
  if(COMPILER STREQUAL "clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-generate=${CMAKE_BINARY_DIR}/pgo-profiles")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fprofile-generate=${CMAKE_BINARY_DIR}/pgo-profiles")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-generate -fprofile-dir=${CMAKE_BINARY_DIR}/pgo-profiles")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fprofile-generate")
  endif()
  message(STATUS "PGO: generating profiles to ${CMAKE_BINARY_DIR}/pgo-profiles")
elseif(PGO STREQUAL "use")
  if(COMPILER STREQUAL "clang")
    # Clang needs llvm-profdata merge first: llvm-profdata merge -output=default.profdata pgo-profiles/
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-use=${CMAKE_BINARY_DIR}/pgo-profiles/default.profdata")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-use -fprofile-dir=${CMAKE_BINARY_DIR}/pgo-profiles -fprofile-correction")
  endif()
  message(STATUS "PGO: using profiles from ${CMAKE_BINARY_DIR}/pgo-profiles")
endif()

# Download and unpack googletest at configure time
configure_file(3rdparty/GTest.CMakeLists.txt googletest-download/CMakeLists.txt)
execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/googletest-download" )
execute_process(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/googletest-download" )

# Prevent GoogleTest from overriding our compiler/linker options
# when building with Visual Studio
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This adds
# the following targets: gtest, gtest_main, gmock
# and gmock_main
add_subdirectory("${CMAKE_BINARY_DIR}/googletest-src"
                 "${CMAKE_BINARY_DIR}/googletest-build")

# Suppress -Werror for googletest/googlemock (old gtest triggers warnings with newer GCC)
target_compile_options(gtest PRIVATE -Wno-error)
target_compile_options(gtest_main PRIVATE -Wno-error)
target_compile_options(gmock PRIVATE -Wno-error)
target_compile_options(gmock_main PRIVATE -Wno-error)

# The gtest/gmock targets carry header search path
# dependencies automatically when using CMake 2.8.11 or
# later. Otherwise we have to add them here ourselves.
if(CMAKE_VERSION VERSION_LESS 2.8.11)
    include_directories("${gtest_SOURCE_DIR}/include"
                        "${gmock_SOURCE_DIR}/include")
endif()

# Now simply link your own targets against gtest, gmock,
# etc. as appropriate

project(exec_engine_comparison)

IF(DATADIR)
    message(STATUS "setting datadir to user defined")
    message(STATUS "${DATADIR}")

    ADD_DEFINITIONS(-DDATADIR="${DATADIR}")
ELSE(DATADIR)
    message(STATUS "setting datadir to proj dir ")
    message(STATUS "${PROJECT_SOURCE_DIR}/data")
    ADD_DEFINITIONS(-DDATADIR="${PROJECT_SOURCE_DIR}/data")
ENDIF(DATADIR)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/3rdparty/")
find_package(TBB)
include_directories(${TBB_INCLUDE_DIRS})


IF(LINUX AND NOT CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
  add_subdirectory(3rdparty/jevents)
  set(JEVENTSLIB "jevents")
ELSE()
  set(JEVENTSLIB "")
ENDIF()



add_library(common
  src/common/algebra/Operators.cpp
  src/common/algebra/Types.cpp
  src/common/runtime/Database.cpp
  src/common/runtime/MemoryPool.cpp
  src/common/runtime/Types.cpp
  src/common/runtime/String.cpp
  src/common/runtime/Import.cpp
  src/common/runtime/Hashmap.cpp
  src/common/runtime/Concurrency.cpp
  src/common/runtime/NumaAlloc.cpp
  src/common/runtime/Profile.cpp
  )
target_include_directories(common PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/simde>
    $<INSTALL_INTERFACE:include>
    PRIVATE src)
target_link_libraries(common)


set(HYPER_TRANSLATORS
  src/hyper/codegen/operators/Map.cpp
  src/hyper/codegen/operators/Print.cpp
  src/hyper/codegen/operators/Scan.cpp
  )

add_library(hyper
  src/hyper/codegen/Translator.cpp
  src/hyper/codegen/TranslatorRegistry.cpp
  )
target_include_directories(hyper PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE src)
target_link_libraries(hyper common)


file(GLOB PRIMITIVES src/vectorwise/primitives/*.cpp)
add_library(vectorwise
  src/vectorwise/Operations.cpp
  src/vectorwise/Operators.cpp
  ${PRIMITIVES}
  src/vectorwise/QueryBuilder.cpp
  )
target_include_directories(vectorwise PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE src)
# Set architecture if AVX512 experiments are requested
# On x86, force skylake-avx512 for native AVX-512; on ARM, use ARCH_FLAGS
# (SIMDe emulates AVX-512 intrinsics portably).
if(AVX512EXPERIMENTS)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    SET_TARGET_PROPERTIES(vectorwise PROPERTIES
      COMPILE_FLAGS "${ARCH_FLAGS}"
    )
  else()
    SET_TARGET_PROPERTIES(vectorwise PROPERTIES
      COMPILE_FLAGS "-march=skylake-avx512"
    )
  endif()
endif()
target_link_libraries(vectorwise common)

add_executable(bench
  ${HYPER_TRANSLATORS}
  src/bench.cpp
  )
target_include_directories(bench PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE src)
target_link_libraries(bench hyper vectorwise common)


add_library(benchmark_config
  src/benchmarks/config.cpp)
target_include_directories(benchmark_config PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE src)
target_link_libraries(benchmark_config vectorwise common)

file(GLOB TPCHQUERIES src/benchmarks/tpch/queries/*.cpp)
add_library(tpch
  ${TPCHQUERIES}
  )
target_include_directories(tpch PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE src)
target_link_libraries(tpch benchmark_config pthread hyper ${TBB_LIBRARIES} vectorwise common)

add_executable(run_tpch
  ${HYPER_TRANSLATORS}
  src/benchmarks/tpch/run.cpp
  )
target_include_directories(run_tpch PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE src)
target_link_libraries(run_tpch tpch hyper vectorwise common ${JEVENTSLIB})

# Embed the git commit so the canary's version is reproducible/attributable.
execute_process(
  COMMAND git rev-parse --short HEAD
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_COMMIT_SHORT
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET)
if(NOT GIT_COMMIT_SHORT)
  set(GIT_COMMIT_SHORT "unknown")
endif()
target_compile_definitions(run_tpch PRIVATE GIT_COMMIT_SHORT="${GIT_COMMIT_SHORT}")

# Opt-in: compile the canary into run_tpch (per-cell bracketing via -c/-C).
if(CANARY_IN_BENCH)
  target_compile_definitions(run_tpch PRIVATE CANARY_IN_BENCH=1)
endif()

# ── Standalone canary tool ────────────────────────────────────────────────
add_executable(canary src/tools/canary.cpp)
target_include_directories(canary PRIVATE include)
target_compile_definitions(canary PRIVATE GIT_COMMIT_SHORT="${GIT_COMMIT_SHORT}")

file(GLOB SBBQUERIES src/benchmarks/sbb/queries/*.cpp)
add_library(ssb
  ${SBBQUERIES}
  )
target_include_directories(ssb PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE src)
target_link_libraries(ssb benchmark_config pthread hyper ${TBB_LIBRARIES} vectorwise common)

add_executable(run_ssb
  ${HYPER_TRANSLATORS}
  src/benchmarks/sbb/run.cpp
  )
target_include_directories(run_ssb PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE src)
target_link_libraries(run_ssb ssb hyper vectorwise common ${JEVENTSLIB})

add_executable(run_prim
  src/benchmarks/primitives/run.cpp
  )
target_include_directories(run_prim PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE src)
target_link_libraries(run_prim vectorwise common ${TBB_LIBRARIES}  ${JEVENTSLIB})

add_executable(run_selbench
  src/benchmarks/primitives/selbench.cpp
  )
target_include_directories(run_selbench PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE src)
target_link_libraries(run_selbench vectorwise common ${TBB_LIBRARIES} ${JEVENTSLIB})

add_executable(run_hashbench
  src/benchmarks/primitives/hashbench.cpp
  )
target_include_directories(run_hashbench PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE src)
target_link_libraries(run_hashbench vectorwise common ${TBB_LIBRARIES} ${JEVENTSLIB})

# Enable tests
enable_testing()
set(CTEST_OUTPUT_ON_FAILURE "1")

add_executable(test_all
  src/test/tpch_expected.cpp
  src/test/ssb_expected.cpp
  src/test/tpch.cpp
  src/test/ssb.cpp
  src/test/vectorwise/primitives.cpp
  src/test/vectorwise/simd_selection.cpp
  src/test/vectorwise/simd_hash.cpp
  src/test/vectorwise/Operators.cpp
  src/test/common/Hashmap.cpp
  src/test/common/Database.cpp
  src/test/common/PartitionedDeque.cpp
  src/test/common/Mmap.cpp
  src/test/common/runtime/Stack.cpp
  )
target_link_libraries(test_all common hyper vectorwise tpch ssb gtest gtest_main)

if(HARDWARE_BENCHMARKS)
  add_executable(latency
    src/benchmarks/hardware/latency.cpp)
  target_link_libraries(latency pthread)
  add_executable(randomWrites
    src/benchmarks/hardware/randomWrites.cpp)
  target_include_directories(randomWrites PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE src)
  target_link_libraries(randomWrites common pthread ${JEVENTSLIB})
endif()


if(AVX512EXPERIMENTS)
  # On x86, enable native AVX-512; on ARM, SIMDe emulates AVX-512 portably
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    set(AVX512_COMPILE_FLAGS "${ARCH_FLAGS}")
  else()
    set(AVX512_COMPILE_FLAGS "-march=skylake-avx512")
  endif()

  add_executable(simdJoin src/benchmarks/simdJoin.cpp)
  SET_TARGET_PROPERTIES(simdJoin PROPERTIES
    COMPILE_FLAGS "${AVX512_COMPILE_FLAGS}"
    )
  target_include_directories(simdJoin PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE src)
  target_link_libraries(simdJoin common ${JEVENTSLIB})

  add_executable(run_simd
    src/benchmarks/primitives/simdselection.cpp
    )
  SET_TARGET_PROPERTIES(run_simd PROPERTIES
    COMPILE_FLAGS "${AVX512_COMPILE_FLAGS}"
    )
  target_include_directories(run_simd PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE src)
  target_link_libraries(run_simd vectorwise common ${TBB_LIBRARIES}  ${JEVENTSLIB})

  add_executable(run_join
    src/benchmarks/primitives/joinmicrobench.cpp
    )
  SET_TARGET_PROPERTIES(run_join PROPERTIES
    COMPILE_FLAGS "${AVX512_COMPILE_FLAGS}"
    )
  target_include_directories(run_join PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE src)
  target_link_libraries(run_join vectorwise common ${TBB_LIBRARIES}  ${JEVENTSLIB})
endif(AVX512EXPERIMENTS)

# Grant cap_sys_nice to all built binaries (requires sudo)
add_custom_target(setcap
  COMMAND sudo ${CMAKE_CURRENT_SOURCE_DIR}/scripts/setcap.sh ${CMAKE_BINARY_DIR}
  COMMENT "Setting cap_sys_nice on executables in build subdirectories"
)
