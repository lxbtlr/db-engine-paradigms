cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

if(UNIX AND NOT APPLE)
  set(LINUX true)
else()
  set(LINUX )
endif()

OPTION(USE_CLANG "Use clang/clang++ instead of gcc/g++" OFF)

if(USE_CLANG)
  set(CMAKE_C_COMPILER clang-18) # clang 18 is leagues better for arm
  set(CMAKE_CXX_COMPILER clang++-18)
else()
  set(CMAKE_C_COMPILER gcc)
  set(CMAKE_CXX_COMPILER g++)
endif()

if(LINUX)
  ADD_DEFINITIONS(-D_LIBCPP_VERSION=1) # tbb clang compatibility
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

OPTION(VECTORWISE_BRANCHING "Use branching vectorwise primitives" OFF)
OPTION(AUTOVECTORIZE "Allow the compiler to autovectorize" OFF)
OPTION(DATADIR "Directory containing testdata" "")
OPTION(AVX512EXPERIMENTS OFF)
OPTION(HARDWARE_BENCHMARKS OFF)
OPTION(INTERPRET_SEPARATE OFF)
OPTION(HASH_SIZE_32 OFF)
OPTION(USE_TCMALLOC "Use tcmalloc from gperftools as the allocator" OFF)
OPTION(NUMA_ALLOC "Replicate base-table columns per NUMA region" OFF)
OPTION(NUMA_SHARD "Shard base-table columns across NUMA regions" OFF)
OPTION(THREAD_PIN_PACKED "Pack threads per socket (vs spread round-robin)" OFF)
OPTION(NUMA_DEBUG "Enable NUMA allocation debug logging and placement verification" OFF)

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
elseif(NOT TARGET_MACHINE STREQUAL "custom")
  message(FATAL_ERROR "Unknown TARGET_MACHINE: ${TARGET_MACHINE}. Valid: custom, dubliner, roquefort, manchego, burrata, kafir")
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
OPTION(HUGE_1GB_MALLOC_HUGE "Use 1GB huge pages for malloc_huge" OFF)
OPTION(HUGE_1GB_MALLOC_NUMA "Use 1GB huge pages for malloc_numa" OFF)
OPTION(WARM_PAGES "Touch all relation pages before measurement to warm page tables" OFF)
OPTION(INTERLEAVE_HT "Interleave hash table pages across NUMA nodes" OFF)
OPTION(HYPER_FLOAT "Let TBB hyper-engine threads float (no pinning)" OFF)
OPTION(SCAN_STATIC_PARTITION "Static per-thread slice partitioning in Scan::next()" OFF)
OPTION(ORIGINAL_GROUPLOOKUP "Use original single-pass GroupLookup (no prefetch split)" OFF)

IF(NUMA_ALLOC AND NUMA_SHARD)
  message(FATAL_ERROR "NUMA_ALLOC and NUMA_SHARD are mutually exclusive. Pick one.")
ENDIF()
IF(SCAN_STATIC_PARTITION AND NUMA_SHARD)
  message(FATAL_ERROR "SCAN_STATIC_PARTITION and NUMA_SHARD are mutually exclusive. Pick one.")
ENDIF()
OPTION(VW_USE_CRC32 "Set the default hashing method to CRC32hash for VectorWise" OFF)

if (VW_USE_CRC32)
  ADD_DEFINITIONS(-DVW_USE_CRC32=1)
  message(STATUS "Building with CRC32Hash ")
else ()
  message(STATUS "Building with MurMurHash")
endif (VW_USE_CRC32)


#set(GROUP_VERSION 0 CACHE STRING "Group Version to use")
#message(STATUS "Building with GROUP_VERSION: ${GROUP_VERSION}")
# add_compile_definitions(GROUP_VERSION=${GROUP_VERSION})

# Compiler flags for the different targets
# TARGET_ARCH selects a machine-specific flag profile.
# "native" (default) auto-detects via -march=native / -mcpu=native.
# Set to a known profile for cross-compilation or reproducible builds.
set(TARGET_ARCH "native" CACHE STRING
    "Target architecture profile: native, skylake-x, icelake, sapphirerapids, neoverse-n1, neoverse-v1, neoverse-v2, ampere-altra, armv8.2-generic, armv9-generic")

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
elseif(TARGET_ARCH STREQUAL "armv8.2-generic")
  set(ARCH_FLAGS "-march=armv8.2-a+simd+crypto")
elseif(TARGET_ARCH STREQUAL "armv9-generic")
  set(ARCH_FLAGS "-march=armv9-a+sve2+crypto")
else()
  message(FATAL_ERROR "Unknown TARGET_ARCH: ${TARGET_ARCH}. See CMakeLists.txt for valid profiles.")
endif()

message(STATUS "TARGET_ARCH=${TARGET_ARCH}  =>  ARCH_FLAGS=${ARCH_FLAGS}")

# Compiler-specific warning flags
if(USE_CLANG)
  # clang doesn't have -Wno-psabi (GCC-specific ABI warning)
  set(WARN_FLAGS "-Wall -Wextra -Wno-unknown-pragmas")
else()
  set(WARN_FLAGS "-Wall -Wextra -Wno-psabi -Wno-unknown-pragmas")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 ${ARCH_FLAGS} -std=c++17 -fPIC ${WARN_FLAGS} -fno-omit-frame-pointer ")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fPIC -Wall -Wextra -fno-omit-frame-pointer ${ARCH_FLAGS} -fdiagnostics-color ")

# Debug sanitizer flags (compiler-specific)
if(LINUX)
  if(USE_CLANG)
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

IF(NUMA_ALLOC)
  ADD_DEFINITIONS(-DNUMA_ALLOC)
ENDIF()

IF(NUMA_SHARD)
  ADD_DEFINITIONS(-DNUMA_SHARD)
ENDIF()

IF(THREAD_PIN_PACKED)
  ADD_DEFINITIONS(-DTHREAD_PIN_PACKED)
ENDIF()

IF(NUMA_DEBUG)
  ADD_DEFINITIONS(-DNUMA_DEBUG)
ENDIF()

IF(NO_HUGE_PAGES)
  ADD_DEFINITIONS(-DNO_HUGE_PAGES)
ENDIF()

IF(HUGE_2MB_MALLOC_HUGE)
  ADD_DEFINITIONS(-DHUGE_2MB_MALLOC_HUGE)
ENDIF()

IF(HUGE_1GB_MALLOC_HUGE)
  ADD_DEFINITIONS(-DHUGE_1GB_MALLOC_HUGE)
ENDIF()

IF(HUGE_1GB_MALLOC_NUMA)
  ADD_DEFINITIONS(-DHUGE_1GB_MALLOC_NUMA)
ENDIF()

IF(WARM_PAGES)
  ADD_DEFINITIONS(-DWARM_PAGES)
ENDIF()

IF(INTERLEAVE_HT)
  ADD_DEFINITIONS(-DINTERLEAVE_HT)
ENDIF()

IF(HYPER_FLOAT)
  ADD_DEFINITIONS(-DHYPER_FLOAT)
ENDIF()

IF(SCAN_STATIC_PARTITION)
  ADD_DEFINITIONS(-DSCAN_STATIC_PARTITION)
ENDIF()

IF(ORIGINAL_GROUPLOOKUP)
  ADD_DEFINITIONS(-DORIGINAL_GROUPLOOKUP)
ENDIF()

IF(NUMA_ALLOC OR NUMA_DEBUG OR NUMA_SHARD OR INTERLEAVE_HT)
  set(NUMA_LIB numa)
ELSE()
  set(NUMA_LIB "")
ENDIF()

IF(AUTOVECTORIZE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftree-vectorize")
ELSE(AUTOVECTORIZE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-tree-vectorize")
  if(USE_CLANG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-vectorize -fno-slp-vectorize")
  endif()
ENDIF(AUTOVECTORIZE)


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

# tcmalloc from gperftools (built out-of-tree to avoid target name collisions)
IF(USE_TCMALLOC)
  include(ExternalProject)
  set(GPERFTOOLS_PREFIX ${CMAKE_BINARY_DIR}/gperftools)
  set(GPERFTOOLS_BUILD_DIR ${GPERFTOOLS_PREFIX}/src/gperftools_external-build)
  ExternalProject_Add(gperftools_external
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/gperftools
    PREFIX ${GPERFTOOLS_PREFIX}
    CMAKE_ARGS
      -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
      -Dgperftools_build_minimal=ON
      -DBUILD_SHARED_LIBS=ON
      -DBUILD_TESTING=OFF
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS ${GPERFTOOLS_BUILD_DIR}/libtcmalloc_minimal.so
  )
  add_library(tcmalloc_minimal SHARED IMPORTED GLOBAL)
  set_target_properties(tcmalloc_minimal PROPERTIES
    IMPORTED_LOCATION ${GPERFTOOLS_BUILD_DIR}/libtcmalloc_minimal.so
    IMPORTED_NO_SONAME TRUE
  )
  add_dependencies(tcmalloc_minimal gperftools_external)
  set(TCMALLOC_LIB tcmalloc_minimal)
  message(STATUS "Using tcmalloc as the allocator")
ELSE()
  set(TCMALLOC_LIB "")
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
target_link_libraries(common ${TCMALLOC_LIB} ${NUMA_LIB})


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
# Set architecture if AVX512 experiments are requested (x86 only)
if(AVX512EXPERIMENTS AND NOT CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
  SET_TARGET_PROPERTIES(vectorwise PROPERTIES
    COMPILE_FLAGS "-march=skylake-avx512"
  )
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

# Enable tests
enable_testing()
set(CTEST_OUTPUT_ON_FAILURE "1")

add_executable(test_all
  src/test/tpch_expected.cpp
  src/test/ssb_expected.cpp
  src/test/tpch.cpp
  src/test/ssb.cpp
  src/test/vectorwise/primitives.cpp
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
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    set(AVX512_COMPILE_FLAGS "-mcpu=native")
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
