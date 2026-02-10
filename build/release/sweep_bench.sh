#!/bin/bash

sudo -v # get perms
REAL_USER=${SUDO_USER:-$USER}

# --- Configuration ---
REPS=${1:-1}
#COMPILERS=( "g++")
COMPILERS=("clang++" "g++")
# Note: We are already IN build/release
DB_PATH="../../data/tpch/"
HOSTNAME=$(hostname -s)
ARCH=$(uname -m)

# --- Shared & Language-Specific Flags ---
COMMON_FLAGS="-O3 -fPIC -fno-omit-frame-pointer -Wall -Wextra -Wno-unknown-pragmas"
BASE_CXX_FLAGS="-std=c++17"
BASE_C_FLAGS="-fdiagnostics-color"

# Ensure results directory exists one level up from build/release
sudo -u "$REAL_USER" mkdir -p "../../results"

for CXX_BIN in "${COMPILERS[@]}"; do
    if [[ "$CXX_BIN" == "clang++" ]]; then
        C_BIN="clang"
        SPECIFIC_CXX=""
        SPECIFIC_C=""
    else
        C_BIN="gcc"
        SPECIFIC_CXX=""
        SPECIFIC_C=""
    fi

    # --- Specific Architecture & Compiler Logic ---
    ARCH_FLAGS=""
    if [[ "$HOSTNAME" == "burrata" ]]; then
        if [[ "$C_BIN" == "clang" ]]; then
            ARCH_FLAGS="-march=armv8-a+simd -mcpu=neoverse-n1"
        else
            ARCH_FLAGS="-march=armv8-a+simd -mtune=native"
        fi
    elif [[ "$HOSTNAME" == *"rpi"* ]]; then
        ARCH_FLAGS="-march=armv8-a+crc+simd -moutline-atomics"
    elif [[ "$ARCH" == "aarch64" ]]; then
        ARCH_FLAGS="-march=armv8.2-a"
    else
        ARCH_FLAGS="-march=native"
    fi

    FINAL_CXX_FLAGS="$COMMON_FLAGS $BASE_CXX_FLAGS $SPECIFIC_CXX $ARCH_FLAGS"
    FINAL_C_FLAGS="$COMMON_FLAGS $BASE_C_FLAGS $SPECIFIC_C $ARCH_FLAGS"

    echo "=========================================================="
    echo " HOST: $HOSTNAME | TOOLCHAIN: $C_BIN / $CXX_BIN"
    echo " CXX FLAGS: $FINAL_CXX_FLAGS"
    echo " C FLAGS:   $FINAL_C_FLAGS"
    echo "=========================================================="

    # 1. Clean Cache - Removing local artifacts without deleting the dir
    echo "[1/3] Clearing local CMake cache..."
    sudo -u "$REAL_USER" rm -rf CMakeCache.txt CMakeFiles

    # 2. Configure and Build 
    # Current directory is build/release, CMakeLiss.txt is two levels up
    sudo -u "$REAL_USER" cmake ../.. \
          -DCMAKE_C_COMPILER="$C_BIN" \
          -DCMAKE_CXX_COMPILER="$CXX_BIN" \
          -DCMAKE_CXX_FLAGS="$FINAL_CXX_FLAGS" \
          -DCMAKE_C_FLAGS="$FINAL_C_FLAGS" \
          -DCMAKE_BUILD_TYPE=Release  > /dev/null 2>&1
          
    echo "[2/3] Building testbench..."
    # Specific target build
    sudo -u "$REAL_USER" make run_tpch -j$nproc > /dev/null 2>&1

    # 3. Execution and Metadata Collection
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    RESULT_DIR="../../results/${HOSTNAME}_${C_BIN}_${TIMESTAMP}"
    sudo -u "$REAL_USER" mkdir -p "$RESULT_DIR"

    # Save metadata
    echo "CXX_FLAGS: $FINAL_CXX_FLAGS" > "$RESULT_DIR/flags.txt"
    echo "C_FLAGS:   $FINAL_C_FLAGS" >> "$RESULT_DIR/flags.txt"
    
    echo "[3/3] Running sweep..."

    for (( VS=1; VS<=4000000; VS*=2 )); do
        echo "  >> Running VectorSize: $VS"
    	sudo ./run_tpch "$REPS" "$DB_PATH" 1 "$VS"> "$RESULT_DIR/${VS}_output.log" 2> "$RESULT_DIR/${VS}_error.log" 
    
    done

    echo ">> Results stored in: $RESULT_DIR"
done


