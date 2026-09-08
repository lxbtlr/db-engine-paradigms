#!/bin/bash
set -euo pipefail

# dd_verify_asm.sh — disassemble the dd-kernel inner loops and report:
#   - inner-loop instruction count
#   - arithmetic by class (scalar mul/add/xor/shift vs vector)
#   - spill stores/reloads verbatim
#   - GP vs vector operand counts
#   - (tier mode) scratch-access instructions
#
# Usage:
#   dd_verify_asm.sh <build_dir>                       # default: all kernels
#   dd_verify_asm.sh <build_dir> --side-by-side        # compare chained vs independent
#   dd_verify_asm.sh <build_dir> --tier                 # tier mode: compare across tiers
#   dd_verify_asm.sh <build_dir> --tier --side-by-side  # tier side-by-side at fixed W

BUILD_DIR="${1:?Usage: $0 <build_dir> [--side-by-side] [--tier]}"
shift
SIDE_BY_SIDE=0
TIER_MODE=0
for arg in "$@"; do
    case "$arg" in
        --side-by-side) SIDE_BY_SIDE=1 ;;
        --tier)         TIER_MODE=1 ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

BINARY="${BUILD_DIR}/run_tpch"
if [ ! -f "$BINARY" ]; then
    echo "Error: binary not found at $BINARY"
    exit 1
fi

OBJDUMP=${OBJDUMP:-objdump}

# Disassemble the full binary
DISASM=$(mktemp)
trap 'rm -f "$DISASM"' EXIT
$OBJDUMP -d -C --no-show-raw-insn "$BINARY" > "$DISASM"

analyze_function() {
    local pattern="$1"
    local label="$2"

    echo "=== $label ==="

    # Find the function and extract its body
    local func_body
    func_body=$(awk "
        /^[0-9a-f]+ <.*${pattern}.*>:/{found=1; next}
        found && /^[0-9a-f]+ </{found=0}
        found{print}
    " "$DISASM")

    if [ -z "$func_body" ]; then
        echo "  (function not found for pattern: $pattern)"
        return
    fi

    # Find the inner loop: look for the backward jump (the morsel loop)
    # The hot loop typically starts after a comparison with the date and
    # ends with a backward branch. We look for the block between the
    # comparison-and-branch to the loop header.

    # Count total instructions in the function
    local total_insn
    total_insn=$(echo "$func_body" | grep -cE '^\s+[0-9a-f]+:' || true)

    # Arithmetic classification
    local scalar_mul scalar_add scalar_xor scalar_shift vector_ops
    scalar_mul=$(echo "$func_body" | grep -ciE '\b(imul|mul)\b' || true)
    scalar_add=$(echo "$func_body" | grep -ciE '\b(add|sub|lea|adc|sbb)\b' || true)
    scalar_xor=$(echo "$func_body" | grep -ciE '\b(xor)\b' || true)
    scalar_shift=$(echo "$func_body" | grep -ciE '\b(shl|shr|sar|sal|rol|ror)\b' || true)
    vector_ops=$(echo "$func_body" | grep -ciE '\b(v?mov[aups]|v?add[ps]|v?mul[ps]|v?sub[ps]|v?xor[ps]|v?padd|v?pmul|v?psub|v?pxor|vmov|vadd|vmul|vsub|vfma)\b' || true)

    # GP vs vector register operands
    local gp_ops vec_ops_count
    gp_ops=$(echo "$func_body" | grep -coE '%r[abcds][xip]|%r[0-9]+[dwb]?|%e[abcds][xip]|%r[0-9]+' || true)
    vec_ops_count=$(echo "$func_body" | grep -coE '%[xy]mm[0-9]+|%zmm[0-9]+' || true)

    # Spill stores (mov to stack via rbp/rsp offset)
    local spill_stores spill_reloads
    spill_stores=$(echo "$func_body" | grep -cE 'mov.*%r.*,.*\(%r[bs]p\)' || true)
    spill_reloads=$(echo "$func_body" | grep -cE 'mov.*\(%r[bs]p\).*,%r' || true)

    # Scratch accesses (for tiered kernel: stores/loads via a non-stack base register)
    local scratch_stores scratch_loads
    scratch_stores=$(echo "$func_body" | grep -cE 'mov.*%r.*,.*0x[0-9a-f]+\(%r' || true)
    scratch_loads=$(echo "$func_body" | grep -cE 'mov.*0x[0-9a-f]+\(%r.*,%r' || true)

    echo "  Total instructions:    $total_insn"
    echo "  Arith: mul=$scalar_mul add=$scalar_add xor=$scalar_xor shift=$scalar_shift"
    echo "  Vector ops:            $vector_ops"
    echo "  GP operand refs:       $gp_ops"
    echo "  Vec operand refs:      $vec_ops_count"
    echo "  Spill stores:          $spill_stores"
    echo "  Spill reloads:         $spill_reloads"
    if [ "$TIER_MODE" = "1" ]; then
        echo "  Scratch stores:        $scratch_stores"
        echo "  Scratch loads:         $scratch_loads"
    fi
    echo ""

    # Print spill lines verbatim
    if [ "$spill_stores" -gt 0 ] || [ "$spill_reloads" -gt 0 ]; then
        echo "  Spill detail:"
        echo "$func_body" | grep -E 'mov.*(%r[bs]p)' | head -40
        echo ""
    fi
}

if [ "$TIER_MODE" = "1" ]; then
    echo "========================================="
    echo "  TIER MODE: Tiered-reload kernel analysis"
    echo "========================================="
    echo ""

    # The tiered kernel functions are instantiated as q1_dd_tiered_impl<W, Tier>
    for W in 8 16 32; do
        echo "--- W=$W ---"
        for tier_idx in 0 1 2 3; do
            case $tier_idx in
                0) tier_name="L1" ;;
                1) tier_name="L2" ;;
                2) tier_name="LLC" ;;
                3) tier_name="DRAM" ;;
            esac
            # The mangled name pattern for q1_dd_tiered_impl<W, (dd::Tier)T>
            analyze_function "q1_dd_tiered_impl.*<${W}," "W=$W Tier=$tier_name"
        done

        if [ "$SIDE_BY_SIDE" = "1" ]; then
            echo "--- Side-by-side instruction counts for W=$W ---"
            printf "%-8s %8s %8s %8s %8s %8s\n" "Tier" "total" "mul" "add" "xor" "vec"
            for tier_idx in 0 1 2 3; do
                case $tier_idx in
                    0) tier_name="L1" ;;
                    1) tier_name="L2" ;;
                    2) tier_name="LLC" ;;
                    3) tier_name="DRAM" ;;
                esac
                func_body=$(awk "
                    /^[0-9a-f]+ <.*q1_dd_tiered_impl.*<${W},.*>:/{found=1; next}
                    found && /^[0-9a-f]+ </{found=0}
                    found{print}
                " "$DISASM")
                if [ -z "$func_body" ]; then
                    printf "%-8s %8s %8s %8s %8s %8s\n" "$tier_name" "N/A" "N/A" "N/A" "N/A" "N/A"
                    continue
                fi
                total=$(echo "$func_body" | grep -cE '^\s+[0-9a-f]+:' || true)
                mul=$(echo "$func_body" | grep -ciE '\b(imul|mul)\b' || true)
                add=$(echo "$func_body" | grep -ciE '\b(add|sub|lea|adc|sbb)\b' || true)
                xor_c=$(echo "$func_body" | grep -ciE '\b(xor)\b' || true)
                vec=$(echo "$func_body" | grep -ciE '\b(v?mov[aups]|v?add[ps]|v?mul[ps]|vmov|vadd|vmul)\b' || true)
                printf "%-8s %8d %8d %8d %8d %8d\n" "$tier_name" "$total" "$mul" "$add" "$xor_c" "$vec"
            done
            echo ""
        fi
    done
else
    echo "========================================="
    echo "  Standard dd-kernel ASM analysis"
    echo "========================================="
    echo ""

    # Analyze chained and independent kernels across W values
    for W in 2 4 6 8 10 12 14 16 20 24 28 32; do
        echo "--- W=$W ---"
        analyze_function "q1_dd_hyper_impl.*<${W},.*Chained" "W=$W Chained"
        analyze_function "q1_dd_hyper_impl.*<${W},.*Independent" "W=$W Independent"
    done

    if [ "$SIDE_BY_SIDE" = "1" ]; then
        echo "========================================="
        echo "  Side-by-side: Chained vs Independent"
        echo "========================================="
        printf "%-6s %-12s %8s %8s %8s %8s %8s %8s\n" "W" "Shape" "total" "mul" "add" "xor" "spill_s" "spill_r"
        for W in 2 4 6 8 10 12 14 16 20 24 28 32; do
            for shape in Chained Independent; do
                func_body=$(awk "
                    /^[0-9a-f]+ <.*q1_dd_hyper_impl.*<${W},.*${shape}.*>:/{found=1; next}
                    found && /^[0-9a-f]+ </{found=0}
                    found{print}
                " "$DISASM")
                if [ -z "$func_body" ]; then
                    printf "%-6d %-12s %8s %8s %8s %8s %8s %8s\n" "$W" "$shape" "N/A" "N/A" "N/A" "N/A" "N/A" "N/A"
                    continue
                fi
                total=$(echo "$func_body" | grep -cE '^\s+[0-9a-f]+:' || true)
                mul=$(echo "$func_body" | grep -ciE '\b(imul|mul)\b' || true)
                add=$(echo "$func_body" | grep -ciE '\b(add|sub|lea|adc|sbb)\b' || true)
                xor_c=$(echo "$func_body" | grep -ciE '\b(xor)\b' || true)
                spill_s=$(echo "$func_body" | grep -cE 'mov.*%r.*,.*\(%r[bs]p\)' || true)
                spill_r=$(echo "$func_body" | grep -cE 'mov.*\(%r[bs]p\).*,%r' || true)
                printf "%-6d %-12s %8d %8d %8d %8d %8d %8d\n" "$W" "$shape" "$total" "$mul" "$add" "$xor_c" "$spill_s" "$spill_r"
            done
        done
    fi
fi
