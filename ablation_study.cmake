# ablation_study.cmake
# Include this into your existing CMakeLists.txt with:
#
#   include(path/to/ablation_study.cmake)
#
# Then attach the ablation definitions to your existing target:
#
#   ablation_configure_target(your_target_name)
#
# To build a single configuration:
#
#   cmake -B build -DABLATION_OPERATOR=0 -DABLATION_PERMUTATION=0
#
# OPERATOR values : 0 = BITWISE (&), 1 = SHORT_CIRCUIT (&&)
# PERMUTATION values: 0-119  (all 5! orderings of the 5 predicates)
#
# To build all 240 configurations as separate targets in one CMake invocation,
# call ablation_add_all_targets(your_target_name) instead. Each variant will
# be built as <target>_op<O>_perm<P> with its own binary.
#
# Regenerate selection_kernel.hpp if predicates change:
#   python3 generate_permutations.py --output selection_kernel.hpp

cmake_minimum_required(VERSION 3.15)

# ---------------------------------------------------------------------------
# Single-configuration mode
# Attach ablation compile definitions to an existing target.
# ---------------------------------------------------------------------------
function(ablation_configure_target TARGET_NAME)

    if(NOT DEFINED ABLATION_OPERATOR)
        message(FATAL_ERROR
            "ABLATION_OPERATOR must be set.\n"
            "  0 = bitwise   (&)\n"
            "  1 = short-circuit (&&)\n"
            "Pass -DABLATION_OPERATOR=<0|1> to cmake.")
    endif()

    if(NOT DEFINED ABLATION_PERMUTATION)
        message(FATAL_ERROR
            "ABLATION_PERMUTATION must be set (0-119).\n"
            "Pass -DABLATION_PERMUTATION=<0..119> to cmake.")
    endif()

    if(ABLATION_OPERATOR LESS 0 OR ABLATION_OPERATOR GREATER 1)
        message(FATAL_ERROR "ABLATION_OPERATOR must be 0 or 1, got: ${ABLATION_OPERATOR}")
    endif()

    if(ABLATION_PERMUTATION LESS 0 OR ABLATION_PERMUTATION GREATER 119)
        message(FATAL_ERROR "ABLATION_PERMUTATION must be 0-119, got: ${ABLATION_PERMUTATION}")
    endif()

    target_compile_definitions(${TARGET_NAME} PRIVATE
        OPERATOR_VARIANT=${ABLATION_OPERATOR}
        PERMUTATION_INDEX=${ABLATION_PERMUTATION}
    )

    if(ABLATION_OPERATOR EQUAL 0)
        set(_op_label "BITWISE")
    else()
        set(_op_label "SHORT_CIRCUIT")
    endif()

    message(STATUS
        "[ablation] ${TARGET_NAME}: "
        "operator=${_op_label}(${ABLATION_OPERATOR}) "
        "permutation=${ABLATION_PERMUTATION}")

endfunction()


# ---------------------------------------------------------------------------
# All-configurations mode
# Clones an existing executable target 240 times (2 operators x 120 perms),
# with OPERATOR_VARIANT and PERMUTATION_INDEX injected as the only difference.
# Every other property — sources, includes, link libraries, compile options,
# compile definitions, compile features, build type — is copied from the base
# target unchanged. Your main target itself is not modified.
#
# Usage (place after your existing add_executable / target_* calls):
#   ablation_add_all_targets(your_target)
#
# Each variant is named: <base>_op<O>_perm<P>
# e.g. myapp_op0_perm0 ... myapp_op1_perm119
# ---------------------------------------------------------------------------
function(ablation_add_all_targets BASE_TARGET)

    # Verify the base target exists and is an executable
    if(NOT TARGET ${BASE_TARGET})
        message(FATAL_ERROR
            "[ablation] Target '${BASE_TARGET}' does not exist. "
            "Call ablation_add_all_targets() after add_executable().")
    endif()

    get_target_property(_target_type ${BASE_TARGET} TYPE)
    if(NOT _target_type STREQUAL "EXECUTABLE")
        message(FATAL_ERROR
            "[ablation] Target '${BASE_TARGET}' is not an executable (type: ${_target_type}).")
    endif()

    # Collect all properties to mirror onto each clone
    get_target_property(_sources   ${BASE_TARGET} SOURCES)
    get_target_property(_includes  ${BASE_TARGET} INCLUDE_DIRECTORIES)
    get_target_property(_links     ${BASE_TARGET} LINK_LIBRARIES)
    get_target_property(_opts      ${BASE_TARGET} COMPILE_OPTIONS)
    get_target_property(_defs      ${BASE_TARGET} COMPILE_DEFINITIONS)
    get_target_property(_features  ${BASE_TARGET} COMPILE_FEATURES)
    get_target_property(_std       ${BASE_TARGET} CXX_STANDARD)
    get_target_property(_std_req   ${BASE_TARGET} CXX_STANDARD_REQUIRED)
    get_target_property(_outdir    ${BASE_TARGET} RUNTIME_OUTPUT_DIRECTORY)

    if(NOT _sources)
        message(FATAL_ERROR
            "[ablation] Could not read SOURCES from '${BASE_TARGET}'. "
            "Ensure ablation_add_all_targets() is called after add_executable().")
    endif()

    foreach(OP RANGE 0 1)
        foreach(PERM RANGE 0 119)

            set(_target "${BASE_TARGET}_op${OP}_perm${PERM}")

            # Clone the executable from the same sources
            add_executable(${_target} ${_sources})

            # Inject the only two things that differ
            target_compile_definitions(${_target} PRIVATE
                OPERATOR_VARIANT=${OP}
                PERMUTATION_INDEX=${PERM}
            )

            # Mirror everything else from the base target
            if(_includes)
                target_include_directories(${_target} PRIVATE ${_includes})
            endif()

            if(_links)
                target_link_libraries(${_target} PRIVATE ${_links})
            endif()

            if(_opts)
                target_compile_options(${_target} PRIVATE ${_opts})
            endif()

            # Forward base compile definitions (excluding any prior
            # OPERATOR_VARIANT / PERMUTATION_INDEX set on the base target,
            # in case this is called more than once)
            if(_defs)
                list(FILTER _defs EXCLUDE REGEX "^OPERATOR_VARIANT=")
                list(FILTER _defs EXCLUDE REGEX "^PERMUTATION_INDEX=")
                target_compile_definitions(${_target} PRIVATE ${_defs})
            endif()

            if(_features)
                target_compile_features(${_target} PRIVATE ${_features})
            endif()

            if(_std)
                set_target_properties(${_target} PROPERTIES
                    CXX_STANDARD ${_std})
            endif()

            if(_std_req)
                set_target_properties(${_target} PROPERTIES
                    CXX_STANDARD_REQUIRED ${_std_req})
            endif()

            # Place variant binaries in a subdirectory to avoid cluttering
            # the same output directory as the base target
            if(_outdir)
                set(_variant_outdir "${_outdir}/ablation")
            else()
                set(_variant_outdir "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ablation")
            endif()
            set_target_properties(${_target} PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${_variant_outdir}")

        endforeach()
    endforeach()

    message(STATUS
        "[ablation] Registered 240 targets cloned from '${BASE_TARGET}' "
        "(OPERATOR 0..1 x PERMUTATION 0..119) → ${_variant_outdir}")

endfunction()
