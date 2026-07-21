# CUB test-registration machinery. Data (quirk declarations, option gates)
# lives in cub/test/CMakeLists.txt; this module holds only functions.
#
# PARITY INVARIANTS — this module was extracted from cub/test/CMakeLists.txt
# under a byte-identical-behavior contract. Do not change any of the
# following without an accompanying parity run
# (ci/util/cmake_registration_replay/run.sh):
#
# 1. Test/target names: `cub.test.<name><suffix>` with `.lid_N` suffixes.
#    CI shards tests with name regexes over `.lid_N` (CMakePresets.json), so
#    names are load-bearing API.
# 2. COMPILE_DEFINITIONS order: CUB_DEBUG_SYNC (legacy) -> quirk definitions
#    -> CCCL_ENABLE_ASSERTIONS -> %PARAM% variant definitions (VAR_IDX last).
# 3. RDC: cub_configure_cuda_target sets CUDA_SEPARABLE_COMPILATION ON+PIC
#    when truthy but leaves PIC UNSET when falsy; call it exactly once per
#    target.
# 4. _cub_register_ctest's first parameter must remain literally named
#    `test_target`: cccl_add_xfail_compile_target_test reads ${test_target}
#    from its caller's scope (cmake/CCCLUtilities.cmake). Fixing that shared
#    function is a separate follow-up.
# 5. This module must not contain cmake_minimum_required() or cmake_policy():
#    include() shares the includer's policy scope, and flipping policies here
#    would silently change behavior for the rest of cub/test/CMakeLists.txt.
# 6. The nvrtc quirk's configure_file() intentionally resolves relative to
#    the *calling* directory (cub/test). Functions execute in the caller's
#    scope; do not "modernize" paths to CMAKE_CURRENT_FUNCTION_LIST_DIR.
include_guard(GLOBAL)

# Quirk registry storage (rows are appended by _cub_declare_quirk from the
# cub/test directory scope):
set(_cub_quirk_count 0)

## _cub_is_catch2_test
#
# If the test_src contains the substring "catch2_test_", `result_var` will
# be set to TRUE.
function(_cub_is_catch2_test result_var test_src)
  string(FIND "${test_src}" "catch2_test_" idx)
  if (idx EQUAL -1)
    set(${result_var} FALSE PARENT_SCOPE)
  else()
    set(${result_var} TRUE PARENT_SCOPE)
  endif()
endfunction()

## _cub_is_fail_test
#
# If the test_src contains the substring "_fail", `result_var` will
# be set to TRUE.
function(_cub_is_fail_test result_var test_src)
  string(FIND "${test_src}" "_fail" idx)
  if (idx EQUAL -1)
    set(${result_var} FALSE PARENT_SCOPE)
  else()
    set(${result_var} TRUE PARENT_SCOPE)
  endif()
endfunction()

## _cub_launcher_requires_rdc
#
# If given launcher id corresponds to a CDP launcher, set `out_var` to 1.
function(_cub_launcher_requires_rdc out_var launcher_id)
  if ("${launcher_id}" STREQUAL "1")
    set(${out_var} 1 PARENT_SCOPE)
  else()
    set(${out_var} 0 PARENT_SCOPE)
  endif()
endfunction()

#[=======================================================================[.rst:
_cub_test_name_from_source
--------------------------

Derive the dotted test name from a source path relative to ``cub/test/``:
strip the directory and ``catch2_test_``/``test_`` prefixes, then convert
the component prefixes (``thread_``, ``warp_``, ``block_``, ``device_``,
``util_``) into dotted metatarget groups.

E.g. ``warp/catch2_test_warp_reduce.cu`` -> ``warp.reduce``.

The result is the suffix-free test name: it keys the metatarget path and the
GPU footprint table (CubTestFootprints.cmake), and is shared by all
lid/%PARAM% variants of a source.

Destiny: CUB-local naming convention; stays here.
#]=======================================================================]
function(_cub_test_name_from_source out_var test_src)
  get_filename_component(test_name "${test_src}" NAME_WE)
  string(REGEX REPLACE "^catch2_test_" "" test_name "${test_name}")
  string(REGEX REPLACE "^test_" "" test_name "${test_name}")

  # Group sets of tests into metatargets based on their prefixes:
  foreach (
    component
    IN
    ITEMS thread warp block device util
  )
    string(
      REGEX REPLACE
      "^${component}_"
      "${component}."
      test_name
      "${test_name}"
    )
  endforeach()

  set(${out_var} "${test_name}" PARENT_SCOPE)
endfunction()

## _cub_test_excluded
#
# Set `out_var` to TRUE if the named test must be skipped entirely in the
# current configuration. Reads the directory-scope `build_nvrtc_tests`
# toggle (OFF when NVHPC is the host compiler).
function(_cub_test_excluded out_var test_name)
  if ("${test_name}" MATCHES "nvrtc" AND NOT build_nvrtc_tests)
    set(${out_var} TRUE PARENT_SCOPE)
  else()
    set(${out_var} FALSE PARENT_SCOPE)
  endif()
endfunction()

#[=======================================================================[.rst:
_cub_launch_policy
------------------

Determine the launcher for a test variant and whether the current CMake
configuration allows building it. Replaces the former ``_cub_has_lid_variant``
/ ``_cub_launcher_id`` / ``_cub_launch_id_enabled`` helpers plus the loop's
``CUB_FORCE_RDC`` fallback and RDC gate.

If ``label`` contains ``lid``, the parameter explicitly tests variants built
with different launchers. The ``values`` for such a parameter must be
``0:1:2``, with:

- ``0`` indicating host launch and CDP disabled (RDC off),
- ``1`` indicating device launch and CDP enabled (RDC on),
- ``2`` indicating graph capture launch and CDP disabled (RDC off).

Tests that do not contain a variant labeled ``lid`` only enable RDC if the
CMake config forces it (``CUB_FORCE_RDC``).

``launcher_id_var``
  Set to the launcher id (0=host, 1=device, 2=graph).
``enabled_var``
  Set to a false value when the configuration disables this variant
  (``CUB_ENABLE_LAUNCH_*_LAUNCHER`` / ``CUB_ENABLE_RDC_TESTS``).
``label``
  The variant label. Pass ``""`` for sources without %PARAM% variants; such
  tests are never gated on launch options or ``CUB_ENABLE_RDC_TESTS``.

Reads from calling scope: ``CUB_FORCE_RDC``, ``CUB_ENABLE_LAUNCH_*``,
``CUB_ENABLE_RDC_TESTS`` (cache options from cub/cmake/CubCudaConfig.cmake).

Parity invariants: the substring ``FIND(lid_)`` + unanchored regex pair is
preserved verbatim (including its false-positive semantics for hypothetical
labels like ``invalid_2``); unknown lid values stay *silently* disabled.

Destiny: hoists to cmake/ (Thrust/cudax reuse) after MR2 pins semantics.
#]=======================================================================]
function(_cub_launch_policy launcher_id_var enabled_var label)
  string(FIND "${label}" "lid_" lid_idx)
  if (NOT lid_idx EQUAL -1)
    # Explicit launcher variant:
    string(REGEX MATCH "lid_([0-9]+)" match_result "${label}")
    if (match_result)
      set(launcher_id ${CMAKE_MATCH_1})
    else()
      set(launcher_id 0)
    endif()

    if ("${launcher_id}" STREQUAL "0")
      set(launch_enabled ${CUB_ENABLE_LAUNCH_HOST_LAUNCHER})
    elseif ("${launcher_id}" STREQUAL "1")
      set(launch_enabled ${CUB_ENABLE_LAUNCH_DEVICE_LAUNCHER})
    elseif ("${launcher_id}" STREQUAL "2")
      set(launch_enabled ${CUB_ENABLE_LAUNCH_GRAPH_LAUNCHER})
    else()
      set(launch_enabled 0)
    endif()
  else()
    if (${CUB_FORCE_RDC})
      set(launcher_id 1)
    else()
      set(launcher_id 0)
    endif()

    if ("${label}" STREQUAL "")
      # No %PARAM% variants: always built, regardless of launch options.
      set(launch_enabled 1)
    else()
      set(launch_enabled "${CUB_ENABLE_LAUNCH_NO_LAUNCHER}")
    endif()
  endif()

  # Variants that require device-side launch (CDP) also require RDC support:
  _cub_launcher_requires_rdc(cdp_val "${launcher_id}")
  if (NOT "${label}" STREQUAL "" AND cdp_val AND NOT CUB_ENABLE_RDC_TESTS)
    set(launch_enabled 0)
  endif()

  set(${launcher_id_var} ${launcher_id} PARENT_SCOPE)
  set(${enabled_var} "${launch_enabled}" PARENT_SCOPE)
endfunction()

## _cub_no_variant_suffix
#
# Compute the synthetic variant suffix for sources without %PARAM% variants.
#
# FIXME: There are a few remaining device algorithm tests that have not been
# ported to use Catch2 and lid variants. Mark these as `lid_0/1` so they'll
# run in the appropriate CI configs:
function(_cub_no_variant_suffix out_var test_name launcher_id)
  set(variant_suffix)
  string(REGEX MATCH "^device\\." is_device_test "${test_name}")
  _cub_is_fail_test(is_fail_test "${test_name}")
  if (is_device_test AND NOT is_fail_test)
    string(APPEND variant_suffix ".lid_${launcher_id}")
  endif()
  set(${out_var} "${variant_suffix}" PARENT_SCOPE)
endfunction()

#[=======================================================================[.rst:
_cub_declare_quirk
------------------

Register a per-source exception applied to Catch2 test targets whose source
path matches a regex.

.. code-block:: cmake

  _cub_declare_quirk(
    [IF <precomputed-boolean-value>]
    SRC_REGEX <regex matched against the test source path>
    WHY <one-line rationale (required)>
    [COMPILE_OPTIONS <opt>...]
    [COMPILE_DEFINITIONS <def>...]
  )

``IF``
  A guard *value* (not a variable name), evaluated and normalized to 0/1 at
  declaration time. Defaults to enabled. Pass e.g.
  ``IF "${CCCL_ENABLE_EXPERIMENTAL_TILE_TRANSFORM_DISPATCH}"``.
``WHY``
  Required. Declarations are the documentation of record for exceptions.

Rows apply in declaration order; options/definitions are stored as lists and
applied UNQUOTED, so a historical two-token generator expression can (and
must) be declared as two list elements. Declarations must run at the
cub/test directory scope (rows are published via PARENT_SCOPE).

Structural quirks (configure_file, extra link edges, non-list target
properties) do not fit rows and live in _cub_apply_source_quirks.

Destiny: transitional scaffolding — if CUB ever moves from glob registration
to explicit per-test calls, rows dissolve into call-site arguments.
#]=======================================================================]
function(_cub_declare_quirk)
  cmake_parse_arguments(
    PARSE_ARGV
    0
    _CUB_QUIRK
    ""
    "IF;SRC_REGEX;WHY"
    "COMPILE_OPTIONS;COMPILE_DEFINITIONS"
  )

  if (_CUB_QUIRK_UNPARSED_ARGUMENTS)
    message(
      FATAL_ERROR
      "_cub_declare_quirk: unparsed arguments: ${_CUB_QUIRK_UNPARSED_ARGUMENTS}"
    )
  endif()
  if (NOT DEFINED _CUB_QUIRK_SRC_REGEX)
    message(FATAL_ERROR "_cub_declare_quirk: SRC_REGEX is required.")
  endif()
  if (NOT DEFINED _CUB_QUIRK_WHY OR "${_CUB_QUIRK_WHY}" STREQUAL "")
    message(FATAL_ERROR "_cub_declare_quirk: WHY is required.")
  endif()
  if (
    NOT DEFINED _CUB_QUIRK_COMPILE_OPTIONS
    AND NOT DEFINED _CUB_QUIRK_COMPILE_DEFINITIONS
  )
    message(
      FATAL_ERROR
      "_cub_declare_quirk: COMPILE_OPTIONS and/or COMPILE_DEFINITIONS "
      "is required (structural quirks belong in _cub_apply_source_quirks)."
    )
  endif()

  # Normalize the guard to a literal 0/1 at declaration time. Storing raw
  # values would make the application-time if() evaluate strings as variable
  # names (and error on empty values).
  #
  # `IF ""` is rejected loudly: under CMP0174's default, an empty value
  # leaves _CUB_QUIRK_IF undefined and would silently mean "enabled". The
  # literal `IF` token is still visible in ARGV, so an empty expansion (an
  # undefined option variable, say) is detectable. Callers must normalize
  # guards to a non-empty boolean first.
  list(FIND ARGV "IF" _cub_quirk_if_token)
  if (NOT _cub_quirk_if_token EQUAL -1 AND NOT DEFINED _CUB_QUIRK_IF)
    message(
      FATAL_ERROR
      "_cub_declare_quirk: IF requires a non-empty boolean value; normalize "
      "the guard (e.g. to TRUE/FALSE) before passing it."
    )
  endif()
  if (NOT DEFINED _CUB_QUIRK_IF)
    set(guard 1)
  elseif (_CUB_QUIRK_IF)
    set(guard 1)
  else()
    set(guard 0)
  endif()

  math(EXPR row "${_cub_quirk_count} + 1")
  set(_cub_quirk_count "${row}" PARENT_SCOPE)
  set(_cub_quirk_${row}_guard "${guard}" PARENT_SCOPE)
  set(_cub_quirk_${row}_regex "${_CUB_QUIRK_SRC_REGEX}" PARENT_SCOPE)
  set(_cub_quirk_${row}_why "${_CUB_QUIRK_WHY}" PARENT_SCOPE)
  set(_cub_quirk_${row}_options "${_CUB_QUIRK_COMPILE_OPTIONS}" PARENT_SCOPE)
  set(
    _cub_quirk_${row}_definitions
    "${_CUB_QUIRK_COMPILE_DEFINITIONS}"
    PARENT_SCOPE
  )
endfunction()

#[=======================================================================[.rst:
_cub_apply_source_quirks
------------------------

Apply per-source special cases to a test target: registry rows declared via
``_cub_declare_quirk`` (Catch2 tests only, in declaration order), plus
structural quirks that need real code (nvrtc header generation, nvtx
linking, future-arch pinning).

``test_type`` is ``Catch2`` or ``none`` (the run_test.cmake vocabulary).

Reads from calling scope: the ``_cub_quirk_*`` registry rows. Must run in
the cub/test directory scope: the nvrtc ``configure_file`` uses a relative
input path and ``CMAKE_CURRENT_BINARY_DIR``.

Parity invariants: registry rows apply only to the Catch2 arm and
nvtx/future-arch only to the legacy arm, mirroring the pre-refactor
branch scoping; options/definitions expand UNQUOTED (empty lists contribute
zero elements; multi-element rows land as multiple property entries).
#]=======================================================================]
function(_cub_apply_source_quirks test_target test_src test_type)
  if (test_type STREQUAL "Catch2")
    # The nvrtc test JIT-compiles CUB at runtime and needs a generated
    # header describing the configure-time include paths:
    if ("${test_target}" MATCHES "nvrtc")
      configure_file(
        "cmake/nvrtc_args.h.in"
        "${CMAKE_CURRENT_BINARY_DIR}/nvrtc_args.h"
      )
      target_include_directories(
        ${test_target}
        PRIVATE "${CMAKE_CURRENT_BINARY_DIR}"
      )
    endif()

    if (_cub_quirk_count GREATER 0)
      foreach (row RANGE 1 ${_cub_quirk_count})
        if (
          _cub_quirk_${row}_guard
          AND "${test_src}" MATCHES "${_cub_quirk_${row}_regex}"
        )
          if (_cub_quirk_${row}_options)
            target_compile_options(
              ${test_target}
              PRIVATE ${_cub_quirk_${row}_options}
            )
          endif()
          if (_cub_quirk_${row}_definitions)
            target_compile_definitions(
              ${test_target}
              PRIVATE ${_cub_quirk_${row}_definitions}
            )
          endif()
        endif()
      endforeach()
    endif()
  else() # Not catch2:
    if ("${test_target}" MATCHES "nvtx_in_usercode")
      target_link_libraries(${test_target} PRIVATE nvtx3-cpp)
    endif()

    if (
      "${test_src}" MATCHES "test_future_arch\\.cu$"
      AND "${CMAKE_CUDA_COMPILER_ID}" STREQUAL "NVIDIA"
      AND "${CMAKE_CUDA_COMPILER_VERSION}" VERSION_GREATER_EQUAL "13.0.0"
    )
      # In future arch test, we replace __CUDA_ARCH__ and __CUDA_ARCH_LIST__ with our own values. However, nvcc
      # preincludes <cuda_runtime.h> header, already working with __CUDA_ARCH__. So, we define the <cuda_runtime.h>'s
      # include guard macro so we can modify the values before including the header inside the source file.
      target_compile_definitions(${test_target} PRIVATE __CUDA_RUNTIME_H__)

      # Force only 1 architecture to be compiled. It must be the newest supported architecture.
      set_target_properties(
        ${test_target}
        PROPERTIES CUDA_ARCHITECTURES "121-virtual"
      )
    endif()
  endif()
endfunction()

#[=======================================================================[.rst:
_cub_add_test_executable
------------------------

Create the executable for a test target, including link/include setup and
(for Catch2 tests) the lazily-created per-launcher helper library.

.. code-block:: cmake

  _cub_add_test_executable(<test_target>
    METATARGET_PATH <path>
    SOURCE <test_src>
    LAUNCHER_ID <0|1|2>
    TYPE <Catch2|none>
    XFAIL <bool>
  )

Parity invariants: the helper library ``cub.test.catch2_helper.lid_<N>`` is
created on first use per launcher id, so configurations that build no lid_1
variants never create the lid_1 helper. Catch2 targets get the test include
dir PUBLIC; legacy targets PRIVATE plus ``CUB_DEBUG_SYNC``. XFAIL targets
are created with ``NO_CLANG_TIDY NO_METATARGETS``.

Destiny: hoists to cmake/ with the launcher-specific pieces parameterized.
#]=======================================================================]
function(_cub_add_test_executable test_target)
  cmake_parse_arguments(
    PARSE_ARGV
    1
    _CUB_EXE
    ""
    "METATARGET_PATH;SOURCE;LAUNCHER_ID;TYPE;XFAIL"
    ""
  )
  if (_CUB_EXE_UNPARSED_ARGUMENTS)
    message(
      FATAL_ERROR
      "_cub_add_test_executable: unparsed arguments: "
      "${_CUB_EXE_UNPARSED_ARGUMENTS}"
    )
  endif()
  foreach (
    required
    IN
    ITEMS METATARGET_PATH SOURCE LAUNCHER_ID TYPE XFAIL
  )
    if (NOT DEFINED _CUB_EXE_${required})
      message(FATAL_ERROR "_cub_add_test_executable: ${required} is required.")
    endif()
  endforeach()

  if (_CUB_EXE_TYPE STREQUAL "Catch2")
    _cub_launcher_requires_rdc(cdp_val "${_CUB_EXE_LAUNCHER_ID}")

    # Per config helper library:
    set(config_c2h_target cub.test.catch2_helper.lid_${_CUB_EXE_LAUNCHER_ID})
    if (NOT TARGET ${config_c2h_target})
      add_library(${config_c2h_target} INTERFACE)
      cccl_configure_target(${config_c2h_target})
      cccl_ensure_metatargets(${config_c2h_target})
      cub_configure_cuda_target(${config_c2h_target} RDC ${cdp_val})
      target_include_directories(
        ${config_c2h_target}
        INTERFACE "${CUB_SOURCE_DIR}/test"
      )
      target_link_libraries(
        ${config_c2h_target}
        INTERFACE #
          cub.compiler_interface
          cccl.c2h
          CUDA::nvrtc
          CUDA::cuda_driver
      )
    endif() # config_c2h_target

    cccl_add_executable(
      ${test_target}
      SOURCES "${_CUB_EXE_SOURCE}"
      METATARGET_PATH ${_CUB_EXE_METATARGET_PATH}
    )
    target_link_libraries(
      ${test_target}
      PRIVATE #
        cub.compiler_interface
        ${config_c2h_target}
        cccl.c2h.main
        Catch2::Catch2
    )
    target_include_directories(${test_target} PUBLIC "${CUB_SOURCE_DIR}/test")
  else() # Not catch2:
    if (_CUB_EXE_XFAIL)
      cccl_add_executable(
        ${test_target}
        SOURCES "${_CUB_EXE_SOURCE}"
        NO_CLANG_TIDY
        NO_METATARGETS
      )
    else()
      cccl_add_executable(
        ${test_target}
        SOURCES "${_CUB_EXE_SOURCE}"
        METATARGET_PATH ${_CUB_EXE_METATARGET_PATH}
      )
    endif()

    target_link_libraries(
      ${test_target}
      PRIVATE #
        cub.compiler_interface
        cccl.c2h
    )
    target_include_directories(${test_target} PRIVATE "${CUB_SOURCE_DIR}/test")
    target_compile_definitions(${test_target} PRIVATE CUB_DEBUG_SYNC)
  endif()
endfunction()

#[=======================================================================[.rst:
_cub_register_ctest
-------------------

Register a built test target with ctest. XFAIL tests become an
expected-compile-failure test pair (via ``cccl_add_xfail_compile_target_test``);
runtime tests are wrapped in ``run_test.cmake`` with the ``CCCL_SKIP_TEST``
skip regex and rapids-cmake GPU resource requirements keyed on the
suffix-free ``NAME``.

.. code-block:: cmake

  _cub_register_ctest(<test_target>
    NAME <suffix-free name>
    SOURCE <test_src>
    TYPE <Catch2|none>
    XFAIL <bool>
  )

Reads from calling scope: ``_cccl_gpu_test_runner_arg`` (directory scope;
a single ``-DGPU_TEST_RUNNER=<path>``
token when ON).

Parity invariants: the COMMAND token vector below is assembled explicitly
and expanded UNQUOTED so that empty optional arguments contribute zero
tokens; token order is the historical one. This block is the designated
swap point for CMake >= 3.29 TEST_LAUNCHER, should CCCL's floor ever allow
it. The first parameter MUST remain literally named `test_target` (see
PARITY INVARIANTS at the top of this file).
#]=======================================================================]
function(_cub_register_ctest test_target)
  cmake_parse_arguments(PARSE_ARGV 1 _CUB_REG "" "NAME;SOURCE;TYPE;XFAIL" "")
  if (_CUB_REG_UNPARSED_ARGUMENTS)
    message(
      FATAL_ERROR
      "_cub_register_ctest: unparsed arguments: ${_CUB_REG_UNPARSED_ARGUMENTS}"
    )
  endif()
  foreach (required IN ITEMS NAME SOURCE TYPE XFAIL)
    if (NOT DEFINED _CUB_REG_${required})
      message(FATAL_ERROR "_cub_register_ctest: ${required} is required.")
    endif()
  endforeach()

  if (_CUB_REG_XFAIL)
    cccl_add_xfail_compile_target_test(
      ${test_target}
      SOURCE_FILE "${_CUB_REG_SOURCE}"
      ERROR_REGEX_LABEL "expected-error"
      ERROR_NUMBER_TARGET_NAME_REGEX "\\.err_([0-9]+)"
    )
  else()
    # Assemble the historical COMMAND token vector explicitly. Optional
    # arguments are lists that contribute zero tokens when empty; the final
    # expansion below is deliberately unquoted for the same reason.
    set(test_command "${CMAKE_COMMAND}")
    list(APPEND test_command "-DCCCL_SOURCE_DIR=${CCCL_SOURCE_DIR}")
    list(APPEND test_command "-DTEST=$<TARGET_FILE:${test_target}>")
    if (_CUB_REG_TYPE STREQUAL "Catch2")
      list(APPEND test_command "-DTYPE=Catch2")
    endif()
    list(APPEND test_command ${_cccl_gpu_test_runner_arg})
    list(APPEND test_command -P "${CUB_SOURCE_DIR}/test/run_test.cmake")

    add_test(NAME ${test_target} COMMAND ${test_command})
    set_tests_properties(
      ${test_target}
      PROPERTIES SKIP_REGULAR_EXPRESSION "CCCL_SKIP_TEST"
    )
    cub_test_gpu_percent(_cub_gpu_percent "${_CUB_REG_NAME}")
    rapids_test_gpu_requirements(
      ${test_target}
      GPUS 1
      PERCENT ${_cub_gpu_percent}
    )
  endif()
endfunction()

#[=======================================================================[.rst:
cub_add_test
------------

Create and register one CUB test variant: executable (+ lazily-created
per-launcher Catch2 helper library), per-source quirks, and ctest entry.

.. code-block:: cmake

  cub_add_test(<target_name_var>
    NAME <suffix-free test name>       # e.g. device.scan
    SUFFIX <variant suffix or "">      # e.g. .lid_0.types_2
    SOURCE <path relative to cub/test>
    LAUNCHER_ID <0|1|2>                # 0=host, 1=device (RDC), 2=graph
    [DEFINITIONS <def>...]             # %PARAM% variant definitions
  )

``target_name_var`` receives the created target's name
(``cub.test.<NAME><SUFFIX>``). The returned name is stable API: post-call
``target_*`` mutation is supported.

``LAUNCHER_ID`` also determines RDC: id 1 enables separable compilation on
the test target and its helper library.

``DEFINITIONS`` are appended after ``CCCL_ENABLE_ASSERTIONS`` to preserve
the historical ``COMPILE_DEFINITIONS`` order (the pre-refactor loop applied
them after the call returned).

NOTE: keyword parsing uses ``PARSE_ARGV``: call sites legitimately pass
``SUFFIX ""`` and classic ``${ARGN}`` parsing would silently drop the empty
argument.
#]=======================================================================]
function(cub_add_test target_name_var)
  cmake_parse_arguments(
    PARSE_ARGV
    1
    _CUB_TEST
    ""
    "NAME;SUFFIX;SOURCE;LAUNCHER_ID"
    "DEFINITIONS"
  )
  if (_CUB_TEST_UNPARSED_ARGUMENTS)
    message(
      FATAL_ERROR
      "cub_add_test: unparsed arguments: ${_CUB_TEST_UNPARSED_ARGUMENTS}"
    )
  endif()
  foreach (required IN ITEMS NAME SOURCE LAUNCHER_ID)
    if (NOT DEFINED _CUB_TEST_${required})
      message(FATAL_ERROR "cub_add_test: ${required} is required.")
    endif()
  endforeach()
  if (NOT DEFINED _CUB_TEST_SUFFIX)
    set(_CUB_TEST_SUFFIX "")
  endif()

  _cub_is_catch2_test(is_catch2_test "${_CUB_TEST_SOURCE}")
  if (is_catch2_test)
    set(test_type "Catch2")
    set(is_fail_test FALSE)
  else()
    set(test_type "none")
    _cub_is_fail_test(is_fail_test "${_CUB_TEST_SOURCE}")
  endif()

  # The actual name of the test's target:
  set(test_target cub.test.${_CUB_TEST_NAME}${_CUB_TEST_SUFFIX})
  set(${target_name_var} ${test_target} PARENT_SCOPE)

  # The metatarget path ignores the variant suffix:
  set(metatarget_path cub.test.${_CUB_TEST_NAME})

  _cub_add_test_executable(
    "${test_target}"
    METATARGET_PATH "${metatarget_path}"
    SOURCE "${_CUB_TEST_SOURCE}"
    LAUNCHER_ID "${_CUB_TEST_LAUNCHER_ID}"
    TYPE "${test_type}"
    XFAIL "${is_fail_test}"
  )
  # Statement order mirrors the pre-refactor file exactly: the Catch2 branch
  # applied per-source quirks after ctest registration, the legacy branch
  # before it. Quirks only mutate target properties, but the parity contract
  # pins the historical order.
  if (test_type STREQUAL "Catch2")
    _cub_register_ctest(
      "${test_target}"
      NAME "${_CUB_TEST_NAME}"
      SOURCE "${_CUB_TEST_SOURCE}"
      TYPE "${test_type}"
      XFAIL "${is_fail_test}"
    )
    _cub_apply_source_quirks(
      "${test_target}"
      "${_CUB_TEST_SOURCE}"
      "${test_type}"
    )
  else()
    _cub_apply_source_quirks(
      "${test_target}"
      "${_CUB_TEST_SOURCE}"
      "${test_type}"
    )
    _cub_register_ctest(
      "${test_target}"
      NAME "${_CUB_TEST_NAME}"
      SOURCE "${_CUB_TEST_SOURCE}"
      TYPE "${test_type}"
      XFAIL "${is_fail_test}"
    )
  endif()

  # Disable build caching for compile tests
  set_tests_properties(${test_target} PROPERTIES ENVIRONMENT SCCACHE_NO_CACHE=1)

  # Ensure that we test with assertions enabled
  target_compile_definitions(${test_target} PRIVATE CCCL_ENABLE_ASSERTIONS)

  # Disable clang-cuda >= 22 warnings regarding failed loop unrolling.
  if (
    "${CMAKE_CUDA_COMPILER_ID}" STREQUAL "Clang"
    AND "${CMAKE_CUDA_COMPILER_VERSION}" VERSION_GREATER_EQUAL "22.0.0"
  )
    target_compile_options(${test_target} PRIVATE "-Wno-pass-failed")
  endif()

  # Variant definitions land after CCCL_ENABLE_ASSERTIONS, matching the
  # pre-refactor ordering where the caller applied them post-return:
  if (_CUB_TEST_DEFINITIONS)
    target_compile_definitions(${test_target} PRIVATE ${_CUB_TEST_DEFINITIONS})
  endif()

  # Enable RDC if the test either:
  # 1. Explicitly requests it (lid_1 label)
  # 2. Does not have an explicit CDP variant (no lid_0, lid_1, or lid_2) but
  #    RDC testing is forced
  #
  # Tests that explicitly request no cdp (lid_0 label) should never enable
  # RDC.
  _cub_launcher_requires_rdc(cdp_val "${_CUB_TEST_LAUNCHER_ID}")
  cub_configure_cuda_target(${test_target} RDC ${cdp_val})
endfunction()
