if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  if (CUB_ENABLE_RDC_TESTS)
    if ("${CMAKE_VERSION}" VERSION_LESS 3.27.5)
      # https://gitlab.kitware.com/cmake/cmake/-/merge_requests/8794
      message(
        WARNING
        "CMake 3.27.5 or newer is required to enable RDC tests in Visual Studio."
      )
      cmake_minimum_required(VERSION 3.27.5)
    endif()
  endif()
endif()

cccl_get_c2h()
cccl_get_cudatoolkit()
cccl_get_nvtx()

# CTest GPU resource scheduling. Each CUB test gets a RESOURCE_GROUPS claim
# from the rapids-cmake test helpers, so parallel ctest runs do not place too
# many GPU-heavy tests on the same device. The generated resource spec needs
# CMake 3.28.
if ("${CMAKE_VERSION}" VERSION_LESS 3.28)
  message(WARNING "CMake 3.28 or newer is required to build CUB tests.")
  cmake_minimum_required(VERSION 3.28)
endif()
cccl_get_rapids_test()
rapids_test_init()
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/CubTestFootprints.cmake")

# Stage the rapids-cmake GPU runner into the build tree. The runner only uses
# CMake features available before 3.28, so keep its local minimum aligned with
# the CUB test floor instead of rapids-cmake's repo-wide floor.
file(
  READ "${rapids-cmake_SOURCE_DIR}/rapids-cmake/test/detail/run_gpu_test.cmake"
  _cub_gpu_test_runner_content
)
string(
  REGEX REPLACE
  "cmake_minimum_required\\(VERSION [0-9.]+"
  "cmake_minimum_required(VERSION 3.28"
  _cub_gpu_test_runner_content
  "${_cub_gpu_test_runner_content}"
)
file(
  WRITE "${PROJECT_BINARY_DIR}/rapids-cmake/run_gpu_test.cmake"
  "${_cub_gpu_test_runner_content}"
)
# Forwarded to run_test.cmake by cub_add_test:
set(
  _cccl_gpu_test_runner_arg
  "-DGPU_TEST_RUNNER=${PROJECT_BINARY_DIR}/rapids-cmake/run_gpu_test.cmake"
)

set(build_nvrtc_tests ON)
if ("NVHPC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
  set(build_nvrtc_tests OFF)
endif()

file(
  GLOB_RECURSE test_srcs
  RELATIVE "${CUB_SOURCE_DIR}/test"
  CONFIGURE_DEPENDS
  test_*.cu
  catch2_test_*.cu
)

# nvtx headers contain a variable named `module`, which breaks nvc++ as that is a keyword
if (
  "NVHPC" STREQUAL "${CMAKE_CXX_COMPILER_ID}"
  AND NOT "${CMAKE_CXX_STANDARD}" MATCHES "17"
)
  list(FILTER test_srcs EXCLUDE REGEX "test_nvtx*")
endif()

# =============================== Registration machinery ======================
# All _cub_* functions live in the module below; this file declares only
# policy: which sources exist (glob above), per-source quirks, and the
# registration loop.
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/CubTestRegistration.cmake")

# ================================ Per-source quirks ===========================
# Exceptions are declarations, applied to Catch2 test targets in declaration
# order by _cub_apply_source_quirks. WHY is required. Structural quirks
# (nvrtc header generation, nvtx linking, future-arch pinning) live in
# _cub_apply_source_quirks itself.

set(_cub_host_is_clang_ge_13 FALSE)
if (
  "Clang" STREQUAL "${CMAKE_CXX_COMPILER_ID}"
  AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13
)
  set(_cub_host_is_clang_ge_13 TRUE)
endif()

# Quirk guards must be non-empty booleans (see _cub_declare_quirk):
set(_cub_tile_transform_enabled FALSE)
if (CCCL_ENABLE_EXPERIMENTAL_TILE_TRANSFORM_DISPATCH)
  set(_cub_tile_transform_enabled TRUE)
endif()

_cub_declare_quirk(
  SRC_REGEX "test_iterator\\.cu$"
  WHY "large type lists exceed the default template instantiation depth"
  COMPILE_OPTIONS -ftemplate-depth=1000
)
_cub_declare_quirk(
  IF "${_cub_tile_transform_enabled}"
  SRC_REGEX "test_device_transform_tile\\.cu$"
  WHY "experimental tile transform dispatch needs the nvcc tile extension"
  COMPILE_OPTIONS "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--enable-tile>"
  COMPILE_DEFINITIONS _CCCL_ENABLE_EXPERIMENTAL_TILE_TRANSFORM_DISPATCH
)
_cub_declare_quirk(
  SRC_REGEX "test.+_api\\.cu$"
  WHY "API examples use extended lambdas"
  COMPILE_OPTIONS "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda>"
)
# When clang >= 13 is used as host compiler, we get the following warning:
#   nvcc_internal_extended_lambda_implementation:312:22: error: definition of implicit copy constructor for '__nv_hdl_wrapper_t<false, true, false, __nv_dl_tag<void (*)(), &TestAddressStabilityLambda, 2>, int (const int &)>' is deprecated because it has a user-declared copy assignment operator [-Werror,-Wdeprecated-copy]
#   312 | __nv_hdl_wrapper_t & operator=(const __nv_hdl_wrapper_t &in) = delete;
#   |                      ^
# Suppress it until NVBug 4980157 is resolved. The option is declared as TWO
# list elements on purpose: the historical call site passed the generator
# expression unquoted with an embedded space, and CMake's argument splitting
# stored two COMPILE_OPTIONS entries. Keeping the same two-element shape is
# part of the byte-parity contract (the parity harness asserts it).
foreach (
  _cub_dc_src
  IN
  ITEMS
    "test_device_segmented_scan_api\\.cu$"
    "test_device_find_env_api\\.cu$"
    "test_device_partition_env_api\\.cu$"
    "test_device_select_env_api\\.cu$"
)
  _cub_declare_quirk(
    IF "${_cub_host_is_clang_ge_13}"
    SRC_REGEX "${_cub_dc_src}"
    WHY
      "NVBug 4980157: -Wdeprecated-copy from nvcc's extended-lambda wrapper on clang >= 13 hosts"
    COMPILE_OPTIONS
      "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:"
      "-Wno-deprecated-copy>"
  )
endforeach()

# ================================ Test registration ===========================
foreach (test_src IN LISTS test_srcs)
  _cub_test_name_from_source(test_name "${test_src}")

  _cub_test_excluded(test_excluded "${test_name}")
  if (test_excluded)
    continue()
  endif()

  cccl_parse_variant_params(
    "${test_src}"
    num_variants
    variant_labels
    variant_defs
  )

  if (num_variants EQUAL 0)
    # Only one version of this test.
    _cub_launch_policy(launcher_id launch_enabled "")
    _cub_no_variant_suffix(variant_suffix "${test_name}" "${launcher_id}")
    cub_add_test(
      test_target
      NAME ${test_name}
      SUFFIX "${variant_suffix}"
      SOURCE "${test_src}"
      LAUNCHER_ID ${launcher_id}
    )
  else() # has variants:
    cccl_log_variant_params(
      "${test_name}"
      ${num_variants}
      variant_labels
      variant_defs
    )

    # Subtract 1 to support the inclusive endpoint of foreach(...RANGE...):
    math(EXPR range_end "${num_variants} - 1")

    # Generate multiple tests, one per variant.
    foreach (var_idx RANGE ${range_end})
      cccl_get_variant_data(variant_labels variant_defs ${var_idx} label defs)
      _cub_launch_policy(launcher_id launch_enabled "${label}")
      if (NOT launch_enabled)
        continue()
      endif()
      cub_add_test(
        test_target
        NAME ${test_name}
        SUFFIX ".${label}"
        SOURCE "${test_src}"
        LAUNCHER_ID ${launcher_id}
        DEFINITIONS ${defs}
      )
    endforeach() # Variant
  endif() # Has variants
endforeach() # Source file

# ================================== Sub-suites ================================
add_subdirectory(cmake)
add_subdirectory(cubin-check)
