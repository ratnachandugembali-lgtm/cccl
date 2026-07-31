# Check all source files for various issues that can be detected using pattern
# matching.
#
# This is run as a ctest test named `cub.test.cmake.check_source_files`, or
# manually with:
# cmake -D "CUB_SOURCE_DIR=<CUB project root>" -P check_source_files.cmake

cmake_minimum_required(VERSION 3.15)

function(count_substrings input search_regex output_var)
  string(REGEX MATCHALL "${search_regex}" matches "${input}")
  list(LENGTH matches num_matches)
  set(${output_var} ${num_matches} PARENT_SCOPE)
endfunction()

set(found_errors 0)
file(
  GLOB_RECURSE cub_srcs
  RELATIVE "${CUB_SOURCE_DIR}"
  "${CUB_SOURCE_DIR}/cub/*.cuh"
  "${CUB_SOURCE_DIR}/cub/*.cu"
  "${CUB_SOURCE_DIR}/cub/*.h"
  "${CUB_SOURCE_DIR}/cub/*.cpp"
)

file(
  GLOB_RECURSE cub_test_srcs
  RELATIVE "${CUB_SOURCE_DIR}"
  "${CUB_SOURCE_DIR}/test/*.cuh"
  "${CUB_SOURCE_DIR}/test/*.cu"
  "${CUB_SOURCE_DIR}/test/*.h"
  "${CUB_SOURCE_DIR}/test/*.cpp"
)

################################################################################
# Namespace checks.
# Check all files in thrust to make sure that they use
# CUB_NAMESPACE_BEGIN/END instead of bare `namespace cub {}` declarations.
set(
  namespace_exclusions
  # This defines the macros and must have bare namespace declarations:
  cub/util_namespace.cuh
)

set(bare_ns_regex "namespace[ \n\r\t]+cub[ \n\r\t]*\\{")

# Validation check for the above regex:
count_substrings([=[
namespace cub{
namespace cub {
namespace  cub  {
 namespace cub {
namespace cub
{
namespace
cub
{
]=]
  ${bare_ns_regex} valid_count
)
if (NOT valid_count EQUAL 6)
  message(
    FATAL_ERROR
    "Validation of bare namespace regex failed: "
    "Matched ${valid_count} times, expected 6."
  )
endif()

################################################################################
# stdpar header checks.
# Check all files in CUB to make sure that they aren't including <algorithm>
# or <memory>, both of which will cause circular dependencies in nvc++'s
# stdpar library.
#
# The headers following headers should be used instead:
# <algorithm> -> <cuda/std/__host_stdlib/algorithm>
# <memory>    -> <cuda/std/__host_stdlib/memory>
# <numeric>   -> <cuda/std/__host_stdlib/numeric>
#
set(
  stdpar_header_exclusions
  # Placeholder -- none yet.
)

set(algorithm_regex "#[ \t]*include[ \t]+<algorithm>")
set(memory_regex "#[ \t]*include[ \t]+<memory>")
set(numeric_regex "#[ \t]*include[ \t]+<numeric>")

# Validation check for the above regex pattern:
count_substrings([=[
#include <algorithm>
# include <algorithm>
#include  <algorithm>
# include  <algorithm>
# include  <algorithm> // ...
]=]
  ${algorithm_regex} valid_count
)
if (NOT valid_count EQUAL 5)
  message(
    FATAL_ERROR
    "Validation of stdpar header regex failed: "
    "Matched ${valid_count} times, expected 5."
  )
endif()

################################################################################
# Test registration checks.
# Catch2 tests must use the CUB wrappers so every test case declares its GPU
# memory class.
set(
  raw_test_registration_regex
  "(^|[\n\r])[ \t]*(C2H_TEST(_LIST)?(_WITH_FIXTURE)?|TEST_CASE(_METHOD)?|SCENARIO(_METHOD)?|TEMPLATE_((PRODUCT_)?TEST_CASE(_METHOD)?(_SIG)?|LIST_TEST_CASE(_METHOD)?))[ \t]*\\("
)

# Validation check for the above regex pattern:
count_substrings([=[
C2H_TEST(
C2H_TEST_LIST(
C2H_TEST_WITH_FIXTURE(
C2H_TEST_LIST_WITH_FIXTURE(
TEST_CASE(
TEST_CASE_METHOD(
SCENARIO(
SCENARIO_METHOD(
TEMPLATE_TEST_CASE(
TEMPLATE_TEST_CASE_SIG(
TEMPLATE_TEST_CASE_METHOD(
TEMPLATE_TEST_CASE_METHOD_SIG(
TEMPLATE_PRODUCT_TEST_CASE(
TEMPLATE_PRODUCT_TEST_CASE_SIG(
TEMPLATE_PRODUCT_TEST_CASE_METHOD(
TEMPLATE_PRODUCT_TEST_CASE_METHOD_SIG(
TEMPLATE_LIST_TEST_CASE(
TEMPLATE_LIST_TEST_CASE_METHOD(
CUB_TEST(
CUB_TEST_CASE(
CUB_TEST_LIST(
]=]
  "${raw_test_registration_regex}" valid_count
)
if (NOT valid_count EQUAL 18)
  message(
    FATAL_ERROR
    "Validation of raw test registration regex failed: "
    "Matched ${valid_count} times, expected 18."
  )
endif()

################################################################################
# Read source files:
foreach (src ${cub_srcs})
  file(READ "${CUB_SOURCE_DIR}/${src}" src_contents)

  if (NOT ${src} IN_LIST namespace_exclusions)
    count_substrings("${src_contents}" "${bare_ns_regex}" bare_ns_count)
    count_substrings("${src_contents}" CUB_NS_PREFIX prefix_count)
    count_substrings("${src_contents}" CUB_NS_POSTFIX postfix_count)
    count_substrings("${src_contents}" CUB_NAMESPACE_BEGIN begin_count)
    count_substrings("${src_contents}" CUB_NAMESPACE_END end_count)

    if (NOT bare_ns_count EQUAL 0)
      message(
        "'${src}' contains 'namespace cub {...}'. Replace with CUB_NAMESPACE macros."
      )
      set(found_errors 1)
    endif()

    if (NOT prefix_count EQUAL 0)
      message(
        "'${src}' contains 'CUB_NS_PREFIX'. Replace with CUB_NAMESPACE macros."
      )
      set(found_errors 1)
    endif()

    if (NOT postfix_count EQUAL 0)
      message(
        "'${src}' contains 'CUB_NS_POSTFIX'. Replace with CUB_NAMESPACE macros."
      )
      set(found_errors 1)
    endif()

    if (NOT begin_count EQUAL end_count)
      message("'${src}' namespace macros are unbalanced:")
      message(" - CUB_NAMESPACE_BEGIN occurs ${begin_count} times.")
      message(" - CUB_NAMESPACE_END   occurs ${end_count} times.")
      set(found_errors 1)
    endif()
  endif()

  if (NOT ${src} IN_LIST stdpar_header_exclusions)
    count_substrings("${src_contents}" "${algorithm_regex}" algorithm_count)
    count_substrings("${src_contents}" "${memory_regex}" memory_count)
    count_substrings("${src_contents}" "${numeric_regex}" numeric_count)

    if (NOT algorithm_count EQUAL 0)
      message(
        "'${src}' includes the <algorithm> header. Replace with <cuda/std/__host_stdlib/algorithm>."
      )
      set(found_errors 1)
    endif()

    if (NOT memory_count EQUAL 0)
      message(
        "'${src}' includes the <memory> header. Replace with <cuda/std/__host_stdlib/memory>."
      )
      set(found_errors 1)
    endif()

    if (NOT numeric_count EQUAL 0)
      message(
        "'${src}' includes the <numeric> header. Replace with <cuda/std/__host_stdlib/numeric>."
      )
      set(found_errors 1)
    endif()
  endif()
endforeach()

foreach (src ${cub_test_srcs})
  file(READ "${CUB_SOURCE_DIR}/${src}" src_contents)
  count_substrings(
    "${src_contents}"
    "${raw_test_registration_regex}"
    raw_test_registration_count
  )
  if (NOT raw_test_registration_count EQUAL 0)
    message("'${src}' registers tests without using a CUB_TEST wrapper.")
    set(found_errors 1)
  endif()
endforeach()

if (NOT found_errors EQUAL 0)
  message(FATAL_ERROR "Errors detected.")
endif()
