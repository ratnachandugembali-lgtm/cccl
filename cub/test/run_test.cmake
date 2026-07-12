#
# Launch a test, optionally enabling runtime sanitizers, etc.
#

function(usage)
  message("Usage:")
  message("  cmake -D CCCL_SOURCE_DIR=/path/to/cccl \\")
  message("        -D TEST=bin/test.exe \\")
  message("        -D ARGS=\"arg1;arg2;arg3\" \\")
  message("        -D TYPE=Catch2 \\")
  message("        -D MODE=compute-sanitizer-memcheck \\")
  message("        -P cccl/cub/test/run_test.cmake")
  message("")
  message("  - CCCL_SOURCE_DIR: Required.  Path to the CCCL source directory.")
  message("  - TEST: Required.  Path to the test executable.")
  message("  - ARGS: Optional.  Arguments to pass to the test executable.")
  message("  - TYPE: Optional.")
  message("    - The test framework used by the test executable.")
  message("    - Must be one of the following:")
  message("    -  \"none\" (default)")
  message("    -  \"Catch2\"")
  message("  - MODE: Optional.")
  message("    - May be set through CCCL_TEST_MODE env var.")
  message("    - Must be one of the following:")
  message("    -  \"none\" (default)")
  message("    -  \"compute-sanitizer-memcheck\"")
  message("    -  \"compute-sanitizer-racecheck\"")
  message("    -  \"compute-sanitizer-initcheck\"")
  message("    -  \"compute-sanitizer-synccheck\"")
endfunction()

# Usage:
#   run_command(COMMAND [ARGS...])
#
# The command is printed before it is executed.
# The new process's stdout, sterr are redirected to the current process.
# The current process will exit with an error if the new process exits with a non-zero status.
function(run_command command)
  set(command_str "${command}")
  list(APPEND command_str ${ARGN})
  list(JOIN command_str " " command_str)
  message(STATUS ">> Running:\n\t${command_str}")
  execute_process(COMMAND ${command} ${ARGN} RESULT_VARIABLE result)
  if (NOT result EQUAL 0)
    message(FATAL_ERROR ">> Exit Status: ${result}")
  else()
    message(STATUS ">> Exit Status: ${result}")
  endif()
endfunction()

######################################################################

# Parse arguments
#
# The normal CUB add_test path passes TEST and ARGS through -D variables.
# Catch2 discovery uses this script as a target launcher, so the executable and
# Catch2 arguments arrive after -P instead. Support both forms so discovered
# Catch2 tests still run through this wrapper.
if (NOT DEFINED TEST AND DEFINED CMAKE_ARGC)
  set(_script_arg_start -1)
  math(EXPR _last_arg "${CMAKE_ARGC} - 1")
  foreach (_arg_idx RANGE 0 ${_last_arg})
    if ("${CMAKE_ARGV${_arg_idx}}" STREQUAL "${CMAKE_SCRIPT_MODE_FILE}")
      math(EXPR _script_arg_start "${_arg_idx} + 1")
      break()
    endif()
  endforeach()

  if (NOT _script_arg_start EQUAL -1 AND _script_arg_start LESS CMAKE_ARGC)
    set(TEST "${CMAKE_ARGV${_script_arg_start}}")
    math(EXPR _arg_idx "${_script_arg_start} + 1")
    while (_arg_idx LESS CMAKE_ARGC)
      list(APPEND ARGS "${CMAKE_ARGV${_arg_idx}}")
      math(EXPR _arg_idx "${_arg_idx} + 1")
    endwhile()
  endif()
endif()

if (NOT DEFINED TEST)
  usage()
  message(FATAL_ERROR "TEST must be defined")
endif()

if (NOT DEFINED ARGS)
  set(ARGS)
endif()

if (NOT DEFINED TYPE)
  set(TYPE "none")
endif()

# Catch2 discovery/listing must produce clean machine-readable output for
# catch_discover_tests. Do not wrap it in compute-sanitizer or print the usual
# command banner.
set(is_catch2_discovery OFF)
if (TYPE STREQUAL "Catch2")
  foreach (arg IN LISTS ARGS)
    if (
      arg STREQUAL "--list-tests"
      OR arg STREQUAL "--list-test-names-only"
      OR arg STREQUAL "--list-reporters"
    )
      set(is_catch2_discovery ON)
    endif()
  endforeach()
endif()

if (is_catch2_discovery)
  set(MODE "none")
elseif (NOT DEFINED MODE)
  if (DEFINED ENV{CCCL_TEST_MODE})
    message(STATUS "Using CCCL_TEST_MODE from env: $ENV{CCCL_TEST_MODE}")
    set(MODE $ENV{CCCL_TEST_MODE})
  else()
    set(MODE "none")
  endif()
elseif (NOT MODE)
  set(MODE "none")
endif()

if (MODE STREQUAL "none")
  if (is_catch2_discovery)
    execute_process(COMMAND ${TEST} ${ARGS} RESULT_VARIABLE result)
    if (NOT result EQUAL 0)
      message(FATAL_ERROR ">> Exit Status: ${result}")
    endif()
  else()
    run_command(${TEST} ${ARGS})
  endif()
elseif (MODE MATCHES "^compute-sanitizer-(.*)$")
  set(tool ${CMAKE_MATCH_1})

  # All test cases in these tests take an excessive amount of time to execute with the compute-sanitizer tools.
  # Just skip the whole test.
  if (
    (
      "${TEST}" MATCHES "/cub.+test.*large_offsets.*"
      AND
        (
          tool STREQUAL "initcheck"
          OR tool STREQUAL "racecheck"
          OR tool STREQUAL "synccheck"
        )
    )
    OR
    # These take >2000s on racecheck to complete, even with reduced seeds, skipping large offsets, etc.
    (
      "${TEST}" MATCHES "/cub.+test.device_segmented_(radix_sort|reduce).*"
      AND (tool STREQUAL "racecheck")
    )
  )
    message(
      FATAL_ERROR
      "CCCL_SKIP_TEST:\n${TEST} takes an excessive amount of time to execute with ${tool}."
    )
  endif()

  # These tests intentionally don't launch any kernels or make any CUDA API calls:
  if ("${TEST}" MATCHES "/cub.*.test.cdp_variant_state.*")
    message(
      FATAL_ERROR
      "CCCL_SKIP_TEST:\n${TEST} intentionally doesn't use CUDA APIs. Skipping."
    )
  endif()

  # The CUB debug test intentionally throws CUDA errors to test error handling:
  if ("${TEST}" MATCHES "/cub.*test.debug$")
    message(
      FATAL_ERROR
      "CCCL_SKIP_TEST:\n${TEST} intentionally throws CUDA errors. Skipping."
    )
  endif()

  if (TYPE STREQUAL "Catch2")
    list(APPEND ARGS "--durations" "yes" "~[skip-cs-${tool}]")
  endif()

  # Compile the compute-sanitizer args:
  # gersemi: off
  set(cs_general_args
    "--tool" "${tool}"
    "--suppressions" "${CCCL_SOURCE_DIR}/ci/compute-sanitizer-suppressions.xml"
    "--check-device-heap" "yes"
    "--check-bulk-copy" "yes"
    "--check-exit-code" "yes"
    "--check-warpgroup-mma" "yes"
    "--error-exitcode" "1"
    "--nvtx" "true"
  )
  # gersemi: on
  if (tool STREQUAL "memcheck")
    # gersemi: off
    set(cs_tool_args
      "--leak-check" "full"
      "--padding" "512"
      "--track-stream-ordered-races" "all"
    )
    # gersemi: on
  elseif (tool STREQUAL "racecheck")
    set(cs_tool_args)
  elseif (tool STREQUAL "initcheck")
    set(cs_tool_args)
  elseif (tool STREQUAL "synccheck")
    set(cs_tool_args)
  endif()

  run_command(compute-sanitizer
    ${cs_general_args}
    ${cs_tool_args}
    ${TEST} ${ARGS}
  )
else()
  usage()
  message(FATAL_ERROR "Invalid MODE: ${MODE}")
endif()
