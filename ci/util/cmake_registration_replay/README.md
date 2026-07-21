# CUB test-registration replay parity harness

Proves that a change to CUB's CMake test registration
(`cub/test/CMakeLists.txt` + `cub/test/cmake/CubTestRegistration.cmake`) is
behavior-identical to a reference git revision — without needing a CUDA
toolchain, a compiler, or a GPU.

## How it works

Both the reference revision's registration file and the working tree's are
replayed inside a `LANGUAGES NONE` CMake project in which every effectful
command (`add_test`, `set_tests_properties`, `set_property(TEST ...)`,
`add_library`, `target_compile_definitions`, `target_compile_options`,
`target_link_libraries`, `configure_file`, ...) is shadowed by a recorder
(`shims.cmake`). The replay runs over the *real* test sources (the real
`%PARAM%` scanner, the real xfail machinery from `cmake/CCCLUtilities.cmake`,
the real RDC helper from `cub/cmake/CubUtilities.cmake`, and the real
`rapids_test_gpu_requirements`), so every observable registration decision —
test names, ctest COMMAND token vectors, properties, compile
definitions/options and their order, RDC calls, GPU resource claims — is
captured as an ordered event stream. The two streams must be byte-identical
per configuration row.

Rows exercised: default options, `CUB_FORCE_RDC`,
`CUB_ENABLE_RDC_TESTS=OFF`, host-only launchers,
`CUB_ENABLE_LAUNCH_NO_LAUNCHER=OFF`, clang>=13 host (exercises the
`-Wno-deprecated-copy` two-element quirk and `future_arch`), the
tile-transform option, and clang-cuda>=22. GPU resource scheduling is
always active, so every row includes the rapids-cmake test functions
(set `RAPIDS_SRC` to a rapids-cmake checkout).

## Usage

```bash
# Compare the working tree against a reference revision:
ci/util/cmake_registration_replay/run.sh <old-git-ref>

# The scheduling row needs a rapids-cmake checkout:
RAPIDS_SRC=/path/to/rapids-cmake ci/util/cmake_registration_replay/run.sh <old-git-ref>
```

Requires CMake >= 3.28 on PATH. Output: `PASS(<row>): N events identical`
per row; on failure, a unified diff of the event streams is printed and kept
under the scratch directory.

## What it cannot see

The harness stubs shared infrastructure (`cccl_add_executable`,
`cccl_configure_target`, metatargets) with recorders — it proves the
registration layer calls them identically, not what they do internally.
Changes to those shared helpers, to real compile/link behavior, or to
`run_test.cmake`'s runtime dispatch need a real configure diff
(`ctest --show-only=json-v1`) on a CUDA machine and/or a smoke test run.
