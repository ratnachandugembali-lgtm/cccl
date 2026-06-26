#!/usr/bin/env bash

# CI wrapper for the `target` project test job.
# Invokes ci/util/build_and_test_targets.sh with the provided arguments to
# build and test selected targets on a GPU runner.

set -euo pipefail

ci_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd "${ci_dir}/.." && pwd)

build_common_args=()
target_args=()
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    -v | --verbose | -verbose | -pedantic | --pedantic)
      build_common_args+=("$1")
      shift
      ;;
    -std | -arch | -cuda | -cxx | -cmake-options)
      build_common_args+=("$1" "$2")
      shift 2
      ;;
    *)
      target_args+=("$1")
      shift
      ;;
  esac
done

set -- "${build_common_args[@]}"
source "${ci_dir}/build_common.sh"

cd "${repo_dir}"
cmd=("${ci_dir}/util/build_and_test_targets.sh" "${target_args[@]}")
if [[ "${#GLOBAL_CMAKE_OPTIONS[@]}" -gt 0 ]]; then
  cmd+=(--cmake-options "${GLOBAL_CMAKE_OPTIONS[*]}")
fi
printf '\033[34m%s\033[0m\n' "${cmd[*]}"
"${cmd[@]}"
