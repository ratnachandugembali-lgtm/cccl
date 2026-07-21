#!/bin/bash
# Replays OLD (git ref) vs NEW (working tree) cub/test/CMakeLists.txt over
# the real test sources with all effectful CMake commands shadowed by
# recorders, then diffs the ordered event streams. See README.md.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <old-git-ref>   (env: RAPIDS_SRC=<rapids-cmake checkout> to enable the 'sched' row)" >&2
  exit 2
fi
old_ref="$1"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(git -C "$here" rev-parse --show-toplevel)"
scratch="${REPLAY_SCRATCH:-$(mktemp -d "${TMPDIR:-/tmp}/cub-reg-replay.XXXXXX")}"
rapids_src="${RAPIDS_SRC:-}"
if [[ -z "$rapids_src" ]]; then
  # GPU scheduling is always on, so every row includes the rapids test
  # functions; a rapids-cmake checkout is required.
  echo "error: set RAPIDS_SRC to a rapids-cmake checkout" >&2
  exit 2
fi

rows=(base forcerdc nordc hostonly nolauncher clanghost tile clangcuda)

stage="$scratch/stage"
rm -rf "$stage"
mkdir -p "$stage" "$scratch/build"

stage_side() { # $1 = side name
  local side="$1"
  local dir="$stage/$side/cub/test"
  mkdir -p "$dir"
  cp -r "$repo/cub/test/." "$dir/"
  rm -f "$dir/CMakeLists.txt"
  cat > "$dir/CMakeLists.txt" <<WRAPPER
cmake_minimum_required(VERSION 3.28)
cmake_policy(SET CMP0077 NEW)
project(replay_${side} LANGUAGES NONE)
set(CCCL_SOURCE_DIR "$stage/$side")
set(CUB_SOURCE_DIR "$stage/$side/cub")
include("$here/shims.cmake")
include("\${CMAKE_CURRENT_SOURCE_DIR}/reg.cmake")
replay_dump()
WRAPPER
}

stage_side old
stage_side new
git -C "$repo" show "$old_ref:cub/test/CMakeLists.txt" \
  > "$stage/old/cub/test/reg.cmake"
# Registration helpers as of the old ref (absent there before the refactor):
if git -C "$repo" cat-file -e "$old_ref:cub/test/cmake/CubTestRegistration.cmake" 2> /dev/null; then
  git -C "$repo" show "$old_ref:cub/test/cmake/CubTestRegistration.cmake" \
    > "$stage/old/cub/test/cmake/CubTestRegistration.cmake"
else
  rm -f "$stage/old/cub/test/cmake/CubTestRegistration.cmake"
fi
cp "$repo/cub/test/CMakeLists.txt" "$stage/new/cub/test/reg.cmake"

fail=0
for row in "${rows[@]}"; do
  for side in old new; do
    b="$scratch/build/$side-$row"
    rm -rf "$b"
    if ! cmake -S "$stage/$side/cub/test" -B "$b" \
      -DROW="$row" -DREPO="$repo" -DRAPIDS_SRC="$rapids_src" \
      > "$b.configure.log" 2>&1; then
      echo "FAIL($row/$side): configure error — see $b.configure.log"
      tail -5 "$b.configure.log"
      fail=1
      continue 2
    fi
  done
  old_log="$scratch/build/old-$row/events.log"
  new_log="$scratch/build/new-$row/events.log"
  # Normalize the intentional path differences: staged side dir + build dir.
  sed -e "s|$stage/old|<SIDE>|g" -e "s|$scratch/build/old-$row|<BUILD>|g" \
    "$old_log" > "$old_log.norm"
  sed -e "s|$stage/new|<SIDE>|g" -e "s|$scratch/build/new-$row|<BUILD>|g" \
    "$new_log" > "$new_log.norm"
  if diff -u "$old_log.norm" "$new_log.norm" > "$scratch/build/diff-$row.txt"; then
    n=$(wc -l < "$new_log")
    echo "PASS($row): $n events identical"
  else
    echo "FAIL($row): event streams differ — $scratch/build/diff-$row.txt"
    head -20 "$scratch/build/diff-$row.txt"
    fail=1
  fi
done

if [[ "$fail" -eq 0 ]]; then
  echo "OK: registration parity holds vs $old_ref (${#rows[@]} rows)"
fi
exit "$fail"
