#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/artifact/scripts/runtime_env.sh"
BUILD_DIR="${BEAM24_BUILD_DIR:-$ROOT/build/sm120}"
cmake -S "$ROOT" -B "$BUILD_DIR" -DBEAM24_ENABLE_CUDA=ON -DBEAM24_CUDA_ARCH=120a -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" -j "${BEAM24_BUILD_JOBS:-4}"

exec 9>"${BEAM24_GPU_LOCK:-/tmp/beam24_gpu_campaign.lock}"
flock -n 9 || { echo 'GPU campaign lock is busy' >&2; exit 7; }
"$PYTHON_BIN" "$ROOT/scripts/check_gpu_idle.py"
"$BUILD_DIR/beam24_operator" 128 128 64 1 1 1 0.02 2
"$BUILD_DIR/beam24_dense_fused" 128 128 64 1 1 1 0.02 2
"$BUILD_DIR/beam24_system" 128 128 64 1 1 1 0.02 2 0
"$BUILD_DIR/beam24_hierarchical_system" 1 0 1 24 128 1 0

if command -v compute-sanitizer >/dev/null; then
  compute-sanitizer --tool memcheck --error-exitcode=99 \
    "$BUILD_DIR/beam24_system" 128 128 64 1 0 0 0.02 2 0
  compute-sanitizer --tool memcheck --error-exitcode=99 \
    "$BUILD_DIR/beam24_hierarchical_system" 1 0 1 24 128 1 0
fi
