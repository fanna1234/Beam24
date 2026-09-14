#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/artifact/scripts/runtime_env.sh"
BUILD_DIR="${BEAM24_BUILD_DIR:-$ROOT/build/sm120}"
cmake -S "$ROOT" -B "$BUILD_DIR" -DBEAM24_ENABLE_CUDA=ON \
  -DBEAM24_CUDA_ARCH=120a -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" -j "${BEAM24_BUILD_JOBS:-4}"
