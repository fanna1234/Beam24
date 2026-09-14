#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/artifact/scripts/runtime_env.sh"
sources="${BEAM24_BASELINE_ROOT:-$ROOT/work/baselines}"
BUILD_DIR="${BEAM24_EXTERNAL_BUILD_DIR:-$ROOT/build/external}"
"$PYTHON_BIN" "$ROOT/artifact/scripts/get_baselines.py" ccglib --root "$sources"
cmake -S "$ROOT" -B "$BUILD_DIR" -DBEAM24_ENABLE_CUDA=ON \
  -DBEAM24_CUDA_ARCH=120a -DCMAKE_BUILD_TYPE=Release \
  -DBEAM24_ENABLE_CCGLIB=ON -DBEAM24_BASELINE_SOURCE_ROOT="$sources"
cmake --build "$BUILD_DIR" --target beam24_ccglib_hierarchy \
  beam24_ccglib_materialized_top1 -j "${BEAM24_BUILD_JOBS:-4}"
