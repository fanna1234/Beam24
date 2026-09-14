#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/artifact/scripts/runtime_env.sh"

MODE="${1:-cpu}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
command -v "$PYTHON_BIN" >/dev/null
"$PYTHON_BIN" -c 'import sys; assert sys.version_info >= (3, 12), "Python 3.12+ is required by the pinned dependencies"'

if [[ "$MODE" == quality ]]; then
  "$PYTHON_BIN" -c 'import numpy, scipy, h5py, soundfile' || {
    echo "quality dependencies missing; run: python3 -m pip install -r requirements.txt" >&2
    exit 3
  }
  exit 0
fi

if [[ "$MODE" == cpu ]]; then
  "$PYTHON_BIN" -c 'import numpy, scipy' || {
    echo 'CPU dependencies missing; install requirements.txt in the selected Python environment' >&2
    exit 3
  }
  exit 0
fi

NVCC="${CUDACXX:-${NVCC:-$(command -v nvcc || true)}}"
if [[ -z "$NVCC" && -n "${CUDA_HOME:-}" && -x "$CUDA_HOME/bin/nvcc" ]]; then
  NVCC="$CUDA_HOME/bin/nvcc"
fi
[[ -n "$NVCC" ]] || { echo "nvcc not found; set CUDA_HOME" >&2; exit 3; }
version="$("$NVCC" --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | tail -1)"
case "$version" in
  13.0|13.1|13.2|13.3) ;;
  *) echo "unsupported CUDA $version; validated range is 13.0-13.3" >&2; exit 3 ;;
esac

command -v nvidia-smi >/dev/null
gpu_selection=()
if [[ -v CUDA_VISIBLE_DEVICES ]]; then
  [[ -n "$CUDA_VISIBLE_DEVICES" && "$CUDA_VISIBLE_DEVICES" != *,* ]] || {
    echo 'select one GPU with CUDA_VISIBLE_DEVICES, preferably its UUID' >&2; exit 3;
  }
  gpu_selection=(--id "$CUDA_VISIBLE_DEVICES")
fi
gpu_info="$(nvidia-smi "${gpu_selection[@]}" --query-gpu=name,compute_cap --format=csv,noheader)"
printf '%s\n' "$gpu_info"
printf '%s\n' "$gpu_info" | grep -q ', 12.0$' || {
  echo "Beam24 kernels require an SM120 GPU" >&2
  exit 3
}
