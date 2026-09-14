#!/usr/bin/env bash
# Shared interpreter and compiler selection; never changes the caller's shell.

if [[ -n "${BEAM24_PYTHON:-}" && -n "${PYTHON_BIN:-}" \
  && "$BEAM24_PYTHON" != "$PYTHON_BIN" ]]; then
  echo 'conflicting BEAM24_PYTHON and PYTHON_BIN settings' >&2
  return 2
fi
PYTHON_BIN="${BEAM24_PYTHON:-${PYTHON_BIN:-python3}}"
export PYTHON_BIN

beam24_nvcc="${CUDACXX:-${NVCC:-}}"
if [[ -z "$beam24_nvcc" && -n "${CUDA_HOME:-}" && -x "$CUDA_HOME/bin/nvcc" ]]; then
  beam24_nvcc="$CUDA_HOME/bin/nvcc"
fi
if [[ -n "$beam24_nvcc" ]]; then
  if [[ "$beam24_nvcc" != */* ]]; then
    beam24_nvcc="$(command -v "$beam24_nvcc" || true)"
  fi
  [[ -x "$beam24_nvcc" ]] || { echo 'configured CUDA compiler does not exist' >&2; return 3; }
  CUDACXX="$beam24_nvcc"
  NVCC="$beam24_nvcc"
  export CUDACXX NVCC
fi
