#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DRY_RUN=0
TARGET=""

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    smoke|gpu-smoke|quality|robustness|finite-gpu-quality|fourier-control|system|hierarchy|all) TARGET="$arg" ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done
TARGET="${TARGET:-smoke}"

run() {
  if [[ "$DRY_RUN" == 1 ]]; then
    printf '[dry-run]'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

run_smoke() {
  run "$ROOT/artifact/scripts/check_env.sh" cpu
  run "$PYTHON_BIN" "$ROOT/scripts/check_reference_anchors.py"
  run "$PYTHON_BIN" "$ROOT/scripts/check_claim_framing.py"
  run "$PYTHON_BIN" -m unittest discover -s "$ROOT/tests" -v
  run "$ROOT/artifact/scripts/quality_smoke.sh"
}

run_gpu_smoke() {
  run "$ROOT/artifact/scripts/check_env.sh" gpu
  run "$ROOT/artifact/scripts/gpu_smoke.sh"
}

run_quality() {
  run "$ROOT/artifact/scripts/check_env.sh" quality
  run "$ROOT/artifact/scripts/run_quality.sh"
}

run_robustness() {
  run "$ROOT/artifact/scripts/check_env.sh" cpu
  run "$ROOT/artifact/scripts/run_robustness.sh"
}

run_finite_gpu_quality() {
  run "$ROOT/artifact/scripts/check_env.sh" gpu
  run "$ROOT/artifact/scripts/run_finite_gpu_quality.sh"
}

run_fourier_control() {
  run "$ROOT/artifact/scripts/check_env.sh" gpu
  run "$ROOT/artifact/scripts/run_fourier_control.sh"
}

run_system() {
  run "$ROOT/artifact/scripts/check_env.sh" gpu
  run "$ROOT/artifact/scripts/run_system.sh"
}

run_hierarchy() {
  run "$ROOT/artifact/scripts/check_env.sh" gpu
  run "$ROOT/artifact/scripts/run_hierarchy.sh"
}

case "$TARGET" in
  smoke) run_smoke ;;
  gpu-smoke) run_gpu_smoke ;;
  quality) run_quality ;;
  robustness) run_robustness ;;
  finite-gpu-quality) run_finite_gpu_quality ;;
  fourier-control) run_fourier_control ;;
  system) run_system ;;
  hierarchy) run_hierarchy ;;
  all) run_smoke; run_robustness; run_gpu_smoke; run_finite_gpu_quality; run_fourier_control; run_quality; run_system; run_hierarchy ;;
esac
