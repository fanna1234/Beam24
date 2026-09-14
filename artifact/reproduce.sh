#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/artifact/scripts/runtime_env.sh"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DRY_RUN=0
TARGET=""

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    smoke|evidence|doctor|build|gpu-smoke|quality|robustness|finite-gpu-quality|fourier-control|system|hierarchy|external-hierarchy|all)
      [[ -z "$TARGET" ]] || { echo 'select one reproduction target' >&2; exit 2; }
      TARGET="$arg" ;;
    help|-h|--help) TARGET=help ;;
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
  help)
    printf '%s\n' 'Usage: ./reproduce.sh TARGET [--dry-run]' \
      'CPU: smoke, evidence, quality, robustness' \
      'GPU: doctor, build, gpu-smoke, finite-gpu-quality' \
      'Compare: external-hierarchy, hierarchy, system, fourier-control' \
      'See docs/REPRODUCIBILITY.md for setup and all target scopes.'
    ;;
  evidence)
    run "$PYTHON_BIN" "$ROOT/scripts/check_reference_anchors.py"
    run "$PYTHON_BIN" "$ROOT/scripts/check_claim_framing.py"
    ;;
  doctor) run "$ROOT/artifact/scripts/check_env.sh" gpu ;;
  build) run bash "$ROOT/artifact/scripts/build.sh" ;;
  external-hierarchy)
    run "$ROOT/artifact/scripts/check_env.sh" gpu
    run bash "$ROOT/artifact/scripts/build.sh"
    run bash "$ROOT/artifact/scripts/build_ccglib.sh"
    run "$PYTHON_BIN" "$ROOT/scripts/run_external_hierarchy.py"
    ;;
  smoke) run_smoke ;;
  gpu-smoke) run_gpu_smoke ;;
  quality) run_quality ;;
  robustness) run_robustness ;;
  finite-gpu-quality) run_finite_gpu_quality ;;
  fourier-control) run_fourier_control ;;
  system) run_system ;;
  hierarchy) run_hierarchy ;;
  all)
    run_smoke; run_robustness; run_gpu_smoke; run_finite_gpu_quality
    run_fourier_control; run_quality
    run bash "$ROOT/artifact/scripts/build_ccglib.sh"
    run "$PYTHON_BIN" "$ROOT/scripts/run_external_hierarchy.py"
    run_system; run_hierarchy
    ;;
esac
