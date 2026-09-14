#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/artifact/scripts/runtime_env.sh"
DATA_ROOT="${BEAM24_DATA_ROOT:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
[[ -n "$DATA_ROOT" ]] || { echo "set BEAM24_DATA_ROOT" >&2; exit 4; }
BEAM24_DATA_ROOT="$DATA_ROOT" "$ROOT/artifact/scripts/get_data.sh" spib
mkdir -p "$ROOT/artifact/runs"
RUN_DIR="$(mktemp -d "$ROOT/artifact/runs/quality.XXXXXX")"

spib_dir="$DATA_ROOT/spib/A2601_1_10"
sensor_file="$DATA_ROOT/spib/SACLANT_sens.dat"
[[ -d "$spib_dir" && -f "$sensor_file" ]] || {
  echo "SPIB data missing; run artifact/scripts/get_data.sh spib" >&2
  exit 4
}

"$PYTHON_BIN" "$ROOT/src/quality/spib_vla_screen.py" \
  --data-dir "$spib_dir" --sensors "$sensor_file" \
  --output "$RUN_DIR/spib48.json"

"$PYTHON_BIN" "$ROOT/scripts/check_real_quality.py" "$RUN_DIR/spib48.json" \
  "$ROOT/artifact/expected/reference_anchors.json"
echo "$RUN_DIR"
