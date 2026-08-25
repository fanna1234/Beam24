#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${BEAM24_DATA_ROOT:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
[[ -n "$DATA_ROOT" ]] || { echo "set BEAM24_DATA_ROOT" >&2; exit 4; }
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

"$PYTHON_BIN" - "$RUN_DIR/spib48.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))["summary"]["Q2_localf4_joint_top2"]
anchor = 0.99998
value = r["mean_map_correlation"]
label = "[OK >=reference]" if round(value, 5) >= anchor else "[~within3%]" if value >= 0.97 * anchor else "[LOW]"
print(f"SPIB48 map correlation measured={value:.6f} reference={anchor:.5f} {label}")
if value < 0.97 * anchor:
    raise SystemExit(5)
PY
