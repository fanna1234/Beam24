#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
mkdir -p "$ROOT/artifact/runs"
RUN_DIR="$(mktemp -d "$ROOT/artifact/runs/smoke.XXXXXX")"
OUT="$RUN_DIR/ideal_ula"

PYTHONPATH="$ROOT/src/quality${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" "$ROOT/src/quality/quality_oracle_greedy_dither.py" \
  --output-dir "$OUT" --k 48 \
  --look-min-deg -5 --look-max-deg 5 --look-step-deg 5 \
  --response-min-deg -90 --response-max-deg 90 --response-step-deg 0.5 \
  --max-substitutions 3 --min-snr-loss-db -1

"$PYTHON_BIN" - "$OUT/summary.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
if not r["decision"]["quality_stop_pass"]:
    raise SystemExit("quality smoke failed its declared stop rule")
print(f"quality smoke: {sys.argv[1]} [OK]")
PY
