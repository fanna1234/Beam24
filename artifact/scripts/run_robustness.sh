#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/artifact/scripts/runtime_env.sh"
PYTHON_BIN="${PYTHON_BIN:-python3}"
mkdir -p "$ROOT/artifact/runs"
RUN_DIR="$(mktemp -d "$ROOT/artifact/runs/robustness.XXXXXX")"

"$PYTHON_BIN" "$ROOT/src/quality/regular_ula_robustness_heldout.py" \
  --output "$RUN_DIR/summary.json" \
  --csv "$RUN_DIR/per_seed.csv" \
  --raw-dir "$RUN_DIR"

"$PYTHON_BIN" - "$RUN_DIR/summary.json" "$ROOT/artifact/expected/reference_anchors.json" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1]))
anchors = json.load(open(sys.argv[2]))["quality"]
practical = [row for row in result["aggregates"] if row["practical"]]
local_top1 = min(row["local_top1_min"] for row in practical)
local_shift = max(row["local_max_shift_deg"] for row in practical)
hierarchy_top1 = min(row["hierarchy_top1_min"] for row in practical)
correlation_gap = min(
    row["local_minus_direct_correlation_min"] for row in practical
)

checks = (
    ("local top1 minimum", local_top1, anchors["regular_ula_robustness_local_top1_min"], 0.99),
    ("local maximum shift", local_shift, anchors["regular_ula_robustness_local_max_shift_deg"], None),
    ("hierarchy top1 minimum", hierarchy_top1, anchors["regular_ula_robustness_hierarchy_top1_min"], 0.999),
    ("local correlation advantage", correlation_gap, anchors["regular_ula_robustness_local_minus_direct_correlation_min"], 0.0),
)
for label, measured, reference, floor in checks:
    print(f"robustness {label}: measured={measured:.9f} reference={reference:.9f}")
    if floor is not None and measured < floor:
        raise SystemExit(f"[LOW] {label}: {measured} < {floor}")
if local_shift > 0.25:
    raise SystemExit(f"[LOW] local shift: {local_shift} > 0.25")
print(f"robustness: {sys.argv[1]} [OK]")
PY

MARGIN_DIR="$RUN_DIR/margin_coverage"
"$PYTHON_BIN" "$ROOT/src/quality/hierarchy_margin_coverage.py" \
  --raw-dir "$MARGIN_DIR"

"$PYTHON_BIN" - "$MARGIN_DIR/summary.json" \
  "$ROOT/artifact/expected/reference_anchors.json" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1]))
anchors = json.load(open(sys.argv[2]))["quality"]
checks = (
    ("top1 recall", result["admitted"]["top_l_recall"]["1"], anchors["hierarchy_margin_admitted_top1_recall"]),
    ("top2 recall", result["admitted"]["top_l_recall"]["2"], anchors["hierarchy_margin_admitted_top2_recall"]),
    ("top4 recall", result["admitted"]["top_l_recall"]["4"], anchors["hierarchy_margin_admitted_top4_recall"]),
    ("top8 recall", result["admitted"]["top_l_recall"]["8"], anchors["hierarchy_margin_admitted_top8_recall"]),
    ("admitted max rank", result["admitted"]["coverage_rank"]["max"], anchors["hierarchy_margin_admitted_max_rank"]),
)
for label, measured, reference in checks:
    print(f"margin {label}: measured={measured:.9f} reference={reference:.9f}")
    if measured != reference:
        raise SystemExit(f"[LOW] margin {label}: {measured} != {reference}")
stress = next(
    row for row in result["per_condition"]
    if row["condition"] == "stress_close_coherent_64"
)
if stress["top_l"]["8"]["recall"] != anchors["hierarchy_margin_stress_top8_recall"]:
    raise SystemExit("[LOW] close coherent top8 recall drift")
if stress["coverage_rank"]["max"] != anchors["hierarchy_margin_stress_max_rank"]:
    raise SystemExit("[LOW] close coherent max-rank drift")
print(f"margin coverage: {sys.argv[1]} [OK]")
PY
