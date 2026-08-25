#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${BEAM24_BUILD_DIR:-$ROOT/build/sm120}"
LOCK="${BEAM24_GPU_LOCK:-/tmp/beam24_gpu_campaign.lock}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
command -v flock >/dev/null || { echo "flock is required" >&2; exit 7; }

cmake -S "$ROOT" -B "$BUILD_DIR" -DBEAM24_ENABLE_CUDA=ON \
  -DBEAM24_CUDA_ARCH=120a -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" -j "${BEAM24_BUILD_JOBS:-4}" --target \
  beam24_cufft_lower_bound beam24_cufft_fp16_rejected \
  beam24_hierarchical_exhaustive beam24_hierarchical_system

mkdir -p "$ROOT/artifact/runs"
RUN_DIR="$(mktemp -d "$ROOT/artifact/runs/fft_lower_bound.XXXXXX")"
mkdir -p "$RUN_DIR/logs"
FFT="$BUILD_DIR/beam24_cufft_lower_bound"
FFT16="$BUILD_DIR/beam24_cufft_fp16_rejected"
EXHAUSTIVE="$BUILD_DIR/beam24_hierarchical_exhaustive"
HIERARCHY="$BUILD_DIR/beam24_hierarchical_system"

{
  date --iso-8601=seconds
  echo "comparison_role=optimistic_uniform_spatial_frequency_fft_lower_bound"
  echo "interpolation_to_uniform_angle_grid=excluded"
  nvidia-smi --query-gpu=name,uuid,compute_cap,driver_version,power.limit --format=csv
  sha256sum "$FFT" "$FFT16" "$EXHAUSTIVE" "$HIERARCHY" \
    "$ROOT/baselines/cuda/cufft_top1_lower_bound.cu"
} > "$RUN_DIR/environment.txt"

exec 9>"$LOCK"
flock -x 9
"$FFT" 2 16 512 1024 1 1 1 > "$RUN_DIR/fp32_correctness.log" 2>&1
set +e
"$FFT16" 2 16 512 1024 1 1 1 > "$RUN_DIR/fp16_rejected.log" 2>&1
fp16_status=$?
set -e
if [[ "$fp16_status" != 3 ]]; then
  echo "expected FP16 numerical gate to return 3, got $fp16_status" >&2
  exit 8
fi
if command -v compute-sanitizer >/dev/null; then
  compute-sanitizer --tool memcheck --error-exitcode=99 \
    "$FFT" 2 16 512 1024 1 0 0 > "$RUN_DIR/fp32_memcheck.log" 2>&1
fi

orders=("F E H" "F H E" "E F H" "E H F" "H F E" "H E F")
for process in 0 1 2 3 4 5; do
  read -r first second third <<< "${orders[$process]}"
  for position in 0 1 2; do
    case "$position" in 0) variant="$first" ;; 1) variant="$second" ;; 2) variant="$third" ;; esac
    log="$RUN_DIR/logs/process_${process}_position_${position}_${variant}.log"
    case "$variant" in
      F)
        "$FFT" 256 1024 512 1024 100 20 0 > "$log" 2>&1
        expected=cufft_spatial_top1
        ;;
      E)
        "$EXHAUSTIVE" 100 20 0 256 1024 0 0 > "$log" 2>&1
        expected=beam24_exhaustive_e2e
        ;;
      H)
        "$HIERARCHY" 100 20 0 256 1024 0 0 > "$log" 2>&1
        expected=beam24_hierarchical_e2e
        ;;
    esac
    "$PYTHON_BIN" - "$process" "$position" "$variant" "$expected" "$log" <<'PY' >> "$RUN_DIR/samples.jsonl"
import json, sys
process, position = map(int, sys.argv[1:3])
variant, expected, path = sys.argv[3:]
records = [json.loads(line) for line in open(path) if f'"kind":"{expected}"' in line]
if len(records) != 1:
    raise SystemExit(f"expected one {expected} record in {path}, got {len(records)}")
record = records[0]
record.update(process=process, position=position, variant=variant, raw_log=path)
print(json.dumps(record, sort_keys=True))
PY
  done
done

"$PYTHON_BIN" - "$RUN_DIR/samples.jsonl" \
  "$ROOT/artifact/expected/reference_anchors.json" "$RUN_DIR/summary.json" <<'PY'
import json, statistics, sys
rows = [json.loads(line) for line in open(sys.argv[1])]
anchors = json.load(open(sys.argv[2]))["system"]
values = {v: [row["milliseconds"] for row in rows if row["variant"] == v] for v in "FEH"}
if any(len(value) != 6 for value in values.values()):
    raise SystemExit("incomplete six-process F/E/H campaign")
medians = {variant: statistics.median(value) for variant, value in values.items()}
reference = {
    "F": anchors["optimistic_cufft_fp32_ms"],
    "E": anchors["fft_campaign_beam24_exhaustive_ms"],
    "H": anchors["fft_campaign_beam24_hierarchy_ms"],
}
ratios = {variant: medians[variant] / reference[variant] for variant in "FEH"}
summary = {
    "comparison_role": "optimistic_uniform_spatial_frequency_fft_lower_bound",
    "interpolation_to_uniform_angle_grid": "excluded",
    "processes": 6,
    "medians_ms": {
        "cufft_fp32": medians["F"],
        "beam24_exhaustive": medians["E"],
        "beam24_hierarchy": medians["H"],
    },
    "reference_ms": {
        "cufft_fp32": reference["F"],
        "beam24_exhaustive": reference["E"],
        "beam24_hierarchy": reference["H"],
    },
    "within_3_percent": all(0.97 <= value <= 1.03 for value in ratios.values()),
    "fp16_status": "rejected_by_expected_numerical_gate",
}
summary["status"] = "accepted" if summary["within_3_percent"] else "LOW"
open(sys.argv[3], "w").write(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
if summary["status"] != "accepted":
    raise SystemExit(5)
PY
echo "$RUN_DIR"
