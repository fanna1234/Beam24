#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${BEAM24_BUILD_DIR:-$ROOT/build/sm120}"
LOCK="${BEAM24_GPU_LOCK:-/tmp/beam24_gpu_campaign.lock}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
command -v flock >/dev/null || { echo "flock is required" >&2; exit 7; }

assert_gpu_idle() {
  local activity
  activity="$(nvidia-smi pmon -c 1 | awk '
    $2 ~ /^[0-9]+$/ && $10 != "-" && $10 !~ /^gnome/ { print }
  ')"
  if [[ -n "$activity" ]]; then
    echo "GPU has active external processes; Fourier timing aborted:" >&2
    printf '%s\n' "$activity" >&2
    return 9
  fi
}

cmake -S "$ROOT" -B "$BUILD_DIR" -DBEAM24_ENABLE_CUDA=ON \
  -DBEAM24_CUDA_ARCH=120a -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" -j "${BEAM24_BUILD_JOBS:-4}" --target \
  beam24_cufft_streamed_top1 beam24_hierarchical_system

mkdir -p "$ROOT/artifact/runs"
RUN_DIR="$(mktemp -d "$ROOT/artifact/runs/fourier_control.XXXXXX")"
mkdir -p "$RUN_DIR/logs"
FOURIER="$BUILD_DIR/beam24_cufft_streamed_top1"
BEAM24="$BUILD_DIR/beam24_hierarchical_system"

{
  date --iso-8601=seconds
  echo "comparison_role=streamed_uniform_angle_fourier_top1"
  echo "full_spectrum_materialization=false"
  nvidia-smi --query-gpu=name,uuid,compute_cap,driver_version,power.limit --format=csv
  sha256sum "$FOURIER" "$BEAM24" \
    "$ROOT/baselines/cuda/cufft_streamed_uniform_angle_top1.cu"
} > "$RUN_DIR/environment.txt"

exec 9>"$LOCK"
flock -x 9
assert_gpu_idle
"$FOURIER" 1 16 512 1024 4096 1 1 1 > "$RUN_DIR/correctness.log" 2>&1
"$PYTHON_BIN" - "$RUN_DIR/correctness.log" <<'PY'
import json
import sys

records = [json.loads(line) for line in open(sys.argv[1]) if '"kind":"streamed_cufft_uniform_angle_top1"' in line]
if len(records) != 1 or records[0]["first_top1"] != 511:
    raise SystemExit("streamed Fourier correctness smoke failed")
PY
if command -v compute-sanitizer >/dev/null; then
  compute-sanitizer --tool memcheck --error-exitcode=99 \
    "$FOURIER" 1 16 512 1024 4096 1 1 0 > "$RUN_DIR/memcheck.log" 2>&1
fi

orders=("F H" "H F" "F H" "H F" "F H" "H F")
for process in 0 1 2 3 4 5; do
  assert_gpu_idle
  read -r first second <<< "${orders[$process]}"
  for position in 0 1; do
    if [[ "$position" == 0 ]]; then variant="$first"; else variant="$second"; fi
    log="$RUN_DIR/logs/process_${process}_position_${position}_${variant}.log"
    case "$variant" in
      F)
        "$FOURIER" 256 1024 512 1024 4096 2 100 20 > "$log" 2>&1
        expected=streamed_cufft_uniform_angle_top1
        ;;
      H)
        "$BEAM24" 100 20 0 256 1024 0 0 > "$log" 2>&1
        expected=beam24_hierarchical_e2e
        ;;
    esac
    "$PYTHON_BIN" - "$process" "$position" "$variant" "$expected" "$log" <<'PY' >> "$RUN_DIR/samples.jsonl"
import json
import sys

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
import json
import math
import statistics
import sys

rows = [json.loads(line) for line in open(sys.argv[1])]
anchors = json.load(open(sys.argv[2]))["system"]
by_process = {}
for row in rows:
    by_process.setdefault(row["process"], {})[row["variant"]] = row["milliseconds"]
ratios = [value["F"] / value["H"] for _, value in sorted(by_process.items())]
if len(ratios) != 6:
    raise SystemExit("incomplete streamed Fourier campaign")
paired = math.exp(sum(math.log(value) for value in ratios) / len(ratios))
reference = anchors["streamed_cufft_over_beam24"]
summary = {
    "comparison_role": "streamed_uniform_angle_fourier_top1",
    "processes": 6,
    "medians_ms": {
        "streamed_cufft": statistics.median(row["milliseconds"] for row in rows if row["variant"] == "F"),
        "beam24": statistics.median(row["milliseconds"] for row in rows if row["variant"] == "H"),
    },
    "paired_geomean_cufft_over_beam24": paired,
    "reference_ratio": reference,
    "beam24_wins": sum(value > 1.0 for value in ratios),
    "within_3_percent": 0.97 <= paired / reference <= 1.03,
}
summary["status"] = "accepted" if summary["beam24_wins"] == 6 and summary["within_3_percent"] else "LOW"
open(sys.argv[3], "w").write(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
if summary["status"] != "accepted":
    raise SystemExit(5)
PY

printf '%s\n' "$RUN_DIR"
