#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${BEAM24_BUILD_DIR:-$ROOT/build/sm120}"
LOCK="${BEAM24_GPU_LOCK:-/tmp/beam24_gpu_campaign.lock}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
command -v flock >/dev/null || { echo "flock is required for the system campaign" >&2; exit 7; }

cmake -S "$ROOT" -B "$BUILD_DIR" -DBEAM24_ENABLE_CUDA=ON -DBEAM24_CUDA_ARCH=120a -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" -j "${BEAM24_BUILD_JOBS:-4}"

mkdir -p "$ROOT/artifact/runs"
RUN_DIR="$(mktemp -d "$ROOT/artifact/runs/system.XXXXXX")"
mkdir -p "$RUN_DIR/logs"
D1="$BUILD_DIR/beam24_dense_fused"
S2="$BUILD_DIR/beam24_system"

{
  date --iso-8601=seconds
  echo "comparison_role=internal_dense_fused_attribution_control"
  nvidia-smi --query-gpu=name,uuid,compute_cap,driver_version,power.limit,clocks.current.sm,clocks.current.memory --format=csv
  sha256sum "$D1" "$S2" "$ROOT/src/cuda/dense_fused_baseline.cu" "$ROOT/src/cuda/beam24_system.cu"
} > "$RUN_DIR/environment.txt"

exec 9>"$LOCK"
flock -x 9
orders=("D1 S2" "S2 D1" "D1 S2" "S2 D1" "D1 S2" "S2 D1")
for process in 0 1 2 3 4 5; do
  read -r first second <<< "${orders[$process]}"
  for position in 0 1; do
    if [[ "$position" == 0 ]]; then variant="$first"; else variant="$second"; fi
    log="$RUN_DIR/logs/process_${process}_position_${position}_${variant}.log"
    case "$variant" in
      D1)
        "$D1" 1024 1024 512 100 20 0 0.02 256 > "$log" 2>&1
        expected=dense_fused_power_top1_e2e
        ;;
      S2)
        "$S2" 1024 1024 512 100 20 0 0.02 256 0 > "$log" 2>&1
        expected=beam24_doa_e2e
        ;;
    esac
    "$PYTHON_BIN" - "$process" "$position" "$variant" "$expected" "$log" <<'PY' >> "$RUN_DIR/samples.jsonl"
import json, sys
process, position = map(int, sys.argv[1:3])
variant, expected, path = sys.argv[3:]
records = [json.loads(line) for line in open(path) if f'"kind":"{expected}"' in line]
if len(records) != 1:
    raise SystemExit(f"expected one {expected} record in {path}, got {len(records)}")
r = records[0]
r.update(process=process, position=position, variant=variant, raw_log=path)
print(json.dumps(r, sort_keys=True))
PY
  done
done

"$PYTHON_BIN" "$ROOT/scripts/summarize_system.py" \
  "$RUN_DIR/samples.jsonl" "$ROOT/artifact/expected/reference_anchors.json" \
  "$RUN_DIR/summary.json"

echo "$RUN_DIR"
