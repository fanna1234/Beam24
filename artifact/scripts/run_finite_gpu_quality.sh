#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${BEAM24_BUILD_DIR:-$ROOT/build/sm120}"
LOCK="${BEAM24_GPU_LOCK:-/tmp/beam24_gpu_campaign.lock}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cmake -S "$ROOT" -B "$BUILD_DIR" -DBEAM24_ENABLE_CUDA=ON \
  -DBEAM24_CUDA_ARCH=120a -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" --target beam24_hierarchical_quality \
  -j "${BEAM24_BUILD_JOBS:-4}"

mkdir -p "$ROOT/artifact/runs"
RUN_DIR="$(mktemp -d "$ROOT/artifact/runs/finite-gpu-quality.XXXXXX")"
BINARY="$BUILD_DIR/beam24_hierarchical_quality"
conditions=(clean snr10 snr0 gain05 gain1 phase1 phase5 position001 position005 reflection10 combined)
seeds=(20260831 20260901 20260902)

{
  date --iso-8601=seconds
  nvidia-smi --query-gpu=name,uuid,compute_cap,driver_version,power.limit --format=csv
  sha256sum "$BINARY" "$ROOT/src/cuda/beam24_hierarchical_system.cu" \
    "$ROOT/src/cuda/beam24_hierarchical_quality.cu" \
    "$ROOT/src/cuda/beam24_system.cu"
} >"$RUN_DIR/environment.txt"

exec 9>"$LOCK"
flock -x 9
for condition_index in "${!conditions[@]}"; do
  condition="${conditions[$condition_index]}"
  for seed in "${seeds[@]}"; do
    for chunk in 0 1 2 3; do
      "$BINARY" 1 0 1 256 1024 0 0 "$condition_index" "$seed" \
        "$((chunk * 256))" 5 1 \
        >"$RUN_DIR/${condition}_seed${seed}_chunk${chunk}.jsonl" 2>&1
    done
  done
done

"$PYTHON_BIN" - "$RUN_DIR" "$ROOT/artifact/expected/reference_anchors.json" <<'PY'
import collections
import json
import pathlib
import sys

run_dir = pathlib.Path(sys.argv[1])
anchors = json.load(open(sys.argv[2]))["quality"]
records = []
for path in sorted(run_dir.glob("*_chunk*.jsonl")):
    for line in path.read_text().splitlines():
        if not line.startswith("{"):
            continue
        row = json.loads(line)
        if row.get("kind") == "beam24_finite_snapshot_quality":
            records.append(row)
if len(records) != 132:
    raise SystemExit(f"[LOW] expected 132 records, got {len(records)}")

groups = collections.defaultdict(list)
for row in records:
    groups[(row["condition"], row["seed"])].append(row)
for key, rows in groups.items():
    trials = sum(row["trials"] for row in rows)
    local = sum(row["local_dense_exact"] for row in rows) / trials
    hierarchy_local = sum(row["hierarchy_local_exact"] for row in rows) / trials
    hierarchy_dense = sum(row["hierarchy_dense_exact"] for row in rows) / trials
    within_one = sum(row["hierarchy_dense_within_one"] for row in rows) / trials
    if local < 0.99 or hierarchy_local < 0.999 or hierarchy_dense < 0.99:
        raise SystemExit(f"[LOW] exact gate {key}: {local=} {hierarchy_local=} {hierarchy_dense=}")
    if within_one != 1.0:
        raise SystemExit(f"[LOW] within-one gate {key}: {within_one}")
    if sum(row["replay_unstable"] + row["packed_a_mismatch"] +
           row["metadata_mismatch"] + row["nonfinite"] for row in rows):
        raise SystemExit(f"[LOW] structural gate {key}")

trials = sum(row["trials"] for row in records)
complete = sum(row["hierarchy_dense_exact"] for row in records) / trials
hierarchy_local = sum(row["hierarchy_local_exact"] for row in records) / trials
within_one = sum(row["hierarchy_dense_within_one"] for row in records) / trials
max_shift = max(row["hierarchy_dense_max_shift_grid"] for row in records)
replay_unstable = sum(row["replay_unstable"] for row in records)
checks = (
    ("complete/dense exact", complete, anchors["finite_gpu_complete_dense_exact_top1"]),
    ("hierarchy/local exact", hierarchy_local, anchors["finite_gpu_hierarchy_local_exact_top1"]),
    ("complete/dense within one", within_one, anchors["finite_gpu_complete_dense_within_one"]),
    ("maximum shift grid", max_shift, anchors["finite_gpu_max_shift_grid"]),
    ("replay unstable", replay_unstable, anchors["finite_gpu_replay_unstable"]),
)
for label, measured, reference in checks:
    print(f"finite GPU {label}: measured={measured} reference={reference}")
    if measured != reference:
        raise SystemExit(f"[LOW] {label}: {measured} != {reference}")
print(f"finite GPU quality: {trials} trials [OK]")
PY

echo "$RUN_DIR"
