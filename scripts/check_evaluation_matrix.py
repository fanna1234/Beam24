#!/usr/bin/env python3
"""Validate full external-baseline by dataset coverage."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "baselines" / "evaluation_matrix.json"
MANIFEST = ROOT / "baselines" / "manifest.json"
VALID_STATES = {"planned", "measured", "rejected_geometry", "not_applicable"}


def check_cells(label: str, rows: set[str], cells: dict[str, str]) -> None:
    if set(cells) != rows:
        raise SystemExit(f"{label}: row coverage mismatch")
    invalid = {row: state for row, state in cells.items() if state not in VALID_STATES}
    if invalid:
        raise SystemExit(f"{label}: invalid states {invalid}")


def main() -> None:
    matrix = json.loads(MATRIX.read_text())
    manifest = json.loads(MANIFEST.read_text())
    if matrix.get("version") != 3:
        raise SystemExit("unsupported evaluation matrix version")
    dataset_ids = {dataset["id"] for dataset in matrix["datasets"]}
    if set(matrix["quality_cells"]) != dataset_ids:
        raise SystemExit("quality matrix does not cover every dataset")

    rows = matrix["main_rows"]
    all_rows = set().union(*map(set, rows.values()))
    if set(matrix["row_bindings"]) != all_rows:
        raise SystemExit("external row bindings are incomplete")
    manifest_ids = {baseline["id"] for baseline in manifest["baselines"]}
    invalid_bindings = {}
    for row, binding in matrix["row_bindings"].items():
        if binding == "candidate":
            continue
        bound_ids = binding.removeprefix("best_of:").split(",")
        missing = [baseline_id for baseline_id in bound_ids if baseline_id not in manifest_ids]
        if missing:
            invalid_bindings[row] = missing
    if invalid_bindings:
        raise SystemExit(f"invalid row bindings: {invalid_bindings}")
    quality_rows = set(rows["quality"])
    for dataset, cells in matrix["quality_cells"].items():
        check_cells(f"quality/{dataset}", quality_rows, cells)

    profiles = {
        profile
        for dataset in matrix["datasets"]
        for profile in dataset["performance_profiles"]
    }
    if set(matrix["operator_profiles"]) != profiles:
        raise SystemExit("operator profiles do not cover every physical-K profile")
    if set(matrix["system_profiles"]) != profiles:
        raise SystemExit("system profiles do not cover every physical-K profile")
    for profile, cells in matrix["operator_profiles"].items():
        check_cells(f"operator/{profile}", set(rows["operator"]), cells)
    for profile, cells in matrix["system_profiles"].items():
        check_cells(f"system/{profile}", set(rows["system"]), cells)
    hybrid = matrix["hybrid_backend_ablation"]
    if set(hybrid["profiles"]) != profiles:
        raise SystemExit("hybrid backend ablation does not cover every profile")
    if any(state not in VALID_STATES for state in hybrid["profiles"].values()):
        raise SystemExit("hybrid backend ablation has invalid state")

    counts = {state: 0 for state in VALID_STATES}
    for cells in matrix["quality_cells"].values():
        for state in cells.values():
            counts[state] += 1
    for group in (matrix["operator_profiles"], matrix["system_profiles"]):
        for cells in group.values():
            for state in cells.values():
                counts[state] += 1
    print(f"evaluation matrix: datasets={len(dataset_ids)}, profiles={len(profiles)}, states={counts} [OK]")


if __name__ == "__main__":
    main()
