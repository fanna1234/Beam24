#!/usr/bin/env python3
"""Validate the complete ten-recording SPIB Local-F4 quality result."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


def validate(result: dict, anchors: dict) -> dict:
    rows = [row for row in result["details"] if row["variant"] == "Q2_localf4_joint_top2"]
    if len(rows) != 10 or len({row["file"] for row in rows}) != 10:
        raise ValueError("SPIB quality requires all ten distinct recordings")
    contract = result["contract"]
    if (contract["K"], contract["sample_rate"], contract["frequency_hz"]) != (48, 1000, [160.0, 180.0]):
        raise ValueError("SPIB quality contract changed")
    correlations = [float(row["map_correlation"]) for row in rows]
    shifts = [float(row["peak_shift_deg"]) for row in rows]
    if not all(math.isfinite(value) and -1 <= value <= 1 for value in correlations):
        raise ValueError("invalid spatial-spectrum correlation")
    if not all(math.isfinite(value) and 0 <= value <= anchors["spib48_vla_max_peak_shift_deg"] + 1e-10 for value in shifts):
        raise ValueError("SPIB peak-shift gate failed")
    mean = statistics.mean(correlations)
    saved = result["summary"]["Q2_localf4_joint_top2"]
    if saved["files"] != 10 or not math.isclose(mean, saved["mean_map_correlation"], abs_tol=1e-12):
        raise ValueError("summary does not match the ten recording-level observations")
    # The reference is rounded to five decimals; latency's 3% tolerance is not a quality gate.
    if mean < anchors["spib48_vla_mean_map_correlation"] - 1e-5:
        raise ValueError("SPIB map correlation did not reproduce the frozen reference")
    return {"recordings": 10, "mean_map_correlation": mean, "max_peak_shift_deg": max(shifts)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("anchors", type=Path)
    args = parser.parse_args()
    report = validate(json.loads(args.result.read_text()), json.loads(args.anchors.read_text())["quality"])
    print("[OK] SPIB48 Local-F4: " + json.dumps(report))


if __name__ == "__main__":
    main()
