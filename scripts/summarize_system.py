#!/usr/bin/env python3
"""Summarize the fixed six-process internal dense-fused attribution campaign."""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

from paired_records import validate_pairs


def main() -> None:
    samples_path, anchors_path, output_path = map(Path, sys.argv[1:4])
    records = [json.loads(line) for line in samples_path.read_text().splitlines() if line.strip()]
    validate_pairs(records, ("D1", "S2"))
    by_process: dict[int, dict[str, float]] = {}
    for record in records:
        by_process.setdefault(int(record["process"]), {})[record["variant"]] = float(record["milliseconds"])
    if len(by_process) != 6 or any(set(row) != {"D1", "S2"} for row in by_process.values()):
        raise SystemExit("campaign requires six complete D1/S2 processes")
    ratios = [row["D1"] / row["S2"] for _, row in sorted(by_process.items())]
    speedup = math.exp(statistics.fmean(math.log(value) for value in ratios))
    anchor = json.loads(anchors_path.read_text())["system"]["same_output_dense_fused_speedup"]
    within = json.loads(anchors_path.read_text())["gates"]["within_fraction"]
    label = "[OK >=reference]" if round(speedup, 5) >= anchor else "[~within3%]" if speedup >= within * anchor else "[LOW]"
    result = {
        "comparison_role": "internal_dense_fused_attribution_control",
        "process_ratios": ratios,
        "process_wins": sum(value > 1.0 for value in ratios),
        "paired_geomean_speedup": speedup,
        "reference_anchor": anchor,
        "status": label,
        "d1_median_ms": statistics.median(row["D1"] for row in by_process.values()),
        "s2_median_ms": statistics.median(row["S2"] for row in by_process.values()),
    }
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"internal attribution speedup measured={speedup:.6f} reference={anchor:.5f} {label}")
    print(output_path)
    if speedup < within * anchor or result["process_wins"] < 6:
        raise SystemExit(8)


if __name__ == "__main__":
    main()
