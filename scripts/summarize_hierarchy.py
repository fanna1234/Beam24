#!/usr/bin/env python3
"""Summarize the fixed exhaustive-versus-hierarchical Beam24 campaign."""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path


def main() -> None:
    samples_path, anchors_path, output_path = map(Path, sys.argv[1:4])
    records = [
        json.loads(line)
        for line in samples_path.read_text().splitlines()
        if line.strip()
    ]
    by_process: dict[int, dict[str, float]] = {}
    for record in records:
        by_process.setdefault(int(record["process"]), {})[
            str(record["variant"])
        ] = float(record["milliseconds"])
    if len(by_process) != 6 or any(
        set(row) != {"E", "H"} for row in by_process.values()
    ):
        raise SystemExit("campaign requires six complete E/H processes")
    ratios = [row["E"] / row["H"] for _, row in sorted(by_process.items())]
    speedup = math.exp(statistics.fmean(math.log(value) for value in ratios))
    anchors = json.loads(anchors_path.read_text())
    anchor = anchors["system"]["hierarchical_exhaustive_ablation_speedup"]
    within = anchors["gates"]["within_fraction"]
    if round(speedup, 5) >= anchor:
        label = "[OK >=reference]"
    elif speedup >= within * anchor:
        label = "[~within3%]"
    else:
        label = "[LOW]"
    result = {
        "comparison_role": "internal_exhaustive_beam24_ablation",
        "process_ratios": ratios,
        "process_wins": sum(value > 1.0 for value in ratios),
        "paired_geomean_speedup": speedup,
        "reference_anchor": anchor,
        "status": label,
        "exhaustive_median_ms": statistics.median(
            row["E"] for row in by_process.values()
        ),
        "hierarchy_median_ms": statistics.median(
            row["H"] for row in by_process.values()
        ),
    }
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"hierarchy speedup measured={speedup:.6f} "
        f"reference={anchor:.5f} {label}"
    )
    print(output_path)
    if speedup < within * anchor or result["process_wins"] < 6:
        raise SystemExit(8)


if __name__ == "__main__":
    main()
