#!/usr/bin/env python3
"""Run the predeclared held-out Beam24 top-1 robustness gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


EXPERIMENT = "beam24_20260824_top1_robustness_heldout"
ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = Path(__file__).with_name("regular_ula_robustness.py")


def load_base():
    spec = importlib.util.spec_from_file_location("beam24_robustness_base", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_seed(result: dict[str, object]) -> list[str]:
    failures: list[str] = []
    for row in result["conditions"]:
        condition = row["condition"]
        if not condition["practical"]:
            continue
        local = row["methods"]["beam24_local_f4"]
        direct = row["methods"]["direct_joint_2of4"]
        hierarchy = row["hierarchy_vs_exhaustive_beam24"]
        prefix = str(condition["id"])
        if local["top1_agreement"] < 0.99:
            failures.append(f"{prefix}: local top1 < 0.99")
        if local["shift_deg"]["max"] > 0.25:
            failures.append(f"{prefix}: local max shift > 0.25 deg")
        if hierarchy["top1_agreement"] < 0.999:
            failures.append(f"{prefix}: hierarchy top1 < 0.999")
        if hierarchy["shift_deg"]["max"] > 0.25:
            failures.append(f"{prefix}: hierarchy max shift > 0.25 deg")
        if (
            local["spectrum_correlation"]["mean"]
            <= direct["spectrum_correlation"]["mean"]
        ):
            failures.append(f"{prefix}: local correlation not above direct")
    return failures


def aggregate(results: list[dict[str, object]]) -> list[dict[str, object]]:
    condition_ids = [row["condition"]["id"] for row in results[0]["conditions"]]
    rows: list[dict[str, object]] = []
    for condition_id in condition_ids:
        matches = [
            next(
                row for row in result["conditions"]
                if row["condition"]["id"] == condition_id
            )
            for result in results
        ]
        local_top1 = np.asarray(
            [row["methods"]["beam24_local_f4"]["top1_agreement"] for row in matches]
        )
        hierarchy_top1 = np.asarray(
            [row["hierarchy_vs_exhaustive_beam24"]["top1_agreement"] for row in matches]
        )
        local_max_shift = np.asarray(
            [row["methods"]["beam24_local_f4"]["shift_deg"]["max"] for row in matches]
        )
        hierarchy_max_shift = np.asarray(
            [row["hierarchy_vs_exhaustive_beam24"]["shift_deg"]["max"] for row in matches]
        )
        correlation_gap = np.asarray(
            [
                row["methods"]["beam24_local_f4"]["spectrum_correlation"]["mean"]
                - row["methods"]["direct_joint_2of4"]["spectrum_correlation"]["mean"]
                for row in matches
            ]
        )
        rows.append(
            {
                "condition": condition_id,
                "practical": bool(matches[0]["condition"]["practical"]),
                "local_top1_min": float(np.min(local_top1)),
                "local_top1_mean": float(np.mean(local_top1)),
                "local_max_shift_deg": float(np.max(local_max_shift)),
                "hierarchy_top1_min": float(np.min(hierarchy_top1)),
                "hierarchy_top1_mean": float(np.mean(hierarchy_top1)),
                "hierarchy_max_shift_deg": float(np.max(hierarchy_max_shift)),
                "local_minus_direct_correlation_min": float(np.min(correlation_gap)),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=1024)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[20260825, 20260826, 20260827]
    )
    args = parser.parse_args()
    if args.seeds != [20260825, 20260826, 20260827]:
        raise ValueError("the frozen held-out seeds are 20260825/26/27")
    if args.trials != 1024:
        raise ValueError("the frozen held-out contract uses 1024 trials per seed")

    base = load_base()
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    failures: list[str] = []
    for seed in args.seeds:
        base_args = argparse.Namespace(
            trials=args.trials,
            seed=seed,
            sensors=512,
            beams=1024,
            coarse_beams=128,
        )
        result = base.run(base_args)
        seed_failures = evaluate_seed(result)
        failures.extend(f"seed {seed}: {failure}" for failure in seed_failures)
        seed_path = args.raw_dir / f"seed_{seed}.json"
        seed_path.write_text(json.dumps(result, indent=2) + "\n")
        results.append(result)

    aggregates = aggregate(results)
    decision = {
        "top1_practical_envelope_gate": "accepted" if not failures else "rejected",
        "gate_failures": failures,
        "implementation": "accepted held-out CPU quality oracle",
        "mechanism": (
            "bounded synthetic regular-array top-1 robustness supported"
            if not failures
            else "bounded top-1 robustness not supported by the frozen gate"
        ),
        "thesis_impact": (
            "may add bounded synthetic top-1 perturbation evidence"
            if not failures
            else "retain the prior clean and real-data scope"
        ),
        "allowed_wording": (
            "Across the declared synthetic regular-array perturbation envelope, Beam24 retains at least 99% dense top-1 agreement with at most 0.25-degree shift, while the hierarchy retains at least 99.9% agreement with exhaustive Beam24."
            if not failures
            else "No positive perturbation-robustness claim is allowed."
        ),
        "boundary": "synthetic continuous-angle K512 ULA top-1; not full-spectrum, real calibration, top-k, or general-array evidence",
    }
    summary = {
        "experiment": EXPERIMENT,
        "state": "measured",
        "contract": {
            "heldout_seeds": args.seeds,
            "trials_per_condition_per_seed": args.trials,
            "total_trials_per_condition": args.trials * len(args.seeds),
            "base_experiment": "beam24_20260824_regular_array_robustness_top1",
            "gate_application": "per seed and practical condition",
        },
        "environment": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "wrapper_source_sha256": sha256(Path(__file__)),
            "base_source_sha256": sha256(BASE_PATH),
        },
        "aggregates": aggregates,
        "decision": decision,
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    with args.csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)
    print(json.dumps(decision, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
