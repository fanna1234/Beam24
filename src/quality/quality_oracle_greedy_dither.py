#!/usr/bin/env python3
"""Scalable greedy group-mask dither for local-DFT exact-2:4 beamforming."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import quality_oracle as base


LEGAL_MASKS = tuple(itertools.combinations(range(4), 2))


def source_sha256() -> str:
    payload = Path(__file__).read_bytes() + Path(base.__file__).read_bytes()
    return hashlib.sha256(payload).hexdigest()


def make_options(
    dense: np.ndarray,
    steering_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    groups = dense.reshape(-1, 4)
    num_groups = groups.shape[0]
    dft = base.unitary_dft4()
    coeff = groups @ dft.conj()
    block_options = np.zeros((num_groups, len(LEGAL_MASKS), 4), dtype=np.complex128)
    energy_ratio = np.zeros((num_groups, len(LEGAL_MASKS)), dtype=np.float64)
    response_options = np.zeros(
        (num_groups, len(LEGAL_MASKS), steering_grid.shape[1]), dtype=np.complex128
    )
    for group in range(num_groups):
        sensor_slice = slice(4 * group, 4 * group + 4)
        denominator = float(np.sum(np.abs(coeff[group]) ** 2))
        for option, pair in enumerate(LEGAL_MASKS):
            sparse = np.zeros(4, dtype=np.complex128)
            sparse[list(pair)] = coeff[group, list(pair)]
            block = sparse @ dft.T
            block_options[group, option] = block
            energy_ratio[group, option] = float(np.sum(np.abs(sparse) ** 2) / denominator)
            response_options[group, option] = np.conj(block) @ steering_grid[sensor_slice]
    top_option = int(np.argmax(np.mean(energy_ratio, axis=0)))
    return block_options, energy_ratio, response_options, top_option


def dense_sidelobe_context(
    dense: np.ndarray,
    look: np.ndarray,
    steering_grid: np.ndarray,
) -> tuple[np.ndarray, float]:
    dense = base.normalize_look_response(dense, look)
    pattern = np.abs(np.conj(dense) @ steering_grid)
    pattern /= np.max(pattern)
    peak = int(np.argmax(pattern))
    left, right = base.find_mainlobe_bounds(pattern, peak)
    sidelobe_mask = np.ones(pattern.size, dtype=bool)
    sidelobe_mask[left : right + 1] = False
    dense_psll = 20.0 * math.log10(float(np.max(pattern[sidelobe_mask])))
    return sidelobe_mask, dense_psll


def psll_db(response: np.ndarray, sidelobe_mask: np.ndarray) -> float:
    pattern = np.abs(response)
    pattern /= np.max(pattern)
    return float(20.0 * math.log10(float(np.max(pattern[sidelobe_mask]))))


def greedy_assignment(
    energy_ratio: np.ndarray,
    response_options: np.ndarray,
    top_option: int,
    sidelobe_mask: np.ndarray,
    max_substitutions: int,
    min_snr_ratio: float,
) -> tuple[np.ndarray, int, float, float, list[dict[str, object]]]:
    num_groups = energy_ratio.shape[0]
    group_index = np.arange(num_groups)
    assignment = np.full(num_groups, top_option, dtype=np.int8)
    changed: set[int] = set()
    response = np.sum(response_options[group_index, assignment], axis=0)
    current_psll = psll_db(response, sidelobe_mask)
    current_snr = float(np.mean(energy_ratio[group_index, assignment]))
    trace: list[dict[str, object]] = []

    for step in range(max_substitutions):
        best: tuple[float, float, int, int, np.ndarray] | None = None
        evaluated = 0
        admissible = 0
        for group in range(num_groups):
            if group in changed:
                continue
            old_option = int(assignment[group])
            for option in range(len(LEGAL_MASKS)):
                if option == old_option:
                    continue
                evaluated += 1
                snr = current_snr + (
                    energy_ratio[group, option] - energy_ratio[group, old_option]
                ) / num_groups
                if snr + 1e-12 < min_snr_ratio:
                    continue
                admissible += 1
                candidate_response = (
                    response
                    - response_options[group, old_option]
                    + response_options[group, option]
                )
                candidate_psll = psll_db(candidate_response, sidelobe_mask)
                candidate = (candidate_psll, -float(snr), group, option, candidate_response)
                if best is None or candidate[:4] < best[:4]:
                    best = candidate
        if best is None or best[0] >= current_psll - 1e-12:
            trace.append(
                {
                    "step": step + 1,
                    "evaluated": evaluated,
                    "admissible": admissible,
                    "accepted": False,
                    "reason": "no_admissible_improvement",
                }
            )
            break
        candidate_psll, negative_snr, group, option, candidate_response = best
        old_option = int(assignment[group])
        assignment[group] = option
        changed.add(group)
        response = candidate_response
        current_psll = float(candidate_psll)
        current_snr = float(-negative_snr)
        trace.append(
            {
                "step": step + 1,
                "evaluated": evaluated,
                "admissible": admissible,
                "accepted": True,
                "group": int(group),
                "old_option": old_option,
                "new_option": int(option),
                "new_mask": list(LEGAL_MASKS[option]),
                "snr_ratio": current_snr,
                "psll_db": current_psll,
            }
        )
    return assignment, len(changed), current_snr, current_psll, trace


def evaluate_case(
    num_sensors: int,
    look_angle: float,
    angle_grid: np.ndarray,
    steering_grid: np.ndarray,
    max_substitutions: int,
    min_snr_ratio: float,
) -> tuple[dict[str, object], dict[str, object]]:
    look = base.steering(num_sensors, np.asarray([look_angle]))[:, 0]
    dense = look / num_sensors
    sidelobe_mask, dense_psll = dense_sidelobe_context(dense, look, steering_grid)
    block_options, energy_ratio, response_options, top_option = make_options(
        dense, steering_grid
    )
    assignment, changed_groups, greedy_snr, greedy_psll, trace = greedy_assignment(
        energy_ratio,
        response_options,
        top_option,
        sidelobe_mask,
        max_substitutions,
        min_snr_ratio,
    )
    group_index = np.arange(num_sensors // 4)
    weight = block_options[group_index, assignment].reshape(-1)
    retained = float(np.mean(energy_ratio[group_index, assignment]))
    metrics = base.pattern_metrics(
        weight,
        dense,
        look,
        angle_grid,
        steering_grid,
        retained,
    )
    if not math.isclose(metrics.snr_ratio, greedy_snr, rel_tol=1e-10, abs_tol=1e-10):
        raise AssertionError("incremental and reconstructed SNR ratios disagree")
    if not math.isclose(metrics.psll_db, greedy_psll, rel_tol=1e-10, abs_tol=1e-10):
        raise AssertionError("incremental and reconstructed PSLL disagree")
    row: dict[str, object] = {
        "K": int(num_sensors),
        "look_angle_deg": float(look_angle),
        "candidate": "local_dft_greedy_dither2",
        "changed_groups": int(changed_groups),
        "top_option": int(top_option),
        "best_assignment": ";".join(map(str, assignment.tolist())),
        "best_masks": "|".join(
            ";".join(map(str, LEGAL_MASKS[int(option)])) for option in assignment
        ),
        **asdict(metrics),
    }
    detail = {
        "K": int(num_sensors),
        "look_angle_deg": float(look_angle),
        "dense_psll_db": float(dense_psll),
        "top_option": int(top_option),
        "assignment": assignment.tolist(),
        "masks": [list(LEGAL_MASKS[int(option)]) for option in assignment],
        "trace": trace,
        "metrics": asdict(metrics),
    }
    return row, detail


def aggregate(rows: list[dict[str, object]], key: str) -> dict[str, float]:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return {
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5.0)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95.0)),
        "max": float(np.max(values)),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if any(k % 4 != 0 for k in args.k):
        raise ValueError("all K values must be divisible by four")
    angle_grid = np.arange(
        args.response_min_deg,
        args.response_max_deg + 0.5 * args.response_step_deg,
        args.response_step_deg,
        dtype=np.float64,
    )
    look_grid = np.arange(
        args.look_min_deg,
        args.look_max_deg + 0.5 * args.look_step_deg,
        args.look_step_deg,
        dtype=np.float64,
    )
    min_snr_ratio = 10.0 ** (args.min_snr_loss_db / 10.0)
    rows: list[dict[str, object]] = []
    details: list[dict[str, object]] = []
    for num_sensors in args.k:
        steering_grid = base.steering(num_sensors, angle_grid)
        for look_angle in look_grid:
            row, detail = evaluate_case(
                num_sensors,
                float(look_angle),
                angle_grid,
                steering_grid,
                args.max_substitutions,
                min_snr_ratio,
            )
            rows.append(row)
            details.append(detail)

    aggregates = {
        metric: aggregate(rows, metric)
        for metric in (
            "snr_ratio",
            "snr_loss_db",
            "psll_delta_db",
            "hpbw_ratio",
            "magnitude_nmse",
            "retained_weight_energy",
            "changed_groups",
        )
    }
    worst_snr = min(float(row["snr_loss_db"]) for row in rows)
    mean_snr = float(np.mean([float(row["snr_loss_db"]) for row in rows]))
    max_psll_delta = max(float(row["psll_delta_db"]) for row in rows)
    stop_pass = worst_snr >= -1.0 and max_psll_delta <= 3.0
    promotion_pass = worst_snr >= -1.0 and mean_snr >= -0.5 and max_psll_delta <= 2.0
    summary: dict[str, object] = {
        "experiment_id": "beam24_20260822_localdft24_ula_tcbf/quality_ula_v4_greedy_dither2",
        "state": "partial",
        "input_scope": "synthetic ideal half-wavelength ULA steering vectors",
        "mechanism": "two-step greedy group-mask substitution under an exact joint-complex local-DFT 2:4 contract",
        "contract": {
            "K": args.k,
            "look_grid_deg": {
                "min": args.look_min_deg,
                "max": args.look_max_deg,
                "step": args.look_step_deg,
                "count": int(look_grid.size),
            },
            "response_grid_deg": {
                "min": args.response_min_deg,
                "max": args.response_max_deg,
                "step": args.response_step_deg,
                "count": int(angle_grid.size),
            },
            "max_substitutions": args.max_substitutions,
            "minimum_snr_loss_db": args.min_snr_loss_db,
            "quality_stop": {
                "worst_snr_loss_db": -1.0,
                "max_psll_delta_db": 3.0,
            },
            "promotion_target": {
                "mean_snr_loss_db": -0.5,
                "worst_snr_loss_db": -1.0,
                "max_psll_delta_db": 2.0,
            },
        },
        "environment": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "source_sha256": source_sha256(),
        },
        "aggregates": aggregates,
        "decision": {
            "quality_stop_pass": bool(stop_pass),
            "ideal_ula_promotion_target_pass": bool(promotion_pass),
            "mechanism_status": "advance_to_nonideal_quality_and_gpu_screen"
            if stop_pass
            else "rejected_on_full_ideal_ula_quality",
            "evidence_scope": "synthetic_only",
        },
        "details": details,
    }
    return {"summary": summary, "rows": rows}


def write_outputs(result: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "summary.json").write_text(
        json.dumps(result["summary"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = result["rows"]
    with (output_dir / "per_angle.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, nargs="+", default=[48, 64, 128, 512])
    parser.add_argument("--look-min-deg", type=float, default=-60.0)
    parser.add_argument("--look-max-deg", type=float, default=60.0)
    parser.add_argument("--look-step-deg", type=float, default=0.5)
    parser.add_argument("--response-min-deg", type=float, default=-90.0)
    parser.add_argument("--response-max-deg", type=float, default=90.0)
    parser.add_argument("--response-step-deg", type=float, default=0.025)
    parser.add_argument("--max-substitutions", type=int, default=2)
    parser.add_argument("--min-snr-loss-db", type=float, default=-1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run(args)
    write_outputs(result, args.output_dir)
    print(json.dumps(result["summary"]["decision"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
