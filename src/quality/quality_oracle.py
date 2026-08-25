#!/usr/bin/env python3
"""Quality oracle for local-DFT-aligned complex 2:4 ULA beamforming."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cusparselt_prune import prune_complex


@dataclass(frozen=True)
class PatternMetrics:
    snr_ratio: float
    snr_loss_db: float
    psll_db: float
    psll_delta_db: float
    hpbw_deg: float
    hpbw_ratio: float
    magnitude_nmse: float
    retained_weight_energy: float


def steering(num_sensors: int, angles_deg: np.ndarray) -> np.ndarray:
    """Return K x A half-wavelength ULA steering vectors."""
    sensor = np.arange(num_sensors, dtype=np.float64)[:, None]
    phase = np.pi * np.sin(np.deg2rad(angles_deg))[None, :]
    return np.exp(1j * sensor * phase)


def unitary_dft4() -> np.ndarray:
    row = np.arange(4, dtype=np.float64)[:, None]
    col = np.arange(4, dtype=np.float64)[None, :]
    return np.exp(-2j * np.pi * row * col / 4.0) / 2.0


def direct_top2(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    groups = weight.reshape(-1, 4)
    # Stable sorting makes tied ULA magnitudes deterministic.
    keep = np.argsort(-np.abs(groups), axis=1, kind="stable")[:, :2]
    mask = np.zeros_like(groups, dtype=bool)
    np.put_along_axis(mask, keep, True, axis=1)
    sparse = np.where(mask, groups, 0.0)
    return sparse.reshape(-1), mask


def local_dft_top2(
    weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups = weight.reshape(-1, 4)
    dft = unitary_dft4()
    coeff = groups @ dft.conj()
    keep = np.argsort(-np.abs(coeff), axis=1, kind="stable")[:, :2]
    mask = np.zeros_like(coeff, dtype=bool)
    np.put_along_axis(mask, keep, True, axis=1)
    sparse_coeff = np.where(mask, coeff, 0.0)
    projected = sparse_coeff @ dft.T
    return projected.reshape(-1), sparse_coeff, mask


def normalize_look_response(weight: np.ndarray, look: np.ndarray) -> np.ndarray:
    response = np.vdot(weight, look)
    if abs(response) < 1e-14:
        raise ValueError("candidate has zero look-direction response")
    # np.vdot(c * w, a) = conj(c) * np.vdot(w, a).
    return weight / np.conj(response)


def find_mainlobe_bounds(pattern: np.ndarray, peak_index: int) -> tuple[int, int]:
    """Use nearest local minima around the dense peak as first-null bounds."""
    left = peak_index
    while left > 1:
        if pattern[left - 1] <= pattern[left - 2] and pattern[left - 1] <= pattern[left]:
            left -= 1
            break
        left -= 1
    right = peak_index
    last = pattern.size - 1
    while right < last - 1:
        if pattern[right + 1] <= pattern[right] and pattern[right + 1] <= pattern[right + 2]:
            right += 1
            break
        right += 1
    return left, right


def interpolate_crossing(
    x0: float, y0: float, x1: float, y1: float, threshold: float
) -> float:
    if y1 == y0:
        return 0.5 * (x0 + x1)
    fraction = (threshold - y0) / (y1 - y0)
    return x0 + fraction * (x1 - x0)


def hpbw(angles: np.ndarray, pattern: np.ndarray, peak_index: int) -> float:
    threshold = 1.0 / math.sqrt(2.0)
    left = peak_index
    while left > 0 and pattern[left] >= threshold:
        left -= 1
    right = peak_index
    last = pattern.size - 1
    while right < last and pattern[right] >= threshold:
        right += 1
    if left == 0 or right == last:
        return float("nan")
    left_cross = interpolate_crossing(
        angles[left], pattern[left], angles[left + 1], pattern[left + 1], threshold
    )
    right_cross = interpolate_crossing(
        angles[right - 1],
        pattern[right - 1],
        angles[right],
        pattern[right],
        threshold,
    )
    return float(right_cross - left_cross)


def pattern_metrics(
    candidate: np.ndarray,
    dense: np.ndarray,
    look: np.ndarray,
    angle_grid: np.ndarray,
    steering_grid: np.ndarray,
    retained_energy: float,
) -> PatternMetrics:
    num_sensors = dense.size
    dense = normalize_look_response(dense, look)
    candidate = normalize_look_response(candidate, look)

    dense_response = np.conj(dense) @ steering_grid
    candidate_response = np.conj(candidate) @ steering_grid
    dense_pattern = np.abs(dense_response)
    candidate_pattern = np.abs(candidate_response)
    dense_pattern /= np.max(dense_pattern)
    candidate_pattern /= np.max(candidate_pattern)

    peak_index = int(np.argmax(dense_pattern))
    left, right = find_mainlobe_bounds(dense_pattern, peak_index)
    sidelobe_mask = np.ones_like(dense_pattern, dtype=bool)
    sidelobe_mask[left : right + 1] = False

    floor = np.finfo(np.float64).tiny
    dense_psll = 20.0 * math.log10(max(float(np.max(dense_pattern[sidelobe_mask])), floor))
    candidate_psll = 20.0 * math.log10(
        max(float(np.max(candidate_pattern[sidelobe_mask])), floor)
    )
    dense_hpbw = hpbw(angle_grid, dense_pattern, peak_index)
    candidate_hpbw = hpbw(angle_grid, candidate_pattern, int(np.argmax(candidate_pattern)))

    dense_snr = abs(np.vdot(dense, look)) ** 2 / float(np.vdot(dense, dense).real)
    candidate_snr = abs(np.vdot(candidate, look)) ** 2 / float(
        np.vdot(candidate, candidate).real
    )
    snr_ratio = float(candidate_snr / dense_snr)

    error = candidate_pattern - dense_pattern
    denominator = float(np.dot(dense_pattern, dense_pattern))
    magnitude_nmse = float(np.dot(error, error) / denominator)

    return PatternMetrics(
        snr_ratio=snr_ratio,
        snr_loss_db=float(10.0 * math.log10(snr_ratio)),
        psll_db=float(candidate_psll),
        psll_delta_db=float(candidate_psll - dense_psll),
        hpbw_deg=float(candidate_hpbw),
        hpbw_ratio=float(candidate_hpbw / dense_hpbw),
        magnitude_nmse=magnitude_nmse,
        retained_weight_energy=float(retained_energy),
    )


def validate_construction(num_sensors: int, look_angle: float) -> None:
    look = steering(num_sensors, np.asarray([look_angle]))[:, 0]
    dense = look / num_sensors
    projected, sparse_coeff, mask = local_dft_top2(dense)
    dft = unitary_dft4()
    dense_groups = dense.reshape(-1, 4)
    dense_coeff = dense_groups @ dft.conj()

    if not np.all(mask.sum(axis=1) == 2):
        raise AssertionError("local DFT mask is not exact 2:4")
    if not np.allclose(sparse_coeff[~mask], 0.0, atol=0.0, rtol=0.0):
        raise AssertionError("masked local DFT entries are not exactly zero")
    if not np.allclose(dense_groups, dense_coeff @ dft.T, atol=1e-12, rtol=1e-12):
        raise AssertionError("unitary DFT round trip failed")
    retained = float(np.vdot(projected, projected).real / np.vdot(dense, dense).real)
    coeff_retained = float(
        np.vdot(sparse_coeff, sparse_coeff).real / np.vdot(dense_coeff, dense_coeff).real
    )
    if not math.isclose(retained, coeff_retained, rel_tol=1e-12, abs_tol=1e-12):
        raise AssertionError("Parseval energy check failed")

    direct, direct_mask = direct_top2(dense)
    if not np.all(direct_mask.sum(axis=1) == 2):
        raise AssertionError("direct mask is not exact 2:4")
    direct_retained = float(np.vdot(direct, direct).real / np.vdot(dense, dense).real)
    if not math.isclose(direct_retained, 0.5, rel_tol=1e-12, abs_tol=1e-12):
        raise AssertionError("direct ULA 2:4 should retain exactly half the weight energy")


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q))


def aggregate(rows: list[dict[str, object]], candidate: str, key: str) -> dict[str, float]:
    values = np.asarray(
        [float(row[key]) for row in rows if row["candidate"] == candidate],
        dtype=np.float64,
    )
    return {
        "min": float(np.min(values)),
        "p05": percentile(values, 5.0),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": percentile(values, 95.0),
        "max": float(np.max(values)),
    }


def source_sha256() -> str:
    payload = Path(__file__).read_bytes()
    return hashlib.sha256(payload).hexdigest()


def run(args: argparse.Namespace) -> dict[str, object]:
    if any(k % 4 != 0 for k in args.k):
        raise ValueError("all K values must be divisible by four")
    response_grid = np.arange(
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

    rows: list[dict[str, object]] = []
    for num_sensors in args.k:
        validate_construction(num_sensors, float(look_grid[look_grid.size // 2]))
        steering_response = steering(num_sensors, response_grid)
        for look_angle in look_grid:
            look = steering(num_sensors, np.asarray([look_angle]))[:, 0]
            dense = look / num_sensors

            direct, _ = direct_top2(dense)
            direct_retained = float(
                np.vdot(direct, direct).real / np.vdot(dense, dense).real
            )
            external_strip = prune_complex(dense[None, :], "strip")[0]
            external_tile = prune_complex(dense[None, :], "tile")[0]
            external_strip_retained = float(
                np.vdot(external_strip, external_strip).real / np.vdot(dense, dense).real
            )
            external_tile_retained = float(
                np.vdot(external_tile, external_tile).real / np.vdot(dense, dense).real
            )
            projected, sparse_coeff, _ = local_dft_top2(dense)
            dense_coeff = dense.reshape(-1, 4) @ unitary_dft4().conj()
            local_retained = float(
                np.vdot(sparse_coeff, sparse_coeff).real
                / np.vdot(dense_coeff, dense_coeff).real
            )

            for name, candidate, retained in (
                ("direct_top2", direct, direct_retained),
                ("cusparselt_prune_strip", external_strip, external_strip_retained),
                ("cusparselt_prune_tile", external_tile, external_tile_retained),
                ("local_dft_top2", projected, local_retained),
            ):
                metrics = pattern_metrics(
                    candidate,
                    dense,
                    look,
                    response_grid,
                    steering_response,
                    retained,
                )
                rows.append(
                    {
                        "K": int(num_sensors),
                        "look_angle_deg": float(look_angle),
                        "candidate": name,
                        **asdict(metrics),
                    }
                )

    summary: dict[str, object] = {
        "experiment_id": "beam24_20260822_localdft24_ula_tcbf/quality_ula_v1",
        "state": "partial",
        "input_scope": "synthetic ideal half-wavelength ULA steering vectors",
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
                "count": int(response_grid.size),
            },
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
        "aggregates": {},
        "decision": {},
    }

    for candidate in (
        "direct_top2",
        "cusparselt_prune_strip",
        "cusparselt_prune_tile",
        "local_dft_top2",
    ):
        summary["aggregates"][candidate] = {
            metric: aggregate(rows, candidate, metric)
            for metric in (
                "snr_ratio",
                "snr_loss_db",
                "psll_delta_db",
                "hpbw_ratio",
                "magnitude_nmse",
                "retained_weight_energy",
            )
        }

    local_rows = [row for row in rows if row["candidate"] == "local_dft_top2"]
    worst_snr = min(float(row["snr_loss_db"]) for row in local_rows)
    max_psll_delta = max(float(row["psll_delta_db"]) for row in local_rows)
    quality_stop_pass = worst_snr >= -1.0 and max_psll_delta <= 3.0
    promotion_pass = (
        float(np.mean([float(row["snr_loss_db"]) for row in local_rows])) >= -0.5
        and worst_snr >= -1.0
        and max_psll_delta <= 2.0
    )
    summary["decision"] = {
        "quality_stop_pass": bool(quality_stop_pass),
        "ideal_ula_promotion_target_pass": bool(promotion_pass),
        "mechanism_status": "advance_to_nonideal_quality_and_gpu_screen"
        if quality_stop_pass
        else "rejected_on_ideal_ula_quality",
        "evidence_scope": "synthetic_only",
    }

    return {"summary": summary, "rows": rows}


def write_outputs(result: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    summary_path = output_dir / "summary.json"
    rows_path = output_dir / "per_angle.csv"
    summary_path.write_text(
        json.dumps(result["summary"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = result["rows"]
    if not rows:
        raise RuntimeError("no result rows")
    with rows_path.open("w", encoding="utf-8", newline="") as handle:
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run(args)
    write_outputs(result, args.output_dir)
    print(json.dumps(result["summary"]["decision"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
