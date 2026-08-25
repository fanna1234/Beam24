#!/usr/bin/env python3
"""Evaluate Beam24 under bounded regular-array deployment perturbations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


EXPERIMENT = "beam24_20260824_regular_array_robustness_top1"


def dft4() -> np.ndarray:
    axis = np.arange(4, dtype=np.float32)
    return (
        np.exp(-2j * np.pi * axis[:, None] * axis[None, :] / 4.0) / 2.0
    ).astype(np.complex64)


def normalize_rows(candidate: np.ndarray, dense: np.ndarray) -> np.ndarray:
    gain = np.sum(candidate * dense.conj(), axis=1)
    dense_gain = np.sum(dense * dense.conj(), axis=1).real
    if np.any(np.abs(gain) < 1e-7):
        raise ValueError("candidate has a zero nominal look-direction gain")
    return (candidate * (dense_gain / gain)[:, None]).astype(np.complex64)


def local_f4_top2(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    groups = weights.reshape(weights.shape[0], -1, 4)
    transform = dft4()
    coefficients = groups @ transform.conj().T
    keep = np.argsort(-np.abs(coefficients), axis=-1, kind="stable")[..., :2]
    mask = np.zeros(coefficients.shape, dtype=bool)
    np.put_along_axis(mask, keep, True, axis=-1)
    projected = (np.where(mask, coefficients, 0.0) @ transform).reshape(weights.shape)
    return normalize_rows(projected, weights), mask


def direct_top2(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    groups = weights.reshape(weights.shape[0], -1, 4)
    keep = np.argsort(-np.abs(groups), axis=-1, kind="stable")[..., :2]
    mask = np.zeros(groups.shape, dtype=bool)
    np.put_along_axis(mask, keep, True, axis=-1)
    sparse = np.where(mask, groups, 0.0).reshape(weights.shape)
    return normalize_rows(sparse, weights), mask


def steering(positions_lambda: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    return np.exp(
        2j
        * np.pi
        * positions_lambda[:, None]
        * np.sin(np.deg2rad(angles_deg))[None, :]
    ).astype(np.complex64)


def condition_table() -> list[dict[str, object]]:
    clean = {
        "snr_db": None,
        "gain_std_db": 0.0,
        "phase_std_deg": 0.0,
        "position_std_lambda": 0.0,
        "reflection_db": None,
    }
    conditions: list[dict[str, object]] = [
        {"id": "clean", "family": "clean", "practical": True, **clean}
    ]
    for value in (10.0, 0.0, -5.0):
        conditions.append(
            {
                "id": f"snr_{value:g}db",
                "family": "snr",
                "practical": value >= 0.0,
                **clean,
                "snr_db": value,
            }
        )
    for value in (0.5, 1.0, 2.0):
        conditions.append(
            {
                "id": f"gain_sigma_{value:g}db",
                "family": "gain",
                "practical": value <= 1.0,
                **clean,
                "gain_std_db": value,
            }
        )
    for value in (1.0, 5.0, 10.0):
        conditions.append(
            {
                "id": f"phase_sigma_{value:g}deg",
                "family": "phase",
                "practical": value <= 5.0,
                **clean,
                "phase_std_deg": value,
            }
        )
    for value in (0.001, 0.005, 0.01):
        conditions.append(
            {
                "id": f"position_sigma_{value:g}lambda",
                "family": "position",
                "practical": value <= 0.005,
                **clean,
                "position_std_lambda": value,
            }
        )
    for value in (-10.0, -6.0):
        conditions.append(
            {
                "id": f"reflection_{value:g}db",
                "family": "reflection",
                "practical": value <= -10.0,
                **clean,
                "reflection_db": value,
            }
        )
    conditions.append(
        {
            "id": "combined_practical",
            "family": "combined",
            "practical": True,
            "snr_db": 10.0,
            "gain_std_db": 1.0,
            "phase_std_deg": 5.0,
            "position_std_lambda": 0.005,
            "reflection_db": -10.0,
        }
    )
    return conditions


def family_seed(family: str, seed: int) -> int:
    offset = int.from_bytes(hashlib.sha256(family.encode()).digest()[:4], "little")
    return (seed + offset) % (2**32)


def received_signal(
    positions: np.ndarray,
    trials: int,
    condition: dict[str, object],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    generator = np.random.default_rng(family_seed(str(condition["family"]), seed))
    source_angles = generator.uniform(-55.0, 55.0, size=trials).astype(np.float32)
    gain_latent = generator.standard_normal((positions.size, trials), dtype=np.float32)
    phase_latent = generator.standard_normal((positions.size, trials), dtype=np.float32)
    position_latent = generator.standard_normal(
        (positions.size, trials), dtype=np.float32
    )

    perturbed_positions = positions[:, None] + position_latent * float(
        condition["position_std_lambda"]
    )
    signal = np.exp(
        2j
        * np.pi
        * perturbed_positions
        * np.sin(np.deg2rad(source_angles))[None, :]
    ).astype(np.complex64)
    gain = np.power(
        10.0, gain_latent * (float(condition["gain_std_db"]) / 20.0)
    ).astype(np.float32)
    phase = np.exp(
        1j * np.deg2rad(phase_latent * float(condition["phase_std_deg"]))
    ).astype(np.complex64)
    calibration = gain * phase
    signal *= calibration

    reflection_db = condition["reflection_db"]
    if reflection_db is not None:
        direction = generator.choice(np.asarray([-1.0, 1.0]), size=trials)
        offset = direction * generator.uniform(5.0, 25.0, size=trials)
        reflection_angles = np.clip(source_angles + offset, -55.0, 55.0)
        reflection = np.exp(
            2j
            * np.pi
            * perturbed_positions
            * np.sin(np.deg2rad(reflection_angles))[None, :]
        ).astype(np.complex64)
        reflection_phase = np.exp(
            1j * generator.uniform(-np.pi, np.pi, size=trials)
        ).astype(np.complex64)
        reflection_scale = np.float32(10.0 ** (float(reflection_db) / 20.0))
        signal += reflection_scale * reflection * calibration * reflection_phase[None, :]

    snr_db = condition["snr_db"]
    noise_variance = 0.0 if snr_db is None else 10.0 ** (-float(snr_db) / 10.0)
    return signal, source_angles, noise_variance


def power_map(weights: np.ndarray, signal: np.ndarray, noise_variance: float) -> np.ndarray:
    response = weights.conj() @ signal
    power = np.abs(response) ** 2
    if noise_variance:
        noise_gain = np.sum(np.abs(weights) ** 2, axis=1, dtype=np.float64)
        power += (noise_variance * noise_gain[:, None]).astype(np.float32)
    return power.astype(np.float32)


def spectrum_metrics(reference: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference64 = reference.astype(np.float64)
    candidate64 = candidate.astype(np.float64)
    reference_centered = reference64 - np.mean(reference64, axis=0, keepdims=True)
    candidate_centered = candidate64 - np.mean(candidate64, axis=0, keepdims=True)
    numerator = np.sum(reference_centered * candidate_centered, axis=0)
    denominator = np.sqrt(
        np.sum(reference_centered**2, axis=0)
        * np.sum(candidate_centered**2, axis=0)
    )
    correlation = np.divide(
        numerator,
        denominator,
        out=np.ones_like(numerator),
        where=denominator > 0,
    )
    error = np.sum((candidate64 - reference64) ** 2, axis=0)
    energy = np.sum(reference64**2, axis=0)
    nmse = np.divide(error, energy, out=np.zeros_like(error), where=energy > 0)
    return np.clip(correlation, -1.0, 1.0), nmse


def select_hierarchy(full_power: np.ndarray, coarse_power: np.ndarray) -> np.ndarray:
    full_beams, trials = full_power.shape
    coarse_beams = coarse_power.shape[0]
    centers = np.rint(np.linspace(0, full_beams - 1, coarse_beams)).astype(np.int64)
    stride = int(round((full_beams - 1) / (coarse_beams - 1)))
    radius = max(1, stride // 2)
    winners = np.empty(trials, dtype=np.int64)
    for trial in range(trials):
        sectors = np.argsort(-coarse_power[:, trial], kind="stable")[:8]
        candidates: set[int] = set()
        for sector in sectors:
            center = int(centers[sector])
            candidates.update(
                range(max(0, center - radius), min(full_beams - 1, center + radius) + 1)
            )
        indices = np.asarray(sorted(candidates), dtype=np.int64)
        winners[trial] = int(indices[np.argmax(full_power[indices, trial])])
    return winners


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "p05": float(np.percentile(values, 5.0)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95.0)),
        "max": float(np.max(values)),
    }


def method_summary(
    dense_power: np.ndarray,
    candidate_power: np.ndarray,
    beam_angles: np.ndarray,
    source_angles: np.ndarray,
) -> dict[str, object]:
    dense_winner = np.argmax(dense_power, axis=0)
    candidate_winner = np.argmax(candidate_power, axis=0)
    shift = np.abs(beam_angles[candidate_winner] - beam_angles[dense_winner])
    correlation, nmse = spectrum_metrics(dense_power, candidate_power)
    true_error = np.abs(beam_angles[candidate_winner] - source_angles)
    return {
        "top1_agreement": float(np.mean(candidate_winner == dense_winner)),
        "shift_deg": quantiles(shift),
        "spectrum_correlation": quantiles(correlation),
        "relative_power_nmse": quantiles(nmse),
        "absolute_error_to_source_deg": quantiles(true_error),
    }


def hierarchy_summary(
    exhaustive_power: np.ndarray,
    coarse_power: np.ndarray,
    beam_angles: np.ndarray,
) -> dict[str, object]:
    exhaustive = np.argmax(exhaustive_power, axis=0)
    hierarchy = select_hierarchy(exhaustive_power, coarse_power)
    shift = np.abs(beam_angles[hierarchy] - beam_angles[exhaustive])
    return {
        "top1_agreement": float(np.mean(hierarchy == exhaustive)),
        "shift_deg": quantiles(shift),
    }


def source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.sensors % 4:
        raise ValueError("sensor count must be divisible by four")
    if args.sensors != 512 or args.beams != 1024 or args.coarse_beams != 128:
        raise ValueError("this frozen experiment admits only K512/M1024/M1=128")

    positions = (0.5 * np.arange(args.sensors, dtype=np.float32)).astype(np.float32)
    beam_angles = np.linspace(-60.0, 60.0, args.beams, dtype=np.float32)
    dense = (steering(positions, beam_angles).T / args.sensors).astype(np.complex64)
    direct, direct_mask = direct_top2(dense)
    local, local_mask = local_f4_top2(dense)
    if not np.all(direct_mask.sum(axis=-1) == 2):
        raise AssertionError("direct support is not exact 2:4")
    if not np.all(local_mask.sum(axis=-1) == 2):
        raise AssertionError("local support is not exact 2:4")

    subset_start = (args.sensors - 128) // 2
    subset = np.arange(subset_start, subset_start + 128)
    coarse_angles = np.linspace(-60.0, 60.0, args.coarse_beams, dtype=np.float32)
    coarse_dense = (
        steering(positions[subset], coarse_angles).T / subset.size
    ).astype(np.complex64)
    coarse_local, coarse_mask = local_f4_top2(coarse_dense)
    if not np.all(coarse_mask.sum(axis=-1) == 2):
        raise AssertionError("coarse support is not exact 2:4")

    per_condition: list[dict[str, object]] = []
    for condition in condition_table():
        signal, source_angles, noise_variance = received_signal(
            positions, args.trials, condition, args.seed
        )
        dense_power = power_map(dense, signal, noise_variance)
        direct_power = power_map(direct, signal, noise_variance)
        local_power = power_map(local, signal, noise_variance)
        coarse_power = power_map(coarse_local, signal[subset], noise_variance)
        row = {
            "condition": condition,
            "trials": args.trials,
            "dense_absolute_error_to_source_deg": quantiles(
                np.abs(beam_angles[np.argmax(dense_power, axis=0)] - source_angles)
            ),
            "methods": {
                "direct_joint_2of4": method_summary(
                    dense_power, direct_power, beam_angles, source_angles
                ),
                "beam24_local_f4": method_summary(
                    dense_power, local_power, beam_angles, source_angles
                ),
            },
            "hierarchy_vs_exhaustive_beam24": hierarchy_summary(
                local_power, coarse_power, beam_angles
            ),
        }
        per_condition.append(row)
        local_summary = row["methods"]["beam24_local_f4"]
        hierarchy = row["hierarchy_vs_exhaustive_beam24"]
        print(
            f"{condition['id']}: local_top1={local_summary['top1_agreement']:.4f} "
            f"corr_p05={local_summary['spectrum_correlation']['p05']:.6f} "
            f"hier_top1={hierarchy['top1_agreement']:.4f}",
            flush=True,
        )

    gate_failures: list[str] = []
    for row in per_condition:
        condition = row["condition"]
        local = row["methods"]["beam24_local_f4"]
        hierarchy = row["hierarchy_vs_exhaustive_beam24"]
        if condition["id"] == "clean":
            if local["top1_agreement"] < 0.99:
                gate_failures.append("clean local top1")
            if local["spectrum_correlation"]["mean"] < 0.999:
                gate_failures.append("clean local mean correlation")
            if hierarchy["top1_agreement"] < 0.999:
                gate_failures.append("clean hierarchy top1")
        if condition["practical"]:
            if local["top1_agreement"] < 0.99:
                gate_failures.append(f"{condition['id']} local top1")
            if local["spectrum_correlation"]["p05"] < 0.995:
                gate_failures.append(f"{condition['id']} local p05 correlation")
            if hierarchy["top1_agreement"] < 0.99:
                gate_failures.append(f"{condition['id']} hierarchy top1")

    result: dict[str, object] = {
        "experiment": EXPERIMENT,
        "state": "measured",
        "contract": {
            "geometry": "nominal half-wavelength ULA",
            "sensors": args.sensors,
            "beams": args.beams,
            "beam_grid_deg": [-60.0, 60.0],
            "source_grid_deg": "continuous uniform [-55, 55]",
            "trials_per_condition": args.trials,
            "seed": args.seed,
            "hierarchy": {
                "stage1_sensors": 128,
                "coarse_beams": args.coarse_beams,
                "top_sectors": 8,
                "stage2_candidate_rule": "integer centers and half-stride union",
            },
            "noise": "deterministic expected output power",
            "oracle": "nominal dense beamformer on the identical perturbed received manifold",
        },
        "environment": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "git_commit": git_commit(),
            "source_sha256": source_sha256(),
        },
        "conditions": per_condition,
        "decision": {
            "practical_envelope_gate": "accepted" if not gate_failures else "rejected",
            "gate_failures": gate_failures,
            "implementation": "measured CPU quality oracle",
            "mechanism": (
                "robustness extension supported inside the declared synthetic envelope"
                if not gate_failures
                else "robustness extension not supported by the predeclared gate"
            ),
            "thesis_impact": (
                "may add bounded synthetic regular-array robustness wording"
                if not gate_failures
                else "retain existing clean/real-data scope; do not add robustness wording"
            ),
            "allowed_wording": (
                "Beam24 preserves dense top-1 and spectrum shape across the declared synthetic regular-array perturbation envelope."
                if not gate_failures
                else "No positive perturbation-robustness claim is allowed."
            ),
            "boundary": "synthetic nominal-codebook ULA; not real calibration, general-array, or top-k evidence",
        },
    }
    return result


def write_outputs(result: dict[str, object], output: Path, csv_path: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    fields = [
        "condition",
        "practical",
        "method",
        "top1_agreement",
        "shift_mean_deg",
        "shift_p95_deg",
        "correlation_mean",
        "correlation_p05",
        "nmse_mean",
    ]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in result["conditions"]:
            condition = row["condition"]
            for method, metrics in row["methods"].items():
                writer.writerow(
                    {
                        "condition": condition["id"],
                        "practical": condition["practical"],
                        "method": method,
                        "top1_agreement": metrics["top1_agreement"],
                        "shift_mean_deg": metrics["shift_deg"]["mean"],
                        "shift_p95_deg": metrics["shift_deg"]["p95"],
                        "correlation_mean": metrics["spectrum_correlation"]["mean"],
                        "correlation_p05": metrics["spectrum_correlation"]["p05"],
                        "nmse_mean": metrics["relative_power_nmse"]["mean"],
                    }
                )
            hierarchy = row["hierarchy_vs_exhaustive_beam24"]
            writer.writerow(
                {
                    "condition": condition["id"],
                    "practical": condition["practical"],
                    "method": "hierarchy_vs_exhaustive_beam24",
                    "top1_agreement": hierarchy["top1_agreement"],
                    "shift_mean_deg": hierarchy["shift_deg"]["mean"],
                    "shift_p95_deg": hierarchy["shift_deg"]["p95"],
                    "correlation_mean": "",
                    "correlation_p05": "",
                    "nmse_mean": "",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--sensors", type=int, default=512)
    parser.add_argument("--beams", type=int, default=1024)
    parser.add_argument("--coarse-beams", type=int, default=128)
    args = parser.parse_args()
    result = run(args)
    write_outputs(result, args.output, args.csv)
    print(json.dumps(result["decision"], indent=2))
    if result["decision"]["practical_envelope_gate"] != "accepted":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
