#!/usr/bin/env python3
"""Run the frozen held-out hierarchy margin and candidate-coverage audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


EXPERIMENT = "beam24_20260825_hierarchy_margin_coverage_heldout"
ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = (
    ROOT
    / "src"
    / "quality"
    / "regular_ula_robustness.py"
)
FROZEN_SEEDS = [20260828, 20260829, 20260830]
FROZEN_TOP_L = [1, 2, 4, 8, 16, 32]
FROZEN_TRIALS = 1024
FROZEN_CONDITIONS = 15
FROZEN_ADMITTED_CONDITIONS = 13


def load_base():
    spec = importlib.util.spec_from_file_location("beam24_margin_base", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def quantiles(values: np.ndarray) -> dict[str, float]:
    values64 = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(values64)),
        "p01": float(np.percentile(values64, 1.0)),
        "p05": float(np.percentile(values64, 5.0)),
        "median": float(np.median(values64)),
        "p95": float(np.percentile(values64, 95.0)),
        "p99": float(np.percentile(values64, 99.0)),
        "max": float(np.max(values64)),
        "mean": float(np.mean(values64)),
    }


def normalized_top1_margin(power: np.ndarray) -> np.ndarray:
    partition = np.partition(power, power.shape[0] - 2, axis=0)[-2:]
    best = np.max(partition, axis=0).astype(np.float64)
    second = np.min(partition, axis=0).astype(np.float64)
    return np.divide(
        best - second,
        np.maximum(np.abs(best), 1e-30),
        out=np.zeros_like(best),
    ).astype(np.float32)


def fixed_work_ratio(top_l: int) -> float:
    return float((128 * 128 + (16 * top_l) * 512) / (1024 * 512))


def candidate_metrics(
    exhaustive_power: np.ndarray,
    coarse_power: np.ndarray,
) -> dict[str, np.ndarray]:
    full_beams, trials = exhaustive_power.shape
    coarse_beams = coarse_power.shape[0]
    centers = np.rint(np.linspace(0, full_beams - 1, coarse_beams)).astype(np.int64)
    stride = int(round((full_beams - 1) / (coarse_beams - 1)))
    radius = max(1, stride // 2)
    exhaustive_winner = np.argmax(exhaustive_power, axis=0).astype(np.int64)
    order = np.argsort(-coarse_power, axis=0, kind="stable")
    ordered_scores = np.take_along_axis(coarse_power, order, axis=0)
    ordered_centers = centers[order]
    covered_by_rank = (
        np.abs(ordered_centers - exhaustive_winner[None, :]) <= radius
    )
    has_cover = np.any(covered_by_rank, axis=0)
    coverage_rank = np.where(
        has_cover,
        np.argmax(covered_by_rank, axis=0) + 1,
        coarse_beams + 1,
    ).astype(np.uint16)

    trial_index = np.arange(trials)
    top_l_recall = np.empty((trials, len(FROZEN_TOP_L)), dtype=bool)
    boundary_margin = np.empty((trials, len(FROZEN_TOP_L)), dtype=np.float32)
    unique_candidates = np.empty((trials, len(FROZEN_TOP_L)), dtype=np.uint16)
    for column, top_l in enumerate(FROZEN_TOP_L):
        mask = np.zeros((full_beams, trials), dtype=bool)
        for rank in range(top_l):
            center = ordered_centers[rank]
            for offset in range(-radius, radius + 1):
                candidate = np.clip(center + offset, 0, full_beams - 1)
                mask[candidate, trial_index] = True
        top_l_recall[:, column] = mask[exhaustive_winner, trial_index]
        unique_candidates[:, column] = np.sum(mask, axis=0).astype(np.uint16)
        denominator = np.maximum(np.abs(ordered_scores[0]), 1e-30)
        boundary_margin[:, column] = (
            (ordered_scores[top_l - 1] - ordered_scores[top_l]) / denominator
        ).astype(np.float32)
    return {
        "exhaustive_winner": exhaustive_winner.astype(np.uint16),
        "coverage_rank": coverage_rank,
        "top_l_recall": top_l_recall,
        "boundary_margin": boundary_margin,
        "unique_candidates": unique_candidates,
    }


def summarize_condition(
    condition: dict[str, object],
    dense_margin: np.ndarray,
    local_margin: np.ndarray,
    local_dense_exact: np.ndarray,
    candidate: dict[str, np.ndarray],
) -> dict[str, object]:
    coverage_rank = candidate["coverage_rank"]
    top_l_rows: dict[str, object] = {}
    for column, top_l in enumerate(FROZEN_TOP_L):
        top_l_rows[str(top_l)] = {
            "recall": float(np.mean(candidate["top_l_recall"][:, column])),
            "boundary_margin": quantiles(candidate["boundary_margin"][:, column]),
            "unique_candidates": quantiles(candidate["unique_candidates"][:, column]),
            "fixed_stage2_slots": 16 * top_l,
            "beam_sensor_work_ratio": fixed_work_ratio(top_l),
        }
    return {
        "condition": condition,
        "trials": int(coverage_rank.size),
        "local_f4_dense_exact_top1": float(np.mean(local_dense_exact)),
        "dense_top1_margin": quantiles(dense_margin),
        "local_f4_top1_margin": quantiles(local_margin),
        "coverage_rank": {
            **quantiles(coverage_rank),
            "histogram": {
                str(rank): int(np.sum(coverage_rank == rank))
                for rank in range(1, int(np.max(coverage_rank)) + 1)
            },
        },
        "top_l": top_l_rows,
    }


def setup_weights(base):
    sensors = 512
    beams = 1024
    coarse_beams = 128
    positions = (0.5 * np.arange(sensors, dtype=np.float32)).astype(np.float32)
    beam_angles = np.linspace(-60.0, 60.0, beams, dtype=np.float32)
    dense = (base.steering(positions, beam_angles).T / sensors).astype(np.complex64)
    local, local_mask = base.local_f4_top2(dense)
    if not np.all(local_mask.sum(axis=-1) == 2):
        raise AssertionError("local support is not exact 2:4")
    subset_start = (sensors - 128) // 2
    subset = np.arange(subset_start, subset_start + 128)
    coarse_angles = np.linspace(-60.0, 60.0, coarse_beams, dtype=np.float32)
    coarse_dense = (
        base.steering(positions[subset], coarse_angles).T / subset.size
    ).astype(np.complex64)
    coarse_local, coarse_mask = base.local_f4_top2(coarse_dense)
    if not np.all(coarse_mask.sum(axis=-1) == 2):
        raise AssertionError("coarse support is not exact 2:4")
    full_steering = base.steering(positions, beam_angles)
    return {
        "positions": positions,
        "dense": dense,
        "local": local,
        "subset": subset,
        "coarse_local": coarse_local,
        "dense_response": (dense.conj() @ full_steering).astype(np.complex64),
        "local_response": (local.conj() @ full_steering).astype(np.complex64),
        "coarse_response": (
            coarse_local.conj() @ full_steering[subset]
        ).astype(np.complex64),
    }


def separated_sources(
    generator: np.random.Generator, count: int, beams: int
) -> np.ndarray:
    while True:
        sources = np.sort(generator.choice(beams, size=count, replace=False))
        if np.min(np.diff(sources)) >= 8:
            return sources


def close_sources(
    generator: np.random.Generator, count: int, beams: int
) -> np.ndarray:
    start = int(generator.integers(0, beams - 8))
    offsets = np.sort(generator.choice(np.arange(8), size=count, replace=False))
    return (start + offsets).astype(np.int64)


def multisource_conditions(
    base,
    setup: dict[str, np.ndarray],
    trials: int,
    seed: int,
) -> list[tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray]]:
    dense_response = setup["dense_response"]
    local_response = setup["local_response"]
    coarse_response = setup["coarse_response"]
    beams = dense_response.shape[0]
    outputs: list[tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray]] = []

    generator = np.random.default_rng(base.family_seed("margin_incoherent", seed))
    dense_power = np.empty((beams, trials), dtype=np.float32)
    local_power = np.empty((beams, trials), dtype=np.float32)
    coarse_power = np.empty((coarse_response.shape[0], trials), dtype=np.float32)
    for trial in range(trials):
        count = int(generator.integers(2, 4))
        sources = separated_sources(generator, count, beams)
        energies = generator.uniform(0.25, 2.25, size=count)
        dense_power[:, trial] = np.sum(
            np.abs(dense_response[:, sources]) ** 2 * energies[None, :], axis=1
        )
        local_power[:, trial] = np.sum(
            np.abs(local_response[:, sources]) ** 2 * energies[None, :], axis=1
        )
        coarse_power[:, trial] = np.sum(
            np.abs(coarse_response[:, sources]) ** 2 * energies[None, :], axis=1
        )
    outputs.append(
        (
            {
                "id": "multisource_incoherent",
                "family": "multisource",
                "practical": True,
                "admitted": True,
                "source_count": [2, 3],
                "minimum_source_separation_beams": 8,
                "snapshot_model": "incoherent expected power",
            },
            dense_power,
            local_power,
            coarse_power,
        )
    )

    generator = np.random.default_rng(base.family_seed("margin_coherent", seed))
    dense_power = np.empty((beams, trials), dtype=np.float32)
    local_power = np.empty((beams, trials), dtype=np.float32)
    coarse_power = np.empty((coarse_response.shape[0], trials), dtype=np.float32)
    snapshots = 64
    for trial in range(trials):
        count = int(generator.integers(2, 4))
        sources = separated_sources(generator, count, beams)
        common = (
            generator.standard_normal((1, snapshots))
            + 1j * generator.standard_normal((1, snapshots))
        ).astype(np.complex64)
        amplitudes = generator.uniform(0.5, 1.5, size=(count, 1)).astype(np.float32)
        phases = np.exp(
            1j * generator.uniform(-np.pi, np.pi, size=(count, 1))
        ).astype(np.complex64)
        signals = amplitudes * phases * common
        dense_output = dense_response[:, sources] @ signals
        local_output = local_response[:, sources] @ signals
        coarse_output = coarse_response[:, sources] @ signals
        dense_power[:, trial] = np.sum(np.abs(dense_output) ** 2, axis=1)
        local_power[:, trial] = np.sum(np.abs(local_output) ** 2, axis=1)
        coarse_power[:, trial] = np.sum(np.abs(coarse_output) ** 2, axis=1)
    outputs.append(
        (
            {
                "id": "multisource_coherent_64",
                "family": "multisource",
                "practical": True,
                "admitted": True,
                "source_count": [2, 3],
                "minimum_source_separation_beams": 8,
                "snapshots": snapshots,
            },
            dense_power,
            local_power,
            coarse_power,
        )
    )

    generator = np.random.default_rng(base.family_seed("margin_close_incoherent", seed))
    dense_power = np.empty((beams, trials), dtype=np.float32)
    local_power = np.empty((beams, trials), dtype=np.float32)
    coarse_power = np.empty((coarse_response.shape[0], trials), dtype=np.float32)
    for trial in range(trials):
        count = int(generator.integers(2, 4))
        sources = close_sources(generator, count, beams)
        energies = generator.uniform(0.8, 1.2, size=count)
        dense_power[:, trial] = np.sum(
            np.abs(dense_response[:, sources]) ** 2 * energies[None, :], axis=1
        )
        local_power[:, trial] = np.sum(
            np.abs(local_response[:, sources]) ** 2 * energies[None, :], axis=1
        )
        coarse_power[:, trial] = np.sum(
            np.abs(coarse_response[:, sources]) ** 2 * energies[None, :], axis=1
        )
    outputs.append(
        (
            {
                "id": "stress_close_incoherent",
                "family": "multisource_stress",
                "practical": False,
                "admitted": False,
                "source_count": [2, 3],
                "maximum_source_span_beams": 7,
                "energy_range": [0.8, 1.2],
                "snapshot_model": "incoherent expected power",
            },
            dense_power,
            local_power,
            coarse_power,
        )
    )

    generator = np.random.default_rng(base.family_seed("margin_close_coherent", seed))
    dense_power = np.empty((beams, trials), dtype=np.float32)
    local_power = np.empty((beams, trials), dtype=np.float32)
    coarse_power = np.empty((coarse_response.shape[0], trials), dtype=np.float32)
    for trial in range(trials):
        count = int(generator.integers(2, 4))
        sources = close_sources(generator, count, beams)
        common = (
            generator.standard_normal((1, snapshots))
            + 1j * generator.standard_normal((1, snapshots))
        ).astype(np.complex64)
        amplitudes = generator.uniform(0.8, 1.2, size=(count, 1)).astype(np.float32)
        phases = np.exp(
            1j * generator.uniform(-np.pi, np.pi, size=(count, 1))
        ).astype(np.complex64)
        signals = amplitudes * phases * common
        dense_output = dense_response[:, sources] @ signals
        local_output = local_response[:, sources] @ signals
        coarse_output = coarse_response[:, sources] @ signals
        dense_power[:, trial] = np.sum(np.abs(dense_output) ** 2, axis=1)
        local_power[:, trial] = np.sum(np.abs(local_output) ** 2, axis=1)
        coarse_power[:, trial] = np.sum(np.abs(coarse_output) ** 2, axis=1)
    outputs.append(
        (
            {
                "id": "stress_close_coherent_64",
                "family": "multisource_stress",
                "practical": False,
                "admitted": False,
                "source_count": [2, 3],
                "maximum_source_span_beams": 7,
                "amplitude_range": [0.8, 1.2],
                "snapshots": snapshots,
                "waveform": "shared coherent waveform with random source phase",
            },
            dense_power,
            local_power,
            coarse_power,
        )
    )
    return outputs


def run_seed(
    base,
    setup: dict[str, np.ndarray],
    seed: int,
    trials: int,
    raw_dir: Path,
) -> dict[str, object]:
    positions = setup["positions"]
    dense = setup["dense"]
    local = setup["local"]
    subset = setup["subset"]
    coarse_local = setup["coarse_local"]
    conditions = [condition for condition in base.condition_table() if condition["practical"]]
    if len(conditions) != 11:
        raise AssertionError(f"expected 11 practical conditions, got {len(conditions)}")

    dense_margins: list[np.ndarray] = []
    local_margins: list[np.ndarray] = []
    local_dense_exact_rows: list[np.ndarray] = []
    coverage_ranks: list[np.ndarray] = []
    top_l_recall_rows: list[np.ndarray] = []
    boundary_margins: list[np.ndarray] = []
    unique_candidates: list[np.ndarray] = []
    summaries: list[dict[str, object]] = []

    def record_condition(
        condition: dict[str, object],
        dense_power: np.ndarray,
        local_power: np.ndarray,
        coarse_power: np.ndarray,
    ) -> None:
        dense_margin = normalized_top1_margin(dense_power)
        local_margin = normalized_top1_margin(local_power)
        local_dense_exact = np.argmax(local_power, axis=0) == np.argmax(
            dense_power, axis=0
        )
        candidate = candidate_metrics(local_power, coarse_power)
        summaries.append(
            summarize_condition(
                condition,
                dense_margin,
                local_margin,
                local_dense_exact,
                candidate,
            )
        )
        dense_margins.append(dense_margin)
        local_margins.append(local_margin)
        local_dense_exact_rows.append(local_dense_exact)
        coverage_ranks.append(candidate["coverage_rank"])
        top_l_recall_rows.append(candidate["top_l_recall"])
        boundary_margins.append(candidate["boundary_margin"])
        unique_candidates.append(candidate["unique_candidates"])
        top8_recall = np.mean(candidate["top_l_recall"][:, FROZEN_TOP_L.index(8)])
        print(
            f"seed={seed} condition={condition['id']} "
            f"top8={top8_recall:.6f} max_rank={int(np.max(candidate['coverage_rank']))} "
            f"dense_margin_p01={np.percentile(dense_margin, 1):.3e}",
            flush=True,
        )

    for condition in conditions:
        signal, _, noise_variance = base.received_signal(
            positions, trials, condition, seed
        )
        record_condition(
            condition,
            base.power_map(dense, signal, noise_variance),
            base.power_map(local, signal, noise_variance),
            base.power_map(coarse_local, signal[subset], noise_variance),
        )
    for condition, dense_power, local_power, coarse_power in multisource_conditions(
        base, setup, trials, seed
    ):
        record_condition(condition, dense_power, local_power, coarse_power)

    if len(summaries) != FROZEN_CONDITIONS:
        raise AssertionError(
            f"expected {FROZEN_CONDITIONS} conditions, got {len(summaries)}"
        )

    arrays = {
        "condition_ids": np.asarray([str(item["condition"]["id"]) for item in summaries]),
        "top_l_values": np.asarray(FROZEN_TOP_L, dtype=np.int16),
        "dense_margin": np.stack(dense_margins),
        "local_margin": np.stack(local_margins),
        "local_dense_exact": np.stack(local_dense_exact_rows),
        "coverage_rank": np.stack(coverage_ranks),
        "top_l_recall": np.stack(top_l_recall_rows),
        "boundary_margin": np.stack(boundary_margins),
        "unique_candidates": np.stack(unique_candidates),
    }
    np.savez_compressed(raw_dir / f"seed_{seed}.npz", **arrays)
    result = {
        "experiment": EXPERIMENT,
        "seed": seed,
        "trials_per_condition": trials,
        "conditions": summaries,
    }
    (raw_dir / f"seed_{seed}.json").write_text(json.dumps(result, indent=2) + "\n")
    return {"summary": result, "arrays": arrays}


def aggregate(seed_results: list[dict[str, object]]) -> dict[str, object]:
    condition_ids = seed_results[0]["arrays"]["condition_ids"].tolist()
    per_condition: list[dict[str, object]] = []
    failures: list[str] = []
    all_coverage: list[np.ndarray] = []
    all_dense_margin: list[np.ndarray] = []
    all_local_margin: list[np.ndarray] = []
    all_local_exact: list[np.ndarray] = []
    all_top_l_recall: list[np.ndarray] = []
    all_boundary: list[np.ndarray] = []
    all_unique: list[np.ndarray] = []
    admitted_coverage_rows: list[np.ndarray] = []
    admitted_top_l_rows: list[np.ndarray] = []

    for condition_index, condition_id in enumerate(condition_ids):
        condition_metadata = seed_results[0]["summary"]["conditions"][
            condition_index
        ]["condition"]
        admitted = bool(condition_metadata.get("admitted", True))
        coverage = np.concatenate(
            [result["arrays"]["coverage_rank"][condition_index] for result in seed_results]
        )
        dense_margin = np.concatenate(
            [result["arrays"]["dense_margin"][condition_index] for result in seed_results]
        )
        local_margin = np.concatenate(
            [result["arrays"]["local_margin"][condition_index] for result in seed_results]
        )
        local_exact = np.concatenate(
            [result["arrays"]["local_dense_exact"][condition_index] for result in seed_results]
        )
        top_l_recall = np.concatenate(
            [result["arrays"]["top_l_recall"][condition_index] for result in seed_results]
        )
        boundary = np.concatenate(
            [result["arrays"]["boundary_margin"][condition_index] for result in seed_results]
        )
        unique = np.concatenate(
            [result["arrays"]["unique_candidates"][condition_index] for result in seed_results]
        )
        top_l_summary: dict[str, object] = {}
        for column, top_l in enumerate(FROZEN_TOP_L):
            recall = float(np.mean(top_l_recall[:, column]))
            top_l_summary[str(top_l)] = {
                "recall": recall,
                "boundary_margin": quantiles(boundary[:, column]),
                "unique_candidates": quantiles(unique[:, column]),
                "fixed_stage2_slots": 16 * top_l,
                "beam_sensor_work_ratio": fixed_work_ratio(top_l),
            }
        if admitted:
            if top_l_summary["8"]["recall"] != 1.0:
                failures.append(f"{condition_id}: aggregate top8 recall < 1")
            if int(np.max(coverage)) > 8:
                failures.append(f"{condition_id}: aggregate max coverage rank > 8")
            for result in seed_results:
                seed = result["summary"]["seed"]
                seed_coverage = result["arrays"]["coverage_rank"][condition_index]
                seed_top8 = result["arrays"]["top_l_recall"][
                    condition_index, :, FROZEN_TOP_L.index(8)
                ]
                if float(np.mean(seed_top8)) != 1.0:
                    failures.append(f"seed {seed} {condition_id}: top8 recall < 1")
                if int(np.max(seed_coverage)) > 8:
                    failures.append(f"seed {seed} {condition_id}: max coverage rank > 8")
            admitted_coverage_rows.append(coverage)
            admitted_top_l_rows.append(top_l_recall)
        per_condition.append(
            {
                "condition": condition_id,
                "admitted": admitted,
                "trials": int(coverage.size),
                "local_f4_dense_exact_top1": float(np.mean(local_exact)),
                "dense_top1_margin": quantiles(dense_margin),
                "local_f4_top1_margin": quantiles(local_margin),
                "coverage_rank": {
                    **quantiles(coverage),
                    "histogram": {
                        str(rank): int(np.sum(coverage == rank))
                        for rank in range(1, int(np.max(coverage)) + 1)
                    },
                },
                "top_l": top_l_summary,
            }
        )
        all_coverage.append(coverage)
        all_dense_margin.append(dense_margin)
        all_local_margin.append(local_margin)
        all_local_exact.append(local_exact)
        all_top_l_recall.append(top_l_recall)
        all_boundary.append(boundary)
        all_unique.append(unique)

    coverage = np.concatenate(all_coverage)
    dense_margin = np.concatenate(all_dense_margin)
    local_margin = np.concatenate(all_local_margin)
    local_exact = np.concatenate(all_local_exact)
    top_l_recall = np.concatenate(all_top_l_recall)
    boundary = np.concatenate(all_boundary)
    unique = np.concatenate(all_unique)
    global_top_l: dict[str, object] = {}
    for column, top_l in enumerate(FROZEN_TOP_L):
        global_top_l[str(top_l)] = {
            "recall": float(np.mean(top_l_recall[:, column])),
            "boundary_margin": quantiles(boundary[:, column]),
            "unique_candidates": quantiles(unique[:, column]),
            "fixed_stage2_slots": 16 * top_l,
            "beam_sensor_work_ratio": fixed_work_ratio(top_l),
        }
    admitted_coverage = np.concatenate(admitted_coverage_rows)
    admitted_top_l = np.concatenate(admitted_top_l_rows)
    admitted_top8_misses = int(
        np.sum(~admitted_top_l[:, FROZEN_TOP_L.index(8)])
    )
    return {
        "per_condition": per_condition,
        "global": {
            "trials": int(coverage.size),
            "local_f4_dense_exact_top1": float(np.mean(local_exact)),
            "dense_top1_margin": quantiles(dense_margin),
            "local_f4_top1_margin": quantiles(local_margin),
            "coverage_rank": {
                **quantiles(coverage),
                "histogram": {
                    str(rank): int(np.sum(coverage == rank))
                    for rank in range(1, int(np.max(coverage)) + 1)
                },
            },
            "top_l": global_top_l,
            "top8_misses": int(np.sum(~top_l_recall[:, FROZEN_TOP_L.index(8)])),
        },
        "admitted": {
            "conditions": len(admitted_coverage_rows),
            "trials": int(admitted_coverage.size),
            "coverage_rank": quantiles(admitted_coverage),
            "top_l_recall": {
                str(top_l): float(np.mean(admitted_top_l[:, column]))
                for column, top_l in enumerate(FROZEN_TOP_L)
            },
            "top8_misses": admitted_top8_misses,
            "rule_of_three_miss_rate_upper_95": (
                3.0 / admitted_coverage.size
                if admitted_top8_misses == 0
                else None
            ),
        },
        "failures": failures,
    }


def write_csv(seed_results: list[dict[str, object]], path: Path) -> None:
    fields = [
        "seed",
        "condition",
        "admitted",
        "top_l",
        "recall",
        "boundary_margin_p01",
        "boundary_margin_median",
        "unique_candidates_mean",
        "fixed_stage2_slots",
        "beam_sensor_work_ratio",
        "max_coverage_rank",
        "dense_margin_p01",
        "local_margin_p01",
        "local_f4_dense_exact_top1",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in seed_results:
            seed = result["summary"]["seed"]
            for condition in result["summary"]["conditions"]:
                for top_l, metrics in condition["top_l"].items():
                    writer.writerow(
                        {
                            "seed": seed,
                            "condition": condition["condition"]["id"],
                            "admitted": condition["condition"].get("admitted", True),
                            "top_l": top_l,
                            "recall": metrics["recall"],
                            "boundary_margin_p01": metrics["boundary_margin"]["p01"],
                            "boundary_margin_median": metrics["boundary_margin"]["median"],
                            "unique_candidates_mean": metrics["unique_candidates"]["mean"],
                            "fixed_stage2_slots": metrics["fixed_stage2_slots"],
                            "beam_sensor_work_ratio": metrics["beam_sensor_work_ratio"],
                            "max_coverage_rank": condition["coverage_rank"]["max"],
                            "dense_margin_p01": condition["dense_top1_margin"]["p01"],
                            "local_margin_p01": condition["local_f4_top1_margin"]["p01"],
                            "local_f4_dense_exact_top1": condition["local_f4_dense_exact_top1"],
                        }
                    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=FROZEN_TRIALS)
    parser.add_argument("--seeds", type=int, nargs="+", default=FROZEN_SEEDS)
    args = parser.parse_args()
    if args.trials != FROZEN_TRIALS:
        raise ValueError(f"frozen trial count is {FROZEN_TRIALS}")
    if args.seeds != FROZEN_SEEDS:
        raise ValueError(f"frozen seeds are {FROZEN_SEEDS}")

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    base = load_base()
    setup = setup_weights(base)
    seed_results = [
        run_seed(base, setup, seed, args.trials, args.raw_dir) for seed in args.seeds
    ]
    aggregated = aggregate(seed_results)
    decision = {
        "top8_coverage_gate": "accepted" if not aggregated["failures"] else "rejected",
        "gate_failures": aggregated["failures"],
        "implementation": "measured held-out CPU hierarchy audit",
        "mechanism": (
            "fixed top-8 empirically covers the exhaustive Local-F4 winner inside the independent K512 envelope"
            if not aggregated["failures"]
            else "fixed top-8 does not cover the independent K512 envelope"
        ),
        "thesis_impact": (
            "may add rank, margin, and recall-work evidence"
            if not aggregated["failures"]
            else "evaluate adaptive top-L or exhaustive fallback"
        ),
        "boundary": "synthetic continuous-angle K512 ULA top-1; no real-array, top-k, arbitrary-signal, or general-geometry guarantee",
    }
    summary = {
        "experiment": EXPERIMENT,
        "state": "measured",
        "contract": {
            "seeds": args.seeds,
            "conditions": FROZEN_CONDITIONS,
            "admitted_conditions": FROZEN_ADMITTED_CONDITIONS,
            "stress_conditions": FROZEN_CONDITIONS - FROZEN_ADMITTED_CONDITIONS,
            "single_source_perturbation_conditions": 11,
            "multisource_conditions": 4,
            "trials_per_condition_per_seed": args.trials,
            "total_trials": args.trials * len(args.seeds) * FROZEN_CONDITIONS,
            "top_l": FROZEN_TOP_L,
            "base_experiment": "beam24_20260824_regular_array_robustness_top1",
            "gate_application": "per seed and condition",
        },
        "environment": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "git_commit": git_commit(),
            "wrapper_source_sha256": sha256(Path(__file__)),
            "base_source_sha256": sha256(BASE_PATH),
        },
        **aggregated,
        "decision": decision,
    }
    (args.raw_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_csv(seed_results, args.raw_dir / "per_condition.csv")
    print(json.dumps({"global": summary["global"], "decision": decision}, indent=2))
    if aggregated["failures"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
