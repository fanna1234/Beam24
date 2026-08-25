#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from cusparselt_prune import complex_matmul_fp16_output, prune_complex


def dft4() -> np.ndarray:
    x = np.arange(4)
    return np.exp(-2j * np.pi * x[:, None] * x[None, :] / 4.0) / 2.0


def keep(values: np.ndarray, score: np.ndarray, count: int) -> np.ndarray:
    idx = np.argpartition(score, score.shape[-1] - count, axis=-1)[..., -count:]
    mask = np.zeros(score.shape, bool)
    np.put_along_axis(mask, idx, True, axis=-1)
    return np.where(mask, values, 0.0)


def sparsify(weights: np.ndarray, variant: str) -> np.ndarray:
    groups = weights.reshape(len(weights), -1, 4)
    if variant == "D0_dense":
        candidate = weights.copy()
    elif variant == "CUSPARSELT_PRUNE_STRIP":
        candidate = prune_complex(weights, "strip")
    elif variant == "CUSPARSELT_PRUNE_TILE":
        candidate = prune_complex(weights, "tile")
    elif variant == "Q1_direct_joint_2of4":
        candidate = keep(groups, abs(groups), 2).reshape(weights.shape)
    else:
        f4 = dft4()
        transformed = groups @ f4.conj().T
        if variant == "Q2_localf4_joint_top2":
            sparse = keep(transformed, abs(transformed), 2)
        elif variant == "Q4_localf4_unstructured50":
            flat = transformed.reshape(weights.shape)
            sparse = keep(flat[:, None], abs(flat[:, None]), weights.shape[1] // 2).reshape(transformed.shape)
        elif variant == "Q5_localf4_independent_ri_2of4":
            sparse = keep(transformed.real, abs(transformed.real), 2) + 1j * keep(
                transformed.imag, abs(transformed.imag), 2
            )
        else:
            raise ValueError(variant)
        candidate = (sparse @ f4).reshape(weights.shape)
    gain = np.sum(candidate * weights.conj(), axis=1)
    dense_gain = np.sum(weights * weights.conj(), axis=1)
    return candidate * (dense_gain / gain)[:, None]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--sensors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frequency-min", type=float, default=160.0)
    parser.add_argument("--frequency-max", type=float, default=180.0)
    parser.add_argument(
        "--experiment", default="beam24_20260823_spib48_vla_realdata"
    )
    args = parser.parse_args()
    sensors = np.loadtxt(args.sensors)
    sensors -= sensors.mean()
    angles = np.deg2rad(np.arange(-90.0, 90.0001, 0.25))
    variants = [
        "D0_dense",
        "CUSPARSELT_PRUNE_STRIP",
        "CUSPARSELT_PRUNE_TILE",
        "Q1_direct_joint_2of4",
        "Q2_localf4_joint_top2",
        "Q4_localf4_unstructured50",
        "Q5_localf4_independent_ri_2of4",
    ]
    rows = []
    for path in sorted(args.data_dir.glob("*.mat")):
        data = np.asarray(loadmat(path)["dat"], dtype=np.float64)
        data[23] *= -1.0
        nfft = 1024
        hop = 512
        starts = np.arange(0, data.shape[1] - nfft + 1, hop)
        frames = np.stack([data[:, start : start + nfft] * np.hanning(nfft) for start in starts])
        spectrum = np.fft.rfft(frames, axis=2)
        frequencies = np.fft.rfftfreq(nfft, 1.0 / 1000.0)
        bins = np.flatnonzero(
            (frequencies >= args.frequency_min)
            & (frequencies <= args.frequency_max)
        )
        if bins.size == 0:
            raise ValueError("frequency range selects no FFT bins")
        maps = {variant: np.zeros(len(angles)) for variant in variants}
        for fi in bins:
            steering = np.exp(-1j * 2.0 * np.pi * frequencies[fi] * np.sin(angles)[:, None] * sensors[None, :] / 1500.0)
            dense = steering.conj() / len(sensors)
            snapshots = spectrum[:, :, fi].T
            for variant in variants:
                candidate = sparsify(dense, variant)
                output = (
                    complex_matmul_fp16_output(candidate, snapshots)
                    if variant.startswith("CUSPARSELT_")
                    else candidate @ snapshots
                )
                maps[variant] += np.sum(abs(output) ** 2, axis=1)
        dense_peak = int(np.argmax(maps["D0_dense"]))
        dense_norm = maps["D0_dense"] / maps["D0_dense"].max()
        for variant in variants:
            peak = int(np.argmax(maps[variant]))
            norm = maps[variant] / maps[variant].max()
            rows.append(
                {
                    "file": path.name,
                    "variant": variant,
                    "dense_peak_deg": float(np.rad2deg(angles[dense_peak])),
                    "peak_deg": float(np.rad2deg(angles[peak])),
                    "peak_shift_deg": float(abs(np.rad2deg(angles[peak] - angles[dense_peak]))),
                    "map_correlation": float(np.corrcoef(dense_norm, norm)[0, 1]),
                    "peak_delta_db_at_dense": float(
                        10.0 * np.log10(maps[variant][dense_peak] / maps["D0_dense"][dense_peak])
                    ),
                }
            )
    summary = {}
    for variant in variants:
        values = [row for row in rows if row["variant"] == variant]
        summary[variant] = {
            "files": len(values),
            "dense_peak_agreement_rate": float(np.mean([row["peak_shift_deg"] == 0 for row in values])),
            "mean_peak_shift_deg": float(np.mean([row["peak_shift_deg"] for row in values])),
            "max_peak_shift_deg": float(np.max([row["peak_shift_deg"] for row in values])),
            "mean_map_correlation": float(np.mean([row["map_correlation"] for row in values])),
            "worst_peak_delta_db_at_dense": float(np.min([row["peak_delta_db_at_dense"] for row in values])),
        }
    result = {
        "experiment": args.experiment,
        "input_kind": "real_underwater_vla",
        "contract": {
            "files": len(list(args.data_dir.glob("*.mat"))),
            "K": 48,
            "sample_rate": 1000,
            "frequency_hz": [args.frequency_min, args.frequency_max],
        },
        "summary": summary,
        "details": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"experiment": result["experiment"], "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
