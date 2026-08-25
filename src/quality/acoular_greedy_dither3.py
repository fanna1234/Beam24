#!/usr/bin/env python3
"""Geometry-generalized three-step joint-complex K4 mask dither."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import h5py
import numpy as np

from acoular_dense_oracle import SOURCE_XY, load_mics, steering_weights, stft
from acoular_quality_screen import dft4, metric_summary, sparsify


LEGAL_MASKS = tuple(itertools.combinations(range(4), 2))


def transfer_matrix(
    grid_xyz: np.ndarray, mic_xyz: np.ndarray, frequency: float, sound_speed: float
) -> np.ndarray:
    center = mic_xyz.mean(axis=0)
    r0 = np.linalg.norm(grid_xyz - center[None, :], axis=1)
    rm = np.linalg.norm(grid_xyz[:, None, :] - mic_xyz[None, :, :], axis=2)
    wave_number = 2.0 * np.pi * frequency / sound_speed
    return (r0[:, None] / rm) * np.exp(-1j * wave_number * (rm - r0[:, None]))


def option_blocks(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    transform = dft4()
    coeff = weight.reshape(-1, 4) @ transform.conj().T
    blocks = np.zeros((coeff.shape[0], len(LEGAL_MASKS), 4), dtype=np.complex128)
    energies = np.zeros((coeff.shape[0], len(LEGAL_MASKS)), dtype=np.float64)
    for option, pair in enumerate(LEGAL_MASKS):
        sparse = np.zeros_like(coeff)
        sparse[:, pair] = coeff[:, pair]
        blocks[:, option, :] = sparse @ transform
        energies[:, option] = np.sum(np.abs(sparse) ** 2, axis=1)
    return blocks, energies


def synthesize_row(
    dense_weight: np.ndarray,
    transfer: np.ndarray,
    focus: int,
    grid_xy: np.ndarray,
    max_substitutions: int,
    min_snr_ratio: float,
    max_psll_delta_db: float,
) -> tuple[np.ndarray, dict]:
    num_groups = dense_weight.size // 4
    blocks, energies = option_blocks(dense_weight)
    transfer_groups = transfer.T.reshape(num_groups, 4, transfer.shape[0])
    responses = np.einsum("gok,gkp->gop", blocks, transfer_groups, optimize=True)

    assignment = np.argmax(energies, axis=1).astype(np.int64)
    group_index = np.arange(num_groups)
    current_response = np.sum(responses[group_index, assignment], axis=0)
    current_norm = float(np.sum(np.abs(blocks[group_index, assignment]) ** 2))
    dense_norm = float(np.sum(np.abs(dense_weight) ** 2))
    changed = set()
    trace = []
    sidelobe = np.linalg.norm(grid_xy - grid_xy[focus], axis=1) > 0.04

    def state_metrics(response: np.ndarray, norm: float) -> tuple[float, float, complex]:
        gain = response[focus]
        snr_ratio = dense_norm * float(np.abs(gain) ** 2) / norm
        psll = 20.0 * np.log10(np.max(np.abs(response[sidelobe] / gain)))
        return snr_ratio, float(psll), gain

    current_snr, current_psll, current_gain = state_metrics(current_response, current_norm)
    initial_snr = current_snr
    initial_psll = current_psll
    dense_response = dense_weight @ transfer.T
    dense_gain = dense_response[focus]
    dense_psll = float(20.0 * np.log10(np.max(np.abs(dense_response[sidelobe] / dense_gain))))
    psll_limit = dense_psll + max_psll_delta_db

    for step in range(max_substitutions):
        if current_snr + 1e-12 >= min_snr_ratio and current_psll <= psll_limit + 1e-12:
            break
        candidate_groups = []
        candidate_options = []
        candidate_response = []
        candidate_norm = []
        for group in range(num_groups):
            if group in changed:
                continue
            old = int(assignment[group])
            for option in range(len(LEGAL_MASKS)):
                if option == old:
                    continue
                candidate_groups.append(group)
                candidate_options.append(option)
                candidate_response.append(current_response - responses[group, old] + responses[group, option])
                candidate_norm.append(
                    current_norm
                    - float(np.sum(np.abs(blocks[group, old]) ** 2))
                    + float(np.sum(np.abs(blocks[group, option]) ** 2))
                )
        if not candidate_response:
            break

        response_batch = np.asarray(candidate_response)
        norm_batch = np.asarray(candidate_norm)
        gains = response_batch[:, focus]
        snr = dense_norm * np.abs(gains) ** 2 / norm_batch
        psll = 20.0 * np.log10(
            np.max(np.abs(response_batch[:, sidelobe] / gains[:, None]), axis=1)
        )

        if current_snr + 1e-12 < min_snr_ratio:
            best = int(np.lexsort((psll, -snr))[0])
            accepted = bool(snr[best] > current_snr + 1e-12)
            reason = "snr_repair"
        else:  # SNR passes, so only a PSLL-budget violation can reach this branch.
            admissible = np.flatnonzero(snr + 1e-12 >= min_snr_ratio)
            if admissible.size == 0:
                accepted = False
                best = 0
            else:
                local = int(np.lexsort((-snr[admissible], psll[admissible]))[0])
                best = int(admissible[local])
                accepted = bool(psll[best] < current_psll - 1e-12)
            reason = "psll_improvement"

        if not accepted:
            trace.append({"step": step + 1, "accepted": False, "reason": "no_improvement"})
            break

        group = int(candidate_groups[best])
        option = int(candidate_options[best])
        old = int(assignment[group])
        assignment[group] = option
        changed.add(group)
        current_response = response_batch[best]
        current_norm = float(norm_batch[best])
        current_snr, current_psll, current_gain = state_metrics(current_response, current_norm)
        trace.append(
            {
                "step": step + 1,
                "accepted": True,
                "reason": reason,
                "group": group,
                "old_option": old,
                "new_option": option,
                "snr_loss_db": float(10.0 * np.log10(current_snr)),
                "psll_db": current_psll,
            }
        )

    final = blocks[group_index, assignment].reshape(-1) / current_gain
    return final, {
        "changed_groups": len(changed),
        "initial_snr_loss_db": float(10.0 * np.log10(initial_snr)),
        "final_snr_loss_db": float(10.0 * np.log10(current_snr)),
        "initial_psll_db": initial_psll,
        "final_psll_db": current_psll,
        "dense_psll_db": dense_psll,
        "psll_limit_db": psll_limit,
        "snr_gate_pass": bool(current_snr + 1e-12 >= min_snr_ratio),
        "psll_gate_pass": bool(current_psll <= psll_limit + 1e-12),
        "trace": trace,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maps", type=Path, required=True)
    parser.add_argument("--max-substitutions", type=int, default=3)
    parser.add_argument("--max-psll-delta-db", type=float, default=3.0)
    args = parser.parse_args()

    with h5py.File(args.data, "r") as handle:
        waveform = np.asarray(handle["time_data"], dtype=np.float64)
        sample_rate = float(handle["time_data"].attrs["sample_freq"])
    mic_xyz = load_mics(args.geometry)

    nfft = 128
    spectrum = stft(waveform, nfft=nfft, hop=64)
    frequencies = np.fft.rfftfreq(nfft, d=1.0 / sample_rate)
    lower = 8000.0 * 2.0 ** (-1.0 / 6.0)
    upper = 8000.0 * 2.0 ** (+1.0 / 6.0)
    bins = np.flatnonzero((frequencies >= lower) & (frequencies < upper))

    axis = np.arange(-0.20, 0.2001, 0.01)
    grid_x, grid_y = np.meshgrid(axis, axis, indexing="xy")
    grid_xy = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    grid_xyz = np.column_stack([grid_xy, np.full(grid_xy.shape[0], -0.30)])
    off_source = np.ones(grid_xy.shape[0], dtype=bool)
    for point in SOURCE_XY:
        off_source &= np.linalg.norm(grid_xy - point, axis=1) > 0.04

    dense_power = np.zeros(grid_xy.shape[0], dtype=np.float64)
    q2_power = np.zeros_like(dense_power)
    q3_power = np.zeros_like(dense_power)
    diagnostics = []
    min_snr_ratio = 10.0 ** (-1.0 / 10.0)

    for freq_bin in bins:
        frequency = float(frequencies[freq_bin])
        transfer = transfer_matrix(grid_xyz, mic_xyz, frequency, 343.0)
        dense_weights = steering_weights(grid_xyz, mic_xyz, frequency, 343.0)
        q2_weights = sparsify(dense_weights, "Q2_localf4_joint_top2")
        q2_gain = np.sum(q2_weights * transfer, axis=1)
        q2_weights /= q2_gain[:, None]
        q3_weights = np.empty_like(dense_weights)

        for focus in range(grid_xy.shape[0]):
            q3_weights[focus], detail = synthesize_row(
                dense_weights[focus],
                transfer,
                focus,
                grid_xy,
                args.max_substitutions,
                min_snr_ratio,
                args.max_psll_delta_db,
            )
            detail["frequency_hz"] = frequency
            detail["focus_index"] = focus
            diagnostics.append(detail)

        snapshots = spectrum[:, freq_bin, :].T
        dense_power += np.sum(np.abs(dense_weights @ snapshots) ** 2, axis=1)
        q2_power += np.sum(np.abs(q2_weights @ snapshots) ** 2, axis=1)
        q3_power += np.sum(np.abs(q3_weights @ snapshots) ** 2, axis=1)

    q2_noise = [detail["initial_snr_loss_db"] * -1.0 for detail in diagnostics]
    q3_noise = [detail["final_snr_loss_db"] * -1.0 for detail in diagnostics]
    variants = [
        metric_summary("D0_dense", dense_power, dense_power, grid_xy, off_source, [0.0]),
        metric_summary("Q2_localf4_joint_top2", q2_power, dense_power, grid_xy, off_source, q2_noise),
        metric_summary("Q3_nonula_budgeted_dither3", q3_power, dense_power, grid_xy, off_source, q3_noise),
    ]
    changed = np.asarray([detail["changed_groups"] for detail in diagnostics])
    snr_pass = np.asarray([detail["snr_gate_pass"] for detail in diagnostics])
    psll_pass = np.asarray([detail["psll_gate_pass"] for detail in diagnostics])
    result = {
        "experiment": "beam24_20260823_acoular64_multisource_quality",
        "stage": "q3_nonula_budgeted_dither3",
        "input_kind": "synthetic_acoustic_waveform",
        "contract": {
            "max_substitutions": args.max_substitutions,
            "minimum_snr_loss_db": -1.0,
            "maximum_psll_delta_db": args.max_psll_delta_db,
            "sidelobe_exclusion_radius_m": 0.04,
            "channel_grouping": "fixed_upstream_order",
        },
        "variants": variants,
        "synthesis_diagnostics": {
            "rows_times_frequencies": len(diagnostics),
            "changed_group_mean": float(changed.mean()),
            "changed_group_max": int(changed.max()),
            "snr_gate_pass_fraction": float(snr_pass.mean()),
            "snr_gate_fail_count": int((~snr_pass).sum()),
            "psll_gate_pass_fraction": float(psll_pass.mean()),
            "psll_gate_fail_count": int((~psll_pass).sum()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(args.maps, grid_xy=grid_xy, dense=dense_power, q2=q2_power, q3=q3_power)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
