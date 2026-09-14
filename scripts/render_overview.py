#!/usr/bin/env python3
"""Render a compact Beam24 overview from the maintained local Fourier transform."""

from __future__ import annotations

import hashlib
from html import escape
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/quality"))
from spib_vla_screen import dft4


def main() -> None:
    phase_step = math.pi / 4
    weights = np.exp(1j * np.arange(4) * phase_step)
    modes = weights @ dft4()
    energy = abs(modes) ** 2 / np.sum(abs(modes) ** 2)
    support = sorted(np.argsort(energy)[-2:].tolist())
    pieces = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1160" height="360" viewBox="0 0 1160 360" role="img" aria-labelledby="title desc">',
        '<title id="title">Beam24: local structure and coarse-to-fine beam search</title>',
        '<desc id="desc">An illustrative regular-array phase progression produces four Fourier modes. Retaining two gives support shared by real and imaginary weights. Both GPU search stages use this representation, and the complete pipeline returns beam index and power without storing the full complex response matrix.</desc>',
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="#202124"/></marker></defs>',
        '<rect width="1160" height="360" rx="8" fill="white"/>',
        '<g font-family="Arial, Helvetica, sans-serif" fill="#111111">',
    ]

    def text(x, y, value, size=19, weight="400", anchor="middle"):
        pieces.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}">{escape(value)}</text>')

    def line(x1, y1, x2, y2, arrow=False, stroke="#202124", width=1.6):
        marker = ' marker-end="url(#arrow)"' if arrow else ""
        pieces.append(f'<path d="M{x1} {y1} L{x2} {y2}" stroke="{stroke}" stroke-width="{width}" fill="none"{marker}/>')

    def box(x, y, width, height, fill="#f2f5f7"):
        pieces.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="6" fill="{fill}"/>')

    text(20, 27, "LOCAL STRUCTURE", 14, "600", "start")
    text(117, 62, "Array phase", 20, "600")
    line(25, 115, 205, 115, stroke="#aab1b8")
    for index, phase in enumerate(np.arange(4) * phase_step):
        x, y = 47 + 47 * index, 112
        pieces.append(f'<circle cx="{x}" cy="{y}" r="16" fill="white" stroke="#7f9cb2" stroke-width="1.5"/>')
        line(x, y, x + 23 * math.cos(phase), y - 23 * math.sin(phase), True)
    text(117, 157, "Four adjacent sensors", 15)
    line(226, 110, 268, 110, True)
    text(352, 62, "Local F4 modes", 20, "600")
    for index, value in enumerate(energy):
        x = 291 + index * 35
        height = 62 * float(value / energy.max())
        fill = "#88aac4" if index in support else "#d9dfe4"
        pieces.append(f'<rect x="{x}" y="{132-height:.3f}" width="25" height="{height:.3f}" fill="{fill}"/>')
        text(x + 12.5, 152, str(index), 14)
    line(286, 132, 431, 132, stroke="#6b7279", width=1)
    line(444, 110, 482, 110, True)
    text(620, 62, "Joint-complex 2:4", 20, "600")
    for row in range(2):
        text(516, 106 + row * 29, "Re" if row == 0 else "Im", 16, anchor="end")
        for column in range(4):
            x, y = 530 + 42 * column, 85 + 29 * row
            fill = "#b7cedf" if column in support else "#f2f4f6"
            pieces.append(f'<rect x="{x}" y="{y}" width="42" height="29" fill="{fill}" stroke="#9ea7b0" stroke-width="0.8"/>')
            if column not in support:
                text(x + 21, y + 21, "0", 17)
    text(614, 163, "One support for both planes", 15)
    line(721, 110, 765, 110, True)
    box(790, 77, 340, 73, "#e8f0f7")
    text(960, 108, "Complex sparse MMA", 22, "600")
    text(960, 134, "FP32 accumulation", 17)

    pieces.append('<path d="M20 190 H1135" stroke="#bcc4cc" stroke-dasharray="5 5"/>')
    text(20, 220, "GPU SEARCH", 14, "600", "start")
    stages = [
        (20, 235, "Coarse search", "K128 · 128 beams"),
        (320, 245, "Select + gather", "8 sectors → 128 rows"),
        (630, 235, "Refine", "K512 full aperture"),
        (930, 205, "Power + top-1", "Beam index + power"),
    ]
    for index, (x, width, label, subtitle) in enumerate(stages):
        box(x, 239, width, 74, "#e8f0f7" if index in (0, 2) else "#f2f5f7")
        text(x + width / 2, 270, label, 21, "600")
        text(x + width / 2, 297, subtitle, 17)
    for x1, x2 in ((269, 306), (578, 616), (879, 916)):
        line(x1, 275, x2, 275, True)
    text(20, 346, "Both search stages use the sparse representation; no full complex response matrix is written.", 15, anchor="start")
    pieces.extend(["</g>", "</svg>"])
    target = ROOT / "docs/assets"
    target.mkdir(exist_ok=True)
    (target / "beam24-overview.svg").write_text("\n".join(pieces) + "\n")
    manifest = {
        "kind": "illustrative phase example and schematic primary K512 pipeline",
        "phase_step_radians": phase_step,
        "mode_energy_fractions": energy.tolist(),
        "retained_modes": support,
        "transform_source": "src/quality/spib_vla_screen.py",
        "transform_source_sha256": hashlib.sha256((ROOT / "src/quality/spib_vla_screen.py").read_bytes()).hexdigest(),
        "scope": "local K4 support, not an MMA tile or a measured real-array recording",
    }
    (target / "beam24-overview.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
