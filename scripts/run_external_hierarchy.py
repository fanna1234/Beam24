#!/usr/bin/env python3
"""Reproduce the fixed ccglib-versus-Beam24 identical-hierarchy comparison."""

from __future__ import annotations

import fcntl
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import tempfile

from check_gpu_idle import assert_idle
from paired_records import summarize_pairs, validate_pairs


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def logged(command: list[str], path: Path, kind: str) -> dict:
    result = subprocess.run(command, text=True, capture_output=True)
    path.write_text("$ " + " ".join(command) + "\n" + result.stdout + "\n" + result.stderr)
    if result.returncode:
        raise RuntimeError(f"command failed; see {path}")
    rows = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    selected = [row for row in rows if row.get("kind") == kind]
    if len(selected) != 1:
        raise RuntimeError(f"expected one {kind} record; see {path}")
    return selected[0]


def validate_timing(row: dict, variant: str) -> None:
    expected = {"batch": 256, "m": 1024, "n": 1024, "k": 512,
                "coarse_m": 128, "coarse_k": 128, "fine_m": 128,
                "warmup": 20, "iterations": 100}
    if any(row.get(key) != value for key, value in expected.items()):
        raise RuntimeError("timing record differs from the fixed shape/schedule")
    if variant == "E" and (row.get("engine"), row.get("launch_mode")) != ("ccglib_basic", "direct"):
        raise RuntimeError("external comparison must use dynamic-A ccglib basic with direct launches")
    if variant == "H" and (row.get("graph") is not False or row.get("dispatch") != "hierarchical"):
        raise RuntimeError("Beam24 comparison must use the direct hierarchical route")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="run output/operand admission without a performance campaign")
    args = parser.parse_args()
    build = Path(os.environ.get("BEAM24_BUILD_DIR", ROOT / "build/sm120"))
    external_build = Path(os.environ.get("BEAM24_EXTERNAL_BUILD_DIR", ROOT / "build/external"))
    external = external_build / "beam24_ccglib_hierarchy"
    beam24 = build / "beam24_hierarchical_system"
    for binary in (external, beam24):
        if not binary.is_file():
            raise FileNotFoundError(f"missing binary: {binary}")
    runs = ROOT / "artifact/runs"
    runs.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="external-hierarchy.", dir=runs))
    lock_path = os.environ.get("BEAM24_GPU_LOCK", "/tmp/beam24_gpu_campaign.lock")
    records = []
    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert_idle()
        correctness = {}
        for variant, binary, kind, arguments in (
            ("E", external, "ccglib_same_hierarchy_correctness", ["1", "0", "1", "481", "128"]),
            ("H", beam24, "beam24_hierarchy_correctness", ["1", "0", "1", "481", "128", "0", "0"]),
        ):
            assert_idle()
            row = logged([str(binary), *arguments], output / f"correctness-{variant}.log", kind)
            if row.get("status") != "PASSED":
                raise RuntimeError(f"{variant} correctness did not pass")
            correctness[variant] = row
        environment = subprocess.check_output(
            ["nvidia-smi", *(["--id", os.environ["CUDA_VISIBLE_DEVICES"]] if os.environ.get("CUDA_VISIBLE_DEVICES") else []),
             "--query-gpu=name,driver_version,compute_cap,power.limit,clocks.current.sm,clocks.current.memory", "--format=csv"],
            text=True,
        )
        (output / "environment.txt").write_text(environment)
        (output / "admission.json").write_text(json.dumps({
            "correctness": correctness,
            "binary_sha256": {"external": digest(external), "beam24": digest(beam24)},
            "ccglib_library_sha256": digest(external_build / "ccglib-build/src/libccglib.so"),
            "baseline_manifest_sha256": digest(ROOT / "baselines/manifest.json"),
        }, indent=2) + "\n")
        if args.check_only:
            print("[OK] external and Beam24 hierarchy correctness; no paired timing campaign")
            print(output)
            return
        for process in range(6):
            order = ("E", "H") if process % 2 == 0 else ("H", "E")
            for position, variant in enumerate(order):
                assert_idle()
                binary = external if variant == "E" else beam24
                args = ["100", "20", "0", "256", "1024"] + ([] if variant == "E" else ["0", "0"])
                kind = "ccglib_same_hierarchy_e2e" if variant == "E" else "beam24_hierarchical_e2e"
                row = logged([str(binary), *args], output / f"process-{process}-{position}-{variant}.log", kind)
                validate_timing(row, variant)
                row.update(process=process, position=position, variant=variant)
                records.append(row)
                with (output / "samples.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
    pairs = validate_pairs(records, ("E", "H"))
    result = summarize_pairs(pairs, "E", "H")
    anchors = json.loads((ROOT / "artifact/expected/reference_anchors.json").read_text())
    anchor = anchors["system"]["hierarchical_external_same_algorithm_speedup"]
    speedup = result["paired_geomean_speedup"]
    passed = speedup >= anchors["gates"]["within_fraction"] * anchor and result["process_wins"] == 6
    result.update(
        comparison_role="external_same_hierarchy", reference_anchor=anchor,
        status="[OK >=reference]" if speedup >= anchor else "[~within3%]" if passed else "[LOW]",
        external_median_ms=statistics.median(pair["E"] for pair in pairs),
        beam24_median_ms=statistics.median(pair["H"] for pair in pairs),
    )
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(output)
    if not passed:
        raise SystemExit(8)


if __name__ == "__main__":
    main()
