#!/usr/bin/env python3
"""Reject external compute activity before running a single-GPU campaign."""

from __future__ import annotations

import json
import os
import subprocess


def foreign_processes(text: str) -> list[str]:
    allowed = ("gnome-remote-desktop-daemon", "Xorg")
    return [line.strip() for line in text.splitlines()
            if line.strip() and not any(name in line for name in allowed)]


def assert_idle() -> None:
    selector = os.environ.get("CUDA_VISIBLE_DEVICES")
    if selector is not None and (not selector.strip() or "," in selector):
        raise RuntimeError("select one GPU with CUDA_VISIBLE_DEVICES, preferably its UUID")
    selection = ["--id", selector] if selector else []
    result = subprocess.run(
        ["nvidia-smi", *selection, "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        check=True, text=True, capture_output=True,
    )
    foreign = foreign_processes(result.stdout)
    if foreign:
        raise RuntimeError("GPU is occupied; no benchmark was launched: " + json.dumps(foreign))


if __name__ == "__main__":
    assert_idle()
