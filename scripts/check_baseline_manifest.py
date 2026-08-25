#!/usr/bin/env python3
"""Validate the pinned Beam24 baseline inventory."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "baselines" / "manifest.json"
REQUIRED_BASELINES = {
    "DENSE_REFERENCE",
    "ABL_DIRECT_ANTENNA_JOINT_2OF4",
    "ABL_LOCAL_F4_UNSTRUCTURED50",
    "ABL_LOCAL_F4_INDEPENDENT_RI",
    "CCGLIB_BASIC",
    "CCGLIB_OPT_STATIC_A",
    "CCGLIB_FULL_PIPELINE",
    "CUBLASLT_COMPLEX_4GEMM",
    "CUSPARSELT_REAL_PRIMITIVE",
    "CUSPARSELT_PRUNE_COMPLEX",
    "BEAM24_REP_CUSPARSELT",
    "CCGLIB_MATERIALIZED_TOP1",
    "CCGLIB_HIERARCHICAL_TOP1",
    "CUFFT_STREAMED_UNIFORM_ANGLE_TOP1",
    "DENSE_FUSED_TOP1_CONTROL",
}
VALID_STATES = {"measured", "measured_primitive", "partial", "pending"}
VALID_OWNERSHIP = {
    "external", "hybrid_ablation", "internal_reference",
    "internal_ablation", "internal_control"
}


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    if manifest.get("version") != 6:
        raise SystemExit("unsupported baseline manifest version")

    sources = manifest["external_sources"]
    source_ids = [source["id"] for source in sources]
    if len(source_ids) != len(set(source_ids)):
        raise SystemExit("duplicate external source id")
    for source in sources:
        if "commit" in source and not re.fullmatch(r"[0-9a-f]{40}", source["commit"]):
            raise SystemExit(f"invalid commit for {source['id']}")
        if "repository" in source and not source["repository"].startswith("https://"):
            raise SystemExit(f"non-HTTPS repository for {source['id']}")

    baselines = manifest["baselines"]
    baseline_ids = [baseline["id"] for baseline in baselines]
    if set(baseline_ids) != REQUIRED_BASELINES or len(baseline_ids) != len(set(baseline_ids)):
        raise SystemExit("baseline id set does not match the frozen suite")
    for baseline in baselines:
        if baseline["status"] not in VALID_STATES:
            raise SystemExit(f"invalid status for {baseline['id']}")
        if baseline["ownership"] not in VALID_OWNERSHIP:
            raise SystemExit(f"invalid ownership for {baseline['id']}")
        if baseline.get("source") not in {None, *source_ids}:
            raise SystemExit(f"unknown source for {baseline['id']}")
        implementation = baseline.get("implementation")
        if implementation and not (ROOT / implementation).is_file():
            raise SystemExit(f"missing implementation for {baseline['id']}: {implementation}")
    incomplete_required = [
        b["id"] for b in baselines
        if b["required"] and b["status"] in {"pending", "partial"}
    ]
    if incomplete_required:
        raise SystemExit(f"unexpected incomplete required baselines: {incomplete_required}")
    external_main = {
        baseline["id"]
        for baseline in baselines
        if baseline["ownership"] == "external" and baseline["table_role"] == "main"
    }
    if external_main != {
        "CUBLASLT_COMPLEX_4GEMM",
        "CUSPARSELT_PRUNE_COMPLEX",
        "CCGLIB_MATERIALIZED_TOP1",
        "CCGLIB_HIERARCHICAL_TOP1",
        "CUFFT_STREAMED_UNIFORM_ANGLE_TOP1",
    }:
        raise SystemExit(f"unexpected external main set: {sorted(external_main)}")
    print(
        f"baseline manifest: {len(sources)} sources, {len(baselines)} entries, "
        "required suite complete [OK]"
    )


if __name__ == "__main__":
    main()
