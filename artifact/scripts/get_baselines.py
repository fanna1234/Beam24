#!/usr/bin/env python3
"""Fetch pinned external baseline sources without modifying existing checkouts."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "baselines" / "manifest.json"


def run(*args: str, dry_run: bool = False) -> None:
    print("+", " ".join(args))
    if not dry_run:
        subprocess.run(args, check=True)


def current_commit(path: Path) -> str | None:
    if not (path / ".git").is_dir():
        return None
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return result.stdout.strip()


def fetch(source: dict[str, object], destination: Path, dry_run: bool) -> None:
    source_id = str(source["id"])
    repository = str(source["repository"])
    commit = str(source["commit"])
    checkout = destination / source_id
    if checkout.exists():
        actual = current_commit(checkout)
        if actual != commit:
            raise SystemExit(
                f"refusing to overwrite {checkout}: expected {commit}, found {actual}"
            )
        print(f"{source_id}: already pinned at {commit}")
        return

    run("git", "init", str(checkout), dry_run=dry_run)
    run("git", "-C", str(checkout), "remote", "add", "origin", repository, dry_run=dry_run)
    sparse_paths = [str(path) for path in source.get("sparse_paths", [])]
    if sparse_paths:
        run("git", "-C", str(checkout), "sparse-checkout", "init", "--cone", dry_run=dry_run)
        run("git", "-C", str(checkout), "sparse-checkout", "set", *sparse_paths, dry_run=dry_run)
    run("git", "-C", str(checkout), "fetch", "--depth", "1", "origin", commit, dry_run=dry_run)
    run("git", "-C", str(checkout), "checkout", "--detach", "FETCH_HEAD", dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", help="source id or 'all'")
    parser.add_argument("--root", type=Path, default=ROOT / "work" / "baselines")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    sources = [source for source in manifest["external_sources"] if "repository" in source]
    selected = sources if args.target == "all" else [s for s in sources if s["id"] == args.target]
    if not selected:
        valid = ", ".join(str(source["id"]) for source in sources)
        raise SystemExit(f"unknown target {args.target!r}; choose one of: {valid}, all")
    if not args.dry_run:
        args.root.mkdir(parents=True, exist_ok=True)
    for source in selected:
        fetch(source, args.root.resolve(), args.dry_run)


if __name__ == "__main__":
    main()
