#!/usr/bin/env python3
"""Check the ten extracted SPIB recordings against their pinned source archive."""

import argparse
from pathlib import Path
import zipfile
import zlib


def verify(archive: Path, directory: Path) -> None:
    with zipfile.ZipFile(archive) as source:
        members = [info for info in source.infolist() if info.filename.endswith(".mat")]
        if len(members) != 10:
            raise ValueError("expected ten SPIB recordings in the source archive")
        expected = set()
        for info in members:
            target = (directory / info.filename).resolve()
            if not target.is_relative_to(directory.resolve()):
                raise ValueError("archive member escapes the extraction directory")
            expected.add(target)
            if not target.is_file() or target.stat().st_size != info.file_size:
                raise ValueError(f"missing or incomplete extracted recording: {target.name}")
            crc = 0
            with target.open("rb") as stream:
                for block in iter(lambda: stream.read(1 << 20), b""):
                    crc = zlib.crc32(block, crc)
            if crc != info.CRC:
                raise ValueError(f"extracted recording differs from source archive: {target.name}")
        if {path.resolve() for path in directory.glob("*.mat")} != expected:
            raise ValueError("unexpected SPIB recording set")
    print("[OK] all ten SPIB recordings match the source archive")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    verify(args.archive, args.directory)
