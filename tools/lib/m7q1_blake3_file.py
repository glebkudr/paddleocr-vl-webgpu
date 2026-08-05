#!/usr/bin/env python3
"""Memory-bounded whole-file BLAKE3 helper for the Node artifact tooling."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from blake3 import blake3


def file_signature(stat: os.stat_result) -> tuple[int, int, int, int]:
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: m7q1_blake3_file.py FILE")

    path = Path(sys.argv[1])
    digest = blake3()
    buffer = bytearray(4 * 1024 * 1024)
    with path.open("rb", buffering=0) as source:
        before = os.fstat(source.fileno())
        while read := source.readinto(buffer):
            digest.update(memoryview(buffer)[:read])
        after = os.fstat(source.fileno())

    if file_signature(before) != file_signature(after):
        raise RuntimeError(f"file changed while hashing: {path}")

    print(
        json.dumps(
            {
                "blake3": digest.hexdigest(),
                "bytes": after.st_size,
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
