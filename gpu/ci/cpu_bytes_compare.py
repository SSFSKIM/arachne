"""Compares the two arms of ``cpu_bytes.sh``: the frame digests must be identical, and each arm's
tripwire log must hold the one deliberate import and nothing else.

``python3 cpu_bytes_compare.py <without.json> <with.json> <without.log> <with.log> <marker>``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def records(log: Path) -> list[dict]:
    return [json.loads(line) for line in log.read_text().splitlines() if line.strip()]


def main() -> None:
    without, with_gpu, log_without, log_with, marker = sys.argv[1:]
    a, b = json.loads(Path(without).read_text()), json.loads(Path(with_gpu).read_text())
    problems = []
    problems += [
        f"{path}: the frames differ with and without the GPU build"
        for path in sorted(set(a) | set(b))
        if a.get(path) != b.get(path)
    ]
    for log in (log_without, log_with):
        found = records(Path(log))
        deliberate = [r for r in found if marker in r["command"]]
        others = [r for r in found if marker not in r["command"]]
        print(f"tripwire {log}: {len(deliberate)} deliberate, {len(others)} other")
        if len(deliberate) != 1:
            problems.append(
                f"{log}: the deliberate import left {len(deliberate)} records, not 1"
            )
        problems += [f"{log}: imported by {r['pid']}: {r['command']}" for r in others]
    for problem in problems:
        print(f"FAIL: {problem}")
    if problems:
        sys.exit(1)
    frames = sum(repeat for frames in a.values() for _, repeat in frames)
    scenes = len({path.split(" ", 1)[1] for path in a})
    print(
        f"cpu-bytes: {frames} frames of {scenes} scenes byte-identical with and without the GPU build"
    )


if __name__ == "__main__":
    main()
