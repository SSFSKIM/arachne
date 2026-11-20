"""Compares the arms of the bytes comparison (``cpu_bytes_arm.sh``): every arm's frame digests
must be identical, each arm's tripwire log must hold its one deliberate import and nothing else,
and at least one arm must be without the GPU build and one with it.

``python3 cpu_bytes_compare.py <arm directory> <arm directory> [<arm directory>...]``
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from tripwire import MARKER


def records(log: Path) -> list[dict]:
    return [json.loads(line) for line in log.read_text().splitlines() if line.strip()]


def main() -> None:
    arms = [Path(arm) for arm in sys.argv[1:]]
    if len(arms) < 2:
        sys.exit(__doc__)
    digests = {arm: json.loads((arm / "digests.json").read_text()) for arm in arms}
    builds = {
        arm: json.loads((arm / "arm.json").read_text())["gpu_build"] for arm in arms
    }
    first = arms[0]
    problems = []
    if set(builds.values()) != {"absent", "present"}:
        problems.append(
            f"the arms are not with and without the GPU build: {sorted(builds.values())}"
        )
    for arm in arms[1:]:
        a, b = digests[first], digests[arm]
        problems += [
            f"{path}: the frames differ between {first.name} and {arm.name}"
            for path in sorted(set(a) | set(b))
            if a.get(path) != b.get(path)
        ]
    for arm in arms:
        log = arm / "tripwire.log"
        found = records(log)
        deliberate = [r for r in found if MARKER in r["command"]]
        others = [r for r in found if MARKER not in r["command"]]
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
    listed = digests[first]
    frames = sum(repeat for frames in listed.values() for _, repeat in frames)
    scenes = len({path.split(" ", 1)[1] for path in listed})
    canonical = json.dumps(listed, sort_keys=True, separators=(",", ":")).encode()
    print(f"arms: {', '.join(f'{arm.name} (build {builds[arm]})' for arm in arms)}")
    print(f"frames sha256: {hashlib.sha256(canonical).hexdigest()}")
    print(
        f"cpu-bytes: {frames} frames of {scenes} scenes byte-identical with and without the GPU build"
    )


if __name__ == "__main__":
    main()
