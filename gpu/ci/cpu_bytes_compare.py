"""Compares the arms of the bytes comparison (``cpu_bytes_arm.sh``): every arm's frame digests
must be identical, each arm's tripwire log must hold its one deliberate import and nothing else,
and at least one arm must be without the GPU build and one with it.

    python3 cpu_bytes_compare.py [--no-hardware ARM]... [--hardware ARM=ADAPTER]...
        <arm directory> <arm directory> [<arm directory>...]

What an arm with the GPU build saw can be held to as well, an arm named by its directory's name:
``--no-hardware ARM``, that it saw no adapter but software ones; ``--hardware ARM=ADAPTER``, that
it saw a hardware adapter of that name. An arm without the build has nothing to see with, and
neither can be asked of it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tripwire import MARKER


def records(log: Path) -> list[dict]:
    return [json.loads(line) for line in log.read_text().splitlines() if line.strip()]


def seen(arms: dict[str, dict], name: str, problems: list[str]) -> list[dict] | None:
    """The adapters arm ``name`` saw; None, with the problem noted, where it cannot say."""
    if name not in arms:
        problems.append(f"no arm is named {name}")
        return None
    if arms[name]["gpu_build"] != "present":
        problems.append(f"{name}: without the GPU build it sees no adapter to hold to")
        return None
    return arms[name]["adapters"]


def hardware(adapters: list[dict]) -> list[dict]:
    return [a for a in adapters if a["device_type"] != "Cpu"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("arms", nargs="+", type=Path)
    parser.add_argument("--no-hardware", action="append", default=[], metavar="ARM")
    parser.add_argument(
        "--hardware", action="append", default=[], metavar="ARM=ADAPTER"
    )
    options = parser.parse_args()
    arms = options.arms
    if len(arms) < 2:
        parser.error("compare two arms or more")
    if len({arm.name for arm in arms}) != len(arms):
        parser.error("each arm's directory needs a name of its own")
    digests = {arm: json.loads((arm / "digests.json").read_text()) for arm in arms}
    records_of = {arm.name: json.loads((arm / "arm.json").read_text()) for arm in arms}
    builds = {arm: records_of[arm.name]["gpu_build"] for arm in arms}
    first = arms[0]
    problems = []
    for name in options.no_hardware:
        adapters = seen(records_of, name, problems)
        if adapters is not None and hardware(adapters):
            problems.append(f"{name}: saw a hardware adapter: {hardware(adapters)}")
    for expectation in options.hardware:
        name, _, adapter = expectation.partition("=")
        adapters = seen(records_of, name, problems)
        if adapters is not None and adapter not in [
            a["name"] for a in hardware(adapters)
        ]:
            problems.append(
                f"{name}: saw no hardware adapter named {adapter}: {adapters}"
            )
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
        raise SystemExit(1)
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
