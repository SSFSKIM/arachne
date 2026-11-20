"""Compares the two commits of ``base_head.sh``, pass by pass: frames, keys, cache look-ups and
published files must be equal, and the warm pass must take plays from the cache.

A scene whose partial-movie keys move between the two passes of the base itself (a key reading
something that differs from process to process) has no keys to compare: its keys, look-ups and
segment files are left out of the comparison, its frames and movie are not. The head must move
the keys of no other scene.

``python3 base_head_compare.py <work> <base> <head>``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PARTS = {
    "frames": "raw frames",
    "session": "session frames",
    "keys": "partial-movie keys",
    "lookups": "cache look-ups",
    "files": "published files",
}


#: What a scene with moving keys is not compared by.
KEYED = ("keys", "lookups")


def moving(work: Path, arm: str) -> set[str]:
    """The scenes whose partial-movie keys differ between ``arm``'s two passes."""
    cold, warm = (
        json.loads((work / f"{arm}-{pass_}.json").read_text())["keys"]
        for pass_ in ("cold", "warm")
    )
    return {name for name in cold if cold[name] != warm.get(name)}


def segment_of(name: str, scenes: set[str]) -> bool:
    """Whether the published file ``name`` is a segment file of one of ``scenes``."""
    return "/partial_movie_files/" in name and name.split("/")[1] in scenes


def main() -> None:
    work, base, head = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    problems: list[str] = []
    unkeyed = moving(work, "base")
    problems.extend(
        f"partial-movie keys of {name} move between passes at the head only"
        for name in sorted(moving(work, "head") - unkeyed)
    )
    for pass_ in ("cold", "warm"):
        a = json.loads((work / f"base-{pass_}.json").read_text())
        b = json.loads((work / f"head-{pass_}.json").read_text())
        for part, said in PARTS.items():
            names = sorted(set(a[part]) | set(b[part]))
            if part in KEYED:
                names = [name for name in names if name not in unkeyed]
            if part == "files":
                names = [name for name in names if not segment_of(name, unkeyed)]
            problems.extend(
                f"{pass_}: {said} of {name} differ"
                for name in names
                if a[part].get(name) != b[part].get(name)
            )
        hits = sum(
            found for found_list in b["lookups"].values() for _, found in found_list
        )
        print(f"{pass_}: {hits} plays from the cache at the head")
        if pass_ == "warm" and not hits:
            problems.append(
                "warm: no play was taken from the cache, so nothing was compared"
            )
    for problem in problems:
        print(f"FAIL: {problem}")
    if problems:
        sys.exit(1)
    aside = (
        f" (keys of {', '.join(sorted(unkeyed))} move between passes at the base too:"
        " compared by frames and movie)"
        if unkeyed
        else ""
    )
    print(
        f"base-head: frames, keys, cache hits and published files equal at {base[:8]} and"
        f" {head[:8]}, cold and warm{aside}"
    )


if __name__ == "__main__":
    main()
