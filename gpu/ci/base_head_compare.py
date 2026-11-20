"""Compares the two commits of ``base_head.sh``, pass by pass: frames, keys, cache look-ups and
published files must be equal, and the warm pass must take plays from the cache.

Every pass runs under one hash seed (``base_head.sh``). A scene whose partial-movie keys still
differ between two passes of one commit leaves the comparison incomplete, and it fails: keys that
move from process to process can't show that the head moved none.

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


def moving(work: Path, arm: str) -> set[str]:
    """The scenes whose partial-movie keys differ between ``arm``'s two passes."""
    cold, warm = (
        json.loads((work / f"{arm}-{pass_}.json").read_text())["keys"]
        for pass_ in ("cold", "warm")
    )
    return {name for name in cold if cold[name] != warm.get(name)}


def main() -> None:
    work, base, head = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    problems: list[str] = []
    for arm in ("base", "head"):
        problems.extend(
            f"incomplete evidence: the partial-movie keys of {name} differ between two"
            f" passes at the {arm} under one hash seed"
            for name in sorted(moving(work, arm))
        )
    for pass_ in ("cold", "warm"):
        a = json.loads((work / f"base-{pass_}.json").read_text())
        b = json.loads((work / f"head-{pass_}.json").read_text())
        for part, said in PARTS.items():
            names = sorted(set(a[part]) | set(b[part]))
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
    print(
        f"base-head: frames, keys, cache hits and published files equal at {base[:8]} and"
        f" {head[:8]}, cold and warm"
    )


if __name__ == "__main__":
    main()
