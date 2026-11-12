"""Reads each XF stream the newer revision wrote with this revision's core (the
older reader), and the same case's stream this revision wrote, and draws every
frame of both: the frames must be the same bytes.

    python old_reader.py <head xf dir> <base xf dir> <out.json>
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import arachne_core
import numpy as np


def frames(path: Path) -> list[str]:
    found = []
    stream = arachne_core.Stream()
    size = None
    try:
        deltas = arachne_core.read_xf(path)
    except Exception as error:
        return [f"read: {type(error).__name__}: {error}"]
    for delta in deltas:
        view = delta.contents()["view"]
        if view is not None:
            size = (view["pixel_width"], view["pixel_height"])
        try:
            stream.apply(delta)
        except Exception as error:
            found.append(f"apply: {type(error).__name__}: {error}")
            continue
        if size is None:
            found.append("no view")
            continue
        try:
            pixels = stream.draw(*size, threads=1, overlay=False)
        except Exception as error:
            found.append(f"draw: {type(error).__name__}: {str(error)[:120]}")
            continue
        data = np.ascontiguousarray(np.asarray(pixels)).tobytes()
        found.append(hashlib.sha256(data).hexdigest())
    return found


def main() -> None:
    head, base, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    print("reader", arachne_core.XF_VERSION, arachne_core.__file__)
    result = {"reader": list(arachne_core.XF_VERSION), "streams": 0, "frames": 0}
    differ, missing, versions = [], [], {}
    for path in sorted(head.glob("*.xf")):
        minor = int.from_bytes(path.read_bytes()[6:8], "little")
        versions[minor] = versions.get(minor, 0) + 1
        mine = base / path.name
        if not mine.exists():
            missing.append(path.name)
            continue
        new, old = frames(path), frames(mine)
        result["streams"] += 1
        result["frames"] += len(new)
        if new != old:
            differ.append({"stream": path.name, "new": new[:3], "old": old[:3]})
    result.update(
        differ=differ, missing=missing, head_file_minors=versions, same=not differ
    )
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "differ"}))
    print("streams differing:", len(differ))
    for d in differ[:20]:
        print(" ", d)


if __name__ == "__main__":
    main()
