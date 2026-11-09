"""A session's render, in a process of its own: ``python -m
arachne.server.render_child``, the request read as JSON from standard input.

The session server starts it for a render drawn by workers
(:mod:`.render`): a process with one thread and none of the registry's
locks, from which the render's workers are forked. The request names the
scene's file and class, the render's options, the name the caller gave the
file, and two paths: ``result``, where the render's video, the plays its
execution played and the summary of that execution are written as JSON
once it is whole; ``failure``, where what it raised is written instead
(:func:`.render.failure`), with exit status 1.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


def main() -> int:
    from .render import failure, rendered

    request: dict[str, Any] = json.loads(sys.stdin.read())
    try:
        video, timeline, summary = rendered(
            Path(request["file"]),
            request["scene"],
            request["options"],
            request["named"],
        )
    except BaseException as error:  # noqa: BLE001 (every failure is the server's to answer)
        _write(Path(request["failure"]), failure(error))
        return 1
    found = {
        "video": os.fspath(video),
        "timeline": timeline,
        "summary": summary.to_json(),
    }
    _write(Path(request["result"]), json.dumps(found).encode())
    return 0


def _write(path: Path, data: bytes) -> None:
    """``data`` at ``path``, whole or not at all."""
    made = path.with_name(path.name + ".part")
    made.write_bytes(data)
    os.replace(made, path)


if __name__ == "__main__":
    sys.exit(main())
