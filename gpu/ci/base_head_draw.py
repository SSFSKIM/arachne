"""One pass of ``base_head.sh``: renders ``cpu_bytes_scenes`` at 480 x 270 into ``<roots>``, and
draws each through a session's ``frame``, with the engine of ``<tree>``; writes what a comparison
of two commits reads.

``python base_head_draw.py <tree> <roots> <out.json>``, from ``<tree>``, with the interpreter of
the environment made from it. A second pass into the same ``<roots>`` finds the cache the first
left.

The JSON holds, per scene: the sha256 of every raw frame as it reaches the file writer (``frames``)
or as the session returns it (``session``); each play's partial-movie key (``keys``); each look-up
of the cache and whether it found the segment (``lookups``). And ``files``: every file the renders
published under ``videos/`` and ``images/`` of each scene's media directory, by path, with the
sha256 of its bytes once ``<roots>`` is written as ``<roots>`` in them (the concat manifest names
the segments by their absolute paths).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def digest(pixels) -> str:
    import numpy as np

    array = np.ascontiguousarray(pixels)
    return hashlib.sha256(
        f"{array.shape} {array.dtype} ".encode() + array.tobytes()
    ).hexdigest()


def published(roots: Path) -> dict[str, str]:
    marker = str(roots).encode()
    found: dict[str, str] = {}
    media = roots / "media"
    for scene in sorted(media.iterdir()) if media.exists() else ():
        for part in ("videos", "images"):
            for path in sorted((scene / part).rglob("*")):
                if path.is_file():
                    data = path.read_bytes().replace(marker, b"<roots>")
                    name = path.relative_to(roots).as_posix()
                    found[name] = hashlib.sha256(data).hexdigest()
    return found


def main() -> None:
    tree, roots, out = (Path(argument).resolve() for argument in sys.argv[1:4])
    os.environ["ARACHNE_STORE"] = str(roots / "store")
    sys.path[:0] = [str(tree), str(HERE)]

    import cpu_bytes_scenes as scenes
    import numpy as np

    import arachne
    from arachne import tempconfig
    from arachne.scene.scene_file_writer import SceneFileWriter
    from arachne.session import Session

    assert Path(arachne.__file__).resolve().is_relative_to(tree), arachne.__file__

    written: list[list] = []
    lookups: list[list] = []
    write_frame, save_image = SceneFileWriter.write_frame, SceneFileWriter.save_image
    is_already_cached = SceneFileWriter.is_already_cached

    def hashed_write_frame(self, pixels, *, repeat=1):
        written.append([digest(pixels), repeat])
        return write_frame(self, pixels, repeat=repeat)

    def hashed_save_image(self, pixels):
        written.append([digest(pixels), 1])
        return save_image(self, pixels)

    def noted_is_already_cached(self, key):
        found = is_already_cached(self, key)
        lookups.append([key, found])
        return found

    SceneFileWriter.write_frame = hashed_write_frame
    SceneFileWriter.save_image = hashed_save_image
    SceneFileWriter.is_already_cached = noted_is_already_cached

    result: dict[str, dict] = {"frames": {}, "session": {}, "keys": {}, "lookups": {}}
    for scene in scenes.SCENES:
        name = scene.__name__
        config = {
            "pixel_width": 480,
            "pixel_height": 270,
            "frame_rate": 15,
            "media_dir": str(roots / "media" / name),
        }
        with tempconfig(config):
            made = scene()
            made.render()
        result["frames"][name] = written[:]
        result["keys"][name] = list(made.renderer.animations_hashes)
        result["lookups"][name] = lookups[:]
        written.clear()
        lookups.clear()

        session = Session.open(Path(scenes.__file__), name, config=config)
        try:
            result["session"][name] = [
                digest(np.asarray(session.frame(t))) for t in scenes.INSTANTS
            ]
        finally:
            session.close()

    result["files"] = published(roots)
    # A warm pass may take every play from the cache and write no frame.
    empty = [
        name
        for name, frames in result["frames"].items()
        if not frames and not any(found for _, found in result["lookups"][name])
    ]
    if empty:
        sys.exit(f"FAIL: nothing rendered or found in the cache for {', '.join(empty)}")
    out.write_text(json.dumps(result, indent=1))
    hits = sum(
        found for found_list in result["lookups"].values() for _, found in found_list
    )
    print(
        f"{sum(len(f) for f in result['frames'].values())} frame writes,"
        f" {hits} plays from the cache, {len(result['files'])} files published"
    )


if __name__ == "__main__":
    main()
