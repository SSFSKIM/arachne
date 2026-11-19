"""Draws ``cpu_bytes_scenes`` at 480 x 270 through the render path and through a session's
``frame``, and writes the sha256 of every raw frame: as it reaches the file writer for a render,
as the session returns it for a frame. Fails if a scene yields no frame from a path.

``python cpu_bytes_draw.py <roots> <digests.json>``, run from the repository's root with the
interpreter of the environment drawn in. ``<roots>`` gets this run's media, partial movies and
session store, so nothing another run drew is reused.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]


def digest(pixels) -> str:
    import numpy as np

    array = np.ascontiguousarray(pixels)
    return hashlib.sha256(
        f"{array.shape} {array.dtype} ".encode() + array.tobytes()
    ).hexdigest()


def main() -> None:
    roots, out = Path(sys.argv[1]), Path(sys.argv[2])
    os.environ["ARACHNE_STORE"] = str(roots / "store")
    sys.path[:0] = [str(REPOSITORY), str(HERE)]

    import cpu_bytes_scenes as scenes
    import numpy as np

    from arachne import tempconfig
    from arachne.scene.scene_file_writer import SceneFileWriter
    from arachne.session import Session

    written: list[list] = []
    write_frame, save_image = SceneFileWriter.write_frame, SceneFileWriter.save_image

    def hashed_write_frame(self, pixels, *, repeat=1):
        written.append([digest(pixels), repeat])
        return write_frame(self, pixels, repeat=repeat)

    def hashed_save_image(self, pixels):
        written.append([digest(pixels), 1])
        return save_image(self, pixels)

    SceneFileWriter.write_frame = hashed_write_frame
    SceneFileWriter.save_image = hashed_save_image

    digests: dict[str, list] = {}
    for scene in scenes.SCENES:
        name = scene.__name__
        config = {
            "pixel_width": 480,
            "pixel_height": 270,
            "frame_rate": 15,
            "media_dir": str(roots / "media" / name),
        }
        with tempconfig(config):
            scene().render()
        digests[f"render {name}"] = written[:]
        written.clear()

        session = Session.open(Path(scenes.__file__), name, config=config)
        try:
            digests[f"frame {name}"] = [
                [digest(np.asarray(session.frame(t))), 1] for t in scenes.INSTANTS
            ]
        finally:
            session.close()

    empty = [path for path, frames in digests.items() if not frames]
    if empty:
        sys.exit(f"FAIL: no frame from {', '.join(empty)}")
    out.write_text(json.dumps(digests, indent=1))
    for path, frames in digests.items():
        print(
            f"{path}: {sum(repeat for _, repeat in frames)} frames, {len(frames)} distinct writes"
        )


if __name__ == "__main__":
    main()
