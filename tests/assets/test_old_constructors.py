"""The engine's existing constructors need no manifest: an SVG and an image
opened by path draw exactly as before, whether or not a ``manifest.json``
sits beside them, and whether or not a scene that used an asset ran before
them in the same process; that scene poses its rig, alone and while it
moves, with a callout on one of its anchors and a readout counting. Nor
does a scene with an effect between them change them: here, with no GPU,
it answers its states and refuses its frames
(``tests/effects/test_drawn.py`` draws it on a device).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import arachne

RESOURCES = Path(__file__).parent.parent / "test_graphical_units" / "img_svg_resources"

OLD = """\
from pathlib import Path

from arachne import *

HERE = Path(__file__).parent


class OldSvg(Scene):
    def construct(self):
        heart = SVGMobject(str(HERE / "heart.svg"))
        self.add(heart)
        self.play(heart.animate.shift(RIGHT).scale(0.5), run_time=0.5)
        self.wait(0.25)


class OldImage(Scene):
    def construct(self):
        tree = ImageMobject(str(HERE / "tree.png")).scale(0.5)
        self.add(tree)
        self.play(tree.animate.shift(LEFT), run_time=0.5)
        self.wait(0.25)
"""

WITH_ASSET = """\
from arachne import *


class WithAsset(Scene):
    def construct(self):
        box = Rig("fixture:lidded-box")
        knob = Callout("knob", at=box.anchor("knob"))
        reading = Readout(0, format="{:.0f} deg", at=anchor(box), offset=DOWN)
        self.add(box, knob, reading)
        self.play(box.animate.shift(RIGHT), run_time=0.5)
        self.play(
            Pose(box, lid_open=80),
            CountUp(reading, 80),
            box.animate.shift(UP),
            run_time=0.5,
        )
        box.pose(lid_open=30)
        self.wait(0.25)
"""

WITH_EFFECT = """\
from arachne import *


class WithEffect(Scene):
    def construct(self):
        square = Square(fill_opacity=1)
        set_blur(square, 0.1)
        self.add(square)
        self.play(ChangeBlur(square, 0.3), run_time=0.5)
        set_mask(square, Circle(fill_opacity=1), feather=0.1)
        self.wait(0.25)
"""

#: Opens each old scene, then a scene that used an asset, then each old
#: scene again, in one process; prints each opening's routes and frames.
PROGRAM = """\
import hashlib, json, sys
from arachne.session import Session

directory, media = sys.argv[1], sys.argv[2]
config = {"pixel_width": 96, "pixel_height": 54, "frame_rate": 4, "media_dir": media}


def opened(file, scene):
    session = Session.open(f"{directory}/{file}", scene, config=config)
    try:
        return {
            "routes": [play.route for play in session.record.plays],
            "frames": [
                hashlib.sha256(session.frame(t).tobytes()).hexdigest()
                for t in (0, 0.25, 0.5, 0.7)
            ],
        }
    finally:
        session.close()


def refused(file, scene):
    session = Session.open(f"{directory}/{file}", scene, config=config)
    try:
        states = [len(session.state_at(t).nodes) for t in (0, 0.25, 0.5, 0.7)]
        try:
            session.frame(0.25)
        except Exception as error:
            frame = getattr(error, "code", type(error).__name__)
        else:
            frame = None
        return {"route": session.route, "states": states, "frame": frame}
    finally:
        session.close()


fresh = {scene: opened("old.py", scene) for scene in ("OldSvg", "OldImage")}
asset = opened("with_asset.py", "WithAsset")
effect = refused("with_effect.py", "WithEffect")
after = {scene: opened("old.py", scene) for scene in ("OldSvg", "OldImage")}
print(json.dumps({"fresh": fresh, "asset": asset, "effect": effect, "after": after}))
"""


def _openings(directory: Path) -> dict:
    """What a fresh process gets opening the scenes in ``directory``."""
    root = Path(arachne.__file__).parent.parent
    done = subprocess.run(
        [sys.executable, "-c", PROGRAM, str(directory), str(directory / "media")],
        capture_output=True,
        text=True,
        cwd=root,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


def _scenes(directory: Path) -> Path:
    directory.mkdir()
    shutil.copy(RESOURCES / "heart.svg", directory / "heart.svg")
    shutil.copy(RESOURCES / "tree_img_640x351.png", directory / "tree.png")
    (directory / "old.py").write_text(textwrap.dedent(OLD))
    (directory / "with_asset.py").write_text(textwrap.dedent(WITH_ASSET))
    (directory / "with_effect.py").write_text(textwrap.dedent(WITH_EFFECT))
    return directory


def test_an_svg_and_an_image_by_path_draw_the_same_with_or_without_a_manifest(
    tmp_path,
):
    bare = _openings(_scenes(tmp_path / "bare"))
    beside = _scenes(tmp_path / "beside")
    shutil.copy(
        Path(arachne.__file__).parent
        / "assets"
        / "fixtures"
        / "dial"
        / "manifest.json",
        beside / "manifest.json",
    )
    with_manifest = _openings(beside)
    for openings in (bare, with_manifest):
        assert openings["fresh"] == openings["after"]
        assert all(len(set(o["frames"])) > 1 for o in openings["fresh"].values())
        assert openings["effect"]["route"] == "gpu"
        assert all(openings["effect"]["states"])
        assert openings["effect"]["frame"] == "render.gpu_required"
    assert bare["fresh"] == with_manifest["fresh"]
    assert bare["asset"] == with_manifest["asset"]
