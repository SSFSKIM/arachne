"""The engine's existing constructors need no manifest: an SVG and an image
opened by path draw exactly as before, whether or not a ``manifest.json``
sits beside them, and whether or not a scene that used an asset ran before
them in the same process; that scene poses its rig, alone and while it
moves, with a callout on one of its anchors and a readout counting, the rig
drawn on twelve drawings a second with line boil. Nor does a scene with an
effect between them change them: here, with no GPU, it answers its states
and refuses its frames (``tests/effects/test_drawn.py`` draws it on a
device). A scene of media objects runs between them too, and one whose clip
runs on its clock: here, on a host with no GPU, their states are answered
and their frames refused (``render.gpu_required``);
``tests/media/test_drawn.py`` draws them on a device between them.
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
        set_timing(box, 12, boil=Boil(rate=6))
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

WITH_MEDIA = """\
from arachne import *


class WithMedia(Scene):
    def construct(self):
        clip = VideoMobject("fixture:numbered-clip")
        still = Still("fixture:swatch").shift(DOWN)
        self.add(still)
        self.play(FadeIn(clip), still.animate.shift(RIGHT), run_time=0.5)
        self.wait(0.25)
"""

WITH_CLOCK = """\
from arachne import *


class WithClock(Scene):
    def construct(self):
        clip = VideoMobject("fixture:numbered-clip")
        clip.run()
        self.play(FadeIn(clip), run_time=0.5)
        clip.loop(rate=-1)
        self.wait(0.25)
"""

#: Opens each old scene, then a scene that used an asset, one with an effect,
#: one of media objects and one whose clip runs on its clock, then each old
#: scene again, in one process, on a host with no GPU; prints each opening's
#: routes and frames, and what the effect and media scenes answered.
PROGRAM = """\
import hashlib, json, sys
from arachne.compose import device
from arachne.routing import GpuRequired
from arachne.session import Session


class NoGpu:
    def pin(self):
        return None

    def open(self, width, height):
        return None


device.door = NoGpu()

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


def media_answers():
    session = Session.open(f"{directory}/with_media.py", "WithMedia", config=config)
    try:
        media = {"Still", "VideoMobject"}
        kinds = [
            sorted(media & {n.type for n in session.state_at(t).nodes}) for t in (0, 0.6)
        ]
        try:
            session.frame(0.25)
        except GpuRequired as refused:
            frame = refused.code
        else:
            frame = "drawn"
        return {"route": session.route, "kinds": kinds, "frame": frame}
    finally:
        session.close()


def clock_answers():
    session = Session.open(f"{directory}/with_clock.py", "WithClock", config=config)
    try:
        times = []
        for t in (0.25, 0.75):
            state = session.state_at(t)
            (clip,) = [n for n in state.nodes if n.type == "VideoMobject"]
            numbers = {q.name: q.value for q in state.quantities[clip.id]}
            times.append([str(state.as_of), numbers["media.time"]])
        try:
            session.frame(0.25)
        except GpuRequired as refused:
            frame = refused.code
        else:
            frame = "drawn"
        return {"route": session.route, "times": times, "frame": frame}
    finally:
        session.close()


fresh = {scene: opened("old.py", scene) for scene in ("OldSvg", "OldImage")}
asset = opened("with_asset.py", "WithAsset")
effect = refused("with_effect.py", "WithEffect")
media = media_answers()
clock = clock_answers()
after = {scene: opened("old.py", scene) for scene in ("OldSvg", "OldImage")}
print(
    json.dumps(
        {
            "fresh": fresh,
            "asset": asset,
            "effect": effect,
            "media": media,
            "clock": clock,
            "after": after,
        }
    )
)
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
    (directory / "with_media.py").write_text(textwrap.dedent(WITH_MEDIA))
    (directory / "with_clock.py").write_text(textwrap.dedent(WITH_CLOCK))
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
    assert bare["media"] == {
        "route": "gpu",
        "kinds": [["Still", "VideoMobject"]] * 2,
        "frame": "render.gpu_required",
    }
    # At 4 frames a second: the clip runs forward, then loops backward from
    # where it stands at 0.5 s.
    assert bare["clock"] == {
        "route": "gpu",
        "times": [["1/4", 0.25], ["3/4", 0.25]],
        "frame": "render.gpu_required",
    }
