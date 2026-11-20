"""A scene that declares no timing plays and draws as it did before stepped
timing existed: it gets no clock and no restorer, its rig's pose settles
and its flows' driver run as the scene's own updaters, and its plays'
routes and frames are the same opened fresh and opened after a declared
scene in the same process. A declaration ends with its scene: a mobject
declared there, a copy of it, and a mobject a group outliving the scene
holds play in a later scene as they did before it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import arachne
from arachne.assets.timing import MARK, restore_true

from .reading import SCENES

#: Opens the scene that declares nothing, then scenes that declare timing
#: (with boil, a callout on the rig, flows), then the first again, in one
#: process; prints each opening's routes and frames.
PROGRAM = """\
import hashlib, json, sys
from arachne.session import Session

scenes, media = sys.argv[1], sys.argv[2]
config = {"pixel_width": 96, "pixel_height": 54, "frame_rate": 15, "media_dir": media}


def opened(scene):
    session = Session.open(scenes, scene, config=config)
    try:
        return {
            "routes": [play.route for play in session.record.plays],
            "frames": [
                hashlib.sha256(session.frame(t).tobytes()).hexdigest()
                for t in (0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2)
            ],
        }
    finally:
        session.close()


fresh = opened("Undeclared")
declared = [opened(s) for s in ("Boiling", "CalloutOnSteppedBox", "FlowBesideSteppedBox")]
after = opened("Undeclared")
print(json.dumps({"fresh": fresh, "after": after}))
"""


def test_a_scene_that_declares_nothing_is_the_same_fresh_and_after_a_declared_one(
    tmp_path,
):
    root = Path(arachne.__file__).parent.parent
    done = subprocess.run(
        [sys.executable, "-c", PROGRAM, str(SCENES), str(tmp_path / "media")],
        capture_output=True,
        text=True,
        cwd=root,
    )
    assert done.returncode == 0, done.stderr
    openings = json.loads(done.stdout.strip().splitlines()[-1])
    assert openings["fresh"] == openings["after"]
    # A flow's driver is a scene updater, so every play ticks, as it did.
    assert openings["fresh"]["routes"] == ["ticked", "ticked", "ticked"]
    assert len(set(openings["fresh"]["frames"])) > 3


#: Mobjects that outlive their scenes: a square, a group holding one, two
#: copies (fresh squares until the declaring scene puts its copies here),
#: and a pair a second declaring scene declares and leaves.
KEPT = """\
from arachne import Circle, Square, VGroup

square = Square()
group = VGroup(Square(), Circle())
copies = [Square(), Square()]
pair = VGroup(Square(), Circle())
"""

#: A scene for each kind of mobject kept, which declares nothing; the scene
#: that declares the square and the group's square, copies the square in
#: the middle of a play and between two, and keeps the copies; and the
#: scene whose one play leaves the declared pair.
REUSING = """\
from arachne import LEFT, RIGHT, UP, Boil, FadeOut, Scene, set_timing

import kept


class _Plain(Scene):
    def things(self):
        raise NotImplementedError

    def construct(self):
        things = self.things()
        self.add(*things)
        self.wait(0.5)
        self.play(things[0].animate.shift(0.1 * RIGHT), run_time=0.5)
        self.wait(0.5)


class PlainSquare(_Plain):
    def things(self):
        return [kept.square]


class PlainCopies(_Plain):
    def things(self):
        return list(kept.copies)


class PlainGroup(_Plain):
    def things(self):
        return [kept.group]


class PlainPair(_Plain):
    def things(self):
        return [kept.pair]


class Declaring(Scene):
    def construct(self):
        made = []

        def copy_in_play(dt):
            if self.time > 0.2 and not made:
                made.append(kept.square.copy())

        self.add_updater(copy_in_play)
        self.add(kept.square, kept.group)
        set_timing(kept.square, 12, boil=Boil(6, 2.0, 3))
        set_timing(kept.group[0], 12)
        self.play(
            kept.square.animate.shift(RIGHT), kept.group.animate.shift(UP),
            run_time=0.45,
        )
        made.append(kept.square.copy())
        self.play(kept.square.animate.shift(LEFT), run_time=0.45)
        kept.copies[:] = made


class DeclaringPair(Scene):
    def construct(self):
        # Its one play fades one of the declared pair out: the scene puts
        # the other in the pair's place, so the pair leaves the scene as the
        # play ends.
        self.add(kept.pair)
        set_timing(kept.pair, 12)
        self.play(FadeOut(kept.pair[0]), run_time=0.45)
"""

#: Opens each plain scene, then the declaring scene, then each plain scene
#: again, in one process; prints each plain opening's routes and holds,
#: and whether anything it played carried a restorer.
REUSE = """\
import json, sys
from arachne.assets.timing import restore_true
from arachne.session import Session

scenes, media = sys.argv[1], sys.argv[2]
config = {"pixel_width": 96, "pixel_height": 54, "frame_rate": 15, "media_dir": media}
PLAIN = ("PlainSquare", "PlainCopies", "PlainGroup", "PlainPair")


def opened(scene):
    session = Session.open(scenes, scene, config=config)
    try:
        return {
            "routes": [play.route for play in session.record.plays],
            "held": [play.held for play in session.record.plays],
            "restorer": any(
                u is restore_true
                for top in session.scene.mobjects
                for m in top.get_family()
                for u in m.updaters
            ),
        }
    finally:
        session.close()


fresh = {s: opened(s) for s in PLAIN}
opened("Declaring")
opened("DeclaringPair")
after = {s: opened(s) for s in PLAIN}
print(json.dumps({"fresh": fresh, "after": after}))
"""


def test_a_declaration_ends_with_its_scene(tmp_path):
    # PlainSquare: the declared mobject itself; PlainCopies: a copy made in
    # the middle of a play and one made between two; PlainGroup: a group
    # outliving the scene that holds a declared mobject; PlainPair: a
    # declared group the scene's last play left, when it faded one of its
    # members out.
    scenes, shared = tmp_path / "scenes", tmp_path / "shared"
    scenes.mkdir()
    shared.mkdir()
    (shared / "kept.py").write_text(KEPT)
    (scenes / "reusing.py").write_text(REUSING)
    root = Path(arachne.__file__).parent.parent
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            REUSE,
            str(scenes / "reusing.py"),
            str(tmp_path / "media"),
        ],
        capture_output=True,
        text=True,
        cwd=root,
        env={**os.environ, "PYTHONPATH": os.pathsep.join([str(shared), str(root)])},
    )
    assert done.returncode == 0, done.stderr
    openings = json.loads(done.stdout.strip().splitlines()[-1])
    for scene, fresh in openings["fresh"].items():
        assert openings["after"][scene] == fresh, scene
        # Nothing ticks a plain scene's waits: they hold.
        assert fresh["held"] == [True, False, True], scene
        assert not fresh["restorer"]


def test_a_scene_that_declares_nothing_holds_no_mark_and_no_restorer(opened):
    session = opened("Undeclared", 30)
    record = session.record
    for play in record.plays:
        for capture in play.samples:
            for mobject in record.mobjects():
                state = record.node_state(mobject, capture)
                if state is None:
                    continue
                assert MARK not in state.keys
                assert all(u is not restore_true for u in state.updaters)
