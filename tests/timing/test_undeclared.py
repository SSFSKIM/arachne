"""A scene that declares no timing plays and draws as it did before stepped
timing existed: it gets no clock and no restorer, its rig's pose settles
and its flows' driver run as the scene's own updaters, and its plays'
routes and frames are the same opened fresh and opened after a declared
scene in the same process.
"""

from __future__ import annotations

import json
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
