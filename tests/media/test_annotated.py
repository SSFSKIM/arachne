"""Callouts on media, with no GPU: the rule ``anchored`` is clean on a
callout kept at a point of a running clip that moves and grows, at the
anchor of a rig of stills while it is posed, and at the anchor of a
layered set's depth member while the camera pans and the set fades in;
each under a moving camera. It reports a callout on a clip that is read a
tick late, so it measures a clip's anchor. And, read apart from the rule,
the leader on the depth member ends where the member's parallax puts the
anchor's point at every tick.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from arachne import Callout
from arachne.lint import lint
from arachne.lint.rules import anchored
from arachne.session import Session
from arachne.timeline import admission
from arachne.timeline.clock import frames_in

from . import annotated_scenes
from .annotated_scenes import PAN
from .raster_scenes import GROUND, canvas_to_scene, depth_of

SCENES = Path(annotated_scenes.__file__)
CLEAN = ["CalloutOnRunningClip", "CalloutOnPosedFlap", "CalloutOnViewedLayer"]
#: The frame height the scenes are configured at: the default viewing
#: distance of a layered set.
DISTANCE = 8.0


@pytest.fixture
def opened():
    sessions: list[Session] = []
    admission.DEMOTED.clear()

    def open_scene(name: str, fps: int = 30) -> Session:
        session = Session.open(
            SCENES,
            name,
            config={"pixel_width": 192, "pixel_height": 108, "frame_rate": fps},
        )
        sessions.append(session)
        return session

    yield open_scene
    for session in sessions:
        session.close()
    admission.DEMOTED.clear()


def _leader_ends(session: Session, fps: int):
    """The scene's callout's leader end at every frame at ``fps`` before
    the scene's end, with the frame's instant.
    """
    record = session.record
    callout = next(m for m in record.mobjects() if isinstance(m, Callout))
    for t in frames_in(Fraction(0), record.duration, Fraction(fps)):
        capture = record.capture_at(t)[1]
        points = np.asarray(record.node_state(callout.leader, capture).get("_points"))
        yield t, points[-1]


@pytest.mark.parametrize("scene", CLEAN)
def test_a_callout_on_media_kept_at_its_anchor_is_clean(opened, scene):
    session = opened(scene)
    found = lint(session, rules=(anchored,)).findings
    assert [f.code for f in found] == [], [f.message for f in found]
    # The callout's point moved: the rule measured a moving anchor.
    ends = [end for _, end in _leader_ends(session, 30)]
    assert np.linalg.norm(ends[-1] - ends[0]) > 0.1


def test_a_callout_on_a_clip_read_a_tick_late_is_detached(opened):
    (finding,) = lint(opened("CalloutBeforeMovingClip"), rules=(anchored,)).findings
    assert finding.code == "annotation.detached"
    assert [o.kind for o in finding.objects] == ["Callout"]


def test_a_leader_on_a_depth_member_ends_on_its_sliding_point(opened):
    session = opened("CalloutOnViewedLayer", fps=60)
    peak = canvas_to_scene(GROUND, np.array([[120.0, 90.0]]))[0]
    slide = 1 - DISTANCE / (DISTANCE + depth_of("middle"))
    for t, end in _leader_ends(session, 60):
        expected = peak + slide * PAN * float(min(t, Fraction(1)))
        # Exact, and so within a pixel of the picture.
        assert np.allclose(end, expected, rtol=0, atol=1e-9), t
