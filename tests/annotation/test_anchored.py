"""The rule ``anchored``: every attachment measured at its anchor, in pixels
of the 1080-tall reference picture, at every instant the rule samples.

Clean on callouts kept at a moving rig's anchor, at a rig posed while it
moves, at a reshaping part's outline anchor and at points of an ordinary
shape, each under a moving camera; failing on a callout added before what
moves its target. And, read apart from the rule and from
:meth:`~arachne.assets.Anchor.point`, a leader's end on each rig scene sits
on the landmark of the part's outline the record holds.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from arachne import Callout, Rig
from arachne.camera.projection import record_view
from arachne.lint import lint
from arachne.lint.rules import anchored
from arachne.session import Session
from arachne.timeline import admission
from arachne.timeline.clock import frames_in

from . import annotation_scenes

SCENES = Path(annotation_scenes.__file__)
CLEAN = [
    "CalloutOnMovingBox",
    "CalloutOnPosedMovingBox",
    "CalloutOnSwellingTip",
    "AnnotatedShape",
    "CalloutAfterMovingDot",
    "FollowingReadoutAndGauge",
    "WideningReadouts",
]


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


@pytest.mark.parametrize("scene", CLEAN)
def test_an_attachment_kept_at_its_anchor_is_clean(opened, scene):
    found = lint(opened(scene), rules=(anchored,)).findings
    assert [f.code for f in found] == [], [f.message for f in found]


def test_a_callout_added_before_what_moves_its_target_is_detached(opened):
    session = opened("CalloutBeforeMovingDot")
    (finding,) = lint(session, rules=(anchored,)).findings
    assert finding.code == "annotation.detached"
    assert finding.severity == "error"
    assert [o.kind for o in finding.objects] == ["Callout"]
    # A tick behind a dot crossing 2.4 units a second: 0.04 units, over five
    # reference pixels.
    assert finding.detail["distance"] == pytest.approx(0.04 * 1080 / 8, rel=0.02)
    assert finding.detail["allowed"] == 1
    assert finding.line is not None
    assert "after" in finding.message or "after" in _fix(finding.code)
    # It holds while the dot moves: from the wait's first moving frame on.
    assert finding.t_start <= Fraction(1, 10)
    assert finding.t_end >= Fraction(9, 10)


def _fix(code: str) -> str:
    from arachne.lint.codes import code as entry

    return entry(code).fix


def test_the_rule_reads_nothing_in_a_scene_with_no_attachment(opened):
    session = opened("StillReadout")
    assert lint(session, rules=(anchored,)).findings == ()


def _bezier(controls: np.ndarray, t: float) -> np.ndarray:
    p0, p1, p2, p3 = controls
    s = 1 - t
    return s**3 * p0 + 3 * s**2 * t * p1 + 3 * s * t**2 * p2 + t**3 * p3


def _landmark(rig: Rig, part: str, point: np.ndarray) -> tuple[int, int, float]:
    """The shape of ``part``, the cubic of it and the place along that cubic
    where its outline passes through ``point``: searched here, densely, then
    by bisection on the place.
    """
    best = None
    places = np.linspace(0, 1, 4001)
    for s, shape in enumerate(rig.part(part).submobjects):
        for c, controls in enumerate(np.asarray(shape.points).reshape(-1, 4, 3)):
            along = np.array([_bezier(controls, t) for t in places])
            k = int(np.argmin(np.linalg.norm(along - point, axis=1)))
            low, high = places[max(k - 1, 0)], places[min(k + 1, len(places) - 1)]
            for _ in range(60):
                a, b = low + (high - low) / 3, high - (high - low) / 3
                if np.linalg.norm(_bezier(controls, a) - point) < np.linalg.norm(
                    _bezier(controls, b) - point
                ):
                    high = b
                else:
                    low = a
            t = (low + high) / 2
            distance = float(np.linalg.norm(_bezier(controls, t) - point))
            if best is None or distance < best[0]:
                best = (distance, s, c, t)
    assert best is not None
    assert best[0] < 1e-9, best
    return best[1], best[2], best[3]


@pytest.mark.parametrize(
    ("scene", "anchor_name", "part"),
    [
        ("CalloutOnMovingBox", "knob", "lid"),
        ("CalloutOnPosedMovingBox", "knob", "lid"),
        ("CalloutOnSwellingTip", "tip", "inner_layer"),
    ],
)
def test_a_leader_ends_on_its_part_s_outline_landmark(opened, scene, anchor_name, part):
    # The landmark is found at rest, before the scene runs, by its place.
    rest = Rig(
        "fixture:lidded-box" if part == "lid" else "fixture:layered-section"
    ).pose(**({"lid_open": 40} if scene == "CalloutOnMovingBox" else {}))
    s, curve, at = _landmark(rest, part, rest.anchor(anchor_name).point())
    session = opened(scene)
    record = session.record
    rig = next(m for m in record.mobjects() if isinstance(m, Rig))
    callout = next(m for m in record.mobjects() if isinstance(m, Callout))
    shape = rig.part(part).submobjects[s]
    moved = False
    for t in frames_in(Fraction(0), record.duration, Fraction(30)):
        _, capture = record.capture_at(t)
        points = np.asarray(record.node_state(shape, capture).get("_points"))
        landmark = _bezier(points[4 * curve : 4 * curve + 4], at)
        leader = np.asarray(record.node_state(callout.leader, capture).get("_points"))
        frame = record.node_state(session.scene.camera.frame, capture)
        height = float(np.ptp(np.asarray(frame.get("_points"))[:, 1]))
        # In reference pixels of the frame on screen then.
        assert np.linalg.norm(leader[-1] - landmark) * 1080 / height < 1e-3, t
        moved |= not np.allclose(landmark, rest.anchor(anchor_name).point())
    assert moved
    assert record_view(record).camera == "moving"


def test_a_counting_readout_and_a_filling_gauge_end_at_their_anchors(opened):
    from arachne import DialGauge, Readout

    session = opened("FollowingReadoutAndGauge")
    record = session.record
    rig = next(m for m in record.mobjects() if isinstance(m, Rig))
    reading = next(m for m in record.mobjects() if isinstance(m, Readout))
    dial = next(m for m in record.mobjects() if isinstance(m, DialGauge))
    knob = rig.anchor("knob").point()
    assert (reading.value, reading.label.original_text) == (90, "90")
    assert dial.value == 0.75
    np.testing.assert_allclose(
        reading.body.get_center() - reading.attached.offset, knob, atol=1e-9
    )
    np.testing.assert_allclose(dial.leader.points[-1], knob, atol=1e-12)


def test_a_fault_at_an_instant_the_lint_samples_is_read(opened):
    # 1/4 s is a frame the lint samples at 60 fps, and none of the rule's
    # own 15 Hz instants.
    session = opened("StepAtAQuarter", fps=60)
    (finding,) = lint(session, rules=(anchored,)).findings
    assert finding.code == "annotation.detached"
    assert (finding.t_start, finding.t_end) == (Fraction(1, 4), Fraction(1, 4))


def test_a_fault_at_an_instant_the_lint_is_asked_for_is_read(opened):
    session = opened("StepAtThirtySevenSixtieths", fps=60)
    asked = Fraction(37, 60)
    assert lint(session, rules=(anchored,)).findings == ()
    (finding,) = lint(session, at=[asked], rules=(anchored,)).findings
    assert (finding.t_start, finding.t_end) == (asked, asked)
    # Asked of the rule itself, it reads that frame too.
    (finding,) = lint(session, rules=(anchored.where(at=[asked]),)).findings
    assert finding.t_start == asked


def test_an_anchor_on_an_object_the_scene_does_not_hold_is_not_measured(opened):
    # The record keeps no state of the dot, so no instant can say where it
    # was: the rule measures only anchors on what the scene holds.
    session = opened("CalloutOnOffSceneDot")
    assert lint(session, rules=(anchored,)).findings == ()


def test_a_lint_s_instants_are_read_only_on_its_frames():
    from arachne.lint.frames import Frames

    frames = Frames(None, lint_instants=())  # type: ignore[arg-type]
    assert frames.lint_instants == ()
    with pytest.raises(AttributeError):
        frames.lint_instants = ()  # type: ignore[misc]
