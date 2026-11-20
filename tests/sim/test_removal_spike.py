"""Spike (do not merge): what ``state_at`` answers at the instant a closed
world is removed (t = 1/2, between two frames of a 15 fps grid), asked alone,
with a later instant, in either order, and what the frame there draws.

Everything is printed; the last assertion states the rule the spike checks:
the answer at 1/2 does not depend on which other instants are asked, and the
frame at 1/2 draws the pose it answers.
"""

from __future__ import annotations

import math
import sys
import traceback
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("arachne_sim")

from arachne.session import Session  # noqa: E402
from arachne.timeline import admission  # noqa: E402

from . import removal_scenes  # noqa: E402

FILE = Path(removal_scenes.__file__)
CONFIG = {"pixel_width": 640, "pixel_height": 360, "frame_rate": 15}
HALF = Fraction(1, 2)
LATER = Fraction(3, 4)


def say(*parts: object) -> None:
    print(*parts, flush=True)
    print(*parts, file=sys.stderr, flush=True)


def _open(scene: str, *, keep_demoted: bool = False, **options) -> Session:
    if not keep_demoted:
        admission.DEMOTED.clear()
    return Session.open(FILE, scene, config=CONFIG, **options)


def _body(session: Session) -> str | None:
    bakes = session.record.bakes
    if bakes is None or not bakes.occurrences:
        return None
    occurrence = bakes.occurrences[0]
    (body,) = [b.id for b in occurrence.bindings if b.channel == "pose"]
    return body


def _poses(session: Session) -> dict[int, tuple[float, float]]:
    bakes = session.record.bakes
    occurrence = bakes.occurrences[0]
    values = bakes.contents[occurrence.content].channels["pose"].values
    return {
        occurrence.first_tick + row: tuple(float(v) for v in values[row, 0, :2])
        for row in range(values.shape[0])
    }


def _describe_record(session: Session) -> None:
    record = session.record
    bakes = record.bakes
    for occurrence in bakes.occurrences:
        say(
            f"    occurrence {occurrence.ordinal}: ticks"
            f" {occurrence.first_tick}..{occurrence.last_tick}"
        )
    for play in record.plays:
        instants = [str(c.instant) for c in play.samples]
        reasons = play.admission.reasons if play.admission is not None else None
        say(
            f"    play {play.index} [{play.start}, {play.end}) kind={play.kind}"
            f" route={play.route} held={play.held} reasons={reasons}"
            f" samples={instants}"
        )


def _summary(state, body: str | None) -> str:
    tick = math.floor(state.as_of * 60)
    head = (
        f"t={state.t} as_of={state.as_of} (tick {tick}) play={state.play}"
        f" route={state.route} answer={state.answer}"
    )
    if body is None:
        return head + " body=?"
    try:
        node = state.node(body)
    except KeyError:
        return head + " body=ABSENT"
    x0, y0, x1, y1 = node.scene_box
    placed = ((x0 + x1) / 2, (y0 + y1) / 2)
    read = {q.name: q for q in state.quantities.get(body, ())}
    position = read.get("sim.position")
    shown = (
        f"{position.value}"
        if position is not None and position.unreadable is None
        else f"unreadable({position.unreadable if position else 'none'})"
    )
    return head + f" sim.position={shown} placement=({placed[0]:.4f}, {placed[1]:.4f})"


def _probe(session: Session, image) -> tuple[float, float] | None:
    """The centre of the body's drawn disc, in scene units: the bright
    pixels of the window it falls through (x in [0.5, 2.5], y in [-0.5, 2.3]).
    """
    camera = session.scene.renderer.camera
    pixels = np.asarray(image.convert("RGB")).astype(float)
    height, width, _ = pixels.shape
    fw, fh = float(camera.frame_width), float(camera.frame_height)
    rows, cols = np.mgrid[0:height, 0:width]
    xs = (cols + 0.5 - width / 2) * fw / width
    ys = (height / 2 - rows - 0.5) * fh / height
    window = (xs > 0.5) & (xs < 2.5) & (ys > -0.5) & (ys < 2.3)
    weight = pixels.mean(axis=2) * window
    total = weight.sum()
    if total < 255:
        return None
    return (float((xs * weight).sum() / total), float((ys * weight).sum() / total))


def _nearest(drawn, poses) -> str:
    if drawn is None:
        return "nothing drawn"
    best = min(poses, key=lambda t: math.dist(drawn, poses[t]))
    return (
        f"drawn=({drawn[0]:.4f}, {drawn[1]:.4f}) nearest tick {best}"
        f" (off {math.dist(drawn, poses[best]):.4f} u)"
    )


def _guard(what: str, run) -> object:
    try:
        return run()
    except Exception:  # noqa: BLE001
        say(f"  {what}: RAISED")
        for line in traceback.format_exc().splitlines():
            say("    " + line)
        return None


def _spike(scene: str) -> dict[str, object]:
    """Every way of asking 1/2, each in a fresh session; what each answered."""
    found: dict[str, object] = {}
    say(f"\n===== {scene} =====")

    base = _open(scene)
    try:
        body = _body(base)
        poses = _poses(base)
        say(f"  body id {body}")
        say(
            "  baked poses: "
            + ", ".join(
                f"tick {t}=({poses[t][0]:.4f}, {poses[t][1]:.4f})"
                for t in (26, 27, 28, 29, 30, 31, 32)
                if t in poses
            )
        )
        _describe_record(base)

        state = _guard("state_at(1/2) alone", lambda: base.state_at(HALF))
        if state is not None:
            say("  (a) state_at(1/2) alone: " + _summary(state, body))
            found["alone"] = state.as_of
        image = _guard("frame(1/2)", lambda: base.frame(HALF))
        if image is not None:
            drawn = _probe(base, image)
            say("  (c) frame(1/2) alone: " + _nearest(drawn, poses))
            found["frame alone"] = drawn
        for t in (Fraction(7, 15), Fraction(8, 15)):
            image = _guard(f"frame({t})", lambda t=t: base.frame(t))
            if image is not None:
                say(
                    f"      frame({t}) for reference: "
                    + _nearest(_probe(base, image), poses)
                )
        state = _guard("state_at(3/4)", lambda: base.state_at(LATER))
        if state is not None:
            say("      then state_at(3/4): " + _summary(state, body))
        state = _guard("state_at(1/2) again", lambda: base.state_at(HALF))
        if state is not None:
            say("      then state_at(1/2) again: " + _summary(state, body))
            found["again"] = state.as_of
    finally:
        base.close()

    for label, instants in (
        ("answers([1/2])", [HALF]),
        ("answers([1/2, 3/4])", [HALF, LATER]),
        ("answers([3/4, 1/2])", [LATER, HALF]),
    ):
        session = _open(scene)
        try:
            body = _body(session)
            answers = _guard(
                label,
                lambda session=session, instants=instants: session.answers(instants),
            )
            if answers is None:
                continue
            for answer in answers:
                drawn = _probe(session, answer.frame) if answer.frame else None
                fallback = answer.fallback.reason if answer.fallback else None
                say(
                    f"  (b) {label} at {answer.t}: "
                    + _summary(answer.state, body)
                    + f" fallback={fallback} | "
                    + _nearest(drawn, poses)
                )
                if answer.t == HALF:
                    found[label] = answer.state.as_of
                    found[label + " frame"] = drawn
            say(f"      forward runs: {session.forward_runs}")
        finally:
            session.close()

    for label, extra in (
        ("open(instants=[3/4])", [LATER]),
        ("open(instants=[1/2, 3/4])", [HALF, LATER]),
    ):
        session = _guard(label, lambda extra=extra: _open(scene, instants=extra))
        if session is None:
            continue
        try:
            body = _body(session)
            _describe_record(session)
            state = _guard(
                label + " state_at(1/2)", lambda session=session: session.state_at(HALF)
            )
            if state is not None:
                say(f"  (b') {label} then state_at(1/2): " + _summary(state, body))
                found[label] = state.as_of
            image = _guard(
                label + " frame(1/2)", lambda session=session: session.frame(HALF)
            )
            if image is not None:
                drawn = _probe(session, image)
                say(f"      {label} frame(1/2): " + _nearest(drawn, poses))
                found[label + " frame"] = drawn
        finally:
            session.close()
    return found


@pytest.mark.parametrize("scene", removal_scenes.SCENES)
def test_state_at_the_removal_instant(scene):
    found = _spike(scene)
    as_of = {k: v for k, v in found.items() if "frame" not in k and v is not None}
    drawn = {k: v for k, v in found.items() if "frame" in k and v is not None}
    say(f"  SUMMARY {scene}: as_of {sorted({str(v) for v in as_of.values()})}")
    for key, value in drawn.items():
        say(f"  SUMMARY {scene}: {key} drawn ({value[0]:.4f}, {value[1]:.4f})")
    assert len(set(as_of.values())) <= 1, as_of
    xs = list(drawn.values())
    assert all(math.dist(xs[0], v) < 0.02 for v in xs), drawn


def test_demotions_carried_between_sessions():
    """The same scene opened three times in one process, demotions kept:
    plain, with a later instant asked of the recorder, plain again.
    """
    admission.DEMOTED.clear()
    say("\n===== ReturnedThenCounted, demotions carried =====")
    for label, extra in (
        ("plain", ()),
        ("instants=[3/4]", (LATER,)),
        ("plain again", ()),
    ):
        session = _open("ReturnedThenCounted", keep_demoted=True, instants=extra)
        try:
            body = _body(session)
            poses = _poses(session)
            say(f"  {label}: probe={session.probe}")
            say(f"  {label}: divergences={session.divergences}")
            _describe_record(session)
            state = session.state_at(HALF)
            say(f"  {label} state_at(1/2): " + _summary(state, body))
            drawn = _probe(session, session.frame(HALF))
            say(f"  {label} frame(1/2): " + _nearest(drawn, poses))
        finally:
            session.close()
    admission.DEMOTED.clear()
