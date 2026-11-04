"""The assertion bridge: checks only the scene can state, recorded as it
runs and merged into the lint.

Some things the lint can't infer from what is drawn: that a point the scene
computes, or a pen its updaters drive, must stay on an axes' plotted region;
that the frame is to be swept now, or two mobjects kept apart; what terms
its contacts are held to. The scene says so through the bridge, and the
record keeps each call (:class:`~arachne.timeline.recorder.BridgeCall`, per
play):

- :func:`record` checks operands at the instant it is called;
- :func:`watch` checks a mobject, or a function of the scene, at every
  frame from the call on, read while the scene records and kept as values,
  so the lint never calls the scene's code; :func:`clear` stops it;
- :func:`sweep` sweeps the scene as it stands at the call (the instant
  rules' sweep, :mod:`.rules`), and :func:`apart` checks that one mobject
  clears others there;
- :func:`declare` states terms the lint's rules are held to from the call
  on (``pad``, ``text_gap``, ``exempt``);
- :func:`arm` asks for the checks made over time from the call on: the
  motion rule, which the lint then runs unasked.

A call returns what it found as violations in the report's shape
(:mod:`.serialise`) and never raises for a finding; :func:`~arachne.lint.lint`
reports each at its line and time. Outside a recording, which every
execution of a scene is, a call raises :class:`RuntimeError`.

An exemption names mobjects, selectors (:mod:`.exempt`) and pairs of them;
each mobject stands for its family's stable ids as it is at the call. A
sweep is held to the terms it is given, then to those declared, then to the
rules' defaults; a rule the lint runs is held to its own terms with the
declared ``pad`` and ``text_gap`` in their place and the declared
exemptions beside its own.

The one code :func:`record` and :func:`watch` check today is
``axes.outside``: every point of the operands inside the rectangle the
axes' ranges span, shrunk by ``margin`` on every side, in scene units. A
mobject operand of :func:`record` is checked along its outline, sampled
about every :data:`SPACING` units, or at its centre when it has no points;
a watched mobject is checked at its centre. A watch's finding is a run of
frames outside, at the scene line running when the run began.
"""

from __future__ import annotations

__all__ = [
    "CHECKED",
    "SPACING",
    "TERMS",
    "TEXT_FLOOR",
    "Checks",
    "Watch",
    "apart",
    "arm",
    "armed",
    "armed_rules",
    "clear",
    "declare",
    "declared",
    "exemptions",
    "findings",
    "record",
    "sweep",
    "terms_at",
    "watch",
]

import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from fractions import Fraction
from types import FrameType
from typing import TYPE_CHECKING, Any

import numpy as np

from arachne._config import config
from arachne.timeline.clock import exact, frames_in
from arachne.timeline.origins import origin_of
from arachne.timeline.recorder import BridgeCall, Reading, Recorder, running
from arachne.timeline.source import names_from_frames, scene_frames

from .exempt import Exemptions
from .findings import Finding
from .parts import Parts, build_parts
from .rules import DEFAULT_PAD, Terms, sweep_findings
from .rules._pairs import NOISE, _annotates, _called
from .rules._pairs import _measure as _clearance
from .rules._units import Measured, measured_units

if TYPE_CHECKING:
    from arachne.timeline.record import Record
    from arachne.timeline.source import SourceLine

    from .core import LintContext
    from .frames import Frames


#: The codes :func:`record` and :func:`watch` check.
CHECKED = ("axes.outside",)

#: The terms :func:`declare`, :func:`sweep` and :func:`arm` take.
TERMS = ("exempt", "pad", "text_gap")

#: The calls that declare terms.
_DECLARING = ("declare", "arm")

#: The least clearance :func:`apart` holds two texts to, in scene units.
TEXT_FLOOR = 0.3

#: Scene units between the samples along a mobject's outline.
SPACING = 0.02
#: Samples along each curve of an outline, at least and at most.
_SAMPLES = (8, 512)


def record(
    code: str,
    operands: Any,
    *,
    reference: Any = None,
    margin: float = 0.0,
) -> list[dict[str, Any]]:
    """Check ``operands`` (points, a mobject, or a sequence of either) for
    ``code`` against ``reference`` now, keep the call on the record, and
    return what it found as report violations.
    """
    recorder = _recorder("record")
    _check_code(code, reference)
    points = _points(operands)
    call = recorder.bridge_call(
        "record",
        sys._getframe(1),
        code=code,
        operands=_stored(operands),
        reference=origin_of(reference).id,
        margin=float(margin),
        subject=_subject(operands),
    )
    call.readings.append(
        Reading(call.t, call.line, _measure(points, reference, margin))
    )
    return _violations(_record_findings(call))


def watch(
    code: str,
    operands: Any,
    *,
    reference: Any = None,
    margin: float = 0.0,
) -> Watch:
    """Check ``operands`` for ``code`` against ``reference`` at every frame
    from now on: a mobject at its centre, or a function of no arguments
    returning a point or points (``None`` for nothing to check), each read
    as the scene records. Returns the watch (:class:`Watch`), which holds
    what the check finds now.
    """
    from arachne.mobject.mobject import Mobject

    recorder = _recorder("watch")
    _check_code(code, reference)
    if isinstance(operands, Mobject):
        mobject = operands

        def probe() -> Any:
            return mobject.get_center()

        stored: tuple[Any, ...] = (origin_of(mobject).id,)
    elif callable(operands):
        probe, stored = operands, ()
    else:
        raise TypeError(
            "watch checks a mobject or a function giving points, not"
            f" {type(operands).__name__}"
        )

    def read() -> Mapping[str, Any] | None:
        found = probe()
        if found is None:
            return None
        return _measure(_as_points(found), reference, margin)

    call = recorder.bridge_call(
        "watch",
        sys._getframe(1),
        code=code,
        operands=stored,
        reference=origin_of(reference).id,
        margin=float(margin),
        subject=_subject(operands),
        reader=read,
    )
    reading = Reading(call.t, call.line, read())
    call.readings.append(reading)
    found = _finding(call, reading, reading.t, call.line)
    return Watch(call, _violations([found] if found is not None else []))


@dataclass(frozen=True, eq=False)
class Watch:
    """What :func:`watch` returns: the watch's call as the record keeps it,
    and what the check found at the call (report violations). :func:`clear`
    takes it to stop the watch.
    """

    call: BridgeCall
    found: list[dict[str, Any]]


def declare(**terms: Any) -> list[dict[str, Any]]:
    """Declare terms the lint's rules are held to from now on: ``pad``,
    ``text_gap`` (``None`` for none) and ``exempt``, as :func:`sweep` takes
    them. They are kept on the record; nothing is found by declaring.
    """
    recorder = _recorder("declare")
    recorder.bridge_call("declare", sys._getframe(1), terms=_terms(terms))
    return []


def clear(handle: Watch) -> list[dict[str, Any]]:
    """Stop the watch ``handle`` (what :func:`watch` returned): it is read
    at no frame after now. Clearing a watch again does nothing more.
    """
    recorder = _recorder("clear")
    if not isinstance(handle, Watch):
        raise TypeError(
            "clear stops a watch: it takes what watch() returned, not"
            f" {type(handle).__name__}"
        )
    bridge = recorder.ensure_record().bridge
    if not any(call is handle.call for call in bridge.calls):
        raise ValueError("clear stops a watch of this recording; this one is another's")
    recorder.bridge_call(
        "clear", sys._getframe(1), operands=(bridge.calls.index(handle.call),)
    )
    bridge.stop(handle.call)
    return []


def sweep(**terms: Any) -> list[dict[str, Any]]:
    """Sweep the scene as it stands now: the instant rules' sweep (texts at
    the frame's edge, every pair on it) under the terms given (``pad``,
    ``text_gap``, ``exempt``), each term not given as declared, else as
    :class:`~.rules.Terms` defaults it. Returns what it finds.
    """
    recorder = _recorder("sweep")
    given = _terms(terms)
    held = _apply(_apply(Terms(), _held(recorder.ensure_record().bridge.calls)), given)
    caller = sys._getframe(1)
    call = recorder.bridge_call("sweep", caller, terms=given)
    parts = _parts_now(recorder, call, _names(recorder, caller))
    return _found(call, sweep_findings(parts, held))


def apart(
    subject: Any,
    against: Any,
    *,
    min_gap: float | None = None,
    pad: float = DEFAULT_PAD,
) -> list[dict[str, Any]]:
    """Check that ``subject`` clears each mobject of ``against`` (one, or a
    sequence) now, added to the scene or not: by ``min_gap`` when given,
    else two texts by a line of the shorter (at least :data:`TEXT_FLOOR`),
    a ring around some of a text's glyphs by touching only, anything else by
    ``pad``. An axis's line and ticks are held clear of a text subject only.
    Returns what it finds.
    """
    from arachne.mobject.mobject import Mobject

    recorder = _recorder("apart")
    others = [against] if isinstance(against, Mobject) else list(against)
    for mob in (subject, *others):
        if not isinstance(mob, Mobject):
            raise TypeError(f"apart checks mobjects, not {type(mob).__name__}")
    caller = sys._getframe(1)
    call = recorder.bridge_call(
        "apart",
        caller,
        operands=tuple(origin_of(mob).id for mob in (subject, *others)),
        terms={"min_gap": min_gap, "pad": float(pad)},
        subject=_subject(subject),
    )
    names = _names(recorder, caller)
    subjects = _units_now(recorder, call, names, subject)
    found = [
        finding
        for other in others
        for o in _units_now(recorder, call, names, other)
        for s in subjects
        if (finding := _apart(s, o, min_gap, float(pad))) is not None
    ]
    return _found(call, found)


def arm(*, motion_jump: float | None = None, **terms: Any) -> list[dict[str, Any]]:
    """Arm the checks a scene asks for over time from now on: the motion
    rule (``motion.jump`` beyond ``motion_jump``, by default the rule's own
    tolerance), which the lint then runs unasked, its record keeping each
    play's exit from the next to end. ``terms`` are declared as by
    :func:`declare`. Arming again declares again and sets the tolerance the
    whole scene is measured at.
    """
    from .motion import JUMP_TOL

    recorder = _recorder("arm")
    kept = _terms(terms)
    kept["motion_jump"] = JUMP_TOL if motion_jump is None else float(motion_jump)
    recorder.bridge_call("arm", sys._getframe(1), terms=kept)
    recorder.arm_motion()
    return []


# The terms declared


def declared(source: Record, t: Fraction | float) -> dict[str, Any]:
    """The terms ``source``'s scene declared at or before ``t`` (by
    :func:`declare` and :func:`arm`): a later ``pad`` or ``text_gap``
    replacing an earlier, exemptions adding up.
    """
    at = exact(t)
    return _held(call for call in source.bridge.calls if call.t <= at)


def terms_at(source: Record, t: Fraction | float, base: Terms) -> Terms:
    """``base`` under the terms declared at or before ``t``: the declared
    ``pad`` and ``text_gap`` replace its own, and the declared exemptions
    hold beside its own.
    """
    return _apply(base, declared(source, t), join=True)


def exemptions(source: Record) -> Exemptions:
    """Every exemption ``source``'s scene stated, declared, armed or swept
    with, each once: what a check of the whole scene, read at its end, is
    held to.
    """
    kept = Exemptions()
    for call in source.bridge.calls:
        if call.kind in _DECLARING + ("sweep",) and "exempt" in call.terms:
            kept = _joined(kept, call.terms["exempt"])
    return kept


def armed(source: Record, t: Fraction | float | None = None) -> BridgeCall | None:
    """The arming in force at ``t`` (by default, at the scene's end): the
    last :func:`arm` call at or before it, whose terms hold its
    ``motion_jump`` and the terms it declared; ``None`` before any.
    """
    at = None if t is None else exact(t)
    found = None
    for call in source.bridge.calls:
        if call.kind == "arm" and (at is None or call.t <= at):
            found = call
    return found


def armed_rules(source: Record, asked: Iterable[Any]) -> list[Any]:
    """The rules ``source``'s scene armed (:func:`arm`), which the lint runs
    unasked, less those the lint was asked for already: each check made over
    time that arming asks for. Today that is the motion rule, at the
    tolerance of the last arming.
    """
    from .motion import MotionJump

    arming = armed(source)
    if arming is None or any(isinstance(rule, MotionJump) for rule in asked):
        return []
    return [MotionJump(arming.terms["motion_jump"])]


class Checks:
    """The lint's reader of the bridge (:mod:`.frames`' client): the findings
    of the bridge's calls, where the render's frames see them, and of the
    rules the scene armed (:func:`armed_rules`, given the lint's own
    ``rules``). Each record, sweep and apart is found at its call, and each
    watch at every frame of the render from its call to its clear, a run of
    frames outside one finding from the first to the last, as a check made
    on every frame finds it.

    The calls and readings are read from the execution the lint's frames
    come from (:attr:`.Frames.execution`). A record its frames are read from
    as kept (one on the render's grid that draws) evaluated every frame as
    the render does, and is read itself. Any other (one kept on another
    grid, whose frame hooks and dt updaters ran at other instants, or one
    that refuses drawing) is read from the lint's one forward run, which is
    asked to run to the scene's end whatever frames the lint reads, so it
    holds every call that execution makes, those its record never made
    included.
    """

    def __init__(self, rules: Iterable[Any] = ()) -> None:
        self._asked = tuple(rules)
        self._armed: list[Any] = []

    def request_frames(self, frames: Frames) -> None:
        """Ask that a forward run, if the lint needs one, run to the end; and
        for the frames the armed rules read.
        """
        frames.complete(self)
        self._armed = armed_rules(frames.session.record, self._asked)
        for rule in self._armed:
            rule.request_frames(frames)

    def __call__(self, context: LintContext) -> list[Finding]:
        found = findings(context.frames.execution)
        for rule in self._armed:
            found.extend(rule(context))
        return found


def findings(source: Record) -> list[Finding]:
    """The findings of the bridge's calls in ``source``, an execution on the
    render's grid.
    """
    from .instants import render_grid

    fps = render_grid(source)
    calls = source.bridge.calls
    stops: dict[int, Fraction] = {}
    for call in calls:
        if call.kind == "clear":
            stops.setdefault(call.operands[0], call.t)
    found: list[Finding] = []
    for index, call in enumerate(calls):
        if call.kind == "record":
            found.extend(_record_findings(call))
        elif call.kind == "watch":
            end = min(source.duration, stops.get(index, source.duration))
            found.extend(_watch_findings(call, fps, end))
        elif call.kind in ("sweep", "apart"):
            (reading,) = call.readings
            assert reading.values is not None
            found.extend(reading.values["findings"])
    return found


# Calls


def _recorder(name: str) -> Recorder:
    recorder = running()
    if recorder is None:
        raise RuntimeError(
            f"bridge.{name} is called while a scene is recording, as every"
            " execution of a scene is: from its setup, construct or tear_down, or"
            " the code they call"
        )
    return recorder


def _check_code(code: str, reference: Any) -> None:
    if code not in CHECKED:
        raise ValueError(f"the bridge checks {', '.join(CHECKED)}; not {code!r}")
    if reference is None or not all(
        hasattr(reference, name) for name in ("c2p", "x_range", "y_range")
    ):
        raise ValueError(
            f"{code} is checked against a reference: the axes whose ranges"
            " the points must lie in"
        )


def _terms(terms: Mapping[str, Any]) -> dict[str, Any]:
    """``terms`` as kept: numbers as floats, exemptions read now."""
    unknown = sorted(set(terms) - set(TERMS))
    if unknown:
        raise TypeError(
            f"the bridge takes {', '.join(TERMS)}; not {', '.join(unknown)}"
        )
    kept: dict[str, Any] = {}
    for name, value in terms.items():
        if name == "exempt":
            kept[name] = _exemptions(value)
        elif name == "text_gap" and value is None:
            kept[name] = None
        else:
            kept[name] = float(value)
    return kept


def _exemptions(exempt: Any) -> Exemptions:
    """``exempt`` (mobjects, selectors, and pairs of them; a lone mobject or
    a lone pair of mobjects is one) parsed now: each mobject stands for its
    family's stable ids as it is at the call.
    """
    from arachne.mobject.mobject import Mobject

    if isinstance(exempt, Mobject) or (
        isinstance(exempt, tuple)
        and len(exempt) == 2
        and all(isinstance(item, Mobject) for item in exempt)
    ):
        exempt = [exempt]
    return Exemptions.parse(exempt or ())


def _joined(first: Exemptions, second: Exemptions) -> Exemptions:
    """The exemptions of both, each once."""
    return Exemptions(
        tuple(dict.fromkeys((*first.bare, *second.bare))),
        tuple(dict.fromkeys((*first.pairs, *second.pairs))),
    )


def _held(calls: Iterable[BridgeCall]) -> dict[str, Any]:
    """The terms ``calls`` declare, in order: a later ``pad`` or
    ``text_gap`` replacing an earlier, exemptions adding up.
    """
    held: dict[str, Any] = {}
    for call in calls:
        if call.kind not in _DECLARING:
            continue
        for name in TERMS:
            if name not in call.terms:
                continue
            value = call.terms[name]
            if name == "exempt" and name in held:
                value = _joined(held[name], value)
            held[name] = value
    return held


def _apply(base: Terms, held: Mapping[str, Any], *, join: bool = False) -> Terms:
    """``base`` with each term ``held`` states in its place; with ``join``,
    its exemptions beside ``base``'s.
    """
    exempt = base.exempt
    if "exempt" in held:
        exempt = _joined(exempt, held["exempt"]) if join else held["exempt"]
    return replace(
        base,
        pad=held.get("pad", base.pad),
        text_gap=held.get("text_gap", base.text_gap),
        exempt=exempt,
    )


# Measuring the scene at a call


@dataclass(frozen=True)
class _Now:
    """The play a call reads the scene in, as :func:`.parts.build_parts`
    reads one: its index, the scene's names for its mobjects now, and no
    static picture (the scene is read whole).
    """

    index: int
    names: dict[int, str]
    static: tuple[Any, ...] = ()


def _names(recorder: Recorder, caller: FrameType) -> dict[int, str]:
    """The names the scene's code gives its mobjects from ``caller`` out."""
    return names_from_frames(list(scene_frames(caller)), recorder.scene)


def _parts_now(
    recorder: Recorder,
    call: BridgeCall,
    names: dict[int, str],
    roots: tuple[Any, ...] | None = None,
) -> Parts:
    """The parts the scene draws now, as its camera views them: the
    scene's mobjects, or ``roots``, drawn or not. Nothing is captured on the
    record.
    """
    from arachne.timeline.record import Capture

    scene = recorder.scene
    if roots is None:
        roots, foreground = tuple(scene.mobjects), tuple(scene.foreground_mobjects)
    else:
        foreground = ()
    capture = Capture(-1, call.t, "bridge", roots, foreground, (), (), None)
    camera = scene.renderer.camera
    return build_parts(
        recorder.ensure_record(),
        call.t,
        _Now(call.play, names),  # type: ignore[arg-type]
        capture,
        answer="recorded",
        target=(int(camera.pixel_width), int(camera.pixel_height)),
        live=True,
    )


def _units_now(
    recorder: Recorder, call: BridgeCall, names: dict[int, str], mob: Any
) -> list[Measured]:
    """The units ``mob`` draws now, drawn or not."""
    return measured_units(_parts_now(recorder, call, names, (mob,)))


def _apart(
    subject: Measured, other: Measured, min_gap: float | None, pad: float
) -> Finding | None:
    """The finding of ``subject`` not clear of ``other``, if it is not."""
    if subject.id == other.id or (other.structural and not subject.text):
        return None
    if min_gap is not None:
        required = float(min_gap)
    elif subject.text and other.text:
        required = max(TEXT_FLOOR, min(_height(subject), _height(other)))
    elif _annotates(subject, other) or _annotates(other, subject):
        required = NOISE
    else:
        required = pad
    measured = _clearance(subject, other)
    if measured >= required:
        return None
    if not (subject.text or other.text):
        code = "contact.held"
    elif subject.text and other.text and measured > 0:
        code = "gap.text"
    else:
        code = "overlap.text"
    return Finding(
        code,
        Fraction(0),
        Fraction(0),
        (subject.object, other.object),
        message=(
            f"{_called(subject)} is not clear of {_called(other)}: {measured:.3f} units"
            f" apart, against {required:.3f}."
        ),
        detail=(
            {"overlap": round(-measured, 4)}
            if code == "overlap.text"
            else {"gap": round(measured, 4), "required": round(required, 4)}
        ),
    )


def _height(unit: Measured) -> float:
    x0, y0, x1, y1 = unit.outer
    return y1 - y0


def _found(call: BridgeCall, found: Iterable[Finding]) -> list[dict[str, Any]]:
    """``found`` at ``call``'s time and line, kept as its reading; returned
    as report violations.
    """
    kept = tuple(
        replace(finding, t_start=call.t, t_end=call.t, line=call.line)
        for finding in found
    )
    call.readings.append(Reading(call.t, call.line, {"findings": kept}))
    return _violations(kept)


def _items(operands: Any) -> list[Any]:
    """The operands one by one: a mobject, an array of points, or each item
    of a sequence holding mobjects.
    """
    from arachne.mobject.mobject import Mobject

    if isinstance(operands, Mobject):
        return [operands]
    try:
        _as_points(operands)
    except (TypeError, ValueError):
        return list(operands)
    return [operands]


def _stored(operands: Any) -> tuple[Any, ...]:
    """The operands as kept: each mobject by stable id, points as values."""
    from arachne.mobject.mobject import Mobject

    return tuple(
        origin_of(item).id if isinstance(item, Mobject) else _as_points(item)
        for item in _items(operands)
    )


def _subject(operands: Any) -> str:
    from arachne.mobject.mobject import Mobject

    if isinstance(operands, Mobject):
        return f"the {type(operands).__name__} {origin_of(operands).id}"
    if callable(operands):
        return "the watched function"
    return "the point set"


# Measuring


def _as_points(value: Any) -> np.ndarray:
    """``value``, a point or points, as an ``(n, 2)`` array of x and y."""
    points = np.atleast_2d(np.asarray(value, dtype=float))
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError(f"not a point or points: {value!r}")
    return points[:, :2].copy()


def _points(operands: Any) -> np.ndarray:
    from arachne.mobject.mobject import Mobject

    found = [
        _outline(item) if isinstance(item, Mobject) else _as_points(item)
        for item in _items(operands)
    ]
    return np.vstack(found) if found else np.empty((0, 2))


def _outline(mobject: Any) -> np.ndarray:
    """Points along ``mobject``'s family: each cubic curve of a member
    sampled about every :data:`SPACING` units of its longest control
    polygon, other points as they are; its centre when it has none.
    """
    out: list[np.ndarray] = []
    for member in mobject.family_members_with_points():
        points = np.asarray(member.points, dtype=float)
        per_curve = getattr(member, "n_points_per_curve", 4)
        whole = len(points) - len(points) % per_curve
        if per_curve != 4 or whole < 4:
            out.append(points[:, :2])
            continue
        curves = points[:whole].reshape(-1, 4, points.shape[1])
        polygon = np.linalg.norm(np.diff(curves, axis=1), axis=2).sum(axis=1)
        low, high = _SAMPLES
        n = int(np.clip(np.ceil(polygon.max() / SPACING), low, high))
        s = np.linspace(0.0, 1.0, n)[:, None]
        weights = np.hstack(
            [(1 - s) ** 3, 3 * s * (1 - s) ** 2, 3 * s**2 * (1 - s), s**3]
        )
        sampled = np.einsum("sk,ckd->csd", weights, curves)
        out.append(sampled.reshape(-1, points.shape[1])[:, :2])
    if not out:
        return _as_points(mobject.get_center())
    return np.vstack(out)


def _measure(points: np.ndarray, axes: Any, margin: float) -> dict[str, Any]:
    """What ``axes.outside`` keeps of one evaluation: the points, the
    plotted rectangle shrunk by ``margin``, how far the worst point lies
    past it and where, in scene and data coordinates.
    """
    x_range = tuple(float(v) for v in axes.x_range[:2])
    y_range = tuple(float(v) for v in axes.y_range[:2])
    corners = np.array(
        [
            axes.c2p(x_range[0], y_range[0])[:2],
            axes.c2p(x_range[1], y_range[1])[:2],
        ],
        dtype=float,
    )
    low = corners.min(axis=0) + margin
    high = corners.max(axis=0) - margin
    past, point, data = 0.0, None, None
    if len(points):
        beyond = np.maximum(np.maximum(low - points, points - high), 0.0).max(axis=1)
        worst = int(np.argmax(beyond))
        past = float(beyond[worst])
        point = tuple(float(v) for v in points[worst])
        if past > 0 and hasattr(axes, "p2c"):
            data = tuple(float(v) for v in axes.p2c(np.array([*point, 0.0]))[:2])
    return {
        "points": points,
        "low": tuple(float(v) for v in low),
        "high": tuple(float(v) for v in high),
        "past": past,
        "point": point,
        "data": data,
        "x_range": x_range,
        "y_range": y_range,
        "axes": type(axes).__name__,
        "px_per_unit": float(config.pixel_width) / float(config.frame_width),
    }


# Findings


def _past(reading: Reading | None) -> bool:
    return (
        reading is not None
        and reading.values is not None
        and reading.values["past"] > 0
    )


def _finding(
    call: BridgeCall,
    reading: Reading,
    t_end: Fraction,
    line: SourceLine | None,
) -> Finding | None:
    """The finding of ``reading``, held until ``t_end`` and found at
    ``line``, if a point lies past the plot.
    """
    if not _past(reading):
        return None
    values = reading.values
    assert values is not None
    x, y = values["point"]
    past = values["past"]
    where = f"({x:.3f}, {y:.3f})"
    data = values["data"]
    if data is not None:
        where += f", data ({data[0]:.3f}, {data[1]:.3f}),"
    since = (
        f", watched since line {call.line.line}"
        if call.kind == "watch" and call.line is not None
        else ""
    )
    margin = f" less a margin of {call.margin:g}" if call.margin else ""
    message = (
        f"{call.subject} is off the plotted region of {values['axes']}"
        f" {call.reference}{since}: its point {where} lies {past:.3f} units"
        f" ({round(past * values['px_per_unit'])} px) past it, where x_range is"
        f" {list(values['x_range'])} and y_range {list(values['y_range'])}{margin}"
    )
    return Finding(
        "axes.outside",
        reading.t,
        t_end,
        message=message,
        line=line,
        detail={"axes": call.reference, "point": [x, y]},
    )


def _record_findings(call: BridgeCall) -> list[Finding]:
    (reading,) = call.readings
    found = _finding(call, reading, reading.t, call.line)
    return [found] if found is not None else []


def _watch_findings(call: BridgeCall, fps: Fraction, end: Fraction) -> list[Finding]:
    """A watch read at each frame of the render's grid (``fps``) from its
    call on, and at the scene's ``end``: each run of frames outside is one
    finding from its first frame to its last, at the line running when it
    began. A reading shows on the frames in the span its state holds for
    (a ticked play's capture is of a tick, which the frames after it show);
    a frame shows the last reading that covers it.
    """
    frames: dict[Fraction, Reading] = {}
    for reading in call.readings:
        if reading.until is not None:
            # No frame past the scene's end, whatever tick the state held to.
            shown = frames_in(reading.t, min(reading.until, end), fps)
        elif (reading.t * fps).denominator == 1 or reading.t == end:
            shown = [reading.t]
        else:
            shown = []
        for t in shown:
            frames[t] = reading
    found: list[Finding] = []
    run: list[tuple[Fraction, Reading]] = []

    def close() -> None:
        if run:
            first_t, first = run[0]
            finding = _finding(call, first, run[-1][0], first.line or call.line)
            assert finding is not None
            found.append(replace(finding, t_start=first_t))
            run.clear()

    for t in sorted(frames):
        if _past(frames[t]):
            run.append((t, frames[t]))
        else:
            close()
    close()
    return found


def _violations(found: Iterable[Finding]) -> list[dict[str, Any]]:
    from .serialise import violation

    return [violation(f) for f in found]
