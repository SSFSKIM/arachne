"""The assertion bridge: checks only the scene can state, recorded as it
runs and merged into the lint.

Some things the lint can't infer from what is drawn: that a point the scene
computes, or a pen its updaters drive, must stay on an axes' plotted region.
The scene says so through the bridge, and the record keeps each call
(:class:`~arachne.timeline.recorder.BridgeCall`, per play):

- :func:`record` checks operands at the instant it is called;
- :func:`watch` checks a mobject, or a function of the scene, at every
  frame from the call on, read while the scene records and kept as values,
  so the lint never calls the scene's code;
- :func:`declare` keeps terms for the lint's rules (``text_gap``, ``pad``,
  ``exempt``).

A call returns what it found as violations in the report's shape
(:mod:`.serialise`) and never raises for a finding; :func:`~arachne.lint.lint`
reports each at its line and time. Outside a recording, which every
execution of a scene is, a call raises :class:`RuntimeError`.

The one code checked today is ``axes.outside``: every point of the operands
inside the rectangle the axes' ranges span, shrunk by ``margin`` on every
side, in scene units. A mobject operand of :func:`record` is checked along
its outline, sampled about every :data:`SPACING` units, or at its centre
when it has no points; a watched mobject is checked at its centre. A
watch's finding is a run of frames outside, at the scene line
running when the run began.
"""

from __future__ import annotations

__all__ = [
    "CHECKED",
    "SPACING",
    "TERMS",
    "Checks",
    "declare",
    "findings",
    "record",
    "watch",
]

import sys
from collections.abc import Iterable, Mapping
from dataclasses import replace
from fractions import Fraction
from typing import TYPE_CHECKING, Any

import numpy as np

from arachne._config import config
from arachne.timeline.clock import frames_in
from arachne.timeline.origins import origin_of
from arachne.timeline.recorder import BridgeCall, Reading, Recorder, running

from .findings import Finding

if TYPE_CHECKING:
    from arachne.timeline.record import Record
    from arachne.timeline.source import SourceLine

    from .core import LintContext
    from .frames import Frames


#: The codes :func:`record` and :func:`watch` check.
CHECKED = ("axes.outside",)

#: The terms :func:`declare` takes.
TERMS = ("exempt", "pad", "text_gap")

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
) -> list[dict[str, Any]]:
    """Check ``operands`` for ``code`` against ``reference`` at every frame
    from now on: a mobject at its centre, or a function of no arguments
    returning a point or points (``None`` for nothing to check), each read
    as the scene records. Returns what the check finds now.
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
    return _violations([found] if found is not None else [])


def declare(**terms: Any) -> list[dict[str, Any]]:
    """Declare terms for the lint's rules from now on (``text_gap``,
    ``pad``, ``exempt``: mobjects, and pairs of them, kept as stable ids).
    They are kept on the record; nothing is found by declaring.
    """
    recorder = _recorder("declare")
    unknown = sorted(set(terms) - set(TERMS))
    if unknown:
        raise TypeError(f"declare takes {', '.join(TERMS)}; not {', '.join(unknown)}")
    kept = {
        name: _ids(value) if name == "exempt" else float(value)
        for name, value in terms.items()
    }
    recorder.bridge_call("declare", sys._getframe(1), terms=kept)
    return []


class Checks:
    """The lint's reader of the bridge (:mod:`.frames`' client): the findings
    of the bridge's calls, where the render's frames see them. Each record is
    found at its call, and each watch at every frame of the render from its
    call on, a run of frames outside one finding from the first to the last,
    as a check made on every frame finds it.

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

    def request_frames(self, frames: Frames) -> None:
        """Ask that a forward run, if the lint needs one, run to the end."""
        frames.complete(self)

    def __call__(self, context: LintContext) -> list[Finding]:
        return findings(context.frames.execution)


def findings(source: Record) -> list[Finding]:
    """The findings of the bridge's calls in ``source``, an execution on the
    render's grid.
    """
    from .instants import render_grid

    fps = render_grid(source)
    found: list[Finding] = []
    for call in source.bridge.calls:
        if call.kind == "record":
            found.extend(_record_findings(call))
        elif call.kind == "watch":
            found.extend(_watch_findings(call, fps, source.duration))
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


def _ids(value: Any) -> Any:
    """``value`` with each mobject in it as its stable id."""
    from arachne.mobject.mobject import Mobject

    if isinstance(value, Mobject):
        return origin_of(value).id
    if isinstance(value, (list, tuple)):
        return tuple(_ids(item) for item in value)
    return value


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
