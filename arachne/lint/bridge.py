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
  motion rule, which the lint then runs unasked, and the states the
  temporal rules read (:mod:`.watch`, :mod:`.departures`), which hold
  the scene to the arming in force.

A call returns what it found as violations in the report's shape
(:mod:`.serialise`) and never raises for a finding; :func:`~arachne.lint.lint`
reports each at its line and time. Outside a recording, which every
execution of a scene is, a call raises :class:`RuntimeError`.

An exemption names mobjects, selectors (:mod:`.exempt`) and pairs of them;
each mobject stands for its family's stable ids as they are where the
exemption is read: at a sweep, the sweep; for a rule the lint runs, the
instant it checks (as the record read the scene there); for a check of the
whole scene, its end (:func:`exemptions`). A sweep is held to the terms it
is given, then to those declared, then to the rules' defaults; a rule the
lint runs is held to its own terms with the declared ``pad`` and
``text_gap`` in their place and the declared exemptions beside its own.
Each arming states ``pad`` and ``text_gap`` afresh.

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
    "armed_entries",
    "armed_terms",
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
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from fractions import Fraction
from types import FrameType
from typing import TYPE_CHECKING, Any

import numpy as np

from arachne._config import config
from arachne.timeline.clock import exact, frames_in
from arachne.timeline.origins import origin_of
from arachne.timeline.recorder import Bridge, BridgeCall, Reading, Recorder, running
from arachne.timeline.source import names_from_frames, scene_frames

from .clearance import TEXT_FLOOR, arguments, clear_of
from .exempt import Exemptions
from .findings import Finding
from .motion import JUMP_TOL, MotionJump
from .parts import Parts, build_parts
from .rules import DEFAULT_PAD, Terms, sweep_findings

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
        Reading(call.t, call.line, _measure(points, reference, margin), seq=call.seq)
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
    reading = Reading(call.t, call.line, read(), seq=call.seq)
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
    _declaring(_recorder("declare"), "declare", sys._getframe(1), terms)
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
    :class:`~.rules.Terms` defaults it; every exemption is read as it
    stands now. Returns what it finds.
    """
    recorder = _recorder("sweep")
    bridge = recorder.ensure_record().bridge
    now = _held(bridge.calls, lambda call: _exempt_now(bridge, call))
    caller = sys._getframe(1)
    call = _declaring(recorder, "sweep", caller, terms)
    held = _apply(_apply(Terms(), now), call.terms)
    parts = _parts_now(recorder, call, _names(recorder, caller))
    return _found(call, sweep_findings(parts, held))


def apart(
    subject: Any,
    against: Any,
    *,
    min_gap: float | None = None,
    pad: float = DEFAULT_PAD,
) -> list[dict[str, Any]]:
    """Check now that ``subject`` stands clear of ``against`` (a mobject, a
    ``(label, mobject)`` pair, or a list of them), each measured whole,
    added to the scene or not (:mod:`.clearance`): by ``min_gap`` when
    given, else two texts by a line of the shorter (at least
    :data:`TEXT_FLOOR`), a ring around some of a text's glyphs by touching
    only, anything else by ``pad``. An axes or number line stands for its
    furniture. Returns what it finds.
    """
    recorder = _recorder("apart")
    caller = sys._getframe(1)
    # Read once: a generator of targets gives them up once.
    given = arguments(subject, against)
    found = clear_of(
        given,
        min_gap=min_gap,
        pad=float(pad),
        names=_names(recorder, caller),
    )
    call = recorder.bridge_call(
        "apart",
        caller,
        operands=tuple(origin_of(mob).id for mob, _ in given),
        terms={"min_gap": min_gap, "pad": float(pad)},
        subject=_subject(given[0][0]),
    )
    return _found(call, found)


def arm(*, motion_jump: float | None = None, **terms: Any) -> list[dict[str, Any]]:
    """Arm the checks a scene asks for over time from now on: the motion
    rule (``motion.jump`` beyond ``motion_jump``, by default the rule's own
    tolerance), which the lint then runs unasked, its record keeping each
    play's exit from the next to end; and for the temporal rules
    (:mod:`.watch`, :mod:`.departures`), held to the arming's terms, the
    record keeps the scene's states from here on (``Record.states``), so
    that they read it rather than a forward run. ``terms`` are declared as by
    :func:`declare`, ``pad`` and ``text_gap`` at their defaults when not
    given: each arming states both afresh, its exemptions adding to those
    before. Arming again sets the tolerance the whole scene is measured at.
    """
    recorder = _recorder("arm")
    _declaring(
        recorder,
        "arm",
        sys._getframe(1),
        {"pad": DEFAULT_PAD, "text_gap": None, **terms},
        motion_jump=JUMP_TOL if motion_jump is None else float(motion_jump),
    )
    recorder.arm_motion()
    recorder.arm_states()
    return []


# The terms declared


def declared(
    source: Record,
    t: Fraction | float,
    *,
    play: int | None = None,
    seq: int | Fraction | None = None,
) -> dict[str, Any]:
    """The terms ``source``'s scene declared at or before ``t`` (by
    :func:`declare` and :func:`arm`): a later ``pad`` or ``text_gap``
    replacing an earlier, exemptions adding up, each read as it stood at
    ``t``. With ``play``, at an instant of that play (as the script called
    it, while it ran, or as it ended): no call made after it, between it and
    the next at its end, holds there. With ``seq``, at the state of that seq
    the record keeps (``Record.states``): only what the script did before
    it holds there, a call or a change made after it at the same instant
    not.
    """
    at = exact(t)
    return _held(
        (call for call in source.bridge.calls if _by(call, at, play, seq)),
        lambda call: _exempt_at(call, at, seq=seq),
    )


def terms_at(
    source: Record,
    t: Fraction | float,
    base: Terms,
    *,
    play: int | None = None,
    seq: int | Fraction | None = None,
) -> Terms:
    """``base`` under the terms declared at or before ``t`` (within
    ``play`` and before ``seq``, as :func:`declared` reads them): the
    declared ``pad`` and ``text_gap`` replace its own, and the declared
    exemptions, as they stood at ``t``, hold beside its own.
    """
    return _apply(base, declared(source, t, play=play, seq=seq), join=True)


def exemptions(source: Record, t: Fraction | float | None = None) -> Exemptions:
    """Every exemption ``source``'s scene stated by ``t`` (by default, its
    end), declared, armed or swept with, each read as it stood at ``t`` and
    kept once: what a check of the scene up to then, read at ``t``, is held
    to.
    """
    at = source.duration if t is None else exact(t)
    kept = Exemptions()
    for call in source.bridge.calls:
        if (
            call.t <= at
            and call.kind in (*_DECLARING, "sweep")
            and "exempt" in call.terms
        ):
            kept = _joined(kept, _exempt_at(call, at, "hold"))
    return kept


def armed_entries(
    source: Record,
    t: Fraction | float,
    *,
    play: int | None = None,
    seq: int | Fraction | None = None,
) -> Exemptions:
    """The entries the value the arming in force at ``t`` was handed held
    there (within ``play`` and before ``seq``, as :func:`declared` reads
    them), each read as it stands at the scene's end: what a check made
    there adds to the declaration a check of the whole scene is held to, as
    the reference checks' departure guard adds its arming's to the hold
    watch's. None before any arming.
    """
    at = exact(t)
    arming = armed(source, at, play=play, seq=seq)
    if arming is None or "exempt" not in arming.terms:
        return Exemptions()
    named: tuple[Any, ...] = ()
    seen: Mapping[Any, Exemptions] = {}
    for reading in arming.readings:
        values = reading.values
        if values is None or "seen" not in values:
            continue
        if reading.t <= at and _before(reading.seq, seq):
            named = values["named"]
        seen = values["seen"]
    kept = Exemptions()
    for key in named:
        kept = _joined(kept, seen[key])
    return kept


def armed(
    source: Record,
    t: Fraction | float | None = None,
    *,
    play: int | None = None,
    seq: int | Fraction | None = None,
) -> BridgeCall | None:
    """The arming in force at ``t`` (by default, at the scene's end; within
    ``play`` and before ``seq``, as :func:`declared` reads them): the last
    :func:`arm` call at or before it, whose terms hold its ``motion_jump``
    and the terms it declared; ``None`` before any.
    """
    at = None if t is None else exact(t)
    found = None
    for call in source.bridge.calls:
        if call.kind == "arm" and (at is None or _by(call, at, play, seq)):
            found = call
    return found


def armed_terms(
    source: Record,
    t: Fraction | float | None = None,
    *,
    play: int | None = None,
    seq: int | Fraction | None = None,
) -> Terms | None:
    """What the arming in force at ``t`` (by default, the scene's end;
    within ``play`` and before ``seq``, as :func:`declared` reads them)
    holds a check at ``t`` to: its own ``pad`` and ``text_gap``, and its
    exemptions as the value it was handed stood at ``t`` (not those of
    earlier armings, which :func:`exemptions` accumulates); ``None`` before
    any arming.
    """
    at = source.duration if t is None else exact(t)
    arming = armed(source, at, play=play, seq=seq)
    if arming is None:
        return None
    exempt = (
        _exempt_at(arming, at, seq=seq) if "exempt" in arming.terms else Exemptions()
    )
    return Terms(arming.terms["pad"], arming.terms["text_gap"], exempt)


class _ArmedMotion(MotionJump):
    """The motion rule a scene armed: one per lint, its frames asked for
    under its own identity, its tolerance set from the arming of the
    execution the lint reads once that is known.
    """

    __hash__ = object.__hash__

    def __eq__(self, other: object) -> bool:
        return self is other


class _IfForward:
    """A lint's frames, as a reader asks for them that needs them only from
    a forward run, if the lint makes one: such a run keeps every play's
    exit, so it asks for none.
    """

    def __init__(self, frames: Frames) -> None:
        self._frames = frames
        self.session = frames.session

    def request(self, owner: Any, times: Any, read: Any) -> None:
        self._frames.request(owner, times, read, forward_only=True)

    def require_forward(self) -> None:
        pass


class Checks:
    """The lint's reader of the bridge (:mod:`.frames`' client): the findings
    of the bridge's calls, where the render's frames see them, and of the
    checks over time the scene armed (:func:`arm`), which run unasked unless
    the lint was given them as ``rules``: today the motion rule, at the
    tolerance of the execution's last arming. Each record, sweep and apart
    is found at its call, and each watch at every frame of the render from
    its call to its clear, a run of frames outside one finding from the
    first to the last, as a check made on every frame finds it.

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
        asked = any(isinstance(rule, MotionJump) for rule in rules)
        self._motion = None if asked else _ArmedMotion()

    def request_frames(self, frames: Frames) -> None:
        """Ask that a forward run, if the lint needs one, run to the end; and
        for the frames the armed motion rule reads: as any rule asks, when
        the session's record armed it, else only from a forward run, whose
        own arming is known only once it has run.
        """
        frames.complete(self)
        if self._motion is None:
            return
        if armed(frames.session.record) is not None:
            self._motion.request_frames(frames)
        else:
            self._motion.request_frames(_IfForward(frames))  # type: ignore[arg-type]

    def __call__(self, context: LintContext) -> list[Finding]:
        execution = context.frames.execution
        found = findings(execution)
        arming = armed(execution)
        motion = self._motion
        if (
            motion is not None
            and arming is not None
            and context.frames.answered(motion)
        ):
            object.__setattr__(motion, "jump_tol", arming.terms["motion_jump"])
            found.extend(motion(context))
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
            found.extend(_call_findings(call))
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


def _by(
    call: BridgeCall, t: Fraction, play: int | None, seq: int | Fraction | None = None
) -> bool:
    """Whether ``call`` was made by ``t``, with ``play`` by that play's end
    (a call between plays belongs to the next, ``BridgeCall.play``), and
    with ``seq`` before the capture of that seq.
    """
    return (
        call.t <= t and (play is None or call.play <= play) and _before(call.seq, seq)
    )


def _before(made: int | Fraction, seq: int | Fraction | None) -> bool:
    """Whether what the script did after the capture ``made`` (a call, a
    reading: their ``seq``) came before the capture ``seq``, when given.
    """
    return seq is None or made < seq


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
    family's stable ids as it is now.
    """
    return Exemptions.parse(_entries(exempt))


def _entries(exempt: Any) -> list[Any]:
    """The entries of ``exempt`` as they are now: a lone mobject or a lone
    pair of mobjects is one.
    """
    from arachne.mobject.mobject import Mobject

    if isinstance(exempt, Mobject) or (
        isinstance(exempt, tuple)
        and len(exempt) == 2
        and all(isinstance(item, Mobject) for item in exempt)
    ):
        return [exempt]
    return list(exempt or ())


def _joined(first: Exemptions, second: Exemptions) -> Exemptions:
    """The exemptions of both, each once."""
    return Exemptions(
        tuple(dict.fromkeys((*first.bare, *second.bare))),
        tuple(dict.fromkeys((*first.pairs, *second.pairs))),
    )


def _held(
    calls: Iterable[BridgeCall], exempt_of: Callable[[BridgeCall], Exemptions]
) -> dict[str, Any]:
    """The terms ``calls`` declare, in order: a later ``pad`` or
    ``text_gap`` replacing an earlier, exemptions (each call's as
    ``exempt_of`` reads it) adding up.
    """
    held: dict[str, Any] = {}
    for call in calls:
        if call.kind not in _DECLARING:
            continue
        for name in TERMS:
            if name not in call.terms:
                continue
            value = exempt_of(call) if name == "exempt" else call.terms[name]
            if name == "exempt" and name in held:
                value = _joined(held[name], value)
            held[name] = value
    return held


def _declaring(
    recorder: Recorder,
    kind: str,
    caller: FrameType,
    terms: Mapping[str, Any],
    **extra: Any,
) -> BridgeCall:
    """Keep a call stating ``terms`` (and ``extra`` beside them). Its
    exemptions name what the scene's mobjects hold as each reader asks, read
    two ways: ``exempt``, what the value the scene handed over holds as it
    stands (a list it changes later names what it holds then), and
    ``hold``, the entries it held at the call, whose mobjects' families are
    read as they stand: what a check of the whole scene accumulates. Beside
    them, ``named`` keys the entries the value holds as it stands, and
    ``seen`` holds, by key, every entry it has held, read as it stands
    (:func:`armed_entries`). All are read at the call and again whenever a
    kept capture finds them changed, and can be read while the scene runs
    (:meth:`.Bridge.read_now`). The value is iterated once at the call: a
    one-shot iterable (a generator) gives up its entries there, to the terms
    and ``hold``, and holds none when read again.
    """
    reader = None
    if "exempt" in terms:
        named = terms["exempt"]
        entries = _entries(named)
        terms = {**terms, "exempt": entries}
        seen: dict[Any, Any] = {}

        def reader() -> Mapping[str, Any]:
            now = _entries(named)
            keys = tuple(_entry_key(entry) for entry in now)
            for key, entry in zip(keys, now, strict=True):
                seen.setdefault(key, entry)
            return {
                "exempt": Exemptions.parse(now),
                "hold": Exemptions.parse(entries),
                "named": keys,
                "seen": {k: Exemptions.parse([e]) for k, e in seen.items()},
            }

    kept = {**_terms(terms), **extra}
    call = recorder.bridge_call(
        kind, caller, terms=kept, reader=reader, changes_only=True
    )
    if reader is not None:
        call.readings.append(Reading(call.t, call.line, reader(), seq=call.seq))
    return call


def _entry_key(entry: Any) -> Any:
    """What identifies an exemption's entry while the scene runs: a mobject
    by its identity, a selector by its text, a pair by its two.
    """
    from arachne.mobject.mobject import Mobject

    if isinstance(entry, Mobject):
        return ("mobject", id(entry))
    if isinstance(entry, (tuple, list)):
        return tuple(_entry_key(item) for item in entry)
    return ("selector", entry)


def _exempt_now(bridge: Bridge, call: BridgeCall) -> Exemptions:
    """``call``'s exemptions as they stand now."""
    read = bridge.read_now(call)
    assert read is not None
    now: Exemptions = read["exempt"]
    return now


def _exempt_at(
    call: BridgeCall,
    t: Fraction,
    way: str = "exempt",
    *,
    seq: int | Fraction | None = None,
) -> Exemptions:
    """``call``'s exemptions as they stood at ``t``, read ``way``
    (:func:`_declaring`): their last reading at or before it, and with
    ``seq`` before the capture of that seq.
    """
    found: Exemptions = call.terms["exempt"]
    for reading in call.readings:
        if reading.t > t or not _before(reading.seq, seq):
            break
        if reading.values is not None and way in reading.values:
            found = reading.values[way]
    return found


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


def _parts_now(recorder: Recorder, call: BridgeCall, names: dict[int, str]) -> Parts:
    """The parts the scene draws now, as its camera views them. Nothing is
    captured on the record.
    """
    from arachne.timeline.record import Capture

    scene = recorder.scene
    capture = Capture(
        -1,
        call.t,
        "bridge",
        tuple(scene.mobjects),
        tuple(scene.foreground_mobjects),
        (),
        (),
        None,
    )
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


def _found(call: BridgeCall, found: Iterable[Finding]) -> list[dict[str, Any]]:
    """``found`` at ``call``'s time and line, kept as its reading; returned
    as report violations.
    """
    kept = tuple(
        replace(finding, t_start=call.t, t_end=call.t, line=call.line)
        for finding in found
    )
    call.readings.append(Reading(call.t, call.line, {"findings": kept}, seq=call.seq))
    return _violations(kept)


def _call_findings(call: BridgeCall) -> tuple[Finding, ...]:
    """What a sweep or an apart found at its call."""
    (reading,) = (r for r in call.readings if r.values and "findings" in r.values)
    assert reading.values is not None
    found: tuple[Finding, ...] = reading.values["findings"]
    return found


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
