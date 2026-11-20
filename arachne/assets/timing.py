"""Stepped timing: an illustration drawn a declared number of times a
second of scene time, the same at every frame rate, while everything else
on screen stays smooth.

:func:`set_timing` declares a mobject and its whole family as one
**illustration** drawn ``drawings_per_second`` times a second (a divisor of
60, so that each boundary ``k / r`` is one of the scene's 60 Hz ticks),
optionally with **line boil** (:class:`Boil`): its outlines redrawn with a
small wobble that changes on a grid of its own. The declaration is an
immutable :class:`Timing` value; the declared mobject holds its
:attr:`~Timing.mark`, a string of its numbers, so the record holds it and a
content key reads it; :func:`timing_of` reads it back.

How it holds
------------

Nothing wraps or patches an animation. The first declaration in a scene
registers one engine callback, the scene's **clock**, last among its
updaters; the scene owns it, through that updater. While a play runs, each
member of a declared family in the scene carries one engine updater, its
**restorer**, which the clock gives it; when the play ends the first to run
puts back what is true and takes every restorer the clock gave out. Both
live in that scene alone: a scene that declares nothing has neither, and
plays and draws as it always did; a mobject declared in a scene, its
copies, and a group holding it play in a later scene as they did before it.

The clock reads :attr:`~.Scene.time`, the exact scene time, and never adds
up the ``dt`` it is handed (it reads it only so that every tick runs it).
With it among the scene's updaters every play of the scene is ticked and no
wait is a frozen frame, so it runs at every 60 Hz tick, in order, at every
frame rate. For each illustration it keeps two states, each the family's
attribute values and lists of submobjects and updaters as the record keeps
them (:class:`~arachne.timeline.record.NodeState`), which are immutable
values, so keeping one copies no geometry:

- **true**: the family as the scene's code and the animations last wrote it;
- **shown**: the family as it is on display.

At every instant, last: what wrote the family since the clock last
installed a state becomes true (only what was written: a held value left in
place is not a write); the rigs of the family are settled on true; then, if
the illustration entered the scene, its family changed members, or the
instant crossed a boundary ``k / r``, shown becomes true; shown is
installed, with the boil of the instant's boil step added. When a play ends
the restorer installs true, so the scene's code and the next play's
animations start from what is true; at a play's first instant the clock
then installs shown again. So at any frame rate the drawing on display at
``t`` is true as it was at the last boundary at or before ``t`` (or at the
instant the illustration entered the scene, or changed members, if that is
later): a function of the instant.

An off-grid play end keeps the drawing on display until the next boundary,
inside whatever play or wait holds it; if the scene ends first, no
delivered frame shows the play's final state. End an illustration's last
move on its grid, or follow it with a wait. The record asked at the
scene's end (``state_at(duration)``) answers the script's final state, as
it does for every scene, not the drawing held: no frame is drawn at an
instant equal to the scene's duration.

An annotation attached to a member of a declared family is put at its
anchor again once the drawing on display is installed, so its leader ends
on the drawing on screen, not on the pose true between two drawings (on a
boiled drawing, on its outline as painted: :mod:`.anchor`); the annotation
itself is never stepped. A copy of a declared mobject carries its mark, and
is declared too when it enters the scene that declared it.

The declared mobject declares its timing (:mod:`arachne.quantities`),
read from the mark a capture holds: ``timing.rate``, its drawings a second,
and with boil ``timing.boil_rate`` and ``timing.boil_amplitude``, in
changes a second (``1/s``) and in ``px``: one 1080th of the frame's shorter
side, ``min(frame_width, frame_height) / 1080`` scene units, not a device
pixel of the rendered frame.

Stated limit. A frozen frame (``pause``, ``wait(frozen_frame=True)``) runs
no updater, so it shows the family as true when it begins, without boil:
after an off-grid play end, the final state before its boundary.

What it refuses
---------------

Each as a raised finding, at the instant the clock sees it. Two declared
families that share a member, at the declaration or once a member joins a
second, are a plain ``ValueError``.

- ``timing.updater``: a scene's own updater on a member of a declared
  family (the hold needs writers that write whole values, and a scene's
  updater may build on the drawing on display); an animation on a member
  that is not one of the engine's classes, or is ``UpdateFromFunc`` or
  ``UpdateFromAlphaFunc``; an engine animation that moves its mobject from
  where it stands (``MoveAlongPath``, ``PhaseFlow``,
  ``MaintainPositionRelativeTo``), which between two drawings is the drawing
  on display; a ``Succession``'s part on a declared family that begins in
  the middle of a play after a part that does not animate that family (only
  that part's end restores what is true before the next begins); a member
  whose updating the scene's code suspended, which keeps its restorer from
  running; an animation that takes a declared family out of the scene as
  the play ends without resuming its updating then (it does not suspend it,
  or it is a Succession's part, which finishes earlier), which leaves no
  member to run its restorer; a scene updater registered after the clock,
  which must run last;
- ``timing.exact_content``: text, a number, a coordinate system or a plot,
  an annotation, a schematic mark, a diagram, or the scene's camera's frame
  in a declared family, when it is declared and at each play's first
  instant. What they show must change at every frame.

Scene drivers
-------------

A **scene driver** is engine work that keeps objects on the scene's clock
through one scene updater of its own (a diagram's flows, the first). The
clock takes every scene driver over, so it alone runs last. A driver is an
object

- whose attribute ``updater`` is a scene updater that reads
  :attr:`~.Scene.time` and never adds up ``dt``: it may run at any instant
  and any number of times there, with the same result;
- which, called with no argument, does its updater's work at the scene's
  current instant;
- which is findable on its scene by a function registered here with
  :func:`register_driver` (it takes the scene and gives its driver, or
  ``None``);
- which accepts later objects without adding its updater again, and is put
  in a scene by :func:`drive`, never by ``scene.add_updater``.

When the clock is registered it takes each found driver's updater out of
the scene's updaters, and at the start of every instant, before its hold,
calls the drivers in the order the scene registered them. :func:`drive`
hands a driver made in a scene that already has a clock to the clock. A
module that defines a driver registers its finder at import, without
editing this one.
"""

from __future__ import annotations

__all__ = ["Boil", "Timing", "set_timing", "timing_of"]

import copy
import functools
import math
import numbers
import weakref
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from .. import quantities
from ..mobject.values import FrozenPoints, adopt
from ..quantities import At, Quantity
from ..timeline.record import NodeState

if TYPE_CHECKING:
    from ..animation.animation import Animation
    from ..animation.composition import Succession
    from ..mobject.mobject import Mobject
    from ..scene.scene import Scene

#: The rates a drawing or a boil may change at, in changes a second: the
#: divisors of 60 below it, so every boundary falls on a tick.
DRAWING_RATES = (1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30)

#: The attribute of a declared mobject holding its :class:`Timing`.
MARK = "_timing"

#: The attribute of a member whose points hold a boiled drawing, while
#: they do: its outline anchors are read off those points.
BOILED = "_timing_boiled"

#: A boil's wobble: the length, in scene units, over which its displacement
#: turns about once, and how many waves make it.
BOIL_WAVELENGTH = 0.75
BOIL_WAVES = 4


def _rate(value: Any, what: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Integral)
        or int(value) not in DRAWING_RATES
    ):
        raise ValueError(
            f"{what} is a whole number of changes a second that divides 60, so"
            f" that every change falls on a tick: one of"
            f" {', '.join(map(str, DRAWING_RATES))}, not {value!r}."
        )
    return int(value)


@dataclass(frozen=True)
class Boil:
    """Line boil: an illustration's outlines redrawn ``rate`` times a second
    (a divisor of 60), each point moved by at most ``amplitude`` ``px``, a
    ``px`` being one 1080th of the frame's shorter side,
    ``min(frame_width, frame_height) / 1080`` scene units, not a device pixel
    of the rendered frame; by a wobble that is a function of the boil's step
    (``floor(t * rate)``), the point's place and ``seed``.

    The wobble is one smooth field over the picture, so points that
    coincide move together: a closed outline stays closed and parts that
    touch stay touching.
    """

    rate: int
    amplitude: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "rate", _rate(self.rate, "A boil's rate"))
        amplitude = self.amplitude
        if (
            isinstance(amplitude, bool)
            or not isinstance(amplitude, numbers.Real)
            or not math.isfinite(amplitude)
            or amplitude < 0
        ):
            raise ValueError(
                f"A boil's amplitude is a finite number of px (1080ths of the frame's"
                f" shorter side), 0 or more, not"
                f" {amplitude!r}."
            )
        object.__setattr__(self, "amplitude", float(amplitude))
        seed = self.seed
        if isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or seed < 0:
            raise ValueError(
                f"A boil's seed is a whole number, 0 or more, not {seed!r}."
            )
        object.__setattr__(self, "seed", int(seed))

    def step(self, t: Fraction) -> int:
        """The boil's step at the instant ``t``."""
        return math.floor(t * self.rate)

    def _wobble(self, step: int) -> Callable[[Any], np.ndarray]:
        """The wobble of the step ``step``: a function giving how far it
        moves each of the scene points ``(n, 3)`` it is handed, in the
        scene's units of the current configuration's frame.
        """
        from arachne import config

        unit = min(config.frame_width, config.frame_height) / 1080
        rng = np.random.default_rng([self.seed, step])
        directions = rng.uniform(0, 2 * math.pi, (2, BOIL_WAVES))
        phases = rng.uniform(0, 2 * math.pi, (2, BOIL_WAVES))
        waves = np.stack([np.cos(directions), np.sin(directions)], axis=-1)
        waves *= 2 * math.pi / BOIL_WAVELENGTH
        # Each axis's displacement is at most 1 before scaling, so a point
        # moves at most the amplitude.
        reach = self.amplitude * unit / math.sqrt(2)

        def displacement(points: Any) -> np.ndarray:
            xy = np.asarray(points, dtype=float)[:, :2]
            angles = np.einsum("akd,nd->ank", waves, xy) + phases[:, None, :]
            moved = np.zeros((len(xy), 3))
            moved[:, :2] = reach * np.sin(angles).mean(axis=2).T
            return moved

        return displacement


@dataclass(frozen=True)
class Timing:
    """An illustration's declared timing: drawn ``drawings_per_second``
    times a second of scene time, with ``boil`` (or none).
    """

    drawings_per_second: int
    boil: Boil | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "drawings_per_second",
            _rate(self.drawings_per_second, "An illustration's drawings_per_second"),
        )
        if self.boil is not None and not isinstance(self.boil, Boil):
            raise TypeError(f"boil is a Boil, not a {type(self.boil).__name__}")

    def step(self, t: Fraction) -> int:
        """The drawing's step at the instant ``t``."""
        return math.floor(t * self.drawings_per_second)

    @property
    def mark(self) -> str:
        """What a declared mobject holds of it, one string a content key
        reads: ``"drawings_per_second"``, or with boil
        ``"drawings_per_second rate amplitude seed"``.
        """
        boil = self.boil
        if boil is None:
            return str(self.drawings_per_second)
        return f"{self.drawings_per_second} {boil.rate} {boil.amplitude!r} {boil.seed}"


def _timing(mark: Any) -> Timing | None:
    """The timing whose :attr:`~Timing.mark` ``mark`` is, or ``None``."""
    if not isinstance(mark, str):
        return None
    fields = mark.split()
    if len(fields) == 1:
        return Timing(int(fields[0]))
    if len(fields) == 4:
        drawings, rate, amplitude, seed = fields
        return Timing(int(drawings), Boil(int(rate), float(amplitude), int(seed)))
    return None


def timing_of(mobject: Mobject) -> Timing | None:
    """The timing ``mobject`` was declared with (:func:`set_timing`), or
    ``None``. The mark is the declared mobject's own, not its members'; on a
    state the record installed, it is the timing declared then.
    """
    return _timing(mobject.__dict__.get(MARK))


# Declared quantities

#: The prefix of an illustration's declared quantities.
PREFIX = "timing."
quantities.register(PREFIX, owner="arachne.assets", units=("1/s", "px"))


@dataclass(frozen=True, slots=True)
class TimingNumbers:
    """The source of a declared mobject's timing quantities. It holds
    nothing: what it answers is read from the mark the capture holds, so a
    mobject declared again answers its new timing from then.
    """


@quantities.source(TimingNumbers)
def _declared(
    found: TimingNumbers, held: Mapping[str, Any], at: At
) -> Iterable[Quantity]:
    timing = _timing(held.get(MARK))
    if timing is None:
        return ()
    numbers = [Quantity(f"{PREFIX}rate", timing.drawings_per_second, "1/s")]
    boil = timing.boil
    if boil is not None:
        numbers += [
            Quantity(f"{PREFIX}boil_rate", boil.rate, "1/s"),
            Quantity(f"{PREFIX}boil_amplitude", boil.amplitude, "px"),
        ]
    return numbers


def _declare_numbers(root: Mobject) -> None:
    """Have ``root`` declare its timing's numbers, once."""
    held = root.__dict__.get(quantities.SOURCES, ())
    if not any(type(each) is TimingNumbers for each in held):
        quantities.attach(root, TimingNumbers())


# Scene drivers


class SceneDriver(Protocol):
    """What the clock takes over (see the module's docs, "Scene drivers")."""

    #: The scene updater the driver runs as without a clock.
    updater: Callable[[float], None]

    def __call__(self) -> None:
        """Do the updater's work at the scene's current instant."""


def _flow_driver(scene: Scene) -> SceneDriver | None:
    from ..mobject.diagram import flow_driver

    return flow_driver(scene)


#: How each kind of scene driver is found on a scene, in the order the
#: kinds were registered: a diagram's flows first.
_FINDERS: list[Callable[[Scene], SceneDriver | None]] = [_flow_driver]


def register_driver(find: Callable[[Scene], SceneDriver | None]) -> None:
    """Have the clock take over the scene drivers ``find`` finds: given a
    scene, it gives that scene's driver, or ``None``. Registering a finder
    twice registers it once.
    """
    if find not in _FINDERS:
        _FINDERS.append(find)


def drive(scene: Scene, driver: SceneDriver) -> None:
    """Put ``driver`` to work in ``scene``: by the scene's clock when it has
    one, else as one of the scene's updaters.
    """
    clock = clock_of(scene)
    if clock is not None:
        clock.adopt(driver)
    elif driver.updater not in scene.updaters:
        scene.add_updater(driver.updater)


def _found_drivers(scene: Scene) -> list[SceneDriver]:
    found: list[SceneDriver] = []
    for find in _FINDERS:
        driver = find(scene)
        if driver is not None and all(driver is not d for d in found):
            found.append(driver)
    return found


# The restorer


def restore_true(mobject: Mobject, dt: float) -> None:
    """The updater of each member of a declared family in the scene while a
    play runs: the first to be handed ``dt == 0`` when the play ends has
    every family hold what is true, so the scene's code and the next play's
    animations start from it, and takes the restorer out of every member
    (:meth:`TimingClock.end_play`). Whichever member still runs then does
    it: one the play took out of the scene runs as its animation resumes
    it. The clock gives the members the restorer again at the next instant
    it holds them, so between two plays, and once their scene has ended,
    they have no updater of stepped timing. It reads ``dt``, so it runs at
    every tick; elsewhere it does nothing.
    """
    if dt != 0:
        return
    scene = _running_scene()
    clock = clock_of(scene) if scene is not None else None
    if clock is not None and clock.arms(mobject):
        clock.end_play()
    _release(mobject)


class _Restoring(list):
    """A declared member's updaters while its clock holds it, the restorer
    last: a copy of it, made in the middle of a play, leaves the restorer
    out, so that no copy outlives the scene with it.
    """

    def __deepcopy__(self, memo: dict[int, Any]) -> list[Any]:
        copied: list[Any] = []
        memo[id(self)] = copied
        copied.extend(copy.deepcopy(u, memo) for u in self if u is not restore_true)
        return copied


def _arm(member: Mobject) -> None:
    """Give the declared ``member`` the restorer, last among its updaters."""
    updaters = member.updaters
    if type(updaters) is _Restoring and updaters and updaters[-1] is restore_true:
        return
    member.updaters = _Restoring([u for u in updaters if u is not restore_true])
    member.updaters.append(restore_true)


def _release(mobject: Mobject) -> None:
    """Take the restorer out of ``mobject``'s updaters."""
    if any(u is restore_true for u in mobject.updaters):
        mobject.updaters = [u for u in mobject.updaters if u is not restore_true]


def _engine_updaters() -> tuple[Callable[..., Any], ...]:
    """The engine's own updaters a member of a declared family may hold."""
    from .rig import keep_settled

    return (restore_true, keep_settled)


# Illustrations


_State = dict[int, tuple["Mobject", NodeState]]


def _capture(members: list[Mobject], previous: _State | None = None) -> _State:
    """Each of ``members``' states now, by id: a previous state itself where
    nothing changed.
    """
    captured: _State = {}
    for member in members:
        before = previous.get(id(member)) if previous is not None else None
        state = NodeState.of(member, before[1] if before is not None else None)
        captured[id(member)] = (member, state)
    return captured


def _members(state: _State) -> tuple[int, ...]:
    return tuple(state)


class Illustration:
    """One declared family, kept by its scene's clock: its root, its
    timing, what is true and what is shown, and the state the clock last
    installed in it.
    """

    def __init__(self, root: Mobject, timing: Timing, now: Fraction) -> None:
        self.root = root
        self.timing = timing
        members = root.get_family()
        state = _capture(members)
        self.true: _State = state
        self.shown: _State = state
        #: What the clock last left the family holding.
        self.installed: _State = state
        self.step = timing.step(now)
        #: Whether the root was in the scene at the last instant.
        self.present: bool | None = None

    # The true state

    def _absorb(self, members: list[Mobject]) -> None:
        """Have every member hold what is true: what was written since the
        clock installed its state, over true's values for the rest.
        """
        installed, true = self.installed, self.true
        for member in members:
            kept = installed.get(id(member))
            held = true.get(id(member))
            if kept is None or held is None:
                # A member that joined since: what it holds is true.
                continue
            was, truth = kept[1], held[1]
            own = member.__dict__
            if was.matches(own):
                if truth is not was:
                    truth.install(member)
                continue
            _merge(own, was, truth)

    def restore(self, posed: frozenset[int] = frozenset()) -> None:
        """Make the family hold what is true now, its rigs settled on it:
        those a pose holds (their ids ``posed``), and those whose own
        updater runs, as it would at a frame.
        """
        members = self.root.get_family()
        self._absorb(members)
        _settle(members, posed)
        self.true = _capture(members, self.true)
        self.installed = self.true

    # One instant

    def at(self, now: Fraction, present: bool, posed: frozenset[int]) -> list[Mobject]:
        """The clock's work at the instant ``now``, the root in the scene
        or not, the rigs ``posed`` held by a pose: the members it armed
        with the restorer (none when the root is not in the scene).
        """
        if not present:
            if self.present:
                self.restore(posed)
            self.present = False
            return []
        members = self.root.get_family()
        for member in members:
            _arm(member)
        self.restore(posed)
        timing = self.timing
        step = timing.step(now)
        entered = not self.present
        self.present = True
        if entered or _members(self.true) != _members(self.shown) or step != self.step:
            self.shown = self.true
            self.step = step
        boil = timing.boil
        if boil is None and self.shown is self.true:
            return members
        true = self.true
        for member, state in self.shown.values():
            if not state.matches(member.__dict__):
                state.install(member)
                _keep_running(member, true[id(member)][1])
        if boil is not None:
            _boiled(self.shown, boil, boil.step(now))
        # A state shown from before a member was armed holds its updaters
        # in a list of another kind.
        for member in members:
            _arm(member)
        self.installed = _capture([m for m, _ in self.shown.values()], self.shown)
        return members


def _keep_running(member: Mobject, truth: NodeState) -> None:
    """Give ``member``, holding the drawing on display, what is true of how
    it runs: whether its updating is suspended and which updaters it has.
    They say what runs now, not what is drawn: held from an earlier drawing,
    they would keep the restorer from running at the end of a play.
    """
    own = member.__dict__
    suspended = truth.get("updating_suspended")
    if own.get("updating_suspended") is not suspended:
        own["updating_suspended"] = suspended
    updaters = own.get("updaters")
    if updaters is not None and (
        len(updaters) != len(truth.updaters)
        or not all(a is b for a, b in zip(updaters, truth.updaters, strict=True))
    ):
        updaters[:] = truth.updaters


def _settle(members: list[Mobject], posed: frozenset[int]) -> None:
    """Settle each rig among ``members`` that a pose holds (its id in
    ``posed``) or whose own updater runs; one an animation drives whole, its
    updating suspended, keeps the art the animation wrote.
    """
    from .rig import Rig, keep_settled

    for member in members:
        if not isinstance(member, Rig):
            continue
        if id(member) in posed or (
            not member.updating_suspended
            and any(u is keep_settled for u in member.updaters)
        ):
            member.settle()


def _merge(own: dict[str, Any], was: NodeState, truth: NodeState) -> None:
    """Put back, in a member's attributes ``own``, true's value of each
    attribute the writer left as the clock installed it (``was``): the
    attributes and lists written since keep what was written.
    """
    installed = dict(zip(was.keys, was.values, strict=True))
    true = dict(zip(truth.keys, truth.values, strict=True))
    for key in list(own):
        if key in installed and own[key] is installed[key]:
            if key in true:
                own[key] = true[key]
            else:
                del own[key]
    for key, value in true.items():
        if key not in own and key not in installed:
            own[key] = value
    for name, then, now in (
        ("submobjects", was.submobjects, truth.submobjects),
        ("updaters", was.updaters, truth.updaters),
    ):
        live = own.get(name)
        if (
            live is not None
            and len(live) == len(then)
            and all(a is b for a, b in zip(live, then, strict=True))
        ):
            live[:] = now


def _boiled(shown: _State, boil: Boil, step: int) -> None:
    """Move each outline point of the family's members, holding ``shown``,
    by the boil's wobble at its step ``step``.
    """
    if boil.amplitude == 0:
        return
    wobble = boil._wobble(step)
    for member, _ in shown.values():
        if getattr(member, "_scaffold", False):
            continue
        points = member.__dict__.get("_points")
        if points is None or not len(points):
            continue
        moved = np.array(points, dtype=float)
        moved += wobble(moved)
        member.points = adopt(moved, FrozenPoints)
        member.__dict__[BOILED] = True


# What is refused


@functools.cache
def _exact_content() -> tuple[type, ...]:
    """The classes whose drawing must change at every frame."""
    from ..mobject import marks
    from ..mobject.annotation.attach import Annotation, LeaderLine
    from ..mobject.diagram import Diagram, Flow, _StraightConnector
    from ..mobject.graphing.coordinate_systems import CoordinateSystem
    from ..mobject.graphing.functions import ImplicitFunction, ParametricFunction
    from ..mobject.graphing.number_line import NumberLine
    from ..mobject.text.code_mobject import Code
    from ..mobject.text.numbers import DecimalNumber, Variable
    from ..mobject.text.tex_mobject import MathTexPart, SingleStringMathTex
    from ..mobject.text.text_mobject import MarkupText, Paragraph, Text
    from ..mobject.text.typst_mobject import Typst

    return (
        Text,
        MarkupText,
        Paragraph,
        SingleStringMathTex,
        MathTexPart,
        Typst,
        Code,
        DecimalNumber,
        Variable,
        CoordinateSystem,
        NumberLine,
        ParametricFunction,
        ImplicitFunction,
        Annotation,
        LeaderLine,
        Diagram,
        Flow,
        _StraightConnector,
        *(getattr(marks, name) for name in marks.__all__),
    )


def _refuse(code: str, what: str, **detail: Any) -> None:
    from ..lint.raised import RaisedFinding

    raise RaisedFinding(code, what, detail=detail)


def _check_content(scene: Scene, root: Mobject, members: list[Mobject]) -> None:
    exact = _exact_content()
    frame = getattr(scene.camera, "frame", None)
    for member in members:
        if frame is not None and member is frame:
            _refuse(
                "timing.exact_content",
                f"the scene's camera's frame (a {type(member).__name__}) is in the"
                f" family declared with stepped timing on a {type(root).__name__}:"
                " the camera moves smoothly",
                object=type(member).__name__,
                camera=True,
            )
        if isinstance(member, exact):
            _refuse(
                "timing.exact_content",
                f"a {type(member).__name__} is in the family declared with stepped"
                f" timing on a {type(root).__name__}: what it shows changes at every"
                " frame",
                object=type(member).__name__,
                camera=False,
            )


def _check_updaters(root: Mobject, members: list[Mobject]) -> None:
    allowed = _engine_updaters()
    for member in members:
        for updater in member.updaters:
            if not any(updater is engine for engine in allowed):
                name = getattr(updater, "__qualname__", type(updater).__name__)
                _refuse(
                    "timing.updater",
                    f"the {type(member).__name__} in the family declared with stepped"
                    f" timing on a {type(root).__name__} has the scene's own updater"
                    f" {name}",
                    kind="mobject updater",
                    name=name,
                )


def _animation_nodes(animations: Any) -> list[Animation]:
    """Every animation of ``animations``, each group and each animation it
    holds (introducers included), in order.
    """
    from ..animation.animation import Animation

    found: list[Animation] = []
    for animation in animations:
        if not isinstance(animation, Animation):
            continue
        found.append(animation)
        found += _animation_nodes(getattr(animation, "animations", None) or ())
        inner = getattr(animation, "anim", None)
        if isinstance(inner, Animation):
            found += _animation_nodes((inner,))
    return found


def _touched(
    animation: Animation, declared: dict[int, Illustration]
) -> Illustration | None:
    """The illustration a member of whose family ``animation``'s own
    mobject holds.
    """
    mobject = getattr(animation, "mobject", None)
    if mobject is None:
        return None
    return next(
        (declared[id(m)] for m in mobject.get_family() if id(m) in declared), None
    )


def _touching(
    animation: Animation, declared: dict[int, Illustration]
) -> list[Illustration]:
    """Every illustration ``animation`` animates a member of, read through
    each animation it holds: a group's own mobject leaves its introducers
    out.
    """
    found: list[Illustration] = []
    for node in _animation_nodes((animation,)):
        mobject = getattr(node, "mobject", None)
        if mobject is None:
            continue
        for member in mobject.get_family():
            illustration = declared.get(id(member))
            if illustration is not None and all(illustration is not f for f in found):
                found.append(illustration)
    return found


def _holds(animation: Animation, root: Mobject) -> bool:
    """Whether ``animation`` suspends, and so its end resumes, the
    updating of a family that holds ``root``.
    """
    mobject = getattr(animation, "mobject", None)
    return (
        bool(animation.suspend_mobject_updating)
        and mobject is not None
        and any(m is root for m in mobject.get_family())
    )


def _restores(animation: Animation, root: Mobject) -> bool:
    """Whether finishing ``animation`` runs ``root``'s updaters (its
    restorer): it, or for a group any animation it holds (for a Succession
    its last part, which finishes last), resumes the updating of a family
    that holds ``root``.
    """
    from ..animation.composition import Succession

    parts = getattr(animation, "animations", None) or []
    if isinstance(animation, Succession):
        return bool(parts) and _restores(parts[-1], root)
    return _holds(animation, root) or any(_restores(p, root) for p in parts)


def _suspending(animations: Any) -> frozenset[int]:
    """The ids of every mobject the animations of a play may suspend: the
    family of each one's mobject when it suspends its updating, and of every
    other mobject it holds (a starting copy, a stand-in it suspends itself).
    """
    from ..mobject.mobject import Mobject

    found: set[int] = set()
    for node in _animation_nodes(animations):
        for value in vars(node).values():
            if not isinstance(value, Mobject):
                continue
            if value is node.mobject and not node.suspend_mobject_updating:
                continue
            found.update(id(member) for member in value.get_family())
    return frozenset(found)


def _check_running(
    root: Mobject, members: list[Mobject], suspending: frozenset[int]
) -> None:
    """Refuse a member whose updating the scene's code suspended: its
    restorer would not run when the play ends.
    """
    for member in members:
        if member.updating_suspended and id(member) not in suspending:
            kind = type(member).__name__
            _refuse(
                "timing.updater",
                f"the {kind} in the family declared with stepped timing on a"
                f" {type(root).__name__} has its updating suspended by the scene's"
                " code: the engine puts back what is true through it when each"
                " play ends",
                kind="suspended updating",
                name=kind,
            )


def _check_animations(animations: Any, declared: dict[int, Illustration]) -> None:
    from ..animation.composition import Succession
    from ..animation.movement import MoveAlongPath, PhaseFlow
    from ..animation.updaters.update import MaintainPositionRelativeTo, UpdateFromFunc
    from ..timeline.frozen import engine_class

    # Engine animations that move their mobject from where it stands, which
    # between two drawings is the drawing on display.
    relative = (MoveAlongPath, PhaseFlow, MaintainPositionRelativeTo)
    for animation in _animation_nodes(animations):
        if isinstance(animation, Succession):
            # Whatever its own mobject holds: it leaves its introducers out.
            _check_parts(animation, declared)
        touched = _touched(animation, declared)
        if touched is None:
            continue
        kind = type(animation).__name__
        root = type(touched.root).__name__
        if isinstance(animation, UpdateFromFunc) or not engine_class(type(animation)):
            why = (
                "runs the scene's own function"
                if isinstance(animation, UpdateFromFunc)
                else "is not one of the engine's"
            )
            _refuse(
                "timing.updater",
                f"the animation {kind} moves the family declared with stepped"
                f" timing on a {root}, and {why}: the engine can hold only"
                " animations it knows write whole values",
                kind="animation",
                name=kind,
            )
        if isinstance(animation, relative):
            _refuse(
                "timing.updater",
                f"the animation {kind} moves the family declared with stepped"
                f" timing on a {root} from where it stands, which between two"
                " drawings is the drawing on display: the engine can hold only"
                " animations that write whole values",
                kind="animation",
                name=kind,
            )
    _check_removals(animations, declared)


def _removals(
    animations: Any, in_succession: bool = False, removing: bool = False
) -> Iterator[tuple[Animation, bool, bool]]:
    """Each animation of ``animations`` and each it holds, with whether it
    is held by a Succession and whether a group holding it removes what it
    animates (a removing group makes its animations removers).
    """
    from ..animation.animation import Animation
    from ..animation.composition import AnimationGroup, Succession

    for animation in animations:
        if not isinstance(animation, Animation):
            continue
        yield animation, in_succession, removing
        inner = in_succession or isinstance(animation, Succession)
        parts = getattr(animation, "animations", None) or ()
        group_removes = isinstance(animation, AnimationGroup) and (
            removing or animation.is_remover()
        )
        yield from _removals(parts, inner, group_removes)
        held = getattr(animation, "anim", None)
        if isinstance(held, Animation):
            yield from _removals((held,), inner, False)


def _check_removals(animations: Any, declared: dict[int, Illustration]) -> None:
    """Refuse an animation that takes a declared family out of the scene as
    the play ends without its updating resumed then: it does not suspend
    it, or it finishes earlier, as a Succession's part. Its members'
    restorer would never run.
    """
    from ..animation.composition import AnimationGroup
    from ..animation.speedmodifier import ChangeSpeed
    from ..animation.transform import FadeTransform
    from ..animation.transform_matching_parts import TransformMatchingAbstractBase

    for animation, in_succession, removing in _removals(animations):
        always = isinstance(animation, (FadeTransform, TransformMatchingAbstractBase))
        if not always and isinstance(animation, (AnimationGroup, ChangeSpeed)):
            # It hands its end to the animations it holds, read on their own.
            continue
        removes = (
            always
            or removing
            or animation.is_remover()
            or bool(getattr(animation, "replace_mobject_with_target_in_scene", False))
        )
        if not removes or (animation.suspend_mobject_updating and not in_succession):
            continue
        touched = _touched(animation, declared)
        if touched is None:
            continue
        kind = type(animation).__name__
        why = (
            "finishes before the play ends, as a Succession's part"
            if in_succession
            else "does not suspend its updating"
        )
        _refuse(
            "timing.updater",
            f"the animation {kind} takes the family declared with stepped timing"
            f" on a {type(touched.root).__name__} out of the scene as the play"
            f" ends, and {why}: nothing would put back what is true there",
            kind="animation",
            name=kind,
        )


def _check_parts(succession: Succession, declared: dict[int, Illustration]) -> None:
    """Refuse a part of ``succession`` that begins in the middle of the
    play and animates a declared family the part before it does not
    restore: it would begin from the drawing on display.
    """
    parts = list(succession.animations)
    for before, part in zip(parts, parts[1:], strict=False):
        for touched in _touching(part, declared):
            if _restores(before, touched.root):
                continue
            kind = type(succession).__name__
            _refuse(
                "timing.updater",
                f"the {kind}'s part {type(part).__name__} animates the family"
                f" declared with stepped timing on a {type(touched.root).__name__}"
                f" and begins in the middle of the play, after a"
                f" {type(before).__name__} that does not animate that family: it"
                " would begin from the drawing on display",
                kind="animation",
                name=kind,
            )


# The clock


class TimingClock:
    """A scene's clock: the engine callback that holds the scene's
    illustrations to their timing, last among the scene's updaters (see the
    module's docs). A scene has at most one, made by its first declaration
    and found with :func:`clock_of`. The scene owns it, through the scene
    updater it runs as; it refers to the scene weakly.
    """

    def __init__(self, scene: Scene) -> None:
        self._scene = weakref.ref(scene)
        self.illustrations: list[Illustration] = []
        #: The scene drivers it runs, in the order the scene registered them.
        self.drivers: list[SceneDriver] = []
        self._animations: Any = None
        #: Every mark declared here, by the string itself: a mobject carrying
        #: one of them is a copy of a mobject declared in this scene (a deep
        #: copy of a string is the string).
        self._marks: list[str] = []
        #: The members it gave the restorer at its last instant, by id.
        self._armed: dict[int, Mobject] = {}

        def hold_illustrations(dt: float) -> None:
            if dt >= 0:  # the read a stepping tick looks for; never a time
                self()

        #: The scene updater the clock runs as.
        self.updater: Callable[[float], None] = hold_illustrations

    def take_over(self, scene: Scene) -> None:
        """Take the scene's drivers and its pose settle out of its updaters,
        to run them itself.
        """
        from .motion import pose_settle

        found = _found_drivers(scene)
        updaters = list(scene.updaters)

        def place(driver: SceneDriver) -> int:
            # A driver's updater is one of the scene's as drive() finds it:
            # by equality (a bound method is a new object at each lookup).
            return next(
                (i for i, f in enumerate(updaters) if f == driver.updater),
                len(updaters),
            )

        found.sort(key=place)
        for driver in found:
            self.adopt(driver)
        settle = pose_settle(scene)
        if settle is not None:
            settle.registers = False
            if settle.updater in scene.updaters:
                scene.remove_updater(settle.updater)

    def adopt(self, driver: SceneDriver) -> None:
        """Run ``driver`` from now on, at the start of each instant."""
        if any(driver is d for d in self.drivers):
            return
        scene = self._scene()
        if scene is not None:
            # The scene removes an updater by identity: hand it the one it
            # holds.
            for held in [f for f in scene.updaters if f == driver.updater]:
                scene.remove_updater(held)
        self.drivers.append(driver)

    def arms(self, mobject: Mobject) -> bool:
        """Whether ``mobject`` holds the restorer this clock gave it."""
        return self._armed.get(id(mobject)) is mobject

    def end_play(self) -> None:
        """A play's end: have every family in the scene hold what is true,
        and take the restorer out of every member it was given to.
        """
        for illustration in self.illustrations:
            if illustration.present:
                illustration.restore()
        self.release()

    def release(self) -> None:
        """Take the restorer out of every member it was given to."""
        armed, self._armed = self._armed, {}
        for member in armed.values():
            _release(member)

    def illustration_of(self, root: Mobject) -> Illustration | None:
        """The illustration declared here on ``root``, or ``None``."""
        return next((i for i in self.illustrations if i.root is root), None)

    def declared(self) -> dict[int, Illustration]:
        """Each member of every declared family, by id, with its
        illustration; families that have come to share a member are refused.
        """
        found: dict[int, Illustration] = {}
        for illustration in self.illustrations:
            for member in illustration.root.get_family():
                other = found.setdefault(id(member), illustration)
                if other is not illustration:
                    raise _shared(illustration.root, other.root)
        return found

    def check(self, scene: Scene, root: Mobject) -> None:
        """Refuse declaring ``root``: a family that shares members with
        another declared, holds exact content or a scene's updater.
        """
        members = root.get_family()
        mine = {id(m) for m in members}
        for illustration in self.illustrations:
            if illustration.root is root:
                continue
            if any(id(m) in mine for m in illustration.root.get_family()):
                raise _shared(root, illustration.root)
        _check_content(scene, root, members)
        _check_updaters(root, members)

    def declare(self, scene: Scene, root: Mobject, timing: Timing) -> None:
        """Hold ``root`` and its family to ``timing`` from the scene's
        instant now.
        """
        self.check(scene, root)
        now = scene.time
        mark = timing.mark
        root.__dict__[MARK] = mark
        _declare_numbers(root)
        self._marks.append(mark)
        illustration = self.illustration_of(root)
        if illustration is not None:
            illustration.restore()
            illustration.timing = timing
            illustration.step = timing.step(now)
            illustration.shown = illustration.true
            return
        self.illustrations.append(Illustration(root, timing, now))

    def __call__(self) -> None:
        """The clock's work at the scene's instant now."""
        scene = self._scene()
        if scene is None:
            return
        try:
            self._hold(scene)
        except BaseException:
            # A refused scene ends here: its mobjects keep no restorer.
            self.release()
            raise

    def _hold(self, scene: Scene) -> None:
        self._last_in(scene)
        for driver in self.drivers:
            driver()
        from .motion import pose_settle

        settle = pose_settle(scene)
        posed: frozenset[int] = frozenset()
        if settle is not None:
            settle()
            posed = frozenset(map(id, settle.rigs))
        family = [m for top in scene.mobjects for m in top.get_family()]
        self._adopt_copies(scene, family)
        declared = self.declared()
        first = scene.animations is not self._animations
        if first:
            self._animations = scene.animations
            for illustration in self.illustrations:
                _check_content(scene, illustration.root, illustration.root.get_family())
            _check_animations(scene.animations or (), declared)
        present = {id(m) for m in family}
        # Read at every instant: an animation begun since, or one that grew
        # its mobject's family as it began, suspends what it holds now.
        suspending = _suspending(scene.animations or ())
        for illustration in self.illustrations:
            members = illustration.root.get_family()
            _check_updaters(illustration.root, members)
            if id(illustration.root) in present:
                _check_running(illustration.root, members, suspending)
        now = scene.time
        armed: dict[int, Mobject] = {}
        for illustration in self.illustrations:
            for member in illustration.at(now, id(illustration.root) in present, posed):
                armed[id(member)] = member
        # A member that left its family or the scene since the last instant
        # gives the restorer back now: nothing may run it when the play ends.
        for key, member in self._armed.items():
            if armed.get(key) is not member:
                _release(member)
        self._armed = armed
        _reattach(family, declared)

    def _last_in(self, scene: Scene) -> None:
        """Refuse a scene updater registered after the clock; a scene
        driver's is taken over instead.
        """
        updaters = scene.updaters
        index = next((i for i, f in enumerate(updaters) if f is self.updater), None)
        if index is None or index == len(updaters) - 1:
            return
        drivers = _found_drivers(scene)
        for late in list(updaters[index + 1 :]):
            driver = next((d for d in drivers if d.updater == late), None)
            if driver is not None:
                self.adopt(driver)
                continue
            name = getattr(late, "__qualname__", type(late).__name__)
            _refuse(
                "timing.updater",
                f"the scene updater {name} was registered after the scene declared"
                " stepped timing, whose clock must run last",
                kind="scene updater",
                name=name,
            )

    def _adopt_copies(self, scene: Scene, family: list[Mobject]) -> None:
        """Declare each mobject in the scene that carries a timing mark it
        was not declared with here: a copy of a declared mobject.
        """
        declared: dict[int, Illustration] | None = None
        for member in family:
            mark = member.__dict__.get(MARK)
            timing = _timing(mark)
            if (
                timing is None
                or not any(mark is ours for ours in self._marks)
                or self.illustration_of(member) is not None
            ):
                continue
            if declared is None:
                declared = self.declared()
            if id(member) in declared:
                continue
            self.declare(scene, member, timing)
            declared = self.declared()


def _reattach(family: list[Mobject], declared: dict[int, Illustration]) -> None:
    """Put each running attachment to a member of a declared family at its
    anchor again, now that the family shows its drawing on display.
    """
    from ..mobject.annotation.attach import Annotation, keep_attached

    for mobject in family:
        if not isinstance(mobject, Annotation) or mobject.updating_suspended:
            continue
        attached = mobject.attached
        if attached is None or id(attached.anchor.mobject) not in declared:
            continue
        if any(u is keep_attached for u in mobject.updaters):
            mobject.attach()


def _shared(root: Mobject, other: Mobject) -> ValueError:
    return ValueError(
        f"The {type(root).__name__} declared with stepped timing shares its"
        f" family with the {type(other).__name__} declared before: declare one"
        " root over the whole illustration."
    )


#: Each scene's clock, by the scene, both held weakly: the scene owns its
#: clock, through the scene updater it runs as.
_CLOCKS: weakref.WeakKeyDictionary[Scene, weakref.ref[TimingClock]] = (
    weakref.WeakKeyDictionary()
)


def clock_of(scene: Scene) -> TimingClock | None:
    """``scene``'s clock, or ``None`` before its first declaration."""
    held = _CLOCKS.get(scene)
    return held() if held is not None else None


def _running_scene() -> Scene | None:
    from ..timeline.recorder import running

    recorder = running()
    return recorder.scene if recorder is not None else None


def set_timing(
    root: Mobject, drawings_per_second: int, boil: Boil | None = None
) -> None:
    """Declare ``root`` and its whole family one illustration, drawn
    ``drawings_per_second`` times a second of scene time (a divisor of 60),
    with ``boil``: whatever animates it, or any part of it, its drawing on
    display changes only at the boundaries ``k / drawings_per_second``, the
    same at every frame rate. It needs the scene whose script is running,
    and holds there from the instant it is called; declaring a mobject again
    gives it the new timing from then.
    """
    from ..mobject.mobject import Mobject

    if not isinstance(root, Mobject):
        raise TypeError(f"set_timing declares a mobject, not a {type(root).__name__}")
    timing = Timing(drawings_per_second, boil)
    scene = _running_scene()
    if scene is None:
        raise RuntimeError(
            "set_timing declares timing in a running scene: call it in the scene's"
            " construct()."
        )
    clock = clock_of(scene)
    if clock is None:
        clock = TimingClock(scene)
        # Checked first, so a refused declaration leaves the scene as it was.
        clock.check(scene, root)
        clock.take_over(scene)
        _CLOCKS[scene] = weakref.ref(clock)
        scene.add_updater(clock.updater)
    clock.declare(scene, root, timing)
