"""Restoring a state-function play (route 2) at an instant its record did
not sample.

A state play is evaluated forward only where its consumer reads it (a
render's frames) and at its events. At any other instant a restore
evaluates it again, after the scene has run: a non-frame evaluation, as the
forward run would have made had it evaluated the play there.

1. The play's latest kept capture at or before ``t`` is installed: every
   mobject it reaches, the camera's inputs, the registered state the play's
   admitted callbacks read and write beyond what the capture reaches, the
   animations' own state there, and the scene's lists and clock. The latest
   capture rather than the start: it keeps whatever the scene's frame hooks
   wrote at earlier frames, and for callbacks that keep no history the two
   are equal.
2. The names the callbacks read (closure cells and the scene's attributes)
   are given the values they held when the play began.
3. The clock is set to ``t`` and the engine's bodies run once, in CE's
   order (:func:`.routes.evaluate_instant`): never the scene's frame hooks,
   which run only where a frame reads the instant. The camera's admitted
   callbacks (fixed-orientation centre functions) are then evaluated at
   ``t``, as the forward run evaluates them after every instant.
4. The live scene, its clock and the names are put back; the mobjects and
   the camera are put back when the drawing that holds them ends. What the
   callbacks build at ``t`` is tagged with its creating line as a recording
   tags it, numbered on from the forward run's last object of that line and
   alike in every restore, after every id the record holds: a node built
   at an instant has no id the forward run gave it.

What the record holds for a restore is the registered-state contract
(:data:`REGISTERED_STATE_VERSION`): the capture's mobject tree with every
node's values (its scalar and cursor fields included: a tracker's value,
``DecimalNumber.number``, ``TracedPath``'s time, suspension flags), the
camera's inputs, and at each kept sample a :class:`RestorePoint`.

A restore is sound only for what it can put back. A global or a default
argument a callback read that holds another value now, or a container it
read whose items changed, makes the play :class:`Unrestorable` at that
instant: those are not the scene's state, and assigning them would reach
code outside the play.
"""

from __future__ import annotations

__all__ = [
    "REGISTERED_STATE_VERSION",
    "RestorePoint",
    "Unrestorable",
    "restore_point",
    "restored",
    "restorer",
]

import os
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Any

import numpy as np

from .draw import CameraExtras, Drawing
from .events import EventPlan
from .origins import ORIGINS
from .record import NodeState
from .routes import Unavailable, _copied, evaluate_instant, snapshot_animations

if TYPE_CHECKING:
    from arachne.mobject.mobject import Mobject
    from arachne.scene.scene import Scene

    from .purity import Binding
    from .record import Capture, PlayRecord, Record

#: The version of what a record keeps for a restore: the capture, the
#: camera's inputs and a :class:`RestorePoint` at every kept sample of a
#: state play. A restore refuses a point of another version.
REGISTERED_STATE_VERSION = 1

_MISSING = object()


class Unrestorable(Unavailable):
    """A state play can't be restored at the instant asked. ``reason`` is
    the stable code: ``restore.global_rebound``,
    ``restore.default_rebound``, ``restore.binding_mutated``,
    ``restore.rebound_in_play`` or ``restore.missing_state``.
    """

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason


@dataclass(frozen=True)
class RestorePoint:
    """What a kept sample of a state play holds beyond its capture.

    ``animations`` are the play's animations with their state there
    (:func:`.routes.snapshot_animations`). ``nodes`` are the states of every
    mobject the admitted callbacks read or write, families included, which
    the capture need not reach (a tracker never added to the scene).
    ``centers`` are, per camera, its fixed-orientation mobjects with their
    centre functions, which the capture holds only as values. ``rebound``
    names the bindings that held another value there than when the play
    began (a frame hook assigned them), from which no restore is sound.
    """

    animations: tuple[tuple[Any, dict[str, Any]], ...]
    nodes: tuple[tuple[Mobject, NodeState], ...]
    centers: tuple[tuple[Any, tuple[tuple[Mobject, Callable[[], Any]], ...]], ...]
    rebound: tuple[str, ...] = ()
    version: int = REGISTERED_STATE_VERSION


def restore_point(
    scene: Scene, play: PlayRecord, previous: RestorePoint | None
) -> RestorePoint:
    """The :class:`RestorePoint` of ``play`` now, sharing what is unchanged
    with ``previous``.
    """
    earlier = {id(mob): state for mob, state in previous.nodes} if previous else {}
    nodes = []
    for mob in _registered_mobjects(play):
        nodes.append((mob, NodeState.of(mob, earlier.get(id(mob)))))
    found = tuple(nodes)
    if previous is not None and len(found) == len(previous.nodes):
        if all(
            a is c and b is d
            for (a, b), (c, d) in zip(found, previous.nodes, strict=True)
        ):
            found = previous.nodes
    centers = _centre_functions(scene.renderer.camera)
    if previous is not None and _same_centers(centers, previous.centers):
        centers = previous.centers
    rebound = tuple(
        binding.name
        for binding in _bindings(play)
        if binding.where in ("cell", "attribute")
        and _current(scene, binding) is not binding.value
    )
    return RestorePoint(
        snapshot_animations(play._animations), found, centers, rebound
    )


def restorer(
    record: Record, play: PlayRecord, t: Fraction
) -> tuple[Capture, Callable[[Drawing], None]]:
    """The capture a restore of ``play`` at ``t`` starts from, and what
    brings it to ``t`` once a drawing shows it (a ``prepare`` of
    :meth:`.Drawing.sample`). Raises :class:`Unrestorable` when it can't be.
    """
    base = play.base_at(t)
    point = play._animation_states.get(base.seq)
    if not isinstance(point, RestorePoint) or point.version != REGISTERED_STATE_VERSION:
        raise Unrestorable(
            "restore.missing_state",
            f"the record kept no restore point for play {play.index} at"
            f" {base.instant}",
        )
    if point.rebound:
        raise Unrestorable(
            "restore.rebound_in_play",
            f"{', '.join(point.rebound)} held another value at {base.instant} than"
            f" when play {play.index} began",
        )
    scene = record._scene
    assignments = _assignments(scene, play)

    def prepare(drawing: Drawing) -> None:
        drawing.show(base, [*base.roots, *base.foreground, *base.moving])
        saved = drawing._saved
        for mob, state in point.nodes:
            if id(mob) not in saved:
                saved[id(mob)] = (mob, NodeState.of(mob))
            state.install(mob)
        with (
            _assigned(scene, assignments),
            _animations_at(point),
            _scene_at(scene, play, base, t),
            ORIGINS.resumed(_base_dir(record), _numbering(record)),
        ):
            plan = _event_plan(point, t)
            camera = scene.renderer.camera
            _install_centre_functions(point.centers)
            evaluate_instant(scene, plan, play.start, t, float(t - base.instant))
            extras = CameraExtras.observe(camera, base.camera)
        if extras is not None:
            # The centres as values, as a capture holds them.
            extras.install(camera)

    return base, prepare


@contextmanager
def restored(record: Record, t: Fraction | float) -> Iterator[Scene]:
    """The scene inside the block holding the state of the state play that
    holds ``t``, restored there; everything is put back after it.
    """
    from .clock import exact

    t = exact(t)
    play = record.play_at(t)
    if play is None or play.route != "state" or play.held:
        raise LookupError(f"No state-function play holds {t}.")
    _, prepare = restorer(record, play, t)
    with Drawing(record) as drawing:
        prepare(drawing)
        yield record._scene


def _base_dir(record: Record) -> str | None:
    """The directory the record's source lines are relative to."""
    if record.scene_file is None:
        return None
    return os.path.realpath(os.path.dirname(record.scene_file))


def _numbering(record: Record) -> dict[str, int]:
    """Per creating line, the ordinal after the last the record's mobjects
    hold: what a restore numbers the objects it builds from, so that no id
    of the record is given again.
    """
    found = _NUMBERING.get(record)
    if found is None:
        found = {}
        for mob in record.mobjects():
            where, _, ordinal = record.origin(mob).id.rpartition("#")
            if ordinal.isdigit():
                found[where] = max(found.get(where, 0), int(ordinal) + 1)
        _NUMBERING[record] = found
    return found


_NUMBERING: weakref.WeakKeyDictionary[Record, dict[str, int]] = (
    weakref.WeakKeyDictionary()
)


# The registered state


def _bindings(play: PlayRecord) -> Iterator[Binding]:
    """The names the play's admitted callbacks read, each once."""
    seen: set[tuple[str, int, str]] = set()
    admission = play.admission
    for callback in admission.callbacks if admission is not None else ():
        if callback.proof is None:
            continue
        for binding in callback.proof.bindings:
            key = (binding.where, id(binding.holder), binding.name)
            if key not in seen:
                seen.add(key)
                yield binding


def _registered_mobjects(play: PlayRecord) -> Iterator[Mobject]:
    """Every mobject the play's admitted callbacks read or write, with their
    families, each once.
    """
    seen: set[int] = set()
    admission = play.admission
    for callback in admission.callbacks if admission is not None else ():
        if callback.proof is None:
            continue
        for mob in (*callback.proof.reads, *callback.proof.writes):
            for member in mob.get_family():
                if id(member) not in seen:
                    seen.add(id(member))
                    yield member


def _centre_functions(
    camera: Any,
) -> tuple[tuple[Any, tuple[tuple[Mobject, Callable[[], Any]], ...]], ...]:
    """Each camera's fixed-orientation mobjects with their centre functions:
    the scene's camera, and its displays' cameras.
    """
    cameras = [camera]
    for display in vars(camera).get("image_mobjects_from_cameras", ()):
        sub_camera = getattr(display, "camera", None)
        if sub_camera is not None and sub_camera is not camera:
            cameras.append(sub_camera)
    return tuple(
        (each, tuple(orientations.items()))
        for each in cameras
        if (orientations := getattr(each, "fixed_orientation_mobjects", None))
    )


def _same_centers(a: Any, b: Any) -> bool:
    return len(a) == len(b) and all(
        x is u
        and len(y) == len(v)
        and all(m is n and f is g for (m, f), (n, g) in zip(y, v, strict=True))
        for (x, y), (u, v) in zip(a, b, strict=True)
    )


def _install_centre_functions(
    centers: tuple[tuple[Any, tuple[tuple[Mobject, Callable[[], Any]], ...]], ...],
) -> None:
    """Give each camera its centre functions back, for the camera's inputs
    to be evaluated at the restored instant. (The drawing showing the
    capture saved what the cameras held.)
    """
    for camera, orientations in centers:
        camera.fixed_orientation_mobjects = dict(orientations)


# Names


def _current(scene: Scene, binding: Binding) -> Any:
    """What the name ``binding`` read holds now, or ``_MISSING``."""
    where, holder = binding.where, binding.holder
    if where == "cell":
        try:
            return holder.cell_contents
        except ValueError:
            return _MISSING
    if where == "global":
        return holder.get(binding.name, _MISSING)
    if where == "attribute":
        owner, name = _attribute_owner(scene, binding)
        return vars(owner).get(name, _MISSING)
    if where == "default":
        return _default(holder, binding.name)
    return _MISSING


def _attribute_owner(scene: Scene, binding: Binding) -> tuple[Any, str]:
    """The object holding an attribute binding and the attribute's name:
    ``self.x`` on the scene, ``camera.x`` on its camera.
    """
    prefix, _, name = binding.name.partition(".")
    if binding.holder is not None:
        return binding.holder, name
    if prefix == "camera":
        return scene.renderer.camera, name
    return scene, name


def _default(function: Any, name: str) -> Any:
    code = function.__code__
    positional = code.co_varnames[: code.co_argcount]
    defaults = function.__defaults__ or ()
    if name in positional:
        index = positional.index(name) - (len(positional) - len(defaults))
        return defaults[index] if index >= 0 else _MISSING
    return (function.__kwdefaults__ or {}).get(name, _MISSING)


def _assignments(scene: Scene, play: PlayRecord) -> list[tuple[Binding, Any]]:
    """The cells and attributes a restore of ``play`` gives back the value
    they held when it began, each with what it holds now. Raises
    :class:`Unrestorable` for a global or default argument rebound since,
    and for a value whose items changed since.
    """
    found = []
    for binding in _bindings(play):
        now = _current(scene, binding)
        if now is not binding.value:
            if binding.where == "global":
                raise Unrestorable(
                    "restore.global_rebound",
                    f"the global {binding.name} a callback of play {play.index}"
                    " read holds another value now",
                )
            if binding.where == "default":
                raise Unrestorable(
                    "restore.default_rebound",
                    f"the default argument {binding.name} a callback of play"
                    f" {play.index} read holds another value now",
                )
            found.append((binding, now))
        if not _same_contents(_contents_of(binding.value), binding.contents):
            raise Unrestorable(
                "restore.binding_mutated",
                f"what {binding.name} holds changed in place since play"
                f" {play.index} began",
            )
    return found


def _contents_of(value: Any) -> Any:
    from .purity import _contents

    return _contents(value)


def _same_contents(a: Any, b: Any) -> bool:
    """Whether two snapshots of a binding's contents hold the same items."""
    if a is b:
        return True
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (
            isinstance(a, np.ndarray)
            and isinstance(b, np.ndarray)
            and a.dtype == b.dtype
            and a.shape == b.shape
            and bool(np.array_equal(a, b))
        )
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(
            _same_contents(x, y) for x, y in zip(a, b, strict=True)
        )
    return False


@contextmanager
def _assigned(scene: Scene, assignments: list[tuple[Binding, Any]]) -> Iterator[None]:
    """The names given the values they held when the play began, and put
    back after.
    """
    done: list[tuple[Binding, Any]] = []
    try:
        for binding, now in assignments:
            _assign(scene, binding, binding.value)
            done.append((binding, now))
        yield
    finally:
        for binding, now in reversed(done):
            _assign(scene, binding, now)


def _assign(scene: Scene, binding: Binding, value: Any) -> None:
    """Make the cell or attribute ``binding`` read hold ``value`` (unbound
    for ``_MISSING``).
    """
    if binding.where == "cell":
        if value is _MISSING:
            del binding.holder.cell_contents
        else:
            binding.holder.cell_contents = value
        return
    owner, name = _attribute_owner(scene, binding)
    if value is _MISSING:
        vars(owner).pop(name, None)
    else:
        vars(owner)[name] = value


# The play's state


@contextmanager
def _animations_at(point: RestorePoint) -> Iterator[None]:
    """Every animation holding its state at the point, and put back after."""
    saved = [(animation, dict(vars(animation))) for animation, _ in point.animations]
    try:
        for animation, state in point.animations:
            vars(animation).clear()
            vars(animation).update(_copied(state))
        yield
    finally:
        for animation, state in saved:
            vars(animation).clear()
            vars(animation).update(state)


def _event_plan(point: RestorePoint, t: Fraction) -> EventPlan:
    """The play's events from the point on: none falls between the point and
    ``t`` (every event instant is kept), so none runs.
    """
    from arachne.animation.composition import Succession

    successions = [a for a, _ in point.animations if isinstance(a, Succession)]
    plan = EventPlan(successions=successions, now=t)
    for succession in successions:
        succession._event_plan = plan
    return plan


@contextmanager
def _scene_at(
    scene: Scene, play: PlayRecord, base: Capture, t: Fraction
) -> Iterator[None]:
    """The scene's lists as they were at ``base``, its play's animations
    running, and its clock at ``t``; put back after.
    """
    renderer = scene.renderer
    saved = (
        scene.mobjects,
        scene.foreground_mobjects,
        scene.moving_mobjects,
        scene.animations,
        scene.updaters,
        renderer.clock,
    )
    try:
        scene.mobjects = list(base.roots)
        scene.foreground_mobjects = list(base.foreground)
        scene.moving_mobjects = list(base.moving)
        scene.animations = list(play._animations)
        scene.updaters = list(base.scene_updaters)
        renderer.time = t
        yield
    finally:
        (
            scene.mobjects,
            scene.foreground_mobjects,
            scene.moving_mobjects,
            scene.animations,
            scene.updaters,
            renderer.time,
        ) = saved
