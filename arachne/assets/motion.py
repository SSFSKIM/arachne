"""Animating rigs: :class:`Pose` moves a rig through its parameters, and
:class:`MorphParts` turns one rig into another part by part.

A :class:`Pose` writes values only. The art follows from the scene's
:class:`PoseSettle`, an engine callback among the scene's updaters while a
play poses a rig: the timeline runs it after every animation and every
mobject's updater at every evaluated instant, and no animation suspends it.
So a whole-rig animation and a pose in one play compose in either order:
the mover writes the carrier (and art posed for its start), the pose writes
the values, and the settle that follows draws the pose in the moved place.
After the play the rig's own updater keeps it settled (:mod:`.rig`).
"""

from __future__ import annotations

__all__ = [
    "MorphParts",
    "Pose",
    "PoseSettle",
    "hold_pose",
    "pose_settle",
    "release_pose",
]

import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from ..animation.animation import Animation
from ..animation.fading import FadeIn, FadeOut
from ..animation.transform import Transform
from ..mobject.types.vectorized_mobject import VMobject
from ..utils.rate_functions import linear
from .manifest import PartSpec
from .rig import Rig, _values

if TYPE_CHECKING:
    from ..mobject.mobject import Mobject
    from ..scene.scene import Scene

#: How close to an end of its run a pose's progress is taken to be there,
#: so that a play's last values are the targets themselves.
_END = 1e-12


class PoseSettle:
    """What keeps the rigs a scene's current play poses settled: each held
    rig settled at every instant the scene's updaters run, after the
    animations and the mobjects' updaters.

    A scene has at most one, made with its first :class:`Pose` and found
    with :func:`pose_settle`. While it holds a rig it runs as the scene
    updater :attr:`updater`, a plain function whose code never reads
    ``dt``. A rig joins it through :func:`hold_pose` when a pose of it
    begins and leaves through :func:`release_pose` when that pose is cleaned
    up. Something else that is to settle the posed rigs itself, at its own
    place in the instant (a clock that runs last), sets :attr:`registers`
    to false, takes :attr:`updater` out of the scene's updaters if it is
    there, and calls the settle itself.

    It is the scene's engine state, kept beside the scene, and no record
    reads it; it refers to the scene and to its rigs weakly.
    """

    def __init__(self, scene: Scene) -> None:
        self._scene = weakref.ref(scene)
        #: The rigs held, by id, with how many poses hold each.
        self._held: dict[int, tuple[weakref.ref[Rig], int]] = {}
        #: Whether the settle adds itself to the scene's updaters while it
        #: holds a rig.
        self.registers = True

        def settle_posed(dt: float) -> None:
            self()

        #: The scene updater it runs as.
        self.updater: Callable[[float], None] = settle_posed

    @property
    def rigs(self) -> tuple[Rig, ...]:
        """The rigs held that still exist, in the order they joined."""
        return tuple(r for ref, _ in self._held.values() if (r := ref()) is not None)

    def hold(self, rig: Rig) -> None:
        """Settle ``rig`` at every instant until it is released as often."""
        ref, count = self._held.get(id(rig), (weakref.ref(rig), 0))
        self._held[id(rig)] = (ref, count + 1)

    def release(self, rig: Rig) -> None:
        """Let go of one hold on ``rig``."""
        held = self._held.get(id(rig))
        if held is None:
            return
        ref, count = held
        if count > 1:
            self._held[id(rig)] = (ref, count - 1)
        else:
            del self._held[id(rig)]

    def __call__(self) -> None:
        """Settle every rig held."""
        for rig in self.rigs:
            rig.settle()


#: Each scene's pose settle, by the scene; gone with the scene.
_SETTLES: weakref.WeakKeyDictionary[Scene, PoseSettle] = weakref.WeakKeyDictionary()


def pose_settle(scene: Scene) -> PoseSettle | None:
    """``scene``'s pose settle, or ``None`` before its first pose."""
    return _SETTLES.get(scene)


def hold_pose(scene: Scene, rig: Rig) -> None:
    """Have ``rig`` settled at every instant of ``scene`` from now until
    :func:`release_pose`: by the scene's pose settle, added to the scene's
    updaters while it holds a rig, or run by the scene's timing clock when
    it has one (:mod:`.timing`).
    """
    settle = _SETTLES.get(scene)
    if settle is None:
        settle = _SETTLES[scene] = PoseSettle(scene)
        from .timing import clock_of

        # A scene that declared stepped timing settles its posed rigs from
        # its clock, which runs last.
        settle.registers = clock_of(scene) is None
    if settle.registers and settle.updater not in scene.updaters:
        scene.add_updater(settle.updater)
    settle.hold(rig)


def release_pose(scene: Scene, rig: Rig) -> None:
    """Undo one :func:`hold_pose` of ``rig``; the scene's pose settle leaves
    its updaters once it holds no rig.
    """
    settle = _SETTLES.get(scene)
    if settle is None:
        return
    settle.release(rig)
    if not settle.rigs and settle.updater in scene.updaters:
        scene.remove_updater(settle.updater)


def _rig(mobject: Any, role: str) -> Rig:
    if not isinstance(mobject, Rig):
        raise TypeError(f"{role} is a Rig, not a {type(mobject).__name__}")
    return mobject


def _no_placement(rig: Rig) -> ValueError:
    return ValueError(
        f"A rig of the asset {rig.asset.name} scaled to nothing has no place to be"
        " posed in: give it a size first."
    )


class Pose(Animation):
    """Move ``rig`` to the parameter values and slot drawings ``targets``
    (names to values, and slot names to drawing names) over the play.

    Each parameter named runs from its value when the play begins to its
    target, through ``rate_func``; the others keep theirs, and a pose writes
    no other: two poses of one rig in one play each move their own. A slot
    named switches at the play's first instant, in the rig and in the
    copies of it the play's animations already hold, so a whole-rig
    animation in the same play pairs each shape with its own copy. A value
    outside its parameter's tested range is ``rig.range``, and a name the
    manifest does not have ``rig.name_unknown``, both when the animation is
    made, at the line that asks; a rate function that carries a value out
    of the range is ``rig.range`` at that instant of the play.
    """

    def __init__(
        self,
        rig: Rig,
        /,
        *,
        run_time: float | None = None,
        rate_func: Callable[[float], float] | None = None,
        name: str | None = None,
        **targets: Any,
    ) -> None:
        rig = _rig(rig, "What Pose poses")
        slots = {slot.name for slot in rig.asset.manifest.slots}
        drawings = {key: value for key, value in targets.items() if key in slots}
        values = {key: value for key, value in targets.items() if key not in slots}
        known = [*rig.model.index, *slots]
        for key in values:
            if key not in rig.model.index:
                rig._unknown("parameter", key, known)
        #: The targets, by parameter index.
        self.targets = rig.checked_values(values)
        #: The drawings switched in, by slot index.
        self.switches = rig.checked_drawings(drawings)
        if rig.placement() is None:
            raise _no_placement(rig)
        given: dict[str, Any] = {"name": name}
        if run_time is not None:
            given["run_time"] = run_time
        if rate_func is not None:
            given["rate_func"] = rate_func
        super().__init__(rig, **given)
        self.rig = rig
        self._scene: weakref.ref[Scene] | None = None

    def create_starting_mobject(self) -> Mobject:
        # The values it starts from are all it reads of the rig's start.
        return self.mobject

    def _setup_scene(self, scene: Scene) -> None:
        super()._setup_scene(scene)
        if scene is not None:
            self._scene = weakref.ref(scene)
            hold_pose(scene, self.rig)

    def begin(self) -> None:
        rig = self.rig
        if rig.placement() is None:
            raise _no_placement(rig)
        if self.switches:
            rig.write_choices(self.switches)
            self._switch_copies()
        indices = sorted(self.targets)
        #: The parameters it moves, by index, and where each starts and ends.
        self._indices = np.array(indices, dtype=int)
        self._start = np.array([rig._values[i] for i in indices], dtype=float)
        self._end = np.array([self.targets[i] for i in indices], dtype=float)
        super().begin()
        # The play's start state shows the drawings switched in.
        rig.settle()

    def _switch_copies(self) -> None:
        """Have the copies of the rig that the play's animations begun
        before this one hold (an animation's starting copy, a transform's
        aligned target) show what the rig shows now: each stands in the
        animation's family where the rig stands in its mobject's.
        """
        scene = self._scene() if self._scene is not None else None
        if scene is None:
            return
        rig = self.rig
        for animation in _flattened(getattr(scene, "animations", None) or ()):
            if isinstance(animation, Pose):
                continue
            family = animation.mobject.get_family()
            index = next((i for i, m in enumerate(family) if m is rig), None)
            if index is None:
                continue
            for held in (
                getattr(animation, "starting_mobject", None),
                getattr(animation, "target_copy", None),
            ):
                if held is None or held is animation.mobject:
                    continue
                members = held.get_family()
                if index < len(members):
                    copy = members[index]
                    if isinstance(copy, Rig) and copy.model is rig.model:
                        copy.adopt_choices(rig)

    def interpolate_mobject(self, alpha: float) -> None:
        if not len(self._indices):
            return
        progress = self.get_sub_alpha(alpha, 0, 1)
        if abs(progress - 1) < _END:
            progress = 1.0
        elif abs(progress) < _END:
            progress = 0.0
        blended = (1 - progress) * self._start + progress * self._end
        rig = self.rig
        parameters = rig.model.parameters
        lows = np.array([parameters[i].min for i in self._indices])
        highs = np.array([parameters[i].max for i in self._indices])
        if 0 <= progress <= 1:
            # Between two values in range: in range, but for rounding.
            blended = np.clip(blended, lows, highs)
        else:
            rig.checked_values(
                {
                    parameters[i].name: value
                    for i, value in zip(self._indices, blended, strict=True)
                },
                eased=True,
            )
        values = np.array(rig._values, dtype=float)
        values[self._indices] = blended
        rig._values = _values(values)

    def clean_up_from_scene(self, scene: Scene) -> None:
        super().clean_up_from_scene(scene)
        release_pose(scene, self.rig)


def _flattened(animations: Any) -> list[Animation]:
    """``animations`` and the animations of every group among them, in the
    order they begin.
    """
    found: list[Animation] = []
    for animation in animations:
        found.append(animation)
        found += _flattened(getattr(animation, "animations", None) or ())
    return found


class MorphParts(Animation):
    """Turn the rig ``source`` into the rig ``destination`` part by part.

    Each part or slot the two share by name is transformed from its drawing
    in ``source`` into its drawing in ``destination``; those only ``source``
    has fade out, and copies of those only ``destination`` has fade in,
    each at its layer in ``destination`` from the first instant. The
    destination is posed at its own values and slot drawings, where the
    source stands: each moved alike from where its manifest places it. When
    the play is done the scene holds ``destination`` where ``source`` was,
    and neither ``source`` nor any part of the morph; ``source`` itself is
    left as it was, out of the scene.

    A still does not morph: a part or slot that draws a raster member in
    either rig (:class:`~.raster.RasterPart`) has no counterpart, so its
    drawing in ``source`` fades out and its drawing in ``destination`` fades
    in, at its layer in ``destination``.

    ``source`` must be in the scene and ``destination`` not.
    """

    def __init__(self, source: Rig, destination: Rig, **kwargs: Any) -> None:
        source = _rig(source, "What MorphParts morphs from")
        destination = _rig(destination, "What MorphParts morphs into")
        if source is destination:
            raise ValueError("MorphParts morphs a rig into another rig, not itself")
        self.source = source
        self.destination = destination
        super().__init__(source, **kwargs)

    def create_starting_mobject(self) -> Mobject:
        return self.mobject

    def _setup_scene(self, scene: Scene) -> None:
        source, destination = self.source, self.destination
        members = scene.get_mobject_family_members()
        if destination in members:
            raise ValueError(
                "MorphParts brings its destination rig into the scene: it must"
                " not be there before"
            )
        if source not in members:
            raise ValueError("MorphParts morphs a rig that is in the scene")
        placement = source.placement()
        if placement is None:
            raise _no_placement(source)
        destination.place(placement.placed_like(source.model, destination.model))
        # The source's parts are morphed in a stand-in, so the source keeps
        # its parts as its asset draws them.
        self._stand_in = source.copy()
        self._stand_in.suspend_updating()
        scene.replace(source, self._stand_in)
        theirs = dict(destination._members)
        ours = dict(self._stand_in._members)
        # A name drawn as a still on either side has no counterpart: its
        # drawing on the other side arrives under a key no name can be.
        stills = {
            name
            for name in ours
            if name in theirs
            and (
                _still(_drawn(self._stand_in, name))
                or _still(_drawn(destination, name))
            )
        }
        arrives = {
            name: _ARRIVING + name if name in stills else name for name in theirs
        }
        self._pairs = [
            (_drawn(self._stand_in, name), _drawn(destination, name))
            for name in ours
            if name in theirs and name not in stills
        ]
        self._leaving = [
            _drawn(self._stand_in, n) for n in ours if n not in theirs or n in stills
        ]
        arriving = {
            arrives[n]: _drawn(destination, n).copy()
            for n in theirs
            if n not in ours or n in stills
        }
        self._arriving = list(arriving.values())
        if arriving:
            # Each comes in at its layer in the destination, among the parts
            # of the stand-in, and stays there.
            groups = [
                ours[name] if name in ours else arriving[name]
                for name in _layered(list(ours), [arrives[n] for n in theirs])
            ]
            self._stand_in.remove(*ours.values())
            self._stand_in.add(*groups)
        super()._setup_scene(scene)

    def begin(self) -> None:
        self._parts: list[Animation] = [
            *(
                Transform(ours, theirs, rate_func=linear)
                for ours, theirs in self._pairs
            ),
            *(FadeOut(leaving, rate_func=linear) for leaving in self._leaving),
            *(FadeIn(arriving, rate_func=linear) for arriving in self._arriving),
        ]
        for part in self._parts:
            part.begin()
        super().begin()

    def interpolate_mobject(self, alpha: float) -> None:
        progress = self.get_sub_alpha(alpha, 0, 1)
        for part in self._parts:
            part.interpolate(progress)

    def update_mobjects(self, dt: float) -> None:
        for part in self._parts:
            part.update_mobjects(dt)

    def finish(self) -> None:
        super().finish()
        for part in self._parts:
            part.finish()

    def clean_up_from_scene(self, scene: Scene) -> None:
        scene.replace(self._stand_in, self.destination)
        super().clean_up_from_scene(scene)


def _layered(ours: list[str], theirs: list[str]) -> list[str]:
    """Names of both rigs' parts and slots in one drawing order: ``ours``
    in their order, each of ``theirs`` that ``ours`` lacks placed where
    ``theirs`` puts it, after the parts of ``ours`` that ``theirs`` puts
    below it.
    """
    pending = list(ours)
    found: list[str] = []
    for name in theirs:
        if name in ours:
            while pending:
                taken = pending.pop(0)
                found.append(taken)
                if taken == name:
                    break
        else:
            found.append(name)
    return found + pending


#: What a destination's drawing of a name drawn as a still on either side
#: arrives under, beside the source's drawing of that name: no name is
#: spelt with it.
_ARRIVING = "\0"


def _still(drawn: VMobject) -> bool:
    """Whether ``drawn`` (a part's group, or a slot's drawing) draws a
    raster member.
    """
    from .raster import RasterPart

    return any(isinstance(shape, RasterPart) for shape in drawn.get_family())


def _drawn(rig: Rig, name: str) -> VMobject:
    """What ``rig`` draws of its part or slot ``name``: the part's group, or
    the group of the drawing the slot shows.
    """
    group = rig.part(name)
    if isinstance(rig.model.placed[name], PartSpec):
        return group
    shown = rig.shown(name)
    drawing: VMobject = next(d for d in group.submobjects if d.name == shown)
    return drawing
