"""The media clock: how a clip's or a sprite's own time runs against scene
time, and the one driver per scene that keeps every running clock on it.

A clocked media object holds a **program**: an immutable tuple of
:class:`Segment`, each starting at a scene instant. Its four calls
(:meth:`~.VideoMobject.run`, ``loop``, ``hold`` and ``scrub``) each append
one segment at the scene's present instant, so an earlier capture keeps the
program as it was then. The program is data, and media time is exact
arithmetic on it (:func:`program_time`)::

    program time at the scene instant t, in the last segment that starts
    at or before t:
        m = media + rate * (min(t, until) - start)      (no until: t)
        wrapping:  m mod duration
        otherwise: in range while 0 <= m <= duration
    the frame for a program time in range:
        min(floor(m * frame_rate), frame_count - 1)

A new object stands still at media time 0. A program that neither wraps nor
holds runs out of range below 0 or past the clip's duration; media time
equal to the duration shows the last frame and is in range. A clip that is
shown while its program is out of range is refused with
``media.clock_past_end``, at the instant the timeline evaluates it, and
again when a frame showing it is drawn.

A scene's **media driver** (:class:`MediaDriver`, found with
:func:`media_driver`) is made with its first running clock: it is a scene
updater, run last at every instant the timeline evaluates, that reads
:attr:`~.Scene.time` and sets each clock it holds to what its program says
there. Its updater reads the ``dt`` it is handed and never uses it as time,
so it runs at every tick, at every frame rate: a clip changes frame, and
runs past its end, at the instant its program names, whichever instants a
frame rate reads. A later object joins it through :func:`hold_media`. A rig
viewed through a moving camera (:meth:`~arachne.assets.Rig.view_through`)
joins it too, and is settled by it at every instant, after the clocks are
set: the rig's art slides with the camera through plays that animate the
rig, whose own updater an animation suspends. Importing this module reads
no asset and builds no object.
"""

from __future__ import annotations

__all__ = [
    "STANDING",
    "MediaDriver",
    "Segment",
    "frame_at",
    "hold_media",
    "in_range",
    "media_driver",
    "program_time",
    "runs",
]

import math
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Any

from ...timeline.record import DEPENDENCIES

if TYPE_CHECKING:
    from ...assets.rig import Rig
    from ...scene.scene import Scene
    from ..mobject import Mobject
    from .objects import MediaObject


@dataclass(frozen=True)
class Segment:
    """One segment of a clock program, from the scene instant ``start`` on:
    media time ``media`` there, running ``rate`` media seconds a scene
    second, wrapping at the clip's ends when ``wrap``, and standing still
    after the scene instant ``until`` when it is given (a scrub's end, or a
    run told to hold where it ends). Every number is exact.
    """

    start: Fraction
    media: Fraction
    rate: Fraction
    wrap: bool = False
    until: Fraction | None = None


#: The program of a new object, and a still's for good: media time 0.
STANDING: tuple[Segment, ...] = (Segment(Fraction(0), Fraction(0), Fraction(0)),)


def _segment_at(program: tuple[Segment, ...], t: Fraction) -> Segment:
    """The last segment of ``program`` that starts at or before ``t``."""
    found = program[0]
    for segment in program[1:]:
        if segment.start > t:
            break
        found = segment
    return found


def program_time(
    program: tuple[Segment, ...], t: Fraction, duration: Fraction | None
) -> Fraction:
    """The media time ``program`` gives at the scene instant ``t``, for a
    clip of ``duration`` seconds (``None`` for a still, whose program never
    wraps): wrapped into the clip where the program wraps, else unclamped.
    """
    segment = _segment_at(program, t)
    end = t if segment.until is None else min(t, segment.until)
    media = segment.media + segment.rate * (end - segment.start)
    if segment.wrap:
        assert duration is not None
        return media % duration
    return media


def in_range(media: Fraction, duration: Fraction) -> bool:
    """Whether the media time ``media`` lies in a clip of ``duration``
    seconds: from 0 to the duration, both included.
    """
    return 0 <= media <= duration


def frame_at(media: Fraction, frame_rate: Fraction, frame_count: int) -> int:
    """The index of the frame the media time ``media`` shows, in a clip of
    ``frame_count`` frames at ``frame_rate``; a time out of range is held to
    the clip's first or last frame.
    """
    return min(max(math.floor(media * frame_rate), 0), frame_count - 1)


def runs(program: tuple[Segment, ...], t: Fraction) -> bool:
    """Whether ``program``'s media time is changing at the scene instant
    ``t``.
    """
    segment = _segment_at(program, t)
    return segment.rate != 0 and (segment.until is None or t < segment.until)


class MediaDriver:
    """What keeps a scene's clocks on its time: each clocked media object
    it holds set, at every instant the timeline evaluates, to the frame its
    program names there.

    A scene has at most one, made with its first running clock and found
    with :func:`media_driver`. It runs as the scene updater :attr:`updater`,
    last in every instant, after the animations and the mobjects' updaters,
    and unsuspended by any animation, so a clip runs on through plays that
    animate other things, through waits and through its own fade.
    :attr:`updater` reads the ``dt`` it is handed and never uses it as
    time: the driver reads :attr:`~.Scene.time`, and a callback that reads
    its ``dt`` runs at every tick, so the driver answers alike at every
    frame rate. Calling the driver sets every object it holds; an object
    joins only through :func:`hold_media`. Something else that is to run
    the driver, such as a clock that runs last itself, takes
    :attr:`updater` out of the scene's updaters and calls the driver at its
    own instant.

    Once set, an object that is shown while its program is out of range is
    refused with ``media.clock_past_end``; one that runs while a layer of
    the scene reads it and the scene does not hold it is refused with a
    plain error, since ``state_at`` would not list it.

    The driver is the scene's engine state, kept beside the scene, and no
    record reads it. It refers to the scene and to its objects weakly: a
    copy an animation makes of a media object is not held, and keeps the
    frame it was copied with.
    """

    def __init__(self, scene: Scene) -> None:
        self._scene = weakref.ref(scene)
        #: The objects held, by id, in the order they joined.
        self._held: dict[int, weakref.ref[MediaObject]] = {}
        #: The viewed rigs held, by id, in the order they joined.
        self._rigs: dict[int, weakref.ref[Rig]] = {}

        def drive_media(dt: float) -> None:
            # Reading dt has the timeline run this at every tick, not only at
            # those a frame reads; it is never time here, Scene.time is.
            _ = dt
            self()

        #: The scene updater the driver runs as.
        self.updater: Callable[[float], None] = drive_media

    @property
    def objects(self) -> tuple[MediaObject, ...]:
        """The objects held that still exist, in the order they joined."""
        return tuple(o for ref in self._held.values() if (o := ref()) is not None)

    @property
    def rigs(self) -> tuple[Rig, ...]:
        """The viewed rigs held that still exist, in the order they joined."""
        return tuple(r for ref in self._rigs.values() if (r := ref()) is not None)

    def holds(self, mobject: Mobject) -> bool:
        """Whether the driver sets ``mobject``."""
        ref = self._held.get(id(mobject)) or self._rigs.get(id(mobject))
        return ref is not None and ref() is mobject

    def hold(self, mobject: MediaObject | Rig) -> None:
        """Set ``mobject`` from now on: a media object's frame, or a viewed
        rig's art.
        """
        from ...assets.rig import Rig

        if self.holds(mobject):
            return
        held: dict[int, Any] = self._rigs if isinstance(mobject, Rig) else self._held
        held[id(mobject)] = weakref.ref(mobject, self._let_go(held, id(mobject)))

    @staticmethod
    def _let_go(held: dict[int, Any], key: int) -> Callable[[weakref.ref[Any]], None]:
        def gone(ref: weakref.ref[Any]) -> None:
            if held.get(key) is ref:
                del held[key]

        return gone

    def __call__(self) -> None:
        """Set every object held to what its program says at the scene's
        time; refuse one that is shown out of range, and a running one that
        a layer shows while the scene does not hold it.
        """
        scene = self._scene()
        if scene is None:
            return
        now = scene.time
        running: list[MediaObject] = []
        past: list[MediaObject] = []
        for mobject in self.objects:
            mobject._follow_clock(now)
            if mobject._ran_out is not None:
                past.append(mobject)
            elif runs(mobject._program, now):
                running.append(mobject)
        for rig in self.rigs:
            rig.settle()
        if not (running or past):
            return
        held, read = _shown(scene)
        for mobject in past:
            if id(mobject) in held or id(mobject) in read:
                raise mobject._past_end()
        for mobject in running:
            if id(mobject) in read and id(mobject) not in held:
                raise ValueError(
                    f"the {type(mobject).__name__} of the asset"
                    f" {mobject.asset.name} runs its clock and is shown only"
                    " through a layer that reads it (a window), which state_at"
                    " does not list: add it to the scene as well"
                    " (self.add(...)); the layer still draws it once"
                )


def _shown(scene: Scene) -> tuple[set[int], set[int]]:
    """The ids of what ``scene`` holds (its mobjects and their families),
    and of what a layer among those reads (a mask, a window's objects, with
    their families and what they read in turn).
    """
    held: set[int] = set()
    reading: list[Mobject] = []
    pending: list[Mobject] = list(scene.mobjects)
    while pending:
        mobject = pending.pop()
        if id(mobject) in held:
            continue
        held.add(id(mobject))
        reading.extend(vars(mobject).get(DEPENDENCIES, ()))
        pending.extend(mobject.submobjects)
    read: set[int] = set()
    while reading:
        mobject = reading.pop()
        if id(mobject) in read:
            continue
        read.add(id(mobject))
        reading.extend(mobject.submobjects)
        reading.extend(vars(mobject).get(DEPENDENCIES, ()))
    return held, read


#: Each scene's media driver, by the scene; gone with the scene.
_DRIVERS: weakref.WeakKeyDictionary[Scene, MediaDriver] = weakref.WeakKeyDictionary()


def media_driver(scene: Scene) -> MediaDriver | None:
    """``scene``'s media driver, or ``None`` before its first running clock."""
    return _DRIVERS.get(scene)


def hold_media(scene: Scene, mobject: MediaObject | Rig) -> None:
    """Have ``mobject`` set on ``scene``'s time: a media object's frame, or
    a rig viewed through a moving camera, settled; by the scene's media
    driver, made and added to the scene's updaters with the first object it
    holds.
    """
    from ...assets.rig import Rig
    from .objects import MediaObject

    if not isinstance(mobject, (MediaObject, Rig)):
        raise TypeError(
            "the media driver sets media objects and viewed rigs, not a"
            f" {type(mobject).__name__}"
        )
    driver = _DRIVERS.get(scene)
    if driver is None:
        driver = _DRIVERS[scene] = MediaDriver(scene)
        scene.add_updater(driver.updater)
    driver.hold(mobject)
