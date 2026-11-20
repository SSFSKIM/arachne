"""Media objects: a still, a clip or a sprite, placed in a scene like any
image by naming its asset.

``Still(ref)``, ``VideoMobject(ref)`` and ``Sprite(ref)`` load the asset
``ref`` names (:func:`~arachne.assets.load_asset`: a directory, or
``fixture:<name>``) of the kind ``still``, ``clip`` or ``sprite``, and stand
at the size its manifest declares, on its first frame. Each is an image
object, so it is placed, scaled, ordered and faded as any image is.

What a media object holds, and the scene's record keeps at every capture,
is never pixels: the asset's digest, the index of the frame on display, an
opacity and its four corners, which decide the picture; and beside them the
asset loaded (an immutable value that knows its directory) and the scene's
line that made the object. Its pixels (:meth:`get_pixel_array`) are that
frame, decoded and checked by the process's frame cache
(:mod:`.frames`), with the opacity applied to their alpha. So a clip's
record stays small whatever the clip's length, and whatever reads the
record gets the pixels the live scene had.

An animation may change where a media object is, how large, in what order
and how opaque, and never which frame it shows. Two media objects of other
assets, or a media object and another kind of object, do not morph: such a
transform is a ``ValueError`` naming the fade that replaces it.

A clip's or a sprite's own time runs on a declared clock (:mod:`.clock`):
``run()``, ``loop()``, ``hold()`` and ``scrub(to, over)`` each say how it
runs from the scene's present instant, and the scene's media driver sets
the frame it names at every instant. A still's clock is the constant 0.
``state_at`` reports every media object's ``media.time``, the program's
time at the answer's instant, and ``media.frame``, the frame on display.

A clip or a sprite declares a GPU-only capability (``media.video``,
``media.sprite``), so a scene that shows one is drawn on the GPU. A still
declares none: a scene whose only new object is a still stays a CPU frame.
"""

from __future__ import annotations

__all__ = ["Sprite", "Still", "VideoMobject"]

import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from PIL.Image import Resampling

from ... import quantities
from ...assets.reader import Asset, load_asset
from ...constants import DOWN, LEFT, RIGHT, UP
from ...lint.raised import RaisedFinding
from ...quantities import At, Quantity
from ...routing import declare
from ...timeline.clock import exact
from ...timeline.origins import ORIGINS, origin_of
from ...utils.bezier import interpolate
from ..mobject import morph_refusal
from ..types.image_mobject import AbstractImageMobject
from ..values import FrozenPixels, adopt
from . import clock, frames
from .clock import Segment

if TYPE_CHECKING:
    from typing import Self

    from ...typing import PixelArray
    from ..mobject import Mobject


@dataclass(frozen=True)
class _Made:
    """The scene's line that made a media object, kept beside what decides
    its picture: a refusal raised while one of its frames is drawn, with no
    scene code running, names it.
    """

    line: int | None


@dataclass(frozen=True)
class _RanOut:
    """A clock found out of its clip's range: at the scene instant ``t`` its
    program read ``media``; ``object`` is the object's stable id.
    """

    t: Fraction
    media: Fraction
    object: str


class MediaObject(AbstractImageMobject):
    """A media object: the frames of one media asset, standing on one."""

    #: The manifest kind this class shows.
    asset_kind: ClassVar[str]
    #: The GPU-only capability its objects declare, if any.
    capability: ClassVar[str | None] = None

    def __init__(
        self,
        asset: Asset | str | os.PathLike[str],
        resampling_algorithm: Resampling = Resampling.BICUBIC,
        **kwargs: Any,
    ) -> None:
        loaded = load_asset(asset)
        if loaded.kind != self.asset_kind:
            shown_by = _CLASSES.get(loaded.kind, "Rig")
            raise ValueError(
                f"the asset {loaded.name} is a {loaded.kind}, not a {self.asset_kind}:"
                f" show it with {shown_by}"
            )
        origin = ORIGINS.get(self)
        self._asset = loaded
        self._digest = loaded.digest
        self._frame = 0
        self._opacity = 1.0
        self._made = _Made(origin.scene_line if origin is not None else None)
        #: The clock program (:mod:`.clock`), and, while it reads outside
        #: the clip, where it was found to.
        self._program: tuple[Segment, ...] = clock.STANDING
        self._ran_out: _RanOut | None = None
        super().__init__(
            scale_to_resolution=0, resampling_algorithm=resampling_algorithm, **kwargs
        )
        if self.capability is not None:
            declare(self, self.capability)

    @property
    def asset(self) -> Asset:
        """The asset this object shows."""
        return self._asset

    @property
    def frame(self) -> int:
        """The index of the frame on display."""
        return self._frame

    @property
    def opacity(self) -> float:
        """How opaque the object is, from 0 to 1."""
        return self._opacity

    @property
    def pixel_array(self) -> FrozenPixels:
        """The frame on display, as :meth:`get_pixel_array` gives it."""
        return self.get_pixel_array()

    @pixel_array.setter
    def pixel_array(self, value: PixelArray) -> None:
        raise AttributeError(
            "a media object's pixels are its asset's frames: show another asset"
            " with an object of its own"
        )

    def get_pixel_array(self) -> FrozenPixels:
        """The frame on display, decoded and checked against its digest, its
        alpha multiplied by the object's opacity where that is below 1. An
        object whose clock reads outside its clip has no frame on display:
        drawing it is refused with ``media.clock_past_end``, at the line
        that made it.
        """
        if self._ran_out is not None:
            raise self._past_end(self._made.line)
        pixels = frames.cache.frame(self._asset, self._frame, self._made.line)
        if self._opacity >= 1:
            return pixels
        faded = np.array(pixels)
        faded[:, :, 3] = pixels[:, :, 3] * self._opacity
        return adopt(faded, FrozenPixels)

    def reset_points(self) -> Self:
        """The four corners of the asset's natural size, about the origin."""
        self.points = np.array([UP + LEFT, UP + RIGHT, DOWN + LEFT, DOWN + RIGHT])
        extent = self._asset.manifest.extent
        self.stretch_to_fit_height(extent.height)
        self.stretch_to_fit_width(extent.width)
        return self

    def set_opacity(self, alpha: float) -> Self:
        """Show the object ``alpha`` opaque, from 0 to 1."""
        self._opacity = min(max(float(alpha), 0.0), 1.0)
        return self

    def fade(self, darkness: float = 0.5, family: bool = True) -> Self:
        """Show the object ``1 - darkness`` opaque."""
        self.set_opacity(1 - darkness)
        super().fade(darkness, family)
        return self

    def interpolate_color(
        self, mobject1: Mobject, mobject2: Mobject, alpha: float
    ) -> Self:
        """The opacity between ``mobject1``'s and ``mobject2``'s, media
        objects of this object's asset; the frame on display is left as it
        is.
        """
        start, end = self._same_picture(mobject1), self._same_picture(mobject2)
        self._opacity = float(interpolate(start._opacity, end._opacity, alpha))
        return self

    def _same_picture(self, other: Mobject) -> MediaObject:
        """``other``, a media object of this object's asset; anything else is
        refused as alignment refuses it.
        """
        if isinstance(other, MediaObject) and other._digest == self._digest:
            return other
        raise morph_refusal(self, other)

    def _follow_clock(self, now: Fraction) -> None:
        """Show the frame the clock program names at the scene instant
        ``now``: a still's is its one frame, always.
        """

    def _past_end(self, line: int | None = None) -> RaisedFinding:
        """The refusal of this object shown while its clock reads outside
        its clip, raised at ``line`` (else at the scene line running).
        """
        ran_out = self._ran_out
        assert ran_out is not None
        assert isinstance(self, _Clocked)
        return RaisedFinding(
            "media.clock_past_end",
            f"the {type(self).__name__} of the asset {self._asset.name} is shown"
            f" while its clock reads {float(ran_out.media):g} s at the scene's"
            f" {float(ran_out.t):g} s, outside its {float(self.duration):g} s,"
            " and the clock neither loops nor holds",
            detail={
                "asset": self._asset.name,
                "object": ran_out.object,
                "t": float(ran_out.t),
                "media_time": float(ran_out.media),
                "duration": float(self.duration),
            },
            line=line,
        )

    def _morph_key(self) -> str:
        """The asset's digest: a media object aligns, and so is transformed,
        only with media objects of its own asset.
        """
        return self._digest

    def _morph_name(self) -> str:
        return f"{super()._morph_name()} of the asset {self._asset.name}"


class Still(MediaObject):
    """A still: one frame of an asset of the kind ``still``, standing at the
    size its manifest declares. It declares no GPU-only capability.
    """

    asset_kind = "still"


class _Clocked(MediaObject):
    """A media object with frames at a rate: a clip or a sprite."""

    @property
    def frame_count(self) -> int:
        """How many frames the asset has."""
        media = self._asset.manifest.media
        assert media is not None
        return len(media.frames)

    @property
    def frame_rate(self) -> Fraction:
        """The asset's frames a second."""
        media = self._asset.manifest.media
        assert media is not None
        assert media.rate is not None
        return media.rate

    @property
    def duration(self) -> Fraction:
        """The asset's length in seconds: its frame count over its rate."""
        return self.frame_count / self.frame_rate

    @property
    def media_time(self) -> Fraction:
        """The clock's media time at the running scene's present instant:
        the program's time, unclamped where it reads outside the clip.
        """
        return clock.program_time(self._program, _present("media_time"), self.duration)

    def run(
        self,
        rate: float | Fraction = 1,
        *,
        at: float | Fraction | None = None,
        then: str | None = None,
    ) -> Self:
        """Run the clock from the scene's present instant at ``rate`` media
        seconds a scene second (negative runs backward), from the media
        time ``at`` (else from where it is). With ``then="hold"`` the clip
        stands on the frame it reaches at its end (its start, run
        backward); with none, a clip shown past either end is refused
        (``media.clock_past_end``).
        """
        if then not in (None, "hold"):
            raise ValueError(f'then is None or "hold", not {then!r}')
        now = _present("run")
        speed = _exact(rate, "rate")
        media = self._media_at(now, at, "at")
        until = None
        if then == "hold" and speed != 0:
            end = self.duration if speed > 0 else Fraction(0)
            until = now + (end - media) / speed
        return self._declare(Segment(now, media, speed, False, until), runs=True)

    def loop(
        self, rate: float | Fraction = 1, *, at: float | Fraction | None = None
    ) -> Self:
        """Run the clock from the scene's present instant at ``rate``,
        wrapping at the clip's ends, from the media time ``at`` (else from
        where it is).
        """
        now = _present("loop")
        speed = _exact(rate, "rate")
        media = self._media_at(now, at, "at")
        return self._declare(Segment(now, media, speed, True), runs=True)

    def hold(
        self, at: float | Fraction | None = None, *, frame: int | None = None
    ) -> Self:
        """Stand the clock still from the scene's present instant: at the
        media time ``at``, or on the frame ``frame``, or where it is.
        """
        now = _present("hold")
        if frame is not None:
            if at is not None:
                raise ValueError("hold takes a media time or a frame, not both")
            if not isinstance(frame, Integral) or isinstance(frame, bool):
                raise TypeError(f"a frame is an index, not {frame!r}")
            if not 0 <= frame < self.frame_count:
                raise ValueError(
                    f"the asset {self._asset.name} has the frames 0 to"
                    f" {self.frame_count - 1} ({float(self.duration):g} s),"
                    f" not {frame}"
                )
            media = Fraction(int(frame)) / self.frame_rate
        else:
            media = self._media_at(now, at, "at")
        return self._declare(Segment(now, media, Fraction(0)), runs=False)

    def scrub(self, to: float | Fraction, over: float | Fraction) -> Self:
        """Move the clock from where it is to the media time ``to`` over
        ``over`` scene seconds from the scene's present instant, evenly, and
        stand it there.
        """
        now = _present("scrub")
        target = self._within(_exact(to, "to"), "to")
        span = _exact(over, "over")
        if span <= 0:
            raise ValueError(f"a scrub lasts more than 0 scene seconds, not {over!r}")
        media = self._media_at(now, None, "at")
        rate = (target - media) / span
        return self._declare(Segment(now, media, rate, False, now + span), runs=True)

    def _media_at(
        self, now: Fraction, at: float | Fraction | None, name: str
    ) -> Fraction:
        """The media time ``at``, checked to lie in the clip; else the
        clock's time at ``now``.
        """
        if at is None:
            return clock.program_time(self._program, now, self.duration)
        return self._within(_exact(at, name), name)

    def _within(self, media: Fraction, name: str) -> Fraction:
        """``media``, a media time ``name`` asked for, checked to lie in the
        clip.
        """
        if not clock.in_range(media, self.duration):
            raise ValueError(
                f"{name} is a media time from 0 to the asset {self._asset.name}'s"
                f" duration, {float(self.duration):g} s, not {float(media):g}"
            )
        return media

    def _declare(self, segment: Segment, *, runs: bool) -> Self:
        """Append ``segment`` to the clock program, show what it names now,
        and, for a clock that is to run, have the scene's media driver set
        the object from now on.
        """
        from ...timeline.recorder import running

        self._program = (*self._program, segment)
        self._follow_clock(segment.start)
        if runs:
            recorder = running()
            assert recorder is not None
            clock.hold_media(recorder.scene, self)
        return self

    def __deepcopy__(self, clone_from_id: dict[int, Any]) -> Self:
        """A copy, standing on the frame this object shows, at the media time
        its clock reads (the start of that frame where the clock has moved
        on, or reads outside the clip). No driver sets a copy, so its own
        clock stands still until one of the four calls runs it.
        """
        copied = super().__deepcopy__(clone_from_id)
        from ...timeline.recorder import running

        recorder = running()
        media = None
        if recorder is not None:
            now = clock.program_time(self._program, recorder.scene.time, self.duration)
            shown = clock.frame_at(now, self.frame_rate, self.frame_count)
            if clock.in_range(now, self.duration) and shown == self._frame:
                media = now
        if media is None:
            media = Fraction(self._frame) / self.frame_rate
        copied._program = (Segment(Fraction(0), media, Fraction(0)),)
        copied._ran_out = None
        return copied

    def _follow_clock(self, now: Fraction) -> None:
        """Show the frame the clock program names at the scene instant
        ``now``, and note whether the program reads outside the clip there.
        """
        media = clock.program_time(self._program, now, self.duration)
        self._frame = clock.frame_at(media, self.frame_rate, self.frame_count)
        if clock.in_range(media, self.duration):
            self._ran_out = None
        else:
            self._ran_out = _RanOut(now, media, origin_of(self).id)


class VideoMobject(_Clocked):
    """A clip: the frames of an asset of the kind ``clip``, standing at the
    size its manifest declares, on its first frame. A scene that shows one
    is drawn on the GPU (``media.video``).
    """

    asset_kind = "clip"
    capability = "media.video"


class Sprite(_Clocked):
    """A sprite: the drawings of an asset of the kind ``sprite``, standing
    at the size its manifest declares, on its first. A scene that shows one
    is drawn on the GPU (``media.sprite``).
    """

    asset_kind = "sprite"
    capability = "media.sprite"


#: The class that shows each media kind.
_CLASSES = {"still": "Still", "clip": "VideoMobject", "sprite": "Sprite"}


def _present(call: str) -> Fraction:
    """The running scene's present instant; a clock is declared and read
    only while a scene runs.
    """
    from ...timeline.recorder import running

    recorder = running()
    if recorder is None:
        raise RuntimeError(
            f"{call} works on the clock from the running scene's present"
            " instant: call it inside the scene's construct"
        )
    return recorder.scene.time


def _exact(value: float | Fraction, name: str) -> Fraction:
    """``value``, a time or a rate given as ``name``, exactly."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} is a number, not {value!r}")
    return exact(value)


#: The prefix of the media quantities.
PREFIX = "media."
quantities.register(PREFIX, owner="arachne.mobject.media", units=("s",))


@quantities.provider(MediaObject)
def _media_numbers(held: Mapping[str, Any], at: At) -> Iterable[Quantity]:
    """A media object's ``media.time``, the program's time at the instant
    read, and ``media.frame``, the frame on display, from its recorded
    program and frame.
    """
    asset, program, frame = held.get("_asset"), held.get("_program"), held.get("_frame")
    if not isinstance(asset, Asset) or not isinstance(program, tuple):
        return ()
    media = asset.manifest.media
    assert media is not None
    duration = None if media.rate is None else len(media.frames) / media.rate
    return (
        Quantity(
            f"{PREFIX}time", float(clock.program_time(program, at.as_of, duration)), "s"
        ),
        Quantity(f"{PREFIX}frame", frame, "1"),
    )
