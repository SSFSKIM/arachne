"""What a comparison is given: thresholds, frames, object maps and changed sets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Literal, Protocol

import numpy as np

Box = tuple[int, int, int, int]
"""``x0, y0, x1, y1`` in pixels of the frame: half-open, y pointing down."""

Mode = Literal["same", "cross"]
"""``same``: both sequences come from one renderer. ``cross``: two renderers or two
drivers; only there do the edge rule and the membership floor apply."""

Side = Literal["reference", "candidate"]


@dataclass(frozen=True)
class EdgeBudget:
    """How many of one object's edge pixels may differ.

    At most ``max(floor_px, share * edge_pixels)`` of an object's edge pixels may
    differ by more than ``delta`` in any channel.
    """

    delta: int
    share: float
    floor_px: int

    def allowed(self, edge_pixels: int) -> float:
        return max(float(self.floor_px), self.share * edge_pixels)


@dataclass(frozen=True)
class Thresholds:
    """The visible-scale thresholds and the sampling settings, at their starting values."""

    cluster_share: float = 0.0002
    strong_delta: int = 16
    cross_floor: int = 8
    edge_budget: EdgeBudget | None = None
    sample_rate: float = 0.05
    replay_cap: int = 32
    replay_reserve: int = 8

    def cluster_pixels(self, height: int, width: int) -> float:
        """The size a cluster must exceed, in pixels, at this frame size."""
        return self.cluster_share * height * width

    def member_floor(self, mode: Mode) -> int:
        """A pixel is changed when its largest channel difference exceeds this."""
        return self.cross_floor if mode == "cross" else 0

    def as_dict(self) -> dict:
        budget = self.edge_budget
        return {
            "cluster_share": self.cluster_share,
            "strong_delta": self.strong_delta,
            "cross_floor": self.cross_floor,
            "edge_budget": None
            if budget is None
            else {
                "delta": budget.delta,
                "share": budget.share,
                "floor_px": budget.floor_px,
            },
            "sample_rate": self.sample_rate,
            "replay_cap": self.replay_cap,
            "replay_reserve": self.replay_reserve,
        }


@dataclass(frozen=True)
class Footprint:
    """Where one drawable covers the frame.

    ``mask`` is a boolean array over ``box``. ``None`` means only the box is known,
    which serves the flicker test only; a mask with no true pixel means the object
    was examined and draws nothing. ``grow`` is the declared radius, in pixels, of a
    blur or glow on the object. ``parent`` is set on a part (a face of a surface): a
    part gives edges and is not tested for presence. ``source`` is the object's
    source line, where the map's producer knows it.
    """

    box: Box
    mask: np.ndarray | None = field(default=None, compare=False, repr=False)
    grow: float = 0.0
    parent: str | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        x0, y0, x1, y1 = (int(v) for v in self.box)
        if x1 < x0 or y1 < y0:
            raise ValueError(
                f"a footprint's box must have x0 <= x1 and y0 <= y1: {self.box}"
            )
        object.__setattr__(self, "box", (x0, y0, x1, y1))
        if self.mask is not None:
            mask = np.asarray(self.mask, dtype=bool)
            if mask.shape != (y1 - y0, x1 - x0):
                raise ValueError(
                    f"a footprint's mask has shape {mask.shape}, its box {self.box} needs "
                    f"{(y1 - y0, x1 - x0)}"
                )
            object.__setattr__(self, "mask", mask)
        if self.grow < 0:
            raise ValueError("a footprint's grow is a radius and cannot be negative")

    @property
    def examined(self) -> bool:
        """Whether the pixels are known, not only the box."""
        return self.mask is not None

    @property
    def empty(self) -> bool:
        """Examined and drawing nothing."""
        return self.mask is not None and not self.mask.any()


@dataclass(frozen=True)
class ObjectMap:
    """Which pixels each drawable covers in one frame, by its stable id.

    Footprints may overlap. ``complete``: an id that is not listed draws nothing. Not
    complete: an id that is not listed is untested.
    """

    size: tuple[int, int]
    objects: Mapping[str, Footprint]
    complete: bool

    def __post_init__(self) -> None:
        height, width = (int(v) for v in self.size)
        object.__setattr__(self, "size", (height, width))
        for name, footprint in self.objects.items():
            x0, y0, x1, y1 = footprint.box
            if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
                raise ValueError(
                    f"{name!r}: box {footprint.box} lies outside a {width} x {height} frame"
                )

    @classmethod
    def from_id_image(
        cls, ids: np.ndarray, table: Mapping[int, str] | Sequence[str]
    ) -> ObjectMap:
        """A complete map from an image holding one object number per pixel.

        ``table`` gives the id of each number (a sequence is indexed by the number).
        Pixels whose number the table does not hold are background.
        """
        ids = np.asarray(ids)
        if ids.ndim != 2:
            raise ValueError("an id image is two-dimensional")
        names = (
            dict(enumerate(table)) if not isinstance(table, Mapping) else dict(table)
        )
        objects: dict[str, Footprint] = {}
        for value in np.unique(ids):
            name = names.get(int(value))
            if name is None:
                continue
            where = ids == value
            rows = np.flatnonzero(where.any(axis=1))
            cols = np.flatnonzero(where.any(axis=0))
            box = (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)
            objects[name] = Footprint(box, where[box[1] : box[3], box[0] : box[2]])
        return cls(ids.shape, objects, complete=True)


@dataclass(frozen=True)
class ChangedSet:
    """The drawables whose recorded values differ between two instants.

    ``undecided``: drawables the record cannot say about (an instant evaluated
    between captures). ``camera``: the camera's recorded inputs differ.
    """

    ids: frozenset[str]
    undecided: frozenset[str] = frozenset()
    camera: bool = False


@dataclass(frozen=True)
class Group:
    """One 8-connected group of changed pixels.

    ``cluster`` is true when the group is over both visible-scale thresholds.
    ``mask``, where kept, is the group's pixels over its box.
    """

    box: Box
    pixels: int
    peak: int
    cluster: bool
    mask: np.ndarray | None = field(default=None, compare=False, repr=False)


class Frames(Protocol):
    """A sequence of frames: ``uint8`` arrays of height x width x 3 or 4."""

    def __len__(self) -> int: ...

    def instant(self, i: int) -> Fraction: ...

    def frame(self, i: int) -> np.ndarray: ...


class MapSource(Protocol):
    """Where a comparison gets the object maps of sampled frames."""

    def boxes(self, frame: int) -> tuple[Mapping[str, Box], Mapping[str, Box]] | None:
        """Each side's drawables with boxes in the frame's pixels.

        ``None``: this source's maps need no list of ids, and every map is asked
        for whole.
        """
        ...

    def map(
        self, side: Side, frame: int, ids: Sequence[str] | None
    ) -> ObjectMap | None:
        """One side's map of one frame, for ``ids`` (``None``: everything).

        ``None``: this side has no map for the frame.
        """
        ...


class ArrayFrames:
    """Frames held as arrays, at instants ``start + i / fps``."""

    def __init__(
        self, frames: Sequence[np.ndarray], fps: Fraction, start: Fraction = Fraction(0)
    ) -> None:
        self._frames = list(frames)
        self._fps = Fraction(fps)
        self._start = Fraction(start)

    def __len__(self) -> int:
        return len(self._frames)

    def instant(self, i: int) -> Fraction:
        return self._start + Fraction(i) / self._fps

    def frame(self, i: int) -> np.ndarray:
        return self._frames[i]


class HandedMaps:
    """Maps handed in by a backend, by frame number for each side.

    A frame missing from a side's mapping has no map on that side.
    """

    def __init__(
        self,
        reference: Mapping[int, ObjectMap],
        candidate: Mapping[int, ObjectMap],
    ) -> None:
        self._maps = {"reference": reference, "candidate": candidate}

    def boxes(self, frame: int) -> None:
        return None

    def map(
        self, side: Side, frame: int, ids: Sequence[str] | None
    ) -> ObjectMap | None:
        return self._maps[side].get(frame)
