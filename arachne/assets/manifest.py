"""The asset manifest, version 1: what the engine reads of an asset made
elsewhere, checked whole before anything of the asset is drawn.

An asset is a directory holding ``manifest.json`` and the files it names.
:func:`parse_manifest` reads the parsed document into a :class:`Manifest`,
refusing (``asset.manifest_invalid``, naming the member and why) a document
of a major version or a kind this reader does not know, one that requires a
feature it does not know, and one that misses a required member or
contradicts itself. Every name it declares (the asset's, and each part's,
slot's, drawing's, parameter's and anchor's) is made of ASCII letters,
digits, ``_`` and ``-``. A member it does not know, in a document whose
``requires`` it satisfies, is kept and ignored. Geometry is in canvas units
(the part files' user units), y down.

The kind ``parts`` is a set of vector parts (a rig). The kinds ``still``,
``clip`` and ``sprite``, from version 1.1, are media: frames of pixels, one
PNG file each, described by the member ``media`` (:class:`Media`). A media
asset has no canvas: its ``size`` is its natural width and height in scene
units alone (:class:`Extent`). ``media.pixels`` is every frame's width and
height in pixels; ``media.alpha`` whether any pixel may be less than
opaque; ``media.rate`` frames per second as ``[numerator, denominator]``,
positive integers (a sprite's a whole number dividing 60, so that every
change of drawing falls on a tick of the scene's 60 Hz clock; a still has
none); ``media.frames`` the frame files in order, each with its frame
digest (:func:`arachne.mobject.media.frames.frame_digest`). A still has
one frame.

:func:`digest` is the asset's digest: ``sha256:`` and the hex SHA-256 of
every file the manifest names, sorted by relative path, each as its path in
UTF-8, a zero byte, its length as 8 bytes big-endian and its bytes; then of
the manifest as JSON with sorted keys and no insignificant whitespace, its
``checks`` member removed, so that the checks can be written without
changing what they attest.
"""

from __future__ import annotations

__all__ = [
    "ANGLE_UNITS",
    "ATTACHMENTS",
    "DEFAULT_UNIT",
    "DRIVER_KINDS",
    "FEATURES",
    "FRAME_DIGEST",
    "KINDS",
    "MAJOR",
    "MANIFEST",
    "MEDIA_KINDS",
    "NAME",
    "PROVENANCE",
    "SPRITE_TICKS",
    "UNIT",
    "AnchorKey",
    "AnchorSpec",
    "DrawingSpec",
    "DriverSpec",
    "Extent",
    "FrameSpec",
    "Key",
    "Lattice",
    "Manifest",
    "Media",
    "ParameterSpec",
    "PartSpec",
    "Size",
    "SlotSpec",
    "MAX_INTEGER",
    "digest",
    "frozen",
    "normalised",
    "parse_manifest",
    "read_json",
    "refuse",
    "thawed",
]

import hashlib
import json
import math
import posixpath
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import Any, NoReturn

#: The file that makes a directory an asset.
MANIFEST = "manifest.json"
#: The major version of the manifest this reader reads.
MAJOR = 1
#: The kinds of media asset this reader reads: frames of pixels.
MEDIA_KINDS = ("still", "clip", "sprite")
#: The kinds of asset this reader reads.
KINDS = ("parts", *MEDIA_KINDS)
#: The feature names a manifest may require that this reader knows: none in 1.0.
FEATURES: frozenset[str] = frozenset()
#: What a parameter can do to a part.
DRIVER_KINDS = ("rotate", "translate", "scale", "warp", "shape")
#: The units a ``rotate`` driver's output may be in.
ANGLE_UNITS = ("deg", "rad")
#: How an anchor is attached to its part.
ATTACHMENTS = ("outline", "interior")
#: The members ``provenance`` must hold; the engine does not interpret them.
PROVENANCE = ("source_url", "author", "license", "credit_line", "retrieved_at")
#: What a frame digest is: ``sha256:`` and 64 lower-case hex digits.
FRAME_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
#: The scene's clock, in ticks a second: a sprite's rate divides it.
SPRITE_TICKS = 60

Point = tuple[float, float]


def refuse(code: str, what: str, **detail: Any) -> NoReturn:
    """Raise the finding ``code``, saying ``what`` is wrong, with ``detail``."""
    from ..lint.raised import RaisedFinding

    raise RaisedFinding(code, what, detail=detail)


@dataclass(frozen=True)
class Extent:
    """An asset's natural size in scene units: ``width`` and ``height``."""

    width: float
    height: float


@dataclass(frozen=True)
class Size(Extent):
    """Where the canvas of an asset of the kind ``parts`` sits in a scene:
    ``scene_units_per_unit`` scene units to a canvas unit, the canvas point
    ``origin`` at the scene's origin, y flipped; ``width`` and ``height``,
    the natural size in scene units at the default pose.
    """

    scene_units_per_unit: float
    origin: Point


@dataclass(frozen=True)
class FrameSpec:
    """One frame of a media asset: its ``file``, and the frame digest of its
    decoded pixels.
    """

    file: str
    digest: str


@dataclass(frozen=True)
class Media:
    """The frames of a media asset: their size in ``pixels`` (width,
    height), whether any pixel may be less than opaque (``alpha``), their
    ``rate`` in frames a second (``None`` for a still), and the ``frames``
    in order.
    """

    pixels: tuple[int, int]
    alpha: bool
    rate: Fraction | None
    frames: tuple[FrameSpec, ...]


@dataclass(frozen=True)
class Lattice:
    """A part's warp lattice: ``cols`` by ``rows`` cells over ``bounds``
    (``x0, y0, x1, y1``), so ``(cols + 1) * (rows + 1)`` points.
    """

    cols: int
    rows: int
    bounds: tuple[float, float, float, float]

    @property
    def points(self) -> int:
        """How many points the lattice has."""
        return (self.cols + 1) * (self.rows + 1)


@dataclass(frozen=True)
class PartSpec:
    """A part: the whole of ``file``, or its element whose ``id`` is
    ``element``; its ``parent`` (a part or slot, or ``None``), ``pivot``,
    drawing ``order`` (lower is drawn first) and warp ``lattice``.
    """

    name: str
    file: str
    element: str | None
    parent: str | None
    pivot: Point
    order: int
    lattice: Lattice | None


@dataclass(frozen=True)
class DrawingSpec:
    """One drawing a slot can show: ``file``, or its element ``element``."""

    name: str
    file: str
    element: str | None


@dataclass(frozen=True)
class SlotSpec:
    """A slot: placed like a part, it shows one of its ``drawings``,
    ``default`` at rest.
    """

    name: str
    parent: str | None
    pivot: Point
    order: int
    default: str
    drawings: tuple[DrawingSpec, ...]

    def drawing(self, name: str) -> DrawingSpec | None:
        """The drawing ``name``, or ``None``."""
        return next((d for d in self.drawings if d.name == name), None)


@dataclass(frozen=True)
class Key:
    """A driver's output ``value`` at the parameter value ``at``: a number
    (``rotate``, ``scale``), a canvas vector (``translate``), the lattice's
    offsets (``warp``) or a file whose outline is the part's there
    (``shape``).
    """

    at: float
    value: float | Point | tuple[Point, ...] | str


@dataclass(frozen=True)
class DriverSpec:
    """What a parameter does to one part: its ``kind``, the ``unit`` of a
    ``rotate``'s angles, and its ``keys``, the lowest at the parameter's
    ``min`` and the highest at its ``max``.
    """

    part: str
    kind: str
    unit: str | None
    keys: tuple[Key, ...]


@dataclass(frozen=True)
class ParameterSpec:
    """A parameter: its lowest and highest tested values, its default, its
    unit (:data:`DEFAULT_UNIT` when the manifest gives none; any spelling
    :data:`UNIT` admits, read by no part of the engine but a ``rotate``
    driver's own) and what it drives.
    """

    name: str
    min: float
    max: float
    default: float
    unit: str
    drives: tuple[DriverSpec, ...]


@dataclass(frozen=True)
class AnchorKey:
    """An interior anchor's ``point`` at the shape key ``at``."""

    at: float
    point: Point


@dataclass(frozen=True)
class AnchorSpec:
    """A named point of a part: on its outline, or inside it; at rest at
    ``at``, and, inside a reshaped part, at each of its shape keys.
    """

    name: str
    part: str
    attach: str
    at: Point
    keys: tuple[AnchorKey, ...]


@dataclass(frozen=True)
class Manifest:
    """An asset's manifest, checked. ``document`` is the parsed JSON as it
    was, unknown members included, deep-immutable (:func:`frozen`), as are
    ``checks`` and ``provenance``; :func:`thawed` gives a plain copy.
    """

    manifest_version: str
    name: str
    version: int
    kind: str
    requires: tuple[str, ...]
    #: The declared ``size``: a :class:`Size` for the kind ``parts``, an
    #: :class:`Extent` for a media kind, which has no canvas.
    extent: Extent
    parts: tuple[PartSpec, ...]
    slots: tuple[SlotSpec, ...]
    parameters: tuple[ParameterSpec, ...]
    anchors: tuple[AnchorSpec, ...]
    checks: Mapping[str, Any] | None
    provenance: Mapping[str, Any]
    document: Mapping[str, Any]
    #: The frames of a media kind; ``None`` for the kind ``parts``.
    media: Media | None = None

    @property
    def size(self) -> Size:
        """Where the canvas of an asset of the kind ``parts`` sits. A media
        asset has none: asking it of one is a ``ValueError``.
        """
        if not isinstance(self.extent, Size):
            raise ValueError(
                f"the asset {self.name} is a {self.kind}, which has no canvas: a"
                " rig is made of an asset of the kind parts, and a still, a clip"
                " or a sprite is shown by a Still, a VideoMobject or a Sprite"
            )
        return self.extent

    def files(self) -> tuple[str, ...]:
        """Every file the manifest names, as written, in the order named,
        each once.
        """
        if self.media is not None:
            return tuple(dict.fromkeys(frame.file for frame in self.media.frames))
        named: list[str] = [part.file for part in self.parts]
        named += [d.file for slot in self.slots for d in slot.drawings]
        named += [
            key.value
            for parameter in self.parameters
            for driver in parameter.drives
            if driver.kind == "shape"
            for key in driver.keys
            if isinstance(key.value, str)
        ]
        return tuple(dict.fromkeys(named))


def normalised(file: str) -> str:
    """``file``, a path relative to the asset's directory, as the digest
    names it.
    """
    return posixpath.normpath(file)


def digest(document: Mapping[str, Any], files: Mapping[str, bytes]) -> str:
    """The digest of the asset whose manifest is ``document`` and whose
    files are ``files`` (contents by relative path, as :func:`normalised`).
    """
    sha = hashlib.sha256()
    for path in sorted(files):
        data = files[path]
        sha.update(path.encode("utf-8"))
        sha.update(b"\0")
        sha.update(len(data).to_bytes(8, "big"))
        sha.update(data)
    attested = {key: thawed(v) for key, v in document.items() if key != "checks"}
    sha.update(
        json.dumps(
            attested,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    )
    return f"sha256:{sha.hexdigest()}"


#: The largest integer a manifest may hold, so that every JSON reader reads
#: it exactly.
MAX_INTEGER = 2**53 - 1


def read_json(data: bytes) -> Any:
    """The JSON document ``data`` holds, in the domain a manifest is read and
    digested in: UTF-8; objects whose members are each named once; strings
    of Unicode scalar values; finite numbers, integers within
    :data:`MAX_INTEGER`; ``true``, ``false`` and ``null``. Raises
    :class:`ValueError`, saying why, for anything else.
    """

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        found: dict[str, Any] = {}
        for key, value in pairs:
            if key in found:
                raise ValueError(f"the member {key!r} is given twice")
            found[key] = value
        return found

    def real(text: str) -> float:
        number = float(text)
        if not math.isfinite(number):
            raise ValueError(f"{text} is not a finite number")
        return number

    def integer(text: str) -> int:
        number = int(text)
        if abs(number) > MAX_INTEGER:
            raise ValueError(f"{text} is beyond the integers JSON readers agree on")
        return number

    def constant(name: str) -> Any:
        raise ValueError(f"{name} is not a JSON number")

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("it is not UTF-8") from None
    try:
        document = json.loads(
            text,
            object_pairs_hook=unique,
            parse_float=real,
            parse_int=integer,
            parse_constant=constant,
        )
    except RecursionError:
        raise ValueError("it is nested too deep") from None
    _scalar_values(document)
    return document


def _scalar_values(value: Any) -> None:
    """Refuse a string, or a member's name, that is not Unicode scalar
    values (a lone surrogate), which no UTF-8 writes.
    """
    if isinstance(value, dict):
        for key, item in value.items():
            _scalar_values(key)
            _scalar_values(item)
    elif isinstance(value, list):
        for item in value:
            _scalar_values(item)
    elif isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            raise ValueError(f"the string {value!r} is not valid Unicode") from None


def frozen(value: Any) -> Any:
    """``value``, parsed JSON, deep-immutable: objects as read-only mappings
    and arrays as tuples.
    """
    if isinstance(value, Mapping):
        return MappingProxyType({key: frozen(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(frozen(item) for item in value)
    return value


def thawed(value: Any) -> Any:
    """``value``, as :func:`frozen` keeps it, as plain JSON again: a detached
    copy.
    """
    if isinstance(value, Mapping):
        return {key: thawed(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [thawed(item) for item in value]
    return value


def parse_manifest(document: Any, asset: str) -> Manifest:
    """The manifest ``document`` (parsed JSON) of the asset ``asset`` (as it
    was asked for), checked; raises ``asset.manifest_invalid``.
    """
    return _Reader(asset).manifest(document)


_VERSION = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")
#: What a declared name may be: it is a segment of the names an asset's
#: values are read by (``rig.param.<parameter>``, ``rig.slot.<slot>``).
NAME = re.compile(r"[A-Za-z0-9_-]+")
#: What a parameter's unit may be: 1 to 16 printable ASCII characters
#: without white space.
UNIT = re.compile(r"[!-~]{1,16}")
#: The unit of a parameter that gives none: a plain number.
DEFAULT_UNIT = "1"


class _Reader:
    """Reads one manifest, naming in each refusal the member it is about."""

    def __init__(self, asset: str) -> None:
        self.asset = asset

    def invalid(self, member: str, reason: str, **name: str) -> NoReturn:
        refuse(
            "asset.manifest_invalid",
            f"the manifest of the asset {self.asset} is invalid at {member}: {reason}",
            asset=self.asset,
            member=member,
            reason=reason,
            **name,
        )

    def name(self, value: Any, member: str) -> str:
        """A declared name: ASCII letters, digits, ``_`` and ``-``."""
        name = self.text(value, member)
        if NAME.fullmatch(name) is None:
            self.invalid(
                member,
                f"the name {name!r} is not made of ASCII letters, digits, _ and -",
                name=name,
            )
        return name

    # Values

    def object(self, value: Any, member: str) -> Mapping[str, Any]:
        if not isinstance(value, dict):
            self.invalid(member, "it must be an object")
        return value

    def member(self, holder: Mapping[str, Any], name: str, at: str) -> Any:
        if name not in holder:
            self.invalid(_join(at, name), "it is required and missing")
        return holder[name]

    def text(self, value: Any, member: str) -> str:
        if not isinstance(value, str) or not value:
            self.invalid(member, "it must be a non-empty string")
        return value

    def optional_text(self, value: Any, member: str) -> str | None:
        return None if value is None else self.text(value, member)

    def number(self, value: Any, member: str) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            self.invalid(member, "it must be a finite number")
        return float(value)

    def positive(self, value: Any, member: str) -> float:
        number = self.number(value, member)
        if number <= 0:
            self.invalid(member, "it must be above 0")
        return number

    def integer(self, value: Any, member: str, least: int | None = None) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            self.invalid(member, "it must be an integer")
        if least is not None and value < least:
            self.invalid(member, f"it must be at least {least}")
        return value

    def point(self, value: Any, member: str) -> Point:
        if not isinstance(value, list) or len(value) != 2:
            self.invalid(member, "it must be a canvas point, two numbers")
        return (self.number(value[0], member), self.number(value[1], member))

    def array(self, value: Any, member: str, *, empty: bool = True) -> list[Any]:
        if not isinstance(value, list) or (not empty and not value):
            self.invalid(
                member, "it must be a list" + ("" if empty else " of one or more")
            )
        return value

    # The document

    def manifest(self, document: Any) -> Manifest:
        top = self.object(document, "the manifest")
        version = self.member(top, "manifest_version", "")
        matched = _VERSION.fullmatch(version) if isinstance(version, str) else None
        if matched is None:
            self.invalid("manifest_version", 'it must be "<major>.<minor>"')
        if int(matched.group(1)) != MAJOR:
            self.invalid(
                "manifest_version",
                f"major version {matched.group(1)} is not one this reader knows"
                f" (it reads {MAJOR}.x)",
            )
        requires = tuple(
            self.text(name, f"requires[{i}]")
            for i, name in enumerate(self.array(top.get("requires", []), "requires"))
        )
        unknown = [name for name in requires if name not in FEATURES]
        if unknown:
            self.invalid(
                "requires",
                f"it requires {', '.join(unknown)}, which this reader does not know",
            )
        kind = self.text(self.member(top, "kind", ""), "kind")
        if kind not in KINDS:
            self.invalid("kind", f"{kind!r} is not a kind this reader knows")
        name = self.name(self.member(top, "name", ""), "name")
        number = self.integer(self.member(top, "version", ""), "version", 0)
        media: Media | None = None
        parts: tuple[PartSpec, ...] = ()
        slots: tuple[SlotSpec, ...] = ()
        parameters: tuple[ParameterSpec, ...] = ()
        anchors: tuple[AnchorSpec, ...] = ()
        extent: Extent
        if kind in MEDIA_KINDS:
            extent = self.media_size(self.member(top, "size", ""))
            media = self.media(self.member(top, "media", ""), kind)
        else:
            extent = self.size(self.member(top, "size", ""))
            parts = tuple(
                self.part(entry, f"parts[{i}]")
                for i, entry in enumerate(
                    self.array(self.member(top, "parts", ""), "parts", empty=False)
                )
            )
            slots = tuple(
                self.slot(entry, f"slots[{i}]")
                for i, entry in enumerate(self.array(top.get("slots", []), "slots"))
            )
            placed = self.placed(parts, slots)
            parameters = tuple(
                self.parameter(entry, f"parameters[{i}]", parts, placed)
                for i, entry in enumerate(
                    self.array(top.get("parameters", []), "parameters")
                )
            )
            self.unique(parameters, "parameters")
            shaped = self.shaped(parameters)
            anchors = tuple(
                self.anchor(entry, f"anchors[{i}]", placed, shaped)
                for i, entry in enumerate(self.array(top.get("anchors", []), "anchors"))
            )
            self.unique(anchors, "anchors")
        checks = top.get("checks")
        if checks is not None:
            self.object(checks, "checks")
        provenance = self.object(self.member(top, "provenance", ""), "provenance")
        for field in PROVENANCE:
            self.member(provenance, field, "provenance")
        return Manifest(
            manifest_version=version,
            name=name,
            version=number,
            kind=kind,
            requires=requires,
            extent=extent,
            parts=parts,
            slots=slots,
            parameters=parameters,
            anchors=anchors,
            checks=None if checks is None else frozen(checks),
            provenance=frozen(provenance),
            document=frozen(top),
            media=media,
        )

    def size(self, value: Any) -> Size:
        size = self.object(value, "size")
        found = Size(
            scene_units_per_unit=self.positive(
                self.member(size, "scene_units_per_unit", "size"),
                "size.scene_units_per_unit",
            ),
            origin=self.point(self.member(size, "origin", "size"), "size.origin"),
            width=self.extent(self.member(size, "width", "size"), "size.width"),
            height=self.extent(self.member(size, "height", "size"), "size.height"),
        )
        if found.width == 0 and found.height == 0:
            self.invalid("size", "width and height are both 0; one may be, for a line")
        return found

    def extent(self, value: Any, member: str) -> float:
        number = self.number(value, member)
        if number < 0:
            self.invalid(member, "it must be 0 or more")
        return number

    # Media

    def media_size(self, value: Any) -> Extent:
        """A media asset's ``size``: its natural width and height, both above
        0, and no canvas.
        """
        size = self.object(value, "size")
        for canvas in ("scene_units_per_unit", "origin"):
            if canvas in size:
                self.invalid(
                    f"size.{canvas}",
                    "a media asset has no canvas: its size is its width and height"
                    " alone",
                )
        return Extent(
            width=self.positive(self.member(size, "width", "size"), "size.width"),
            height=self.positive(self.member(size, "height", "size"), "size.height"),
        )

    def media(self, value: Any, kind: str) -> Media:
        """The member ``media`` of an asset of the media kind ``kind``."""
        entry = self.object(value, "media")
        pixels = self.array(self.member(entry, "pixels", "media"), "media.pixels")
        if len(pixels) != 2:
            self.invalid("media.pixels", "it must be two integers, width and height")
        width, height = (self.integer(p, "media.pixels", 1) for p in pixels)
        alpha = self.member(entry, "alpha", "media")
        if not isinstance(alpha, bool):
            self.invalid("media.alpha", "it must be true or false")
        rate = self.rate(entry, kind)
        listed = self.array(self.member(entry, "frames", "media"), "media.frames")
        if kind == "still" and len(listed) != 1:
            self.invalid(
                "media.frames", f"a still has one frame, and {len(listed)} are listed"
            )
        if not listed:
            self.invalid("media.frames", f"a {kind} has one frame or more")
        frames = tuple(
            self.frame(item, f"media.frames[{i}]") for i, item in enumerate(listed)
        )
        digests: dict[str, str] = {}
        for i, frame in enumerate(frames):
            if digests.setdefault(normalised(frame.file), frame.digest) != frame.digest:
                self.invalid(
                    f"media.frames[{i}].digest",
                    f"the file {frame.file} is listed again with another digest",
                )
        return Media(pixels=(width, height), alpha=alpha, rate=rate, frames=frames)

    def rate(self, entry: Mapping[str, Any], kind: str) -> Fraction | None:
        """A clip's or a sprite's frames a second, ``[numerator,
        denominator]``, from the ``media`` member ``entry``; a still has
        none, so it writes no ``rate`` at all (not even ``null``).
        """
        if kind == "still":
            if "rate" in entry:
                self.invalid("media.rate", "a still has one frame and no rate")
            return None
        if "rate" not in entry:
            self.invalid("media.rate", "it is required and missing")
        found = self.array(entry["rate"], "media.rate")
        if len(found) != 2:
            self.invalid(
                "media.rate", "it must be two integers, numerator and denominator"
            )
        numerator, denominator = (self.integer(n, "media.rate", 1) for n in found)
        rate = Fraction(numerator, denominator)
        if kind == "sprite" and (
            rate.denominator != 1 or SPRITE_TICKS % rate.numerator != 0
        ):
            self.invalid(
                "media.rate",
                f"a sprite's rate is a whole number of frames a second that divides"
                f" {SPRITE_TICKS}, so every drawing changes on a tick; {rate} does not",
            )
        return rate

    def frame(self, value: Any, at: str) -> FrameSpec:
        entry = self.object(value, at)
        file = self.text(self.member(entry, "file", at), f"{at}.file")
        digest = self.text(self.member(entry, "digest", at), f"{at}.digest")
        if FRAME_DIGEST.fullmatch(digest) is None:
            self.invalid(
                f"{at}.digest", "it must be sha256: and 64 lower-case hex digits"
            )
        return FrameSpec(file=file, digest=digest)

    def part(self, value: Any, at: str) -> PartSpec:
        entry = self.object(value, at)
        lattice = entry.get("lattice")
        return PartSpec(
            name=self.name(self.member(entry, "name", at), f"{at}.name"),
            file=self.text(self.member(entry, "file", at), f"{at}.file"),
            element=self.optional_text(entry.get("element"), f"{at}.element"),
            parent=self.optional_text(self.member(entry, "parent", at), f"{at}.parent"),
            pivot=self.point(self.member(entry, "pivot", at), f"{at}.pivot"),
            order=self.integer(self.member(entry, "order", at), f"{at}.order"),
            lattice=None if lattice is None else self.lattice(lattice, f"{at}.lattice"),
        )

    def lattice(self, value: Any, at: str) -> Lattice:
        entry = self.object(value, at)
        bounds = self.array(self.member(entry, "bounds", at), f"{at}.bounds")
        if len(bounds) != 4:
            self.invalid(f"{at}.bounds", "it must be four numbers, x0, y0, x1, y1")
        x0, y0, x1, y1 = (self.number(b, f"{at}.bounds") for b in bounds)
        if not (x1 > x0 and y1 > y0):
            self.invalid(f"{at}.bounds", "x1 must be above x0 and y1 above y0")
        return Lattice(
            cols=self.integer(self.member(entry, "cols", at), f"{at}.cols", 1),
            rows=self.integer(self.member(entry, "rows", at), f"{at}.rows", 1),
            bounds=(x0, y0, x1, y1),
        )

    def slot(self, value: Any, at: str) -> SlotSpec:
        entry = self.object(value, at)
        drawings = tuple(
            self.drawing(d, f"{at}.drawings[{i}]")
            for i, d in enumerate(
                self.array(
                    self.member(entry, "drawings", at), f"{at}.drawings", empty=False
                )
            )
        )
        self.unique(drawings, f"{at}.drawings")
        default = self.text(self.member(entry, "default", at), f"{at}.default")
        if default not in {d.name for d in drawings}:
            self.invalid(f"{at}.default", f"the slot has no drawing {default!r}")
        return SlotSpec(
            name=self.name(self.member(entry, "name", at), f"{at}.name"),
            parent=self.optional_text(self.member(entry, "parent", at), f"{at}.parent"),
            pivot=self.point(self.member(entry, "pivot", at), f"{at}.pivot"),
            order=self.integer(self.member(entry, "order", at), f"{at}.order"),
            default=default,
            drawings=drawings,
        )

    def drawing(self, value: Any, at: str) -> DrawingSpec:
        entry = self.object(value, at)
        return DrawingSpec(
            name=self.name(self.member(entry, "name", at), f"{at}.name"),
            file=self.text(self.member(entry, "file", at), f"{at}.file"),
            element=self.optional_text(entry.get("element"), f"{at}.element"),
        )

    def placed(
        self, parts: Sequence[PartSpec], slots: Sequence[SlotSpec]
    ) -> Mapping[str, PartSpec | SlotSpec]:
        """Every part and slot by name, names unique and every parent chain
        ending at a root.
        """
        placed: dict[str, PartSpec | SlotSpec] = {}
        members: dict[str, str] = {}
        entries: list[PartSpec | SlotSpec] = [*parts, *slots]
        for i, entry in enumerate(entries):
            at = f"parts[{i}]" if i < len(parts) else f"slots[{i - len(parts)}]"
            if entry.name in placed:
                self.invalid(f"{at}.name", f"{entry.name!r} names another part or slot")
            placed[entry.name] = entry
            members[entry.name] = at
        for name, entry in placed.items():
            seen = {name}
            parent = entry.parent
            while parent is not None:
                if parent not in placed:
                    self.invalid(
                        f"{members[name]}.parent",
                        f"there is no part or slot {parent!r}",
                    )
                if parent in seen:
                    self.invalid(
                        f"{members[name]}.parent", "the parent chain is a loop"
                    )
                seen.add(parent)
                parent = placed[parent].parent
        return placed

    def unique(self, entries: Sequence[Any], at: str) -> None:
        names = [entry.name for entry in entries]
        for i, name in enumerate(names):
            if name in names[:i]:
                self.invalid(f"{at}[{i}].name", f"{name!r} is named twice")

    def parameter(
        self,
        value: Any,
        at: str,
        parts: Sequence[PartSpec],
        placed: Mapping[str, PartSpec | SlotSpec],
    ) -> ParameterSpec:
        entry = self.object(value, at)
        name = self.name(self.member(entry, "name", at), f"{at}.name")
        unit = entry.get("unit", DEFAULT_UNIT)
        if not isinstance(unit, str) or UNIT.fullmatch(unit) is None:
            self.invalid(
                f"{at}.unit",
                f"the unit {unit!r} is not 1 to 16 printable ASCII characters without"
                " white space",
                name=name,
            )
        low = self.number(self.member(entry, "min", at), f"{at}.min")
        high = self.number(self.member(entry, "max", at), f"{at}.max")
        if not low < high:
            self.invalid(f"{at}.max", "it must be above min")
        default = self.number(self.member(entry, "default", at), f"{at}.default")
        if not low <= default <= high:
            self.invalid(f"{at}.default", "it must lie from min to max")
        drives = tuple(
            self.driver(d, f"{at}.drives[{i}]", (low, high), placed)
            for i, d in enumerate(
                self.array(
                    self.member(entry, "drives", at), f"{at}.drives", empty=False
                )
            )
        )
        return ParameterSpec(
            name=name, min=low, max=high, default=default, unit=unit, drives=drives
        )

    def driver(
        self,
        value: Any,
        at: str,
        span: tuple[float, float],
        placed: Mapping[str, PartSpec | SlotSpec],
    ) -> DriverSpec:
        entry = self.object(value, at)
        part = self.text(self.member(entry, "part", at), f"{at}.part")
        target = placed.get(part)
        if target is None:
            self.invalid(f"{at}.part", f"there is no part or slot {part!r}")
        kind = self.text(self.member(entry, "kind", at), f"{at}.kind")
        if kind not in DRIVER_KINDS:
            self.invalid(
                f"{at}.kind", f"{kind!r} is not one of {', '.join(DRIVER_KINDS)}"
            )
        unit = None
        if kind == "rotate":
            unit = self.text(self.member(entry, "unit", at), f"{at}.unit")
            if unit not in ANGLE_UNITS:
                self.invalid(
                    f"{at}.unit", f"a rotation's unit is {' or '.join(ANGLE_UNITS)}"
                )
        if kind in ("warp", "shape"):
            if not isinstance(target, PartSpec):
                self.invalid(
                    f"{at}.part", f"a {kind} drives a part, and {part!r} is a slot"
                )
            if kind == "warp" and target.lattice is None:
                self.invalid(
                    f"{at}.part",
                    f"a warp drives a part with a lattice; {part!r} has none",
                )
        keys_at = f"{at}.keys"
        found = self.array(self.member(entry, "keys", at), keys_at)
        if len(found) < 2:
            self.invalid(keys_at, "a driver has two keys or more")
        keys = tuple(
            self.key(k, f"{keys_at}[{i}]", kind, target) for i, k in enumerate(found)
        )
        if any(b.at <= a.at for a, b in zip(keys, keys[1:], strict=False)):
            self.invalid(keys_at, "their values of at must rise")
        if (keys[0].at, keys[-1].at) != span:
            self.invalid(
                keys_at, "the lowest key must be at min and the highest at max"
            )
        return DriverSpec(part=part, kind=kind, unit=unit, keys=keys)

    def key(self, value: Any, at: str, kind: str, target: PartSpec | SlotSpec) -> Key:
        entry = self.object(value, at)
        where = self.number(self.member(entry, "at", at), f"{at}.at")
        output: float | Point | tuple[Point, ...] | str
        if kind in ("rotate", "scale"):
            output = self.number(self.member(entry, "value", at), f"{at}.value")
        elif kind == "translate":
            output = self.point(self.member(entry, "value", at), f"{at}.value")
        elif kind == "warp":
            assert isinstance(target, PartSpec)
            assert target.lattice is not None
            offsets = self.array(self.member(entry, "offsets", at), f"{at}.offsets")
            if len(offsets) != target.lattice.points:
                self.invalid(
                    f"{at}.offsets",
                    f"the lattice has {target.lattice.points} points, and"
                    f" {len(offsets)} offsets are given",
                )
            output = tuple(self.point(o, f"{at}.offsets") for o in offsets)
        else:
            output = self.text(self.member(entry, "file", at), f"{at}.file")
        return Key(at=where, value=output)

    def shaped(self, parameters: Sequence[ParameterSpec]) -> Mapping[str, DriverSpec]:
        """The ``shape`` driver of each part that has one; at most one each."""
        shaped: dict[str, DriverSpec] = {}
        for i, parameter in enumerate(parameters):
            for j, driver in enumerate(parameter.drives):
                if driver.kind != "shape":
                    continue
                if driver.part in shaped:
                    self.invalid(
                        f"parameters[{i}].drives[{j}]",
                        f"the part {driver.part!r} has another shape driver",
                    )
                shaped[driver.part] = driver
        return shaped

    def anchor(
        self,
        value: Any,
        at: str,
        placed: Mapping[str, PartSpec | SlotSpec],
        shaped: Mapping[str, DriverSpec],
    ) -> AnchorSpec:
        entry = self.object(value, at)
        part = self.text(self.member(entry, "part", at), f"{at}.part")
        if part not in placed:
            self.invalid(f"{at}.part", f"there is no part or slot {part!r}")
        attach = self.text(self.member(entry, "attach", at), f"{at}.attach")
        if attach not in ATTACHMENTS:
            self.invalid(f"{at}.attach", f"it is {' or '.join(ATTACHMENTS)}")
        found = entry.get("keys")
        reshaped = shaped.get(part)
        keys: tuple[AnchorKey, ...] = ()
        if attach == "interior" and reshaped is not None:
            if found is None:
                self.invalid(
                    f"{at}.keys",
                    f"an interior anchor on the reshaped part {part!r} gives its"
                    " point at each key of the part's shape driver",
                )
            keys = tuple(
                AnchorKey(
                    at=self.number(
                        self.member(
                            self.object(k, f"{at}.keys[{i}]"), "at", f"{at}.keys[{i}]"
                        ),
                        f"{at}.keys[{i}].at",
                    ),
                    point=self.point(
                        self.member(k, "point", f"{at}.keys[{i}]"),
                        f"{at}.keys[{i}].point",
                    ),
                )
                for i, k in enumerate(self.array(found, f"{at}.keys"))
            )
            if [k.at for k in keys] != [k.at for k in reshaped.keys]:
                self.invalid(
                    f"{at}.keys",
                    "they must be at the keys of the part's shape driver, one each,"
                    " in order",
                )
        elif found is not None:
            self.invalid(
                f"{at}.keys",
                "only an interior anchor on a reshaped part gives keyed points",
            )
        return AnchorSpec(
            name=self.name(self.member(entry, "name", at), f"{at}.name"),
            part=part,
            attach=attach,
            at=self.point(self.member(entry, "at", at), f"{at}.at"),
            keys=keys,
        )


def _join(at: str, name: str) -> str:
    return f"{at}.{name}" if at else name
