"""Admission by geometry: an asset is drawn at the poses its maker tested,
and admitted or refused by its shape alone.

:func:`admit` draws a rig of the asset at its default pose; at the lowest,
middle and highest value of each parameter, the others at their defaults;
and with each drawing of each slot, one at a time, at the default pose. The
number of poses grows with the asset, never with combinations of its
parameters. Each pose is drawn by the engine's own Cairo camera into alpha
masks :data:`MASK_SIZE` pixels on their longer side, one of each part, one
of each slot (the drawing it shows) and one of the whole figure, every mask
of one asset on one grid, so that its pixels compare from pose to pose. A
pixel a shape touches at all is covered: transparency in a mask is the
absence of paint, never its opacity. Three checks are run over them:

- ``hole``: the figure's enclosed transparent pixels (those no path of
  transparent pixels joins to the mask's edge, stepping across sides) beyond
  as many as the default pose encloses, more than :data:`HOLE_ALLOWANCE` and
  the rim of the default pose's openings, which a moved opening can gain by
  where its edges fall between pixels;
  or more openings of more than :data:`HOLE_ALLOWANCE` pixels each than the
  default pose has, whatever their area, so a seam that opens while another
  opening narrows is found. A hole the asset is drawn with, such as a
  ring's, is not one it opens, and an opening that only moves is none;
- ``detached_part``: a part or slot whose mask, grown by :data:`REACH`
  pixels, no longer meets its parent's. A part drawn apart from its parent
  at the default pose (a label floating beside its body) meets nothing
  there, and is never found to leave it;
- ``anchor_off_part``: an anchor whose point lies outside its part's mask
  grown by :data:`REACH` pixels, at any pose, the default included.

A media asset (a still, a clip or a sprite) is admitted by its frames, not
its geometry, with two checks:

- ``frames_exact``: every frame file is in a form the engine decodes exactly
  (:data:`~arachne.mobject.media.frames.ADMITTED`, which loading holds too)
  and decodes, twice, to the frame digest its manifest lists;
- ``frames_uniform``: every frame is the manifest's ``media.pixels`` in
  size, and, where the manifest says the asset has no alpha, no pixel of it
  is less than opaque.

The tool writes nothing but, asked to, the record of what it found into the
asset's manifest (:func:`write_checks`): the tool version
:data:`~.reader.TOOL`, the asset's digest, and each check's result by name.
An asset loads only with that record complete and passing at its digest as
it is (:func:`~.reader.load_asset`), so changing one byte of it asks for its
admission again. From a shell::

    python -m arachne.assets admit <the asset's directory> [--write]

It judges shape and nothing else: neither style, nor identity, nor whether
a person also looks. What the asset reader refuses is refused before any
pose is drawn, as loading refuses it (:mod:`.svg_parts`): text, embedded
images, gradient and pattern paint, dashes, clip paths, masks and filters;
a stroked shape under a transform that is not a similarity; a stroked open
subpath whose ends meet, and a stroked subpath that begins where the one
before it ended (an editor's combine command writes these); a miter join
cut otherwise by the file's ``stroke-miterlimit`` than by the engine's 10
(state ``stroke-miterlimit="10"``, or use a round or bevel join); a
``miter-clip`` or ``arcs`` join. Two forms pass both the reader and this
tool and are drawn otherwise than SVG draws them, so a supplier's export
should avoid them: subpath ends that nearly meet, which the camera may join
once the asset is drawn small; and drawing on after ``Z`` with no move,
which the engine runs on from the closed outline where SVG starts afresh.
"""

from __future__ import annotations

__all__ = [
    "HOLE_ALLOWANCE",
    "MASK_MARGIN",
    "MASK_SIZE",
    "REACH",
    "AdmissionReport",
    "CheckResult",
    "admit",
    "write_checks",
]

import contextlib
import json
import math
import os
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..camera.camera import Camera
from ..constants import LineJointType
from ..mobject.types.vectorized_mobject import VMobject
from ..utils.color import BLACK, WHITE
from .anchor import attachments
from .joins import ENGINE_LIMIT
from .manifest import MANIFEST, MEDIA_KINDS, Media, PartSpec, normalised, read_json
from .reader import TOOL, Asset, checks_for, read_asset, read_file
from .rig import Rig, unadmitted_rig

#: The longer side of every mask, in pixels.
MASK_SIZE = 512
#: The clear pixels kept round the drawing on every side of a mask, so the
#: transparency outside the figure is one region along the mask's edge.
MASK_MARGIN = 8
#: How far, in mask pixels, a part is grown to meet its parent, and an
#: anchor may lie from its part: one pixel, at any side or corner.
REACH = 1
#: How many enclosed transparent pixels a pose may have beyond those of the
#: default pose: a pinhole where two outlines cross at a point, never a seam.
#: The engine's fixture assets enclose none beyond their default pose at any
#: pose; a seam a few pixels wide down one side of a part encloses thousands.
HOLE_ALLOWANCE = 4
#: The line width the camera strokes with, as a share of a shape's stroke
#: width, in scene units: how far a stroke reaches past its outline.
_LINE = 0.01


@dataclass(frozen=True)
class _Pose:
    """One pose the tool draws: its ``label``, as a result names it, and
    each parameter's value and each slot's drawing, by index.
    """

    label: str
    values: Mapping[int, float]
    choices: Mapping[int, str]


@dataclass(frozen=True)
class CheckResult:
    """One check's result: whether it ``passed``; ``worst``, the largest
    measure it took at any pose, in mask pixels (enclosed pixels beyond the
    default pose's; pixels between a part and its parent, for the parts that
    meet their parents at the default pose; pixels between an anchor and its
    part), and the ``pose`` it was taken at; ``failure``, what is wrong
    where it failed. A frame check's ``worst`` is how many frames fail it,
    and its ``pose`` the first failing frame's file.
    """

    name: str
    passed: bool
    worst: float
    pose: str | None
    failure: str | None = None

    def line(self) -> str:
        """The tool's line for the check: ``PASS <name>`` or ``FAIL <name>:
        <what is wrong>``.
        """
        if self.passed:
            return f"PASS {self.name}"
        return f"FAIL {self.name}: {self.failure}"


@dataclass(frozen=True)
class AdmissionReport:
    """What :func:`admit` found of an asset: its name, version, kind and
    digest, and one :class:`CheckResult` per check of its kind's check set
    (:func:`~.reader.checks_for`), in that order.
    """

    name: str
    version: int
    kind: str
    digest: str
    results: tuple[CheckResult, ...]

    @property
    def passed(self) -> bool:
        """Whether every check passed."""
        return all(result.passed for result in self.results)

    def result(self, name: str) -> CheckResult:
        """The result of the check ``name``."""
        return next(result for result in self.results if result.name == name)

    def checks(self) -> dict[str, Any]:
        """The manifest's ``checks`` record of this admission."""
        return {
            "tool": TOOL,
            "digest": self.digest,
            "results": [
                {"name": result.name, "pass": result.passed} for result in self.results
            ],
        }

    def lines(self) -> list[str]:
        """The tool's output: the asset, one line per check by name, and
        the count of checks passed.
        """
        passed = sum(result.passed for result in self.results)
        return [
            f"asset: {self.name} {self.version} ({self.kind}) {self.digest}",
            *(result.line() for result in sorted(self.results, key=_by_name)),
            f"admission: {passed}/{len(self.results)} pass",
        ]


def _by_name(result: CheckResult) -> str:
    return result.name


def admit(asset: Asset | str | os.PathLike[str]) -> AdmissionReport:
    """Draw ``asset`` (an :class:`~.reader.Asset`, or a reference
    :func:`~.reader.load_asset` takes, read without asking for its
    admission) at every pose the module names, and run the three checks; or
    decode the frames of a media asset and run its two. What makes the
    asset unreadable is raised as loading raises it. The poses are drawn one
    at a time: only the default pose's masks and the current pose's are
    held.
    """
    if not isinstance(asset, Asset):
        asset = read_asset(asset)
    manifest = asset.manifest
    if manifest.kind in MEDIA_KINDS:
        assert manifest.media is not None
        results = _frame_checks(asset, manifest.media)
        assert tuple(result.name for result in results) == checks_for(manifest)
        return AdmissionReport(
            manifest.name, manifest.version, manifest.kind, asset.digest, results
        )
    rig = unadmitted_rig(asset)
    poses = list(_poses(rig))
    grid = _Grid(rig, poses)
    default = grid.masks(rig, poses[0])
    checks = (_Holes(default), _Detached(rig, default), _Anchors(rig, grid))
    for pose in poses:
        masks = default if pose is poses[0] else grid.masks(rig, pose)
        for check in checks:
            check.see(pose, masks)
    results = tuple(check.result() for check in checks)
    assert tuple(result.name for result in results) == checks_for(manifest)
    return AdmissionReport(
        manifest.name, manifest.version, manifest.kind, asset.digest, results
    )


def write_checks(directory: str | os.PathLike[str], report: AdmissionReport) -> Path:
    """Write ``report``'s record as the ``checks`` of the manifest in
    ``directory``, its other members as they were, and give the manifest's
    path. The record attests the digest the report was made at, so an
    asset changed since loads as unadmitted. The manifest is written whole
    beside the old one, in UTF-8, and replaces it only once written: a
    write that fails leaves the old manifest as it was.
    """
    path = Path(directory) / MANIFEST
    document = read_json(path.read_bytes())
    document = {key: value for key, value in document.items() if key != "checks"}
    ordered: dict[str, Any] = {}
    for key, value in document.items():
        if key == "provenance":
            ordered["checks"] = report.checks()
        ordered[key] = value
    ordered.setdefault("checks", report.checks())
    text = json.dumps(ordered, indent=2, ensure_ascii=False) + "\n"
    handle, temporary = tempfile.mkstemp(
        prefix=f".{MANIFEST}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as written:
            written.write(text)
            written.flush()
            os.fsync(written.fileno())
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise
    return path


# Poses


def _poses(rig: Rig) -> Iterator[_Pose]:
    """The default pose, then each parameter at its lowest, middle and
    highest tested value, then each slot's drawings one at a time.
    """
    manifest = rig.asset.manifest
    defaults = {i: float(p.default) for i, p in enumerate(manifest.parameters)}
    shown = {i: slot.default for i, slot in enumerate(manifest.slots)}
    yield _Pose("the default pose", defaults, shown)
    for i, parameter in enumerate(manifest.parameters):
        low, high = float(parameter.min), float(parameter.max)
        for value in (low, (low + high) / 2, high):
            yield _Pose(f"{parameter.name}={value:g}", defaults | {i: value}, shown)
    for i, slot in enumerate(manifest.slots):
        for drawing in slot.drawings:
            if drawing.name != slot.default:
                yield _Pose(
                    f"{slot.name}={drawing.name}", defaults, shown | {i: drawing.name}
                )


def _posed(rig: Rig, pose: _Pose) -> Rig:
    rig.write_values(dict(pose.values))
    rig.write_choices(dict(pose.choices))
    return rig.settle()


def _members(rig: Rig) -> list[str]:
    """Every part and slot of ``rig``, in its manifest's order."""
    manifest = rig.asset.manifest
    return [p.name for p in manifest.parts] + [s.name for s in manifest.slots]


def _drawn(group: VMobject) -> list[VMobject]:
    """The shapes of ``group`` that draw now, its rig's carrier left out."""
    return [
        shape
        for shape in group.family_members_with_points()
        if isinstance(shape, VMobject) and not getattr(shape, "_scaffold", False)
    ]


# Masks


@dataclass
class _Masks:
    """One pose's masks: each part's and slot's, by name, and the figure's,
    each ``True`` where paint touches a pixel.
    """

    members: dict[str, np.ndarray]
    figure: np.ndarray


class _Grid:
    """The pixel grid every pose of one rig is drawn on: the box of the
    art at all of ``poses``, as far as its strokes can reach past its points
    (:func:`_stroke_reach`), fitted with a margin of :data:`MASK_MARGIN`
    into :data:`MASK_SIZE` pixels on its longer side.
    """

    def __init__(self, rig: Rig, poses: Sequence[_Pose]) -> None:
        low, high = np.full(2, math.inf), np.full(2, -math.inf)
        for pose in poses:
            shapes = _drawn(_posed(rig, pose))
            for shape in shapes:
                points = np.asarray(shape.points)[:, :2]
                reach = _stroke_reach(shape)
                low = np.minimum(low, points.min(axis=0) - reach)
                high = np.maximum(high, points.max(axis=0) + reach)
        extent = np.maximum(high - low, 1e-9)
        scale = (MASK_SIZE - 2 * MASK_MARGIN) / float(extent.max())
        width, height = (
            int(min(MASK_SIZE, math.ceil(e * scale) + 2 * MASK_MARGIN)) for e in extent
        )
        centre = (low + high) / 2
        self.camera = Camera(
            pixel_width=width,
            pixel_height=height,
            frame_width=width / scale,
            frame_height=height / scale,
            frame_center=np.array([centre[0], centre[1], 0.0]),
            background_color=BLACK,
            background_opacity=0.0,
        )

    def draw(self, shapes: Sequence[VMobject]) -> np.ndarray:
        """The pixels ``shapes`` paint, all opaque, drawn by the camera."""
        camera = self.camera
        camera.reset()
        camera.capture_mobjects([_opaque(shape) for shape in shapes])
        return np.array(camera.pixel_array[:, :, 3] > 0)

    def masks(self, rig: Rig, pose: _Pose) -> _Masks:
        """Every mask of ``rig`` at ``pose``."""
        _posed(rig, pose)
        members = {name: self.draw(_drawn(rig.part(name))) for name in _members(rig)}
        return _Masks(members, self.draw(_drawn(rig)))

    def pixel(self, point: np.ndarray) -> tuple[int, int]:
        """The column and row of the pixel the scene point ``point`` is in."""
        x, y = self.camera.points_to_subpixel_coords(None, np.array([point]))[0]  # type: ignore[arg-type]
        return math.floor(x), math.floor(y)


def _stroke_reach(shape: VMobject) -> float:
    """How far, in scene units, ``shape``'s stroke can reach past its
    points, for every join and cap the asset reader admits: a miter join up
    to the engine's miter limit, :data:`~.joins.ENGINE_LIMIT` half-widths,
    past its corner; any other join, and a square cap at its corner, up to
    the square root of 2 half-widths.
    """
    half = _LINE * float(shape.get_stroke_width()) / 2
    if shape.joint_type in (LineJointType.ROUND, LineJointType.BEVEL):
        return half * math.sqrt(2)
    return half * max(ENGINE_LIMIT, math.sqrt(2))


def _opaque(shape: VMobject) -> VMobject:
    """A copy of ``shape`` painted opaque wherever it is painted at all."""
    copy = shape.copy()
    fill = bool(np.any(np.asarray(shape.get_fill_opacities()) > 0))
    stroke = bool(np.any(np.asarray(shape.get_stroke_opacities()) > 0))
    copy.set_fill(WHITE, opacity=1.0 if fill else 0.0)
    copy.set_stroke(WHITE, opacity=1.0 if stroke else 0.0)
    copy.set_stroke(WHITE, width=0, background=True)
    return copy


# The checks


@dataclass(frozen=True)
class _Openings:
    """The transparent pixels of a figure's mask that no path of
    transparent pixels, stepping across sides, joins to the mask's edge:
    how many there are (``area``), how many of them lie on an opening's rim,
    next across a side to paint (``rim``), and how many openings of more
    than :data:`HOLE_ALLOWANCE` pixels they make (``count``).
    """

    area: int
    rim: int
    count: int


def _openings(figure: np.ndarray) -> _Openings:
    from scipy import ndimage

    labels, _ = ndimage.label(~figure)
    edge = np.unique(
        np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    )
    sizes = np.bincount(labels.ravel())
    sizes[edge] = 0
    sizes[0] = 0
    enclosed = sizes[labels] > 0
    inner = ndimage.binary_erosion(enclosed)
    return _Openings(
        int(sizes.sum()),
        int(np.count_nonzero(enclosed & ~inner)),
        int(np.count_nonzero(sizes > HOLE_ALLOWANCE)),
    )


class _Holes:
    """``hole``: more enclosed pixels than the default pose's, beyond
    :data:`HOLE_ALLOWANCE` and the rim of the default pose's openings (an
    opening drawn at another place on the grid gains or loses up to a pixel
    all round, by where its edges fall between pixels); or more openings of
    more than :data:`HOLE_ALLOWANCE` pixels than the default pose has,
    whatever the area, so that a seam opening while another opening narrows
    is still found. An opening that only moves passes.
    """

    def __init__(self, default: _Masks) -> None:
        self.default = _openings(default.figure)
        self.allowed = HOLE_ALLOWANCE + self.default.rim
        self.worst = 0
        self.where: str | None = None
        self.failure: str | None = None

    def see(self, pose: _Pose, masks: _Masks) -> None:
        found = _openings(masks.figure)
        beyond = found.area - self.default.area
        if beyond > self.worst:
            self.worst, self.where = beyond, pose.label
            if beyond > self.allowed:
                self.failure = (
                    f"{beyond} px open inside the figure at {pose.label}"
                    f" (allowed {self.allowed})"
                )
        if found.count > self.default.count and self.failure is None:
            self.where = pose.label
            self.failure = (
                f"{found.count} openings of more than {HOLE_ALLOWANCE} px inside"
                f" the figure at {pose.label}, where the default pose has"
                f" {self.default.count}"
            )

    def result(self) -> CheckResult:
        return CheckResult(
            "hole", self.failure is None, self.worst, self.where, self.failure
        )


def _apart(mask: np.ndarray, other: np.ndarray) -> float:
    """How many pixels apart, stepping across sides or corners, the
    nearest pixels of ``mask`` and ``other`` are: 0 where they overlap, 1
    where they touch; infinite where one is empty.
    """
    if not mask.any() or not other.any():
        return math.inf
    from scipy import ndimage

    distance = ndimage.distance_transform_cdt(~other, metric="chessboard")
    return float(distance[mask].min())


def _parents(rig: Rig) -> list[tuple[str, str]]:
    """Every part and slot with a parent, and its parent."""
    placed = rig.model.placed
    return [
        (name, parent)
        for name in _members(rig)
        if (parent := placed[name].parent) is not None
    ]


class _Detached:
    """``detached_part``: a part or slot that meets its parent at the
    default pose, more than :data:`REACH` pixels from it at a pose.
    """

    def __init__(self, rig: Rig, default: _Masks) -> None:
        self.joined = [
            (name, parent)
            for name, parent in _parents(rig)
            if _apart(default.members[name], default.members[parent]) <= REACH
        ]
        self.worst: float = 0.0
        self.where: str | None = None
        self.failure: str | None = None

    def see(self, pose: _Pose, masks: _Masks) -> None:
        for name, parent in self.joined:
            gap = _apart(masks.members[name], masks.members[parent])
            if gap > self.worst:
                self.worst, self.where = gap, pose.label
                if gap > REACH:
                    apart = (
                        "one of them draws nothing"
                        if math.isinf(gap)
                        else f"{gap:g} px apart"
                    )
                    self.failure = (
                        f"{name} no longer meets its parent {parent} at"
                        f" {pose.label}: {apart} (allowed {REACH})"
                    )

    def result(self) -> CheckResult:
        return CheckResult(
            "detached_part", self.failure is None, self.worst, self.where, self.failure
        )


class _Anchors:
    """``anchor_off_part``: an anchor more than :data:`REACH` pixels from
    its part's mask at a pose. It reads the rig as the pose's masks left it.
    """

    def __init__(self, rig: Rig, grid: _Grid) -> None:
        self.rig, self.grid = rig, grid
        self.anchors = attachments(rig.model)
        self.worst: float = 0.0
        self.where: str | None = None
        self.failure: str | None = None

    def see(self, pose: _Pose, masks: _Masks) -> None:
        rig = self.rig
        _posed(rig, pose)
        for spec in rig.asset.manifest.anchors:
            point = self.anchors[spec.name].point(rig)
            assert point is not None
            covered = np.argwhere(masks.members[spec.part])
            if not len(covered):
                off = math.inf
            else:
                column, row = self.grid.pixel(point)
                off = float(np.abs(covered - np.array([row, column])).max(axis=1).min())
            if off > self.worst:
                self.worst, self.where = off, pose.label
                if off > REACH:
                    kind = (
                        "part"
                        if isinstance(rig.model.placed[spec.part], PartSpec)
                        else "slot"
                    )
                    self.failure = (
                        f"{spec.name} lies {off:g} px off its {kind} {spec.part}"
                        f" at {pose.label} (allowed {REACH})"
                    )

    def result(self) -> CheckResult:
        return CheckResult(
            "anchor_off_part",
            self.failure is None,
            self.worst,
            self.where,
            self.failure,
        )


# The frame checks


def _frame_checks(asset: Asset, media: Media) -> tuple[CheckResult, ...]:
    """``frames_exact`` and ``frames_uniform`` over every frame of the media
    asset ``asset``, each frame file decoded twice. A check's ``worst`` is
    how many frames fail it, and its ``pose`` the first failing frame's file.
    """
    from ..mobject.media.frames import EXPECTED_FORM, decode, form_of, frame_digest

    inexact: list[tuple[str, str]] = []
    unlike: list[tuple[str, str]] = []
    width, height = media.pixels
    for frame in media.frames:
        data = read_file(asset.directory, normalised(frame.file), asset.name)
        found = form_of(data)
        if isinstance(found, str):
            inexact.append(
                (frame.file, f"{frame.file} is {found}, and not {EXPECTED_FORM}")
            )
            continue
        try:
            pixels, again = decode(data), decode(data)
        except ValueError as error:
            inexact.append((frame.file, f"{frame.file} is {error}"))
            continue
        digests = {frame_digest(pixels), frame_digest(again)}
        if digests != {frame.digest}:
            inexact.append(
                (
                    frame.file,
                    f"{frame.file} decodes to {' and '.join(sorted(digests))}, and"
                    f" its listed digest is {frame.digest}",
                )
            )
        rows, columns = pixels.shape[:2]
        if (columns, rows) != (width, height):
            unlike.append(
                (
                    frame.file,
                    f"{frame.file} is {columns} x {rows} pixels, and the manifest"
                    f" says {width} x {height}",
                )
            )
        elif not media.alpha and int(pixels[:, :, 3].min()) < 255:
            unlike.append(
                (
                    frame.file,
                    f"{frame.file} has a pixel of alpha {int(pixels[:, :, 3].min())},"
                    " and the manifest says the asset has no alpha",
                )
            )
    return (
        _frames_result("frames_exact", inexact, len(media.frames)),
        _frames_result("frames_uniform", unlike, len(media.frames)),
    )


def _frames_result(
    name: str, failing: list[tuple[str, str]], frames: int
) -> CheckResult:
    if not failing:
        return CheckResult(name, True, 0, None)
    first_file, first = failing[0]
    more = f" ({len(failing)} of {frames} frames fail)" if len(failing) > 1 else ""
    return CheckResult(name, False, len(failing), first_file, first + more)
