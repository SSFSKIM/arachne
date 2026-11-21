"""Loading an asset: its manifest, its files and their shapes, read whole
before anything is drawn.

:func:`load_asset` takes a reference to an asset: a path to its directory
or to its ``manifest.json``, resolved as the engine's image and SVG
constructors resolve a file name (as given, then under the configuration's
``assets_dir``), or ``fixture:<name>`` for one of the engine's own fixture
assets. Nothing else is searched and nothing is fetched: an address, or a
file the manifest names by anything but a plain relative path inside the
asset's directory (an address, an absolute path, a ``..``, a link), and a
manifest that is a link, is ``asset.remote``. Every way an asset can be
unusable is a finding raised with its code: no manifest
(``asset.manifest_missing``), a manifest that is not valid
(``asset.manifest_invalid``, an anchor placed off its part's outline and a
name a pose could not tell apart included), a file not there
(``asset.file_missing``), one that cannot be read or drawn exactly
(``asset.unreadable``), and, once all of that holds, an asset not admitted
(``asset.unadmitted``): it loads only when its manifest's ``checks`` record
the tool version :data:`TOOL`, the digest of its files and manifest as they
are, and every check of the asset's check set by name, each passing
(:func:`checks_for`; :mod:`.admission` writes that record). Nothing is
cached between loads.

A media asset (a still, a clip or a sprite) is frames of pixels, each a
file in a form the engine decodes exactly
(:data:`~arachne.mobject.media.frames.ADMITTED`). Loading reads every file's
bytes, for the asset's digest, and each frame file's header, for its form: a
file in no admitted form is ``media.inexact`` (reason ``format``), before
admission is asked. It decodes no frame: each frame is decoded, and checked
against its digest, when it is first drawn (:mod:`arachne.mobject.media.frames`).

An asset of the kind ``parts`` may hold raster members (its manifest
requires ``raster``): a part or a slot's drawing that is one still, a PNG
file in an admitted form, read as a media frame is. Loading checks each
such file's form from its header and decodes none; the member stands in the
rig as the quad of its ``box`` (:class:`~.raster.RasterPart`), whose
picture is decoded, and checked against the member's digest, when it is
first drawn. Such an asset holds the three checks of its kind and
``frames_exact`` over its raster files (:data:`RASTER_CHECKS`).

A ``shape`` driver's stored outlines are the rest outline moved: a supplier
keeps, shape for shape, the contours in the same order, each with as many
points, open or closed alike, and each point the same landmark of the
drawing, since the rig interpolates point to point and an anchor on the
outline is a place along one contour.
"""

from __future__ import annotations

__all__ = [
    "CHECKS",
    "FIXTURES",
    "MEDIA_CHECKS",
    "RASTER_CHECKS",
    "TOOL",
    "Asset",
    "checks_for",
    "load_asset",
    "read_file",
]

import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any, NoReturn
from urllib.parse import urlsplit

from .. import config, quantities
from ..mobject.types.vectorized_mobject import VMobject
from .manifest import (
    MANIFEST,
    MEDIA_KINDS,
    AnchorSpec,
    Manifest,
    ParameterSpec,
    PartSpec,
    SlotSpec,
    digest,
    normalised,
    parse_manifest,
    read_json,
    refuse,
    thawed,
)
from .svg_parts import SvgFile, Unreadable, read_svg, structure

#: The admission tool's version this engine supports: an asset loads only
#: with the passing checks of this version recorded at its digest. It moves
#: when a check comes to measure otherwise, so a record of the old checker
#: admits nothing: 2 draws the masks on a grid of painted geometry alone.
TOOL = "arachne.assets 2"
#: The versions before :data:`TOOL`, whose records are refused with the
#: command that admits the asset again.
_OLDER = ("arachne.assets 1",)
#: The checks of :data:`TOOL` an asset of the kind ``parts`` holds, in the
#: order its record lists them.
CHECKS = ("hole", "detached_part", "anchor_off_part")
#: The checks of :data:`TOOL` an asset of a media kind holds, in the order
#: its record lists them.
MEDIA_CHECKS = ("frames_exact", "frames_uniform")
#: The checks of :data:`TOOL` an asset of the kind ``parts`` that requires
#: ``raster`` holds: the three of its kind, measured as they are for one
#: without raster members, and ``frames_exact`` over its raster files.
RASTER_CHECKS = (*CHECKS, "frames_exact")
#: The engine's own fixture assets, each a directory named as it is asked
#: for: ``fixture:<name>``.
FIXTURES = Path(__file__).resolve().parent / "fixtures"
_FIXTURE = "fixture:"
#: Opens a file's bytes as they are, without waiting for a writer, where
#: the system has these flags.
_OPEN = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)


@dataclass(frozen=True, eq=False)
class Asset:
    """An asset, loaded: its manifest, its digest, and the shapes of its
    parts, drawings and stored outlines at rest, in canvas coordinates (a
    media asset has none: its frames are read when drawn).

    It is a value: nothing of it can be written once loaded (its fields are
    frozen and its manifest deep-immutable), and a copy of it is itself. The
    shapes it gives are fresh copies.
    """

    directory: Path
    manifest: Manifest
    digest: str
    #: The part files' shared ``viewBox``, ``None`` when they have none.
    canvas: tuple[float, ...] | None
    _shapes: Mapping[tuple[str, ...], tuple[VMobject, ...]]

    @property
    def name(self) -> str:
        """The asset's name."""
        return self.manifest.name

    @property
    def kind(self) -> str:
        """The asset's kind: ``parts``, or a media kind."""
        return self.manifest.kind

    @property
    def version(self) -> int:
        """The asset's version."""
        return self.manifest.version

    @property
    def parts(self) -> Mapping[str, PartSpec]:
        """Its parts, by name, in the manifest's order."""
        return MappingProxyType({part.name: part for part in self.manifest.parts})

    @property
    def parameters(self) -> Mapping[str, ParameterSpec]:
        """Its parameters, by name, in the manifest's order."""
        return MappingProxyType({p.name: p for p in self.manifest.parameters})

    @property
    def slots(self) -> Mapping[str, SlotSpec]:
        """Its slots, by name, in the manifest's order."""
        return MappingProxyType({slot.name: slot for slot in self.manifest.slots})

    @property
    def anchors(self) -> Mapping[str, AnchorSpec]:
        """Its anchors, by name, in the manifest's order."""
        return MappingProxyType({a.name: a for a in self.manifest.anchors})

    def rest(self, part: str) -> tuple[VMobject, ...]:
        """The shapes of the part ``part`` as its file draws them."""
        return _copies(self._shapes[("part", part)])

    def drawing(self, slot: str, drawing: str) -> tuple[VMobject, ...]:
        """The shapes of the slot ``slot``'s drawing ``drawing``."""
        return _copies(self._shapes[("drawing", slot, drawing)])

    def outline(self, file: str) -> tuple[VMobject, ...]:
        """The shapes of a stored outline's file, as the manifest names it."""
        return _copies(self._shapes[("outline", normalised(file))])

    def __copy__(self) -> Asset:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> Asset:
        return self

    def __repr__(self) -> str:
        return f"Asset({self.name!r}, version={self.version}, {self.digest})"


def load_asset(ref: str | os.PathLike[str] | Asset) -> Asset:
    """The asset ``ref`` names (see the module), checked and read whole, and
    admitted; raises the finding of the first thing that makes it unusable.
    An :class:`Asset` given is admitted as one read from a path is, and
    returned as it is.
    """
    if isinstance(ref, Asset):
        _admitted(ref, str(ref.directory))
        return _units_registered(ref)
    asset = read_asset(ref)
    _admitted(asset, os.fspath(ref))
    return _units_registered(asset)


def _units_registered(asset: Asset) -> Asset:
    """``asset``, its parameters' units registered as units of declared
    quantities (:mod:`arachne.quantities`), so a rig's values read in them.
    """
    for parameter in asset.manifest.parameters:
        quantities.register_unit(parameter.unit)
    return asset


def checks_for(manifest: Manifest) -> tuple[str, ...]:
    """The checks of :data:`TOOL` an asset whose manifest is ``manifest``
    must hold, by its kind and its ``requires``: :data:`CHECKS` for the
    kind ``parts``, :data:`RASTER_CHECKS` for one that requires ``raster``,
    :data:`MEDIA_CHECKS` for a media kind.
    """
    if manifest.kind in MEDIA_KINDS:
        return MEDIA_CHECKS
    return RASTER_CHECKS if "raster" in manifest.requires else CHECKS


def read_asset(ref: str | os.PathLike[str]) -> Asset:
    """The asset ``ref`` names, checked and read whole as :func:`load_asset`
    reads it, its admission left unasked: the admission tool's own entry
    (:mod:`.admission`), which admits what it reads. An asset it gives loads,
    and makes a rig, only once admitted.
    """
    label = os.fspath(ref)
    directory = _directory(label)
    _unlinked_manifest(directory, label)
    document = _document(_bytes(directory, MANIFEST, label), label)
    manifest = parse_manifest(document, label)
    contents: dict[str, bytes] = {}
    for file in manifest.files():
        _local(file, directory, label)
    for file in manifest.files():
        key = normalised(file)
        if not (directory / key).exists():
            refuse(
                "asset.file_missing",
                f"the asset {label} names the file {file}, which is not in its"
                " directory",
                asset=label,
                file=file,
            )
        contents[key] = _bytes(directory, key, label)
    if manifest.kind in MEDIA_KINDS:
        from ..mobject.media.frames import refuse_unadmitted_form

        for file, data in contents.items():
            refuse_unadmitted_form(data, label, file)
        return Asset(
            directory,
            manifest,
            digest(manifest.document, contents),
            None,
            MappingProxyType({}),
        )
    from ..mobject.media.frames import refuse_unadmitted_form

    asset_digest = digest(manifest.document, contents)
    raster = raster_files(manifest)
    for file in raster:
        refuse_unadmitted_form(contents[file], label, file)
    files = {
        file: _read(contents[file], file, label)
        for file in contents
        if file not in raster
    }
    canvas = _canvas(files, label) if files else None
    quads = _Quads(directory, manifest.name, asset_digest)
    shapes = _shapes(manifest, files, label, quads)
    asset = Asset(directory, manifest, asset_digest, canvas, MappingProxyType(shapes))
    # What needs the asset's geometry: its names as a pose reads them, and
    # every anchor resolved on its part, so that an asset loaded is whole.
    from .anchor import attachments
    from .pose import model_of

    attachments(model_of(asset))
    return asset


def raster_files(manifest: Manifest) -> tuple[str, ...]:
    """The files of the raster members of ``manifest`` (an asset of the
    kind ``parts``), each once, as :func:`~.manifest.normalised` gives them.
    """
    found = [normalised(part.file) for part in manifest.parts if part.raster]
    found += [
        normalised(drawing.file)
        for slot in manifest.slots
        for drawing in slot.drawings
        if drawing.raster
    ]
    return tuple(dict.fromkeys(found))


@dataclass(frozen=True)
class _Quads:
    """What makes a raster member's quad: the asset's ``directory``, its
    ``name`` and its ``digest``.
    """

    directory: Path
    name: str
    digest: str

    def quad(
        self, file: str, box: tuple[float, float, float, float], frame: str
    ) -> tuple[VMobject, ...]:
        from .raster import RasterPart

        return (
            RasterPart(
                box,
                asset_digest=self.digest,
                file=file,
                digest=frame,
                asset=self.name,
                directory=self.directory,
            ),
        )


def _admitted(asset: Asset, label: str) -> None:
    """Refuse ``asset`` (asked for as ``label``) unless its manifest records
    the passing checks of :data:`TOOL` at its digest, every one of its check
    set (:func:`checks_for`): ``asset.unadmitted``, whose ``reason`` says
    what is not so.
    """
    checks = asset.manifest.checks
    if checks is None:
        _unadmitted(
            label,
            f"the asset {label} carries no admission record",
            "unchecked",
        )
    tool = checks.get("tool")
    if tool in _OLDER:
        _unadmitted(
            label,
            f"the asset {label} was admitted by {tool!r}, an older checker than"
            f" {TOOL!r}, the admission this engine supports, whose checks measure"
            " otherwise: admit it again with python -m arachne.assets admit"
            f" {asset.directory} --write",
            "tool",
            expected=TOOL,
            found=tool,
        )
    if tool != TOOL:
        _unadmitted(
            label,
            f"the asset {label} was admitted by {tool!r}, not by {TOOL!r}, the"
            " admission this engine supports",
            "tool",
            expected=TOOL,
            found=thawed(tool),
        )
    recorded = checks.get("digest")
    if recorded != asset.digest:
        _unadmitted(
            label,
            f"the asset {label} has changed since it was admitted: its digest is"
            f" {asset.digest}, and its checks were recorded at {recorded}",
            "digest",
            expected=asset.digest,
            found=thawed(recorded),
        )
    results = checks.get("results")
    held = checks_for(asset.manifest)
    passing: dict[str, bool] = {}
    for entry in results if isinstance(results, (list, tuple)) else ():
        if isinstance(entry, Mapping) and entry.get("name") in held:
            name = entry["name"]
            passing[name] = passing.get(name, True) and entry.get("pass") is True
    missing = [name for name in held if name not in passing]
    if missing:
        _unadmitted(
            label,
            f"the admission record of the asset {label} lacks the check"
            f" {', '.join(missing)} of {TOOL}",
            "missing",
            checks=missing,
        )
    failing = [name for name in held if not passing[name]]
    if failing:
        _unadmitted(
            label,
            f"the asset {label} fails the admission check {', '.join(failing)}",
            "failing",
            checks=failing,
        )


def _unadmitted(label: str, what: str, reason: str, **detail: Any) -> NoReturn:
    refuse("asset.unadmitted", what, asset=label, reason=reason, **detail)


def read_file(directory: Path, file: str, label: str) -> bytes:
    """The bytes of the file ``file`` (a path as :func:`~.manifest.normalised`
    gives it) of the asset in ``directory``, asked for as ``label``, read as
    loading reads them: ``asset.unreadable`` for one that cannot be read,
    ``asset.file_missing`` for one that is gone, and, as at loading,
    ``asset.remote`` for one now reached through a link: the file read after
    loading is the one loading read.
    """
    _local(file, directory, label)
    if not (directory / file).exists():
        refuse(
            "asset.file_missing",
            f"the asset {label} names the file {file}, which is not in its directory",
            asset=label,
            file=file,
        )
    return _bytes(directory, file, label)


def _bytes(directory: Path, file: str, label: str) -> bytes:
    """The bytes of the asset's file ``file``; one that cannot be read (not
    a regular file, one the process may not read) is ``asset.unreadable``.
    It is opened without waiting, so a pipe or a device never blocks the
    read, and checked to be a regular file before a byte is read.
    """
    try:
        handle = os.open(directory / file, _OPEN)
    except OSError as error:
        _unreadable(label, file, f"it cannot be read ({error.strerror or error})")
    try:
        if not stat.S_ISREG(os.fstat(handle).st_mode):
            _unreadable(label, file, "it cannot be read: it is not a regular file")
        with os.fdopen(handle, "rb", closefd=False) as opened:
            return opened.read()
    except OSError as error:
        _unreadable(label, file, f"it cannot be read ({error.strerror or error})")
    finally:
        os.close(handle)


def _copies(shapes: tuple[VMobject, ...]) -> tuple[VMobject, ...]:
    return tuple(shape.copy() for shape in shapes)


def _directory(label: str) -> Path:
    """The asset directory ``label`` names, which holds a manifest."""
    if label.startswith(_FIXTURE):
        name = label[len(_FIXTURE) :]
        directory = FIXTURES / name
        if "/" in name or "\\" in name or not (directory / MANIFEST).is_file():
            known = sorted(p.parent.name for p in FIXTURES.glob(f"*/{MANIFEST}"))
            refuse(
                "asset.manifest_missing",
                f"the engine has no fixture asset {name!r}; it has {', '.join(known)}",
                asset=label,
                searched=[label],
            )
        return directory
    if _addressed(label):
        refuse(
            "asset.remote",
            f"the asset {label} is named by an address; assets are read from"
            " local directories only",
            asset=label,
            file=label,
        )
    given = Path(label).expanduser()
    candidates = [given, Path(config.get_dir("assets_dir")) / given]
    for candidate in candidates:
        if candidate.is_dir():
            if (candidate / MANIFEST).is_file():
                return candidate
            refuse(
                "asset.manifest_missing",
                f"the directory {candidate} holds no {MANIFEST}",
                asset=label,
                searched=[str(candidate / MANIFEST)],
            )
        if candidate.is_file():
            if candidate.name == MANIFEST:
                return candidate.parent
            refuse(
                "asset.manifest_missing",
                f"{candidate} is a file, not an asset: an asset is a directory"
                f" holding {MANIFEST} beside its files",
                asset=label,
                searched=[str(candidate)],
            )
    refuse(
        "asset.manifest_missing",
        f"no asset {label} was found",
        asset=label,
        searched=[str(candidate) for candidate in candidates],
    )


def _addressed(reference: str) -> bool:
    """Whether ``reference`` is an address with a scheme (a drive letter is
    no scheme).
    """
    return len(urlsplit(reference).scheme) > 1


def _local(file: str, directory: Path, label: str) -> None:
    """Refuse the file ``file`` a manifest names unless it is a plain path in
    the asset's directory ``directory``: relative, with ``/`` alone between
    names (a backslash separates on one system and not on another), with no
    ``..`` and through no link, so that the path the manifest writes, the one opened and the
    one the digest names are one file.
    """
    parts = PurePosixPath(file).parts
    outside = (
        _addressed(file)
        or PurePosixPath(file).is_absolute()
        or PureWindowsPath(file).is_absolute()
        or PureWindowsPath(file).drive != ""
        or ".." in parts
        or "\\" in file
    )
    if not outside:
        place = directory
        for part in PurePosixPath(normalised(file)).parts:
            place = place / part
            if place.is_symlink():
                outside = True
                break
    if outside:
        refuse(
            "asset.remote",
            f"the asset {label} names the file {file}, which is not a plain path"
            " inside its directory (relative, with / alone between names, with no .."
            " and through no link)",
            asset=label,
            file=file,
        )


def _unlinked_manifest(directory: Path, label: str) -> None:
    """Refuse the asset in ``directory`` when its manifest is a link, as
    :func:`_local` refuses a part file reached through one: the manifest
    read, the one beside the files and the one the digest names are one.
    """
    if (directory / MANIFEST).is_symlink():
        refuse(
            "asset.remote",
            f"the asset {label} has its {MANIFEST} through a link; an asset's"
            " manifest, like its files, is a plain file inside its directory",
            asset=label,
            file=MANIFEST,
        )


def _document(data: bytes, label: str) -> Any:
    """The manifest's JSON, in the domain :func:`~.manifest.read_json`
    reads; anything else is ``asset.manifest_invalid``.
    """
    try:
        return read_json(data)
    except ValueError as error:
        reason = f"it is not valid JSON in the manifest's domain ({error})"
        refuse(
            "asset.manifest_invalid",
            f"the manifest of the asset {label} cannot be read: {reason}",
            asset=label,
            member=MANIFEST,
            reason=reason,
        )


def _read(data: bytes, file: str, label: str) -> SvgFile:
    try:
        return read_svg(data)
    except Unreadable as error:
        _unreadable(label, file, error.reason)


def _unreadable(label: str, file: str, reason: str) -> NoReturn:
    refuse(
        "asset.unreadable",
        f"the file {file} of the asset {label} cannot be drawn: {reason}",
        asset=label,
        file=file,
        reason=reason,
    )


def _canvas(files: Mapping[str, SvgFile], label: str) -> tuple[float, ...] | None:
    """The canvas every part file shares, by its ``viewBox``."""
    boxes = {file: svg.view_box for file, svg in files.items()}
    canvas = next(iter(boxes.values()))
    for file, box in boxes.items():
        if box != canvas:
            reason = (
                f"the part files do not share one canvas: {file} has the viewBox"
                f" {box} and another {canvas}"
            )
            refuse(
                "asset.manifest_invalid",
                f"the asset {label} is invalid: {reason}",
                asset=label,
                member="parts",
                reason=reason,
            )
    return canvas


def _shapes(
    manifest: Manifest, files: Mapping[str, SvgFile], label: str, quads: _Quads
) -> dict[tuple[str, ...], tuple[VMobject, ...]]:
    """Every part's, drawing's and stored outline's shapes, by key: a
    raster member's, its quad (made by ``quads``).
    """
    shapes: dict[tuple[str, ...], tuple[VMobject, ...]] = {}
    for i, part in enumerate(manifest.parts):
        if part.box is not None:
            assert part.digest is not None
            shapes[("part", part.name)] = quads.quad(part.file, part.box, part.digest)
            continue
        shapes[("part", part.name)] = _element(
            files,
            part.file,
            part.element,
            f"parts[{i}]",
            f"the part {part.name!r}",
            label,
        )
    for i, slot in enumerate(manifest.slots):
        for j, drawing in enumerate(slot.drawings):
            if drawing.box is not None:
                assert drawing.digest is not None
                shapes[("drawing", slot.name, drawing.name)] = quads.quad(
                    drawing.file, drawing.box, drawing.digest
                )
                continue
            shapes[("drawing", slot.name, drawing.name)] = _element(
                files,
                drawing.file,
                drawing.element,
                f"slots[{i}].drawings[{j}]",
                f"the drawing {drawing.name!r} of the slot {slot.name!r}",
                label,
            )
    for i, parameter in enumerate(manifest.parameters):
        for j, driver in enumerate(parameter.drives):
            if driver.kind != "shape":
                continue
            rest = shapes[("part", driver.part)]
            for k, key in enumerate(driver.keys):
                assert isinstance(key.value, str)
                at = f"parameters[{i}].drives[{j}].keys[{k}].file"
                outline = _element(
                    files, key.value, None, at, f"the outline in {key.value}", label
                )
                _same_structure(rest, outline, at, key.value, label)
                shapes[("outline", normalised(key.value))] = outline
    return shapes


def _element(
    files: Mapping[str, SvgFile],
    file: str,
    element: str | None,
    member: str,
    drawn: str,
    label: str,
) -> tuple[VMobject, ...]:
    """The shapes ``element`` of ``file`` draws, as ``drawn`` (what the
    manifest at ``member`` makes of them, named in a refusal).
    """
    try:
        found = files[normalised(file)].shapes(element)
    except Unreadable as error:
        _unreadable(label, file, f"{drawn}: {error.reason}")
    if found is None:
        reason = f"the file {file} has no element whose id is {element!r}"
        refuse(
            "asset.manifest_invalid",
            f"the manifest of the asset {label} is invalid at {member}: {reason}",
            asset=label,
            member=member,
            reason=reason,
        )
    if not found:
        what = f"its element {element!r}" if element is not None else "it"
        _unreadable(label, file, f"{what} draws nothing")
    return tuple(found)


def _same_structure(
    rest: tuple[VMobject, ...],
    outline: tuple[VMobject, ...],
    member: str,
    file: str,
    label: str,
) -> None:
    """Refuse a stored outline unlike the rest outline: shape for shape, the
    same subpaths in the same order, each with as many points and open or
    closed alike (:func:`~.svg_parts.structure`).
    """
    if structure(outline) != structure(rest):
        reason = (
            f"the outline in {file} has not the rest outline's structure (shape"
            " for shape, the same subpaths in order, each with as many points and"
            " open or closed alike)"
        )
        refuse(
            "asset.manifest_invalid",
            f"the manifest of the asset {label} is invalid at {member}: {reason}",
            asset=label,
            member=member,
            reason=reason,
        )
