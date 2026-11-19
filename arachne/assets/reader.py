"""Loading an asset: its manifest, its files and their shapes, read whole
before anything is drawn.

:func:`load_asset` takes a reference to an asset: a path to its directory
or to its ``manifest.json``, resolved as the engine's image and SVG
constructors resolve a file name (as given, then under the configuration's
``assets_dir``), or ``fixture:<name>`` for one of the engine's own fixture
assets. Nothing else is searched and nothing is fetched: an address, or a
file the manifest names by an address, an absolute path or a path out of the
asset's directory, is ``asset.remote``. Every way an asset can be unusable
is a finding raised with its code: no manifest (``asset.manifest_missing``),
a manifest that is not valid (``asset.manifest_invalid``), a file not there
(``asset.file_missing``), one that cannot be drawn exactly
(``asset.unreadable``). Nothing is cached between loads.
"""

from __future__ import annotations

__all__ = ["FIXTURES", "Asset", "load_asset"]

import json
import os
import posixpath
from collections.abc import Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any, NoReturn
from urllib.parse import urlsplit

from .. import config
from ..mobject.types.vectorized_mobject import VMobject
from .manifest import (
    MANIFEST,
    AnchorSpec,
    Manifest,
    ParameterSpec,
    PartSpec,
    SlotSpec,
    digest,
    normalised,
    parse_manifest,
    refuse,
)
from .svg_parts import SvgFile, Unreadable, read_svg

#: The engine's own fixture assets, each a directory named as it is asked
#: for: ``fixture:<name>``.
FIXTURES = Path(__file__).resolve().parent / "fixtures"
_FIXTURE = "fixture:"


class Asset:
    """An asset, loaded: its manifest, its digest, and the shapes of its
    parts, drawings and stored outlines at rest, in canvas coordinates.

    It is a value: nothing of it changes once loaded, and a copy of it is
    itself. The shapes it gives are fresh copies.
    """

    def __init__(
        self,
        directory: Path,
        manifest: Manifest,
        digest: str,
        canvas: tuple[float, ...] | None,
        shapes: Mapping[tuple[str, ...], tuple[VMobject, ...]],
    ) -> None:
        self.directory = directory
        self.manifest = manifest
        self.digest = digest
        #: The part files' shared ``viewBox``, ``None`` when they have none.
        self.canvas = canvas
        self._shapes = shapes

    @property
    def name(self) -> str:
        """The asset's name."""
        return self.manifest.name

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
    """The asset ``ref`` names (see the module), checked and read whole;
    raises the finding of the first thing that makes it unusable.
    """
    if isinstance(ref, Asset):
        return ref
    label = os.fspath(ref)
    directory = _directory(label)
    data = (directory / MANIFEST).read_bytes()
    document = _document(data, label)
    manifest = parse_manifest(document, label)
    named = manifest.files()
    for file in named:
        _local(file, directory, label)
    contents: dict[str, bytes] = {}
    for file in named:
        path = directory / file
        if not path.is_file():
            refuse(
                "asset.file_missing",
                f"the asset {label} names the file {file}, which is not in its"
                " directory",
                asset=label,
                file=file,
            )
        contents[normalised(file)] = path.read_bytes()
    files = {file: _read(contents[file], file, label) for file in contents}
    canvas = _canvas(files, label)
    shapes = _shapes(manifest, files, label)
    return Asset(
        directory, manifest, digest(manifest.document, contents), canvas, shapes
    )


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
    """Refuse the file ``file`` a manifest names unless it is a path in the
    asset's directory ``directory``.
    """
    outside = (
        _addressed(file)
        or PurePosixPath(file).is_absolute()
        or PureWindowsPath(file).is_absolute()
        or PureWindowsPath(file).drive != ""
        or posixpath.normpath(file).split("/")[0] == ".."
    )
    if not outside:
        root = os.path.realpath(directory)
        found = os.path.realpath(directory / file)
        outside = os.path.commonpath([root, found]) != root
    if outside:
        refuse(
            "asset.remote",
            f"the asset {label} names the file {file}, which is not a path inside"
            " its directory",
            asset=label,
            file=file,
        )


def _document(data: bytes, label: str) -> Any:
    """The manifest's JSON; a document that is not JSON, or that repeats a
    member, is ``asset.manifest_invalid``.
    """

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        found: dict[str, Any] = {}
        for key, value in pairs:
            if key in found:
                raise ValueError(f"the member {key!r} is given twice")
            found[key] = value
        return found

    def no_constant(name: str) -> Any:
        raise ValueError(f"{name} is not a JSON number")

    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=no_constant,
        )
    except ValueError as error:
        reason = f"it is not valid JSON ({error})"
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
    manifest: Manifest, files: Mapping[str, SvgFile], label: str
) -> dict[tuple[str, ...], tuple[VMobject, ...]]:
    """Every part's, drawing's and stored outline's shapes, by key."""
    shapes: dict[tuple[str, ...], tuple[VMobject, ...]] = {}
    for i, part in enumerate(manifest.parts):
        shapes[("part", part.name)] = _element(
            files, part.file, part.element, f"parts[{i}]", label
        )
    for i, slot in enumerate(manifest.slots):
        for j, drawing in enumerate(slot.drawings):
            shapes[("drawing", slot.name, drawing.name)] = _element(
                files,
                drawing.file,
                drawing.element,
                f"slots[{i}].drawings[{j}]",
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
                outline = _element(files, key.value, None, at, label)
                _same_structure(rest, outline, at, key.value, label)
                shapes[("outline", normalised(key.value))] = outline
    return shapes


def _element(
    files: Mapping[str, SvgFile],
    file: str,
    element: str | None,
    member: str,
    label: str,
) -> tuple[VMobject, ...]:
    found = files[normalised(file)].shapes(element)
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
    """Refuse a stored outline whose shapes and points do not match the
    rest outline's one for one.
    """
    if [len(shape.points) for shape in outline] != [
        len(shape.points) for shape in rest
    ]:
        reason = (
            f"the outline in {file} has not the rest outline's structure (the"
            " same shapes, with as many points each)"
        )
        refuse(
            "asset.manifest_invalid",
            f"the manifest of the asset {label} is invalid at {member}: {reason}",
            asset=label,
            member=member,
            reason=reason,
        )
