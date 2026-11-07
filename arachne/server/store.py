"""What a session server keeps after it ends: the answers it gave, and the
summary of each execution, keyed by the execution they are of.

A record holds live mobjects and the scene's own callables, so no record is
kept: the store keeps what can be written down. For each session opened (a
scene class of a file, under a configuration, from a working directory,
which a relative path the scene reads or its configuration names is read
from) its **manifest**: the files its execution came from, by digest (its
source and every file its code read, as its module loaded too), the
identity of the execution, and the record's form as ``open`` answered it.
For each execution, the **answers** given, each by its request (the method
and its params); and its **summary**
(:class:`~arachne.timeline.execution.ExecutionSummary`), kept as it
executes and before its manifest, which a render checks itself against.

A server asked to open a session whose manifest's files still hold what
they held, by digest, answers from the store until a request needs what the
store does not hold, and only then executes the scene. Everything is kept
under the engine that made it (:func:`engine_fingerprint`, its native core
included): another engine, or this one edited or built again, reads none of
it.

The store is the directory ``ARACHNE_STORE`` names, else the user's cache
directory's ``arachne/store``. Each file is written whole, under another
name, then renamed: a reader finds a whole file or none. Nothing is ever
removed from it.
"""

from __future__ import annotations

__all__ = [
    "Manifest",
    "Store",
    "dependency_reads",
    "engine_fingerprint",
    "identity_form",
    "settings_of",
    "store_dir",
    "summary_key",
    "working_dir",
]

import contextlib
import functools
import hashlib
import importlib.machinery
import importlib.metadata
import importlib.util
import json
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .._config import config as global_config
from .._config import tempconfig
from ..timeline.execution import ExecutionSummary
from ..timeline.inputs import file_digests, source_files

if TYPE_CHECKING:
    from .._config.utils import ManimConfig
    from ..session import Session
    from ..timeline.record import Identity
    from ..typing import StrPath

#: The configuration's options of a session's pixel size: an execution
#: summary leaves the pixel shape out, and of the size only the aspect
#: changes an execution (the camera fits its frame to it).
PIXEL_OPTIONS = ("pixel_width", "pixel_height", "quality")


def store_dir() -> Path:
    """Where the store is: ``ARACHNE_STORE``, else the user's cache
    directory's ``arachne/store``.
    """
    named = os.environ.get("ARACHNE_STORE")
    if named:
        return Path(named)
    if sys.platform == "darwin":
        cache = Path.home() / "Library" / "Caches"
    else:
        cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return cache / "arachne" / "store"


def _core_identity() -> str:
    """The native core this process would load: ``arachne_core``'s version
    and the sha256 of each of its compiled extensions, by name; ``"absent"``
    when it is not installed. Read without importing it.
    """
    spec = importlib.util.find_spec("arachne_core")
    if spec is None or not spec.submodule_search_locations:
        return "absent"
    try:
        version = importlib.metadata.version("arachne-core")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    built = tuple(importlib.machinery.EXTENSION_SUFFIXES)
    extensions = [
        path
        for location in spec.submodule_search_locations
        for path in sorted(Path(location).iterdir())
        if path.name.endswith(built)
    ]
    found = [f"arachne-core {version}"]
    found += [
        f"{p.name} {hashlib.sha256(p.read_bytes()).hexdigest()}" for p in extensions
    ]
    return "\n".join(found)


@functools.cache
def engine_fingerprint() -> str:
    """The digest of the engine answering: the interpreter's version, every
    source file of the ``arachne`` package, by path and content, and its
    native core (:func:`_core_identity`).
    """
    root = Path(__file__).resolve().parent.parent
    digest = hashlib.sha256(sys.version.encode())
    digest.update(_core_identity().encode() + b"\0")
    for directory, names, files in os.walk(root):
        names[:] = sorted(n for n in names if n != "__pycache__")
        for name in sorted(files):
            if not name.endswith((".py", ".json")):
                continue
            path = Path(directory, name)
            digest.update(path.relative_to(root).as_posix().encode() + b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def settings_of(config: Mapping[str, Any]) -> ManimConfig:
    """The whole configuration a session opened with ``config`` executes
    under, as :meth:`~arachne.session.Session.open` makes it.
    """
    with tempconfig({**config, "dry_run": True, "disable_caching": True}):
        return global_config.copy()


def _digest(*parts: Any) -> str:
    text = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=repr)
    return hashlib.sha256(text.encode()).hexdigest()[:32]


def _settings_digest(settings: ManimConfig, *, pixels: bool = True) -> str:
    """A digest of every option of ``settings``; without ``pixels``, of the
    pixel size only its aspect.
    """
    options = sorted(
        (name, repr(settings[name]))
        for name in settings
        if pixels or name not in PIXEL_OPTIONS
    )
    if not pixels:
        aspect = Fraction(int(settings.pixel_width), int(settings.pixel_height))
        options.append(("pixel_aspect", str(aspect)))
    return _digest(options)


def identity_form(identity: Identity) -> dict[str, Any]:
    """An execution's identity as JSON values, as the record's form has it."""
    return {
        "source_digest": identity.source_digest,
        "input_digests": [list(pair) for pair in identity.input_digests],
        "seed": identity.seed,
    }


def summary_key(
    file: str,
    scene: str,
    identity: Mapping[str, Any],
    config: Mapping[str, Any],
    reads: Sequence[tuple[str, str]] = (),
) -> str:
    """The key of the summary of ``identity``'s execution of ``scene`` of
    ``file`` (a real path) under ``config``, having read ``reads``
    (:func:`dependency_reads`): every option of the configuration but the
    pixel size, of which only the aspect, since the camera fits its frame
    to it; the frame rate among them, since a play's states depend on the
    instants it was evaluated at. A render at another frame rate or aspect
    than an execution kept is another execution, never checked against it.
    The working directory is the process's, which a relative path is read
    from.
    """
    settings = _settings_digest(settings_of(config), pixels=False)
    return _digest(
        "summary",
        file,
        scene,
        dict(identity),
        [list(read) for read in reads],
        settings,
        working_dir(),
    )


def dependency_reads(
    session: Session, loaded: Sequence[tuple[str, str]] = ()
) -> tuple[tuple[str, str], ...]:
    """Every file ``session``'s execution read, by ``(path relative to the
    scene's directory, sha256)``: those its code read as it ran, and
    ``loaded``, those read as its modules loaded, which its identity leaves
    out.
    """
    reads = dict(loaded)
    source = session._source
    if source is not None:
        for read in source.reads:
            reads.setdefault(read.path, read.digest)
    return tuple(sorted(reads.items()))


def working_dir() -> str:
    """The process's working directory, a real path: what a relative path the
    scene opens, or its configuration names, is read from.
    """
    return os.path.realpath(os.getcwd())


@dataclass(frozen=True)
class Manifest:
    """A session as the store keeps it: what it was opened as (``file``, a
    real path, ``scene`` and ``config``), the files it came from (``files``,
    its source, and ``reads``, the files its execution read, each as
    ``(path relative to the scene's directory, sha256)``), its execution's
    ``identity``, and ``record``, the record's form ``open`` answered.
    """

    key: str
    file: str
    scene: str
    config: dict[str, Any]
    files: tuple[tuple[str, str], ...]
    reads: tuple[tuple[str, str], ...]
    identity: dict[str, Any]
    record: dict[str, Any]

    @property
    def answers_key(self) -> str:
        """The key of the answers given of this execution: of its identity
        and every file it came from, those read as its modules loaded among
        them, so an answer is never given of other files.
        """
        files = [list(pair) for pair in self.files]
        reads = [list(pair) for pair in self.reads]
        return _digest("answers", self.key, self.identity, files, reads)

    def fresh(self) -> bool:
        """Whether the files the session came from hold what they held, by
        digest, as :meth:`~arachne.session.Session.changed` reads them.
        """
        scene_dir = os.path.dirname(self.file)
        try:
            if file_digests(source_files(self.file), scene_dir) != self.files:
                return False
            return all(
                file_digests([os.path.join(scene_dir, path)], scene_dir)[0][1] == digest
                for path, digest in self.reads
            )
        except OSError:
            return False

    def to_json(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "file": self.file,
            "scene": self.scene,
            "config": self.config,
            "files": [list(pair) for pair in self.files],
            "reads": [list(pair) for pair in self.reads],
            "identity": self.identity,
            "record": self.record,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Manifest:
        return cls(
            key=data["key"],
            file=data["file"],
            scene=data["scene"],
            config=data["config"],
            files=tuple((path, digest) for path, digest in data["files"]),
            reads=tuple((path, digest) for path, digest in data["reads"]),
            identity=data["identity"],
            record=data["record"],
        )


def _request_key(method: str, params: Mapping[str, Any]) -> str:
    return _digest("request", method, dict(params))


class Store:
    """The store at ``root`` (else :func:`store_dir`), as the engine
    answering now keeps it.
    """

    def __init__(self, root: StrPath | None = None) -> None:
        self.root = (Path(root) if root is not None else store_dir()) / (
            engine_fingerprint()[:16]
        )

    # Manifests

    @staticmethod
    def manifest_key(file: str, scene: str, config: Mapping[str, Any]) -> str:
        """The key of the session of ``scene`` of ``file`` (a real path)
        opened with ``config`` from the process's working directory: the whole
        configuration it executes under, and where relative paths are read.
        """
        settings = _settings_digest(settings_of(config))
        return _digest("manifest", file, scene, settings, working_dir())

    def manifest(
        self, file: str, scene: str, config: Mapping[str, Any]
    ) -> Manifest | None:
        """The manifest kept of the session, ``None`` when none is."""
        found = self._read(self._manifest_path(self.manifest_key(file, scene, config)))
        return None if found is None else Manifest.from_json(found)

    def keep_manifest(
        self,
        file: str,
        scene: str,
        config: Mapping[str, Any],
        session: Session,
        record: dict[str, Any],
        loaded: tuple[tuple[str, str], ...] = (),
    ) -> Manifest | None:
        """Keep ``session``'s manifest, ``record`` its record's form, and
        ``loaded`` among the files it read: those its code read whenever it
        ran, as its modules loaded too, as ``(path relative to the scene's
        directory, sha256)``. ``None`` when its files can't be shown to be
        what it executed.
        """
        source = session._source
        if source is None or source.files is None or session.stale:
            return None
        manifest = Manifest(
            key=self.manifest_key(file, scene, config),
            file=file,
            scene=scene,
            config=dict(config),
            files=source.files,
            reads=dependency_reads(session, loaded),
            identity=identity_form(session.record.identity),
            record=record,
        )
        self._write(self._manifest_path(manifest.key), manifest.to_json())
        return manifest

    def _manifest_path(self, key: str) -> Path:
        return self.root / "manifests" / f"{key}.json"

    # Answers

    def answer(
        self, manifest: Manifest, method: str, params: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        """The forms answered to ``method`` asked with ``params`` of
        ``manifest``'s execution, ``None`` when none were kept.
        """
        return self._read(self._answer_path(manifest, method, params, ".json"))

    def answer_file(
        self, manifest: Manifest, method: str, params: Mapping[str, Any]
    ) -> bytes | None:
        """The file kept with the answer (a frame's PNG), if any."""
        path = self._answer_path(manifest, method, params, ".bin")
        with contextlib.suppress(OSError):
            return path.read_bytes()
        return None

    def keep_answer(
        self,
        manifest: Manifest,
        method: str,
        params: Mapping[str, Any],
        forms: dict[str, Any],
        file: bytes | None = None,
    ) -> None:
        """Keep ``forms``, the answer to ``method`` asked with ``params``,
        and ``file``, written with it (a frame's PNG): the file first, so
        that an answer kept is never one without its file.
        """
        if file is not None:
            self._write_bytes(self._answer_path(manifest, method, params, ".bin"), file)
        self._write(self._answer_path(manifest, method, params, ".json"), forms)

    def _answer_path(
        self,
        manifest: Manifest,
        method: str,
        params: Mapping[str, Any],
        suffix: str,
    ) -> Path:
        name = _request_key(method, params) + suffix
        return self.root / "answers" / manifest.answers_key / name

    # Summaries

    def summary(self, key: str) -> ExecutionSummary | None:
        """The execution summary kept under ``key`` (:func:`summary_key`)."""
        path = self.root / "summaries" / f"{key}.json"
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return None
        try:
            return ExecutionSummary.from_json(text)
        except (ValueError, KeyError, TypeError):
            return None

    def has_summary(self, key: str) -> bool:
        return (self.root / "summaries" / f"{key}.json").exists()

    def keep_summary(self, key: str, summary: ExecutionSummary) -> None:
        self._write_bytes(
            self.root / "summaries" / f"{key}.json", summary.to_json().encode()
        )

    # Files

    def _read(self, path: Path) -> dict[str, Any] | None:
        try:
            found = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return found if isinstance(found, dict) else None

    def _write(self, path: Path, value: dict[str, Any]) -> None:
        self._write_bytes(
            path, json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
        )

    def _write_bytes(self, path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(handle, "wb") as file:
                file.write(data)
            os.replace(temporary, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(temporary)
            raise
