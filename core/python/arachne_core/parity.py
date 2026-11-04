"""The parity kit's runner: a backend draws the kit's fixtures and each frame is held to the core's
reference frames, pixel by pixel, within the bound ``tolerances.json`` sets for the resolve class
the reference pixel took.

The kit (``core/parity/``) holds, per fixture, its reference frames (``references/<fixture>.npz``,
``frame_data``: uint8, frames x height x width x 4) and their resolve classes
(``references/<fixture>.resolves.npz``, ``frame_data``: uint8, frames x height x width x 3, the
index into ``RESOLVES`` in every channel), the sha256 of the stream each fixture's references
were drawn from (``references/streams.json``), and ``tolerances.json``. The fixtures are the
class probes' streams, built in code by ``probe_fixtures`` and never stored; ``--write-streams
DIR`` writes them as XF files for a backend that reads streams from disk.

    python -m arachne_core.parity --backend cpu [--kit DIR] [--regenerate] [--write-streams DIR]

prints ``PASS <fixture>`` or ``FAIL <fixture>: ...`` per fixture, then ``parity: <passed>/<n>
pass``, and exits 0 when every fixture passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from arachne_core._core import RESOLVES, Stream, probe_fixtures, write_xf

# A backend draws a fixture's frames: (deltas, width, height) -> [(rgba, resolves)] per frame.
Backend = Callable[[list, int, int], list[tuple[np.ndarray, np.ndarray]]]


def _cpu(deltas: list, width: int, height: int) -> list[tuple[np.ndarray, np.ndarray]]:
    stream = Stream()
    frames = []
    for delta in deltas:
        stream.apply(delta)
        frames.append(stream.draw_resolves(width, height))
    return frames


BACKENDS: dict[str, Backend] = {"cpu": _cpu}


def stream_sha256(deltas: list) -> str:
    """The sha256 of a fixture's stream: its deltas' XF bytes, in order."""
    digest = hashlib.sha256()
    for delta in deltas:
        digest.update(delta.to_bytes())
    return digest.hexdigest()


def default_kit() -> Path:
    for candidate in (Path("parity"), Path("core/parity")):
        if (candidate / "tolerances.json").is_file():
            return candidate
    return Path("parity")


def load_bounds(kit: Path) -> np.ndarray:
    """Per resolve class, in ``RESOLVES`` order, the largest difference a channel may show."""
    bounds = json.loads((kit / "tolerances.json").read_text())["bounds"]
    return np.array([int(bounds[name]) for name in RESOLVES], dtype=np.int16)


def _paths(kit: Path, name: str) -> tuple[Path, Path]:
    refs = kit / "references"
    return refs / f"{name}.npz", refs / f"{name}.resolves.npz"


def load_reference(kit: Path, name: str) -> tuple[np.ndarray, np.ndarray] | None:
    """A fixture's reference frames and their resolve classes, or None when it has none."""
    rgba_path, resolves_path = _paths(kit, name)
    if not (rgba_path.is_file() and resolves_path.is_file()):
        return None
    with np.load(rgba_path) as rgba, np.load(resolves_path) as resolves:
        return rgba["frame_data"], resolves["frame_data"][..., 0]


def load_shas(kit: Path) -> dict[str, str]:
    path = kit / "references" / "streams.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def compare(
    frames: Sequence[tuple[np.ndarray, np.ndarray]],
    reference: tuple[np.ndarray, np.ndarray],
    bounds: np.ndarray,
) -> str | None:
    """What fails in a fixture's frames against its reference, or None when they pass."""
    rgba, resolves = reference
    if len(frames) != len(rgba):
        return f"{len(frames)} frames drawn where the reference has {len(rgba)}"
    for i, (drawn, _) in enumerate(frames):
        if drawn.shape != rgba[i].shape:
            return f"frame {i}: shape {drawn.shape} against {rgba[i].shape}"
        diff = np.abs(drawn.astype(np.int16) - rgba[i].astype(np.int16)).max(axis=2)
        over = diff > bounds[resolves[i]]
        if over.any():
            y, x = np.argwhere(over)[0]
            return (
                f"frame {i}: {int(over.sum())} pixels beyond their class's bound, the "
                f"first at {x},{y} ({RESOLVES[resolves[i][y, x]]}, off by "
                f"{int(diff[y, x])})"
            )
    return None


def regenerate(kit: Path, name: str, frames: list) -> None:
    """Writes a fixture's reference, unless the one there already holds these frames."""
    rgba = np.stack([f[0] for f in frames])
    resolves = np.stack([f[1] for f in frames])
    old = load_reference(kit, name)
    if (
        old is not None
        and np.array_equal(old[0], rgba)
        and np.array_equal(old[1], resolves)
    ):
        return
    rgba_path, resolves_path = _paths(kit, name)
    rgba_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(rgba_path, frame_data=rgba)
    classes = np.repeat(resolves[..., np.newaxis], 3, axis=3)
    np.savez_compressed(resolves_path, frame_data=classes)


def run(
    backend: str = "cpu",
    kit: Path | None = None,
    regenerate_references: bool = False,
    write_streams: Path | None = None,
    out=sys.stdout,
) -> bool:
    """Runs the kit; True when every fixture passes."""
    kit = kit or default_kit()
    draw = BACKENDS[backend]
    bounds = load_bounds(kit)
    shas = {} if regenerate_references else load_shas(kit)
    fixtures = probe_fixtures()
    passed = 0
    for name, width, height, deltas in fixtures:
        sha = stream_sha256(deltas)
        if write_streams is not None:
            write_streams.mkdir(parents=True, exist_ok=True)
            write_xf(write_streams / f"{name}.xf", deltas)
        frames = draw(deltas, width, height)
        if regenerate_references:
            regenerate(kit, name, frames)
            shas[name] = sha
        reference = load_reference(kit, name)
        if reference is None:
            failure = f"no reference in {kit / 'references'}"
        elif shas.get(name) != sha:
            failure = "the reference was drawn from another stream (regenerate it)"
        else:
            failure = compare(frames, reference, bounds)
        if failure is None:
            passed += 1
            print(f"PASS {name}", file=out)
        else:
            print(f"FAIL {name}: {failure}", file=out)
    if regenerate_references:
        text = json.dumps(dict(sorted(shas.items())), indent=2) + "\n"
        (kit / "references" / "streams.json").write_text(text)
    print(f"parity: {passed}/{len(fixtures)} pass", file=out)
    return passed == len(fixtures)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m arachne_core.parity", description=__doc__
    )
    parser.add_argument("--backend", choices=sorted(BACKENDS), required=True)
    parser.add_argument(
        "--kit", type=Path, help="the kit's directory (default: ./parity)"
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="write the references from this backend's frames first (the core's only)",
    )
    parser.add_argument(
        "--write-streams", type=Path, metavar="DIR", help="write the fixtures' XF"
    )
    args = parser.parse_args(argv)
    return 0 if run(args.backend, args.kit, args.regenerate, args.write_streams) else 1


if __name__ == "__main__":
    raise SystemExit(main())
