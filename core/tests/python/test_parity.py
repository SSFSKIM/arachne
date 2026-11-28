"""The parity kit: the core passes its own references, a frame off by more than its class's bound
fails, a probe changed since its reference fails, and every bound is a placeholder.
"""

from __future__ import annotations

import io
import json
import shutil
from pathlib import Path

import arachne_core
import numpy as np
from arachne_core import parity

KIT = Path(__file__).resolve().parents[2] / "parity"


def run(kit: Path, **kwargs) -> tuple[bool, list[str]]:
    out = io.StringIO()
    ok = parity.run("cpu", kit, out=out, **kwargs)
    return ok, out.getvalue().splitlines()


def test_the_core_passes_every_fixture():
    names = [f[0] for f in arachne_core.probe_fixtures()]
    assert len(names) == 10
    ok, lines = run(KIT)
    assert ok, lines
    assert lines == [f"PASS {n}" for n in names] + [
        f"parity: {len(names)}/{len(names)} pass"
    ]


def test_every_bound_is_a_placeholder():
    # The bounds are drift between backends against the core's own frames, not the core's
    # coverage-to-sample constant; every one, coverage among them, waits for the
    # cross-architecture measurement.
    tolerances = json.loads((KIT / "tolerances.json").read_text())
    bounds = tolerances["bounds"]
    assert set(bounds) == set(arachne_core.RESOLVES)
    assert set(tolerances["placeholder"]) == set(arachne_core.RESOLVES)
    assert "coverage" in tolerances["placeholder"]


def test_a_pixel_beyond_its_class_bound_fails_its_fixture(tmp_path):
    kit = tmp_path / "kit"
    shutil.copytree(KIT, kit)
    rgba, resolves = parity.load_reference(kit, "seam")
    coverage = arachne_core.RESOLVES.index("coverage")
    y, x = np.argwhere(resolves[1] == coverage)[0]
    # Within the bound passes; one past it fails.
    bound = json.loads((KIT / "tolerances.json").read_text())["bounds"]["coverage"]
    for off, passes in [(bound, True), (bound + 1, False)]:
        moved = rgba.copy()
        v = int(rgba[1, y, x, 0])
        moved[1, y, x, 0] = v - off if v >= off else v + off
        np.savez_compressed(kit / "references" / "seam.npz", frame_data=moved)
        ok, lines = run(kit)
        assert ok == passes, lines
        if not passes:
            assert any(
                line.startswith(
                    f"FAIL seam: frame 1: 1 pixels beyond their class's bound, the first at {x},{y}"
                )
                for line in lines
            ), lines
            assert lines[-1] == "parity: 9/10 pass"


def test_a_pixel_resolved_in_another_class_fails_the_cpu_backend(tmp_path):
    kit = tmp_path / "kit"
    shutil.copytree(KIT, kit)
    path = kit / "references" / "seam.resolves.npz"
    with np.load(path) as ref:
        classes = ref["frame_data"].copy()
    coverage = arachne_core.RESOLVES.index("coverage")
    sample = arachne_core.RESOLVES.index("sample")
    y, x = np.argwhere(classes[1, :, :, 0] == coverage)[0]
    classes[1, y, x] = sample
    np.savez_compressed(path, frame_data=classes)
    ok, lines = run(kit)
    assert not ok
    assert any(
        line.startswith(
            f"FAIL seam: frame 1: 1 pixels resolved in another class, the first at {x},{y} "
            "(coverage where the reference has sample)"
        )
        for line in lines
    ), lines
    assert lines[-1] == "parity: 9/10 pass"


def test_a_reference_from_another_stream_fails_until_regenerated(tmp_path):
    kit = tmp_path / "kit"
    shutil.copytree(KIT, kit)
    shas = kit / "references" / "streams.json"
    pinned = json.loads(shas.read_text())
    shas.write_text(json.dumps({**pinned, "orbit": "0" * 64}))
    ok, lines = run(kit)
    assert not ok
    assert (
        "FAIL orbit: the reference was drawn from another stream (regenerate it)"
        in lines
    )
    ok, lines = run(kit, regenerate_references=True)
    assert ok, lines
    assert json.loads(shas.read_text()) == pinned
    orbit = next(f for f in arachne_core.probe_fixtures() if f[0] == "orbit")
    assert pinned["orbit"] == parity.stream_sha256(orbit[3])


def test_references_are_one_uint8_frame_array_per_file():
    # The only .npz the repository's leak check admits (scripts/leak_check.py): one uint8
    # `frame_data` of frames x height x width x 3 or 4.
    refs = sorted((KIT / "references").glob("*.npz"))
    assert len(refs) == 20
    for path in refs:
        with np.load(path) as ref:
            assert list(ref.keys()) == ["frame_data"]
            data = ref["frame_data"]
            assert data.dtype == np.uint8
            assert data.ndim == 4
            assert data.shape[-1] == (3 if path.name.endswith(".resolves.npz") else 4)


def test_the_fixtures_can_be_written_as_xf_and_read_back(tmp_path):
    ok, _ = run(KIT, write_streams=tmp_path)
    assert ok
    for name, _, _, deltas in arachne_core.probe_fixtures():
        back = arachne_core.read_xf(tmp_path / f"{name}.xf")
        assert [d.to_bytes() for d in back] == [d.to_bytes() for d in deltas]
