"""CE's light model stays selectable on the core: under ``core_light = ce`` the
3D examples are handed the stream, byte for byte, and drawn the frames, byte for
byte, that the core was handed and drew before the soft light became its
default.

``ce_light.json`` holds, for each 3D example drawn on the core, how many frames
the core drew and a digest of every delta the core was handed (as XF bytes, in
order) and of every frame it drew, recorded at the engine revision before the
soft light (07b24b12) on the CI runner the documentation examples run on. Run
with ``--set_test`` to record them again. The frames are the same bytes on
other machines; a few examples' deltas are not, since they carry values the
engine computes in floating point down to the last bit, so the digests hold on
the runner they were recorded on.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
from pathlib import Path

import pytest

from arachne.frame_ir.core_painter import CorePainter

from . import test_docs_examples as docs
from .collect import DocsExample

DIGESTS = Path(__file__).parent / "ce_light.json"
#: The light the examples are drawn in.
LIGHT = {"core_light": "ce"}


def _record(name: str, entry: dict) -> None:
    # Written by parallel workers: update the shared file under a lock.
    DIGESTS.touch()
    with DIGESTS.open("r+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        text = handle.read()
        digests = json.loads(text) if text.strip() else {}
        digests[name] = entry
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps(dict(sorted(digests.items())), indent=1) + "\n")


@pytest.mark.parametrize(
    "example", docs.CORE_EXAMPLES, ids=lambda example: example.name
)
def test_a_3d_example_lit_by_ce_is_handed_and_draws_what_it_did(
    example: DocsExample, request, tmp_path: Path, monkeypatch
) -> None:
    pytest.importorskip("arachne_core")
    settings = docs._config_test

    def lit(last_frame: bool):
        made = settings(last_frame)
        for key, value in LIGHT.items():
            made[key] = value
        return made

    monkeypatch.setattr(docs, "_config_test", lit)
    sent = hashlib.sha256()
    drawn = hashlib.sha256()
    frames = 0
    draw = CorePainter.draw

    def digested(self, delta, **kwargs):
        nonlocal frames
        frame = draw(self, delta, **kwargs)
        made = self.last.to_core().to_bytes()
        sent.update(len(made).to_bytes(8, "little"))
        sent.update(made)
        drawn.update(frame.tobytes())
        frames += 1
        return frame

    monkeypatch.setattr(CorePainter, "draw", digested)
    docs.render(example, tmp_path, draw_path="core")
    assert frames, f"{example.name}: the core drew no frame"
    entry = {"frames": frames, "deltas": sent.hexdigest(), "pixels": drawn.hexdigest()}
    if request.config.getoption("--set_test"):
        _record(example.name, entry)
        return
    assert entry == json.loads(DIGESTS.read_text())[example.name]
