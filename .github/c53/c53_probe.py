"""A pytest plugin measuring, per test, what the core is handed and draws.

For each test (a case) it records, as JSON lines under $C53_OUT:
- the sha256 of every frame a frames comparison is handed (``check_frame``),
  and of every frame a scene corpus test hands it (``record_frames``);
- per core painter, in order: each delta's bytes as the core receives them
  (their length, and the length and sha256 of the same delta with every
  record's ``source`` and ``facing_group`` removed), and the sha256 of each
  frame the painter returns;
and writes each core painter's stream as an XF file ($C53_OUT/xf/), for an
older core to read.

With C53_FORCE_CORE=1, every frames comparison runs on the core alone, its
control-frame comparison replaced by the recording.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path

import numpy as np

OUT = Path(os.environ["C53_OUT"])
STRIP = ("source", "facing_group")
STATE = {"case": None, "painters": {}, "frames": [], "flags": {"draw": 0, "canon": 0}}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def record_frames(frames) -> None:
    """Record the frames a scene delivered, in order."""
    STATE["frames"].extend(
        _sha(np.ascontiguousarray(np.asarray(f)).tobytes()) for f in frames
    )


def _stripped(delta):
    from arachne.frame_ir.xf import Delta, Record

    bare = Delta(
        t=delta.t,
        seq=delta.seq,
        view=delta.view,
        objects=delta.objects,
        upload=delta.upload,
        upload_paints=delta.upload_paints,
        release=delta.release,
        set=[
            (slot, Record(r.kind, {k: v for k, v in r.fields.items() if k not in STRIP}))
            for slot, r in delta.set
        ],
        remove=delta.remove,
        order=delta.order,
    )
    return bare


def _install(force_core: bool) -> None:
    from arachne.frame_ir import canonical, core_painter, xf

    flags = STATE["flags"]
    to_core = xf.Delta.to_core

    def to_core_kept(self):
        core = to_core(self)
        if flags["draw"] and not flags["canon"]:
            flags["sent"] = (self, core)
        return core

    xf.Delta.to_core = to_core_kept

    call = canonical.Canonical.__call__

    def canon(self, delta):
        flags["canon"] += 1
        try:
            return call(self, delta)
        finally:
            flags["canon"] -= 1

    canonical.Canonical.__call__ = canon

    draw = core_painter.CorePainter.draw

    def drawn(self, delta, **kwargs):
        flags["draw"] += 1
        flags["sent"] = None
        try:
            frame = draw(self, delta, **kwargs)
        finally:
            flags["draw"] -= 1
        sent = flags.get("sent")
        painters = STATE["painters"]
        entry = painters.setdefault(
            id(self), {"index": len(painters), "deltas": [], "frames": [], "core": []}
        )
        if sent is not None:
            made, core = sent
            full = bytes(core.to_bytes())
            bare = bytes(to_core(_stripped(made)).to_bytes())
            entry["deltas"].append([len(full), len(bare), _sha(bare)])
            entry["core"].append(core)
        entry["frames"].append(_sha(np.ascontiguousarray(frame).tobytes()))
        return frame

    core_painter.CorePainter.draw = drawn

    if force_core:
        from arachne.utils.testing import _frames_testers
        from arachne.utils.testing import frames_comparison as fc

        compare = fc.frames_comparison

        def forced(func=None, **kwargs):
            kwargs["draw_paths"] = ("core",)
            if callable(func):
                return compare(func, **kwargs)
            return compare(**kwargs)

        fc.frames_comparison = forced
        # No control frame is read: the recording replaces the comparison.
        fc._control_data_path = lambda *_args, **_kw: Path("unread.npz")

        @contextlib.contextmanager
        def testing(self):
            yield

        def check_frame(self, frame_number, frame):
            STATE["frames"].append(_sha(np.ascontiguousarray(frame).tobytes()))

        _frames_testers._FramesTester.testing = testing
        _frames_testers._FramesTester.check_frame = check_frame


def pytest_configure(config):
    _install(os.environ.get("C53_FORCE_CORE") == "1")
    (OUT / "xf").mkdir(parents=True, exist_ok=True)


def pytest_runtest_setup(item):
    STATE["case"] = item.nodeid
    STATE["painters"] = {}
    STATE["frames"] = []


def pytest_runtest_teardown(item, nextitem):
    import arachne_core

    case = STATE["case"]
    if case is None:
        return
    painters = []
    name = _sha(case.encode())[:24]
    for entry in sorted(STATE["painters"].values(), key=lambda e: e["index"]):
        stream = None
        if entry["core"]:
            stream = f"{name}-{entry['index']}.xf"
            arachne_core.write_xf(OUT / "xf" / stream, entry["core"])
        painters.append(
            {"deltas": entry["deltas"], "frames": entry["frames"], "xf": stream}
        )
    line = {"case": case, "frames": STATE["frames"], "painters": painters}
    with open(OUT / f"cases-{os.getpid()}.jsonl", "a") as f:
        f.write(json.dumps(line) + "\n")
    STATE["case"] = None
    STATE["painters"] = {}
    STATE["frames"] = []
