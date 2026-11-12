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

For classification it also keeps every frame's pixels ($C53_OUT/frames/), and
beside each core painter a shadow stream: the same deltas without the path and
mesh records whose every layer Cairo paints nothing of (no paint, an alpha
that premultiplies to 0 at 8 bits, a stroke of no width). Where such a record
is held, it draws the frame on the core alone from both streams, and records
both (``shadow``): what this core draws with those records absent.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
from pathlib import Path

import numpy as np

OUT = Path(os.environ["C53_OUT"])
STRIP = ("source", "facing_group")
STATE = {
    "case": None,
    "painters": {},
    "frames": [],
    "arrays": [],
    "flags": {"draw": 0, "canon": 0},
}
PAINTS = ("background_stroke", "fill", "stroke")
WIDTHS = {"background_stroke": "background_stroke_width", "stroke": "stroke_width"}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _pixels(frame) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(frame))


def record_frames(frames) -> None:
    """Record the frames a scene delivered, in order."""
    for f in frames:
        data = _pixels(f)
        STATE["frames"].append(_sha(data.tobytes()))
        STATE["arrays"].append(data.copy())


def _alpha8(a: float) -> int:
    """Cairo's 8-bit alpha of a solid source (composite::premultiply)."""
    a = min(max(float(a), 0.0), 1.0)
    return int(a * 65535.0 + 0.5) >> 8


def _rows(paint, tables):
    if paint is None:
        return None
    if isinstance(paint, (bytes, bytearray)):
        table = tables.get(bytes(paint))
        return "unknown" if table is None else np.asarray(table.rows, float).reshape(-1, 4)
    return np.asarray(paint, float).reshape(-1, 4)


def _drawn_layers(fields, opacity, tables, widths_from=None) -> list[str]:
    """The layers of a paint block Cairo paints something of; an unknown
    paint counts as painted."""
    widths_from = fields if widths_from is None else widths_from
    found = []
    for name in PAINTS:
        rows = _rows(fields.get(name), tables)
        if rows is None:
            continue
        if name in WIDTHS:
            width = float(widths_from.get(WIDTHS[name], 0.0) or 0.0)
            if not width > 0.0:
                continue
        if isinstance(rows, str) or any(_alpha8(r[3] * opacity) > 0 for r in rows):
            found.append(name)
    return found


def paints_nothing(record, tables) -> bool:
    """Whether a path or mesh record has no layer Cairo paints anything of."""
    if record.kind not in ("path", "mesh"):
        return False
    f = record.fields
    if f.get("fill_image") is not None:
        return False
    opacity = float(f.get("opacity", 1.0))
    if _drawn_layers(f, opacity, tables):
        return False
    paint_b = f.get("paint_b")
    if paint_b and float(f.get("mix", 0.0)) != 0.0 and _drawn_layers(paint_b, opacity, tables):
        return False
    pre = f.get("pre_projected")
    if pre and any(pre.get(n) is not None for n in PAINTS):
        if _drawn_layers(pre, opacity, tables, widths_from=f):
            return False
    return True


def stroke_px(record, view) -> float | None:
    """A drawn stroke's width in pixels, as the flat frame mapping scales it
    (approximate under a 3D camera's zoom); None without a stroke width."""
    f = record.fields
    width = max(
        float(f.get("stroke_width", 0.0) or 0.0),
        float(f.get("background_stroke_width", 0.0) or 0.0),
    )
    if not width > 0.0 or not view or not view.get("frame_width"):
        return None
    mult = float(view.get("cairo_line_width_multiple") or 0.01)
    return width * mult * float(view.get("pixel_width", 0)) / float(view["frame_width"])


def _shadow(painter, entry, made, to_core) -> None:
    """Feeds the painter's shadow stream this delta without the records that
    paint nothing, and draws both where one is held."""
    import arachne_core

    sh = entry.get("sh")
    if sh is None:
        sh = entry["sh"] = {"core": arachne_core.Stream(), "live": set(), "order": None}
    if sh.get("dead"):
        return
    try:
        tables = painter.state.paints
        known = sh.setdefault("known", {})  # slot -> (record, whether it paints nothing)
        hidden = set()
        for slot, record in painter.state.records.items():
            seen = known.get(slot)
            if seen is None or seen[0] is not record:
                seen = known[slot] = (record, paints_nothing(record, tables))
            if seen[1]:
                hidden.add(slot)
        keep, remove = [], []
        for slot, record in made.set:
            if slot in hidden:
                if slot in sh["live"]:
                    remove.append(slot)
                    sh["live"].discard(slot)
            else:
                keep.append((slot, record))
                sh["live"].add(slot)
        for slot in made.remove:
            if slot in sh["live"]:
                remove.append(slot)
                sh["live"].discard(slot)
        if made.order is not None:
            sh["order"] = list(made.order)
        changes = {"set": keep, "remove": remove}
        if sh["order"] is not None and (made.order is not None or hidden or sh.get("hid")):
            changes["order"] = [e for e in sh["order"] if e[1] in sh["live"]]
        if hasattr(made, "source"):
            changes["source"] = [r for r in made.source if r[0] in sh["live"]]
        sh["hid"] = bool(hidden)
        sent = to_core(dataclasses.replace(made, **changes))
        sh["core"].apply(arachne_core.FrameDelta.from_bytes(sent.to_bytes()))
        if not hidden:
            return
        view = painter.state.view
        size = (view["pixel_width"], view["pixel_height"])
        real = _pixels(painter.core.draw(*size, threads=1, overlay=False))
        bare = _pixels(sh["core"].draw(*size, threads=1, overlay=False))
        k = len(entry["frames"]) - 1
        kinds = sorted({painter.state.records[s].kind for s in hidden})
        entry["shadow"].append(
            [k, _sha(real.tobytes()), _sha(bare.tobytes()), len(hidden), kinds]
        )
        entry["arrays"][f"r{k}"] = real.copy()
        entry["arrays"][f"s{k}"] = bare.copy()
    except Exception as error:  # recorded, never raised: the measurement goes on
        sh["dead"] = True
        entry["shadow_error"] = f"{type(error).__name__}: {str(error)[:300]}"


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
            id(self),
            {
                "index": len(painters),
                "deltas": [],
                "frames": [],
                "core": [],
                "arrays": {},
                "shadow": [],
            },
        )
        if sent is not None:
            made, core = sent
            full = bytes(core.to_bytes())
            bare = bytes(to_core(_stripped(made)).to_bytes())
            entry["deltas"].append([len(full), len(bare), _sha(bare)])
            entry["core"].append(core)
        pixels = _pixels(frame)
        entry["arrays"][f"p{len(entry['frames'])}"] = pixels.copy()
        entry["frames"].append(_sha(pixels.tobytes()))
        if sent is not None:
            _shadow(self, entry, sent[0], to_core)
        elif "sh" in entry:  # a delta the shadow did not see: it no longer follows
            entry["sh"]["dead"] = True
            entry["shadow_error"] = "a frame drawn without its delta seen"
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
            data = _pixels(frame)
            STATE["frames"].append(_sha(data.tobytes()))
            STATE["arrays"].append(data.copy())

        _frames_testers._FramesTester.testing = testing
        _frames_testers._FramesTester.check_frame = check_frame


def pytest_configure(config):
    _install(os.environ.get("C53_FORCE_CORE") == "1")
    (OUT / "xf").mkdir(parents=True, exist_ok=True)
    (OUT / "frames").mkdir(parents=True, exist_ok=True)


def pytest_runtest_setup(item):
    STATE["case"] = item.nodeid
    STATE["painters"] = {}
    STATE["frames"] = []
    STATE["arrays"] = []


def pytest_runtest_teardown(item, nextitem):
    import arachne_core

    case = STATE["case"]
    if case is None:
        return
    painters = []
    name = _sha(case.encode())[:24]
    arrays = {f"f{i}": a for i, a in enumerate(STATE["arrays"])}
    for entry in sorted(STATE["painters"].values(), key=lambda e: e["index"]):
        stream = None
        if entry["core"]:
            stream = f"{name}-{entry['index']}.xf"
            arachne_core.write_xf(OUT / "xf" / stream, entry["core"])
        for key, a in entry["arrays"].items():
            arrays[f"{key[0]}{entry['index']}_{key[1:]}"] = a
        painters.append(
            {
                "deltas": entry["deltas"],
                "frames": entry["frames"],
                "xf": stream,
                "shadow": entry["shadow"],
                "shadow_error": entry.get("shadow_error"),
            }
        )
    if arrays:
        np.savez_compressed(OUT / "frames" / f"{name}.npz", **arrays)
    line = {"case": case, "frames": STATE["frames"], "painters": painters}
    with open(OUT / f"cases-{os.getpid()}.jsonl", "a") as f:
        f.write(json.dumps(line) + "\n")
    STATE["case"] = None
    STATE["painters"] = {}
    STATE["frames"] = []
    STATE["arrays"] = []
