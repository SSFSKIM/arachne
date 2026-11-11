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

With C53_PLACEMENT=off, every core stream draws its labels unplaced (where the
core has the switch). With C53_FRAMES=1, a case that draws a facing group keeps
its frames ($C53_OUT/frames/<case>.npz: each core painter's, the scene's own,
and a graphical unit's Cairo control frames), and its labels' plan outcomes
per frame, for the before/after comparison against CE.
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
STATE = {
    "case": None,
    "painters": {},
    "frames": [],
    "arrays": [],
    "control": None,
    "flags": {"draw": 0, "canon": 0},
}
PLACEMENT = os.environ.get("C53_PLACEMENT", "on")
KEEP = os.environ.get("C53_FRAMES") == "1"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def record_frames(frames) -> None:
    """Record the frames a scene delivered, in order."""
    STATE["frames"].extend(
        _sha(np.ascontiguousarray(np.asarray(f)).tobytes()) for f in frames
    )
    if KEEP:
        STATE["arrays"].extend(np.asarray(f).copy() for f in frames)


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
        if PLACEMENT == "off" and hasattr(self.core, "set_label_placement"):
            self.core.set_label_placement(False)
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
                "facing": False,
                "arrays": [],
                "plans": [],
                "tables": {},
                "faint": set(),
                "band": {},
                "blends": {},
                "blend_sets": 0,
                "band_log": [],
            },
        )
        for table in delta.upload_paints:
            entry["tables"][table.key] = np.asarray(table.rows, dtype=float)
        def fill_rows(fill):
            if fill is None:
                return None
            rows = entry["tables"].get(fill) if isinstance(fill, bytes) else fill
            if rows is None:
                return None
            return np.asarray(rows, dtype=float).reshape(-1, 4)

        for slot, r in delta.set:
            paint_b = r.fields.get("paint_b")
            if paint_b:
                a, b = fill_rows(r.fields.get("fill")), fill_rows(paint_b.get("fill"))
                mix = float(r.fields.get("mix", 0.0))
                alphas = [x[:, 3] * 255 for x in (a, b) if x is not None]
                both = np.concatenate(alphas) if alphas else np.zeros(0)
                blend = None
                if a is not None and b is not None and len(a) == len(b):
                    blend = (a[:, 3] * (1 - mix) + b[:, 3] * mix) * 255
                rng = lambda v: None if v is None or len(v) == 0 else [
                    round(float(v.min()), 4),
                    round(float(v.max()), 4),
                ]
                entry["blend_sets"] += 1
                if len(both) >= 2 and both.min() < 1.5 and both.max() >= 0.5:
                    if mix == 1.0:
                        mixed = b
                    elif a is None or b is None or len(a) == 0 or len(b) == 0:
                        mixed = None
                    else:
                        n = max(len(a), len(b))
                        pick = lambda x, i: x[0 if len(x) == 1 else i]
                        mixed = np.array(
                            [(1 - mix) * pick(a, i) + mix * pick(b, i) for i in range(n)]
                        )
                    entry["band_log"].append(
                        {
                            "seq": int(delta.seq),
                            "slot": int(slot),
                            "mix": float(mix),
                            "gradient": r.fields.get("gradient") is not None,
                            "fill_a": None if a is None else a.tolist(),
                            "fill_b": None if b is None else b.tolist(),
                            "blend_alpha_x255": None
                            if mixed is None
                            else [float(v) * 255 for v in mixed[:, 3]],
                        }
                    )
                form = lambda f: (
                    None if f is None else "table " + f.hex()[:16] if isinstance(f, bytes)
                    else "inline"
                )
                rows_of = lambda x: None if x is None else [
                    [round(float(v), 6) for v in row] for row in x
                ]
                entry["blends"][slot] = {
                    "fill_a_form": form(r.fields.get("fill")),
                    "fill_a_rows": rows_of(a),
                    "fill_b_form": form(paint_b.get("fill")),
                    "fill_b_rows": rows_of(b),
                    "kind": r.kind,
                    "mix": round(mix, 4),
                    "fill_a": rng(None if a is None else a[:, 3] * 255),
                    "fill_b": rng(None if b is None else b[:, 3] * 255),
                    "blend": rng(blend),
                    "band_capable": bool(
                        len(both) >= 2 and both.min() < 1.5 and both.max() >= 0.5
                    ),
                    "source": str(r.fields.get("source"))[:120],
                }
            for fields in (r.fields, r.fields.get("pre_projected") or {}):
                fill = fields.get("fill")
                if fields.get("gradient") is None or fill is None:
                    continue
                rows = entry["tables"].get(fill) if isinstance(fill, bytes) else fill
                if rows is None:
                    continue
                rows = np.asarray(rows, dtype=float).reshape(-1, 4)
                if len(rows) >= 2 and rows[:, 3].min() * 255 < 2:
                    entry["faint"].add(slot)
                a = rows[:, 3] * 255 if len(rows) >= 2 else None
                if a is not None and a.min() < 1.5 and a.max() >= 0.5:
                    entry["band"][slot] = {
                        "kind": r.kind,
                        "alphas": [round(float(v), 4) for v in rows[:, 3]],
                        "source": str(r.fields.get("source"))[:160],
                    }
        if any(r.fields.get("facing_group") for _, r in delta.set):
            entry["facing"] = True
        if sent is not None:
            made, core = sent
            full = bytes(core.to_bytes())
            bare = bytes(to_core(_stripped(made)).to_bytes())
            entry["deltas"].append([len(full), len(bare), _sha(bare)])
            entry["core"].append(core)
        entry["frames"].append(_sha(np.ascontiguousarray(frame).tobytes()))
        if KEEP:
            entry["arrays"].append(np.asarray(frame).copy())
            plan = []
            if entry["facing"] and hasattr(self.core, "label_plan"):
                h, w = np.asarray(frame).shape[:2]
                plan = [list(p) for p in self.core.label_plan(w, h)]
            entry["plans"].append(plan)
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
        # No control frame is read: the recording replaces the comparison. The Cairo
        # control file is noted, for the comparison against CE.
        control_path = fc._control_data_path

        def unread(test_file_path, module_name, test_name, setting_test, *_a, **_kw):
            STATE["control"] = control_path(
                test_file_path, module_name, test_name, setting_test, draw_path="cairo"
            )
            return Path("unread.npz")

        fc._control_data_path = unread

        @contextlib.contextmanager
        def testing(self):
            yield

        def check_frame(self, frame_number, frame):
            STATE["frames"].append(_sha(np.ascontiguousarray(frame).tobytes()))
            if KEEP:
                STATE["arrays"].append(np.asarray(frame).copy())

        _frames_testers._FramesTester.testing = testing
        _frames_testers._FramesTester.check_frame = check_frame


def pytest_configure(config):
    _install(os.environ.get("C53_FORCE_CORE") == "1")
    (OUT / "xf").mkdir(parents=True, exist_ok=True)


def pytest_runtest_setup(item):
    STATE["case"] = item.nodeid
    STATE["painters"] = {}
    STATE["frames"] = []
    STATE["arrays"] = []
    STATE["control"] = None


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
    entries = sorted(STATE["painters"].values(), key=lambda e: e["index"])
    facing = any(e["facing"] for e in entries)
    corpus = "test_c53_corpus" in case
    if KEEP and (facing or corpus):
        (OUT / "frames").mkdir(exist_ok=True)
        kept = {}
        for e in entries:
            if e["arrays"]:
                kept[f"core{e['index']}"] = np.stack(e["arrays"])
        if STATE["arrays"]:
            kept["scene"] = np.stack(STATE["arrays"])
        control = STATE["control"]
        if control is not None and Path(control).exists():
            with np.load(control) as data:
                kept["control"] = np.asarray(data["frame_data"])
        if kept:
            np.savez_compressed(OUT / "frames" / f"{name}.npz", **kept)
        plans = {str(e["index"]): e["plans"] for e in entries if e["facing"]}
        with open(OUT / f"plans-{os.getpid()}.jsonl", "a") as f:
            f.write(json.dumps({"case": case, "name": name, "plans": plans}) + "\n")
    faint = sum(len(e["faint"]) for e in entries if e["facing"])
    if facing:
        with open(OUT / f"census-{os.getpid()}.jsonl", "a") as f:
            band, blends = {}, {}
            for e in entries:
                if e["facing"]:
                    band.update(e["band"])
                    blends.update({(e["index"], k): v for k, v in e["blends"].items()})
            f.write(
                json.dumps(
                    {
                        "case": case,
                        "faint_gradient_records": faint,
                        "band_capable": len(band),
                        "examples": list(band.values())[:3],
                        "blend_records": len(blends),
                        "blend_band_capable": sum(
                            1 for v in blends.values() if v["band_capable"]
                        ),
                        "blend_examples": list(blends.values())[:3],
                    }
                )
                + "\n"
            )
    every = {}
    for e in entries:
        every.update({(e["index"], k): v for k, v in e["blends"].items()})
    if every:
        with open(OUT / f"blends-{os.getpid()}.jsonl", "a") as f:
            f.write(
                json.dumps(
                    {
                        "case": case,
                        "facing": facing,
                        "blend_records": len(every),
                        "blend_sets": sum(e["blend_sets"] for e in entries),
                        "band_capable": sum(1 for v in every.values() if v["band_capable"]),
                        "example": next(iter(every.values())),
                        "band_log": [x for e in entries for x in e["band_log"]],
                    }
                )
                + "\n"
            )
    line = {
        "case": case,
        "frames": STATE["frames"],
        "painters": painters,
        "facing": facing,
        "name": name,
    }
    with open(OUT / f"cases-{os.getpid()}.jsonl", "a") as f:
        f.write(json.dumps(line) + "\n")
    STATE["case"] = None
    STATE["painters"] = {}
    STATE["frames"] = []
