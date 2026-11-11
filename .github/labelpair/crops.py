"""Scratch, never merged: crops around one pixel of chosen core draws, core (placement on and
off) beside Manim CE at the same frame, with their differences and a full frame marking the box.

    python crops.py <work> <out>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

work = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)

# scene: (painter, seqs, pixel x, pixel y)
TARGETS = {
    "SteadyProjectingSolids": (0, [16, 18, 19, 20], 123, 51),
    "ShiftedSolids": (0, [21, 23, 24, 25], 109, 47),
    "PatchedSolids": (0, [16, 18, 19, 20], 136, 56),
    "MidPlayStateSolids": (0, [16, 18, 19, 20], 123, 51),
}
HALF = 12


def save(a: np.ndarray, path: Path, scale: int = 1) -> None:
    img = Image.fromarray(np.ascontiguousarray(a.astype(np.uint8)), "RGBA")
    if scale > 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    img.save(path)


def absdiff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).astype(np.uint8)
    d[..., 3] = 255
    return d


def gained(d: np.ndarray) -> np.ndarray:
    peak = int(d[..., :3].max())
    g = d.copy()
    if peak:
        g[..., :3] = np.clip(d[..., :3].astype(np.int32) * (255 // peak), 0, 255)
    return g


report = {}
for scene, (painter, seqs, px, py) in TARGETS.items():
    runs = {}
    for mode in ("on", "off"):
        data = np.load(work / f"{scene}-{mode}.npz")
        meta = json.loads((work / f"{scene}-{mode}.json").read_text())
        runs[mode] = (data["scene"], data["draws"], meta)
    ce = np.load(work / f"{scene}-ce.npz")["scene"]
    cenl = np.load(work / f"{scene}-cenl.npz")["scene"]
    on_scene, on_draws, on_meta = runs["on"]
    off_scene, off_draws, off_meta = runs["off"]
    entry = {
        "frames": {"core_on": len(on_scene), "core_off": len(off_scene), "ce": len(ce)},
        "targets": [],
    }
    for seq in seqs:
        found = [
            i for i, d in enumerate(on_meta) if d["painter"] == painter and d["seq"] == seq
        ]
        if not found:
            entry["targets"].append({"seq": seq, "error": "no such draw"})
            continue
        k = found[0]
        d = on_meta[k]
        on = on_draws[k]
        koff = [
            i for i, e in enumerate(off_meta) if e["painter"] == painter and e["seq"] == seq
        ]
        off = off_draws[koff[0]] if koff else None
        index = d["scene_index"][0] if d["scene_index"] else None
        h, w = on.shape[:2]
        x0, y0 = max(px - HALF, 0), max(py - HALF, 0)
        x1, y1 = min(px + HALF, w), min(py + HALF, h)
        near = {}
        if index is not None:
            for j in range(index - 2, index + 3):
                if 0 <= j < len(ce):
                    near[j] = float(np.abs(ce[j].astype(int) - on.astype(int)).mean())
        best = min(near, key=near.get) if near else None
        stem = f"{scene}_seq{seq}"
        crop = lambda a: a[y0:y1, x0:x1]  # noqa: E731
        save(crop(on), out / f"{stem}_on.png")
        save(crop(on), out / f"{stem}_on_x8.png", 8)
        t = {
            "seq": seq,
            "draw": k,
            "frame_index": index,
            "box": [x0, y0, x1, y1],
            "pixel": [px, py],
            "plan_on": d["plan"],
            "plan_off": off_meta[koff[0]]["plan"] if off is not None else None,
            "records": d["records"],
            "on_px": on[py, px].tolist(),
            "ce_mean_abs_by_index": near,
            "ce_index": best,
        }
        if off is not None:
            save(crop(off), out / f"{stem}_off.png")
            save(crop(off), out / f"{stem}_off_x8.png", 8)
            dd = absdiff(on, off)
            save(crop(dd), out / f"{stem}_diff_on_off_x8.png", 8)
            t["off_px"] = off[py, px].tolist()
            t["on_off_px_differ_frame"] = int((dd[..., :3].max(axis=2) > 0).sum())
            t["on_off_px_differ_box"] = int((crop(dd)[..., :3].max(axis=2) > 0).sum())
        for label, j in (("ce", index), ("cebest", best)):
            if j is None or j >= len(ce) or (label == "cebest" and j == index):
                continue
            c = ce[j]
            dc = absdiff(on, c)
            save(crop(c), out / f"{stem}_{label}{j}.png")
            save(crop(c), out / f"{stem}_{label}{j}_x8.png", 8)
            save(crop(dc), out / f"{stem}_diff_on_{label}{j}.png")
            save(crop(dc), out / f"{stem}_diff_on_{label}{j}_x8.png", 8)
            save(crop(gained(dc)), out / f"{stem}_diff_on_{label}{j}_gain_x8.png", 8)
            t[f"{label}{j}_px"] = c[py, px].tolist()
            t[f"on_{label}{j}_max_box"] = int(crop(dc)[..., :3].max())
            t[f"on_{label}{j}_differ_box"] = int((crop(dc)[..., :3].max(axis=2) > 0).sum())
            t[f"on_{label}{j}_differ_frame"] = int((dc[..., :3].max(axis=2) > 0).sum())
        if index is not None and index < len(cenl):
            n = cenl[index]
            save(crop(n), out / f"{stem}_ce_nolabel{index}_x8.png", 8)
            t["ce_nolabel_px"] = n[py, px].tolist()
            if index < len(ce):
                t["ce_vs_nolabel_differ_box"] = int(
                    (crop(absdiff(ce[index], n))[..., :3].max(axis=2) > 0).sum()
                )
        full = Image.fromarray(np.ascontiguousarray(on), "RGBA").resize((w * 4, h * 4), Image.NEAREST)
        ImageDraw.Draw(full).rectangle([x0 * 4, y0 * 4, x1 * 4 - 1, y1 * 4 - 1], outline=(255, 0, 0, 255))
        ImageDraw.Draw(full).rectangle([px * 4, py * 4, px * 4 + 3, py * 4 + 3], outline=(255, 0, 255, 255))
        full.save(out / f"{stem}_context_on_x4.png")
        entry["targets"].append(t)
    report[scene] = entry
(out / "report.json").write_text(json.dumps(report, indent=1))
print(json.dumps({s: [(t.get("seq"), t.get("frame_index"), t.get("ce_index"), t.get("plan_on"), t.get("on_px"), t.get("off_px")) for t in e["targets"]] for s, e in report.items()}))
