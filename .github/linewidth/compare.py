"""Scratch, never merged: per-frame differences between the renders, and crops of the frames
where the core and Manim CE differ most.

    python compare.py <work> <out>
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
SCENES = ["MidPlayStateSolids", "SteadyProjectingSolids"]
RUNS = ["new-core", "old-core", "new-cairo", "old-cairo", "ce"]
PAIRS = [
    ("new-core", "ce"),
    ("old-core", "ce"),
    ("new-core", "old-core"),
    ("new-cairo", "ce"),
    ("new-cairo", "new-core"),
    ("old-cairo", "old-core"),
]
HALF = 16


def save(a: np.ndarray, path: Path, scale: int = 1) -> None:
    img = Image.fromarray(np.ascontiguousarray(a.astype(np.uint8)), "RGBA")
    if scale > 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    img.save(path)


def absdiff(a, b):
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).astype(np.uint8)
    d[..., 3] = 255
    return d


def gained(d):
    peak = int(d[..., :3].max())
    g = d.copy()
    if peak:
        g[..., :3] = np.clip(d[..., :3].astype(np.int32) * (255 // peak), 0, 255)
    return g


report = {}
for scene in SCENES:
    frames, logs = {}, {}
    for run in RUNS:
        p = work / f"{scene}-{run}.npz"
        if p.exists():
            frames[run] = np.load(p)["frames"]
            logs[run] = json.loads(p.with_suffix(".json").read_text())
    entry = {"frames": {r: len(f) for r, f in frames.items()}, "logs": logs, "pairs": {}}
    for a, b in PAIRS:
        if a not in frames or b not in frames:
            continue
        n = min(len(frames[a]), len(frames[b]))
        rows = []
        for i in range(n):
            d = np.abs(frames[a][i].astype(int) - frames[b][i].astype(int))[..., :3]
            rows.append([i, round(float(d.mean()), 4), int((d.max(axis=2) > 0).sum()),
                         int(d.max())])
        entry["pairs"][f"{a} vs {b}"] = rows
    if scene == "MidPlayStateSolids" and "new-core" in frames and "ce" in frames:
        rows = entry["pairs"]["new-core vs ce"]
        ranked = sorted(rows, key=lambda r: -r[1])
        first = [r[0] for r in rows if r[1] > 0.5][:1]
        chosen = sorted({ranked[0][0], ranked[1][0], *first})
        crops = []
        for i in chosen:
            a, c = frames["new-core"][i], frames["ce"][i]
            d = absdiff(a, c)[..., :3].sum(axis=2).astype(np.int64)
            h, w = d.shape
            s = d.cumsum(0).cumsum(1)
            s = np.pad(s, ((1, 0), (1, 0)))
            k = 2 * HALF
            best, at = -1, (0, 0)
            for y in range(0, h - k + 1):
                for x in range(0, w - k + 1):
                    v = s[y + k, x + k] - s[y, x + k] - s[y + k, x] + s[y, x]
                    if v > best:
                        best, at = v, (x, y)
            x0, y0 = at
            x1, y1 = x0 + k, y0 + k
            stem = f"{scene}_f{i}"
            for run in ("new-core", "old-core", "new-cairo", "ce"):
                if run in frames and i < len(frames[run]):
                    save(frames[run][i][y0:y1, x0:x1], out / f"{stem}_{run}.png")
                    save(frames[run][i][y0:y1, x0:x1], out / f"{stem}_{run}_x8.png", 8)
            dd = absdiff(a, c)[y0:y1, x0:x1]
            save(dd, out / f"{stem}_diff_newcore_ce.png")
            save(dd, out / f"{stem}_diff_newcore_ce_x8.png", 8)
            save(gained(dd), out / f"{stem}_diff_newcore_ce_gain_x8.png", 8)
            for run in ("new-core", "ce"):
                full = Image.fromarray(np.ascontiguousarray(frames[run][i]), "RGBA")
                full = full.resize((w * 4, h * 4), Image.NEAREST)
                ImageDraw.Draw(full).rectangle(
                    [x0 * 4, y0 * 4, x1 * 4 - 1, y1 * 4 - 1], outline=(255, 0, 0, 255)
                )
                full.save(out / f"{stem}_context_{run}_x4.png")
                save(frames[run][i], out / f"{stem}_full_{run}.png")
            crops.append({"frame": i, "box": [x0, y0, x1, y1], "diff_sum": int(best)})
        entry["crops"] = crops
    report[scene] = entry
(out / "report.json").write_text(json.dumps(report, indent=1))
for scene, e in report.items():
    print("==", scene, e["frames"])
    for pair, rows in e["pairs"].items():
        print(" ", pair, [(r[0], r[1]) for r in rows])
    print("  crops", e.get("crops"))
