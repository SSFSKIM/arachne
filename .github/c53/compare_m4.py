"""Label placement's evidence from three runs of the corpus: this revision with
placement on, with it off, and the base revision.

    python compare_m4.py <on> <off> <base> <summary.json>

Identity: with placement off every frame is the base's; with it on, every frame
of a case that draws no camera-facing label is. Before/after: for each case that
draws one, the pixels where the core's frames differ from CE's (a scene's
ir_cairo frames, a graphical unit's Cairo control frames) by more than the
graphical units' tolerance, before (base) and after (on), and the pixels each
side alone gets wrong; with the labels' plan outcomes that case's frames
counted.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

TOLERANCE = 1.01


def load(root: Path, pattern: str) -> dict[str, dict]:
    found = {}
    for path in sorted(root.glob(pattern)):
        for line in path.read_text().splitlines():
            entry = json.loads(line)
            found[entry["case"]] = entry
    return found


def drawn(entry) -> list:
    return [entry["frames"]] + [p["frames"] for p in entry["painters"]]


def frames(root: Path, name: str):
    path = root / "frames" / f"{name}.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        return {k: np.asarray(data[k]) for k in data.files}


def wrong(core: np.ndarray, ce: np.ndarray) -> np.ndarray:
    """Per pixel, whether any channel differs by more than the tolerance."""
    return (np.abs(core.astype(np.int32) - ce.astype(np.int32)) > TOLERANCE).any(axis=-1)


def main() -> int:
    on, off, base = (Path(p) for p in sys.argv[1:4])
    a, b, c = load(on, "cases-*.jsonl"), load(off, "cases-*.jsonl"), load(base, "cases-*.jsonl")
    plans = load(on, "plans-*.jsonl")
    common = sorted(set(a) & set(b) & set(c))
    off_differ = [k for k in common if drawn(b[k]) != drawn(c[k])]
    on_differ = [k for k in common if drawn(a[k]) != drawn(c[k])]
    facing = sorted(k for k in common if a[k].get("facing"))
    unlabelled_differ = [k for k in on_differ if k not in facing]
    print(f"cases: {len(common)} common ({len(a)} on, {len(b)} off, {len(c)} base)")
    print(f"placement off against the base: {len(off_differ)} cases draw differently")
    for k in off_differ:
        print("   ", k)
    print(f"placement on against the base: {len(on_differ)} cases draw differently, "
          f"{len(unlabelled_differ)} of them without a facing group")
    for k in unlabelled_differ:
        print("    UNLABELLED", k)
    print(f"cases drawing a camera-facing label: {len(facing)}")
    rows = []
    for k in facing:
        name = a[k]["name"]
        mine, theirs = frames(on, name), frames(base, name)
        if mine is None or theirs is None:
            rows.append({"case": k, "error": "frames not kept"})
            continue
        if "control" in mine:
            ce, after, before = mine["control"], mine.get("scene"), theirs.get("scene")
        else:
            twins = [k.replace("[core-", "[ir_cairo-"), k.replace("-core]", "-ir_cairo]")]
            twin = next((t for t in twins if t != k and t in a), None)
            if twin is None:
                rows.append({"case": k, "error": "no CE twin"})
                continue
            ce_frames = frames(on, a[twin]["name"])
            ce = None if ce_frames is None else ce_frames.get("scene")
            after, before = mine.get("scene"), theirs.get("scene")
        if ce is None or after is None or before is None:
            rows.append({"case": k, "error": "missing frames"})
            continue
        n = min(len(ce), len(after), len(before))
        if n == 0:
            rows.append({"case": k, "error": "no frames"})
            continue
        ce, after, before = ce[-n:], after[-n:], before[-n:]
        if ce.shape != after.shape or ce.shape != before.shape:
            rows.append({"case": k, "error": f"shapes {ce.shape} {after.shape} {before.shape}"})
            continue
        bad_after, bad_before = wrong(after, ce), wrong(before, ce)
        tally = collections.Counter()
        for painter in plans.get(k, {}).get("plans", {}).values():
            for frame in painter:
                for pair in frame:
                    outcome, cause = pair[4], pair[5]
                    tally[f"{outcome}:{cause}" if cause else outcome] += 1
        rows.append(
            {
                "case": k,
                "frames": int(n),
                "before": int(bad_before.sum()),
                "after": int(bad_after.sum()),
                "newly_wrong": int((bad_after & ~bad_before).sum()),
                "newly_right": int((bad_before & ~bad_after).sum()),
                "changed": int(wrong(after, before).sum()),
                "plan": dict(tally),
            }
        )
    print("--- before/after against CE, every case drawing a facing group")
    print(f"{'before':>8} {'after':>8} {'+wrong':>8} {'+right':>8} {'changed':>8}  case  plan")
    for r in rows:
        if "error" in r:
            print(f"{'':>44}  {r['case']}  ERROR {r['error']}")
            continue
        print(
            f"{r['before']:>8} {r['after']:>8} {r['newly_wrong']:>8} {r['newly_right']:>8} "
            f"{r['changed']:>8}  {r['case']}  {r['plan']}"
        )
    worse = [r for r in rows if "error" not in r and r["after"] > r["before"]]
    print(f"worse after: {len(worse)} of {len(rows)}")
    total = collections.Counter()
    for r in rows:
        total.update(r.get("plan", {}))
    print(f"plan outcomes over every frame: {dict(total)}")
    Path(sys.argv[4]).write_text(
        json.dumps(
            {
                "off_differ": off_differ,
                "on_differ": on_differ,
                "unlabelled_differ": unlabelled_differ,
                "rows": rows,
                "outcomes": dict(total),
            },
            indent=1,
        )
    )
    return 1 if off_differ or unlabelled_differ else 0


if __name__ == "__main__":
    sys.exit(main())
