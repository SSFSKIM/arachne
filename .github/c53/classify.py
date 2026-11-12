"""What differs between two revisions' frames, case by case, and what the
newer revision's records hold where it differs: the reading a difference's
class is decided from.

    python classify.py <head out dir> <base out dir> <summary.json>

Run with the newer revision's core. For every case whose frames, core frames
or stripped stream differ, it prints:
- each differing frame's pixels: how many differ (any channel), the greatest
  channel difference, the mean over all channel values, and their box;
- the zero-alpha reading (c53_probe's shadow): where a record that paints
  nothing is held, whether each revision's core draws the frame the same with
  those records absent ("absent"), and whether the older core so drawn equals
  the newer core's frame ("older without = newer");
- the newer stream's records at each differing core frame: how many paint
  nothing, how many carry a drawn stroke under 0.8 px (the thin-stroke series'
  bound; flat mapping), a facing group or a fixed orientation (labels);
- at up to three differing core frames per stream, which records the newer
  core composites at the differing pixels (``Stream.contributors``).

Then, read-only and for every case on each side, its distance from CE: a scene
corpus case's core frames against its own ir_cairo frames, a graphical unit's
against its Cairo control frames (pixels differing, summed levels, per frame);
and each differing pixel of the cases in ``DUMPED``, with the records under it.
"""

from __future__ import annotations

import collections
import hashlib
import json
import sys
from pathlib import Path

import arachne_core
import numpy as np
from c53_probe import paints_nothing, stroke_px

from arachne.frame_ir.xf import Delta, StreamState

THIN = 0.8
LOOKED = 3


def load(root: Path) -> dict[str, dict]:
    found = {}
    for path in sorted(root.glob("cases-*.jsonl")):
        for line in path.read_text().splitlines():
            entry = json.loads(line)
            found[entry["case"]] = entry
    return found


def arrays(root: Path, case: str):
    path = root / "frames" / f"{hashlib.sha256(case.encode()).hexdigest()[:24]}.npz"
    return np.load(path) if path.exists() else {}


def stats(a, b) -> dict:
    if a is None or b is None:
        return {"missing": [a is None, b is None]}
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return {"shape": [list(a.shape), list(b.shape)]}
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    per = d.max(axis=-1) if d.ndim == 3 else d
    ys, xs = np.nonzero(per)
    out = {
        "px": int(len(xs)),
        "max": int(d.max()),
        "mean": round(float(d.mean()), 5),
        "levels": {str(k): int((per == k).sum()) for k in (1, 2, 3)},
        "over3": int((per > 3).sum()),
    }
    if len(xs):
        out["box"] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    return out, per


def get(store, key):
    try:
        return store[key]
    except KeyError:
        return None


def differing(x: list, y: list) -> list[int]:
    return [i for i in range(max(len(x), len(y))) if i >= len(x) or i >= len(y) or x[i] != y[i]]


def facts(state: StreamState) -> tuple[dict, set, set]:
    """Counts over the records held, and the slots with a thin stroke and
    with a label's marks."""
    count = collections.Counter()
    thin, label = set(), set()
    for slot, record in state.records.items():
        count[record.kind] += 1
        f = record.fields
        if paints_nothing(record, state.paints):
            count["paints_nothing"] += 1
        px = stroke_px(record, state.view)
        if px is not None and px < THIN and f.get("stroke") is not None:
            count["thin_stroke"] += 1
            thin.add(slot)
        if f.get("facing_group") or f.get("fixed_orientation_centre") is not None:
            count["label_marks"] += 1
            label.add(slot)
        if f.get("pre_projected") is not None:
            count["pre_projected"] += 1
    return dict(count), thin, label


def contributors(deltas, k: int, per: np.ndarray, view: dict, thin: set, label: set) -> dict:
    stream = arachne_core.Stream()
    for delta in deltas[: k + 1]:
        stream.apply(delta)
    ys, xs = np.nonzero(per)
    found = stream.contributors(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    tally = collections.Counter()
    for x, y in zip(xs.tolist(), ys.tolist(), strict=True):
        leaves = found.get((x, y))
        if leaves is None:
            tally["unread"] += 1
            continue
        slots = {leaf[0] for leaf in leaves[0]}
        if not slots:
            tally["no_leaf"] += 1
            continue
        tally["on_thin_stroke_record"] += bool(slots & thin)
        tally["on_label_record"] += bool(slots & label)
        tally["on_neither"] += not (slots & thin or slots & label)
        tally["crossing"] += bool(leaves[2])
    return dict(tally)


def wire_differs(p: dict, q: dict) -> bool:
    return [d[2] for d in p["deltas"]] != [d[2] for d in q["deltas"]] or [
        d[1] for d in p["deltas"]
    ] != [d[0] for d in q["deltas"]]


def shadow_reading(p: dict, q: dict, ha, ba, index: int, frames: list[int]) -> dict:
    """The zero-alpha reading of one stream at its differing frames."""
    hs = {row[0]: row for row in p.get("shadow") or []}
    bs = {row[0]: row for row in q.get("shadow") or []}
    out = {
        "frames_holding_one": [len(hs), len(bs)],
        "errors": [p.get("shadow_error"), q.get("shadow_error")],
        "kinds": sorted({k for row in bs.values() for k in row[4]}),
    }
    explained, unexplained, none_held, same_core = [], [], [], []
    residue = []
    for k in frames:
        if k not in bs or k not in hs:
            none_held.append(k)
            continue
        _, hr, hsh, n, _ = hs[k]
        _, br, bsh, _, _ = bs[k]
        row = {
            "frame": k,
            "held": n,
            "core_alone_same": br == hr,
            "newer_absent_same": hsh == hr,
            "older_absent_same": bsh == br,
        }
        if br == hr:
            same_core.append(k)
        elif bsh == hr:
            explained.append(k)
        else:
            unexplained.append(k)
            got = stats(get(ba, f"s{index}_{k}"), get(ha, f"r{index}_{k}"))
            row["older_without_vs_newer"] = got[0] if isinstance(got, tuple) else got
        residue.append(row)
    out.update(
        older_without_equals_newer=explained,
        older_without_differs=unexplained,
        no_record_painting_nothing=none_held,
        core_alone_same=same_core,
        rows=residue[:12],
    )
    return out


def main() -> None:
    head, base = Path(sys.argv[1]), Path(sys.argv[2])
    a, b = load(head), load(base)
    common = sorted(set(a) & set(b))
    report = []
    zero_cases = []
    for case in common:
        x, y = a[case], b[case]
        held = [len(p.get("shadow") or []) for p in x["painters"]]
        if any(held):
            zero_cases.append(case)
        frames = differing(x["frames"], y["frames"])
        streams = []
        same_count = len(x["painters"]) == len(y["painters"])
        for k, (p, q) in enumerate(zip(x["painters"], y["painters"])):
            core = differing(p["frames"], q["frames"])
            wire = wire_differs(p, q)
            if core or wire:
                streams.append((k, p, q, core, wire))
        if not frames and not streams and same_count:
            continue
        ha, ba = arrays(head, case), arrays(base, case)
        entry = {"case": case, "frames_total": len(x["frames"]), "painters": len(x["painters"])}
        if not same_count:
            entry["painter_count"] = [len(x["painters"]), len(y["painters"])]
        rows = []
        for i in frames:
            got = stats(get(ha, f"f{i}"), get(ba, f"f{i}"))
            rows.append({"frame": i, **(got[0] if isinstance(got, tuple) else got)})
        entry["frames"] = rows
        entry["streams"] = []
        for k, p, q, core, wire in streams:
            s = {"stream": k, "core_frames_total": len(p["frames"]), "stripped_wire_differs": wire}
            pixel_rows, pers = [], {}
            for j in core:
                got = stats(get(ha, f"p{k}_{j}"), get(ba, f"p{k}_{j}"))
                if isinstance(got, tuple):
                    pixel_rows.append({"frame": j, **got[0]})
                    pers[j] = got[1]
                else:
                    pixel_rows.append({"frame": j, **got})
            s["core_frames"] = pixel_rows
            s["zero_alpha"] = shadow_reading(p, q, ha, ba, k, core)
            try:
                if p["xf"] and core:
                    deltas = arachne_core.read_xf(head / "xf" / p["xf"])
                    state = StreamState()
                    seen = {}
                    for j, delta in enumerate(deltas):
                        if j > max(core):
                            break
                        state.apply(Delta.from_contents(delta.contents()))
                        if j in core:
                            seen[j] = facts(state)
                    s["records"] = {str(j): seen[j][0] for j in core[:LOOKED] if j in seen}
                    most = collections.Counter()
                    for j in seen:
                        for key, n in seen[j][0].items():
                            most[key] = max(most[key], n)
                    s["records_most"] = dict(most)
                    looked = {}
                    for j in [j for j in core if j in pers and pers[j].any()][:LOOKED]:
                        _, thin, label = seen[j]
                        looked[str(j)] = contributors(deltas, j, pers[j], state.view, thin, label)
                    s["pixels_on"] = looked
            except Exception as error:  # the reading is evidence, never the verdict
                s["reading_error"] = f"{type(error).__name__}: {str(error)[:300]}"
            entry["streams"].append(s)
        report.append(entry)

    Path(sys.argv[3]).write_text(json.dumps(report, indent=1))
    print(f"cases compared: {len(common)}; differing: {len(report)}")
    print(f"cases holding a record that paints nothing (newer): {len(zero_cases)}")
    for case in zero_cases:
        print("   ", case)
    for entry in report:
        print("=" * 100)
        print(entry["case"])
        print(json.dumps({k: v for k, v in entry.items() if k not in ("case", "frames", "streams")}))
        for row in entry["frames"]:
            print("  frame", json.dumps(row))
        for s in entry["streams"]:
            print(f"  stream #{s['stream']} ({s['core_frames_total']} frames), stripped wire differs: "
                  f"{s['stripped_wire_differs']}")
            for row in s["core_frames"]:
                print("    core frame", json.dumps(row))
            for key in ("zero_alpha", "records", "records_most", "pixels_on", "reading_error"):
                if key in s:
                    print(f"    {key}:", json.dumps(s[key]))
    print("=" * 100)
    print("summary: case | frames differing/total | px per frame min-max | max level | "
          "core frames differing | older core without zero-alpha records = newer | wire")
    for entry in report:
        rows = entry["frames"] or [r for s in entry["streams"] for r in s["core_frames"]]
        px = [r["px"] for r in rows if "px" in r]
        top = max([r["max"] for r in rows if "max" in r], default=None)
        core = sum(len(s["core_frames"]) for s in entry["streams"])
        za = [
            f"{len(s['zero_alpha']['older_without_equals_newer'])}/{len(s['core_frames'])}"
            for s in entry["streams"]
        ]
        wire = any(s["stripped_wire_differs"] for s in entry["streams"])
        span = f"{min(px)}-{max(px)}" if px else "-"
        print(f"  {entry['case']} | {len(entry['frames'])}/{entry['frames_total']} | {span} | "
              f"{top} | {core} | {','.join(za) or '-'} | {wire}")


# ---------------------------------------------------------------------------------------------
# Read-only: each revision's distance from CE, and one case's differing pixels named.

DUMPED = (":MovedThreeDCentre-core]",)


def distance(a, b) -> tuple:
    """Pixels differing (any channel) and the summed level difference, or why not."""
    if a is None or b is None:
        return ("missing",)
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return ("shape", list(a.shape), list(b.shape))
    d = np.abs(a.astype(np.int32) - b.astype(np.int32))
    per = d.max(axis=-1) if d.ndim == 3 else d
    return (int((per > 0).sum()), int(d.sum()))


def control_frames(case: str):
    """A graphical unit's Cairo control frames (control_data/<module>/<name>.npz)."""
    path, _, test = case.partition("::")
    module = Path(path).stem.removeprefix("test_")
    name = test.split("[")[0].removeprefix("test_")
    found = Path("tests/test_graphical_units/control_data") / module / f"{name}.npz"
    if not found.exists():
        return None
    with np.load(found) as data:
        frames = np.asarray(data["frame_data"])
    return frames[None] if frames.ndim == 3 else frames


def ce_print(label: str, root: Path, cases: dict) -> None:
    """Every core case against CE: a scene corpus case against its own ir_cairo frames, a
    graphical unit against its Cairo control frames; per frame (pixels, summed levels)."""
    print(f"--- CE distance, {label}: case | against | frames | px total | levels total | per frame")
    for case in sorted(cases):
        if case.endswith("-core]"):
            twin = case[: -len("-core]")] + "-ir_cairo]"
            if twin not in cases:
                continue
            mine, theirs = arrays(root, case), arrays(root, twin)
            n = max(len(cases[case]["frames"]), len(cases[twin]["frames"]))
            ref = [get(theirs, f"f{i}") for i in range(n)]
            against = "ir_cairo"
        elif case.endswith("[core]") and "test_graphical_units" in case:
            mine = arrays(root, case)
            control = control_frames(case)
            if control is None:
                continue
            n = max(len(cases[case]["frames"]), len(control))
            ref = [control[i] if i < len(control) else None for i in range(n)]
            against = "control"
        else:
            continue
        rows = [distance(get(mine, f"f{i}"), ref[i]) for i in range(n)]
        whole = [r for r in rows if len(r) == 2]
        px = sum(r[0] for r in whole)
        levels = sum(r[1] for r in whole)
        print(f"  CE {label} | {case} | {against} | {n} | {px} | {levels} | {json.dumps(rows)}")


def pixel_dump(head: Path, base: Path, x: dict, y: dict) -> None:
    """Each differing pixel of a case's core frames: where, both colours, the levels, and the
    newer records the core composites there."""
    case = x["case"]
    ha, ba = arrays(head, case), arrays(base, case)
    print(f"--- pixels of {case}")
    for k, (p, q) in enumerate(zip(x["painters"], y["painters"])):
        core = differing(p["frames"], q["frames"])
        if not core or not p["xf"]:
            continue
        deltas = arachne_core.read_xf(head / "xf" / p["xf"])
        state = StreamState()
        for j, delta in enumerate(deltas):
            if j > max(core):
                break
            state.apply(Delta.from_contents(delta.contents()))
            if j not in core:
                continue
            new, old = get(ha, f"p{k}_{j}"), get(ba, f"p{k}_{j}")
            if new is None or old is None or new.shape != old.shape:
                print(f"  stream {k} frame {j}: frames missing or shaped apart")
                continue
            d = np.abs(new.astype(np.int16) - old.astype(np.int16))
            ys, xs = np.nonzero(d.max(axis=-1))
            if not len(xs):
                continue
            stream = arachne_core.Stream()
            for earlier in deltas[: j + 1]:
                stream.apply(earlier)
            found = stream.contributors(
                int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
            )
            for px, py in zip(xs.tolist(), ys.tolist(), strict=True):
                leaves = found.get((px, py))
                under = []
                for leaf in (leaves[0] if leaves else []):
                    record = state.records.get(leaf[0])
                    f = record.fields if record else {}
                    under.append(
                        {
                            "slot": leaf[0],
                            "part": leaf[1],
                            "coverage": leaf[2],
                            "drawn": list(leaf[4]),
                            "shown": leaf[5],
                            "kind": record.kind if record else None,
                            "stroke_px": stroke_px(record, state.view) if record else None,
                            "has_fill": f.get("fill") is not None,
                            "has_stroke": f.get("stroke") is not None,
                            "opacity": f.get("opacity", 1.0),
                        }
                    )
                print(
                    f"  stream {k} frame {j} ({px}, {py}): newer {new[py, px].tolist()} "
                    f"older {old[py, px].tolist()} levels {d[py, px].tolist()} "
                    f"under {json.dumps(under, default=str)}"
                )


def read_only(head: Path, base: Path) -> None:
    a, b = load(head), load(base)
    for name, root, cases in (("newer", head, a), ("older", base, b)):
        try:
            ce_print(name, root, cases)
        except Exception as error:  # read-only: never the verdict
            print(f"  CE {name} failed: {type(error).__name__}: {str(error)[:300]}")
    for case in sorted(set(a) & set(b)):
        if any(mark in case for mark in DUMPED):
            try:
                pixel_dump(head, base, a[case], b[case])
            except Exception as error:
                print(f"  pixels failed: {type(error).__name__}: {str(error)[:300]}")


if __name__ == "__main__":
    main()
    read_only(Path(sys.argv[1]), Path(sys.argv[2]))
