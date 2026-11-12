"""Compares what two revisions drew and sent for the same cases, and measures
the newer revision's SOURCE bytes against its whole XF streams.

    python compare.py <head out dir> <base out dir> <old-reader json> <summary.json>

Exit 1 when any frame, any stripped stream byte or any old-reader frame differs.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def load(root: Path, pattern: str) -> dict[str, dict]:
    found = {}
    for path in sorted(root.glob(pattern)):
        for line in path.read_text().splitlines():
            entry = json.loads(line)
            found[entry["case"]] = entry
    return found


def main() -> int:
    head, base = Path(sys.argv[1]), Path(sys.argv[2])
    reader = json.loads(Path(sys.argv[3]).read_text())
    a, b = load(head, "cases-*.jsonl"), load(base, "cases-*.jsonl")
    ea, eb = load(head, "errors-*.jsonl"), load(base, "errors-*.jsonl")
    common = sorted(set(a) & set(b))
    frames = core_frames = deltas = 0
    differ_frames, differ_core, differ_wire = [], [], []
    streams = []
    for case in common:
        x, y = a[case], b[case]
        frames += len(x["frames"])
        if x["frames"] != y["frames"]:
            differ_frames.append(case)
        if len(x["painters"]) != len(y["painters"]):
            differ_core.append(f"{case}: {len(x['painters'])} painters vs {len(y['painters'])}")
            continue
        for k, (p, q) in enumerate(zip(x["painters"], y["painters"], strict=True)):
            name = f"{case}#{k}"
            core_frames += len(p["frames"])
            deltas += len(p["deltas"])
            if p["frames"] != q["frames"]:
                differ_core.append(name)
            # The newer stream without its SOURCE is the older stream, byte for byte.
            if [d[2] for d in p["deltas"]] != [d[2] for d in q["deltas"]] or [
                d[1] for d in p["deltas"]
            ] != [d[0] for d in q["deltas"]]:
                differ_wire.append(name)
            total = sum(d[0] for d in p["deltas"])
            bare = sum(d[1] for d in p["deltas"])
            older = sum(d[0] for d in q["deltas"])
            if total:
                streams.append(
                    {
                        "stream": name,
                        "total": total,
                        "source": total - bare,
                        "older": older,
                        "deltas": len(p["deltas"]),
                        "percent": 100.0 * (total - bare) / total,
                    }
                )
    errors_differ = sorted(
        c for c in set(ea) | set(eb) if ea.get(c, {}).get("error") != eb.get(c, {}).get("error")
    )
    pcts = [s["percent"] for s in streams]
    worst = max(streams, key=lambda s: s["percent"]) if streams else None
    labelled = [s for s in streams if s["source"]]
    summary = {
        "cases": {"head": len(a), "base": len(b), "common": len(common)},
        "only_head": sorted(set(a) - set(b))[:20],
        "only_base": sorted(set(b) - set(a))[:20],
        "errors": {"head": len(ea), "base": len(eb), "differ": errors_differ[:20]},
        "frames_compared": frames,
        "core_frames_compared": core_frames,
        "deltas_compared": deltas,
        "differ": {
            "frames": differ_frames[:20],
            "core_frames": differ_core[:20],
            "stripped_wire": differ_wire[:20],
            "counts": [len(differ_frames), len(differ_core), len(differ_wire)],
        },
        "source_percent": {
            "streams": len(streams),
            "median": statistics.median(pcts) if pcts else None,
            "max": max(pcts) if pcts else None,
            "worst": worst,
            "total_bytes_with_source": sum(s["total"] for s in streams),
            "total_source_bytes": sum(s["source"] for s in streams),
            "total_bytes_older_revision": sum(s["older"] for s in streams),
        },
        "top10": sorted(streams, key=lambda s: -s["percent"])[:10],
        "streams_carrying_source": len(labelled),
        "old_reader": {k: v for k, v in reader.items() if k != "differ"},
    }
    Path(sys.argv[4]).write_text(json.dumps({**summary, "all": streams}, indent=1))
    print(json.dumps(summary, indent=1))
    bad = differ_frames or differ_core or differ_wire or errors_differ or not reader["same"]
    print("IDENTITY", "FAIL" if bad else "OK")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
