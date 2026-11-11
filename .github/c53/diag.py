"""Where a newer stream without its SOURCE first differs from the older one.

    python diag.py <head out dir> <base out dir> <old-reader json> [name substring ...]

Reads both revisions' XF files with this (the newer) core, removes ``source``
and ``facing_group`` from every record, and reports per stream the first delta
and part that differ, counts by part, and details for the first few streams.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import arachne_core
import numpy as np

STRIP = ("source", "facing_group")


def same(a, b) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def bare(record: dict) -> dict:
    return {k: v for k, v in record.items() if k not in STRIP}


def short(value, n=160) -> str:
    text = repr(value)
    return text if len(text) <= n else text[:n] + "..."


def cases(root: Path) -> dict[str, str]:
    found = {}
    for path in root.glob("cases-*.jsonl"):
        for line in path.read_text().splitlines():
            entry = json.loads(line)
            for k, p in enumerate(entry["painters"]):
                if p["xf"]:
                    found[p["xf"]] = f"{entry['case']}#{k}"
    return found


def sets(contents) -> list[tuple[int, dict]]:
    return [(slot, {"kind": kind, **fields}) for kind, slot, _, fields in contents["set"]]


def compare(new, old) -> tuple[str, str] | None:
    if len(new) != len(old):
        return "delta count", f"{len(new)} vs {len(old)}"
    for i, (a, b) in enumerate(zip(new, old, strict=True)):
        ca, cb = a.contents(), b.contents()
        for part in ("t", "seq", "view", "objects", "remove", "release", "order"):
            if not same(ca.get(part), cb.get(part)):
                return f"{part}", f"delta {i}: {short(ca.get(part))} vs {short(cb.get(part))}"
        for part in ("upload", "upload_paints"):
            if not same(ca.get(part), cb.get(part)):
                return part, f"delta {i}: {len(ca.get(part) or [])} vs {len(cb.get(part) or [])}"
        sa = [(s, bare(r)) for s, r in sets(ca)]
        sb = [(s, bare(r)) for s, r in sets(cb)]
        slots_a, slots_b = [s for s, _ in sa], [s for s, _ in sb]
        if slots_a != slots_b:
            extra = sorted(set(slots_a) - set(slots_b))
            missing = sorted(set(slots_b) - set(slots_a))
            detail = f"delta {i}: newer sets {extra[:8]} more, {missing[:8]} fewer"
            if extra:
                slot = extra[0]
                rec = dict(sets(ca))[slot]
                detail += f"; slot {slot} kind {rec.get('kind')} source {short(rec.get('source'))}"
                prev = [
                    dict(sets(new[j].contents())).get(slot)
                    for j in range(i)
                    if slot in dict(sets(new[j].contents()))
                ]
                if prev:
                    p = prev[-1]
                    diff = [k for k in set(p) | set(rec) if not same(p.get(k), rec.get(k))]
                    detail += f"; differs from its last send in {sorted(diff)}"
                    for k in sorted(diff)[:3]:
                        detail += f"\n      {k}: {short(p.get(k))} -> {short(rec.get(k))}"
            return "set slots", detail
        for (slot, ra), (_, rb) in zip(sa, sb, strict=True):
            if not same(ra, rb):
                keys = [k for k in set(ra) | set(rb) if not same(ra.get(k), rb.get(k))]
                return "set fields", f"delta {i} slot {slot}: {sorted(keys)} " + "; ".join(
                    f"{k}: {short(ra.get(k))} vs {short(rb.get(k))}" for k in sorted(keys)[:3]
                )
    return None


def main() -> None:
    head, base = Path(sys.argv[1]), Path(sys.argv[2])
    reader = json.loads(Path(sys.argv[3]).read_text())
    names = cases(head)
    parts = collections.Counter()
    details = []
    total = 0
    # Held records' new sources: the deltas carrying rows, and those whose
    # only SOURCE content they are, by stream.
    rowed: dict[str, tuple[int, int]] = {}
    for path in sorted((head / "xf").glob("*.xf")):
        other = base / "xf" / path.name
        if not other.exists():
            continue
        total += 1
        newer = arachne_core.read_xf(path)
        carrying = alone = 0
        for delta in newer:
            contents = delta.contents()
            if contents["source"]:
                carrying += 1
                if not any("source" in f or "facing_group" in f for _, f in sets(contents)):
                    alone += 1
        if carrying:
            rowed[path.name] = (carrying, alone)
        found = compare(newer, arachne_core.read_xf(other))
        if found is None:
            continue
        parts[found[0]] += 1
        details.append((names.get(path.name, path.name), *found))
    print(f"streams read: {total}, differing without SOURCE: {len(details)}")
    print("first difference by part:", dict(parts))
    by_part = collections.defaultdict(list)
    for d in details:
        by_part[d[1]].append(d)
    for part, rows in by_part.items():
        print(f"--- {part}: {len(rows)}")
        for name, _, detail in rows[:6]:
            print(f"  {name}\n    {detail}")
    differ = {d["stream"] for d in reader.get("differ", [])}
    print(
        f"--- held rows: {len(rowed)} streams carry them, in "
        f"{sum(c for c, _ in rowed.values())} deltas, "
        f"{sum(a for _, a in rowed.values())} of them their only SOURCE content; "
        f"the 1.4 reader draws {len([n for n in rowed if n in differ])} of these "
        f"streams differently ({reader.get('streams')} streams read)"
    )
    for name, (carrying, alone) in sorted(rowed.items())[:10]:
        print(f"  {names.get(name, name)}: {carrying} deltas, {alone} row-only")
    shown = sys.argv[4:]
    if shown:
        print("--- named streams")
        for name, part, detail in details:
            if any(s in name for s in shown):
                print(f"  {name} [{part}]\n    {detail}")


if __name__ == "__main__":
    main()
