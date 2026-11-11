"""Every record facing the camera carries a facing group, in every frame of every corpus stream,
with a planted control proving the same sweep goes RED.

    python facing_sweep.py <out dir>

<out dir> is what the c53_probe plugin recorded (cases-*.jsonl and xf/). The sweep reads each
stream with this revision's core and, applying its deltas as the core does (records set, records
removed, then held records' source rows), checks each record a delta sets or gives a source row:
when it keeps ``fixed_orientation_centre``, its facing group after the delta must not be 0.

The control: the stream, frame and slot rule below are fixed here, before any run. The stream is
read again, copied, the resolved record's facing group zeroed on the copy alone, and the same
sweep must name exactly that record.

Exit 0 only when all EXPECTED_STREAMS streams are read, none has a record of group 0, and the
control goes RED on exactly the planted record.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import arachne_core
import numpy as np

#: The corpus streams the probe records at this revision's engine.
EXPECTED_STREAMS = 551
#: The planted control's stream: a fixed-orientation label case of the scene corpus on the core,
#: the case's first painter.
PLANT_CASE = (
    "tests/frame_ir/test_c53_corpus.py::test_scene"
    "[tests.frame_ir.probes.scenes:FixedLabelOverSurface-core]"
)
PLANT_PAINTER = 0
PLANT_FRAME = 0
PLANT_RULE = "the lowest-slot record set in that frame with fixed_orientation_centre"


def sweep(deltas: list[dict]) -> tuple[dict[str, int], list[tuple[int, int, str]]]:
    """Counts and findings (frame, slot, how) of one stream's deltas, as ``contents()`` reads them."""
    held: dict[int, list] = {}  # slot -> [facing the camera, facing group]
    counts = {"frames": 0, "set": 0, "fo_set": 0, "fo_held_rows": 0}
    findings = []
    for frame, delta in enumerate(deltas):
        counts["frames"] += 1
        touched: dict[int, str] = {}
        for _kind, slot, _object, fields in delta["set"]:
            counts["set"] += 1
            held[slot] = [
                fields.get("fixed_orientation_centre") is not None,
                fields.get("facing_group", 0),
            ]
            touched[slot] = "set"
        for slot in delta["remove"]:
            held.pop(slot, None)
            touched.pop(slot, None)
        for slot, _source, facing_group in delta["source"]:
            held[slot][1] = facing_group
            touched.setdefault(slot, "source row")
        for slot, how in sorted(touched.items()):
            facing, group = held[slot]
            if not facing:
                continue
            counts["fo_set" if how == "set" else "fo_held_rows"] += 1
            if group == 0:
                findings.append((frame, slot, how))
    return counts, findings


def read(path: Path) -> list[dict]:
    return [delta.contents() for delta in arachne_core.read_xf(path)]


def differences(a, b, path=()) -> list[tuple]:
    """Every place two ``contents()`` values differ."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape or a.dtype != b.dtype:
            return [path]
        try:
            return [] if np.array_equal(a, b, equal_nan=True) else [path]
        except TypeError:
            return [] if np.array_equal(a, b) else [path]
    if isinstance(a, dict) and isinstance(b, dict):
        found = []
        for key in sorted(set(a) | set(b), key=str):
            if key not in a or key not in b:
                found.append((*path, key))
            else:
                found.extend(differences(a[key], b[key], (*path, key)))
        return found
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            return [path]
        found = []
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            found.extend(differences(x, y, (*path, i)))
        return found
    if isinstance(a, float) and isinstance(b, float) and a != a and b != b:
        return []
    return [] if type(a) is type(b) and a == b else [path]


def streams(root: Path) -> dict[str, str]:
    """Each stream file the cases name, and its case and painter."""
    found = {}
    for path in sorted(root.glob("cases-*.jsonl")):
        for line in path.read_text().splitlines():
            entry = json.loads(line)
            for k, painter in enumerate(entry["painters"]):
                if painter["xf"]:
                    found[painter["xf"]] = f"{entry['case']}#{k}"
    return found


def plant(root: Path, named: dict[str, str]) -> list[str]:
    """The control; returns why it failed, empty when it went RED on exactly the planted record."""
    stream = f"{PLANT_CASE}#{PLANT_PAINTER}"
    files = [name for name, case in named.items() if case == stream]
    print(f"plant: stream {stream}")
    print(f"plant: frame {PLANT_FRAME}, {PLANT_RULE}")
    if len(files) != 1:
        return [f"plant: the stream is recorded {len(files)} times, not once"]
    print(f"plant: file {files[0]}")
    original = read(root / "xf" / files[0])
    if len(original) <= PLANT_FRAME:
        return [
            f"plant: the stream has {len(original)} frames, none is frame {PLANT_FRAME}"
        ]
    candidates = sorted(
        (slot, i)
        for i, (_kind, slot, _object, fields) in enumerate(original[PLANT_FRAME]["set"])
        if fields.get("fixed_orientation_centre") is not None
    )
    if not candidates:
        return [
            f"plant: frame {PLANT_FRAME} sets no record with fixed_orientation_centre"
        ]
    slot, index = candidates[0]
    kind, _, _, fields = original[PLANT_FRAME]["set"][index]
    group = fields.get("facing_group", 0)
    print(
        f"plant: resolved slot {slot} (set entry {index}, kind {kind}, of {len(candidates)} "
        f"facing records set in the frame), facing_group {group} before the plant"
    )
    if group == 0:
        return [
            "plant: the resolved record's facing group is already 0 (a sweep finding)"
        ]
    planted = copy.deepcopy(original)
    planted[PLANT_FRAME]["set"][index][3]["facing_group"] = 0
    changed = differences(original, planted)
    expected = [(PLANT_FRAME, "set", index, 3, "facing_group")]
    print(f"plant: the copy differs from the original at {changed}")
    failures = []
    if changed != expected:
        failures.append(f"plant: the copy differs at {changed}, not only at {expected}")
    _, red = sweep(planted)
    for frame, s, how in red:
        print(f"  RED {stream} ({files[0]}) frame {frame} slot {s} ({how}): group 0")
    if red != [(PLANT_FRAME, slot, "set")]:
        failures.append(
            f"plant: the sweep named {red}, not exactly frame {PLANT_FRAME} slot {slot}"
        )
    else:
        print(
            f"plant: RED on exactly the planted record, frame {PLANT_FRAME} slot {slot}"
        )
    return failures


def main() -> int:
    root = Path(sys.argv[1])
    print(f"reader {arachne_core.XF_VERSION}")
    print(
        f"plant target, fixed before the run: {PLANT_CASE}#{PLANT_PAINTER}, frame {PLANT_FRAME}, "
        f"{PLANT_RULE}"
    )
    named = streams(root)
    on_disk = sorted(p.name for p in (root / "xf").glob("*.xf"))
    cases = sum(len(p.read_text().splitlines()) for p in root.glob("cases-*.jsonl"))
    failures = []
    unnamed = sorted(set(on_disk) - set(named))
    missing = sorted(set(named) - set(on_disk))
    if unnamed or missing:
        failures.append(
            f"{len(unnamed)} stream files no case names, {len(missing)} named but absent"
        )
    totals = {"frames": 0, "set": 0, "fo_set": 0, "fo_held_rows": 0}
    zero, unread = [], []
    for name in sorted(named):
        try:
            counts, findings = sweep(read(root / "xf" / name))
        except Exception as error:  # an unread stream fails the gate
            unread.append(
                f"{named[name]} ({name}): {type(error).__name__}: {str(error)[:160]}"
            )
            continue
        for k, v in counts.items():
            totals[k] += v
        zero.extend(
            (named[name], name, frame, slot, how) for frame, slot, how in findings
        )
    read_ok = len(named) - len(unread)
    print(
        f"facing sweep: streams read {read_ok}/{EXPECTED_STREAMS} ({len(named)} named by "
        f"{cases} cases, {len(on_disk)} files)"
    )
    print(f"facing sweep: frames {totals['frames']}, records set {totals['set']}")
    print(
        f"facing sweep: facing records checked {totals['fo_set'] + totals['fo_held_rows']} "
        f"({totals['fo_set']} set, {totals['fo_held_rows']} given a source row), "
        f"group 0: {len(zero)}"
    )
    for case, name, frame, slot, how in zero:
        print(f"  GROUP 0 {case} ({name}) frame {frame} slot {slot} ({how})")
    for line in unread:
        print(f"  UNREAD {line}")
    if read_ok != EXPECTED_STREAMS or len(named) != EXPECTED_STREAMS:
        failures.append(f"streams read {read_ok}/{EXPECTED_STREAMS}")
    if zero:
        failures.append(f"{len(zero)} facing records of group 0")
    failures.extend(plant(root, named))
    for failure in failures:
        print("FAIL:", failure)
    print("FACING SWEEP", "FAIL" if failures else "OK")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
