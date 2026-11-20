"""Pairs of frames just under and just over each threshold, for a person to look at.

Usage::

    python -m arachne_compare.calibration <base 1920x1080 png> <base 480x270 png> <out dir>

From each base frame it makes one pair just under and one just over the cluster
share, the strength, the floor and the edge rule, and three more at 480 x 270: 26
connected pixels changed by 7 with one at 17, a one-pixel stripe just inside a
silhouette's boundary pixels, and a candidate that covers one more pixel outside the
reference's footprint along a 40-pixel edge. Every pair's verdict in each mode is
computed by the comparator and checked against what the pair is built to show. It
writes each pair's two frames and the reference's edge mask as PNG files, and
``pairs.json``.
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from .comparison import compare
from .model import ArrayFrames, ChangedSet, Footprint, HandedMaps, ObjectMap, Thresholds
from .pixels import object_edges
from .png import read_png, write_png

SIZES = ((1080, 1920), (270, 480))
EDGE_DELTA = 60
BLOB_DELTA = 40
SILHOUETTE = "silhouette"


@dataclass(frozen=True)
class Pair:
    """Two frames on either side of one threshold, and what each mode decides."""

    name: str
    threshold: str
    side: str
    value: float | str
    limit: float | str
    note: str
    reference: np.ndarray
    candidate: np.ndarray
    reference_map: ObjectMap
    candidate_map: ObjectMap
    expected: dict[str, str]

    @property
    def size(self) -> tuple[int, int]:
        return self.reference_map.size

    def edges(self) -> np.ndarray:
        found = np.zeros(self.size, dtype=bool)
        footprint = self.reference_map.objects[SILHOUETTE]
        region, mask = object_edges(footprint, self.size)  # type: ignore[misc]
        x0, y0, x1, y1 = region
        found[y0:y1, x0:x1] = mask
        return found


def verdict(pair: Pair, mode: str, thresholds: Thresholds) -> str:
    """The comparator's verdict on the pair, as one frame of a one-play sequence."""
    result = compare(
        ArrayFrames([pair.reference], Fraction(1)),
        ArrayFrames([pair.candidate], Fraction(1)),
        mode=mode,  # type: ignore[arg-type]
        maps=HandedMaps({0: pair.reference_map}, {0: pair.candidate_map}),
        changed=lambda a, b: ChangedSet(frozenset()),
        plays=[(0, 1)],
        fps=Fraction(1),
        seed=0,
        key=pair.name,
        thresholds=thresholds,
    )
    return result.verdict


def _shift(values: np.ndarray, delta: np.ndarray | int) -> np.ndarray:
    """Move every colour channel by exactly ``delta``: up where it fits, else down."""
    values = values.astype(np.int16)
    delta = np.broadcast_to(
        np.asarray(delta, dtype=np.int16)[..., None], values[..., :3].shape
    )
    rgb = values[..., :3]
    moved = np.where(rgb + delta <= 255, rgb + delta, rgb - delta)
    out = values.copy()
    out[..., :3] = moved
    return out.astype(np.uint8)


class _Scene:
    """A base frame with a silhouette painted on it: the reference of every pair."""

    def __init__(self, base: np.ndarray) -> None:
        self.height, self.width = base.shape[:2]
        h, w = self.height, self.width
        self.box = (round(0.3 * w), round(0.35 * h), round(0.7 * w), round(0.65 * h))
        x0, y0, x1, y1 = self.box
        around = base[max(y0 - 4, 0) : y1 + 4, max(x0 - 4, 0) : x1 + 4, :3].astype(
            np.int16
        )
        corners = [np.array(c, np.int16) for c in itertools.product((0, 255), repeat=3)]
        # The corner of the colour cube farthest from every pixel around it.
        self.colour = max(
            corners, key=lambda c: (int(np.abs(around - c).max(axis=2).min()), tuple(c))
        )
        self.reference = base.copy()
        self.reference[y0:y1, x0:x1, :3] = self.colour
        self.footprint = Footprint(self.box, np.ones((y1 - y0, x1 - x0), dtype=bool))
        self.map = ObjectMap((h, w), {SILHOUETTE: self.footprint}, complete=False)
        region, mask = object_edges(self.footprint, (h, w))  # type: ignore[misc]
        self.edges = np.zeros((h, w), dtype=bool)
        self.edges[region[1] : region[3], region[0] : region[2]] = mask

    @property
    def label(self) -> str:
        return f"{self.width}x{self.height}"

    def blob(self, count: int, deltas: Sequence[int]) -> np.ndarray:
        """A candidate with ``count`` connected pixels changed, the first by ``deltas[0]``
        and the rest by ``deltas[1]``, away from the silhouette.
        """
        side = math.ceil(math.sqrt(count))
        x, y = round(0.08 * self.width), round(0.15 * self.height)
        delta = np.zeros((side * side,), np.int16)
        delta[:count] = deltas[1]
        delta[0] = deltas[0]
        delta = delta.reshape(side, side)
        candidate = self.reference.copy()
        region = candidate[y : y + side, x : x + side]
        changed = _shift(region, delta)
        keep = delta[..., None] == 0
        candidate[y : y + side, x : x + side] = np.where(keep, region, changed)
        return candidate

    def changed_where(self, where: np.ndarray, delta: int = EDGE_DELTA) -> np.ndarray:
        candidate = self.reference.copy()
        candidate[where] = _shift(self.reference[where], delta)
        return candidate


def _pairs_at(base: np.ndarray, thresholds: Thresholds) -> list[Pair]:
    scene = _Scene(base)
    h, w = scene.height, scene.width
    limit = thresholds.cluster_pixels(h, w)
    over = math.floor(limit) + 1
    strong = thresholds.strong_delta
    floor = thresholds.cross_floor
    made: list[Pair] = []

    def add(
        name,
        threshold,
        side,
        value,
        limit_value,
        note,
        candidate,
        expected,
        cand_map=None,
    ):
        made.append(
            Pair(
                name=f"{name}-{scene.label}",
                threshold=threshold,
                side=side,
                value=value,
                limit=limit_value,
                note=note,
                reference=scene.reference,
                candidate=candidate,
                reference_map=scene.map,
                candidate_map=cand_map or scene.map,
                expected=expected,
            )
        )

    both_pass = {"same": "pass", "cross": "pass"}
    both_block = {"same": "block", "cross": "block"}
    cross_passes = {"same": "block", "cross": "pass"}
    add(
        "share-under",
        "cluster_share",
        "under",
        over - 1,
        limit,
        f"{over - 1} connected pixels changed by {BLOB_DELTA}: one short of a cluster",
        scene.blob(over - 1, (BLOB_DELTA, BLOB_DELTA)),
        both_pass,
    )
    add(
        "share-over",
        "cluster_share",
        "over",
        over,
        limit,
        f"{over} connected pixels changed by {BLOB_DELTA}: the smallest cluster",
        scene.blob(over, (BLOB_DELTA, BLOB_DELTA)),
        both_block,
    )
    add(
        "strength-under",
        "strong_delta",
        "under",
        strong,
        strong,
        f"{over} connected pixels, none changed by more than {strong}",
        scene.blob(over, (strong, strong)),
        both_pass,
    )
    add(
        "strength-over",
        "strong_delta",
        "over",
        strong + 1,
        strong,
        f"{over} connected pixels changed by {strong}, one of them by {strong + 1}",
        scene.blob(over, (strong + 1, strong)),
        both_block,
    )
    add(
        "floor-under",
        "cross_floor",
        "under",
        floor,
        floor,
        f"{over} connected pixels changed by {floor}, one by {strong + 1}: one renderer "
        "blocks, two renderers' rounding passes",
        scene.blob(over, (strong + 1, floor)),
        cross_passes,
    )
    add(
        "floor-over",
        "cross_floor",
        "over",
        floor + 1,
        floor,
        f"{over} connected pixels changed by {floor + 1}, one by {strong + 1}",
        scene.blob(over, (strong + 1, floor + 1)),
        both_block,
    )
    ring = 0 if h < 1080 else h // 1080
    rule = "boundary pixels" if ring == 0 else f"boundary pixels and {ring} beyond"
    add(
        "edge-under",
        "edge",
        "under",
        "edge pixels only",
        rule,
        f"every edge pixel of the silhouette ({rule}) changed by {EDGE_DELTA}: "
        "two renderers' antialiasing",
        scene.changed_where(scene.edges),
        cross_passes,
    )
    x0, y0, x1, y1 = scene.box
    if ring:
        stripe = scene.edges.copy()
        stripe[y0 - ring - 1, x0:x1] = True
        add(
            "edge-over",
            "edge",
            "over",
            f"a stripe {ring + 1} pixels outside",
            rule,
            f"the edge pixels and a stripe {ring + 1} pixels outside the top edge changed "
            f"by {EDGE_DELTA}",
            scene.changed_where(stripe),
            both_block,
        )
        return made
    inside = scene.edges.copy()
    inside[y0 + 1, x0 + 1 : x1 - 1] = True
    add(
        "edge-over-inside",
        "edge",
        "over",
        "a stripe just inside the boundary pixels",
        rule,
        f"the boundary pixels and a one-pixel stripe just inside them changed by "
        f"{EDGE_DELTA}: a wrong interior, not edge noise",
        scene.changed_where(inside),
        both_block,
    )
    outside = scene.reference.copy()
    outside[y0 - 1, x0 : x0 + 40, :3] = scene.colour
    grown = np.ones((y1 - y0 + 1, x1 - x0), dtype=bool)
    grown[0, 40:] = False
    add(
        "edge-over-outside",
        "edge",
        "over",
        "one pixel outside along 40",
        rule,
        "the candidate covers one more pixel outside the reference's footprint along a "
        "40-pixel edge: below 1080p no ring exempts it",
        outside,
        both_block,
        ObjectMap(
            (h, w), {SILHOUETTE: Footprint((x0, y0 - 1, x1, y1), grown)}, complete=False
        ),
    )
    add(
        "floor-under-7",
        "cross_floor",
        "under",
        7,
        floor,
        f"{over} connected pixels changed by 7, one by {strong + 1}: one renderer blocks, "
        "two renderers' rounding passes",
        scene.blob(over, (strong + 1, 7)),
        cross_passes,
    )
    return made


def pairs(
    base_1080: np.ndarray, base_270: np.ndarray, thresholds: Thresholds = Thresholds()
) -> list[Pair]:
    """Every pair from the two base frames, each checked against the comparator."""
    made = []
    for base, (h, w) in zip((base_1080, base_270), SIZES, strict=True):
        if base.shape[:2] != (h, w):
            raise ValueError(
                f"a base frame here is {w} x {h}, not {base.shape[1]} x {base.shape[0]}"
            )
        made += _pairs_at(base, thresholds)
    for pair in made:
        for mode, want in pair.expected.items():
            got = verdict(pair, mode, thresholds)
            if got != want:
                raise RuntimeError(
                    f"{pair.name}: in {mode} mode the comparator says {got}, not {want}"
                )
    return made


def write(
    made: Sequence[Pair], out: Path, thresholds: Thresholds = Thresholds()
) -> Path:
    """Write each pair's frames and edge mask, and ``pairs.json`` describing them."""
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    for pair in made:
        files = {
            "reference": f"{pair.name}.reference.png",
            "candidate": f"{pair.name}.candidate.png",
            "edges": f"{pair.name}.edges.png",
        }
        write_png(out / files["reference"], pair.reference)
        write_png(out / files["candidate"], pair.candidate)
        write_png(out / files["edges"], pair.edges())
        entries.append(
            {
                "name": pair.name,
                "width": pair.size[1],
                "height": pair.size[0],
                "threshold": pair.threshold,
                "side": pair.side,
                "value": pair.value,
                "limit": pair.limit,
                "note": pair.note,
                "files": files,
                "verdict": dict(pair.expected),
            }
        )
    path = out / "pairs.json"
    document = {"thresholds": thresholds.as_dict(), "pairs": entries}
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 3:
        print(
            "usage: python -m arachne_compare.calibration <base 1920x1080 png> "
            "<base 480x270 png> <out dir>",
            file=sys.stderr,
        )
        return 2
    base_1080, base_270, out = read_png(args[0]), read_png(args[1]), Path(args[2])
    made = pairs(base_1080, base_270)
    path = write(made, out)
    for pair in made:
        print(
            f"{pair.name}: same {pair.expected['same']}, cross {pair.expected['cross']}"
        )
    print(f"{len(made)} pairs written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
