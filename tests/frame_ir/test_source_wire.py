"""XF 1.5's SOURCE on the Python side: the mirror's records carry a record's
source and facing group to the core and back, and the canonical snap hands
both on as they were given.
"""

from __future__ import annotations

import numpy as np
import pytest

from arachne.frame_ir.canonical import Canonical
from arachne.frame_ir.xf import Delta, Record, Shape

pytest.importorskip("arachne_core")

SOURCE = (
    (
        (4, "arachne.mobject.geometry.tips.ArrowTriangleFilledTip"),
        (2, "arachne.mobject.geometry.line.Vector"),
    ),
)

VIEW = {
    "pixel_width": 8,
    "pixel_height": 8,
    "frame_width": 8.0,
    "frame_height": 8.0,
    "camera": "flat",
}


def _delta() -> Delta:
    corners = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=float)
    points = np.concatenate(
        [
            [a, a + (b - a) / 3, a + 2 * (b - a) / 3, b]
            for a, b in zip(corners, np.roll(corners, -1, axis=0), strict=True)
        ]
    )
    shape = Shape("path", points)
    record = Record(
        "path",
        {
            "shape": shape.key,
            "fill": np.array([[1.0, 0.0, 0.0, 1.0]]),
            "source": SOURCE,
            "facing_group": 2,
        },
    )
    return Delta(
        view=dict(VIEW),
        upload=[shape],
        set=[(0, record)],
        order=[("moving", 0, 0)],
    )


def test_the_mirror_carries_a_record_s_source_to_the_core_and_back():
    import arachne_core

    core = _delta().to_core()
    assert core.version == (1, 5)
    read = arachne_core.FrameDelta.from_bytes(core.to_bytes())
    [(slot, record)] = Delta.from_contents(read.contents()).set
    assert slot == 0
    assert record.fields["source"] == SOURCE
    assert record.fields["facing_group"] == 2


def test_the_canonical_snap_hands_a_record_s_source_on_as_given():
    [(_, record)] = Canonical()(_delta()).set
    assert record.fields["source"] == SOURCE
    assert record.fields["facing_group"] == 2
