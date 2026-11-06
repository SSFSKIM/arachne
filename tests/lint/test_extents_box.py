"""The box around a path's Cairo extents is NumPy's box of them, bit for
bit: the same corners, a zero's sign included.
"""

from __future__ import annotations

import numpy as np

from arachne.camera.projection import _extents_box


def _numpy_box(found: list[tuple[float, float, float, float]]) -> np.ndarray:
    box = np.array(found)
    return np.array(
        [
            [box[:, 0].min(), box[:, 1].min()],
            [box[:, 2].max(), box[:, 3].max()],
        ]
    )


def test_the_extents_box_is_numpy_s_box_of_the_extents_bit_for_bit():
    rng = np.random.default_rng(20261106)
    values = [0.0, -0.0, 1.5, -1.5, 1e-300, -1e-300, 4.0e6]
    for _ in range(4000):
        found = []
        for _ in range(int(rng.integers(1, 4))):
            if rng.random() < 0.5:
                extent = [float(v) for v in rng.choice(values, size=4)]
            else:
                extent = [float(v) for v in rng.normal(scale=100.0, size=4)]
            found.append(tuple(extent))
        ours = np.array(_extents_box(found))
        assert ours.dtype == np.float64
        assert ours.tobytes() == _numpy_box(found).tobytes(), found
