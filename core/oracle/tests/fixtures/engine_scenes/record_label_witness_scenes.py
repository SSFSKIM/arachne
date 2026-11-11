"""Renders the two label witness scenes on the core.

Copied with ``label_witness_scenes.py`` into ``tests/frame_ir/`` of a checkout at
``30538b2e``, and run by pytest with ``PYTHONHASHSEED=0`` and a plugin that writes each
core painter's stream as an XF file, SOURCE included. The streams are the fixtures
``no_pair_centre_without_points_engine`` and ``pair_cause_label_engine``.
"""

from __future__ import annotations

import sys

import pytest

from ..timeline.conftest import render
from . import label_witness_scenes as scenes

assert sys.flags.hash_randomization == 0, "run with PYTHONHASHSEED=0"


@pytest.mark.parametrize("name", ["RingCentreInHole", "LabelPreProjected"])
def test_scene(name):
    render(
        getattr(scenes, name),
        scenes.FPS,
        draw_path="core",
        pixel_width=64,
        pixel_height=64,
        frame_width=16.0,
        frame_height=16.0,
    )
