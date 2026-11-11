"""Renders the registry witness scene on the core.

Copied with ``registry_witness_scene.py`` into ``tests/frame_ir/`` of a checkout at
``30538b2e``, and run by pytest with ``PYTHONHASHSEED=0`` and a plugin that writes each
core painter's stream as an XF file, SOURCE included. The stream is the fixture
``invisible_registry_engine``.
"""

from __future__ import annotations

import sys

from ..timeline.conftest import render
from . import registry_witness_scene as scene

assert sys.flags.hash_randomization == 0, "run with PYTHONHASHSEED=0"


def test_scene():
    render(
        scene.InvisibleRegistryOffer,
        scene.FPS,
        draw_path="core",
        pixel_width=160,
        pixel_height=90,
    )
