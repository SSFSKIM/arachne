"""The eight class probes, drawn on the core from the engine's scenes.

At each check the core draws what the scene draws without the objects the
check hides (a depth class CE's painter order gets wrong, or right where its
per-face sort orders the faces as depth does), or, where it hides nothing,
CE's pixel. Cairo's pixel is CE's, the bytes the check records, which
``ce_check.py`` holds against real CE 0.20.1.
"""

from __future__ import annotations

import functools

import pytest

from ...timeline.conftest import render
from . import scenes

pytest.importorskip("arachne_core")

SIZE = {"pixel_width": scenes.WIDTH, "pixel_height": scenes.HEIGHT}


@functools.cache
def _frames(probe: type, path: str, hide: tuple[str, ...] = ()) -> list:
    scene_class = scenes.twin(probe, hide) if hide else probe
    _, drawn = render(scene_class, scenes.FPS, draw_path=path, **SIZE)
    return drawn


def _cases():
    for probe in scenes.PROBES:
        for index, check in enumerate(probe.CHECKS):
            yield pytest.param(probe, check, id=f"{probe.__name__}-{index}")


def _pixel(frames: list, check: scenes.Check) -> tuple[int, ...]:
    return tuple(int(v) for v in frames[check.frame][check.y, check.x])


@pytest.mark.parametrize(("probe", "check"), list(_cases()))
def test_the_core_draws_each_check_by_depth(probe, check):
    core = _pixel(_frames(probe, "core"), check)
    cairo = _pixel(_frames(probe, "cairo"), check)
    assert cairo == check.ce, "Cairo draws what CE draws there"
    if check.hide:
        assert core == _pixel(_frames(probe, "core", check.hide), check), check.what
        assert (core == cairo) == ("(CE right)" in check.what), (core, cairo)
    else:
        assert core == cairo, check.what
