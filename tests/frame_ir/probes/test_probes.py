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

#: Checks inside a lit face, where the core's lighting gradient lands one
#: level per channel off pixman's: there the core is held to CE's bytes within
#: one level. (Matching pixman inside a face is a look, not depth: a depth
#: fault there shows another sheet's colour. A translucent sheet's pixel is a
#: blend, so no twin hides what lies behind it.)
LIT_FACE = {("TranslucentOverOpaque", 1)}


@functools.cache
def _frames(probe: type, path: str, hide: tuple[str, ...] = ()) -> list:
    scene_class = scenes.twin(probe, hide) if hide else probe
    _, drawn = render(scene_class, scenes.FPS, draw_path=path, **SIZE)
    return drawn


def _cases():
    for probe in scenes.PROBES:
        for index, check in enumerate(probe.CHECKS):
            near = (probe.__name__, index) in LIT_FACE
            yield pytest.param(probe, check, near, id=f"{probe.__name__}-{index}")


def _pixel(frames: list, check: scenes.Check) -> tuple[int, ...]:
    return tuple(int(v) for v in frames[check.frame][check.y, check.x])


@pytest.mark.parametrize(("probe", "check", "near"), list(_cases()))
def test_the_core_draws_each_check_by_depth(probe, check, near):
    core = _pixel(_frames(probe, "core"), check)
    cairo = _pixel(_frames(probe, "cairo"), check)
    assert cairo == check.ce, "Cairo draws what CE draws there"
    if check.hide:
        assert core == _pixel(_frames(probe, "core", check.hide), check), check.what
    if near:
        apart = max(abs(a - b) for a, b in zip(core, cairo, strict=True))
        assert apart <= 1, (core, cairo)
    elif check.hide:
        assert (core == cairo) == ("(CE right)" in check.what), (core, cairo)
    else:
        assert core == cairo, check.what
