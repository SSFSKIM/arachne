"""A mobject's deep copy made by Arachne's own copier (``fast_copy``) changes
no record and no frame: each scene rendered with the copier on and off has
equal content keys at every capture, the same recorded values in type and
bytes, the same animation states kept, and equal bytes at every frame.
"""

from __future__ import annotations

import functools
import importlib.util
import sys
from pathlib import Path

import pytest

from . import route_scenes, vectorized_scenes
from .conftest import render
from .test_vectorized import animation_states, keys, recorded_values, same_value

TEACHING = Path(__file__).parents[2] / "benchmarks" / "teaching_scenes" / "scenes.py"


@functools.cache
def teaching_scenes():
    spec = importlib.util.spec_from_file_location("teaching_scenes", TEACHING)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TEACHING_NAMES = [
    "PlotWithTangent",
    "BulletLayout",
    "FormulaSteps",
    "AreaUnderCurve",
    "NumberLineJumps",
    "SpinningPhasor",
    "SurfaceOrbit",
    "LinearMap",
    "GrowingBars",
    "CameraOverGrid",
]


def assert_same_with_the_copier_on_and_off(scene_class, fps: int = 15):
    # A scene's first render in a process can record other content keys than
    # its later renders do, with either copier (what the process's caches
    # hold differs): one render first, so that both compared renders are
    # later ones.
    render(scene_class, fps)
    on_scene, on = render(scene_class, fps, fast_copy=True)
    off_scene, off = render(scene_class, fps, fast_copy=False)
    assert keys(on_scene) == keys(off_scene)
    on_values, off_values = recorded_values(on_scene), recorded_values(off_scene)
    for index, (a, b) in enumerate(zip(on_values, off_values, strict=True)):
        assert same_value(a, b), f"capture {index}"
    assert same_value(animation_states(on_scene), animation_states(off_scene))
    assert [p.route for p in on_scene.timeline.ensure_record().plays] == [
        p.route for p in off_scene.timeline.ensure_record().plays
    ]
    assert len(on) == len(off)
    for index, (a, b) in enumerate(zip(on, off, strict=True)):
        assert a.tobytes() == b.tobytes(), f"frame {index}"


@pytest.mark.parametrize("name", sorted(route_scenes.TRUSTED_PLAYS))
def test_each_entry_of_the_trusted_algebra(name):
    assert_same_with_the_copier_on_and_off(getattr(route_scenes, f"Trusted{name}"))


@pytest.mark.parametrize(
    "name",
    [
        "Grouped",
        "Lagged",
        "Nested",
        "Rated",
        "Revealed",
        "Copied",
        "Successions",
        "Mixed",
    ],
)
def test_compositions(name):
    assert_same_with_the_copier_on_and_off(getattr(vectorized_scenes, name))


@pytest.mark.parametrize("name", TEACHING_NAMES)
def test_the_teaching_scenes(name):
    assert_same_with_the_copier_on_and_off(getattr(teaching_scenes(), name), fps=5)
