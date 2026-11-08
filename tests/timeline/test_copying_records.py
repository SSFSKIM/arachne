"""A mobject's deep copy made by Arachne's own copier (``fast_copy``) changes
no record and no frame: each scene rendered with the copier on and off has
equal content keys at every capture, the same recorded values in type and
bytes, the same animation states kept, and equal bytes at every frame.
"""

from __future__ import annotations

import copyreg
import functools
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from arachne.mobject.placement import Placement
from arachne.mobject.values import Paint
from arachne.timeline.record import IDENTITY_FIELDS

from . import route_scenes, vectorized_scenes
from .conftest import render
from .test_vectorized import animation_states, keys, recorded_values, same_value

_MISSING = object()
_SCALARS = (bool, int, float, complex, str, bytes, np.generic)


def _held(x) -> tuple[dict | None, list]:
    """What an object holds: its instance ``__dict__``, but for the fields
    naming an object of its process (:data:`IDENTITY_FIELDS`), and its slots.
    """
    own = getattr(x, "__dict__", None)
    if isinstance(own, dict):
        own = {k: v for k, v in own.items() if k not in IDENTITY_FIELDS}
    slots = copyreg._slotnames(type(x))
    return own, [getattr(x, name, _MISSING) for name in slots]


def state_difference(
    a, b, path: str = "", seen: set[tuple[int, int]] | None = None
) -> str | None:
    """Where two recorded values differ, or ``None``: :func:`same_value`'s
    comparison, except that an object of a class it does not know, which it
    matches by type alone, is matched by what it holds too: its instance
    ``__dict__`` and slots, value by value. A pair of objects in ``seen`` is
    taken as matched.
    """
    seen = set() if seen is None else seen
    if type(a) is not type(b):
        return f"{path}: {type(a).__name__} against {type(b).__name__}"
    if a is None or isinstance(a, _SCALARS):
        return None if repr(a) == repr(b) else f"{path}: {a!r:.40} against {b!r:.40}"
    if isinstance(a, (Paint, Placement, type)) or getattr(a, "__code__", None):
        same = a is b if isinstance(a, type) else same_value(a, b)
        return None if same else f"{path}: {type(a).__name__} differs"
    if isinstance(a, np.ndarray) and not a.dtype.hasobject:
        return None if same_value(a, b) else f"{path}: array differs"
    if isinstance(a, np.ndarray):
        if a.dtype != b.dtype or a.shape != b.shape:
            return f"{path}: array differs"
        a, b = a.ravel().tolist(), b.ravel().tolist()
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return f"{path}: length {len(a)} against {len(b)}"
        pairs = [
            (f"{path}[{i}]", x, y) for i, (x, y) in enumerate(zip(a, b, strict=True))
        ]
    elif isinstance(a, dict):
        if list(a) != list(b):
            return f"{path}: keys differ"
        pairs = [(f"{path}[{k!r:.30}]", a[k], b[k]) for k in a]
    else:
        if (id(a), id(b)) in seen:
            return None
        seen.add((id(a), id(b)))
        pairs = [(f"{path}<held>", _held(a), _held(b))]
    for where, x, y in pairs:
        found = state_difference(x, y, where, seen)
        if found:
            return found
    return None


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
    # What a record holds of its scene is the scene itself: the two scenes
    # are counterparts, not walked into.
    scenes = (id(on_scene), id(off_scene))
    on_values, off_values = recorded_values(on_scene), recorded_values(off_scene)
    for index, (a, b) in enumerate(zip(on_values, off_values, strict=True)):
        found = state_difference(a, b, seen={scenes})
        assert found is None, f"capture {index}: {found}"
    found = state_difference(
        animation_states(on_scene), animation_states(off_scene), seen={scenes}
    )
    assert found is None, found
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


class Holding:
    def __init__(self, held):
        self.held = held


class Slotted:
    __slots__ = ("held",)

    def __init__(self, held):
        self.held = held


@pytest.mark.parametrize("kind", [Holding, Slotted], ids=lambda k: k.__name__)
def test_the_check_reads_what_an_unfamiliar_object_holds(kind):
    assert state_difference([kind([1.0])], [kind([1.0])]) is None
    assert state_difference([kind([1.0])], [kind([2.0])])
    assert state_difference({"a": kind(kind(1))}, {"a": kind(kind(2))})


def test_the_check_ends_on_an_object_that_holds_itself():
    a, b = Holding(None), Holding(None)
    a.held, b.held = a, b
    assert state_difference(a, b) is None
