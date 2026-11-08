"""The tables that decide what the engine trusts find the engine's own
objects by identity: a class whose metaclass makes it equal to an engine
class, or a value equal to an engine value, is not taken for it.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from arachne import (
    AnimationGroup,
    Camera,
    Create,
    Dot,
    PointCloudDot,
    Scene,
    Square,
    UpdateFromAlphaFunc,
)
from arachne.animation.animation import Add, prepare_animation
from arachne.timeline import (
    admission,
    bodies,
    dispatch,
    draw,
    frozen,
    purity,
    restore,
    setters,
    trusted,
)
from arachne.timeline.identity import ByIdentity, one_of

from . import identity_scenes

IDENTITY_SCENES = Path(identity_scenes.__file__)


def equal_to(target: type, base: type) -> type:
    """A metaclass (over ``base``'s) whose classes hash and compare equal to
    ``target``.
    """

    class EqualTo(type(base)):
        def __hash__(cls):
            return hash(target)

        def __eq__(cls, other):
            return other is target or other is cls

    return EqualTo


class OwnCreate(Create, metaclass=equal_to(Create, Create)):
    """A reveal of a scene's own that interpolates its own way."""

    def interpolate_mobject(self, alpha):
        super().interpolate_mobject(1 - alpha)


class OwnGroup(AnimationGroup, metaclass=equal_to(AnimationGroup, AnimationGroup)):
    def interpolate(self, alpha):
        super().interpolate(1 - alpha)


class OwnUpdate(
    UpdateFromAlphaFunc, metaclass=equal_to(UpdateFromAlphaFunc, UpdateFromAlphaFunc)
):
    def interpolate_mobject(self, alpha):
        super().interpolate_mobject(1 - alpha)


class OwnSquare(Square, metaclass=equal_to(Square, Square)):
    pass


class OwnDot(Dot, metaclass=equal_to(Dot, Dot)):
    pass


class OwnCloud(PointCloudDot, metaclass=equal_to(Square, PointCloudDot)):
    """A point cloud equal to ``Square``."""


class OwnFloat(float, metaclass=equal_to(float, float)):
    """A float equal to ``float`` as a type, whose repr hides its value."""

    def __repr__(self):
        return "OwnFloat"


def test_a_table_finds_only_the_object_it_holds():
    table = ByIdentity({Create: 1})
    assert table[Create] == 1
    assert Create in table
    assert OwnCreate == Create
    assert hash(OwnCreate) == hash(Create)
    assert OwnCreate not in table
    assert table.get(OwnCreate) is None
    with pytest.raises(KeyError):
        table[OwnCreate]
    with pytest.raises(KeyError):
        del table[OwnCreate]
    table[OwnCreate] = 2
    assert (table[Create], table[OwnCreate], len(table)) == (1, 2, 2)
    del table[Create]
    assert list(table) == [OwnCreate]
    assert ByIdentity.of([Create]) == {Create: None}
    assert one_of(Create, (Square, Create))
    assert not one_of(OwnCreate, (Square, Create))
    assert OwnCreate in (Square, Create)


def test_a_leaf_class_equal_to_an_engine_leaf_is_outside_the_algebra():
    reasons = trusted.refusals([prepare_animation(OwnCreate(Square()))])
    assert any("not in the trusted algebra" in r for r in reasons), reasons
    assert trusted.refusals([prepare_animation(Create(Square()))]) == []


def test_a_group_class_equal_to_an_engine_group_is_outside_the_algebra():
    group = OwnGroup(Create(Square()), Create(Dot()))
    reasons = trusted.refusals([prepare_animation(group)])
    assert any("not in the trusted algebra" in r for r in reasons), reasons


def test_a_class_equal_to_a_callback_animation_is_not_one():
    animation = OwnUpdate(Square(), lambda m, a: None)
    assert not admission._is_callback_animation(animation)
    reasons = trusted.refusals([prepare_animation(animation)])
    assert any("not in the trusted algebra" in r for r in reasons), reasons
    assert admission._is_callback_animation(
        UpdateFromAlphaFunc(Square(), lambda m, a: None)
    )


def test_a_class_equal_to_an_engine_class_is_not_the_engines():
    assert not frozen.engine_class(OwnSquare)
    with pytest.raises(KeyError):
        frozen.class_members(OwnSquare)
    assert frozen.engine_class(Square)
    assert not setters.builtin_class(OwnSquare)
    assert setters.setter_for(Square(), "move_to") is not None
    assert setters.setter_for(OwnSquare(), "move_to") is None
    assert setters.read_kind(OwnSquare(), "get_center") is None
    assert setters.read_kind(Square(), "get_center") == "value"


def test_a_class_equal_to_an_admitted_constructor_is_not_admitted():
    assert setters.constructor_kind(Dot) is not None
    assert setters.constructor_kind(OwnDot) is None


def test_a_value_of_a_type_equal_to_a_plain_type_is_compared_as_an_object():
    assert restore._same_contents(1.0, 1.0)
    assert not restore._same_contents(OwnFloat(1.0), OwnFloat(2.0))


def test_a_mobject_type_equal_to_another_takes_its_own_draw_kind():
    bodies._look(Square())
    cloud = OwnCloud()
    assert len(bodies._look(cloud)) == len(bodies._look(PointCloudDot()))


class _EqualToFunction:
    """A callable whose hash and equality are ``function``'s."""

    def __init__(self, function):
        self.function = function

    def __hash__(self):
        return hash(self.function)

    def __eq__(self, other):
        return other is self.function or other is self

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)


@pytest.mark.parametrize("name", ["move_to", "get_center"])
def test_an_engine_method_replaced_by_an_equal_callable_is_not_admitted(
    monkeypatch, name
):
    engine = setters._method_function(Square(), name)
    before = (setters.setter_for(Square(), name), setters.read_kind(Square(), name))
    assert before != (None, None)
    monkeypatch.setattr(Square, name, _EqualToFunction(engine))
    assert setters.setter_for(Square(), name) is None
    assert setters.read_kind(Square(), name) is None
    assert setters.fresh_method(Square(), name) is None


class OwnAdd(Add, metaclass=equal_to(Add, Add)):
    def interpolate_mobject(self, alpha):
        self.mobject.shift(0.01 * alpha)


def test_an_animation_class_equal_to_add_writes_its_mobject():
    square = Square()
    assert trusted.writes(Add(square)) == set()
    assert id(square) in trusted.writes(OwnAdd(square))


class OwnCamera(Camera, metaclass=equal_to(Camera, Camera)):
    def capture_mobjects(self, mobjects, **kwargs):
        super().capture_mobjects(list(mobjects)[::-1], **kwargs)


def test_a_camera_class_equal_to_the_engines_is_not_supported():
    assert admission.camera_refusal(Camera()) is None
    assert admission.camera_refusal(OwnCamera()).startswith("camera.unsupported")
    assert draw._paints_its_list(Camera())
    assert not draw._paints_its_list(OwnCamera())


class OwnCentre(Square, metaclass=equal_to(Square, Square)):
    def get_center(self):
        return super().get_center() * 2


def test_a_class_equal_to_one_seen_before_is_checked_for_itself():
    reason = dispatch.family_refusal([Square(), OwnCentre()])
    assert reason is not None
    assert "OwnCentre" in reason
    reason = dispatch.draw_refusal(Camera(), [Dot(), identity_scenes.OwnFill()])
    assert reason is not None
    assert reason.startswith("draw.user_override: OwnFill.")


class OwnTracked(Square, metaclass=equal_to(Square, Square)):
    def update(self, dt=0, recursive=True):
        return super().update(dt, recursive)


def test_an_update_of_a_class_equal_to_one_seen_before_is_checked_for_itself():
    scene = Scene()
    scene.add(Square(), OwnTracked())
    reason = admission._update_replaced(scene)
    assert reason is not None
    assert "OwnTracked" in reason


@pytest.mark.parametrize("name", ["OwnFillBesideDot", "OwnFillVisits"])
def test_a_dot_class_equal_to_dot_is_drawn_as_the_scene_runs(session_of, name):
    session = session_of(name, IDENTITY_SCENES)
    assert session.record.refusal is not None
    assert session.record.refusal.startswith("draw.user_override: OwnFill.")


class _ListLike(list, metaclass=equal_to(list, list)):
    pass


class _FloatLike(float, metaclass=equal_to(float, float)):
    pass


class _NamedNumpy(np.float64):
    __module__ = "numpy"


def test_a_value_of_a_type_only_equal_to_a_plain_type_is_not_a_constant():
    assert purity._constant(1.5)
    assert purity._constant((1, "a"))
    assert purity._constant(np.float64(1.5))
    assert not purity._constant(_FloatLike(1.5))
    assert not purity._constant(_NamedNumpy(1.5))


def _subclasses(cls: type) -> set[type]:
    found = set()
    for sub in cls.__subclasses__():
        found |= {sub, *_subclasses(sub)}
    return found


def test_numpys_own_scalar_types_are_the_ones_it_names():
    """The constants table takes numpy's scalar types by the names numpy
    gives them: every scalar type numpy defines is among them.
    """
    own = {
        cls
        for cls in _subclasses(np.generic) | {np.generic}
        if cls.__module__ == "numpy" and cls is not _NamedNumpy
    }
    assert own <= set(purity._NUMPY_SCALARS)


PLAIN_ITEMS = [1.0, 2.0]
OWN_ITEMS = _ListLike([1.0, 2.0])


def _reads_plain_items():
    return PLAIN_ITEMS[0]


def _reads_own_items():
    return OWN_ITEMS[0]


def test_a_list_type_only_equal_to_list_is_not_read_as_a_list():
    scene = Scene()
    purity.prove_readout(_reads_plain_items, scene)
    with pytest.raises(purity.Refused):
        purity.prove_readout(_reads_own_items, scene)


class DriftingModule(ModuleType):
    """A module of a scene's own whose attribute is its own code."""

    reads = 0

    @property
    def value(self):
        self.reads += 1
        return self.reads / 100


DRIFTING = DriftingModule("math")
#: A plain module object named ``math`` whose missing names are its own code.
MADE = ModuleType("math")
MADE.__getattr__ = lambda name: len(name) / 100


def _reads_math():
    return math.pi


def _reads_drifting():
    return DRIFTING.value


def _reads_made():
    return MADE.pi


def test_only_the_interpreters_own_module_of_a_name_is_read():
    assert setters.pure_module(math)
    assert not setters.pure_module(DRIFTING)
    assert not setters.pure_module(MADE)
    scene = Scene()
    purity.prove_readout(_reads_math, scene)
    reads = DRIFTING.reads
    for readout in (_reads_drifting, _reads_made):
        with pytest.raises(purity.Refused):
            purity.prove_readout(readout, scene)
    assert DRIFTING.reads == reads
