"""The fields an interpolation overwrites from what it holds alone
(:func:`arachne.timeline.overwrites.overwritten_fields`): each declaration is
checked by a property that can fail. Whatever the target's members held
before the call on the declared fields (other values, other point counts),
the call leaves them bit-identical, at every alpha and at both ends of each
rate function.
"""

from __future__ import annotations

import numpy as np
import pytest

from arachne import (
    RIGHT,
    UP,
    Circle,
    Create,
    Dot,
    FadeIn,
    FadeOut,
    Line,
    Square,
    Text,
    Transform,
    Triangle,
    VGroup,
)
from arachne.animation.creation import ShowPartial
from arachne.mobject.types.vectorized_mobject import VMobject
from arachne.mobject.values import Paint
from arachne.timeline.overwrites import overwritten_fields
from arachne.utils.rate_functions import linear, smooth, there_and_back

POINTS = frozenset({"_points"})
POINTS_AND_PAINT = frozenset({"_points", "_paint"})
ALPHAS = (0.0, 0.1, 0.37, 0.5, 0.83, 1.0)
RATES = (smooth, linear, there_and_back)
TRIALS = 6


def _group():
    return VGroup(Circle(), Square().shift(RIGHT), Triangle().shift(UP))


def _text():
    return Text("1.25")


#: Each declared animation, by how its target is made, with its fields.
DECLARED = {
    "Create circle": (lambda rate: Create(Circle(), rate_func=rate), POINTS),
    "Create line": (lambda rate: Create(Line(), rate_func=rate), POINTS),
    "Create group": (lambda rate: Create(_group(), rate_func=rate), POINTS),
    "Create text": (lambda rate: Create(_text(), rate_func=rate), POINTS),
    "FadeIn group": (lambda rate: FadeIn(_group(), rate_func=rate), POINTS_AND_PAINT),
    "FadeIn shifted": (
        lambda rate: FadeIn(Circle(), shift=UP, scale=1.5, rate_func=rate),
        POINTS_AND_PAINT,
    ),
    "FadeIn text": (lambda rate: FadeIn(_text(), rate_func=rate), POINTS_AND_PAINT),
    "FadeOut circle": (
        lambda rate: FadeOut(Circle(), rate_func=rate),
        POINTS_AND_PAINT,
    ),
    "FadeOut group": (
        lambda rate: FadeOut(_group(), shift=RIGHT, rate_func=rate),
        POINTS_AND_PAINT,
    ),
    "Transform aligning families": (
        lambda rate: Transform(
            VGroup(Circle()), VGroup(Square(), Triangle(), Dot()), rate_func=rate
        ),
        POINTS_AND_PAINT,
    ),
}


def _scramble(member, field: str, rng: np.random.Generator) -> None:
    """Give ``member`` another value of ``field``, of another shape."""
    if field == "_points":
        member.points = rng.normal(size=(int(rng.integers(1, 60)), 3))
    elif field == "_paint":
        member._paint = member._paint.replace(
            fill_rgbas=rng.random((int(rng.integers(1, 4)), 4)),
            stroke_rgbas=rng.random((int(rng.integers(1, 4)), 4)),
            background_stroke_rgbas=rng.random((int(rng.integers(1, 4)), 4)),
            stroke_width=float(rng.random() * 8),
            background_stroke_width=float(rng.random() * 8),
            sheen_direction=rng.normal(size=3),
            sheen_factor=float(rng.random()),
        )
    elif field == "stroke_width":
        member._paint = member._paint.replace(stroke_width=float(rng.random() * 8))
    else:
        raise KeyError(field)


def _value(value) -> object:
    if isinstance(value, np.ndarray):
        return (value.dtype.str, value.shape, value.tobytes())
    return repr(value)


def _read(member, field: str) -> object:
    if field == "_points":
        return _value(np.asarray(member.points))
    if field == "_paint":
        return tuple(
            (name, _value(np.asarray(getattr(member._paint, name, None))))
            for name in Paint.FIELDS
        )
    if field == "stroke_width":
        return _value(np.asarray(member._paint.stroke_width))
    raise KeyError(field)


def assert_overwrites(make, fields: frozenset[str]) -> None:
    """``fields`` of every member with points, after ``interpolate(alpha)``,
    are bit-identical whatever the members held there before: the value the
    previous call left (the first trial) and scrambled ones.
    """
    for rate in RATES:
        for alpha in ALPHAS:
            animation = make(rate)
            animation.begin()
            members = animation.mobject.family_members_with_points()
            outputs = []
            for trial in range(TRIALS):
                rng = np.random.default_rng(trial)
                if trial:
                    for member in members:
                        for field in sorted(fields):
                            _scramble(member, field, rng)
                animation.interpolate(alpha)
                outputs.append([_read(m, f) for m in members for f in sorted(fields)])
            assert all(out == outputs[0] for out in outputs), (rate.__name__, alpha)


@pytest.mark.parametrize("name", sorted(DECLARED))
def test_a_declaration_is_what_the_interpolation_overwrites(name):
    make, fields = DECLARED[name]
    animation = make(smooth)
    assert overwritten_fields(animation) == fields
    animation.begin()
    assert overwritten_fields(animation) == fields
    assert_overwrites(make, fields)


def test_a_declaration_that_claims_too_much_fails_its_property():
    """The negative control: ``Create`` keeps the stroke width its target
    had, so declaring it as overwritten fails.
    """
    make, _ = DECLARED["Create group"]
    with pytest.raises(AssertionError):
        assert_overwrites(make, frozenset({"stroke_width"}))


class _OwnPartial(Create):
    def interpolate_submobject(self, submobject, starting_submobject, alpha):
        ShowPartial.interpolate_submobject(self, submobject, starting_submobject, alpha)


class _OwnPoints(Circle):
    def pointwise_become_partial(self, vmobject, a, b):
        return VMobject.pointwise_become_partial(self, vmobject, a, b)


def _stub():
    """A VMobject with points but no whole curve."""
    stub = VMobject()
    stub.points = np.array([[0.0, 0, 0], [1, 0, 0]])
    return stub


def test_a_partial_of_a_copy_with_no_whole_curve_is_not_declared():
    """``pointwise_become_partial`` from a copy with points but no whole
    curve leaves the target's points as they were: no declaration, and the
    property it would claim fails.
    """
    assert overwritten_fields(Create(_stub())) == frozenset()
    with pytest.raises(AssertionError):
        assert_overwrites(lambda rate: Create(_stub(), rate_func=rate), POINTS)


def _instance_shadow():
    animation = Create(Circle())
    animation.interpolate = animation.interpolate
    return animation


@pytest.mark.parametrize(
    "make",
    [
        lambda: _OwnPartial(Circle()),
        _instance_shadow,
        lambda: Create(Circle(), rate_func=lambda t: t),
        lambda: Create(_OwnPoints()),
        lambda: Transform(
            Circle(), Square(), path_func=lambda a, b, t: a + (b - a) * t
        ),
    ],
    ids=[
        "subclass overriding interpolate_submobject",
        "instance replacing interpolate",
        "scene rate function",
        "member class overriding pointwise_become_partial",
        "scene path function",
    ],
)
def test_what_replaces_a_method_on_the_path_inherits_no_declaration(make):
    assert overwritten_fields(make()) == frozenset()
