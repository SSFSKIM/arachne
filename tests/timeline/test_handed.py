"""Which parameters a function may call (:mod:`arachne.timeline.handed`): the
refinements that let the write check tell what a callee never calls, each
with the case it must still count as called.
"""

from __future__ import annotations

import numpy as np

from arachne.timeline.ticking import _Writes


def _called(function, cls=None):
    return _Writes().callers.called_parameters(function, cls, 0)


def _inner(scale=1.0, act=None):
    sorted(["a", "b"], key=act)


def _outer(**kwargs):
    _inner(**kwargs)


def _gathers(**kwargs):
    kwargs.get("act")()


def _hands_on(**kwargs):
    _gathers(**kwargs)


def _repeats(act=None):
    np.piecewise(np.array([1.0]), [np.array([True])], [act] * 1)


def _negates(act=None):
    sorted(["a", "b"], key=-act)


def _counts(scale=1.0):
    return np.sin(np.zeros(3) * scale + 2 * 3)


def test_what_an_operator_makes_carries_what_its_operands_hold():
    """A list an operator repeats holds its items, and any operator on a
    parameter may give back what the parameter holds; only constants are
    shown to hold no callable.
    """
    assert _called(_repeats) == {"act"}
    assert _called(_negates) == {"act"}
    assert _called(_counts) == set()


def test_kwargs_handed_on_are_called_key_by_key():
    """``**kwargs`` handed on as ``**kwargs``: only the keyword the callee
    calls is called, and a callee calling what it gathers calls every one.
    """
    assert _called(_outer) == {"kwargs:act"}
    assert _called(_hands_on) == {"kwargs"}
