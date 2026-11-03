"""Which parameters a function may call (:mod:`arachne.timeline.handed`): the
refinements that let the write check tell what a callee never calls, each
with the case it must still count as called.
"""

from __future__ import annotations

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


def test_kwargs_handed_on_are_called_key_by_key():
    """``**kwargs`` handed on as ``**kwargs``: only the keyword the callee
    calls is called, and a callee calling what it gathers calls every one.
    """
    assert _called(_outer) == {"kwargs:act"}
    assert _called(_hands_on) == {"kwargs"}
