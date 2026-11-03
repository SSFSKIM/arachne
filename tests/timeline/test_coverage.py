"""The operation table (:mod:`arachne.timeline.coverage`): every node class of
the running Python has a rule, a form without one falls back, and the two
queries built on it (:func:`proven_pure`, :func:`result_kept`) answer from
its rules.
"""

from __future__ import annotations

import ast
import math

import numpy as np

from arachne.timeline.coverage import (
    OPERATIONS,
    Rule,
    node_classes,
    proven_pure,
    result_kept,
    rule_of,
)


def test_every_node_class_has_a_rule():
    """A class this Python's ``ast`` defines and the table does not name
    would fall back without anyone deciding it should.
    """
    missing = sorted(
        cls.__name__ for cls in node_classes() if cls.__name__ not in OPERATIONS
    )
    assert missing == []


def test_a_form_without_a_rule_falls_back():
    class Unlisted(ast.expr):
        pass

    assert rule_of(Unlisted()) is Rule.FALLBACK
    assert rule_of(ast.Name(id="x", ctx=ast.Load())) is Rule.NAME


def test_every_rule_says_what_it_does():
    assert all(rule.value for rule in Rule)
    used = set(OPERATIONS.values())
    assert set(Rule) - used == {Rule.FALLBACK}


_SCALE = 2.0
_TABLE = [1.0, 2.0]


def _pure(x, y=1.0):
    return math.sqrt(x * x + y * y) * _SCALE + np.sin(x)


def _reads_a_list(i):
    return _TABLE[i]


def _reads_an_argument(m):
    return m.width


def _calls_pure(x):
    return _pure(x) + 1


def test_a_function_of_its_arguments_and_inert_globals_is_pure():
    assert proven_pure(_pure) is True
    assert proven_pure(_calls_pure) is True


def test_a_function_reading_a_mutable_global_is_not_pure():
    why = proven_pure(_reads_a_list)
    assert isinstance(why, str)
    assert "_TABLE" in why
    assert "width" in proven_pure(_reads_an_argument)


def _corners(n):
    return [float(i) for i in range(n)]


def _length(n):
    return float(n) * 2.0


def _appends(m):
    points = _corners(3)
    points.append(1.0)


def _reads(m):
    points = _corners(3)
    return len(points) + sum(points) + points[0]


def _uses_length(m):
    return _length(3) + 1


def test_a_result_a_caller_appends_to_is_kept():
    why = result_kept(_corners, [_appends])
    assert isinstance(why, str)
    assert "append" in why


def test_an_inert_result_is_never_kept():
    assert result_kept(_length, [_uses_length, _appends]) is None


def test_a_mutable_result_no_code_changes_is_not_kept():
    assert result_kept(_corners, [_reads]) is None
