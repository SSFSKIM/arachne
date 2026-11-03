"""The operation table (:mod:`arachne.timeline.coverage`): every node class of
the running Python has a rule, a form without one falls back, and the two
queries built on it (:func:`proven_pure`, :func:`result_kept`) answer from
its rules, as filters for precision only.
"""

from __future__ import annotations

import ast
import math

import numpy as np
import pytest

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


# Known limits of the two queries, which only filter for precision (nothing
# in the stepping check rests on them): each reproducer expects the answer a
# sound query gives, and fails until the query gives it.

_SCALE_SET_ELSEWHERE = 2.0


def _scaled(x):
    return x * _SCALE_SET_ELSEWHERE


def _pair_distance(pair):
    a, b = pair
    return np.linalg.norm(a.get_center() - b.get_center())


def _sum_of_xs(group):
    total = 0.0
    for m in group:
        total += m.get_x()
    return total


def _aliased_shift(m):
    n = m
    n.shift(1)
    return 1


def _counter():
    count = 0

    def bump():
        nonlocal count
        count += 1

    def read():
        return count

    return bump, read


_bump, _read_count = _counter()


def _saves(x):
    np.save("unused.npy", x)
    return x


def _loads(path):
    return np.load(path)


@pytest.mark.xfail(strict=True, reason="precision filter: a known limit")
@pytest.mark.parametrize(
    "function",
    [_scaled, _pair_distance, _sum_of_xs, _aliased_shift, _read_count, _saves, _loads],
    ids=["global", "unpacked", "loop", "alias", "nonlocal", "save", "load"],
)
def test_what_a_function_reads_beyond_its_arguments_is_not_pure(function):
    """A global another module may rebind, an attribute of what an argument
    holds (unpacked, iterated, aliased), a closure's name another function
    rebinds, NumPy's input and output: none of these functions is pure.
    """
    assert proven_pure(function) is not True


def _flags(t):
    return np.array(t) > 0


def _changes_flags(t):
    found = _flags(t)
    found[0] = False


def _tables(t):
    return {"a": [1, 2], "b": [3]}


def _changes_through_a_comprehension(t):
    ys = [x for x in _tables(t).values()]  # noqa: C416 (the comprehension is the case)
    ys[0].append(9)


@pytest.mark.xfail(strict=True, reason="precision filter: a known limit")
@pytest.mark.parametrize(
    ("function", "reachable"),
    [(_flags, _changes_flags), (_tables, _changes_through_a_comprehension)],
    ids=["comparison", "comprehension"],
)
def test_a_result_changed_through_what_holds_it_is_kept(function, reachable):
    """An array a comparison makes, changed in place; a list a comprehension
    takes out of a result, changed: each result is kept.
    """
    assert result_kept(function, [reachable]) is not None
