"""Finding a callback's source: the node a function's code was compiled from,
whichever CPython release placed its instructions.

CPython 3.11 and some 3.12 releases (3.12.3 among them) place a lambda's
implicit return at column 0 of its first line, outside the lambda; later
releases place it on the lambda's body. ``as_on_3_12_3`` gives every
interpreter the earlier placement, so these tests hold on each.
"""

from __future__ import annotations

import ast
import contextlib
import dis
from pathlib import Path

import pytest

from arachne import Dot
from arachne.timeline.purity import (
    _PROLOGUE,
    Refused,
    _function_node,
    prove_setter_body,
)

from . import route_scenes

ROUTE_SCENES = Path(route_scenes.__file__)
_RETURNS = {"RETURN_VALUE", "RETURN_CONST"}


@pytest.fixture
def as_on_3_12_3(monkeypatch):
    """Place every lambda's return at column 0 of the lambda's first line."""
    real = dis.get_instructions

    def get_instructions(code, *args, **kwargs):
        for instruction in real(code, *args, **kwargs):
            if code.co_name == "<lambda>" and instruction.opname in _RETURNS:
                line = code.co_firstlineno
                instruction = instruction._replace(
                    positions=dis.Positions(line, line, 0, 0)
                )
            yield instruction

    monkeypatch.setattr(dis, "get_instructions", get_instructions)


def _body(function) -> str:
    return ast.unparse(_function_node(function).body)


def test_a_lambda_whose_return_has_no_column_is_found(as_on_3_12_3):
    tracker = Dot()
    follow = lambda m: m.set_x(tracker.get_x())  # noqa: E731
    node = _function_node(follow)
    assert isinstance(node, ast.Lambda)
    assert node.lineno == follow.__code__.co_firstlineno
    assert ast.unparse(node.body) == "m.set_x(tracker.get_x())"
    prove_setter_body(follow, Dot(), None)


def test_a_multiline_lambda_whose_return_has_no_column_is_found(as_on_3_12_3):
    tracker = Dot()
    follow = lambda m: m.set_x(  # noqa: E731
        tracker.get_x()
    )
    assert _body(follow) == "m.set_x(tracker.get_x())"


@pytest.mark.parametrize("placement", ["interpreter", "column 0"])
def test_each_of_two_constant_lambdas_on_one_line_is_itself_or_refused(
    placement, request
):
    if placement == "column 0":
        request.getfixturevalue("as_on_3_12_3")
    pair = (lambda: 1, lambda: 10)
    for function, body in zip(pair, ["1", "10"], strict=True):
        with contextlib.suppress(Refused):
            assert _body(function) == body


def _located(function) -> bool:
    """Whether an instruction of the function's body is placed on its source:
    on 3.12.3 a constant lambda's only one, its return, is not.
    """
    code = function.__code__
    first = code.co_firstlineno
    return any(
        instruction.opname not in _PROLOGUE
        and instruction.positions.col_offset is not None
        and tuple(instruction.positions) != (first, first, 0, 0)
        for instruction in dis.get_instructions(code)
    )


@pytest.mark.parametrize("placement", ["interpreter", "column 0"])
def test_a_lambda_whose_body_is_a_lambda_on_its_line_is_refused(placement, request):
    # Every located instruction of the outer lambda lies inside the inner one,
    # and the outer makes a lambda of that line: either could be its source.
    if placement == "column 0":
        request.getfixturevalue("as_on_3_12_3")
    outer = lambda: lambda: 1  # noqa: E731
    with pytest.raises(Refused, match="ambiguous"):
        _function_node(outer)
    # The inner makes no lambda, so the narrower node is its source; where no
    # instruction of it is located, it is refused.
    if _located(outer()):
        assert _body(outer()) == "1"
    else:
        with pytest.raises(Refused, match="ambiguous"):
            _function_node(outer())
    building = lambda: lambda: Dot()  # noqa: E731
    with pytest.raises(Refused, match="ambiguous"):
        _function_node(building)
    assert _body(building()) == "Dot()"


def _redrawn(function):
    """A stand-in for ``always_redraw`` that gives back the function it was handed."""
    return function


def _redrawn_pair():
    make = lambda: _redrawn(lambda: Dot())  # noqa: E731
    return make, "_redrawn(lambda: Dot())", make(), "Dot()"


def _updater_pair():
    attach = lambda m: m.add_updater(lambda m: m)  # noqa: E731
    dot = Dot()
    attach(dot)
    return attach, "m.add_updater(lambda m: m)", dot.get_updaters()[-1], "m"


def _updater_making_a_lambda_pair():
    # The inner makes a lambda too, of other arguments: no node alike to its own.
    attach = lambda m: m.add_updater(lambda m: m.set(f=lambda x: x))  # noqa: E731
    dot = Dot()
    attach(dot)
    inner = dot.get_updaters()[-1]
    return (
        attach,
        "m.add_updater(lambda m: m.set(f=lambda x: x))",
        inner,
        "m.set(f=lambda x: x)",
    )


@pytest.mark.parametrize("placement", ["interpreter", "column 0"])
@pytest.mark.parametrize(
    "pair", [_redrawn_pair, _updater_pair, _updater_making_a_lambda_pair]
)
def test_a_lambda_held_by_a_lambda_on_its_line_and_its_holder_are_each_found(
    pair, placement, request
):
    # The outer is located outside the lambda it holds, so it is the only
    # candidate; the inner is located inside both, makes no lambda of its own
    # line and arguments, and is the narrower.
    if placement == "column 0":
        request.getfixturevalue("as_on_3_12_3")
    outer, outer_body, inner, inner_body = pair()
    assert _body(outer) == outer_body
    assert _body(inner) == inner_body


@pytest.mark.parametrize("placement", ["interpreter", "column 0"])
def test_code_compiled_from_other_source_is_refused(placement, request):
    if placement == "column 0":
        request.getfixturevalue("as_on_3_12_3")
    decoy = lambda m: m.set_x(1)  # noqa: E731
    line = decoy.__code__.co_firstlineno
    # Code that claims the decoy's line of this file, from another text: its
    # body starts left of the decoy.
    text = "\n" * (line - 1) + "lambda m: m.set_y(0)"
    other = eval(compile(text, __file__, "eval"), {})
    with pytest.raises(Refused, match="does not match its code"):
        _function_node(other)


def test_a_lambda_updater_routes_like_its_def_twin(as_on_3_12_3, session_of):
    routes = {
        name: [
            play.route
            for play in session_of(name, ROUTE_SCENES, probe=False).record.plays
        ]
        for name in ("LambdaUpdater", "DefUpdater")
    }
    assert routes == {"LambdaUpdater": ["state"], "DefUpdater": ["state"]}
