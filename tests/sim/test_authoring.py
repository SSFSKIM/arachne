"""The authoring layer's rules that need no solver: the one conversion of a
number to the stored type, what a callback force may read, a world without
the simulation extra, and an open stay's rows read through the bakes.
"""

from __future__ import annotations

import functools
import math
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from arachne import Dot, Scene
from arachne.lint.raised import RaisedFinding
from arachne.simulation import inputs
from arachne.simulation.declaration import stored, stored_vector
from arachne.timeline.bake import BakeContent, BakeIdentity, Bakes, Channel, Row

GLOBAL_DOT = Dot()
GLOBAL_DOTS = [Dot(), Dot()]


def test_a_number_is_stored_once_rounded_to_single_precision():
    assert stored(0.1, "x") == float(np.float32(0.1))
    assert stored(0.1, "x") != 0.1
    # Round to nearest, ties to even: 1 + 2**-24 is halfway to the next float32.
    assert stored(1 + 2**-24, "x") == 1.0
    assert stored(1 + 3 * 2**-24, "x") == 1 + 2**-22
    # A negative zero is stored as it was given.
    assert math.copysign(1, stored(-0.0, "x")) == -1
    assert stored(3, "x") == 3.0
    stored_again = stored(stored(0.1, "x"), "x")
    assert stored_again == stored(0.1, "x")


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan, 1e39])
def test_a_number_single_precision_cannot_hold_is_refused(value):
    with pytest.raises(ValueError, match="A mass is a finite number"):
        stored(value, "A mass")


@pytest.mark.parametrize("value", [True, "1", None])
def test_a_value_that_is_not_a_number_is_refused(value):
    with pytest.raises(TypeError, match="is a number"):
        stored(value, "A mass")


def test_a_2d_world_takes_vectors_in_its_plane():
    assert stored_vector((1.0, 2.0, 0.0), 2, "v") == (1.0, 2.0)
    assert stored_vector(np.array([1.0, 2.0]), 2, "v") == (1.0, 2.0)
    with pytest.raises(ValueError, match="plane"):
        stored_vector((1.0, 2.0, 0.5), 2, "v")
    with pytest.raises(ValueError, match="takes 3 numbers"):
        stored_vector((1.0, 2.0), 3, "v")


def test_a_callback_force_that_can_reach_a_mobject_or_the_scene_is_refused():
    dot = Dot()
    scene = Scene.__new__(Scene)

    def through_closure(state):
        return dot

    def through_a_global(state):
        return GLOBAL_DOT

    def through_a_list(state):
        return GLOBAL_DOTS[0]

    def through_a_default(state, shown=dot):
        return shown

    def through_the_scene(state):
        return scene

    def nested(state):
        return [GLOBAL_DOT for _ in range(1)]

    class Pusher:
        def __init__(self):
            self.shown = dot

        def __call__(self, state):
            return None

    cases = {
        through_closure: ["dot"],
        through_a_global: ["GLOBAL_DOT"],
        through_a_list: ["GLOBAL_DOTS"],
        through_a_default: ["shown"],
        through_the_scene: ["scene"],
        nested: ["GLOBAL_DOT"],
        functools.partial(through_closure, dot): ["argument 0", "dot"],
        Pusher(): ["shown"],
    }
    for force, names in cases.items():
        with pytest.raises(RaisedFinding) as raised:
            inputs.check_force(force)
        assert raised.value.code == "sim.force_input"
        assert raised.value.detail["names"] == names, force


def test_a_callback_force_that_reads_only_what_it_is_handed_is_declared():
    handle = SimpleNamespace(index=0)
    holder = SimpleNamespace(mob=Dot())
    count = []

    def handed(state):
        count.append(state.tick)
        return {handle: state.position(handle) * -1.0}

    # A mobject behind another object's attribute is beyond the check (and
    # not supported).
    def behind_an_object(state):
        return holder.mob

    inputs.check_force(handed)
    inputs.check_force(behind_an_object)
    with pytest.raises(TypeError, match="callable"):
        inputs.check_force(3)


def test_nothing_is_withheld_at_the_solver_pin():
    assert dict(inputs.WITHHELD) == {}
    inputs.check_feature("joint.pin.limits")


def test_a_world_without_the_simulation_extra_names_it(monkeypatch):
    from arachne import World

    monkeypatch.setitem(sys.modules, "arachne_sim", None)
    with pytest.raises(ImportError, match="'sim' extra"):
        World()


def test_the_engine_imports_no_simulation_until_a_world_is_made():
    root = Path(__file__).resolve().parents[2]
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, arachne; print(sorted(m for m in sys.modules"
            " if m.split('.')[0] == 'arachne_sim'))",
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=root,
    )
    assert done.stdout.splitlines()[-1] == "[]"


class _OpenStay:
    """What computes a stay, as the bakes see it: rows it has computed, and
    whether it is done.
    """

    def __init__(self, bakes, rows):
        self.bakes = bakes
        self.rows = rows
        self.done = False
        self.ordinal = bakes.begin(self)

    def row(self, tick):
        if not 0 <= tick < len(self.rows):
            return None
        return Row(tick, tick, MappingProxyType({"x": self.rows[tick]}))

    def settle(self):
        if self.done:
            content = BakeContent(
                BakeIdentity("test", "1", "rows", "1", "none", "python"),
                len(self.rows),
                {"x": Channel("f64", "u", np.array(self.rows, dtype=float))},
            )
            self.bakes.seal(
                self.ordinal, content, first_tick=0, closed=True, declaration=b"d"
            )


def test_an_open_stay_s_rows_are_read_before_it_is_sealed():
    bakes = Bakes()
    stay = _OpenStay(bakes, [np.array([0.0]), np.array([1.5])])
    assert bakes.occurrences == ()
    assert bakes.row(stay.ordinal, 1)["x"].tolist() == [1.5]
    assert bakes.at(stay.ordinal, Fraction(1, 60))["x"].tolist() == [1.5]
    assert bakes.row(stay.ordinal, 2) is None
    stay.done = True
    # The next read lets it seal itself: the same rows, now sealed.
    (occurrence,) = bakes.occurrences
    assert occurrence.ordinal == stay.ordinal
    assert bakes.row(stay.ordinal, 1)["x"].tolist() == [1.5]
    assert bakes.contents[occurrence.content].ticks == 2


@pytest.mark.parametrize(
    ("given", "error", "message"),
    [
        ((1.0, 0.0, 1.0), ValueError, "plane of a 2D world"),
        ((1.0, 0.0, float("nan")), ValueError, "finite"),
        ((1.0, float("inf")), ValueError, "finite"),
        (("1.0", 0.0), TypeError, "is a number"),
        ((True, 0.0), TypeError, "is a number"),
        ((1.0,), ValueError, "takes 2 numbers"),
    ],
)
def test_a_vector_a_scene_gives_is_checked_before_any_arithmetic(given, error, message):
    from arachne.simulation.declaration import given_vector

    with pytest.raises(error, match=message):
        given_vector(given, 2, "A pushed force")


def test_a_checked_vector_keeps_the_scene_s_numbers():
    from arachne.simulation.declaration import given_angular, given_vector

    assert given_vector((0.1, 2, 0.0), 2, "v") == (0.1, 2.0)
    assert given_vector(np.array([1.0, 2.0, 3.0]), 3, "v") == (1.0, 2.0, 3.0)
    assert given_angular(0.5, 2, "w") == (0.5,)


def test_an_occurrence_s_contacts_begin_with_the_pairs_touching_as_it_begins():
    from arachne.simulation.stay import _window_contacts

    solver = np.array(
        [
            [0, 0, 1, 1],  # 0-1 touch from the first row
            [3, 1, 2, 1],
            [5, 1, 2, 0],  # 1-2 part at the window's first row
            [6, 0, 1, 0],
            [7, 0, 2, 1],
        ],
        dtype=np.uint32,
    )
    assert _window_contacts(solver, 0, 4).tolist() == [[0, 0, 1, 1], [3, 1, 2, 1]]
    assert _window_contacts(solver, 5, 9).tolist() == [
        [0, 0, 1, 1],
        [1, 0, 1, 0],
        [2, 0, 2, 1],
    ]
    assert _window_contacts(solver[:0], 2, 4).shape == (0, 4)
