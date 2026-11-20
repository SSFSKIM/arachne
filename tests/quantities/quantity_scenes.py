"""Scenes the declared quantities' tests execute: engine-owned objects that
declare quantities (none a simulation), and literals attached to ordinary
mobjects.
"""

from __future__ import annotations

from arachne import LEFT, RIGHT, UP, Dot, Scene, Square, ValueTracker
from arachne.quantities import Literal, attach
from arachne.utils.testing.declared_probe import (
    BakedNumber,
    DeclaredProbe,
    ProbeGrid,
    bake_rows,
)

#: Read by the updater of :class:`ForwardRead`, rebound between its plays.
SHIFT = 0.0


class ChangedBetweenPlays(Scene):
    """A probe whose number and name change between two waits, and a square
    that holds one literal, then two.
    """

    def construct(self):
        probe = DeclaredProbe(amount=1.0, setting="off").shift(UP)
        square = Square(side_length=1).shift(LEFT * 2)
        plain = Square(side_length=1).shift(RIGHT * 2)
        attach(square, Literal("probe.size", 2.0, "1"))
        self.add(probe, square, plain)
        self.wait(1)
        probe.set_amount(2.5).set_setting("on")
        attach(square, Literal("probe.count", 3, "1", tolerance=0.5))
        self.wait(1)


class ForwardRead(Scene):
    """A probe changed between two state plays, and an updater reading a
    global rebound between them: the first play is answered by a forward
    run between its samples, the second restored.
    """

    def construct(self):
        global SHIFT
        SHIFT = 0.0
        probe = DeclaredProbe(amount=1.0)
        tracker = ValueTracker(0)
        dot = Dot().add_updater(lambda m: m.set_x(tracker.get_value() + SHIFT))
        attach(dot, Literal("probe.size", 1.0, "1"))
        self.add(probe, dot)
        self.play(tracker.animate.set_value(2), run_time=1)
        SHIFT = 1.0
        probe.set_amount(2.0)
        attach(dot, Literal("probe.count", 2, "1"))
        self.play(tracker.animate.set_value(0), run_time=1)


class OverridingProbe(DeclaredProbe):
    """A scene's subclass naming, on the class, what the engine's provider is
    called, to stand in for it.
    """

    @staticmethod
    def _declared(held, at):
        return ()

    def quantities(self):
        return ()


class SubclassedProbe(Scene):
    def construct(self):
        self.add(OverridingProbe(amount=5.0, setting="on"))
        self.wait(0.5)


class Unreadables(Scene):
    """A quantity of each fault on its own square: an unregistered prefix, an
    unregistered unit, a value that is not finite, and a source with no value
    before its bake's stay begins at 1 s.
    """

    def construct(self):
        later = bake_rows([[1.0]] * 30, channel="level", unit="1", first_tick=60)
        prefix = Square().shift(LEFT * 3)
        attach(prefix, Literal("nothing.x", 1.0, "1"))  # @prefix
        unit = Square().shift(LEFT)
        attach(unit, Literal("probe.length", 1.0, "furlong"))  # @unit
        finite = Square().shift(RIGHT)
        attach(finite, Literal("probe.length", float("nan"), "1"))  # @finite
        missing = Square().shift(RIGHT * 3)
        attach(missing, BakedNumber(later, "level", "probe.level", "1", (0,)))
        self.add(prefix, unit, finite, missing)
        self.wait(1)


class Readables(Scene):
    """:class:`Unreadables` with each fault mended: the clean twin."""

    def construct(self):
        always = bake_rows([[1.0]] * 61, channel="level", unit="1")
        prefix = Square().shift(LEFT * 3)
        # A vector with nothing in it yet is a value: no contact so far.
        attach(prefix, Literal("probe.x", 1.0, "1"), Literal("probe.times", [], "1"))
        unit = Square().shift(LEFT)
        attach(unit, Literal("probe.length", 1.0, "1"))
        finite = Square().shift(RIGHT)
        attach(finite, Literal("probe.length", 1.5, "1"))
        missing = Square().shift(RIGHT * 3)
        attach(missing, BakedNumber(always, "level", "probe.level", "1", (0,)))
        self.add(prefix, unit, finite, missing)
        self.wait(1)


#: A field of two cells along x: 3 and 7 at tick 0, 4 and 8 from tick 1 to
#: tick 120.
TWO_CELLS = [[[[3.0]], [[7.0]]]] + [[[[4.0]], [[8.0]]]] * 120


class TwoPointField(Scene):
    """The two-point fixture: one object with one channel, ``temperature``
    in K, on a grid of two cells along x, its origin at (-1, 0, 0) and its
    spacing 2, read nearest; the same read linear; a vector channel; and a
    square with no point source. The scene lasts 3 s.
    """

    def construct(self):
        field = bake_rows(TWO_CELLS, channel="temperature", unit="K")
        grid = ProbeGrid(
            field, channel="temperature", unit="K", origin=(-1, 0, 0), spacing=2
        )
        linear = ProbeGrid(
            field,
            channel="temperature",
            unit="K",
            origin=(-1, 0, 0),
            spacing=2,
            interpolation="linear",
        )
        arrows = bake_rows([[[[[1.0, 2.0]]]]] * 181, channel="flow", unit="1")
        flow = ProbeGrid(arrows, channel="flow", unit="1")
        plain = Square().shift(RIGHT * 3)
        self.add(grid, linear, flow, plain)
        self.wait(3)


class Malformed(Scene):
    """A malformed name, a malformed unit and a value of no kind, each on its
    own square: each is answered unreadable with the name and unit declared.
    """

    def construct(self):
        name = Square().shift(LEFT * 3)
        attach(name, Literal("probe.two words", 1.0, "1"))
        unit = Square().shift(LEFT)
        attach(unit, Literal("probe.x", 1.0, ""))
        kind = Square().shift(RIGHT)
        attach(kind, Literal("probe.x", {"a": 1.0}, "1"))
        empty = Square().shift(RIGHT * 3)
        attach(empty, Literal("probe.x", None, "1"))
        self.add(name, unit, kind, empty)
        self.wait(0.5)
