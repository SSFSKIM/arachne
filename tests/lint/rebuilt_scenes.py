"""Scenes for ``motion.jump`` on groups ``become`` rebuilds
(``test_motion_rebuilt.py``): streams whose list of items grows or turns over
between plays, rigs redrawn about a fixed centre, and controls.

A stream is a beam of particles leaving a source: a trail per particle, then
an arrow per particle, each arrow longer as its particle ages, so no two
look alike. A particle spawned between plays is a newcomer at the source.
"""

from __future__ import annotations

from arachne import (
    PI,
    RIGHT,
    Arrow,
    Circle,
    Dot,
    Line,
    Mobject,
    Scene,
    ValueTracker,
    VGroup,
    always_redraw,
    linear,
    rotate_vector,
)

#: Where particles leave from, and how fast they go, in scene units.
SOURCE = -5.0
SPEED = 2.0


def _x(clock: dict[str, float], born: float) -> float:
    return SOURCE + SPEED * (clock["t"] - born)


def _arrow(clock: dict[str, float], born: float, growth: float = 0.1) -> Arrow:
    x = _x(clock, born)
    return Arrow([x, 0, 0], [x + 0.4 + growth * (clock["t"] - born), 0, 0], buff=0)


def beam(clock: dict[str, float], born: list[float], growth: float = 0.1) -> VGroup:
    """A trail per particle, then an arrow per particle, ``growth`` longer
    for each second of its age: a spawn puts a trail before the arrows, so
    every arrow's place in the list moves on by one.
    """
    trails = [Line([SOURCE, -0.6, 0], [_x(clock, b), -0.6, 0]) for b in born]
    return VGroup(*trails, *(_arrow(clock, b, growth) for b in born))


def arrows(
    clock: dict[str, float], born: list[float], *, before: float = 99.0
) -> VGroup:
    """An arrow per particle that has not yet passed ``before``."""
    return VGroup(*(_arrow(clock, b) for b in born if _x(clock, b) < before))


def _clock(scene: Scene) -> dict[str, float]:
    """A clock the scene's frames advance."""
    clock = {"t": 0.0}
    ticker = Mobject().add_updater(
        lambda mob, dt: clock.__setitem__("t", clock["t"] + dt)
    )
    scene.add(ticker)
    return clock


class StreamGrows(Scene):
    """A particle spawns between plays: the redrawn beam gains a trail and an
    arrow, and hands its old children each other's items.
    """

    def construct(self):
        clock, born = _clock(self), [-2.0, -1.4, -0.8]
        self.add(always_redraw(lambda: beam(clock, born)))
        self.wait(1.0)
        born.append(clock["t"])
        self.wait(1.0)


class StreamGrowsByItsOwnBecome(Scene):
    """The same stream, rebuilt by an updater of its own calling ``become``."""

    def construct(self):
        clock, born = _clock(self), [-2.0, -1.4, -0.8]
        self.add(beam(clock, born).add_updater(lambda m: m.become(beam(clock, born))))
        self.wait(1.0)
        born.append(clock["t"])
        self.wait(1.0)


class StreamOfUnlikeArrowsGrows(Scene):
    """A stream whose arrows grow fast with age: an arrow that takes the item
    of the one after it is no look-alike of it.
    """

    def construct(self):
        clock, born = _clock(self), [-3.0, -1.5, -0.5]
        self.add(always_redraw(lambda: beam(clock, born, growth=0.5)))
        self.wait(1.0)
        born.append(clock["t"])
        self.wait(1.0)


class StreamTurnsOver(Scene):
    """Between plays the oldest particle leaves as a newcomer spawns: the
    count holds.
    """

    def construct(self):
        clock, born = _clock(self), [-2.0, -1.4, -0.8]
        self.add(always_redraw(lambda: arrows(clock, born)))
        self.wait(1.0)
        born.pop(0)
        born.append(clock["t"])
        self.wait(1.0)


#: Where an arrow of :class:`StreamRefillsAVacatedSlot` leaves the list.
LEAVES_AT = 1.0


class StreamRefillsAVacatedSlot(Scene):
    """The oldest arrow passes :data:`LEAVES_AT` within the first play and
    leaves the list; ``become`` keeps the group at its larger count, padded,
    so its count holds. A newcomer spawns at the cut into that slot.
    """

    def construct(self):
        clock, born = _clock(self), [-2.0, -1.4, -0.8]
        self.add(always_redraw(lambda: arrows(clock, born, before=LEAVES_AT)))
        self.wait(1.0)
        born.append(clock["t"])
        self.wait(1.0)


class RebuiltGroupTeleports(Scene):
    """A redrawn row of three arrows slides, is set back, and slides again."""

    def construct(self):
        x = ValueTracker(-3.0)
        self.add(
            always_redraw(
                lambda: VGroup(
                    *(
                        Arrow(
                            [x.get_value() + k, 1, 0],
                            [x.get_value() + k + 0.5, 1, 0],
                            buff=0,
                        )
                        for k in range(3)
                    )
                )
            )
        )
        self.play(x.animate.set_value(0.0), run_time=1.0)
        x.set_value(-3.0)
        self.play(x.animate.set_value(0.0), run_time=1.0)


#: The angles of a rig's outer atoms about its fixed centre atom.
RIG_ANGLES = (0.0, 2.0, 4.0)


def rig(theta: float, angles: tuple[float, ...] = RIG_ANGLES) -> VGroup:
    """A rig of outer atoms at ``angles`` on the unit circle, turned by
    ``theta`` about a centre atom, in a ring that holds the group's box.
    """
    atoms = [Dot(rotate_vector(RIGHT, a + theta), radius=0.12) for a in angles]
    return VGroup(Circle(radius=1.6), Dot([0, 0, 0], radius=0.2), *atoms)


#: An evenly spaced rig's outer atoms, and the spacing between them.
SPACING = 2 * PI / 3
EVEN_ANGLES = (0.0, SPACING, 2 * SPACING)


class RigTurnsByItsSpacing(Scene):
    """Known limit (a permutation coincidence): an evenly spaced rig turns by
    exactly its spacing, snaps back at the cut, and turns again. Each atom set
    back lands where a sibling stood, and a sibling stands where it left, so
    the picture is the same rig and no atom is read as jumping. Telling the
    snap from a relabelling needs where each atom's item came from: a
    source-aware rule is the open fix.
    """

    def construct(self):
        theta = ValueTracker(0.0)
        self.add(always_redraw(lambda: rig(theta.get_value(), EVEN_ANGLES)))
        self.play(theta.animate.set_value(SPACING), rate_func=linear, run_time=1.0)
        theta.set_value(0.0)
        self.play(theta.animate.set_value(SPACING), rate_func=linear, run_time=1.0)


class RigSnapsBack(Scene):
    """The rig turns, snaps back at the cut, and turns again."""

    def construct(self):
        theta = ValueTracker(0.0)
        self.add(always_redraw(lambda: rig(theta.get_value())))
        self.play(theta.animate.set_value(1.0), rate_func=linear, run_time=1.0)
        theta.set_value(0.0)
        self.play(theta.animate.set_value(1.0), rate_func=linear, run_time=1.0)


#: The three atoms of :func:`one_loose` that hold.
HELD = ((0.0, 1.0, 0.0), (-1.0, 0.6, 0.0), (1.0, 0.6, 0.0))


def one_loose(x: float) -> VGroup:
    """Four atoms in a ring: three hold, the fourth rides ``x`` along
    y = -0.8.
    """
    return VGroup(
        Circle(radius=1.6),
        *(Dot(p, radius=0.12) for p in HELD),
        Dot((x, -0.8, 0.0), radius=0.12),
    )


class OneAtomSnapsBack(Scene):
    """One atom of four snaps back at the cut while the other three hold."""

    def construct(self):
        x = ValueTracker(-1.0)
        self.add(always_redraw(lambda: one_loose(x.get_value())))
        self.play(x.animate.set_value(0.6), rate_func=linear, run_time=1.0)
        x.set_value(-1.0)
        self.play(x.animate.set_value(0.6), rate_func=linear, run_time=1.0)


#: Where a newcomer of :class:`GrowsAsOneSnapsBack` appears.
NEWCOMER = (-5.0, -2.5, 0.0)


class GrowsAsOneSnapsBack(Scene):
    """At the cut the riding atom, first of the group, snaps back and a
    newcomer appears far from the rest: two children land where none stood,
    and the group gained one.
    """

    def construct(self):
        x = ValueTracker(-1.0)
        extra: list[tuple[float, float, float]] = []

        def atoms():
            return VGroup(
                Dot((x.get_value(), -0.8, 0.0), radius=0.12),
                Circle(radius=1.6),
                *(Dot(p, radius=0.12) for p in (*HELD, *extra)),
            )

        self.add(always_redraw(atoms))
        self.play(x.animate.set_value(0.6), rate_func=linear, run_time=1.0)
        x.set_value(-1.0)
        extra.append(NEWCOMER)
        self.play(x.animate.set_value(0.6), rate_func=linear, run_time=1.0)


#: Where :class:`TieBrokenByFirstSight`'s items stand, before their drift.
LEFT_PLACE, RIGHT_PLACE = (-1.0, 0.0), (1.0, 0.0)
FAR_BELOW, FAR_ABOVE = (0.0, -3.0), (0.0, 3.0)


class TieBrokenByFirstSight(Scene):
    """A drifting group of dots, rebuilt twice. The first cut grows it from
    two items to three, so ``become`` pads it by repeating its first child:
    the padded child is first seen after the second, though it stands before
    it in the list. The second cut grows it to four with its padded child
    and its second both landing on one place far from where any child stood,
    as far as each other from every place, and room for one birth only.
    """

    def construct(self):
        clock = _clock(self)
        items = [LEFT_PLACE, RIGHT_PLACE]
        self.add(
            always_redraw(
                lambda: VGroup(
                    *(
                        Dot((x + 0.1 * clock["t"], y, 0.0), radius=0.12)
                        for x, y in items
                    )
                )
            )
        )
        self.wait(1.0)
        items[:] = [LEFT_PLACE, RIGHT_PLACE, LEFT_PLACE]
        self.wait(1.0)
        items[:] = [LEFT_PLACE, FAR_BELOW, FAR_ABOVE, FAR_ABOVE]
        self.wait(1.0)


class PlainGroupPutBack(Scene):
    """Control: a group nothing rebuilds, one of whose arrows the script puts
    back between plays.
    """

    def construct(self):
        arrow = Arrow([-2, 0, 0], [-1, 0, 0], buff=0)
        group = VGroup(arrow, Dot([0, 2, 0]))
        self.add(group)
        self.play(arrow.animate.shift(RIGHT * 4), run_time=1.0)
        arrow.shift(RIGHT * -4)
        self.play(arrow.animate.shift(RIGHT * 4), run_time=1.0)
