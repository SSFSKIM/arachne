"""A scene of the marks that carry a number, each declaring it; the
vessel is filled further between two waits.
"""

from __future__ import annotations

from arachne import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    ChargedParticle,
    Dot,
    DottedTrail,
    FlowArrow,
    Particles,
    Scene,
    Vessel,
    WavefrontArcs,
    WavyLine,
)


class NumberedMarks(Scene):
    """Every mark that carries a number, at rest; the vessel's level is
    raised from 0.5 to 1.5 at half a second.
    """

    def construct(self) -> None:
        vessel = Vessel(level=0.5).shift(LEFT * 4)
        scatter = Particles(count=12, width=2, height=1).shift(UP * 2)
        fronts = WavefrontArcs(count=3, spacing=0.25).shift(DOWN * 2)
        walker = Dot().shift(RIGHT * 4)
        trail = DottedTrail(walker, interval=0.25)
        wave = WavyLine(LEFT, RIGHT, wavelengths=3)
        arrow = FlowArrow(LEFT + UP, RIGHT + UP, magnitude=2.5)
        ion = ChargedParticle(-2).shift(RIGHT * 2 + DOWN)
        self.add(vessel, scatter, fronts, walker, trail, wave, arrow, ion)
        self.wait(0.5)
        vessel.set_level(1.5)
        self.wait(0.5)
