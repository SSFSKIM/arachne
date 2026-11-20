"""Scenes whose readout and gauges declare their values: each counted
linearly over one second, then one gauge set by the scene's code.
"""

from __future__ import annotations

from arachne import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    BarGauge,
    ColumnGauge,
    CountUp,
    DialGauge,
    Readout,
    Scene,
)
from arachne.utils.rate_functions import linear


class Counted(Scene):
    """A readout counted from 0 to 10 and three gauges counted in the same
    play, between two waits; the dial is then set to 60.
    """

    def construct(self) -> None:
        reading = Readout(0, "{:.1f} mV").shift(UP)
        bar = BarGauge(0.25, value_range=(0, 1)).shift(LEFT * 3)
        column = ColumnGauge(2, value_range=(0, 4)).shift(DOWN)
        dial = DialGauge(30, value_range=(0, 100)).shift(RIGHT * 3)
        self.add(reading, bar, column, dial)
        self.wait(0.5)
        self.play(
            CountUp(reading, 10, rate_func=linear),
            CountUp(bar, 0.75, rate_func=linear),
            CountUp(column, 4, rate_func=linear),
            CountUp(dial, 50, rate_func=linear),
            run_time=1,
        )
        dial.set_value(60)
        self.wait(0.5)
