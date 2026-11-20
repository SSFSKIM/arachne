"""A gauge shows a declared value: a bar's or a column's fill is the value's
share of the gauge's length, a dial's needle turns through the value's
share of its sweep, wherever the gauge stands and however it is turned.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from arachne import DEGREES, LEFT, UP, BarGauge, ColumnGauge, DialGauge


def _extent(mobject, axis: int) -> float:
    points = np.asarray(mobject.points)
    return float(points[:, axis].max() - points[:, axis].min())


@pytest.mark.parametrize("value", [0.0, 3.0, 7.5, 10.0])
def test_a_bar_s_fill_is_the_value_s_share_of_its_length(value):
    gauge = BarGauge(value, (0, 10), length=4).shift(LEFT + UP)
    assert gauge.value == value
    assert _extent(gauge.fill, 0) == pytest.approx(4 * value / 10, abs=1e-9)
    # It starts at the track's low end.
    assert gauge.fill.points[:, 0].min() == pytest.approx(
        gauge.track.points[:, 0].min(), abs=1e-9
    )
    gauge.set_value(10 - value)
    assert _extent(gauge.fill, 0) == pytest.approx(4 * (10 - value) / 10, abs=1e-9)


@pytest.mark.parametrize("value", [-20.0, 0.0, 35.0])
def test_a_column_s_fill_is_the_value_s_share_of_its_height(value):
    gauge = ColumnGauge(value, (-20, 40), length=3)
    assert _extent(gauge.fill, 1) == pytest.approx(3 * (value + 20) / 60, abs=1e-9)
    assert gauge.fill.points[:, 1].min() == pytest.approx(
        gauge.track.points[:, 1].min(), abs=1e-9
    )


def _needle_angle(gauge: DialGauge) -> float:
    centre = gauge.hub.get_center()
    tip = gauge.needle.points[-1]
    return math.atan2(tip[1] - centre[1], tip[0] - centre[0])


@pytest.mark.parametrize("value", [0.0, 0.25, 1.0])
def test_a_dial_s_needle_turns_through_the_value_s_share_of_its_sweep(value):
    gauge = DialGauge(value, (0, 1))
    # From 225 degrees clockwise to -45 by default.
    expected = math.radians(225 - 270 * value)
    turned = (_needle_angle(gauge) - expected + math.pi) % (2 * math.pi) - math.pi
    assert turned == pytest.approx(0, abs=1e-9)
    # Turned with the gauge, the needle keeps its share of the sweep.
    gauge.rotate(30 * DEGREES).shift(LEFT)
    gauge.set_value(value)
    turned = (_needle_angle(gauge) - expected - math.radians(30) + math.pi) % (
        2 * math.pi
    ) - math.pi
    assert turned == pytest.approx(0, abs=1e-9)


@pytest.mark.parametrize("make", [BarGauge, ColumnGauge, DialGauge])
def test_a_value_outside_a_gauge_s_scale_is_refused(make):
    with pytest.raises(ValueError, match="scale"):
        make(1.5, (0, 1))
    gauge = make(0.5, (0, 1))
    with pytest.raises(ValueError, match="scale"):
        gauge.set_value(-0.1)
    with pytest.raises(ValueError, match="min to a larger max"):
        make(1, (1, 1))
