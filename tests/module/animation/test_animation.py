from __future__ import annotations

from fractions import Fraction

import pytest

from arachne import FadeIn, Scene


def test_animation_zero_total_run_time():
    test_scene = Scene()
    with pytest.raises(
        ValueError, match="The total run_time must be a positive number."
    ):
        test_scene.play(FadeIn(None, run_time=0))


def test_single_animation_zero_run_time_with_more_animations():
    test_scene = Scene()
    test_scene.play(FadeIn(None, run_time=0), FadeIn(None, run_time=1))


def test_animation_negative_run_time():
    with pytest.raises(ValueError, match="The run_time of FadeIn cannot be negative."):
        FadeIn(None, run_time=-1)


# CE raised a play shorter than one frame to one frame, with a warning.
# Arachne keeps every duration exact, whatever the frame rate (deliberate
# break 3).
def test_animation_run_time_shorter_than_frame_rate(manim_caplog, config):
    test_scene = Scene()
    run_time = 1 / (config.frame_rate + 1)
    test_scene.play(FadeIn(None, run_time=run_time))
    assert "too short" not in manim_caplog.text
    assert test_scene.time == Fraction(1, int(config.frame_rate) + 1)


@pytest.mark.parametrize("duration", [0, -1])
def test_wait_invalid_duration(duration):
    test_scene = Scene()
    with pytest.raises(ValueError, match="The duration must be a positive number."):
        test_scene.wait(duration)


@pytest.mark.parametrize("frozen_frame", [False, True])
def test_wait_duration_shorter_than_frame_rate(manim_caplog, frozen_frame):
    test_scene = Scene()
    test_scene.wait(1e-9, frozen_frame=frozen_frame)
    assert "too short" not in manim_caplog.text
    assert test_scene.time == Fraction(1, 10**9)


@pytest.mark.parametrize("duration", [0, -1])
def test_pause_invalid_duration(duration):
    test_scene = Scene()
    with pytest.raises(ValueError, match="The duration must be a positive number."):
        test_scene.pause(duration)


@pytest.mark.parametrize("max_time", [0, -1])
def test_wait_until_invalid_max_time(max_time):
    test_scene = Scene()
    with pytest.raises(ValueError, match="The max_time must be a positive number."):
        test_scene.wait_until(lambda: True, max_time)
