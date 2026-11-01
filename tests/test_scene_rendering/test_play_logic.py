from __future__ import annotations

from fractions import Fraction
from unittest.mock import Mock

import pytest

from arachne import (
    Dot,
    Mobject,
    Scene,
    ValueTracker,
    Wait,
)

from .simple_scenes import (
    SceneForFrozenFrameTests,
    SceneWithMultipleCalls,
    SceneWithNonStaticWait,
    SceneWithSceneUpdater,
    SceneWithStaticWait,
    SquareToCircle,
)

# CE evaluated each play once per output frame on the play's own grid, and a
# skipped, cached or still play once, at its end. Arachne evaluates every play
# at its start and at each 60 Hz tick of the scene clock whatever is drawn
# (deliberate break 4), and frames read the record on one global grid
# (break 3). The t-value tests below hold the new rule.


def _frames_written(scene):
    """The repeat counts of the frames the scene's renderer writes."""
    writer = scene.renderer.file_writer
    written = []
    write_frame = writer.write_frame

    def counting(pixels, *, repeat=1):
        written.append(repeat)
        return write_frame(pixels, repeat=repeat)

    writer.write_frame = counting
    return written


def _instants(play):
    return [capture.instant for capture in play.samples]


@pytest.mark.parametrize("frame_rate", argvalues=[15, 30, 60])
def test_t_values(config, using_temp_config, disabling_caching, frame_rate):
    """A play is evaluated on the tick; the frame rate only picks which of
    its instants are kept and drawn, one frame per instant of its grid.
    """
    config.frame_rate = frame_rate
    scene = SquareToCircle()
    written = _frames_written(scene)
    scene.render()
    (play,) = scene.record.plays
    assert sum(written) == frame_rate
    assert _instants(play) == [Fraction(k, frame_rate) for k in range(frame_rate)]
    assert play.route == "ticked"


def test_t_values_with_skip_animations(using_temp_config, disabling_caching):
    """A skipped play runs in full, and draws nothing."""
    scene = SquareToCircle()
    scene.renderer._original_skipping_status = True
    written = _frames_written(scene)
    scene.render()
    (play,) = scene.record.plays
    assert play.skipped
    assert play.end == 1
    assert sum(written) == 0


def test_static_wait_detection(using_temp_config, disabling_caching):
    """Test if a static wait (wait that freeze the frame) is correctly detected"""
    scene = SceneWithStaticWait()
    scene.render()
    # Test is is_static_wait of the Wait animation has been set to True by compile_animation_ata
    assert scene.animations[0].is_static_wait
    assert scene.is_current_animation_frozen_frame()


def test_non_static_wait_detection(using_temp_config, disabling_caching):
    scene = SceneWithNonStaticWait()
    scene.render()
    assert not scene.animations[0].is_static_wait
    assert not scene.is_current_animation_frozen_frame()
    scene = SceneWithSceneUpdater()
    scene.render()
    assert not scene.animations[0].is_static_wait
    assert not scene.is_current_animation_frozen_frame()


def test_wait_with_stop_condition(using_temp_config, disabling_caching):
    class TestScene(Scene):
        def construct(self):
            self.wait_until(lambda: self.time >= 1)
            assert self.time >= 1
            d = Dot()
            d.add_updater(lambda mobj, dt: self.add(Mobject()))
            self.add(d)
            self.play(Wait(run_time=5, stop_condition=lambda: len(self.mobjects) > 5))
            assert len(self.mobjects) > 5
            assert self.time < 2

    scene = TestScene()
    scene.render()


def test_frozen_frame(using_temp_config, disabling_caching):
    scene = SceneForFrozenFrameTests()
    scene.render()
    assert scene.mobject_update_count == 0
    assert scene.scene_update_count == 0


def test_t_values_with_cached_data(using_temp_config):
    """Cached plays run forward like any other; their frames are not drawn."""
    scene = SceneWithMultipleCalls()
    # Mocking the file_writer will skip all the writing process.
    scene.renderer.file_writer = Mock(scene.renderer.file_writer)
    scene.renderer.update_skipping_status = Mock()
    # Simulate that all animations are cached.
    scene.renderer.file_writer.is_already_cached.return_value = True

    scene.render()
    plays = scene.record.plays
    assert [(play.start, play.end) for play in plays] == [
        (Fraction(i), Fraction(i + 1)) for i in range(10)
    ]
    assert all(not play.skipped for play in plays)
    scene.renderer.file_writer.write_frame.assert_not_called()


@pytest.mark.parametrize(
    "still_config",
    [{"save_last_frame": True}, {"format": "png"}],
)
def test_t_values_save_last_frame(config, using_temp_config, still_config):
    """Still output runs each play forward in full, draws none of its frames,
    and draws the last frame from the record's final state.
    """
    config.update(still_config)
    scene = SquareToCircle()
    written = _frames_written(scene)
    scene.render()
    (play,) = scene.record.plays
    assert play.skipped
    assert play.end == 1
    assert sum(written) == 0
    assert scene.record.final is not None


def test_png_sequence_evaluates_every_frame(config, using_temp_config):
    config.format = "png-sequence"
    scene = SquareToCircle()

    scene.render()

    assert scene.renderer.file_writer.frame_count == config.frame_rate
    assert scene.renderer.file_writer.final_file_path.is_dir()
    assert (scene.renderer.file_writer.final_file_path / "0000.png").is_file()


def test_animate_with_changed_custom_attribute(using_temp_config):
    """Test that animating the change of a custom attribute
    using the animate syntax works correctly.
    """

    class CustomAnimateScene(Scene):
        def construct(self):
            vt = ValueTracker(0)
            vt.custom_attribute = "hello"
            self.play(vt.animate.set_value(42).set(custom_attribute="world"))
            assert vt.get_value() == 42
            assert vt.custom_attribute == "world"

    CustomAnimateScene().render()
