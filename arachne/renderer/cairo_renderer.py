from __future__ import annotations

from collections.abc import Iterable
from fractions import Fraction
from typing import TYPE_CHECKING, Any

from arachne.utils.hashing import get_hash_from_play_call

from .. import config, logger
from .._config.video_encoder import video_encoder_fingerprint
from ..camera.camera import Camera
from ..mobject.mobject import Mobject, _AnimationBuilder
from ..scene.scene_file_writer import SceneFileWriter
from ..timeline.clock import TICK_RATE, exact, frames_in
from ..timeline.draw import Drawing, LiveFrames
from ..timeline.execution import Reexecute
from ..utils.exceptions import EndSceneEarlyException
from ..utils.iterables import list_update
from .protocol import RendererCapabilities

if TYPE_CHECKING:
    from arachne._config.render_session import RenderSessionSpec
    from arachne.animation.animation import Animation
    from arachne.scene.scene import Scene
    from arachne.scene.scene_file_writer import _SceneFileWriterSettings
    from arachne.timeline.record import PlayRecord

    from ..typing import PixelArray

__all__ = ["CairoRenderer"]


class CairoRenderer:
    """A renderer using Cairo.

    Attributes
    ----------
    num_plays : int
        Number of play() functions in the scene.

    time : float
        Time elapsed since initialisation of scene.
    """

    capabilities = RendererCapabilities(live_preview=False)

    def __init__(
        self,
        file_writer_class: type[SceneFileWriter] = SceneFileWriter,
        camera_class: type[Camera] | None = None,
        skip_animations: bool = False,
        **kwargs: Any,
    ):
        # All of the following are set to EITHER the value passed via kwargs,
        # OR the value stored in the global config dict at the time of
        # _instance construction_.
        self._file_writer_class = file_writer_class
        camera_cls = camera_class if camera_class is not None else Camera
        self.camera = camera_cls()
        self._original_skipping_status = skip_animations
        self.skip_animations = skip_animations
        self.animations_hashes: list[str | None] = []
        self.num_plays = 0
        self._clock = Fraction(0)
        self.static_image: PixelArray | None = None

    def renew(self) -> None:
        """Start afresh for another execution of the scene: a new camera of
        the same class, no plays, the clock at zero.
        """
        self.__init__(  # type: ignore[misc]
            self._file_writer_class,
            type(self.camera),
            self._original_skipping_status,
        )

    def init_scene(
        self,
        scene: Scene,
        session_spec: RenderSessionSpec,
        file_writer_settings: _SceneFileWriterSettings,
    ) -> None:
        self.file_writer: Any = self._file_writer_class(file_writer_settings)

    @property
    def time(self) -> float:
        """The scene clock as a float: the exact :attr:`clock`'s value."""
        return float(self._clock)

    @time.setter
    def time(self, value: float | Fraction) -> None:
        self._clock = exact(value)

    @property
    def clock(self) -> Fraction:
        """The scene clock, exactly."""
        return self._clock

    def play(
        self,
        scene: Scene,
        *args: Animation | Mobject | _AnimationBuilder,
        **kwargs: Any,
    ) -> None:
        """Run one play forward, record it, and draw its frames from the record.

        Skipping (a skipped section, ``-n``, a still output) and cached frames
        leave the play's frames undrawn; the play runs all the same, so what
        follows it never depends on whether it was drawn.
        """
        # Reset skip_animations to the original state.
        # Needed when rendering only some animations, and skipping others.
        self.skip_animations = self._original_skipping_status
        self.update_skipping_status()

        scene.compile_animation_data(*args, **kwargs)
        skipped = self.skip_animations
        cache_key: str | None = None
        if skipped:
            logger.debug(f"Skipping animation {self.num_plays}")
        elif config["disable_caching"]:
            logger.info("Caching disabled.")
            cache_key = f"uncached_{self.num_plays:05}"
        else:
            assert scene.animations is not None
            cache_key = get_hash_from_play_call(
                scene,
                self.camera,
                scene.animations,
                scene.mobjects,
                backend="cairo",
                encoder_fingerprint=video_encoder_fingerprint(
                    scene.session_spec.video_encoder,
                ),
                renderer_state=self._grid_phase(scene),
            )

        timeline = scene.timeline
        fps = exact(self.camera.frame_rate)
        if timeline.refusal() is not None:
            # The record can't draw this camera's frames: draw them as the
            # play runs, as CE does.
            self._play_live(scene, skipped, cache_key, fps)
            return
        play = timeline.play(skipped=skipped, cache_key=cache_key)

        frames = frames_in(play.start, play.end, fps)
        if not frames:
            # A play shorter than the gap between two frames shows in none.
            cache_key = None
        draw = cache_key is not None and timeline.draws
        if draw and play.draw_refusal is not None:
            # A mobject the record can't draw joined while the play ran: its
            # frames are drawn as the scene runs, in an execution that knows.
            raise Reexecute(play.index, play.draw_refusal)
        if draw and self.file_writer.is_already_cached(cache_key):
            logger.info(
                f"Animation {self.num_plays} : Using cached data (hash : %(hash_current_animation)s)",
                {"hash_current_animation": cache_key},
            )
            draw = False
        self._register_play(cache_key)
        logger.debug(
            "List of the first few animation hashes of the scene: %(h)s",
            {"h": str(self.animations_hashes[:5])},
        )
        static = None
        if draw:
            static = self._draw_play(scene, play, frames)
        elif play.held and play.static and scene.timeline.draws:
            # CE draws a frozen frame's static picture whatever it skips.
            with Drawing(scene.timeline.ensure_record()) as drawing:
                static = drawing.static(play)
        # As in CE, a frozen frame's static picture stays the renderer's
        # background after it (what update_frame draws over), and no other
        # play's does.
        self.static_image = static if play.held else None
        self.num_plays += 1

    def _play_live(
        self, scene: Scene, skipped: bool, cache_key: str | None, fps: Fraction
    ) -> None:
        """Run one play of a scene whose record can't draw its frames
        (``Record.refusal``), drawing each frame from the live scene while
        the play runs.
        """
        timeline = scene.timeline
        if not frames_in(scene.time, timeline.requested_end(), fps):
            cache_key = None
        draw = cache_key is not None and timeline.draws
        if draw and self.file_writer.is_already_cached(cache_key):
            draw = False
        live = None
        opened = False
        if draw:
            # The play's segment opens at its first frame: a stop condition
            # can end the play before the frame grid reaches it, and then it
            # has no segment, as a play drawn from the record would not.

            def write(pixels: PixelArray, repeat: int) -> None:
                nonlocal opened
                if not opened:
                    opened = True
                    # Registered before the segment opens, which looks its
                    # path up by play index.
                    self._register_play(cache_key)
                    self.file_writer.begin_animation(
                        True, animation_index=self.num_plays
                    )
                self.file_writer.write_frame(pixels, repeat=repeat)

            live = LiveFrames(scene, fps, write)
        play = timeline.play(skipped=skipped, cache_key=cache_key, live=live)
        if opened:
            self.file_writer.end_animation(True)
        else:
            self._register_play(cache_key if not draw else None)
        self.static_image = live._static if live is not None and play.held else None
        self.num_plays += 1

    def _register_play(self, cache_key: str | None) -> None:
        """Give the play about to be drawn its place in the writer's list of
        segments (``None``: no segment) and in the list of play hashes.
        """
        # adding None as a partial movie file will make file_writer ignore the latter.
        self.file_writer.add_partial_movie_file(cache_key)
        self.animations_hashes.append(cache_key)

    def _grid_phase(self, scene: Scene) -> tuple[str, str]:
        """Where the play about to run falls on the frame grid and on the tick:
        a cached play's frames are only reusable at the same phases.
        """
        start = scene.time
        fps = exact(self.camera.frame_rate)
        return (str((start * fps) % 1), str((start * TICK_RATE) % 1))

    def _draw_play(
        self, scene: Scene, play: PlayRecord, frames: list[Fraction]
    ) -> PixelArray | None:
        """Draw the play's frames, one per instant of the frame grid in its
        span, from its recorded samples, and return its static picture.
        """
        record = scene.timeline.ensure_record()
        self.file_writer.begin_animation(True, animation_index=self.num_plays)
        with Drawing(record) as drawing:
            static = drawing.static(play)
            if play.held:
                pixels = drawing.sample(play, play.samples[0], static)
                self.file_writer.write_frame(pixels, repeat=len(frames))
            else:
                for t in frames:
                    pixels = drawing.sample(play, play.capture_at(t), static)
                    self.file_writer.write_frame(pixels)
        self.file_writer.end_animation(True)
        return static

    def update_frame(  # TODO Description in Docstring
        self,
        scene: Scene,
        mobjects: Iterable[Mobject] | None = None,
        include_submobjects: bool = True,
        ignore_skipping: bool = True,
        **kwargs: Any,
    ) -> None:
        """Update the frame.

        Parameters
        ----------
        scene

        mobjects
            list of mobjects

        include_submobjects

        ignore_skipping

        **kwargs
        """
        if self.skip_animations and not ignore_skipping:
            return
        if not mobjects:
            mobjects = list_update(
                scene.mobjects,
                scene.foreground_mobjects,
            )
        if self.static_image is not None:
            self.camera.set_frame_to_background(self.static_image)
        else:
            self.camera.reset()

        kwargs["include_submobjects"] = include_submobjects
        self.camera.capture_mobjects(mobjects, **kwargs)

    def render(
        self,
        scene: Scene,
        time: float,
        moving_mobjects: Iterable[Mobject] | None = None,
    ) -> None:
        self.update_frame(scene, moving_mobjects)
        self.add_frame(self.get_frame())

    def get_frame(self) -> PixelArray:
        """Gets the current frame as NumPy array.

        Returns
        -------
        PixelArray
            NumPy array of pixel values of each pixel in screen.
            The shape of the array is height x width x 3.
        """
        return self.camera.pixel_array.copy()

    def add_frame(self, frame: PixelArray, num_frames: int = 1) -> None:
        """Adds a frame to the video_file_stream

        Parameters
        ----------
        frame
            The frame to add, as a pixel array.
        num_frames
            The number of times to add frame.

        The scene clock is the timeline's: writing frames does not move it.
        """
        if self.skip_animations:
            return
        self.file_writer.write_frame(frame, repeat=num_frames)

    def freeze_current_frame(self, duration: float) -> None:
        """Adds a static frame to the movie for a given duration. The static frame is the current frame.

        Parameters
        ----------
        duration
            [description]
        """
        dt = 1 / self.camera.frame_rate
        self.add_frame(
            self.get_frame(),
            num_frames=int(duration / dt),
        )

    def show_frame(self, scene: Scene) -> None:
        """Opens the current frame in the Default Image Viewer
        of your system.
        """
        self.update_frame(scene, ignore_skipping=True)
        self.camera.get_image().show()

    def save_static_frame_data(
        self,
        scene: Scene,
        static_mobjects: Iterable[Mobject],
    ) -> PixelArray | None:
        """Compute and save the static frame, that will be reused at each frame
        to avoid unnecessarily computing static mobjects.

        Parameters
        ----------
        scene
            The scene played.
        static_mobjects
            Static mobjects of the scene. If None, self.static_image is set to None.

        Returns
        -------
        PixelArray | None
            The static image computed. The return value is None if there are no static mobjects in the scene.
        """
        self.static_image = None
        if not static_mobjects:
            return None
        self.update_frame(scene, mobjects=static_mobjects)
        self.static_image = self.get_frame()
        return self.static_image

    def update_skipping_status(self) -> None:
        """This method is used internally to check if the current
        animation needs to be skipped or not. It also checks if
        the number of animations that were played correspond to
        the number of animations that need to be played, and
        raises an EndSceneEarlyException if they don't correspond.
        """
        # there is always at least one section -> no out of bounds here
        if self.file_writer.sections[-1].skip_animations:
            self.skip_animations = True
        if self.file_writer.output_spec.is_still:
            self.skip_animations = True
        if (
            config.from_animation_number > 0
            and self.num_plays < config.from_animation_number
        ):
            self.skip_animations = True
        if (
            config.upto_animation_number >= 0
            and self.num_plays > config.upto_animation_number
        ):
            self.skip_animations = True
            raise EndSceneEarlyException()

    def scene_finished(self, scene: Scene) -> None:
        output = self.file_writer.output_spec
        if self.num_plays and (output.is_video or output.is_image_sequence):
            self.file_writer.finish()
        elif not self.num_plays:
            self._draw_final(scene)

        # Automatically selected video output falls back to a last-frame PNG
        # when a scene has no play calls.
        if output.is_still or (not self.num_plays and output.fallback_to_still):
            if self.num_plays:
                self._draw_final(scene)
            self.file_writer.save_image(self.get_frame())

    def _draw_final(self, scene: Scene) -> None:
        """Draw the scene as its script left it into the camera: from the
        record's final capture once the run has ended.
        """
        self.static_image = None
        record = scene.timeline.record
        if record is None or record.final is None or record.refusal is not None:
            # Nothing recorded, or a camera the record can't draw: the live
            # scene is the script's last state.
            self.update_frame(scene)
            return
        with Drawing(record) as drawing:
            drawing.final()
