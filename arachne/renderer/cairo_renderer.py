from __future__ import annotations

import functools
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Any

from arachne.utils.hashing import get_hash_from_play_call

from .. import config, logger
from .._config.video_encoder import video_encoder_fingerprint, with_thread_share
from ..camera.camera import Camera
from ..mobject.mobject import Mobject, _AnimationBuilder
from ..scene.scene_file_writer import SceneFileWriter, uncached_key
from ..timeline.clock import TICK_RATE, exact, frames_in
from ..timeline.draw import Drawing, DrawReuse, LiveFrames
from ..timeline.execution import UNREPEATABLE, Reexecute
from ..utils.exceptions import EndSceneEarlyException
from ..utils.iterables import list_update
from . import workers
from .protocol import RendererCapabilities

if TYPE_CHECKING:
    from arachne._config.render_session import RenderSessionSpec
    from arachne.animation.animation import Animation
    from arachne.scene.scene import Scene
    from arachne.scene.scene_file_writer import _SceneFileWriterSettings
    from arachne.timeline.record import PlayRecord

    from ..typing import PixelArray

__all__ = ["CairoRenderer"]


@dataclass(frozen=True)
class _Deferred:
    """A play of a render whose frames wait to be drawn by its workers:
    ``number``, its index in the render; ``play`` and ``frames``, the play
    and its instants of the frame grid, ``None`` and none when it draws
    nothing; ``static``, the static picture the renderer kept after it, for
    a play that draws nothing.
    """

    number: int
    play: PlayRecord | None
    frames: list[Fraction]
    static: PixelArray | None


def _stop_progress_monitor() -> None:
    """Stop the progress bars' monitor thread while no bar is open: tqdm
    starts it with the first bar of the process and keeps it to the end,
    and starts it again with the next bar.
    """
    from tqdm import tqdm

    with tqdm.get_lock():
        monitor = tqdm.monitor
        if monitor is None or tqdm._instances:
            return
        tqdm.monitor = None
    # Outside the lock, which the monitor takes as it wakes.
    monitor.exit()


class CairoRenderer:
    """A renderer using Cairo.

    Attributes
    ----------
    num_plays : int
        Number of play() functions in the scene.

    time : float
        Time elapsed since initialisation of scene.

    draws : DrawReuse
        The frame drawn last from the record, which the next frame drawn
        from the same recorded state reuses (``draw_reuse``): kept for one
        execution of the scene.

    frame_runs : dict[int, tuple[int, ...]]
        Per play drawn from the record (by index), its frames as runs of
        equal frames, in order: a frozen frame's are one run, and a frame
        that reused the frame before it extends that frame's run. Every
        frame is written all the same.

    drawn_by : tuple[workers.WorkerReport, ...]
        With ``render_workers`` above 1, the worker processes that drew and
        encoded the render's frames, each a contiguous range of its plays,
        in order (:mod:`.workers`); none when the render's own process drew
        them.

    worker_fallback : str | None
        Why a render that asked for workers drew its frames in its own
        process, as one would: a scene with a play drawn as it runs, cached
        or sectioned output, a GIF, no video, one play to draw, a camera
        whose drawing runs code no record holds, or a thread alive beside
        the render's. ``None`` when it asked for none, or they drew it.
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
        #: Whether the capture being made is a play's static picture.
        self._capturing_static = False
        self._clock = Fraction(0)
        self.static_image: PixelArray | None = None
        self.draws = DrawReuse(1)
        self.frame_runs: dict[int, tuple[int, ...]] = {}
        #: The plays run so far, while their frames wait for the workers;
        #: ``None`` when each play is drawn as the render reaches it.
        self._deferred: list[_Deferred] | None = None
        self.drawn_by: tuple[workers.WorkerReport, ...] = ()
        self.worker_fallback: str | None = None

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
        self._deferred = None
        if config.render_workers > 1:
            reason = self._no_workers(self.file_writer.output_spec)
            if reason is None:
                self._deferred = []
            else:
                self._fall_back(reason)

    def _no_workers(self, output: Any) -> str | None:
        """Why a render of ``output`` draws its frames in its own process
        whatever its scene, or ``None``.
        """
        if not output.is_video:
            return "no video is written"
        if output.is_gif:
            return "a GIF is written from one stream of every frame"
        if output.save_sections:
            return "sections are saved, each joined from its plays' segments"
        if not config.disable_caching:
            return "caching is on: each play's segment is kept for later renders"
        return None

    def _fall_back(self, reason: str) -> None:
        """The render draws its frames in its own process, for ``reason``."""
        self.worker_fallback = reason
        logger.info(
            "Drawn in this process, not by %(n)s workers: %(reason)s.",
            {"n": config.render_workers, "reason": reason},
        )

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

        # The scene as the script called the play, before it adds what the
        # animations animate.
        scene._arachne_timeline.calling()
        scene.compile_animation_data(*args, **kwargs)
        skipped = self.skip_animations
        cache_key: str | None = None
        if skipped:
            logger.debug(f"Skipping animation {self.num_plays}")
        elif config["disable_caching"]:
            logger.info("Caching disabled.")
            cache_key = uncached_key(self.num_plays)
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
                renderer_state=(*self._grid_phase(scene), *self._painter_state()),
            )

        timeline = scene._arachne_timeline
        fps = exact(self.camera.frame_rate)
        refusal = timeline.refusal()
        if refusal is not None:
            # The record can't draw this camera's frames: draw them as the
            # play runs, as CE does.
            self._draw_here(
                scene, f"play {self.num_plays} is drawn as it runs ({refusal})"
            )
            self._play_live(scene, skipped, cache_key, fps)
            return
        manager = scene.manager
        if timeline.draws and (manager is None or not manager.reexecutes):
            # Played outside a render: a refusal found midway could not be
            # drawn as it runs in another execution.
            self._draw_here(
                scene, f"play {self.num_plays} is drawn as it runs ({UNREPEATABLE})"
            )
            self._play_live(scene, skipped, cache_key, fps, UNREPEATABLE)
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
        deferred = draw and self._deferred is not None
        if deferred:
            # Drawn by the render's workers once the scene has run.
            pass
        elif draw:
            static = self._draw_play(scene, play, frames)
        elif play.held and play.static and scene._arachne_timeline.draws:
            # CE draws a frozen frame's static picture whatever it skips.
            with Drawing(scene._arachne_timeline.ensure_record()) as drawing:
                static = drawing.static(play)
        # As in CE, a frozen frame's static picture stays the renderer's
        # background after it (what update_frame draws over), and no other
        # play's does; with the static layer off, none does.
        self.static_image = static if play.held and config.static_layer else None
        if self._deferred is not None:
            self._deferred.append(
                _Deferred(
                    self.num_plays,
                    play if deferred else None,
                    frames if deferred else [],
                    self.static_image,
                )
            )
        self.num_plays += 1

    def _play_live(
        self,
        scene: Scene,
        skipped: bool,
        cache_key: str | None,
        fps: Fraction,
        reason: str | None = None,
    ) -> None:
        """Run one play of a scene whose record can't draw its frames
        (``Record.refusal``), or that is drawn as it runs for ``reason``,
        drawing each frame from the live scene while the play runs.
        """
        timeline = scene._arachne_timeline
        draw = cache_key is not None and timeline.draws
        if draw and self.file_writer.is_already_cached(cache_key):
            draw = False
        live = None
        opened = False
        if draw:
            # The play's segment opens at its first frame: a stop condition
            # can end the play before the frame grid reaches it, and then it
            # has no segment, like a frameless play drawn from the record.
            # Whether it has frames is known only once it has run: its
            # play_internal decides its span.

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
        play = timeline.play(
            skipped=skipped, cache_key=cache_key, live=live, live_reason=reason
        )
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

    def _painter_state(self) -> tuple[str, ...]:
        """The painter a play's frames are asked of, where it is not Cairo: a
        segment drawn on the core is not Cairo's, and a render on one path
        never reuses the other's (a scene the core refused is drawn on Cairo
        again from its start, past the core's segments of the plays before).
        Cairo's two paths draw the same pixels, so share their segments.
        """
        from ..frame_ir.painter import scene_path

        return ("core",) if scene_path(self.camera, config.draw_path) == "core" else ()

    def _draw_play(
        self,
        scene: Scene,
        play: PlayRecord,
        frames: list[Fraction],
        number: int | None = None,
    ) -> PixelArray | None:
        """Draw the play's frames, one per instant of the frame grid in its
        span, from its recorded samples, and return its static picture.
        ``number`` is the play's index in the render, by default the play
        being run.
        """
        record = scene._arachne_timeline.ensure_record()
        if number is None:
            number = self.num_plays
        self.file_writer.begin_animation(True, animation_index=number)
        reuse = self.draws if config.draw_reuse else None
        runs: list[int] = []
        with Drawing(record, reuse=reuse) as drawing:
            static = drawing.static(play)
            if play.held:
                pixels = drawing.sample(play, play.samples[0], static)
                self.file_writer.write_frame(pixels, repeat=len(frames))
                runs.append(len(frames))
            else:
                for t in frames:
                    pixels = drawing.sample(play, play.capture_at(t), static)
                    # One write per frame, reused or drawn.
                    self.file_writer.write_frame(pixels)
                    if runs and drawing.reused:
                        runs[-1] += 1
                    else:
                        runs.append(1)
        self.file_writer.end_animation(True)
        self.frame_runs[play.index] = tuple(runs)
        return static

    # The render's workers

    def _draw_here(self, scene: Scene, reason: str) -> None:
        """Draw, in this process, the frames of the plays run so far that wait
        for the render's workers, as each would have been drawn when it ran,
        and draw every later play as the render reaches it: for ``reason``.
        """
        deferred, self._deferred = self._deferred, None
        if deferred is None:
            return
        self._fall_back(reason)
        self._draw_in_order(scene, deferred)

    def _draw_in_order(self, scene: Scene, deferred: Sequence[_Deferred]) -> None:
        """Draw the frames of ``deferred`` in this process, in order,
        leaving the renderer as drawing each as it ran would have.
        """
        for each in deferred:
            if each.play is None:
                self.static_image = each.static
                continue
            static = self._draw_play(scene, each.play, each.frames, each.number)
            held = each.play.held and config.static_layer
            self.static_image = static if held else None

    def _draw_deferred(self, scene: Scene) -> None:
        """Once the scene has run, draw the frames of its plays: cut into
        contiguous ranges balanced by what each play costs to draw and
        encode, each range drawn and encoded into a stream of its own by a
        worker forked for it, the streams joined in play order as the
        render finishes. A render with one play to draw, a camera whose
        drawing runs code no record holds, or a thread alive beside this
        one draws them here instead.
        """
        deferred, self._deferred = self._deferred, None
        assert deferred is not None
        drawn = [each for each in deferred if each.play is not None]
        if not drawn:
            return
        gaps = self.file_writer.video_encoder.gaps
        costs = [
            1 if gaps and each.play.held else len(each.frames)  # type: ignore[union-attr]
            for each in drawn
        ]
        spans = workers.ranges(costs, config.render_workers)
        reason = self._refused_by_workers(scene, len(spans))
        if reason is None:
            # The render's encoders are all done, and the progress bars'
            # monitor stopped: no thread of its own is left, and nothing it
            # encodes is half written.
            self.file_writer.join_all_encode_jobs()
            _stop_progress_monitor()
            alive = workers.threads_alive()
            if alive:
                reason = (
                    "a thread other than the render's is alive, beside which"
                    f" no worker is forked: {', '.join(alive)}"
                )
        if reason is not None:
            self._fall_back(reason)
            self._draw_in_order(scene, deferred)
            return
        writer = self.file_writer
        writer.cut_streams_before({drawn[start].number for start, _ in spans})
        tasks = [
            functools.partial(self._draw_range, scene, drawn[start:stop], len(spans))
            for start, stop in spans
        ]
        try:
            results = workers.run(tasks)
        except workers.WorkerFailed as failed:
            writer.discard_streams()
            from ..frame_ir.extract import Unable

            if failed.kind == f"{Unable.__module__}.{Unable.__qualname__}":
                # The core refused a frame: the render draws the scene on
                # Cairo instead, as it would have had it drawn it here.
                raise Unable(failed.message) from failed
            raise
        except BaseException:
            writer.discard_streams()
            raise
        self._took(drawn, spans, results)

    def _refused_by_workers(self, scene: Scene, ranges: int) -> str | None:
        """Why the scene's frames are drawn in this process once it has run,
        cut into ``ranges`` ranges, or ``None``: one range to draw, or a
        camera whose drawing runs code at each frame that can read what no
        record holds, which a process drawing after the scene ran would
        read as the scene left it.
        """
        from ..frame_ir.painter import user_camera
        from ..timeline.draw import _paints_its_list

        if ranges < 2:
            return "one play to draw"
        camera = self.camera
        if not _paints_its_list(camera):
            return (
                f"its camera, a {type(camera).__name__}, is not one of the"
                " engine's flat, moving and 3D cameras, whose frames are drawn"
                " from the record alone"
            )
        own = user_camera(camera)
        if own is not None:
            return f"the scene's code draws through its camera ({own[1].rstrip(',')})"
        replaced = scene._arachne_timeline.ensure_record().camera_replaced
        if replaced:
            names = ", ".join(each.qualname for each in replaced)
            return f"the scene's code replaced its camera's {names}"
        return None

    def _draw_range(
        self, scene: Scene, deferred: Sequence[_Deferred], count: int
    ) -> dict[str, Any]:
        """In one of ``count`` workers: draw and encode the plays of
        ``deferred``, a contiguous range of the render's, into the stream
        planned for them, as the render's own process would have drawn
        them; what it drew, as a JSON form.

        The worker begins where the render left off, the scene run: each
        play's frames, and its static picture, are drawn from the record
        alone. Paths map around the centre the render's first path draw
        froze, as a frame drawn after the scene ran always does
        (:mod:`.timeline.mapping`): a worker whose range begins after the
        camera moved draws as a process that drew every frame before it.
        """
        from ..frame_ir.lint import raised
        from ..server import registry
        from ..timeline.mapping import freeze_mapping_centre

        writer = self.file_writer
        writer.video_encoder = with_thread_share(writer.video_encoder, count)
        freeze_mapping_centre(scene._arachne_timeline.ensure_record())
        found = len(raised)
        writer.take_stream(deferred[0].number)
        frames = 0
        for each in deferred:
            assert each.play is not None
            self._draw_play(scene, each.play, each.frames, each.number)
            frames += len(each.frames)
        writer.join_all_encode_jobs()
        return {
            "frame_runs": [
                [each.play.index, list(self.frame_runs[each.play.index])]  # type: ignore[union-attr]
                for each in deferred
            ],
            "frames": frames,
            "peak_mb": registry._rss_mb(),
            "findings": [[f.code, f.subject, f.message] for f in raised[found:]],
        }

    def _took(
        self,
        drawn: Sequence[_Deferred],
        spans: Sequence[tuple[int, int]],
        results: Sequence[dict[str, Any]],
    ) -> None:
        """Take what the workers drew (``results``, one per range of
        ``spans`` over ``drawn``) as the render's own.
        """
        from ..frame_ir.lint import Finding, raised

        reports = []
        for (start, stop), result in zip(spans, results, strict=True):
            for index, runs in result["frame_runs"]:
                self.frame_runs[index] = tuple(runs)
            for code, subject, message in result["findings"]:
                # Logged where the worker raised it.
                finding = Finding(code, subject, message)
                if finding not in raised:
                    raised.append(finding)
            reports.append(
                workers.WorkerReport(
                    first=drawn[start].number,
                    last=drawn[stop - 1].number,
                    frames=result["frames"],
                    peak_mb=result["peak_mb"],
                )
            )
        self.drawn_by = tuple(reports)
        for index, report in enumerate(reports):
            logger.info(
                "Worker %(index)s drew plays %(first)s to %(last)s: %(frames)s"
                " frames, at most %(peak).0f MB.",
                {
                    "index": index,
                    "first": report.first,
                    "last": report.last,
                    "frames": report.frames,
                    "peak": report.peak_mb,
                },
            )
        logger.info(
            "%(n)s workers drew the render, at most %(peak).0f MB in all.",
            {"n": len(reports), "peak": sum(r.peak_mb for r in reports)},
        )

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
        kwargs["include_submobjects"] = include_submobjects
        from ..frame_ir.live import live

        # A play's static picture is told apart from its frames: the core
        # watches the first of each (``LivePainter._sample``). The picture the
        # frame starts on is set within, so the frame is described over it.
        play = ("static", self.num_plays) if self._capturing_static else self.num_plays
        with live(self.camera, self.clock, play=play):
            if self.static_image is not None:
                self.camera.set_frame_to_background(self.static_image)
            else:
                self.camera.reset()
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
        self._capturing_static = True
        try:
            self.update_frame(scene, mobjects=static_mobjects)
        finally:
            self._capturing_static = False
        self.static_image = self.get_frame()
        from ..frame_ir.live import keep_static_picture

        keep_static_picture(self.camera, self.static_image)
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
        if self._deferred is not None:
            self._draw_deferred(scene)
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
        record = scene._arachne_timeline.record
        if record is None or record.final is None or record.refusal is not None:
            # Nothing recorded, or a camera the record can't draw: the live
            # scene is the script's last state.
            self.update_frame(scene)
            return
        with Drawing(record) as drawing:
            drawing.final()
