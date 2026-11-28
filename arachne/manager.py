"""Orchestration for rendering a scene."""

from __future__ import annotations

import datetime
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

import srt

from . import logger
from .mobject.construction_caches import (
    construction_caches,
    restore_construction_caches,
)
from .scene.section import DefaultSectionType
from .timeline.clock import exact
from .timeline.execution import Reexecute, play_keys
from .utils.exceptions import EndSceneEarlyException
from .utils.file_ops import open_media_file

if TYPE_CHECKING:
    from ._config.output import OutputSpec
    from ._config.render_session import RenderSessionSpec
    from .animation.animation import Animation
    from .camera.camera import Camera
    from .mobject.mobject import Mobject, _AnimationBuilder
    from .renderer.cairo_renderer import CairoRenderer
    from .scene.scene import Scene
    from .scene.scene_file_writer import SceneFileWriter

__all__ = ["Manager"]

SceneT = TypeVar("SceneT", bound="Scene")


class Manager(Generic[SceneT]):
    """Coordinate the render lifecycle for a single scene.

    A manager is attached to exactly one :class:`~arachne.scene.scene.Scene`. It
    coordinates the scene lifecycle, delegates animation playback to the current
    renderer, and routes section, subcaption, and audio operations to the current
    file writer. Calling :meth:`~arachne.scene.scene.Scene.render` creates a manager
    lazily when one has not already been attached.

    Parameters
    ----------
    scene
        The scene coordinated by this manager. A scene that already has a manager
        cannot be attached to another one.

    Attributes
    ----------
    scene
        The managed scene.

    Notes
    -----
    This class is the coordination boundary for an incremental render-flow
    refactor. Renderer, camera, clock, and output ownership still remain on the
    scene and renderer for compatibility. The :attr:`renderer`, :attr:`camera`,
    :attr:`file_writer`, :attr:`time`, :attr:`num_plays`, and
    :attr:`skip_animations` properties are forwarding views for now.

    Examples
    --------
    A scene normally creates its manager when :meth:`~arachne.scene.scene.Scene.render`
    is called. It can also be attached explicitly::

        scene = Scene()
        manager = Manager(scene)
        manager.render()
    """

    def __init__(self, scene: SceneT) -> None:
        if scene.manager is not None:
            raise ValueError("A manager is already attached to this scene.")
        self.scene = scene
        scene.manager = self
        # The source line of the play running now, or None between plays.
        self._playing: str | None = None

    @property
    def renderer(self) -> CairoRenderer:
        """Return the scene's current renderer."""
        return self.scene.renderer

    @property
    def camera(self) -> Camera:
        """Return the current renderer's camera."""
        return self.renderer.camera

    @property
    def file_writer(self) -> SceneFileWriter:
        """Return the current renderer's file writer."""
        return cast("SceneFileWriter", self.renderer.file_writer)

    @property
    def output_spec(self) -> OutputSpec:
        """Return the immutable output intent captured for this session."""
        return self.session_spec.output

    @property
    def session_spec(self) -> RenderSessionSpec:
        """Return the immutable artifact, presentation, and execution intent."""
        return self.scene.session_spec

    @property
    def time(self) -> float:
        """Return the current renderer time."""
        return self.renderer.time

    @time.setter
    def time(self, value: float) -> None:
        self.renderer.time = value

    @property
    def num_plays(self) -> int:
        """Return the current renderer's play count."""
        return self.renderer.num_plays

    @num_plays.setter
    def num_plays(self, value: int) -> None:
        self.renderer.num_plays = value

    @property
    def skip_animations(self) -> bool:
        """Return the current renderer's animation skip state."""
        return self.renderer.skip_animations

    @skip_animations.setter
    def skip_animations(self, value: bool) -> None:
        self.renderer.skip_animations = value

    def render(self, preview: bool = False) -> None:
        """Run the complete render lifecycle for the managed scene.

        The lifecycle invokes :meth:`setup`, :meth:`construct`,
        :meth:`tear_down`, and :meth:`post_construct`, in that order. Reaching a
        configured animation boundary ends construction normally. Exceptions
        raised while constructing the scene abort active encoding jobs before
        they are propagated.

        A play that was to be drawn from the scene's record and that a mobject
        the record can't draw joined mid-play (:mod:`.timeline.execution`)
        has the scene executed again from the start: the same scene object,
        initialised anew with the arguments it was made with, draws that play
        and every later one as it runs. Each such refusal costs at most one
        execution. The interrupted execution is not torn down, as CE tears
        down no run that raised: :meth:`tear_down` runs once, for the
        execution that completes.

        A scene drawn on the core that the core cannot draw
        (``render.unable``: never drawn wrong, and never one frame by another
        painter among the core's) is executed again the same way, from the
        start, drawn on Cairo: the core may refuse a scene, but never makes
        an unrenderable one of a scene Cairo renders. A ``render.fallback``
        finding carries the refusal; the draw path asked for is the
        option's again once the render ends.

        Parameters
        ----------
        preview
            Whether the rendered media should be opened after rendering.
        """
        presentation = self.session_spec.presentation
        open_after_render = preview or presentation.open_after_render
        if open_after_render and not self.output_spec.enabled:
            raise ValueError("Previewing after render requires a media artifact.")
        known: dict[int, str] = {}
        earlier: list[tuple[int, tuple[bytes, ...]]] = []
        # The construction caches as a scene rendered earlier in this process
        # left them: an execution again finds them as the first execution did.
        caches = construction_caches()
        # While the scene is drawn on Cairo after the core refused it: the
        # draw path asked for, the option's again once the render ends.
        asked: str | None = None
        try:
            while True:
                try:
                    self._executions(known, earlier, caches)
                    self.post_construct()
                    break
                except Exception as unable:
                    if asked is not None or not self._refused_by_the_core(unable):
                        raise
                    asked = self._draw_on_cairo(unable)
                    # As for a play's refusal: the interrupted execution's
                    # encoders were aborted and its run abandoned.
                    record = self.scene._arachne_timeline.ensure_record()
                    earlier.append((record.execution.number, play_keys(record)))
                    restore_construction_caches(caches)
                    self._renew_scene()
                    self.scene._arachne_timeline.reexecuting(known, earlier)
        finally:
            if asked is not None:
                from . import config

                config.draw_path = asked

        if open_after_render or presentation.show_in_file_browser:
            open_media_file(
                self.file_writer,
                preview=open_after_render,
                show_in_file_browser=presentation.show_in_file_browser,
            )

    def _executions(
        self,
        known: dict[int, str],
        earlier: list[tuple[int, tuple[bytes, ...]]],
        caches: Any,
    ) -> None:
        """Execute the scene until an execution completes: again from the
        start for each play it meant to draw from its record that a mobject
        the record can't draw joined mid-play (``known``, by play).
        """
        while True:
            try:
                self._execute()
                break
            except Reexecute as again:
                record = self.scene._arachne_timeline.ensure_record()
                if known and again.play >= min(known):
                    # Every play from the earliest known refusal on is drawn
                    # as it runs and never raises this.
                    raise RuntimeError(
                        f"play {again.play} was to be drawn as it runs"
                    ) from again
                # The interrupted execution's encoders were aborted and its
                # run abandoned on the way out; the scene's tear_down is for
                # an execution that completes.
                known[again.play] = again.reason
                earlier.append((record.execution.number, play_keys(record)))
                logger.info(
                    "Play %(play)s: %(reason)s. Executing %(scene)s again (execution"
                    " %(n)s), drawing that play as it runs.",
                    {
                        "play": again.play,
                        "reason": again.reason,
                        "scene": type(self.scene).__name__,
                        "n": len(earlier) + 1,
                    },
                )
                restore_construction_caches(caches)
                self._renew_scene()
                self.scene._arachne_timeline.reexecuting(known, earlier)

    def _refused_by_the_core(self, error: Exception) -> bool:
        """Whether ``error`` is the core's refusal (``render.unable``) of a
        scene whose frames are asked for on the core (``core``, or ``auto``
        for its camera). Read only once raised: the drawing imported it by
        then, and the engine's code is read before (``timeline.frozen``).
        """
        from . import config
        from .frame_ir.extract import Unable
        from .frame_ir.painter import scene_path

        if not isinstance(error, Unable):
            return False
        return scene_path(self.scene.renderer.camera, config.draw_path) == "core"

    def _draw_on_cairo(self, unable: Exception) -> str:
        """The scene, refused by the core (``unable``), is drawn on Cairo from
        now on, with a ``render.fallback`` finding: return the draw path that
        was asked for.
        """
        from . import config
        from .frame_ir.lint import raise_finding

        name = type(self.scene).__name__
        raise_finding(
            "render.fallback",
            name,
            f"{name} is drawn on Cairo for the whole scene, executed again from"
            f" its start: the core could not draw it ({unable})",
        )
        asked = config.draw_path
        config.draw_path = "cairo"
        return asked

    def _execute(self) -> None:
        """Execute the scene's script once, recording it."""
        timeline = self.scene._arachne_timeline
        timeline.begin_run()
        try:
            self.setup()
            try:
                self.construct()
            except EndSceneEarlyException:
                # Reaching the configured animation boundary ends the scene normally.
                pass
            except BaseException:
                # A mid-play exception leaves an unsealed encode job whose
                # non-daemon worker would hang the process at exit.
                self.file_writer.abort_encode_jobs()
                raise
            self.tear_down()
        except BaseException:
            timeline.abandon_run()
            raise
        timeline.end_run()

    def _renew_scene(self) -> None:
        """Initialise the scene anew, as it was made, for another execution:
        the same object, so whoever holds it holds the execution that ends.
        """
        scene = self.scene
        arguments = scene._arguments
        # A renderer the scene was given (rather than made) is used again.
        scene.renderer.renew()
        vars(scene).clear()
        scene._arguments = arguments
        args, kwargs = arguments
        type(scene).__init__(scene, *args, **kwargs)
        scene.manager = self

    def setup(self) -> None:
        """Run the managed scene's :meth:`~arachne.scene.scene.Scene.setup` hook."""
        self.scene.setup()

    def construct(self) -> None:
        """Run the managed scene's :meth:`~arachne.scene.scene.Scene.construct` hook."""
        self.scene.construct()

    def post_construct(self) -> None:
        """Finalize output after scene construction and tear-down.

        This validates empty video output, asks the renderer to finish the scene,
        and logs the number of played animations. It intentionally runs after
        :meth:`tear_down` to preserve the established render lifecycle.
        """
        output = self.output_spec
        empty_video_output = self.num_plays == 0 and output.is_video
        if empty_video_output and not output.fallback_to_still:
            raise RuntimeError(
                f"{self.scene} has no play calls, so the explicitly requested "
                f"{output.format.value.upper()} output cannot be produced. "
                "Use --format=png to save its last frame.",
            )

        # We have to reset these settings in case of multiple renders.
        self.renderer.scene_finished(self.scene)

        if (
            empty_video_output
            and output.fallback_to_still
            and getattr(self.file_writer, "final_file_path", None) is not None
        ):
            logger.warning(
                f"{self.scene} has no play calls. Automatic video output has "
                "been saved as a PNG instead.",
            )

        # Show info only if animations are rendered or to get image.
        if self.num_plays or output.enabled:
            logger.info(
                f"Rendered {str(self.scene)}\nPlayed {self.num_plays} animations",
            )

    def tear_down(self) -> None:
        """Run the managed scene's :meth:`~arachne.scene.scene.Scene.tear_down` hook."""
        self.scene.tear_down()

    def play(
        self,
        *args: Animation | Mobject | _AnimationBuilder,
        subcaption: str | None = None,
        subcaption_duration: float | Fraction | None = None,
        subcaption_offset: float = 0,
        **kwargs: Any,
    ) -> None:
        """Coordinate an animation request and its optional subcaption.

        Parameters
        ----------
        args
            Animations, mobjects, or animation builders passed by
            :meth:`~arachne.scene.scene.Scene.play`.
        subcaption
            Content to add to the external subcaption file for this play call.
        subcaption_duration
            Duration of the subcaption. When omitted, the elapsed animation time
            is used.
        subcaption_offset
            Offset in seconds from the beginning of the play call.
        kwargs
            Additional animation arguments forwarded to the renderer.

        Raises
        ------
        RuntimeError
            When called while another play or wait runs, as from an updater.
        """
        # The re-entrancy guard follows ManimCommunity/manim PR #5006's.
        timeline = self.scene._arachne_timeline
        if self._playing is not None:
            line = timeline.location()
            raise RuntimeError(
                "play() and wait() can't be called while another play or wait "
                "runs, as from an updater"
                + (f", at {line}" if line is not None else "")
                + f" (the play running is the one at {self._playing})."
                " Call it from construct() once that play has ended."
            )
        start_time = self.scene.time
        line = timeline.location()
        self._playing = str(line) if line is not None else "an unknown line"
        try:
            self.renderer.play(self.scene, *args, **kwargs)
        finally:
            self._playing = None
        run_time = self.scene.time - start_time

        if subcaption:
            if subcaption_duration is None:
                subcaption_duration = run_time
            # The start of the subcaption needs to be offset by the run time
            # because it is added after the animation has already played.
            # Route through Scene's public API to preserve its customization hook.
            self.scene.add_subcaption(
                content=subcaption,
                duration=subcaption_duration,
                offset=exact(subcaption_offset) - run_time,
            )

    def next_section(
        self,
        name: str = "unnamed",
        section_type: str = DefaultSectionType.NORMAL,
        skip_animations: bool = False,
    ) -> None:
        """Create a new output section.

        Parameters
        ----------
        name
            The section name.
        section_type
            The section type stored in the section manifest.
        skip_animations
            Whether animation output in this section should be skipped.
        """
        self.file_writer.next_section(name, section_type, skip_animations)
        self.scene._arachne_timeline.mark(
            "section",
            {"name": name, "type": str(section_type), "skip": skip_animations},
        )

    def add_subcaption(
        self, content: str, duration: float | Fraction = 1, offset: float | Fraction = 0
    ) -> None:
        """Add a subcaption at the current scene time.

        Parameters
        ----------
        content
            The subcaption text.
        duration
            The duration in seconds for which the subcaption is displayed.
        offset
            The offset in seconds from the current scene time.
        """
        subtitle = srt.Subtitle(
            index=len(self.file_writer.subcaptions),
            content=content,
            start=datetime.timedelta(seconds=float(self.time + offset)),
            end=datetime.timedelta(seconds=float(self.time + offset + duration)),
        )
        self.file_writer.subcaptions.append(subtitle)
        self.scene._arachne_timeline.mark(
            "subcaption",
            {"content": content, "duration": duration},
            t=max(self.scene.time + exact(offset), Fraction(0)),
        )

    def add_sound(
        self,
        sound_file: str,
        time_offset: float = 0,
        gain: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Add sound to the output at the current scene time.

        No sound is added while animations are being skipped.

        Parameters
        ----------
        sound_file
            The path to the sound file.
        time_offset
            The offset in seconds from the current scene time.
        gain
            The gain adjustment applied to the sound.
        kwargs
            Additional arguments forwarded to
            :meth:`~arachne.scene.scene_file_writer.SceneFileWriter.add_sound`.
        """
        if self.skip_animations:
            return
        self.file_writer.add_sound(sound_file, self.time + time_offset, gain, **kwargs)
        self.scene._arachne_timeline.mark(
            "sound",
            {"file": str(sound_file), "gain": gain},
            t=max(self.scene.time + exact(time_offset), Fraction(0)),
        )
