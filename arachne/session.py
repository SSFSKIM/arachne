"""A scene opened in process: executed once, recorded, then asked about.

:meth:`Session.open` imports a scene file, executes one of its scenes
forward with nothing written, and keeps the scene's :class:`Record` with
every instant the run evaluated. :meth:`Session.state_at` answers from the
record, and :meth:`Session.frame` draws one frame from it; neither runs the
scene's code again. :meth:`Session.answers` answers a batch of instants, and
what the record can't answer there a forward run does: the scene executed
once more, for the whole batch (:mod:`.timeline.fallback`).
:meth:`Session.compare` compares the session's execution with another of
the same source, perhaps from another process, and keeps where they differ
in :attr:`Session.divergences`.
"""

from __future__ import annotations

__all__ = ["Session"]

import importlib.util
import itertools
import os
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from ._config import config as global_config
from ._config import tempconfig
from .constants import QUALITIES
from .timeline.clock import exact
from .timeline.draw import Drawing
from .timeline.execution import Divergence, ExecutionSummary, compare, summarize
from .timeline.fallback import Answer, ForwardRun, answer_instants
from .timeline.origins import ORIGINS
from .timeline.recorder import Recorder
from .timeline.routes import _draw, _resolve, _Resolved

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from types import ModuleType

    from PIL import Image

    from .scene.scene import Scene
    from .timeline.record import PlayRecord, Record
    from .timeline.state import State
    from .typing import PixelArray, StrPath

_MODULE_NUMBERS = itertools.count()


class Session:
    """One scene of one file, executed and recorded.

    Open it with :meth:`open`; it holds the executed scene, whose camera
    draws its frames.
    """

    def __init__(self, scene: Scene, record: Record, module: ModuleType) -> None:
        self.scene = scene
        self.record = record
        self.module = module
        self._quality: str | None = None
        self._static: tuple[int, str | None, PixelArray | None] | None = None
        camera = scene.renderer.camera
        self._executed_shape = (camera.pixel_height, camera.pixel_width)
        #: Where the session's execution was found to differ from another of
        #: its source, in the order found (``replay.diverged``).
        self.divergences: list[Divergence] = []
        #: The record last summarised, and its summary.
        self._summary: tuple[Record, ExecutionSummary] | None = None
        #: What :meth:`open` executed, which a forward run executes again.
        self._source: Source | None = None
        #: The forward runs that answered what the record could not, in order.
        self.forward_runs: list[ForwardRun] = []
        #: Whether a forward run found the scene's source changed since the
        #: record was made: the record is of a program the file no longer is.
        self.stale = False

    @classmethod
    def open(
        cls,
        file: StrPath,
        scene: str,
        *,
        grid: Fraction | float | None | Literal["render"] = "render",
        config: dict[str, Any] | None = None,
        stepping: bool | Literal["brute_force"] = True,
        on_step: Callable[[Scene, Fraction], object] | None = None,
        on_tick: Callable[[Scene, Fraction, bool], object] | None = None,
        on_start: Callable[[Scene, Fraction], object] | None = None,
        instants: Iterable[Fraction | float] = (),
    ) -> Session:
        """Execute the scene class named ``scene`` from ``file`` and record it.

        ``grid`` is the frame rate whose instants the record keeps: by
        default the frame rate of the render the configuration describes, so
        the session's answers equal that render's frames (a ticked play's
        dt-free callbacks run at the instants a frame reads,
        :mod:`.timeline.ticking`); ``None`` keeps and fully evaluates every
        instant the run reaches. ``stepping`` is ``Recorder``'s switch:
        ``False`` runs every tick of a ticked play in full, and
        ``"brute_force"`` is the verification mode of
        :mod:`.timeline.reference`; ``on_step``,
        ``on_tick`` and ``on_start`` are ``Recorder``'s diagnostic hooks.
        ``instants`` are extra instants the trusted and state-function plays
        are evaluated and recorded at, between frames (``Recorder``'s): the
        forward evaluation a restore there is compared with. ``config``
        overrides the configuration the scene is executed under; nothing is
        written either way.
        """
        path = Path(file).resolve()
        settings = {
            **(config or {}),
            "dry_run": True,
            "disable_caching": True,
        }

        def recorder(instance: Scene) -> Recorder:
            return Recorder(
                instance,
                grid=global_config.frame_rate if grid == "render" else grid,
                draws=False,
                stepping=stepping,
                on_step=on_step,
                on_tick=on_tick,
                on_start=on_start,
                instants=instants,
            )

        instance, module = execute(path, scene, settings, recorder)
        record = instance.record
        assert record is not None
        session = cls(instance, record, module)
        session._source = Source(path, scene, settings, stepping)
        return session

    def state_at(self, t: Fraction | float) -> State:
        """The scene at ``t``, from the record: restored there when a
        state-function play holds ``t`` and the record did not sample it.
        """
        return self.record.state_at(t)

    def frame(self, t: Fraction | float, quality: str | None = None) -> Image.Image:
        """The frame at ``t``, drawn from the record as a render draws it;
        at ``quality`` (a key of :data:`~arachne.constants.QUALITIES`) when
        given, else at the pixel size the scene was executed with.

        A state-function play is restored at an instant it was not sampled
        at; raises :class:`~arachne.timeline.restore.Unrestorable` where it
        can't be, and :class:`~arachne.timeline.routes.RecordRefused` for a
        scene whose camera the record can't draw.
        """
        self._use_quality(quality)
        record = self.record
        (pixels,) = self._drawn(
            record, [_resolve(record, exact(t))], self._static_picture
        )
        return self.scene.renderer.camera.get_image(pixels)

    def _drawn(
        self,
        record: Record,
        answers: Iterable[_Resolved],
        static: Callable[[Drawing, PlayRecord], PixelArray | None],
    ) -> list[PixelArray]:
        """The frames of ``answers``, instants ``record`` answers, drawn as
        every frame a session gives is drawn: the session's own record's, and
        a forward run's (:mod:`.timeline.fallback`). ``static(drawing, play)``
        gives a play's static picture.
        """
        with Drawing(record) as drawing:
            return [
                _draw(drawing, record, found, lambda play: static(drawing, play))
                for found in answers
            ]

    def answers(
        self,
        instants: Iterable[Fraction | float],
        quality: str | None = None,
        *,
        frames: bool = True,
    ) -> list[Answer]:
        """The state at each of ``instants``, and its frame unless ``frames``
        is false (at ``quality`` as :meth:`frame` draws it), in order.

        The record answers what it can. The instants it can't (a state play
        that can't be restored there, a record that refuses drawing) are
        answered by one forward run for them all (:mod:`.timeline.fallback`):
        a new execution of the scene, whose answers say ``"forward"`` and
        name why the record could not answer, and which is compared with
        the record's execution (:attr:`forward_runs`, :attr:`divergences`).
        A forward run of a source changed since the record is compared with
        nothing: its answers say ``source-changed``, and :attr:`stale` is set.
        """
        return answer_instants(self, instants, quality, frames=frames)

    def summary(self) -> ExecutionSummary:
        """The summary of the session's execution, which another execution
        of its source, in this process or another, is compared with
        (:meth:`compare`, :meth:`.ExecutionSummary.to_json`).
        """
        if self._summary is None or self._summary[0] is not self.record:
            self._summary = (self.record, summarize(self.record))
        return self._summary[1]

    def compare(self, summary: ExecutionSummary) -> Divergence | None:
        """Compare the session's execution with ``summary``'s, an execution of
        the same source (:func:`.execution.compare`): the first play where they
        differ, also kept in :attr:`divergences`, or ``None``. Raises
        :class:`.execution.Incomparable` for another source, or an unknown one.
        """
        found = compare(self.summary(), summary)
        if found is not None:
            self.divergences.append(found)
        return found

    def _static_picture(self, drawing: Drawing, play: Any) -> PixelArray | None:
        """The play's static picture, kept for the next frame of the same play."""
        key = (play.index, self._quality)
        if self._static is not None and self._static[:2] == key:
            return self._static[2]
        static = drawing.static(play)
        self._static = (*key, static)
        return static

    def _use_quality(self, quality: str | None) -> None:
        if quality == self._quality:
            return
        camera = self.scene.renderer.camera
        if quality is None:
            height, width = self._executed_shape
        else:
            if quality not in QUALITIES:
                raise ValueError(
                    f"Unknown quality {quality!r}: one of {', '.join(QUALITIES)}."
                )
            height = QUALITIES[quality]["pixel_height"]
            width = QUALITIES[quality]["pixel_width"]
        resize(camera, height, width)
        self._quality = quality
        self._static = None

    def close(self) -> None:
        """Let go of the scene and its record."""
        sys.modules.pop(self.module.__name__, None)


def resize(camera: Any, height: int, width: int) -> None:
    """Give ``camera`` ``height`` by ``width`` pixels. The frame keeps its
    size in scene units, as a render at another quality has it; only the
    pixels change.
    """
    camera.pixel_height, camera.pixel_width = height, width
    camera.init_background()
    camera.pixel_array_to_cairo_context.clear()
    camera.reset()


@dataclass(frozen=True)
class Source:
    """What a session executed: the scene class ``scene`` of ``path``, under
    ``settings``, its ticked plays stepping as ``stepping`` says.
    """

    path: Path
    scene: str
    settings: dict[str, Any]
    stepping: bool | Literal["brute_force"]


def execute(
    path: Path,
    scene: str,
    settings: dict[str, Any],
    recorder: Callable[[Scene], Recorder],
) -> tuple[Scene, ModuleType]:
    """Import ``path`` under a module of its own and execute its scene class
    ``scene`` under ``settings``, recorded by ``recorder(scene)``: the scene
    and its module. An execution that raises leaves its module unimported.
    """
    base_dir = str(path.parent)
    with tempconfig(settings):
        ORIGINS.start(base_dir)
        added_path = base_dir not in sys.path
        if added_path:
            sys.path.insert(0, base_dir)
        try:
            module = _import(path)
            try:
                scene_class = getattr(module, scene, None)
                if not isinstance(scene_class, type):
                    raise LookupError(f"{path.name} defines no scene named {scene!r}.")
                instance: Scene = scene_class()
                instance.timeline = recorder(instance)
                instance.render()
            except BaseException:
                sys.modules.pop(module.__name__, None)
                raise
        finally:
            ORIGINS.stop()
            if added_path and base_dir in sys.path:
                sys.path.remove(base_dir)
    return instance, module


def _import(path: Path) -> ModuleType:
    """Import ``path`` under a name of its own, so two sessions of one file
    never share a module.
    """
    name = f"arachne_session_{next(_MODULE_NUMBERS)}_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, os.fspath(path))
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"{path} is not a Python file.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module
