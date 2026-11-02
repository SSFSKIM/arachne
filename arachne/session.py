"""A scene opened in process: executed once, recorded, then asked about.

:meth:`Session.open` imports a scene file, executes one of its scenes
forward with nothing written, and keeps the scene's :class:`Record` with
every instant the run evaluated. :meth:`Session.state_at` answers from the
record, and :meth:`Session.frame` draws one frame from it; neither runs the
scene's code again.
"""

from __future__ import annotations

__all__ = ["Session"]

import importlib.util
import itertools
import os
import sys
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._config import tempconfig
from .constants import QUALITIES
from .timeline.clock import exact
from .timeline.draw import Drawing
from .timeline.origins import ORIGINS
from .timeline.recorder import Recorder
from .timeline.routes import draw

if TYPE_CHECKING:
    from types import ModuleType

    from PIL import Image

    from .scene.scene import Scene
    from .timeline.record import Record
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

    @classmethod
    def open(
        cls,
        file: StrPath,
        scene: str,
        *,
        grid: Fraction | float | None = None,
        config: dict[str, Any] | None = None,
    ) -> Session:
        """Execute the scene class named ``scene`` from ``file`` and record it.

        ``grid`` is the frame rate whose instants the record keeps; ``None``
        keeps every instant the run evaluated. ``config`` overrides the
        configuration the scene is executed under; nothing is written either
        way.
        """
        path = Path(file).resolve()
        base_dir = str(path.parent)
        settings = {
            **(config or {}),
            "dry_run": True,
            "disable_caching": True,
        }
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
                        raise LookupError(
                            f"{path.name} defines no scene named {scene!r}."
                        )
                    instance: Scene = scene_class()
                    instance.timeline = Recorder(instance, grid=grid, draws=False)
                    instance.render()
                except BaseException:
                    sys.modules.pop(module.__name__, None)
                    raise
            finally:
                ORIGINS.stop()
                if added_path and base_dir in sys.path:
                    sys.path.remove(base_dir)
        record = instance.record
        assert record is not None
        return cls(instance, record, module)

    def state_at(self, t: Fraction | float) -> State:
        """The scene at ``t``, from the record."""
        return self.record.state_at(t)

    def frame(self, t: Fraction | float, quality: str | None = None) -> Image.Image:
        """The frame at ``t``, drawn from the record as a render draws it;
        at ``quality`` (a key of :data:`~arachne.constants.QUALITIES`) when
        given, else at the pixel size the scene was executed with.

        Raises :class:`~arachne.timeline.routes.Unavailable` at an instant a
        state-function play was not sampled at, and
        :class:`~arachne.timeline.routes.RecordRefused` for a scene whose
        camera the record can't draw.
        """
        self._use_quality(quality)
        record = self.record
        with Drawing(record) as drawing:
            pixels = draw(
                drawing,
                record,
                exact(t),
                lambda play: self._static_picture(drawing, play),
            )
        return self.scene.renderer.camera.get_image(pixels)

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
        # The frame keeps its size in scene units, as a render at that
        # quality has it; only the pixels change.
        camera.pixel_height, camera.pixel_width = height, width
        camera.init_background()
        camera.pixel_array_to_cairo_context.clear()
        camera.reset()
        self._quality = quality
        self._static = None

    def close(self) -> None:
        """Let go of the scene and its record."""
        sys.modules.pop(self.module.__name__, None)


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
