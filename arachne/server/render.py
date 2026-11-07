"""``clip`` and ``render``: the session's scene executed afresh through the
render path, into the media file the caller names, and checked against the
execution summary the store keeps of the session's execution.

The render writes into a directory of its own beside ``out``; its video is
renamed onto ``out`` only once the render succeeded, and only once the
request has claimed its answer, which its deadline's error then can't take
from it: a render that ends any other way (an error, a refusal, the
request's deadline ending the process before the claim) leaves ``out`` as
it was, and the rename is atomic, so ``out`` is only ever the file before
or the render. A render is a render as the command line makes it: the
scene's own recorder, at the quality's pixel size and frame rate, the plays
it leaves out executed all the same, as ``-n first,last`` executes them.
"""

from __future__ import annotations

__all__ = ["SelfCheck", "Rendered", "plays_meeting", "render"]

import dataclasses
import os
import shutil
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .._config import config as global_config
from .._config import tempconfig
from ..constants import QUALITIES
from ..errors import explain_on
from ..session import execute
from ..timeline.execution import Divergence, Incomparable, compare, summarize

if TYPE_CHECKING:
    from ..scene.scene import Scene
    from ..timeline.execution import ExecutionSummary
    from ..timeline.record import Record


#: Why a render with no summary like its own execution was not checked.
NO_REFERENCE = (
    "No summary of this execution at the render's frame rate and pixel "
    "aspect: open the session at them to check its renders."
)


@dataclass(frozen=True)
class SelfCheck:
    """Whether a render was checked against an execution summary: ``status``,
    ``"checked"`` or ``"unchecked"``; for an unchecked one, ``reason``; for a
    checked one, ``reference``, the execution it was checked against:
    ``"lint"``, the one a lint in the session judged, else ``"own"``, the
    session's own.
    """

    status: str
    reason: str | None = None
    reference: str | None = None


@dataclass(frozen=True)
class Rendered:
    """A render written: ``path``, ``out``; ``record``, its execution's
    record; ``self_check``, whether it was checked; ``divergence``, where it
    differs from the summary it was checked against, if it does.
    """

    path: Path
    record: Record
    self_check: SelfCheck
    divergence: Divergence | None


def plays_meeting(
    spans: Sequence[tuple[Fraction, Fraction]],
    t0: Fraction | float,
    t1: Fraction | float,
) -> tuple[int, int] | None:
    """The first and last of the plays (``spans``, each ``(start, end)``, by
    index) that meet ``[t0, t1]``: from the first that ends at or after
    ``t0`` to the last that starts at or before ``t1``. ``None`` when none
    does.
    """
    first = next((i for i, (_, end) in enumerate(spans) if end >= t0), None)
    last = max((i for i, (start, _) in enumerate(spans) if start <= t1), default=None)
    if first is None or last is None or first > last:
        return None
    return first, last


def quality_options(quality: str | None) -> dict[str, Any]:
    """The configuration options of ``quality``, a key of
    :data:`~arachne.constants.QUALITIES`: its pixel size and frame rate.
    """
    if quality is None:
        return {}
    found = QUALITIES[quality]
    return {
        "pixel_width": found["pixel_width"],
        "pixel_height": found["pixel_height"],
        "frame_rate": found["frame_rate"],
    }


def render(
    file: str,
    scene: str,
    config: Mapping[str, Any],
    *,
    named: str,
    quality: str | None,
    plays: tuple[int, int] | None,
    out: Path,
    reference: ExecutionSummary | None,
    scratch: Callable[[Path | None], None],
    publish: Callable[[Callable[[], None]], bool],
    linted: bool = False,
) -> Rendered:
    """Render ``scene`` of ``file`` (a real path, ``named`` so by the
    caller) under ``config`` (the session's), at ``quality`` (else the
    session's pixel size and frame rate), the plays ``plays`` (first and
    last, by index; every play when ``None``), to ``out``, checked against
    ``reference``: the plays it rendered, up to its last, and with every
    play its end too. ``scratch`` is told the directory the render writes
    in while it does, and ``None`` once it is gone. ``publish`` runs the
    rename onto ``out`` once the request has claimed its answer, false
    when the request's deadline passed first and the rename never ran.
    ``linted``: whether ``reference`` is the execution a lint in the
    session judged.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    media = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    scratch(media)
    try:
        options = {
            **config,
            **quality_options(quality),
            "dry_run": False,
            "disable_caching": True,
            "media_dir": os.fspath(media),
            "progress_bar": "none",
            "preview": False,
            "show_in_file_browser": False,
            "write_all": False,
            "save_sections": False,
            "log_to_file": False,
        }
        if options.get("format", "auto") == "auto":
            options["format"] = "mp4"
        if plays is not None:
            options["from_animation_number"], options["upto_animation_number"] = plays
        with tempconfig(options):
            settings = global_config.copy()
        with tempconfig(settings):
            made, record = _execute(Path(file), scene, settings, named)
            summary = summarize(record)
        video = Path(made.renderer.file_writer.movie_file_path)
        checked, divergence = _check(summary, reference, plays)
        if checked.status == "checked":
            judged = "lint" if linted else "own"
            checked = dataclasses.replace(checked, reference=judged)
        if not publish(lambda: os.replace(video, out)):
            raise RuntimeError(
                "The request's deadline passed before its render was written."
            )
        return Rendered(out, record, checked, divergence)
    finally:
        shutil.rmtree(media, ignore_errors=True)
        scratch(None)


def _execute(path: Path, scene: str, settings: Any, named: str) -> tuple[Scene, Record]:
    """Execute and render ``scene`` of ``path`` under ``settings``, recorded
    by the scene's own recorder, as the command line renders it; what it
    raises is raised as it was, explained (:func:`~arachne.errors.explain_on`).
    """
    made: list[Scene] = []

    def own(instance: Scene) -> Any:
        made.append(instance)
        return instance._arachne_timeline

    try:
        instance, module = execute(path, scene, settings, own)
    except BaseException as error:
        explain_on(error, named, made[-1].time if made else None)
        raise
    sys.modules.pop(module.__name__, None)
    record = instance.record
    assert record is not None
    return instance, record


def _check(
    summary: ExecutionSummary,
    reference: ExecutionSummary | None,
    plays: tuple[int, int] | None,
) -> tuple[SelfCheck, Divergence | None]:
    """Whether ``summary``, the render's, was checked against ``reference``,
    and where it differs from it: up to the last play rendered, and where
    the scripts ended when every play was executed. Unchecked with nothing
    to check against, or a reference :func:`compare` refuses.
    """
    if reference is None:
        return SelfCheck("unchecked", NO_REFERENCE), None
    before = None
    if plays is not None and plays[1] + 1 < len(reference.plays):
        before = plays[1] + 1
    try:
        return SelfCheck("checked"), compare(summary, reference, before_play=before)
    except Incomparable as refused:
        reason = f"The summary can't be compared with the render's: {refused}"
        return SelfCheck("unchecked", reason), None
