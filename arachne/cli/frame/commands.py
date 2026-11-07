"""``arachne frame FILE SCENE --t T``: the picture at one instant of a
scene, drawn from its session and written as a PNG.
"""

from __future__ import annotations

__all__ = ["frame"]

from pathlib import Path
from typing import Any

import click
import cloup

from ..asking import (
    QUALITY_FLAGS,
    RESIDENT,
    ask,
    instant,
    printed,
    seconds,
    session_options,
)


@cloup.command(epilog=RESIDENT)
@session_options
@cloup.option(
    "--t",
    "t",
    required=True,
    callback=instant,
    help="The instant, in seconds or exactly as 'n/d'.",
)
@cloup.option(
    "-q",
    "--quality",
    type=click.Choice(list(QUALITY_FLAGS)),
    default=None,
    help="Draw at this quality's size (as render's -q); else at the session's.",
)
@cloup.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="The PNG to write; else SCENE_T.png here.",
)
def frame(
    file: Path,
    scene: str,
    config: dict[str, Any],
    once: bool,
    deadline_s: float,
    as_json: bool,
    t: float | str,
    quality: str | None,
    output: Path | None,
) -> None:
    """Draw the frame at time T of SCENE in FILE and write it as a PNG."""
    if output is None:
        output = Path(f"{scene}_{t}.png".replace("/", "-"))
    answer = ask(
        file,
        scene,
        config,
        once=once,
        deadline_s=deadline_s,
        as_json=as_json,
        method="frame",
        params={
            "t": t,
            "path": str(output.resolve()),
            "quality": None if quality is None else QUALITY_FLAGS[quality],
        },
    )
    printed(answer, as_json, lambda: formatted(answer["frame"]))


def formatted(found: dict[str, Any]) -> str:
    """The frame's form, ``found``, for reading."""
    play = "" if found["play"] is None else f", play {found['play']}"
    return (
        f"{found['path']}: t = {seconds(found['t'])} s, as of"
        f" {seconds(found['as_of'])} s{play} ({found['answer']})"
    )
