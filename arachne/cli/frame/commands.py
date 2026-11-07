"""``arachne frame FILE SCENE --t T``: the picture at one instant of a
scene, drawn from its session and written as a PNG.
"""

from __future__ import annotations

__all__ = ["frame"]

import math
from pathlib import Path
from typing import Any

import click
import cloup

from ...constants import QUALITIES
from ..asking import ask, printed, seconds, session_options

#: The quality keys, by their flag, as ``arachne render -q`` names them.
FLAGS: dict[str, str] = {
    flag: name for name, q in QUALITIES.items() if (flag := q["flag"]) is not None
}


def _instant(ctx: click.Context, param: click.Parameter, value: str) -> float | str:
    """Seconds, as a number, else the text as given: an exact instant,
    ``"n/d"``, or what the session refuses as no time.
    """
    try:
        found = float(value)
    except ValueError:
        return value
    return found if math.isfinite(found) else value


@cloup.command()
@session_options
@cloup.option(
    "--t",
    "t",
    required=True,
    callback=_instant,
    help="The instant, in seconds or exactly as 'n/d'.",
)
@cloup.option(
    "-q",
    "--quality",
    type=click.Choice(list(FLAGS)),
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
            "quality": None if quality is None else FLAGS[quality],
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
