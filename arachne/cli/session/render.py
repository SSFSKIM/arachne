"""``arachne session render`` and ``arachne session clip``: a scene's
plays rendered through its session, checked against the summary of the
session's execution.

A written file is an answer, whether its render was checked or not: the
command exits 0, and with ``--strict`` 1 when the render was unchecked or
diverged. The human form ends with the check: ``self_check: checked``, or
``self_check: unchecked (<why>)``.
"""

from __future__ import annotations

__all__ = ["clip", "render"]

from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import click
import cloup

from ..asking import (
    EXIT_STRICT,
    QUALITY_FLAGS,
    RESIDENT,
    ask,
    instant,
    printed,
    render_options,
    seconds,
)

F = TypeVar("F", bound=Callable[..., Any])


def _span(
    ctx: click.Context, param: click.Parameter, value: str | None
) -> tuple[int, int | None] | None:
    """``A,B`` as render's ``-n`` reads it: play A through play B; ``A``,
    play A through the last.
    """
    if value is None:
        return None
    try:
        found = [int(each) for each in value.split(",")]
    except ValueError:
        found = []
    if (
        len(found) not in (1, 2)
        or found[0] < 0
        or (len(found) == 2 and found[1] < found[0])
    ):
        raise click.BadParameter(
            "A,B: play A through play B, counted from 0, B not before A;"
            " or A, play A through the last."
        )
    return found[0], found[1] if len(found) == 2 else None


def _rendering(command: F) -> F:
    """What a render and a clip take besides the session: its quality, the
    file to write and ``--strict``.
    """
    decorators = [
        cloup.option(
            "-q",
            "--quality",
            type=click.Choice(list(QUALITY_FLAGS)),
            default=None,
            help="Render at this quality (as render's -q); else at the session's.",
        ),
        cloup.option(
            "-o",
            "--output",
            required=True,
            type=click.Path(dir_okay=False, path_type=Path),
            help="The media file to write, once the render is whole.",
        ),
        cloup.option(
            "--strict",
            is_flag=True,
            help="Exit 1 when the render was not checked, or diverged.",
        ),
    ]
    for decorator in reversed(decorators):
        command = decorator(command)
    return command


@cloup.command(epilog=RESIDENT)
@render_options
@cloup.option(
    "--plays",
    metavar="A[,B]",
    callback=_span,
    help="Render play A through play B, counted from 0, waits included, as"
    " render's -n does; A alone, through the last. Else every play.",
)
@_rendering
def render(
    file: Path,
    scene: str,
    config: dict[str, Any],
    once: bool,
    deadline_s: float,
    as_json: bool,
    plays: tuple[int, int | None] | None,
    quality: str | None,
    output: Path,
    strict: bool,
) -> None:
    """Render SCENE of FILE through its session, every play or those
    --plays names, checked against the summary of the session's execution.
    """

    def params(opened: dict[str, Any]) -> dict[str, Any]:
        asked: dict[str, Any] = {"out": str(output.resolve()), **_quality(quality)}
        if plays is not None:
            asked["plays"] = _run(plays, len(opened["record"]["plays"]))
        return asked

    answer = ask(
        file,
        scene,
        config,
        once=once,
        deadline_s=deadline_s,
        as_json=as_json,
        method="render",
        params=params,
    )
    _done(answer, "render", as_json, strict)


@cloup.command(epilog=RESIDENT)
@render_options
@cloup.option(
    "--t0",
    "t0",
    required=True,
    callback=instant,
    help="The span's start, in seconds or exactly as 'n/d'.",
)
@cloup.option(
    "--t1",
    "t1",
    required=True,
    callback=instant,
    help="The span's end, in seconds or exactly as 'n/d'.",
)
@_rendering
def clip(
    file: Path,
    scene: str,
    config: dict[str, Any],
    once: bool,
    deadline_s: float,
    as_json: bool,
    t0: float | str,
    t1: float | str,
    quality: str | None,
    output: Path,
    strict: bool,
) -> None:
    """Render the plays of SCENE in FILE that meet T0 to T1, through its
    session, checked against the summary of the session's execution.
    """
    answer = ask(
        file,
        scene,
        config,
        once=once,
        deadline_s=deadline_s,
        as_json=as_json,
        method="clip",
        params={"t0": t0, "t1": t1, "out": str(output.resolve()), **_quality(quality)},
    )
    _done(answer, "clip", as_json, strict)


def _run(plays: tuple[int, int | None], count: int) -> list[int]:
    """The plays ``--plays`` names, of a scene of ``count`` plays: play A
    through play B, or the last play when B is past it or not given, as
    render's ``-n`` takes them. An A past the last play is a usage error.
    """
    first, last = plays
    if first >= count:
        where = (
            "the scene has no plays" if count == 0 else f"the last play is {count - 1}"
        )
        raise click.UsageError(
            f"Invalid value for '--plays': play {first} is past the end; {where}."
        )
    return list(range(first, count if last is None else min(last, count - 1) + 1))


def _quality(flag: str | None) -> dict[str, Any]:
    return {} if flag is None else {"quality": QUALITY_FLAGS[flag]}


def _done(answer: dict[str, Any], form: str, as_json: bool, strict: bool) -> None:
    """Print the render form of ``answer``; with ``strict``, exit 1 when it
    was unchecked or diverged.
    """
    found = answer[form]
    printed(answer, as_json, lambda: formatted(found))
    if strict and (found["self_check"]["status"] != "checked" or "divergence" in found):
        raise click.exceptions.Exit(EXIT_STRICT)


def formatted(found: dict[str, Any]) -> str:
    """A render or clip's form, ``found``, for reading: what it wrote, the
    divergence its check found, and its check, last.
    """
    shown = [play for play in found["timeline"] if not play["skipped"]]
    if shown:
        first, last = shown[0], shown[-1]
        what = (
            f"plays {first['index']}-{last['index']}"
            f" ({seconds(first['start'])}-{seconds(last['end'])} s)"
        )
    else:
        what = "no plays"
    lines = [f"{found['path']}: {what}"]
    divergence = found.get("divergence")
    if divergence is not None:
        lines.append(
            f"divergence {divergence['code']} ({divergence['kind']})"
            f" at play {divergence['play']}: {divergence['reason']}"
        )
    check = found["self_check"]
    lines.append(
        "self_check: checked"
        if check["status"] == "checked"
        else f"self_check: unchecked ({check['reason']})"
    )
    return "\n".join(lines)
