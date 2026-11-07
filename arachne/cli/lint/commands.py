"""``arachne lint FILE SCENE``: a scene's lint, as its session runs it:
the scene's own checks and the temporal rules it armed.

A finding is an answer: the command exits 0 whether the lint passed or
not, and 1 with ``--strict`` when it did not.
"""

from __future__ import annotations

__all__ = ["lint"]

import math
from pathlib import Path
from typing import Any

import click
import cloup

from ..asking import (
    EXIT_STRICT,
    ask,
    finite,
    json_value,
    printed,
    seconds,
    session_options,
)


def _span(
    ctx: click.Context, param: click.Parameter, value: str | None
) -> list[float] | None:
    if value is None:
        return None
    try:
        start, end = (float(each) for each in value.split(","))
    except ValueError:
        raise click.BadParameter("two times in seconds, A,B") from None
    if not (math.isfinite(start) and math.isfinite(end)):
        raise click.BadParameter("two finite times in seconds, A,B")
    return [start, end]


def _exemptions(ctx: click.Context, param: click.Parameter, value: str | None) -> Any:
    if value is None:
        return None
    found = json_value(value)
    if not isinstance(found, list):
        raise click.BadParameter("a JSON list of selectors and pairs of them")
    return found


@cloup.command()
@session_options
@cloup.option(
    "--fps",
    type=click.FloatRange(min=0, min_open=True),
    default=None,
    callback=finite,
    help="The rate of the lint's instants, per second (the session's default: 4).",
)
@cloup.option(
    "--t-range",
    "t_range",
    metavar="A,B",
    callback=_span,
    help="Keep the findings that hold at some time from A to B, in seconds.",
)
@cloup.option(
    "--exempt",
    metavar="JSON",
    callback=_exemptions,
    help="Exemptions, a JSON list: selectors ('path:line#n', 'line:N',"
    " 'name:<variable>') and pairs of them.",
)
@cloup.option("--strict", is_flag=True, help="Exit 1 when the lint does not pass.")
def lint(
    file: Path,
    scene: str,
    config: dict[str, Any],
    once: bool,
    deadline_s: float,
    as_json: bool,
    fps: float | None,
    t_range: list[float] | None,
    exempt: list[Any] | None,
    strict: bool,
) -> None:
    """Lint SCENE of FILE: the scene's own checks and the temporal rules it
    armed, over its whole timeline.
    """
    answer = ask(
        file,
        scene,
        config,
        once=once,
        deadline_s=deadline_s,
        as_json=as_json,
        method="lint",
        params={"fps": fps, "t_range": t_range, "exempt": exempt},
    )
    found = answer["lint_result"]
    printed(answer, as_json, lambda: formatted(found, scene))
    if strict and not found["report"]["pass"]:
        raise click.exceptions.Exit(EXIT_STRICT)


def formatted(found: dict[str, Any], scene: str) -> str:
    """The lint's form, ``found``, of ``scene``, for reading."""
    report = found["report"]
    size = report["target_resolution"]
    violations = report["violations"]
    count = f"{len(violations)} finding{'' if len(violations) == 1 else 's'}"
    lines = [
        f"{'PASS' if report['pass'] else 'FAIL'}  {scene}: {count} at"
        f" {size['width']}x{size['height']},"
        f" completed in {seconds(report['duration_s'])} s"
    ]
    for kind, entries in (("", violations), ("note ", found["notes"])):
        for entry in entries:
            where = "" if entry["line"] is None else f"  line {entry['line']}"
            lines.append(
                f"  {kind}{entry['code']}  {entry['severity']}"
                f"  {seconds(entry['t_start'])}-{seconds(entry['t_end'])} s{where}"
            )
            lines.append(f"      {entry['message']}")
            if entry["fix"]:
                lines.append(f"      fix: {entry['fix']}")
    return "\n".join(lines)
