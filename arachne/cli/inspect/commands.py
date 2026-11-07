"""``arachne inspect FILE SCENE``: a scene's record, its timeline and where
its mobjects were made, as a session holds it.
"""

from __future__ import annotations

__all__ = ["inspect"]

from pathlib import Path
from typing import Any

import cloup

from ..asking import RESIDENT, ask, printed, seconds, session_options


@cloup.command(epilog=RESIDENT)
@session_options
def inspect(
    file: Path,
    scene: str,
    config: dict[str, Any],
    once: bool,
    deadline_s: float,
    as_json: bool,
) -> None:
    """Execute SCENE of FILE in a session and show its record: each play
    and wait, with its times and the scene line that ran it.
    """
    answer = ask(file, scene, config, once=once, deadline_s=deadline_s, as_json=as_json)
    printed(answer, as_json, lambda: formatted(answer["record"]))


def formatted(record: dict[str, Any]) -> str:
    """The record's form, ``record``, for reading."""
    size = record["target_resolution"]
    grid = record["grid"]
    head = (
        f"{record['scene_class']} ({record['scene_file']}):"
        f" {seconds(record['duration'])} s at {size['width']}x{size['height']},"
        f" {'every instant' if grid is None else f'{seconds(grid)} fps'},"
        f" generation {record['generation']}"
    )
    lines = [head + (", stale" if record["stale"] else "")]
    for play in record["plays"]:
        what = ", ".join(play["animations"]) or play["kind"]
        flags = [name for name in ("held", "skipped") if play[name]]
        where = "" if play["source_line"] is None else f"  line {play['source_line']}"
        lines.append(
            f"  {play['index']:>3}  {play['kind']:<5}"
            f" {seconds(play['start'])}-{seconds(play['end'])} s"
            f"  route {play['route']}{where}  {what}"
            + (f"  ({', '.join(flags)})" if flags else "")
        )
    lines.extend(
        f"  divergence {found['code']} ({found['kind']}): {found['reason']}"
        for found in record["divergences"]
    )
    return "\n".join(lines)
