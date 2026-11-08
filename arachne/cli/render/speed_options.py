from __future__ import annotations

from typing import TYPE_CHECKING

from click import BadParameter
from cloup import Path, option, option_group

if TYPE_CHECKING:
    from click import Context, Option

__all__ = ["speed_options"]


def validate_workers(
    ctx: Context, param: Option, value: str | None
) -> int | str | None:
    """A positive worker count, or ``"auto"``."""
    if value is None or value == "auto":
        return value
    if value.isdecimal() and int(value) >= 1:
        return int(value)
    raise BadParameter(
        f"{value!r} is neither a positive integer nor 'auto'", ctx=ctx, param=param
    )


speed_options = option_group(
    "Speed options",
    option(
        "--speed-levers/--no-speed-levers",
        "speed_levers",
        default=None,
        help="Turn every speed lever on (as each is set) or off (each as Manim "
        "CE does it). The levers change how fast a render is made, never what "
        "it shows.",
    ),
    option(
        "--vectorize-trusted/--no-vectorize-trusted",
        "vectorize_trusted",
        default=None,
        help="Compute a trusted play's values for all its instants at once.",
    ),
    option(
        "--construction-memo/--no-construction-memo",
        "construction_memo",
        default=None,
        help="Reuse what an equal text or number construction built.",
    ),
    option(
        "--draw-reuse/--no-draw-reuse",
        "draw_reuse",
        default=None,
        help="Reuse the pixels of a frame whose recorded state is unchanged.",
    ),
    option(
        "--hold-gaps/--no-hold-gaps",
        "hold_gaps",
        default=None,
        help="Write a run of equal frames as one frame lasting the run "
        "(variable frame rate).",
    ),
    option(
        "--workers",
        "render_workers",
        callback=validate_workers,
        default=None,
        metavar="N|auto",
        help="How many processes draw and encode the render; auto is the number "
        "of processors this process may run on. Defaults to 1.",
    ),
    option(
        "--typeset_dir",
        type=Path(file_okay=False),
        default=None,
        help="Directory where typeset Tex and Text are kept and shared across "
        "scenes, renders and processes (also ARACHNE_TYPESET_DIR).",
    ),
)
