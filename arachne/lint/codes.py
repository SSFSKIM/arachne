"""The lint's codes: stable strings, append-only.

Each code has its severity (``error``, ``warning`` or ``note``), its
``class`` (the name a report gives its kind, as the exception names of the
checks the clearance family comes from), what it means, how to fix it, and
the fields its findings' ``detail`` may carry. A code is never renamed,
removed or given another meaning; new ones are appended. Thresholds are in
scene units of the home frame, with pixel forms stated at a picture 1080
pixels tall (:data:`THRESHOLD_HEIGHT`).

``render.unsupported`` and ``render.fallback`` are the drawing's codes,
here with the strings, severities and meanings the drawing gave them; the
Frame IR's own table re-exports them from here once both live in one tree.
The ``image.`` codes are reserved for the image checks.
"""

from __future__ import annotations

__all__ = ["CODES", "RESERVED", "SEVERITIES", "THRESHOLD_HEIGHT", "Code", "code"]

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

#: The severities, gravest first. Only an ``error`` fails a report.
SEVERITIES = ("error", "warning", "note")

#: Code prefixes kept for checks to come.
RESERVED = ("image.",)

#: The picture height, in pixels, the table's pixel thresholds are stated
#: at; a rule scales them to the home frame.
THRESHOLD_HEIGHT = 1080


@dataclass(frozen=True)
class Code:
    """One lint code: its string, severity, class, meaning and fix, the
    names of the ``detail`` fields its findings may carry (besides
    ``folded``, which a report adds to any finding look-alikes fold into),
    and whether look-alikes of it fold: a code whose every finding is an
    event of its own, at a time of its own, never does.
    """

    code: str
    severity: str
    cls: str
    meaning: str
    fix: str
    detail: tuple[str, ...] = ()
    folds: bool = True


_CLEARANCE = "Clearance"

_TABLE = (
    Code(
        "clip.frame_edge",
        "error",
        _CLEARANCE,
        "a text's glyphs, or a shape that was wholly inside, run past the frame's edge",
        "Move or shrink it so it sits inside the frame with a margin, or give the"
        " layout room before it is placed.",
        ("edge", "past"),
    ),
    Code(
        "overlap.text",
        "error",
        _CLEARANCE,
        "a text overlaps another text or a shape",
        "Move one of the two apart, or place one against the other with a gap.",
        ("overlap",),
    ),
    Code(
        "gap.text",
        "error",
        _CLEARANCE,
        "two texts sit closer than the gap the scene declared for texts",
        "Widen the space between the two texts to the declared gap.",
        ("gap", "required"),
    ),
    Code(
        "contact.held",
        "error",
        _CLEARANCE,
        "two things that are not text touch or overlap",
        "Move them apart, or declare the contact meant by exempting the pair.",
        ("gap", "required"),
    ),
    Code(
        "occlusion.glyph",
        "error",
        _CLEARANCE,
        "something drawn after a text paints over its glyphs",
        "Draw the text after what covers it, raise it in the paint order, or move"
        " the covering shape off it.",
        ("covered",),
    ),
    Code(
        "label.left_behind",
        "error",
        _CLEARANCE,
        "a label stays where its referent was after the referent moved away",
        "Move the label with its referent: carry it in the same animation, or"
        " place it from the referent's position each frame.",
        ("referent", "distance"),
    ),
    Code(
        "axes.outside",
        "error",
        _CLEARANCE,
        "a point or mobject asserted to lie on the axes' plotted region lies outside it",
        "Keep the point within the axes' ranges: clamp it, widen the ranges, or"
        " stop drawing where it leaves them.",
        ("axes", "point"),
    ),
    Code(
        "replay.diverged",
        "error",
        "ReplayDiverged",
        "two executions of one source differ at an instant both evaluated (an"
        " error: what is shown may not be what was checked), or a play restored"
        " at a recorded instant differs from what was recorded there and was"
        " ticked instead (a note: the engine corrected itself)",
        "Make the scene deterministic: seed its random generators and keep the"
        " files it reads unchanged between runs.",
        ("kind", "play", "reason", "changed_inputs"),
    ),
    Code(
        "source.changed",
        "note",
        "SourceChanged",
        "the scene's source changed since its record was made, so the record"
        " is of a program the file no longer is",
        "Open the scene again to lint the source as it is now.",
    ),
    Code(
        "motion.jump",
        "warning",
        "MotionJump",
        "a body visible on both sides of the boundary between two plays is in"
        " another place after it, with nothing carrying it, while it moves on"
        " either side",
        "Animate the move, or fade the body out and in again if the cut is meant.",
        ("side", "distance", "from", "to", "boundary"),
        folds=False,
    ),
    Code(
        "render.unsupported",
        "error",
        "Unsupported",
        "a frame's description carries a record kind or feature bit the drawer"
        " does not know; the frame is refused, never drawn partly",
        "Draw the scene with Cairo, or leave out the feature the drawer does not know.",
        ("subject",),
    ),
    Code(
        "render.fallback",
        "warning",
        "RenderFallback",
        "a scene was drawn with Cairo because the Frame IR's path refused"
        " something in it (a camera class of the scene's own; for a 3D scene,"
        " anything the core refuses); the finding names the cause and its"
        " source line",
        "Use the engine's own camera classes and methods unchanged for the"
        " faster path.",
        ("camera", "cause", "methods", "where"),
    ),
    Code(
        "text.small",
        "note",
        "TextSmall",
        "a text is shorter than 20 pixels at 1080 pixels tall, hard to read",
        "Scale the text up, or give it a larger font size.",
        ("height_px",),
    ),
)

#: Every code, by its string, in the order they were added.
CODES: Mapping[str, Code] = MappingProxyType({entry.code: entry for entry in _TABLE})


def code(name: str) -> Code:
    """The code ``name``; a :class:`KeyError` naming it for one the table
    does not hold, reserved ones included.
    """
    try:
        return CODES[name]
    except KeyError:
        reserved = name.startswith(RESERVED)
        why = "is reserved" if reserved else "is not a lint code"
        raise KeyError(f"{name!r} {why}") from None
