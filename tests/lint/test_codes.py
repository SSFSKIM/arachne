"""The lint code table: stable strings, append-only, each with its
severity, class, meaning, fix and the detail fields its findings carry.
"""

from __future__ import annotations

import pytest

from arachne.lint import CODES, Code
from arachne.lint.codes import RESERVED, SEVERITIES, code

#: Every code the table has held, in order, with its severity and class:
#: the table may grow after these, never change or lose them.
PUBLISHED = [
    ("clip.frame_edge", "error", "Clearance"),
    ("overlap.text", "error", "Clearance"),
    ("gap.text", "error", "Clearance"),
    ("contact.held", "error", "Clearance"),
    ("occlusion.glyph", "error", "Clearance"),
    ("label.left_behind", "error", "Clearance"),
    ("axes.outside", "error", "Clearance"),
    ("replay.diverged", "error", "ReplayDiverged"),
    ("source.changed", "note", "SourceChanged"),
    ("motion.jump", "warning", "MotionJump"),
    ("render.unsupported", "error", "Unsupported"),
    ("render.fallback", "warning", "RenderFallback"),
    ("text.small", "note", "TextSmall"),
]


def test_the_table_only_grows():
    held = [(c.code, c.severity, c.cls) for c in CODES.values()]
    assert held[: len(PUBLISHED)] == PUBLISHED
    assert list(CODES) == [c.code for c in CODES.values()]


def test_every_code_is_complete():
    for entry in CODES.values():
        assert isinstance(entry, Code)
        assert entry.severity in SEVERITIES
        assert entry.cls
        assert entry.meaning
        assert entry.fix
        assert all(isinstance(name, str) for name in entry.detail)


def test_the_drawing_codes_keep_their_strings_and_meanings():
    """The two codes the drawing raises keep the strings, severities and
    meanings it gave them.
    """
    unsupported = CODES["render.unsupported"]
    assert unsupported.severity == "error"
    assert unsupported.meaning.startswith(
        "a frame's description carries a record kind or feature bit the drawer"
        " does not know; the frame is refused, never drawn partly"
    )
    fallback = CODES["render.fallback"]
    assert fallback.severity == "warning"
    assert fallback.meaning.startswith(
        "a scene was drawn with Cairo because the Frame IR's path refused"
        " something in it (a camera class of the scene's own; for a 3D scene,"
        " anything the core refuses); the finding names the cause and its"
        " source line"
    )
    assert fallback.detail == ("camera", "cause", "methods", "where")


def test_motion_jump_says_which_side_moved():
    assert "side" in CODES["motion.jump"].detail


def test_image_codes_are_reserved():
    assert RESERVED == ("image.",)
    assert not any(name.startswith(RESERVED) for name in CODES)


def test_an_unknown_code_is_refused_by_name():
    with pytest.raises(KeyError, match="no.such"):
        code("no.such")
