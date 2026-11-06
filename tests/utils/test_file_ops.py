"""The output-format helpers Manim CE 0.20.1 exports from ``utils.file_ops``,
answered from the output the configuration resolves to: whether a render
writes a movie, and which format it writes.
"""

from __future__ import annotations

import pytest

from arachne import tempconfig
from arachne.utils import file_ops

HELPERS = ("mp4", "gif", "png", "webm", "mov")


def _answers() -> dict[str, bool]:
    found = {name: getattr(file_ops, f"is_{name}_format")() for name in HELPERS}
    found["movie"] = file_ops.write_to_movie()
    return found


@pytest.mark.parametrize(
    ("settings", "movie", "format"),
    [
        ({"format": "mp4"}, True, "mp4"),
        ({"format": "webm"}, True, "webm"),
        ({"format": "mov"}, True, "mov"),
        ({"format": "gif"}, True, "gif"),
        ({"format": "png"}, False, "png"),
        ({"format": "png-sequence"}, False, "png"),
        ({"format": "none"}, False, None),
        # The default: a movie, mp4, or mov when it is transparent.
        ({"format": "auto"}, True, "mp4"),
        ({"format": "auto", "transparent": True}, True, "mov"),
        # A dry run writes nothing.
        ({"format": "mp4", "dry_run": True}, False, None),
    ],
)
def test_the_format_helpers_answer_from_the_resolved_output(settings, movie, format):
    with tempconfig(settings):
        found = _answers()
    assert found.pop("movie") is movie
    assert found == {name: name == format for name in HELPERS}


def test_from_arachne_import_star_binds_the_format_helpers():
    bound: dict[str, object] = {}
    exec("from arachne import *", bound)
    for name in ("write_to_movie", *(f"is_{name}_format" for name in HELPERS)):
        assert bound[name] is getattr(file_ops, name)
