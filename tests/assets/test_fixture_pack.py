"""The fixture pack and the planted defects are what their one sequence
draws: drawn again into a directory of the test's own, each rig is the
committed one, file for file, admission's record included, with the same
structure and numbers equal to within 1e-6. A media asset drawn again is
admitted on its own and has the committed one's files, manifest (but its
admission record) and frame digests; its files' bytes, and so its
admission's digest, move with the PNG encoder's compression library, and are
not compared. So it is with a rig of stills: each member's frame digest is
the manifest's, and is its file's.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from arachne.assets import admit
from arachne.assets.reader import FIXTURES, read_asset
from arachne.mobject.media.frames import decode, frame_digest

from . import draw_fixture_pack

#: The planted defects admission must find.
DEFECTS = Path(__file__).parent / "defects"
#: The planted media defects the frame checks must find.
MEDIA_DEFECTS = Path(__file__).parents[1] / "media" / "broken"

#: A number standing alone: not a run of digits inside a word, a colour or
#: a digest, which are compared as text.
_NUMBER = re.compile(
    r"(?<![0-9A-Za-z#])-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![0-9A-Za-z])"
)


def _files(root: Path) -> list[str]:
    return sorted(
        p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
    )


def _same(drawn: str, committed: str) -> bool:
    """Whether two texts differ only in numbers, by at most 1e-6 each."""
    drawn_numbers = _NUMBER.findall(drawn)
    committed_numbers = _NUMBER.findall(committed)
    return (
        _NUMBER.split(drawn) == _NUMBER.split(committed)
        and len(drawn_numbers) == len(committed_numbers)
        and all(
            abs(float(a) - float(b)) <= 1e-6
            for a, b in zip(drawn_numbers, committed_numbers, strict=True)
        )
    )


def _drawn_as_committed(drawn: Path, committed: Path) -> None:
    assert _files(drawn) == _files(committed)
    for file in _files(committed):
        text = (drawn / file).read_text()
        assert _same(text, (committed / file).read_text()), file


def test_the_pack_is_what_its_sequence_draws(tmp_path):
    draw_fixture_pack.draw(tmp_path)
    assert sorted(p.name for p in FIXTURES.iterdir() if p.is_dir()) == sorted(
        draw_fixture_pack.NAMES + draw_fixture_pack.MEDIA + draw_fixture_pack.RASTER
    )
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(draw_fixture_pack.NAMES)
    for name in draw_fixture_pack.NAMES:
        _drawn_as_committed(tmp_path / name, FIXTURES / name)


def _without_checks(directory: Path) -> dict:
    document = json.loads((directory / "manifest.json").read_text())
    document.pop("checks")
    return document


def _frames_drawn_as_committed(drawn: Path, committed: Path) -> None:
    """The media asset ``drawn`` has ``committed``'s files, manifest but its
    admission record, and frame digests; and ``committed``'s record is at its
    own digest.
    """
    assert _files(drawn) == _files(committed)
    assert _without_checks(drawn) == _without_checks(committed)
    for frame in _without_checks(committed)["media"]["frames"]:
        found = {
            frame_digest(decode((root / frame["file"]).read_bytes()))
            for root in (drawn, committed)
        }
        assert len(found) == 1, frame["file"]
    record = json.loads((committed / "manifest.json").read_text())["checks"]
    assert record["digest"] == read_asset(committed).digest


def test_the_media_are_what_their_sequence_draws_and_are_admitted_alone(tmp_path):
    draw_fixture_pack.draw_media(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(draw_fixture_pack.MEDIA)
    for name in draw_fixture_pack.MEDIA:
        assert admit(tmp_path / name).passed
        assert admit(FIXTURES / name).passed
        _frames_drawn_as_committed(tmp_path / name, FIXTURES / name)


def _members(directory: Path) -> list[dict]:
    """Every raster member of the rig in ``directory``: its parts, and its
    slots' drawings.
    """
    document = _without_checks(directory)
    slots = document.get("slots", [])
    return document["parts"] + [d for slot in slots for d in slot["drawings"]]


def test_the_rigs_of_stills_are_what_their_sequence_draws_and_are_admitted_alone(
    tmp_path,
):
    draw_fixture_pack.draw_raster(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        draw_fixture_pack.RASTER
    )
    for name in draw_fixture_pack.RASTER:
        drawn, committed = tmp_path / name, FIXTURES / name
        assert admit(drawn).passed
        assert admit(committed).passed
        assert _files(drawn) == _files(committed)
        assert _without_checks(drawn) == _without_checks(committed)
        for member in _members(committed):
            for root in (drawn, committed):
                pixels = decode((root / member["file"]).read_bytes())
                assert frame_digest(pixels) == member["digest"], member["file"]
        record = json.loads((committed / "manifest.json").read_text())["checks"]
        assert record["digest"] == read_asset(committed).digest


def test_the_planted_media_defects_are_what_their_sequence_draws(tmp_path):
    draw_fixture_pack.draw_media_defects(tmp_path)
    assert sorted(p.name for p in MEDIA_DEFECTS.iterdir() if p.is_dir()) == sorted(
        draw_fixture_pack.MEDIA_DEFECTS
    )
    for name, check in draw_fixture_pack.MEDIA_DEFECTS.items():
        _frames_drawn_as_committed(tmp_path / name, MEDIA_DEFECTS / name)
        for root in (tmp_path, MEDIA_DEFECTS):
            results = json.loads((root / name / "manifest.json").read_text())
            failing = [r["name"] for r in results["checks"]["results"] if not r["pass"]]
            assert failing == [check]


def test_the_planted_defects_are_what_their_sequence_draws(tmp_path):
    draw_fixture_pack.draw_defects(tmp_path)
    assert sorted(p.name for p in DEFECTS.iterdir() if p.is_dir()) == sorted(
        draw_fixture_pack.DEFECTS
    )
    _drawn_as_committed(tmp_path, DEFECTS)


@pytest.mark.parametrize("root", [FIXTURES, DEFECTS, MEDIA_DEFECTS])
def test_every_file_is_as_the_hooks_leave_it(root):
    for file in _files(root):
        if file.endswith(".png"):
            continue  # a frame file is binary, which the hooks leave alone
        text = (root / file).read_text()
        assert text.endswith("\n"), file
        assert not text.endswith("\n\n"), file
        assert all(line == line.rstrip() for line in text.splitlines()), file
        assert b"\r" not in (root / file).read_bytes(), file
