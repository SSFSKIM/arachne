"""The fixture pack and the planted defects are what their one sequence
draws: drawn again into a directory of the test's own, each is the committed
one, file for file, admission's record included, with the same structure and
numbers equal to within 1e-6.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from arachne.assets.reader import FIXTURES

from . import draw_fixture_pack

#: The planted defects admission must find.
DEFECTS = Path(__file__).parent / "defects"

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
        draw_fixture_pack.NAMES
    )
    _drawn_as_committed(tmp_path, FIXTURES)


def test_the_planted_defects_are_what_their_sequence_draws(tmp_path):
    draw_fixture_pack.draw_defects(tmp_path)
    assert sorted(p.name for p in DEFECTS.iterdir() if p.is_dir()) == sorted(
        draw_fixture_pack.DEFECTS
    )
    _drawn_as_committed(tmp_path, DEFECTS)


@pytest.mark.parametrize("root", [FIXTURES, DEFECTS])
def test_every_file_is_as_the_hooks_leave_it(root):
    for file in _files(root):
        text = (root / file).read_text()
        assert text.endswith("\n"), file
        assert not text.endswith("\n\n"), file
        assert all(line == line.rstrip() for line in text.splitlines()), file
        assert b"\r" not in (root / file).read_bytes(), file
