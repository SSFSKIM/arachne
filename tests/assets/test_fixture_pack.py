"""The fixture pack is what its one sequence draws: drawn again into a
directory of the test's own, it is the committed pack, file for file, with
the same structure and numbers equal to within 1e-6.
"""

from __future__ import annotations

import re
from pathlib import Path

from arachne.assets.reader import FIXTURES

from . import draw_fixture_pack

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


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


def test_the_pack_is_what_its_sequence_draws(tmp_path):
    draw_fixture_pack.draw(tmp_path)
    assert sorted(p.name for p in FIXTURES.iterdir() if p.is_dir()) == sorted(
        draw_fixture_pack.NAMES
    )
    assert _files(tmp_path) == _files(FIXTURES)
    for file in _files(FIXTURES):
        drawn, committed = (tmp_path / file).read_text(), (FIXTURES / file).read_text()
        assert _same(drawn, committed), file


def test_every_file_is_as_the_hooks_leave_it():
    for file in _files(FIXTURES):
        text = (FIXTURES / file).read_text()
        assert text.endswith("\n"), file
        assert not text.endswith("\n\n"), file
        assert all(line == line.rstrip() for line in text.splitlines()), file
        assert b"\r" not in (FIXTURES / file).read_bytes(), file
