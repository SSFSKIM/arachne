"""Tests read :func:`~arachne.timeline.record.content_fields` by its first two parts.

``content_fields`` yields a field's name and its key part, and a branch of the engine may yield more parts after
those two. A test that builds a ``dict`` straight from it, or unpacks each item into exactly two names, breaks on
such a branch while passing here. This scan names every such line under ``tests/``.
"""

from __future__ import annotations

import re
from pathlib import Path

TESTS = Path(__file__).resolve().parent

#: A dict built straight from the items, or a loop that unpacks each item into two names.
PAIRS = re.compile(
    r"dict\(\s*content_fields\(|for\s+\w+\s*,\s*\w+\s+in\s+content_fields\("
)


def test_no_test_reads_content_fields_as_pairs() -> None:
    found = [
        f"{path.relative_to(TESTS.parent)}:{number}: {line.strip()}"
        for path in sorted(TESTS.rglob("*.py"))
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        )
        if PAIRS.search(line)
    ]
    assert not found, (
        "content_fields read as pairs; use the tolerant form"
        " `for key, part, *_ in content_fields(...)` (or the same in a comprehension):\n"
        + "\n".join(found)
    )
