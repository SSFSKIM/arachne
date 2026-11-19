"""What the asset tests share: fixture assets copied into a test's own
directory, to be broken there one way at a time.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from arachne.assets.reader import FIXTURES
from arachne.lint.raised import RaisedFinding

#: The broken part files the refusal tests put into an asset.
BROKEN = Path(__file__).parent / "broken"


def copied(tmp_path: Path, name: str, as_name: str | None = None) -> Path:
    """The fixture asset ``name``, copied to ``tmp_path / as_name``."""
    target = tmp_path / (as_name or name)
    shutil.copytree(FIXTURES / name, target)
    return target


def edited(directory: Path, edit: Callable[[dict[str, Any]], object]) -> Path:
    """``directory``'s manifest rewritten by ``edit``, which changes the
    parsed document in place.
    """
    path = directory / "manifest.json"
    document = json.loads(path.read_text())
    edit(document)
    path.write_text(json.dumps(document, indent=2) + "\n")
    return directory


def refused(code: str, load: Callable[[], object]) -> RaisedFinding:
    """The finding ``load()`` raises, which must be of ``code``."""
    with pytest.raises(RaisedFinding) as raised:
        load()
    assert raised.value.code == code, str(raised.value)
    return raised.value
