"""The simulation crate's Python tests need the built extension.

Arachne's own test run collects the whole repository; where the extension is not installed, these
tests are left out by collection, never skipped. The gate ``sim-binding`` installs the extension
and runs them.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

if importlib.util.find_spec("arachne_sim") is None:
    collect_ignore_glob = ["*"]

#: Timed tests, collected only when this crate's tests run on their own (rooted here), never
#: inside the engine's parallel run, where other tests share the machine.
_ALONE = {"test_pile.py"}


def pytest_ignore_collect(collection_path, config):
    if collection_path.name in _ALONE and config.rootpath != Path(__file__).parent:
        return True
    return None
