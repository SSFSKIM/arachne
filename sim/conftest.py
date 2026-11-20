"""The simulation crate's Python tests need the built extension.

Arachne's own test run collects the whole repository; where the extension is not installed, these
tests are left out by collection, never skipped. The gate ``sim-binding`` installs the extension
and runs them.
"""

from __future__ import annotations

import importlib.util

if importlib.util.find_spec("arachne_sim") is None:
    collect_ignore_glob = ["*"]
