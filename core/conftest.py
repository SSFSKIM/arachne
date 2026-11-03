"""The core's Python tests need the built extension.

Arachne's own test run collects the whole repository; there the extension is not installed, so the
core's tests are left out. The core's CI gates install it and run them from this directory.
"""

from __future__ import annotations

import importlib.util

if importlib.util.find_spec("arachne_core") is None:
    collect_ignore_glob = ["*"]
