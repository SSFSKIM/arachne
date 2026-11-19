"""The GPU build's Python tests need the built extension.

Arachne's own test run collects the whole repository; there the extension is not installed, so these
tests are left out by collection, never skipped. The gate ``gpu-binding`` installs the extension and
runs them.
"""

from __future__ import annotations

import importlib.util

if importlib.util.find_spec("arachne_gpu") is None:
    collect_ignore_glob = ["*"]
