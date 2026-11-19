"""The comparator's tests need the comparator installed.

Arachne's own test run collects the whole repository; where the comparator is not
installed its tests are left out. Its CI gate installs the built wheel and runs them
outside the checkout.
"""

from __future__ import annotations

import importlib.util

if importlib.util.find_spec("arachne_compare") is None:
    collect_ignore_glob = ["*"]
