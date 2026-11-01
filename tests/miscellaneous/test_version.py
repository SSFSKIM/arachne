from __future__ import annotations

from importlib.metadata import version

from arachne import __version__


def test_version():
    assert __version__ == version("arachne-engine")
