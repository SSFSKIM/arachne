"""The one door to a software adapter (such as Mesa's lavapipe), for tests only.

No draft and no final is drawn on a software adapter. Nothing in the engine imports this module;
a test checks that.
"""

from __future__ import annotations

from arachne_gpu._gpu import software_compositor

__all__ = ["software_compositor"]
