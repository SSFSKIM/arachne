"""A request's deadline, kept by a thread of its own.

The scene's code a request runs can't be interrupted from inside the
process: a scene that sleeps, loops or waits holds the thread serving the
request. So the deadline is kept beside it: when a request is still running
at its deadline, the watchdog calls what the server gave it for that
request, which answers the request with its error and ends the process
(:func:`os._exit`, the only end scene code can't stand in the way of).
"""

from __future__ import annotations

__all__ = ["Watchdog"]

import threading
import time
from collections.abc import Callable


class Watchdog:
    """One deadline at a time: :meth:`arm` it as a request starts and
    :meth:`disarm` it as the request ends. Past the deadline, the
    ``overran`` the request was armed with is called, once, on the
    watchdog's thread.
    """

    def __init__(self) -> None:
        self._changed = threading.Condition()
        self._deadline: float | None = None
        self._overran: Callable[[], object] | None = None
        threading.Thread(target=self._keep, name="deadline", daemon=True).start()

    def arm(self, seconds: float, overran: Callable[[], object]) -> None:
        with self._changed:
            self._deadline = time.monotonic() + seconds
            self._overran = overran
            self._changed.notify()

    def disarm(self) -> None:
        """No deadline. Once it returns, the deadline that was armed can't
        fire: a firing holds the watchdog until it ends the process.
        """
        with self._changed:
            self._deadline = self._overran = None
            self._changed.notify()

    def _keep(self) -> None:
        with self._changed:
            while True:
                if self._deadline is None:
                    self._changed.wait()
                    continue
                left = self._deadline - time.monotonic()
                if left > 0:
                    self._changed.wait(left)
                    continue
                overran, self._deadline, self._overran = self._overran, None, None
                assert overran is not None
                overran()
