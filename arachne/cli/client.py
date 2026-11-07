"""How the session commands reach a session: a resident one, held by a
server of the host's registry (:mod:`arachne.server.registry`), found there
or started; or, with ``--once``, one held in the command's own process,
answered by the same dispatcher a server answers with. Either way an
answer is the server's result, a JSON object; an error answer is raised as
:class:`Failed`, carrying the protocol's error object, and a server that
could not be reached, or ended without answering, as :class:`Gone`.
"""

from __future__ import annotations

__all__ = ["Client", "Failed", "Gone", "InProcess", "Resident", "reached"]

import contextlib
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol

from ..server import protocol, registry
from ..server.protocol import RpcError


class Failed(Exception):
    """An error answer: ``error`` is the protocol's error object, ``{code,
    message, data?}``.
    """

    def __init__(self, error: dict[str, Any]) -> None:
        super().__init__(error.get("message"))
        self.error = error

    @property
    def code(self) -> int:
        code: int = self.error["code"]
        return code


class Gone(Exception):
    """The server could not be reached, or ended without answering."""


class Client(Protocol):
    def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """The result of ``method`` asked with ``params``; raises
        :class:`Failed` for an error answer.
        """
        ...


def _checked(method: str, params: dict[str, Any]) -> None:
    """Raise :class:`Failed`, the caller's mistake (``-32602``), for params
    that are no JSON, a number that is not finite among them.
    """
    request = {"jsonrpc": "2.0", "id": 0, "method": method, "params": params}
    try:
        protocol.encode(request)
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise Failed(
            {
                "code": protocol.INVALID_PARAMS,
                "message": f"The request is no JSON: {error}.",
            }
        ) from None


@contextlib.contextmanager
def _reaching() -> Iterator[None]:
    """A server's error answer as :class:`Failed`; a server out of reach, or
    gone, as :class:`Gone`.
    """
    try:
        yield
    except registry.RemoteError as error:
        found: dict[str, Any] = {"code": error.code, "message": error.message}
        if error.data is not None:
            found["data"] = error.data
        raise Failed(found) from None
    except (OSError, RuntimeError, ValueError) as error:
        # Gone, timed out, refused, no socket to be had, a server that did
        # not start, or a line that is no JSON.
        raise Gone(str(error)) from None


class InProcess:
    """A session held in this process, answered by the server's own
    dispatcher. ``deadline_s`` is read and checked as a server reads it,
    but no watchdog keeps it: nothing ends this process for a request.
    Whatever the scene prints goes to standard error, as a server's does.
    """

    def __init__(self) -> None:
        from ..server.dispatch import Dispatcher

        self._dispatcher = Dispatcher()

    def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        from ..server.dispatch import deadline, internal_error

        _checked(method, params)
        try:
            _, given = deadline(params)
            with contextlib.redirect_stdout(sys.stderr):
                found = self._dispatcher.handle(method, given)
        except RpcError as error:
            raise Failed(protocol.error(None, error)["error"]) from None
        try:
            # The answer as a server would send it.
            result: dict[str, Any] = json.loads(json.dumps(found, allow_nan=False))
        except (TypeError, ValueError, OverflowError, RecursionError) as error:
            raise Failed(protocol.error(None, internal_error(error))["error"]) from None
        return result


class Resident:
    """A session held by a server of the registry, over ``connection``."""

    def __init__(self, connection: registry.Connection) -> None:
        self._connection = connection

    def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        _checked(method, params)
        with _reaching():
            result = self._connection.call(method, **params)
        return result


@contextlib.contextmanager
def reached(
    file: Path,
    scene: str,
    config: dict[str, Any],
    *,
    once: bool,
    deadline_s: float,
) -> Iterator[tuple[Client, dict[str, Any]]]:
    """The session of ``scene`` in ``file`` under ``config``, opened, and
    what its ``open`` answered, each request by ``deadline_s``. With
    ``once``, it is held in this process and let go at the end; else by the
    registry's server holding it, started when none does, which goes on
    holding it.
    """
    named = str(file.resolve())
    opening: dict[str, Any] = {"file": named, "scene": scene, "config": config}
    _checked("open", {**opening, "deadline_s": deadline_s})
    if once:
        here = InProcess()
        try:
            yield here, here.call("open", {**opening, "deadline_s": deadline_s})
        finally:
            with contextlib.suppress(Failed):
                here.call("close", {"deadline_s": deadline_s})
        return
    with _reaching():
        connection = registry.connect_or_start(
            named, scene, config, deadline_s=deadline_s
        )
    with connection:
        yield Resident(connection), connection.opened
