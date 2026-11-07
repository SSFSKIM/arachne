"""How the session commands reach a session: a resident one, held by a
server of the host's registry (:mod:`arachne.server.registry`), found there
or started; or, with ``--once``, one held in the command's own process,
answered by the same dispatcher a server answers with. Either way an
answer is the server's result, a JSON object; an error answer is raised as
:class:`Failed`, carrying the protocol's error object (a request that can't
be sent is the caller's mistake, ``-32602``), and a server that could not
be reached, or did not answer, as :class:`Gone`.
"""

from __future__ import annotations

__all__ = ["Client", "Failed", "Gone", "InProcess", "Resident", "reached"]

import contextlib
import functools
import json
import os
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn, Protocol

from ..server import protocol, registry
from ..server.protocol import RpcError

if TYPE_CHECKING:
    from ..server.watchdog import Ticket


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
    that are no JSON, a number that is not finite among them: what a
    registry's client refuses to send (:class:`~.registry.BadRequest`),
    refused here too.
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
    """What a registry's client raises: an error answer as :class:`Failed`;
    a request it can't send (its params, or a registry directory no server
    can serve in) as :class:`Failed`, the caller's mistake (``-32602``);
    and a server gone, silent past the deadline, never come to serve, or
    answering outside the protocol as :class:`Gone`.
    """
    try:
        yield
    except registry.RemoteError as error:
        found: dict[str, Any] = {"code": error.code, "message": error.message}
        if error.data is not None:
            found["data"] = error.data
        raise Failed(found) from None
    except registry.BadRequest as error:
        raise Failed({"code": protocol.INVALID_PARAMS, "message": str(error)}) from None
    except registry.ClientError as error:
        raise Gone(str(error)) from None


class InProcess:
    """A session held in this process, answered by the server's own
    dispatcher as a server answers: with the store of answers
    (``ARACHNE_STORE``), so that a render checks itself against the summary
    of the session's execution kept there, as a server's does, and each
    request kept to its deadline by the server's watchdog. A request past its deadline ends the process, as it
    ends a server: ``expired`` is given the deadline's error object, unless
    the answer had been claimed first; what a render running wrote is
    removed, and the process exits with status 3. Whatever the scene prints
    goes to standard error, as a server's does to its log.
    """

    def __init__(
        self, expired: Callable[[dict[str, Any]], object] = lambda error: None
    ) -> None:
        from ..server.dispatch import Dispatcher
        from ..server.store import Store
        from ..server.watchdog import Watchdog

        self._dispatcher = Dispatcher(Store())
        self._watchdog = Watchdog()
        self._expired = expired

    def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        from ..server.dispatch import deadline, internal_error

        _checked(method, params)
        try:
            given_s, given = deadline(params)
        except RpcError as error:
            raise Failed(protocol.error(None, error)["error"]) from None
        limit = protocol.DEADLINE_S if given_s is None else given_s
        ticket = self._watchdog.arm(
            time.monotonic() + limit,
            lambda respond: self._end(method, limit, respond),
        )
        # What the request publishes (a render's file) is published under the
        # claim of its answer, so a deadline that passes first leaves it unmade.
        self._dispatcher.publish = functools.partial(self._watchdog.claim, ticket)
        try:
            with contextlib.redirect_stdout(sys.stderr):
                found = self._dispatcher.handle(method, given)
            self._answering(ticket)
        except RpcError as error:
            self._answering(ticket)
            raise Failed(protocol.error(None, error)["error"]) from None
        finally:
            self._dispatcher.publish = None
            self._watchdog.end(ticket)
        try:
            # The answer as a server would send it.
            result: dict[str, Any] = json.loads(json.dumps(found, allow_nan=False))
        except (TypeError, ValueError, OverflowError, RecursionError) as error:
            raise Failed(protocol.error(None, internal_error(error))["error"]) from None
        return result

    def _answering(self, ticket: Ticket) -> None:
        """Claim ``ticket``'s answer, an error's as a result's: from here its
        deadline can't pass unanswered. A request that expired first answers
        nothing: the deadline's error is its answer, and the process is
        ending, so this waits for that end.
        """
        if not self._watchdog.answering(ticket):
            threading.Event().wait()

    def _end(self, method: str, limit: float, respond: bool) -> NoReturn:
        """End the process: ``method`` ran past its deadline, ``limit`` s.
        Each step is bounded as a server's end is.
        """
        from ..server.serve import _bounded, _flush

        try:
            if respond:
                error = {
                    "code": protocol.DEADLINE_EXCEEDED,
                    "message": f"{method} ran past its deadline of {limit:g} s.",
                    "data": {"method": method, "deadline_s": limit},
                }
                _bounded(lambda: self._expired(error))
            _bounded(self._dispatcher.abandon)
            _bounded(_flush)
        finally:
            os._exit(protocol.EXIT_DEADLINE)


class Resident:
    """A session held by a server of the registry, over ``connection``."""

    def __init__(self, connection: registry.Connection) -> None:
        self._connection = connection

    def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
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
    expired: Callable[[dict[str, Any]], object] = lambda error: None,
) -> Iterator[tuple[Client, dict[str, Any]]]:
    """The session of ``scene`` in ``file`` under ``config``, opened, and
    what its ``open`` answered, each request by ``deadline_s``. With
    ``once``, it is held in this process and let go at the end; else by the
    registry's server holding it, started when none does, which goes on
    holding it. ``expired`` is told of a request in this process past its
    deadline, as it ends the process (:class:`InProcess`).
    """
    named = str(file.resolve())
    opening: dict[str, Any] = {"file": named, "scene": scene, "config": config}
    if once:
        here = InProcess(expired)
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
