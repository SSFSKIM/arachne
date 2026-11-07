"""The session server's wire: JSON-RPC 2.0 messages, one per line, and the
error codes and exit statuses the server answers with.

The protocol's description is :mod:`arachne.server`'s.
"""

from __future__ import annotations

__all__ = [
    "DEADLINE_EXCEEDED",
    "DEADLINE_S",
    "DEADLINE_WRITE_S",
    "EXIT_DEADLINE",
    "INTERNAL_ERROR",
    "INVALID_PARAMS",
    "INVALID_REQUEST",
    "METHOD_NOT_FOUND",
    "NO_SESSION",
    "OTHER_SESSION",
    "PARSE_ERROR",
    "PROTOCOL_VERSION",
    "REFUSED",
    "SCENE_ERROR",
    "TRACEBACK_FRAMES",
    "Request",
    "RpcError",
    "encode",
    "error",
    "parse",
    "ready",
    "result",
]

import json
from dataclasses import dataclass, field
from typing import Any

#: The protocol's version, which the ready line names: a method or field
#: added is a minor version, one changed or removed a major one.
PROTOCOL_VERSION = "1.0"

# JSON-RPC 2.0's own codes.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

#: ``INTERNAL_ERROR``'s ``data`` is ``{type, message, traceback}``: what the
#: server raised unforeseen, its traceback trimmed to the last
#: :data:`TRACEBACK_FRAMES` frames.
TRACEBACK_FRAMES = 8

# The server's.
#: The scene's code raised, whatever it raised (an exit too): ``data`` is
#: the ``scene_error`` form.
SCENE_ERROR = -32000
#: The engine refused the request, by one of its refusals
#: (:data:`arachne.server.dispatch.REFUSALS`): ``data`` is ``{type, message,
#: reason}``, the refusal's class name, its message and its stable code, or
#: null.
REFUSED = -32001
#: The request ran past its deadline: ``data`` is ``{method, deadline_s}``.
#: The server exits (:data:`EXIT_DEADLINE`) once it is sent, or once
#: :data:`DEADLINE_WRITE_S` has passed trying to send it.
DEADLINE_EXCEEDED = -32002
#: The method asks about a session, and none is open.
NO_SESSION = -32003
#: ``open`` asked for another scene than the session the server holds.
OTHER_SESSION = -32004

#: The exit status of a server ended by a request past its deadline.
EXIT_DEADLINE = 3

#: How long a request may run, writing its answer included, in seconds,
#: when neither it (``deadline_s``) nor the server's start says otherwise.
DEADLINE_S = 600.0

#: How long the server tries to send a deadline's error before it exits
#: anyway, in seconds: a client that stopped reading can't hold it.
DEADLINE_WRITE_S = 1.0


class RpcError(Exception):
    """An error answer: its ``code``, ``message`` and ``data``; and, for a
    line that is no request, the id it was read with, if any.
    """

    def __init__(
        self, code: int, message: str, data: Any = None, *, request_id: Any = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data
        self.request_id = request_id


@dataclass(frozen=True)
class Request:
    """A request read off the wire. ``id`` is absent (``has_id`` false) for
    a notification, which is never answered.
    """

    method: str
    params: dict[str, Any] = field(default_factory=dict)
    id: Any = None
    has_id: bool = True


def parse(line: bytes) -> Request:
    """The request ``line`` holds; raises :class:`RpcError` for a line that
    is not one.
    """
    try:
        message = json.loads(line)
    except ValueError as found:
        raise RpcError(PARSE_ERROR, f"Not JSON: {found}") from None
    if not isinstance(message, dict):
        raise RpcError(
            INVALID_REQUEST, "A request is one JSON object; batches are not served."
        )
    has_id = "id" in message
    request_id = message.get("id")
    if isinstance(request_id, bool) or not isinstance(
        request_id, (str, int, float, type(None))
    ):
        raise RpcError(INVALID_REQUEST, "A request's id is a string or a number.")
    method = message.get("method")
    params = message.get("params", {})
    if message.get("jsonrpc") != "2.0" or not isinstance(method, str):
        raise RpcError(
            INVALID_REQUEST,
            'A request has "jsonrpc": "2.0" and a method.',
            request_id=request_id,
        )
    if not isinstance(params, dict):
        raise RpcError(
            INVALID_PARAMS,
            "A request's params are named: an object.",
            request_id=request_id,
        )
    return Request(method, params, request_id, has_id)


def result(request_id: Any, value: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": value}


def error(request_id: Any, found: RpcError) -> dict[str, Any]:
    body: dict[str, Any] = {"code": found.code, "message": found.message}
    if found.data is not None:
        body["data"] = found.data
    return {"jsonrpc": "2.0", "id": request_id, "error": body}


def ready(transport: str, pid: int, socket: str | None = None) -> dict[str, Any]:
    """The line a server writes once it serves: a notification, ``ready``."""
    params: dict[str, Any] = {
        "protocol": PROTOCOL_VERSION,
        "pid": pid,
        "transport": transport,
    }
    if socket is not None:
        params["socket"] = socket
    return {"jsonrpc": "2.0", "method": "ready", "params": params}


def encode(message: dict[str, Any]) -> bytes:
    """``message`` as one line of the wire."""
    return json.dumps(message, allow_nan=False, separators=(",", ":")).encode() + b"\n"
