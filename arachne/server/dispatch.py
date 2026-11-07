"""What each method of the session server does, read from its params and
answered as :mod:`arachne.api_json`'s forms.

:data:`METHODS` is the table the server dispatches by: a method's
:class:`Method` names what runs it, whether it needs an open session, and
whether the session is brought up to date with its files first, once its
params are read (:meth:`Params.done`). A method is added by adding its
entry.
"""

from __future__ import annotations

__all__ = ["METHODS", "Dispatcher", "Method", "Params", "deadline"]

import json
import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from .. import api_json
from ..session import Session
from .protocol import (
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
    NO_SESSION,
    OTHER_SESSION,
    REFUSED,
    SCENE_ERROR,
    RpcError,
)

_REQUIRED: Any = object()


class Params:
    """A request's params, read one by one. Every method reads them all
    before it does anything, then calls :meth:`done`, which refuses any it
    did not read, then runs ``then`` (the session's reload, for a method
    answered from the files as they are now).
    """

    def __init__(
        self, given: dict[str, Any], then: Callable[[], object] | None = None
    ) -> None:
        self._given = given
        self._read: set[str] = set()
        self._then = then

    def get(
        self, name: str, read: Callable[[str, Any], Any], default: Any = _REQUIRED
    ) -> Any:
        """Param ``name``, as ``read(name, value)`` reads it; ``default`` when
        not given (or given as null), and required when there is none.
        """
        self._read.add(name)
        value = self._given.get(name)
        if value is None:
            if default is _REQUIRED:
                raise _invalid(f"{name} is required.")
            return default
        return read(name, value)

    def done(self) -> None:
        unknown = sorted(set(self._given) - self._read)
        if unknown:
            raise _invalid(f"No param {', '.join(map(repr, unknown))}.")
        if self._then is not None:
            self._then()


def _invalid(message: str) -> RpcError:
    return RpcError(INVALID_PARAMS, message)


def text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise _invalid(f"{name} is a non-empty string.")
    return value


def flag(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise _invalid(f"{name} is true or false.")
    return value


def number(name: str, value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise _invalid(f"{name} is a number.")
    return value


def positive(name: str, value: Any) -> float:
    found = number(name, value)
    if found <= 0:
        raise _invalid(f"{name} is a number above 0.")
    return found


def instant(name: str, value: Any) -> Fraction | float:
    """A time: seconds, or the exact instant as a string, ``"n/d"``."""
    if isinstance(value, str):
        try:
            return Fraction(value)
        except ValueError:
            raise _invalid(f"{name} is seconds or an exact instant, 'n/d'.") from None
    return number(name, value)


def instants(name: str, value: Any) -> list[Fraction | float]:
    if not isinstance(value, list):
        raise _invalid(f"{name} is a list of times.")
    return [instant(name, each) for each in value]


def span(name: str, value: Any) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise _invalid(f"{name} is [start, end], in seconds.")
    start, end = (number(name, each) for each in value)
    return start, end


def array(name: str, value: Any) -> list[Any]:
    if not isinstance(value, list):
        raise _invalid(f"{name} is a list.")
    return value


def mapping(name: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _invalid(f"{name} is an object.")
    return value


def deadline(params: dict[str, Any]) -> tuple[float | None, dict[str, Any]]:
    """A request's ``deadline_s``, in seconds, and its other params."""
    rest = dict(params)
    given = rest.pop("deadline_s", None)
    return (None if given is None else positive("deadline_s", given)), rest


def _rules(name: str, value: Any) -> Any:
    """``"armed"``, or the rules a list names, each armed from the start."""
    from ..lint.departures import departures
    from ..lint.watch import hold_watch

    named = {"hold_watch": hold_watch, "departures": departures}
    if value == "armed":
        return value
    if not isinstance(value, list) or any(each not in named for each in value):
        raise _invalid(f"{name} is 'armed' or a list of {', '.join(map(repr, named))}.")
    return tuple(named[each] for each in value)


def _answers(name: str, value: Any) -> list[str]:
    asked = [value] if isinstance(value, str) else value
    if (
        not isinstance(asked, list)
        or not asked
        or any(each not in api_json.QUERY_ANSWERS for each in asked)
    ):
        raise _invalid(f"{name} is one or more of {', '.join(api_json.QUERY_ANSWERS)}.")
    return asked


def _ids(name: str, value: Any) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(e, str) for e in value):
        raise _invalid(f"{name} is a list of stable ids.")
    return value


@dataclass(frozen=True)
class Method:
    """A method: ``run(dispatcher, params)`` answers it. ``session``: it
    asks about the open session. ``fresh``: once its params are read, the
    session is executed again when its files changed (:meth:`Session.reload`).
    """

    run: Callable[[Dispatcher, Params], dict[str, Any]]
    session: bool = True
    fresh: bool = True


class Dispatcher:
    """The session a server holds, and what each request asks of it."""

    def __init__(self) -> None:
        self.session: Session | None = None
        #: What the open session was opened as: its file, scene and config.
        self._opened: tuple[str, str, str] | None = None
        #: Whether a request asked the server to end, once it answered.
        self.ending = False

    def handle(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """The result of ``method`` asked with ``params``; raises
        :class:`RpcError` for its error.
        """
        found = METHODS.get(method)
        if found is None:
            raise RpcError(METHOD_NOT_FOUND, f"No method {method!r}.")
        session = self.session
        if found.session and session is None:
            raise RpcError(NO_SESSION, "No session is open: open one first.")
        # The params read and checked, the session catches up with its files.
        then = session.reload if found.fresh and session is not None else None
        try:
            return found.run(self, Params(params, then))
        except RpcError:
            raise
        except Exception as error:
            raise _error_answer(error) from error

    def answer(self, **forms: Any) -> dict[str, Any]:
        """A result: ``forms``, with the generation of the session they are
        of (null with none open).
        """
        session = self.session
        return {"generation": None if session is None else session.generation, **forms}

    @property
    def opened(self) -> Session:
        assert self.session is not None
        return self.session


def _error_answer(error: Exception) -> RpcError:
    """``error``'s answer: a scene's, explained, or the engine's refusal."""
    explained = getattr(error, "scene_error", None)
    if explained is not None:
        message = (
            f"{explained.cls}: {explained.message}"
            if explained.message
            else explained.cls
        )
        return RpcError(SCENE_ERROR, message, api_json.scene_error(explained))
    return RpcError(
        REFUSED,
        f"{type(error).__name__}: {error}",
        {"type": type(error).__name__, "message": str(error)},
    )


# The methods


def _open(server: Dispatcher, params: Params) -> dict[str, Any]:
    file = params.get("file", text)
    scene = params.get("scene", text)
    config = params.get("config", mapping, {})
    params.done()
    opened = (os.path.realpath(file), scene, json.dumps(config, sort_keys=True))
    if server.session is None:
        server.session = Session.open(file, scene, config=config or None)
        server._opened = opened
    elif opened != server._opened:
        assert server._opened is not None
        held_file, held_scene, held_config = server._opened
        raise RpcError(
            OTHER_SESSION,
            f"This server holds {held_scene} of {held_file}, config"
            f" {held_config}: close it before opening another.",
        )
    else:
        server.opened.reload()
    return server.answer(record=api_json.inspect(server.opened))


def _inspect(server: Dispatcher, params: Params) -> dict[str, Any]:
    params.done()
    return server.answer(record=api_json.inspect(server.opened))


def _state_at(server: Dispatcher, params: Params) -> dict[str, Any]:
    t = params.get("t", instant)
    params.done()
    session = server.opened
    (found,) = session.answers([t], frames=False)
    return server.answer(state=api_json.state(found.state, session))


def _lint(server: Dispatcher, params: Params) -> dict[str, Any]:
    options: dict[str, Any] = {
        "fps": params.get("fps", positive, 4),
        "t_range": params.get("t_range", span, None),
        "exempt": params.get("exempt", array, ()),
        "at": params.get("at", _at, None),
        "rules": params.get("rules", _rules, "armed"),
        "sample_fps": params.get("sample_fps", positive, None),
    }
    out_dir = params.get("key_frames_dir", text, None)
    params.done()
    session = server.opened
    forms = {"lint_result": api_json.lint(session.lint(**options))}
    if out_dir is not None:
        forms["key_frames"] = api_json.key_frames(session.key_frames(out_dir))
    return server.answer(**forms)


def _at(name: str, value: Any) -> str | list[Fraction | float]:
    return value if isinstance(value, str) else instants(name, value)


def _query(server: Dispatcher, params: Params) -> dict[str, Any]:
    t = params.get("t", instant)
    what = params.get("what", _answers)
    ids = params.get("ids", _ids, [])
    params.done()
    found = server.opened.query(t)
    return server.answer(query=api_json.query(found, what, ids=ids))


def _frame(server: Dispatcher, params: Params) -> dict[str, Any]:
    t = params.get("t", instant)
    path = Path(params.get("path", text))
    quality = params.get("quality", text, None)
    params.done()
    (found,) = server.opened.answers([t], quality)
    assert found.frame is not None
    path.parent.mkdir(parents=True, exist_ok=True)
    found.frame.save(path, format="PNG")
    return server.answer(frame=api_json.frame(found, path))


def _key_frames(server: Dispatcher, params: Params) -> dict[str, Any]:
    out_dir = params.get("out_dir", text)
    quality = params.get("quality", text, None)
    params.done()
    found = server.opened.key_frames(out_dir, quality)
    return server.answer(key_frames=api_json.key_frames(found))


def _reload(server: Dispatcher, params: Params) -> dict[str, Any]:
    force = params.get("force", flag, False)
    params.done()
    reloaded = server.opened.reload(force)
    return server.answer(reloaded=reloaded)


def _close(server: Dispatcher, params: Params) -> dict[str, Any]:
    params.done()
    if server.session is not None:
        server.session.close()
        server.session = server._opened = None
    return server.answer()


def _ping(server: Dispatcher, params: Params) -> dict[str, Any]:
    params.done()
    return server.answer()


def _shutdown(server: Dispatcher, params: Params) -> dict[str, Any]:
    params.done()
    server.ending = True
    return server.answer()


#: The methods, by name.
METHODS: dict[str, Method] = {
    "open": Method(_open, session=False, fresh=False),
    "inspect": Method(_inspect),
    "state_at": Method(_state_at),
    "lint": Method(_lint),
    "query": Method(_query),
    "frame": Method(_frame),
    "key_frames": Method(_key_frames),
    "reload": Method(_reload, fresh=False),
    "close": Method(_close, session=False, fresh=False),
    "ping": Method(_ping, session=False, fresh=False),
    "shutdown": Method(_shutdown, session=False, fresh=False),
}
