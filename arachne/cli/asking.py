"""What the session commands (``inspect``, ``frame``, ``lint``) share: the
options that name a session and how it is reached, asking it, and turning
its answer into output and an exit status.

``--json`` prints the answer as the server gives it, an error's too;
without it, the command formats the answer for reading. The exit status:

- ``0``: answered (a lint's findings are an answer);
- ``1``: ``lint --strict``, and the lint did not pass;
- ``2``: the scene raised (its trimmed report on standard error), or the
  command line asked for what the session can't take;
- ``3``: the engine or the session failed: a refusal, a request past its
  deadline, a server that ended or failed unforeseen.
"""

from __future__ import annotations

__all__ = [
    "EXIT_FAILED",
    "QUALITY_FLAGS",
    "RENDER_DEADLINE_S",
    "RESIDENT",
    "EXIT_SCENE_ERROR",
    "EXIT_STRICT",
    "ask",
    "finite",
    "instant",
    "json_value",
    "printed",
    "render_options",
    "seconds",
    "session_options",
]

import json
import math
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn, TypeVar

import click
import cloup

from ..constants import QUALITIES
from ..server import protocol
from . import client

EXIT_STRICT = 1
EXIT_SCENE_ERROR = 2
EXIT_FAILED = 3

#: How long each request of a render or a clip may run by default, in
#: seconds: a long render's, not a lint's.
RENDER_DEADLINE_S = 6 * 3600.0

#: The quality keys, by their flag, as ``arachne render -q`` names them.
QUALITY_FLAGS: dict[str, str] = {
    flag: name for name, q in QUALITIES.items() if (flag := q["flag"]) is not None
}

#: Where a session asked of a server lives, for the commands' help.
RESIDENT = (
    "A session is held by a server of this host's registry"
    " (ARACHNE_SESSION_DIR), started for it when none holds it, which"
    " outlives the command: the next command for the same file, scene,"
    " config and working directory is answered by it. It ends once idle"
    " (ARACHNE_SESSION_IDLE_S, 600 s by default), or at"
    " 'arachne session stop-all'. What the scene prints, and the engine's"
    " logging, go to the server's log, <pid>.log in the registry. --once"
    " holds the session in this process instead, and prints there."
)

F = TypeVar("F", bound=Callable[..., Any])


def finite(
    ctx: click.Context, param: click.Parameter, value: float | None
) -> float | None:
    """A number option's ``value``, refused unless finite."""
    if value is not None and not math.isfinite(value):
        raise click.BadParameter(f"{value!r} is not a finite number.")
    return value


def json_value(text: str) -> Any:
    """``text`` read as JSON, each number finite (``NaN``, ``Infinity`` and a
    number too large for a float are not); raises :class:`click.BadParameter`.
    """

    def constant(name: str) -> Any:
        raise ValueError(f"{name} is not a finite number")

    def number(written: str) -> float:
        found = float(written)
        if not math.isfinite(found):
            raise ValueError(f"{written} is not a finite number")
        return found

    try:
        return json.loads(text, parse_constant=constant, parse_float=number)
    except ValueError as error:
        raise click.BadParameter(f"not JSON: {error}") from None


def instant(ctx: click.Context, param: click.Parameter, value: str) -> float | str:
    """Seconds, as a finite number, else the text as given: an exact
    instant, ``"n/d"``, or what the session refuses as no time.
    """
    try:
        found = float(value)
    except ValueError:
        return value
    if not math.isfinite(found):
        raise click.BadParameter(f"{value!r} is not a finite time.")
    return found


def _json_object(ctx: click.Context, param: click.Parameter, value: str | None) -> Any:
    if value is None:
        return {}
    found = json_value(value)
    if not isinstance(found, dict):
        raise click.BadParameter("a JSON object of configuration options")
    return found


def _options(deadline_s: float) -> Callable[[F], F]:
    """The session's file and scene, the configuration it executes under,
    how it is reached, the deadline of each request (``deadline_s`` by
    default), and ``--json``.
    """

    def decorated(command: F) -> F:
        for decorator in reversed(_decorators(deadline_s)):
            command = decorator(command)
        return command

    return decorated


def _decorators(deadline_s: float) -> list[Callable[[Any], Any]]:
    return [
        cloup.argument(
            "file",
            type=click.Path(exists=True, dir_okay=False, path_type=Path),
        ),
        cloup.argument("scene"),
        cloup.option(
            "--config",
            "config",
            metavar="JSON",
            callback=_json_object,
            help="Configuration options the scene executes under, a JSON object"
            ' (for example \'{"pixel_width": 854, "pixel_height": 480}\').',
        ),
        cloup.option(
            "--once",
            is_flag=True,
            help="Execute the scene in this process, with no session server.",
        ),
        cloup.option(
            "--deadline-s",
            type=click.FloatRange(min=0, min_open=True, max=protocol.MAX_DEADLINE_S),
            default=deadline_s,
            show_default=True,
            callback=finite,
            help="How long each request may run, in seconds; a server ends past it.",
        ),
        cloup.option(
            "--json", "as_json", is_flag=True, help="Print the answer as JSON."
        ),
    ]


#: The options of a command asking a session one request.
session_options = _options(protocol.DEADLINE_S)
#: The same, for a render or a clip: a long deadline by default.
render_options = _options(RENDER_DEADLINE_S)


def ask(
    file: Path,
    scene: str,
    config: dict[str, Any],
    *,
    once: bool,
    deadline_s: float,
    as_json: bool,
    method: str | None = None,
    params: dict[str, Any] | Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Open the session of ``scene`` in ``file`` under ``config``, then ask
    it ``method`` with ``params`` (given ones only; or as a function of the
    open's answer makes them; none: the open's answer): the answer. An
    error answer ends the command with its status, printed as ``as_json``
    says.
    """
    # A request past its deadline in this process ends it, from another
    # thread, while the scene's prints go to standard error: its error is
    # printed where this command prints.
    stdout = sys.stdout

    def expired(error: dict[str, Any]) -> None:
        _report(error, as_json, stdout)

    try:
        with client.reached(
            file, scene, config, once=once, deadline_s=deadline_s, expired=expired
        ) as (session, answer):
            if method is not None:
                asked = params(answer) if callable(params) else params or {}
                given = {k: v for k, v in asked.items() if v is not None}
                answer = session.call(method, {**given, "deadline_s": deadline_s})
    except client.Failed as failed:
        _failed(failed, as_json)
    except client.Gone as gone:
        click.echo(f"arachne: {gone}", err=True)
        raise click.exceptions.Exit(EXIT_FAILED) from None
    return answer


def _failed(failed: client.Failed, as_json: bool) -> NoReturn:
    error, code = failed.error, failed.code
    if code == protocol.INVALID_PARAMS:
        if as_json:
            click.echo(json.dumps(error))
        raise click.UsageError(error["message"])
    _report(error, as_json)
    data = error.get("data")
    if code == protocol.SCENE_ERROR and isinstance(data, dict):
        raise click.exceptions.Exit(EXIT_SCENE_ERROR)
    raise click.exceptions.Exit(EXIT_FAILED)


def _report(error: dict[str, Any], as_json: bool, stdout: Any = None) -> None:
    """Print the error object ``error``: as JSON on ``stdout`` (else the
    standard output) with ``as_json``, and for reading on standard error,
    a scene's error as its trimmed report.
    """
    if as_json:
        click.echo(json.dumps(error), file=stdout)
    data = error.get("data")
    if error["code"] == protocol.SCENE_ERROR and isinstance(data, dict):
        click.echo(data["report"], err=True)
        return
    click.echo(f"arachne: {error['message']}", err=True)
    if error["code"] == protocol.INTERNAL_ERROR and isinstance(data, dict):
        click.echo(data.get("traceback", ""), err=True, nl=False)


def printed(
    answer: dict[str, Any], as_json: bool, formatted: Callable[[], str]
) -> None:
    """Print ``answer``: as JSON, else as ``formatted`` writes it."""
    click.echo(json.dumps(answer) if as_json else formatted())


def seconds(value: float | None) -> str:
    """``value`` seconds, briefly."""
    return "-" if value is None else f"{value:g}"
