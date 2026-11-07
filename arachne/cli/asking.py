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
    "EXIT_SCENE_ERROR",
    "EXIT_STRICT",
    "ask",
    "printed",
    "seconds",
    "session_options",
]

import contextlib
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, NoReturn, TypeVar

import click
import cloup

from ..server import protocol
from . import client

EXIT_STRICT = 1
EXIT_SCENE_ERROR = 2
EXIT_FAILED = 3

F = TypeVar("F", bound=Callable[..., Any])


def _json_object(ctx: click.Context, param: click.Parameter, value: str | None) -> Any:
    if value is None:
        return {}
    try:
        found = json.loads(value)
    except ValueError as error:
        raise click.BadParameter(f"not JSON: {error}") from None
    if not isinstance(found, dict):
        raise click.BadParameter("a JSON object of configuration options")
    return found


def session_options(command: F) -> F:
    """The session's file and scene, the configuration it executes under,
    how it is reached, the deadline of each request, and ``--json``.
    """
    decorators = [
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
            default=protocol.DEADLINE_S,
            show_default=True,
            help="How long each request may run, in seconds; a server ends past it.",
        ),
        cloup.option(
            "--json", "as_json", is_flag=True, help="Print the answer as JSON."
        ),
    ]
    for decorator in reversed(decorators):
        command = decorator(command)
    return command


@contextlib.contextmanager
def _reached(
    file: str, scene: str, config: dict[str, Any], once: bool
) -> Iterator[client.Client]:
    if not once:
        with client.connect_or_start(file, scene, config) as remote:
            yield remote
        return
    here = client.InProcess()
    try:
        yield here
    finally:
        with contextlib.suppress(client.Failed):
            here.call("close", {})


def ask(
    file: Path,
    scene: str,
    config: dict[str, Any],
    *,
    once: bool,
    deadline_s: float,
    as_json: bool,
    method: str | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Open the session of ``scene`` in ``file`` under ``config``, then ask
    it ``method`` with ``params`` (given ones only; none: the open's
    answer): the answer. An error answer ends the command with its status,
    printed as ``as_json`` says.
    """
    try:
        with _reached(str(file.resolve()), scene, config, once) as session:
            answer = session.call(
                "open",
                {
                    "file": str(file.resolve()),
                    "scene": scene,
                    "config": config,
                    "deadline_s": deadline_s,
                },
            )
            if method is not None:
                given = {k: v for k, v in (params or {}).items() if v is not None}
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
        raise click.UsageError(error["message"])
    if as_json:
        click.echo(json.dumps(error))
    data = error.get("data")
    if code == protocol.SCENE_ERROR and isinstance(data, dict):
        click.echo(data["report"], err=True)
        raise click.exceptions.Exit(EXIT_SCENE_ERROR)
    click.echo(f"arachne: {error['message']}", err=True)
    if code == protocol.INTERNAL_ERROR and isinstance(data, dict):
        click.echo(data.get("traceback", ""), err=True, nl=False)
    raise click.exceptions.Exit(EXIT_FAILED)


def printed(
    answer: dict[str, Any], as_json: bool, formatted: Callable[[], str]
) -> None:
    """Print ``answer``: as JSON, else as ``formatted`` writes it."""
    click.echo(json.dumps(answer) if as_json else formatted())


def seconds(value: float | None) -> str:
    """``value`` seconds, briefly."""
    return "-" if value is None else f"{value:g}"
