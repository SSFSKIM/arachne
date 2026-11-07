"""``arachne session``: serve a session, render it or a clip of it, and
list and stop the session servers of this host's registry
(:mod:`arachne.server.registry`, ``ARACHNE_SESSION_DIR``).
"""

from __future__ import annotations

__all__ = ["session"]

import contextlib
import json
from collections.abc import Iterator
from dataclasses import asdict

import click
import cloup

from ...server import protocol, registry
from ..asking import EXIT_FAILED, RESIDENT, finite
from .render import clip, render


@cloup.group(epilog=RESIDENT)
def session() -> None:
    """Serve a scene's session, render it or a clip of it, and list and stop
    the servers running.
    """


session.add_command(render)
session.add_command(clip)


@session.command()
@cloup.option("--stdio", is_flag=True, help="On standard input and output.")
@cloup.option(
    "--socket",
    is_flag=True,
    help="On a Unix socket in ARACHNE_SESSION_DIR, else a private directory.",
)
@cloup.option(
    "--deadline-s",
    type=click.FloatRange(min=0, min_open=True, max=protocol.MAX_DEADLINE_S),
    default=None,
    callback=finite,
    help="How long a request that gives no deadline may run, in seconds:"
    f" else ARACHNE_SESSION_DEADLINE_S, else {protocol.DEADLINE_S:g}.",
)
def serve(stdio: bool, socket: bool, deadline_s: float | None) -> None:
    """Serve one session over JSON-RPC 2.0, one message a line, until it is
    shut down or idle: on --stdio or --socket.
    """
    from ...server.serve import main

    if stdio == socket:
        raise click.UsageError("Serve on exactly one of --stdio and --socket.")
    transport = "stdio" if stdio else "socket"

    main(
        [
            f"--{transport}",
            *([] if deadline_s is None else ["--deadline-s", repr(deadline_s)]),
        ]
    )


@contextlib.contextmanager
def _registry() -> Iterator[None]:
    """A registry no server can serve in (its directory too long for a
    socket, another user's, or not to be made) is a usage error, status 2;
    a server that could not be reached ends the command, status 3.
    """
    try:
        yield
    except registry.BadRequest as error:
        raise click.UsageError(str(error)) from None
    except registry.ClientError as error:
        click.echo(f"arachne: {error}", err=True)
        raise click.exceptions.Exit(EXIT_FAILED) from None


def _line(entry: registry.Entry) -> str:
    """A server's entry, for reading."""
    if entry.key is None:
        held = "no session"
    else:
        held = (
            f"{entry.key}  {entry.scene} of {entry.file}  generation {entry.generation}"
        )
    busy = "  busy" if entry.busy else ""
    return f"{held}  pid {entry.pid}  {entry.rss_mb:.0f} MB{busy}"


@session.command("list")
@cloup.option("--json", "as_json", is_flag=True, help="Print the servers as JSON.")
def list_(as_json: bool) -> None:
    """List the session servers running, each with the session it holds."""
    with _registry():
        found = registry.sessions()
    if as_json:
        click.echo(json.dumps([asdict(entry) for entry in found]))
        return
    if not found:
        click.echo("no sessions")
    for entry in found:
        click.echo(_line(entry))


@session.command()
@cloup.argument("target")
def stop(target: str) -> None:
    """Shut down the session server TARGET names: a session's key, or a
    server's pid, as `session list` shows them.
    """
    with _registry():
        ended = registry.stop(target)
    if not ended:
        click.echo(f"arachne: no session server {target}", err=True)
        raise click.exceptions.Exit(1)
    for entry in ended:
        click.echo(f"stopped {_line(entry)}")


@session.command("stop-all")
def stop_all() -> None:
    """Shut down every session server running."""
    with _registry():
        ended = registry.stop_all()
    for entry in ended:
        click.echo(f"stopped {_line(entry)}")
