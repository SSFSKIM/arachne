"""``arachne session``: serve a session, and list and stop the session
servers running on this host.
"""

from __future__ import annotations

__all__ = ["session"]

import json

import click
import cloup

from ...server import protocol
from .. import client
from ..asking import finite


@cloup.group()
def session() -> None:
    """Serve a scene's session, and list and stop the servers running."""


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


@session.command("list")
@cloup.option("--json", "as_json", is_flag=True, help="Print the servers as JSON.")
def list_(as_json: bool) -> None:
    """List the session servers running."""
    found = client.sessions()
    if as_json:
        click.echo(json.dumps(found))
        return
    if not found:
        click.echo("no sessions")
    for entry in found:
        click.echo(
            "  ".join(
                f"{name} {value}" for name, value in entry.items() if value is not None
            )
        )


@session.command()
@cloup.argument("key")
def stop(key: str) -> None:
    """Shut down the session server KEY names, as `session list` shows it."""
    if not client.stop(key):
        click.echo(f"arachne: no session {key} answered", err=True)
        raise click.exceptions.Exit(1)


@session.command("stop-all")
def stop_all() -> None:
    """Shut down every session server running."""
    failed = [
        entry["key"]
        for entry in client.sessions()
        if not client.stop(str(entry["key"]))
    ]
    if failed:
        click.echo(f"arachne: no answer from {', '.join(map(str, failed))}", err=True)
        raise click.exceptions.Exit(1)
