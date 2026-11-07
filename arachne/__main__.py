from __future__ import annotations

import click
import cloup

from arachne import __version__
from arachne._config import cli_ctx_settings, console
from arachne.cli.cache.commands import cache
from arachne.cli.cfg.group import cfg
from arachne.cli.checkhealth.commands import checkhealth
from arachne.cli.default_group import DefaultGroup
from arachne.cli.frame.commands import frame
from arachne.cli.init.commands import init
from arachne.cli.inspect.commands import inspect
from arachne.cli.lint.commands import lint
from arachne.cli.plugins.commands import plugins
from arachne.cli.render.commands import render
from arachne.cli.session.commands import session
from arachne.constants import EPILOG, FORKED_FROM

#: The commands whose standard output is their answer, which no splash
#: message comes before: the session's.
ANSWERING = ("frame", "inspect", "lint", "session")


def _print_splash() -> None:
    console.print(
        f"Arachne [green]v{__version__}[/green] (forked from {FORKED_FROM})\n"
    )


def show_splash(ctx: click.Context, param: click.Option, value: str | None) -> None:
    """When giving a value by console, show an initial message with the Manim
    version before executing any other command:
    ``Arachne vA.B.C (forked from Manim Community vX.Y.Z)``. It is shown
    once the command is known (:func:`main`), and not before the commands
    that answer on their standard output (:data:`ANSWERING`).

    Parameters
    ----------
    ctx
        The Click context.
    param
        A Click option.
    value
        A string value given by console, or None.
    """
    ctx.meta["arachne.splash"] = bool(value)


def print_version_and_exit(
    ctx: click.Context, param: click.Option, value: str | None
) -> None:
    """When giving a value by console, show the message :func:`show_splash`
    shows, then exit.

    Parameters
    ----------
    ctx
        The Click context.
    param
        A Click option.
    value
        A string value given by console, or None.
    """
    if value:
        _print_splash()
        ctx.exit()


@cloup.group(
    context_settings=cli_ctx_settings,
    cls=DefaultGroup,
    default="render",
    no_args_is_help=True,
    help="Animation engine for explanatory math videos.",
    epilog="See 'arachne <command>' to read about a specific subcommand.\n\n"
    "Note: the subcommand 'arachne render' is called if no other subcommand "
    "is specified. Run 'arachne render --help' if you would like to know what the "
    f"'-ql' or '-p' flags do, for example.\n\n{EPILOG}",
)
@cloup.option(
    "--version",
    is_flag=True,
    help="Show version and exit.",
    callback=print_version_and_exit,
    is_eager=True,
    expose_value=False,
)
@click.option(
    "--show-splash/--hide-splash",
    is_flag=True,
    default=True,
    help="Print splash message with version information.",
    callback=show_splash,
    is_eager=True,
    expose_value=False,
)
@cloup.pass_context
def main(ctx: click.Context) -> None:
    """The entry point for Manim.

    Parameters
    ----------
    ctx
        The Click context.
    """
    if ctx.meta.get("arachne.splash") and ctx.invoked_subcommand not in ANSWERING:
        _print_splash()


main.add_command(checkhealth)
main.add_command(cache)
main.add_command(cfg)
main.add_command(plugins)
main.add_command(init)
main.add_command(render)
main.add_command(inspect)
main.add_command(frame)
main.add_command(lint)
main.add_command(session)

if __name__ == "__main__":
    main()
