from __future__ import annotations

import click
import cloup

from arachne import __version__
from arachne._config import cli_ctx_settings, console
from arachne.cli.cache.commands import cache
from arachne.cli.cfg.group import cfg
from arachne.cli.checkhealth.commands import checkhealth
from arachne.cli.default_group import DefaultGroup
from arachne.cli.init.commands import init
from arachne.cli.plugins.commands import plugins
from arachne.cli.render.commands import render
from arachne.constants import EPILOG, FORKED_FROM


def show_splash(ctx: click.Context, param: click.Option, value: str | None) -> None:
    """When giving a value by console, show an initial message with the Manim
    version before executing any other command:
    ``Arachne vA.B.C (forked from Manim Community vX.Y.Z)``.

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
        console.print(
            f"Arachne [green]v{__version__}[/green] (forked from {FORKED_FROM})\n"
        )


def print_version_and_exit(
    ctx: click.Context, param: click.Option, value: str | None
) -> None:
    """Same as :func:`show_splash`, but also exit when giving a value by
    console.

    Parameters
    ----------
    ctx
        The Click context.
    param
        A Click option.
    value
        A string value given by console, or None.
    """
    show_splash(ctx, param, value)
    if value:
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
    pass


main.add_command(checkhealth)
main.add_command(cache)
main.add_command(cfg)
main.add_command(plugins)
main.add_command(init)
main.add_command(render)

if __name__ == "__main__":
    main()
