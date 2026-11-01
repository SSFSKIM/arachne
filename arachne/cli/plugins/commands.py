"""Manim's plugin subcommand.

Manim's plugin subcommand is accessed in the command-line interface via ``arachne
plugin``. Here you can specify options, subcommands, and subgroups for the plugin
group.

"""

from __future__ import annotations

import cloup

from arachne.constants import CONTEXT_SETTINGS, EPILOG
from arachne.plugins.plugins_flags import list_plugins

__all__ = ["plugins"]


@cloup.command(
    context_settings=CONTEXT_SETTINGS,
    no_args_is_help=True,
    epilog=EPILOG,
    help="Manages Manim plugins.",
)
@cloup.option(
    "-l",
    "--list",
    "list_available",
    is_flag=True,
    help="List available plugins.",
)
def plugins(list_available: bool) -> None:
    """Print a list of all available plugins when calling ``arachne plugins -l``
    or ``arachne plugins --list``.

    Parameters
    ----------
    list_available
        If the ``-l`` or ``-list`` option is passed to ``arachne plugins``, this
        parameter will be set to ``True``, which will print a list of all
        available plugins.
    """
    if list_available:
        list_plugins()
