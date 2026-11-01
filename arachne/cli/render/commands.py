"""Manim's default subcommand, render.

Manim's render subcommand is accessed in the command-line interface via
``arachne``, but can be more explicitly accessed with ``arachne render``. Here you
can specify options, and arguments for the render command.

"""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import cloup

from arachne._config import (
    config,
    error_console,
    tempconfig,
)
from arachne.cli.render.ease_of_access_options import ease_of_access_options
from arachne.cli.render.global_options import global_options
from arachne.cli.render.output_options import output_options
from arachne.cli.render.render_options import render_options
from arachne.constants import EPILOG
from arachne.manager import Manager
from arachne.utils.module_ops import scene_classes_from_file

__all__ = ["render"]


class ClickArgs(Namespace):
    def __init__(self, args: dict[str, Any]) -> None:
        for name in args:
            setattr(self, name, args[name])

    def _get_kwargs(self) -> list[tuple[str, Any]]:
        return list(self.__dict__.items())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ClickArgs):
            return NotImplemented
        return vars(self) == vars(other)

    def __contains__(self, key: str) -> bool:
        return key in self.__dict__

    def __repr__(self) -> str:
        return str(self.__dict__)


def _validate_scene_batch_output_name(scene_classes: list[type]) -> None:
    if config.output_file and (config.write_all or len(scene_classes) != 1):
        raise ValueError(
            "--output_file can only be used when rendering exactly one scene. "
            "Remove --write_all or select a single scene.",
        )


@cloup.command(
    context_settings=None,
    no_args_is_help=True,
    epilog=EPILOG,
)
@cloup.argument("file", type=cloup.Path(path_type=Path), required=True)
@cloup.argument("scene_names", required=False, nargs=-1)
@global_options
@output_options
@render_options
@ease_of_access_options
def render(**kwargs: Any) -> ClickArgs | dict[str, Any]:
    """Render SCENE(S) from the input FILE.

    FILE is the file path of the script or a config file.

    SCENES is an optional list of scenes in the file.
    """
    click_args = ClickArgs(kwargs)
    if kwargs["jupyter"]:
        return click_args

    config.digest_args(click_args)
    file = Path(config.input_file)
    try:
        scene_classes = scene_classes_from_file(file)
        _validate_scene_batch_output_name(scene_classes)

        for SceneClass in scene_classes:
            with tempconfig({}):
                scene = SceneClass()
                # Attach explicitly, but preserve custom Scene.render overrides.
                Manager(scene)
                scene.render()
    except Exception:
        error_console.print_exception()
        sys.exit(1)

    return kwargs
