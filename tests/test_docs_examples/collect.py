"""Collect the documentation examples: every ``.. arachne::`` block that Sphinx renders.

Blocks come from the package's docstrings (every bare string statement, which covers
module, class, function and attribute docstrings) and from ``docs/source/**/*.rst``.
A ``.. arachne::`` line inside a literal block (after a paragraph ending in ``::`` or
inside a ``code``/``code-block`` directive) only shows the syntax and is not a block.

Each block becomes one :class:`DocsExample`. Names are the block's scene class name;
a class name used by more than one block gets ``-2``, ``-3``, ... in collection order.
Names never contain the package's import name, so the control data does not depend on it.
"""

from __future__ import annotations

import ast
import inspect
import re
from dataclasses import dataclass
from pathlib import Path

import arachne

PACKAGE_DIR = Path(arachne.__file__).parent
REPO_DIR = PACKAGE_DIR.parent
DOCS_DIR = REPO_DIR / "docs" / "source"

# Blocks that cannot render at all, by name, with the reason.
_NOT_EXPORTED = (
    "uses an OpenGL mobject class that `from arachne import *` does not export; its "
    "module is not in the rendered documentation, so the block has never run"
)
EXCLUDED: dict[str, str] = {
    "Example-2": _NOT_EXPORTED,
    "ShuffleSubmobjectsExample-2": _NOT_EXPORTED,
    "DotInterpolation": _NOT_EXPORTED,
    "ArcShapeIris": _NOT_EXPORTED,
    "AddToOpenGLVGroup": _NOT_EXPORTED,
    "DashedVMobjectExample": _NOT_EXPORTED,
    "ConvexHull3DExample": (
        "renders differently from run to run: ConvexHull3D orders its faces by "
        "iterating a set of objects, whose order follows their memory addresses"
    ),
    "LabeledCountryExample": (
        "downloads its outline from a CDN while rendering, so its frame follows that "
        "file and the network, not the engine"
    ),
}

# Directive syntax as docutils reads it: one optional space before the ``::``.
_DIRECTIVE = re.compile(r"^(?P<indent>\s*)\.\.[ ]+arachne[ ]?::[ ]+(?P<scene>\S+)\s*$")
_LITERAL_DIRECTIVE = re.compile(
    r"^\s*\.\.[ ]+(code|code-block|sourcecode|literalinclude|parsed-literal)[ ]?::"
)
_ANY_DIRECTIVE = re.compile(r"^\s*\.\.[ ]+[\w:+.-]+[ ]?::")


@dataclass(frozen=True)
class DocsExample:
    """One ``.. arachne::`` block."""

    name: str
    scene: str
    origin: str
    still: bool
    code: str


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _in_literal_block(lines: list[str], index: int) -> bool:
    """Whether ``lines[index]`` sits in a literal block, judged by its enclosing lines."""
    level = _indent(lines[index])
    for line in reversed(lines[:index]):
        if not line.strip() or _indent(line) >= level:
            continue
        if _LITERAL_DIRECTIVE.match(line):
            return True
        if not _ANY_DIRECTIVE.match(line) and line.rstrip().endswith("::"):
            return True
        level = _indent(line)
        if level == 0:
            return False
    return False


def _parse_block(lines: list[str], index: int) -> tuple[set[str], list[str]]:
    """The option names and the content lines of the directive at ``lines[index]``."""
    level = _indent(lines[index])
    body: list[str] = []
    for line in lines[index + 1 :]:
        if line.strip() and _indent(line) <= level:
            break
        body.append(line)
    options: set[str] = set()
    while body and body[0].strip().startswith(":"):
        options.add(body.pop(0).strip().split(":")[1])
    while body and not body[0].strip():
        body.pop(0)
    while body and not body[-1].strip():
        body.pop()
    content_indent = min((_indent(line) for line in body if line.strip()), default=0)
    content = [line[content_indent:] for line in body]
    if content and content[0].startswith(">>> "):
        # A block written as a doctest, as the directive reads it.
        content = [line[4:] for line in content if line.startswith((">>> ", "... "))]
    return options, content


def _blocks_in_text(text: str, origin: str) -> list[tuple[str, str, bool, str]]:
    lines = text.splitlines()
    found = []
    for index, line in enumerate(lines):
        match = _DIRECTIVE.match(line)
        if not match or _in_literal_block(lines, index):
            continue
        options, content = _parse_block(lines, index)
        found.append(
            (
                match["scene"],
                f"{origin}:{index + 1}",
                "save_last_frame" in options,
                "\n".join(content) + "\n",
            )
        )
    return found


def _docstrings(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        (node.lineno, inspect.cleandoc(node.value.value))
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    ]


def _raw_blocks() -> list[tuple[str, str, bool, str]]:
    raw = []
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        relative = path.relative_to(PACKAGE_DIR).as_posix()
        for lineno, doc in sorted(_docstrings(path)):
            raw.extend(_blocks_in_text(doc, f"{relative}@{lineno}"))
    for path in sorted(DOCS_DIR.rglob("*.rst")):
        relative = path.relative_to(REPO_DIR).as_posix()
        raw.extend(_blocks_in_text(path.read_text(encoding="utf-8"), relative))
    return raw


def collect() -> list[DocsExample]:
    """Every documentation example, in a stable order, uniquely named."""
    seen: dict[str, int] = {}
    examples = []
    for scene, origin, still, code in _raw_blocks():
        seen[scene] = seen.get(scene, 0) + 1
        name = scene if seen[scene] == 1 else f"{scene}-{seen[scene]}"
        examples.append(DocsExample(name, scene, origin, still, code))
    return examples


def sample_indices(frame_count: int) -> list[int]:
    """The frames an animated example compares: 4 spread evenly over it, then its last."""
    if frame_count < 1:
        raise ValueError("an animated example must write at least one frame")
    return [frame_count * k // 4 for k in range(4)] + [frame_count - 1]
