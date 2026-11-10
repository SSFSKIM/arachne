"""The oracle's independence: it depends on the core's XF wire schema and nothing else.

``python3 ci/oracle_check.py [--oracle DIR] [--xf DIR] [--no-tree]``
``python3 ci/oracle_check.py --isolated-checkout ROOT``

Fails (exit 1, each problem on its own line) when:

- a source under the oracle's ``src/`` or ``tests/`` names a path of ``arachne_core`` that is not under
  ``arachne_core::xf``, or names the crate bare or renamed (``use arachne_core as …``), which would let
  a later path reach any module;
- the oracle's ``Cargo.toml`` renames the core (``package = "arachne-core"``);
- a ``#[path]`` attribute or an ``include!`` macro reaches outside the oracle's directory, or its file
  is not a plain relative literal (an escape, an absolute path, a macro);
- the core's ``src/xf/`` names another core module (``crate::`` outside ``crate::xf``, ``super::`` out
  of the module), the self-containment the oracle's independence rests on;
- the oracle's dependency tree holds a GPU crate, or kurbo built with ``std`` (``tree_check.py``'s
  rules; skipped with ``--no-tree``).

A lexical check cannot be closed against every spelling; it is a courtesy pass. The proof is
``ci/oracle_isolated.py``, which builds the oracle in a checkout holding no core source but the
schema. ``--isolated-checkout ROOT`` asserts that checkout holds exactly ``ISOLATED_CHECKOUT``: every
entry present, and no file outside it, ``.git`` included (the build's ``target/`` aside).

Prints what it allowed. Defaults: the oracle at ``core/oracle``, the schema at ``core/src/xf``.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
from pathlib import Path

CORE = Path(__file__).resolve().parent.parent

#: What the isolated build's checkout holds, and nothing else: the oracle, the XF schema, and the
#: manifest and toolchain the stand-in core needs, with the scripts that build it. The permission
#: list: a file the oracle could reach in the core is a file this list holds. Each entry is a file,
#: or a directory (ending ``/``) taken whole.
ISOLATED_CHECKOUT: tuple[str, ...] = (
    "core/Cargo.toml",
    "core/rust-toolchain.toml",
    "core/src/xf/",
    "core/tests/fixtures/xf15/",
    "core/oracle/",
    "core/ci/oracle_check.py",
    "core/ci/oracle_isolated.py",
    "core/ci/toolchain.sh",
)

#: Inside the list, the test files that may reach the core: they never enter the isolated checkout,
#: and run in the normal tree.
ISOLATED_EXCLUDED: tuple[str, ...] = (
    "core/oracle/tests/contract*.rs",
    "core/oracle/tests/common/fixtures.rs",
)

#: What a build adds to a checkout, never part of it (``core/src/lib.rs`` the build writes as
#: ``pub mod xf;``, over whatever stood there). A ``.git`` is not among them: the checkout holds
#: no repository a build could read the core's history from.
ISOLATED_IGNORED: tuple[str, ...] = ("core/oracle/target/", "core/src/lib.rs")


def isolated_listed(rel: str) -> bool:
    """Whether the file ``rel`` (POSIX, from the checkout's root) belongs in the isolated checkout."""
    if any(Path(rel).match(pattern) for pattern in ISOLATED_EXCLUDED):
        return False
    return any(
        rel == e or (e.endswith("/") and rel.startswith(e)) for e in ISOLATED_CHECKOUT
    )


def check_isolated_checkout(root: Path) -> tuple[list[str], list[str]]:
    """Problems with ``root`` as the isolated checkout, and the files it holds."""
    files = sorted(
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if (p.is_file() or p.is_symlink())
        and not any(
            p.relative_to(root).as_posix().startswith(i) for i in ISOLATED_IGNORED
        )
    )
    problems = [
        f"{rel}: outside the isolated checkout's list"
        for rel in files
        if not isolated_listed(rel)
    ]

    def held(e: str) -> bool:
        return any(rel == e or (e.endswith("/") and rel.startswith(e)) for rel in files)

    problems += [
        f"{e}: in the list, missing from the checkout"
        for e in ISOLATED_CHECKOUT
        if not held(e)
    ]
    return problems, files


#: Sources outside ``src/`` that may name core paths beyond ``xf``: contract tests, which hold the
#: core to the contract through its public interface.
CORE_PATHS_ALLOWED_IN: tuple[str, ...] = ("tests/contract_join_hypot.rs",)

CRATE = re.compile(r"\barachne_core\b")
INCLUDE = re.compile(r"\b(include|include_str|include_bytes)\s*!\s*\(\s*")
PATH_ATTR = re.compile(r"#\s*\[\s*path\s*=\s*")
STRING = re.compile(r'"((?:[^"\\]|\\.)*)"')
RAW = re.compile(r'b?r(#*)"')
CHAR = re.compile(r"'(?:\\.|[^\\'])'")


def blank(text: str) -> tuple[str, str]:
    """The source with its comments blanked, and with its comments and the contents of its string
    literals blanked. Both keep every offset and newline, so a match in one is a position in the
    other and in the source, and line numbers hold.
    """
    code, bare = [], []

    def keep(s: str) -> None:
        code.append(s)
        bare.append(s)

    def blank_all(s: str) -> None:
        b = "".join("\n" if ch == "\n" else " " for ch in s)
        code.append(b)
        bare.append(b)

    i, n, depth = 0, len(text), 0
    while i < n:
        if depth:
            j = i + 2 if text.startswith(("/*", "*/"), i) else i + 1
            depth += (
                1 if text.startswith("/*", i) else -1 if text.startswith("*/", i) else 0
            )
            blank_all(text[i:j])
            i = j
            continue
        word_before = i > 0 and (text[i - 1].isalnum() or text[i - 1] == "_")
        raw = None if word_before else RAW.match(text, i)
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            blank_all(text[i:j])
            i = j
        elif text.startswith("/*", i):
            depth = 1
            blank_all("/*")
            i += 2
        elif raw or text[i] == '"':
            if raw:
                close = '"' + raw.group(1)
                body = raw.end()
                k = text.find(close, body)
                j = n if k < 0 else k + len(close)
            else:
                m = STRING.match(text, i)
                body = i + 1
                j = m.end() if m else n
            code.append(text[i:j])
            inner = text[body : max(body, j - 1 - (len(raw.group(1)) if raw else 0))]
            bare.append(
                text[i:body]
                + "".join("\n" if ch == "\n" else " " for ch in inner)
                + text[body + len(inner) : j]
            )
            i = j
        elif text[i] == "'" and CHAR.match(text, i):
            j = CHAR.match(text, i).end()
            keep(text[i:j])
            i = j
        else:
            keep(text[i])
            i += 1
    return "".join(code), "".join(bare)


def line_of(text: str, at: int) -> int:
    return text.count("\n", 0, at) + 1


def group_items(text: str, start: int) -> tuple[list[str], int]:
    """The top-level items of the brace group opening at ``start``, and where it ends."""
    depth, item, items = 0, [], []
    i = start
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
            if depth == 1:
                i += 1
                continue
        elif c == "}":
            depth -= 1
            if depth == 0:
                items.append("".join(item).strip())
                return [it for it in items if it], i + 1
        elif c == "," and depth == 1:
            items.append("".join(item).strip())
            item = []
            i += 1
            continue
        item.append(c)
        i += 1
    return items, i


def core_paths(text: str) -> list[tuple[int, str, bool]]:
    """Every ``arachne_core`` path in the source: its line, as written, and whether it stays
    within ``arachne_core::xf``.
    """
    found = []
    for m in CRATE.finditer(text):
        rest = text[m.end() :]
        sep = re.match(r"\s*::\s*", rest)
        if not sep:
            found.append(
                (line_of(text, m.start()), "arachne_core (bare or renamed)", False)
            )
            continue
        after = m.end() + sep.end()
        if text.startswith("{", after):
            items, end = group_items(text, after)
            shown = text[m.start() : end]
            ok = bool(items) and all(re.match(r"xf\b", it) for it in items)
            found.append((line_of(text, m.start()), " ".join(shown.split()), ok))
            continue
        word = re.match(r"[A-Za-z_][A-Za-z0-9_]*|\*", text[after:])
        name = word.group(0) if word else ""
        found.append((line_of(text, m.start()), f"arachne_core::{name}", name == "xf"))
    return found


def outside_reaches(
    path: Path, text: str, bare: str, root: Path
) -> list[tuple[int, str]]:
    """Every ``#[path]`` or ``include!`` in the code (``bare``: no comments, no string contents) whose
    file, the literal at the same place in ``text``, lies outside ``root``.
    """
    bad = []
    for pattern in (PATH_ATTR, INCLUDE):
        for m in pattern.finditer(bare):
            lit = STRING.match(text, m.end())
            line = line_of(text, m.start())
            if not lit:
                bad.append((line, "a #[path] or include! whose file is not a literal"))
                continue
            if "\\" in lit.group(1):
                bad.append(
                    (line, "a #[path] or include! whose literal holds an escape")
                )
                continue
            if Path(lit.group(1)).is_absolute():
                bad.append((line, f"reaches {lit.group(1)!r}, an absolute path"))
                continue
            target = (path.parent / lit.group(1)).resolve()
            if not target.is_relative_to(root.resolve()):
                bad.append((line, f"reaches {lit.group(1)!r}, outside {root.name}/"))
    return bad


def rust_files(*dirs: Path) -> list[Path]:
    return sorted(p for d in dirs if d.is_dir() for p in d.rglob("*.rs"))


def check_oracle(oracle: Path) -> tuple[list[str], list[str]]:
    problems, allowed = [], []
    for path in rust_files(oracle / "src", oracle / "tests"):
        rel = path.relative_to(oracle).as_posix()
        text, bare = blank(path.read_text(encoding="utf-8"))
        names = core_paths(bare)
        for line, shown, ok in names:
            if not ok and rel not in CORE_PATHS_ALLOWED_IN:
                problems.append(
                    f"{rel}:{line}: names {shown}, outside arachne_core::xf"
                )
        if names and all(ok for _, _, ok in names):
            allowed.append(rel)
        for line, what in outside_reaches(path, text, bare, oracle):
            problems.append(f"{rel}:{line}: {what}")
    manifest = oracle / "Cargo.toml"
    if re.search(r'package\s*=\s*"arachne-core"', manifest.read_text(encoding="utf-8")):
        problems.append('Cargo.toml: the core is renamed (package = "arachne-core")')
    return problems, allowed


def check_xf(xf: Path) -> tuple[list[str], list[str]]:
    problems, files = [], []
    for path in rust_files(xf):
        rel = path.relative_to(xf).as_posix()
        files.append(rel)
        text, bare = blank(path.read_text(encoding="utf-8"))
        for m in re.finditer(r"\bcrate\s*::\s*", bare):
            nxt = re.match(r"[A-Za-z_][A-Za-z0-9_]*|\{|\*", bare[m.end() :])
            if not (nxt and nxt.group(0) == "xf"):
                problems.append(
                    f"src/xf/{rel}:{line_of(bare, m.start())}: names a core module outside xf"
                )
        for m in re.finditer(r"\bsuper\s*::", bare):
            # super:: from xf/mod.rs is the crate root; from a file under it, xf itself.
            escapes = path.name == "mod.rs" or re.match(
                r"\s*super\s*::", bare[m.end() :]
            )
            if escapes:
                problems.append(
                    f"src/xf/{rel}:{line_of(bare, m.start())}: super:: leaves the xf module"
                )
        for line, what in outside_reaches(path, text, bare, xf):
            problems.append(f"src/xf/{rel}:{line}: {what}")
    return problems, files


def check_tree(oracle: Path) -> list[str]:
    spec = importlib.util.spec_from_file_location(
        "tree_check", Path(__file__).resolve().parent / "tree_check.py"
    )
    tree_check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tree_check)
    here = os.getcwd()
    os.chdir(oracle)
    try:
        return tree_check.check_deps()
    finally:
        os.chdir(here)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--oracle", type=Path, default=CORE / "oracle")
    ap.add_argument("--xf", type=Path, default=CORE / "src" / "xf")
    ap.add_argument(
        "--no-tree", action="store_true", help="skip the dependency tree (no cargo)"
    )
    ap.add_argument(
        "--isolated-checkout",
        type=Path,
        metavar="ROOT",
        help="assert ROOT holds exactly the list",
    )
    args = ap.parse_args(argv)
    if args.isolated_checkout:
        problems, files = check_isolated_checkout(args.isolated_checkout)
        for p in problems:
            print(f"oracle: FAIL {p}")
        if problems:
            return 1
        print(
            f"oracle: the isolated checkout holds exactly its list ({len(files)} files)"
        )
        return 0
    problems, allowed = check_oracle(args.oracle)
    xf_problems, xf_files = check_xf(args.xf)
    problems += xf_problems
    if not args.no_tree:
        problems += [f"cargo tree: {p}" for p in check_tree(args.oracle)]
    for p in problems:
        print(f"oracle: FAIL {p}")
    if problems:
        return 1
    print(f"oracle: allowed arachne_core::xf ({', '.join(allowed)})")
    print(f"oracle: src/xf self-contained ({', '.join(xf_files)})")
    if not args.no_tree:
        print("oracle: no GPU crate in the tree; kurbo without std")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
