"""The oracle's independence, proven by absence: the oracle builds and passes its tests in a checkout
that holds no core source but the XF schema, so there is nothing else for it to reach.

``python3 ci/oracle_isolated.py --tree ROOT``
``python3 ci/oracle_isolated.py --from-git REV``
``python3 ci/oracle_isolated.py --prune ROOT``

``--tree ROOT`` takes a checkout that holds exactly ``oracle_check.ISOLATED_CHECKOUT`` and no
``.git`` (a directory ``git archive`` filled with those paths alone, or a checkout ``--prune``
emptied of the rest). ``--from-git REV`` makes one in a fresh temporary directory, by
``git archive`` of the list at ``REV`` from this repository: the local run. ``--prune ROOT``, the
CI gate's first step, deletes from a GitHub-hosted job's own checkout every file the list does not
hold, ``.git`` included, prints what is left and passes only if that is the list exactly.
Then, in ``ROOT``:

1. ``ROOT/core/ci/oracle_check.py --isolated-checkout ROOT`` asserts the checkout holds exactly the
   list, before anything is built;
2. ``core/src/lib.rs`` is written as ``pub mod xf;``, so the core's own manifest builds a stand-in
   ``arachne-core`` of the schema alone;
3. ``cargo test --release --locked`` runs in ``core/oracle``;
4. each plant reaches for a core file the full checkout would hold at ``ROOT`` — a ``#[path]`` and an
   ``include_str!`` with escaped literals, an absolute ``include_str!``, a ``cfg_attr(path)`` and an
   ``include_str!`` renamed by ``use`` — and passes only if the library then fails to build because
   the file is not there.

The proof is only as strong as the checkout: where the full tree exists on the same machine (the
local ``--from-git`` run beside a full checkout), an absolute reach into it still builds. The CI
gate ``core-oracle-isolated`` proves it in a job whose checkout, pruned, holds the list alone.
"""

from __future__ import annotations

import argparse
import io
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent

#: The messages the compiler gives for a file that is not there.
ABSENT = ("No such file or directory", "couldn't find file", "couldn't read")


def plants(root: Path) -> dict[str, str]:
    """Each plant's Rust, appended to the oracle's ``lib.rs``: a reach into the core files a full
    checkout at ``root`` holds.
    """
    core = (root / "core" / "src").resolve().as_posix()
    return {
        "escaped-path": '#[path = "\\x2e\\x2e/\\x2e\\x2e/src/composite.rs"]\nmod reached;\n',
        "escaped-include": 'pub const REACHED: &str = include_str!("\\x2e\\x2e/\\x2e\\x2e/src/path.rs");\n',
        "absolute-include": f'pub const REACHED: &str = include_str!("{core}/path.rs");\n',
        "cfg-attr-path": f'#[cfg_attr(all(), path = "{core}/composite.rs")]\nmod reached;\n',
        "aliased-include": f'use std::include_str as read_core;\npub const REACHED: &str = read_core!("{core}/path.rs");\n',
    }


def oracle_check():
    """The list's module, imported without writing bytecode into the checkout it describes."""
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import oracle_check

    return oracle_check


def pathspecs() -> list[str]:
    check = oracle_check()
    specs = [e.rstrip("/") for e in check.ISOLATED_CHECKOUT]
    specs += [f":(exclude,glob){pattern}" for pattern in check.ISOLATED_EXCLUDED]
    return specs


def prune(root: Path) -> int:
    """Deletes from ``root`` every file the list does not hold, ``.git`` first, and prints what is
    left; passes only if what is left is the list exactly. It runs before anything is built.
    """
    if os.environ.get("RUNNER_ENVIRONMENT") != "github-hosted":
        print(
            "isolated: --prune deletes the checkout it is given; it runs only in a GitHub-hosted job",
            flush=True,
        )
        return 2
    check = oracle_check()
    git = root / ".git"
    if git.is_dir() and not git.is_symlink():
        shutil.rmtree(git)
    elif git.exists() or git.is_symlink():
        git.unlink()
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        here = Path(dirpath)
        for name in filenames:
            p = here / name
            if not check.isolated_listed(p.relative_to(root).as_posix()):
                p.unlink()
        for name in dirnames:
            p = here / name
            if p.is_symlink():
                if not check.isolated_listed(p.relative_to(root).as_posix()):
                    p.unlink()
            elif not any(p.iterdir()):
                p.rmdir()
    left = sorted(
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() or p.is_symlink()
    )
    print(f"isolated: the checkout after pruning holds {len(left)} files:")
    for rel in left:
        print(f"  {rel}")
    problems, _ = check.check_isolated_checkout(root)
    problems += [
        f"{rel}: left by the pruning, outside the list"
        for rel in left
        if not check.isolated_listed(rel)
    ]
    for p in problems:
        print(f"isolated: FAIL {p}", flush=True)
    if problems:
        return 1
    print("isolated: what is left is the isolated checkout's list exactly", flush=True)
    return 0


def from_git(rev: str, root: Path) -> None:
    """Fills ``root`` with the list's paths at ``rev``, and nothing else."""
    tar = subprocess.run(
        ["git", "archive", "--format=tar", rev, "--", *pathspecs()],
        cwd=REPO,
        capture_output=True,
        check=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(tar)) as t:
        t.extractall(root, filter="data")


def cargo(root: Path, *args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.pop("CARGO_TARGET_DIR", None)
    return subprocess.run(
        ["cargo", *args],
        cwd=root / "core" / "oracle",
        env=env,
        capture_output=True,
        text=True,
    )


def prove(root: Path) -> int:
    listed = subprocess.run(
        [
            sys.executable,
            "-I",
            str(root / "core" / "ci" / "oracle_check.py"),
            "--isolated-checkout",
            str(root),
        ],
        capture_output=True,
        text=True,
    )
    sys.stdout.write(listed.stdout)
    if listed.returncode != 0:
        print("isolated: FAIL the checkout does not hold exactly its list", flush=True)
        return 1
    (root / "core" / "src" / "lib.rs").write_text("pub mod xf;\n", encoding="utf-8")
    run = cargo(root, "test", "--release", "--locked")
    if run.returncode != 0:
        sys.stdout.write(run.stdout)
        sys.stderr.write(run.stderr)
        print(
            "isolated: FAIL the oracle does not build and pass with the XF schema alone",
            flush=True,
        )
        return 1
    passed = [
        line for line in run.stdout.splitlines() if line.startswith("test result:")
    ]
    print(
        f"isolated: the oracle builds and passes its tests with the XF schema alone ({len(passed)} targets)"
    )
    lib = root / "core" / "oracle" / "src" / "lib.rs"
    text = lib.read_text(encoding="utf-8")
    failed = 0
    for name, rust in plants(root).items():
        lib.write_text(text + rust, encoding="utf-8")
        try:
            run = cargo(root, "build", "--release", "--locked", "--lib")
        finally:
            lib.write_text(text, encoding="utf-8")
        errors = [line for line in run.stderr.splitlines() if line.startswith("error")]
        if run.returncode != 0 and any(a in run.stderr for a in ABSENT):
            print(
                f"isolated: the {name} plant does not build: the file is absent ({errors[0]})"
            )
        else:
            failed += 1
            why = (
                "it built"
                if run.returncode == 0
                else f"it failed otherwise ({errors[0] if errors else '?'})"
            )
            print(f"isolated: FAIL the {name} plant: {why}", flush=True)
    return 1 if failed else 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    where = ap.add_mutually_exclusive_group(required=True)
    where.add_argument(
        "--tree", type=Path, metavar="ROOT", help="a checkout holding exactly the list"
    )
    where.add_argument(
        "--from-git", metavar="REV", help="make one by git archive of the list at REV"
    )
    where.add_argument(
        "--prune",
        type=Path,
        metavar="ROOT",
        help="delete from a CI job's checkout all the list does not hold, .git included",
    )
    args = ap.parse_args(argv)
    if args.prune:
        return prune(args.prune.resolve())
    if args.tree:
        return prove(args.tree.resolve())
    root = Path(tempfile.mkdtemp(prefix="oracle-isolated-"))
    try:
        from_git(args.from_git, root)
        return prove(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
