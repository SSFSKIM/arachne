"""The oracle's independence, proven by absence: the oracle builds and passes its tests in a checkout
that holds no core source but the XF schema, so there is nothing else for it to reach.

``python3 ci/oracle_isolated.py --tree ROOT``
``python3 ci/oracle_isolated.py --from-git REV``

``--tree ROOT`` takes a checkout that holds exactly ``oracle_check.ISOLATED_CHECKOUT`` (a sparse
checkout, or a directory ``git archive`` filled with those paths alone). ``--from-git REV`` makes
one in a fresh temporary directory, by ``git archive`` of the list at ``REV`` from this repository.
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

The proof is only as strong as the checkout: where the full tree exists on the same machine (an
interim run from a full checkout), an absolute reach into it still builds. The CI job that proves it
runs in a checkout holding the list alone.
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


def pathspecs() -> list[str]:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from oracle_check import ISOLATED_CHECKOUT, ISOLATED_EXCLUDED

    specs = [e.rstrip("/") for e in ISOLATED_CHECKOUT]
    specs += [f":(exclude,glob){pattern}" for pattern in ISOLATED_EXCLUDED]
    return specs


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
    args = ap.parse_args(argv)
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
