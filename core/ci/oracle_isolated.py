"""The oracle's independence, proven by the compiler: the oracle builds and passes its tests in a tree
that holds no core source but the XF schema.

``python3 ci/oracle_isolated.py [--plant path|include] [--keep]``

It builds, in a fresh temporary directory:

- ``core/oracle/``: the oracle's manifest, lock, sources, tests and other files, less ``target/``
  and the test files that may reach the core (``tests/contract*.rs``, ``tests/common/fixtures.rs``),
  which run in the normal tree;
- ``core/``: a stand-in ``arachne-core`` package, the core's own manifest with a ``src/`` of exactly
  ``src/xf/**`` and a ``lib.rs`` of ``pub mod xf;``, and the core's toolchain and lint files;
- ``core/ci/``: the oracle's checker, which its tests run.

Then ``cargo test --release --locked`` there. Whatever reaches into the core — a ``#[path]`` or an
``include!`` to a core file however its literal is spelled, a ``arachne_core::`` path outside
``xf``, a path dependency, a build script that reads the core — has nothing to reach and fails to
compile. The ``--plant`` forms prove it: each plants an escaped reach into a core file in the
isolated copy, builds the library, and passes only if the build fails.

``CARGO_TARGET_DIR`` defaults to the oracle's ``target/isolated``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CORE = Path(__file__).resolve().parent.parent
ORACLE = CORE / "oracle"

#: Test files the contract permits core access in: run in the normal tree, never in the isolated one.
NORMAL_TREE_TESTS = ("tests/contract*.rs", "tests/common/fixtures.rs")

#: An escaped reach into a core file, per form: `..` spelt `\\x2e\\x2e`, which a lexical check reads
#: as other characters and the compiler reads as `..`.
PLANTS = {
    "path": '#[path = "\\x2e\\x2e/\\x2e\\x2e/src/composite.rs"]\nmod reached;\n',
    "include": 'pub const REACHED: &str = include_str!("\\x2e\\x2e/\\x2e\\x2e/src/path.rs");\n',
}


def excluded(rel: str) -> bool:
    return any(Path(rel).match(pattern) for pattern in NORMAL_TREE_TESTS)


def build_tree(root: Path) -> Path:
    """The isolated tree under ``root``; returns its oracle directory."""
    core = root / "core"
    (core / "src").mkdir(parents=True)
    shutil.copytree(CORE / "src" / "xf", core / "src" / "xf")
    (core / "src" / "lib.rs").write_text("pub mod xf;\n", encoding="utf-8")
    for name in ("Cargo.toml", "rust-toolchain.toml", "clippy.toml"):
        shutil.copy2(CORE / name, core / name)
    (core / "ci").mkdir()
    for name in ("oracle_check.py", "tree_check.py"):
        shutil.copy2(CORE / "ci" / name, core / "ci" / name)
    oracle = core / "oracle"
    for path in sorted(ORACLE.rglob("*")):
        rel = path.relative_to(ORACLE).as_posix()
        if rel.split("/")[0] == "target" or path.is_dir() or excluded(rel):
            continue
        dest = oracle / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    return oracle


def cargo(oracle: Path, *args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.setdefault("CARGO_TARGET_DIR", str(ORACLE / "target" / "isolated"))
    return subprocess.run(["cargo", *args], cwd=oracle, env=env, capture_output=True, text=True)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--plant", choices=sorted(PLANTS), help="plant an escaped reach; pass if the build fails")
    ap.add_argument("--keep", action="store_true", help="keep the isolated tree and print where it is")
    args = ap.parse_args(argv)
    root = Path(tempfile.mkdtemp(prefix="oracle-isolated-"))
    try:
        oracle = build_tree(root)
        if args.keep:
            print(f"isolated: tree at {root}")
        if args.plant:
            lib = oracle / "src" / "lib.rs"
            lib.write_text(lib.read_text(encoding="utf-8") + PLANTS[args.plant], encoding="utf-8")
            run = cargo(oracle, "build", "--release", "--locked", "--lib")
            if run.returncode == 0:
                print(f"isolated: FAIL the planted {args.plant} reach built", flush=True)
                return 1
            errors = [line for line in run.stderr.splitlines() if line.startswith("error")]
            print(f"isolated: the planted {args.plant} reach does not build ({errors[0] if errors else 'error'})")
            return 0
        run = cargo(oracle, "test", "--release", "--locked")
        sys.stdout.write(run.stdout)
        sys.stderr.write(run.stderr)
        if run.returncode != 0:
            print("isolated: FAIL the oracle does not build and pass with the XF schema alone", flush=True)
            return 1
        print("isolated: the oracle builds and passes its tests with the XF schema alone")
        return 0
    finally:
        if not args.keep:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
