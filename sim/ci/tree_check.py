"""The simulation crate is CPU only, stands apart from the core, and keeps the solver's
determinism feature on.

``tree_check.py deps``
    Fails if the crate's dependency tree (every kind, every target) holds a GPU crate (the
    pattern is the core's, read from ``core/ci/tree_check.py``), if ``arachne-core`` is in it,
    or if ``simba`` (the solver's scalar maths) is built without ``libm_force``, which routes
    its transcendental functions to the pure-Rust ``libm``. Prints which of the solver's
    crates are built with their ``std`` feature.

Run from ``sim/``.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

CORE_CHECK = Path(__file__).resolve().parents[2] / "core" / "ci" / "tree_check.py"
SOLVER_CRATES = (
    "rapier2d",
    "rapier3d",
    "parry2d",
    "parry3d",
    "glamx",
    "glam",
    "simba",
    "nalgebra",
)


def core_gpu_pattern() -> re.Pattern[str]:
    spec = importlib.util.spec_from_file_location("core_tree_check", CORE_CHECK)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pattern: re.Pattern[str] = module.GPU_CRATES
    return pattern


def cargo(*args: str) -> str:
    out = subprocess.run(["cargo", *args], capture_output=True, text=True, check=False)
    if out.returncode != 0:
        sys.exit(f"cargo {' '.join(args)} failed:\n{out.stderr}")
    return out.stdout


def features_of(crate: str) -> str | None:
    """``cargo tree -e features -i <crate>`` output, or None when the crate is not in the tree."""
    out = subprocess.run(
        ["cargo", "tree", "--locked", "--target", "all", "-e", "features", "-i", crate],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        if "did not match any packages" in out.stderr:
            return None
        sys.exit(f"cargo tree -i {crate} failed:\n{out.stderr}")
    if not out.stdout.startswith(f"{crate} v"):
        sys.exit(
            f"cargo tree -i {crate} printed what this check does not read:\n{out.stdout}"
        )
    return out.stdout


def check_deps() -> list[str]:
    tree = cargo(
        "tree",
        "--locked",
        "--target",
        "all",
        "-e",
        "all",
        "--prefix",
        "none",
        "--format",
        "{p}",
    )
    crates = sorted({line.split()[0] for line in tree.splitlines() if line})
    print(f"deps: {len(crates)} crates: {' '.join(crates)}")
    gpu = core_gpu_pattern()
    problems = [f"a GPU crate in the tree: {c}" for c in crates if gpu.match(c)]
    if "arachne-core" in crates:
        problems.append("arachne-core is in the tree")

    simba = features_of("simba")
    if simba is None:
        problems.append("simba is not in the tree")
    elif 'simba feature "libm_force"' not in simba:
        problems.append("simba is built without libm_force (the platform's maths)")
    else:
        print("deps: simba is built with libm_force")

    with_std = [
        c for c in SOLVER_CRATES if f'{c} feature "std"' in (features_of(c) or "")
    ]
    print(f"deps: built with std: {' '.join(with_std) or 'none'}")
    return problems


def main() -> None:
    if sys.argv[1:] != ["deps"]:
        sys.exit(__doc__)
    problems = check_deps()
    for problem in problems:
        print(f"FAIL: {problem}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
