"""The simulation crate is CPU only, stands apart from the core, and keeps the solver's
determinism feature on.

``tree_check.py deps``
    Fails if the crate's dependency tree (every kind, every target) holds a GPU crate (the
    pattern is the core's, read from ``core/ci/tree_check.py``), if ``arachne-core`` is in it,
    or if ``simba`` (the solver's scalar maths) is built without ``libm_force``, which routes
    its transcendental functions to the pure-Rust ``libm``. Prints which crates are built with
    their ``std`` feature.

Run from ``sim/``.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

CORE_CHECK = Path(__file__).resolve().parents[2] / "core" / "ci" / "tree_check.py"


def core_gpu_pattern() -> re.Pattern[str]:
    spec = importlib.util.spec_from_file_location("core_tree_check", CORE_CHECK)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pattern: re.Pattern[str] = module.GPU_CRATES
    return pattern


def features_by_package(listing: str) -> dict[str, set[str]]:
    """``cargo tree --format '{p}|{f}'`` output: each package (name and version) and its
    enabled features.
    """
    packages: dict[str, set[str]] = {}
    for line in listing.splitlines():
        if "|" not in line:
            continue
        package, features = line.split("|", 1)
        name, version = package.split()[:2]
        enabled = features.removesuffix(" (*)").strip()
        packages.setdefault(f"{name} {version}", set()).update(
            f for f in enabled.split(",") if f
        )
    return packages


def check_deps() -> list[str]:
    out = subprocess.run(
        [
            "cargo",
            "tree",
            "--locked",
            "--target",
            "all",
            "-e",
            "all",
            "--prefix",
            "none",
            "--format",
            "{p}|{f}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        sys.exit(f"cargo tree failed:\n{out.stderr}")
    packages = features_by_package(out.stdout)
    names = sorted({p.split()[0] for p in packages})
    print(f"deps: {len(names)} crates: {' '.join(names)}")
    gpu = core_gpu_pattern()
    problems = [f"a GPU crate in the tree: {n}" for n in names if gpu.match(n)]
    if "arachne-core" in names:
        problems.append("arachne-core is in the tree")

    simba = {p: f for p, f in packages.items() if p.split()[0] == "simba"}
    if not simba:
        problems.append("simba is not in the tree")
    for package, features in sorted(simba.items()):
        if "libm_force" in features:
            print(f"deps: {package} is built with libm_force")
        else:
            problems.append(
                f"{package} is built without libm_force (the platform's maths)"
            )

    with_std = sorted(p for p, f in packages.items() if "std" in f)
    print(f"deps: built with std: {', '.join(with_std) or 'none'}")
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
