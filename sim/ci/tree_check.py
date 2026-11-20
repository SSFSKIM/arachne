"""The simulation crate is CPU only, stands apart from the core, and keeps the solver's
determinism feature on.

``tree_check.py deps``
    Fails if the crate's dependency tree (every kind, every target, with the Python binding's
    feature: the tree the shipped extension is built from) holds a GPU crate (the pattern is the
    core's, read from ``core/ci/tree_check.py``), if ``arachne-core`` is in it, if ``simba``
    (the solver's scalar maths) is built without ``libm_force``, which routes its
    transcendental functions to the pure-Rust ``libm``, or if one of the solver's vendored
    crates (``vendor/VENDORED.md``) resolves from anywhere but its directory under ``vendor/``,
    or the lock file holds a git source. Prints which crates are built with their ``std``
    feature.

Run from ``sim/``.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

SIM = Path(__file__).resolve().parents[1]
CORE_CHECK = SIM.parent / "core" / "ci" / "tree_check.py"
#: Each vendored crate and its directory under vendor/.
VENDORED = {
    "rapier2d": "rapier/crates/rapier2d",
    "rapier3d": "rapier/crates/rapier3d",
    "parry2d": "parry/crates/parry2d",
    "parry3d": "parry/crates/parry3d",
    "glamx": "glamx",
}


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
            "--features",
            "python",
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

    out = subprocess.run(
        [
            "cargo",
            "metadata",
            "--locked",
            "--format-version",
            "1",
            "--features",
            "python",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        sys.exit(f"cargo metadata failed:\n{out.stderr}")
    found = check_vendored(
        json.loads(out.stdout), (SIM / "Cargo.lock").read_text(), SIM / "vendor"
    )
    if not found:
        print(f"deps: {', '.join(VENDORED)} from vendor/ only; no git source locked")
    return problems + found


def check_vendored(metadata: dict[str, Any], lock: str, vendor: Path) -> list[str]:
    """Each vendored crate is one package, built from its directory under ``vendor``, and
    the lock file names no git source.
    """
    problems = []
    for name, directory in VENDORED.items():
        packages = [p for p in metadata["packages"] if p["name"] == name]
        expected = (vendor / directory / "Cargo.toml").resolve()
        if len(packages) != 1:
            problems.append(f"{name}: {len(packages)} packages in the tree, not 1")
            continue
        package = packages[0]
        if package["source"] is not None:
            problems.append(f"{name} resolves from {package['source']}, not vendor/")
        elif Path(package["manifest_path"]).resolve() != expected:
            problems.append(
                f"{name} is built from {package['manifest_path']}, not {expected}"
            )
    problems += [
        f"Cargo.lock holds a git source: {package['name']} {package['source']}"
        for package in tomllib.loads(lock).get("package", [])
        if package.get("source", "").startswith("git+")
    ]
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
