"""Fails unless the GPU crate's Cargo.lock holds exactly one wgpu, at the version Cargo.toml pins.

Run from ``gpu/``.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path


def main() -> None:
    manifest = tomllib.loads(Path("Cargo.toml").read_text())
    pinned = manifest["dependencies"]["wgpu"]["version"]
    match = re.fullmatch(r"=(\d+\.\d+\.\d+)", pinned)
    if match is None:
        sys.exit(f"wgpu is not pinned to one version in Cargo.toml: {pinned!r}")
    lock = tomllib.loads(Path("Cargo.lock").read_text())
    versions = {}
    for package in lock["package"]:
        versions.setdefault(package["name"], []).append(package["version"])
    for name in ("wgpu", "wgpu-core", "wgpu-hal", "naga", "ash"):
        print(f"lock: {name} {' '.join(versions.get(name, ['absent']))}")
    if versions.get("wgpu") != [match.group(1)]:
        sys.exit(
            f"FAIL: the lock holds wgpu {versions.get('wgpu')}, not exactly {match.group(1)}"
        )


if __name__ == "__main__":
    main()
