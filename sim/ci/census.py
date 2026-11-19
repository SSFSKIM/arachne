"""Nothing in the crate's tree picks its arithmetic from the CPU it runs on.

``census.py``
    Searches the compiled sources of every package of the shipped tree (this crate and its
    normal and build dependencies, on every target; not the probe's dev-dependencies) for
    run-time CPU feature detection, and fails unless each hit is on the reviewed list,
    ``ci/census.txt``, with the reason it cannot reach a result, and every entry of the list is
    still a hit. The sources searched are those Cargo compiles: the directory of each library
    target's root file, wherever it is (rapier's and parry's are outside their manifests'
    directories), and each build script. An entry names a line, or every hit of a file
    (``:*``), and the locked source it was reviewed in (``sha256:<checksum>`` from the lock
    file, ``git:<commit>``, or ``local``), so a changed source is reviewed again. Prints
    ``census: <n> hits in <m> files of <k> packages, all listed``.

Choices made when the crate is compiled (``cfg(target_feature)``) are not run-time detection:
the identity gate compares a baseline build with an x86-64-v3 one for those.

Run from ``sim/``.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DETECTION = re.compile(
    r"is_(x86|aarch64|arm|riscv|powerpc64|loongarch)_feature_detected!"
    r"|\bstd_detect\b|\bcpufeatures::new!|\b__cpuid(_count)?\b|\b_xgetbv\b"
    r"|\bgetauxval\b|\belf_aux_info\b|hw\.optional|\bIsProcessorFeaturePresent\b"
)
REVIEWED = Path(__file__).resolve().parent / "census.txt"
LOCK = Path(__file__).resolve().parents[1] / "Cargo.lock"
COMPILED = {"lib", "rlib", "dylib", "cdylib", "staticlib", "proc-macro", "custom-build"}


@dataclass(frozen=True)
class Hit:
    package: str
    path: str
    line: int
    text: str

    @property
    def key(self) -> str:
        return f"{self.package} {self.path}:{self.line}"


def detects(line: str) -> bool:
    return DETECTION.search(line) is not None


def parse_reviewed(text: str) -> dict[str, str]:
    """The reviewed list: ``<name> <version> <source> <path>:<line>  <reason>`` per line."""
    entries = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(None, 4)
        if len(fields) < 5:
            raise ValueError(f"census.txt: an entry without its reason: {line}")
        entries[" ".join(fields[:4])] = fields[4]
    return entries


def compare(hits: list[Hit], reviewed: dict[str, str]) -> tuple[list[Hit], list[str]]:
    """The hits the list lacks, and the entries that are no longer hits."""
    used: set[str] = set()
    unlisted = []
    for hit in hits:
        entries = {hit.key, f"{hit.package} {hit.path}:*"} & set(reviewed)
        used |= entries
        if not entries:
            unlisted.append(hit)
    return unlisted, sorted(set(reviewed) - used)


def shipped(metadata: dict[str, Any]) -> set[str]:
    """The package ids reached from the root through normal and build dependencies."""
    nodes = {node["id"]: node for node in metadata["resolve"]["nodes"]}
    seen: set[str] = set()
    stack = [metadata["resolve"]["root"]]
    while stack:
        package = stack.pop()
        if package in seen:
            continue
        seen.add(package)
        stack.extend(
            dep["pkg"]
            for dep in nodes[package]["deps"]
            if any(kind["kind"] != "dev" for kind in dep["dep_kinds"])
        )
    return seen


def lock_index(text: str) -> dict[tuple[str, str, str | None], str | None]:
    """Each locked package's checksum, by name, version and source."""
    return {
        (p["name"], p["version"], p.get("source")): p.get("checksum")
        for p in tomllib.loads(text).get("package", [])
    }


def source_id(
    index: dict[tuple[str, str, str | None], str | None],
    name: str,
    version: str,
    source: str | None,
) -> str:
    """What identifies a package's locked source: its checksum, its commit, or ``local``."""
    if source is None:
        return "local"
    if source.startswith("git+"):
        return "git:" + source.rsplit("#", 1)[1]
    checksum = index.get((name, version, source))
    if not checksum:
        sys.exit(f"census: no checksum in Cargo.lock for {name} {version} ({source})")
    return f"sha256:{checksum}"


def compiled_roots(package: dict[str, Any]) -> tuple[Path, list[Path]]:
    """What Cargo compiles of a package: the directory of each library target's root file, and
    each build script (its directory when it has one of its own). Paths are reported relative
    to the first path returned, the common ancestor of these and the manifest's directory.
    """
    manifest_dir = Path(os.path.normpath(Path(package["manifest_path"]).parent))
    roots: list[Path] = []
    for target in package["targets"]:
        if not COMPILED & set(target["kind"]):
            continue
        root = Path(os.path.normpath(target["src_path"]))
        if "custom-build" not in target["kind"] or root.parent != manifest_dir:
            root = root.parent
        if root not in roots:
            roots.append(root)
    base = Path(os.path.commonpath([manifest_dir, *roots]))
    return base, roots


def packages() -> list[tuple[str, Path, list[Path]]]:
    out = subprocess.run(
        ["cargo", "metadata", "--locked", "--format-version", "1"],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        sys.exit(f"cargo metadata failed:\n{out.stderr}")
    metadata = json.loads(out.stdout)
    index = lock_index(LOCK.read_text())
    wanted = shipped(metadata)
    found = []
    for p in metadata["packages"]:
        if p["id"] not in wanted:
            continue
        locked = source_id(index, p["name"], p["version"], p["source"])
        base, roots = compiled_roots(p)
        found.append((f"{p['name']} {p['version']} {locked}", base, roots))
    return found


def search(package: str, base: Path, roots: list[Path]) -> list[Hit]:
    files = sorted(
        {
            f
            for root in roots
            for f in ([root] if root.is_file() else root.rglob("*.rs"))
        }
    )
    hits = []
    for source in files:
        path = source.relative_to(base).as_posix()
        text = source.read_text(errors="replace")
        for number, line in enumerate(text.splitlines(), 1):
            if detects(line):
                hits.append(Hit(package, path, number, line.strip()))
    return hits


def main() -> None:
    if len(sys.argv) != 1:
        sys.exit(__doc__)
    found = packages()
    hits = [
        hit for package, base, roots in found for hit in search(package, base, roots)
    ]
    unlisted, stale = compare(hits, parse_reviewed(REVIEWED.read_text()))
    for hit in unlisted:
        print(
            f"FAIL: run-time detection not on the reviewed list: {hit.key}  {hit.text}"
        )
    for key in stale:
        print(f"FAIL: on the reviewed list but no longer in the tree: {key}")
    if unlisted or stale:
        sys.exit(1)
    files = len({(h.package, h.path) for h in hits})
    print(
        f"census: {len(hits)} hits in {files} files of {len(found)} packages, all listed"
    )


if __name__ == "__main__":
    main()
