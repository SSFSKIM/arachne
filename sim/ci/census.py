"""Nothing in the crate's tree picks its arithmetic from the CPU it runs on.

``census.py``
    Searches the sources of every package in the locked tree (every target, every kind; not a
    dependency's tests, benches or examples, which are never built here) for run-time CPU
    feature detection, and fails unless each hit is on the reviewed list,
    ``ci/census.txt``, with the reason it cannot reach a result, and every entry of the list is
    still a hit. A changed lock file moves the lines, so the list is reviewed again. Prints
    ``census: <n> hits, all listed``.

Choices made when the crate is compiled (``cfg(target_feature)``) are not run-time detection:
the identity gate compares a baseline build with an x86-64-v3 one for those.

Run from ``sim/``.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

DETECTION = re.compile(
    r"is_(x86|aarch64|arm|riscv|powerpc64|loongarch)_feature_detected!"
    r"|\bstd_detect\b|\bcpufeatures::new!|\b__cpuid(_count)?\b|\b_xgetbv\b"
    r"|\bgetauxval\b|\belf_aux_info\b|hw\.optional|\bIsProcessorFeaturePresent\b"
)
REVIEWED = Path(__file__).resolve().parent / "census.txt"


@dataclass(frozen=True)
class Hit:
    package: str
    path: str
    line: int
    text: str

    @property
    def key(self) -> str:
        return f"{self.package} {self.path}:{self.line}"


def built(path: str) -> bool:
    """Whether a dependency's source file is built for us: its tests, benches and examples are not."""
    return path.split("/", 1)[0] not in ("tests", "benches", "examples")


def detects(line: str) -> bool:
    return DETECTION.search(line) is not None


def parse_reviewed(text: str) -> dict[str, str]:
    """The reviewed list: ``<name> <version> <path>:<line>  <reason>`` per line."""
    entries = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(None, 3)
        if len(fields) < 4:
            raise ValueError(f"census.txt: an entry without its reason: {line}")
        entries[" ".join(fields[:3])] = fields[3]
    return entries


def compare(hits: list[Hit], reviewed: dict[str, str]) -> tuple[list[Hit], list[str]]:
    """The hits the list lacks, and the entries that are no longer hits."""
    keys = {hit.key for hit in hits}
    return [h for h in hits if h.key not in reviewed], sorted(set(reviewed) - keys)


def packages() -> list[tuple[str, Path]]:
    out = subprocess.run(
        ["cargo", "metadata", "--locked", "--format-version", "1"],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        sys.exit(f"cargo metadata failed:\n{out.stderr}")
    metadata = json.loads(out.stdout)
    members = set(metadata["workspace_members"])
    return [
        (f"{p['name']} {p['version']}", Path(p["manifest_path"]).parent)
        for p in metadata["packages"]
        if p["id"] not in members
    ]


def search(package: str, root: Path) -> list[Hit]:
    hits = []
    for source in sorted(root.rglob("*.rs")):
        path = source.relative_to(root).as_posix()
        if not built(path):
            continue
        for number, line in enumerate(
            source.read_text(errors="replace").splitlines(), 1
        ):
            if detects(line):
                hits.append(
                    Hit(
                        package,
                        source.relative_to(root).as_posix(),
                        number,
                        line.strip(),
                    )
                )
    return hits


def main() -> None:
    if len(sys.argv) != 1:
        sys.exit(__doc__)
    hits = [hit for package, root in packages() for hit in search(package, root)]
    unlisted, stale = compare(hits, parse_reviewed(REVIEWED.read_text()))
    for hit in unlisted:
        print(
            f"FAIL: run-time detection not on the reviewed list: {hit.key}  {hit.text}"
        )
    for key in stale:
        print(f"FAIL: on the reviewed list but no longer in the tree: {key}")
    if unlisted or stale:
        sys.exit(1)
    print(f"census: {len(hits)} hits, all listed")


if __name__ == "__main__":
    main()
