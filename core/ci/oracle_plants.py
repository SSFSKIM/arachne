"""The oracle's plants, run: each check the oracle makes is removed in turn and the tests meant for
it must fail.

``python3 core/ci/oracle_plants.py [NAME ...]``   (from the repository root; no names: every plant)

Each plant in ``core/ci/oracle_plants.toml`` names a source file, a text found in it exactly once
and what replaces it (the check, removed), and the tests that must fail without the check, as
``TARGET::TEST`` (``TARGET`` a test file of ``core/oracle/tests``, or ``lib`` for the library's own
tests). For each plant the file is changed, the target's tests run, the file restored, and one line
printed; the last line is the total. It exits 1 if any plant's tests do not all fail (or its text
is not found exactly once, or the plant does not build), 2 if a source is not as it was when the
run began.

It builds the oracle once per plant: a development check, run on a build machine, not a CI gate.
The plants are never built into anything: this script writes each into the source and restores it.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
ORACLE = ROOT / "core" / "oracle"
PLANTS = Path(__file__).resolve().parent / "oracle_plants.toml"


def failing(target: str) -> tuple[set[str], str]:
    """The tests of ``target`` that fail, and the build's error where it does not build."""
    which = ["--lib"] if target == "lib" else ["--test", target]
    env = dict(os.environ)
    env.pop("CARGO_TARGET_DIR", None)
    run = subprocess.run(
        ["cargo", "test", "--release", "--locked", *which],
        cwd=ORACLE,
        env=env,
        capture_output=True,
        text=True,
    )
    if "could not compile" in run.stderr:
        errors = [line for line in run.stderr.splitlines() if line.startswith("error")]
        return set(), errors[0] if errors else "did not build"
    return set(re.findall(r"^test (\S+) \.\.\. FAILED$", run.stdout, re.M)), ""


def main(argv: list[str]) -> int:
    plants = tomllib.loads(PLANTS.read_text(encoding="utf-8"))["plant"]
    if argv:
        plants = [p for p in plants if p["name"] in argv]
    sources = {f: (ROOT / f).read_bytes() for f in {p["file"] for p in plants}}
    failed = 0
    for p in plants:
        path = ROOT / p["file"]
        text = path.read_text(encoding="utf-8")
        found = text.count(p["find"])
        if found != 1:
            failed += 1
            print(
                f"FAIL {p['name']} [{p['item']}] {p['check']}: its text occurs {found} times"
            )
            continue
        path.write_text(text.replace(p["find"], p["replace"]), encoding="utf-8")
        try:
            red: set[str] = set()
            broke = ""
            for target in sorted({t.split("::", 1)[0] for t in p["red"]}):
                tests, broke = failing(target)
                red |= {f"{target}::{t}" for t in tests}
                if broke:
                    break
        finally:
            path.write_text(text, encoding="utf-8")
        missing = [t for t in p["red"] if t not in red]
        if broke or missing:
            failed += 1
            why = broke or "still passing: " + ", ".join(missing)
            print(f"FAIL {p['name']} [{p['item']}] {p['check']}: {why}", flush=True)
        else:
            others = sorted(red - set(p["red"]))
            also = f" (also {len(others)} more)" if others else ""
            print(
                f"RED  {p['name']} [{p['item']}] {p['check']}: {', '.join(p['red'])}{also}",
                flush=True,
            )
    if any((ROOT / f).read_bytes() != b for f, b in sources.items()):
        print("plants: a source was not restored")
        return 2
    print(f"plants: {len(plants) - failed} of {len(plants)} RED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
