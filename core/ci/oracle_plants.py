"""The oracle's plants, run: each check the oracle makes is removed in turn and the tests meant for
it must fail.

``python3 core/ci/oracle_plants.py [NAME ...]``   (from the repository root; no names: every plant)

Each plant in ``core/ci/oracle_plants.toml`` names a source file, a text found in it exactly once
and what replaces it (the check, removed), and the tests that must fail without the check, as
``TARGET::TEST`` (``TARGET`` a test file of ``core/oracle/tests``, or ``lib`` for the library's own
tests). For each plant the file is changed, the target's tests run, the file restored, and one line
printed; the last line is the total. A plant that cannot be applied (its file is gone, or its text
does not occur there exactly once) removes no check: it is printed ``STALE`` before any build and
fails the run as a plant whose tests pass does. It exits 1 if any plant is stale, does not build,
or leaves a test it names passing, 2 if a source is not as it was when the run began.

Before any plant, every target the applying plants name is run once on the unchanged tree. A test
that fails there would read as a kill under any plant, so the run is refused (exit 3, no plant
applied) unless each target builds, none of its tests fails, and every test a plant names passes.
A name given that matches no plant refuses the run too (exit 3).

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
PLANTS = Path(__file__).resolve().parent / "oracle_plants.toml"


def run_target(target: str, root: Path) -> tuple[set[str], set[str], str]:
    """The tests of ``target`` that pass and that fail, and the build's error where it does not
    build.
    """
    which = ["--lib"] if target == "lib" else ["--test", target]
    env = dict(os.environ)
    env.pop("CARGO_TARGET_DIR", None)
    run = subprocess.run(
        ["cargo", "test", "--release", "--locked", *which],
        cwd=root / "core" / "oracle",
        env=env,
        capture_output=True,
        text=True,
    )
    if "could not compile" in run.stderr:
        errors = [line for line in run.stderr.splitlines() if line.startswith("error")]
        return set(), set(), errors[0] if errors else "did not build"
    passed = set(re.findall(r"^test (\S+) \.\.\. ok$", run.stdout, re.M))
    failed = set(re.findall(r"^test (\S+) \.\.\. FAILED$", run.stdout, re.M))
    if run.returncode != 0 and not failed:
        return passed, failed, f"exit {run.returncode} with no test failing"
    return passed, failed, ""


def failing(target: str, root: Path) -> tuple[set[str], str]:
    """The tests of ``target`` that fail, and the build's error where it does not build."""
    _, failed, broke = run_target(target, root)
    return failed, broke


def baseline(plants: list[dict], root: Path) -> list[str]:
    """Why the unchanged tree cannot carry the plants: each target that does not build, a test of
    it that fails, or a named test that does not pass. Empty when every target passes.
    """
    named: dict[str, set[str]] = {}
    for p in plants:
        for t in p["red"]:
            target, test = t.split("::", 1)
            named.setdefault(target, set()).add(test)
    problems = []
    for target in sorted(named):
        passed, failed, broke = run_target(target, root)
        if broke:
            problems.append(f"{target}: {broke}")
        problems += [
            f"{target}::{t}: fails on the unchanged tree" for t in sorted(failed)
        ]
        problems += [
            f"{target}::{t}: named by a plant, does not pass on the unchanged tree"
            for t in sorted(named[target] - passed - failed)
        ]
        print(f"BASE {target}: {len(passed)} passed, {len(failed)} failed", flush=True)
    return problems


def not_applied(plant: dict, root: Path) -> str:
    """Why ``plant`` cannot be written into the tree at ``root``: its file is absent or its text does
    not occur there exactly once. Empty when it applies.
    """
    path = root / plant["file"]
    if not path.is_file():
        return f"{plant['file']} does not exist"
    found = path.read_text(encoding="utf-8").count(plant["find"])
    if found == 0:
        return f"its text is not in {plant['file']}"
    if found > 1:
        return f"its text occurs {found} times in {plant['file']}"
    return ""


def main(argv: list[str], plants_file: Path = PLANTS, root: Path = ROOT) -> int:
    plants = tomllib.loads(plants_file.read_text(encoding="utf-8"))["plant"]
    if argv:
        unknown = sorted(set(argv) - {p["name"] for p in plants})
        if unknown:
            print(f"plants: refused, no plant is named {', '.join(unknown)}")
            return 3
        plants = [p for p in plants if p["name"] in argv]
    # A plant whose text the tree no longer holds exactly once removes no check: it fails the run
    # as a plant left GREEN does, and is never skipped.
    stale = {p["name"]: why for p in plants if (why := not_applied(p, root))}
    for p in plants:
        if p["name"] in stale:
            print(
                f"STALE {p['name']} [{p['item']}] {p['check']}: {stale[p['name']]}",
                flush=True,
            )
    live = [p for p in plants if p["name"] not in stale]
    sources = {f: (root / f).read_bytes() for f in {p["file"] for p in live}}
    problems = baseline(live, root)
    if problems:
        for problem in problems:
            print(f"BASE {problem}", flush=True)
        print(
            "plants: refused, the unchanged tree does not pass every target they name"
        )
        return 3
    failed = len(stale)
    for p in live:
        path = root / p["file"]
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace(p["find"], p["replace"]), encoding="utf-8")
        try:
            red: set[str] = set()
            broke = ""
            for target in sorted({t.split("::", 1)[0] for t in p["red"]}):
                tests, broke = failing(target, root)
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
    if any((root / f).read_bytes() != b for f, b in sources.items()):
        print("plants: a source was not restored")
        return 2
    not_red = f", {len(stale)} not applied" if stale else ""
    print(f"plants: {len(plants) - failed} of {len(plants)} RED{not_red}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
