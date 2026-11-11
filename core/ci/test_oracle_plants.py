"""The plant runner, and its anchor check, fail a plant that does not apply:
``python3 core/ci/test_oracle_plants.py``."""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oracle_plants  # noqa: E402

SOURCE = "fn check() -> bool {\n    true\n}\n"


def plant(name: str, find: str, file: str = "core/oracle/src/lib.rs") -> str:
    return (
        "[[plant]]\n"
        f'name = "{name}"\n'
        'item = "§0"\n'
        'check = "a check"\n'
        f'file = "{file}"\n'
        f'find = "{find}"\n'
        'replace = "false"\n'
        'red = ["lib::a_test"]\n'
    )


def run_plants(*plants: str, names: tuple[str, ...] = ()) -> tuple[int, str]:
    """The runner's exit and output over ``plants`` in a tree holding ``SOURCE`` alone, where any
    build fails the test: a plant that does not apply must be judged before anything builds.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "core" / "oracle" / "src").mkdir(parents=True)
        (root / "core" / "oracle" / "src" / "lib.rs").write_text(SOURCE)
        toml = root / "plants.toml"
        toml.write_text("\n".join(plants))
        out = io.StringIO()

        def no_build(target: str, root: Path) -> None:
            raise AssertionError(f"built {target} for a plant that does not apply")

        run_target, oracle_plants.run_target = oracle_plants.run_target, no_build
        try:
            with contextlib.redirect_stdout(out):
                code = oracle_plants.main(list(names), plants_file=toml, root=root)
        finally:
            oracle_plants.run_target = run_target
        assert (root / "core" / "oracle" / "src" / "lib.rs").read_text() == SOURCE
    return code, out.getvalue()


def test_a_plant_whose_text_is_gone_fails_the_run_by_name() -> None:
    code, out = run_plants(plant("gone", "    if checked {"))
    assert code == 1, out
    assert "STALE gone [§0] a check: its text is not in core/oracle/src/lib.rs" in out
    assert out.rstrip().endswith("plants: 0 of 1 RED, 1 not applied"), out


def test_a_plant_whose_text_occurs_twice_fails_the_run_by_name() -> None:
    code, out = run_plants(plant("twice", "e"))
    assert code == 1, out
    assert "STALE twice [§0] a check: its text occurs 2 times in" in out


def test_a_missing_file_fails_the_run_by_name() -> None:
    code, out = run_plants(plant("nowhere", "true", file="core/oracle/src/gone.rs"))
    assert code == 1, out
    assert "STALE nowhere [§0] a check: core/oracle/src/gone.rs does not exist" in out


def test_a_name_no_plant_has_applies_nothing() -> None:
    code, out = run_plants(plant("real", "true"), names=("unreal",))
    assert code == 3, out
    assert "plants: refused, no plant is named unreal" in out


def anchors(*plants: str) -> tuple[int, str]:
    """The anchor check's exit and output over ``plants``: it reads every plant's text and builds
    nothing.
    """
    return run_plants(*plants, names=("--anchors",))


def test_the_anchor_check_fails_a_plant_whose_text_is_gone_by_name() -> None:
    code, out = anchors(plant("real", "true"), plant("gone", "    if checked {"))
    assert code == 1, out
    assert "STALE gone [§0] a check: its text is not in core/oracle/src/lib.rs" in out
    assert "real" not in out, out
    assert out.rstrip().endswith("anchors: 1 of 2 plants do not apply"), out


def test_the_anchor_check_fails_a_plant_whose_text_occurs_twice_by_name() -> None:
    code, out = anchors(plant("real", "true"), plant("twice", "e"))
    assert code == 1, out
    assert "STALE twice [§0] a check: its text occurs 2 times in core/oracle/src/lib.rs" in out
    assert out.rstrip().endswith("anchors: 1 of 2 plants do not apply"), out


def test_the_anchor_check_passes_where_every_text_occurs_once() -> None:
    code, out = anchors(plant("real", "true"), plant("other", "fn check"))
    assert code == 0, out
    assert out.rstrip() == "anchors: each of 2 plants applies", out


if __name__ == "__main__":
    tests = [f for n, f in sorted(globals().items()) if n.startswith("test_")]
    for test in tests:
        test()
    print(f"plant runner: {len(tests)} tests passed")
