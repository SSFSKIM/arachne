"""Every world of the two determinism scenes bakes the committed declaration
digest and the committed trace digest, compared apart (:mod:`.determinism`);
and the rule names a solver that computed other traces from equal
declarations apart from inputs that were computed otherwise.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("arachne_sim")

from . import determinism  # noqa: E402


@pytest.mark.parametrize("scene", determinism.SCENES)
def test_every_world_bakes_its_committed_declaration_and_trace(scene):
    case = determinism.bake(scene)
    found = determinism.problems(case)
    assert not found, determinism.report(case, found)


def _case(scene: str, **changed: tuple[str, str]) -> dict:
    """A baked scene as the committed digests say, with ``changed`` worlds'
    (declaration, trace) in their place.
    """
    worlds = []
    for (s, world), (declaration, trace) in determinism.COMMITTED.items():
        if s == scene:
            declaration, trace = changed.get(world, (declaration, trace))
            worlds.append({"world": world, "declaration": declaration, "trace": trace})
    return {"scene": scene, "worlds": worlds}


def _committed(scene: str, world: str) -> tuple[str, str]:
    return determinism.COMMITTED[(scene, world)]


def test_the_committed_digests_are_no_problem():
    for scene in determinism.SCENES:
        assert determinism.problems(_case(scene)) == []


def test_an_equal_declaration_with_another_trace_is_the_solver_s():
    declaration, _ = _committed("LiteralWorlds", "space")
    (found,) = determinism.problems(
        _case("LiteralWorlds", space=(declaration, "f" * 32))
    )
    assert found.startswith("LiteralWorlds space: SOLVER:")


def test_another_declaration_of_computed_inputs_is_an_input_mismatch():
    _, trace = _committed("ComputedWorlds", "open")
    (found,) = determinism.problems(_case("ComputedWorlds", open=("e" * 32, "f" * 32)))
    assert found.startswith("ComputedWorlds open: INPUT MISMATCH:")
    # The trace of other inputs is not compared, even when it is the committed one.
    (same_trace,) = determinism.problems(
        _case("ComputedWorlds", open=("e" * 32, trace))
    )
    assert same_trace.startswith("ComputedWorlds open: INPUT MISMATCH:")


def test_another_declaration_of_literal_inputs_is_named_as_such():
    (found,) = determinism.problems(_case("LiteralWorlds", plane=("e" * 32, "f" * 32)))
    assert found.startswith("LiteralWorlds plane: DECLARATION:")


def test_a_missing_world_is_a_problem():
    case = _case("ComputedWorlds")
    case["worlds"] = [w for w in case["worlds"] if w["world"] != "open"]
    (found,) = determinism.problems(case)
    assert "open" in found


def test_the_script_writes_the_scene_s_digests(tmp_path, monkeypatch):
    """The entry point a driver runs: one bake, its JSON written whole."""
    made = {
        "scene": "LiteralWorlds",
        "worlds": [{"world": "plane", "declaration": "a", "trace": "b"}],
    }
    monkeypatch.setattr(determinism, "bake", lambda scene: dict(made))
    out = tmp_path / "case.json"
    assert determinism.main(["LiteralWorlds", "--out", str(out)]) == 0
    written = json.loads(out.read_text())
    assert written["scene"] == "LiteralWorlds"
    assert written["worlds"][0]["declaration"] == "a"
    assert written["problems"]
    assert [p.name for p in tmp_path.iterdir()] == ["case.json"]
