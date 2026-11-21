"""Every world of the two determinism scenes bakes the committed declaration
and motion digests, and on the committed build the committed trace digest,
compared apart (:mod:`.determinism`); and the rule names a solver that
computed other motion from equal declarations apart from an identity that
alone differs and from inputs that were computed otherwise.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("arachne_sim")

from . import determinism  # noqa: E402


@pytest.mark.parametrize("scene", determinism.SCENES)
def test_every_world_bakes_its_committed_digests(scene):
    case = determinism.bake(scene)
    found = determinism.problems(case)
    assert not found, determinism.report(case, found)


def _case(scene: str, **changed: dict) -> dict:
    """A baked scene as the committed digests say, on the committed build,
    with ``changed`` worlds' fields (declaration, motion, trace, build) in
    their place.
    """
    worlds = []
    for (s, world), (declaration, trace) in determinism.COMMITTED.items():
        if s == scene:
            baked = {
                "world": world,
                "declaration": declaration,
                "motion": determinism.MOTION[(s, world)],
                "trace": trace,
                "build": determinism.BUILD,
                "identity": "the identity",
            }
            worlds.append({**baked, **changed.get(world, {})})
    return {"scene": scene, "worlds": worlds}


OTHER_BUILD = "rustc 1.99.1 (58cf8f9c3 2026-10-13); aarch64-apple-darwin; neon"


def test_the_committed_digests_are_no_problem():
    for scene in determinism.SCENES:
        assert determinism.problems(_case(scene)) == []


def test_an_equal_declaration_with_other_motion_is_the_solver_s_on_any_build():
    for build in (determinism.BUILD, OTHER_BUILD):
        (found,) = determinism.problems(
            _case("LiteralWorlds", space={"motion": "f" * 32, "build": build})
        )
        assert found.startswith("LiteralWorlds space: MOTION:")


def test_equal_motion_with_another_trace_on_the_committed_build_is_the_identity_alone():
    (found,) = determinism.problems(_case("LiteralWorlds", plane={"trace": "f" * 32}))
    assert found.startswith("LiteralWorlds plane: IDENTITY:")
    assert "different motion" not in found


def test_on_another_build_the_trace_is_not_compared():
    changed = {"trace": "f" * 32, "build": OTHER_BUILD}
    assert determinism.problems(_case("ComputedWorlds", open=changed)) == []


def test_another_declaration_of_computed_inputs_is_an_input_mismatch():
    (found,) = determinism.problems(
        _case("ComputedWorlds", open={"declaration": "e" * 32, "motion": "f" * 32})
    )
    assert found.startswith("ComputedWorlds open: INPUT MISMATCH:")
    # The motion and trace of other inputs are not compared, even when committed.
    (same,) = determinism.problems(
        _case("ComputedWorlds", open={"declaration": "e" * 32})
    )
    assert same.startswith("ComputedWorlds open: INPUT MISMATCH:")


def test_another_declaration_of_literal_inputs_is_named_as_such():
    (found,) = determinism.problems(
        _case("LiteralWorlds", plane={"declaration": "e" * 32})
    )
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
        "worlds": [{"world": "plane", "declaration": "a", "motion": "m", "trace": "b"}],
    }
    monkeypatch.setattr(determinism, "bake", lambda scene: dict(made))
    out = tmp_path / "case.json"
    assert determinism.main(["LiteralWorlds", "--out", str(out)]) == 0
    written = json.loads(out.read_text())
    assert written["scene"] == "LiteralWorlds"
    assert written["worlds"][0]["declaration"] == "a"
    assert written["problems"]
    assert [p.name for p in tmp_path.iterdir()] == ["case.json"]
