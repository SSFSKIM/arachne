"""What the manager decides for a render is not scene state.

A scene the core refuses is executed again on Cairo, and a play found
refusing mid-play has it executed again once more. Every execution again
begins from the configuration the first one found (the scene may have
changed it as it ran), and the manager's own decisions, the draw path among
them, hold over it until the render ends.
"""

from __future__ import annotations

import pytest

from arachne import Scene, Square, config, tempconfig
from arachne.frame_ir.extract import Unable
from arachne.manager import Manager
from arachne.timeline.execution import Reexecute

from .conftest import SMALL

#: What each execution of ``Refused`` found in the configuration.
FOUND: list[tuple[str, int]] = []


class Refused(Scene):
    """Refused by the core in its first execution, found refusing mid-play in
    its second; every execution changes the configuration as it runs.
    """

    def construct(self):
        FOUND.append((config.draw_path, config.frame_rate))
        config.frame_rate = 5
        if len(FOUND) == 1:
            raise Unable("render.unable: the core cannot draw this")
        if len(FOUND) == 2:
            raise Reexecute(0, "a mobject joined mid-play")
        self.add(Square())
        self.wait(0.2)


@pytest.fixture
def refused_by_the_core(monkeypatch):
    """The core's refusal is the manager's to answer, as on the core."""
    from arachne.frame_ir import lint

    monkeypatch.setattr(lint, "raised", [])
    monkeypatch.setattr(
        Manager, "_refused_by_the_core", lambda self, error: isinstance(error, Unable)
    )
    FOUND.clear()
    return lint.raised


def test_a_scene_refused_by_the_core_that_executes_again_keeps_cairo(
    refused_by_the_core,
):
    with tempconfig({**SMALL, "frame_rate": 15, "dry_run": True, "draw_path": "core"}):
        Manager(Refused()).render()
        assert config.draw_path == "core", "the path asked for, once the render ends"
        assert config.frame_rate == 5, "the last execution's own change stays"
    assert FOUND == [("core", 15), ("cairo", 15), ("cairo", 15)]
    [finding] = refused_by_the_core
    assert finding.code == "render.fallback"


def test_a_decision_holds_over_what_the_scene_set_and_ends_as_asked():
    with tempconfig({**SMALL, "frame_rate": 15, "draw_path": "core"}):
        manager = Manager(Refused())
        configured = config.copy()
        manager._decide("draw_path", "cairo")
        config.draw_path = "ir_cairo"
        config.frame_rate = 5
        manager._reconfigure(configured)
        assert (config.draw_path, config.frame_rate) == ("cairo", 15)
        # Decided again, it is still the value asked for that comes back.
        manager._decide("draw_path", "cairo")
        manager._undecide()
        assert config.draw_path == "core"
        manager._undecide()
        assert config.draw_path == "core"
