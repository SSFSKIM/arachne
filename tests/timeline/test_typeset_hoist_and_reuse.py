"""Shared typesetting, the families zip taken once, and draw reuse, together.

Each is exact alone (its own tests); a render with all of them on must still
write the frames a render with none of them writes, whichever of the zip hoist
(``vectorize_trusted``) and draw reuse (``draw_reuse``) are on, from a cold
shared typeset directory (``typeset_dir``) and from a warm one, and each lever
must actually run: reuse draws frames from reused pixels, the hoist takes fewer
families zips, and a warm directory is read, never rewritten.
"""

from __future__ import annotations

import pytest

from arachne import (
    RIGHT,
    MarkupText,
    MathTex,
    Scene,
    Text,
    Transform,
    VGroup,
    linear,
)
from arachne.mobject.construction_caches import (
    construction_caches,
    forget_construction_caches,
    restore_construction_caches,
)
from arachne.timeline.execution import play_keys

from .test_draw_reuse import assert_same_writes, render_writes
from .test_updater_kind import zips_taken


class TypesetAndMove(Scene):
    """Typeset mobjects of three kinds, moved, held under an updater, then
    transformed into a copy of themselves: a trusted, a ticked and a trusted
    play.
    """

    def construct(self):
        group = VGroup(MathTex("x"), Text("axis"), MarkupText("<b>A</b>")).arrange()
        self.add(group)
        self.play(group.animate.shift(RIGHT), run_time=0.4, rate_func=linear)
        group.add_updater(lambda m, dt: None)
        self.wait(0.4)
        group.clear_updaters()
        target = VGroup(MathTex("x"), Text("axis"), MarkupText("<b>A</b>")).arrange()
        self.play(Transform(group, target), run_time=0.4, rate_func=linear)


@pytest.fixture
def construction_caches_kept():
    """The construction caches as they were, restored after the test."""
    saved = construction_caches()
    yield
    restore_construction_caches(saved)


def shared_svgs(shared):
    """The shared directory's SVGs, as (name, bytes, modification time)."""
    return sorted(
        (path.relative_to(shared), path.read_bytes(), path.stat().st_mtime_ns)
        for path in shared.glob("t*/*.svg")
    )


def test_typesetting_cold_and_warm_with_hoist_and_reuse_independently_toggled(
    tmp_path, construction_caches_kept
):
    serial = 0

    def draw(**options):
        nonlocal serial
        serial += 1
        forget_construction_caches()
        with zips_taken() as zips:
            scene, writes = render_writes(
                TypesetAndMove,
                media_dir=str(tmp_path / f"media-{serial}"),
                verbosity="ERROR",
                **options,
            )
        record = scene.timeline.record
        assert record.refusal is None
        assert [p.route for p in record.plays] == ["trusted", "ticked", "trusted"]
        return play_keys(record), writes, scene.renderer.draws.reused, zips[0]

    unused = tmp_path / "levers-off"
    reference = draw(
        speed_levers=False,
        vectorize_trusted=True,
        draw_reuse=True,
        typeset_dir=str(unused),
    )
    assert not unused.exists(), "with the levers off nothing is shared"
    assert reference[2] == 0
    assert len(reference[1]) == 18
    for hoist in (False, True):
        for reuse in (False, True):
            shared = tmp_path / f"typeset-{hoist}-{reuse}"
            options = {
                "speed_levers": True,
                "vectorize_trusted": hoist,
                "draw_reuse": reuse,
                "typeset_dir": str(shared),
            }
            cold = draw(**options)
            entries = shared_svgs(shared)
            assert len(entries) == 3
            warm = draw(**options)
            assert shared_svgs(shared) == entries, "a warm directory is not rewritten"
            for result in (cold, warm):
                assert result[0] == reference[0], "the recorded content changed"
                assert_same_writes(result[1], reference[1])
                assert (result[2] > 0) is reuse, "draw reuse runs only when on"
            assert cold[3] == warm[3]
            assert (cold[3] < reference[3]) is hoist, "the hoist runs only when on"
