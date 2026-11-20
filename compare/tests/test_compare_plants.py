"""Planted defects, each with the exact set of codes it must give, in both modes."""

from __future__ import annotations

import json
from fractions import Fraction

import numpy as np
import pytest
from arachne_compare import (
    ArrayFrames,
    ChangedSet,
    Footprint,
    ObjectMap,
    Thresholds,
    cheap_self,
    choose_objects,
    compare,
    flicker,
)
from compare_plants import (
    DOT,
    GLYPH,
    LINE,
    SMALL,
    SURFACE,
    Shape,
    busy,
    handed,
    nothing_changed,
    rect,
    render,
    renderers,
    single,
    stripes,
)

MODES = ("same", "cross")


def run(reference_shapes, candidate_shapes, mode, *, reference_map=True):
    ref_renderer, cand_renderer = renderers(mode)
    reference, ref_map = render(reference_shapes, ref_renderer)
    candidate, cand_map = render(candidate_shapes, cand_renderer)
    return compare(
        single(reference),
        single(candidate),
        mode=mode,
        maps=handed([ref_map] if reference_map else [], [cand_map]),
        changed=nothing_changed,
        plays=[(0, 1)],
        fps=Fraction(30),
        seed=0,
        key="plant",
    )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("plant", SMALL, ids=lambda s: s.name)
def test_a_dropped_small_object_is_missing_and_nothing_else(mode, plant):
    scene = [*busy(), *SMALL]
    verdict = run(scene, [s for s in scene if s is not plant], mode)
    assert verdict.codes == {"gate.presence.missing"}
    (finding,) = verdict.findings
    assert finding.object == plant.name


@pytest.mark.parametrize("mode", MODES)
def test_an_object_only_the_candidate_draws_is_extra(mode):
    scene = [*busy(), DOT]
    verdict = run(scene[:-1], scene, mode)
    assert verdict.codes == {"gate.presence.extra"}


def test_the_small_plants_are_under_the_cluster_threshold():
    # So presence, not a cluster, is what catches them.
    for plant in (LINE, DOT, GLYPH):
        _, found = render([plant])
        assert found.objects[plant.name].mask.sum() <= 25


@pytest.mark.parametrize("mode", MODES)
def test_a_shifted_region_is_a_cluster(mode):
    panel = rect("panel", 200, 30, 240, 52, (250, 120, 40))
    scene = busy()
    verdict = run([*scene, panel], [*scene, panel.moved(4)], mode)
    assert verdict.codes == {"gate.cluster"}
    assert "panel" in {f.object for f in verdict.findings}
    for finding in verdict.findings:
        x0, y0, x1, y1 = finding.box
        assert min(x0 - 200, y0 - 30, 244 - x1, 52 - y1) >= 0


def test_antialiasing_along_every_edge_passes_between_two_renderers():
    scene = busy()
    assert len(scene) >= 51
    verdict = run(scene, scene, "cross")
    assert verdict.verdict == "pass"
    assert verdict.counts["edge_changed"] > 1000
    assert verdict.counts["edge_strong"] > 500


def test_the_same_antialiasing_blocks_in_same_mode():
    reference, ref_map = render(busy(), "fine")
    candidate, cand_map = render(busy(), "sparse")
    verdict = compare(
        single(reference),
        single(candidate),
        mode="same",
        maps=handed([ref_map], [cand_map]),
        changed=nothing_changed,
        plays=[(0, 1)],
        fps=Fraction(30),
        seed=0,
        key="plant",
    )
    assert verdict.codes == {"gate.cluster"}


def test_the_same_antialiasing_blocks_between_two_renderers_without_a_reference_map():
    verdict = run(busy(), busy(), "cross", reference_map=False)
    assert verdict.codes == {"gate.cluster"}
    assert verdict.counts["edge_pixels"] == 0


def corrupted(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = stripes(x, y)
    patch = (x >= 340) & (x < 352) & (y >= 190) & (y < 202)
    out[patch] = stripes(x + 6, y)[patch]
    return out


def test_a_corrupted_interior_of_a_textured_surface_is_a_cluster_between_two_renderers():
    scene = busy()
    assert scene[-1] is SURFACE
    broken = Shape("surface", SURFACE.polygons, texture=corrupted)
    verdict = run(scene, [*scene[:-1], broken], "cross")
    assert verdict.codes == {"gate.cluster"}
    (finding,) = verdict.findings
    assert finding.object == "surface"
    x0, y0, x1, y1 = finding.box
    assert min(x0 - 338, y0 - 188, 354 - x1, 204 - y1) >= 0


def test_a_thin_glyph_of_edge_pixels_only_is_still_missing_between_two_renderers():
    _, found = render([GLYPH])
    verdict = run([*busy(), GLYPH], busy(), "cross")
    assert verdict.codes == {"gate.presence.missing"}
    # Every pixel of the glyph is an edge pixel: the edge rule exempts none from presence.
    assert verdict.edges["glyph"]["pixels"] == found.objects["glyph"].mask.sum()


FLASH = (slice(20, 32), slice(436, 448))


def mover(i: int) -> Shape:
    return rect("mover", 20 + 3 * i, 200, 44 + 3 * i, 218, (240, 90, 90))


def sequence(mode, *, flash_at=None, flash=FLASH, camera=False):
    """Six frames of a busy scene with one moving object; a flash put into one drawn frame."""
    ref_renderer, cand_renderer = renderers(mode)
    background = busy()
    references, candidates, ref_maps, cand_maps = [], [], [], []
    for i in range(6):
        frame, found = render([*background, mover(i)], ref_renderer)
        references.append(frame)
        ref_maps.append(found)
        frame, found = render([*background, mover(i)], cand_renderer)
        if i == flash_at:
            frame = frame.copy()
            frame[flash] = (255, 255, 255)
        candidates.append(frame)
        cand_maps.append(found)

    def changed(a: int, b: int) -> ChangedSet:
        return ChangedSet(frozenset({"mover"}), camera=camera)

    verdict = compare(
        ArrayFrames(references, Fraction(30)),
        ArrayFrames(candidates, Fraction(30)),
        mode=mode,
        maps=handed(ref_maps, cand_maps),
        changed=changed,
        plays=[(0, 6)],
        fps=Fraction(30),
        seed=0,
        key="flash",
    )
    return verdict, candidates, cand_maps


@pytest.mark.parametrize("mode", MODES)
def test_a_one_frame_flash_is_flicker(mode):
    verdict, _, _ = sequence(mode, flash_at=3)
    assert verdict.codes == {"gate.flicker"}
    assert {f.frame for f in verdict.findings} == {3, 4}
    assert all(f.box == (436, 20, 448, 32) for f in verdict.findings)


@pytest.mark.parametrize("mode", MODES)
def test_a_moving_object_alone_passes(mode):
    verdict, _, _ = sequence(mode)
    assert verdict.verdict == "pass"


def test_a_flash_inside_a_changed_objects_footprint_is_not_flicker():
    inside = (slice(203, 213), slice(32, 42))  # inside the mover at frame 3
    verdict, candidates, maps = sequence("same", flash_at=3, flash=inside)
    # Against the reference it is still a difference; as flicker it is not flagged.
    assert verdict.codes == {"gate.cluster"}
    changed = ChangedSet(frozenset({"mover"}))
    previous = {"mover": maps[2].objects["mover"]}
    current = {"mover": maps[3].objects["mover"]}
    assert flicker(candidates[2], candidates[3], changed, previous, current) == ()
    assert (
        flicker(
            candidates[3],
            candidates[4],
            changed,
            current,
            {"mover": maps[4].objects["mover"]},
        )
        == ()
    )


def test_flicker_takes_a_changed_object_out_by_its_box_or_its_grown_mask():
    _, candidates, maps = sequence(
        "same", flash_at=3, flash=(slice(219, 222), slice(29, 49))
    )
    changed = ChangedSet(frozenset({"mover"}))
    box_only = {"mover": Footprint(maps[3].objects["mover"].box)}
    before = {"mover": maps[2].objects["mover"]}
    # The flash sits just below the mover: its mask does not hold it, a grow of 4 does.
    assert (
        len(flicker(candidates[2], candidates[3], changed, before, maps[3].objects))
        == 1
    )
    mask = maps[3].objects["mover"]
    grown = Footprint(mask.box, mask.mask, grow=4)
    assert (
        flicker(candidates[2], candidates[3], changed, before, {"mover": grown}) == ()
    )
    assert len(flicker(candidates[2], candidates[3], changed, before, box_only)) == 1


def test_a_pair_where_the_camera_moved_is_not_judged_and_is_counted():
    verdict, candidates, maps = sequence("same", flash_at=3, camera=True)
    assert "gate.flicker" not in verdict.codes
    assert verdict.counts["flicker_unjudged"] == len(
        [i for i in verdict.sampled if i > 0]
    )
    assert (
        flicker(
            candidates[2], candidates[3], ChangedSet(frozenset(), camera=True), {}, {}
        )
        is None
    )


def test_one_sequence_alone_nominates_the_flash():
    _, candidates, _ = sequence("same", flash_at=3)
    found = cheap_self(ArrayFrames(candidates, Fraction(30)))
    assert [f.index for f in found if f.flagged] == [3]


def test_a_verdicts_json_is_byte_equal_across_two_runs():
    first = sequence("cross", flash_at=3)[0].to_json()
    second = sequence("cross", flash_at=3)[0].to_json()
    assert first == second
    document = json.loads(first)
    assert first == json.dumps(document, sort_keys=True, separators=(",", ":"))
    assert document["verdict"] == "block"
    assert document["findings"][0]["t"] == "1/10"
    assert document["findings"][0]["fix"]
    assert set(document) == {
        "verdict",
        "mode",
        "thresholds",
        "frames",
        "sampled",
        "findings",
        "counts",
        "edges",
    }


class ListedMaps:
    """A map source that lists boxes and draws only the objects asked for, as a replay does."""

    def __init__(
        self, reference: ObjectMap, candidate: ObjectMap, listed: dict[str, tuple]
    ):
        self.full = {"reference": reference, "candidate": candidate}
        self.listed = listed
        self.asked: list[list[str]] = []

    def boxes(self, frame):
        return self.listed["reference"], self.listed["candidate"]

    def map(self, side, frame, ids):
        self.asked.append(list(ids))
        full = self.full[side].objects
        return ObjectMap(
            self.full[side].size,
            {
                n: full.get(n, Footprint((0, 0, 0, 0), np.zeros((0, 0), bool)))
                for n in ids
            },
            complete=False,
        )


def test_a_line_drawn_as_nothing_among_crowded_objects_is_examined_and_missing():
    # The candidate still holds the line (its box is listed) but draws nothing of it.
    scene = [*busy(), *SMALL]
    reference, ref_map = render(scene)
    candidate, cand_map = render([s for s in scene if s is not LINE])
    boxes = {n: f.box for n, f in ref_map.objects.items()}
    source = ListedMaps(ref_map, cand_map, {"reference": boxes, "candidate": boxes})
    thresholds = Thresholds(replay_cap=8, replay_reserve=2)
    verdict = compare(
        single(reference),
        single(candidate),
        mode="same",
        maps=source,
        changed=nothing_changed,
        plays=[(0, 1)],
        fps=Fraction(30),
        seed=0,
        key="crowd",
        thresholds=thresholds,
    )
    assert verdict.codes == {"gate.presence.missing"}
    assert all(len(ids) <= 8 for ids in source.asked)
    assert verdict.counts["objects_unexamined"] == len(boxes) - 8
    # Without the group of 20 changed pixels the line is not among the first picks.
    alone = choose_objects(boxes, boxes, [], cap=8, reserve=2, seed=0, key="crowd:0")
    assert "line" not in alone
