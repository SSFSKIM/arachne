"""What the rules read at a lint instant: the scene's drawn parts, with
families flattened, texts as their glyph ink, strokes at their drawn width,
fills, axis furniture, and neither padding nor a second copy of one text.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from arachne.lint import lint
from arachne.lint.parts import Parts

from .conftest import line_of


def _parts_of(session, **kwargs) -> list[Parts]:
    seen: list[Parts] = []

    def keep(context):
        seen.extend(context.parts)
        return ()

    lint(session, rules=(keep,), **kwargs)
    return seen


def _final(session) -> Parts:
    (last,) = _parts_of(session, at=[session.record.duration])
    return last


def test_groups_are_flattened_into_their_drawn_parts(session_of):
    session = session_of("Furnished", frame_rate=15)
    parts = _final(session)
    shapes = [p for p in parts.parts if p.within and p.within[0].name == "shapes"]
    assert [p.type for p in shapes] == ["Square", "Circle"]
    assert all(p.root == shapes[0].within[0].id for p in shapes)
    assert {p.within[0].line for p in shapes} == {line_of("shapes")}


def test_a_stroke_is_as_wide_as_it_is_drawn_and_a_fill_is_a_fill(session_of):
    session = session_of("Furnished", frame_rate=15)
    parts = _final(session)
    square, circle = (p for p in parts.parts if p.type in ("Square", "Circle"))
    (edge,) = square.leaves
    # Width 8 at Cairo's 0.01 units per width, on a frame the home size.
    assert edge.stroke_width == pytest.approx(0.08)
    assert edge.fill_opacity == 0
    assert edge.closed
    # Its ink is Cairo's extents of the stroke, to Cairo's tolerance.
    assert square.ink_box[0] == pytest.approx(square.screen_box[0] - 0.04, abs=1e-3)
    (disc,) = circle.leaves
    assert disc.stroke_width == 0
    assert disc.fill_opacity == 1
    assert circle.ink_box == circle.screen_box


def test_screen_and_scene_boxes_agree_under_a_still_camera(session_of):
    session = session_of("Furnished", frame_rate=15)
    for part in _final(session).parts:
        assert part.screen_box == pytest.approx(part.scene_box)


def test_axis_furniture_is_marked(session_of):
    session = session_of("Furnished", frame_rate=15)
    parts = _final(session).parts
    furniture = [p for p in parts if p.furniture]
    assert {p.type for p in furniture} >= {"NumberLine", "ArrowTriangleFilledTip"}
    assert all(p.within[0].type == "Axes" for p in furniture)
    assert not any(p.furniture for p in parts if p.root != furniture[0].root)


def test_a_text_is_one_part_of_glyph_ink(session_of):
    session = session_of("Furnished", frame_rate=15)
    (text,) = _final(session).texts
    assert text.kind == "text"
    assert text.text == "hi"
    assert text.name == "label"
    assert text.scene_line == line_of("label")
    assert len(text.leaves) == 2
    assert all(leaf.points.shape[1] == 2 for leaf in text.leaves)
    glyphs = [leaf.screen_box for leaf in text.leaves]
    assert text.screen_box == (
        min(b[0] for b in glyphs),
        min(b[1] for b in glyphs),
        max(b[2] for b in glyphs),
        max(b[3] for b in glyphs),
    )


def test_two_copies_of_one_text_in_one_place_are_one_text(session_of):
    session = session_of("Furnished", frame_rate=15)
    (text,) = _final(session).texts
    assert text.copies == 2
    assert text.name == "label"
    # The copy, drawn after it, is on top.
    assert text.top > max(leaf.order for leaf in text.leaves)


def test_parts_are_in_paint_order(session_of):
    session = session_of("Furnished", frame_rate=15)
    orders = [p.order for p in _final(session).parts]
    assert orders == sorted(orders)
    assert len(set(orders)) == len(orders)


def test_what_draws_nothing_is_no_part(session_of):
    """Three dots transformed into one: two of them end fully transparent
    where the one is, as the target's padding is. ``state_at`` reports all
    three; only the one that draws is a part.
    """
    session = session_of("Shrunk", frame_rate=15)
    record = session.record
    (last,) = _parts_of(session, at=[record.duration])
    state = session.state_at(record.duration)
    dots = [node.id for node in state.nodes if node.type == "Dot"]
    assert len(dots) == 3
    (part,) = last.parts
    assert part.id in dots
    assert part.leaves[0].fill_opacity == 1


def test_every_lint_instant_has_its_parts(session_of):
    session = session_of("Furnished", frame_rate=15)
    seen = _parts_of(session)
    assert [p.t for p in seen] == sorted({p.t for p in seen})
    assert all(p.answer == "recorded" for p in seen)
    assert seen[-1].t == session.record.duration
    assert seen[-1].play is None


def test_a_text_s_backdrop_is_its_own_part_not_its_ink(session_of):
    session = session_of("Backdropped", frame_rate=15)
    parts = _final(session)
    (text,) = parts.texts
    assert {leaf.type for leaf in text.leaves} == {"VMobjectFromSVGPath"}
    (backdrop,) = [p for p in parts.parts if p.type == "BackgroundRectangle"]
    assert backdrop.kind == "path"
    assert backdrop.order < text.order
    # The rectangle reaches a unit past the glyphs.
    assert backdrop.screen_box[0] < text.screen_box[0] - 0.9


def test_a_copy_drawn_wider_is_a_text_of_its_own(session_of):
    """A copy with a thick stroke lays other ink than its original: two
    texts, the copy's ink the wider, neither a copy of the other.
    """
    session = session_of("StyledCopy", frame_rate=15)
    plain, thick = _final(session).texts
    assert (plain.copies, thick.copies) == (1, 1)
    assert max(leaf.stroke_width for leaf in plain.leaves) == 0
    assert min(leaf.stroke_width for leaf in thick.leaves) > 0
    assert thick.ink_box[0] < plain.ink_box[0]
    assert thick.ink_box[2] > plain.ink_box[2]


def test_a_transparent_image_is_no_part(session_of):
    session = session_of("InvisibleImage", frame_rate=15)
    assert _final(session).parts == ()


def test_a_point_cloud_s_ink_is_as_thick_as_it_is_drawn(session_of):
    session = session_of("WideCloud", frame_rate=15)
    (cloud,) = _final(session).parts
    (leaf,) = cloud.leaves
    assert leaf.draw == "points"
    assert leaf.stroke_width > 0
    width = leaf.ink_box[2] - leaf.ink_box[0]
    assert width > leaf.screen_box[2] - leaf.screen_box[0]
    assert width == pytest.approx(leaf.stroke_width, abs=0.2)


def test_a_zoomed_display_and_its_frames_are_parts(session_of):
    """A zoomed scene draws its display (an image of the zoomed camera's
    picture) with its border, and the zoomed camera's frame over the scene:
    all on screen, all parts.
    """
    from ..timeline import camera_scenes

    session = session_of("ZoomStill", Path(camera_scenes.__file__), frame_rate=15)
    parts = _final(session).parts
    kinds = {(p.type, p.kind) for p in parts}
    assert ("ImageMobjectFromCamera", "image") in kinds
    display = next(p for p in parts if p.kind == "image")
    (border,) = [p for p in parts if p.within and p.within[-1].id == display.id]
    assert border.type == "SurroundingRectangle"
    assert border.order > display.order
    # The zoomed camera's frame, drawn over the scene.
    assert ("ScreenRectangle", "path") in kinds


def test_copies_in_another_colour_or_opacity_are_one_text(session_of):
    session = session_of("Recoloured", frame_rate=15)
    (text,) = _final(session).texts
    assert text.copies == 3


def test_a_cloud_s_stroke_is_as_drawn_where_its_ink_is_cut_to_the_frame(session_of):
    """A cloud's points on the frame's corners paint only the quarter of
    each square inside it, but are drawn as wide as one in the middle.
    """
    inside = _final(session_of("WideCloud", frame_rate=15)).parts[0].leaves[0]
    corners = _final(session_of("CornerCloud", frame_rate=15)).parts[0].leaves[0]
    assert inside.stroke_width > 0
    assert corners.stroke_width == pytest.approx(inside.stroke_width, abs=1e-9)
    width = corners.ink_box[2] - corners.ink_box[0]
    assert width == pytest.approx(corners.screen_box[2] - corners.screen_box[0])
