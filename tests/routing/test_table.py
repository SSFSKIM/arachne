"""The table of GPU-only capabilities: data, one row per id, and the only
names an object may declare.
"""

from __future__ import annotations

import pytest

from arachne import Square
from arachne.routing import GPU_ONLY, Capability, declare, held, table, withdraw


def test_the_rows_are_the_compositing_effects_a_layers_depth_and_media():
    assert list(GPU_ONLY) == [
        "effect.blur",
        "effect.depth_of_field",
        "effect.soft_mask",
        "effect.window",
        "layer.depth",
        "media.video",
        "media.sprite",
        "media.raster_part",
    ]


def test_each_row_is_an_id_and_one_sentence():
    assert len({row.id for row in table.ROWS}) == len(table.ROWS) == len(GPU_ONLY)
    for key, row in GPU_ONLY.items():
        assert isinstance(row, Capability)
        assert row.id == key
        assert row.summary[0].isupper()
        assert row.summary.endswith(".")
        assert row.summary.count(". ") == 0


def test_the_table_cannot_be_changed_in_place():
    with pytest.raises(TypeError):
        GPU_ONLY["effect.new"] = Capability("effect.new", "Something.")  # type: ignore[index]


def test_a_plain_still_has_no_row():
    assert not any("image" in key or "still" in key for key in GPU_ONLY)


def test_declare_refuses_an_id_that_is_no_row_and_declares_nothing():
    square = Square()
    with pytest.raises(ValueError, match="render.gpu_live"):
        declare(square, "effect.blur", "render.gpu_live")
    assert held(square) == frozenset()


def test_declaring_adds_rows_and_withdrawing_removes_them():
    square = Square()
    before = dict(vars(square))
    declare(square, "effect.blur")
    declare(square, "layer.depth", "effect.blur")
    assert held(square) == {"effect.blur", "layer.depth"}
    withdraw(square, "effect.blur")
    assert held(square) == {"layer.depth"}
    withdraw(square, "layer.depth")
    # Holding nothing, the object is as one that never declared anything.
    assert held(square) == frozenset()
    assert vars(square).keys() == before.keys()


def test_a_declaration_is_a_new_value_each_time_it_changes():
    """The record keeps a new version of an object only when one of its
    attributes holds another object.
    """
    square = Square()
    declare(square, "effect.blur")
    first = vars(square)[table.ATTRIBUTE]
    declare(square, "effect.blur")
    assert vars(square)[table.ATTRIBUTE] is first
    declare(square, "effect.window")
    assert vars(square)[table.ATTRIBUTE] is not first


def test_a_copy_holds_what_its_original_held():
    square = Square()
    declare(square, "effect.soft_mask")
    assert held(square.copy()) == {"effect.soft_mask"}
