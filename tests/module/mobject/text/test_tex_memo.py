"""The TeX construction memo: a ``MathTex``, ``Tex`` or ``SingleStringMathTex``
built once and handed out as a copy of what it built, exactly as a fresh
construction builds it.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest

from arachne import (
    BLUE,
    RED,
    DecimalNumber,
    MathTex,
    SingleStringMathTex,
    Tex,
    TexTemplate,
    VMobject,
    config,
    tempconfig,
)
from arachne.mobject.construction_caches import forget_construction_caches
from arachne.mobject.svg.svg_mobject import SVG_HASH_TO_MOB_MAP, SVGMobject
from arachne.mobject.text.tex_memo import TEX_MEMO
from arachne.timeline.inputs import Inputs
from arachne.timeline.origins import ORIGINS
from tests.helpers.recorded_state import mobject_state

HERE = os.path.dirname(os.path.realpath(__file__))


@pytest.fixture(autouse=True)
def empty_memo():
    """Each test starts from an empty memo and empty construction caches."""
    TEX_MEMO.clear()
    forget_construction_caches()
    yield
    TEX_MEMO.clear()
    forget_construction_caches()


#: Constructions of each memoized class, and of a number whose glyphs are.
MAKERS: dict[str, Callable[[], VMobject]] = {
    "math": lambda: MathTex(r"a^2 + b^2 = c^2"),
    "parts": lambda: MathTex(
        r"\frac{x}{2}", "+", "y", substrings_to_isolate=["x"], font_size=30
    ),
    "colored": lambda: MathTex(
        "e^{i", r"\pi}", "+ 1 = 0", tex_to_color_map={r"\pi}": BLUE, "e^{i": RED}
    ),
    "tex": lambda: Tex("Hello", " world", arg_separator="", color=BLUE),
    "single": lambda: SingleStringMathTex(r"\sqrt{2}", stroke_width=1),
    "number": lambda: DecimalNumber(-3.25, num_decimal_places=2),
}


def _three(make: Callable[[], VMobject]) -> list[VMobject]:
    """Three constructions in a row from empty construction caches: the
    first finds no parsed image, the others find it.
    """
    forget_construction_caches()
    return [make() for _ in range(3)]


def _states(mobs, origins=False):
    return [mobject_state(mob, origins=origins) for mob in mobs]


@pytest.mark.parametrize("name", MAKERS)
def test_a_hit_builds_what_a_construction_without_the_memo_builds(name):
    make = MAKERS[name]
    with tempconfig({"construction_memo": False}):
        off = _three(make)
        off_images = set(SVG_HASH_TO_MOB_MAP)
    assert len(TEX_MEMO) == 0
    cold = _three(make)
    cold_images = set(SVG_HASH_TO_MOB_MAP)
    hits = TEX_MEMO.hits
    warm = _three(make)
    warm_images = set(SVG_HASH_TO_MOB_MAP)
    assert TEX_MEMO.hits > hits
    assert _states(off) == _states(cold) == _states(warm)
    assert off_images == cold_images == warm_images


@pytest.mark.parametrize("name", MAKERS)
def test_a_hit_gives_the_origins_a_fresh_construction_gives(name):
    """Every mobject a construction makes is numbered at its line, kept or
    not; a hit numbers what it hands out, and the line, as a miss does.
    """
    make = MAKERS[name]

    def run(memo: bool) -> tuple[list, dict]:
        with ORIGINS.kept(), tempconfig({"construction_memo": memo}):
            ORIGINS.start(HERE, __file__)
            mobs = _three(make)
            mobs.append(VMobject())
            return _states(mobs, origins=True), dict(ORIGINS._ordinals)

    off = run(False)
    cold = run(True)
    hits = TEX_MEMO.hits
    warm = run(True)
    assert TEX_MEMO.hits > hits
    assert off == cold == warm


def test_a_hit_built_while_nothing_tagged_is_built_again_while_tagging():
    MathTex("x")
    forget_construction_caches()
    with ORIGINS.kept():
        ORIGINS.start(HERE, __file__)
        hits = TEX_MEMO.hits
        tagged = MathTex("x")
        assert TEX_MEMO.hits == hits
        assert all(ORIGINS.get(m) is not None for m in tagged.get_family())
    forget_construction_caches()
    with ORIGINS.kept():
        ORIGINS.start(HERE, __file__)
        MathTex("x")
        assert TEX_MEMO.hits == hits + 1


def test_the_image_cache_is_filled_as_a_fresh_construction_fills_it():
    MathTex("y^3")
    forget_construction_caches()
    assert not SVG_HASH_TO_MOB_MAP
    hits = TEX_MEMO.hits
    MathTex("y^3")
    assert TEX_MEMO.hits == hits + 1
    assert len(SVG_HASH_TO_MOB_MAP) == 1


# Separation: a change in what construction reads is a miss.


def _hit(make: Callable[[], object]) -> bool:
    forget_construction_caches()
    hits = TEX_MEMO.hits
    make()
    return TEX_MEMO.hits > hits


def _primed(make: Callable[[], object]) -> None:
    _hit(make)
    assert _hit(make), "the same construction again is a hit"


@pytest.mark.parametrize(
    "other",
    [
        lambda: MathTex("a", "b"),
        lambda: MathTex("ab", font_size=48.0),
        lambda: MathTex("ab", font_size=30),
        lambda: MathTex("ab", color=RED),
        lambda: MathTex("ab", arg_separator=""),
        lambda: MathTex("ab", substrings_to_isolate=["a"]),
        lambda: MathTex("ab", tex_to_color_map={"a": RED}),
        lambda: MathTex("ab", tex_environment="gather*"),
        lambda: MathTex("ab", stroke_width=2),
        lambda: MathTex(
            "ab", tex_template=TexTemplate(preamble=r"\usepackage{amsmath}")
        ),
        lambda: Tex("ab"),
        lambda: SingleStringMathTex("ab"),
    ],
    ids=[
        "strings",
        "font_size_type",
        "font_size",
        "color",
        "arg_separator",
        "substrings_to_isolate",
        "tex_to_color_map",
        "tex_environment",
        "stroke_width",
        "tex_template",
        "class_tex",
        "class_single",
    ],
)
def test_each_constructor_argument_and_the_class_key_apart(other):
    _primed(lambda: MathTex("ab", font_size=48))
    assert not _hit(other)


def test_the_order_of_a_colour_map_keys_apart():
    _primed(lambda: MathTex("a", "b", tex_to_color_map={"a": RED, "b": BLUE}))
    assert not _hit(lambda: MathTex("a", "b", tex_to_color_map={"b": BLUE, "a": RED}))


@pytest.mark.parametrize(
    "change",
    [
        lambda t: t.add_to_preamble(r"\usepackage{amssymb}"),
        lambda t: setattr(t, "post_doc_commands", r"\boldmath"),
        lambda t: setattr(t, "tex_compiler", "xelatex"),
        lambda t: setattr(t, "output_format", ".xdv"),
        lambda t: setattr(t, "body", "% a body of its own\n" + t.body),
    ],
    ids=["preamble", "post_doc_commands", "compiler", "output_format", "body"],
)
def test_the_configured_template_keys_apart(change):
    template = TexTemplate()
    with tempconfig({"tex_template": template}):
        _primed(lambda: MathTex("q"))
        changed = template.copy()
        change(changed)
        config.tex_template = changed
        # A compiler or output format changes no file name: the image found
        # is the same one, and the memo still tells them apart.
        assert not _hit(lambda: MathTex("q"))


def test_another_media_directory_keys_apart(tmp_path):
    _primed(lambda: MathTex("m"))
    with tempconfig({"media_dir": str(tmp_path / "other")}):
        assert not _hit(lambda: MathTex("m"))


def test_the_image_s_content_keys_apart():
    first = MathTex("w")
    svg = Path(first.file_name)
    _primed(lambda: MathTex("w"))
    other = MathTex("v")
    svg.write_bytes(Path(other.file_name).read_bytes())
    assert not _hit(lambda: MathTex("w"))
    with tempconfig({"construction_memo": False}):
        forget_construction_caches()
        expected = mobject_state(MathTex("w"))
    forget_construction_caches()
    assert mobject_state(MathTex("w")) == expected


def test_whether_the_image_cache_holds_the_image_keys_apart():
    forget_construction_caches()
    MathTex("k")
    MathTex("k")
    assert len(TEX_MEMO) == 2


def test_a_default_set_on_the_class_or_a_base_keys_apart():
    _primed(lambda: MathTex("d"))
    try:
        MathTex.set_default(font_size=20)
        assert not _hit(lambda: MathTex("d"))
        MathTex.set_default()
        assert _hit(lambda: MathTex("d"))
        VMobject.set_default(color=RED)
        assert not _hit(lambda: MathTex("d"))
        with tempconfig({"construction_memo": False}):
            forget_construction_caches()
            expected = mobject_state(MathTex("d"))
        forget_construction_caches()
        assert mobject_state(MathTex("d")) == expected
    finally:
        MathTex.set_default()
        VMobject.set_default()


def test_an_engine_method_replaced_is_a_miss(monkeypatch):
    _primed(lambda: MathTex("p"))
    original = SVGMobject.flip

    def flip(self, *args, **kwargs):
        return original(self, *args, **kwargs).shift([1, 0, 0])

    monkeypatch.setattr(SVGMobject, "flip", flip)
    assert not _hit(lambda: MathTex("p"))
    with tempconfig({"construction_memo": False}):
        forget_construction_caches()
        expected = mobject_state(MathTex("p"))
    forget_construction_caches()
    assert mobject_state(MathTex("p")) == expected


def test_a_subclass_is_built_as_it_always_was():
    class Mine(MathTex):
        pass

    Mine("s")
    forget_construction_caches()
    hits = TEX_MEMO.hits
    Mine("s")
    assert TEX_MEMO.hits == hits
    assert len(TEX_MEMO) == 0


@pytest.mark.parametrize(
    "settings", [{"construction_memo": False}, {"speed_levers": False}]
)
def test_the_lever_off_neither_reads_nor_fills_the_memo(settings):
    MathTex("o")
    stored = len(TEX_MEMO)
    with tempconfig(settings):
        for _ in range(2):
            forget_construction_caches()
            hits = TEX_MEMO.hits
            MathTex("o")
            MathTex("u")
            assert TEX_MEMO.hits == hits
    assert len(TEX_MEMO) == stored


def test_a_watched_image_directory_is_never_served(tmp_path):
    """A recording that would see the image read (its directory not
    excluded) sees what a fresh construction reads.
    """
    MathTex("r")
    forget_construction_caches()

    def reads(memo: bool) -> tuple:
        watcher = Inputs(HERE, excluded=())
        forget_construction_caches()
        with tempconfig({"construction_memo": memo}):
            watcher.start()
            try:
                hits = TEX_MEMO.hits
                MathTex("r")
                served = TEX_MEMO.hits > hits
            finally:
                watcher.stop()
        return served, watcher._read

    served, seen = reads(True)
    assert not served
    assert seen == reads(False)[1]


def test_a_failing_formula_raises_as_it_does_without_the_memo():
    with (
        pytest.raises(ValueError) as without,
        tempconfig({"construction_memo": False}),
    ):
        MathTex(r"\undefinedcontrolsequence")
    with pytest.raises(ValueError) as memoized:
        MathTex(r"\undefinedcontrolsequence")
    assert str(memoized.value) == str(without.value)
    assert len(TEX_MEMO) == 0


# Caller safety: what a construction hands out never reaches the memo.


MUTATIONS: dict[str, Callable[[MathTex], object]] = {
    "set_color": lambda m: m.set_color(RED),
    "set_color_by_tex": lambda m: m.set_color_by_tex("y", BLUE),
    "set_opacity_by_tex": lambda m: m.set_opacity_by_tex("x", 0.2, 0.7),
    "font_size": lambda m: setattr(m, "font_size", 12),
    "shift": lambda m: m.shift([1, 2, 0]),
    "leaf_points": lambda m: m.family_members_with_points()[0].set_points(
        m.family_members_with_points()[0].points * 2
    ),
    "remove_part": lambda m: m.remove(m.submobjects[0]),
    "submobject_list": lambda m: m.submobjects[-1].submobjects.clear(),
    "part_group": lambda m: m.id_to_vgroup_dict["root"].set_fill(BLUE, 0.3),
    "attribute": lambda m: m.tex_strings.append("z"),
    "colour_map": lambda m: m.tex_to_color_map.update(z=RED),
}


def _parts() -> MathTex:
    return MathTex("x", "+", "y", tex_to_color_map={"y": RED})


@pytest.mark.parametrize("name", MUTATIONS)
def test_changing_what_a_hit_or_a_miss_hands_out_never_reaches_the_memo(name):
    with tempconfig({"construction_memo": False}):
        expected = _states(_three(_parts))
    forget_construction_caches()
    for mob in [_parts() for _ in range(3)]:
        # Misses: the first two; the third a hit.
        MUTATIONS[name](mob)
    forget_construction_caches()
    hits = TEX_MEMO.hits
    handed = [_parts() for _ in range(3)]
    for mob in handed:
        MUTATIONS[name](mob)
    assert TEX_MEMO.hits == hits + 3
    assert _states(_three(_parts)) == expected


def test_a_hit_holds_the_caller_s_own_objects_where_a_miss_does():
    colours = {"y": RED}
    template = TexTemplate()
    first = MathTex("y", tex_to_color_map=colours, tex_template=template)
    assert first.tex_to_color_map is colours
    assert first.tex_template is template
    forget_construction_caches()
    again, other = {"y": RED}, TexTemplate()
    hits = TEX_MEMO.hits
    second = MathTex("y", tex_to_color_map=again, tex_template=other)
    assert TEX_MEMO.hits == hits + 1
    assert second.tex_to_color_map is again
    assert second.tex_template is other
    forget_construction_caches()
    MathTex("y")
    forget_construction_caches()
    hits = TEX_MEMO.hits
    third = MathTex("y")
    assert TEX_MEMO.hits == hits + 1
    assert third.tex_template is config.tex_template


def test_the_memo_is_bounded_and_eviction_costs_only_a_hit(monkeypatch):
    monkeypatch.setattr(TEX_MEMO, "bound", 3)
    for k in range(5):
        forget_construction_caches()
        MathTex(f"{k}")
    assert len(TEX_MEMO) == 3
    assert not _hit(lambda: MathTex("0"))
    assert _hit(lambda: MathTex("4"))
    with tempconfig({"construction_memo": False}):
        forget_construction_caches()
        expected = mobject_state(MathTex("0"))
    forget_construction_caches()
    assert mobject_state(MathTex("0")) == expected
