"""An image built from the image cache is the image parsing builds: every
member of its family, its own root included, holds the same values, so a
scene's first render in a process (which parses) and its later renders
(which find the image cached) record and draw alike.
"""

from __future__ import annotations

import pytest

from arachne import BLUE, UP, MathTex, Scene, SVGMobject
from arachne.mobject.construction_caches import forget_construction_caches
from arachne.timeline.keys import value_key
from arachne.timeline.record import NodeState, covered_fields
from tests.helpers.path_utils import get_svg_resource

from .conftest import render


def heart() -> SVGMobject:
    return SVGMobject(get_svg_resource("heart.svg"), fill_color=BLUE, fill_opacity=1)


def formula() -> MathTex:
    return MathTex("x^2", color=BLUE)


IMAGES = {"svg": heart, "formula": formula}


def member_values(image) -> list[tuple[str, list[tuple[str, bytes]]]]:
    """Each member of ``image``'s family, with every field a content key
    covers and that field's key.
    """
    return [
        (
            type(member).__qualname__,
            [
                (name, value_key(value)[0])
                for name, value in covered_fields(NodeState.of(member))
            ],
        )
        for member in image.get_family()
    ]


@pytest.mark.parametrize("make", IMAGES.values(), ids=IMAGES.keys())
def test_an_image_from_the_cache_is_the_image_parsing_built(make):
    forget_construction_caches()
    parsed = make()
    cached = make()
    assert member_values(cached) == member_values(parsed)


class LitAlongItsSheen(Scene):
    """An image lit along its own sheen direction, which it reads from its
    root: the root's direction reaches every part it draws.
    """

    make = staticmethod(heart)

    def construct(self):
        image = self.make().scale(1.5)
        image.set_sheen(0.75, image.get_sheen_direction())
        self.add(image)
        self.play(image.animate.shift(UP), run_time=0.4)


class FormulaLitAlongItsSheen(LitAlongItsSheen):
    make = staticmethod(formula)


@pytest.mark.parametrize(
    "scene_class", [LitAlongItsSheen, FormulaLitAlongItsSheen], ids=["svg", "formula"]
)
def test_a_scene_s_first_render_in_a_process_draws_as_its_later_ones(scene_class):
    forget_construction_caches()
    _, first = render(scene_class, 5)  # parses the image
    _, later = render(scene_class, 5)  # finds it cached
    assert len(first) == len(later)
    for index, (a, b) in enumerate(zip(first, later, strict=True)):
        assert a.tobytes() == b.tobytes(), f"frame {index}"
