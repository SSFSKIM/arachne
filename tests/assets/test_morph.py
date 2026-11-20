"""Two rigs morph part by part, and the play leaves one rig: the destination,
once, where the source stood, with parts, slots and anchors that work and a
pose that follows; neither the source nor any part of the morph stays.
"""

from __future__ import annotations

import textwrap
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from arachne import LEFT
from arachne.assets import MorphParts, Rig
from arachne.session import Session
from arachne.timeline import admission
from arachne.timeline.draw import Drawing

from . import rig_scenes
from .helpers import copied, edited

SCENES = Path(rig_scenes.__file__)
SMALL = {"pixel_width": 96, "pixel_height": 54}

PAIR = """\
from arachne import *


class Pair(Scene):
    def construct(self):
        source = Rig({source!r}, name="source").shift(LEFT)
        source.pose(reading=20).show(lamp="on")
        destination = Rig({destination!r}, name="destination")
        self.add(source)
        self.play(MorphParts(source, destination), run_time=0.5)
        destination.show(lamp="off")
        self.play(Pose(destination, reading=80), run_time=0.5)
"""


def _renamed(document: dict) -> None:
    """The dial with its needle called a pointer: a part the dial lacks."""
    document["name"] = "pointer-dial"
    document["parts"][1]["name"] = "pointer"
    document["parameters"][0]["drives"][0]["part"] = "pointer"
    document["anchors"][0].update(name="pointer_tip", part="pointer")


@pytest.fixture
def pair(tmp_path: Path):
    """A dial morphed into a dial whose needle is named otherwise: each has a
    part the other lacks, and they share a face and a lamp slot.
    """
    source = copied(tmp_path, "dial")
    destination = edited(copied(tmp_path, "dial", "pointer-dial"), _renamed)
    scene = tmp_path / "pair.py"
    scene.write_text(
        textwrap.dedent(PAIR.format(source=str(source), destination=str(destination)))
    )
    admission.DEMOTED.clear()
    session = Session.open(
        scene,
        "Pair",
        config={**SMALL, "frame_rate": 10, "media_dir": str(tmp_path / "media")},
    )
    yield session
    session.close()
    admission.DEMOTED.clear()


def _roots(session: Session, t: Fraction) -> list:
    _, capture = session.record.capture_at(t)
    return list(capture.roots)


def test_a_morph_leaves_the_destination_once_and_nothing_else(pair):
    scene = pair.scene
    (destination,) = scene.mobjects
    assert isinstance(destination, Rig)
    assert destination.name == "destination"
    family = scene.get_mobject_family_members()
    assert sum(m is destination for m in family) == 1
    # Every member of the scene is the destination's.
    assert set(map(id, family)) == set(map(id, destination.get_family()))
    # And the record says the same at the end of the morph's play.
    assert [m.name for m in _roots(pair, Fraction(1, 2))] == ["destination"]


def test_the_destination_works_after_the_morph(pair):
    (destination,) = pair.scene.mobjects
    # Its slot switched, and its pose followed, after the morph.
    assert destination.shown("lamp") == "off"
    assert destination.value("reading") == 80
    reference = Rig(destination.asset).shift(LEFT).pose(reading=80)
    for got, want in zip(
        destination.family_members_with_points(),
        reference.family_members_with_points(),
        strict=True,
    ):
        np.testing.assert_allclose(got.points, want.points, atol=1e-9)
    np.testing.assert_allclose(
        destination.anchor("pointer_tip").point(),
        reference.anchor("pointer_tip").point(),
        atol=1e-9,
    )


def test_a_morph_draws_its_parts_between_the_two_rigs(pair):
    record = pair.record
    play = record.plays[0]
    assert play.route == "ticked"
    # Mid-morph, the source's part only it has is fading and the
    # destination's is coming in, both in the one rig the morph draws; at
    # the end, only the destination is drawn.
    _, capture = record.capture_at(Fraction(3, 10))
    assert [m.name for m in capture.roots] == ["source"]
    with Drawing(record) as drawing:
        drawing.show(capture, capture.roots)
        (morphing,) = capture.roots
        arriving = next(m for m in morphing.submobjects if m.name == "pointer")
        opacity = arriving.submobjects[0].get_fill_opacity()
    assert 0 < opacity < 1


BACKED = """\
from arachne import *


class Backed(Scene):
    def construct(self):
        source = Rig({source!r}, name="source")
        destination = Rig({destination!r}, name="destination")
        self.add(source)
        self.play(MorphParts(source, destination), run_time=0.5)
"""


def _backed(document: dict) -> None:
    """The dial with a plate drawn behind its face: a part the dial lacks,
    lowest in the drawing order.
    """
    document["name"] = "backed-dial"
    for part in document["parts"]:
        part["order"] += 1
    for slot in document["slots"]:
        slot["order"] += 1
    document["parts"].insert(
        0,
        {
            "name": "backplate",
            "file": "pieces/face.svg",
            "parent": None,
            "pivot": [150, 160],
            "order": 0,
        },
    )


def _layers(roots) -> list[str]:
    """The names of the groups ``roots`` draw, in the order drawn."""
    return [
        m.name
        for root in roots
        for m in root.get_family()
        if m.name in ("backplate", "face", "lamp", "needle")
    ]


def test_a_part_only_the_destination_has_comes_in_at_its_own_layer(tmp_path):
    source = copied(tmp_path, "dial")
    destination = edited(copied(tmp_path, "dial", "backed-dial"), _backed)
    scene = tmp_path / "backed.py"
    scene.write_text(BACKED.format(source=str(source), destination=str(destination)))
    admission.DEMOTED.clear()
    session = Session.open(
        scene,
        "Backed",
        config={**SMALL, "frame_rate": 10, "media_dir": str(tmp_path / "media")},
    )
    try:
        record = session.record
        expected = ["backplate", "face", "lamp", "needle"]
        with Drawing(record) as drawing:
            for capture in record.plays[0].samples:
                drawing.show(capture, capture.roots)
                assert _layers(capture.roots) == expected
        assert _layers(session.scene.mobjects) == expected
    finally:
        session.close()
        admission.DEMOTED.clear()


def test_a_destination_already_in_the_scene_is_refused():
    from arachne import Scene

    box, crate = Rig("fixture:lidded-box"), Rig("fixture:lidded-crate")
    scene = Scene()
    scene.add(box, crate)
    with pytest.raises(ValueError, match="must not be there"):
        MorphParts(box, crate)._setup_scene(scene)


@pytest.fixture
def morphing():
    admission.DEMOTED.clear()
    session = Session.open(SCENES, "Morphing", config={**SMALL, "frame_rate": 10})
    yield session
    session.close()
    admission.DEMOTED.clear()


def test_the_destination_stands_where_the_source_stood(morphing):
    (crate,) = morphing.scene.mobjects
    assert crate.name == "crate"
    # Moved alike from where its own manifest places it, and posed after.
    reference = Rig("fixture:lidded-crate").shift(LEFT).pose(lid_open=80)
    for got, want in zip(
        crate.family_members_with_points(),
        reference.family_members_with_points(),
        strict=True,
    ):
        np.testing.assert_allclose(got.points, want.points, atol=1e-9)
