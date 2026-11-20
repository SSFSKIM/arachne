"""The scene's membership notice: a mobject whose class has a
``_scene_membership`` method and that registered itself is told when it enters or leaves the family of
the scene's mobjects, by whichever path, at the scene's instant, or with no
instant when scene time passed before the scene saw a change no method of
its reported.
"""

from __future__ import annotations

from fractions import Fraction

from arachne import FadeIn, FadeOut, Group, Mobject, Scene, Square
from arachne.scene.scene import _take_membership_notice


class Carrier(Mobject):
    """A mobject that takes the notice, recording each."""

    def __init__(self) -> None:
        super().__init__()
        self.told: list[tuple[bool, Fraction | None]] = []
        _take_membership_notice(self)

    def _scene_membership(self, scene, present, at):
        self.told.append((present, at))


def test_add_and_remove_tell_at_the_scene_s_instant(dry_run):
    scene, carrier = Scene(), Carrier()
    assert scene.add(carrier) is scene
    scene.wait(1)
    assert scene.remove(carrier) is scene
    assert carrier.told == [(True, 0), (False, 1)]


def test_a_member_of_a_group_is_told_with_its_group(dry_run):
    scene, carrier, other = Scene(), Carrier(), Square()
    group = Group(Group(carrier), other)
    scene.add(group)
    scene.wait(0.5)
    # Removing the carrier from a group that stays dissolves the group.
    scene.remove(carrier)
    assert other in scene.get_mobject_family_members()
    scene.wait(0.5)
    scene.add(group)
    scene.wait(0.5)
    scene.remove(group)
    assert carrier.told == [
        (True, 0),
        (False, Fraction(1, 2)),
        (True, 1),
        (False, Fraction(3, 2)),
    ]


def test_replace_clear_and_the_foreground_tell(dry_run):
    scene, carrier, other = Scene(), Carrier(), Square()
    scene.add(carrier)
    scene.replace(carrier, other)
    scene.wait(0.5)
    scene.replace(other, carrier)
    scene.wait(0.5)
    scene.clear()
    scene.wait(0.5)
    scene.add_foreground_mobject(carrier)
    scene.wait(0.5)
    scene.remove_foreground_mobject(carrier)
    scene.remove(carrier)
    assert carrier.told == [
        (True, 0),
        (False, 0),
        (True, Fraction(1, 2)),
        (False, 1),
        (True, Fraction(3, 2)),
        (False, 2),
    ]


def test_moving_a_member_within_the_scene_tells_nothing(dry_run):
    scene, carrier = Scene(), Carrier()
    scene.add(carrier, Square())
    scene.bring_to_front(carrier)
    scene.bring_to_back(carrier)
    scene.add(carrier)
    assert carrier.told == [(True, 0)]


def test_a_remove_and_an_add_at_one_instant_tell_both_at_it(dry_run):
    scene, carrier = Scene(), Carrier()
    scene.add(carrier)
    scene.wait(1)
    scene.remove(carrier)
    scene.add(carrier)
    assert carrier.told == [(True, 0), (False, 1), (True, 1)]


def test_an_animation_s_entry_and_clean_up_tell(dry_run):
    scene, carrier = Scene(), Carrier()
    carrier.add(Square())
    scene.play(FadeIn(carrier), run_time=0.5)
    scene.play(FadeOut(carrier), run_time=0.5)
    assert carrier.told == [(True, 0), (False, 1)]


def test_a_change_by_another_path_between_plays_is_told_at_its_instant(dry_run):
    scene, carrier = Scene(), Carrier()
    group = Group(Square())
    scene.add(group)
    scene.wait(0.5)
    group.add(carrier)
    scene.wait(0.5)
    scene.mobjects = []
    scene.wait(0.5)
    assert carrier.told == [(True, Fraction(1, 2)), (False, 1)]


def test_a_change_by_another_path_during_a_play_has_no_instant(dry_run):
    scene, carrier = Scene(), Carrier()
    scene.add(carrier)
    seen = []

    def drop(dt):
        if dt and not seen:
            seen.append(scene.time)
            scene.mobjects = []

    scene.add_updater(drop)
    scene.wait(0.5)
    assert seen
    assert carrier.told == [(True, 0), (False, None)]


def test_the_look_walks_the_engine_s_family_not_a_mobject_s(dry_run):
    class Hiding(Group):
        def get_family(self, recurse=True):
            return [self]

    scene, carrier = Scene(), Carrier()
    scene.add(Hiding(carrier))
    assert carrier.told == [(True, 0)]


def test_a_scene_with_nothing_that_takes_the_notice_only_checks(dry_run, monkeypatch):
    import weakref

    from arachne.scene import scene as scene_module

    def no_walk(mobject):
        raise AssertionError("walked a family with nothing to tell")

    monkeypatch.setattr(scene_module, "_NOTICE_TAKERS", weakref.WeakSet())
    monkeypatch.setattr(scene_module, "_family", no_walk)
    scene = Scene()
    scene.add(Group(Square()))
    scene.wait(0.5)
    scene.clear()


def test_only_a_registered_mobject_is_told(dry_run):
    """A mobject whose class has the method but that never registered is not
    told, whatever else in the process registered.
    """

    class Unregistered(Mobject):
        def __init__(self) -> None:
            super().__init__()
            self.told: list[tuple[bool, Fraction | None]] = []

        def _scene_membership(self, scene, present, at):
            self.told.append((present, at))

    scene, carrier, passive = Scene(), Carrier(), Unregistered()
    scene.add(carrier, passive)
    scene.wait(0.5)
    scene.remove(passive)
    assert carrier.told == [(True, 0)]
    assert passive.told == []
