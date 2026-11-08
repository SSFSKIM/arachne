"""A mobject's deep copy made by Arachne's own copier (``fast_copy``) is the
copy :func:`copy.deepcopy` makes: the same objects are new and the same are
shared, aliasing inside the original is kept, and the memo ends holding what
:func:`copy.deepcopy`'s would.
"""

from __future__ import annotations

import copy
import enum
import functools
from collections import Counter

import numpy as np
import pytest

from arachne import (
    BLUE,
    RED,
    RIGHT,
    UP,
    Arrow,
    Axes,
    BraceLabel,
    Circle,
    DecimalNumber,
    Dot,
    ImageMobject,
    Integer,
    Line,
    MathTex,
    NumberPlane,
    PMobject,
    Square,
    Tex,
    Text,
    ValueTracker,
    VGroup,
    VMobject,
    always_redraw,
)
from arachne.mobject import copying
from arachne.utils.testing.copy_graph import copy_differences, reachable


@pytest.fixture
def fast(config):
    config.fast_copy = True
    return config


def made_by(config, fast_copy: bool, copier, value):
    config.fast_copy = fast_copy
    try:
        return copier(value)
    finally:
        config.fast_copy = True


# ── what a mobject may hold ──────────────────────────────────────────────────


class Plain:
    """A plain Python object, which copy.deepcopy rebuilds from its reduce."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


class Slotted:
    __slots__ = ("a", "b")

    def __init__(self, a, b):
        self.a, self.b = a, b


class Reducing:
    def __init__(self, items):
        self.items = items

    def __reduce__(self):
        return (Reducing, (self.items,))


class OwnCopy:
    """Its own __deepcopy__, which shares one field and copies the other."""

    def __init__(self, kept, copied):
        self.kept, self.copied = kept, copied

    def __deepcopy__(self, memo):
        return OwnCopy(self.kept, copy.deepcopy(self.copied, memo))


class Looking:
    """An attribute lookup that runs the class's own code."""

    def __init__(self):
        self.data = [1, 2]

    def __getattr__(self, name):
        raise AttributeError(name)


class Listed(list):
    pass


class Keyed(dict):
    pass


class Shade(enum.Enum):
    LIGHT = 1
    DARK = 2


class Holder(VMobject):
    """A user subclass holding values of every kind copy.deepcopy treats apart."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shared_list = [1, 2, 3]
        self.plain = Plain(xs=[1.0, 2.0], nested=Plain(tag="n"), empty=Plain())
        self.again = self.plain.nested
        self.plain.nested.back = self.plain  # a cycle through plain objects
        self.slotted = Slotted([1], (2, [3]))
        self.reducing = Reducing([Plain(v=1)])
        self.own = OwnCopy([1], [2])
        self.looking = Looking()
        self.listed = Listed([1, [2]])
        self.keyed = Keyed(a=[1])
        self.shade = Shade.DARK
        self.array = np.arange(6.0).reshape(2, 3)
        self.view = self.array[0]
        self.scalar = np.float64(2.5)
        self.number = 3, 4.5, "five", None, True, 2j, b"six"
        self.mixed = (1, shared_list)
        self.aliases = [shared_list, shared_list, {"again": shared_list}]
        self.sets = ({1, 2}, frozenset({3}))
        self.things = (range(3), int, len, lambda x: x, functools.partial(max, 1))
        self.method = self.get_center
        self.myself = self
        self.by_mobject = {Dot(): "a dot"}
        self.add(Square(), Circle())
        self.pointer = self.submobjects[0]


def updated() -> Dot:
    dot = Dot()
    dot.add_updater(lambda m, dt: m)
    dot.add_updater(dot.get_center)
    return dot


def targeted() -> Square:
    square = Square().set_fill(RED, 0.5).set_stroke(BLUE, 3)
    square.generate_target().shift(UP)
    return square


def own_copier() -> VGroup:
    square = Square()
    square.__dict__["__deepcopy__"] = lambda memo: "an instance's own copier"
    return VGroup(square)


#: What each case of the gallery builds.
MAKERS = {
    "Square": Square,
    "VGroup": lambda: VGroup(Circle(), VGroup(Square(), Line())),
    "Arrow": Arrow,
    "Axes": lambda: Axes().add(Axes().plot(lambda x: x**2)),
    "NumberPlane": NumberPlane,
    "DecimalNumber": lambda: DecimalNumber(3.14159),
    "Integer": lambda: Integer(42),
    "MathTex": lambda: MathTex(r"\int_0^1 x^2\,dx", "=", r"\frac{1}{3}"),
    "Tex": lambda: Tex("Two words"),
    "Text": lambda: Text("hello world"),
    "BraceLabel": lambda: BraceLabel(Square(), "label"),
    "ValueTracker": lambda: ValueTracker(2),
    "Updaters": updated,
    "Target": targeted,
    "Redraw": lambda: always_redraw(lambda: Dot()),
    "PMobject": lambda: PMobject().add_points(np.zeros((4, 3))),
    "ImageMobject": lambda: ImageMobject(np.zeros((4, 4, 3), dtype=np.uint8)),
    "Holder": Holder,
    "InstanceCopier": own_copier,
}


GALLERY = sorted(MAKERS)


@pytest.mark.parametrize("name", GALLERY)
@pytest.mark.parametrize(
    "copier",
    [lambda m: m.copy(), copy.deepcopy, lambda m: copy.deepcopy([m, m, {"k": m}])],
    ids=["copy", "deepcopy", "in a container"],
)
def test_the_copy_is_copy_deepcopys(fast, name, copier):
    original = MAKERS[name]()
    reference = made_by(fast, False, copier, original)
    made = made_by(fast, True, copier, original)
    opaque: Counter[str] = Counter()
    assert copy_differences(original, reference, made, opaque=opaque) == []
    # The only objects compared by type alone are the partial in Holder.
    assert set(opaque) <= {"functools.partial"}


def test_generate_target_copies_as_copy_deepcopy_does(fast):
    original = MathTex("a^2")
    reference = made_by(fast, False, lambda m: m.generate_target(), original)
    made = made_by(fast, True, lambda m: m.generate_target(), original)
    original.target = None  # the target is the copy, not part of the original
    assert copy_differences(original, reference, made) == []


@pytest.mark.parametrize("name", GALLERY)
def test_the_memo_ends_as_copy_deepcopys(fast, name):
    """The memo maps the same originals, and keeps the same ones alive in
    the same order, so a caller that shares a memo across copies sees no
    difference. (A reduce's temporaries, new in every copy, are counted.)
    """
    original = MAKERS[name]()
    originals = reachable(original)
    memos = []
    for fast_copy in (False, True):
        memo: dict = {}
        made_by(fast, fast_copy, lambda m, memo=memo: copy.deepcopy(m, memo), original)
        alive = memo.pop(id(memo))
        memos.append(
            (
                sorted(key for key in memo if key in originals),
                len(memo),
                [id(x) for x in alive if id(x) in originals],
                len(alive),
            )
        )
    assert memos[0] == memos[1]


def test_a_memo_already_holding_the_mobject_returns_its_entry(fast):
    square, stand_in = Square(), Circle()
    memo = {id(square): stand_in}
    assert copy.deepcopy([square], memo)[0] is stand_in


# ── a class's copy hooks changed while the scene runs ────────────────────────


def _shares_its_data(self, memo):
    """A __deepcopy__ that shares what the class holds."""
    new = Plain()
    new.__dict__.update(self.__dict__)
    return new


def _rebuilds_from_a_list(self):
    return (Plain, (), {"rebuilt": list(self.__dict__)})


def _states_only_its_keys(self):
    return {"keys": sorted(self.__dict__)}


def _sets_state_in_reverse(self, state):
    self.__dict__.update(dict(reversed(list(state.items()))), restored=True)


def _reduces_by_registration(obj):
    return (Plain, (), {"registered": True})


def _copies_as_none(self, memo):
    return None


#: Each hook a scene may give its class after the class was first copied:
#: (the class it is set on, the hook, its new value).
HOOKS = {
    "mobject __deepcopy__": ("mobject", "__deepcopy__", _copies_as_none),
    "__deepcopy__": ("plain", "__deepcopy__", _shares_its_data),
    "__reduce_ex__": (
        "plain",
        "__reduce_ex__",
        lambda self, protocol: _rebuilds_from_a_list(self),
    ),
    "__reduce__": ("plain", "__reduce__", _rebuilds_from_a_list),
    "__getstate__": ("plain", "__getstate__", _states_only_its_keys),
    "__setstate__": ("plain", "__setstate__", _sets_state_in_reverse),
    "copyreg.dispatch_table": ("plain", None, _reduces_by_registration),
}


@pytest.mark.parametrize("hook", sorted(HOOKS))
def test_a_hook_changed_after_the_first_copy_is_the_one_used(fast, monkeypatch, hook):
    """A scene's own classes are copied by copy.deepcopy's rules as they are
    when the copy is made, not as they were at the class's first copy.
    """
    import copyreg

    class Part(VMobject):
        pass

    class Note:
        def __init__(self):
            self.text, self.marks = "note", [1, 2]

    def build():
        group = VGroup(Part(), Square())
        group.note = Note()
        group.submobjects[0].note = Note()
        return group

    original = build()
    made_by(fast, True, copy.deepcopy, original)  # each class's first copy
    where, name, value = HOOKS[hook]
    cls = Part if where == "mobject" else Note
    if name is None:
        monkeypatch.setitem(copyreg.dispatch_table, cls, value)
    else:
        monkeypatch.setattr(cls, name, value, raising=False)
    reference = made_by(fast, False, copy.deepcopy, original)
    made = made_by(fast, True, copy.deepcopy, original)
    assert copy_differences(original, reference, made) == []

    # And back: the hook removed, the class is copied as before.
    monkeypatch.undo()
    reference = made_by(fast, False, copy.deepcopy, original)
    made = made_by(fast, True, copy.deepcopy, original)
    assert copy_differences(original, reference, made) == []


# ── the user's code changing the copy in place ───────────────────────────────


def made_by_off(mob):
    from arachne import config

    config.fast_copy = False
    try:
        return copy.deepcopy(mob)
    finally:
        config.fast_copy = True


snapshot = made_by_off


@pytest.mark.parametrize(
    "name", ["VGroup", "MathTex", "Text", "DecimalNumber", "Holder"]
)
def test_changing_the_copy_in_place_leaves_the_original_as_it_was(fast, name):
    original = MAKERS[name]()
    before = snapshot(original)
    clone = original.copy()

    # Points: the engine's own changes, and a write into the array, which a
    # read-only value refuses.
    clone.shift(RIGHT).rotate(1.0)
    for leaf in clone.family_members_with_points():
        with pytest.raises(ValueError):
            leaf.points[0, 0] = 99.0
    # Colours, through the API and into what a colour holds.
    clone.set_color(RED).set_fill(BLUE, 0.3).set_stroke(width=7)
    for mob in clone.get_family():
        colour = mob.__dict__.get("background_stroke_color")
        if colour is not None:
            colour._internal_value[:] = 0.25
    # The submobject list, and what the copy's parts keep.
    clone.submobjects.append(Dot())
    clone.submobjects.reverse()
    clone.submobjects[-1].submobjects.clear()
    for mob in clone.get_family():
        mob.updaters.append(lambda m: m)
        path = mob.__dict__.get("path_obj")
        if path is not None and len(path):
            path[-1].end.x += 1000
    if name == "Holder":
        clone.plain.xs.append(3.0)
        clone.plain.nested.tag = "changed"
        clone.slotted.a.append(2)
        clone.aliases[0].append(4)
        clone.array[0, 0] = -1.0
        clone.listed[1].append(3)

    # The original's copy now and before, compared value by value with
    # nothing assumed shared, are the same: the original holds what it held.
    assert copy_differences(object(), before, snapshot(original)) == []


# ── the lever ────────────────────────────────────────────────────────────────


def counting(monkeypatch) -> list[int]:
    calls: list[int] = []
    real = copying.copy_mobject

    def counted(mob, memo):
        calls.append(1)
        return real(mob, memo)

    monkeypatch.setattr("arachne.mobject.mobject.copy_mobject", counted)
    return calls


def test_the_lever_makes_the_copy_with_the_copier(config, monkeypatch):
    calls = counting(monkeypatch)
    config.fast_copy = True
    VGroup(Square(), Circle()).copy()
    assert calls  # the root; what it holds is copied inside it
    assert len(calls) == 1


@pytest.mark.parametrize(
    "settings", [{"fast_copy": False}, {"speed_levers": False}], ids=["lever", "switch"]
)
def test_off_the_copy_is_copy_deepcopys_own(config, monkeypatch, settings):
    calls = counting(monkeypatch)
    for key, value in settings.items():
        setattr(config, key, value)
    VGroup(Square(), Circle()).copy()
    assert calls == []


# ── the check itself ─────────────────────────────────────────────────────────


def test_the_check_finds_a_copy_that_shares_or_copies_what_the_reference_does_not():
    square = Square()
    reference = made_by_off(square)
    sharing = made_by_off(square)
    sharing.submobjects = square.submobjects
    assert copy_differences(square, reference, sharing) == [
        "<__dict__>['submobjects']: shares list that the reference copies"
    ]
    copying_paint = made_by_off(square)
    copying_paint._paint = square._paint.replace()
    assert copy_differences(square, reference, copying_paint) == [
        "<__dict__>['_paint']: copies Paint that the reference shares"
    ]


def test_the_check_finds_aliasing_the_reference_does_not_have():
    group = VGroup(Square(), Square())
    reference = made_by_off(group)
    aliased = made_by_off(group)
    aliased.submobjects[1] = aliased.submobjects[0]
    found = copy_differences(group, reference, aliased)
    assert found
    assert "aliases differently" in found[0]


def test_the_check_finds_a_different_value():
    dot = Dot()
    reference = made_by_off(dot)
    moved = made_by_off(dot).shift(RIGHT)
    assert copy_differences(dot, reference, moved)
