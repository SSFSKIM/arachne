"""Scenes whose callbacks or mobjects look admissible but are not: each
changes what a frame shows depending on which instants are evaluated, or
runs scene code during an evaluation, so its play must tick.
"""

from __future__ import annotations

import collections
import functools
import operator
import random
import re

import numpy as np

from arachne import (
    BLUE,
    LEFT,
    RED,
    RIGHT,
    UP,
    AnimationGroup,
    Axes,
    Dot,
    Mobject,
    NumberLine,
    Rotate,
    Scene,
    Square,
    Transform,
    UpdateFromAlphaFunc,
    ValueTracker,
    VGroup,
    ZoomedScene,
    always_redraw,
    color_to_rgb,
    linear,
    random_color,
)
from tests.timeline import rebind_shared


class _Ramp(Scene):
    """A dot whose updater runs while a tracker ramps over one second."""

    def attach(self, dot: Dot, t: ValueTracker) -> None:
        raise NotImplementedError

    def construct(self):
        t = ValueTracker(0)
        dot = Dot()
        self.add(dot)
        self.attach(dot, t)
        self.play(t.animate.set_value(1))


class ArrayAugmented(_Ramp):
    """An alias of a closure array changed in place: it drifts by one step
    per evaluation.
    """

    def attach(self, dot, t):
        p = np.zeros(3)

        def step(m):
            q = p
            q += 0.02 * RIGHT
            m.move_to(q)

        dot.add_updater(step)


class ArrayOutKeyword(_Ramp):
    """A ufunc writing into a closure array through ``out=``."""

    def attach(self, dot, t):
        p = np.zeros(3)
        dot.add_updater(lambda m: m.move_to(np.add(p, 0.02 * RIGHT, out=p)))


class ArrayOutPositional(_Ramp):
    """The same write, with the output array passed positionally."""

    def attach(self, dot, t):
        p = np.zeros(3)
        dot.add_updater(lambda m: m.move_to(np.add(p, 0.02 * RIGHT, p)))


class _OutRead(Scene):
    """A dt-free updater writes a closure array through an argument NumPy
    writes its result into; a dt updater integrates the array's item, and
    an idle updater keeps the play stepping.
    """

    def writer(self, values: np.ndarray):
        raise NotImplementedError

    def construct(self):
        values = np.array([1.0])
        producer = Dot(radius=0.05)
        producer.add_updater(self.writer(values))
        output = Dot(color=RED, radius=0.05)
        output.add_updater(lambda m, dt: m.shift(dt * values[0] * RIGHT))
        idle = Dot(radius=0.05)
        idle.add_updater(lambda m: None)
        self.add(producer, output, idle)
        self.wait(1)


class ArrayOutPositionalRead(_OutRead):
    """``np.add(values, step, values)``: past the ufunc's two inputs."""

    def writer(self, values):
        return lambda m: np.add(values, 0.01, values)


class SecondOutRead(_OutRead):
    """The second output of a two-output ufunc, by position."""

    def writer(self, values):
        spare = np.zeros(1)
        return lambda m: np.divmod(values + 7.01, 7.0, spare, values)


class HeldUfuncOutRead(_OutRead):
    """The ufunc called under a name of the scene's own."""

    def writer(self, values):
        add = np.add
        return lambda m: add(values, 0.01, values)


class UfuncAtRead(_OutRead):
    """``np.add.at``: a ufunc's method writing its first argument in place."""

    def writer(self, values):
        return lambda m: np.add.at(values, 0, 0.01)


class FunctionOutRead(_OutRead):
    """``np.clip``'s fourth parameter, ``out``, given by position."""

    def writer(self, values):
        return lambda m: np.clip(values + 0.01, 0, 9, values)


class MethodOutRead(_OutRead):
    """An array's method on a local array the check can't resolve, with
    its ``out`` by position.
    """

    def writer(self, values):
        def step(m):
            moved = values + 0.01
            moved.clip(0, 9, values)

        return step


class BoundMethodOutRead(_OutRead):
    """The closure array's own method writing into it: ``values.dot(scale,
    values)``.
    """

    def writer(self, values):
        scale = np.array([[1.01]])
        return lambda m: values.dot(scale, values)


class StarredOutRead(_OutRead):
    """The arguments unpacked from a tuple: which lands in ``out`` is not
    in the source.
    """

    def writer(self, values):
        arguments = (values, 0.01, values)
        return lambda m: np.add(*arguments)


class KeywordsOutRead(_OutRead):
    """``out`` unpacked from a dict of keywords."""

    def writer(self, values):
        keywords = {"out": values}
        return lambda m: np.add(values, 0.01, **keywords)


class InputsOnlyRead(_OutRead):
    """The positive control: the same calls given only inputs, their results
    put on the updater's own mobject; the closure array never changes.
    """

    def writer(self, values):
        scale = np.array([[1.01]])
        add = np.add

        def step(m):
            moved = values + 0.01
            m.set_x(
                np.add(values, 0.01)[0]
                + add(values, 0.01)[0]
                + np.divmod(values, 7.0)[1][0]
                + np.clip(values, 0, 9)[0]
                + values.sum(0)
                + values.dot(scale)[0]
                + moved.clip(0, 9)[0]
            )

        return step


class _SharedRead(Scene):
    """A dt-free updater changes a dot the scene shows, through some path
    a narrowed write check must still see; a dt updater integrates the
    dot's x, and an idle updater keeps the play stepping.
    """

    def writer(self, shared: Dot):
        raise NotImplementedError

    def construct(self):
        shared = Dot(radius=0.05).move_to(RIGHT)
        producer = Dot(radius=0.05)
        producer.add_updater(self.writer(shared))
        output = Dot(color=RED, radius=0.05)
        output.add_updater(lambda m, dt: m.shift(dt * shared.get_x() * UP))
        idle = Dot(radius=0.05)
        idle.add_updater(lambda m: None)
        self.add(shared, producer, output, idle)
        self.wait(1)


class MadeGroupRead(_SharedRead):
    """A function of the scene's makes a group holding the shown dot and
    hands it back; the callback moves the group.
    """

    def writer(self, shared):
        def labelled(size):
            return VGroup(shared, Dot(radius=size))

        return lambda m: labelled(0.05).shift(0.01 * RIGHT)


class InlineGroupRead(_SharedRead):
    """A group made in the callback around the shown dot, moved at once."""

    def writer(self, shared):
        return lambda m: VGroup(shared, Dot()).shift(0.01 * RIGHT)


class MadeListRead(_SharedRead):
    """A list made in the callback holding the shown dot; its item moved."""

    def writer(self, shared):
        def step(m):
            parts = [shared]
            parts[0].shift(0.01 * RIGHT)

        return step


class MadeClassRead(_SharedRead):
    """An object of the scene's own class, whose constructor puts the shown
    dot into it; the callback moves the object.
    """

    def writer(self, shared):
        class Holder(VGroup):
            def __init__(self):
                super().__init__(shared, Dot())

        return lambda m: Holder().shift(0.01 * RIGHT)


class MadeSetterRead(_SharedRead):
    """A group made around the shown dot, resized through a property: the
    engine's setter scales what it holds.
    """

    def writer(self, shared):
        def step(m):
            group = VGroup(shared)
            group.width = group.width * 1.01

        return step


class CopiedDictRead(_SharedRead):
    """A shallow copy of a dict holding the shown dot; the copy's item moved."""

    def writer(self, shared):
        registry = {"dot": shared}

        def step(m):
            found = registry.copy()
            found["dot"].shift(0.01 * RIGHT)

        return step


class ArgumentRead(_SharedRead):
    """A function of the scene's that moves what it is handed, handed the
    shown dot.
    """

    def writer(self, shared):
        def nudge(mob):
            mob.shift(0.01 * RIGHT)

        return lambda m: nudge(shared)


class MemoValueRead(_SharedRead):
    """A memo dict whose value is the shown dot, moved through the dict."""

    def writer(self, shared):
        cache = {"dot": shared}
        return lambda m: cache["dot"].shift(0.01 * RIGHT)


class MemoHelperRead(_SharedRead):
    """A helper handing back the memo dict's value, not a copy of it."""

    def writer(self, shared):
        cache = {"dot": shared}

        def cached():
            return cache["dot"]

        return lambda m: cached().shift(0.01 * RIGHT)


class OutsideMemberRead(_SharedRead):
    """A group the shown dot was added to from outside, moved."""

    def writer(self, shared):
        group = VGroup()
        group.add(shared)
        return lambda m: group.shift(0.01 * RIGHT)


class ByNameMethodRead(_SharedRead):
    """The shown dot's method fetched by ``getattr`` and called."""

    def writer(self, shared):
        return lambda m: getattr(shared, "shift")(0.01 * RIGHT)  # noqa: B009


class GrownGroupRead(_SharedRead):
    """A group made around the shown dot, grown by ``add`` (its own list),
    then moved: the move reaches the dot.
    """

    def writer(self, shared):
        def step(m):
            group = VGroup(shared)
            group.add(Dot())
            group.shift(0.01 * RIGHT)

        return step


class MadeGroupSuspends(_SharedRead):
    """A group made around the shown dot, whose updating it suspends: that
    reaches the dot's own updaters, a change of structure.
    """

    def writer(self, shared):
        return lambda m: VGroup(shared).suspend_updating()


class LoopLocalRead(_SharedRead):
    """A loop's variable, bound to each item of a list holding the shown
    dot, moved.
    """

    def writer(self, shared):
        held = [shared]

        def step(m):
            for each in held:
                each.shift(0.01 * RIGHT)

        return step


class PickedLocalRead(_SharedRead):
    """A local bound to an item of a list holding the shown dot, moved."""

    def writer(self, shared):
        held = [shared]

        def step(m):
            pick = held[0]
            pick.shift(0.01 * RIGHT)

        return step


class MadeAroundArgumentRead(_SharedRead):
    """A function of the scene's makes a group around what it is handed and
    moves it, handed the shown dot.
    """

    def writer(self, shared):
        def nudge(mob):
            VGroup(mob).shift(0.01 * RIGHT)

        return lambda m: nudge(shared)


class LoopOwnerRead(_SharedRead):
    """The updater's own parameter rebound, by a loop, to the shown dot."""

    def writer(self, shared):
        held = [shared]

        def step(m):
            for m in held:  # noqa: B020, PLR1704
                m.shift(0.01 * RIGHT)

        return step


class ComprehensionOwnerRead(_SharedRead):
    """A comprehension whose variable has the updater's parameter's name,
    bound to the shown dot.
    """

    def writer(self, shared):
        held = [shared]
        return lambda m: [m.shift(0.01 * RIGHT) for m in held]


class LambdaOwnerRead(_SharedRead):
    """A lambda whose parameter has the updater's parameter's name, run on
    the shown dot.
    """

    def writer(self, shared):
        held = [shared]
        return lambda m: list(map(lambda m: m.shift(0.01 * RIGHT), held))  # noqa: C417


class _Mover:
    """A helper of the scene's whose static method moves what it is handed."""

    @staticmethod
    def move(mob):
        mob.shift(0.01 * RIGHT)


class StaticMethodRead(_SharedRead):
    """A held helper's static method, handed the shown dot."""

    def writer(self, shared):
        helper = _Mover()
        return lambda m: helper.move(shared)


class SuperInitRead(_SharedRead):
    """An object of a class of the scene's whose parent's constructor, run
    through ``super()``, moves the shown dot.
    """

    def writer(self, shared):
        class Base(VGroup):
            def __init__(self):
                super().__init__()
                shared.shift(0.01 * RIGHT)

        class Child(Base):
            def __init__(self):
                super().__init__()

        return lambda m: Child()


class DrawnItemRead(_SharedRead):
    """A generator the callback seeds draws the shown dot from an array
    holding it; the draw is moved.
    """

    def writer(self, shared):
        pool = np.empty(1, dtype=object)
        pool[0] = shared

        def step(m):
            rng = np.random.default_rng(0)
            rng.choice(pool).shift(0.01 * RIGHT)

        return step


class MethodReturnRead(_SharedRead):
    """A held helper's method hands back the shown dot, which is moved."""

    def writer(self, shared):
        class Holder:
            def __init__(self):
                self.target = shared

            def pick(self):
                return self.target

        holder = Holder()

        def step(m):
            found = holder.pick()
            found.shift(0.01 * RIGHT)

        return step


class AttributeByNameRead(_SharedRead):
    """The shown dot read by ``getattr`` with a constant name, then moved."""

    def writer(self, shared):
        class Holder:
            pass

        holder = Holder()
        holder.target = shared

        def step(m):
            found = getattr(holder, "target")  # noqa: B009
            found.shift(0.01 * RIGHT)

        return step


class LargestRead(_SharedRead):
    """``max`` hands back one of its arguments' items: the shown dot."""

    def writer(self, shared):
        held = [shared, Dot(radius=0.01)]

        def step(m):
            found = max(held, key=lambda d: d.width)
            found.shift(0.01 * RIGHT)

        return step


class StarTailRead(_SharedRead):
    """The shown dot handed after an unpacked argument, to the parameter a
    function of the scene's moves.
    """

    def writer(self, shared):
        def nudge(offset, target):
            target.shift(offset)

        prefix = (0.01 * RIGHT,)
        return lambda m: nudge(*prefix, shared)


class _HandedCallableRead(_SharedRead):
    """A function of the scene's moving the shown dot, handed to a call that
    calls it in turn.
    """

    def handing(self, move):
        raise NotImplementedError

    def writer(self, shared):
        def move(value):
            shared.shift(0.01 * RIGHT)
            return "x"

        return self.handing(move)


class PatternReplacementRead(_HandedCallableRead):
    """A compiled pattern's ``sub``, given a replacement function."""

    def handing(self, move):
        pattern = re.compile("x")
        return lambda m: pattern.sub(move, "x")


class SortKeyRead(_HandedCallableRead):
    """``sorted``'s key."""

    def handing(self, move):
        return lambda m: sorted(["a", "b"], key=move)


class MappedRead(_HandedCallableRead):
    """``map``'s function, run as the map is read."""

    def handing(self, move):
        return lambda m: list(map(move, [1]))


class AlongAxisRead(_HandedCallableRead):
    """``np.apply_along_axis``'s function."""

    def handing(self, move):
        return lambda m: np.apply_along_axis(move, 0, np.zeros(1))


class DictHeldCallableRead(_HandedCallableRead):
    """``sorted``'s key, read out of a dict: the check can't resolve it."""

    def handing(self, move):
        table = {"key": move}
        return lambda m: sorted(["a", "b"], key=table["key"])


class ReturnedCallableRead(_HandedCallableRead):
    """``sorted``'s key, handed back by a function of the scene's that reads
    it out of a dict: the check can't resolve it.
    """

    def handing(self, move):
        table = {"key": move}

        def pick():
            return table["key"]

        return lambda m: sorted(["a", "b"], key=pick())


class EngineAppliedCallableRead(_HandedCallableRead):
    """A function read out of a dict, handed to the engine's
    ``apply_function``, whose code calls it on the points.
    """

    def handing(self, move):
        def keep(points):
            move(points)
            return points

        table = {"f": keep}
        return lambda m: m.apply_function(table["f"])


class NativeUnlistedCallableRead(_HandedCallableRead):
    """A function read out of a dict, made a ``defaultdict``'s factory, which
    a missing key's lookup calls: a native listed neither way.
    """

    def handing(self, move):
        table = {"f": lambda: move(None)}
        return lambda m: collections.defaultdict(table["f"])["missing"]


class KeywordForwardedCallableRead(_HandedCallableRead):
    """A function read out of a dict, handed by keyword to a function that
    hands its ``**kwargs`` on to one handing that keyword's parameter to
    ``sorted`` as its key.
    """

    def handing(self, move):
        def inner(scale=1.0, act=None):
            sorted(["a", "b"], key=act)

        def outer(**kwargs):
            inner(**kwargs)

        table = {"f": move}
        return lambda m: outer(act=table["f"])


class _Runner:
    def run(self, function):
        function(None)


class UnknownCalleeCallableRead(_HandedCallableRead):
    """A function read out of a dict, handed to a method of an object read
    out of a list: the check can't tell what the method does with it.
    """

    def handing(self, move):
        runners = [_Runner()]
        table = {"f": move}
        return lambda m: runners[0].run(table["f"])


class _PiecewiseCallableRead(_SharedRead):
    """A function of the scene's moving the shown dot and giving a number,
    held in something handed to ``np.piecewise`` as its functions.
    """

    def handing(self, move):
        raise NotImplementedError

    def writer(self, shared):
        def move(value):
            shared.shift(0.01 * RIGHT)
            return 0.0

        return self.handing(move)


def _piecewise(functions):
    return np.piecewise(np.array([1.0]), [np.array([True])], functions)


class ListedCallableRead(_PiecewiseCallableRead):
    """A list the check resolves, holding the function."""

    def handing(self, move):
        functions = [move]
        return lambda m: np.piecewise(np.array([1.0]), [np.array([True])], functions)


class DisplayedCallableRead(_PiecewiseCallableRead):
    """A list written out at the call, holding the function."""

    def handing(self, move):
        return lambda m: np.piecewise(np.array([1.0]), [np.array([True])], [move])


class UnpackedCallableRead(_HandedCallableRead):
    """A pattern's ``sub``, its replacement in a dict written out at the
    call and unpacked as its keywords.
    """

    def handing(self, move):
        pattern = re.compile("x")
        return lambda m: pattern.sub(**{"repl": move, "string": "x"})


class NamedUnpackedCallableRead(_HandedCallableRead):
    """A pattern's ``sub``, its replacement in a dict the check resolves,
    unpacked as its keywords.
    """

    def handing(self, move):
        pattern = re.compile("x")
        keywords = {"repl": move, "string": "x"}
        return lambda m: pattern.sub(**keywords)


class RepeatedCallableRead(_PiecewiseCallableRead):
    """A function read out of a dict, handed by keyword through a
    ``**kwargs`` to a function repeating it into the list it hands
    ``np.piecewise``: a sequence an operator makes holds what its operands
    hold.
    """

    def handing(self, move):
        def inner(act=None):
            _piecewise([act] * 1)

        def outer(**kwargs):
            inner(**kwargs)

        table = {"f": move}
        return lambda m: outer(act=table["f"])


class ReturnedListCallableRead(_PiecewiseCallableRead):
    """A function of the scene's handing back a list holding the function:
    what it returns is no number, so the list is what it may hold.
    """

    def handing(self, move):
        def picked():
            return [move]

        return lambda m: _piecewise(picked())


def _direction(angle):
    return np.array([np.cos(angle), np.sin(angle), 0])


class ComputedPointMove(_SharedRead):
    """A control: a point a function of the scene's computes from numbers,
    in arithmetic handed to the engine's ``move_to``: every value is shown
    a number or an array of them, and the updater is stepped past.
    """

    def writer(self, shared):
        centre = np.array([0.5, 0.0, 0.0])
        offsets = [0.1, 0.2]
        return lambda m: m.move_to(
            centre + offsets[0] * _direction(len(offsets) * 0.3) + 0.2 * LEFT
        )


class UnresolvedPointMove(_SharedRead):
    """The control: a point read out of a dict, handed to the engine's
    ``move_to`` on the updater's own dot, whose code never calls it.
    """

    def writer(self, shared):
        table = {"p": 0.5 * LEFT}
        return lambda m: m.move_to(table["p"])


class MappedEngineMethodRead(_SharedRead):
    """The engine's ``Mobject.shift``, mapped over the shown dot."""

    def writer(self, shared):
        shift = Mobject.shift
        return lambda m: list(map(shift, [shared], [0.01 * RIGHT]))


class MethodCallerRead(_SharedRead):
    """``operator.methodcaller("shift", …)``, mapped over the shown dot."""

    def writer(self, shared):
        return lambda m: list(
            map(operator.methodcaller("shift", 0.01 * RIGHT), [shared])
        )


class AliasingArray(np.ndarray):
    """An array whose ``+`` hands back itself."""

    def __add__(self, other):
        return self


class _ArraySum(Scene):
    """A dt-free updater stores into the sum of two arrays; a dt updater
    integrates the first array's item, and an idle updater keeps the play
    stepping.
    """

    def values(self):
        raise NotImplementedError

    def construct(self):
        values = self.values()
        offset = np.array([0.0])

        def produce(m):
            result = values + offset
            result[0] += 0.01

        producer = Dot(radius=0.05)
        producer.add_updater(produce)
        output = Dot(color=RED, radius=0.05)
        output.add_updater(lambda m, dt: m.shift(dt * values[0] * UP))
        idle = Dot(radius=0.05)
        idle.add_updater(lambda m: None)
        self.add(producer, output, idle)
        self.wait(1)


class AliasingSumRead(_ArraySum):
    """The sum is the first array itself, whose class's ``+`` hands it back."""

    def values(self):
        return np.array([1.0]).view(AliasingArray)


class NumPySum(_ArraySum):
    """The sum of two of NumPy's own arrays: a new array."""

    def values(self):
        return np.array([1.0])


class HookRebound(Scene):
    """A dt updater integrates the x of the dot its mobject's ``source``
    names; a dt-free updater moves another dot, ``other``, which an
    ``on_start`` hook may make the source.
    """

    def construct(self):
        source = Dot(radius=0.05).move_to(RIGHT)
        self.other = Dot(radius=0.05).move_to(2 * RIGHT)
        producer = Dot(radius=0.05)
        producer.add_updater(lambda m: self.other.shift(0.01 * RIGHT))
        self.output = Dot(color=RED, radius=0.05)
        self.output.source = source
        self.output.add_updater(lambda m, dt: m.shift(dt * m.source.get_x() * UP))
        idle = Dot(radius=0.05)
        idle.add_updater(lambda m: None)
        self.add(source, self.other, producer, self.output, idle)
        self.wait(1)


class MadeHelperMethodRead(_SharedRead):
    """An object of the scene's own class, made with nothing, whose method
    moves the shown dot: what the class's code reads, its methods' among it.
    """

    def writer(self, shared):
        class Pusher:
            def push(self):
                shared.shift(0.01 * RIGHT)

        def make():
            return Pusher()

        return lambda m: make().push()


class ConstantItemInPlaceRead(_SharedRead):
    """An engine constant held in a list, added to in place through its
    item (by zero: the constant is shared by the whole process): a write
    into an array no key names, which may be any a dt updater reads.
    """

    def writer(self, shared):
        held = [UP]

        def step(m):
            held[0] += 0

        return step


class PlacedArgumentRead(_SharedRead):
    """A function of the scene's places what it is handed into the group it
    hands back (beside a radius it only reads); the group is moved.
    """

    def writer(self, shared):
        def badge(center, radius=0.1):
            return VGroup(center, Dot(radius=radius))

        return lambda m: badge(radius=0.05, center=shared).shift(0.01 * RIGHT)


class StoredArgumentRead(_SharedRead):
    """A function of the scene's makes an object of the scene's class that
    stores what it is handed; the stored object is moved.
    """

    def writer(self, shared):
        class Holder:
            def __init__(self, target):
                self.target = target

        def make(size, target):
            return Holder(target)

        return lambda m: make(0.1, shared).target.shift(0.01 * RIGHT)


class GatheredArgumentsRead(_SharedRead):
    """A function of the scene's gathers its arguments into a group it
    hands back, handed the shown dot among them; the group is moved.
    """

    def writer(self, shared):
        def gather(*parts):
            return VGroup(*parts)

        return lambda m: gather(Dot(), shared).shift(0.01 * RIGHT)


class ReboundArgumentRead(_SharedRead):
    """A function of the scene's whose parameter, handed the shown dot, is
    rebound and moved under either binding.
    """

    def writer(self, shared):
        def nudge(mob, twice=False):
            if twice:
                mob = mob.copy()
            mob.shift(0.01 * RIGHT)

        return lambda m: nudge(shared)


class SharedRngRead(Scene):
    """A dt-free updater draws from a generator the scene made, which a dt
    updater draws from too: each draw changes what the other draws next.
    """

    def construct(self):
        rng = np.random.default_rng(1)
        producer = Dot(radius=0.05)
        producer.add_updater(lambda m: rng.uniform())
        output = Dot(color=RED, radius=0.05)
        output.add_updater(lambda m, dt: m.shift(dt * rng.random() * RIGHT))
        idle = Dot(radius=0.05)
        idle.add_updater(lambda m: None)
        self.add(producer, output, idle)
        self.wait(1)


LEVEL = 1.0


def _raise_level(m):
    global LEVEL
    LEVEL += 0.01


class _LevelRead(Scene):
    """A dt updater integrates a module global a dt-free updater rebinds."""

    def construct(self):
        global LEVEL
        LEVEL = 1.0
        rebind_shared.LEVEL = 1.0
        producer = Dot(radius=0.05)
        producer.add_updater(self.writer())
        output = Dot(color=RED, radius=0.05)
        output.add_updater(self.reader())
        idle = Dot(radius=0.05)
        idle.add_updater(lambda m: None)
        self.add(producer, output, idle)
        self.wait(1)


class GlobalRebindRead(_LevelRead):
    """``global LEVEL; LEVEL += …``."""

    def writer(self):
        return _raise_level

    def reader(self):
        return lambda m, dt: m.shift(dt * LEVEL * RIGHT)


def _raise_shared(m):
    rebind_shared.LEVEL = rebind_shared.LEVEL + 0.01


class ModuleStoreRead(_LevelRead):
    """``module.LEVEL = …``, read as ``module.LEVEL``."""

    def writer(self):
        return _raise_shared

    def reader(self):
        return lambda m, dt: m.shift(dt * rebind_shared.LEVEL * RIGHT)


class ModuleSetattrRead(_LevelRead):
    """``setattr(module, "LEVEL", …)``."""

    def writer(self):
        return lambda m: setattr(rebind_shared, "LEVEL", rebind_shared.LEVEL + 0.01)

    def reader(self):
        return lambda m, dt: m.shift(dt * rebind_shared.LEVEL * RIGHT)


class PrivateWorkRead(Scene):
    """The positive control: a redraw that builds its picture through a
    function of the scene's handing back a group it made, a memo dict of
    copies, a generator it seeds itself, ``getattr`` by a constant name and
    ``np.linalg.norm``, from a tracker a dt updater advances. None of it is
    read by stepping work, so the redraw waits for the frames.
    """

    def construct(self):
        phase = ValueTracker(0)
        phase.add_updater(lambda m, dt: m.increment_value(dt))
        cache = {}

        def badge(size):
            return VGroup(Dot(radius=size), Square(side_length=size))

        def cached(key):
            if key not in cache:
                cache[key] = badge(0.1)
            return cache[key].copy()

        def picture():
            rng = np.random.default_rng(3)
            jitter = rng.uniform(-0.01, 0.01)
            x = phase.get_value() + getattr(self, "offset", 0.0) + jitter
            length = np.linalg.norm(np.array([x, 1.0]))
            return VGroup(badge(0.05).shift(x * RIGHT), cached("a").shift(length * UP))

        self.add(phase, always_redraw(picture))
        self.wait(1)


_PAUSE = re.compile(r"\[pause\s+(\d+)\]")


class PatternRead(Scene):
    """A redraw that reads a module's compiled regular expression (as a
    scene's cue parser does), beside a dt updater: a pattern holds nothing
    that changes, so the plan is bounded and the redraw waits for the
    frames.
    """

    def construct(self):
        phase = ValueTracker(0)
        phase.add_updater(lambda m, dt: m.increment_value(dt))

        def marker():
            gaps = len(_PAUSE.findall("a [pause 1] b [pause 2] c"))
            return Dot().shift(gaps * phase.get_value() * RIGHT)

        self.add(phase, always_redraw(marker))
        self.wait(1)


class _ItemStores(Scene):
    """A dt updater storing into each item of ``parts`` (a dot at key or
    index 0), beside a redraw that reads none of it.
    """

    def stores(self, parts):
        clock = ValueTracker(0)

        def step(m, dt):
            m.increment_value(dt)
            for q in parts:
                q[0] = Dot().shift(m.get_value() * UP)

        clock.add_updater(step)
        self.add(clock, always_redraw(lambda: Square(side_length=0.2)))
        return clock


class MobjectItemStore(_ItemStores):
    """The list holds a group shown: ``group[0] = other`` replaces one of its
    members as stepping work runs.
    """

    def construct(self):
        group = VGroup(Dot(), Dot().shift(RIGHT))
        self.add(group)
        self.stores([group])
        self.wait(1)


class MembersListStore(_ItemStores):
    """The list holds a shown group's list of members: a store into it
    replaces one of them.
    """

    def construct(self):
        group = VGroup(Dot(), Dot().shift(RIGHT))
        self.add(group)
        self.stores([group.submobjects])
        self.wait(1)


class AttributeDictStore(Scene):
    """The list holds a shown group's attribute dict (``vars(group)``): a
    store of its ``submobjects`` replaces its members.
    """

    def construct(self):
        group = VGroup(Dot(), Dot().shift(RIGHT))
        parts = [vars(group)]
        clock = ValueTracker(0)

        def step(m, dt):
            m.increment_value(dt)
            for q in parts:
                q["submobjects"] = [Dot().shift(m.get_value() * UP)]

        clock.add_updater(step)
        self.add(group, clock, always_redraw(lambda: Square(side_length=0.2)))
        self.wait(1)


class MobjectJoinsDicts(_ItemStores):
    """The list holds dicts until an updater appends a group shown, half-way:
    the next store into its items replaces one of the group's members.
    """

    def construct(self):
        group = VGroup(Dot(), Dot().shift(RIGHT))
        parts: list = [{}]
        self.add(group)
        clock = self.stores(parts)

        def join(m, dt):
            if m.get_value() > 0.5 and len(parts) == 1:
                parts.append(group)

        clock.add_updater(join)
        self.wait(1)


class DictStateIntegrator(Scene):
    """A dt updater integrating particles whose state a list of dicts holds
    (``p, v = q["p"], q["v"]``, ``pn = p + v * dt``, ``pn[1] = …``,
    ``q["p"] = pn``), beside a redraw drawing them: the check can't tell
    that ``q`` is a dict nor that ``pn`` is a new array, so the play ticks
    everything (planned precision work, ``tech-debt-tracker.md``).
    """

    def construct(self):
        parts = [{"p": np.zeros(3), "v": np.array([1.0, 0.5, 0.0])} for _ in range(2)]
        clock = ValueTracker(0)

        def step(m, dt):
            for q in parts:
                p, v = q["p"], q["v"]
                pn = p + v * dt
                pn[1] = min(pn[1], 0.3)
                q["p"] = pn

        clock.add_updater(step)
        drawn = always_redraw(lambda: VGroup(*[Dot().move_to(q["p"]) for q in parts]))
        self.add(clock, drawn)
        self.wait(1)


def _place(group, centre):
    for dot, offset in zip(group, (LEFT, RIGHT), strict=True):
        dot.move_to(centre + offset)


class _Pair(VGroup):
    def __init__(self):
        super().__init__()
        self.add(Dot(), Dot())
        self.x = 0.0

    def drift(self, dt):
        self.x += dt
        _place(self, self.x * RIGHT)


class UnresolvedMethodRead(Scene):
    """A dt updater moving the members of a group of the scene's class
    through a helper's loop (``dot.move_to(…)``, ``dot`` unresolved), beside
    a redraw that reads none of it: the check can't tell that no code of the
    scene's runs as ``move_to``, and the class's code adds members, so the
    play ticks everything (a precision gap, ``tech-debt-tracker.md``).
    """

    def construct(self):
        pair = _Pair()
        pair.add_updater(lambda m, dt: m.drift(dt))
        self.add(pair, always_redraw(lambda: Square(side_length=0.2)))
        self.wait(1)


class PartialFill(Scene):
    """A fill set only between alpha 0.3 and 0.4, left as it was elsewhere."""

    def construct(self):
        d = Square(fill_opacity=1)
        self.add(d)
        self.play(
            UpdateFromAlphaFunc(
                d, lambda m, a: m.set_fill(RED if 0.3 < a < 0.4 else None)
            )
        )


class NoneFill(Scene):
    """A fill colour that is None: the setter leaves the colour as it was."""

    def construct(self):
        d = Square(fill_opacity=1)
        colour = None
        self.add(d)
        self.play(UpdateFromAlphaFunc(d, lambda m, a: m.set_fill(colour, a)))


class NarrowedColour(Scene):
    """``family=False``: outside the domain the setter's test covers."""

    def construct(self):
        d = Square()
        self.add(d)
        self.play(UpdateFromAlphaFunc(d, lambda m, a: m.set_color(RED, family=False)))


class RandomRate(Scene):
    """A callback animation timed by a rate function that draws at random."""

    def construct(self):
        d = Dot()
        self.add(d)
        self.play(
            AnimationGroup(
                UpdateFromAlphaFunc(
                    d, lambda m, a: m.set_x(a), rate_func=lambda t: random.random()
                )
            )
        )


class CallbackOverlap(Scene):
    """A Transform of ``d`` over 2 s while a callback part also sets ``d``
    for its first 0.8 s: CE stops the shorter part at the first frame after
    0.8 s, and the Transform overwrites from then on.
    """

    def construct(self):
        d = Dot()
        self.add(d)
        self.play(
            AnimationGroup(
                Transform(d, Dot(RIGHT), run_time=2),
                UpdateFromAlphaFunc(
                    Mobject(), lambda m, a: d.set_x(2 * a), run_time=0.8
                ),
            )
        )


class CountingGraph(Scene):
    """A dot on a graph whose function counts its calls."""

    def construct(self):
        calls = []

        def f(x):
            calls.append(1)
            return x * x / 4 + len(calls) / 100

        t = ValueTracker(0)
        axes = Axes(x_range=[0, 3], y_range=[0, 3], x_length=5, y_length=3)
        graph = axes.plot(lambda x: x * x / 4)
        graph.underlying_function = f
        dot = Dot()
        dot.add_updater(lambda m: m.move_to(axes.i2gp(t.get_value(), graph)))
        self.add(axes, graph, dot)
        self.play(t.animate.set_value(2))


class CountingDot(Dot):
    """A user's dot whose ``interpolate`` counts its calls."""

    calls: list[int] = []

    def interpolate(self, *args, **kwargs):
        CountingDot.calls.append(1)
        return super().interpolate(*args, **kwargs)


class CountedTransform(Scene):
    """A linear Transform of that dot: every type and function in it is the
    algebra's but the dot's own ``interpolate``.
    """

    def construct(self):
        d = CountingDot()
        self.add(d)
        self.play(Transform(d, Dot(RIGHT * 2)), rate_func=linear)


class ViewCentre(Scene):
    """``Rotate`` about a read-only view of an array the script changes
    after the play.
    """

    def construct(self):
        p = np.array([1.0, 0.0, 0.0])
        c = p.view()
        c.flags.writeable = False
        square = Square()
        self.add(square)
        self.play(Rotate(square, angle=np.pi / 2, about_point=c))
        p[0] = 5


class ZoomBackground(ZoomedScene):
    """The zoomed camera's background turns from red to blue at 1/2 s."""

    def __init__(self, **kwargs):
        super().__init__(
            zoom_factor=0.3, zoomed_display_height=3, zoomed_display_width=3, **kwargs
        )

    def construct(self):
        dot = Dot(UP * 0.1)
        self.add(dot)
        self.zoomed_camera.frame.move_to(dot)
        self.activate_zooming()
        self.zoomed_camera.background_color = RED
        self.play(dot.animate.shift(RIGHT * 0.1), run_time=0.5)
        self.zoomed_camera.background_color = BLUE
        self.play(dot.animate.shift(RIGHT * 0.1), run_time=0.5)


# What the prover once admitted by default.


class PutAlongAxis(_Ramp):
    """A numpy function that writes into the closure array it is given."""

    def attach(self, dot, t):
        p = np.zeros(3)

        def drift():
            np.put_along_axis(p, np.array([0]), p[:1] + 0.02, axis=0)
            return p

        dot.add_updater(lambda m: m.move_to(drift()))


class RandomColour(_Ramp):
    """An engine helper that draws from the random generator's hidden state."""

    def attach(self, dot, t):
        dot.add_updater(lambda m: m.set_color(random_color()))


class PickedNone(Scene):
    """A fill picked from ``[None, RED]``: None outside 0.3 < alpha < 0.4,
    which leaves the fill as it was.
    """

    def construct(self):
        d = Square(fill_opacity=1)
        self.add(d)
        self.play(
            UpdateFromAlphaFunc(
                d, lambda m, a: m.set_fill([None, RED][int(0.3 < a < 0.4)])
            )
        )


class BoundReceiver(Scene):
    """A graph whose function is a number line's ``n2p``: the dot reading the
    graph reads the line, whose own updater runs after the dot's.
    """

    def construct(self):
        t = ValueTracker(0)
        ax = Axes(x_range=[0, 3], y_range=[0, 3], x_length=5, y_length=3)
        graph = ax.plot(lambda x: x / 2)
        line = NumberLine(x_range=[0, 3], length=3)
        graph.function = line.n2p
        dot = Dot()
        dot.add_updater(lambda m: m.move_to(ax.i2gp(0, graph)))
        line.add_updater(lambda m: m.set_y(t.get_value()))
        self.add(ax, graph, dot, line)
        self.play(t.animate.set_value(2))


class InstanceInterpolate(Scene):
    """A linear Transform of a dot whose ``interpolate`` is replaced on the
    instance by one that counts its calls.
    """

    calls: list[int] = []

    def construct(self):
        d = Dot()
        original = d.interpolate

        def counted(*args, **kwargs):
            InstanceInterpolate.calls.append(1)
            return original(*args, **kwargs)

        d.interpolate = counted
        self.add(d)
        self.play(Transform(d, Dot(RIGHT * 2)), rate_func=linear)


class InstanceSetFill(_Ramp):
    """A group ``set_fill`` updater whose child's ``set_fill`` is replaced on
    the instance by one that also moves it a step per call.
    """

    def attach(self, dot, t):
        d = Square(fill_opacity=1)
        original = d.set_fill

        def stepping(*args, **kwargs):
            d.shift(0.02 * RIGHT)
            return original(*args, **kwargs)

        d.set_fill = stepping
        group = VGroup(d)
        group.add_updater(lambda m: m.set_fill(RED, t.get_value()))
        self.add(group)


# Found reviewing what else the prover admitted by default.


class InstanceAnimation(Scene):
    """A linear Transform whose own ``interpolate_mobject`` is replaced on
    the animation, counting its calls.
    """

    calls: list[int] = []

    def construct(self):
        d = Dot()
        animation = Transform(d, Dot(RIGHT * 2), rate_func=linear)
        original = animation.interpolate_mobject

        def counted(alpha):
            InstanceAnimation.calls.append(1)
            return original(alpha)

        animation.interpolate_mobject = counted
        self.add(d)
        self.play(animation)


class ShuffledBuilder(_Ramp):
    """A redraw that shuffles the group it builds: a draw from the random
    generator, at every evaluation.
    """

    def attach(self, dot, t):
        shown = always_redraw(
            lambda: (
                VGroup(*[Dot(radius=0.04 * k) for k in range(1, 6)]).shuffle().arrange()
            )
        )
        self.add(shown)


class InstanceUpdate(_Ramp):
    """A dot whose ``update``, which the scene calls at every tick, is
    replaced on the instance by one that also moves it a step.
    """

    def attach(self, dot, t):
        original = dot.update

        def stepping(dt=0, recursive=True):
            dot.shift(0.02 * RIGHT)
            return original(dt, recursive)

        dot.update = stepping


class PatchedEngine(Scene):
    """A linear Transform while the engine's ``Dot.interpolate`` is replaced
    by one that counts its calls (put back once the scene ends).
    """

    calls: list[int] = []

    def construct(self):
        original = Dot.interpolate

        def counted(mob, *args, **kwargs):
            PatchedEngine.calls.append(1)
            return original(mob, *args, **kwargs)

        Dot.interpolate = counted
        try:
            d = Dot()
            self.add(d)
            self.play(Transform(d, Dot(RIGHT * 2)), rate_func=linear)
        finally:
            del Dot.interpolate


class CountingFill(Dot):
    """A user's dot whose fill the camera reads from a class value the
    script rebinds, counting the reads.
    """

    colour = RED
    calls: list[int] = []

    def get_fill_rgbas(self):
        CountingFill.calls.append(1)
        return np.array([[*color_to_rgb(CountingFill.colour), 1.0]])


class OverriddenFill(Scene):
    """That dot drawn through two waits, red then blue: drawn after the
    scene ran, the first wait would show blue.
    """

    def construct(self):
        CountingFill.colour = RED
        d = CountingFill(radius=0.3)
        self.add(d)
        self.wait(0.5)
        CountingFill.colour = BLUE
        self.wait(0.5)


class FillJoinsMidPlay(Scene):
    """That dot, red, added by an updater partway through a play."""

    def construct(self):
        CountingFill.colour = RED
        t = ValueTracker(0)
        holder = VGroup()

        def join(m, dt):
            if not m.submobjects and t.get_value() > 0.4:
                m.add(CountingFill(radius=0.3))

        holder.add_updater(join)
        self.add(holder)
        self.play(t.animate.set_value(1), rate_func=linear)


# Identity, not metadata.


class WrappedInterpolate(Scene):
    """``InstanceInterpolate`` with its counter dressed as the engine's
    method by ``functools.wraps``.
    """

    calls: list[int] = []

    def construct(self):
        d = Dot()
        original = d.interpolate

        @functools.wraps(original)
        def counted(*args, **kwargs):
            WrappedInterpolate.calls.append(1)
            return original(*args, **kwargs)

        d.interpolate = counted
        self.add(d)
        self.play(Transform(d, Dot(RIGHT * 2)), rate_func=linear)


class WrappedPatch(Scene):
    """``PatchedEngine`` with its counter dressed as the engine's
    ``interpolate`` by ``functools.wraps``.
    """

    calls: list[int] = []

    def construct(self):
        original = Dot.interpolate

        @functools.wraps(original)
        def counted(mob, *args, **kwargs):
            WrappedPatch.calls.append(1)
            return original(mob, *args, **kwargs)

        Dot.interpolate = counted
        try:
            d = Dot()
            self.add(d)
            self.play(Transform(d, Dot(RIGHT * 2)), rate_func=linear)
        finally:
            del Dot.interpolate


# What an array holds.


class ObjectArrayFill(Scene):
    """``PickedNone`` with the choice made by an array of objects."""

    def construct(self):
        d = Square(fill_opacity=1)
        self.add(d)
        self.play(
            UpdateFromAlphaFunc(
                d,
                lambda m, a: m.set_fill(
                    np.array([None, RED], dtype="O")[int(0.3 < a < 0.4)]
                ),
            )
        )


# What drawing runs: protocols, and mobjects that come and go mid-play.


class HashDot(Dot):
    """A user's dot whose class defines ``__hash__``, counting its calls."""

    calls: list[int] = []

    def __hash__(self):
        HashDot.calls.append(1)
        return object.__hash__(self)


class HashedDot(Scene):
    """That dot moved, then held."""

    dot_class: type = HashDot

    def construct(self):
        d = self.dot_class(radius=0.3)
        self.add(d)
        self.play(d.animate.shift(RIGHT), rate_func=linear)
        self.wait(0.5)


def play_fill_between(scene, low, high):
    """A second's play during which an updater holds a ``CountingFill``
    while its tracker is in ``[low, high)``: the dot joins and leaves within
    the play.
    """
    t = ValueTracker(0)
    holder = VGroup()

    def show(m):
        inside = low <= t.get_value() < high
        if inside and not m.submobjects:
            m.add(CountingFill(radius=0.3))
        elif not inside and m.submobjects:
            m.remove(*m.submobjects)

    holder.add_updater(show)
    scene.add(holder)
    scene.play(t.animate.set_value(1), rate_func=linear)
    scene.remove(holder)


class TransientFill(Scene):
    """A red ``CountingFill`` that joins and leaves within play 0, then a
    wait.
    """

    def construct(self):
        CountingFill.colour = RED
        play_fill_between(self, 0.3, 0.6)
        self.wait(0.5)


class TwoTransientFills(Scene):
    """``TransientFill``'s play, a wait, and the same play in blue."""

    def construct(self):
        CountingFill.colour = RED
        play_fill_between(self, 0.3, 0.6)
        self.wait(0.5)
        CountingFill.colour = BLUE
        play_fill_between(self, 0.3, 0.6)


class CaughtTransientFill(Scene):
    """``TransientFill``'s play inside a ``try`` that catches every error."""

    def construct(self):
        CountingFill.colour = RED
        try:  # noqa: SIM105 (written as scenes write it)
            play_fill_between(self, 0.3, 0.6)
        except Exception:  # noqa: BLE001, S110 (a scene that swallows errors)
            pass
        self.wait(0.5)


class DriftingTransientFill(Scene):
    """``TransientFill`` beside a square placed by how many times the scene
    has run: each execution begins its play from another state.
    """

    runs: list[int] = []

    def construct(self):
        DriftingTransientFill.runs.append(1)
        CountingFill.colour = RED
        self.add(Square().shift(RIGHT * len(DriftingTransientFill.runs)))
        play_fill_between(self, 0.3, 0.6)


class ShiftingTransientFill(Scene):
    """``TransientFill``'s play after one more wait each time the scene runs:
    its one refusal is found at another play index in each execution.
    Stops at its fifth run, rather than run on.
    """

    runs: list[int] = []

    def construct(self):
        ShiftingTransientFill.runs.append(1)
        if len(ShiftingTransientFill.runs) > 4:
            raise RuntimeError("executed a fifth time")
        CountingFill.colour = RED
        for _ in ShiftingTransientFill.runs:
            self.wait(0.5)
        play_fill_between(self, 0.3, 0.6)


class SetUpTransientFill(Scene):
    """``TransientFill`` whose ``setup`` opens a resource its ``tear_down``
    closes, noting each step.
    """

    steps: list[str] = []
    open: list[int] = []

    def setup(self):
        SetUpTransientFill.steps.append("setup")
        SetUpTransientFill.open.append(1)

    def tear_down(self):
        SetUpTransientFill.steps.append("tear_down")
        SetUpTransientFill.open.pop()

    def construct(self):
        CountingFill.colour = RED
        play_fill_between(self, 0.3, 0.6)


class AnchoredTransientFill(Scene):
    """``TransientFill`` reading an attribute its caller set on the scene
    after making it, before rendering it.
    """

    def construct(self):
        CountingFill.colour = RED
        self.add(Square().shift(self.anchor))
        play_fill_between(self, 0.3, 0.6)
