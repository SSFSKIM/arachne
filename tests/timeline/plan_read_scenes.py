"""Scenes whose plan reads, when the play begins, a value a callback changes
later in the play: a list, a dict, an object's or a class's attribute, an
item or a function's default. Each changes what a stepping reader reads.
Running through a generator's items runs the generator's code, so iterating
one is a call.

Each scene has one producer callback that changes the value after 0.3 s and
one that uses it; ``output`` is a dt updater reading ``shared``, so a frame
shows whether the user's effect on ``shared`` ran at every tick.
"""

from __future__ import annotations

import operator

from arachne import LEFT, RED, RIGHT, UP, Dot, Mobject, Scene, ValueTracker
from arachne.utils.config_ops import update_dict_recursively


def idle(value=None):
    return 0.0


HANDLER = idle


class Settings:
    """A class whose attribute holds a function."""


class Holder:
    """An object whose attribute holds a function, set by a method."""

    def __init__(self, function):
        self.function = function

    def set(self, function):
        self.function = function


class PropertyHolder:
    """An object whose property gives back what it holds."""

    def __init__(self, function):
        self._function = function

    @property
    def handler(self):
        return self._function


class Target:
    """An object whose attribute holds a mobject."""


class _LateRead(Scene):
    """``shared`` and ``other`` dots; ``move`` shifts ``shared``; ``clock``
    counts the play's time; the callbacks :meth:`build` returns update dots
    of their own.
    """

    def build(self, shared, other, move, clock):
        raise NotImplementedError

    def construct(self):
        shared = Dot(radius=0.05).move_to(RIGHT)
        other = Dot(radius=0.05).move_to(LEFT)

        def move(value=None):
            shared.shift(0.01 * RIGHT)
            return 0.0

        clock = ValueTracker(0)
        clock.add_updater(lambda m, dt: m.increment_value(dt))
        built = self.build(shared, other, move, clock)
        for callback in built if isinstance(built, tuple) else (built,):
            producer = Mobject()
            producer.add_updater(callback)
            self.add(producer)
        output = Dot(color=RED, radius=0.05)
        output.add_updater(lambda m, dt: m.shift(dt * shared.get_x() * UP))
        spare = Mobject()
        spare.add_updater(lambda m: None)
        self.add(clock, shared, other, output, spare)
        self.wait(1)


class StoredLambdaRead(_LateRead):
    """A lambda moving ``shared``, appended to a list empty at the plan."""

    def build(self, shared, other, move, clock):
        registry: list = []

        def fill(m):
            if clock.get_value() > 0.3 and not registry:
                registry.append(lambda value=None: (shared.shift(0.01 * RIGHT), 0.0)[1])

        def use(m):
            if registry:
                sorted(["a", "b"], key=registry[0])

        return fill, use


class StoredLambdaItemRead(_LateRead):
    """A lambda moving ``shared``, stored as a dict's item."""

    def build(self, shared, other, move, clock):
        table = {"f": idle}

        def fill(m):
            if clock.get_value() > 0.3:
                table["f"] = lambda value=None: (shared.shift(0.01 * RIGHT), 0.0)[1]

        def use(m):
            sorted(["a", "b"], key=table["f"])

        return fill, use


class SetterMethodRead(_LateRead):
    """A function set on an object by a method of its class."""

    def build(self, shared, other, move, clock):
        holder = Holder(idle)

        def fill(m):
            if clock.get_value() > 0.3:
                holder.set(move)

        def use(m):
            sorted(["a", "b"], key=holder.function)

        return fill, use


class ClassAttributeRead(_LateRead):
    """A function stored on a class, read through an instance."""

    def build(self, shared, other, move, clock):
        Settings.function = staticmethod(idle)
        settings = Settings()

        def fill(m):
            if clock.get_value() > 0.3:
                Settings.function = staticmethod(move)

        def use(m):
            sorted(["a", "b"], key=settings.function)

        return fill, use


class GlobalsItemRead(_LateRead):
    """A module global rebound through ``globals()``."""

    def build(self, shared, other, move, clock):
        def fill(m):
            sorted([], key=move)
            if clock.get_value() > 0.3:
                globals()["HANDLER"] = move

        def use(m):
            sorted(["a", "b"], key=HANDLER)

        return fill, use


class OperatorItemRead(_LateRead):
    """A dict's item stored by ``operator.setitem``."""

    def build(self, shared, other, move, clock):
        table = {"f": idle}

        def fill(m):
            if clock.get_value() > 0.3:
                operator.setitem(table, "f", move)

        def use(m):
            sorted(["a", "b"], key=table["f"])

        return fill, use


class EngineStoredItemRead(_LateRead):
    """A dict's item stored by a function of the engine's."""

    def build(self, shared, other, move, clock):
        table = {"f": idle}

        def fill(m):
            if clock.get_value() > 0.3:
                update_dict_recursively(table, {"f": move})

        def use(m):
            sorted(["a", "b"], key=table["f"])

        return fill, use


class AliasedAppendRead(_LateRead):
    """A list's ``append`` held under another name."""

    def build(self, shared, other, move, clock):
        registry: list = []

        def fill(m):
            push = registry.append
            if clock.get_value() > 0.3 and not registry:
                push(move)

        def use(m):
            if registry:
                sorted(["a", "b"], key=registry[0])

        return fill, use


class UnboundAppendRead(_LateRead):
    """``list.append`` called on the list as its argument."""

    def build(self, shared, other, move, clock):
        registry: list = []

        def fill(m):
            if clock.get_value() > 0.3 and not registry:
                list.append(registry, move)

        def use(m):
            if registry:
                sorted(["a", "b"], key=registry[0])

        return fill, use


class PropertyRead(_LateRead):
    """A function given back by a property, its holder's attribute set."""

    def build(self, shared, other, move, clock):
        holder = PropertyHolder(idle)

        def fill(m):
            sorted([], key=move)
            if clock.get_value() > 0.3:
                holder._function = move

        def use(m):
            sorted(["a", "b"], key=holder.handler)

        return fill, use


def _ticker(move):
    def ticks():
        while True:
            move()
            yield 1

    return ticks()


class LoopedGeneratorRead(_LateRead):
    """A generator moving ``shared`` with each item, run by a ``for``."""

    def build(self, shared, other, move, clock):
        made = _ticker(move)

        def step(m):
            for _ in made:
                break

        return step


class ComprehendedGeneratorRead(_LateRead):
    """A generator run by a generator expression handed to ``any``."""

    def build(self, shared, other, move, clock):
        made = _ticker(move)
        return lambda m: any(True for _ in made)


class SearchedGeneratorRead(_LateRead):
    """A generator run by ``in``."""

    def build(self, shared, other, move, clock):
        made = _ticker(move)
        return lambda m: 1 in made


class DefaultedGeneratorRead(_LateRead):
    """A generator a parameter falls back to, run by a ``for``: the
    updater is called without it.
    """

    def build(self, shared, other, move, clock):
        made = _ticker(move)

        def step(m, source=made):
            for _ in source:
                break

        return step


class ReboundReceiverWrite(_LateRead):
    """An object's attribute holding the mobject a callback shifts, rebound
    from ``other`` to ``shared``.
    """

    def build(self, shared, other, move, clock):
        target = Target()
        target.mobject = other

        def fill(m):
            if clock.get_value() > 0.3:
                target.mobject = shared

        def use(m):
            target.mobject.shift(0.01 * RIGHT)

        return fill, use


class ReboundItemWrite(_LateRead):
    """A list's item holding the mobject a callback shifts, rebound."""

    def build(self, shared, other, move, clock):
        slots = [other]

        def fill(m):
            if clock.get_value() > 0.3:
                slots[0] = shared

        def use(m):
            slots[0].shift(0.01 * RIGHT)

        return fill, use


class ReboundDefaultWrite(_LateRead):
    """A function's default, the mobject it shifts, rebound through
    ``__defaults__``.
    """

    def build(self, shared, other, move, clock):
        def go(target=other):
            target.shift(0.01 * RIGHT)

        def fill(m):
            if clock.get_value() > 0.3:
                go.__defaults__ = (shared,)

        def use(m):
            go()

        return fill, use


class ReboundCellWrite(_LateRead):
    """A closure's cell, holding the mobject a callback shifts, rebound
    through ``cell_contents``.
    """

    def build(self, shared, other, move, clock):
        target = other

        def go():
            target.shift(0.01 * RIGHT)

        def fill(m):
            if clock.get_value() > 0.3:
                go.__closure__[0].cell_contents = shared

        def use(m):
            go()

        return fill, use


class ReboundMethodWrite(_LateRead):
    """A class's method rebound to one shifting ``shared``."""

    def build(self, shared, other, move, clock):
        class Mover:
            def go(self):
                other.shift(0.01 * RIGHT)

        def shifting(self):
            shared.shift(0.01 * RIGHT)

        mover = Mover()

        def fill(m):
            if clock.get_value() > 0.3:
                Mover.go = shifting

        def use(m):
            mover.go()

        return fill, use


class OwnFamilyWrite(_LateRead):
    """The control: a callback shifting only the mobject it updates, which
    nothing stepping reads, and another doing nothing.
    """

    def build(self, shared, other, move, clock):
        def own(m):
            m.shift(0.01 * RIGHT)

        return own


class OperatorHookRead(_LateRead):
    """An object of the scene's class whose ``__add__`` moves ``shared``,
    added to.
    """

    def build(self, shared, other, move, clock):
        class Vector:
            def __add__(self, value):
                move()
                return value

        vector = Vector()
        return lambda m: vector + 1


class TruthHookRead(_LateRead):
    """An object whose ``__bool__`` moves ``shared``, tested."""

    def build(self, shared, other, move, clock):
        class Flag:
            def __bool__(self):
                move()
                return True

        flag = Flag()

        def test(m):
            if flag:
                pass

        return test


class FormatHookRead(_LateRead):
    """An object whose ``__format__`` moves ``shared``, formatted."""

    def build(self, shared, other, move, clock):
        class Shown:
            def __format__(self, spec):
                move()
                return ""

        shown = Shown()
        return lambda m: f"{shown}"


class PropertyHookRead(_LateRead):
    """An object whose property's getter moves ``shared``, read."""

    def build(self, shared, other, move, clock):
        class Watched:
            @property
            def value(self):
                move()
                return 1

        watched = Watched()
        return lambda m: watched.value


class MissingAttributeHookRead(_LateRead):
    """An object whose ``__getattr__`` moves ``shared``, read for an
    attribute it does not hold.
    """

    def build(self, shared, other, move, clock):
        class Lazy:
            def __getattr__(self, name):
                move()
                return 1

        lazy = Lazy()
        return lambda m: lazy.anything


class DefaultedOperatorHookRead(_LateRead):
    """An object whose ``__add__`` moves ``shared``, a parameter's default,
    added to.
    """

    def build(self, shared, other, move, clock):
        class Vector:
            def __add__(self, value):
                move()
                return value

        return lambda m, vector=Vector(): vector + 1


class DefaultedPropertyHookRead(_LateRead):
    """An object whose property's getter moves ``shared``, a parameter's
    default, read.
    """

    def build(self, shared, other, move, clock):
        class Watched:
            @property
            def value(self):
                move()
                return 1

        return lambda m, watched=Watched(): watched.value
