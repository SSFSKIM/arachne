"""The TeX construction memo: a :class:`~.MathTex`, :class:`~.Tex` or
:class:`~.SingleStringMathTex` built once, then handed out as a copy of
what it built (the ``construction_memo`` lever).

A construction is served from the memo only when it would build exactly
what it built before. Its key is everything the construction reads:

- its class and every argument it was called with, by value and type
  (``48`` and ``48.0`` key apart, as do two orders of a colour map);
- the configured TeX template, every field of it (its full text, its
  compiler and output format);
- the image it builds from: its file, the file the parse would read, and
  the sha256 of that file's content;
- whether the parsed-image cache holds that image now. A construction that
  finds the image there builds otherwise than one that parses it (the
  parse flips the whole mobject, its own sheen direction included; a hit
  in that cache flips only the parts it copies), so the two are two
  entries, and a hit on an image the cache lacks puts the cache's entry
  back, as the parse would have.

An entry also holds the engine's classes as the construction found them:
the code of every class whose instances it made or holds (and the defaults
``set_default`` gave them). A class changed since is a miss, and the
construction is built and stored again.

What a hit hands out is the very object being constructed, holding a deep
copy of what the entry holds: a hit costs that one copy, in place of the
parsed-image cache's own copy, and no parse or layout. Everything a
recording reads of a fresh construction is put back: the origin of every
mobject it made (its line and its place among what was made there,
temporaries counted, :mod:`arachne.timeline.origins`), which mobjects
carry an ``original_id``, and which attributes hold the caller's own
arguments (the template, a colour map) rather than copies.

The memo survives executions (it is not among the construction caches a
session forgets each execution, :mod:`.construction_caches`): what a
construction builds is a function of its key, and the key names the state
of those caches it reads. It holds at most :attr:`TexMemo.bound` entries,
the least recently used dropped first; dropping one costs a hit, never
correctness.

Nothing is served while a recording would count the image's file as a
scene input (a media directory inside the scene's own), since a hit reads
no file.
"""

from __future__ import annotations

__all__ = ["TEX_MEMO", "TexMemo", "memoized"]

import copy
import dataclasses
import enum
import functools
import hashlib
from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from operator import is_
from pathlib import PurePath
from typing import TYPE_CHECKING, Any

import numpy as np

from arachne import config
from arachne.mobject.svg.svg_mobject import SVG_HASH_TO_MOB_MAP
from arachne.timeline import inputs
from arachne.timeline.origins import ORIGINS, Origin
from arachne.utils.iterables import hash_obj
from arachne.utils.tex import TexTemplate

if TYPE_CHECKING:
    from arachne.mobject.svg.svg_mobject import SVGMobject

#: Class attributes the engine rebinds while it runs, which no construction
#: reads (a record's stepping tick), or which ``set_default`` replaces and
#: an entry compares by its defaults (``__init__``).
_UNREAD = frozenset({"_tick_steps", "__init__"})


class _Unkeyable(Exception):
    """A value the key can't hold: the construction is not memoized."""


class _Served(Exception):
    """The construction was served from the memo; unwinds its body."""


def _key(value: Any) -> Any:
    """``value`` as a hashable key equal only to the key of a value that
    builds alike: by type and content.
    """
    kind = type(value)
    if value is None or kind in (bool, str, bytes):
        return (kind, value)
    if isinstance(value, float):
        # The bits: 0.0 and -0.0 key apart.
        return (kind, float.hex(value))
    if isinstance(value, int):
        return (kind, int(value))
    if isinstance(value, complex):
        return (kind, value.real.hex(), value.imag.hex())
    if isinstance(value, (tuple, list)):
        return (kind, tuple(map(_key, value)))
    if isinstance(value, dict):
        return (kind, tuple((_key(k), _key(v)) for k, v in value.items()))
    if isinstance(value, np.ndarray):
        return (kind, value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, np.generic):
        return (kind, _key(value.item()))
    if isinstance(value, TexTemplate):
        return (
            kind,
            tuple(
                (field.name, _key(getattr(value, field.name)))
                for field in dataclasses.fields(value)
            ),
        )
    if isinstance(value, PurePath):
        return (kind, str(value))
    if isinstance(value, enum.Enum):
        return (kind, value.name)
    internal = getattr(value, "_internal_value", None)
    if isinstance(internal, np.ndarray) and type(value).__module__.startswith(
        "arachne"
    ):
        # A colour.
        return (kind, internal.tobytes())
    raise _Unkeyable(kind)


def _defaults(init: Any) -> Any:
    """An ``__init__`` as a key: a function by identity, and one
    ``set_default`` wrapped by the defaults it gave.
    """
    if isinstance(init, functools.partialmethod):
        try:
            return (_defaults(init.func), _key(init.args), _key(init.keywords))
        except _Unkeyable:
            # Defaults no key holds: never equal to another.
            return object()
    return init


@dataclasses.dataclass(frozen=True)
class _ClassState:
    """An engine class as a construction found it."""

    cls: type
    size: int
    names: tuple[str, ...]
    values: tuple[Any, ...]
    init: Any

    @classmethod
    def of(cls, klass: type) -> _ClassState:
        own = vars(klass)
        names = tuple(name for name in own if name not in _UNREAD)
        return cls(
            klass,
            len(own),
            names,
            tuple(own[name] for name in names),
            _defaults(own.get("__init__")),
        )

    def holds(self) -> bool:
        own = vars(self.cls)
        return (
            len(own) == self.size
            and all(map(is_, map(own.get, self.names), self.values))
            and _defaults(own.get("__init__")) == self.init
        )


@dataclasses.dataclass(frozen=True)
class _Placement:
    """How the construction numbered what it made at its line: ``count``
    objects after the root, and the offset among them of each mobject the
    prototype holds.
    """

    count: int
    offsets: tuple[tuple[Any, int], ...]


@dataclasses.dataclass(frozen=True)
class _Entry:
    """What a construction built (``prototype``, never changed and never
    handed out) and what a hit puts back with it.
    """

    prototype: Any
    #: The prototype's mobjects whose originals carried no ``original_id``.
    bare: tuple[Any, ...]
    #: The root's attributes that held one of the caller's arguments, with
    #: where to find it (``("arg", i)``, ``("kw", name)``, ``("template",)``).
    aliases: tuple[tuple[str, tuple[Any, ...]], ...]
    #: The parsed-image cache's entry the construction made, if it made one.
    image: Any
    classes: tuple[_ClassState, ...]
    #: ``None`` when it was built while nothing was tagged.
    placement: _Placement | None

    def serves(self, root: Any) -> bool:
        if ORIGINS.active and (self.placement is None or ORIGINS.get(root) is None):
            return False
        return all(state.holds() for state in self.classes)


@dataclasses.dataclass
class _Construction:
    """A construction under way: its root, its arguments and, once its
    image is known, its key.
    """

    root: Any
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    template: TexTemplate
    base: Any
    key: Any = None
    image_hash: int | None = None
    #: Whether the parsed-image cache held the image (``None``: not used).
    image_held: bool | None = None
    #: Not to be stored (its file is watched, or its key couldn't be read).
    skip: bool = False

    @property
    def image_missing(self) -> int | None:
        """The image cache's key for the image, when the cache lacked it."""
        return self.image_hash if self.image_held is False else None

    def argument(self, where: tuple[Any, ...]) -> Any:
        if where[0] == "arg":
            return self.args[where[1]]
        if where[0] == "kw":
            return self.kwargs[where[1]]
        return self.template


class TexMemo:
    """The memo of TeX constructions, bounded by entry count."""

    def __init__(self, bound: int) -> None:
        #: How many entries it keeps.
        self.bound = bound
        #: Constructions served from it, and those built and looked up.
        self.hits = 0
        self.misses = 0
        self._entries: OrderedDict[Any, _Entry] = OrderedDict()
        self._open: list[_Construction] = []

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        """Drop every entry."""
        self._entries.clear()

    # The constructor's side.

    def _begin(
        self, root: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> _Construction | None:
        """The construction of ``root`` starting, if the memo may serve it:
        ``None`` for a construction already under way (a base class's
        ``__init__``), another class, or the lever off.
        """
        if self._open and self._open[-1].root is root:
            return None
        if type(root) not in SERVED or not config.construction_memo:
            return None
        template = config.tex_template
        try:
            base = (type(root), _key(args), _key(kwargs), _key(template))
        except _Unkeyable:
            return None
        return _Construction(root, args, kwargs, template, base)

    def reached(self, mob: SVGMobject, use_svg_cache: bool) -> None:
        """``mob``'s image file is known and about to be parsed or taken
        from the parsed-image cache. Serve the construction under way from
        the memo if it can: ``mob`` then holds what it built, and this
        raises, unwinding the construction's body.
        """
        construction = self._open[-1] if self._open else None
        if construction is None or construction.root is not mob or construction.key:
            return
        try:
            path = mob.get_file_path()
            if inputs.watched(path):
                construction.skip = True
                return
            with inputs.unwatched():
                digest = hashlib.sha256(path.read_bytes()).digest()
        except (OSError, ValueError):
            construction.skip = True
            return
        if use_svg_cache:
            construction.image_hash = hash_obj(mob.hash_seed)
            construction.image_held = construction.image_hash in SVG_HASH_TO_MOB_MAP
        construction.key = key = (
            construction.base,
            str(mob.file_name),
            str(path),
            digest,
            construction.image_hash,
            construction.image_held,
        )
        entry = self._entries.get(key)
        if entry is None or not entry.serves(mob):
            self.misses += 1
            return
        self._entries.move_to_end(key)
        self.hits += 1
        self._install(entry, mob, construction)
        raise _Served

    def _install(self, entry: _Entry, mob: Any, construction: _Construction) -> None:
        memo: dict[int, Any] = {id(entry.prototype): mob}
        with ORIGINS.paused():
            held = copy.deepcopy(entry.prototype.__dict__, memo)
        own = mob.__dict__
        own.clear()
        own.update(held)
        for bare in entry.bare:
            del memo[id(bare)].__dict__["original_id"]
        for name, where in entry.aliases:
            own[name] = construction.argument(where)
        if entry.placement is not None:
            ORIGINS.place(
                mob,
                (
                    (memo[id(proto)], offset)
                    for proto, offset in entry.placement.offsets
                ),
                entry.placement.count,
            )
        missing = construction.image_missing
        if missing is not None:
            # The parsed-image cache's entries are never changed: shared.
            SVG_HASH_TO_MOB_MAP[missing] = entry.image

    def _store(
        self, construction: _Construction, made: list[tuple[object, Origin]]
    ) -> None:
        """Keep what ``construction`` built, ``made`` being every object it
        gave an origin, in order.
        """
        if construction.key is None or construction.skip:
            return
        root = construction.root
        memo: dict[int, Any] = {}
        with ORIGINS.paused():
            prototype = copy.deepcopy(root, memo)
        from arachne.mobject.mobject import Mobject

        pairs = [
            (original, memo[key])
            for key, original in _originals(root, memo)
            if isinstance(original, Mobject)
        ]
        bare = []
        for original, proto in pairs:
            if "original_id" not in original.__dict__:
                del proto.__dict__["original_id"]
                if original is not root:
                    bare.append(proto)
        candidates = [
            *((("arg", i), value) for i, value in enumerate(construction.args)),
            *((("kw", name), value) for name, value in construction.kwargs.items()),
            (("template",), construction.template),
        ]
        aliases = []
        for name, value in root.__dict__.items():
            if type(value) in _ATOMS:
                continue
            for where, argument in candidates:
                if value is argument:
                    aliases.append((name, where))
                    break
        classes: dict[type, None] = {}
        for klass in (type(root), *map(type, memo.values())):
            for base in klass.__mro__:
                if base.__module__.startswith("arachne"):
                    classes[base] = None
        from arachne.mobject.types.vectorized_mobject import VMobject

        # The colour a construction takes by default is a VMobject's.
        classes.update(dict.fromkeys(VMobject.__mro__[:-1]))
        missing = construction.image_missing
        image = SVG_HASH_TO_MOB_MAP.get(missing) if missing is not None else None
        entry = _Entry(
            prototype,
            tuple(bare),
            tuple(aliases),
            image,
            tuple(_ClassState.of(klass) for klass in classes),
            _placement(root, made, {id(o): p for o, p in pairs if o is not root}),
        )
        entries = self._entries
        entries[construction.key] = entry
        entries.move_to_end(construction.key)
        while len(entries) > self.bound:
            entries.popitem(last=False)

    @contextmanager
    def _constructing(self, construction: _Construction) -> Iterator[None]:
        self._open.append(construction)
        try:
            yield
        finally:
            self._open.pop()


#: Values a construction holds that no copy distinguishes from the caller's.
_ATOMS = frozenset({type(None), bool, int, float, complex, str, bytes})


def _originals(root: Any, memo: dict[int, Any]) -> Iterator[tuple[int, Any]]:
    """Each object ``copy.deepcopy(root, memo)`` copied, by id, with the
    original: the copy keeps every original alive in its keep-alive list.
    """
    yield id(root), root
    for original in memo.get(id(memo), ()):
        if original is not root and id(original) in memo:
            yield id(original), original


def _placement(
    root: Any, made: list[tuple[object, Origin]], prototypes: dict[int, Any]
) -> _Placement | None:
    """How the construction of ``root`` numbered the objects it ``made``
    (each original mobject in ``prototypes`` being copied there), or
    ``None`` when nothing was tagged or not all of it at ``root``'s line
    right after it.
    """
    origin = ORIGINS.get(root)
    if origin is None:
        return None
    where, _, ordinal = origin.id.rpartition("#")
    first = int(ordinal) + 1
    offsets = []
    for offset, (obj, given) in enumerate(made):
        if (
            given.id != f"{where}#{first + offset}"
            or given.source_line != origin.source_line
            or given.scene_line != origin.scene_line
        ):
            return None
        proto = prototypes.get(id(obj))
        if proto is not None:
            offsets.append((proto, offset))
    if len(offsets) != len(prototypes):
        # A mobject it holds that it did not make.
        return None
    return _Placement(len(made), tuple(offsets))


#: The memo of every TeX construction in the process.
TEX_MEMO = TexMemo(bound=512)

#: The classes whose own constructions the memo serves (filled by
#: :mod:`.tex_mobject`): the engine's, never a subclass, whose code may do
#: anything.
SERVED: set[type] = set()


def memoized(init: Callable[..., None]) -> Callable[..., None]:
    """Make the ``__init__`` of a class in :data:`SERVED` construct through
    the memo.
    """

    @functools.wraps(init)
    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        construction = TEX_MEMO._begin(self, args, kwargs)
        if construction is None:
            init(self, *args, **kwargs)
            return
        with TEX_MEMO._constructing(construction), ORIGINS.counting() as made:
            try:
                init(self, *args, **kwargs)
            except _Served:
                return
            TEX_MEMO._store(construction, made)

    return __init__
