"""Tables of the engine's own objects, looked up by identity.

A set or dict finds a key by its ``__hash__`` and ``__eq__``, which a
scene's object may define to match anything: a callable whose hash and
equality are ``linear``'s, a class whose metaclass makes it equal to
``Create``. A table that decides what the engine trusts must find only the
very object it holds, and must not run the object's own code to look.
"""

from __future__ import annotations

__all__ = ["ByIdentity", "one_of"]

from collections.abc import Iterable, Iterator, Mapping, MutableMapping
from typing import Any, TypeVar

K = TypeVar("K")
V = TypeVar("V")
T = TypeVar("T")


def one_of(value: Any, candidates: Iterable[Any]) -> bool:
    """Whether ``value`` is one of ``candidates`` itself (``in`` on a tuple
    would also take one that only compares equal).
    """
    return any(value is candidate for candidate in candidates)


class ByIdentity(MutableMapping[K, V]):
    """A mapping that finds a key only as the object it was stored as,
    whatever that object's ``__hash__`` and ``__eq__``. It holds its keys,
    so their ids stay theirs while they are in it.
    """

    __slots__ = ("_entries",)

    def __init__(self, items: Mapping[K, V] | Iterable[tuple[K, V]] = ()) -> None:
        pairs = items.items() if isinstance(items, Mapping) else items
        self._entries: dict[int, tuple[K, V]] = {}
        for key, value in pairs:
            self[key] = value

    @staticmethod
    def of(keys: Iterable[T]) -> ByIdentity[T, None]:
        """A set by identity: each of ``keys``, mapped to ``None``."""
        return ByIdentity((key, None) for key in keys)

    def __getitem__(self, key: K) -> V:
        entry = self._entries.get(id(key))
        if entry is None or entry[0] is not key:
            raise KeyError(key)
        return entry[1]

    def __setitem__(self, key: K, value: V) -> None:
        self._entries[id(key)] = (key, value)

    def __delitem__(self, key: K) -> None:
        self[key]  # noqa: B018 — a KeyError unless it is this very key
        del self._entries[id(key)]

    def __contains__(self, key: Any) -> bool:
        entry = self._entries.get(id(key))
        return entry is not None and entry[0] is key

    def __iter__(self) -> Iterator[K]:
        return (key for key, _ in list(self._entries.values()))

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"ByIdentity({dict(self.items())!r})"
