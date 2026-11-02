"""The library table admits each entry by identity, in its call form only."""

from __future__ import annotations

import numpy as np
import pytest

from arachne import random_bright_color, random_color
from arachne.timeline.library import Entry, entry, library
from arachne.timeline.purity import Refused, prove_readout

ENTRIES = library()
MOST = 6


def _calls(function):
    """Readouts calling ``function`` with 0 to ``MOST`` positional values."""
    x = 1.0
    f = function
    return [
        lambda: f(),
        lambda: f(x),
        lambda: f(x, x),
        lambda: f(x, x, x),
        lambda: f(x, x, x, x),
        lambda: f(x, x, x, x, x),
        lambda: f(x, x, x, x, x, x),
        lambda: f(x, x, x, x, x, x, x),
    ]


def _with_out(function):
    x = 1.0
    f = function
    return lambda: f(x, out=x)


def _id(item: tuple[object, Entry]) -> str:
    return item[1].name


def test_no_entry_takes_more_positional_arguments_than_the_test_writes():
    assert all(e.positional is None or e.positional < MOST for _, e in ENTRIES)


@pytest.mark.parametrize("item", ENTRIES, ids=_id)
def test_an_entry_is_admitted_in_its_call_form_and_refused_past_it(item):
    function, found = item
    assert entry(function) is found
    calls = _calls(function)
    most = 2 if found.positional is None else found.positional
    prove_readout(calls[most], None)
    if found.positional is not None:
        with pytest.raises(Refused, match="past what its entry admits"):
            prove_readout(calls[most + 1], None)
    with pytest.raises(Refused, match="out="):
        prove_readout(_with_out(function), None)


def test_functions_alike_in_name_or_module_are_not_entries():
    assert entry(np.put_along_axis) is None
    assert entry(np.frompyfunc(lambda v: v, 1, 1)) is None
    assert entry(random_color) is None
    assert entry(random_bright_color) is None
    assert entry(np.random.random) is None
