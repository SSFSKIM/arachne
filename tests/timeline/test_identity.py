"""Admission vouches for the engine's and the libraries' code by identity,
taken when ``arachne`` is imported: a name a scene rebinds, or a function
dressed as the engine's, is only a callable the engine does not list.
"""

from __future__ import annotations

import functools
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from arachne import Dot
from arachne.timeline.dispatch import engine_patches, instance_shadows
from arachne.timeline.frozen import engine_code

from . import soundness_scenes

SOUNDNESS_SCENES = Path(soundness_scenes.__file__)

#: Run in a fresh interpreter: each rebinds a name after ``import arachne``
#: and before anything has asked admission about it.
_REBOUND = {
    "numpy.copy": """
import numpy as np
import arachne
from arachne.timeline import library
original = np.copy
np.copy = lambda a: a
assert library.entry(np.copy) is None, "the rebound np.copy is an entry"
assert library.entry(original) is not None
""",
    "arachne.Line": """
import arachne
from arachne.timeline import setters
original = arachne.Line
class Line(original):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
arachne.Line = Line
assert setters.constructor_kind(Line) is None, "the rebound Line is admitted"
assert setters.constructor_kind(original) == "reads"
""",
    "numpy.linalg.norm": """
import numpy as np
import arachne
from arachne.timeline import library
np.linalg.norm = lambda a, *args: 1.0
assert library.entry(np.linalg.norm) is None, "the rebound norm is an entry"
""",
}


@pytest.mark.parametrize("name", sorted(_REBOUND))
def test_a_name_rebound_before_its_first_use_is_not_admitted(name):
    done = subprocess.run(
        [sys.executable, "-c", _REBOUND[name]],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr


def test_functools_wraps_makes_no_function_the_engine_s():
    original = Dot.interpolate

    @functools.wraps(original)
    def dressed(*args, **kwargs):
        return original(*args, **kwargs)

    assert engine_code(original)
    assert dressed.__module__ == original.__module__
    assert not engine_code(dressed)
    d = Dot()
    d.interpolate = functools.wraps(d.interpolate)(lambda *a, **k: None)
    assert instance_shadows(d) == ("interpolate",)


def test_an_instance_interpolate_dressed_as_the_engine_s_ticks(session_of):
    session = session_of("WrappedInterpolate", SOUNDNESS_SCENES)
    play = session.record.plays[-1]
    assert play.route == "ticked", play.admission.reasons
    assert any("on the instance" in r for r in play.admission.reasons)
    counter = session.module.WrappedInterpolate.calls
    before = len(counter)
    session.state_at(Fraction(1, 7))
    session.frame(Fraction(1, 7))
    assert len(counter) == before


def test_an_engine_method_patched_with_a_dressed_function_ticks(session_of):
    session = session_of("WrappedPatch", SOUNDNESS_SCENES)
    play = session.record.plays[-1]
    assert play.route == "ticked", play.admission.reasons
    assert any(
        "replaces the engine's Dot.interpolate" in r for r in play.admission.reasons
    ), play.admission.reasons
    assert engine_patches() == ()
