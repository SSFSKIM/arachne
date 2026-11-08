"""Every graphical unit drawn three ways, frame for frame the same bytes: the
frames comparison draws each unit's scene directly, checks it against its
control frames, then draws it again on ``ir_cairo``, where its frames come
from the record or are drawn live, both through the Frame IR, and requires
that pass to have drawn from XF and each frame's sha256 to equal the
first's (``frames_comparison.BOTH_PATHS_ENV``); and draws it once more with
the speed levers off and requires the same of each frame of the frame grid
(``frames_comparison.LEVERS_OFF_ENV``). One run of the units serves both
checks.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from arachne.utils.testing.frames_comparison import BOTH_PATHS_ENV, LEVERS_OFF_ENV

pytest.importorskip("arachne_core")

UNITS = Path(__file__).parents[1] / "test_graphical_units"


@pytest.mark.slow
def test_every_graphical_unit_is_drawn_from_the_ir_and_with_the_levers_off_byte_for_byte():
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(UNITS),
            "-q",
            "-p",
            "no:cacheprovider",
            "-p",
            "no:xdist",
            "--no-cov",
            "-o",
            "addopts=",
        ],
        env={**os.environ, BOTH_PATHS_ENV: "1", LEVERS_OFF_ENV: "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-2000:]
