"""The suite runs at one hash seed: ``str`` and ``bytes`` hashes are salted per
process unless ``PYTHONHASHSEED`` is set, and iterating a set of them orders
some scenes' drawing (TransformMatchingParts' keys, QuickHull's facets), so a
frame compares byte for byte only at one seed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _session(seed: str | None) -> subprocess.CompletedProcess[str]:
    """A pytest session collecting this file, started at ``seed`` (unset:
    ``None``).
    """
    env = {k: v for k, v in os.environ.items() if k != "PYTHONHASHSEED"}
    if seed is not None:
        env["PYTHONHASHSEED"] = seed
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-o",
            "addopts=",
            "-p",
            "no:cacheprovider",
            "--collect-only",
            "-q",
            str(Path(__file__)),
        ],
        env=env,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_this_session_runs_with_hash_randomization_off():
    assert sys.flags.hash_randomization == 0


def test_a_session_started_at_another_seed_is_refused():
    assert _session("0").returncode == 0
    for seed in (None, "1"):
        refused = _session(seed)
        assert refused.returncode != 0, seed
        assert "PYTHONHASHSEED=0" in refused.stdout + refused.stderr, seed
