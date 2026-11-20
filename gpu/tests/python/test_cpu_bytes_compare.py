"""The bytes comparison's comparer, on arms written here: it passes only when every arm drew the
same frames, each tripwire log holds its deliberate import alone, and the arms are with and
without the GPU build; then it prints the gate's line.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

CI = Path(__file__).resolve().parents[2] / "ci"
sys.path.insert(0, str(CI))
from tripwire import MARKER  # noqa: E402

FRAMES = {
    "render ShapesAndText": [["aa", 3], ["bb", 1]],
    "frame ShapesAndText": [["cc", 1]],
}


def arm(root: Path, name: str, build: str, frames=FRAMES, log=None) -> Path:
    directory = root / name
    directory.mkdir()
    (directory / "digests.json").write_text(json.dumps(frames))
    (directory / "arm.json").write_text(
        json.dumps({"python": "python", "gpu_build": build, "adapters": None})
    )
    if log is None:
        log = [{"pid": 1, "command": f"python -c import arachne_gpu  # {MARKER}"}]
    (directory / "tripwire.log").write_text(
        "".join(json.dumps(record) + "\n" for record in log)
    )
    return directory


def compare(*arms: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(CI / "cpu_bytes_compare.py"), *map(str, arms)],
        capture_output=True,
        text=True,
    )


def test_two_or_three_equal_arms_print_the_gates_line(tmp_path):
    two = [arm(tmp_path, "without", "absent"), arm(tmp_path, "with", "present")]
    three = [*two, arm(tmp_path, "with-card", "present")]
    for arms in (two, three):
        done = compare(*arms)
        assert done.returncode == 0, done.stdout
        lines = done.stdout.splitlines()
        assert lines[-1] == (
            "cpu-bytes: 5 frames of 1 scenes byte-identical with and without the GPU build"
        )
        assert lines[-2].startswith("frames sha256: ")
    assert (
        compare(*two).stdout.splitlines()[-2] == compare(*three).stdout.splitlines()[-2]
    )


def test_a_differing_frame_fails(tmp_path):
    other = {**FRAMES, "frame ShapesAndText": [["dd", 1]]}
    done = compare(
        arm(tmp_path, "without", "absent"),
        arm(tmp_path, "with", "present"),
        arm(tmp_path, "with-card", "present", frames=other),
    )
    assert done.returncode == 1
    assert "frame ShapesAndText: the frames differ between without and with-card" in (
        done.stdout
    )


def test_arms_all_with_or_all_without_the_build_fail(tmp_path):
    done = compare(arm(tmp_path, "a", "present"), arm(tmp_path, "b", "present"))
    assert done.returncode == 1
    assert "not with and without the GPU build" in done.stdout


def test_an_import_other_than_the_deliberate_one_or_none_fails(tmp_path):
    stray = {"pid": 2, "command": "python render.py"}
    done = compare(
        arm(tmp_path, "without", "absent"),
        arm(tmp_path, "with", "present", log=[stray]),
    )
    assert done.returncode == 1
    assert "the deliberate import left 0 records" in done.stdout
    assert "imported by 2: python render.py" in done.stdout
