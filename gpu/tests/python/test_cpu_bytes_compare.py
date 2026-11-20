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


SOFTWARE = {"name": "llvmpipe", "device_type": "Cpu"}
CARD = {"name": "A Card", "device_type": "DiscreteGpu"}


def arm(
    root: Path, name: str, build: str, frames=FRAMES, log=None, adapters=None
) -> Path:
    directory = root / name
    directory.mkdir()
    (directory / "digests.json").write_text(json.dumps(frames))
    if adapters is None and build == "present":
        adapters = [SOFTWARE]
    (directory / "arm.json").write_text(
        json.dumps({"python": "python", "gpu_build": build, "adapters": adapters})
    )
    if log is None:
        log = [{"pid": 1, "command": f"python -c import arachne_gpu  # {MARKER}"}]
    (directory / "tripwire.log").write_text(
        "".join(json.dumps(record) + "\n" for record in log)
    )
    return directory


def compare(*arms: Path, flags: tuple[str, ...] = ()) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(CI / "cpu_bytes_compare.py"), *flags, *map(str, arms)],
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


def three_arms(root: Path, card=CARD) -> list[Path]:
    return [
        arm(root, "without-build", "absent"),
        arm(root, "build-no-card", "present"),
        arm(root, "build-and-card", "present", adapters=[card, SOFTWARE]),
    ]


def test_the_arms_can_be_held_to_the_adapters_they_saw(tmp_path):
    arms = three_arms(tmp_path)
    flags = ("--no-hardware", "build-no-card", "--hardware", "build-and-card=A Card")
    done = compare(*arms, flags=flags)
    assert done.returncode == 0, done.stdout
    assert done.stdout.splitlines()[-1].startswith("cpu-bytes: 5 frames")


def test_an_arm_that_saw_other_adapters_than_held_fails(tmp_path):
    arms = three_arms(tmp_path)
    done = compare(*arms, flags=("--no-hardware", "build-and-card"))
    assert done.returncode == 1
    assert "build-and-card: saw a hardware adapter" in done.stdout
    done = compare(*arms, flags=("--hardware", "build-no-card=A Card"))
    assert done.returncode == 1
    assert "build-no-card: saw no hardware adapter named A Card" in done.stdout
    done = compare(*arms, flags=("--hardware", "build-and-card=Another Card"))
    assert done.returncode == 1


def test_an_arm_without_the_build_or_unnamed_cannot_be_held(tmp_path):
    arms = three_arms(tmp_path)
    done = compare(*arms, flags=("--no-hardware", "without-build"))
    assert done.returncode == 1
    assert "without the GPU build it sees no adapter" in done.stdout
    done = compare(*arms, flags=("--no-hardware", "elsewhere"))
    assert done.returncode == 1
    assert "no arm is named elsewhere" in done.stdout


def test_the_tripwire_comes_out_of_the_environment_whole(tmp_path):
    from tripwire import HOOK_FILES, install, uninstall

    site = tmp_path / "site"
    site.mkdir()
    install(site, tmp_path / "a.log")
    assert sorted(p.name for p in site.iterdir()) == sorted(HOOK_FILES)
    install(site, tmp_path / "b.log")
    assert str(tmp_path / "b.log") in (site / "_arachne_gpu_tripwire.py").read_text()
    uninstall(site)
    uninstall(site)
    assert list(site.iterdir()) == []
