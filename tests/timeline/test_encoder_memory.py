"""Peak memory of a long render at 1080p, written as one stream.

Each render runs in a fresh process, which reports its own peak: on Linux
its high-water mark (``VmHWM``), since ``ru_maxrss`` carries the peak of the
process that started it across fork and exec, which here would be the test
runner's; elsewhere ``ru_maxrss``, which on macOS is the process's own
(:func:`arachne.server.registry._rss_mb`). With every frame of the grid
encoded the encoder works through the whole scene; with gaps, through one
frame per run.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest

#: The peak measured when this test was written was 1.0 to 1.16 GiB on Linux
#: and macOS alike, a spread of about 180 MiB between runs (and before the
#: render was written as one stream, the same). The ceiling, an absolute
#: guard on the whole render's peak, leaves three times that spread above
#: the highest. It fails a render that keeps every frame of the scene
#: (14 GB), not one whose encoder needs twice its memory: libx264's share of
#: the peak is about 550 MiB, and twice that still comes in under it.
CEILING_MB = 1800

#: A parent that holds ``argv[1]`` MiB, every page written, and then runs the
#: Python script ``argv[2]`` with the arguments after it, passing on what it
#: printed.
PARENT = textwrap.dedent(
    """
    import subprocess, sys
    held = bytearray(b"x") * (int(sys.argv[1]) << 20)
    done = subprocess.run([sys.executable, "-c", *sys.argv[2:]], capture_output=True, text=True)
    sys.stderr.write(done.stderr)
    sys.stdout.write(done.stdout)
    sys.exit(done.returncode)
    """
)

#: The peak of a process that has imported the engine and done nothing else.
IMPORTED = textwrap.dedent(
    """
    import json
    import arachne
    from arachne.server.registry import _rss_mb
    print(json.dumps({"peak_mb": _rss_mb()}))
    """
)

#: The peak of a process that rendered a 28 s scene at 1080p60 (four 2 s
#: plays, each followed by a 5 s hold), with gaps or with every frame.
RENDER = textwrap.dedent(
    """
    import json, sys
    import av
    from arachne import *
    from arachne.server.registry import _rss_mb

    class LongNarration(Scene):
        def construct(self):
            shapes = [Square(color=BLUE), Circle(color=RED), Triangle(color=GREEN), Star(color=YELLOW)]
            caption = Text("one long stream", font_size=36).to_edge(UP)
            self.add(caption)
            current = shapes[0].scale(2)
            self.add(current)
            for shape in [*shapes[1:], Square(color=WHITE)]:
                self.play(Transform(current, shape.scale(2)), run_time=2)
                self.wait(5)

    with tempconfig({"quality": "production_quality", "media_dir": sys.argv[1],
                     "disable_caching": True, "hold_gaps": sys.argv[2] == "gaps",
                     "progress_bar": "none", "verbosity": "WARNING"}):
        scene = LongNarration()
        scene.render()
        movie = scene.renderer.file_writer.movie_file_path
    peak = _rss_mb()
    with av.open(str(movie)) as container:
        stream = container.streams.video[0]
        decoded = sum(1 for _ in container.decode(video=0))
        seconds = float(stream.duration * stream.time_base)
    print(json.dumps({"peak_mb": peak, "decoded": decoded, "seconds": seconds}))
    """
)


def run(script: str, *args: str, held_mb: int = 0) -> dict:
    """What ``script`` printed, run with ``args`` by a parent holding
    ``held_mb`` MiB (:data:`PARENT`).
    """
    done = subprocess.run(
        [sys.executable, "-c", PARENT, str(held_mb), script, *args],
        capture_output=True,
        text=True,
        timeout=1200,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


@pytest.mark.slow
def test_the_peak_reported_is_the_process_s_own():
    """Started by a parent holding 1.5 GiB, a process reports the peak a
    process started by a small parent does, not the parent's.
    """
    small = run(IMPORTED)["peak_mb"]
    large = run(IMPORTED, held_mb=1536)["peak_mb"]
    assert abs(large - small) < 150, (small, large)


@pytest.mark.slow
@pytest.mark.parametrize("gaps", ["gaps", "every_frame"])
def test_a_long_1080p_render_stays_within_its_memory(tmp_path, gaps):
    measured = run(RENDER, str(tmp_path), gaps)
    print(measured)  # the measurement, shown with -rP
    assert measured["seconds"] == pytest.approx(28, abs=1e-3)
    if gaps == "gaps":
        assert measured["decoded"] < 28 * 60
    else:
        assert measured["decoded"] == 28 * 60
    assert measured["peak_mb"] < CEILING_MB, measured
