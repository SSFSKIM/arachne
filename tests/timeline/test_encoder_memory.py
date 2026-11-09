"""Peak memory of a long render at 1080p, written as one stream.

Each render runs in a fresh process, so its peak resident size is the
render's own. With every frame of the grid encoded (no gaps) the encoder
works through the whole scene; with gaps, through one frame per run.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

#: Twice the peak measured when this test was written (about 1 GB, the
#: process's whole resident size), and far below the gigabytes an encoder
#: that kept its frames would reach over a scene this long.
CEILING_MB = 2048

SCRIPT = textwrap.dedent(
    """
    import resource, sys
    from pathlib import Path
    import av
    from arachne import *

    class LongNarration(Scene):
        # 28 s: four 2 s plays, each followed by a 5 s hold.
        def construct(self):
            shapes = [Square(color=BLUE), Circle(color=RED), Triangle(color=GREEN), Star(color=YELLOW)]
            caption = Text("one long stream", font_size=36).to_edge(UP)
            self.add(caption)
            current = shapes[0].scale(2)
            self.add(current)
            for shape in [*shapes[1:], Square(color=WHITE)]:
                self.play(Transform(current, shape.scale(2)), run_time=2)
                self.wait(5)

    gaps = sys.argv[2] == "gaps"
    with tempconfig({"quality": "production_quality", "media_dir": sys.argv[1],
                     "disable_caching": True, "hold_gaps": gaps, "progress_bar": "none",
                     "verbosity": "WARNING"}):
        scene = LongNarration()
        scene.render()
        movie = Path(scene.renderer.file_writer.movie_file_path)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = peak / 2**20 if sys.platform == "darwin" else peak / 2**10
    with av.open(str(movie)) as container:
        stream = container.streams.video[0]
        decoded = sum(1 for _ in container.decode(video=0))
        seconds = float(stream.duration * stream.time_base)
    print(f"peak_mb={peak_mb:.0f} decoded={decoded} seconds={seconds:.3f}")
    """
)


@pytest.mark.slow
@pytest.mark.parametrize("gaps", ["gaps", "every_frame"])
def test_a_long_1080p_render_stays_within_its_memory(tmp_path, gaps):
    done = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(tmp_path), gaps],
        capture_output=True,
        text=True,
        timeout=1200,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    print(done.stdout.strip())  # the measurement, shown with -rP
    fields = dict(
        item.split("=") for item in done.stdout.strip().splitlines()[-1].split()
    )
    assert float(fields["seconds"]) == pytest.approx(28, abs=1e-3)
    if gaps == "gaps":
        assert int(fields["decoded"]) < 28 * 60
    else:
        assert int(fields["decoded"]) == 28 * 60
    assert float(fields["peak_mb"]) < CEILING_MB, done.stdout
