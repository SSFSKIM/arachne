"""Scenes ``cpu_bytes.sh`` draws besides ``cpu_bytes_scenes``: each uses nothing of the GPU build,
and each exists only at the head. ``base_head.sh`` draws ``cpu_bytes_scenes`` with a base commit's
engine too, where a name added since cannot be imported, so a scene of a name added since lives
here: drawn with and without the GPU build at one revision, and never at the base.
"""

from __future__ import annotations

from arachne import DOWN, RIGHT, FadeIn, Scene, Still


class ManifestStill(Scene):
    """A still opened through its asset's manifest, with no effect: a CPU frame."""

    def construct(self):
        still = Still("fixture:swatch").shift(DOWN)
        self.play(FadeIn(still), run_time=0.5)
        self.play(still.animate.shift(2 * RIGHT).scale(0.5), run_time=0.5)
        self.wait(0.5)


SCENES = (ManifestStill,)
