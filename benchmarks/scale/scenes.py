"""The scale scenes: one shape at many placements, each a stream of frames
the core draws.

Each scene is the XF of its frames, built before anything is timed: a list of
encoded deltas, the first setting the scene up and each later one moving it.

- ``spheres``: shaded spheres orbiting a centre, engine-authored XF: one
  sphere mesh at ``n`` placements in one ``instances`` record, each frame
  moving every sphere along its orbit.
- ``arrows``: an :class:`~arachne.ArrowVectorField` of ``n`` arrows, extracted
  as the core path extracts a scene (its arrows factored and batched,
  :mod:`arachne.frame_ir.instancing`; its geometry snapped,
  :mod:`arachne.frame_ir.canonical`), the 3D camera turning about it.
- ``points``: a :class:`~arachne.PMobject` of ``n`` points, one ``points``
  record, extracted the same way, the camera turning about it.

Only the arrows and points scenes need Arachne's extraction; every scene
needs ``arachne_core`` to encode its frames, except where ``encode`` is
False (a smoke run without the core: the arrows and points scenes are then
the batched deltas themselves, the spheres scene its placements).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import numpy as np

#: The frame's extent in scene units (CE's default at 16:9).
FRAME = {"frame_width": 14.222222222222221, "frame_height": 8.0}

#: The light models a scene can be drawn under, as the view names them.
LIGHTS: dict[str, Any] = {"ce": "ce", "soft": (1, [0.35, 0.4])}


@dataclass
class Built:
    """A scene's frames and what they hold."""

    name: str
    count: int
    frames: list[Any] = field(default_factory=list)
    #: Records and ``instances`` records the first frame sets.
    records: int = 0
    instances: int = 0
    #: Leaves whose shape was factored (arrows).
    factored: int = 0


def _sphere(lat: int = 8, lon: int = 16) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A unit UV sphere: positions, triangles, normals."""
    polar = np.linspace(0.0, np.pi, lat + 1)
    azimuth = np.linspace(0.0, 2.0 * np.pi, lon + 1)[:-1]
    p, a = np.meshgrid(polar, azimuth, indexing="ij")
    positions = np.stack(
        [np.sin(p) * np.cos(a), np.sin(p) * np.sin(a), np.cos(p)], axis=-1
    ).reshape(-1, 3)
    triangles = []
    for i in range(lat):
        for j in range(lon):
            a0, a1 = i * lon + j, i * lon + (j + 1) % lon
            b0, b1 = a0 + lon, a1 + lon
            if i > 0:
                triangles.append((a0, b0, a1))
            if i < lat - 1:
                triangles.append((a1, b0, b1))
    return positions, np.array(triangles, dtype=np.uint32), positions.copy()


def _orbits(n: int, seed: int = 7) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    radius = 0.6 + 3.2 * np.sqrt(rng.random(n))
    angle = rng.random(n) * 2.0 * np.pi
    height = rng.normal(0.0, 0.25, n)
    speed = 0.35 / np.sqrt(radius)
    return radius, angle, height, speed


def sphere_placements(n: int, frame: int, scale: float = 0.06) -> np.ndarray:
    """Each sphere's placement at ``frame`` (15 per second of orbit)."""
    radius, angle, height, speed = _orbits(n)
    turned = angle + speed * frame / 15.0
    placements = np.zeros((n, 3, 4))
    placements[:, 0, 0] = placements[:, 1, 1] = placements[:, 2, 2] = scale
    placements[:, 0, 3] = radius * np.cos(turned)
    placements[:, 1, 3] = radius * np.sin(turned)
    placements[:, 2, 3] = height
    return placements


def spheres(
    n: int, frames: int, *, size: tuple[int, int], light: str, encode: bool = True
) -> Built:
    built = Built("spheres", n, records=1, instances=1)
    if not encode:
        built.frames = [sphere_placements(n, k) for k in range(frames)]
        return built
    import arachne_core as ac

    palette = np.array(
        [[0.36, 0.58, 0.92, 1.0], [0.95, 0.55, 0.25, 1.0], [0.45, 0.8, 0.5, 1.0]]
    )
    rows = palette[np.arange(n) % len(palette)]
    positions, triangles, normals = _sphere()
    mesh = fill = None
    for k in range(frames):
        delta = ac.FrameDelta(t=(k, 15), seq=k)
        if k == 0:
            delta.set_view(
                pixel_width=size[0],
                pixel_height=size[1],
                camera="three_d",
                phi=1.05,
                theta=-1.2,
                zoom=1.0,
                background=(0.0, 0.0, 0.0, 1.0),
                light_model=LIGHTS[light],
                **FRAME,
            )
            mesh = delta.upload_mesh(positions, triangles, normals=normals)
            fill = delta.upload_paint(rows)
        delta.set_record(
            "instances",
            1,
            1,
            shape=mesh,
            placements=sphere_placements(n, k),
            instance_fill=fill,
            instance_stroke=fill,
            objects=np.arange(1, n + 1, dtype=np.uint32),
            fill=(1.0, 1.0, 1.0, 1.0),
            space="world",
            shade_in_3d=True,
        )
        if k == 0:
            delta.set_order([("moving", 1, 0)])
        built.frames.append(delta.to_bytes())
    return built


def _extracted(
    name: str,
    mobject: Any,
    count: int,
    frames: int,
    *,
    size: tuple[int, int],
    light: str,
    encode: bool,
) -> Built:
    """``mobject``'s frames as the core path extracts them, a 3D camera
    turning about it.
    """
    from arachne import tempconfig
    from arachne.camera.three_d_camera import ThreeDCamera
    from arachne.frame_ir.canonical import Canonical, canonical_order, frame_grid
    from arachne.frame_ir.extract import FrameStream, batching
    from arachne.frame_ir.instancing import Instancing

    built = Built(name, count)
    # Arrows and points are not shaded: ``light`` does not reach them.
    del light
    with tempconfig({"pixel_width": size[0], "pixel_height": size[1]}):
        camera = ThreeDCamera(phi=1.05, theta=-np.pi / 2)
        stream, batches, canonical = FrameStream(), Instancing(), Canonical()
        groups = [("moving", [mobject])]
        with batching() as tally:
            for k in range(frames):
                camera.set_theta(-np.pi / 2 + 0.01 * k)
                grid = frame_grid(camera)
                with canonical_order(camera, grid):
                    made = stream.frame(camera, groups, t=Fraction(k, 15))
                canonical.frame_grid = grid
                sent = batches(made, grid)
                if k == 0:
                    built.records = len(sent.set)
                    built.instances = sum(r.kind == "instances" for _, r in sent.set)
                if encode:
                    sent = canonical(sent).to_core().to_bytes()
                built.frames.append(sent)
        built.factored = tally["factored"]
    return built


def arrows(
    n: int, frames: int, *, size: tuple[int, int], light: str, encode: bool = True
) -> Built:
    from arachne import ArrowVectorField

    side = max(2, round(np.sqrt(n)))
    field_ = ArrowVectorField(
        lambda p: np.array([np.sin(p[1]) + 0.3, np.cos(p[0]), 0.0]) * 0.5,
        x_range=[-6.5, 6.5, 13.0 / (side - 1)],
        y_range=[-3.6, 3.6, 7.2 / (side - 1)],
    )
    return _extracted(
        "arrows",
        field_,
        len(field_.submobjects),
        frames,
        size=size,
        light=light,
        encode=encode,
    )


def points(
    n: int, frames: int, *, size: tuple[int, int], light: str, encode: bool = True
) -> Built:
    from arachne import PMobject

    rng = np.random.default_rng(11)
    cloud = PMobject(stroke_width=2)
    at = rng.normal(0.0, 1.0, (n, 3)) * np.array([2.2, 1.4, 0.8])
    rgbas = np.ones((n, 4))
    rgbas[:, :3] = 0.5 + 0.5 * np.tanh(at / 2.0)
    cloud.add_points(at, rgbas=rgbas)
    return _extracted("points", cloud, n, frames, size=size, light=light, encode=encode)


SCENES = {"spheres": spheres, "arrows": arrows, "points": points}

#: Each scene's count as the benchmark states it.
COUNTS = {"spheres": 10_000, "arrows": 10_000, "points": 100_000}
