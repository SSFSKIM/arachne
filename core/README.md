# arachne-core

Arachne's CPU core: it draws frames described in XF, the Frame IR (`XF.md`). It is a Rust crate with a
Python binding (`arachne_core`, built by maturin as one abi3 wheel for CPython 3.12 and later). It
links no GPU library, and its output may not depend on the machine: no runtime CPU-feature dispatch,
no fused multiply-add, transcendental functions from the pure-Rust `libm` crate (`clippy.toml` forbids
the platform's), and every pixel a function of its own fragments only.

```python
import arachne_core as ac
from arachne.camera.camera import Camera

# Take the frame's size from the camera: CE fits its height to the pixels' aspect ratio
# (7.99375... at 854x480, not the configured 8.0).
camera = Camera(pixel_width=854, pixel_height=480)
delta = ac.FrameDelta(t=(0, 1), seq=0)
delta.set_view(
    pixel_width=camera.pixel_width,
    pixel_height=camera.pixel_height,
    frame_width=camera.frame_width,
    frame_height=camera.frame_height,
    background=(0.0, 0.0, 0.0, 1.0),
)
stream = ac.Stream()  # make it in the process that draws
stream.apply(delta)
frame = stream.draw(854, 480)  # uint8 (480, 854, 4), the layout of CE's pixel array
```

Today the core draws the background and `mesh` records, depth-tested, antialiased and composited in
depth order (`XF.md`, "Drawing"), and refuses, whole, any frame whose draw order holds a record it
cannot draw yet (`arachne_core.Unsupported`, `render.unsupported`). `stream.draw_resolves(...)` also
returns how each pixel was resolved.

- `src/xf/`: XF's types, content keys and binary form.
- `src/stream.rs`: the stream, applying deltas and drawing.
- `src/project.rs`: CE's camera maths (3D projection, the frame-to-pixel mappings), tested against
  Arachne's own cameras (`tests/data/gen_projection.py` writes the expected values).
- `src/shade.rs`: the one place lighting lives: CE's per-face model (tested against Arachne's own
  `ThreeDCamera`, `tests/data/gen_shading.py`) and soft diffuse light in linear light.
- `src/composite.rs`: the one place blending lives: fragments, the opaque and transparent passes, the
  coverage and sample resolves, emitted light, and the frame buffer. 2D paint blends with the
  arithmetic of Cairo's ARGB32 surface (`tests/data/gen_blend.py` records Cairo's bytes).
- `src/mesh.rs`: the triangle rasterizer, a producer of mask fragments with depth.
- `src/tiles.rs`: the frame cut into tiles, drawn on threads scoped to the draw call; the bytes are
  the same on any number of threads.

Build and test with the toolchain `rust-toolchain.toml` pins: `cargo test --release` in `core/`; the
binding with `uv build --wheel .` and `pytest` in `core/`. CI runs `ci/*.sh` (`.github/ci-kit.yml`),
and `.github/workflows/core-wheels.yml` builds the wheels.
