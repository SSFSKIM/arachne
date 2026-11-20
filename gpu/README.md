# arachne-gpu

Arachne's GPU build: pictures stacked as layers on a GPU through [wgpu](https://wgpu.rs) on
Vulkan, and the frame read back. It is a crate and a Python package of its own (`arachne_gpu`),
built and installed apart from the engine and from the CPU core (`core/`): the engine's package
never depends on it, a host with no GPU never needs it, and a scene that uses nothing of it is
drawn by Cairo and the core, to the same bytes whether it is installed or not.

## What it draws

A `Compositor(width, height)` holds textures of two kinds, each under a content key of its own:

- an **image**: straight RGBA, as an image object holds it, uploaded with `upload_image` under the
  Frame IR's image-shape key (`core/XF.md`, "Content keys"). Its values are converted to painted
  ones on upload, in integer arithmetic: `c' = (c * a + 127) / 255`;
- a **painted picture**: colour already multiplied by alpha, as Cairo leaves objects drawn on a
  transparent backdrop, uploaded with `upload_painted` under `painted_key(rgba)`, the same
  canonical form with a leading tag no Frame IR kind uses. Equal bytes of the two kinds are two
  textures.

`draw(Frame(background, layers))` stacks the layers bottom first over a painted background colour:
a `Picture` covers the frame (its texture has the frame's size); a `Quad` places a texture at four
corners in pixels, in CE's image order (upper left, upper right, lower left, lower right), with
nearest resampling. A layer's `opacity` scales it. Layers stack with the painted "over" rule in
Cairo's 8-bit integer arithmetic on the encoded values (never in linear light, never through an
sRGB format). A frame comes back painted, as `uint8` (height, width, 4) in the layout of CE's
pixel array. The stacking is integer arithmetic on integer textures, so the same layers stack to
the same bytes on every adapter. Where a quad lands is computed in floating point: on a pixel whose
centre falls within rounding of a texel's or the quad's edge, two adapters may choose different
texels. A frame asking for something this build does not draw (another resampling, a quad whose
outline is crossed, concave or has three corners on one line) is refused whole with `Unsupported`
(`render.unsupported`); corners on one line or at one point draw nothing, as CE draws such an
image.
A compositor is shared safely between threads: one draw holds it from its first pass to its
read-back.

## Effects

A `Layer` may carry effects, applied in this order: the layer is placed alone (its picture, quad or
window, at the frame's size), blurred, multiplied by its soft mask, then scaled by its opacity and
stacked as above.

- **Blur** (`Blur(sigma_px)`): a Gaussian of standard deviation `sigma_px` pixels, separable (a
  pass along rows, then one along columns), its kernel reaching `ceil(3 * sigma)` pixels each way.
  At the frame's border the picture's own edge value continues outward. A blurred step edge is
  2.563 sigma wide from its 10% to its 90% level. A blur keeps a constant picture exactly; it does
  not keep every picture's mean, since the continued border counts more than once (a bright border
  row raises the mean of a dark picture).
- **Depth under a focus** (`Layer(..., depth=d)` in `Frame(..., focus=Focus(depth, aperture))`): a
  blur of `aperture * |d - depth|` pixels, combined with the layer's own blur as the root of the sum
  of their squares; it is resolved here and nowhere else. A layer at the focus depth, or with a
  depth and no focus, is not defocused; its own blur, if it has one, still applies.
- **Soft mask** (`SoftMask(key, feather_px, invert=False)`): a picture of the frame's size, of
  either kind, whose alpha (and only its alpha) multiplies the layer. Its alpha is feathered by a
  Gaussian of sigma `feather_px / 3`, run along rows then columns, whose taps reach the whole
  pixels within `feather_px` along each axis. So the mask turns half way at its own edge, and the
  feather's cutoff is per axis: a pixel whose centre is farther than `feather_px` from every pixel
  across the edge, along both axes, is kept or removed whole. Near an edge at an angle to the axes,
  a pixel may be partly kept though it is farther than `feather_px` from the edge in a straight
  line: the feather reaches a square, as the comparator's grown footprint does. `invert` keeps
  what it would remove.
- **Window** (`Window(frame, corners, resampling)`): another stack, composed at the frame's size and
  placed at four corners as a quad is. Windows nest at most four deep.

**Arithmetic.** Three parts, and what each promises across adapters:

- *Kernel set-up*, on the CPU: each kernel's weights are computed in 64-bit floating point and
  rounded once to fixed point with 16 fractional bits, summing to exactly 1 (so a constant picture
  stays exactly constant). Every adapter is handed the same integer weights.
- *Pixel filtering and the final rounding*, on the device, in integers: identical on every
  adapter.
- *Placement* of a quad or a window, on the device, in floating point: which already-rounded texel
  a pixel copies may differ at a texel's or the outline's edge between adapters, and is not
  promised identical. It rounds no value.

The row pass reads the 8-bit placed layer and keeps 8 fractional bits; the column pass reads those
32-bit values and keeps 8 fractional bits again, so the second pass reads no 8-bit rounding of the
first. The mask's alpha is feathered the same way, into its own 32-bit values. The one rounding
back to 8 bits, half up, comes last, after the mask has multiplied the blurred layer; opacity then
scales the 8-bit result as it scales any layer. A layer whose effects do nothing (a blur of 0, a
depth at the focus with no blur of its own) is stacked as a plain layer, its bytes untouched.

A frame is refused whole for an effect out of range: a negative or non-finite sigma, feather or
aperture, a non-finite depth, a mask of another size (`ValueError`); a blur or feather that reaches
past the frame's longer side, or windows nested deeper than four (`Unsupported`).

## The object map

`object_map(frame)` returns one `Footprint` per layer that names an object (`object_id`), in the
frame's order: `object_id`; `layer`, its index; `box`, `(x0, y0, x1, y1)` in pixels, half-open,
y down; `mask`, a bool array of the box's size, true where the layer drawn with no effect puts any
alpha (its texels scaled by its opacity, as it would be stacked); and `grow_px`, how far its
effects spread it: three sigma of its blur, or its mask's feather, whichever is larger. Drawing the
frame without the layer changes pixels only inside its footprint grown by `grow_px` (to the next
whole pixel, as a square). Objects overlap and each keeps its whole footprint. A window is one
object to the frame that holds it: the objects inside it are not listed. The coverage is drawn on
the device, a bit per layer, 32 layers to a read-back, so the map is for sampled frames.

## A proposed Frame IR record

`arachne_gpu::composite::read_composite` reads the body of a proposed Frame IR record kind,
`composite`: the entries drawn into a picture of their own, with a blur, a depth, a focus, a window
and a soft mask. Its layout is stated in `src/composite.rs`; it is not yet part of `core/XF.md`,
and its kind code is not yet allocated. `tests/composite.rs` pins its fixtures' bytes.

## Which device draws

`Compositor(...)` draws on a GPU only: a discrete, integrated or virtual adapter on Vulkan. With
none it raises `NoGpu`, naming every adapter it saw, and nothing in the environment can make it
take a software adapter; a frame size the device cannot draw raises `ValueError`. `arachne_gpu.testing.software_compositor` is the one door to a software adapter
(Mesa's lavapipe), for tests, and its compositor says so (`software` is true). `Compositor.pin`
names the device that draws: `adapter`, `backend`, `driver` (the version string the adapter
reports) and `wgpu` (the locked version, also `arachne_gpu.__wgpu_version__`). `adapters()` lists
what the machine offers.

## Building and testing

The toolchain is pinned in `rust-toolchain.toml`; the build is Linux x86_64. A host needs the
Vulkan loader (`libvulkan1`) and a driver at run time; the build links neither.

    cargo test --release --locked                # Rust; needs ARACHNE_GPU_TEST_MODE
    uv build --wheel gpu                         # the wheel, from the repository's root

Every drawing test, Rust and Python, takes its compositor from one fixture, which reads the launch
mode from the test run's environment and fails with none set:

- `ARACHNE_GPU_TEST_MODE=software`: the testing door; asserts that the device that drew is of
  CPU type. This is CI's mode, on lavapipe (`mesa-vulkan-drivers`).
- `ARACHNE_GPU_TEST_MODE=hardware` with `ARACHNE_GPU_TEST_PIN` (a JSON object with `adapter`,
  `backend`, `driver` and `wgpu`): the ordinary door; asserts for every compositor a test draws
  with that it is no software adapter and that its pin is the one given.

The tests that draw frames of a frame's size draw at `ARACHNE_GPU_TEST_SIZE` (`<width>x<height>`,
at least 480 x 270; 480 x 270 when unset), so the same probes run at CI's size and at
`ARACHNE_GPU_TEST_SIZE=1920x1080` on a GPU; their closed-form bounds hold at both. Like the launch
mode, it is read by test code only. The CI gates (`.github/ci-kit.yml`): `gpu-lint` and `gpu-test`
when this directory changes; `gpu-binding` (the Python tests against the installed wheel) and
`cpu-bytes` (`ci/cpu_bytes.sh`: scenes using nothing new drawn with and without this build, byte
for byte) on every pull request.

`cpu-bytes` runs its two arms through `ci/cpu_bytes_arm.sh`, which installs and syncs nothing, so a
host that already has its environments can draw more arms, for example the same scenes with the
build and a GPU beside the two without one, and compare them all:

    bash gpu/ci/cpu_bytes_arm.sh <python> <arm directory> absent|present    # one arm
    python3 gpu/ci/cpu_bytes_compare.py <arm directory> <arm directory>...  # every arm equal

## Measuring a device: the frame set and the read-back benchmark

Two tools for a GPU host, built with the `bench` feature (`src/bench/`) and never part of the
wheel. Each opens its compositors through the ordinary door, so neither draws on a software
adapter; the crate's own tests run both on one at 480 x 270.

    cargo run --release --locked --features bench --example frameset -- --out <dir>
    cargo run --release --locked --features bench --example readback_bench -- --out <file.json>

- **The frame set** (`frameset`): the reference stack (six pictures of the frame's size: an
  opaque backdrop, two blurred at sigma 8 px, one under a soft mask, a window onto a stack of
  three, one plain) and the heavy stack (24 pictures, four blurred at sigma 24 px), each at
  1920 x 1080 and 3840 x 2160, and one frame per effect probe at 1920 x 1080. Every picture is
  drawn by code; the stacks' sigmas and feather are stated at 1080 px high and scale with the
  frame's height. Each frame is written as `<name>.rgba` (raw RGBA8 rows, top first, painted) with
  `manifest.json`: the set's version, the pin and adapter, and each frame's SHA-256 over its file's
  bytes. `--size WxH` draws every frame at one size; `--digests-only` writes the manifest alone.
- **The read-back benchmark** (`readback_bench`): for the reference and heavy stacks at each size,
  the mean and 95th percentile time per frame over `--frames` (300) after `--warmup` (30), with
  `draw`'s one-frame copy and with a pipelined copy (frame n copied out while frame n + 1 draws,
  through a ring of `--ring` (3) buffers); and the time of one frame from a warm compositor after
  every picture of the stack is uploaded anew, over `--uploads` (30). Every frame it reads back is
  checked against the stack's first draw (of the pipelined copy's timed frames, the last). Each
  part's start and end, in Unix seconds, are in the report, to read a GPU monitor sampled beside
  it against. The pipelined copy is benchmark code: `Compositor` has no call for it.
