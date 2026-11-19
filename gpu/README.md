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

Tests draw at 480 x 270 or smaller. The CI gates (`.github/ci-kit.yml`): `gpu-lint` and `gpu-test`
when this directory changes; `gpu-binding` (the Python tests against the installed wheel) and
`cpu-bytes` (`ci/cpu_bytes.sh`: scenes using nothing new drawn with and without this build, byte
for byte) on every pull request.
