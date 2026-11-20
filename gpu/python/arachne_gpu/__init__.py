"""Arachne's GPU build: pictures stacked as layers on a GPU, and the frame read back.

A :class:`Compositor` holds textures of two kinds. An **image** is straight RGBA, as an image
object holds it, uploaded under its Frame IR image-shape key (``upload_image``) and converted to
painted values there, once. A **painted picture** holds painted values, colour already multiplied
by alpha, as Cairo leaves a run of objects drawn on a transparent backdrop, uploaded under
:func:`painted_key` (``upload_painted``). ``Compositor.draw`` stacks a :class:`Frame`'s layers in
drawing order with the painted "over" rule in Cairo's 8-bit arithmetic and returns a uint8 array
(height, width, 4), painted, in the layout of CE's pixel array.

A layer is a :class:`Picture`, a :class:`Quad` or a :class:`Window` onto another stack, and can
carry a :class:`Blur`, a declared ``depth`` blurred under the frame's :class:`Focus`, and a
:class:`SoftMask`. Pixels are filtered on integers with weights set up once on the CPU, and rounded
back to 8 bits once, so filtering and rounding are the same on every adapter; only where a quad or
a window lands is floating point, and may pick another texel at an edge on another adapter.
``Compositor.object_map`` gives a :class:`Footprint` for each layer that names an object: the
pixels it covers before any effect, and how far its effects spread it.

``Compositor(width, height)`` draws only on a GPU (a discrete, integrated or virtual adapter on
Vulkan) and raises :class:`NoGpu` otherwise. A software adapter is for tests, and the one door to
it is ``arachne_gpu.testing``. ``Compositor.pin`` names the device that draws: the adapter, the
backend, the driver version string and the wgpu version. In a process forked after its parent first
used this build, opening a compositor or listing ``adapters()`` raises
:class:`ForkedProcessError`.
"""

from __future__ import annotations

from arachne_gpu._errors import ForkedProcessError, NoGpu, Unsupported
from arachne_gpu._frame import (
    Blur,
    Focus,
    Footprint,
    Frame,
    Layer,
    Picture,
    Quad,
    SoftMask,
    Window,
)
from arachne_gpu._gpu import WGPU_VERSION as __wgpu_version__
from arachne_gpu._gpu import Compositor, adapters, painted_key

__all__ = [
    "Blur",
    "Compositor",
    "Focus",
    "Footprint",
    "ForkedProcessError",
    "Frame",
    "Layer",
    "NoGpu",
    "Picture",
    "Quad",
    "SoftMask",
    "Unsupported",
    "Window",
    "__wgpu_version__",
    "adapters",
    "painted_key",
]
