"""Arachne's GPU build: pictures stacked as layers on a GPU, and the frame read back.

A :class:`Compositor` holds textures of two kinds. An **image** is straight RGBA, as an image
object holds it, uploaded under its Frame IR image-shape key (``upload_image``) and converted to
painted values there, once. A **painted picture** holds painted values, colour already multiplied
by alpha, as Cairo leaves a run of objects drawn on a transparent backdrop, uploaded under
:func:`painted_key` (``upload_painted``). ``Compositor.draw`` stacks a :class:`Frame`'s layers in
drawing order with the painted "over" rule in Cairo's 8-bit arithmetic and returns a uint8 array
(height, width, 4), painted, in the layout of CE's pixel array.

``Compositor(width, height)`` draws only on a GPU (a discrete, integrated or virtual adapter on
Vulkan) and raises :class:`NoGpu` otherwise. A software adapter is for tests, and the one door to
it is ``arachne_gpu.testing``. ``Compositor.pin`` names the device that draws: the adapter, the
backend, the driver version string and the wgpu version.
"""

from __future__ import annotations

from arachne_gpu._errors import NoGpu, Unsupported
from arachne_gpu._frame import Frame, Layer, Picture, Quad
from arachne_gpu._gpu import WGPU_VERSION as __wgpu_version__
from arachne_gpu._gpu import Compositor, adapters, painted_key

__all__ = [
    "Compositor",
    "Frame",
    "Layer",
    "NoGpu",
    "Picture",
    "Quad",
    "Unsupported",
    "__wgpu_version__",
    "adapters",
    "painted_key",
]
