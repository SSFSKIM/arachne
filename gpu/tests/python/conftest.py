"""The compositor every drawing test draws with, chosen by the run's launch mode.

- ``ARACHNE_GPU_TEST_MODE=software`` (CI): the testing door; every compositor handed out is
  checked to be on a CPU-type adapter.
- ``ARACHNE_GPU_TEST_MODE=hardware`` with ``ARACHNE_GPU_TEST_PIN`` (a JSON object with
  ``adapter``, ``backend``, ``driver`` and ``wgpu``): the ordinary door; every compositor is checked
  to be on no software adapter and at that pin.

With no mode set every drawing test fails. Test code is the only reader of the mode, and of the
size frame-sized tests draw at (``gpu_helpers.drawing_size``).
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping

import arachne_gpu
import pytest
from gpu_helpers import H, W

MODE = "ARACHNE_GPU_TEST_MODE"
PIN = "ARACHNE_GPU_TEST_PIN"

_drew_on: list[dict] = []


def launch_mode(environ: Mapping[str, str]) -> tuple[str, dict | None]:
    """The launch mode and, in hardware mode, the pin it must find; raises when unset or wrong."""
    mode = environ.get(MODE)
    if mode == "software":
        return mode, None
    if mode == "hardware":
        if PIN not in environ:
            raise RuntimeError(f"hardware mode needs {PIN}")
        pin = json.loads(environ[PIN])
        if set(pin) != {"adapter", "backend", "driver", "wgpu"}:
            raise RuntimeError(
                f"{PIN} must name adapter, backend, driver and wgpu: {pin}"
            )
        return mode, pin
    if mode is None:
        raise RuntimeError(
            f"{MODE} is not set: drawing tests run in software mode (CI) or hardware mode"
            " (a GPU host)"
        )
    raise RuntimeError(f"{MODE}={mode}: the modes are software and hardware")


def checked_compositor(width: int, height: int) -> arachne_gpu.Compositor:
    mode, pin = launch_mode(os.environ)
    if mode == "software":
        from arachne_gpu.testing import software_compositor

        compositor = software_compositor(width, height)
        assert compositor.software
        assert compositor.adapter["device_type"] == "Cpu", compositor.adapter
    else:
        compositor = arachne_gpu.Compositor(width, height)
        assert not compositor.software
        assert compositor.adapter["device_type"] != "Cpu", compositor.adapter
        assert compositor.pin == pin, (compositor.pin, pin)
    _drew_on.append(compositor.adapter)
    return compositor


@pytest.fixture
def make_compositor() -> Callable[[int, int], arachne_gpu.Compositor]:
    return checked_compositor


def adapter_line(adapter: dict) -> str:
    return (
        f'ADAPTER: name="{adapter["name"]}" backend={adapter["backend"]}'
        f" device_type={adapter['device_type']} vendor=0x{adapter['vendor']:04x}"
        f' device=0x{adapter["device"]:04x} driver="{adapter["driver"]}"'
        f' driver_info="{adapter["driver_info"]}"'
    )


#: The adapter lines xdist's workers drew on, as they handed them to the controller.
_workers_drew_on: set[str] = set()
WORKER_OUTPUT = "arachne_gpu_drew_on"


def pytest_sessionfinish(session):
    # An xdist worker hands the adapters it drew on to the controller, whose summary prints them.
    output = getattr(session.config, "workeroutput", None)
    if output is not None:
        output[WORKER_OUTPUT] = sorted({adapter_line(a) for a in _drew_on})


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node, error):
    _workers_drew_on.update(getattr(node, "workeroutput", {}).get(WORKER_OUTPUT, []))


def pytest_terminal_summary(terminalreporter):
    for line in sorted({adapter_line(a) for a in _drew_on} | _workers_drew_on):
        terminalreporter.write_line(line)
    terminalreporter.write_line(f"SIZE: {W}x{H}")
