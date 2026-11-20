"""The device a drawing test above the GPU build draws on, chosen by the run's
launch mode as the GPU build's own tests choose theirs
(``gpu/tests/python/conftest.py``).

A scene's frames find their device through the engine's door
(:data:`arachne.compose.device.door`); :func:`install` replaces it, for one
test, with a :class:`LaunchDoor` of this run's mode:

- ``ARACHNE_GPU_TEST_MODE=software`` (CI): the GPU build's testing door, and
  every compositor it opens is checked to be on a CPU-type adapter;
- ``ARACHNE_GPU_TEST_MODE=hardware`` with ``ARACHNE_GPU_TEST_PIN`` (a JSON
  object with ``adapter``, ``backend``, ``driver`` and ``wgpu``): the GPU
  build's ordinary constructor, and every compositor it opens is checked to
  be on no software adapter and at that pin.

With no mode set the door refuses to open, so a drawing test fails. The GPU
build is imported only when a door opens: a test file that draws is left out
by collection where the build is not installed, never skipped.

A conftest of drawing tests calls :func:`report_adapters` from its
``pytest_configure``, so the run ends with one line per device that drew.

A second process is started by exec (:func:`run_child`), never by fork: a
process that has drawn holds a Vulkan instance a forked child may not use.
"""

from __future__ import annotations

__all__ = [
    "MODE",
    "PIN",
    "Drawn",
    "LaunchDoor",
    "install",
    "launch_mode",
    "report_adapters",
    "run_child",
]

import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

MODE = "ARACHNE_GPU_TEST_MODE"
PIN = "ARACHNE_GPU_TEST_PIN"
ROOT = Path(__file__).resolve().parents[2]

#: The adapters this process's compositors were opened on.
_drew_on: list[dict] = []


def launch_mode(environ: Mapping[str, str]) -> tuple[str, dict | None]:
    """The launch mode and, in hardware mode, the pin it must find; raises
    when it is unset or wrong.
    """
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
            f"{MODE} is not set: drawing tests run in software mode (CI) or"
            " hardware mode (a GPU host)"
        )
    raise RuntimeError(f"{MODE}={mode}: the modes are software and hardware")


def _opened(width: int, height: int) -> Any:
    """A compositor of this run's mode, checked to be on its kind of
    adapter.
    """
    import arachne_gpu

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


class Drawn:
    """A compositor that counts the frames it draws and, while its door
    keeps maps, keeps the object map of each (``Compositor.object_map``),
    taken before the frame is drawn, while what it stacks is uploaded.
    """

    def __init__(self, inner: Any, door: LaunchDoor) -> None:
        self._inner = inner
        self._door = door
        self.draws = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def draw(self, frame: Any) -> Any:
        if self._door.keep_maps:
            self._door.maps.append(list(self._inner.object_map(frame)))
        self.draws += 1
        return self._inner.draw(frame)


class LaunchDoor:
    """The engine's door (:class:`arachne.compose.device.Door`) onto this
    run's device. ``opened`` holds every compositor it opened; with
    ``keep_maps`` set, ``maps`` gets each drawn frame's object map.
    """

    def __init__(self) -> None:
        launch_mode(os.environ)
        self.opened: list[Drawn] = []
        self.keep_maps = False
        self.maps: list[list[Any]] = []
        self._pin: dict[str, str] | None = None

    def pin(self) -> dict[str, str] | None:
        if self._pin is None:
            mode, _ = launch_mode(os.environ)
            if mode == "software":
                from arachne_gpu.testing import software_compositor

                self._pin = dict(software_compositor(4, 4).pin)
            else:
                from arachne import routing

                self._pin = routing.gpu()
        return self._pin

    def open(self, width: int, height: int) -> Drawn:
        self.opened.append(Drawn(_opened(width, height), self))
        return self.opened[-1]

    @property
    def draws(self) -> int:
        """How many frames its compositors drew."""
        return sum(opened.draws for opened in self.opened)


def install(monkeypatch: pytest.MonkeyPatch) -> LaunchDoor:
    """Make this run's device the one the engine draws on, for one test."""
    from arachne.compose import device

    door = LaunchDoor()
    monkeypatch.setattr(device, "door", door)
    return door


def run_child(code: str, *args: str, env: Mapping[str, str] | None = None) -> str:
    """Run ``code`` with ``args`` in a fresh interpreter of this environment,
    started by exec from the repository's root, and return what it printed.
    """
    done = subprocess.run(
        [sys.executable, "-c", code, *args],
        cwd=ROOT,
        env={**os.environ, **(env or {})},
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    return done.stdout


def _adapter_line(adapter: dict) -> str:
    return (
        f'ADAPTER: name="{adapter["name"]}" backend={adapter["backend"]}'
        f" device_type={adapter['device_type']} vendor=0x{adapter['vendor']:04x}"
        f' device=0x{adapter["device"]:04x} driver="{adapter["driver"]}"'
        f' driver_info="{adapter["driver_info"]}"'
    )


class _Adapters:
    """The hooks that print, once a run ends, one line per device its
    drawing tests drew on, gathered from xdist's workers too.
    """

    OUTPUT = "arachne_drawing_tests_drew_on"

    def __init__(self) -> None:
        self.from_workers: set[str] = set()

    def pytest_sessionfinish(self, session: pytest.Session) -> None:
        output = getattr(session.config, "workeroutput", None)
        if output is not None:
            output[self.OUTPUT] = sorted({_adapter_line(a) for a in _drew_on})

    @pytest.hookimpl(optionalhook=True)
    def pytest_testnodedown(self, node: Any, error: Any) -> None:
        self.from_workers.update(getattr(node, "workeroutput", {}).get(self.OUTPUT, []))

    def pytest_terminal_summary(self, terminalreporter: Any) -> None:
        lines = {_adapter_line(a) for a in _drew_on} | self.from_workers
        for line in sorted(lines):
            terminalreporter.write_line(line)


def report_adapters(config: pytest.Config) -> None:
    """Have the run print the adapters its drawing tests drew on, once
    however many conftests ask.
    """
    name = "arachne-drawing-adapters"
    if not config.pluginmanager.has_plugin(name):
        config.pluginmanager.register(_Adapters(), name)
