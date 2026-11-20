"""The device a scene draws on, and the two refusals: ``render.gpu_required``
where a GPU-route scene meets a host with no GPU, ``render.gpu_live`` where
it must be drawn as it runs.
"""

from __future__ import annotations

import importlib.util
import sys
import types

import pytest

from arachne.lint import CODES
from arachne.lint.findings import Finding
from arachne.routing import (
    GpuLive,
    GpuRequired,
    cache_token,
    capabilities,
    gpu,
    pin_for,
)
from arachne.server.dispatch import REFUSALS, _error_answer
from arachne.server.protocol import REFUSED

from .conftest import line_of

PIN = {
    "adapter": "A Card",
    "backend": "Vulkan",
    "driver": "1.2.3",
    "wgpu": "30.0.2",
}


def _adapter(name: str, device_type: str, driver_info: str = "1.2.3") -> dict:
    return {
        "name": name,
        "backend": "Vulkan",
        "device_type": device_type,
        "driver": name.lower(),
        "driver_info": driver_info,
        "vendor": 0,
        "device": 0,
    }


@pytest.fixture
def no_build(monkeypatch):
    """This process as one with no GPU build: importing it fails, whether
    or not it is installed.
    """
    for name in [n for n in sys.modules if n.split(".")[0] == "arachne_gpu"]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "arachne_gpu", None)


@pytest.fixture
def build_offering(monkeypatch):
    """This process with a GPU build that offers the given adapters."""

    def offer(*adapters: dict) -> None:
        module = types.ModuleType("arachne_gpu")
        module.__wgpu_version__ = "30.0.2"  # type: ignore[attr-defined]
        module.adapters = lambda: [dict(a) for a in adapters]  # type: ignore[attr-defined]
        for name in [n for n in sys.modules if n.split(".")[0] == "arachne_gpu"]:
            monkeypatch.delitem(sys.modules, name)
        monkeypatch.setitem(sys.modules, "arachne_gpu", module)

    return offer


class _Broken:
    """A finder for an installed GPU build whose import fails, as one whose
    native part cannot load does.
    """

    def __init__(self, error: ImportError) -> None:
        self.error = error

    def find_spec(self, name, path=None, target=None):
        if name != "arachne_gpu":
            return None
        return importlib.util.spec_from_loader(name, self)

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        raise self.error


def test_without_the_gpu_build_there_is_no_gpu(no_build):
    assert gpu() is None


@pytest.mark.parametrize(
    "error",
    [
        ModuleNotFoundError(
            "No module named 'arachne_gpu._gpu'", name="arachne_gpu._gpu"
        ),
        ImportError("libgcc_s.so.1: cannot open shared object file"),
    ],
)
def test_a_broken_gpu_build_is_raised_not_taken_for_no_build(monkeypatch, error):
    for name in [n for n in sys.modules if n.split(".")[0] == "arachne_gpu"]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, "meta_path", [_Broken(error), *sys.meta_path])
    with pytest.raises(ImportError) as raised:
        gpu()
    assert raised.value is error


def test_a_cpu_route_scene_needs_no_device(session_of, no_build):
    assert pin_for(session_of("Plain").record) is None


def test_with_no_gpu_build_a_gpu_route_scene_is_refused_with_its_finding(
    session_of, no_build
):
    record = session_of("LastPlay").record
    with pytest.raises(GpuRequired) as raised:
        pin_for(record)
    error = raised.value
    assert error.code == "render.gpu_required"
    assert error.capabilities == capabilities(record)
    finding = error.finding
    assert isinstance(finding, Finding)
    assert (finding.code, finding.severity) == ("render.gpu_required", "error")
    assert finding.line == line_of("last-play")
    assert finding.t_start == record.plays[1].start
    assert dict(finding.detail) == {
        "capabilities": ["effect.blur"],
        "play": 1,
        "object": capabilities(record)["effect.blur"].object,
    }
    assert "lint" in finding.fix
    assert "state_at" in finding.fix
    message = str(error)
    assert message.startswith("render.gpu_required: ")
    assert "effect.blur" in message
    assert str(line_of("last-play")) in message


@pytest.mark.parametrize(
    "adapters",
    [
        (),
        (_adapter("llvmpipe", "Cpu"),),
        (_adapter("llvmpipe", "Cpu"), _adapter("Other", "Other")),
    ],
)
def test_a_build_offering_no_gpu_adapter_is_no_gpu(
    session_of, build_offering, adapters
):
    build_offering(*adapters)
    assert gpu() is None
    with pytest.raises(GpuRequired):
        pin_for(session_of("LastPlay").record)


def test_the_pin_is_of_the_adapter_a_compositor_would_open(build_offering):
    """Discrete before integrated before virtual, the first seen within one
    kind; never a software adapter.
    """
    build_offering(
        _adapter("llvmpipe", "Cpu"),
        _adapter("Virtual", "VirtualGpu"),
        _adapter("Integrated", "IntegratedGpu"),
        _adapter("First Card", "DiscreteGpu", "580.1"),
        _adapter("Second Card", "DiscreteGpu", "580.2"),
    )
    assert gpu() == {
        "adapter": "First Card",
        "backend": "Vulkan",
        "driver": "580.1",
        "wgpu": "30.0.2",
    }
    build_offering(_adapter("llvmpipe", "Cpu"), _adapter("Virtual", "VirtualGpu"))
    assert gpu()["adapter"] == "Virtual"


def test_a_gpu_route_scene_with_a_gpu_is_given_its_pin(session_of, build_offering):
    build_offering(_adapter("A Card", "DiscreteGpu"))
    assert pin_for(session_of("LastPlay").record) == PIN
    assert pin_for(session_of("Plain").record) is None


def test_the_live_refusal_names_the_capability_and_why_it_draws_live(session_of):
    record = session_of("Refused").record
    error = GpuLive(capabilities(record), record.refusal)
    assert error.code == "render.gpu_live"
    assert (error.finding.code, error.finding.severity) == ("render.gpu_live", "error")
    assert error.finding.line == line_of("refused")
    assert error.finding.detail["cause"] == record.refusal
    message = str(error)
    assert message.startswith("render.gpu_live: ")
    assert "effect.blur" in message
    assert str(line_of("refused")) in message
    # The cause travels in the message, the one text a refusal's answer carries.
    assert record.refusal in message
    assert record.refusal in _error_answer(error, None).data["message"]


def test_the_live_refusal_with_no_known_cause_says_none(session_of):
    error = GpuLive(capabilities(session_of("LastPlay").record), None)
    assert "None" not in str(error)
    assert "It draws as it runs:" not in str(error)


def test_both_codes_are_errors_appended_last_to_the_table():
    *_, required, live = CODES.values()
    assert (required.code, required.severity, required.cls) == (
        "render.gpu_required",
        "error",
        "GpuRequired",
    )
    assert (live.code, live.severity, live.cls) == (
        "render.gpu_live",
        "error",
        "GpuLive",
    )


def test_the_session_server_answers_both_as_refusals(session_of, no_build):
    record = session_of("LastPlay").record
    for error in (
        GpuRequired(capabilities(record)),
        GpuLive(capabilities(record), "drawn as it runs"),
    ):
        assert isinstance(error, REFUSALS)
        answer = _error_answer(error, None)
        assert answer is not None
        assert answer.code == REFUSED
        assert answer.data["type"] == type(error).__name__
        assert answer.data["reason"] == error.code


def test_the_cache_token_is_empty_on_the_cpu_route_and_names_the_pin_on_the_gpu():
    assert cache_token("cpu", None) == ""
    token = cache_token("gpu", PIN)
    assert token
    assert token == cache_token("gpu", dict(reversed(PIN.items())))
    for field in PIN:
        assert cache_token("gpu", {**PIN, field: PIN[field] + "x"}) != token
    with pytest.raises(ValueError):
        cache_token("gpu", None)
    with pytest.raises(ValueError):
        cache_token("cpu", PIN)
    with pytest.raises(ValueError):
        cache_token("metal", PIN)
