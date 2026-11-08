from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import cairo
import pytest

import arachne


def _cpu() -> str:
    """The CPU's model, and whether it has AVX-512 (numpy's vectorised loops
    differ with it in their last bits).
    """
    try:
        info = Path("/proc/cpuinfo").read_text()
    except OSError:
        import platform

        return platform.processor() or platform.machine()
    model = next(
        (
            line.split(":", 1)[1].strip()
            for line in info.splitlines()
            if line.startswith("model name")
        ),
        "unknown",
    )
    flags = next((line for line in info.splitlines() if line.startswith("flags")), "")
    return f"{model} ({'avx512f' if ' avx512f' in flags else 'no avx512f'})"


@pytest.fixture(autouse=True)
def _snapping_keeps_ces_point_equality():
    """No path the test draws on the core has a join or a closed end that CE's
    point-equality rule decides otherwise on its snapped points
    (``frame_ir.canonical``).
    """
    from arachne.frame_ir import canonical

    canonical.point_equality_disagreements.clear()
    yield
    found = list(canonical.point_equality_disagreements)
    assert not found, f"(points, coordinates, boundaries) decided otherwise: {found}"


def pytest_report_header(config):
    return f"\nCairo Version: {cairo.cairo_version()}\nCPU: {_cpu()}"


def pytest_addoption(parser):
    parser.addoption(
        "--skip_slow",
        action="store_true",
        default=False,
        help="Will skip all the slow marked tests. Slow tests are arbitrarily marked as such.",
    )
    parser.addoption(
        "--show_diff",
        action="store_true",
        default=False,
        help="Will show a visual comparison if a graphical unit test fails.",
    )
    parser.addoption(
        "--set_test",
        action="store_true",
        default=False,
        help="Will create the control data for EACH running tests. ",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "skip_end_to_end: mark test as end_to_end test")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--skip_slow"):
        return
    else:
        slow_skip = pytest.mark.skip(
            reason="Slow test skipped due to --disable_slow flag.",
        )
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(slow_skip)


@pytest.fixture(autouse=True)
def draw_reads():
    """With ``ARACHNE_DRAW_READS`` set, every frame the test draws from the
    record is traced, and a read the record does not hold fails the test
    (:mod:`tests.helpers.draw_reads`).
    """
    from tests.helpers.draw_reads import ENV

    if not os.environ.get(ENV):
        yield
        return
    from tests.helpers.draw_reads import tracer

    traced = tracer()
    traced.take()
    yield
    found = traced.take()
    if found:
        pytest.fail(
            "drawing from the record read what the record does not hold:\n"
            + "\n".join(f"  {read}" for read in found),
            pytrace=False,
        )


@pytest.fixture(autouse=True)
def answer_store(tmp_path_factory, monkeypatch):
    """A store of answers of the test's own (``ARACHNE_STORE``), which every
    session server it starts inherits: no test reads or fills the user's.
    """
    monkeypatch.setenv("ARACHNE_STORE", str(tmp_path_factory.mktemp("store")))


@pytest.fixture(autouse=True)
def temp_media_dir(tmpdir, monkeypatch, request):
    if isinstance(request.node, pytest.DoctestItem):
        monkeypatch.chdir(tmpdir)
        yield tmpdir
    else:
        with arachne.tempconfig({"media_dir": str(tmpdir)}):
            assert arachne.config.media_dir == str(tmpdir)
            yield tmpdir


@pytest.fixture
def manim_caplog(caplog):
    logger = logging.getLogger("arachne")
    logger.propagate = True
    caplog.set_level(logging.INFO, logger="arachne")
    yield caplog
    logger.propagate = False


@pytest.fixture
def config():
    saved = arachne.config.copy()
    # we need to return the actual config so that tests
    # using tempconfig pass
    yield arachne.config
    arachne.config.update(saved)


@pytest.fixture
def dry_run(config):
    config.dry_run = True


@pytest.fixture(scope="session")
def python_version():
    # use the same python executable as it is running currently
    # rather than randomly calling using python or python3, which
    # may create problems.
    return sys.executable


@pytest.fixture
def reset_cfg_file():
    cfgfilepath = Path(__file__).parent / "test_cli" / "arachne.cfg"
    original = cfgfilepath.read_text()
    yield
    cfgfilepath.write_text(original)
