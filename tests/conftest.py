from __future__ import annotations

import contextlib
import logging
import sys
from collections.abc import Iterator
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


@contextlib.contextmanager
def point_equality_guard(*, unmeasured: bool = False) -> Iterator[None]:
    """No path snapped inside it has a join or a closed end that CE's
    point-equality rule decides otherwise on its snapped points
    (``frame_ir.canonical``), and every path shape snapped inside it was
    measured, so it never passes on nothing: production does not measure
    (``canonical.MEASURE``), and the test session turns measuring on
    (:func:`pytest_configure`). ``unmeasured``: a test that snaps unmeasured
    on purpose (``@pytest.mark.snaps_unmeasured``).
    """
    from arachne.frame_ir import canonical

    assert canonical.MEASURE, (
        "the test session measures point equality: canonical.MEASURE is off"
    )
    canonical.point_equality_disagreements.clear()
    counts = canonical.point_equality_counts
    counts.update(measured=0, unmeasured=0)
    try:
        yield
    finally:
        found = list(canonical.point_equality_disagreements)
        missed = counts["unmeasured"]
        canonical.point_equality_disagreements.clear()
        counts.update(measured=0, unmeasured=0)
    assert not found, f"(points, coordinates, boundaries) decided otherwise: {found}"
    assert unmeasured or not missed, (
        f"{missed} path shapes were snapped without the point-equality measure"
    )


@pytest.fixture(autouse=True)
def _snapping_keeps_ces_point_equality(request):
    """Every test runs inside :func:`point_equality_guard`."""
    with point_equality_guard(
        unmeasured=request.node.get_closest_marker("snaps_unmeasured") is not None
    ):
        yield


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
    # str and bytes hashes are salted per process unless PYTHONHASHSEED is set,
    # and iterating a set of them orders some scenes' drawing: frames compare
    # byte for byte only at one seed, which takes effect at interpreter start.
    if sys.flags.hash_randomization:
        raise pytest.UsageError(
            "the test suite runs with hash randomization off: run it as "
            "PYTHONHASHSEED=0 uv run pytest"
        )
    config.addinivalue_line("markers", "skip_end_to_end: mark test as end_to_end test")
    config.addinivalue_line(
        "markers",
        "snaps_unmeasured: the test snaps without the point-equality measure on purpose",
    )
    # Production snaps without the point-equality measure; every test measures.
    from arachne.frame_ir import canonical

    canonical.MEASURE = True


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
