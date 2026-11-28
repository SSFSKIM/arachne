from __future__ import annotations

import contextlib
import dataclasses
import importlib
import logging
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

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


#: The module whose evidence the guard reads, always through ``sys.modules``:
#: a reload re-executes it in place and replaces what it holds.
_CANONICAL = "arachne.frame_ir.canonical"


def _canonical() -> Any:
    importlib.import_module(_CANONICAL)
    return sys.modules[_CANONICAL]


@dataclasses.dataclass(frozen=True)
class _Mark:
    """Where the snap's evidence stood: its own list and counters, how many
    disagreements the list held and the shapes counted. The evidence is
    never cleared by a guard, so marks nest.
    """

    state: tuple[list, dict]
    found: int
    measured: int
    unmeasured: int


def _mark() -> _Mark:
    module = _canonical()
    found, counts = module.point_equality_disagreements, module.point_equality_counts
    return _Mark((found, counts), len(found), counts["measured"], counts["unmeasured"])


def _since(mark: _Mark, *, unmeasured: bool) -> list[str]:
    """What fails since ``mark``: a reload, evidence cleared, a disagreement,
    or (unless ``unmeasured``) a shape snapped without the measure.
    """
    module = _canonical()
    found, counts = module.point_equality_disagreements, module.point_equality_counts
    if found is not mark.state[0] or counts is not mark.state[1]:
        return [
            f"{_CANONICAL} was reloaded: the point-equality evidence and "
            "canonical.MEASURE it held were replaced"
        ]
    if (
        len(found) < mark.found
        or counts["measured"] < mark.measured
        or counts["unmeasured"] < mark.unmeasured
    ):
        return [
            "the point-equality evidence was cleared or reset: consume a "
            "disagreement made on purpose with expected_disagreements()"
        ]
    problems = []
    made = found[mark.found :]
    if made:
        problems.append(f"(points, coordinates, boundaries) decided otherwise: {made}")
    missed = counts["unmeasured"] - mark.unmeasured
    if missed and not unmeasured:
        problems.append(
            f"{missed} path shapes were snapped without the point-equality measure"
        )
    return problems


@contextlib.contextmanager
def point_equality_guard(*, unmeasured: bool = False) -> Iterator[None]:
    """No path snapped inside it has a join or a closed end that CE's
    point-equality rule decides otherwise on its snapped points
    (``frame_ir.canonical``), and every path shape snapped inside it was
    measured, so it never passes on nothing: production does not measure
    (``canonical.MEASURE``), and the test session turns measuring on
    (:func:`pytest_configure`). ``unmeasured``: snapping unmeasured is
    expected. It reads the evidence and clears none of it, so a guard
    inside it leaves it every observation; the session checks every test's
    phases the same way (:func:`pytest_runtest_makereport`).
    """
    assert _canonical().MEASURE, (
        "the test session measures point equality: canonical.MEASURE is off"
    )
    mark = _mark()
    yield
    problems = _since(mark, unmeasured=unmeasured)
    if not problems and not _canonical().MEASURE:
        problems.append("canonical.MEASURE was turned off inside the guard")
    assert not problems, "; ".join(problems)


@contextlib.contextmanager
def expected_disagreements() -> Iterator[list[tuple[int, int, int]]]:
    """The disagreements recorded inside it, made on purpose: handed to the
    test in the list it yields and taken off the evidence, so no guard
    counts them. Every other observation stays.
    """
    module = _canonical()
    found = module.point_equality_disagreements
    start = len(found)
    taken: list[tuple[int, int, int]] = []
    yield taken
    assert _canonical().point_equality_disagreements is found, (
        f"{_CANONICAL} was reloaded inside expected_disagreements()"
    )
    assert len(found) >= start, "the point-equality evidence was cleared"
    taken.extend(found[start:])
    del found[start:]


#: Where the session's evidence was last checked: every phase of every test
#: (its setup with the higher-scoped fixtures it sets up, its call, its
#: teardown with those it tears down) is checked from there, and what was
#: snapped outside any test (collection, between tests) fails the next one.
#: Zero: the module's counters start at its import.
_checked: _Mark | None = None
#: What the session found outside any test, by test, before its setup.
_outside: dict[str, list[str]] = {}


def _check(*, unmeasured: bool) -> list[str]:
    global _checked
    mark = _checked
    if mark is None:
        module = _canonical()
        state = (module.point_equality_disagreements, module.point_equality_counts)
        mark = _Mark(state, 0, 0, 0)
    problems = _since(mark, unmeasured=unmeasured)
    _checked = _mark()
    if any("reloaded" in p for p in problems):
        # The reload is this test's failure, not every later one's.
        _canonical().MEASURE = True
    return problems


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    _outside[item.nodeid] = [
        f"outside any test (collection or between tests): {p}"
        for p in _check(unmeasured=False)
    ]
    yield
    _outside.pop(item.nodeid, None)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """A phase fails on what it snapped: a disagreement, a reload, evidence
    cleared, or a shape snapped unmeasured in a test not marked
    ``snaps_unmeasured``; its setup also on what was snapped outside any
    test before it, and on a session that does not measure.
    """
    outcome = yield
    report = outcome.get_result()
    problems = _check(
        unmeasured=item.get_closest_marker("snaps_unmeasured") is not None
    )
    if call.when == "setup":
        problems = _outside.pop(item.nodeid, []) + problems
        if not _canonical().MEASURE:
            problems.append(
                "the test session measures point equality: canonical.MEASURE is off"
            )
    if not problems:
        return
    text = f"point equality, {call.when}: " + "; ".join(problems)
    if report.failed:
        report.sections.append(("point equality", text))
        return
    report.outcome = "failed"
    report.longrepr = text
    if hasattr(report, "wasxfail"):
        del report.wasxfail


def pytest_sessionfinish(session, exitstatus):
    """What was snapped after the last test's teardown, or in a session that
    ran none, fails the session.
    """
    problems = _check(unmeasured=False)
    if problems:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            reporter.write_line(
                "point equality, outside any test: " + "; ".join(problems), red=True
            )


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
        "snaps_unmeasured(reason): the test snaps without the point-equality measure "
        "on purpose",
    )
    # Production snaps without the point-equality measure; every test measures.
    global _measure_before
    module = _canonical()
    _measure_before = module.MEASURE
    module.MEASURE = True


#: ``canonical.MEASURE`` before the session turned it on.
_measure_before = False


def pytest_unconfigure(config):
    _canonical().MEASURE = _measure_before


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
