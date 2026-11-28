"""The session's point-equality check (``tests/conftest.py``) covers what
no test body does: collection, the setup and teardown of fixtures of a
wider scope, a reload of the snap's module, and what is left after the last
test. Each case runs a session of its own, with this suite's conftest.
"""

from __future__ import annotations

import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

CONFTEST = Path(__file__).with_name("conftest.py")

HELPERS = """
import importlib

import pytest

from arachne.frame_ir import canonical


def unmeasured():
    canonical.point_equality_counts["unmeasured"] += 1


def disagree():
    canonical.point_equality_disagreements.append((8, 2, 1))
"""

#: The session run in-process, so what it leaves ``canonical.MEASURE`` at
#: shows.
RUN = (
    "import sys, pytest\n"
    "from arachne.frame_ir import canonical\n"
    "code = pytest.main(sys.argv[1:])\n"
    "print('MEASURE after the session:', canonical.MEASURE)\n"
    "sys.exit(code)\n"
)


def _session(tmp_path: Path, files: dict[str, str]) -> tuple[int, str, dict]:
    """A session over ``files``: its exit code, its output, and each test's
    outcomes by name (``failure``, ``error`` with its message, ``passed``).
    """
    root = tmp_path / "suite"
    root.mkdir()
    (root / "conftest.py").write_text(CONFTEST.read_text())
    for name, body in files.items():
        (root / name).write_text(HELPERS + body)
    junit = tmp_path / "junit.xml"
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST")}
    env["PYTHONHASHSEED"] = "0"
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            RUN,
            "-p",
            "no:cacheprovider",
            "-o",
            "addopts=",
            f"--junitxml={junit}",
            str(root),
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
    )
    cases: dict[str, list[tuple[str, str]]] = {}
    if junit.exists():
        for case in ET.parse(junit).iter("testcase"):
            found = [
                (child.tag, f"{child.get('message', '')} {child.text or ''}")
                for child in case
                if child.tag in ("failure", "error", "skipped")
            ]
            cases[case.get("name")] = found or [("passed", "")]
    return out.returncode, out.stdout + out.stderr, cases


def _only(cases: dict, name: str, tag: str, *says: str) -> None:
    (found,) = cases[name]
    assert found[0] == tag, (name, found)
    for said in says:
        assert said in found[1], (name, said, found)


def test_wider_fixtures_a_reload_and_the_last_teardown_are_checked(tmp_path):
    code, output, cases = _session(
        tmp_path,
        {
            "test_a.py": """
@pytest.fixture(scope="module")
def unmeasured_at_setup():
    unmeasured()
    yield


def test_never_snaps():
    pass


def test_a_module_fixture_set_up(unmeasured_at_setup):
    pass
""",
            "test_b.py": """
@pytest.fixture(scope="module")
def disagrees_at_teardown():
    yield
    disagree()


@pytest.fixture(scope="session")
def disagrees_at_session_end():
    yield
    disagree()


def test_a_module_fixture_torn_down(disagrees_at_teardown, disagrees_at_session_end):
    pass
""",
            "test_c.py": """
@pytest.fixture(scope="module")
def unmeasured_set_up():
    unmeasured()
    yield


@pytest.mark.snaps_unmeasured
def test_a_marked_test_owns_its_setup(unmeasured_set_up):
    pass


def test_an_unmarked_test_snaps_unmeasured():
    unmeasured()


@pytest.mark.snaps_unmeasured
def test_a_marked_test_snaps_unmeasured():
    unmeasured()


@pytest.mark.snaps_unmeasured
def test_a_marked_test_disagrees():
    disagree()
""",
            "test_d.py": """
def test_a_reload():
    from conftest import point_equality_guard

    with pytest.raises(AssertionError, match="reloaded"), point_equality_guard():
        importlib.reload(canonical)


def test_after_a_reload_the_session_measures():
    assert canonical.MEASURE is True


def test_last():
    pass
""",
        },
    )
    assert code == 1, output
    _only(cases, "test_never_snaps", "passed")
    _only(
        cases,
        "test_a_module_fixture_set_up",
        "error",
        "setup",
        "without the point-equality measure",
    )
    _only(
        cases,
        "test_a_module_fixture_torn_down",
        "error",
        "teardown",
        "decided otherwise",
    )
    _only(cases, "test_a_marked_test_owns_its_setup", "passed")
    _only(
        cases,
        "test_an_unmarked_test_snaps_unmeasured",
        "failure",
        "without the point-equality measure",
    )
    _only(cases, "test_a_marked_test_snaps_unmeasured", "passed")
    _only(cases, "test_a_marked_test_disagrees", "failure", "decided otherwise")
    # The guard read the reload itself (no DID NOT RAISE), and the session
    # failed the test that reloaded, then measured again.
    _only(cases, "test_a_reload", "failure", "reloaded")
    assert "DID NOT RAISE" not in cases["test_a_reload"][0][1]
    _only(cases, "test_after_a_reload_the_session_measures", "passed")
    # The session fixture is torn down after the last test.
    _only(cases, "test_last", "error", "teardown", "decided otherwise")
    assert "MEASURE after the session: False" in output


def test_a_snap_at_collection_fails_the_next_test_marked_or_not(tmp_path):
    code, output, cases = _session(
        tmp_path,
        {
            "test_collect.py": """
unmeasured()


@pytest.mark.snaps_unmeasured
def test_first():
    pass


def test_second():
    pass
""",
        },
    )
    assert code == 1, output
    _only(cases, "test_first", "error", "outside any test", "setup")
    _only(cases, "test_second", "passed")


def test_a_snap_in_a_session_that_runs_no_test_fails_it(tmp_path):
    code, output, cases = _session(tmp_path, {"test_none.py": "disagree()\n"})
    assert code == 1, output
    assert "point equality, outside any test" in output
    assert "decided otherwise" in output
    assert cases == {}
