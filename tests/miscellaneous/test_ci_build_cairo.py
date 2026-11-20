"""The CI's Cairo download: retried through a reset connection or a timeout,
taken from the next source when one fails, checked against the pinned checksum. ``urlopen`` and the wait are stubbed; nothing
reaches the network.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import urllib.error
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2] / ".github" / "scripts" / "ci_build_cairo.py"
)


@pytest.fixture
def cairo():
    spec = importlib.util.spec_from_file_location("ci_build_cairo", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Response(io.BytesIO):
    """A response whose body is ``data``, read in blocks."""

    def __init__(self, data: bytes, fail_after: int | None = None):
        super().__init__(data)
        self.fail_after = fail_after

    def read(self, size=-1):
        if self.fail_after is not None and self.tell() >= self.fail_after:
            raise ConnectionResetError(54, "Connection reset by peer")
        return super().read(min(size, 4))


def _opener(outcomes):
    """A stand-in ``urlopen``: each call takes the next outcome, raising it
    if it is an error, else answering with its bytes.
    """
    calls = []

    def urlopen(url, timeout=None):
        assert timeout is not None, "every download has a timeout"
        calls.append(url)
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    return urlopen, calls


def _reset():
    return urllib.error.URLError(ConnectionResetError(54, "Connection reset by peer"))


def test_a_reset_connection_is_retried_with_a_growing_wait(
    cairo, monkeypatch, tmp_path
):
    urlopen, calls = _opener([_reset(), _reset(), _Response(b"the tarball")])
    monkeypatch.setattr(cairo.urllib.request, "urlopen", urlopen)
    waits = []
    target = tmp_path / "cairo.tar.xz"
    cairo.download_file(
        "https://example.invalid/cairo.tar.xz",
        target,
        attempts=3,
        sleep=waits.append,
    )
    assert target.read_bytes() == b"the tarball"
    assert len(calls) == 3
    assert len(waits) == 2
    assert 0 < waits[0] < waits[1]


def test_a_reset_mid_body_starts_the_file_again(cairo, monkeypatch, tmp_path):
    urlopen, calls = _opener(
        [_Response(b"the tarball", fail_after=4), _Response(b"the tarball")]
    )
    monkeypatch.setattr(cairo.urllib.request, "urlopen", urlopen)
    target = tmp_path / "cairo.tar.xz"
    cairo.download_file(
        "https://example.invalid/cairo.tar.xz", target, sleep=lambda s: None
    )
    assert target.read_bytes() == b"the tarball"
    assert len(calls) == 2


def test_the_last_reset_is_raised_once_the_attempts_are_spent(
    cairo, monkeypatch, tmp_path
):
    urlopen, calls = _opener([_reset() for _ in range(10)])
    monkeypatch.setattr(cairo.urllib.request, "urlopen", urlopen)
    with pytest.raises(urllib.error.URLError):
        cairo.download_file(
            "https://example.invalid/cairo.tar.xz",
            tmp_path / "cairo.tar.xz",
            attempts=3,
            sleep=lambda s: None,
        )
    assert len(calls) == 3


@pytest.mark.parametrize(
    "error",
    [
        urllib.error.HTTPError("https://example.invalid", 404, "Not Found", {}, None),
        ValueError("unknown url type"),
    ],
    ids=["not found", "another error"],
)
def test_another_error_fails_at_once(cairo, monkeypatch, tmp_path, error):
    urlopen, calls = _opener([error, _Response(b"the tarball")])
    monkeypatch.setattr(cairo.urllib.request, "urlopen", urlopen)
    with pytest.raises(type(error)):
        cairo.download_file(
            "https://example.invalid/cairo.tar.xz",
            tmp_path / "cairo.tar.xz",
            sleep=lambda s: None,
        )
    assert len(calls) == 1


def test_a_server_error_is_retried(cairo, monkeypatch, tmp_path):
    busy = urllib.error.HTTPError("https://example.invalid", 503, "Busy", {}, None)
    urlopen, calls = _opener([busy, _Response(b"the tarball")])
    monkeypatch.setattr(cairo.urllib.request, "urlopen", urlopen)
    target = tmp_path / "cairo.tar.xz"
    cairo.download_file(
        "https://example.invalid/cairo.tar.xz", target, sleep=lambda s: None
    )
    assert target.read_bytes() == b"the tarball"
    assert len(calls) == 2


def test_the_tarball_is_checked_against_the_pinned_checksum(cairo, tmp_path):
    assert len(cairo.CAIRO_SHA256) == 64
    tarball = tmp_path / "cairo.tar.xz"
    tarball.write_bytes(b"the tarball")
    with pytest.raises(Exception, match="SHA256"):
        cairo.verify_sha256sum(tarball, cairo.CAIRO_SHA256)
    cairo.verify_sha256sum(tarball, hashlib.sha256(b"the tarball").hexdigest())


def test_a_timeout_is_a_network_failure(cairo, monkeypatch, tmp_path):
    urlopen, calls = _opener(
        [
            urllib.error.URLError(TimeoutError("timed out")),
            TimeoutError("The read operation timed out"),
            _Response(b"the tarball"),
        ]
    )
    monkeypatch.setattr(cairo.urllib.request, "urlopen", urlopen)
    target = tmp_path / "cairo.tar.xz"
    cairo.download_file(
        "https://example.invalid/cairo.tar.xz",
        target,
        attempts=3,
        sleep=lambda s: None,
    )
    assert target.read_bytes() == b"the tarball"
    assert len(calls) == 3


SOURCES = ["https://first.invalid/cairo.tar.xz", "https://second.invalid/cairo.tar.xz"]
CHECKSUM = hashlib.sha256(b"the tarball").hexdigest()


def _silent():
    return urllib.error.URLError(TimeoutError("timed out"))


def test_a_source_that_never_answers_gives_way_to_the_next(
    cairo, monkeypatch, tmp_path
):
    urlopen, calls = _opener([_silent(), _silent(), _Response(b"the tarball")])
    monkeypatch.setattr(cairo.urllib.request, "urlopen", urlopen)
    target = tmp_path / "cairo.tar.xz"
    source = cairo.download_cairo(
        target, SOURCES, CHECKSUM, attempts=2, sleep=lambda s: None
    )
    assert source == SOURCES[1]
    assert target.read_bytes() == b"the tarball"
    assert calls == [SOURCES[0], SOURCES[0], SOURCES[1]]


def test_a_source_with_the_wrong_checksum_gives_way_to_the_next(
    cairo, monkeypatch, tmp_path
):
    urlopen, calls = _opener([_Response(b"other bytes"), _Response(b"the tarball")])
    monkeypatch.setattr(cairo.urllib.request, "urlopen", urlopen)
    target = tmp_path / "cairo.tar.xz"
    source = cairo.download_cairo(
        target, SOURCES, CHECKSUM, attempts=2, sleep=lambda s: None
    )
    assert source == SOURCES[1]
    assert target.read_bytes() == b"the tarball"
    assert calls == SOURCES


def test_a_missing_file_gives_way_to_the_next_source_at_once(
    cairo, monkeypatch, tmp_path
):
    missing = urllib.error.HTTPError(SOURCES[0], 404, "Not Found", {}, None)
    urlopen, calls = _opener([missing, _Response(b"the tarball")])
    monkeypatch.setattr(cairo.urllib.request, "urlopen", urlopen)
    source = cairo.download_cairo(
        tmp_path / "cairo.tar.xz", SOURCES, CHECKSUM, sleep=lambda s: None
    )
    assert source == SOURCES[1]
    assert calls == SOURCES


def test_every_source_failing_names_each_one(cairo, monkeypatch, tmp_path):
    missing = urllib.error.HTTPError(SOURCES[1], 404, "Not Found", {}, None)
    urlopen, calls = _opener([_silent(), _silent(), missing])
    monkeypatch.setattr(cairo.urllib.request, "urlopen", urlopen)
    with pytest.raises(Exception) as raised:
        cairo.download_cairo(
            tmp_path / "cairo.tar.xz",
            SOURCES,
            CHECKSUM,
            attempts=2,
            sleep=lambda s: None,
        )
    message = str(raised.value)
    assert SOURCES[0] in message
    assert "timed out" in message
    assert SOURCES[1] in message
    assert "404" in message
    assert len(calls) == 3


def test_the_sources_and_their_bounds(cairo):
    assert cairo.CAIRO_SOURCES[0].startswith("https://cairographics.org/")
    assert len(cairo.CAIRO_SOURCES) >= 2
    worst = len(cairo.CAIRO_SOURCES) * (
        cairo.DOWNLOAD_ATTEMPTS * cairo.DOWNLOAD_TIMEOUT
        + cairo.DOWNLOAD_WAIT * (2 ** (cairo.DOWNLOAD_ATTEMPTS - 1) - 1)
    )
    assert worst < 300
