"""What one lint does per instant, not per leaf: the work of reading a frame
that its leaves share is done once for the instant. Each test counts the
operation; none times one.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from arachne import Square
from arachne.camera import projection, three_d_camera
from arachne.camera.projection import _rotate, engine_trace, held_still
from arachne.camera.three_d_camera import ThreeDCamera
from arachne.lint import RULES, core, lint
from arachne.lint import parts as parts_module
from arachne.lint.departures import departures
from arachne.lint.parts import PartsCache, build_parts
from arachne.lint.watch import hold_watch
from arachne.timeline.routes import _resolve

from . import lint_scenes, read_scenes, watch_scenes

SCENES = Path(read_scenes.__file__)
WATCH_SCENES = Path(watch_scenes.__file__)
LINT_SCENES = Path(lint_scenes.__file__)


def _rotations_made(monkeypatch) -> list[float]:
    """The angles a ThreeDCamera's rotation matrix is made from, from now on:
    two for every matrix made.
    """
    made: list[float] = []
    about_z = three_d_camera.rotation_about_z

    def counted(angle):
        made.append(angle)
        return about_z(angle)

    monkeypatch.setattr(three_d_camera, "rotation_about_z", counted)
    return made


def test_a_held_3d_camera_s_rotation_is_made_once_for_the_lint(session_of, monkeypatch):
    """Every leaf, at every instant, is projected through the one rotation
    a camera that never turns holds: it is not made again per leaf.
    """
    session = session_of("HeldSquares3D", SCENES, frame_rate=15)
    made = _rotations_made(monkeypatch)
    result = lint(session)
    assert len(result.instants) > 1
    assert len(made) / 2 <= 1


def test_a_turning_3d_camera_s_rotation_is_made_once_per_lint_instant(
    session_of, monkeypatch
):
    """The camera turns between every two instants: each is projected
    through its own rotation, made once for all its leaves.
    """
    session = session_of("TurningSquares3D", SCENES, frame_rate=15)
    made = _rotations_made(monkeypatch)
    result = lint(session)
    assert len(made) / 2 == len(result.instants)


def test_a_projection_s_rotation_is_made_again_when_anything_it_reads_moved(
    monkeypatch,
):
    """Made again when the camera's angles moved or something else set its
    matrix since; never kept for a camera that makes its matrix its own way.
    """
    camera = ThreeDCamera()
    made = _rotations_made(monkeypatch)
    _rotate(camera)
    _rotate(camera)
    assert len(made) == 2
    camera.set_phi(1.0)
    _rotate(camera)
    assert len(made) == 4
    camera.rotation_matrix = np.identity(3)
    _rotate(camera)
    assert len(made) == 6
    assert np.array_equal(camera.rotation_matrix, camera.generate_rotation_matrix())

    class OwnRotation(ThreeDCamera):
        def generate_rotation_matrix(self):
            return super().generate_rotation_matrix()

    own = OwnRotation()
    made.clear()
    _rotate(own)
    _rotate(own)
    assert len(made) == 4


def test_a_camera_held_still_makes_its_rotation_once_and_not_before_it_is_asked(
    monkeypatch,
):
    """Held still, the engine's rotation is made at the first projection
    that makes it and not again; nothing is made on entry, so a read that
    projects nothing leaves the matrix as it was. A camera that makes its
    matrix its own way makes it at every projection still.
    """
    camera = ThreeDCamera()
    made = _rotations_made(monkeypatch)
    read: list[object] = []
    get_phi = ThreeDCamera.get_phi

    def counted_phi(self):
        read.append(self)
        return get_phi(self)

    monkeypatch.setattr(ThreeDCamera, "get_phi", counted_phi)
    camera.set_phi(0.4)
    with held_still(camera):
        assert made == []
        _rotate(camera)
        first = len(read)
        _rotate(camera)
        assert len(made) == 2
        # The second projection asks the camera nothing.
        assert len(read) == first
    camera.set_phi(0.9)
    _rotate(camera)
    assert len(made) == 4
    assert np.array_equal(camera.rotation_matrix, camera.generate_rotation_matrix())

    class OwnRotation(ThreeDCamera):
        def generate_rotation_matrix(self):
            return super().generate_rotation_matrix()

    own = OwnRotation()
    made.clear()
    with held_still(own):
        _rotate(own)
        _rotate(own)
    assert len(made) == 4


def test_a_camera_held_still_is_asked_once_whether_it_traces_with_the_engine(
    monkeypatch,
):
    camera = ThreeDCamera()
    asked: list[object] = []

    def ask(of):
        asked.append(of)
        return True

    monkeypatch.setattr(projection, "_engine_trace", ask)
    with held_still(camera):
        assert engine_trace(camera)
        assert engine_trace(camera)
    assert len(asked) == 1
    assert engine_trace(camera)
    assert len(asked) == 2


def _parts_built(monkeypatch) -> list[tuple]:
    """Every build of a frame's parts from now on, by everything it reads."""
    built: list[tuple] = []
    build = parts_module.build_parts

    def counted(record, t, play, capture, *, answer, target, live=False, memo=None):
        built.append((id(record), t, id(play), id(capture), answer, target, live))
        return build(
            record, t, play, capture, answer=answer, target=target, live=live, memo=memo
        )

    monkeypatch.setattr(parts_module, "build_parts", counted)
    monkeypatch.setattr(core, "build_parts", counted, raising=False)
    return built


@pytest.mark.parametrize(
    ("path", "scene"),
    [
        (WATCH_SCENES, "WordSlipsToEdge"),
        (WATCH_SCENES, "StatesArmedLate"),
        (LINT_SCENES, "Instants"),
    ],
)
def test_a_lint_builds_the_parts_of_each_state_it_reads_once(
    session_of, monkeypatch, path, scene
):
    """The lint's instants, an armed rule's own frames and the states it
    reads between them overlap: each state is built once for all of them.
    """
    session = session_of(scene, path, frame_rate=15)
    built = _parts_built(monkeypatch)
    lint(session, rules=(*RULES, hold_watch.where(), departures.where()))
    assert built
    assert len(built) == len(set(built))


def _boxes(parts) -> list:
    return [part.screen_box for part in parts.parts]


def test_another_instant_or_state_is_never_served_parts_built_for_one(session_of):
    """One capture held at two instants is built at each; a capture of
    another state shown at the same instant is built again; only the same
    read is handed what was built.
    """
    session = session_of("Instants", frame_rate=15)
    record = session.record
    cache = PartsCache()
    held = [_resolve(record, Fraction(n, 15)) for n in (9, 12)]
    assert held[0].capture is held[1].capture
    moving = _resolve(record, Fraction(3, 15))
    assert moving.capture is not held[0].capture
    target = (96, 54)

    def built(t, resolved, capture=None):
        return cache.build(
            record,
            t,
            resolved.play,
            capture or resolved.capture,
            answer="recorded",
            target=target,
        )

    first = built(Fraction(9, 15), held[0])
    later = built(Fraction(12, 15), held[1])
    assert later is not first
    assert (first.t, later.t) == (Fraction(9, 15), Fraction(12, 15))
    other = built(Fraction(9, 15), held[0], moving.capture)
    assert other is not first
    fresh = build_parts(
        record,
        Fraction(9, 15),
        held[0].play,
        moving.capture,
        answer="recorded",
        target=target,
    )
    assert _boxes(other) == _boxes(fresh) != _boxes(first)
    assert built(Fraction(9, 15), held[0]) is first


def _closed_curve_by_curve(points: np.ndarray, n: int = 4) -> bool:
    """Whether every subpath ends where it starts, a curve at a time."""
    curves = points[: len(points) // n * n].reshape(-1, n, points.shape[1])
    start = 0
    for index in range(len(curves)):
        last = index == len(curves) - 1
        if last or not np.allclose(curves[index, -1], curves[index + 1, 0], atol=1e-6):
            if not np.allclose(curves[start, 0], curves[index, -1], atol=1e-6):
                return False
            start = index + 1
    return True


def _ring(centre, radius: float, curves: int, *, gap: float = 0.0) -> np.ndarray:
    """A closed polygonal subpath of ``curves`` straight cubic curves, its
    last point ``gap`` short of its first.
    """
    angles = np.linspace(0, 2 * np.pi, curves + 1)
    corners = np.array(
        [[np.cos(a) * radius, np.sin(a) * radius, 0.0] for a in angles]
    ) + np.asarray(centre)
    corners[-1, 0] += gap
    return np.concatenate(
        [np.linspace(corners[i], corners[i + 1], 4) for i in range(curves)]
    )


SUBPATHS = {
    "one ring": [_ring((0, 0, 0), 1, 200)],
    "two rings": [_ring((0, 0, 0), 1, 100), _ring((3, 0, 0), 1, 100)],
    "a ring then an open one": [
        _ring((0, 0, 0), 1, 100),
        _ring((3, 0, 0), 1, 100, gap=0.1),
    ],
    "a gap within tolerance": [_ring((0, 0, 0), 1, 200, gap=5e-7)],
    "a gap past tolerance": [_ring((0, 0, 0), 1, 200, gap=5e-5)],
    "far from the origin": [_ring((1e3, 0, 0), 1, 200, gap=5e-3)],
    "an undefined point": [_ring((0, 0, 0), 1, 200, gap=np.nan)],
}


@pytest.mark.parametrize("name", SUBPATHS)
def test_a_path_s_closure_is_read_in_one_pass_over_its_curves(monkeypatch, name):
    """As closed or open as read a curve at a time, without a comparison per
    curve.
    """
    points = np.concatenate(SUBPATHS[name])
    expected = _closed_curve_by_curve(points)
    compared = []
    for function in ("allclose", "isclose"):
        real = getattr(np, function)

        def counted(*args, _real=real, **kwargs):
            compared.append(1)
            return _real(*args, **kwargs)

        monkeypatch.setattr(np, function, counted)
    assert parts_module._closed(Square(), points) is expected
    assert len(compared) <= 2
