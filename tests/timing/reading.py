"""What the timing tests share: the timing scenes, a rig's values and
points as a capture of the record holds them, and the drawing a declared rig
must show at an instant, computed here from the scene's own terms.
"""

from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path

import numpy as np

from arachne import Rig
from arachne.session import Session
from arachne.timeline.clock import frames_in

from . import timing_scenes
from .timing_scenes import RATE

SCENES = Path(timing_scenes.__file__)
#: The frame rates every statement is held at; 60 is also a session's
#: 60 Hz record.
RATES = (15, 30, 60)
SMALL = {"pixel_width": 96, "pixel_height": 54}


def boundary(t: Fraction, rate: int = RATE) -> Fraction:
    """The last boundary ``k / rate`` at or before ``t``."""
    return Fraction(math.floor(t * rate), rate)


def box_of(session: Session) -> Rig:
    return next(m for m in session.record.mobjects() if isinstance(m, Rig))


def frame_captures(session: Session, fps: int):
    """Each instant a frame at ``fps`` is delivered at, before the scene's
    end, with the capture of the record that answers it.
    """
    record = session.record
    for t in frames_in(Fraction(0), record.duration, Fraction(fps)):
        yield t, record.capture_at(t)[1]


def value_at(session: Session, rig: Rig, capture, parameter: str = "lid_open") -> float:
    state = session.record.node_state(rig, capture)
    return float(state.get("_values")[rig.model.index[parameter]])


def points_at(session: Session, mobjects, capture) -> list[np.ndarray]:
    record = session.record
    return [np.asarray(record.node_state(m, capture).get("_points")) for m in mobjects]


def art(rig: Rig) -> list:
    """Every shape of ``rig``'s parts, in order."""
    return [
        shape
        for _, group in rig._members
        for shape in group.family_members_with_points()
    ]


def scene_box_of(rig: Rig) -> tuple[float, ...]:
    """The box of ``rig``'s art, as a state's ``scene_box`` gives it."""
    points = np.concatenate([shape.points[:, :2] for shape in art(rig)])
    (x0, y0), (x1, y1) = points.min(axis=0), points.max(axis=0)
    return (x0, y0, x1, y1)


def close(got: list[np.ndarray], want: list[np.ndarray], tol: float = 1e-7) -> None:
    assert len(got) == len(want)
    for a, b in zip(got, want, strict=True):
        assert a.shape == b.shape
        assert np.abs(a - b).max() <= tol
