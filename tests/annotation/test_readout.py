"""A readout's number, counted by :class:`CountUp`, held to three statements
about time, kept apart, at 15, 30 and 60 frames per second:

1. ``state_at(t)`` answers the last frame instant of the record's grid at
   or before ``t``, reports it as ``as_of``, and its text is the format of
   the declared value there;
2. every frame a render delivers draws the format of the declared value at
   its own instant, the same at every rate that delivers that instant;
3. a render's last frame is its last frame instant before the scene's end:
   the exact target is in a delivered frame only when a frame instant at or
   after the count's end exists.

Statements 2 and 3 are read off the frames a render writes (a PNG sequence),
each matched against texts drawn here of every number the count shows. An
image sequence writes a held wait's picture once, where a video repeats it. The
declared value at ``t`` is computed here, from the count's own terms:
``start + (to - start) * rate_func(t)`` during its one-second play, and the
target once it is done.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from arachne import DOWN, LEFT, ORIGIN, RIGHT, UL, UP, UR, Readout, Text, tempconfig
from arachne.camera.camera import Camera
from arachne.session import Session
from arachne.timeline import admission
from arachne.timeline.clock import frames_in

from . import annotation_scenes
from .annotation_scenes import COUNT_FONT_SIZE, COUNTS

SCENES = Path(annotation_scenes.__file__)
RATES = (15, 30, 60)
#: The rendered picture: wide enough that one number's frame is told from
#: the next's.
PICTURE = {"pixel_width": 192, "pixel_height": 108}
#: How far, in mean levels per channel, a frame may lie from the text it
#: shows, drawn here: rounding, never another number.
SAME = 2.0


@pytest.fixture
def opened():
    sessions: list[Session] = []
    admission.DEMOTED.clear()

    def open_scene(name: str, fps: int) -> Session:
        session = Session.open(
            SCENES,
            name,
            config={"pixel_width": 96, "pixel_height": 54, "frame_rate": fps},
        )
        sessions.append(session)
        return session

    yield open_scene
    for session in sessions:
        session.close()
    admission.DEMOTED.clear()


def _declared(key: str, t: Fraction) -> float:
    start, to, _, rate = COUNTS[key]
    if t >= 1:
        return to
    return start + (to - start) * rate(float(t))


def _shown(key: str, t: Fraction) -> str:
    return COUNTS[key][2].format(_declared(key, t))


@pytest.mark.parametrize("key", list(COUNTS))
def test_as_of_answers_the_last_frame_of_the_record_s_grid(opened, key):
    asked = Fraction(1, 12)
    answers = {}
    for fps in RATES:
        session = opened(f"Count{key}", fps)
        state = session.state_at(asked)
        label = state.node(state.node(state.roots[0]).children[0])
        answers[fps] = (state.as_of, label.text)
    # 15 and 30 fps records answer from 1/15, a 60 from 1/12 itself: all
    # right.
    assert answers[15] == (Fraction(1, 15), _shown(key, Fraction(1, 15)))
    assert answers[30] == (Fraction(1, 15), _shown(key, Fraction(1, 15)))
    assert answers[60] == (Fraction(1, 12), _shown(key, Fraction(1, 12)))


def _rendered(name: str, fps: int, media: Path) -> list[np.ndarray]:
    """The frames a render of the scene ``name`` at ``fps`` delivers, in
    order.
    """
    scene = getattr(annotation_scenes, name)
    with tempconfig(
        {
            **PICTURE,
            "frame_rate": fps,
            "media_dir": str(media),
            "format": "png-sequence",
            "disable_caching": True,
        }
    ):
        scene().render()
    files = sorted(media.rglob("*.png"))
    return [np.asarray(Image.open(f).convert("RGB"), dtype=float) for f in files]


class _Texts:
    """Each number's text, drawn here as a render draws a frame: centred
    where the readout stands, at its size, on the scene's background.
    """

    def __init__(self) -> None:
        self._drawn: dict[str, np.ndarray] = {}

    def __call__(self, said: str) -> np.ndarray:
        found = self._drawn.get(said)
        if found is None:
            with tempconfig(PICTURE):
                camera = Camera()
                text = Text(said, font_size=COUNT_FONT_SIZE).move_to(ORIGIN)
                camera.capture_mobjects([text])
                found = np.asarray(camera.pixel_array[..., :3], dtype=float)
            self._drawn[said] = found
        return found


def _says(frame: np.ndarray, candidates: set[str], texts: _Texts) -> str:
    """The candidate text ``frame`` draws: the nearest, which must lie
    within :data:`SAME` of it.
    """
    distance = {said: float(np.abs(frame - texts(said)).mean()) for said in candidates}
    nearest = sorted(distance, key=lambda c: distance[c])
    said = nearest[0]
    assert distance[said] <= SAME, [(c, distance[c]) for c in nearest[:2]]
    return said


@pytest.mark.parametrize("key", list(COUNTS))
@pytest.mark.parametrize("waits", [False, True])
def test_every_delivered_frame_draws_the_declared_value_at_its_instant(
    tmp_path, key, waits
):
    name = f"Count{key}{'ThenWait' if waits else ''}"
    texts = _Texts()
    drawn: dict[int, dict[Fraction, np.ndarray]] = {}
    for fps in RATES:
        frames = _rendered(name, fps, tmp_path / f"{fps}")
        # The count's frames, then the wait's held picture, written once.
        times = frames_in(Fraction(0), Fraction(1), Fraction(fps))
        times += [Fraction(1)] if waits else []
        assert len(frames) == len(times), fps
        candidates = {_shown(key, Fraction(k, 60)) for k in range(91)}
        for t, frame in zip(times, frames, strict=True):
            assert _says(frame, candidates, texts) == _shown(key, t), (fps, t)
        drawn[fps] = dict(zip(times, frames, strict=True))
    # The same drawing wherever two rates deliver the same instant.
    for t, frame in drawn[15].items():
        assert np.abs(drawn[30][t] - frame).max() <= 1, t
        assert np.abs(drawn[60][t] - frame).max() <= 1, t


@pytest.mark.parametrize("key", list(COUNTS))
@pytest.mark.parametrize("fps", RATES)
def test_the_target_is_delivered_only_when_a_frame_follows_the_count(
    tmp_path, key, fps
):
    _, to, fmt, _ = COUNTS[key]
    texts = _Texts()
    candidates = {_shown(key, Fraction(k, 60)) for k in range(91)}
    last = 1 - Fraction(1, fps)

    ends = _rendered(f"Count{key}", fps, tmp_path / "ends")
    # The last frame is the last frame instant before the scene's end, and
    # shows the value there.
    assert len(ends) == fps
    said = _says(ends[-1], candidates, texts)
    assert said == _shown(key, last)
    if key != "WithUnit":
        # Short of the target: no delivered frame shows it.
        assert said != fmt.format(to)

    waits = _rendered(f"Count{key}ThenWait", fps, tmp_path / "waits")
    # The count's frames, then the wait's held picture, which an image
    # sequence writes once: it shows the target.
    assert len(waits) == fps + 1
    assert _says(waits[-1], candidates, texts) == fmt.format(to)


def test_the_count_s_value_is_exact_once_it_ends(opened):
    for key, (_, to, fmt, _) in COUNTS.items():
        session = opened(f"Count{key}", 30)
        reading = next(m for m in session.record.mobjects() if isinstance(m, Readout))
        assert reading.value == to
        state = session.state_at(1)
        assert state.as_of == 1
        label = state.node(state.node(state.roots[0]).children[0])
        assert label.text == fmt.format(to)


def test_a_readout_s_format_and_value_are_its_own():
    reading = Readout(2.345, format="{:.2f} s")
    assert reading.value == 2.345
    assert reading.label.original_text == "2.35 s"
    reading.set_value(10)
    assert reading.value == 10
    assert reading.label.original_text == "10.00 s"
    # A colour given since is kept by the next value.
    reading.set_color("#FF0000")
    reading.set_value(11)
    assert reading.label.submobjects[0].get_color().to_hex() == "#FF0000"
    with pytest.raises(ValueError, match="finite"):
        reading.set_value(float("nan"))
    with pytest.raises(ValueError, match="format"):
        Readout(1, format="{:.1f} and {:.1f}")


@pytest.mark.parametrize(
    "pattern",
    ["0 mV", "no field", "{:.1f} / {:.1f}", "{0} and {0}", "{value}", "{:{width}}"],
)
def test_a_format_holds_exactly_one_field_for_the_number(pattern):
    with pytest.raises(ValueError, match="one field"):
        Readout(1, format=pattern)


@pytest.mark.parametrize("pattern", ["{}", "{:.2f} s", "{0:,.0f} km", "~{:.0%}"])
def test_a_format_with_one_field_is_taken(pattern):
    assert Readout(0.5, format=pattern).label.original_text == pattern.format(0.5)


def test_a_count_s_value_between_its_ends_is_the_eased_value_itself():
    from arachne import CountUp

    almost = 1 - 1e-13
    reading = Readout(0, format="{:.1f}")
    count = CountUp(reading, 100, rate_func=lambda a: almost if 0 < a < 1 else a)
    count.begin()
    count.interpolate(0.5)
    # Within a hair of the target, but not the target: not snapped to it.
    assert reading.value == 100 * almost
    assert reading.value != 100


def test_a_count_ends_at_its_target_when_the_play_is_done():
    from arachne import CountUp
    from arachne.utils.rate_functions import there_and_back

    reading = Readout(20, format="{:.1f}")
    count = CountUp(reading, 100, rate_func=there_and_back)
    count.begin()
    count.interpolate(0.5)
    assert reading.value == pytest.approx(100)
    # The rate function comes back to the start at the play's last instant...
    count.interpolate(1)
    assert reading.value == 20
    # ...and the count, done, holds its target.
    count.finish()
    assert reading.value == 100
    assert reading.label.original_text == "100.0"


def test_a_readout_keeps_its_style_across_a_new_digit():
    from arachne import CountUp

    reading = Readout(9, format="{:.0f}", color="#FFCC00")
    reading.set_opacity(0.5).set_z_index(3)
    reading.set_stroke("#00FF00", width=1.5)
    count = CountUp(reading, 20, rate_func=lambda a: a)
    count.begin()
    count.interpolate(0.6)
    assert reading.label.original_text == "16"
    label = reading.label
    assert label.z_index == 3
    glyphs = label.family_members_with_points()
    assert len(glyphs) == 2
    for glyph in glyphs:
        assert glyph.get_fill_opacity() == pytest.approx(0.5)
        assert glyph.get_fill_color().to_hex() == "#FFCC00"
        assert glyph.get_stroke_color().to_hex() == "#00FF00"
        assert glyph.get_stroke_width() == pytest.approx(1.5)
        assert glyph.z_index == 3


def _held(record, mob, capture) -> dict:
    state = record.node_state(mob, capture)
    return dict(zip(state.keys, state.values, strict=True))


def test_a_play_draws_only_the_glyphs_the_readout_shows(opened):
    # A play draws its moving mobjects as they were listed when it began
    # (CE's moving list): the glyphs a new text replaced must draw nothing.
    session = opened("CountRounded", 15)
    record = session.record
    play = record.plays[-1]
    for t in frames_in(Fraction(0), Fraction(1), Fraction(15)):
        capture = play.capture_at(t)
        readout = capture.moving[0]
        assert isinstance(readout, Readout)
        label = record.node_state(readout, capture).submobjects[0]
        shown: set[int] = set()
        stack = [label]
        while stack:
            mob = stack.pop()
            shown.add(id(mob))
            stack.extend(record.node_state(mob, capture).submobjects)
        drawn = [
            mob
            for mob in capture.moving
            if id(mob) not in shown and len(_held(record, mob, capture)["_points"])
        ]
        assert drawn == [], t


def test_a_count_s_target_is_set_without_its_easing_s_last_value():
    from arachne import BarGauge, CountUp

    # An easing that leaves the scale only at its last value: the count is
    # done at its target, which the easing's last value never reaches.
    gauge = BarGauge(0.2)
    count = CountUp(gauge, 1.0, rate_func=lambda a: 1.5 if a >= 1 else a)
    count.begin()
    count.interpolate(0.5)
    count.finish()
    assert gauge.value == 1.0


#: A format with a leading word, as a sheet labels a number.
WORDED = "swell {:.0f}%"


def _drawn_alone(mobject) -> np.ndarray:
    """``mobject`` drawn by the camera alone, on the scene's background:
    what a frame shows of it.
    """
    with tempconfig(PICTURE):
        camera = Camera()
        camera.capture_mobjects([mobject])
        return np.asarray(camera.pixel_array[..., :3], dtype=float)


def _pixel(frame: np.ndarray, point: np.ndarray) -> np.ndarray:
    """The colour ``frame`` draws at the scene point ``point``."""
    from arachne import config

    height, width = frame.shape[:2]
    col = int((point[0] / config.frame_width + 0.5) * width)
    row = int((0.5 - point[1] / config.frame_height) * height)
    return frame[row, col]


def _over_first_glyph(reading: Readout) -> np.ndarray:
    """A point of the plate that no glyph covers: above the first glyph (a
    lower-case letter), under the plate's top edge.
    """
    label = reading.label
    plates = {id(m) for m in label.background_rectangle.get_family()}
    first = next(m for m in label.family_members_with_points() if id(m) not in plates)
    top = label.background_rectangle.get_top()[1]
    return np.array([first.get_center()[0], (first.get_top()[1] + top) / 2, 0])


def test_a_readout_s_plate_is_drawn_through_its_count():
    from arachne import CountUp
    from arachne.mobject.annotation.callout import PILL_COLOR
    from arachne.utils.color import ManimColor

    reading = Readout(0, format=WORDED, pill=True, font_size=COUNT_FONT_SIZE)
    plate_rgb = np.array(ManimColor(PILL_COLOR).to_int_rgb(), dtype=float)
    spot = _over_first_glyph(reading)
    assert np.abs(_pixel(_drawn_alone(reading), spot) - plate_rgb).max() <= 1
    count = CountUp(reading, 100, rate_func=lambda a: a)
    count.begin()
    for alpha in (0.05, 0.5, 1.0):
        count.interpolate(alpha)
        plate = reading.label.background_rectangle
        assert plate in reading.label.family_members_with_points()
        drawn = _pixel(_drawn_alone(reading), _over_first_glyph(reading))
        assert np.abs(drawn - plate_rgb).max() <= 1, (alpha, drawn)


def _prefix_xs(reading: Readout, glyphs: int) -> np.ndarray:
    """The left edges of the label's first ``glyphs`` glyphs."""
    label = reading.label
    plate = getattr(label, "background_rectangle", None)
    plates = set() if plate is None else {id(m) for m in plate.get_family()}
    shown = [m for m in label.family_members_with_points() if id(m) not in plates]
    return np.array([m.get_left()[0] for m in shown[:glyphs]])


@pytest.mark.parametrize("pill", [False, True])
def test_a_leading_word_stays_still_as_the_number_widens(pill):
    from arachne import CountUp

    reading = Readout(0, format=WORDED, pill=pill).shift(LEFT + UP)
    before = _prefix_xs(reading, 5)
    middle = reading.label.get_center()[1]
    count = CountUp(reading, 100, rate_func=lambda a: a)
    count.begin()
    for alpha in (0.05, 0.5, 1.0):
        count.interpolate(alpha)
        assert np.abs(_prefix_xs(reading, 5) - before).max() < 1e-9, alpha
        assert reading.label.get_center()[1] == pytest.approx(middle, abs=0.02)
    assert reading.label.original_text == "swell 100%"


def test_a_bare_number_stays_centred_where_it_was_put():
    from arachne import CountUp

    reading = Readout(0, format="{:.0f}").move_to(2 * RIGHT)
    count = CountUp(reading, 1000, rate_func=lambda a: a)
    count.begin()
    count.interpolate(1.0)
    assert reading.label.original_text == "1000"
    assert reading.label.get_center()[0] == pytest.approx(2.0, abs=1e-9)


def test_a_readout_held_by_its_right_edge_grows_leftward():
    from arachne import CountUp

    reading = Readout(0, format="{:.0f} km", hold=RIGHT)
    right = reading.label.get_right()[0]
    count = CountUp(reading, 1000, rate_func=lambda a: a)
    count.begin()
    count.interpolate(1.0)
    assert reading.label.get_right()[0] == pytest.approx(right, abs=1e-9)


@pytest.mark.parametrize(
    ("offset", "edge"),
    [
        (UR, LEFT),
        (RIGHT, LEFT),
        (UL, RIGHT),
        (LEFT, RIGHT),
        (UP, ORIGIN),
        (DOWN, ORIGIN),
    ],
)
def test_an_attached_readout_grows_away_from_its_anchor(offset, edge):
    from arachne import CountUp, Dot, anchor

    dot = Dot(LEFT)
    reading = Readout(0, format="{:.0f}", at=anchor(dot), offset=offset)
    gap = reading.label.get_critical_point(edge)[0] - dot.get_center()[0]
    count = CountUp(reading, 1000, rate_func=lambda a: a)
    count.begin()
    for alpha in (0.5, 1.0):
        dot.shift(0.25 * RIGHT)
        count.interpolate(alpha)
        reading.attach()  # its updater, which a count keeps running
        held = reading.label.get_critical_point(edge)[0] - dot.get_center()[0]
        assert held == pytest.approx(gap, abs=1e-9), alpha
    assert reading.label.original_text == "1000"


def test_a_count_on_a_plate_draws_the_plate_in_every_frame(tmp_path):
    from arachne.mobject.annotation.callout import PILL_COLOR
    from arachne.utils.color import ManimColor

    plate_rgb = np.array(ManimColor(PILL_COLOR).to_int_rgb(), dtype=float)
    placed = Readout(0, format=WORDED, pill=True, font_size=COUNT_FONT_SIZE)
    spot = _over_first_glyph(placed)
    frames = _rendered("CountOnAPlate", 15, tmp_path)
    assert len(frames) == 15
    for index, frame in enumerate(frames):
        assert np.abs(_pixel(frame, spot) - plate_rgb).max() <= 2, index


def test_a_count_with_a_leading_word_draws_the_word_still(tmp_path):
    frames = _rendered("CountAfterAWord", 15, tmp_path)
    assert len(frames) == 15
    placed = Readout(0, format=WORDED, font_size=COUNT_FONT_SIZE)
    # The columns the word "swell" covers, and a margin left of it.
    from arachne import config

    width = frames[0].shape[1]
    first = _prefix_xs(placed, 5)
    word_end = placed.label.family_members_with_points()[4].get_right()[0]
    left = int((first[0] / config.frame_width + 0.5) * width) - 2
    right = int((word_end / config.frame_width + 0.5) * width) + 1
    word = frames[0][:, max(left, 0) : right]
    assert word.max() > 0  # the word is drawn there
    for index, frame in enumerate(frames[1:], start=1):
        assert np.abs(frame[:, max(left, 0) : right] - word).max() <= 1, index
