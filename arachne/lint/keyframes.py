"""Key frames and storyboard sheets: the picture each play or wait ends on.

A play's key frame is the last frame of the render's grid it holds, the
last before its end: for a frozen wait, its one frame. A play no frame of the
grid falls in (shorter than a frame, between two of them) shows nothing in
the render and gets no picture; nor does one whose frames the render leaves
out. Each is listed with the reason.

Frames are drawn as :meth:`~arachne.session.Session.frame` draws them,
through the session's painter, and what the record can't draw (a camera it
refuses, an instant it can't restore) comes from one forward run, as
:meth:`~arachne.session.Session.answers` gives it. A painter that refuses a
frame (it raises an error whose ``code`` is ``render.unsupported``) leaves
that play without a picture and gives a ``render.unsupported`` finding.
A session recorded on another grid than the render's gives that recording's
own frames at the render's frame times, which may differ from the render's
where a frame hook or dt updater ran at other instants.

:func:`key_frames` writes one PNG per play; :func:`storyboard` lays the same
frames out six to a sheet, each a :data:`TILE` of 960 by 540 pixels (large
enough that a defect a viewer would see is seen), labelled under it with its
time, play and line.
"""

from __future__ import annotations

__all__ = [
    "COLUMNS",
    "LABEL_HEIGHT",
    "ROWS",
    "TILE",
    "KeyFrame",
    "KeyFrames",
    "Sheet",
    "Storyboard",
    "Tile",
    "Unshown",
    "key_frames",
    "storyboard",
]

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from arachne.timeline.clock import frames_in
from arachne.timeline.restore import Unrestorable
from arachne.timeline.routes import RecordRefused

from .findings import Finding
from .instants import render_grid

if TYPE_CHECKING:
    from arachne.session import Session
    from arachne.timeline.record import PlayRecord, Record
    from arachne.timeline.source import SourceLine
    from arachne.typing import StrPath

#: A storyboard tile, ``(width, height)`` in pixels.
TILE = (960, 540)
#: Tiles across a sheet, and down it: six to a sheet.
COLUMNS = 2
ROWS = 3
#: The band under each tile its label is written in, in pixels.
LABEL_HEIGHT = 32

#: The error code a painter's refusal of a frame carries.
_UNSUPPORTED = "render.unsupported"
_BACKGROUND = (0, 0, 0)
_LABEL_BACKGROUND = (24, 24, 24)
_LABEL_INK = (235, 235, 235)
#: The space, in pixels, a label keeps from its tile's sides.
_LABEL_INSET = 8


@dataclass(frozen=True)
class KeyFrame:
    """A play's key frame: the play (its index and kind, ``"play"`` or
    ``"wait"``), the frame's time, the play's line, what answered the frame
    (``"recorded"`` or ``"forward"``), and the PNG it was written to
    (``None`` on a storyboard, which writes sheets only).
    """

    play: int
    kind: str
    t: Fraction
    line: SourceLine | None
    answer: str
    path: Path | None = None


@dataclass(frozen=True)
class Unshown:
    """A play that has no key frame, and why."""

    play: int
    kind: str
    line: SourceLine | None
    reason: str


@dataclass(frozen=True)
class KeyFrames:
    """The key frames written, in play order; the plays with none; and the
    ``render.unsupported`` findings of frames the painter refused.
    """

    frames: tuple[KeyFrame, ...]
    unshown: tuple[Unshown, ...]
    findings: tuple[Finding, ...]


@dataclass(frozen=True)
class Tile:
    """A key frame on a sheet: its box there, ``(left, top, right, bottom)``
    in pixels, and the label written under it.
    """

    key: KeyFrame
    box: tuple[int, int, int, int]
    label: str


@dataclass(frozen=True)
class Sheet:
    """One storyboard sheet: its PNG, and its tiles in order."""

    path: Path
    tiles: tuple[Tile, ...]


@dataclass(frozen=True)
class Storyboard:
    """The sheets written, the plays with no key frame, and the
    ``render.unsupported`` findings of frames the painter refused.
    """

    sheets: tuple[Sheet, ...]
    unshown: tuple[Unshown, ...]
    findings: tuple[Finding, ...]


def key_frames(
    session: Session, out_dir: StrPath, quality: str | None = None
) -> KeyFrames:
    """Write each play's key frame to ``out_dir`` as ``play-NNN.png``, at
    ``quality`` (a key of :data:`~arachne.constants.QUALITIES`; by default
    the size the scene was executed at).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    drawn, unshown, findings = _draw(session, quality)
    frames = []
    for key, image in drawn:
        path = out / f"play-{key.play:03d}.png"
        image.save(path)
        frames.append(_with_path(key, path))
    return KeyFrames(tuple(frames), unshown, findings)


def storyboard(
    session: Session, out_dir: StrPath, quality: str | None = None
) -> Storyboard:
    """Write the key frames to ``out_dir`` as sheets of six,
    ``storyboard-NN.png``: each frame (drawn at ``quality``, as
    :func:`key_frames` draws it) scaled whole into a :data:`TILE`, centred,
    and labelled under it with its time, play and line.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    drawn, unshown, findings = _draw(session, quality)
    width, height = TILE
    cell = height + LABEL_HEIGHT
    per_sheet = COLUMNS * ROWS
    font = ImageFont.load_default(size=LABEL_HEIGHT * 5 // 8)
    sheets = []
    for first in range(0, len(drawn), per_sheet):
        sheet = Image.new("RGB", (COLUMNS * width, ROWS * cell), _BACKGROUND)
        pen = ImageDraw.Draw(sheet)
        tiles = []
        for place, (key, image) in enumerate(drawn[first : first + per_sheet]):
            left = place % COLUMNS * width
            top = place // COLUMNS * cell
            sheet.paste(_fitted(image), (left, top))
            label = _label(key, font, width - 2 * _LABEL_INSET)
            pen.rectangle(
                (left, top + height, left + width - 1, top + cell - 1),
                fill=_LABEL_BACKGROUND,
            )
            pen.text(
                (left + _LABEL_INSET, top + height + LABEL_HEIGHT // 2),
                label,
                fill=_LABEL_INK,
                font=font,
                anchor="lm",
            )
            tiles.append(Tile(key, (left, top, left + width, top + height), label))
        path = out / f"storyboard-{first // per_sheet + 1:02d}.png"
        sheet.save(path)
        sheets.append(Sheet(path, tuple(tiles)))
    return Storyboard(tuple(sheets), unshown, findings)


def _draw(
    session: Session, quality: str | None
) -> tuple[
    list[tuple[KeyFrame, Image.Image]], tuple[Unshown, ...], tuple[Finding, ...]
]:
    """Each play's key frame drawn, in play order; the plays without one;
    and the painter's refusals as findings.
    """
    record = session.record
    grid = render_grid(record)
    keys: list[tuple[PlayRecord, Fraction]] = []
    unshown: dict[int, Unshown] = {}
    for play in record.plays:
        frames = frames_in(play.start, play.end, grid)
        if play.skipped:
            unshown[play.index] = _unshown(
                play, "the render leaves its frames out (a skipped section)"
            )
        elif not frames:
            unshown[play.index] = _unshown(
                play,
                f"no frame of the render's {grid} fps grid falls in its"
                f" {float(play.end - play.start):.3f} s",
            )
        else:
            keys.append((play, frames[-1]))
    images: dict[int, tuple[Image.Image, str]] = {}
    findings: list[Finding] = []
    forward: list[tuple[PlayRecord, Fraction]] = []
    for play, t in keys:
        try:
            images[play.index] = (session.frame(t, quality), "recorded")
        except (RecordRefused, Unrestorable):
            forward.append((play, t))
        except Exception as error:
            if getattr(error, "code", None) != _UNSUPPORTED:
                raise
            findings.append(_refused(play, t, error))
            unshown[play.index] = _unshown(play, str(error))
    if forward:
        with _refusals_apart(session) as refused:
            answers = session.answers([t for _, t in forward], quality)
        for (play, t), found in zip(forward, answers, strict=True):
            refusal = refused.get(play.index)
            if refusal is not None:
                findings.append(_refused(play, t, refusal))
                unshown[play.index] = _unshown(play, str(refusal))
                continue
            assert found.frame is not None
            images[play.index] = (found.frame, "forward")
    findings.sort(key=lambda f: f.t_start or 0)
    drawn = [
        (
            KeyFrame(
                play.index,
                play.kind,
                t,
                play.source_line,
                images[play.index][1],
            ),
            images[play.index][0],
        )
        for play, t in keys
        if play.index in images
    ]
    return (
        drawn,
        tuple(unshown[index] for index in sorted(unshown)),
        tuple(findings),
    )


@contextmanager
def _refusals_apart(session: Session) -> Iterator[dict[int, Exception]]:
    """Within the block, the session draws a batch of frames one frame at a
    time, so that the painter refusing one (an error whose ``code`` is
    ``render.unsupported``) is kept, by its play's index, and the others are
    still drawn: one forward run serves the batch, whatever the painter
    refuses. A refused frame's pixels are left blank.
    """
    refused: dict[int, Exception] = {}
    held = vars(session).get("_drawn")
    painter = session._drawn

    def drawn(
        record: Record, answers: Iterable[Any], static: Callable[..., Any]
    ) -> list[Any]:
        pixels = []
        for answer in answers:
            try:
                (one,) = painter(record, [answer], static)
            except Exception as error:
                if getattr(error, "code", None) != _UNSUPPORTED:
                    raise
                if answer.play is not None:
                    refused[answer.play.index] = error
                one = np.zeros_like(record._scene.renderer.camera.pixel_array)
            pixels.append(one)
        return pixels

    session._drawn = drawn
    try:
        yield refused
    finally:
        if held is None:
            del session._drawn
        else:
            session._drawn = held


def _unshown(play: PlayRecord, reason: str) -> Unshown:
    return Unshown(play.index, play.kind, play.source_line, reason)


def _refused(play: PlayRecord, t: Fraction, error: Exception) -> Finding:
    """The ``render.unsupported`` finding of a frame the painter refused."""
    subject = getattr(error, "subject", None) or getattr(error, "kind", None)
    return Finding(
        _UNSUPPORTED,
        t,
        t,
        message=f"the key frame of play {play.index} is not drawn: {error}",
        line=play.source_line,
        detail={"subject": subject},
    )


def _with_path(key: KeyFrame, path: Path) -> KeyFrame:
    return KeyFrame(key.play, key.kind, key.t, key.line, key.answer, path)


def _fitted(image: Image.Image) -> Image.Image:
    """``image`` scaled whole into a :data:`TILE`, centred on black."""
    width, height = TILE
    scale = min(width / image.width, height / image.height)
    size = (round(image.width * scale), round(image.height * scale))
    if size == TILE:
        return image.convert("RGB").resize(TILE, Image.Resampling.LANCZOS)
    tile = Image.new("RGB", TILE, _BACKGROUND)
    scaled = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    tile.paste(scaled, ((width - size[0]) // 2, (height - size[1]) // 2))
    return tile


def _label(
    key: KeyFrame, font: ImageFont.FreeTypeFont | ImageFont.ImageFont, room: int
) -> str:
    """The tile's label, its time, play and line, within ``room`` pixels in
    ``font``: a path too long for it loses its start, never its line.
    """
    said = f"t={float(key.t):.3f}s   play {key.play} ({key.kind})   "
    if key.line is None:
        return said + "line unknown"
    path, line = key.line.path, f":{key.line.line}"
    label = said + path + line
    cut = 0
    while font.getlength(label) > room and cut < len(path):
        cut += 1
        label = f"{said}\u2026{path[cut:]}{line}"
    return label
