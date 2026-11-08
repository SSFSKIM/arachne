"""Numbers about one frame of a session, measured as the lint measures it.

:func:`query` reads the frame on screen at a time (the last frame of the
render's grid at or before it, or the scene's end past it) the way a lint
asked ``at`` that time reads it: the parts it draws (:mod:`.parts`), from the
record or, where the record can't answer, one forward run. Read from the
record, nothing is drawn. A record that refuses drawing (a camera it can't
draw) is read from a forward run that draws its frames live as the render
does, so a query of it pays that run, drawing included. Its answers are those parts measured by the lint's own instant rules
(:mod:`.rules`) under the terms the scene declared by then, so a query and a
lint of the same frame never disagree:

- :meth:`Query.bbox`: a mobject's box, its family's points where the picture
  shows them, as :meth:`~arachne.session.Session.state_at` gives it;
- :meth:`Query.overlaps`: the pairs that touch (``overlap.text``,
  ``gap.text``, ``contact.held``), and :meth:`Query.occlusions`, the texts
  something is drawn over (``occlusion.glyph``);
- :meth:`Query.offscreen`: the texts the picture's edge cuts or leaves out
  (``clip.frame_edge``);
- :meth:`Query.text_heights`: each text's height, as ``text.small`` reads it;
- :meth:`Query.markers`: the record's named markers in force there (its
  section, the subcaptions showing, the moment tags and sounds of its play),
  as the record keeps them.

Every number about the picture is in home-frame units: scene units of the
frame the configuration describes, centred, y up, the camera's projection applied
(:mod:`arachne.camera.projection`). The findings each answer carries are the
rules' own, with boxes as a report gives them (scene units, and pixels of
the session's resolution).

Those numbers are the engine camera's. A scene whose code replaced a method
of its camera, or an entry of its display functions
(:attr:`~arachne.timeline.record.Record.camera_replaced` for the methods,
:func:`~arachne.timeline.camera_code.replaced_displays` for the entries), may
draw its picture elsewhere, so each answer that measures the picture
(:meth:`Query.bbox`, :meth:`Query.overlaps`, :meth:`Query.occlusions`,
:meth:`Query.offscreen`, :meth:`Query.text_heights`) is a
:class:`CameraReplaced` in its place, unless every method replaced only
paints (:data:`DRAW_ONLY`). The time, the play, the ids a frame holds and the
markers are still answered.
"""

from __future__ import annotations

__all__ = [
    "DRAW_ONLY",
    "CameraReplaced",
    "Contact",
    "Occlusion",
    "Offscreen",
    "Query",
    "TextHeight",
    "Thing",
    "query",
]

from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property
from typing import TYPE_CHECKING

from arachne.camera.projection import Box, home_box, record_view
from arachne.timeline.camera_code import replaced_displays
from arachne.timeline.clock import exact
from arachne.timeline.draw import Drawing, camera_nodes

from .bridge import terms_at
from .core import parts_reader
from .findings import Finding
from .frames import FrameReader, Frames
from .instants import lint_instants
from .parts import Part, Parts
from .rules import Terms
from .rules._geometry import union
from .rules._pairs import pair_findings
from .rules._units import Measured, measured_units
from .rules.frame_edge import edge_findings
from .rules.text_small import text_heights

if TYPE_CHECKING:
    from arachne.camera.projection import View
    from arachne.session import Session
    from arachne.timeline.camera_code import Replaced
    from arachne.timeline.record import Capture, Marker, PlayRecord, Record

#: The codes of a pair that touches, as :meth:`Query.overlaps` gives them.
_TOUCHING = ("overlap.text", "gap.text", "contact.held")

#: The camera methods that only paint: each puts down the pixels of shapes
#: whose place on the picture the projection and the path setter decide
#: (neither is listed: the path setter builds each outline from the projected
#: points), so a replacement that wraps one, to count or to restyle, changes
#: how the picture looks and not where anything on it is. Closed: a replaced
#: method not listed here, a projection or the path setter among them, marks
#: the answers that measure the picture.
DRAW_ONLY = (
    # Paints one vectorized shape: its path, stroke and fill on the context.
    "display_vectorized",
    # Paint the points of point clouds at the pixels they fall on.
    "display_multiple_point_cloud_mobjects",
    "display_point_cloud",
    # Paint images over the corners they are placed at.
    "display_multiple_image_mobjects",
    "display_image_mobject",
    # Sets the Cairo context's colour for the paint that follows.
    "set_cairo_context_color",
)


@dataclass(frozen=True)
class Thing:
    """A thing a query names: its stable id, type, the scene's name for it,
    its creating line in the scene file, its text when it is one, and its
    box on the picture (its points', in home-frame units).
    """

    id: str
    type: str
    name: str | None
    line: int | None
    text: str | None
    box: Box


@dataclass(frozen=True)
class Contact:
    """Two things that touch: the ``code`` a sweep gives the pair, how far
    apart they are (``gap``; below zero, how deep they overlap) and the
    clearance they are held to when the finding states it.
    """

    code: str
    a: Thing
    b: Thing
    gap: float
    required: float | None
    finding: Finding


@dataclass(frozen=True)
class Occlusion:
    """A text drawn under something that lies over its glyphs by ``depth``."""

    text: Thing
    over: Thing
    depth: float
    finding: Finding


@dataclass(frozen=True)
class Offscreen:
    """A text the picture's edge cuts or leaves out: the edge its glyphs
    are furthest past, and by how much.
    """

    thing: Thing
    edge: str
    past: float
    finding: Finding


@dataclass(frozen=True)
class TextHeight:
    """A text's ink height, in home-frame units and in pixels of a picture
    1080 pixels on its shorter side.
    """

    thing: Thing
    units: float
    pixels: float


@dataclass(frozen=True)
class CameraReplaced:
    """Given in place of a measurement of the picture: the scene replaced
    camera methods, or entries of the camera's display functions
    (``display_funcs[Kind]``), that place what it draws (``replaced``, each
    as ``Owner.name (on the instance|class|display_funcs)``), so the picture
    need not be where the engine's camera would put it.
    """

    replaced: tuple[str, ...]
    code: str = "query.camera_replaced"

    @property
    def message(self) -> str:
        return (
            f"{self.code}: the scene replaced {', '.join(self.replaced)}, so the"
            " picture is not measured: it may not be where the engine's camera"
            " puts it"
        )


@dataclass(frozen=True, eq=False)
class _Read:
    parts: Parts
    boxes: dict[str, Box | None]


class Query:
    """The answers about one frame: the frame on screen at the time asked
    (``t``), the play holding it (``None`` at the scene's end), and what
    answered it (``"recorded"`` or ``"forward"``).
    """

    def __init__(
        self,
        parts: Parts,
        boxes: dict[str, Box | None],
        terms: Terms,
        markers: tuple[Marker, ...],
        replaced: CameraReplaced | None = None,
    ):
        self.parts = parts
        self._boxes = boxes
        self._terms = terms
        self._markers = markers
        #: What each measure of the picture is in its place, if anything.
        self.replaced = replaced

    @property
    def t(self) -> Fraction:
        return self.parts.t

    @property
    def play(self) -> int | None:
        return self.parts.play

    @property
    def answer(self) -> str:
        return self.parts.answer

    @property
    def frame(self) -> Box:
        """The picture's box, in home-frame units."""
        return self.parts.frame

    def bbox(self, node_id: str) -> Box | None | CameraReplaced:
        """The box of the mobject with stable id ``node_id`` and its family,
        their points where the picture shows them; ``None`` when they have
        no points. Raises :class:`KeyError` for an id the frame doesn't hold.
        """
        try:
            box = self._boxes[node_id]
        except KeyError:
            raise KeyError(f"no mobject {node_id!r} at t={self.t}") from None
        return self.replaced or box

    def overlaps(self) -> tuple[Contact, ...] | CameraReplaced:
        """The pairs that touch."""
        if self.replaced:
            return self.replaced
        return tuple(
            _contact(finding, self._things(finding))
            for finding in self._pairs
            if finding.code in _TOUCHING
        )

    def occlusions(self) -> tuple[Occlusion, ...] | CameraReplaced:
        """The texts something drawn after them lies over."""
        if self.replaced:
            return self.replaced
        found = []
        for finding in self._pairs:
            if finding.code != "occlusion.glyph":
                continue
            text, over = self._things(finding)
            found.append(Occlusion(text, over, finding.detail["covered"], finding))
        return tuple(found)

    def offscreen(self) -> tuple[Offscreen, ...] | CameraReplaced:
        """The texts the picture's edge cuts, or that lie wholly outside it."""
        if self.replaced:
            return self.replaced
        found = []
        for finding in edge_findings(self.parts, self._terms):
            (thing,) = self._things(finding)
            detail = finding.detail
            found.append(Offscreen(thing, detail["edge"], detail["past"], finding))
        return tuple(found)

    def text_heights(self) -> tuple[TextHeight, ...] | CameraReplaced:
        """Every text on the picture, with its height."""
        if self.replaced:
            return self.replaced
        return tuple(
            TextHeight(_part_thing(part), units, pixels)
            for part, units, pixels in text_heights(self.parts)
        )

    def markers(self) -> tuple[Marker, ...]:
        """The named markers in force at the frame, in the order they were
        placed: the section it is in, each subcaption whose span covers it,
        and the moment tags (``mark``) and sounds of the play holding it (at
        the scene's end, those placed there). They are the record's own
        (:class:`~arachne.timeline.record.Marker`): kind, time, data, line.
        """
        return self._markers

    @cached_property
    def _pairs(self) -> list[Finding]:
        return pair_findings(self.parts, self._terms)

    @cached_property
    def _units(self) -> dict[str, Measured]:
        return {unit.id: unit for unit in measured_units(self.parts)}

    def _things(self, finding: Finding) -> tuple[Thing, ...]:
        return tuple(_unit_thing(self._units[o.id]) for o in finding.objects)


def query(session: Session, t: Fraction | float) -> Query:
    """The answers about the frame on screen at ``t`` in ``session``."""
    record = session.record
    (instant,) = lint_instants(record, at=[exact(t)])
    frames = Frames(session)
    frames.request(query, [instant.t], _reader(session))
    frames.read()
    (read,) = frames.result(query)
    execution = frames.execution
    terms = terms_at(execution, instant.t, Terms())
    return Query(
        read.parts,
        read.boxes,
        terms,
        _in_force(execution, instant.t),
        _replaced(
            record.camera_replaced + replaced_displays(session.scene.renderer.camera)
        ),
    )


def _replaced(found: tuple[Replaced, ...]) -> CameraReplaced | None:
    """What the picture's measures are in their place, when a replaced
    camera method is not one that only paints.
    """
    placing = tuple(
        f"{each.qualname} (on the {each.form})"
        for each in found
        if each.name not in DRAW_ONLY
    )
    return CameraReplaced(placing) if placing else None


def _in_force(record: Record, t: Fraction) -> tuple[Marker, ...]:
    """The named markers of ``record`` in force at ``t``, in the order they
    were placed: the last section placed at or before ``t``, each subcaption
    whose span holds ``t``, and the moment tags and sounds placed in the span
    of the play holding ``t``, later in it included (after the last play,
    those placed after it). A play's span decides, not its own list, which a
    forward run's record does not fill.
    """
    play = record.play_at(t)
    if play is not None:
        start, end = play.start, play.end
    else:
        start = record.plays[-1].end if record.plays else Fraction(0)
        end = None
    section: Marker | None = None
    found: list[Marker] = []
    for marker in record.markers:
        if marker.kind == "section":
            if marker.t <= t and (section is None or marker.t >= section.t):
                section = marker
        elif marker.kind == "subcaption":
            if marker.t <= t < marker.t + exact(marker.data["duration"]):
                found.append(marker)
        elif marker.kind in ("mark", "sound") and (
            start <= marker.t and (end is None or marker.t < end)
        ):
            found.append(marker)
    chosen = {id(m) for m in found} | ({id(section)} if section else set())
    return tuple(m for m in record.markers if id(m) in chosen)


def _reader(session: Session) -> FrameReader[_Read]:
    parts_of = parts_reader(session)

    def read(
        record: Record,
        t: Fraction,
        play: PlayRecord | None,
        capture: Capture,
        live: bool,
    ) -> _Read:
        parts = parts_of(record, t, play, capture, live)
        with Drawing(record) as drawing:
            if live:
                # The forward run's scene holds the capture now, its camera
                # the map the frame is drawn with (the parts' view), through
                # the inputs the capture recorded.
                drawing.show_camera(capture)
                return _Read(parts, _boxes(record, capture, parts.view))
            drawing.show(capture, capture.draw_list())
            return _Read(parts, _boxes(record, capture, record_view(record)))

    return read


def _boxes(record: Record, capture: Capture, view: View) -> dict[str, Box | None]:
    """The box of every mobject the capture's scene holds, by stable id, as
    ``state_at`` boxes it: the camera's own mobjects left out.
    """
    camera = record._scene.renderer.camera
    hidden = {id(mob) for mob in camera_nodes(camera)}
    boxes: dict[str, Box | None] = {}
    for root in capture.draw_list():
        for mob in root.get_family():
            if id(mob) in hidden:
                continue
            key = record.origin(mob).id
            if key not in boxes:
                boxes[key] = home_box(camera, view, mob)
    return boxes


def _contact(finding: Finding, things: tuple[Thing, ...]) -> Contact:
    a, b = things
    detail = finding.detail
    gap = detail["gap"] if "gap" in detail else -detail["overlap"]
    return Contact(finding.code, a, b, gap, detail.get("required"), finding)


def _unit_thing(unit: Measured) -> Thing:
    return Thing(
        unit.id,
        unit.unit.type,
        unit.unit.name,
        unit.unit.scene_line,
        unit.said if unit.text else None,
        union([leaf.screen_box for leaf in unit.leaves]),
    )


def _part_thing(part: Part) -> Thing:
    return Thing(
        part.id, part.type, part.name, part.scene_line, part.text, part.screen_box
    )
