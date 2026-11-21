"""Diagram blocks: pieces joined at their ports, tokens flowing along the
joins, and a camera move into one piece.

- :class:`Diagram`: a group of pieces (blocks, nodes, gates, any mobject)
  joined by connectors from port to port, each connector optionally
  labelled with a text on a pill. It lays nothing out beyond the grid its
  caller names (:meth:`Diagram.place`).
- :class:`Flow`: tokens along a connector at a declared speed, in scene
  units per second, by arc length, kept on the tick.
- :class:`ZoomInto`: the camera's frame moves onto a block while the
  block's inside diagram fades in, once its labels are large enough to
  read, and what the frame's edge would cut fades out before it is cut.
- :class:`ZoomOut`: a played zoom's move backwards, its fades reversed,
  back to where it began.

A flow declares its speed as ``diagram.flow_speed``, in scene units a
second (:mod:`arachne.quantities`), read from what a capture holds: the
declared advance of its tokens along the edge, by arc length, while it runs,
and 0 once it is stopped. It is not each token's visible velocity, which
also carries the edge's own motion when the edge moves or deforms. Its
tolerance, 1 % of it, is a claim about that declared advance.

A diagram's labels are measured against each other, against the frame's
edge and for their height by the lint rule
:data:`~arachne.lint.rules.diagram_clear`, which the sweep's pair rules,
exempting two pieces of one top-level mobject, leave unasked.
"""

from __future__ import annotations

__all__ = ["Diagram", "Flow", "ZoomInto", "ZoomOut"]

import math
import weakref
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from arachne import config, quantities
from arachne.animation.animation import Animation
from arachne.constants import DL, DOWN, LEFT, ORIGIN, RIGHT, UP, UR
from arachne.mobject.geometry.arc import Dot
from arachne.mobject.marks._paths import ArcLength, as_point, direction, head, line
from arachne.mobject.marks._running import running_scene
from arachne.mobject.marks._style import (
    Opacities,
    apply_style,
    opacities_of,
    scaled,
    set_opacities,
    take_style,
)
from arachne.mobject.marks.arrows import ElbowConnector
from arachne.mobject.marks.blocks import SIDES, Block
from arachne.mobject.marks.ports import Ported
from arachne.mobject.text.text_mobject import Text
from arachne.mobject.types.vectorized_mobject import VGroup, VMobject
from arachne.quantities import At, Quantity
from arachne.utils.color import WHITE, YELLOW, ParsableManimColor
from arachne.utils.rate_functions import smooth

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Self

    from arachne.mobject.mobject import Mobject
    from arachne.scene.scene import Scene


class Diagram(VGroup):
    """Pieces joined at ports.

    ``cell`` is the grid :meth:`place` puts pieces on, ``(width, height)``
    of a cell, with column 0, row 0 at ``origin``, columns to the right and
    rows downward. :meth:`connect` joins two pieces with a connector and
    returns it; ``edges`` holds every connector in the order made.
    """

    def __init__(
        self,
        *pieces: VMobject,
        cell: tuple[float, float] = (2.5, 1.6),
        origin: Sequence[float] = ORIGIN,
        **kwargs: Any,
    ) -> None:
        super().__init__(*pieces, **kwargs)
        self.cell = (float(cell[0]), float(cell[1]))
        self.origin = tuple(float(c) for c in as_point(origin))
        self.edges: tuple[VMobject, ...] = ()
        #: Every piece a connector of this diagram has joined, in the order
        #: first joined: a piece's key here is its place in it, from 1. It
        #: only grows, so no key is given twice, and it is this diagram's
        #: alone: a piece joined in two diagrams has a key in each. A
        #: connector holds the keys of its two pieces (``start_key``,
        #: ``end_key``) and, for the content key, their places in
        #: :attr:`submobjects` now (``start_piece``, ``end_piece``, -1 for
        #: one the diagram no longer holds), kept by every change the
        #: diagram's own methods make to its children (add, insert,
        #: add_to_back, remove, item assignment, sort, shuffle, invert).
        #: A change made to ``submobjects`` directly is not kept.
        self.joined: tuple[Mobject, ...] = ()

    def grid_point(self, col: float, row: float) -> np.ndarray:
        """The centre of the grid cell at ``col``, ``row``."""
        width, height = self.cell
        return np.array(self.origin) + col * width * RIGHT + row * height * DOWN

    def place(self, piece: VMobject, col: float, row: float) -> VMobject:
        """Centre ``piece`` on the grid cell at ``col``, ``row`` and add it
        to the diagram (if it is not in it already).
        """
        piece.move_to(self.grid_point(col, row))
        if piece not in self.submobjects:
            self.add(piece)
        return piece

    def connect(
        self,
        start: Any,
        end: Any,
        *,
        route: str = "elbow",
        label: str | None = None,
        tip: bool = True,
        font_size: float = 24,
        corner_radius: float = 0.15,
        stub: float = 0.3,
        end_stub: float | None = None,
        color: ParsableManimColor = WHITE,
        label_color: ParsableManimColor | None = None,
        label_side: str = "above",
        **kwargs: Any,
    ) -> VMobject:
        """Join ``start`` to ``end`` (each a piece and one of its port
        names, or a piece alone, which is joined on the side facing the
        other end) with a connector, add it, and return it.

        ``route`` is ``"elbow"`` (right angles, rounded corners) or
        ``"straight"``. ``label`` puts a text on a pill (a plate the
        background's colour) at the connector's middle, which the
        connector keeps as ``label``: a connector's label is the one this
        makes; attaching one by hand (assigning ``label``) is unsupported.

        A :class:`Flow` put on the connector moves the label beside the
        line, to ``label_side``: a side of the line as drawn on screen,
        ``"above"`` (the default), ``"below"``, ``"left"`` or ``"right"``.
        The label goes to the side of the line, at its middle, that faces
        that way; where the line runs that way itself (an upright line
        asked ``"above"``), to the side a quarter-turn clockwise from it
        (``"above"`` is then right, ``"right"`` below, ``"below"`` left,
        ``"left"`` above). Which end the connector starts from does not
        change it. The lint :data:`~arachne.lint.rules.diagram_clear`
        reports a label set beside its line that lands on another piece,
        connector or label of the diagram; ``label_side`` is the fix.

        An elbow runs ``stub`` straight out of its start before it turns,
        and ``end_stub`` (``stub`` when ``None``) into its end, each cut to
        half the clear space when the piece at the other end lies across it.
        """
        if route not in ("elbow", "straight"):
            raise ValueError(
                f"a connector's route is 'elbow' or 'straight', not {route!r}"
            )
        if label_side not in LABEL_SIDES:
            raise ValueError(
                f"a connector's label_side is one of {', '.join(map(repr, LABEL_SIDES))},"
                f" not {label_side!r}"
            )
        a, leaving = _end(start, end)
        b, arriving_from = _end(end, start)
        edge: ElbowConnector | _StraightConnector
        if route == "elbow":
            edge = ElbowConnector(
                a,
                b,
                start_direction=leaving,
                end_direction=arriving_from,
                detour=_detour(_piece(start), _piece(end), a, b, leaving),
                stub=_stub(a, leaving, _piece(end), stub),
                end_stub=_stub(
                    b,
                    arriving_from,
                    _piece(start),
                    stub if end_stub is None else end_stub,
                ),
                corner_radius=corner_radius,
                tip=tip,
                color=color,
                **kwargs,
            )
        else:
            edge = _StraightConnector(a, b, tip=tip, color=color, **kwargs)
        self.add(edge)
        self.edges = (*self.edges, edge)
        # What it joins, by its pieces' keys in this diagram (``joined``).
        held: Any = edge
        held.start_key = self._key_of(_piece(start))
        held.end_key = self._key_of(_piece(end))
        self._place_ends()
        if label is not None:
            text = Text(
                label,
                font_size=font_size,
                color=color if label_color is None else label_color,
            )
            text.add_background_rectangle(opacity=1.0, buff=0.06, corner_radius=0.08)
            path = ArcLength(edge.points)
            text.move_to(path.point(path.length / 2))
            edge.label = text
            # The side of the line its label stands on beside a flow, and
            # how far a flow's tokens reach from the line, in scene units
            # (0 without one).
            held.label_side = label_side
            held.flow_reach = 0.0
            self.add(text)
        return edge

    def _place_ends(self) -> None:
        """Give each connector the places in :attr:`submobjects` of the
        pieces it joins, as it holds them now: ``start_piece`` and
        ``end_piece``, -1 for a piece the diagram no longer holds.
        """
        for edge in getattr(self, "edges", ()):
            held: Any = edge
            for key, place in (("start_key", "start_piece"), ("end_key", "end_piece")):
                piece = self._joined_piece(getattr(edge, key, None))
                at = -1
                if piece is not None:
                    at = next(i for i, m in enumerate(self.submobjects) if m is piece)
                if getattr(edge, place, None) != at:
                    setattr(held, place, at)

    # Every change Diagram makes to its children re-places the ends.

    def _insert_submobjects(self, index: int, mobjects: Sequence[Mobject]) -> Self:
        super()._insert_submobjects(index, mobjects)
        self._place_ends()
        return self

    def remove(self, *mobjects: Mobject) -> Self:
        super().remove(*mobjects)
        self._place_ends()
        return self

    def __setitem__(self, key: int, value: VMobject | Sequence[VMobject]) -> None:
        super().__setitem__(key, value)
        self._place_ends()

    def sort(self, *args: Any, **kwargs: Any) -> Self:
        super().sort(*args, **kwargs)
        self._place_ends()
        return self

    def shuffle(self, recursive: bool = False) -> Self:
        super().shuffle(recursive)
        self._place_ends()
        return self

    def invert(self, recursive: bool = False) -> Self:
        super().invert(recursive)
        self._place_ends()
        return self

    def _key_of(self, piece: Mobject) -> int:
        """``piece``'s key in this diagram (:attr:`joined`), given it the
        first time a connector joins it.
        """
        for place, held in enumerate(self.joined, start=1):
            if held is piece:
                return place
        self.joined = (*self.joined, piece)
        return len(self.joined)

    def _joined_piece(self, key: Any) -> Mobject | None:
        """The piece ``key`` names in this diagram (:attr:`joined`), if it is
        one of its pieces still; else ``None``.
        """
        if not isinstance(key, int) or not 1 <= key <= len(self.joined):
            return None
        piece = self.joined[key - 1]
        return piece if any(piece is m for m in self.submobjects) else None


class _StraightConnector(VMobject):
    """A straight connector from ``start`` to ``end``, its own points the
    wire, with a filled head at ``end`` when ``tip``.
    """

    def __init__(
        self,
        start: np.ndarray,
        end: np.ndarray,
        tip: bool = True,
        tip_length: float = 0.2,
        color: ParsableManimColor = WHITE,
        **kwargs: Any,
    ) -> None:
        super().__init__(color=color, **kwargs)
        #: The text a diagram set on it, if any.
        self.label: VMobject | None = None
        u = direction(start, end)
        stop = end
        if tip:
            reach = min(tip_length, float(np.linalg.norm(end - start)) / 2)
            stop = end - u * reach
            cap = VMobject(color=color, fill_opacity=1.0, stroke_width=0)
            cap.set_points(head(end, u, reach, reach * 0.9))
            self.add(cap)
        self.set_points(line(start, stop))


def _piece(spec: Any) -> VMobject:
    piece: VMobject = spec[0] if isinstance(spec, tuple) else spec
    return piece


#: How far past a piece a connector going round it runs.
_ROUND = 0.3


def _stub(port: np.ndarray, out: np.ndarray, other: VMobject, wanted: float) -> float:
    """How far a connector can run straight out of ``port`` along ``out``
    before it turns: ``wanted``, or less, half the clear space up to
    ``other`` (the piece at the connector's other end), when that piece lies
    across its course. A port ``other`` covers has no route out: refused.
    """
    axis = 0 if abs(out[0]) >= abs(out[1]) else 1
    across = 1 - axis
    sign = math.copysign(1.0, out[axis])
    low, high = other.get_critical_point(DL), other.get_critical_point(UR)
    if not low[across] - 1e-9 <= port[across] <= high[across] + 1e-9:
        return wanted
    near = low[axis] if sign > 0 else high[axis]
    far = high[axis] if sign > 0 else low[axis]
    if (far - port[axis]) * sign <= 0:
        return wanted  # the other piece is behind the port
    clear = (near - port[axis]) * sign
    if clear <= 1e-9:
        raise ValueError(
            f"a connector cannot leave the port at {np.round(port[:2], 3).tolist()}:"
            f" the {type(other).__name__} at its other end covers it; move the"
            " two pieces apart"
        )
    return float(min(wanted, clear / 2))


def _detour(
    first: VMobject, second: VMobject, a: np.ndarray, b: np.ndarray, leaving: np.ndarray
) -> float:
    """How far past the further of the ports ``a`` and ``b`` a connector
    going round two side-by-side pieces runs: clear of both pieces, above
    them for ports on left and right sides, right of them for ports on top
    and bottom.
    """
    other = 1 if abs(leaving[0]) >= abs(leaving[1]) else 0
    toward = UP if other == 1 else RIGHT
    beyond = max(p.get_critical_point(toward)[other] for p in (first, second))
    return float(beyond - max(a[other], b[other]) + _ROUND)


def _end(spec: Any, other: Any) -> tuple[np.ndarray, np.ndarray]:
    """Where a connector meets the piece ``spec`` names, and the direction
    it leaves in there.
    """
    if isinstance(spec, tuple):
        piece, name = spec
        if not isinstance(piece, Ported):
            raise ValueError(
                f"a {type(piece).__name__} has no ports: join it by itself, not by"
                f" the port {name!r}"
            )
        return piece.port(name), piece.port_direction(name)
    piece = spec
    toward = other[0] if isinstance(other, tuple) else other
    course = toward.get_center() - piece.get_center()
    if abs(course[0]) >= abs(course[1]):
        side, name = (RIGHT, "right") if course[0] > 0 else (LEFT, "left")
    else:
        side, name = (UP, "top") if course[1] > 0 else (DOWN, "bottom")
    if isinstance(piece, Ported) and name in piece.port_names:
        return piece.port(name), piece.port_direction(name)
    return piece.get_critical_point(side), np.array(SIDES[name], dtype=float)


class Flow(VGroup):
    """``tokens`` dots running along ``edge`` at ``speed`` scene units per
    second, by arc length, the first ``offset`` along it at the start, the
    others behind it at equal shares of the edge's length; each wraps from
    the edge's end to its start.

    At every instant a token is at ``(offset + speed * elapsed + i * L / n)
    mod L`` along the edge, ``L`` the edge's length then and ``elapsed`` the
    scene time since the flow started, read from the scene's own clock
    (:attr:`~.Scene.time`), never summed from steps. It starts when it is
    made in a running scene, or else the first time it runs in one: when its
    updater first runs there, or a play first takes it over and suspends
    that updater (a flow made before the scene, a copy). From then on two engine updaters keep
    it there: its own, which a play's end runs, and the scene's
    :class:`FlowDriver`, which the scene runs after every animation and
    mobject updater at every instant and no animation suspends. So the
    speed and the spacing hold at any frame rate, through plays and waits,
    whatever animates the edge, the diagram the flow is in or the flow
    itself, until :meth:`stop`. It reads the edge's own points (a
    connector's wire, not its head) as they are. Style keywords go to the
    tokens.

    A connector's label (the one :meth:`Diagram.connect` makes) moves
    beside the line as the flow is made, so the tokens never pass under
    it: at the line's middle, to the connector's ``label_side``, by half
    the pill's extent along the line's normal there, the tokens' reach
    (``flow_reach``, in scene units: the largest of every flow on the
    connector, whatever order they were made in) and :data:`LABEL_GAP`.
    Its place hangs on the connector and its flows alone. Make the flow
    before the diagram is shown, or the label moves as it is made. The
    label is also drawn above the tokens.
    """

    def __init__(
        self,
        edge: VMobject,
        speed: float = 1.0,
        tokens: int = 3,
        offset: float = 0.0,
        radius: float = 0.07,
        color: ParsableManimColor = YELLOW,
        **kwargs: Any,
    ) -> None:
        if tokens < 1:
            raise ValueError("a flow has at least one token")
        style = take_style(kwargs)
        super().__init__(**kwargs)
        self.edge = edge
        self.speed = float(speed)
        self.offset = float(offset)
        #: The scene time the flow started at; ``None`` until it runs in a
        #: scene.
        self.start: float | None = None
        #: Seconds of scene time the flow has run, as last shown.
        self.elapsed = 0.0
        self.stopped = False
        self.add(*(Dot(radius=radius, color=color) for _ in range(tokens)))
        apply_style(self.submobjects, style)
        _carry_label(edge, self.submobjects)
        _lift_label(edge, self.submobjects)
        self.show_at(0.0)
        self.add_updater(_start_flow)
        _run_in_scene(self)

    def travelled(self) -> float:
        """How far the first token has gone along the edge from its start,
        unwrapped: ``offset + speed * elapsed``.
        """
        return self.offset + self.speed * self.elapsed

    def show_at(self, seconds: float) -> Flow:
        """Put the tokens where they are ``seconds`` after the flow started."""
        self.elapsed = float(seconds)
        path = ArcLength(self.edge.points)
        if not path.length > _NO_LENGTH:
            raise ValueError("a path of zero length has no length")
        share = path.length / len(self.submobjects)
        for index, token in enumerate(self.submobjects):
            along = (self.travelled() + index * share) % path.length
            token.move_to(path.point(along))
        return self

    def follow(self, now: float) -> None:
        """Put the tokens where the scene time ``now`` puts them, unless the
        flow is stopped or has not started.
        """
        if not self.stopped and self.start is not None:
            self.show_at(now - self.start)

    def stop(self) -> Flow:
        """Stop the tokens where they are."""
        self.stopped = True
        return self

    def suspend_updating(self, recursive: bool = True) -> Self:
        # A play that takes the flow over suspends its own updater, which
        # may not have run in the scene yet: the scene's driver must hold
        # the flow by then.
        _run_in_scene(self)
        return super().suspend_updating(recursive)


#: The clear space, in scene units, between a flow's tokens and the label
#: of the connector they run along, set beside it.
LABEL_GAP = 0.06


def _carry_label(edge: VMobject, tokens: Sequence[VMobject]) -> None:
    """Have ``edge``'s label, if it has one, stand beside the line, clear of
    ``tokens`` (:class:`Flow`).
    """
    if getattr(edge, "label", None) is None:
        return
    reach = max(max(t.width, t.height) / 2 for t in tokens)
    cast(Any, edge).flow_reach = max(float(getattr(edge, "flow_reach", 0.0)), reach)
    _place_label(edge)


def _line_frame(edge: VMobject) -> tuple[np.ndarray, np.ndarray]:
    """The middle of ``edge``'s line and its normal there."""
    path = ArcLength(edge.points)
    middle = path.length / 2
    reach = min(middle, 1e-3)
    tangent = path.point(middle + reach) - path.point(middle - reach)
    normal = np.array([-tangent[1], tangent[0], 0.0]) / np.linalg.norm(tangent[:2])
    return path.point(middle), normal


#: The sides a label stands on beside its line, each the screen direction
#: it faces and the one a quarter-turn clockwise, for a line running along
#: the first.
LABEL_SIDES = ("above", "below", "left", "right")
_FACING = {
    "above": (UP, RIGHT),
    "below": (DOWN, LEFT),
    "left": (LEFT, UP),
    "right": (RIGHT, DOWN),
}


def _facing(normal: np.ndarray, side: str) -> np.ndarray:
    """Of ``normal`` and its opposite, the one facing ``side``."""
    for toward in _FACING[side]:
        along = float(np.dot(normal, toward))
        if abs(along) > 1e-9:
            return normal if along > 0 else -normal
    return normal


def _beside(edge: VMobject, reach: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Where ``edge``'s label stands beside the line, for tokens reaching
    ``reach`` from it: the line's middle, the normal there, and how far
    from the middle along it: half the pill's extent that way, the tokens'
    and :data:`LABEL_GAP`.
    """
    at, normal = _line_frame(edge)
    label: Mobject = cast(Any, edge).label
    across = abs(normal[0]), abs(normal[1])
    pill = (across[0] * label.width + across[1] * label.height) / 2
    # A token's box reaches ``reach`` along each axis.
    return at, normal, float(pill + reach * (across[0] + across[1]) + LABEL_GAP)


def _place_label(edge: VMobject) -> None:
    """Put ``edge``'s label beside its line on its side."""
    at, normal, offset = _beside(edge, float(getattr(edge, "flow_reach", 0.0)))
    toward = _facing(normal, str(getattr(edge, "label_side", "above")))
    label: Mobject = cast(Any, edge).label
    label.move_to(at + toward * offset)


def _lift_label(edge: VMobject, tokens: Sequence[VMobject]) -> None:
    """Put ``edge``'s label, if it has one, a depth above ``tokens``, so
    they pass under its pill. The scene draws by depth, and a frame's moving
    mobjects are its depth-ordered list from the first that moves, so the
    label, deeper than a moving token, is drawn after it wherever the flow
    stands in the scene. Nothing else moves in depth.
    """
    label = getattr(edge, "label", None)
    if label is None:
        return
    above = max(token.z_index for token in tokens) + 1
    if min(member.z_index for member in label.get_family()) < above:
        label.set_z_index(above)


quantities.register("diagram.", owner="arachne.mobject.diagram", units=("u/s",))

#: The length, in scene units, at or below which an edge has none to flow along
#: (a line collapsed to a point measures a rounding error, not 0).
_NO_LENGTH = 1e-9

#: The share of a flow's declared speed within which its tokens' advance
#: along the edge, by arc length, holds.
FLOW_TOLERANCE = 0.01


@quantities.provider(Flow)
def _declared(held: Mapping[str, Any], at: At) -> Iterable[Quantity]:
    """``diagram.flow_speed``: the tokens' declared advance along the edge,
    in u/s, not their visible velocity on a moving edge.
    """
    speed = 0.0 if held["stopped"] else held["speed"]
    return (
        Quantity(
            "diagram.flow_speed", speed, "u/s", tolerance=FLOW_TOLERANCE * abs(speed)
        ),
    )


def _start_flow(flow: Mobject, dt: float) -> None:
    """A flow's own updater until it starts in a scene. It reads ``dt``,
    which has the scene tick every instant it runs at, a wait's too, so it
    runs at the first instant the flow is in a running scene; there the
    flow starts and its updater becomes :func:`_keep_flow`.
    """
    assert isinstance(flow, Flow)
    if dt >= 0:  # the read the scene's ticking looks for
        _keep_flow(flow)


def _keep_flow(flow: Mobject) -> None:
    """A flow's own updater: the flow joins the running scene's driver, if
    it has not, and goes where the scene's time puts it.
    """
    assert isinstance(flow, Flow)
    scene = _run_in_scene(flow)
    if scene is not None:
        flow.follow(float(scene.time))


def _run_in_scene(flow: Flow) -> Scene | None:
    """Start ``flow`` in the scene running now, if any, and have the
    scene's driver hold it; the scene, or ``None``.
    """
    scene = running_scene()
    if scene is None:
        return None
    driver = flow_driver(scene)
    if driver is None or not driver.holds(flow):
        if flow.start is None:
            flow.start = float(scene.time)
        if _start_flow in flow.updaters:
            # The scene's driver keeps the scene ticking from now on.
            flow.remove_updater(_start_flow)
            flow.add_updater(_keep_flow)
        hold_flow(scene, flow)
    return scene


class FlowDriver:
    """What keeps a scene's flows on its clock: the flows started in the
    scene, each moved to where :attr:`~.Scene.time` puts it at every
    instant, at a cost of the flows held, whatever else the scene holds.

    A scene has at most one, made with its first flow and found with
    :func:`flow_driver`. It runs as the scene updater :attr:`updater`, last
    in every instant, after the animations and the mobjects' updaters, and
    unsuspended by any animation. :attr:`updater` is a plain function whose
    code never reads ``dt``, so it changes no play's ticking more than any
    scene updater does. Calling the driver moves, as one unit, every flow it
    holds that is not stopped; a flow that has left the scene is moved too
    and draws nothing. A flow joins its scene's driver only through
    :func:`hold_flow`. It is the first of the scene drivers a timing clock
    takes over (:mod:`arachne.assets.timing`): the clock takes
    :attr:`updater` out of the scene's updaters and calls the driver at the
    start of each of its instants.

    The driver is the scene's engine state, kept beside the scene (not on
    it, nor on any mobject), and no record reads it. It refers to the scene
    and to its flows weakly, so it keeps neither alive: a flow nothing else
    holds (an animation's copy, once the animation is done) leaves it.
    """

    def __init__(self, scene: Scene) -> None:
        self._scene = weakref.ref(scene)
        #: The flows held, by id, in the order they joined.
        self._held: dict[int, weakref.ref[Flow]] = {}

        def drive_flows(dt: float) -> None:
            self()

        #: The scene updater the driver runs as.
        self.updater: Callable[[float], None] = drive_flows

    @property
    def flows(self) -> tuple[Flow, ...]:
        """The flows held that still exist, in the order they joined."""
        return tuple(f for ref in self._held.values() if (f := ref()) is not None)

    def holds(self, flow: Flow) -> bool:
        """Whether the driver moves ``flow``."""
        ref = self._held.get(id(flow))
        return ref is not None and ref() is flow

    def hold(self, flow: Flow) -> None:
        """Move ``flow`` from now on."""
        self._held[id(flow)] = weakref.ref(flow, self._let_go(id(flow)))

    def _let_go(self, key: int) -> Callable[[weakref.ref[Flow]], None]:
        def gone(ref: weakref.ref[Flow]) -> None:
            if self._held.get(key) is ref:
                del self._held[key]

        return gone

    def __call__(self) -> None:
        """Put every flow held that is not stopped where the scene's time
        puts it.
        """
        scene = self._scene()
        if scene is None:
            return
        now = float(scene.time)
        for flow in self.flows:
            flow.follow(now)


#: Each scene's flow driver, by the scene; gone with the scene.
_DRIVERS: weakref.WeakKeyDictionary[Scene, FlowDriver] = weakref.WeakKeyDictionary()


def flow_driver(scene: Scene) -> FlowDriver | None:
    """``scene``'s flow driver, or ``None`` before its first flow."""
    return _DRIVERS.get(scene)


def hold_flow(scene: Scene, flow: Flow) -> None:
    """Have ``flow``, started in ``scene``, moved on the scene's clock: by
    the scene's flow driver, made with the scene's first flow and added to
    the scene's updaters, or handed to the scene's timing clock when it has
    one (:func:`arachne.assets.timing.drive`).
    """
    driver = _DRIVERS.get(scene)
    if driver is None:
        from arachne.assets.timing import drive

        driver = _DRIVERS[scene] = FlowDriver(scene)
        drive(scene, driver)
    driver.hold(flow)


#: The share of a zoom over which a piece fades in or out.
FADE = 0.25
#: How many instants of a zoom are read to find when a piece is cut.
_CUT_STEPS = 240
#: A label fades in only once this many times the least legible height.
LEGIBLE_MARGIN = 1.1


class ZoomInto(Animation):
    """Move the camera's frame onto ``block`` (a :class:`~.Block`) until it
    fills the frame but for ``margin`` of the frame on each side, while the
    block's inside diagram fades in and what the frame's edge would cut
    fades out, everything updated at every frame.

    The frame's height shrinks geometrically, about the one point the move
    holds still on screen, so the zoom keeps one speed. The inside fades in
    from the instant its smallest label reads at the lint's legible height
    (:data:`~arachne.lint.rules.text_small.SMALL_PIXELS`, with
    :data:`LEGIBLE_MARGIN`), or over the zoom's last :data:`FADE` if it
    never does. Each other piece of the scene the frame's edge would cut
    (a top-level mobject, or a piece of a group that holds the block)
    fades out over the :data:`FADE` before it is cut and stays hidden. A
    diagram's connector, with its label, fades no later than the
    earliest-fading piece it joins (the block itself never fades), so no
    arrow shows without its ends.

    The frame is the scene's camera's, which only a moving camera has (a
    :class:`~.MovingCameraScene`); a ``frame`` passed must be that one.
    ``rate_func`` eases the zoom once, for the frame and every fade alike.
    """

    def __init__(
        self,
        block: Block,
        frame: Mobject | None = None,
        margin: float = 0.08,
        run_time: float = 2.0,
        rate_func: Callable[[float], float] = smooth,
        **kwargs: Any,
    ) -> None:
        if not isinstance(block, Block):
            raise TypeError(
                f"ZoomInto zooms into a Block, not a {type(block).__name__}"
            )
        if frame is None:
            frame = _camera_frame()
        super().__init__(frame, run_time=run_time, rate_func=rate_func, **kwargs)
        self.block = block
        self.margin = float(margin)
        self._scene_mobjects: tuple[Mobject, ...] = ()

    def _setup_scene(self, scene: Scene) -> None:
        from arachne.camera.moving_camera import MovingCamera

        camera = scene.camera
        if not isinstance(camera, MovingCamera) or camera.frame is not self.mobject:
            raise ValueError(
                "ZoomInto moves the scene's camera's frame, and only a moving"
                " camera has one: make the scene a MovingCameraScene, and pass"
                " no frame or that camera's"
            )
        self._scene_mobjects = tuple(scene.mobjects)
        super()._setup_scene(scene)

    def begin(self) -> None:
        frame, block = self.mobject, self.block
        #: The frame as the move begins, which a return puts back.
        self._from_points = frame.points
        self._from = (frame.get_center(), float(frame.height))
        aspect = frame.width / frame.height
        height = max(block.height, block.width / aspect) * (1 + 2 * self.margin)
        self._to = (block.get_center(), float(height))
        self._aspect = aspect
        inside = block.inside
        self._fade_in = (1.0, 1.0)
        if inside is not None:
            if len(inside.get_family()) != len(block.inside_shown):
                raise ValueError(
                    "a block's inside changed after the block was made: make the"
                    " block from the inside as it is to be shown"
                )
            self._fade_in = _window(self._legible_from(inside))
        #: Each piece the frame's edge cuts, or a connector or label whose
        #: ends it cuts: its family, how each member shows now, and the
        #: share of the zoom it fades out over.
        self._fading: list[
            tuple[list[Mobject], list[Opacities], tuple[float, float]]
        ] = []
        pieces = _pieces(self._scene_mobjects, block, frame)
        cuts = {id(piece): self._cut_at(piece) for piece in pieces}
        joins = _joins(self._scene_mobjects)
        for piece in pieces:
            # A connector, and its label, fades no later than what it joins.
            held = [
                cuts[id(piece)],
                *(cuts.get(id(m)) for m in joins.get(id(piece), ())),
            ]
            cut = min((c for c in held if c is not None), default=None)
            if cut is None:
                continue
            family = list(piece.get_family())
            self._fading.append(
                (family, [opacities_of(m) for m in family], (max(0.0, cut - FADE), cut))
            )
        super().begin()

    def _height(self, alpha: float) -> float:
        h0, h1 = self._from[1], self._to[1]
        return float(h0 * (h1 / h0) ** alpha)

    def _center(self, alpha: float) -> np.ndarray:
        (c0, h0), (c1, h1) = self._from, self._to
        if math.isclose(h0, h1):
            return c0 + (c1 - c0) * alpha
        return c0 + (c1 - c0) * (h0 - self._height(alpha)) / (h0 - h1)

    def _legible_from(self, inside: Mobject) -> float:
        """The first share of the zoom at which every label inside the block
        reads at the legible height; past the end when none does.
        """
        from arachne.lint.codes import THRESHOLD_HEIGHT
        from arachne.lint.rules.text_small import SMALL_PIXELS

        heights = [_glyph_height(text) for text in _revealed_texts(inside)]
        heights = [h for h in heights if h > 0]
        if not heights:
            return 0.0
        width, tall = int(config.pixel_width), int(config.pixel_height)
        pixels_per_unit_at_height_one = THRESHOLD_HEIGHT * tall / min(width, tall)
        # A text h tall shows h * that / H pixels when the frame is H tall.
        needed = (
            min(heights)
            * pixels_per_unit_at_height_one
            / (SMALL_PIXELS * LEGIBLE_MARGIN)
        )
        h0, h1 = self._from[1], self._to[1]
        if needed >= h0:
            return 0.0
        if needed < h1 or math.isclose(h0, h1):
            return math.inf
        return math.log(needed / h0) / math.log(h1 / h0)

    def _cut_at(self, piece: Mobject) -> float | None:
        """The first share of the zoom at which the frame's edge cuts
        ``piece``, or ``None`` if it never does.
        """
        if not any(len(m.points) for m in piece.get_family()):
            return None
        low = piece.get_critical_point(LEFT + DOWN)
        high = piece.get_critical_point(RIGHT + UP)
        for step in range(_CUT_STEPS + 1):
            alpha = step / _CUT_STEPS
            centre, half_h = self._center(alpha), self._height(alpha) / 2
            half_w = half_h * self._aspect
            if (
                low[0] < centre[0] - half_w
                or high[0] > centre[0] + half_w
                or low[1] < centre[1] - half_h
                or high[1] > centre[1] + half_h
            ):
                return alpha
        return None

    def interpolate_mobject(self, alpha: float) -> None:
        # Eased once, for the frame and every fade alike.
        share = self.get_sub_alpha(alpha, 0, 1)
        self.show_share(share)
        #: The share the move last showed, and the frame's points then:
        #: where it left the frame, which a return starts from.
        self._reached = share
        self._left_points = self.mobject.points
        # Only a move standing on its block holds the frame.
        if abs(share - 1.0) <= _EASED:
            _hold_zoom(self)
        else:
            _release_zoom(self)

    def show_share(self, share: float) -> None:
        """Show the move at ``share`` of its way: the frame, the inside's
        fade and each cut piece's, as the move in and the return
        (:class:`ZoomOut`) both draw it.
        """
        frame = self.mobject
        frame.scale_to_fit_height(self._height(share))
        frame.move_to(self._center(share))
        inside = self.block.inside
        if inside is not None:
            shown_share = _ramp(share, self._fade_in)
            for member, shown in zip(
                inside.get_family(), self.block.inside_shown, strict=True
            ):
                set_opacities(member, scaled(shown, shown_share))
        for family, faded, window in self._fading:
            keep = 1.0 - _ramp(share, window)
            for member, opacities in zip(family, faded, strict=True):
                set_opacities(member, scaled(opacities, keep))


#: The zooms that hold each camera frame, by the frame's id: each entry
#: holds its frame and its zooms weakly, innermost last, and goes with its
#: frame. A zoom joins once its move stands on its block, and leaves when a
#: later share of it, or its return, takes the frame off.
_HOLDING: dict[int, tuple[weakref.ref[Mobject], list[weakref.ref[ZoomInto]]]] = {}


def _holders(frame: Mobject) -> list[weakref.ref[ZoomInto]]:
    """The zooms holding ``frame``, innermost last, those gone left out."""
    key = id(frame)
    entry = _HOLDING.get(key)
    if entry is None or entry[0]() is not frame:
        weakref.finalize(frame, _forget_frame, key)
        entry = _HOLDING[key] = (weakref.ref(frame), [])
    entry[1][:] = [held for held in entry[1] if held() is not None]
    return entry[1]


def _forget_frame(key: int) -> None:
    """Drop the entry of a frame that went, if no later frame took its id."""
    entry = _HOLDING.get(key)
    if entry is not None and entry[0]() is None:
        del _HOLDING[key]


def _release_zoom(zoom: ZoomInto) -> None:
    """Have ``zoom`` hold its frame no more."""
    holders = _holders(zoom.mobject)
    holders[:] = [held for held in holders if held() is not zoom]


def _hold_zoom(zoom: ZoomInto) -> None:
    """Have ``zoom`` hold its frame, innermost, once."""
    holders = _holders(zoom.mobject)
    if not holders or holders[-1]() is not zoom:
        holders.append(weakref.ref(zoom))


#: How far, as a share of the height the zoom left it at, the camera's frame
#: may stand from where a zoom left it for the zoom to come back.
RETURN_TOLERANCE = 1e-6
#: How near an eased share must come to 0 or 1 to count as there.
_EASED = 1e-9


class ZoomOut(Animation):
    """Undo a played :class:`ZoomInto`: the camera's frame takes the
    move-in's own path backwards, from the block to where the move-in began,
    while the block's inside fades out over its fade-in window reversed and
    each piece the move-in faded comes back over its own window reversed, to
    the opacities it showed before the move-in.

    At eased progress ``p`` the frame takes the height and centre the
    move-in gave it at share ``1 - p``. At the end the frame is exactly where
    the move-in began, the inside is hidden, and every faded piece shows as
    before. ``run_time`` and ``rate_func`` are the move-in's unless given,
    read when the return plays, so a play's own ``run_time`` for the move-in
    counts.

    Nested zooms come back in order, the inner first. A return is refused,
    by name, when its zoom was never played, or ended short of its block;
    when its own easing (``rate_func``, after ``reverse_rate_function``)
    does not take 0 to 0 and 1 to 1; when the
    camera's frame does not stand where that zoom left it, every point
    within :data:`RETURN_TOLERANCE` of its height (a camera moved,
    stretched or turned since); when a zoom played after it on the same
    frame, its move ended on its block, has not come back (the zooms
    holding a frame are kept beside it, by identity, innermost last; one
    that went out and came back, a ``there_and_back``, holds nothing);
    when it has already come back; when a
    piece the zoom faded has left the scene; and in a scene without that
    zoom's moving camera. The move in and the return draw each share alike
    (:meth:`ZoomInto.show_share`).
    """

    def __init__(
        self,
        zoom: ZoomInto,
        run_time: float | None = None,
        rate_func: Callable[[float], float] | None = None,
        **kwargs: Any,
    ) -> None:
        if not isinstance(zoom, ZoomInto):
            raise TypeError(
                f"ZoomOut returns from a ZoomInto, not a {type(zoom).__name__}"
            )
        self.zoom = zoom
        super().__init__(zoom.mobject, **kwargs)
        # Unset, each is the move-in's, read when the return plays.
        self.run_time, self.rate_func = run_time, rate_func
        self._scene: weakref.ref[Scene] | None = None

    @property
    def run_time(self) -> float:
        own = self._own_run_time
        return self.zoom.run_time if own is None else own

    @run_time.setter
    def run_time(self, value: float | None) -> None:
        if value is not None and value < 0:
            raise ValueError(
                f"The run_time of ZoomOut cannot be negative. The given value was"
                f" {value}."
            )
        self._own_run_time = value

    @property
    def rate_func(self) -> Callable[[float], float]:
        own = self._own_rate_func
        return self.zoom.rate_func if own is None else own

    @rate_func.setter
    def rate_func(self, value: Callable[[float], float] | None) -> None:
        self._own_rate_func = value

    def _setup_scene(self, scene: Scene) -> None:
        from arachne.camera.moving_camera import MovingCamera

        camera = scene.camera
        if not isinstance(camera, MovingCamera) or camera.frame is not self.mobject:
            raise ValueError(
                "ZoomOut moves the scene's camera's frame back, and only a moving"
                " camera has one: make the scene a MovingCameraScene, and return"
                " from a ZoomInto of that camera's frame"
            )
        self._scene = weakref.ref(scene)
        super()._setup_scene(scene)

    def begin(self) -> None:
        zoom, frame = self.zoom, self.mobject
        if not hasattr(zoom, "_from_points"):
            raise ValueError(
                "ZoomOut returns from a played ZoomInto, and this one was never"
                " played: play the ZoomInto first"
            )
        if abs(zoom._reached - 1.0) > _EASED:
            raise ValueError(
                f"ZoomOut returns from a ZoomInto that ended on its block, and this"
                f" one never reached its block (it ended at {zoom._reached:.4g} of"
                " its way): give the move-in a rate_func that ends at 1"
            )
        # The easing as the return applies it, reverse_rate_function and all.
        first, last = self.get_sub_alpha(0.0, 0, 1), self.get_sub_alpha(1.0, 0, 1)
        if abs(first) > _EASED or abs(last - 1.0) > _EASED:
            raise ValueError(
                "a ZoomOut's easing (its rate_func, after reverse_rate_function)"
                " must take 0 to 0 and 1 to 1, so the return starts on the block"
                " and ends where the move-in began"
            )
        left = np.asarray(zoom._left_points)
        tolerance = RETURN_TOLERANCE * zoom._height(1.0)
        if left.shape != frame.points.shape or (
            float(np.max(np.abs(frame.points - left), initial=0.0)) > tolerance
        ):
            raise ValueError(
                "ZoomOut returns the camera's frame from where its ZoomInto left it,"
                f" {float(zoom._height(1.0)):.4g} tall about"
                f" {np.round(zoom._center(1.0), 4).tolist()}, but it is"
                f" {float(frame.height):.4g} tall about"
                f" {np.round(frame.get_center(), 4).tolist()}"
                f" ({float(frame.width):.4g} wide): return from an inner zoom before"
                " an outer one, and move the camera only after the return"
            )
        holders = _holders(frame)
        if not any(held() is zoom for held in holders):
            raise ValueError(
                "this ZoomInto has already come back: play it again before"
                " another ZoomOut of it"
            )
        if holders[-1]() is not zoom:
            raise ValueError(
                "ZoomOut returns from the innermost zoom first: a zoom played"
                " after this one holds the frame, so return from it before this"
                " one"
            )
        scene = self._scene() if self._scene is not None else None
        if scene is not None:
            present = {id(m) for m in scene.get_mobject_family_members()}
            for family, _, _ in zoom._fading:
                if id(family[0]) not in present:
                    raise ValueError(
                        f"a {type(family[0]).__name__} the ZoomInto faded has left"
                        " the scene: ZoomOut brings back each piece its ZoomInto"
                        " faded, so keep it in the scene until the return"
                    )
        holders.pop()
        super().begin()

    def interpolate_mobject(self, alpha: float) -> None:
        # Eased once, for the frame and every fade alike; the move-in's share,
        # none at the end.
        share = 0.0 if alpha >= 1.0 else 1.0 - self.get_sub_alpha(alpha, 0, 1)
        self.zoom.show_share(max(share, 0.0))
        if share <= 0.0:
            # Exactly where the move-in began.
            self.mobject.points = self.zoom._from_points


def _window(start: float) -> tuple[float, float]:
    """The share of a zoom over which something that may show from
    ``start`` on fades in: from ``start``, or over the last :data:`FADE`
    when ``start`` is past the end.
    """
    begin = start if start <= 1.0 else 1.0 - FADE
    return (begin, min(1.0, begin + FADE))


def _ramp(alpha: float, window: tuple[float, float]) -> float:
    """0 before ``window``, 1 after it, linear through it."""
    low, high = window
    if alpha <= low:
        return 0.0
    if alpha >= high or high <= low:
        return 1.0
    return (alpha - low) / (high - low)


def _pieces(
    mobjects: Iterable[Mobject], block: Mobject, frame: Mobject
) -> list[Mobject]:
    """What a zoom onto ``block`` may fade: each of ``mobjects`` but the
    frame, or, for one that holds the block, its pieces that do not, at
    every depth down to the block.
    """
    found: list[Mobject] = []

    def walk(pieces: Iterable[Mobject]) -> None:
        for piece in pieces:
            if piece is block or piece is frame:
                continue
            if block in piece.get_family():
                walk(piece.submobjects)
            else:
                found.append(piece)

    walk(mobjects)
    return found


def _joins(mobjects: Iterable[Mobject]) -> dict[int, tuple[Mobject, ...]]:
    """What each connector of a diagram among ``mobjects`` joins, by its
    id: the pieces of the diagram it was made to join and still holds,
    found by their keys there (:attr:`Diagram.joined`); and for its label,
    the connector and those pieces.
    """
    found: dict[int, tuple[Mobject, ...]] = {}
    seen: set[int] = set()
    for mob in mobjects:
        for diagram in mob.get_family():
            if not isinstance(diagram, Diagram) or id(diagram) in seen:
                continue
            seen.add(id(diagram))
            for edge in diagram.edges:
                keys = (getattr(edge, name, None) for name in ("start_key", "end_key"))
                ends = tuple(
                    end
                    for key in keys
                    if (end := diagram._joined_piece(key)) is not None
                )
                found[id(edge)] = ends
                label = getattr(edge, "label", None)
                if label is not None:
                    found[id(label)] = (edge, *ends)
    return found


def _revealed_texts(inside: Mobject) -> list[Mobject]:
    """The texts opening a block shows: those in its inside, but for the
    hidden insides of the blocks within it.
    """
    found: list[Mobject] = []

    def walk(mob: Mobject) -> None:
        if _is_text(mob):
            found.append(mob)
            return
        for sub in mob.submobjects:
            if isinstance(mob, Block) and sub is mob.inside:
                continue
            walk(sub)

    walk(inside)
    return found


def _glyph_height(text: Mobject) -> float:
    """The height of a text's glyphs, its plate (a pill) left out, as the
    lint measures a text's ink.
    """
    plate = getattr(text, "background_rectangle", None)
    skip = {id(m) for m in plate.get_family()} if plate is not None else set()
    points = [
        np.asarray(m.points)
        for m in text.get_family()
        if id(m) not in skip and len(m.points)
    ]
    if not points:
        return 0.0
    ys = np.concatenate(points)[:, 1]
    return float(ys.max() - ys.min())


def _is_text(mob: Mobject) -> bool:
    from arachne.mobject.text.numbers import DecimalNumber
    from arachne.mobject.text.tex_mobject import SingleStringMathTex
    from arachne.mobject.text.text_mobject import MarkupText, Paragraph
    from arachne.mobject.text.typst_mobject import Typst

    return isinstance(
        mob, (Text, MarkupText, Paragraph, SingleStringMathTex, DecimalNumber, Typst)
    )


def _camera_frame() -> Mobject:
    """The frame of the running scene's camera, which must move."""
    from arachne.camera.moving_camera import MovingCamera
    from arachne.timeline.recorder import running

    recorder = running()
    camera = getattr(recorder.scene, "camera", None) if recorder is not None else None
    if not isinstance(camera, MovingCamera):
        raise ValueError(
            "ZoomInto moves the camera's frame, and only a moving camera has one:"
            " make the scene a MovingCameraScene"
        )
    frame: Mobject = camera.frame
    return frame
