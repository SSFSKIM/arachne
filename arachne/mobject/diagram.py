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

A diagram's labels are measured against each other, against the frame's
edge and for their height by the lint rule
:data:`~arachne.lint.rules.diagram_clear`, which the sweep's pair rules,
exempting two pieces of one top-level mobject, leave unasked.
"""

from __future__ import annotations

__all__ = ["Diagram", "Flow", "ZoomInto"]

import math
import weakref
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from arachne import config
from arachne.animation.animation import Animation
from arachne.constants import DOWN, LEFT, ORIGIN, RIGHT, UP
from arachne.mobject.geometry.arc import Dot
from arachne.mobject.marks._paths import ArcLength, as_point, direction, head, line
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
from arachne.utils.color import WHITE, YELLOW, ParsableManimColor
from arachne.utils.rate_functions import smooth

if TYPE_CHECKING:
    from collections.abc import Callable

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
        color: ParsableManimColor = WHITE,
        label_color: ParsableManimColor | None = None,
        **kwargs: Any,
    ) -> VMobject:
        """Join ``start`` to ``end`` (each a piece and one of its port
        names, or a piece alone, which is joined on the side facing the
        other end) with a connector, add it, and return it.

        ``route`` is ``"elbow"`` (right angles, rounded corners) or
        ``"straight"``. ``label`` puts a text on a pill (a plate the
        background's colour) at the connector's middle, which the
        connector keeps as ``label``.
        """
        if route not in ("elbow", "straight"):
            raise ValueError(
                f"a connector's route is 'elbow' or 'straight', not {route!r}"
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
                corner_radius=corner_radius,
                tip=tip,
                color=color,
                **kwargs,
            )
        else:
            edge = _StraightConnector(a, b, tip=tip, color=color, **kwargs)
        self.add(edge)
        self.edges = (*self.edges, edge)
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
            self.add(text)
        return edge


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
    scene time since the flow was made, read from the scene's own clock
    (:attr:`~.Scene.time`), never summed from steps. Two engine updaters
    keep it there: its own, which a play's end runs, and the scene's
    :class:`FlowDriver`, which the scene runs after every animation and
    mobject updater at every instant and no animation suspends. So the speed and the spacing hold at any frame
    rate, through plays and waits, whatever animates the edge, the diagram
    the flow is in or the flow itself, until :meth:`stop`. It reads the
    edge's own points (a connector's wire, not its head) as they are.
    Style keywords go to the tokens.
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
        self.speed = float(speed)
        self.offset = float(offset)
        scene = _running_scene()
        #: The scene time the flow was made at, from which it runs.
        self.start = float(scene.time) if scene is not None else 0.0
        #: Seconds of scene time the flow has run, as last shown.
        self.elapsed = 0.0
        self.stopped = False
        self.add(*(Dot(radius=radius, color=color) for _ in range(tokens)))
        apply_style(self.submobjects, style)

        def place(flow: Flow, seconds: float) -> None:
            flow.elapsed = seconds
            path = ArcLength(edge.points)
            share = path.length / len(flow.submobjects)
            for index, token in enumerate(flow.submobjects):
                along = (flow.travelled() + index * share) % path.length
                token.move_to(path.point(along))

        def follow(flow: Mobject) -> None:
            assert isinstance(flow, Flow)
            if scene is not None and not flow.stopped:
                place(flow, float(scene.time) - flow.start)

        self._show = place
        self._follow = follow
        place(self, 0.0)
        self.add_updater(follow)
        if scene is not None:
            hold_flow(scene, self)

    def travelled(self) -> float:
        """How far the first token has gone along the edge from its start,
        unwrapped: ``offset + speed * elapsed``.
        """
        return self.offset + self.speed * self.elapsed

    def show_at(self, seconds: float) -> Flow:
        """Put the tokens where they are ``seconds`` after the flow started."""
        self._show(self, float(seconds))
        return self

    def stop(self) -> Flow:
        """Stop the tokens where they are."""
        self.stopped = True
        return self


def _running_scene() -> Scene | None:
    """The scene whose script is running now, if any."""
    from arachne.timeline.recorder import running

    recorder = running()
    return recorder.scene if recorder is not None else None


class FlowDriver:
    """What keeps a scene's flows on its clock: the flows made in the
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
    :func:`hold_flow`. Something else that is to run the flows, such as a
    clock that runs last itself, takes :attr:`updater` out of the scene's
    updaters, calls the driver at its own instant, and has
    :func:`hold_flow` hand it later flows.

    The driver is the scene's engine state, kept beside the scene (not on
    it, nor on any mobject); :attr:`flows` is replaced, never changed in
    place.
    """

    def __init__(self) -> None:
        #: The flows held, in the order they were made.
        self.flows: tuple[Flow, ...] = ()

        def drive_flows(dt: float) -> None:
            self()

        #: The scene updater the driver runs as.
        self.updater: Callable[[float], None] = drive_flows

    def hold(self, flow: Flow) -> None:
        """Move ``flow`` from now on."""
        self.flows = (*self.flows, flow)

    def __call__(self) -> None:
        """Put every flow held that is not stopped where the scene's time
        puts it.
        """
        for flow in self.flows:
            if not flow.stopped:
                flow._follow(flow)


#: Each scene's flow driver, by the scene; gone with the scene.
_DRIVERS: weakref.WeakKeyDictionary[Scene, FlowDriver] = weakref.WeakKeyDictionary()


def flow_driver(scene: Scene) -> FlowDriver | None:
    """``scene``'s flow driver, or ``None`` before its first flow."""
    return _DRIVERS.get(scene)


def hold_flow(scene: Scene, flow: Flow) -> None:
    """Have ``flow``, made in ``scene``, moved on the scene's clock: by the
    scene's flow driver, made and added to the scene's updaters with the
    scene's first flow.
    """
    driver = _DRIVERS.get(scene)
    if driver is None:
        driver = _DRIVERS[scene] = FlowDriver()
        scene.add_updater(driver.updater)
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
    fades out over the :data:`FADE` before it is cut and stays hidden.

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
        #: Each piece the frame's edge cuts: its family, how each member
        #: shows now, and the share of the zoom it fades out over.
        self._fading: list[
            tuple[list[Mobject], list[Opacities], tuple[float, float]]
        ] = []
        for piece in _pieces(self._scene_mobjects, block, frame):
            cut = self._cut_at(piece)
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
        alpha = self.get_sub_alpha(alpha, 0, 1)
        frame = self.mobject
        frame.scale_to_fit_height(self._height(alpha))
        frame.move_to(self._center(alpha))
        inside = self.block.inside
        if inside is not None:
            share = _ramp(alpha, self._fade_in)
            for member, shown in zip(
                inside.get_family(), self.block.inside_shown, strict=True
            ):
                set_opacities(member, scaled(shown, share))
        for family, faded, window in self._fading:
            keep = 1.0 - _ramp(alpha, window)
            for member, opacities in zip(family, faded, strict=True):
                set_opacities(member, scaled(opacities, keep))


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
