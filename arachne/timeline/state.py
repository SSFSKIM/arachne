"""The scene's state at an instant, as ``state_at`` answers it."""

from __future__ import annotations

__all__ = ["NodeView", "State", "build_state"]

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np

from arachne.camera.projection import home_box, render_mapping_centre, view_of

from .draw import Drawing, camera_nodes
from .record import NodeState

if TYPE_CHECKING:
    from arachne.mobject.mobject import Mobject

    from .record import Capture, PlayRecord, Record
    from .source import SourceLine


@dataclass(frozen=True)
class NodeView:
    """One mobject of the scene at an instant.

    ``box`` is ``(x_min, y_min, x_max, y_max)`` over the points of the
    mobject and its family where the picture shows them, in home-frame
    units: scene units of the frame the configuration describes, centred,
    y up (:mod:`arachne.camera.projection`). They are scene units under a
    camera that never moves, follow the view under a MovingCamera, and are
    projected under a ThreeDCamera. ``None`` when the family has no points.
    """

    id: str
    source_line: SourceLine | None
    name: str | None
    type: str
    box: tuple[float, float, float, float] | None
    z: float
    color: str | None
    text: str | None
    children: tuple[str, ...]


@dataclass(frozen=True)
class State:
    """The scene at ``t``: its mobjects depth first from the top-level ones
    (trackers, the camera's own mobjects and the fully transparent padding
    that family alignment adds are left out), the play whose state it is and
    the route that answered.

    ``answer`` says how: ``"recorded"`` is a capture of the forward run,
    ``"analytic"`` a trusted play evaluated at ``t`` from its last capture
    before it, ``"restored"`` a state-function play evaluated at ``t`` from
    its last kept capture before it (:mod:`.restore`), and ``"forward"`` a
    new execution of the scene evaluated at ``t``.

    ``as_of`` is the instant whose state it is: the instant of the capture a
    recorded answer reads, and ``t`` for an evaluated one. A recorded answer
    reflects an instant before ``t`` in a frozen frame, and in a ticked play
    whose record kept no tick at ``t``: there the answer is the last capture
    a frame read at or before ``t``, the frame a render shows, an earlier
    play's before the play's first frame. ``play`` is then that play.
    """

    t: Fraction
    play: int | None
    route: str | None
    answer: str
    roots: tuple[str, ...]
    nodes: tuple[NodeView, ...]
    as_of: Fraction

    def node(self, node_id: str) -> NodeView:
        """The node with stable id ``node_id``."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise KeyError(node_id)

    def named(self, name: str) -> NodeView:
        """The node the scene's code named ``name``."""
        for node in self.nodes:
            if node.name == name:
                return node
        raise KeyError(name)


def build_state(
    record: Record,
    t: Fraction,
    play: PlayRecord | None,
    capture: Capture,
    *,
    route: str | None,
    answer: str,
    as_of: Fraction,
    prepare: Callable[[Drawing], None] | None = None,
) -> State:
    """The :class:`State` a capture holds, read with every reachable mobject
    given its captured values; ``prepare`` runs once they are (a trusted
    play's evaluation at an instant between its captures, or a state play's
    restore), and the tree is then read as the evaluation left it: a
    callback may rebuild a mobject's submobjects.
    """
    camera = record._scene.renderer.camera
    hidden = {id(mob) for mob in camera_nodes(camera)}
    names = play.names if play is not None else record.final_names
    views: list[NodeView] = []
    frozen = render_mapping_centre(record)
    with Drawing(record) as drawing:
        drawing.show(capture, capture.draw_list())
        if prepare is not None:
            prepare(drawing)
        # Paths map around the centre the render froze; where it froze none,
        # a first draw here would freeze the centre the capture holds.
        view = view_of(
            camera,
            mapping_centre=frozen if frozen is not None else camera.frame_center,
            home=record._scene.timeline.home,
        )
        if prepare is None:

            def state_of(mob: Mobject) -> NodeState | None:
                return record.node_state(mob, capture)

        else:

            def state_of(mob: Mobject) -> NodeState | None:
                return NodeState.of(mob)

        visible = _visible_ids(capture, state_of, hidden)
        for mob, state in _walk(capture.draw_list(), state_of):
            if id(mob) not in visible:
                continue
            origin = record.origin(mob)
            children = tuple(
                record.origin(child).id
                for child in state.submobjects
                if id(child) in visible
            )
            views.append(
                NodeView(
                    id=origin.id,
                    source_line=origin.source_line,
                    name=names.get(id(mob)),
                    type=type(mob).__name__,
                    box=home_box(camera, view, mob),
                    z=float(getattr(mob, "z_index", 0)),
                    color=_color(mob),
                    text=_text(mob),
                    children=children,
                )
            )
        roots = tuple(
            record.origin(mob).id for mob in capture.draw_list() if id(mob) in visible
        )
    return State(
        t, play.index if play else None, route, answer, roots, tuple(views), as_of
    )


def _walk(
    roots: list[Mobject], state_of: Callable[[Mobject], NodeState | None]
) -> Iterator[tuple[Mobject, NodeState]]:
    """The mobjects reachable from ``roots``, depth first, each once, with
    their states by ``state_of`` (as :meth:`.Record._walk` reads a capture).
    """
    seen: set[int] = set()
    stack = list(reversed(roots))
    while stack:
        mob = stack.pop()
        if id(mob) in seen:
            continue
        seen.add(id(mob))
        state = state_of(mob)
        if state is None:
            continue
        yield mob, state
        stack.extend(reversed(state.submobjects))


def _visible_ids(
    capture: Capture,
    state_of: Callable[[Mobject], NodeState | None],
    hidden: set[int],
) -> set[int]:
    """The mobjects ``state_at`` reports: every reachable one but trackers,
    the camera's own, and padding that is fully transparent with all it holds.
    """
    from arachne.mobject.mobject import is_alignment_padding
    from arachne.mobject.value_tracker import ValueTracker

    visible: set[int] = set()

    def keep(mob: Mobject, state: NodeState, seen: set[int]) -> bool:
        if id(mob) in seen:
            return id(mob) in visible
        seen.add(id(mob))
        shown = False
        for child in state.submobjects:
            child_state = state_of(child)
            if child_state is not None and keep(child, child_state, seen):
                shown = True
        if isinstance(mob, ValueTracker) or id(mob) in hidden:
            return False
        if shown or not (is_alignment_padding(mob) and _transparent(mob)):
            visible.add(id(mob))
            return True
        return False

    seen: set[int] = set()
    for mob in capture.draw_list():
        state = state_of(mob)
        if state is not None:
            keep(mob, state, seen)
    return visible


def _transparent(mob: Mobject) -> bool:
    """Whether nothing of ``mob`` itself can be seen: every colour it holds
    has zero opacity.
    """
    for getter in ("get_fill_opacities", "get_stroke_opacities"):
        method = getattr(mob, getter, None)
        if method is None:
            continue
        try:
            opacities = np.asarray(method(), dtype=float)
        except Exception:
            return False
        if opacities.size and np.any(opacities > 0):
            return False
    opacity = getattr(mob, "opacity", None)
    return not (isinstance(opacity, (int, float)) and opacity > 0)


def _color(mob: Mobject) -> str | None:
    try:
        return mob.get_color().to_hex()
    except Exception:
        return None


def _text(mob: Mobject) -> str | None:
    from arachne.mobject.text.numbers import DecimalNumber

    if isinstance(mob, DecimalNumber):
        return mob._get_num_string(mob.number)
    for attribute in ("original_text", "tex_string", "text", "code_string"):
        value = mob.__dict__.get(attribute)
        if isinstance(value, str):
            return value
    return None
