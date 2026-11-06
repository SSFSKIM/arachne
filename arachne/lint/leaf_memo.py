"""What a lint reads off a leaf, kept across its instants.

A leaf is read as the camera draws it (:class:`~.parts.Leaf`). What it lays
(its paint, its strokes as set, whether its subpaths close, its box in scene
units) follows from its own state alone. Where the camera puts it (its ink
box, its points on the screen and their box) follows from its state and the
camera's. A lint reads one leaf at many instants, most of them holding it as
it was, so :class:`LeafMemo` keeps each read by what it follows from and
hands it to a later read of the same.

A leaf's state is the record's version of it (:class:`~.record.NodeState`).
A version's values are immutable, and a mobject holding anything else holds
another version, so one version is one key. The camera's key is made of:

- its recorded inputs (:class:`~.draw.CameraExtras`: its plain values, what
  is fixed in its frame and each fixed-orientation centre), one object while
  they stay the same, and inputs observed anew but the same keyed as the
  ones before them;
- the versions of its own mobjects (its frame, trackers and light);
- the view a read maps through (:class:`~.projection.View`), by value.

A read is kept only where that key sees everything it depends on. These are
read every time:

- a live read of a forward run's scene, which holds no recorded versions
  (its caller passes no memo);
- every read of a record whose camera runs the scene's code
  (``Record.camera_replaced``);
- every read of a capture with scene updaters, or whose camera draws other
  cameras' displays, whose pictures are shown apart from the versions;
- every read where one of the camera's own mobjects has no version at the
  capture;
- a leaf that has no version there, or that has updaters itself or through
  a mobject holding it;
- where an image is placed: its draw can skip the projection, so a kept
  place could not leave the camera as the read it stands for would.
"""

from __future__ import annotations

__all__ = ["CAMERAS_KEPT", "LeafMemo", "Lays", "Placed", "Viewing"]

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from arachne.camera.projection import Box, View, hold_rotation
from arachne.timeline.draw import camera_nodes

if TYPE_CHECKING:
    import numpy as np

    from arachne.mobject.mobject import Mobject
    from arachne.timeline.draw import CameraExtras
    from arachne.timeline.record import Capture, NodeState, Record

#: How many camera keys' places are kept, the latest read first. A lint
#: reads its instants in order, so a camera held still is read again soon;
#: one read again after this many others is placed again.
CAMERAS_KEPT = 16

#: A leaf as one version: the mobject and its version at the capture read.
Node = tuple["Mobject", "NodeState"]
_Key = tuple[int, int]
_ABSENT: Any = object()


@dataclass(frozen=True)
class Lays:
    """What a leaf lays whatever the camera: its paint (:func:`~.parts._paint`),
    its strokes as set (:func:`~.parts._strokes`), whether its subpaths close,
    its box in scene units, the opacity of its fill's first row, and the
    points per curve it is drawn with.
    """

    fill: float
    stroke: float
    width: float
    line: float
    halo: float
    halo_opacity: float
    closed: bool
    scene_box: Box
    first_fill: float
    curve_size: int


@dataclass(frozen=True)
class Placed:
    """Where the camera puts a leaf: the box of what its draw paints, its
    points on the screen (read-only, as reads share them), and their box.
    """

    ink: Box
    screen: np.ndarray
    box: Box


class LeafMemo:
    """One lint's kept leaf reads: what each version of a leaf lays, and
    where each version is placed under each of the latest camera keys.
    """

    def __init__(self) -> None:
        #: By leaf version: the version's mobject and state (held, so no id
        #: in a key is reused while it is kept) and what it lays.
        self._lays: dict[_Key, tuple[Mobject, NodeState, Any]] = {}
        #: By camera key, latest last: the key's objects (held likewise) and
        #: each leaf version's place under it (``None``: it paints nothing).
        self._placed: OrderedDict[
            tuple[Any, ...], tuple[Any, dict[_Key, Placed | None]]
        ] = OrderedDict()
        #: The camera inputs the latest read was keyed by: inputs observed
        #: anew but the same (:meth:`~.CameraExtras.same`) are keyed as these.
        self._extras: CameraExtras | None = None

    def __len__(self) -> int:
        """How many leaf versions' lays are kept."""
        return len(self._lays)

    @property
    def cameras(self) -> int:
        """How many camera keys' places are kept."""
        return len(self._placed)

    def viewing(
        self, record: Record, capture: Capture, camera: Any, view: View
    ) -> Viewing | None:
        """The memo for a read of ``capture`` (shown) through ``camera`` and
        ``view``, or ``None`` when nothing read there may be kept.
        """
        if record.camera_replaced or capture.scene_updaters:
            return None
        extras = capture.camera
        if extras is not None and (extras.displays is not None or extras.sub_cameras):
            return None
        last = self._extras
        if extras is not None and last is not None and last.same(extras):
            # The same inputs, observed anew (a play's first capture).
            extras = last
        self._extras = extras
        states = []
        for mob in camera_nodes(camera):
            state = record.node_state(mob, capture)
            if state is None:
                return None
            states.append(state)
        key = (id(extras), tuple(map(id, states)), view)
        found = self._placed.get(key)
        if found is None:
            found = ((extras, tuple(states)), {})
            self._placed[key] = found
            while len(self._placed) > CAMERAS_KEPT:
                self._placed.popitem(last=False)
        else:
            self._placed.move_to_end(key)
        return Viewing(self, record, capture, camera, found[1])


class Viewing:
    """The memo as one read sees it: the capture it shows, through its
    camera key.
    """

    def __init__(
        self,
        memo: LeafMemo,
        record: Record,
        capture: Capture,
        camera: Any,
        placed: dict[_Key, Placed | None],
    ) -> None:
        self._memo = memo
        self._record = record
        self._capture = capture
        self._camera = camera
        self._placed = placed
        self._rotated = False

    def node(self, mob: Mobject, chain: tuple[Mobject, ...]) -> Node | None:
        """``mob`` (held by ``chain``) as its version at the capture, or
        ``None`` when its reads may not be kept: it or a mobject holding it
        has no version there, or has updaters.
        """
        record, capture = self._record, self._capture
        state = record.node_state(mob, capture)
        if state is None or state.updaters:
            return None
        for holder in chain:
            held = record.node_state(holder, capture)
            if held is None or held.updaters:
                return None
        return mob, state

    def lays(self, node: Node, read: Callable[[], Lays]) -> Lays:
        """What ``node`` lays: kept, or ``read`` once."""
        mob, state = node
        key = (id(mob), id(state))
        found = self._memo._lays.get(key)
        if found is None:
            found = (mob, state, read())
            self._memo._lays[key] = found
        lays: Lays = found[2]
        return lays

    def placed(self, node: Node, read: Callable[[], Placed | None]) -> Placed | None:
        """Where ``node`` is placed under this camera key: kept, or ``read``
        once. Its :meth:`lays` are read first, which holds what the key names.

        A projection leaves a ThreeDCamera's rotation made for the angles it
        read through, and the next read's depth sort reads that rotation. So a
        read served from here makes it as reading the place would have.
        """
        mob, state = node
        key = (id(mob), id(state))
        assert key in self._memo._lays, "a place kept before its lays"
        found = self._placed.get(key, _ABSENT)
        if found is _ABSENT:
            found = read()
            self._placed[key] = found
        elif not self._rotated:
            hold_rotation(self._camera)
        self._rotated = True
        placed: Placed | None = found
        return placed
