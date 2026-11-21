"""A rig: an asset's parts as named vector objects in a scene, moved through
the parameters its maker tested.

``Rig(ref)`` loads the asset (:func:`~.reader.load_asset`) and holds one
group per part and per slot, in drawing order (the manifest's ``order``,
then its order of entries), each the part's shapes as ordinary
:class:`~arachne.mobject.types.vectorized_mobject.VMobject` objects: crisp
at any zoom, styled and transformed as any other. A slot's group holds one
group per drawing; the drawing it shows has its points, and every member of
the others holds none, so nothing draws them, while a style given to the
rig still reaches them. A file's stroke widths are in canvas units, drawn at
the manifest's canvas scale as its points are (a unit of stroke width draws
:data:`STROKE_SCALE` scene units, the fixtures' scale); scaling a placed rig
leaves them as scaling any vector object does.

Three things are kept apart, each with one writer:

- the parameter values and slot choices, immutable values replaced on
  write, written by :meth:`Rig.pose`, :meth:`Rig.show` and the animation
  :class:`~.motion.Pose`;
- where the rig is placed, ``F``, the affine map from the canvas to the
  scene: its **carrier**, a first child of three points with no paint, the
  images under ``F`` of three canvas points 10⁻⁴ scene units apart at the
  asset's size (:class:`~.pose.Model`). Whatever moves the rig as a whole
  moves it, and nothing that moves one part does;
- the art, every part's points, written by :meth:`Rig.settle` alone, from
  rest (:mod:`.pose`).

An animation pairs a rig with its copies member by member, by the members
that hold points. A copy shows what its rig shows: a copy aligned with the
rig for an animation takes its slot choices (:meth:`Rig.align_data`), and a
:class:`~.motion.Pose` that switches a slot switches it in the copies the
play's animations already hold. So a whole-rig animation and a slot switch
in one play pair each shape with its own copy, in either order.

A raster member (a part or a slot's drawing that is a still) is held as the
quad of its box (:class:`~.raster.RasterPart`), a vector object like any
other, whose picture is drawn at its corners. A part or slot with a depth is
a layer of its own at that depth (:func:`~arachne.compose.set_layer`), so
the camera's focus blurs it by its depth; and once the rig is viewed
through a moving camera (:meth:`Rig.view_through`), each such member slides
by its parallax as the camera moves (:func:`~.pose.parallax`): the scene's
media driver settles the rig at every instant, after the animations, so the
art follows the camera through plays that animate the rig too, and an
anchor's point is computed for the camera as it is when asked.

A rig settles at once when it is posed or shown, from its own updater
(:func:`keep_settled`: after a play, and after the scene's code moved it),
and during a play that poses it from the scene's settle (:mod:`.motion`). A
part moved directly returns to its posed place at the next settle: parts
move through parameters. A rig scaled to nothing has no ``F`` and is not
settled until it has a size again.

A rig declares its values (:mod:`arachne.quantities`), read from the values
a capture holds: ``rig.param.<parameter>`` for each parameter, in its
manifest's unit, and ``rig.slot.<slot>`` for each slot, the name of the
drawing it shows (:func:`~.reader.load_asset` registers the units). On a rig held to stepped timing (:mod:`.timing`) they are
the values of the drawing on display.
"""

from __future__ import annotations

__all__ = ["CarrierPoint", "Rig", "keep_settled"]

import difflib
import math
import os
import weakref
from typing import TYPE_CHECKING, Any, ClassVar, NoReturn

import numpy as np

from .. import quantities
from ..mobject.types.vectorized_mobject import VectorizedPoint, VGroup, VMobject
from ..mobject.values import FrozenArray, FrozenPoints, adopt
from ..quantities import Quantity
from ..utils.paths import straight_path
from .manifest import QUANTITY_PREFIX as PREFIX
from .manifest import PartSpec, refuse
from .pose import (
    CanvasMap,
    Model,
    RigValues,
    model_of,
    parallax,
    posed_with,
    transforms,
)
from .reader import Asset, load_asset

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from typing import Self

    from ..camera.camera import Camera
    from ..mobject.mobject import Mobject
    from ..quantities import At
    from .anchor import Anchor

#: The points of a shape that draws nothing.
_EMPTY = adopt(np.zeros((0, 3)), FrozenPoints)
#: The scene units a unit of stroke width draws (the camera's line width
#: multiple): the canvas scale at which a file's stroke width, in canvas
#: units, is the engine's stroke width as it stands, the fixtures' scale.
STROKE_SCALE = 0.01


class ViewThrough(FrozenArray):
    """How a rig is viewed through a moving camera: the camera frame's
    centre when the view began (its home, three numbers), then the viewing
    distance.
    """

    value_name = "A rig's view through a camera"
    replacement = "view the rig again with rig.view_through(camera, distance=...)"


class CarrierPoint(VectorizedPoint):
    """One of a carrier's three points: moved with its rig, drawn by
    nothing, and no part of any box or measure of a family that holds it
    (its bounds, extremes and anchors are none), so a rig's box, its size
    and where it stands are its art's.
    """

    #: Passed over by the boxes the lint and the state read of a family.
    _scaffold = True

    def _point_bounds(self, anchors_every: int | None = None) -> Any:
        return None

    def reduce_across_dimension(self, reduce_func: Any, dim: int) -> float | None:
        return None

    def get_merged_array(self, array_attr: str) -> np.ndarray:
        if array_attr == "points":
            return np.zeros((0, self.dim))
        return super().get_merged_array(array_attr)

    def get_anchors(self) -> np.ndarray:
        return np.zeros((0, self.dim))

    def interpolate(
        self,
        mobject1: Mobject,
        mobject2: Mobject,
        alpha: float,
        path_func: Any = straight_path(),
    ) -> Self:
        super().interpolate(mobject1, mobject2, alpha, path_func)
        # A carrier's first point runs between its rig's copies: so do the
        # paints of the drawings its rig hides, which no animation pairs.
        rig, start, end = _rig_of(self), _rig_of(mobject1), _rig_of(mobject2)
        if rig is not None and start is not None and end is not None:
            rig._interpolate_hidden(start, end, alpha)
        return self


#: The rig of each rig's carrier's first point, made or copied, held weakly.
_CARRIED: weakref.WeakKeyDictionary[Mobject, weakref.ref[Rig]] = (
    weakref.WeakKeyDictionary()
)


def _rig_of(point: Mobject) -> Rig | None:
    held = _CARRIED.get(point)
    return None if held is None else held()


def _carry(rig: Rig) -> None:
    _CARRIED[rig.carrier.submobjects[0]] = weakref.ref(rig)


def keep_settled(rig: Mobject) -> None:
    """A rig's own updater: settle it (:meth:`Rig.settle`). It runs after a
    play's animations are done, and wherever the mobjects' updaters run; it
    takes no ``dt``, so it ticks no wait.
    """
    assert isinstance(rig, Rig)
    rig.settle()


class Rig(VGroup):
    """The asset ``asset`` (an :class:`~.reader.Asset`, or a reference
    :func:`~.reader.load_asset` takes) at its default pose, placed as its
    manifest's ``size`` places it. ``vmobject_kwargs`` go to the group, as a
    ``VGroup``'s do.
    """

    #: What a content key leaves out: the frame of the camera the rig is
    #: viewed through, a handle on another object, not content (the view
    #: itself, its home and distance, is keyed); and the asset and its model,
    #: which ``_asset_digest`` stands for (the asset's directory is where it
    #: was read, not what it is).
    _content_key_omits: ClassVar[frozenset[str]] = frozenset(
        {"_view_frame", "asset", "model"}
    )

    def __init__(
        self, asset: Asset | str | os.PathLike[str], **vmobject_kwargs: Any
    ) -> None:
        self._make(load_asset(asset), **vmobject_kwargs)

    def _make(self, asset: Asset, **vmobject_kwargs: Any) -> None:
        """Build the rig of ``asset``, read whole, whatever its admission."""
        model = model_of(asset)
        super().__init__(**vmobject_kwargs)
        self.asset = asset
        self.model = model
        # The asset as content keys know it, by its digest, as a raster
        # member's is: ``asset`` and ``model`` are omitted by declaration
        # (``_content_key_omits``), the digest standing for both, so two rigs
        # whose assets differ only where their points and values show nothing
        # (a drive's unit) key apart, and one asset read from two directories
        # keys alike.
        self._asset_digest: str = asset.digest
        manifest = asset.manifest
        self.carrier = VGroup(
            *(CarrierPoint(point) for point in model.natural.carrier(model)),
            name="carrier",
        )
        entries: list[tuple[int, int, str, VGroup]] = [
            (part.order, i, part.name, VGroup(*asset.rest(part.name), name=part.name))
            for i, part in enumerate(manifest.parts)
        ]
        entries += [
            (
                slot.order,
                len(manifest.parts) + i,
                slot.name,
                VGroup(
                    *(
                        VGroup(*asset.drawing(slot.name, d.name), name=d.name)
                        for d in slot.drawings
                    ),
                    name=slot.name,
                ),
            )
            for i, slot in enumerate(manifest.slots)
        ]
        members = [
            (name, group)
            for *_, name, group in sorted(entries, key=lambda entry: entry[:2])
        ]
        self._members = tuple(members)
        # A file's stroke widths are canvas units, scaled to the scene as its
        # points are; a shape with no stroke (a still's quad) is left as made.
        widen = manifest.size.scene_units_per_unit / STROKE_SCALE
        if widen != 1:
            for shape in self._shapes():
                width = shape.get_stroke_width()
                background = shape.get_stroke_width(background=True)
                if width:
                    shape.set_stroke(width=width * widen, family=False)
                if background:
                    shape.set_stroke(
                        width=background * widen, background=True, family=False
                    )
        groups = [group for _, group in members]
        self.add(self.carrier, *groups)
        # What a settle needs whole: each group the rig made, with its
        # children as made.
        holders: list[VMobject] = [self.carrier, *groups]
        for name, group in members:
            if not isinstance(model.placed[name], PartSpec):
                holders += group.submobjects
        self._skeleton = tuple(
            (holder, tuple(holder.submobjects)) for holder in holders
        )
        if model.depths:
            from ..compose import LayerSpec, set_layer

            for name, group in members:
                depth = model.depths.get(name)
                if depth is not None:
                    set_layer(group, LayerSpec(depth=depth))
        self._values = _values(model.defaults)
        self._shown = ",".join(slot.default for slot in manifest.slots)
        self._settled_from: tuple[Any, ...] | None = None
        self.add_updater(keep_settled)
        _carry(self)
        self.settle()

    def __deepcopy__(self, clone_from_id: dict[int, Any]) -> Self:
        frame = vars(self).get("_view_frame")
        if frame is not None:
            # A copy is viewed through the same camera, never a copy of its
            # frame, which would stand still.
            clone_from_id.setdefault(id(frame), frame)
        copied = super().__deepcopy__(clone_from_id)
        _carry(copied)
        return copied

    # Names

    def part(self, name: str) -> VMobject:
        """The group of the part or slot ``name``; ``rig.name_unknown`` for
        a name the asset's manifest does not have.
        """
        for found, group in self._members:
            if found == name:
                return group
        self._unknown("part", name, [found for found, _ in self._members])

    def _unknown(self, kind: str, name: str, names: list[str]) -> NoReturn:
        nearest = difflib.get_close_matches(name, names, n=3, cutoff=0.6)
        hint = f", and the nearest is {' or '.join(nearest)}" if nearest else ""
        what = "part or slot" if kind == "part" else kind
        refuse(
            "rig.name_unknown",
            f"the asset {self.asset.name} has no {what} {name!r}: it has"
            f" {', '.join(names) or 'none'}{hint}",
            asset=self.asset.name,
            kind=kind,
            name=name,
            nearest=nearest,
        )

    # Values

    def value(self, parameter: str) -> float:
        """The value of ``parameter`` now."""
        return float(self._values[self._index(parameter)])

    def shown(self, slot: str) -> str:
        """The drawing the slot ``slot`` shows now."""
        return self._choices()[self._slot(slot)]

    def _index(self, parameter: str) -> int:
        index = self.model.index.get(parameter)
        if index is None:
            self._unknown("parameter", parameter, list(self.model.index))
        return index

    def _slot(self, slot: str) -> int:
        names = [s.name for s in self.asset.manifest.slots]
        if slot not in names:
            self._unknown("slot", slot, names)
        return names.index(slot)

    def _choices(self) -> list[str]:
        return self._shown.split(",") if self._shown else []

    def checked_values(
        self, values: dict[str, Any], *, eased: bool = False
    ) -> dict[int, float]:
        """``values``, parameter names to values, as values by parameter
        index, each in its tested range: ``rig.name_unknown`` for a name the
        manifest does not have, ``rig.range`` for a value outside the range
        (``eased``: one a pose's rate function carried there).
        """
        checked: dict[int, float] = {}
        for name, given in values.items():
            index = self._index(name)
            spec = self.model.parameters[index]
            value = float(given)
            if not (math.isfinite(value) and spec.min <= value <= spec.max):
                how = (
                    f"carried to {value:g} by its pose's rate function"
                    if eased
                    else f"given {given!r}"
                )
                refuse(
                    "rig.range",
                    f"the parameter {name} of the asset {self.asset.name} is {how},"
                    f" outside its tested range from {spec.min:g} to {spec.max:g}",
                    asset=self.asset.name,
                    parameter=name,
                    value=value if math.isfinite(value) else repr(value),
                    min=spec.min,
                    max=spec.max,
                )
            checked[index] = value
        return checked

    def checked_drawings(self, drawings: dict[str, Any]) -> dict[int, str]:
        """``drawings``, slot names to drawing names, by slot index:
        ``rig.name_unknown`` for a slot or a drawing the manifest does not
        have.
        """
        checked: dict[int, str] = {}
        for slot, drawing in drawings.items():
            index = self._slot(slot)
            names = [d.name for d in self.asset.manifest.slots[index].drawings]
            if drawing not in names:
                self._unknown("drawing", str(drawing), names)
            checked[index] = drawing
        return checked

    def pose(self, /, **values: float) -> Self:
        """Set the parameters ``values`` (names to values) and settle."""
        self.write_values(self.checked_values(values))
        return self.settle()

    def show(self, /, **drawings: str) -> Self:  # type: ignore[override]
        """Have each slot named in ``drawings`` show the drawing given, and
        settle.
        """
        self.write_choices(self.checked_drawings(drawings))
        return self.settle()

    def write_values(self, values: dict[int, float]) -> None:
        """Replace the values at the indices of ``values`` (checked, as
        :meth:`checked_values` gives them); nothing is settled.
        """
        if values:
            updated = np.array(self._values, dtype=float)
            for index, value in values.items():
                updated[index] = value
            self._values = _values(updated)

    def write_choices(self, drawings: dict[int, str]) -> None:
        """Replace the choices of the slots at the indices of ``drawings``
        (checked, as :meth:`checked_drawings` gives them); nothing is
        settled.
        """
        if drawings:
            choices = self._choices()
            for index, drawing in drawings.items():
                choices[index] = drawing
            self._shown = ",".join(choices)

    # Anchors

    def anchor(self, name: str) -> Anchor:
        """The anchor ``name`` of the asset: the point of its part it names,
        followed wherever the part goes (:class:`~.anchor.Anchor`). A name
        the manifest does not have is found by the lint
        (``annotation.anchor_unknown``) and refuses drawing; it gives an
        anchor with no point.
        """
        from .anchor import Anchor, attachments

        attachment = attachments(self.model).get(name)
        if attachment is None:
            from ..lint.raised import found

            names = list(self.asset.anchors)
            nearest = difflib.get_close_matches(name, names, n=3, cutoff=0.6)
            hint = f", and the nearest is {' or '.join(nearest)}" if nearest else ""
            found(
                "annotation.anchor_unknown",
                f"the asset {self.asset.name} has no anchor {name!r}: it has"
                f" {', '.join(names) or 'none'}{hint}",
                detail={"asset": self.asset.name, "name": name, "nearest": nearest},
                objects=(self,),
            )
        return Anchor(self, attachment, name)

    # Viewing through a camera

    def view_through(self, camera: Camera, *, distance: float | None = None) -> Self:
        """View the rig through ``camera``, the running scene's camera, whose
        frame moves (a ``MovingCameraScene``'s ``self.camera``): from now on
        each part or slot with a depth ``d`` slides by ``(1 - D / (D + d))``
        of the camera frame's move from where it is now, its home, with ``D``
        the viewing distance ``distance`` (the frame's height now, unless
        given). The scene's media driver settles the rig at every instant
        from now on, so the scene's plays are evaluated at every tick. Any
        other camera is a ``ValueError``: the scene records its own camera's
        frame, and no other.

        ``D`` must be finite and above 0, and ``D + d`` above 0 for every
        depth: a ``ValueError`` otherwise, checked at each call. Called
        again, it sets a new home and distance.
        """
        from ..camera.moving_camera import MovingCamera
        from ..mobject.media.clock import hold_media
        from ..timeline.recorder import running

        if not isinstance(camera, MovingCamera):
            raise ValueError(
                f"A rig is viewed through a camera whose frame moves, and"
                f" {type(camera).__name__} has none: make the scene a"
                " MovingCameraScene and view the rig through self.camera."
            )
        seen = float(camera.frame.height) if distance is None else distance
        if isinstance(seen, bool) or not isinstance(seen, (int, float)):
            raise TypeError(f"A viewing distance is a number, not {distance!r}.")
        seen = float(seen)
        if not (math.isfinite(seen) and seen > 0):
            raise ValueError(
                f"A viewing distance is a finite length above 0, not {distance!r}."
            )
        for name, depth in self.model.depths.items():
            if not seen + depth > 0:
                raise ValueError(
                    f"The part {name!r} of the asset {self.asset.name} is at the"
                    f" depth {depth:g}, which a viewing distance of {seen:g} puts"
                    f" at or behind the eye: D + d must be above 0, so view it from"
                    f" farther than {-depth:g}."
                )
        recorder = running()
        if recorder is None:
            raise RuntimeError(
                "view_through follows the camera at every instant of the running"
                " scene: call it inside the scene's construct"
            )
        if camera is not getattr(recorder.scene, "camera", None):
            raise ValueError(
                "A rig is viewed through the running scene's own camera, whose"
                " frame the scene records: view it through self.camera, not"
                " another camera."
            )
        home = np.asarray(camera.frame.get_center(), dtype=float)
        self._view_frame = camera.frame
        self._view = adopt(np.array([*home, seen], dtype=float), ViewThrough)
        hold_media(recorder.scene, self)
        return self.settle()

    def _parallax(self, name: str) -> np.ndarray:
        """How far the part or slot ``name`` stands, now, from where the
        rig's placement puts it: its parallax under the camera it is viewed
        through (:func:`~.pose.parallax`), 0 for a rig not viewed or a
        member with no depth.
        """
        depth = self.model.depths.get(name)
        view = vars(self).get("_view")
        if depth is None or view is None:
            return np.zeros(3)
        centre = self._view_frame.get_center()
        return parallax(depth, float(view[3]), centre, view[:3])

    def _viewed_from(self) -> tuple[float, ...] | None:
        """What the parallax is computed from now: the camera frame's
        centre, the home and the distance; ``None`` for a rig not viewed.
        """
        view = vars(self).get("_view")
        if view is None:
            return None
        return (*map(float, self._view_frame.get_center()), *map(float, view))

    # Placement and art

    def placement(self) -> CanvasMap | None:
        """``F``, where the rig is placed, read from its carrier; ``None``
        for a rig scaled to nothing.
        """
        return CanvasMap.from_carrier(
            self.model, [point.points[0] for point in self.carrier.submobjects]
        )

    def place(self, placement: CanvasMap) -> Self:
        """Put the rig where ``placement`` places its canvas, and settle."""
        for point, at in zip(
            self.carrier.submobjects, placement.carrier(self.model), strict=True
        ):
            point.points = at[None]
        return self.settle()

    def settle(self) -> Self:
        """Compute the art from rest, the values, the slots' choices and the
        carrier (:mod:`.pose`). Nothing is done when those, and the art,
        are what the art was last settled from, or when the rig has no
        placement. A rig changed into something its asset does not draw (a
        transform into another shape) raises :class:`ValueError`.
        """
        self._whole()
        carrier = tuple(point.points for point in self.carrier.submobjects)
        shapes = self._shapes()
        last = self._settled_from
        viewed = self._viewed_from()
        if (
            last is not None
            and last[0] is self._values
            and last[1] == self._shown
            and _same(last[2], carrier)
            and _same(last[3], tuple(shape.points for shape in shapes))
            and last[4:] == (viewed,)
        ):
            return self
        placement = self.placement()
        if placement is None:
            return self
        model, values = self.model, np.asarray(self._values)
        affines = transforms(model, values)
        choices = dict(
            zip(
                (slot.name for slot in self.asset.manifest.slots),
                self._choices(),
                strict=True,
            )
        )
        for name, group in self._members:
            placed = placement
            if viewed is not None and name in model.depths:
                placed = placement.shifted(self._parallax(name))
            if isinstance(model.placed[name], PartSpec):
                posed = posed_with(model, name, values, affines[name])
                _write(group.submobjects, posed, placed)
                continue
            for drawing in group.submobjects:
                if drawing.name == choices[name]:
                    posed = posed_with(model, name, values, affines[name], drawing.name)
                    _write(drawing.submobjects, posed, placed)
                else:
                    for shape in drawing.submobjects:
                        shape.points = _EMPTY
        self._settled_from = (
            self._values,
            self._shown,
            carrier,
            tuple(shape.points for shape in shapes),
            viewed,
        )
        return self

    # Pairing with copies

    def align_data(self, mobject: Mobject, skip_point_alignment: bool = False) -> Self:
        """Align with ``mobject`` for an animation between them, as any
        group is. A copy of the rig (a rig of its asset, both whole) first
        takes the rig's slot choices: which drawing a slot shows is the
        rig's to say, through :meth:`show` and :class:`~.motion.Pose`, never
        an animation's, so each shape is paired with its own copy and keeps
        its own paint.
        """
        if (
            isinstance(mobject, Rig)
            and mobject.model is self.model
            and mobject._shown != self._shown
            and self._intact()
            and mobject._intact()
        ):
            mobject.adopt_choices(self)
        return super().align_data(mobject, skip_point_alignment)

    def _interpolate_hidden(self, start: Rig, end: Rig, alpha: float) -> None:
        """Give each shape this rig hides the paint between its own in the
        copies ``start`` and ``end``, which hide it too, at ``alpha``: an
        animation pairs only shapes with points, so a drawing switched in
        later shows the paint the animation left on the rig.
        """
        if not (
            start.model is self.model
            and end.model is self.model
            and self._intact()
            and start._intact()
            and end._intact()
        ):
            return
        for shape, first, last in zip(
            self._shapes(), start._shapes(), end._shapes(), strict=True
        ):
            if not (
                shape.get_num_points()
                or first.get_num_points()
                or last.get_num_points()
            ):
                shape.interpolate_color(first, last, alpha)

    def adopt_choices(self, rig: Rig) -> None:
        """Show what ``rig``, a rig of the same asset, shows, and settle."""
        self._shown = rig._shown
        self.settle()

    def _shapes(self) -> list[VMobject]:
        """Every shape of the art, the parts' and every drawing's."""
        found: list[VMobject] = []
        for name, group in self._members:
            if isinstance(self.model.placed[name], PartSpec):
                found += group.submobjects
            else:
                for drawing in group.submobjects:
                    found += drawing.submobjects
        return found

    def _intact(self) -> bool:
        """Whether the rig holds its carrier and parts as it made them."""
        return (
            self.carrier in self.submobjects
            and all(group in self.submobjects for _, group in self._members)
            and all(
                _same(tuple(holder.submobjects), children)
                for holder, children in self._skeleton
            )
            and all(len(point.points) == 1 for point in self.carrier.submobjects)
        )

    def _whole(self) -> None:
        """Raise unless the rig holds its carrier and parts as it made them."""
        if not self._intact():
            raise ValueError(
                f"A rig of the asset {self.asset.name} was changed into something its"
                " asset does not draw (its carrier or its parts are not as the rig"
                " made them, as a transform into another shape leaves them), so it"
                " can't be posed or settled. Transform a copy, or morph two rigs"
                " part by part with MorphParts."
            )


def unadmitted_rig(asset: Asset) -> Rig:
    """A rig of ``asset`` that loading has not admitted: the admission tool's
    own, which draws an asset to decide its admission. Nothing else makes
    one, and nothing public reaches it.
    """
    rig = Rig.__new__(Rig)
    rig._make(asset)
    return rig


def _values(values: np.ndarray) -> RigValues:
    return adopt(np.array(values, dtype=float), RigValues)


def _same(a: tuple[Any, ...], b: tuple[Any, ...]) -> bool:
    return len(a) == len(b) and all(x is y for x, y in zip(a, b, strict=True))


def _write(
    shapes: list[VMobject], posed: tuple[np.ndarray, ...], placement: CanvasMap
) -> None:
    for shape, points in zip(shapes, posed, strict=True):
        shape.points = adopt(placement(points), FrozenPoints)


@quantities.provider(Rig)
def _declared(held: Mapping[str, Any], at: At) -> Iterable[Quantity]:
    """A rig's parameter values and slot choices, as ``held`` holds them."""
    model: Model = held["model"]
    found = [
        Quantity(f"{PREFIX}param.{spec.name}", float(value), spec.unit)
        for spec, value in zip(model.parameters, held["_values"], strict=True)
    ]
    shown: str = held["_shown"]
    slots = held["asset"].manifest.slots
    found += [
        Quantity(f"{PREFIX}slot.{slot.name}", drawing, "name")
        for slot, drawing in zip(slots, shown.split(",") if shown else (), strict=True)
    ]
    return found
