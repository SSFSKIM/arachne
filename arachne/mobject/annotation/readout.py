"""A running number: a declared value shown in a declared format, and the
animation that counts it to a target.

A :class:`Readout` holds its value as a plain number and its label as a
text of the value in its format (a Python format string, ``"{:.1f} mV"``),
rebuilt whenever the value changes. :class:`CountUp` runs the value of a
readout, or of a gauge (:mod:`.gauge`), to a target: at ``alpha`` the
declared value is ``start + (to - start) * rate_func(alpha)``, and once the
play is done it is the target itself, whatever the rate function's last
value.

What a viewer sees of it is three statements about time, never one: the
text ``state_at(t)`` answers is the format of the value at its ``as_of``
(the last frame of the record's grid at or before ``t``); a delivered frame
draws the format of the declared value at its own instant; and the exact
target is in a delivered frame only when a frame instant at or after the
play's end exists, so a count followed by a wait shows it.

As its text widens or narrows, a readout keeps one point still, its
*hold*, so what the reader has found does not move under the eye. The full
rule, on both axes, is :class:`Readout`'s.

A readout declares its value as ``annotation.count``
(:mod:`arachne.quantities`), read from the value a capture holds: the
number itself, not its text.
"""

from __future__ import annotations

__all__ = ["CountUp", "Readout"]

import math
import string
from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from arachne import quantities
from arachne.animation.animation import Animation
from arachne.constants import LEFT, ORIGIN, RIGHT
from arachne.mobject.text.text_mobject import Text
from arachne.mobject.types.vectorized_mobject import VMobject
from arachne.mobject.values import FrozenPoints, freeze
from arachne.quantities import At, Quantity
from arachne.utils.color import WHITE, ParsableManimColor

from .attach import DEFAULT_OFFSET, Annotation
from .callout import LabelPill

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Self

    import numpy as np

    from arachne.assets.anchor import Anchor
    from arachne.mobject.mobject import Mobject


def _finite(value: Any, what: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{what} is a finite number, not {value!r}.")
    return number


def checked_format(pattern: str) -> str:
    """``pattern`` when it is a Python format string with exactly one
    replacement field, for the number (``"{:.1f} mV"``); :class:`ValueError`
    for none (a label that never changes), for more than one, for a named or
    nested field, or for a format spec no number takes.
    """
    pattern = str(pattern)
    try:
        fields = [
            (name, spec)
            for _, name, spec, _ in string.Formatter().parse(pattern)
            if name is not None
        ]
    except ValueError as error:
        fields, broken = [], str(error)
    else:
        broken = ""
    if (
        broken
        or len(fields) != 1
        or fields[0][0] not in ("", "0")
        or "{" in (fields[0][1] or "")
    ):
        found = broken or f"it holds {len(fields)} fields"
        raise ValueError(
            "A readout's format is a Python format string with exactly one field"
            f" for its number, as '{{:.1f}} mV'; {pattern!r} is not ({found})."
        )
    try:
        pattern.format(0.0)
    except (IndexError, KeyError, ValueError, TypeError) as error:
        raise ValueError(
            "A readout's format is a Python format string with exactly one field"
            f" for its number, as '{{:.1f}} mV'; {pattern!r} is not ({error})."
        ) from None
    return pattern


def _leads_with_text(pattern: str) -> bool:
    """Whether ``pattern`` shows something before its number."""
    literal = next(iter(string.Formatter().parse(pattern)))[0]
    return bool(literal.strip())


def _fixed_glyph(pattern: str) -> int | None:
    """The index, among a text's glyphs, of one its format shows in every
    text: the first when it shows something before the number, the last
    when only after it; ``None`` for a bare number.
    """
    if _leads_with_text(pattern):
        return 0
    parts = list(string.Formatter().parse(pattern))
    trailing = "".join(literal for literal, *_ in parts[1:])
    return -1 if trailing.strip() else None


def _glyphs(label: Text) -> list[Any]:
    """``label``'s glyphs, without its plate."""
    plates = _plate_ids(label)
    return [m for m in label.family_members_with_points() if id(m) not in plates]


def _hold_for(
    pattern: str,
    at: Anchor | None,
    offset: Sequence[float] | np.ndarray,
    hold: Sequence[float] | np.ndarray | None,
) -> Sequence[float] | np.ndarray:
    """The point of a readout's box that stays still as its text changes
    (:class:`Readout`'s rule): ``hold`` when given.
    """
    if hold is not None:
        return hold
    if at is not None:
        across = float(np.asarray(offset, dtype=float).ravel()[0])
        return ORIGIN if abs(across) < 1e-9 else (LEFT if across > 0 else RIGHT)
    return LEFT if _leads_with_text(pattern) else ORIGIN


class Readout(Annotation):
    """The number ``value`` shown in ``format``, optionally attached to an
    anchor ``at`` plus ``offset`` (following it unless ``follow`` is false,
    with a leader line with ``leader``), its text on a pill with ``pill``.

    Its value is :attr:`value`; :meth:`set_value` changes it and rebuilds
    the text, and :class:`CountUp` runs it to a target.

    As the text changes it keeps one point still, its :attr:`hold`, one of
    the nine points of its box (:func:`~.attach.checked_hold`; ``ValueError``
    for anything else). Across, the hold's first coordinate picks the box's
    left edge (-1), centre (0) or right edge (1):

    - attached to an anchor, by default the edge facing the anchor: an
      offset with a part to the right holds the left edge, one to the left
      the right edge, one straight up or down (no part across) the centre.
      The text grows away from what it labels, never over it;
    - placed by hand (``move_to``, ``next_to``), by default its start (the
      left edge) when its format shows something before the number
      (``"swell {:.0f}%"``), which the reader has already read; otherwise
      (a bare number, a number before its unit) its centre, where
      ``move_to`` put it;
    - ``hold`` given names it outright, for a placement the readout cannot
      see: a number right-aligned in a column holds ``RIGHT``.

    Up and down, a hold with no part up or down (every default) keeps the
    baseline: the bottom of a glyph the format shows in every text (the
    first of what comes before the number, else the last of what comes
    after it), which digits of other heights never move; a bare number,
    which has no such glyph, keeps its box's middle. A hold with a part up
    or down (``UP``, ``DL``) keeps that edge of the box, the top or the
    bottom, for any format.

    Attached and following, the held point keeps where it stood from the
    anchor's point plus the offset as the anchor moves, so the gap across
    from the anchor to the held edge stays as set. With ``follow`` false
    the readout is placed once and the held point stays where it was put,
    in the scene: the gap stays only while the anchor does not move. A
    leader, which runs from the box's edge toward the anchor's point, keeps
    its length only if neither the box's width nor, unfollowed, the anchor
    changes; as the box widens it changes unless it runs straight across. :meth:`held_point` reads the held point of a text; a subclass
    that overrides it keeps its signature.
    """

    def __init__(
        self,
        value: float = 0,
        format: str = "{:.1f}",  # noqa: A002 (CE's and the format's own word)
        at: Anchor | None = None,
        *,
        offset: Sequence[float] | np.ndarray = DEFAULT_OFFSET,
        follow: bool = True,
        leader: bool = False,
        font_size: float = 36,
        color: ParsableManimColor = WHITE,
        pill: bool = False,
        hold: Sequence[float] | np.ndarray | None = None,
        name: str | None = None,
    ) -> None:
        number = _finite(value, "A readout's value")
        self.format = checked_format(format)
        self._label_class: type[Text] = LabelPill if pill else Text
        #: The text of the value in its format.
        self.label = self._label_class(
            self.format.format(number), font_size=font_size, color=color
        )
        self._value = number
        #: The points the label's text drew as made, upright, before
        #: anything moved it: what :meth:`set_value` reads its turn from.
        self._text_as_made = _drawn_points(self.label)
        super().__init__(
            self.label,
            at=at,
            offset=offset,
            follow=follow,
            leader=leader,
            leader_color=color,
            hold=_hold_for(self.format, at, offset, hold),
            name=name,
        )

    def held_point(self, body: Mobject) -> np.ndarray:
        """Where the text ``body`` (the label, or a text about to replace
        it) puts the point the readout holds (the class's rule): across,
        the point :attr:`hold` of its box; up and down, the baseline or the
        edge the hold names.
        """
        point = np.array(body.get_critical_point(self.hold), dtype=float)
        fixed = _fixed_glyph(self.format)
        glyphs = _glyphs(body) if isinstance(body, Text) else []
        if self.hold[1] == 0 and fixed is not None and glyphs:
            point[1] = glyphs[fixed].get_bottom()[1]
        return point

    @property
    def value(self) -> float:
        """The declared value now."""
        return self._value

    def set_value(self, value: float) -> Self:
        """Declare ``value`` and show it in the format, in place of the
        text shown: only the text changes, at the label's size and turned
        as it is (rotated or flipped as a whole, the new text is too) with
        its :attr:`hold` point where it was, its glyphs styled as its first
        glyph was (fill, stroke, opacity, depth) and its pill as its pill
        was. A label reshaped otherwise than by one linear map is shown
        upright, at the size its height reads.
        """
        number = _finite(value, "A readout's value")
        said = self.format.format(number)
        self._value = number
        label = self.label
        if said != label.original_text:
            hidden = self._unplaced
            if hidden:
                # Waiting for its anchor: rebuilt from the points it will draw.
                self._show_once_placed()
            turn = self._turn(label)
            if turn is None:
                made = self._label_class(said, font_size=label.font_size)
                upright = _drawn_points(made)
            else:
                made = self._label_class(said, font_size=label._font_size)
                upright = _drawn_points(made)
                made.apply_matrix(turn)
            held = self.held_point(label)
            made.shift(held - self.held_point(made))
            self.note_held(made, held)
            _restyle(made, label)
            # A play draws its moving mobjects as listed when it began (CE's
            # moving list), so the glyphs replaced draw nothing from now on,
            # as DecimalNumber.set_value leaves its own.
            for replaced in label.get_family()[1:]:
                replaced.points = np.zeros((0, 3))
            # The label keeps its identity and its own attributes, so the
            # record follows one text: only what the string makes is new.
            for key in _TEXT_DATA:
                if key in made.__dict__:
                    label.__dict__[key] = made.__dict__[key]
            self._text_as_made = upright
            if hidden:
                self._hide_until_placed()
        return self

    def _turn(self, label: Text) -> np.ndarray | None:
        """The linear map (3 x 3) that took ``label``'s text, as made
        upright, to how it stands now, when that map turns or flips it;
        ``None`` when it stands upright (moved and scaled alone) or when no
        one map takes the text as made to its points now.
        """
        made, now = np.asarray(self._text_as_made), _drawn_points(label)
        if made.shape != now.shape or len(made) < 3:
            return None
        design = np.column_stack([made[:, :2], np.ones(len(made))])
        fit, *_ = np.linalg.lstsq(design, now, rcond=None)
        size = max(float(np.ptp(now, axis=0).max()), 1e-12)
        if float(np.abs(design @ fit - now).max()) > _FIT * size:
            return None
        turn = np.identity(3)
        turn[:, 0], turn[:, 1] = fit[0], fit[1]
        scale = turn[0, 0]
        upright = np.zeros((3, 2))
        upright[0, 0] = upright[1, 1] = scale
        if scale > 0 and float(np.abs(turn[:, :2] - upright).max()) <= _FIT * scale:
            return None
        return turn


@quantities.provider(Readout)
def _declared(held: Mapping[str, Any], at: At) -> Iterable[Quantity]:
    return (Quantity("annotation.count", held["_value"], "1"),)


#: What a text's string makes, which a readout's label takes from a text
#: built anew; everything else of the label stays its own.
_TEXT_DATA = (
    "original_text",
    "text",
    "submobjects",
    "chars",
    "file_name",
    "id_to_vgroup_dict",
    "initial_height",
    "_font_size",
    "background_rectangle",
)


#: How far, as a share of a text's size, its points may stand from where one
#: linear map of its upright text puts them, or from upright, as floating
#: point's error leaves them.
_FIT = 1e-7


def _drawn_points(label: Text) -> FrozenPoints:
    """Every point ``label`` draws, its plate's and its glyphs', in order."""
    members = label.family_members_with_points()
    points = [np.asarray(m.points) for m in members]
    return freeze(np.concatenate(points) if points else np.zeros((0, 3)), FrozenPoints)


def _plate_ids(label: Text) -> set[int]:
    plate = getattr(label, "background_rectangle", None)
    return set() if plate is None else {id(m) for m in plate.get_family()}


def _restyle(made: Text, label: Text) -> None:
    """Style ``made``'s glyphs as ``label``'s first glyph, and its pill as
    ``label``'s pill.
    """
    plates = _plate_ids(label)
    old = label.family_members_with_points()
    glyph = next((m for m in old if id(m) not in plates), None)
    plate = getattr(label, "background_rectangle", None)
    new_plates = _plate_ids(made)
    for member in made.family_members_with_points():
        if id(member) in new_plates:
            continue
        if glyph is not None:
            member.match_style(glyph)
            member.z_index = glyph.z_index
    made_plate = getattr(made, "background_rectangle", None)
    if made_plate is not None and plate is not None:
        # A plate's own set_style keeps only its opacity and paints it the
        # background's black, so match_style would blacken it: its style is
        # set as any shape's.
        VMobject.set_style(made_plate, **plate.get_style(), family=False)
        made_plate.original_fill_opacity = plate.original_fill_opacity
        made_plate.z_index = plate.z_index


class CountUp(Animation):
    """Run the declared value of ``mobject`` (a :class:`Readout` or a
    gauge) from its value when the play begins to ``to``, through
    ``rate_func``; when the play ends the value is ``to`` itself.

    The counted object's own updater keeps running, so an attached readout
    or gauge follows its anchor while it counts.
    """

    def __init__(
        self,
        mobject: Mobject,
        to: float,
        *,
        run_time: float | None = None,
        rate_func: Callable[[float], float] | None = None,
        name: str | None = None,
    ) -> None:
        if not (isinstance(mobject, Annotation) and hasattr(mobject, "set_value")):
            raise TypeError(
                f"CountUp counts a readout or a gauge, not a {type(mobject).__name__}"
            )
        #: The target value.
        self.to = _finite(to, "A count's target")
        given: dict[str, Any] = {"name": name, "suspend_mobject_updating": False}
        if run_time is not None:
            given["run_time"] = run_time
        if rate_func is not None:
            given["rate_func"] = rate_func
        super().__init__(mobject, **given)
        counted: Any = mobject
        #: What it counts.
        self.counted = counted
        #: The value it counts from, read again when the play begins.
        self.start = float(counted.value)

    def create_starting_mobject(self) -> Mobject:
        # The value it starts from is all it reads of the start.
        return self.mobject

    def begin(self) -> None:
        self.start = float(self.counted.value)
        super().begin()

    def interpolate_mobject(self, alpha: float) -> None:
        progress = self.get_sub_alpha(alpha, 0, 1)
        self.counted.set_value(self.start + (self.to - self.start) * progress)

    def finish(self) -> None:
        # Done, the count holds its target exactly: finishing sets it without
        # evaluating the rate function at its end.
        self.counted.set_value(self.to)
        if self.suspend_mobject_updating and self.mobject is not None:
            self.mobject.resume_updating()
