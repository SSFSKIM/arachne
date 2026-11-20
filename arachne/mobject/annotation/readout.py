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

As its text widens or narrows, a readout keeps one point of its box still,
its *hold*, so what the reader has found does not move under the eye:

- attached to an anchor, the edge facing the anchor: an offset to the right
  holds its left edge, one to the left its right edge, one straight up or
  down its centre. The gap to what it labels (and its leader's length) stays
  as it was set, and the text grows away from its anchor, never over it;
- placed by hand, its start (left edge) when its format begins with a word
  or a sign before the number (``"swell {:.0f}%"``), which the reader has
  already read and must not slide; otherwise (a bare number, a number
  before its unit) its centre, where ``move_to`` put it;
- ``hold`` names it outright, for a placement the readout cannot see: a
  number right-aligned in a column holds ``RIGHT``.

Up and down it keeps its baseline: the bottom of a glyph its format shows
in every text (the first of a leading word, else the last of a trailing
unit), which digits of other heights or a dipping round digit never move;
a bare number keeps its centre's height.

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


def _held_point(
    pattern: str,
    at: Anchor | None,
    offset: Sequence[float] | np.ndarray,
    hold: Sequence[float] | np.ndarray | None,
) -> Sequence[float] | np.ndarray:
    """The point of a readout's box that stays still as its text changes
    (the module's rule): ``hold`` when given.
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
    the text, and :class:`CountUp` runs it to a target. As the text changes
    it keeps the point ``hold`` of its box still (``LEFT``, ``ORIGIN`` or
    ``RIGHT``); by default the edge facing its anchor, or, placed by hand,
    its start when its format begins with a word and its centre otherwise.
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
        super().__init__(
            self.label,
            at=at,
            offset=offset,
            follow=follow,
            leader=leader,
            leader_color=color,
            hold=_held_point(self.format, at, offset, hold),
            name=name,
        )

    def _held_point(self, label: Text | None = None) -> np.ndarray:
        """Where the label's (or ``label``'s) :attr:`hold` point stands:
        across, that point of its box; up and down, the bottom of the glyph
        its format always shows (its baseline), or its box's when it shows
        none.
        """
        label = self.label if label is None else label
        point = np.array(label.get_critical_point(self.hold), dtype=float)
        fixed = _fixed_glyph(self.format)
        glyphs = _glyphs(label)
        if fixed is not None and glyphs:
            point[1] = glyphs[fixed].get_bottom()[1]
        return point

    @property
    def value(self) -> float:
        """The declared value now."""
        return self._value

    def set_value(self, value: float) -> Self:
        """Declare ``value`` and show it in the format, in place of the
        text shown: only the text changes, at the label's size with its
        :attr:`hold` point where it was, its glyphs styled as its first
        glyph was (fill, stroke, opacity, depth) and its pill as its pill
        was.
        """
        number = _finite(value, "A readout's value")
        said = self.format.format(number)
        self._value = number
        label = self.label
        if said != label.original_text:
            made = self._label_class(said, font_size=label.font_size)
            made.shift(self._held_point() - self._held_point(made))
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
        return self


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
