"""A running number: a declared value shown in a declared format, and the
animation that counts it to a target.

A :class:`Readout` holds its value as a plain number and its label as a
text of the value in its format (a Python format string, ``"{:.1f} mV"``),
rebuilt whenever the value changes. :class:`CountUp` runs the value of a
readout, or of a gauge (:mod:`.gauge`), to a target: at ``alpha`` the
declared value is ``start + (to - start) * rate_func(alpha)``, and when the
play ends it is the target itself.

What a viewer sees of it is three statements about time, never one: the
text ``state_at(t)`` answers is the format of the value at its ``as_of``
(the last frame of the record's grid at or before ``t``); a delivered frame
draws the format of the declared value at its own instant; and the exact
target is in a delivered frame only when a frame instant at or after the
play's end exists, so a count followed by a wait shows it.
"""

from __future__ import annotations

__all__ = ["CountUp", "Readout"]

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from arachne.animation.animation import Animation
from arachne.mobject.text.text_mobject import Text
from arachne.utils.color import WHITE, ParsableManimColor

from .attach import DEFAULT_OFFSET, Annotation
from .callout import LabelPill

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Self

    import numpy as np

    from arachne.assets.anchor import Anchor
    from arachne.mobject.mobject import Mobject

#: How close to an end of its run a count's progress is taken to be there,
#: so that a play's last value is the target itself.
_END = 1e-12


def _finite(value: Any, what: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{what} is a finite number, not {value!r}.")
    return number


def formatted(pattern: str, value: float) -> str:
    """``value`` in ``pattern``, a Python format string with one field;
    :class:`ValueError` for a format that does not take one number.
    """
    try:
        return pattern.format(value)
    except (IndexError, KeyError, ValueError, TypeError) as error:
        raise ValueError(
            f"A readout's format is a Python format string with one field for its"
            f" number, as '{{:.1f}} mV'; {pattern!r} is not ({error})."
        ) from None


class Readout(Annotation):
    """The number ``value`` shown in ``format``, optionally attached to an
    anchor ``at`` plus ``offset`` (following it unless ``follow`` is false,
    with a leader line with ``leader``), its text on a pill with ``pill``.

    Its value is :attr:`value`; :meth:`set_value` changes it and rebuilds
    the text, and :class:`CountUp` runs it to a target.
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
        name: str | None = None,
    ) -> None:
        number = _finite(value, "A readout's value")
        self.format = str(format)
        self._label_class: type[Text] = LabelPill if pill else Text
        #: The text of the value in its format.
        self.label = self._label_class(
            formatted(self.format, number), font_size=font_size, color=color
        )
        self._value = number
        super().__init__(
            self.label,
            at=at,
            offset=offset,
            follow=follow,
            leader=leader,
            leader_color=color,
            name=name,
        )

    @property
    def value(self) -> float:
        """The declared value now."""
        return self._value

    def set_value(self, value: float) -> Self:
        """Declare ``value`` and show it in the format, in place of the
        text shown: at its centre, its size and its glyphs' colour.
        """
        number = _finite(value, "A readout's value")
        said = formatted(self.format, number)
        self._value = number
        label = self.label
        if said != label.original_text:
            made = self._label_class(
                said, font_size=label.font_size, color=_glyph_color(label)
            )
            made.move_to(label.get_center())
            # The label keeps its identity, so the record follows one text.
            label.__dict__.update(made.__dict__)
        return self


def _glyph_color(label: Text) -> ParsableManimColor:
    """The colour of ``label``'s first glyph (not its pill's)."""
    plate = getattr(label, "background_rectangle", None)
    skipped = set() if plate is None else {id(m) for m in plate.get_family()}
    for member in label.family_members_with_points():
        if id(member) not in skipped:
            return member.get_color()
    return WHITE


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
        if abs(progress - 1) < _END:
            value = self.to
        elif abs(progress) < _END:
            value = self.start
        else:
            value = self.start + (self.to - self.start) * progress
        self.counted.set_value(value)
