"""A label on a pill, joined by a leader line to the point it names."""

from __future__ import annotations

__all__ = ["Callout", "LabelPill"]

from typing import TYPE_CHECKING, Any

from arachne.mobject.text.text_mobject import Text
from arachne.utils.color import WHITE, ParsableManimColor

from .attach import DEFAULT_OFFSET, Annotation

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

    from arachne.assets.anchor import Anchor

#: A pill's fill, a dark slate that holds light text on light or dark art.
PILL_COLOR = "#20242C"


class LabelPill(Text):
    """``text`` on a rounded plate: the text's own background rectangle, so
    the lint measures the label by its pill's boundary.
    """

    def __init__(
        self,
        text: str,
        *,
        font_size: float = 24,
        color: ParsableManimColor = WHITE,
        pill_color: ParsableManimColor = PILL_COLOR,
        pill_opacity: float = 1.0,
        buff: float = 0.1,
        corner_radius: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(text, font_size=font_size, color=color, **kwargs)
        self.add_background_rectangle(
            color=pill_color,
            opacity=pill_opacity,
            buff=buff,
            corner_radius=corner_radius,
        )
        # Its size is the pill's: a pill scaled to twice its height says its
        # font size is twice as large.
        self.initial_height = self.height


class Callout(Annotation):
    """``text`` on a :class:`LabelPill` at the anchor ``at`` plus
    ``offset``, joined to the anchor's point by a leader line (unless
    ``leader`` is false), and kept there as the anchor moves (unless
    ``follow`` is false, when the pill stays where it stands and the leader
    stretches). ``pill_kwargs`` go to the pill.
    """

    def __init__(
        self,
        text: str,
        at: Anchor | None = None,
        *,
        offset: Sequence[float] | np.ndarray = DEFAULT_OFFSET,
        follow: bool = True,
        leader: bool = True,
        font_size: float = 24,
        color: ParsableManimColor = WHITE,
        name: str | None = None,
        **pill_kwargs: Any,
    ) -> None:
        #: The label on its pill.
        self.pill = LabelPill(text, font_size=font_size, color=color, **pill_kwargs)
        super().__init__(
            self.pill,
            at=at,
            offset=offset,
            follow=follow,
            leader=leader,
            leader_color=color,
            name=name,
        )
