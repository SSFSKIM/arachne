"""Annotation: the engine-drawn marks that carry a picture's claims, each
tied to a point of an asset or of any drawn object.

- :func:`anchor`: an anchor on any drawn object (a rig's are its
  manifest's, :meth:`~arachne.assets.Rig.anchor`);
- :class:`Callout`: a label on a :class:`LabelPill`, joined to its anchor
  by a :class:`LeaderLine`;
- :class:`Readout` and :class:`CountUp`: a running number in a declared
  format, counted to its value;
- :class:`BarGauge`, :class:`ColumnGauge`, :class:`DialGauge`: a declared
  value shown as a share of a scale.

Every one attaches the same way (:mod:`.attach`), and the lint rule
:data:`~arachne.lint.rules.anchored` measures each at its anchor. A readout
and a gauge declare their value (:mod:`arachne.quantities`): ``annotation.count``
and ``annotation.gauge``, read from the value a capture holds.
"""

from __future__ import annotations

__all__ = [
    "BarGauge",
    "Callout",
    "ColumnGauge",
    "CountUp",
    "DialGauge",
    "LabelPill",
    "LeaderLine",
    "Readout",
    "anchor",
]

from arachne import quantities

from .attach import LeaderLine, anchor
from .callout import Callout, LabelPill
from .gauge import BarGauge, ColumnGauge, DialGauge
from .readout import CountUp, Readout

quantities.register("annotation.", owner="arachne.mobject.annotation")
