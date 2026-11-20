"""Schematic marks: engine-drawn vector objects for what a science diagram
draws when no asset fits, from arrows that carry an amount to the
circuit and logic symbols.

Every mark is an ordinary vector object with the usual style arguments;
one that carries a number keeps it as a plain attribute (a vessel's
``level``, a scatter's ``count``, wavefronts' ``spacing``, a trail's
``interval``, a wavy line's ``wavelengths``, a flow arrow's ``magnitude``,
a particle's ``charge``) and declares it as ``mark.<attribute>``
(:mod:`._declared`).
A block, a node and a logic gate have named ports a connector attaches to
(:meth:`~.ports.Ported.port`).
"""

from __future__ import annotations

from . import _declared  # noqa: F401  (registers the marks' declared numbers)
from .arrows import *
from .blocks import *
from .circuit import *
from .lines import *
from .particles import *
from .shapes import *
from .views import *

__all__ = [
    "FlowArrow",
    "ElbowConnector",
    "CycleArrow",
    "ReversibleArrows",
    "BluntArrow",
    "MultiLine",
    "Ray",
    "Zigzag",
    "Coil",
    "WavyLine",
    "Capsule",
    "DoubleBoundary",
    "Vessel",
    "Drum",
    "LayerStack",
    "Block",
    "CellGrid",
    "Node",
    "ChargedParticle",
    "Particles",
    "Burst",
    "WavefrontArcs",
    "DottedTrail",
    "DimensionLine",
    "Magnifier",
    "PlatePair",
    "Switch",
    "OneWayMark",
    "LogicGate",
]
