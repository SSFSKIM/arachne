"""Compare two sequences of frames at the scale a viewer can see.

A difference blocks when it is visible: an object drawn on one side only (presence),
a connected cluster of changed pixels large and strong enough to see, or such a
cluster appearing between consecutive frames where no object changed (flicker).
Everything smaller is counted and never blocks. When two renderers are compared,
pixels on the boundary of an object's footprint are counted per object against a
budget instead of joining clusters.

The cheap pass reads every frame and needs frames only. The structural pass reads
sampled frames and is also given each side's object map and the objects that changed.
"""

from __future__ import annotations

from .cheap import CheapFrame, cheap, cheap_self
from .comparison import compare
from .model import (
    ArrayFrames,
    Box,
    ChangedSet,
    EdgeBudget,
    Footprint,
    Frames,
    Group,
    HandedMaps,
    MapSource,
    Mode,
    ObjectMap,
    Thresholds,
)
from .sampling import choose_objects, sample
from .structural import Structural, flicker, structural
from .verdict import Finding, Verdict

__all__ = [
    "ArrayFrames",
    "Box",
    "ChangedSet",
    "CheapFrame",
    "EdgeBudget",
    "Finding",
    "Footprint",
    "Frames",
    "Group",
    "HandedMaps",
    "MapSource",
    "Mode",
    "ObjectMap",
    "Structural",
    "Thresholds",
    "Verdict",
    "cheap",
    "cheap_self",
    "choose_objects",
    "compare",
    "flicker",
    "sample",
    "structural",
]
