"""The geometry lint: findings about what a scene shows, read from its
session without drawing.

:func:`lint` reads a :class:`~arachne.session.Session` at its lint
instants (:mod:`.instants`), where its rules measure the drawn parts
(:mod:`.parts`); its findings (:mod:`.findings`) carry codes from the
append-only table :data:`CODES` (:mod:`.codes`), and the report gives them
as :mod:`.serialise` says.
"""

from __future__ import annotations

__all__ = [
    "CODES",
    "RULES",
    "Code",
    "Finding",
    "FindingObject",
    "LintContext",
    "LintInstant",
    "LintResult",
    "Part",
    "Parts",
    "armed",
    "lint",
]

from .armed import armed
from .codes import CODES, Code
from .core import RULES, LintContext, LintResult, lint
from .findings import Finding, FindingObject
from .instants import LintInstant
from .parts import Part, Parts
