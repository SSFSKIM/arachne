"""Ports: named points on a mark's outline where connectors attach.

A port is kept as two invisible points in the mark's family, one on the
outline and one just inside it, so whatever moves, turns or scales the
mark moves its ports, and the pair gives the direction a connector leaves
in. Neither point draws or widens the mark's box.
"""

from __future__ import annotations

__all__ = ["Ported"]

from collections.abc import Mapping

import numpy as np

from arachne.mobject.types.vectorized_mobject import VectorizedPoint, VGroup

#: How far inside the outline a port's second point sits, in scene units.
_INSET = 1e-3


class Ported(VGroup):
    """A vector group with named ports: :meth:`port` and
    :meth:`port_direction`.
    """

    def _set_ports(self, ports: Mapping[str, tuple[np.ndarray, np.ndarray]]) -> None:
        """Give the mark ``ports``: each name's point on the outline and its
        outward direction there.
        """
        marks = VGroup()
        for point, outward in ports.values():
            unit = outward / np.linalg.norm(outward)
            marks.add(
                VGroup(VectorizedPoint(point), VectorizedPoint(point - unit * _INSET))
            )
        self.port_marks = marks
        self.port_names = tuple(ports)
        self.add(marks)

    def _pair(self, name: str) -> tuple[VectorizedPoint, VectorizedPoint]:
        if name not in self.port_names:
            raise ValueError(
                f"{type(self).__name__} has the ports {', '.join(self.port_names)};"
                f" not {name!r}"
            )
        pair = self.port_marks[self.port_names.index(name)]
        on, inside = pair.submobjects
        assert isinstance(on, VectorizedPoint)
        assert isinstance(inside, VectorizedPoint)
        return on, inside

    def port(self, name: str) -> np.ndarray:
        """Where the port ``name`` is now."""
        return self._pair(name)[0].get_location()

    def port_direction(self, name: str) -> np.ndarray:
        """The unit direction a connector leaves the port ``name`` in now."""
        on, inside = self._pair(name)
        outward = on.get_location() - inside.get_location()
        return outward / np.linalg.norm(outward)
