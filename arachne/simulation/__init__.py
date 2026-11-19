"""Simulated motion: bodies, joints and forces that a physics engine steps on
the scene's 60 Hz tick, their result kept in the record as a bake
(:mod:`arachne.timeline.bake`).

Its declared quantities (:mod:`arachne.quantities`) are named under the
prefix ``sim.``, which this package owns, in the units it registers here:
``u`` (a scene unit), ``u/s``, ``rad``, ``rad/s`` and ``s``.
"""

from __future__ import annotations

from arachne import quantities

#: The units a simulated body's quantities are in.
UNITS = ("u", "u/s", "rad", "rad/s", "s")

quantities.register("sim.", owner="arachne.simulation", units=UNITS)
