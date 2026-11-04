"""XF, the Frame IR: frames described for a drawer, and the drawers' seam.

- :mod:`.xf`: XF's records, shapes, deltas and a stream, as Python values.
- :mod:`.extract`: :class:`~.extract.FrameStream`, a shown frame as an XF delta.
- :mod:`.cairo_painter`: :class:`~.cairo_painter.CairoDrawer`, Cairo drawing
  from XF alone with CE's camera code.
- :mod:`.painter`: the painter a draw path names (``draw_path``).
- :mod:`.lint`: the drawing's lint codes.

Nothing here imports the core (``arachne_core``) but the ``ir_cairo`` painter,
on use.
"""
