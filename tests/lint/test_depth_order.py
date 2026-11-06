"""Under a ThreeDCamera the lint reads the order a frame paints in: the
camera's depth sort with the rotation of the angles on show, as a frame's
capture remakes it before sorting, whatever rotation the camera last held.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

from arachne.lint import lint
from arachne.lint.rules import sweep
from arachne.timeline.draw import Drawing
from arachne.timeline.routes import _resolve

from . import defect_scenes

SCENES = Path(defect_scenes.__file__)


def _read_at_end(session):
    seen = []
    lint(
        session,
        rules=(lambda c: seen.extend(c.parts) or (),),
        at=[session.record.duration],
    )
    return seen[-1]


def test_the_read_paints_in_the_order_the_frame_does_under_a_turned_camera(
    session_of,
):
    session = session_of("ArrowBeforeARod", SCENES, frame_rate=15)
    record = session.record
    parts = _read_at_end(session)
    leaves = sorted(
        (leaf for part in parts.parts for leaf in part.leaves),
        key=lambda leaf: leaf.order,
    )
    read = [leaf.id for leaf in leaves]
    camera = record._scene.renderer.camera
    capture = _resolve(record, record.duration).capture
    with Drawing(record) as drawing:
        drawing.show(capture, capture.draw_list())
        # A frame's capture remakes the rotation before it sorts.
        camera.reset_rotation_matrix()
        painted = camera.get_mobjects_to_display(capture.draw_list())
        framed = [record.origin(m).id for m in painted]
    assert read == [i for i in framed if i in set(read)]


_FIRST_READ = textwrap.dedent(
    """
    import json
    import sys

    import numpy as np

    from arachne.lint import lint
    from arachne.lint.rules._units import measured_units
    from arachne.session import Session

    session = Session.open(
        sys.argv[1],
        "ArrowBeforeARod",
        config={"pixel_width": 96, "pixel_height": 54, "frame_rate": 15},
    )
    camera = session.record._scene.renderer.camera
    # Nothing has projected through the camera yet: the rotation it holds is
    # the one it was made with, not the side view's.
    held = not np.allclose(camera.rotation_matrix, camera.generate_rotation_matrix())
    seen = []
    lint(
        session,
        rules=(lambda c: seen.extend(c.parts) or (),),
        at=[session.record.duration],
    )
    orders = {}
    for unit in measured_units(seen[-1]):
        solid = unit.unit.within[0].type if unit.unit.within else unit.unit.type
        orders.setdefault(solid, []).extend(leaf.order for leaf in unit.leaves)
    print(json.dumps({"held": held, "orders": orders}))
    """
)


def test_the_first_read_in_a_fresh_process_sorts_by_the_view_shown():
    done = subprocess.run(
        [sys.executable, "-c", _FIRST_READ, str(SCENES)],
        capture_output=True,
        text=True,
        timeout=300,
        check=True,
    )
    found = json.loads(done.stdout.strip().splitlines()[-1])
    assert found["held"], "the camera's rotation was remade before the read"
    orders = found["orders"]
    # The arrow lies wholly in front of the rod: the frame paints all of it
    # after all of the rod.
    assert min(orders["Arrow3D"]) > max(orders["Cylinder"])


def test_a_word_ahead_of_a_solid_is_not_read_as_drawn_under_it(session_of):
    session = session_of("WordAheadOfAFineRod", SCENES, frame_rate=15)
    # Only the closing instant is read, so no earlier read has projected
    # through the camera.
    end = [session.record.duration]
    result = lint(session, rules=(sweep.where(at=end),), at=end)
    codes = {f.code for f in result.findings}
    assert "occlusion.glyph" not in codes
    assert "overlap.text" in codes
