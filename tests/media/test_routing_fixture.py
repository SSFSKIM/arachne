"""The routing fixture: two scenes that differ only in a plain still against
a clip on its first frame, at the same corners and resampling. Where the
Frame IR is written, both are one ``image`` record with equal bytes: nothing
in it tells a still from a clip, and the timeline tells them apart. The
first is a CPU frame and imports nothing of the GPU build; the second is on
the ``gpu`` route, and refused with ``render.gpu_required`` on a host with
no GPU. (What the two drawing paths emit is not shown here: on this branch
a GPU-route frame is drawn through the camera, not from the Frame IR. Its
frames on a device are in ``test_drawn.py``.)
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from arachne.frame_ir.extract import FrameStream
from arachne.frame_ir.xf import Delta, Record, Shape
from arachne.routing import GpuRequired
from arachne.timeline.draw import Drawing, static_list
from arachne.timeline.routes import _resolve

from .conftest import SCENES, SMALL

TRIPWIRE = Path(__file__).resolve().parents[2] / "gpu" / "ci" / "tripwire.py"
QUARTER = Fraction(1, 4)


def _images(session, t: Fraction) -> list[tuple[Record, Shape]]:
    """The ``image`` records the Frame IR's extractor writes for the frame
    at ``t``, each with the image it uploads, from the lists a drawing
    paints that frame from.
    """
    record = session.record
    resolved = _resolve(record, t)
    capture = resolved.capture
    with Drawing(record) as drawing:
        roots = capture.draw_list()
        drawing.show(capture, [*roots, *capture.moving])
        if resolved.prepare is not None:
            resolved.prepare(drawing)
        camera = drawing.camera
        if resolved.play is None:
            groups = [("static", roots)]
        else:
            groups = [
                ("static", static_list(camera, roots, capture.moving)),
                ("moving", list(capture.moving) or roots),
            ]
        delta = FrameStream().frame(camera, groups)
    uploaded = {shape.key: shape for shape in delta.upload}
    return [
        (found, uploaded[found.fields["shape"]])
        for _, found in delta.set
        if found.kind == "image"
    ]


def _encoded(record: Record, shape: Shape) -> bytes:
    """``record`` and its image alone as one frame of XF, in the core's
    bytes, under one object id.
    """
    delta = Delta(
        objects=[(0, "the picture", "")],
        upload=[shape],
        set=[(0, record)],
        order=[("static", 0, 0)],
    )
    return bytes(delta.to_core().to_bytes())


def test_a_still_and_a_clip_are_one_image_record_with_equal_bytes(session_of):
    ((image, image_shape),) = _images(session_of("FixtureImage"), QUARTER)
    ((clip, clip_shape),) = _images(session_of("FixtureClip"), QUARTER)
    assert image.fields == clip.fields
    assert image_shape.data.tobytes() == clip_shape.data.tobytes()
    assert _encoded(image, image_shape) == _encoded(clip, clip_shape)


def test_the_timeline_routes_the_still_to_the_cpu_and_the_clip_to_the_gpu(
    session_of,
):
    assert session_of("FixtureImage").route == "cpu"
    assert session_of("FixtureClip").route == "gpu"
    # A still of the asset API, in the image's place, is a CPU frame too.
    assert session_of("FixtureStill").route == "cpu"


def test_with_no_gpu_the_clip_s_frame_is_refused(session_of, no_gpu):
    session = session_of("FixtureClip")
    with pytest.raises(GpuRequired) as refused:
        session.frame(QUARTER)
    assert refused.value.code == "render.gpu_required"
    assert list(refused.value.capabilities) == ["media.video"]


CHILD = """
import site, sys
site.addsitedir({site!r})
from fractions import Fraction
from arachne.session import Session
from arachne import routing
for scene in ("FixtureImage", "FixtureStill"):
    session = Session.open({scenes!r}, scene, config={config!r})
    assert routing.route(session.record) == "cpu"
    session.frame(Fraction(1, 4))
    session.close()
print("cpu route drawn", flush=True)
with open({log!r}) as log:
    print("records:", len(log.readlines()), flush=True)
"""


def test_the_still_s_frames_import_nothing_of_the_gpu_build(tmp_path):
    from tests.routing.conftest import run_child

    site, log = tmp_path / "site", tmp_path / "tripwire.log"
    site.mkdir()
    namespace: dict = {}
    exec(compile(TRIPWIRE.read_text(), str(TRIPWIRE), "exec"), namespace)
    namespace["install"](site, log)
    printed = run_child(
        CHILD.format(site=str(site), scenes=str(SCENES), config=SMALL, log=str(log))
    )
    assert "cpu route drawn" in printed
    assert "records: 0" in printed
