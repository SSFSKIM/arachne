"""The TeX construction memo, proven on what a session keeps: a scene
executed with the memo off, on and empty, and on and full gives the same
record (every node's recorded state and origin, every capture's content
key unrounded) and the same frames; and a reload served by the memo gives
the record a fresh process gives.
"""

from __future__ import annotations

import pickle
import subprocess
import sys
from pathlib import Path

from arachne.mobject.text.tex_memo import TEX_MEMO
from arachne.session import Session
from tests.helpers.recorded_state import frame_digests, record_state

SCENES = Path(__file__).parent / "tex_memo_scenes.py"
ROOT = Path(__file__).parents[2]
FPS = 10
SETTINGS = {"pixel_width": 96, "pixel_height": 54, "frame_rate": FPS}


def _open(media_dir: str, **settings) -> Session:
    return Session.open(
        SCENES,
        "Formulas",
        config={**SETTINGS, "media_dir": media_dir, **settings},
    )


def _kept(session: Session) -> tuple[dict, list[str]]:
    return record_state(session.record), frame_digests(session, FPS)


def test_the_memo_off_empty_and_full_give_one_record_and_its_frames(tmp_path):
    media = str(tmp_path / "media")
    TEX_MEMO.clear()
    off = _open(media, construction_memo=False)
    try:
        assert len(TEX_MEMO) == 0
        expected = _kept(off)
    finally:
        off.close()
    empty = _open(media)
    try:
        cold = _kept(empty)
    finally:
        empty.close()
    hits = TEX_MEMO.hits
    full = _open(media)
    try:
        assert TEX_MEMO.hits > hits
        warm = _kept(full)
    finally:
        full.close()
    assert cold[0] == expected[0]
    assert warm[0] == expected[0]
    assert cold[1] == expected[1] == warm[1]


_FRESH = """
import pickle, sys
from pathlib import Path
from arachne.session import Session
from tests.helpers.recorded_state import frame_digests, record_state
from tests.timeline.test_tex_memo_records import FPS, SCENES, SETTINGS
session = Session.open(
    SCENES,
    "Formulas",
    config={**SETTINGS, "media_dir": sys.argv[1], "construction_memo": False},
)
Path(sys.argv[2]).write_bytes(
    pickle.dumps((record_state(session.record), frame_digests(session, FPS)))
)
"""


def test_a_reload_the_memo_serves_records_what_a_fresh_process_records(tmp_path):
    """The fresh process builds without the memo: with it, the execution
    its own open makes again (the probe's) would be served too.
    """
    media = str(tmp_path / "media")
    TEX_MEMO.clear()
    session = _open(media)
    try:
        hits, misses = TEX_MEMO.hits, TEX_MEMO.misses
        assert session.reload(force=True)
        # Every construction of the reload was served.
        assert TEX_MEMO.hits > hits
        assert TEX_MEMO.misses == misses
        reloaded = _kept(session)
    finally:
        session.close()
    out = tmp_path / "fresh.pickle"
    subprocess.run(
        [sys.executable, "-c", _FRESH, media, str(out)],
        check=True,
        cwd=ROOT,
    )
    fresh = pickle.loads(out.read_bytes())
    assert reloaded[0] == fresh[0]
    assert reloaded[1] == fresh[1]
