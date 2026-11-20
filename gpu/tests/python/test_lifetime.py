"""Many compositors in one process, from Python: opened and dropped one after another, kept open
together, and drawing on several threads at once, as a session server opens them over its life.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from arachne_gpu import Blur, Frame, Layer, Picture, painted_key
from gpu_helpers import CLEAR, painted_picture

MANY = 100
SIDE = 48


def passes_a_picture_through(compositor, seed: int) -> None:
    picture = painted_picture(SIDE, SIDE, seed)
    key = painted_key(picture)
    compositor.upload_painted(key, picture)
    np.testing.assert_array_equal(
        compositor.draw(Frame(CLEAR, [Layer(Picture(key))])), picture
    )


def test_a_process_opens_and_drops_a_hundred_compositors(make_compositor):
    for seed in range(MANY):
        passes_a_picture_through(make_compositor(SIDE, SIDE), seed)


def test_a_process_keeps_a_hundred_compositors_open_and_each_draws(make_compositor):
    compositors = [make_compositor(SIDE, SIDE) for _ in range(MANY)]
    for seed, compositor in enumerate(compositors):
        passes_a_picture_through(compositor, seed)
    passes_a_picture_through(compositors[0], 0)


def test_compositors_on_several_threads_draw_at_once(make_compositor):
    threads = 8
    picture = painted_picture(SIDE, SIDE, 3)
    key = painted_key(picture)
    frame = Frame(CLEAR, [Layer(Picture(key), blur=Blur(2.0), object_id="blurred")])
    alone = make_compositor(SIDE, SIDE)
    alone.upload_painted(key, picture)
    expected = alone.draw(frame)
    # A thread whose open or upload fails breaks the barrier, so the others stop waiting and the
    # failure is reported; the timeout bounds a thread that never arrives.
    start = threading.Barrier(threads, timeout=120)

    def draw_ten() -> list[bool]:
        try:
            compositor = make_compositor(SIDE, SIDE)
            compositor.upload_painted(key, picture)
        except BaseException:
            start.abort()
            raise
        start.wait()
        return [
            np.array_equal(compositor.draw(frame), expected)
            and len(compositor.object_map(frame)) == 1
            for _ in range(10)
        ]

    with ThreadPoolExecutor(max_workers=threads) as pool:
        futures = [pool.submit(draw_ten) for _ in range(threads)]
    errors = [f.exception() for f in futures if f.exception() is not None]
    if errors:
        # The setup failure, not the broken barrier it left its peers.
        first = [e for e in errors if not isinstance(e, threading.BrokenBarrierError)]
        raise (first or errors)[0]
    assert [f.result() for f in futures] == [[True] * 10] * threads
