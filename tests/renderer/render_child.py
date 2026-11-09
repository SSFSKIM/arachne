"""Renders made in a process of their own, as the command line makes them,
for the render-worker tests: ``python render_child.py SPEC OUT``.

A render's workers are forked only from a process whose one thread is the
main one, and a test runner's worker process keeps threads of its own, so
each render here runs in this script's process, which starts none. ``SPEC``
names a scene file and class, the configuration shared by every run, and
the runs, each with the configuration it adds; ``OUT`` gets, per run, what
the render did and wrote:

- ``forks``: how many processes the render forked (``os.register_at_fork``,
  counted while it ran);
- ``fallback``, ``workers``: why it drew its frames in this process, and the
  workers that drew them (the renderer's ``worker_fallback`` and
  ``drawn_by``);
- ``drawn``: the frames handed to the writer in this process, repeats
  expanded, by digest;
- ``movie``: the movie decoded and expanded on the frame grid by its frames'
  times, by digest; how many frames were decoded; how many times fell off
  the grid;
- ``session``: with ``session`` in the run, the frames a session of the
  scene draws at each instant of the grid;
- ``error``, ``error_type``: what the render raised, if it did; then
  ``movie_exists`` and ``left``, the files left in the segment directory,
  ``children``, the processes the render started that are still alive;
  and ``movie_sha``, the movie file's digest.

A run may ``inject`` a failure into the drawing of play ``play`` in a
worker (``how``: ``raise``, ``kill`` or ``unable``), a thread alive through
the render (``thread``), or a memory reader that reads the process id
(``rss_pid``); and ``repeat`` the render, reporting each. With
``config_reads`` in the run, ``config_reads`` reports the configuration
keys read while the plays that waited for the workers were drawn: ``here``,
in this process, and ``workers``, one list per worker. With ``drawings``,
``drawings`` reports likewise what each play was asked to be drawn under
(:func:`record_drawings`), painted on ``paint_on`` when the run names one.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import threading
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

FORKS = [0]
COUNTING = [False]
os.register_at_fork(before=lambda: FORKS.__setitem__(0, FORKS[0] + COUNTING[0]))


def digest(pixels: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(pixels[..., :3]).tobytes()).hexdigest()


def load(file: str, name: str) -> type:
    spec = importlib.util.spec_from_file_location(f"{Path(file).stem}_rendered", file)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return getattr(module, name)


def movie_of(path: Path, fps: Fraction) -> dict:
    """The movie's frames expanded on the grid: each from its time to the
    next frame's, the last to the stream's end (the last packet's end).
    """
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        base = Fraction(stream.time_base)
        packets = [
            (p.pts, p.duration) for p in container.demux(stream) if p.pts is not None
        ]
    with av.open(str(path)) as container:
        frames = [
            (f.pts, f.to_ndarray(format="rgb24")) for f in container.decode(video=0)
        ]
    origin = min(pts for pts, _ in packets)
    times = [(pts - origin) * base * fps for pts, _ in frames]
    end = (max(p + d for p, d in packets) - origin) * base * fps
    shown: list[str] = []
    off = 0
    for start, stop, (_, pixels) in zip(times, [*times[1:], end], frames, strict=True):
        if start.denominator != 1 or stop.denominator != 1:
            off += 1
        shown += [digest(pixels)] * round(stop - start)
    return {"frames": shown, "decoded": len(frames), "off_grid": off}


def inject_failure(how: str, at: int):
    """Make the drawing of play ``at`` in a worker fail ``how``, in the
    scene's first execution; return what undoes it.
    """
    from arachne.renderer import cairo_renderer

    parent = os.getpid()
    draw = cairo_renderer.CairoRenderer._draw_play

    def failing(self, scene, play, frames, *args, **kwargs):
        first = scene._arachne_timeline.ensure_record().execution.number == 1
        if os.getpid() != parent and play.index == at and first:
            if how == "kill":
                os.kill(os.getpid(), signal.SIGKILL)
            if how == "unable":
                from arachne.frame_ir.extract import Unable

                raise Unable("render.unable: a frame the worker could not draw")
            raise RuntimeError("the worker's drawing failed")
        return draw(self, scene, play, frames, *args, **kwargs)

    cairo_renderer.CairoRenderer._draw_play = failing
    return lambda: setattr(cairo_renderer.CairoRenderer, "_draw_play", draw)


class _ReadsRecorded(dict):
    """A configuration store that records each key read from it; ``*`` for a
    read of every key.
    """

    def __init__(self, store: dict, reads: set[str]) -> None:
        super().__init__(store)
        self.reads = reads

    def __getitem__(self, key):
        self.reads.add(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self.reads.add(key)
        return super().get(key, default)

    def __contains__(self, key):
        self.reads.add(key)
        return super().__contains__(key)

    def __iter__(self):
        self.reads.add("*")
        return super().__iter__()

    def items(self):
        self.reads.add("*")
        return super().items()

    def values(self):
        self.reads.add("*")
        return super().values()


def _recording(function, reads: set[str]):
    """``function``, adding every configuration key it reads to ``reads``."""
    from arachne import config

    def recorded(*args, **kwargs):
        store = config._d
        config._d = _ReadsRecorded(store, reads)
        try:
            return function(*args, **kwargs)
        finally:
            config._d = store

    return recorded


def record_config_reads(found: dict):
    """Record the configuration keys read while waiting plays are drawn:
    in this process into ``found["here"]``, in each worker into
    ``found["workers"]``. Returns the undo.
    """
    from arachne.renderer import cairo_renderer, workers

    renderer = cairo_renderer.CairoRenderer
    in_order, in_range, run = renderer._draw_in_order, renderer._draw_range, workers.run
    here: set[str] = set()
    found["here"], found["workers"] = here, []

    def draw_range(self, *args, **kwargs):
        reads: set[str] = set()
        result = _recording(in_range, reads)(self, *args, **kwargs)
        return {**result, "config_reads": sorted(reads)}

    def run_workers(tasks):
        results = run(tasks)
        found["workers"].extend(each.pop("config_reads") for each in results)
        return results

    renderer._draw_in_order = _recording(in_order, here)
    renderer._draw_range = draw_range
    workers.run = run_workers

    def undo() -> None:
        renderer._draw_in_order = in_order
        renderer._draw_range = in_range
        workers.run = run

    return undo


def record_drawings(found: dict, paint_on: str | None = None):
    """Record what each play's frames or static picture were asked to be
    drawn under, ``[what, play, static_layer, draw_path, reuse]``, in this
    process into ``found["here"]``, in each worker into ``found["workers"]``;
    with ``paint_on``, paint them on that path whatever was asked. Returns
    the undo.
    """
    from arachne.renderer import cairo_renderer, workers

    renderer = cairo_renderer.CairoRenderer
    drawing, in_range, run = cairo_renderer.Drawing, renderer._draw_range, workers.run
    found["here"], found["workers"] = [], []
    into = [found["here"]]

    def asked(record, **kwargs):
        caller = sys._getframe(1)
        into[0].append(
            [
                caller.f_code.co_name,
                caller.f_locals["play"].index,
                kwargs["static_layer"],
                kwargs["draw_path"],
                kwargs.get("reuse") is not None,
            ]
        )
        if paint_on is not None:
            kwargs["draw_path"] = paint_on
        return drawing(record, **kwargs)

    def draw_range(self, *args, **kwargs):
        into[0] = []
        return {**in_range(self, *args, **kwargs), "drawings": into[0]}

    def run_workers(tasks):
        results = run(tasks)
        found["workers"].extend(each.pop("drawings") for each in results)
        return results

    cairo_renderer.Drawing = asked
    renderer._draw_range = draw_range
    workers.run = run_workers

    def undo() -> None:
        cairo_renderer.Drawing = drawing
        renderer._draw_range = in_range
        workers.run = run

    return undo


def children() -> set[int]:
    """The processes alive whose parent is this one."""
    ps = subprocess.Popen(
        ["ps", "-A", "-o", "pid=,ppid="], stdout=subprocess.PIPE, text=True
    )
    listed, _ = ps.communicate()
    me = str(os.getpid())
    return {
        int(pid)
        for pid, parent in (
            line.split() for line in listed.splitlines() if line.strip()
        )
        if parent == me and int(pid) != ps.pid
    }


def render(scene_class: type, settings: dict, run: dict) -> dict:
    from arachne import tempconfig
    from arachne.manager import Manager
    from arachne.timeline.clock import exact

    out: dict = {"error": None, "error_type": None}
    with tempconfig(settings):
        scene = scene_class()
        manager = Manager(scene)
        drawn: list[str] = []

        def collect() -> None:
            writer = scene.renderer.file_writer
            write = writer.write_frame

            def write_frame(pixels, *, repeat=1):
                drawn.extend([digest(pixels)] * repeat)
                write(pixels, repeat=repeat)

            writer.write_frame = write_frame

        renew = manager._renew_scene

        def renewed() -> None:
            renew()
            drawn.clear()
            collect()

        manager._renew_scene = renewed
        collect()
        before = children()
        FORKS[0] = 0
        COUNTING[0] = True
        try:
            scene.render()
        except BaseException as error:  # noqa: BLE001
            out["error"] = str(error)
            out["error_type"] = type(error).__name__
        finally:
            COUNTING[0] = False
        renderer = scene.renderer
        writer = renderer.file_writer
        fps = exact(renderer.camera.frame_rate)
        path = Path(writer.movie_file_path)
        out["forks"] = FORKS[0]
        out["fallback"] = getattr(renderer, "worker_fallback", None)
        out["workers"] = [
            {
                "first": each.first,
                "last": each.last,
                "frames": each.frames,
                "peak_mb": each.peak_mb,
            }
            for each in getattr(renderer, "drawn_by", ())
        ]
        out["drawn"] = drawn
        out["frame_runs"] = {str(k): list(v) for k, v in renderer.frame_runs.items()}
        out["movie_exists"] = path.is_file()
        out["movie"] = movie_of(path, fps) if out["movie_exists"] else None
        out["movie_sha"] = (
            hashlib.sha256(path.read_bytes()).hexdigest()
            if out["movie_exists"]
            else None
        )
        segments = writer.partial_movie_directory
        out["left"] = (
            sorted(p.name for p in segments.iterdir()) if segments.is_dir() else []
        )
        out["children"] = sorted(children() - before)
        out["pid"] = os.getpid()
        if run.get("session"):
            count = len(out["movie"]["frames"]) if out["movie"] else len(drawn)
            out["session"] = session_frames(scene_class, settings, fps, count)
    return out


def session_frames(scene_class: type, settings: dict, fps: Fraction, count: int):
    from arachne.session import Session

    file = sys.modules[scene_class.__module__].__file__
    config = {k: settings[k] for k in ("pixel_width", "pixel_height", "frame_rate")}
    session = Session.open(Path(file), scene_class.__name__, config=config)
    try:
        return [
            digest(np.asarray(session.frame(Fraction(k) / fps))) for k in range(count)
        ]
    finally:
        session.close()


def main() -> int:
    from tqdm import tqdm

    # The progress bars' lock starts multiprocessing's resource tracker, a
    # child of this process for good: started before any render's children
    # are counted.
    tqdm.get_lock()
    spec = json.loads(Path(sys.argv[1]).read_text())
    scene_class = load(spec["file"], spec["scene"])
    results = []
    for index, run in enumerate(spec["runs"]):
        undo = []
        if "inject" in run:
            undo.append(inject_failure(run["inject"]["how"], run["inject"]["play"]))
        if run.get("rss_pid"):
            from arachne.server import registry

            reader = registry._rss_mb
            registry._rss_mb = lambda: float(os.getpid())
            undo.append(lambda reader=reader: setattr(registry, "_rss_mb", reader))
        reads: dict = {}
        if run.get("config_reads"):
            undo.append(record_config_reads(reads))
        drawings: dict = {}
        if run.get("drawings"):
            undo.append(record_drawings(drawings, run.get("paint_on")))
        held = threading.Event()
        if run.get("thread"):
            threading.Thread(target=held.wait, name="held", daemon=True).start()
        try:
            for repeat in range(run.get("repeat", 1)):
                media = Path(spec["media"]) / f"run-{index}-{repeat}"
                settings = {**spec["config"], **run["config"], "media_dir": str(media)}
                results.append(render(scene_class, settings, run))
                if reads:
                    results[-1]["config_reads"] = {
                        "here": sorted(reads["here"]),
                        "workers": list(reads["workers"]),
                    }
                if drawings:
                    results[-1]["drawings"] = {
                        "here": list(drawings["here"]),
                        "workers": list(drawings["workers"]),
                    }
        finally:
            held.set()
            for each in reversed(undo):
                each()
    Path(sys.argv[2]).write_text(json.dumps(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
