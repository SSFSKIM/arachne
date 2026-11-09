"""Work run in forked processes, one per task, all at once: how a render's
frames are drawn and encoded by workers (``render_workers``).

A render that asks for workers runs its scene forward with its frames left
undrawn, then cuts its plays into contiguous ranges (:func:`ranges`) and
forks one worker per range (:func:`run`). A forked process starts from
where its parent stands, so each worker holds the whole record, the scene
and the camera as the forward run left them, and draws its range from the
record as the render's own process would have. Forking is safe only from a
process whose one thread is the one forking: the render waits on every
encoder of its own first, and a render whose process keeps another thread
(:func:`threads_alive`) is drawn in its own process instead.

Each worker ends with :func:`os._exit`, so nothing of the parent's runs in
it on the way out (its exit handlers, its buffered files). A worker that
raises, or dies, fails the whole run: the others are killed, every one is
waited on, and :class:`WorkerFailed` says which failed and how.
"""

from __future__ import annotations

__all__ = ["WorkerFailed", "WorkerReport", "ranges", "run", "threads_alive"]

import contextlib
import gc
import json
import os
import shutil
import signal
import sys
import tempfile
import threading
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: How long the parent sleeps between two looks at its workers, in seconds.
POLL_S = 0.01


@dataclass(frozen=True)
class WorkerReport:
    """What one worker of a render drew: its range, plays ``first`` to
    ``last`` by index; the ``frames`` it wrote; and ``peak_mb``, the most
    memory its process held, in megabytes, read in the worker (on Linux its
    own high-water mark, ``VmHWM``, which counts the pages it shares with
    the render's process as well as its own).
    """

    first: int
    last: int
    frames: int
    peak_mb: float


class WorkerFailed(RuntimeError):
    """A worker of :func:`run` failed: ``index``, its task's; ``kind``, the
    qualified name of the exception it raised (``None`` when it died
    without raising, by a signal or an exit status); ``message``, what it
    raised; ``trace``, its traceback as the worker formatted it.
    """

    def __init__(
        self,
        index: int,
        why: str,
        *,
        kind: str | None = None,
        message: str | None = None,
        trace: str | None = None,
    ) -> None:
        super().__init__(why)
        self.index = index
        self.kind = kind
        self.message = message
        self.trace = trace


def ranges(costs: Sequence[int], workers: int) -> list[tuple[int, int]]:
    """Contiguous ranges ``(start, stop)`` of the items whose costs are
    ``costs`` (each at least 1), in order, at most ``workers`` of them and
    none empty, balanced by cost: with the whole cost cut in ``workers``
    equal parts, an item belongs to the part its middle falls in. Items of
    equal cost no more than the workers make a range each; with costs apart,
    items a costly one outweighs may share a range even then.
    """
    total = sum(costs)
    parts: list[int] = []
    before = 0
    for cost in costs:
        middle = 2 * workers * before + workers * cost
        parts.append(min(workers - 1, middle // (2 * total)))
        before += cost
    found: list[tuple[int, int]] = []
    for index, part in enumerate(parts):
        if found and parts[index - 1] == part:
            found[-1] = (found[-1][0], index + 1)
        else:
            found.append((index, index + 1))
    return found


def threads_alive() -> list[str]:
    """The names of this process's threads alive now, other than the one
    calling.
    """
    me = threading.current_thread()
    return [t.name for t in threading.enumerate() if t is not me and t.is_alive()]


def run(tasks: Sequence[Callable[[], dict[str, Any]]]) -> list[dict[str, Any]]:
    """Run each of ``tasks`` in a process forked for it, all at once, and
    return what each returned (a JSON form), in order. The objects alive
    when the workers are forked are kept out of the collector's reach
    (:func:`gc.freeze`) until the last is forked, so that a collection in a
    worker writes nothing to the pages it shares with this process. A
    worker that fails raises :class:`WorkerFailed` once every worker is
    gone; anything that interrupts the wait kills them first.
    """
    alive = threads_alive()
    if alive:
        raise RuntimeError(
            "workers are forked only from a process with one thread; alive: "
            + ", ".join(alive)
        )
    directory = Path(tempfile.mkdtemp(prefix="arachne-workers-"))
    pids: dict[int, int] = {}
    try:
        _flush()
        gc.freeze()
        try:
            for index, task in enumerate(tasks):
                pid = os.fork()
                if pid == 0:
                    _work(task, directory / f"{index}.json")
                pids[pid] = index
        finally:
            gc.unfreeze()
        return _wait(pids, directory, len(tasks))
    finally:
        _end(pids)
        shutil.rmtree(directory, ignore_errors=True)


def _flush() -> None:
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.flush()


def _work(task: Callable[[], dict[str, Any]], out: Path) -> None:
    """In a worker: run ``task``, write what it returned or raised to
    ``out`` (whole, or not at all), and exit, running nothing else.
    """
    status = 1
    try:
        try:
            found: dict[str, Any] = {"result": task()}
            status = 0
        except BaseException as error:  # noqa: BLE001 (every failure is reported)
            kind = type(error)
            found = {
                "kind": f"{kind.__module__}.{kind.__qualname__}",
                "message": str(error),
                "trace": "".join(traceback.format_exception(error)),
            }
        made = out.with_suffix(".part")
        made.write_text(json.dumps(found))
        os.replace(made, out)
    except BaseException:  # noqa: BLE001
        status = 1
    finally:
        _flush()
        os._exit(status)


def _wait(pids: dict[int, int], directory: Path, count: int) -> list[dict[str, Any]]:
    """Wait on every worker (``pids``: by process, its task's index) and
    return each one's result in order; on the first failure, kill the rest
    and raise it once they are gone.
    """
    results: dict[int, dict[str, Any]] = {}
    failed: WorkerFailed | None = None
    while pids:
        for pid in list(pids):
            done, status = os.waitpid(pid, os.WNOHANG)
            if done == 0:
                continue
            index = pids.pop(pid)
            found = _read(directory / f"{index}.json")
            if os.waitstatus_to_exitcode(status) == 0 and "result" in found:
                results[index] = found["result"]
            elif failed is None:
                failed = _failure(index, status, found)
                _kill(pids)
        if pids:
            time.sleep(POLL_S)
    if failed is not None:
        raise failed
    return [results[index] for index in range(count)]


def _read(path: Path) -> dict[str, Any]:
    try:
        found = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return found if isinstance(found, dict) else {}


def _failure(index: int, status: int, found: dict[str, Any]) -> WorkerFailed:
    """How worker ``index`` failed: what it raised, else how it died."""
    if "kind" in found:
        return WorkerFailed(
            index,
            f"worker {index} raised {found['kind']}: {found['message']}",
            kind=found["kind"],
            message=found["message"],
            trace=found.get("trace"),
        )
    if os.WIFSIGNALED(status):
        number = os.WTERMSIG(status)
        try:
            name = signal.Signals(number).name
        except ValueError:
            name = str(number)
        return WorkerFailed(index, f"worker {index} died of signal {name}")
    code = os.waitstatus_to_exitcode(status)
    return WorkerFailed(index, f"worker {index} exited with status {code}")


def _kill(pids: dict[int, int]) -> None:
    for pid in pids:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)


def _end(pids: dict[int, int]) -> None:
    """Kill the workers still running and wait on each: none is left."""
    _kill(pids)
    for pid in list(pids):
        with contextlib.suppress(ChildProcessError):
            os.waitpid(pid, 0)
        pids.pop(pid)
