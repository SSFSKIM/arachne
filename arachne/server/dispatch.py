"""What each method of the session server does, read from its params and
answered as :mod:`arachne.api_json`'s forms.

:data:`METHODS` is the table the server dispatches by: a method's
:class:`Method` names what runs it, whether it needs an open session, and
whether the session is brought up to date with its files first, once its
params are read (:meth:`Params.done`). A method is added by adding its
entry.

With a :class:`~.store.Store`, a dispatcher keeps the answers it gives, and
opens a session the store holds, its files unchanged, without executing
it: it answers from the store until a request needs what the store does
not hold, and executes the scene then (:class:`Dispatcher`).
"""

from __future__ import annotations

__all__ = [
    "METHODS",
    "REFUSALS",
    "Dispatcher",
    "Method",
    "Params",
    "deadline",
    "internal_error",
]

import contextlib
import json
import math
import os
import shutil
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from .. import api_json
from ..constants import QUALITIES
from ..errors import SceneError, explain, message_of
from ..frame_ir.cairo_painter import Unsupported
from ..lint.exempt import Exemptions
from ..lint.instants import DEFAULT as LINT_AT
from ..session import Session
from ..timeline.execution import Incomparable
from ..timeline.hooks import SceneHookRefused
from ..timeline.routes import RecordRefused, Unavailable
from . import render as rendering
from .protocol import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    MAX_DEADLINE_S,
    METHOD_NOT_FOUND,
    NO_SESSION,
    OTHER_SESSION,
    REFUSED,
    SCENE_ERROR,
    TRACEBACK_FRAMES,
    RpcError,
)
from .store import Manifest, Store, identity_form, summary_key

_REQUIRED: Any = object()


class Params:
    """A request's params, read one by one. Every method reads them all
    before it does anything, then calls :meth:`done`, which refuses any it
    did not read, then runs ``then`` (the session's reload, for a method
    answered from the files as they are now).
    """

    def __init__(
        self, given: dict[str, Any], then: Callable[[], object] | None = None
    ) -> None:
        self._given = given
        self._read: set[str] = set()
        self._then = then

    def get(
        self, name: str, read: Callable[[str, Any], Any], default: Any = _REQUIRED
    ) -> Any:
        """Param ``name``, as ``read(name, value)`` reads it; ``default`` when
        not given (or given as null), and required when there is none.
        """
        self._read.add(name)
        value = self._given.get(name)
        if value is None:
            if default is _REQUIRED:
                raise _invalid(f"{name} is required.")
            return default
        return read(name, value)

    @property
    def given(self) -> dict[str, Any]:
        """The params as given, ``deadline_s`` aside."""
        return dict(self._given)

    def done(self) -> None:
        unknown = sorted(set(self._given) - self._read)
        if unknown:
            raise _invalid(f"No param {', '.join(map(repr, unknown))}.")
        if self._then is not None:
            self._then()


def _invalid(message: str) -> RpcError:
    return RpcError(INVALID_PARAMS, message)


def text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise _invalid(f"{name} is a non-empty string.")
    return value


def flag(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise _invalid(f"{name} is true or false.")
    return value


def number(name: str, value: Any) -> float:
    """A finite number, as a float: an integer too large for one is none."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _invalid(f"{name} is a number.")
    try:
        found = float(value)
    except OverflowError:
        found = math.inf
    if not math.isfinite(found):
        raise _invalid(f"{name} is a finite number.")
    return found


def positive(name: str, value: Any) -> float:
    found = number(name, value)
    if found <= 0:
        raise _invalid(f"{name} is a number above 0.")
    return found


def instant(name: str, value: Any) -> Fraction | float:
    """A time: seconds, or the exact instant as a string, ``"n/d"`` (or any
    rational :class:`~fractions.Fraction` reads), finite as a float.
    """
    if not isinstance(value, str):
        return number(name, value)
    try:
        found = Fraction(value)
        float(found)
    except (ValueError, ZeroDivisionError, OverflowError):
        raise _invalid(f"{name} is seconds or an exact instant, 'n/d'.") from None
    return found


def instants(name: str, value: Any) -> list[Fraction | float]:
    if not isinstance(value, list):
        raise _invalid(f"{name} is a list of times.")
    return [instant(name, each) for each in value]


def span(name: str, value: Any) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise _invalid(f"{name} is [start, end], in seconds.")
    start, end = (number(name, each) for each in value)
    if start > end:
        raise _invalid(f"{name} is [start, end], start not after end.")
    return start, end


def array(name: str, value: Any) -> list[Any]:
    if not isinstance(value, list):
        raise _invalid(f"{name} is a list.")
    return value


def mapping(name: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _invalid(f"{name} is an object.")
    return value


def deadline(params: dict[str, Any]) -> tuple[float | None, dict[str, Any]]:
    """A request's ``deadline_s``, in seconds, and its other params."""
    rest = dict(params)
    given = rest.pop("deadline_s", None)
    if given is None:
        return None, rest
    found = positive("deadline_s", given)
    if found > MAX_DEADLINE_S:
        raise _invalid(f"deadline_s is at most {MAX_DEADLINE_S:g} s.")
    return found, rest


def _rules(name: str, value: Any) -> Any:
    """``"armed"``, or the rules a list names, each armed from the start."""
    from ..lint.departures import departures
    from ..lint.watch import hold_watch

    named = {"hold_watch": hold_watch, "departures": departures}
    if value == "armed":
        return value
    if not isinstance(value, list) or not all(
        isinstance(each, str) and each in named for each in value
    ):
        raise _invalid(f"{name} is 'armed' or a list of {', '.join(map(repr, named))}.")
    return tuple(named[each] for each in value)


def _answers(name: str, value: Any) -> list[str]:
    asked = [value] if isinstance(value, str) else value
    if (
        not isinstance(asked, list)
        or not asked
        or not all(
            isinstance(each, str) and each in api_json.QUERY_ANSWERS for each in asked
        )
    ):
        raise _invalid(f"{name} is one or more of {', '.join(api_json.QUERY_ANSWERS)}.")
    return asked


def _ids(name: str, value: Any) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(e, str) for e in value):
        raise _invalid(f"{name} is a list of stable ids.")
    return value


@dataclass(frozen=True)
class Method:
    """A method: ``run(dispatcher, params)`` answers it. ``session``: it
    asks about the open session. ``fresh``: once its params are read, the
    session is executed again when its files changed (:meth:`Session.reload`).
    """

    run: Callable[[Dispatcher, Params], dict[str, Any]]
    session: bool = True
    fresh: bool = True


class Dispatcher:
    """The session a server holds, and what each request asks of it.

    With a ``store``, the session is held one of two ways. Executed, as
    :attr:`session`: each answer it gives that the store can keep is kept,
    by the request and the execution it is of, and its manifest with it
    (:class:`~.store.Manifest`), with the execution summary of its latest
    lint. Or, opened from the store, as :attr:`stored`, the manifest of a
    session the store holds whose files are unchanged: each request the
    store holds the answer of is answered from it, and the first that it
    does not hold (or a file changed) executes the scene, as the session's
    next generation when its files changed.
    """

    def __init__(self, store: Store | None = None) -> None:
        self.session: Session | None = None
        #: The session opened from the store and not executed, if it is.
        self.stored: Manifest | None = None
        self.store = store
        #: What the open session was opened as: its file, scene and config;
        #: the file as ``open`` named it, and its config.
        self._opened: tuple[str, str, str] | None = None
        self._named = ""
        self._config: dict[str, Any] = {}
        #: The manifest the store keeps of the session's execution, if any.
        self._manifest: Manifest | None = None
        #: The generation of the session opened from the store.
        self._stored_generation = 1
        #: The key of the summary kept of the session's execution, if any.
        self._summarised: str | None = None
        #: The directory a render writes into while it runs.
        self._scratch: Path | None = None
        #: Whether a request asked the server to end, once it answered.
        self.ending = False

    def handle(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """The result of ``method`` asked with ``params``; raises
        :class:`RpcError` for its error.
        """
        found = METHODS.get(method)
        if found is None:
            raise RpcError(METHOD_NOT_FOUND, f"No method {method!r}.")
        holding = self.session is not None or self.stored is not None
        if found.session and not holding:
            raise RpcError(NO_SESSION, "No session is open: open one first.")
        # The params read and checked, the session catches up with its files.
        then = self._catch_up if found.fresh and holding else None
        try:
            return found.run(self, Params(params, then))
        except RpcError:
            raise
        except BaseException as error:
            # Whatever the scene's code raised, its exit too, is an answer:
            # the session it ran in is resident. What the server itself
            # raised unforeseen is an answer too; anything else ends it.
            scene_file = None if self._opened is None else self._opened[0]
            answer = _error_answer(error, scene_file)
            if answer is None:
                raise
            raise answer from error

    def answer(self, **forms: Any) -> dict[str, Any]:
        """A result: ``forms``, with the generation of the session they are
        of (null with none open).
        """
        return {"generation": self.generation, **forms}

    @property
    def generation(self) -> int | None:
        """The generation of the session held, ``None`` with none."""
        if self.session is not None:
            return self.session.generation
        return self._stored_generation if self.stored is not None else None

    @property
    def held(self) -> tuple[str, str, dict[str, Any]] | None:
        """The session held: its file (a real path), scene and config."""
        if self._opened is None:
            return None
        return self._opened[0], self._opened[1], dict(self._config)

    def live(self) -> Session:
        """The session, executed: a session opened from the store is
        executed now, as the generation it is.
        """
        if self.session is None:
            self._execute(next_generation=False)
        assert self.session is not None
        return self.session

    def _execute(self, *, next_generation: bool) -> None:
        """Execute the session opened from the store, as its next generation
        when ``next_generation``.
        """
        assert self.stored is not None
        session = Session.open(
            self._named, self.stored.scene, config=self._config or None
        )
        session.generation = self._stored_generation + int(next_generation)
        self.session, self.stored = session, None
        self._executed()

    def _catch_up(self) -> None:
        """Execute the session again when its files changed."""
        if self.session is not None:
            if self.session.reload():
                self._executed()
        elif self.stored is not None and not self.stored.fresh():
            self._execute(next_generation=True)

    def _executed(self, record: dict[str, Any] | None = None) -> None:
        """Keep the manifest of the execution the session now holds,
        ``record`` its record's form if made already.
        """
        self._manifest = self._summarised = None
        if self.store is None or self._opened is None or self.session is None:
            return
        file, scene, _ = self._opened
        if record is None:
            record = api_json.inspect(self.session)
        with contextlib.suppress(OSError):
            self._manifest = self.store.keep_manifest(
                file, scene, self._config, self.session, record
            )

    def record(self) -> dict[str, Any]:
        """The record's form of the session held."""
        if self.session is not None:
            return api_json.inspect(self.session)
        assert self.stored is not None
        return {**self.stored.record, "generation": self.generation}

    def kept(
        self,
        method: str,
        given: dict[str, Any],
        answer: Callable[[Session], dict[str, Any]],
    ) -> dict[str, Any]:
        """The result of ``method`` asked with ``given``: from the store while
        the session is opened from it and the store holds the answer; else
        ``answer(session)``'s forms, kept in the store.
        """
        if self.session is None and self.store is not None:
            assert self.stored is not None
            found = self.store.answer(self.stored, method, given)
            if found is not None:
                return self.answer(**found)
        forms = answer(self.live())
        self.keep(method, given, forms)
        return self.answer(**forms)

    def keep(
        self,
        method: str,
        given: dict[str, Any],
        forms: dict[str, Any],
        file: bytes | None = None,
    ) -> None:
        """Keep the answer ``forms`` to ``method`` asked with ``given`` (and
        ``file``, written with it) as of the execution the session holds.
        """
        manifest, session = self._manifest, self.session
        if self.store is None or manifest is None or session is None or session.stale:
            return
        with contextlib.suppress(OSError):
            self.store.keep_answer(manifest, method, given, forms, file)

    def keep_summary(self, session: Session) -> None:
        """Keep the summary of the session's execution, which its lint was
        of, as the latest lint's, once per execution.
        """
        if self.store is None or self._opened is None or session.stale:
            return
        file, scene, _ = self._opened
        key = summary_key(
            file, scene, identity_form(session.record.identity), self._config
        )
        if key == self._summarised:
            return
        with contextlib.suppress(OSError):
            self.store.keep_summary(key, session.summary())
            self._summarised = key

    def spans(self) -> list[tuple[Fraction, Fraction]]:
        """Each play's span, by index, of the session held."""
        if self.session is not None:
            return [(play.start, play.end) for play in self.session.record.plays]
        assert self.stored is not None
        return [
            (Fraction(play["start_exact"]), Fraction(play["end_exact"]))
            for play in self.stored.record["plays"]
        ]

    def render(
        self, quality: str | None, plays: tuple[int, int] | None, out: Path
    ) -> rendering.Rendered:
        """Render the session's scene afresh, ``plays`` of it (first and
        last) or every play, at ``quality``, to ``out``, checked against the
        summary of the latest lint of the session's execution at the
        render's frame rate and pixel aspect: the store's, else the session's
        own when it was executed so.
        """
        assert self._opened is not None
        file, scene, _ = self._opened
        if self.session is not None:
            identity = identity_form(self.session.record.identity)
        else:
            assert self.stored is not None
            identity = self.stored.identity
        config = {**self._config, **rendering.quality_options(quality)}
        key = summary_key(file, scene, identity, config)
        reference = None if self.store is None else self.store.summary(key)
        if (
            reference is None
            and self.session is not None
            and key == summary_key(file, scene, identity, self._config)
        ):
            reference = self.session.summary()
        return rendering.render(
            file,
            scene,
            self._config,
            named=self._named,
            quality=quality,
            plays=plays,
            out=out,
            reference=reference,
            scratch=self._scratching,
        )

    def _scratching(self, directory: Path | None) -> None:
        self._scratch = directory

    def abandon(self) -> None:
        """Remove what a render running now has written: the server ends."""
        if self._scratch is not None:
            shutil.rmtree(self._scratch, ignore_errors=True)


#: The engine's deliberate refusals, answered as ``-32001`` with their type:
#: a scene hook a play can't honour (``SceneHookRefused``), a record that
#: answers no instant (``RecordRefused``), an instant it can't answer
#: (``Unavailable``, ``Unrestorable`` among them), a frame the drawer does
#: not draw (``Unsupported``, ``render.unsupported``) and two executions that
#: can't be compared (``Incomparable``).
REFUSALS: tuple[type[Exception], ...] = (
    SceneHookRefused,
    RecordRefused,
    Unavailable,
    Unsupported,
    Incomparable,
)


def _error_answer(error: BaseException, scene_file: str | None) -> RpcError | None:
    """``error``'s answer: the engine's refusal; else what the scene's code
    raised, explained, as the session explained it or, for an execution it
    did not explain (a forward run's), by the frames of the scene's own
    code ``error`` went through (of ``scene_file``, the session's); else,
    for an :class:`Exception`, an internal error. ``None``: none of these, so
    not the server's to answer.
    """
    kind = type(error).__name__
    if isinstance(error, REFUSALS):
        reason = getattr(error, "reason", None) or getattr(error, "code", None)
        message = message_of(error)
        return RpcError(
            REFUSED,
            f"{kind}: {message}",
            {
                "type": kind,
                "message": message,
                "reason": reason if isinstance(reason, str) else None,
            },
        )
    explained = getattr(error, "scene_error", None)
    if explained is None and scene_file is not None:
        found = explain(error, scene_file)
        explained = found if found.frames else None
    if isinstance(explained, SceneError):
        message = (
            f"{explained.cls}: {explained.message}"
            if explained.message
            else explained.cls
        )
        return RpcError(SCENE_ERROR, message, api_json.scene_error(explained))
    if not isinstance(error, Exception):
        return None
    return internal_error(error)


def internal_error(error: BaseException) -> RpcError:
    """The server's own failure, ``error``: ``INTERNAL_ERROR`` with its type,
    message and traceback, trimmed to its last
    :data:`~.protocol.TRACEBACK_FRAMES` frames.
    """
    kind, message = type(error).__name__, message_of(error)
    try:
        trace = "".join(
            traceback.format_exception(
                type(error),
                error,
                error.__traceback__,
                limit=-TRACEBACK_FRAMES,
                chain=False,
            )
        )
    except BaseException:
        trace = f"{kind}: {message}\n"
    return RpcError(
        INTERNAL_ERROR,
        f"{kind}: {message}",
        {"type": kind, "message": message, "traceback": trace},
    )


# The methods


def _open(server: Dispatcher, params: Params) -> dict[str, Any]:
    file = params.get("file", text)
    scene = params.get("scene", text)
    config = params.get("config", mapping, {})
    params.done()
    real = os.path.realpath(file)
    opened = (real, scene, json.dumps(config, sort_keys=True))
    if server.session is None and server.stored is None:
        manifest = None
        if server.store is not None:
            with contextlib.suppress(OSError):
                manifest = server.store.manifest(real, scene, config)
        if manifest is not None and manifest.fresh():
            server.stored = manifest
            server._stored_generation = 1
        else:
            server.session = Session.open(file, scene, config=config or None)
        server._opened, server._named, server._config = opened, file, config
        if server.session is None:
            return server.answer(record=server.record())
        record = api_json.inspect(server.session)
        server._executed(record)
        return server.answer(record=record)
    if opened != server._opened:
        assert server._opened is not None
        held_file, held_scene, held_config = server._opened
        raise RpcError(
            OTHER_SESSION,
            f"This server holds {held_scene} of {held_file}, config"
            f" {held_config}: close it before opening another.",
        )
    server._catch_up()
    return server.answer(record=server.record())


def _inspect(server: Dispatcher, params: Params) -> dict[str, Any]:
    params.done()
    return server.answer(record=server.record())


def _state_at(server: Dispatcher, params: Params) -> dict[str, Any]:
    t = params.get("t", instant)
    params.done()

    def answer(session: Session) -> dict[str, Any]:
        (found,) = session.answers([t], frames=False)
        return {"state": api_json.state(found.state, session)}

    return server.kept("state_at", params.given, answer)


def _lint(server: Dispatcher, params: Params) -> dict[str, Any]:
    options: dict[str, Any] = {
        "fps": params.get("fps", positive, 4),
        "t_range": params.get("t_range", span, None),
        "exempt": params.get("exempt", _exempt, ()),
        "at": params.get("at", _at, None),
        "rules": params.get("rules", _rules, "armed"),
        "sample_fps": params.get("sample_fps", positive, None),
    }
    if options["sample_fps"] is not None and options["rules"] != "armed":
        raise _invalid("sample_fps is the rate of the armed rules: rules='armed'.")
    out_dir = params.get("key_frames_dir", text, None)
    params.done()

    def answer(session: Session) -> dict[str, Any]:
        result = session.lint(**options)
        server.keep_summary(session)
        return {"lint_result": api_json.lint(result)}

    if out_dir is None:
        return server.kept("lint", params.given, answer)
    # Key frames are files the store does not keep.
    session = server.live()
    forms = answer(session)
    forms["key_frames"] = api_json.key_frames(session.key_frames(out_dir))
    return server.answer(**forms)


def _at(name: str, value: Any) -> str | list[Fraction | float]:
    if isinstance(value, str):
        if value != LINT_AT:
            raise _invalid(f"{name} is {LINT_AT!r} or a list of times.")
        return value
    return instants(name, value)


def _exempt(name: str, value: Any) -> list[Any]:
    """Exemptions as the lint reads them: selectors, and pairs of two."""
    try:
        Exemptions.parse(array(name, value))
    except (ValueError, TypeError) as found:
        raise _invalid(f"{name}: {found}") from None
    return list(value)


def _quality(name: str, value: Any) -> str:
    if not isinstance(value, str) or value not in QUALITIES:
        raise _invalid(f"{name} is one of {', '.join(QUALITIES)}.")
    return value


def _query(server: Dispatcher, params: Params) -> dict[str, Any]:
    t = params.get("t", instant)
    what = params.get("what", _answers)
    ids = params.get("ids", _ids, [])
    if "bbox" in what and not ids:
        raise _invalid("A query's bbox needs ids, the mobjects to box.")
    params.done()
    return server.kept(
        "query",
        params.given,
        lambda session: {"query": api_json.query(session.query(t), what, ids=ids)},
    )


def _frame(server: Dispatcher, params: Params) -> dict[str, Any]:
    t = params.get("t", instant)
    path = Path(params.get("path", text))
    quality = params.get("quality", _quality, None)
    params.done()
    # The answer is the same picture wherever it is written.
    asked = {name: value for name, value in params.given.items() if name != "path"}
    store, stored = server.store, server.stored
    if server.session is None and store is not None and stored is not None:
        found_form = store.answer(stored, "frame", asked)
        picture = store.answer_file(stored, "frame", asked)
        if found_form is not None and picture is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(picture)
            return server.answer(frame={**found_form["frame"], "path": os.fspath(path)})
    (found,) = server.live().answers([t], quality)
    assert found.frame is not None
    path.parent.mkdir(parents=True, exist_ok=True)
    found.frame.save(path, format="PNG")
    form = api_json.frame(found, path)
    server.keep("frame", asked, {"frame": form}, path.read_bytes())
    return server.answer(frame=form)


def _key_frames(server: Dispatcher, params: Params) -> dict[str, Any]:
    out_dir = params.get("out_dir", text)
    quality = params.get("quality", _quality, None)
    params.done()
    found = server.live().key_frames(out_dir, quality)
    return server.answer(key_frames=api_json.key_frames(found))


def _reload(server: Dispatcher, params: Params) -> dict[str, Any]:
    force = params.get("force", flag, False)
    params.done()
    if server.session is None:
        assert server.stored is not None
        changed = not server.stored.fresh()
        if changed or force:
            server._execute(next_generation=True)
        return server.answer(reloaded=changed or force)
    reloaded = server.session.reload(force)
    if reloaded:
        server._executed()
    return server.answer(reloaded=reloaded)


def _close(server: Dispatcher, params: Params) -> dict[str, Any]:
    params.done()
    if server.session is not None:
        server.session.close()
    server.session = server.stored = server._opened = server._manifest = None
    return server.answer()


def _out(name: str, value: Any) -> Path:
    """A media file to write: a path naming no directory."""
    found = Path(text(name, value))
    if found.is_dir():
        raise _invalid(f"{name} names a directory: name the media file to write.")
    return found


def _plays(name: str, value: Any) -> tuple[int, int]:
    """Plays by index, in order, one after another: the first and last."""
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(i, int) and not isinstance(i, bool) for i in value)
    ):
        raise _invalid(f"{name} is a list of play indices.")
    first = value[0]
    if first < 0 or value != list(range(first, first + len(value))):
        raise _invalid(
            f"{name} is plays one after another, in order, as a render takes"
            f" them: {value} is not."
        )
    return first, value[-1]


def _clip(server: Dispatcher, params: Params) -> dict[str, Any]:
    t0 = params.get("t0", instant)
    t1 = params.get("t1", instant)
    if t0 > t1:
        raise _invalid("A clip's t0 is not after its t1.")
    quality = params.get("quality", _quality, None)
    out = params.get("out", _out)
    params.done()
    spans = server.spans()
    plays = rendering.plays_meeting(spans, t0, t1)
    if plays is None:
        end = float(spans[-1][1]) if spans else 0.0
        raise _invalid(
            f"No play meets [{float(t0):g}, {float(t1):g}] s: the scene runs {end:g} s."
        )
    return server.answer(clip=api_json.render(server.render(quality, plays, out)))


def _render(server: Dispatcher, params: Params) -> dict[str, Any]:
    plays = params.get("plays", _plays, None)
    quality = params.get("quality", _quality, None)
    out = params.get("out", _out)
    params.done()
    if plays is not None:
        count = len(server.spans())
        if plays[1] >= count:
            raise _invalid(f"The scene has {count} plays, 0 to {count - 1}.")
    return server.answer(render=api_json.render(server.render(quality, plays, out)))


def _ping(server: Dispatcher, params: Params) -> dict[str, Any]:
    params.done()
    return server.answer()


def _shutdown(server: Dispatcher, params: Params) -> dict[str, Any]:
    params.done()
    server.ending = True
    return server.answer()


#: The methods, by name.
METHODS: dict[str, Method] = {
    "open": Method(_open, session=False, fresh=False),
    "inspect": Method(_inspect),
    "state_at": Method(_state_at),
    "lint": Method(_lint),
    "query": Method(_query),
    "frame": Method(_frame),
    "key_frames": Method(_key_frames),
    "clip": Method(_clip),
    "render": Method(_render),
    "reload": Method(_reload, fresh=False),
    "close": Method(_close, session=False, fresh=False),
    "ping": Method(_ping, session=False, fresh=False),
    "shutdown": Method(_shutdown, session=False, fresh=False),
}
