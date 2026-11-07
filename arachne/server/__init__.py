"""The session server: one scene's :class:`~arachne.session.Session`, served
to clients outside the process.

A server is one process holding at most one session. It is started as
``python -m arachne.server --stdio`` or ``--socket`` (with ``--deadline-s``
optionally, see Deadlines) and speaks JSON-RPC 2.0,
one JSON message per line, UTF-8, on its standard input and output
(``--stdio``) or on any number of connections to a Unix socket
(``--socket``). Requests are answered one at a time, in the order read.

The ready line
--------------

Once it serves, the server writes one line to its standard output, a
notification:

.. code-block:: json

    {"jsonrpc": "2.0", "method": "ready",
     "params": {"protocol": "1.0", "pid": 4242, "transport": "socket",
                "socket": "/path/4242.sock"}}

``socket`` is given for the socket transport only. The socket is
``<pid>.sock`` in ``ARACHNE_SESSION_DIR`` when that is set (made, mode 0700,
when missing), else in a private temporary directory of the server's own;
the server removes the socket, and its own directory, when it ends. With
``--socket``, the standard output holds the ready line only; with
``--stdio``, the protocol's lines only. Anything else written to standard
output (a scene's prints, the engine's logging) goes to standard error.

Requests
--------

Params are named (an object); a param given as null is not given. Times
are seconds, or exact instants as the string ``"n/d"``. Every method takes
``deadline_s`` too: see Deadlines.

======================  ========================================  =========================
method                  params                                    result, beside generation
======================  ========================================  =========================
``open``                ``file``, ``scene``, ``config`` ({})      ``record``
``inspect``                                                       ``record``
``state_at``            ``t``                                     ``state``
``lint``                ``fps`` (4), ``t_range``, ``exempt``,     ``lint_result``, and
                        ``at``, ``rules`` ("armed"),              ``key_frames`` when
                        ``sample_fps``, ``key_frames_dir``        ``key_frames_dir`` is given
``query``               ``t``, ``what``, ``ids`` ([])             ``query``
``frame``               ``t``, ``path``, ``quality``              ``frame``
``key_frames``          ``out_dir``, ``quality``                  ``key_frames``
``reload``              ``force`` (false)                         ``reloaded``, a boolean
``close``
``ping``
``shutdown``
======================  ========================================  =========================

Every result is an object holding ``generation``, the generation of the
session the answer is of (:attr:`~arachne.session.Session.generation`; null
with no session open), and the forms named above, each as
:mod:`arachne.api_json` writes it and as its schema in ``arachne/schemas/``
describes it.

- ``open`` executes ``scene`` of ``file`` (relative to the server's working
  directory) under ``config``, the configuration's options to override
  (:meth:`~arachne.session.Session.open`). Asked again for the session it
  holds, it answers that session; for another, it refuses (``-32004``):
  ``close`` it first.
- ``state_at`` and ``frame`` answer as
  :meth:`~arachne.session.Session.answers` does: from the record, else from a
  forward run. ``frame`` writes the PNG ``path``, at ``quality`` (a key of
  :data:`~arachne.constants.QUALITIES`), else at the session's pixel size.
- ``lint`` lints as :func:`arachne.lint.lint` does. ``rules`` is ``"armed"``
  (the temporal rules the scene armed) or a list of ``"hold_watch"`` and
  ``"departures"``, each armed from the scene's start; ``[]`` runs the
  scene's own checks only. ``sample_fps`` is the rate the armed hold watch
  samples at. ``at`` is ``"held+sampled"`` (the default) or a list of times.
  A finding is an answer, never an error.
- ``query``'s ``what`` is one or more of
  :data:`~arachne.api_json.QUERY_ANSWERS`, ``ids`` the stable ids ``bbox``
  boxes.
- ``key_frames`` writes the key frames into ``out_dir``.
- ``reload`` executes the scene again when its files changed, or with
  ``force``, as the session's next generation
  (:meth:`~arachne.session.Session.reload`).
- ``close`` lets go of the session; the server goes on, and can open
  another.
- ``ping`` answers; ``shutdown`` answers, then the server ends.

Before answering any request but ``open``, ``reload``, ``close``, ``ping`` and
``shutdown``, the server reloads the session: when the scene's source or a
file it read holds anything else, by digest, the scene is executed again,
and the answer is of the new generation. A request without an ``id`` is a
notification, run and never answered.

Errors
------

=========  =================================================================
code       meaning, and ``data``
=========  =================================================================
``-32700`` the line is not JSON
``-32600`` the line is not a request (a batch is not served)
``-32601`` no such method
``-32602`` a param is missing, unknown, or not of its kind (a quality
           :data:`~arachne.constants.QUALITIES` lacks, an exemption the lint
           can't read, ``sample_fps`` beside rules other than ``"armed"``,
           ``bbox`` asked without ``ids``)
``-32603`` the server failed unforeseen: ``data`` is ``{type, message,
           traceback}``, its traceback trimmed to its last 8 frames; or the
           answer could not be written as JSON, with no ``data``
``-32000`` the scene's code raised, whatever it raised, an exit
           (``SystemExit``) too, while opening, reloading or answering:
           ``data`` is the ``scene_error`` form
           (:class:`~arachne.errors.SceneError`), its ``report`` the error as
           it went through the scene's code. The server goes on.
``-32001`` the engine refused the request, by one of its refusals: a scene
           hook a play can't honour (``SceneHookRefused``), a record that
           answers no instant (``RecordRefused``), an instant it can't answer
           (``Unavailable``, ``Unrestorable`` among them), a frame the drawer
           does not draw (``Unsupported``, ``render.unsupported``), two
           executions that can't be compared (``Incomparable``). ``data`` is
           ``{type, message, reason}``: the refusal's class name, its message
           and its stable code, or null
``-32002`` the request ran past its deadline: ``data`` is ``{method,
           deadline_s}``; the server ends once it is sent
``-32003`` no session is open
``-32004`` ``open`` asked for another scene, file or config than the session
           the server holds
=========  =================================================================

Deadlines and the server's end
------------------------------

Every request has a deadline: its ``deadline_s`` (seconds above 0) when it
gives one, longer or shorter, else the server's: ``--deadline-s`` when
started with it, else ``ARACHNE_SESSION_DEADLINE_S``, else 600 s. It counts
from the line's reading to its answer's writing, so a client that stopped
reading holds the server no longer than a scene that never ends. A thread of
its own keeps it (:mod:`.watchdog`): a request still running at its
deadline is answered with ``-32002``, for at most 1 s of trying to send
it, and the process exits with status 3, whatever the scene's code is
doing. A client that gives up on a request needs nothing else to stop the
work it asked for.

The server ends by itself when no request has come for
``ARACHNE_SESSION_IDLE_S`` seconds (600 by default), on ``shutdown``, and,
with ``--stdio``, at the end of its standard input, each with status 0.
"""

from __future__ import annotations

__all__ = ["PROTOCOL_VERSION"]

from .protocol import PROTOCOL_VERSION
