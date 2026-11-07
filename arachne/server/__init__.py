"""The session server: one scene's :class:`~arachne.session.Session`, served
to clients outside the process. This is the protocol's description, version
1.0; a method or field added is a minor version, one changed or removed a
major one.

Starting a server
-----------------

A server is one process holding at most one session::

    python -m arachne.server --stdio  [--deadline-s S]
    python -m arachne.server --socket [--deadline-s S]

It speaks JSON-RPC 2.0 on its standard input and output (``--stdio``), or
on any number of connections to a Unix socket (``--socket``). Requests are
answered one at a time, in the order read, whichever connection sent them.
What it reads from its environment:

- ``ARACHNE_SESSION_DIR``: where the socket goes, as ``<pid>.sock``; made,
  mode 0700, when missing. Unset, the socket goes in a private temporary
  directory of the server's own, removed when it ends. The socket's whole
  path must fit a Unix socket's address (104 bytes on macOS, 108 on Linux,
  its terminating NUL included), so a directory named here is best kept
  under about 90 bytes: a path too long ends the server at its start, with
  status 1 and no ready line. A server on a socket there keeps its entry in
  the host's registry beside it, ``<pid>.json`` (:mod:`.registry`).
- ``ARACHNE_SESSION_IDLE_S``: see The server's end (600 by default).
- ``ARACHNE_SESSION_DEADLINE_S``: see Deadlines (600 by default;
  ``--deadline-s`` comes first).
- ``ARACHNE_SESSION_MEMORY_MB``: the memory the servers of its
  ``ARACHNE_SESSION_DIR`` may hold together, in megabytes; see The server's
  end (no limit by default).
- ``ARACHNE_STORE``: the store of answers (:mod:`.store`), else the user's
  cache directory's ``arachne/store``.

The two times, and ``--deadline-s``, are seconds above 0 and at most a week
(604800), and the memory megabytes above 0; the server refuses any other
value at its start, with status 2.

A client finds the server holding a session in the registry, or starts one
there, with :func:`.registry.connect_or_start`. A server it starts writes its
standard error, the scene's prints and the engine's logging among them, to
``<pid>.log`` beside its entry, never to the caller's; it outlives the
command that started it until its idle end, a ``stop`` or ``stop_all``.

Lines
-----

Each message is one line: a JSON object, UTF-8, ended by a newline. ``NaN``,
``Infinity`` and ``-Infinity`` are not JSON, nor is a number too large to be
finite, nor anything nested too deep to read. A message is at most 64 MiB
(:data:`~.protocol.MAX_MESSAGE_BYTES`), its line's end aside: a longer line
is answered ``-32600`` with a null id, and dropped as it is read. A line is
read until its end comes; a line whose end never comes is waited for no
longer than the server's idle time, which only an answer resets.

The ready line
--------------

Once it serves, the server writes one line to its standard output, a
notification:

.. code-block:: json

    {"jsonrpc": "2.0", "method": "ready",
     "params": {"protocol": "1.0", "pid": 4242, "transport": "socket",
                "socket": "/path/4242.sock"}}

``socket`` is given for the socket transport only. With ``--socket``, the
standard output holds the ready line only and is closed after it; with
``--stdio``, it holds the protocol's lines only. Anything else written to
standard output (a scene's prints, the engine's logging) goes to standard
error.

Requests
--------

A request is ``{"jsonrpc": "2.0", "id": ..., "method": ..., "params":
{...}}``: an id that is a string or a finite number (or null), a method,
and params by name, an object; params may be left out when there are none.
A param given as null is not given. A line with a string method and no
``id`` is a notification: it is run, and never answered, whatever is wrong
with it. Batches are not served.

A time is seconds (a number) or the exact instant, a string a
:class:`~fractions.Fraction` reads (``"n/d"``, as every answer's ``_exact``
fields write it): ``t``, ``t0``, ``t1`` and the times ``at`` lists. Every
other number (``fps``, ``sample_fps``, ``t_range``'s ends, ``deadline_s``)
is seconds, or a rate, as a number only. Paths are strings; a relative
one is relative to the server's working directory. Every method takes
``deadline_s`` too (see Deadlines).

==============  ========================================  ===========================
method          params, (default) when optional           result, beside generation
==============  ========================================  ===========================
``open``        ``file``, ``scene``, ``config`` ({})      ``record``
``inspect``                                               ``record``
``state_at``    ``t``                                     ``state``
``lint``        ``fps`` (4), ``t_range`` (none),          ``lint_result``, and
                ``exempt`` ([]), ``at``                   ``key_frames`` with
                ("held+sampled"), ``rules`` ("armed"),    ``key_frames_dir``
                ``sample_fps`` (none),
                ``key_frames_dir`` (none)
``query``       ``t``, ``what``, ``ids`` ([])             ``query``
``frame``       ``t``, ``path``, ``quality`` (none)       ``frame``
``key_frames``  ``out_dir``, ``quality`` (none)           ``key_frames``
``reload``      ``force`` (false)                         ``reloaded``, a boolean
``close``
``ping``
``shutdown``
``clip``        ``t0``, ``t1``, ``quality`` (none),       ``clip``
                ``out``
``render``      ``plays`` (all), ``quality`` (none),      ``render``
                ``out``
==============  ========================================  ===========================

A param with no default is required.

Every result is an object holding ``generation``, the generation of the
session the answer is of (:attr:`~arachne.session.Session.generation`; null
with no session open), and the forms named above, each as
:mod:`arachne.api_json` writes it and as its schema in ``arachne/schemas/``
describes it.

- ``open`` executes the scene class ``scene`` (a string) of ``file`` under
  ``config``, an object of the configuration's options to override
  (:meth:`~arachne.session.Session.open`). Asked again for the session it
  holds (the same file, by its real path, scene and config), it answers
  that session; for another it refuses (``-32004``): ``close`` it first.
  A session the store holds (:mod:`.store`), its files unchanged by digest,
  is opened without executing its scene: every answer the store holds of it
  is given from there, and the first request whose answer it does not hold
  executes the scene, as the generation it is (the next one when its files
  changed). Answers so given are the answers the session gave. The store
  keeps a session under all its execution was asked with (the file's real
  path, the scene, the config, the working directory's real path) and the
  engine that executed it (the interpreter's version, the engine's source
  and its native core's build); its files are every file the execution
  read, as its module loaded included.
- ``state_at`` and ``frame`` answer as
  :meth:`~arachne.session.Session.answers` does: from the record, else from a
  forward run. ``frame`` writes the PNG ``path`` (its directory made when
  missing), at ``quality``, a key of :data:`~arachne.constants.QUALITIES`,
  else at the session's pixel size.
- ``lint`` lints as :func:`arachne.lint.lint` does. ``fps`` (seconds' rate,
  above 0) is the rate of its instants. ``t_range`` is ``[start, end]`` in
  seconds, start not after end, both ends included: a finding is kept when it
  holds at some time in it; none keeps all. ``exempt`` lists exemptions, as
  :mod:`arachne.lint.exempt` reads them: a selector is a string, the stable
  id ``"path:line#n"``, ``"line:N"`` (the scene file's line that made it)
  or ``"name:<variable>"`` (the scene's name for it); a bare selector
  exempts every finding about an object it selects, a pair ``[a, b]`` of
  selectors a finding about exactly two objects, one selected by each.
  ``at`` is ``"held+sampled"`` or a list of times, the frames on screen
  there. ``rules`` is ``"armed"`` (the temporal rules the scene armed) or a
  list of ``"hold_watch"`` and ``"departures"``, each armed from the scene's
  start; ``[]`` runs the scene's own checks only. ``sample_fps`` (above 0)
  is the rate the armed hold watch samples at, given with ``"armed"`` only.
  ``key_frames_dir`` writes the key frames there too, at the session's pixel
  size. A finding is an answer, never an error.
- ``query``'s ``what`` is one of :data:`~arachne.api_json.QUERY_ANSWERS`, or
  a list of them; ``ids`` are the stable ids ``bbox`` boxes, required with
  it.
- ``key_frames`` writes the key frames into the directory ``out_dir``, at
  ``quality``, else the session's pixel size.
- ``reload`` executes the scene again when its files changed, or with
  ``force``, as the session's next generation
  (:meth:`~arachne.session.Session.reload`).
- ``close`` lets go of the session; the server goes on, and can open
  another. With none open it does nothing.
- ``ping`` answers; ``shutdown`` answers, then the server ends.
- ``clip`` and ``render`` render the session's scene. Renders are serial,
  as every request is.

  - Each executes the scene afresh through Arachne's render path, at
    ``quality`` (its pixel size and frame rate; else the session's), into
    the media file ``out`` (an MP4 unless ``config`` names another format),
    and checks itself against the execution summary the session holds of
    its source: ``{generation, clip | render: {path, timeline, self_check,
    divergence?}}``, the ``render`` form. ``timeline`` is every play the
    render's execution played, in the record's form, those ``out`` leaves
    out ``skipped``. ``self_check`` is ``{status, reason}``: ``"checked"``
    (reason null), or ``"unchecked"`` with why, so that a render with no
    divergence is a clean comparison only when it is checked.
    ``divergence`` (the ``replay.diverged`` the check found) is given only
    when it found one; a render is written all the same. The summary
    checked against is that of the session's latest lint (the store's, else
    the session's own), of an execution at the render's frame rate and
    pixel aspect: a render at another frame rate or aspect is another
    execution, and is unchecked. Another pixel aspect is another execution
    because the camera keeps its configured frame width and derives its
    frame height from the aspect (854 by 480 gives 7.99375 against 16:9's
    8): the camera's frame differs while mobject geometry, placed by the
    configuration's frame, does not. Checking across aspects, by comparing
    mobject states in scene units, is not done yet.
  - ``clip`` renders the plays that meet ``[t0, t1]``, ``t0`` not after
    ``t1``: from the first play whose span reaches ``t0`` (it ends at or
    after it) to the last whose span reaches back to ``t1`` (it starts at
    or before it), both included, so a play that touches the span at one
    instant is in it; the clip runs from the first one's start through the
    last one's end. A span no play meets is refused. ``render`` renders the
    plays ``plays`` lists, by index, which must be a contiguous ascending
    run (``[2, 3, 4]``; the render path takes a first and a last play), else
    every play; any other list, gaps, repeats or descents, is refused
    ``-32602``, saying so.
  - ``out`` is the file the caller binds, whatever it holds: it is written
    as a temporary file beside it, renamed onto ``out`` only once the render
    succeeded, and only once the request, still live, has claimed its
    answer from the deadline's watchdog: the rename and the deadline's error
    are one or the other. So a deadline's error (whenever it comes, the
    moment before the rename included), a refusal or an error leaves
    ``out`` as it was, a file there before untouched. Claimed, the request's
    deadline still bounds the rename and the answer's writing: a deadline
    passing then ends the server without the error, and since the rename is
    atomic, ``out`` is the file before or the render, never a part of one.
    ``out`` naming a directory is refused.
  - Each sets its own ``deadline_s`` from what it expects to cost: the
    server's default does not fit a render. A caller gives what is left of
    its own monotonic deadline, less a small margin, so that the server ends
    before the caller gives up and no render outlives its caller. The test
    that pins both: a request that runs past its deadline leaves nothing at
    ``out``, and the server's process is gone before the caller returns.

The session is caught up with its files before every request but
``open``, ``reload``, ``close``, ``ping`` and ``shutdown``: once the
request's params are read and checked, when the scene's source or a file it
read holds anything else, by digest, the scene is executed again, and the
answer is of the new generation. ``open`` catches up when it answers the
session it holds, ``reload`` is the catching up, and ``close``, ``ping``
and ``shutdown`` read nothing of the session's files.

Errors
------

=========  =================================================================
code       meaning, and ``data``
=========  =================================================================
``-32700`` the line is not JSON: no ``data``
``-32600`` the line is not a request, or past the longest message (a batch
           is not served): no ``data``
``-32601`` no such method: no ``data``
``-32602`` a param is missing, unknown, or not of its kind (a time that is
           not one, ``1/0`` among them; a quality
           :data:`~arachne.constants.QUALITIES` lacks; a rule, answer or
           exemption the method does not know; ``t_range`` backwards;
           ``sample_fps`` beside rules other than ``"armed"``; ``bbox``
           asked without ``ids``; a ``deadline_s`` past a week; ``plays``
           not a contiguous ascending run): no ``data``
``-32603`` the server failed unforeseen, or its answer could not be
           written as JSON: ``data`` is ``{type, message, traceback}``, the
           traceback trimmed to its last 8 frames
``-32000`` an error was raised through the scene's code (whatever it was,
           an exit, ``SystemExit``, too) while opening, reloading or
           answering, a forward run's included: ``data`` is the
           ``scene_error`` form (:class:`~arachne.errors.SceneError`), its
           ``t`` the scene clock at the raise whenever the execution had
           one, its ``report`` the error as it went through the scene's
           code, its ``raised_by`` whose code raised it (``"scene"`` or
           ``"engine"``, by the traceback's innermost frame). That the
           error went through the scene's code does not by itself make it
           the scene's fault. The server goes on.
``-32001`` the engine refused the request, by one of its refusals: a scene
           hook a play can't honour (``SceneHookRefused``), a record that
           answers no instant (``RecordRefused``), an instant it can't answer
           (``Unavailable``, ``Unrestorable`` among them), a frame the drawer
           does not draw (``Unsupported``, ``render.unsupported``), two
           executions that can't be compared (``Incomparable``). ``data`` is
           ``{type, message, reason}``: the refusal's class name, its message
           and its stable code, or null
``-32002`` the request ran past its deadline: ``data`` is ``{method,
           deadline_s}``; the server ends
``-32003`` no session is open: no ``data``
``-32004`` ``open`` asked for another scene, file or config than the session
           the server holds: no ``data``
=========  =================================================================

Deadlines
---------

Every request has a deadline: its ``deadline_s`` (seconds above 0, at most
a week) when it gives one, longer or shorter than the server's; else the
server's, ``--deadline-s``, else ``ARACHNE_SESSION_DEADLINE_S``, else 600.
Its clock starts when the request's line has been read whole, so a request
queued behind another counts its wait, and it runs to its answer's
writing, so a client that stopped reading holds the server no longer than a
scene that never ends. A thread of its own keeps it (:mod:`.watchdog`):
whatever the scene's code is doing, a request past its deadline is answered
``-32002`` (unless it had begun writing its own answer: a request is never
answered twice), and the process exits with status 3. A request already
past its deadline when its turn comes does not run. A client that gives up
on a request needs nothing else to stop the work it asked for.

The server's end
----------------

The server ends by itself when no request has been answered for
``ARACHNE_SESSION_IDLE_S`` seconds, on ``shutdown``, and, with ``--stdio``,
at the end of its standard input, each with status 0; past a deadline with
status 3; with status 1 if it fails. However it ends, each step on its way
out (a deadline's error sent, what a render running wrote removed, its
socket and registry entry removed, a word on standard error, its streams
flushed) takes at most 1 s (:data:`~.protocol.ENDING_S`), and then the
process exits whatever that step did: no client, full pipe or file system
can keep it alive.

A server on a socket in ``ARACHNE_SESSION_DIR`` is ended by another
process only once it is reserved for its end, ``<pid>.lock`` beside its
entry: the server holds that itself while it answers a request, so a server
answering is never ended under it, nor killed. A reserved server answers
nothing more: the next request it reads it drops, unanswered, and it ends
with status 0; its caller sees the connection end (a client of
:mod:`.registry` raises ``ServerGone``, and ``connect_or_start`` starts the
session afresh).

Under ``ARACHNE_SESSION_MEMORY_MB``, such a server, once it has answered,
ends the least recently used of the directory's other servers it can reserve
while together they hold more than that: it asks each to ``shutdown``, then
kills it if it has not ended within a few seconds (:mod:`.registry`). Still
over the budget with no other server to end, it ends itself, its answer
sent, with status 0. Neither this nor the idle end waits on a client.
"""

from __future__ import annotations

__all__ = ["PROTOCOL_VERSION"]

from .protocol import PROTOCOL_VERSION
