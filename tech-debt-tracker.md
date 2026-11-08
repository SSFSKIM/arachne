# Tech debt

Known gaps kept on record rather than fixed, each with the test that
reproduces it; first, what the stepping tick rests on, which the gaps are
measured against.

## What a stepping tick rests on (`arachne/timeline/ticking.py`)

Not debt: the semantics of the stepping tick, each part with how it holds.

1. **Which ticks step.** A tick no frame of the grid reads, no event falls
   on, not the play's last, in a play without a stop condition
   (`PlayRun.stepping_ticks`, `recorder.py`). The others are full: every
   callback runs, in CE's phase order.
2. **What a stepping tick runs:** every interpolation (the functions of
   `UpdateFromFunc` and `UpdateFromAlphaFunc` among it), the camera's
   inputs (its fixed-orientation centre functions among them), and the
   callbacks whose own code loads the `dt` they are handed; nothing else.
   A mobject's updater is handed `dt` when its signature names it, as its
   second positional argument (`loads_dt`, as CE's `Mobject.update` calls
   it); a scene's updater always is, as its first (`scene_updater_loads_dt`,
   as CE's `Scene.update_self` calls it). Whatever the parameter receiving
   that argument is named, a callback counts as loading `dt` unless its own
   code has a positional parameter there and never reads it
   (`_handed_slot_loaded`, `_slot_loaded`: past a bound method's receiver;
   an argument landing in `*args` counts as read; the 3.13
   superinstructions decoded slot by slot; an inner function closing over
   it or a read of its own frame counts as reading). Each is judged as it
   is called (`skipping`, consulted by `Mobject.update` and
   `Scene.update_self`), so one that joins as the play runs is judged alike.
3. **A play whose updaters a stepping tick could not leave out ticks
   everything** (`tick.ungoverned`): a mobject's own `update`, a camera
   drawing other cameras. Checked when the play is planned and again when
   the structure (`current_structure`) differs from its plan's, before a
   stepping tick (`PlayRun.stepping_plan`): a play found ungoverned then
   ticks everything from that tick on; otherwise the tick steps by the new
   plan, a structure change alone making no tick full. The scene's own
   `update_to_time`, `update_mobjects` and `update_self` (on its class or
   its object) are frame hooks (`arachne/timeline/hooks.py`): they run at
   the full ticks a frame reads and a stepping tick runs the engine's
   bodies, so they make no tick full. Tests:
   `tests/timeline/test_soundness.py::test_a_mobject_whose_update_is_replaced_ticks`,
   `tests/timeline/test_stepping.py::test_a_scene_running_its_own_update_mobjects_steps_and_runs_it_at_frames`.
4. **What a dt reader of a dt-free value sees between frames** is that value
   as of the last full tick, as CE shows it once per frame: a dt-loading
   callback sees a dt-free output one frame old between frames (an approved
   break, 2026-11-03). So a dt updater checking a dt-free updater's output
   against what interpolation moves fails between frames where ticking
   everything holds; a check meant to read what the viewer sees reads at
   frames
   (`tests/timeline/test_stepping.py::test_approved_break_a_dt_check_sees_a_dt_free_output_a_frame_old`).
5. **The stepping run is held against the same semantics computed by brute
   force** (`arachne/timeline/reference.py`, `Recorder(stepping="brute_force")`,
   a verification mode): every scene of the stepping fixtures keeps the same
   captures and tick counts both ways
   (`tests/timeline/test_brute_force.py::test_stepping_keeps_what_brute_force_keeps`).
   The brute-force run makes and reads no plan and has its own judges of
   `dt` (`::test_brute_force_never_enters_the_plan` patches planning, the
   stepping path's judges and the plan to raise); the judges are held alike
   on the forms of `::test_brute_force_and_stepping_judge_dt_alike` and
   `::test_brute_force_and_stepping_judge_scene_updaters_alike`. Its limits
   are listed under "What the brute-force check does not check".
6. **Hooks only observe** (`on_tick`, `on_step`): not enforced; see Hooks.

## The stepping tick's known limits (`arachne/timeline/ticking.py`)

Each limit states its direction against the stated semantics: **runs every
tick** (conservative: the callback runs as when ticking everything, so no
integration is lost) or **skipped** (work the semantics runs between frames
is left out: not conservative). Forms that take deliberate construction are
recorded here rather than chased; where a probe scene exists it is kept as a
strict xfail in `tests/timeline/test_stepping.py`, passing once the limit is
lifted.

- **A callee reading its caller's frame is not seen. Direction: skipped, not
  conservative.** An updater that never loads `dt` but calls a helper reading
  `sys._getframe(1).f_locals["dt"]` is judged dt-free and runs at the full
  ticks only, while CE hands it `dt` at every frame and ticking everything at
  every tick: a real integrator loses its steps between frames. Only the
  updater's own code is read; following every callee for frame reads is not
  attempted. Met by no measured scene.
  (`::test_known_limit_a_callee_reading_its_callers_dt_integrates`, scene
  `CalleeReadsCallersDt`.)
- **A callable whose executed code is not certain counts as loading dt.
  Direction: runs every tick, conservative.** Only a plain function's code,
  or a bound method's whose function is plain, matched by exact type, is
  read (`_handed_slot_loaded`, `reference._own_code`). A `partial`, a
  callable object, a builtin, or a proxy exposing some function's
  `__code__` while its own `__call__` runs, counts as loading, so one with a
  dead `dt` runs at every tick, as ticking everything runs it, where the
  semantics would run it at frames only. Unwrapping a `partial`, or reading
  a callable object's `__call__`, is precision work.
  (`::test_known_limit_a_dead_dt_partial_runs_at_frames_only`, scene
  `DeadDtPartial`.)
- **Fallbacks that turn on mid-play are not all re-checked. Direction:
  skipped, not conservative.** The stated semantics ticks everything once a
  play's updaters may run where a stepping tick could not leave one out;
  missed, the play keeps stepping, its dt-free updaters running at frames
  only as in a governed play (whatever the replacement runs itself still
  runs at every tick). The stepping run re-checks the ungoverned cases only
  when the structure changes before a stepping tick: a mobject's `update`
  replaced on an instance mid-play, with no structure change, keeps the play
  stepping; the brute-force run re-checks at every tick, so the gate's
  comparison catches that case.
- **History-keeping interpolations run per 60 Hz tick where CE runs them
  per frame. Direction: runs every tick** (break 4, approved). The functions
  of `UpdateFromFunc` and `UpdateFromAlphaFunc` are interpolation and run at
  every tick: one that keeps history (appends, counts, integrates on its
  own) is called 60 times a second where CE calls it once a frame, twice
  CE's count at a 30 fps final, not only at draft rates. To measure: how
  often production scenes run a history-keeping one.
- **Per-tick costs left as they are. Direction: none** (cost, not frames).
  Each callback at a stepping tick costs an `inspect.signature` (a mobject
  updater, again in `Mobject.update` to choose the call form), a cache
  lookup by code object in `_slot_loaded`, and before every stepping tick a
  `current_structure` walk of every family. None is cached further: a
  signature depends on the callable (a wrapper, a `partial`, a
  `__signature__`), not on its code. To measure on production scenes before
  any is optimised.

## What the brute-force check does not check (`arachne/timeline/reference.py`)

- **It checks the optimisation, not conformance with CE.** Both runs follow
  the same stated semantics; a semantics that departs from CE departs in
  both. What the two runs share, and so cannot disagree on:
  - the assumption that a mobject updater is handed `dt` when its signature
    names it, as its second positional argument, and that a callback reads
    it only through its own code's positional parameter receiving it. The
    judges themselves are written apart (`ticking._handed_slot_loaded` and
    `reference._reads_argument` share no code or cache) and each follows the
    argument into the positional parameters of code it is certain the call
    runs (a plain function, a bound method of one, by exact type), so a
    `functools.wraps` wrapper, a proxy, or a dt landing in `*args`, counts as
    loading in both;
  - the list of names through which code reads its own frame
    (`_FRAME_ACCESS` and `_FRAME_READERS`, one list copied);
  - where events and stop conditions come from (`PlayRecord.events`,
    `Scene.stop_condition`) and which tick a frame reads (`Recorder.keeps`);
  - which plays are ungoverned (`admission._update_replaced`,
    `dispatch.user_overrides`, the camera's displays);
  - where the scene's frame hooks run (`Recorder.reads`, `PlayRun.frame`):
    both runs dispatch them at the same instants, which are full ticks in
    both.

## Hooks (`arachne/timeline/recorder.py`)

- **`on_tick` and `on_step` must only observe, and nothing enforces it.**
  They are diagnostics, called after each tick of a ticked play. A hook
  that writes what a dt reader reads, or what a dt-free callback left out
  would have written, leaves the frames undefined: neither those of
  stepping nor those of ticking everything. Enforcing it would take a
  digest of the scene at every tick; their one consumer is an instrument
  that only reads. `on_start` is safe: the play's ticks begin from whatever
  it left.

## Content keys (`arachne/timeline/record.py`)

- **A rounded key tells numbers apart by which side of a rounding boundary
  they fall on, not by how far apart they are.** `_value_key` rounds a
  value's floats to `decimals` and hashes the result, so two keys are equal
  exactly when every number rounds alike. Two coordinates one ulp apart
  (2.2e-16) on either side of a boundary key apart (`-0.9213813375` and
  `-0.9213813374999997` round to 9 decimals differently), while two 9e-10
  apart inside one bin key alike. `PlayRecord.sample_keys`
  (`SAMPLE_DECIMALS` = 9) decides with it whether a stepping run's captures
  match another run's. On one measured scene, differences of
  one ulp at a few play starts put glyph coordinates on a 10th-decimal
  half-way point, and rounding flipped some of them, so their nodes keyed
  apart. At 6, 7, 8, 10, 11 or 12 decimals none flip. An execution's
  `divergence` compares unrounded keys (`content_key()`), which tell apart
  any two values that differ at all. The planned restore work inherits the
  limit if it compares state by key. Every flip fails safe:
  - a sample-key comparison reports a difference stepping did not make;
  - a divergence is logged where nothing a frame reads moved;
  - a restore is refused and the scene runs forward.

  No key agrees where values differ by more than a bin's width. The fix:
  when two keys differ, compare the raw values the captures still hold,
  within 1e-9, before reporting a difference, and keep the key as the fast
  path for equality. Reproducer:
  `tests/timeline/test_sample_keys.py::test_a_sample_key_rounds_numbers_so_nearness_does_not_decide_equality`
  (it asserts today's behaviour).
- **A content key does not tell a re-pointed shared child apart.** The
  walk a key digests visits each mobject once, in the order it first
  meets it, and keys each node by its type, its number of submobjects and
  its values; which mobjects its children are is not in it. In
  `Group(Group(a, b), Group(a))`, pointing the second group at `b` keeps
  every count and value, so the key stays the same though the tree
  changed. That deduplicated first-occurrence order is also the draw
  order, so for a tree with shared children equal keys do not imply an
  equal draw order. Every comparison by key inherits the limit:
  `RegisteredState.content_key`, `PlayRecord.sample_keys`, and an
  execution summary's state key (`execution.py`), which the comparison of
  two executions trusts when keys are equal. So would any reuse of draws
  by key, and until the fix lands nothing may reuse draws by content key.
  The probe compares each node's children by their positions in the walk
  (`probe._tree_difference`), so it is not affected. The change to the
  record's content key now in progress closes it: each node's children
  digested as positions in the walk. The shape it misses:
  `tests/timeline/test_probe.py::test_a_re_pointed_shared_child_is_a_tree_difference`.

## The draw route (`arachne/timeline/dispatch.py`)

- **A refused draw costs the scene a second execution.** A class whose
  drawing calls the scene's code (`draw_refusal`: an override of a draw
  method or a drawn protocol, `__deepcopy__` and `__hash__` among them)
  refuses the record, and the scene is executed again, drawn as it runs
  from the first refused play. A scene that ticks every play then pays for
  its ticks twice: its record run and its live run. Reproducer:
  `tests/timeline/test_draw_refusals.py::test_every_play_from_the_first_refused_on_is_drawn_as_the_scene_runs`
  (`execution.number == 2`). The fix is to keep the first execution's
  plays up to the refusal, or to know the refusal before the first run
  (the class's definitions are readable once the scene module is loaded).

## Execution summaries (`arachne/timeline/execution.py`)

- **A supplied camera background is hashed again at every capture
  summarised.** `summarize` digests `CameraExtras.background` (a full
  picture, read-only) once per state it walks, though a scene that set it
  once shares one array across every capture. Cache the digest by the
  array's identity before summaries are measured at corpus scale.

## The forward fallback (`arachne/timeline/fallback.py`)

- **A record refusing drawing from its first play is not compared with its
  forward run at all.** A play drawn as it runs may write what it draws
  into the scene, from its static picture on (a zoomed display's picture,
  the points of what a `MappingCamera` with `allow_object_intrusion`
  maps), which the record's execution, drawing nothing, never holds; so
  only the plays before the first such play are compared, and a changed
  input goes unreported there (`ForwardRun.compared_plays`; a
  `MappingCamera` scene compares none,
  `test_a_mapping_camera_scene_answers_its_grid_instants_with_one_execution`).
  Direction: compare in full, per refusal reason, for cameras whose drawing
  writes nothing into the scene (a `MappingCamera` that copies what it
  maps).

## Scene hooks (`arachne/timeline/hooks.py`)

- **A frame hook's effects are recorded where a frame reads them, not
  between.** A trusted play's analytic answer between its samples
  interpolates its animations from the last sample and runs no hook, so
  what a hook writes shows there as of that sample. Direction: none; a
  render's frames are all samples.
- **At a rate that does not divide 60 a ticked play's hook sees the tick a
  frame reads,** up to 1/60 s before the frame (`t` and the clock are the
  tick's), and above 60 fps one hook serves several frames.
- **A stop condition that first holds at a tick a frame reads ends the play
  there, after that tick's hooks ran**: the hook's `t` is the play's own
  length, where CE's frame lies a frame inside its longer play. CE also
  calls the hooks at the frame its condition first holds.
- **An execution abandoned for re-execution is not torn down**, as CE tears
  down no run that raised: what its `setup` opened stays open
  (`tests/timeline/test_draw_refusals.py::test_a_retry_tears_down_only_the_execution_that_completes`).
- **A play's cache key is computed before `play_internal` runs, as in CE.**
  A run time the scene chooses inside the hook (its own `get_run_time`
  answering otherwise the second time, an animation lengthened there) is
  outside the cache identity, so a warm cache hit reuses the segment an
  earlier render wrote under that key, whatever its length, an empty one
  included. Direction: key the segment on the span the play's body takes,
  which CE would need as well.
- **A hook with effects outside the scene (a print, a file) runs once per
  execution**, so twice for a scene executed again.

## The core's depth keys (`core/src/points.rs`, `core/src/project.rs`)

- **A point within about 1e-6 of an off-grid eye can key behind it.** A
  projected point takes the key of the camera-facing plane through it
  (`FacingPlanes::at`, called from `core/src/points.rs` where a projected
  point's depth is chosen). The plane's offset is rounded to the shared grid,
  so for a point that close to an eye off that grid the rounded plane can
  cross the eye, and the point keys as if behind the camera. No known scene puts
  a point there, and no test reproduces it yet. Direction: when
  the rounded plane crosses the eye, fall back to the point's own key, as
  `at` already does when no shared plane applies.

## Live frames in XF (`arachne/frame_ir/live.py`)

- **A live play's static picture and its frames share one stream, and each
  capture prunes the other's records.** `LiveFrames` draws a frozen play's
  static picture (`begin`) and then its frames through the same camera, so
  the camera's live stream sees the two kinds of capture alternate;
  `keep_only` keeps only the slots the last capture drew, and the next one
  uploads the other's records again. The frames are right (each delta is
  whole against the stream), but the uploads repeat, which costs the core's
  no-draw path the work the IR is meant to save. The fix is a stream
  per capture role (static, frames), as `painter_for` keeps for record
  drawing. Reproducer: a live scene with a frozen wait over static
  mobjects, its deltas kept with `IrCairoPainter.keep_all` (the static
  records set again in each frame's delta).

## Replaced camera methods (`arachne/timeline/camera_code.py`)

- **The replaced-camera detector covers every engine camera member.**
  `replaced_methods` counts any member of an engine camera class the
  scene's code replaced (on the class or the instance), not only the
  methods a draw calls, so a replaced member that never affects drawing
  still sends the camera to Cairo (`render.fallback`) or, where a record
  reads the detector, refuses it. A concrete case: an instance patch of
  `display_multiple_vectorized_mobjects` falls back, though the camera
  never calls it through the instance (`display_funcs` binds the method
  when the camera is made). Narrow it to the display path's methods only
  if a case is ever refused over a member that does not affect drawing
  (the refusal names the member).

## Restored frames on a described path (`arachne/timeline/draw.py`)

- **A restore on `ir_cairo` repaints from scratch.** A restored frame is
  painted by a painter of its own (`Drawing.restored`), so that nothing a
  restore builds stays in the record's stream; it extracts and uploads the
  whole frame each time. That is about 4x per restore on image-heavy scenes
  (92 ms against 22.5 ms at 854x480) and about 1x in 3D. Remedy: cache image
  uploads across restores (they are immutable), or give the restore a
  copy-on-write view of the record's stream.

## The image cache (`arachne/mobject/svg/svg_mobject.py`)

- **An image built from the cache is not the image parsing built.** Inherited from Manim Community Edition: a
  cache hit adds copies of the cached parts, but the image itself never gets parsing's `flip(RIGHT)`, so its
  own `sheen_direction` stays unflipped (its parts match). Invisible unless a sheen is set on the image, and
  within one process the first of two equal images differs from the second, as in CE. Each execution now
  finds the construction caches (`arachne/mobject/construction_caches.py`: this one and a number's glyph
  cache, which fills it) as a fresh process (a session) or as the scene's first execution (a render executed again)
  found it, so two executions of one source agree. The TeX construction memo (`arachne/mobject/text/tex_memo.py`)
  keeps this: whether the cache holds a formula's image is part of its key, and a hit on an image the cache lacks
  puts the cache's entry back. Direction: none while CE's behaviour is the reference.

## The TeX construction memo (`arachne/mobject/text/tex_memo.py`)

- **What its key does not cover.** Every argument, the configured template, the image (path and content), the
  image cache's state and the code and defaults of the engine classes the construction made or holds are keyed. Not
  keyed: an engine module's function replaced by the scene (`tex_mobject.tex_to_svg_file = f`), and a class the
  scene defines whose name is an engine class's, which shares that class's image cache entries (the image cache keys
  by class name, as in CE). Direction: none unless a scene is found doing either.
- **Its bound is an entry count.** 512 entries. An entry is the formula's mobjects plus the image cache's entry:
  about 0.2 MiB for a one-line formula and 2 MiB for a large one (a 4×4 matrix with a sum and an integral), measured
  as deep size on CPython 3.13. Full of large formulas, the memo would hold about 1 GiB. Direction: bound it by size
  when sessions run under a memory budget.
- **A hit still costs one deep copy.** What it hands out is a copy of the prototype (it must be, so that the caller's
  changes never reach it), and the copy dominates a hit: each path's parsed `svgelements` objects are copied with it.
  Direction: cheaper copies of mobjects (below).

## Speed: measured and not built

Measured on a sample of scenes, each opened in a session, edited to make its last play 0.5 s longer, and executed
again; "the floor" is the time from the edit to the new record.

- **The deep copies in a play's `begin`.** Taken across every route (`Transform.begin`, `create_starting_mobject`,
  fades, `Write`'s outlines), they are about 11–14% of the floor by the per-scene median, under the bar for a lever of
  their own. Every deep copy, whoever asks for it, is about 20% by the median. Direction: make mobject copies cheaper
  rather than fewer.
- **`DrawBorderThenFill` builds its outline twice.** `__init__` builds it (`arachne/animation/creation.py:243`) and
  `begin` builds it again (`:252`), which wastes about 1% of the floor. Direction: build it once, in `begin`, once
  nothing reads the first.
- **Fast-forward of trusted plays.** After their values are vectorized, evaluating trusted plays is about 19% of the
  floor by the per-scene median and 12% pooled. About half of that is the play's `begin`, so a fast-forward that
  still runs `begin` saves little. Not built.

## The session server (`arachne/server/`)

- **A render at another pixel aspect than its session's execution is not checked.** The camera keeps its configured
  `frame_width` and derives `frame_height` from the pixel aspect (`Camera.resize_frame_shape`, fixed dimension
  0): at 854x480 the frame is 7.99375 high against 16:9's 8, so the execution summaries differ in
  `camera.frame_height` (by 0.00625) while mobject geometry, placed by the configuration's frame, does not. A
  render or clip at a quality whose aspect differs from every execution kept answers `self_check: unchecked`
  (`tests/session/test_render.py::test_a_render_with_nothing_like_it_to_check_against_says_so`). Direction:
  compare mobject states in scene units, leaving the camera's frame out, so one execution checks every quality.
- **The store has no disk budget.** `arachne/server/store.py` keeps every manifest, answer and summary it
  writes, under each engine fingerprint it has seen, and never prunes; an engine upgraded leaves the old
  fingerprint's directory behind whole. Direction: a size budget with least-recently-read eviction, and old
  fingerprints removed.
- **One unexplained failure of the unread-answers test under load.**
  `tests/session/test_server.py::test_a_client_that_stops_reading_cannot_hold_the_server` (100 `inspect`s
  sent, none read, on stdio) failed once on a heavily loaded machine and has not failed since; the cause is
  not known. Direction: if it recurs, keep the server's standard error and the exit timing of that run.
- **An in-process session call swaps standard output for the whole process.** `arachne/cli/client.py`'s
  `InProcess.call` runs a request under `contextlib.redirect_stdout(sys.stderr)`, so the scene's prints stay off
  the command's output. That swap is process-wide: another thread printing during the call writes to standard
  error. The CLI is safe, because `asking.ask` captures standard output before it calls, and nothing else calls
  `InProcess`. Direction: route the scene's output through the session's own stream, not `sys.stdout`, before
  any programmatic or multi-threaded caller uses the in-process path. Also, each `InProcess` starts a watchdog
  daemon thread that never ends, since the watchdog has no close; it is harmless for one command but leaves
  one idle thread per in-process use.
- **A server's start is not bounded by its caller's deadline.** `connect_or_start` waits up to `START_S` (60 s)
  for a started server's ready line (`registry._ready_line`), whatever is left of `deadline_s`; the open after it
  then gets only what is left, or `ClientTimeout`. Bounding the wait itself would leave a server that comes ready
  after its caller gave up, unreachable to end, since its pid is known only from its ready line. Direction: give
  the server its caller's deadline at its start (an environment stamp on a clock both share) and have it exit
  unregistered when it comes ready past it; then bound the wait by the deadline.
- **A line's arrival is stamped late while the scene holds the interpreter.** The server's reading thread
  (`serve._reading`) stamps each line as it reads it; a scene running a long C call that keeps the GIL delays that
  read, and the line's deadline starts that much late. Direction: read on a thread that never needs the GIL for
  the stamp (or in a helper process), if a scene's long native calls ever make the delay matter.
- **A forked child of the scene's code outlives its server, holding its lock.** A server's `<pid>.alive` lock is
  non-inheritable across exec (PEP 446) and the engine's subprocesses run with `close_fds`, but a child forked
  without exec by the scene's own code (`os.fork`, `pty.fork`, multiprocessing's "fork" start method, Linux's
  default through 3.13) shares the lock's open file and holds it after the server exits; the server's `os._exit`
  also skips the cleanup that would end leaked multiprocessing workers. Since liveness checks the server's own pid
  first (`registry._serving`), a dead server's entry is forgotten at once all the same; what remains is the leaked
  workers themselves, and a pid reused by another process while the child still holds the lock, which keeps the
  entry until the child ends. Direction: end the scene's children on the way out, and check the entry's pid against
  the lock's holder (or take the lock in a process the scene can't fork from), if a scene that forks long-lived
  children turns up.
- **The reading thread's queue has no bound.** `serve._reading` queues every line read while a request runs, so a
  client flooding lines grows the server's memory without limit. Direction: bound the queue and stop reading a
  channel while it is full, so the flood waits in the socket instead.
- **A server's configuration digest is taken as it registers, not as the engine read its files.**
  `Registration` digests `config_file_paths()` after the engine's configuration was parsed at import, so a file
  edited between the two leaves the server keyed by the edit while it runs under the earlier text. Direction:
  digest the files where `arachne._config` reads them, and key the server by that digest.
- **A server of an earlier version, with no `.alive` lock, is forgotten by a newer registry reader.** Liveness is
  read from `<pid>.alive`, which servers before it never made, so after an upgrade a newer client sweeps a still
  running older server's entry and starts a second server beside it; the older one idles out. Direction: none
  while servers are short-lived; a version field in the entry if mixed versions ever share a registry for long.
- **Polish.** `arachne/server/registry.py`'s module docstring wraps one sentence oddly near line 44, and a
  `.alive-XXXX` temporary file a server left between making its lock and naming it (a crash in that instant) is
  never swept. Direction: rewrap; sweep `.alive-*` files that are old and unlocked.

## Configuration (`arachne/_config/utils.py`)

- **A misspelt key in a config file is ignored.** Setting an unknown option from code raises (deliberate break 7:
  as an attribute, an item, in `tempconfig` or a session's configuration), but an unknown key in a `.cfg` file's
  `[CLI]` section is read by nothing and passes silently, as in Manim Community Edition: `vectorise_trusted = False`
  leaves `vectorize_trusted` on. Only the removed options are refused there (`tests/test_config_removed.py`).
  Direction: refuse an unknown `[CLI]` key, naming the nearest option, once the keys that sections legitimately carry
  beside the options (plugins' among them) are listed.
