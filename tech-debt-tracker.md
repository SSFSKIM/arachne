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
