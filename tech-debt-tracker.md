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
2. **What a stepping tick runs.** The updaters whose own code loads `dt`
   (`loads_dt`: any use of the local but a plain store or delete, an inner
   function closing over it, a read of its own frame; code it can't read
   counts as loading), judged as each is called (`skipping`, consulted by
   `Mobject.update`), so one that joins as the play runs is judged alike;
   the scene's updaters; the camera's inputs; every interpolation. No
   dt-free callback runs there, whoever reads what it writes.
3. **A play whose updaters a stepping tick could not leave out ticks
   everything** (`tick.unbounded_read`): the scene's own `update_mobjects`
   or `update_self`, a mobject's own `update`, a camera drawing other
   cameras. Checked when the play is planned and again whenever the
   structure (`current_structure`) differs from its plan's, before a
   stepping tick (`PlayRun.stepping_plan`). Test:
   `tests/timeline/test_stepping.py::test_a_scene_running_its_own_update_mobjects_ticks_everything`.
4. **What a dt reader of a dt-free value sees between frames** is that value
   as of the last full tick, as CE shows it once per frame. One class of it
   awaits the owner's ruling: a dt updater checking a dt-free updater's
   output against what interpolation moves fails between frames where
   ticking everything holds
   (`tests/timeline/test_stepping.py::test_break_awaiting_the_owner_a_dt_check_sees_a_dt_free_output_a_frame_old`,
   which flips if the class is ruled a break).
5. **The stepping run is held against the same semantics computed by brute
   force** (`arachne/timeline/reference.py`, `Recorder(stepping="brute_force")`,
   a verification mode): every scene of the stepping fixtures keeps the same
   captures and tick counts both ways
   (`tests/timeline/test_brute_force.py::test_stepping_keeps_what_brute_force_keeps`).
   The brute-force run makes and reads no plan and has its own `dt` judge
   (`::test_brute_force_never_enters_the_plan` patches planning, the
   stepping path's judge and the plan to raise); the two judges are held
   alike on the forms of `::test_brute_force_and_stepping_judge_dt_alike`.
6. **Hooks only observe** (`on_tick`, `on_step`): not enforced; see Hooks.

## The diagnostic analysis (`arachne/timeline/ticking.py`, `handed.py`, `coverage.py`)

- **The analysis of what callbacks reach and write is kept for diagnostics
  only, and carries what only the stepping plan it once fed used.** A
  comparison of two runs asks it (`TickPlan.stepping_families`, worked out
  the first time it is asked, and `reach`); nothing that plans or runs a
  play consults it, which
  `tests/timeline/test_brute_force.py::test_stepping_never_enters_the_analysis`
  holds by patching it to raise. Inside it, the tracking of which code may
  change the structure (`_Effect.restructures`, `unknown`, `_RESTRUCTURERS`)
  is computed and read by nothing, and `_Writes.plan_reads_hold` now only
  widens the families to every member when the analysis read a value of a
  mutable kind. Taking the structure tracking out means rewriting `_Writes`
  and `_Reach` throughout; worth it if the analysis is kept long, or moot
  if it is removed with its last diagnostic consumer.

## Hooks (`arachne/timeline/recorder.py`)

- **`on_tick` and `on_step` must only observe, and nothing enforces it.**
  They are diagnostics, called after each tick of a ticked play. A hook
  that writes what a dt reader reads, or what a dt-free callback left out
  would have written, leaves the frames undefined: neither those of
  stepping nor those of ticking everything. Enforcing it would take a
  digest of the scene at every tick; their one consumer is an instrument
  that only reads. `on_start` is safe: the play's ticks begin from whatever
  it left.

## Queries over the table (`arachne/timeline/coverage.py`)

- **`proven_pure` and `result_kept` answer from code, and miss.** Filters
  for precision only, with no use in the analysis. `proven_pure`
  answers `True` for a function reading an attribute of what an argument
  holds (unpacked, iterated, aliased: `n = m; n.shift(1)`), a closure's name
  another function rebinds (`nonlocal`), a global another module may
  rebind, and NumPy's input and output (`np.save`, `np.load`); on a cached
  function's `__wrapped__` it reads the cache's own decorator.
  `result_kept` answers `None` for an array a comparison makes
  (`np.array(t) > 0`) and a list a comprehension takes out of a result,
  each changed in place. It is conservative elsewhere: a helper returning
  `np.array(...).reshape(-1, 2)` is reported as letting its result out.
  Reproducers, strict expected failures:
  `tests/timeline/test_coverage.py::test_what_a_function_reads_beyond_its_arguments_is_not_pure`,
  `::test_a_result_changed_through_what_holds_it_is_kept`.

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
