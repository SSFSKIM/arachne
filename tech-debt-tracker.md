# Tech debt

Known gaps kept on record rather than fixed, each with the test that
reproduces it; first, what the stepping check's soundness rests on, which
the gaps are measured against.

## What a stepping plan rests on (`arachne/timeline/ticking.py`)

Not debt: the assumptions a stepping plan makes, each with how it holds:
*observed* (read when the plan is made, or at each stepping tick), *closed*
(by a value's kind or by the operation table, with the test that enforces
it) or *fallback* (the play ticks everything, or the callback is not
skipped). A stepping tick runs the stepping work (dt updaters that read
`dt`, scene updaters, camera inputs), every interpolation, and the dt-free
callbacks the plan pulls; it skips every other dt-free callback, which runs
at the full ticks (each frame's, each event's, the last), as CE runs it.

1. **Which ticks step.** Observed: a tick no frame of the grid reads, no
   event falls on, not the last, in a play without a stop condition
   (`PlayRun.stepping_ticks`, `recorder.py:119`).
2. **What a skipped callback writes is read by no work the tick runs.**
   `_closure` (`ticking.py:4648`) skips a dt-free callback only when what it
   may write (`_Node.written`, `ticking.py:4452`) meets nothing the
   stepping work reads, directly or through what it pulls. What it may
   write is closed to: its own mobject's family, whose members every
   stepping tick compares (entry 6), and the arrays its changes go into,
   by the memory each holds (an array's memory is fixed; a write in place
   to one not resolved writes every array). Anything else is a fallback: a
   change into any other object, or anything it holds, or writes bounded
   only by what the callback reaches, raise `_UnboundedWrites`
   (`_effect_writes`, `ticking.py:4468`), and the callback is pulled.
   Tests: `tests/timeline/test_plan_reads.py::test_a_receiver_rebound_after_the_plan_is_made_is_written`,
   `::test_a_callable_stored_after_the_plan_is_made_is_called`.
3. **The plan rests on no value of a mutable kind.** Fallback: every value
   a proof reads when the plan is made, beyond a binding (entry 5) and an
   inert value, is noted (`_Site.resolve`, `ticking.py:2191`: an object's
   own attribute; `handed.Callers.held_by`: what an object, a list, a dict
   holds), and `_Writes.plan_reads_hold` (`ticking.py:1592`, called at
   `ticking.py:4747`) makes the play tick everything when there is any,
   whatever could change it (a store, an alias, a function of the engine's
   or `operator`, a lambda, a property). A mobject's member list is
   exempt: entry 6 observes it. Tests: `test_plan_reads.py` (every scene
   of `plan_read_scenes.py`), `test_stepping.py::test_state_changed_after_the_plan_is_made_ends_what_it_read`.
4. **What a callback runs is read from its code, all of it.** Closed by the
   table (`coverage.OPERATIONS`, `_Site.read_table` at `ticking.py:3041`):
   every node class has a rule, and one the table does not name is a
   fallback (`test_coverage.py::test_every_node_class_has_a_rule`,
   `::test_a_form_without_a_rule_falls_back`). Code Python runs without a
   call in the source is a call: iterating a value (`_iterating`,
   `ticking.py:3300`; `test_plan_reads.py::test_iterating_a_generator_runs_its_code`),
   an operator's, a truth test's, a formatting's, a key's or an attribute
   read's method of an object whose class the scene defines (`_protocol`,
   `ticking.py:3096`, `_attribute_read`, `ticking.py:3142`;
   `test_plan_reads.py::test_what_python_runs_on_an_object_of_the_scenes_class_is_called`),
   an item of a base not shown to run no code (`_item_code`,
   `ticking.py:3435`), a callable handed to a call, and a value handed that
   is not shown free of callables, which makes the call unknown whatever
   its callee (`_handed_callables`, `ticking.py:2738`;
   `test_stepping.py::test_a_value_not_shown_free_of_callables_is_an_unknown_call`).
   A parameter's value is, by that rule, shown free of callables when its
   caller hands it; the value it falls back to (a default, the object a
   method is called on) is judged as it is (`_Site.fallback`;
   `[DefaultedGeneratorRead]`, `[DefaultedOperatorHookRead]`).
   Entering a `with` and awaiting are unknown calls. Code the check does
   not read (past `MAX_DEPTH`, or without source) is a fallback
   (`_Writes.effect`, `ticking.py:1674`, then `plan_reads_hold`). An
   unknown call writes whatever the callback reaches: by entry 2, the
   callback is pulled.
5. **A binding the plan reads stays bound to what it was.** Closed by the
   table: a closure's cell changes only by `nonlocal` (a fallback unless
   every value bound is inert; `test_stepping.py::test_state_changed_after_the_plan_is_made_ends_what_it_read[LateNonlocalRead…]`)
   or through `cell_contents` or `__closure__`; a module's global by
   `global` (the same rule), a store into an attribute of that name
   (`_rebinds`, `ticking.py:3264`: on a module named directly, a fallback;
   on any other object, a fallback when the plan read a global or an
   attribute through a class of that name) or a namespace builtin
   (`globals`, `vars`, `setattr`, `exec`; scene `GlobalsItemRead`);
   a function's default only through `__defaults__` or `__kwdefaults__`
   (`ReboundDefaultWrite`); a class's attribute, a method among them, by a
   store into the class named directly (`ReboundMethodWrite`,
   `ClassAttributeRead`), by the name rule above, or through `__dict__`,
   `__class__` or `__bases__`. Every namespace attribute and builtin is a
   fallback (`_Site._namespace`, `ticking.py:3082`;
   `coverage.NAMESPACE_ATTRIBUTES`, `NAMESPACE_BUILTINS`;
   `ReboundCellWrite`; the scenes named here are
   `plan_read_scenes.py`'s, each a case of `test_plan_reads.py`).
6. **The structure is what the plan was made for.** Observed at each
   stepping tick: the scene's updaters, the roots, every mobject's updaters
   and every family's members are compared with the plan's
   (`current_structure`, `ticking.py:369`, in `PlayRun.stepping_plan`,
   `recorder.py:136`), and the play plans again first when they differ,
   unless the change lies where no stepping work reads or writes
   (`unaffected`, `ticking.py:393`; when what a running callback writes
   can't be bounded, every family member counts as written,
   `_stepped_ids`, `ticking.py:4788`). Work the tick runs that may change
   the structure within the tick is a fallback (`tick.structural_mutation`).
   Tests: `test_stepping.py::test_a_skipped_callback_that_changes_the_structure_is_never_stepped_past`,
   `::test_structure_changed_before_a_stepping_tick_plans_the_play_again`.
7. **What stepping work reads is all it reads** (`reach`). Closed by kind,
   default-deny: closure cells, globals by binding, defaults, the mobject an
   updater is handed with its family, containers followed whole, objects
   through the attributes the code names, arrays by memory; anything it
   can't follow is unbounded, a fallback for work that runs. Read when the
   plan is made, and it stays complete as the play runs: a value can come
   to be read only by a store, the code storing it reaches it, and that code
   is stepping work (its reads join what is needed) or a callback pulled for
   writing into an object (entry 2) or a binding (entry 5, a fallback).
8. **The engine's code is modelled, not read through the table.** Closed by
   kind, assumptions of the model: the engine's methods change their own
   mobject's family and arrays (an animation, `trusted.writes`); its code
   calls what it is handed only where `handed.Callers` reads that it does,
   reading the engine's source with every unresolved callee taken to call;
   a native function the check lists as never calling is handed only
   values shown free of callables (entry 4); the engine's results are of
   the engine's or native kinds; the engine rebinds no binding of the
   scene's. A mobject's updaters are the plan's own work, not held
   callables.
9. **Hooks only observe** (`on_tick`, `on_step`): not enforced; see Hooks.

The static proofs over a function's code stand for nothing in the plan:
`coverage.proven_pure` and `coverage.result_kept` have no use in the
stepping check (their known limits are below), and a function of the
scene's is never taken to return an inert value (`_Site._inert_call`,
`ticking.py:2990`): its result is a value the plan may not observe (see the
runtime-check item under planned precision work).

## Hooks (`arachne/timeline/recorder.py`)

- **`on_tick` and `on_step` must only observe, and nothing enforces it.**
  They are diagnostics, called after each tick of a ticked play, between
  two stepping ticks whose plan nothing re-checks for what a hook changed. A
  hook that writes what a stepping reader reads, or what a skipped callback
  would have written, leaves the frames undefined: neither those of
  stepping nor those of ticking everything. Enforcing it would take a
  digest of the plan's reach at every tick; their one consumer is an
  instrument that only reads. `on_start` is safe: the play plans its ticks
  after it, from whatever it left
  (`tests/timeline/test_stepping.py::test_a_hook_that_rebinds_what_a_reader_reads_is_planned_for`).

## Planned precision work (`arachne/timeline/ticking.py`)

Work to do, not debt to wait on: each item leaves a play ticking everything
(as CE does: precision, not soundness) where stepping would be sound.

- **Type a loop's variable over a list of plain dicts from its items.** A
  store into such an item (`q["p"] = pn`, `for q in parts`) goes through
  `_Site.into` on a receiver the check can't resolve, which may be a
  mobject whose `[k] = other` replaces a member, so a dt updater doing it
  leaves its play structural. Built and then reverted for want of a measured
  play it steps without the item below: restore it from commit
  `d8a068f3` (on `c52-m4-precision`, reverted by the commit after it). It
  types the variable when every item of what it loops over is a plain dict:
  not a module's, nor the attribute dict of the scene, its renderer or
  camera, an animation or a mobject of the scene (a store there rebinds
  `submobjects` or `updaters`); a list item is never typed, since it may be
  a mobject's members. Through a typed variable, stores and methods change
  no structure and write into the items, not the list. Its two guards:
  - per tick: the plan keeps each list it typed by with its items' ids and
    types (`TickPlan.typings`), and every stepping tick checks them
    (`typings_hold`, in `recorder.stepping_plan`) before the structure, so
    another length, item or type has the play plan again even when no
    structure changed;
  - at plan time: a plan whose stepping work may itself write such a list
    (an append, a store into it) is made again without typing
    (`_changes_typed`), since that change would come after the tick's check.

  Its soundness scenes stay on the branch, each never stepped past without
  typing and with it: `MobjectItemStore`, `MembersListStore`,
  `AttributeDictStore`, `MobjectJoinsDicts`
  (`tests/timeline/test_stepping.py::test_a_store_into_an_item_that_is_or_may_become_a_mobject_is_never_stepped_past`).
  The commit's other scenes come back with it: `DictItemRead` (it steps,
  with the dt-free updater writing the dicts pulled), `DictSwappedForMobject`
  and `DictsBesideGroup` with a tick hook (a dict swapped for a group
  mid-play, no structure changed: the next tick plans again before it
  steps).
- **Type what a dict holds from its values, so that arithmetic on NumPy
  arrays makes a new array.** With the item above alone, `p, v = q["p"],
  q["v"]; pn = p + v * h; pn[1] = …` still leaves the play structural:
  `p` and `v` are not typed, so `_number_array` can't show `pn` is a new
  array, and `pn[1] = …` goes through `_Site.into` on an unresolved
  receiver. State arrays held in dicts, `p + v * h`, and a store into the
  result are ordinary code in a physics scene, and scenes of physical motion
  will meet the pattern far more often than the measured scenes do. The trap: an
  operator may hand back an operand (a class whose `__add__` returns
  `self`, or an in-place result), so a sum is new only when its operands'
  types are known to make one, here NumPy arrays of numbers. The guard it
  needs is the per-tick check above extended to the values a typing read:
  the ids and types of each typed dict's values at the keys the code reads.
  Reproducer, both items together:
  `tests/timeline/test_stepping.py::test_a_store_into_a_sum_of_what_a_dict_holds_is_taken_for_a_restructure`
  (scene `DictStateIntegrator`; it asserts today's fallback).
- **Drop an interpolation's target from its read set where it overwrites
  what it reads there.** The next lever after typing a dict's values. An
  interpolation's node reads everything its animation reaches, its target
  among it, so a skipped callback writing the target's fields (an updater
  on the mobject a `Create` or a `FadeIn` draws) overlaps it.
  `arachne/timeline/overwrites.py` declares, per method object, the fields
  each interpolation sets from its held copies and alpha alone (`Create`'s
  partial: `_points`; a `Transform`'s, `FadeIn` and `FadeOut` among them:
  `_points` and `_paint`), each declaration checked by a property test
  (`tests/timeline/test_overwrites.py`) that scrambles the target's prior
  values and point counts. The plan does not read them yet. To use them,
  `_animation_node` reads the target minus the declared fields, under the
  declaration's own condition: the target's members with points pair with
  the held copy's as they did when the animation began (a member that lost
  its points, or gained some, pairs the families otherwise), checked where
  the plan is made and kept true by the structure guard or checked at each
  stepping tick. The field-level read is new to the plan, which reads by
  object today, so the change carries a soundness scene with a skipped
  writer of an undeclared field (a stroke width a `Create` keeps) that must
  stay pulled.
- **Take an updater's `dt` for a number.** The engine hands every dt
  updater a float, but the check reads `dt` as a parameter it can't
  resolve, so arithmetic on it (`h = dt * speed / n`) is no number for
  `_number_array`, and a sum `v + a * h` of a NumPy array and that product
  may be either operand: a store into the sum is taken for one into `v`.
  The sum rule needs both operands known to be numbers or NumPy's own
  arrays, since a subclass's operator, or its reflected form, may hand back
  an operand (`AliasingSumRead`). One measured scene pulls callbacks in
  several plays for it, its stepping ticks unchanged. The
  fix types `dt` (and an alpha a rate-function updater is handed) as a
  float, and an arithmetic expression of numbers and NumPy's own arrays as
  one of them.
- **Follow engine code precisely enough to tell what it never calls.** An
  argument is free of callables only when its kind shows it, and the
  engine's own constructors are not shown never to call what they are
  handed: `VGroup(b, label)`, `Line(...)`, `Axes.c2p(*xs)` with any argument
  not shown free are unknown calls. Precision this leaves, each fix of which
  must stay sound:
  - a method of a local the function made as a list, dict or set display
    (`valid_vmobjects.append(m)` in `VGroup.add`) is a callee it can't
    resolve; such a local's class is the display's, whose methods never
    call what they are handed (but a list's `sort` with a key);
  - a method of what the object it is called on holds
    (`self.x_axis.number_to_point(xs)` in `Axes.coords_to_point`) is a
    callee it can't resolve, since called parameters are read per class,
    not per object;
  - at the call site, a method of an object a call makes
    (`Dot().move_to(p)`) is a callee it can't resolve: the class
    `_Site._class_made` tells is not used;
  - inside engine code, a parameter stored into an attribute or an item
    counts as called (`VGroup.add`'s members, `ManimColor`'s value); telling
    which stores are later called needs every reader of the attribute, so it
    waits for a whole-program index;
  - `functools.cache` (`_bezier_parameters`) is a C wrapper, so it can't be
    read as forwarding without listing it; a Python wrapper is followed
    through its closure already.

  Reproducers, strict expected failures:
  `tests/timeline/test_stepping.py::test_a_member_swapped_where_no_stepping_reaches_keeps_the_plan`
  and `::test_work_private_to_a_redraw_waits_for_the_frames`.
- **Check at each stepping tick what the plan read, so that a plan may rest
  on it.** A plan rests on no value of a mutable kind (what a list, a dict,
  an object or a class holds when the plan is made): reading one makes the
  play tick everything, whatever could change it, and a callback whose
  writes are bounded only by such contents is pulled rather than skipped
  (the inventory above, entries 2 and 3). That is sound by construction and
  costs most of the stepping a measured set of scenes had: handing a
  mobject to a call reads what it holds, as does resolving `self.dot` or
  `holder.target`, and a dt-free callback moving any mobject but its own
  family is pulled. The fix observes instead of forbidding: the plan records
  what it read (each object with the attributes read, each container with
  its items, each value's kind), digests it when made, and compares the
  digest before each stepping tick, planning again when it differs, as it
  does for the structure (`current_structure`). Its cost is a digest per
  stepping tick of what the plan read. Deferred until before the planned
  precision work above. It also wins back what proofs over code gave and no
  longer may: a function of the scene's whose every return is a number,
  once the plan observes the value rather than proving it, and a scene
  whose callbacks hand values to the engine running more of its callbacks
  than before the handed-callable rule. Reproducers: every test of
  `tests/timeline/` reporting the expected failure `precision: the plan
  rests on no value that may change as the play runs`
  (`tests/timeline/conftest.py::expect_stepping`), which passes again once
  a plan may rest on what it read.
- **A method called on a receiver the check can't resolve is taken for the
  scene's code.** `dot.move_to(…)` with `dot` a loop's variable over
  `zip(group, offsets)` may be any class's `move_to`, so `_Site._method_code`
  marks the call `unknown`; when the callback reaches code of the scene's
  that adds members (a group subclass whose constructor calls `self.add`),
  `_Node.may_restructure` holds, and a play whose dt updater runs it ticks
  everything though nothing of the scene's defines `move_to`. Reproducer:
  `tests/timeline/test_stepping.py::test_a_method_of_an_unresolved_receiver_is_taken_for_the_scenes_code`
  (scene `UnresolvedMethodRead`; it asserts today's fallback). The fix is
  to take such a call as no code of the scene's when no class outside the
  engine defines the method's name. The trap: a scene may set a function of
  its own onto an engine class (`SomeEngineClass.move_to = user_fn`) or
  patch a base class's methods (a scene base class replacing `Scene`
  methods), so "no user class defines the name" must cover every class's
  current attributes, the engine's included, not only the classes the
  scene's module defines.

## Queries over the table (`arachne/timeline/coverage.py`)

- **`proven_pure` and `result_kept` answer from code, and miss.** Filters
  for precision only, with no use in the stepping check. `proven_pure`
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
  match a run that ticks everything. On one measured scene, differences of
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
