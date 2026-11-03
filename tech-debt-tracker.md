# Tech debt

Known gaps kept on record rather than fixed, each with the test that
reproduces it.

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
- **Follow engine code precisely enough to tell what it never calls.** The
  handed-callable analysis reads scene code through one closed table,
  `OPERATIONS` in `arachne/timeline/coverage.py`. Every node class of the
  running Python has a `Rule` saying what a node of it holds, whether it is
  inert, what it carries and, as a statement, what it stores; the value
  kinds those rules name are listed beside it. `rule_of` gives `FALLBACK`
  to a class the table does not name, so a form nobody listed makes the
  play tick everything;
  `tests/timeline/test_coverage.py::test_every_node_class_has_a_rule` holds
  the table against the interpreter. `_Site._held`, `_Site._inert`,
  `handed._carried` and `_Site.read_table` (run on every function of the
  scene's a play reaches) consult it, and so do the two queries
  `proven_pure` and `result_kept`. What the table makes fall back: a value
  sent into or out of a generator or coroutine holds anything; a `global`
  or `nonlocal` name bound to a value not shown inert ticks everything; an
  item of a base that is not a builtin item kind, an inert value, a
  display, a local list or dict, a mobject's members or a parameter is an
  unknown call; a value not shown free of callables makes its call unknown
  whatever the callee, so a native listed as never calling still counts as
  calling a generator or iterator it is handed; modules, other native kinds
  and instances of classes with methods in scene code may hold anything; a
  function the check does not read may store anywhere; and a value not
  shown free of callables stored into what a proof read at plan time (the
  attribute it read, anything an object it read holds, any object for a
  receiver it can't resolve, any module) ends that snapshot, so the play
  ticks everything.

  The engine is modelled, not read through this table: its methods keep
  what they are handed only in the object they are called on or the object
  they make, its functions keep nothing, and a mobject's updaters are not
  counted among what it holds (the plan schedules them). An argument is
  free of callables only when its kind shows it, and the engine's own
  constructors are not shown never to call what they are handed:
  `VGroup(b, label)`, `Line(...)`, `Axes.c2p(*xs)` with any argument not
  shown free are unknown calls. Precision this leaves, each fix of which
  must stay sound:
  - a method of a local the function made as a list, dict or set display
    (`valid_vmobjects.append(m)` in `VGroup.add`) is a callee it can't
    resolve; such a local's class is the display's, whose methods never
    call what they are handed (but a list's `sort` with a key);
  - a method of what the object it is called on holds
    (`self.x_axis.number_to_point(xs)` in `Axes.coords_to_point`) is a
    callee it can't resolve, since called parameters are read per class,
    not per object; reading them for the resolved object at the call site
    resolves `self.x_axis` as `_Site.resolve` does;
  - at the call site, a method of an object a call makes
    (`Dot().move_to(p)`) is a callee it can't resolve: the class
    `_Site._class_made` tells is not used;
  - the store rule inside engine code: a parameter stored into an
    attribute or an item counts as called (`VGroup.add`'s members,
    `ManimColor`'s value). A store into `self.X` may count only where
    analysed code calls `self.X`, hands it to something that may call it,
    or the attribute reaches a callee it can't resolve. That needs every
    reader of `X`, including getters returning it and code outside the
    class, so it waits for a whole-program index;
  - `functools.cache` (`_bezier_parameters`) is a C wrapper, so it can't be
    read as forwarding without listing it; a Python wrapper is followed
    through its closure already;
  - a method a callee calls on what it is handed (`graph.function(x)` in
    engine code) is not taken for calling the argument in
    `handed.Callers`: sound by another route the probes found, with no
    fixture of its own.

  Reproducers of the cost, strict xfails until it is won back:
  `tests/timeline/test_stepping.py::test_a_member_swapped_where_no_stepping_reaches_keeps_the_plan`
  and `::test_work_private_to_a_redraw_waits_for_the_frames`. What must
  hold meanwhile: the fixtures of
  `::test_a_callable_the_check_cannot_resolve_is_called_where_its_callee_may`
  and `::test_a_value_holding_a_callable_is_called_where_its_callee_may`
  stay pulled, and the controls
  `::test_a_value_the_check_cannot_resolve_handed_to_code_that_never_calls_it`
  step.
- **Track a store through engine code by the attribute it lands in.** A
  value handed to an engine method is taken as kept anywhere in the object
  it is called on or the object it makes. When a plan-time proof read any
  attribute of that object, the store ends the snapshot and the play ticks
  everything, though the engine may keep the value in an attribute no
  proof read. On a measured set of scenes this is one of the two causes of
  the plays the closed table stopped stepping, and with the item below it
  leaves a scene whose callbacks hand values to the engine running more of
  the scene's callbacks than it did before the handed-callable rule. The fix reads, per engine method, which attributes of its
  receiver a parameter can reach, and ends only those snapshots; a method
  that stores through a helper it can't resolve, or through `setattr` with
  a computed name, still counts as storing into any attribute.
- **Read what a scene helper's methods change.** An instance of a class
  with methods in scene code may hold anything, and a store into an object
  a proof read at plan time ends the snapshot whatever the method changes.
  So a callback calling a method of a helper object (a state holder or
  tracker class the scene wrote) leaves its play ticking everything. This
  is the other cause of the plays the closed table stopped stepping. The
  fix reads the helper's methods through the same table, as functions of
  the scene's, and ends only the snapshots of the attributes they store
  into; a method the check does not read, or one reached through a
  receiver it can't resolve, still counts as a store anywhere.
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
