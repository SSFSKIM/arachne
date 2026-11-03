# Tech debt

Known gaps kept on record rather than fixed, each with the test that
reproduces it.

## The timeline's stepping tick (`arachne/timeline/ticking.py`)

- **A method called on a receiver the check can't resolve is taken for the
  scene's code.** `dot.move_to(…)` with `dot` a loop's variable over
  `zip(group, offsets)` may be any class's `move_to`, so `_Site._method_code`
  marks the call `unknown`; when the callback reaches code of the scene's
  that adds members (a group subclass whose constructor calls `self.add`),
  `_Node.may_restructure` holds, and a play whose dt updater runs it ticks
  everything though nothing of the scene's defines `move_to`. A precision
  gap, not a soundness one: the play ticks as CE does. Reproducer:
  `tests/timeline/test_stepping.py::test_a_method_of_an_unresolved_receiver_is_taken_for_the_scenes_code`
  (scene `UnresolvedMethodRead`; it asserts today's fallback). The fix is
  to take such a call as no code of the scene's when no class outside the
  engine defines the method's name. The trap: a scene may set a user
  function onto an engine class (`SomeEngineClass.move_to = user_fn`) or
  patch a base class's methods (a scene base class replacing `Scene`
  methods), so "no user class defines the name" must cover every class's
  current attributes, the engine's included, not only the classes the
  scene's module defines.

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
