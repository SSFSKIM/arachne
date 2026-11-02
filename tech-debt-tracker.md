# Tech debt

Known gaps kept on record rather than fixed, each with the test that
reproduces it.

## The timeline's stepping tick (`arachne/timeline/ticking.py`)

- **A method of an object the callback made is not read for its writes.**
  A dt-free updater that builds a helper of the scene's own class
  (`helper = Pusher(source)`) and calls a method of it (`helper.push(v)`)
  is judged by the constructor alone, so a write the method makes elsewhere
  (to a tracker a dt updater integrates) goes unseen and the updater is
  skipped at stepping ticks. This form is unlikely: user objects built per
  frame whose methods write shared state appear in neither the corpus nor
  the plugin's guides. Reproducer:
  `tests/timeline/test_stepping.py::test_a_method_of_an_object_the_callback_made_is_read_for_its_writes`
  (strict xfail, scene `FreshMethod`). The fix is to follow the made class's
  method in `_Site.call` for a fresh receiver (`_Site.classes` already
  knows the class) as `_Site.assigning` does for its setters.
