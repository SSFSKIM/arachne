# Thanks for your interest in contributing!

Please read our contributing guidelines which are hosted at https://docs.manim.community/en/latest/contributing.html

## Names the engine adds to Scene, Mobject and Camera

A scene's code subclasses `Scene`, `Mobject` and `Camera`, so any name on them can be taken by that code: a
scene may define its own `record` method or set `self.timeline`. The engine never reads its own state through
such a name. It reads from the objects it made and holds (a session reads its record from the recorder it
created, not from `scene.record`). Any name the engine adds to `Scene`, `Mobject` or `Camera` that Manim
Community Edition lacks is checked against this rule when it is added. One name still breaks it: the renderer,
the manager and `Scene` read the recorder as `scene.timeline`, which a scene that sets `self.timeline` replaces
(tech-debt-tracker.md).
