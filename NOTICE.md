# Notice

Arachne is a visual engine for teaching, developed as a fork of
[Manim Community Edition](https://github.com/ManimCommunity/manim) (forked at v0.22.1).

## Third-party code

Portions of this repository are derived from Manim Community Edition and, through it, from
[3Blue1Brown's Manim](https://github.com/3b1b/manim). That code remains under its original MIT licenses,
whose copyright and permission notices are kept in full:

- `LICENSES/manim-3b1b-MIT.txt`: Copyright (c) 2018 3Blue1Brown LLC
- `LICENSES/manim-community-MIT.txt`: Copyright (c) 2024, the Manim Community Developers

Those notices apply to the code that came from those projects, including the history of this repository up to and
including the Manim Community Edition v0.22.1 release commit.

## Ideas credited

These ideas came from other projects; no code was taken from them.

- Immutable geometry and paint values behind a mutable-looking mobject API (read-only points, style as one immutable
  `Paint` value): [ManimGX](https://github.com/academa-labs/manimgx) by Academa, Inc. (MIT).
- Holding cost laws with tests that count work (values built, hashed and copied) instead of timing it: ManimGX, as
  above.
- A rational scene clock, on which durations become simple fractions and spans line up exactly
  (`arachne/timeline/clock.py`): ManimGX's `clock.py`, as above.
- Naming the mobjects a finding is about from the local variables of the scene's frames
  (`arachne/timeline/source.py`): ManimGX's `cli/diagnostics.py`, as above.
- Finding the scene's own line by walking the stack out of the library, and refusing `play()` while another play
  runs, as from an updater (`arachne/timeline/source.py`, `arachne/manager.py`): Manim Community Edition
  [pull request #5006](https://github.com/ManimCommunity/manim/pull/5006).

## Arachne's own work

No license is granted for the changes and additions made in this repository after the fork. All rights to them are
reserved by their authors. The repository is public so that its development and CI are visible, not as an offer of a
license.
