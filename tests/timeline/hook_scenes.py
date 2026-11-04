"""Scenes the scene-hook tests execute: what CE calls on a scene during its
plays (``play_internal``, and per frame ``update_to_time`` with
``update_mobjects``, ``update_meshes`` and ``update_self`` inside it), seen
through overrides and through wrappers set on the scene object.

``HookLog`` wraps the hooks on the scene object from ``setup``, which every
execution runs, and logs each call with its argument. ``GuardLike`` stands
for a checker a scene arms in ``construct``: it wraps ``play``, ``wait``,
``remove``, ``clear``, ``update_to_time`` and ``tear_down`` on the scene
object, samples a probe at the frame hooks (one sample per ``SAMPLE``
seconds), and at ``tear_down`` reports each span the probe held for
``PERSIST`` seconds.

``ce_play`` is CE's loop, written out: what CE calls during one play of
``n`` frames. Every run time here is a whole number of frames at the rates
the tests use, so a play's frames fall on the grid and its sequence holds
exactly.
"""

from __future__ import annotations

from fractions import Fraction

from arachne import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Dot,
    Mobject,
    Scene,
    Square,
    Succession,
    ValueTracker,
    VGroup,
    always_redraw,
    config,
    linear,
)

#: The hooks ``HookLog`` wraps, beside ``play`` and ``tear_down``.
HOOKS = (
    "play_internal",
    "update_to_time",
    "update_mobjects",
    "update_meshes",
    "update_self",
)
_ARGUMENT = {
    "update_to_time": "t",
    "update_mobjects": "dt",
    "update_meshes": "dt",
    "update_self": "dt",
    "play_internal": "skip_rendering",
}

#: Every ``HookLog`` made, in order: a re-executed scene makes one per execution.
LOGS: list[HookLog] = []


class HookLog:
    """The hooks of one execution of a scene, as the scene object saw them."""

    def __init__(self, scene: Scene, execution: int) -> None:
        self.execution = execution
        self.calls: list[tuple[str, object]] = []
        for name in ("play", "tear_down", *HOOKS):
            if hasattr(scene, name):
                setattr(scene, name, self._logged(name, getattr(scene, name)))

    def _logged(self, name: str, inner):
        key = _ARGUMENT.get(name)

        def logged(*args, **kwargs):
            value = args[0] if args and key else kwargs.get(key) if key else None
            if isinstance(value, (int, float, Fraction)) and not isinstance(
                value, bool
            ):
                value = float(value)
            self.calls.append((name, value))
            return inner(*args, **kwargs)

        return logged

    def plays(self) -> list[list[tuple[str, object]]]:
        """The hook calls of each play (a wait is a play), in order."""
        out: list[list[tuple[str, object]]] = []
        for name, value in self.calls:
            if name == "play":
                out.append([])
            elif name != "tear_down" and out:
                out[-1].append((name, value))
        return out


def frame_times(calls) -> list[float]:
    return [value for name, value in calls if name == "update_to_time"]


def ce_play(
    frames: int, fps: int, *, wait_updates: bool = False
) -> list[tuple[str, object]]:
    """CE's hook calls for one play of ``frames`` frames at ``fps``: a wait
    that updates first calls ``update_mobjects(0)`` (compiling it); then
    ``play_internal()``; per frame ``j``, ``update_to_time(j / fps)`` with the
    frame's dt (0 at the first) passed on inside it; after the last frame,
    ``update_mobjects(0)``.
    """
    calls: list[tuple[str, object]] = [("update_mobjects", 0.0)] if wait_updates else []
    calls.append(("play_internal", None))
    for j in range(frames):
        dt = 0.0 if j == 0 else float(Fraction(1, fps))
        calls.append(("update_to_time", float(Fraction(j, fps))))
        calls += [("update_mobjects", dt), ("update_meshes", dt), ("update_self", dt)]
    calls.append(("update_mobjects", 0.0))
    return calls


class Logged(Scene):
    """A scene whose hooks ``HookLog`` sees, from every execution."""

    def setup(self):
        super().setup()
        self.log = HookLog(self, len(LOGS) + 1)
        LOGS.append(self.log)


def clock(scene: Scene) -> ValueTracker:
    """A tracker that integrates time: its plays tick."""
    tracker = ValueTracker(0)
    tracker.add_updater(lambda m, dt: m.increment_value(dt))
    scene.add(tracker)
    return tracker


# (a) one play on each route


class TrustedHooks(Logged):
    def construct(self):
        square = Square(side_length=1)
        self.add(square)
        self.play(square.animate.shift(RIGHT), rate_func=linear)


class StateHooks(Logged):
    def construct(self):
        t = ValueTracker(0)
        bar = always_redraw(lambda: Square(side_length=0.2 + t.get_value()))
        self.add(bar)
        self.play(t.animate.set_value(1), rate_func=linear)


class TickedHooks(Logged):
    """A dt updater makes the play tick; the follower waits for the frames."""

    def construct(self):
        clock(self)
        dot = Dot(LEFT)
        follower = Dot(color="#ff0000", radius=0.05)
        follower.add_updater(lambda m: m.next_to(dot, UP))
        self.add(dot, follower)
        self.play(dot.animate.shift(2 * RIGHT), rate_func=linear)


# (b) a scene clock kept by class overrides


class PlayClock(Logged):
    """``now()`` reads the renderer's clock plus the play-local time the
    overrides keep, and a redraw places a dot by it: at frame ``j`` of a play
    CE shows ``start + 2 j / fps`` (its clock has advanced ``j`` frames when
    it calls ``update_to_time(j / fps)``), and after a play of ``n`` frames,
    until the next frame hook, ``end + (n - 1) / fps``.
    """

    SCALE = 0.5

    def play_internal(self, skip_rendering=False):
        self._pt = 0.0
        super().play_internal(skip_rendering)
        self._pt = 0.0

    def update_to_time(self, t):
        self._pt = t
        super().update_to_time(t)

    def now(self):
        return float(self.renderer.time) + getattr(self, "_pt", 0.0)

    def construct(self):
        self._pt = 0.0
        marker = always_redraw(
            lambda: Dot(radius=0.05).move_to([self.now() * self.SCALE - 3, 0, 0])
        )
        square = Square(side_length=1)
        self.add(marker, square)
        self.play(square.animate.shift(RIGHT), run_time=1)
        self.wait(0.4)
        self.play(square.animate.shift(LEFT), run_time=0.6)


# (c) a checker armed on the scene object


class GuardLike:
    """Wraps the scene's methods on the instance and watches a probe."""

    SAMPLE = 0.25
    PERSIST = 1.0

    def __init__(self, scene: Scene, probe) -> None:
        self.scene, self.probe = scene, probe
        self.samples: list[float] = []
        self.report: list[tuple[float, float]] | None = None
        self.departures = 0
        self._open: list[float] | None = None
        self._held: list[tuple[float, float]] = []
        self._last: float | None = None
        self._play_start = 0.0
        inner = {
            name: getattr(scene, name)
            for name in (
                "play",
                "wait",
                "remove",
                "clear",
                "update_to_time",
                "tear_down",
            )
        }

        def now() -> float:
            return float(scene.renderer.time)

        def on_play(*args, **kwargs):
            self.sample(now())
            self._play_start = now()
            result = inner["play"](*args, **kwargs)
            self.sample(now())
            return result

        def on_wait(*args, **kwargs):
            return inner["wait"](*args, **kwargs)

        def on_remove(*mobjects):
            self.departures += 1
            return inner["remove"](*mobjects)

        def on_clear():
            self.departures += 1
            return inner["clear"]()

        def on_frame(t):
            inner["update_to_time"](t)
            at = self._play_start + float(t)
            if self._last is None or at - self._last >= self.SAMPLE:
                self.sample(at)

        def on_tear_down():
            result = inner["tear_down"]()
            self.sample(now())
            self._close()
            self.report = list(self._held)
            return result

        wrappers = {
            "play": on_play,
            "wait": on_wait,
            "remove": on_remove,
            "clear": on_clear,
            "update_to_time": on_frame,
            "tear_down": on_tear_down,
        }
        for name, wrapper in wrappers.items():
            setattr(scene, name, wrapper)

    def sample(self, at: float) -> None:
        self._last = at
        self.samples.append(at)
        if self.probe():
            if self._open is None:
                self._open = [at, at]
            self._open[1] = at
        else:
            self._close()

    def _close(self) -> None:
        if (
            self._open is not None
            and self._open[1] - self._open[0] >= self.PERSIST - 1e-9
        ):
            self._held.append((self._open[0], self._open[1]))
        self._open = None


def ce_guard_samples(plays, fps: int, sample: float = GuardLike.SAMPLE) -> list[float]:
    """The instants ``GuardLike`` samples on CE for ``plays``, each
    ``(start, frames, end)``: a play's entry; its frames ``start + j / fps``,
    one per ``sample`` seconds; its exit; and, after the last, ``tear_down``.
    """
    out: list[float] = []
    for start, frames, end in plays:
        last = float(start)
        out.append(last)
        for j in range(frames):
            at = float(start) + float(Fraction(j, fps))
            if at - last >= sample:
                out.append(at)
                last = at
        out.append(float(end))
    out.append(float(plays[-1][2]))
    return out


def off_frame(mobject: Mobject) -> bool:
    return mobject.get_right()[0] > config.frame_width / 2


class DriftingCaption(Scene):
    """A caption that hangs over the right edge for 1.5 s in the middle of
    one play and is back before it ends: only samples taken during the play
    see it. At 15 fps CE samples it at frames 16 to 36.
    """

    def construct(self):
        caption = Square(side_length=2)
        self.guard = GuardLike(self, lambda: off_frame(caption))
        level = ValueTracker(0.0)

        def place(m):
            s = level.get_value()
            m.move_to([6.5 if 1.0 <= s < 2.5 else 0.0, 2.0, 0.0])

        caption.add_updater(place)
        place(caption)
        self.add(caption)
        self.play(level.animate.set_value(4.0), run_time=4.0, rate_func=linear)
        self.wait(0.6)


# (d) waits


class Waits(Logged):
    """A frozen wait; a wait a dt updater keeps running; a wait whose stop
    condition, read at every tick, first holds at the 26th (0.43 s reached at
    0.4333 s), between the frames 0.4 and 0.4667 of 15 fps.
    """

    def construct(self):
        self.add(Square(side_length=1))
        self.wait(0.6)
        clock(self)
        self.wait(0.6)
        level = clock(self)
        self.wait_until(lambda: level.get_value() >= 0.43, max_time=2)


# (e) an event between frames


class OffGridEvent(Logged):
    """A ``Succession`` whose second part begins at 0.35 s, between the
    frames of 15 fps.
    """

    def construct(self):
        a, b = Square(side_length=0.5).shift(LEFT), Square(side_length=0.5).shift(RIGHT)
        self.add(a, b)
        self.play(
            Succession(
                a.animate(run_time=0.35, rate_func=linear).shift(UP),
                b.animate(run_time=0.65, rate_func=linear).shift(DOWN),
            )
        )


# (f) overrides that can't be dispatched


class _OnePlay(Scene):
    def construct(self):
        square = Square(side_length=1)
        self.add(square)
        self.play(square.animate.shift(RIGHT), rate_func=linear)


class ProgressionOverride(_OnePlay):
    def get_time_progression(self, run_time, *args, **kwargs):
        return super().get_time_progression(run_time, *args, **kwargs)


class PlayInternalSkipped(_OnePlay):
    def play_internal(self, skip_rendering=False):
        pass


class PlayInternalTwice(_OnePlay):
    def play_internal(self, skip_rendering=False):
        super().play_internal(skip_rendering)
        super().play_internal(skip_rendering)


class PlayInternalSkipRendering(_OnePlay):
    def play_internal(self, skip_rendering=False):
        super().play_internal(skip_rendering=True)


class UpdateToTimeWarp(_OnePlay):
    def update_to_time(self, t):
        super().update_to_time(t / 2)


class UpdateToTimeReplaced(_OnePlay):
    def update_to_time(self, t):
        pass


class UpdateMobjectsReplaced(_OnePlay):
    def update_mobjects(self, dt):
        pass


class UpdateSelfReplaced(_OnePlay):
    def update_self(self, dt):
        pass


class UpdateToTimeTwice(_OnePlay):
    def update_to_time(self, t):
        super().update_to_time(t)
        super().update_to_time(t)


class UpdateMobjectsTwice(_OnePlay):
    def update_mobjects(self, dt):
        super().update_mobjects(dt)
        super().update_mobjects(dt)


# (g) a run time the scene decides


class RunTimeOverride(Logged):
    def get_run_time(self, animations):
        return 2.0

    def construct(self):
        square = Square(side_length=1)
        self.add(square)
        self.play(square.animate.shift(RIGHT), run_time=1)


class RunTimeChangesInPlay(Scene):
    """``get_run_time`` gives 1 s when the play is compiled and 2 s when
    ``play_internal`` asks again: CE plays the second, and the event at 1.5 s
    lies beyond the first. A dt updater makes the play tick.
    """

    def get_run_time(self, animations):
        self.run_times.append(1.0 if not self.run_times else 2.0)
        return self.run_times[-1]

    def update_to_time(self, t):
        self.frames.append(float(t))
        super().update_to_time(t)

    def construct(self):
        self.run_times: list[float] = []
        self.frames: list[float] = []
        clock(self)
        a, b = Square(side_length=0.5).shift(LEFT), Square(side_length=0.5).shift(RIGHT)
        self.add(a, b)
        self.play(
            Succession(
                a.animate(run_time=1.5, rate_func=linear).shift(UP),
                b.animate(run_time=0.5, rate_func=linear).shift(DOWN),
            )
        )


class RunTimeScaledInPlay(Scene):
    """``play_internal`` doubles what ``get_run_time`` gives before its body
    asks: CE plays 2 s of a 1 s animation.
    """

    def get_run_time(self, animations):
        self.run_times.append(super().get_run_time(animations) * self.factor)
        return self.run_times[-1]

    def play_internal(self, skip_rendering=False):
        self.factor = 2
        super().play_internal(skip_rendering)

    def update_to_time(self, t):
        self.frames.append(float(t))
        super().update_to_time(t)

    def construct(self):
        self.factor = 1
        self.run_times: list[float] = []
        self.frames: list[float] = []
        clock(self)
        dot = Dot()
        self.add(dot)
        self.play(dot.animate.shift(RIGHT), rate_func=linear)


class AnimationLengthenedInPlay(Scene):
    """``play_internal`` lengthens the animation to 2 s before its body asks
    the (engine's) ``get_run_time``: CE plays 2 s.
    """

    def play_internal(self, skip_rendering=False):
        self.animations[0].run_time = 2.0
        super().play_internal(skip_rendering)

    def update_to_time(self, t):
        self.frames.append(float(t))
        super().update_to_time(t)

    def construct(self):
        self.run_times: list[float] = []
        self.frames: list[float] = []
        clock(self)
        dot = Dot()
        self.add(dot)
        self.play(dot.animate.shift(RIGHT), rate_func=linear)


# (h) an update_mobjects override beside a dt updater


class UpdateOverride(Scene):
    """The override sees CE's frame dt; the dt updater integrates the play's
    exact length.
    """

    def update_mobjects(self, dt):
        self.seen.append(float(dt))
        super().update_mobjects(dt)

    def construct(self):
        self.seen: list[float] = []
        self.total = 0.0

        def integrate(m, dt):
            self.total += dt

        holder = Mobject()
        holder.add_updater(integrate)
        dot = Dot()
        follower = Dot(radius=0.05)
        follower.add_updater(lambda m: m.next_to(dot, UP))
        self.add(holder, dot, follower)
        self.play(dot.animate.shift(RIGHT), rate_func=linear)


# (j) a scene executed again


class OwnFill(Dot):
    """A user's dot whose fill the camera reads through the user's code."""

    def get_fill_rgbas(self):
        import numpy as np

        return np.array([[1.0, 0.0, 0.0, 1.0]])


class Reexecuted(Logged):
    """``OwnFill`` joins and leaves within play 0, so the record can't draw
    that play and the scene runs a second time.
    """

    def construct(self):
        t = ValueTracker(0)
        holder = VGroup()

        def show(m):
            inside = 0.3 <= t.get_value() < 0.6
            if inside and not m.submobjects:
                m.add(OwnFill(radius=0.3))
            elif not inside and m.submobjects:
                m.remove(*m.submobjects)

        holder.add_updater(show)
        self.add(holder)
        self.play(t.animate.set_value(1), rate_func=linear)
        self.remove(holder)
        self.wait(0.5)


# The guarded dt integrator


class DtIntegrator(Scene):
    """A dt updater integrates time and moves a dot by it, through two plays
    and a wait it keeps running.
    """

    guarded = False

    def construct(self):
        if self.guarded:
            self.guard = GuardLike(self, lambda: False)
        self.total = 0.0
        self.ends: list[float] = []
        self.mover = Dot().move_to(LEFT * 3)

        def integrate(m, dt):
            self.total += dt
            m.shift(RIGHT * dt)

        self.mover.add_updater(integrate)
        marker = Square(side_length=0.5).move_to(DOWN * 2)
        self.add(self.mover, marker)
        self.play(marker.animate.shift(RIGHT * 2), run_time=1)
        self.ends.append(self.total)
        self.wait(1)
        self.ends.append(self.total)
        self.play(marker.animate.shift(LEFT * 2), run_time=0.6)
        self.ends.append(self.total)

    def state(self) -> dict:
        return {
            "total": self.total,
            "ends": self.ends,
            "x": float(self.mover.get_center()[0]),
        }


class DtIntegratorGuarded(DtIntegrator):
    guarded = True


# A frame hook that changes what the play's ticks run


class HookAddsReader(Scene):
    """At its first frame from 0.5 s on, the scene's ``update_to_time`` adds
    a tracker whose dt updater reads a dot a dt-free updater moves: the play
    plans again at its next stepping tick.
    """

    def update_to_time(self, t):
        super().update_to_time(t)
        if t >= 0.5 and self.reader not in self.mobjects:
            self.add(self.reader)

    def construct(self):
        clock(self)
        t = ValueTracker(0)
        source = Dot()
        source.add_updater(lambda m: m.set_x(t.get_value()))
        self.reader = ValueTracker(0)
        self.reader.add_updater(lambda m, dt: m.increment_value(dt * source.get_x()))
        self.add(t, source)
        self.play(t.animate.set_value(1), rate_func=linear)
