"""Tools for displaying multiple animations at once."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from fractions import Fraction
from typing import TYPE_CHECKING, Any

import numpy as np

from arachne.animation.animation import Animation, prepare_animation
from arachne.mobject.mobject import Group, Mobject
from arachne.scene.scene import Scene
from arachne.timeline.clock import exact
from arachne.utils.iterables import remove_list_redundancies
from arachne.utils.parameter_parsing import flatten_iterable_parameters
from arachne.utils.rate_functions import linear

if TYPE_CHECKING:
    from arachne.mobject.types.vectorized_mobject import VGroup
    from arachne.timeline.events import EventPlan

__all__ = ["AnimationGroup", "Succession", "LaggedStart", "LaggedStartMap"]


DEFAULT_LAGGED_START_LAG_RATIO: float = 0.05


class AnimationGroup(Animation):
    """Plays a group or series of :class:`~.Animation`.

    Parameters
    ----------
    animations
        Sequence of :class:`~.Animation` objects to be played.
    group
        A group of multiple :class:`~.Mobject`.
    run_time
        The duration of the animation in seconds.
    rate_func
        The function defining the animation progress based on the relative
        runtime (see :mod:`~.rate_functions`) .
    lag_ratio
        Defines the delay after which the animation is applied to submobjects. A lag_ratio of
        ``n.nn`` means the next animation will play when ``nnn%`` of the current animation has played.
        Defaults to 0.0, meaning that all animations will be played together.

        This does not influence the total runtime of the animation. Instead the runtime
        of individual animations is adjusted so that the complete animation has the defined
        run time.
    """

    def __init__(
        self,
        *animations: Animation | Iterable[Animation],
        group: Group | VGroup | None = None,
        run_time: float | None = None,
        rate_func: Callable[[float], float] = linear,
        lag_ratio: float = 0,
        **kwargs: Any,
    ):
        arg_anim = flatten_iterable_parameters(animations)
        self.animations = [prepare_animation(anim) for anim in arg_anim]
        self.rate_func = rate_func
        if group is None:
            mobjects = remove_list_redundancies(
                [anim.mobject for anim in self.animations if not anim.is_introducer()],
            )
            self.group: Group | VGroup = Group(*mobjects)
        else:
            self.group = group
        super().__init__(
            self.group, rate_func=self.rate_func, lag_ratio=lag_ratio, **kwargs
        )
        self.run_time: float = self.init_run_time(run_time)

    def get_all_mobjects(self) -> Sequence[Mobject]:
        return list(self.group)

    def begin(self) -> None:
        if not self.animations:
            raise ValueError(
                f"Trying to play {self} without animations, this is not supported. "
                "Please add at least one subanimation."
            )
        self.anim_group_time = 0.0
        if self.suspend_mobject_updating:
            self.group.suspend_updating()
        for anim in self.animations:
            anim.begin()

    def _setup_scene(self, scene: Scene) -> None:
        for anim in self.animations:
            anim._setup_scene(scene)

    def finish(self) -> None:
        for anim in self.animations:
            anim.finish()
        self.anims_begun[:] = True
        self.anims_finished[:] = True
        if self.suspend_mobject_updating:
            self.group.resume_updating()

    def clean_up_from_scene(self, scene: Scene) -> None:
        self._on_finish(scene)
        for anim in self.animations:
            if self.remover:
                anim.remover = self.remover
            anim.clean_up_from_scene(scene)

    def update_mobjects(self, dt: float) -> None:
        for anim in self.anims_with_timings["anim"][
            self.anims_begun & ~self.anims_finished
        ]:
            anim.update_mobjects(dt)

    def init_run_time(self, run_time: float | None) -> float:
        """Calculates the run time of the animation, if different from ``run_time``.

        Parameters
        ----------
        run_time
            The duration of the animation in seconds.

        Returns
        -------
        run_time
            The duration of the animation in seconds.
        """
        self.build_animations_with_timings()
        # Note: if lag_ratio < 1, then not necessarily the final animation's
        # end time will be the max end time! Therefore we must calculate the
        # maximum over all the end times, and not just take the last one.
        # Example: if you want to play 2 animations of 10s and 1s with a
        # lag_ratio of 0.1, the 1st one will end at t=10 and the 2nd one will
        # end at t=2, so the AnimationGroup will end at t=10.
        self.max_end_time = max(self.anims_with_timings["end"], default=0)
        return self.max_end_time if run_time is None else run_time

    def build_animations_with_timings(self) -> None:
        """Creates a list of triplets of the form (anim, start_time, end_time)."""
        run_times = np.array([anim.run_time for anim in self.animations])
        num_animations = run_times.shape[0]
        dtype = [("anim", "O"), ("start", "f8"), ("end", "f8")]
        self.anims_with_timings: np.ndarray = np.zeros(num_animations, dtype=dtype)
        self.anims_begun: np.ndarray = np.zeros(num_animations, dtype=bool)
        self.anims_finished: np.ndarray = np.zeros(num_animations, dtype=bool)
        if num_animations == 0:
            return

        lags = run_times[:-1] * self.lag_ratio
        self.anims_with_timings["anim"] = self.animations
        self.anims_with_timings["start"][1:] = np.add.accumulate(lags)
        self.anims_with_timings["end"] = self.anims_with_timings["start"] + run_times

    def _exact_timings(self) -> tuple[list[Fraction], list[Fraction]]:
        """The parts' start and end in group time, as exact fractions.

        They are the timings of :meth:`build_animations_with_timings`, computed
        on the scene clock's exact durations instead of in floating point.
        """
        run_times = [exact(anim.run_time) for anim in self.animations]
        lag = exact(self.lag_ratio)
        starts: list[Fraction] = []
        start = Fraction(0)
        for run_time in run_times:
            starts.append(start)
            start += run_time * lag
        ends = [s + run_time for s, run_time in zip(starts, run_times, strict=True)]
        return starts, ends

    def interpolate(self, alpha: float) -> None:
        # Note, if the run_time of AnimationGroup has been
        # set to something other than its default, these
        # times might not correspond to actual times,
        # e.g. of the surrounding scene.  Instead they'd
        # be a rescaled version.  But that's okay!
        anim_group_time = self.rate_func(alpha) * self.max_end_time
        time_goes_back = anim_group_time < self.anim_group_time

        # Only update ongoing animations
        awt = self.anims_with_timings
        new_begun = anim_group_time >= awt["start"]
        new_finished = anim_group_time > awt["end"]
        to_update = awt[
            (self.anims_begun | new_begun) & (~self.anims_finished | ~new_finished)
        ]

        run_times = to_update["end"] - to_update["start"]
        with_zero_run_time = run_times == 0
        run_times[with_zero_run_time] = 1
        sub_alphas = (anim_group_time - to_update["start"]) / run_times
        if time_goes_back:
            sub_alphas[(sub_alphas < 0) | with_zero_run_time] = 0
        else:
            sub_alphas[(sub_alphas > 1) | with_zero_run_time] = 1

        for anim_to_update, sub_alpha in zip(
            to_update["anim"], sub_alphas, strict=True
        ):
            anim_to_update.interpolate(sub_alpha)

        self.anim_group_time = anim_group_time
        self.anims_begun = new_begun
        self.anims_finished = new_finished


class Succession(AnimationGroup):
    """Plays a series of animations in succession.

    Parameters
    ----------
    animations
        Sequence of :class:`~.Animation` objects to be played.
    lag_ratio
        Defines the delay after which the animation is applied to submobjects. A lag_ratio of
        ``n.nn`` means the next animation will play when ``nnn%`` of the current animation has played.
        Defaults to 1.0, meaning that the next animation will begin when 100% of the current
        animation has played.

        This does not influence the total runtime of the animation. Instead the runtime
        of individual animations is adjusted so that the complete animation has the defined
        run time.

    Examples
    --------
    .. arachne:: SuccessionExample

        class SuccessionExample(Scene):
            def construct(self):
                dot1 = Dot(point=LEFT * 2 + UP * 2, radius=0.16, color=BLUE)
                dot2 = Dot(point=LEFT * 2 + DOWN * 2, radius=0.16, color=MAROON)
                dot3 = Dot(point=RIGHT * 2 + DOWN * 2, radius=0.16, color=GREEN)
                dot4 = Dot(point=RIGHT * 2 + UP * 2, radius=0.16, color=YELLOW)
                self.add(dot1, dot2, dot3, dot4)

                self.play(Succession(
                    dot1.animate.move_to(dot2),
                    dot2.animate.move_to(dot3),
                    dot3.animate.move_to(dot4),
                    dot4.animate.move_to(dot1)
                ))
    """

    def __init__(self, *animations: Animation, lag_ratio: float = 1, **kwargs: Any):
        super().__init__(*animations, lag_ratio=lag_ratio, **kwargs)

    def begin(self) -> None:
        if not self.animations:
            raise ValueError(
                f"Trying to play {self} without animations, this is not supported. "
                "Please add at least one subanimation."
            )
        self.update_active_animation(0)

    def finish(self) -> None:
        while self.active_animation is not None:
            self.next_animation()

    def update_mobjects(self, dt: float) -> None:
        if self.active_animation:
            self.active_animation.update_mobjects(dt)

    def _setup_scene(self, scene: Scene | None) -> None:
        if scene is None:
            return
        if self.is_introducer():
            for anim in self.animations:
                if not anim.is_introducer() and anim.mobject is not None:
                    scene.add(anim.mobject)

        self.scene = scene

    def update_active_animation(self, index: int) -> None:
        self.active_index = index
        if index >= len(self.animations):
            self.active_animation: Animation | None = None
            self.active_start_time: float | None = None
            self.active_end_time: float | None = None
        else:
            self.active_animation = self.animations[index]
            self.active_animation._setup_scene(self.scene)
            self.active_animation.begin()
            self.active_start_time = self.anims_with_timings[index]["start"]
            self.active_end_time = self.anims_with_timings[index]["end"]

    def next_animation(self) -> None:
        """Proceeds to the next animation.

        This method is called right when the active animation finishes.
        """
        if self.active_animation is not None:
            self.active_animation.finish()
        self.update_active_animation(self.active_index + 1)

    # While a scene plays this Succession, its events (finishing the active part
    # and beginning the next) are run by the play, through these, in the phase
    # order of arachne.timeline.events; outside a play it handles them itself.
    _event_plan: EventPlan | None = None
    _pending_time: float | None = None
    _next_index: int | None = None
    _event_instants: list[Fraction | None]
    _exact_ends: list[Fraction]

    def interpolate(self, alpha: float) -> None:
        current_time = self.rate_func(alpha) * self.max_end_time
        if self._event_plan is not None:
            # Phase 1 only: evaluate the running part at this instant. A part
            # whose window ends here is left to its finish (phase 2), which
            # evaluates it at 1 once, as CE's does.
            self._pending_time = current_time
            active = self.active_animation
            if (
                active is not None
                and self._next_index is None
                and not self._active_part_due()
            ):
                active.interpolate(self._active_subalpha(current_time))
            return
        while self.active_end_time is not None and current_time >= self.active_end_time:
            self.next_animation()
        if self.active_animation is not None and self.active_start_time is not None:
            self.active_animation.interpolate(self._active_subalpha(current_time))

    def _active_subalpha(self, current_time: float) -> float:
        assert self.active_animation is not None
        assert self.active_start_time is not None
        elapsed = current_time - self.active_start_time
        active_run_time = self.active_animation.run_time
        return elapsed / active_run_time if active_run_time != 0.0 else 1.0

    def _active_part_due(self) -> bool:
        """Whether the active part's window has ended at the play's instant."""
        plan = self._event_plan
        if (
            plan is None
            or self._pending_time is None
            or self._next_index is not None
            or getattr(self, "active_animation", None) is None
        ):
            return False
        index = self.active_index
        instant = self._event_instants[index]
        if instant is not None:
            return instant <= plan.now
        # No exact instant: due at the first instant its group time has crossed.
        return Fraction(self._pending_time) >= self._exact_ends[index]

    def _finish_due_part(self) -> bool:
        """Phase 2: finish the active part if its window has ended."""
        if not self._active_part_due():
            return False
        assert self.active_animation is not None
        self.active_animation.finish()
        self._next_index = self.active_index + 1
        return True

    def _begin_next_part(self) -> bool:
        """Phase 3: begin the part after the one just finished."""
        index = self._next_index
        if index is None:
            return False
        self._next_index = None
        self.update_active_animation(index)
        return self.active_animation is not None

    def _evaluate_begun_part(self) -> None:
        """Evaluate a part begun at this instant, as CE evaluates it in the
        sample that began it, unless it is already due to finish.
        """
        active = self.active_animation
        if active is None or self._pending_time is None or self._active_part_due():
            return
        active.interpolate(max(0.0, self._active_subalpha(self._pending_time)))


class LaggedStart(AnimationGroup):
    """Adjusts the timing of a series of :class:`~.Animation` according to ``lag_ratio``.

    Parameters
    ----------
    animations
        Sequence of :class:`~.Animation` objects to be played.
    lag_ratio
        Defines the delay after which the animation is applied to submobjects. A lag_ratio of
        ``n.nn`` means the next animation will play when ``nnn%`` of the current animation has played.
        Defaults to 0.05, meaning that the next animation will begin when 5% of the current
        animation has played.

        This does not influence the total runtime of the animation. Instead the runtime
        of individual animations is adjusted so that the complete animation has the defined
        run time.

    Examples
    --------
    .. arachne:: LaggedStartExample

        class LaggedStartExample(Scene):
            def construct(self):
                title = Text("lag_ratio = 0.25").to_edge(UP)

                dot1 = Dot(point=LEFT * 2 + UP, radius=0.16)
                dot2 = Dot(point=LEFT * 2, radius=0.16)
                dot3 = Dot(point=LEFT * 2 + DOWN, radius=0.16)
                line_25 = DashedLine(
                    start=LEFT + UP * 2,
                    end=LEFT + DOWN * 2,
                    color=RED
                )
                label = Text("25%", font_size=24).next_to(line_25, UP)
                self.add(title, dot1, dot2, dot3, line_25, label)

                self.play(LaggedStart(
                    dot1.animate.shift(RIGHT * 4),
                    dot2.animate.shift(RIGHT * 4),
                    dot3.animate.shift(RIGHT * 4),
                    lag_ratio=0.25,
                    run_time=4
                ))
    """

    def __init__(
        self,
        *animations: Animation,
        lag_ratio: float = DEFAULT_LAGGED_START_LAG_RATIO,
        **kwargs: Any,
    ):
        super().__init__(*animations, lag_ratio=lag_ratio, **kwargs)


class LaggedStartMap(LaggedStart):
    """Plays a series of :class:`~.Animation` while mapping a function to submobjects.

    Parameters
    ----------
    animation_class
        :class:`~.Animation` to apply to mobject.
    mobject
        :class:`~.Mobject` whose submobjects the animation, and optionally the function,
        are to be applied.
    arg_creator
        Function which will be applied to :class:`~.Mobject`.
    run_time
        The duration of the animation in seconds.
    lag_ratio
        Defines the delay after which the animation is applied to submobjects. A lag_ratio of
        ``n.nn`` means the next animation will play when ``nnn%`` of the current animation has played.
        Defaults to 0.05, meaning that the next animation will begin when 5% of the current
        animation has played.

        This does not influence the total runtime of the animation. Instead the runtime
        of individual animations is adjusted so that the complete animation has the defined
        run time.
    kwargs
        Further keyword arguments that are passed to `animation_class`.

    Examples
    --------
    .. arachne:: LaggedStartMapExample

        class LaggedStartMapExample(Scene):
            def construct(self):
                title = Tex("LaggedStartMap").to_edge(UP, buff=LARGE_BUFF)
                dots = VGroup(
                    *[Dot(radius=0.16) for _ in range(35)]
                    ).arrange_in_grid(rows=5, cols=7, buff=MED_LARGE_BUFF)
                self.add(dots, title)

                # Animate yellow ripple effect
                for mob in dots, title:
                    self.play(LaggedStartMap(
                        ApplyMethod, mob,
                        lambda m : (m.set_color, YELLOW),
                        lag_ratio = 0.1,
                        rate_func = there_and_back,
                        run_time = 2
                    ))
    """

    def __init__(
        self,
        animation_class: type[Animation],
        mobject: Mobject,
        arg_creator: Callable[[Mobject], Iterable[Any]] | None = None,
        run_time: float = 2,
        lag_ratio: float = DEFAULT_LAGGED_START_LAG_RATIO,
        **kwargs: Any,
    ):
        if arg_creator is None:

            def identity(mob: Mobject) -> tuple[Mobject]:
                return (mob,)

            arg_creator = identity

        args_list = [arg_creator(submob) for submob in mobject]
        anim_kwargs = dict(kwargs)
        if "lag_ratio" in anim_kwargs:
            anim_kwargs.pop("lag_ratio")
        animations = [animation_class(*args, **anim_kwargs) for args in args_list]
        super().__init__(*animations, run_time=run_time, lag_ratio=lag_ratio)
