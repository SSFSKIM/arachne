"""Which route a play takes, decided when it begins.

:func:`admit` reads the play about to run and returns its
:class:`Admission`: the route, why it is not on a higher one, and, for a
state-function play, the proof of every callback.

1. **Trusted** (route 1): every animation is in the trusted algebra
   (:mod:`.trusted`), its events have instants, and nothing else runs: no
   updater on anything (an updater on a mobject whose updating the play
   suspends throughout does not run) and no camera callback.
2. **State function** (route 2): the animations are trusted or
   ``UpdateFromFunc``/``UpdateFromAlphaFunc``; every callback that runs
   during the play (those, the mobjects' dt-free updaters, ``always_redraw``,
   the camera's centre functions) is proven a function of the instant
   (:mod:`.purity`); and no callback reads what another writes after it in
   CE's update order, which would hand it the previous sample's value.
3. **Ticked** (route 3): everything else, stepped on the 60 Hz tick; at a
   tick no frame reads, only the work that integrates and what it may read
   runs (:mod:`.ticking`). A dt updater whose code never reads ``dt`` still
   ticks its play: the tick plan, not admission, treats it as dt-free.

A play whose camera draws through inputs the record can't hold
(:func:`camera_refusal`) ticks too, and its frames are drawn while it runs.
So does a play the probe demoted (:mod:`.probe`): restored at an instant
its record kept, it differed from what was kept there, so its callbacks keep
history their proofs did not see. A demotion is for good: every later
recording a session makes of that scene of the same source ticks the play
(:data:`DEMOTED`); a render reads none.
"""

from __future__ import annotations

__all__ = [
    "DEMOTED",
    "PROBE_DEMOTED",
    "Admission",
    "Callback",
    "PlayIdentity",
    "admit",
    "camera_refusal",
    "demote",
    "demoted",
    "line_plays",
    "play_identities",
    "play_identity",
    "whole_line",
]

import inspect
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from typing import TYPE_CHECKING, Any

from .dispatch import engine_patches, user_overrides
from .identity import ByIdentity, one_of
from .purity import Proof, Refused, prove_builder, prove_readout, prove_setter_body
from .source import SourceLine, relative_path

if TYPE_CHECKING:
    from arachne.animation.animation import Animation
    from arachne.mobject.mobject import Mobject
    from arachne.scene.scene import Scene

    from .events import EventPlan

TRUSTED = "trusted"
STATE = "state"
TICKED = "ticked"


@dataclass(frozen=True)
class Callback:
    """Scene code that runs while a play is evaluated.

    ``kind`` is ``"updater"`` (a mobject's dt-free updater), ``"redraw"``
    (``always_redraw``'s), ``"alpha_func"`` or ``"func"``
    (``UpdateFromAlphaFunc``'s and ``UpdateFromFunc``'s function),
    ``"center"`` (a camera's centre function) or ``"dt_updater"``.
    ``function`` is the code that was proven: for ``always_redraw`` the
    function it rebuilds from. ``owner`` is the mobject it updates.
    ``proof`` holds what it reads, writes and the names it read, when it was
    proven; ``refusal`` says why it was not.
    """

    kind: str
    function: Callable[..., Any]
    owner: Mobject | None
    source_line: SourceLine | None
    proof: Proof | None = None
    refusal: str | None = None


@dataclass(frozen=True)
class Admission:
    """The route of one play and the reasons for it."""

    route: str
    #: Why the play is not on a higher route: empty for a trusted play.
    reasons: tuple[str, ...] = ()
    #: The callbacks that run while it plays, in CE's update order, the
    #: camera's last.
    callbacks: tuple[Callback, ...] = field(default=())


#: The code of a demoted play's reason for ticking.
PROBE_DEMOTED = "probe.demoted"

#: A play as executions of one scene know it: its source line, its kind
#: (``"play"`` or ``"wait"``) and how many plays of that line came before
#: it. Its index is not: an execution that plays one play more before it
#: shifts every index after. ``(line, None, None)`` stands for every play of
#: the line (:func:`whole_line`).
PlayIdentity = tuple[SourceLine | None, str | None, int | None]


def play_identity(play: Any, counts: dict[SourceLine | None, int]) -> PlayIdentity:
    """The :data:`PlayIdentity` of ``play``, given how many plays of each
    line came before it (``counts``, which it counts on).
    """
    count = counts.get(play.source_line, 0)
    counts[play.source_line] = count + 1
    return (play.source_line, play.kind, count)


def play_identities(plays: Sequence[Any]) -> list[PlayIdentity]:
    """The :data:`PlayIdentity` of each of ``plays`` (a record's, in order)."""
    counts: dict[SourceLine | None, int] = {}
    return [play_identity(play, counts) for play in plays]


def line_plays(plays: Sequence[Any]) -> dict[SourceLine | None, tuple[str, ...]]:
    """The kinds of the plays of each line, in order: what tells whether the
    plays of a line in two executions correspond one to one.
    """
    found: dict[SourceLine | None, tuple[str, ...]] = {}
    for play in plays:
        found[play.source_line] = (*found.get(play.source_line, ()), play.kind)
    return found


def whole_line(line: SourceLine | None) -> PlayIdentity:
    """The identity that stands for every play of ``line``."""
    return (line, None, None)


#: The plays the probe demoted, as (source digest, scene class, play
#: identity, the kinds of that line's plays in the execution it was found
#: in): those a session's recording of that scene ticks, for the rest of the
#: process. A source holds many scenes, so the scene's name is part of the
#: key. A play is known by its identity, so a demotion ticks the play it was
#: found in wherever it falls in another execution; while that execution's
#: plays of the line are not the same kinds in the same order, which play it
#: was is unknown, and the session demotes the whole line (``None`` kinds).
DEMOTED: set[tuple[str, str, PlayIdentity, tuple[str, ...] | None]] = set()


def demote(
    source_digest: str,
    scene_class: str,
    identity: PlayIdentity,
    line: tuple[str, ...] | None,
) -> None:
    """Tick the play ``identity`` of the scene ``scene_class`` of the source
    ``source_digest``, found where its line played ``line``, in every
    recording a session makes of it from now on.
    """
    DEMOTED.add((source_digest, scene_class, identity, line))


def demoted(
    source_digest: str | None, scene_class: str
) -> dict[PlayIdentity, tuple[str, ...] | None]:
    """The plays of the scene ``scene_class`` of the source
    ``source_digest`` the probe demoted, each with its line's plays where it
    was found; none for an unknown source.
    """
    if source_digest is None:
        return {}
    return {
        identity: line
        for digest, name, identity, line in DEMOTED
        if digest == source_digest and name == scene_class
    }


def camera_refusal(camera: Any) -> str | None:
    """Why frames of a scene drawn by ``camera`` can't come from its record:
    the camera draws through an input the record does not hold. ``None`` for
    the cameras whose every drawing input is a recorded value.
    """
    from arachne.camera.camera import Camera
    from arachne.camera.mapping_camera import MappingCamera
    from arachne.camera.moving_camera import MovingCamera
    from arachne.camera.multi_camera import MultiCamera
    from arachne.camera.three_d_camera import ThreeDCamera

    kind = type(camera)
    if kind is MultiCamera:
        # It draws each display through that display's own camera.
        for display in camera.image_mobjects_from_cameras:
            reason = camera_refusal(display.camera)
            if reason is not None:
                return reason
        return None
    if one_of(kind, (Camera, MovingCamera, ThreeDCamera)):
        return None
    if isinstance(camera, MappingCamera):
        return (
            "camera.mapping_func: a MappingCamera draws through its mapping_func, scene"
            " code read at draw time, so its frames are drawn while the scene runs and"
            " its record answers no instant"
        )
    return (
        f"camera.unsupported: a {kind.__name__} draws through inputs a record does not"
        " hold, so its frames are drawn while the scene runs and its record answers"
        " no instant"
    )


def admit(
    scene: Scene,
    plan: EventPlan,
    *,
    held: bool,
    base_dir: str | None = None,
    refusal: str | None = None,
    demoted: bool = False,
) -> Admission:
    """The route of the play ``scene`` is about to evaluate; callbacks' source
    lines are relative to ``base_dir``, as the record's are. A record that
    draws no frame (``refusal``) ticks, and so does a play the probe
    ``demoted`` that would otherwise be a state-function play.
    """
    from . import trusted

    camera = scene.renderer.camera
    refusal = refusal or camera_refusal(camera)
    if refusal is not None:
        return Admission(TICKED, (refusal,))
    patched = engine_patches()
    if patched:
        return Admission(
            TICKED,
            (
                f"the scene replaces the engine's {', '.join(patched)}, which every"
                " instance then runs in place of what the proofs vouch for",
            ),
        )
    animations = list(scene.animations or [])
    reasons: list[str] = []
    callbacks: list[Callback] = []
    if not held:
        if scene.updaters:
            reasons.append("the scene has an updater, which takes dt")
        if scene.stop_condition is not None:
            reasons.append("a stop condition ends it at the first tick it holds")
        if not plan.determined:
            reasons.append(
                "a Succession's timing runs back or is a custom function, so its parts"
                " begin at ticks"
            )
        callbacks.extend(_animation_callbacks(scene, animations, reasons))
        callbacks.extend(_updater_callbacks(scene, animations))
        updating = _update_replaced(scene)
        if updating is not None:
            reasons.append(updating)
    callbacks.extend(_camera_callbacks(scene, camera))
    callbacks = [_relative_line(callback, base_dir) for callback in callbacks]
    callback_animations = [
        a for a in trusted.iter_animations(animations) if _is_callback_animation(a)
    ]
    algebra = (
        []
        if held
        else trusted.refusals(animations, _callback_writes(animations, callbacks))
    )
    if not reasons and not algebra and not callbacks and not callback_animations:
        return Admission(TRUSTED)
    refused = [c for c in callbacks if c.refusal is not None]
    state_reasons = [
        *reasons,
        *algebra,
        *(f"{c.kind} at {c.source_line or '?'}: {c.refusal}" for c in refused),
    ]
    if not state_reasons:
        order = _dependency_refusal(scene, animations, callbacks)
        if order is not None:
            state_reasons.append(order)
    if state_reasons:
        return Admission(TICKED, tuple(state_reasons), tuple(callbacks))
    if demoted:
        return Admission(
            TICKED,
            (
                f"{PROBE_DEMOTED}: restored at an instant its record kept, it differed"
                " from what was kept there, so its callbacks keep history their"
                " proofs did not see",
            ),
            tuple(callbacks),
        )
    return Admission(
        STATE,
        ("callbacks run while it plays",),
        tuple(callbacks),
    )


# Collecting callbacks


def _callback_writes(
    animations: Sequence[Animation], callbacks: Sequence[Callback]
) -> dict[int, set[int]]:
    """What each callback animation's function writes, by the animation's
    id, from its proof: the mobjects it sets, the families included.
    """
    from . import trusted

    by_function = {id(c.function): c for c in callbacks if c.proof is not None}
    found: dict[int, set[int]] = {}
    for animation in trusted.iter_animations(animations):
        if _is_callback_animation(animation):
            callback = by_function.get(id(animation.update_function))  # type: ignore[attr-defined]
            if callback is not None and callback.proof is not None:
                found[id(animation)] = _family_ids(callback.proof.writes)
    return found


def _is_callback_animation(animation: Animation) -> bool:
    from .trusted import CALLBACK_ANIMATIONS

    return type(animation) in CALLBACK_ANIMATIONS


def _relative_line(callback: Callback, base_dir: str | None) -> Callback:
    line = callback.source_line
    if line is None:
        return callback
    return replace(
        callback, source_line=SourceLine(relative_path(line.path, base_dir), line.line)
    )


def _line(function: Any) -> SourceLine | None:
    code = getattr(function, "__code__", None)
    if code is None:
        return None
    return SourceLine(code.co_filename, code.co_firstlineno)


def _animation_callbacks(
    scene: Scene, animations: Sequence[Animation], reasons: list[str]
) -> Iterator[Callback]:
    """``UpdateFromFunc`` and ``UpdateFromAlphaFunc`` functions, and updaters
    that the animations' own copies carry (CE runs those on the copies).
    """
    from arachne.animation.composition import AnimationGroup
    from arachne.animation.updaters.update import UpdateFromAlphaFunc

    from . import trusted

    for animation in trusted.iter_animations(animations):
        if isinstance(animation, AnimationGroup):
            continue
        if _is_callback_animation(animation):
            alpha = type(animation) is UpdateFromAlphaFunc
            function = animation.update_function  # type: ignore[attr-defined]
            yield _prove(
                "alpha_func" if alpha else "func",
                function,
                animation.mobject,
                partial(
                    prove_setter_body,
                    function,
                    animation.mobject,
                    scene,
                    takes_alpha=alpha,
                ),
            )
        if _copies_carry_updaters(animation):
            reasons.append(
                f"{type(animation).__name__}'s copies carry updaters, which CE runs on them"
            )
            return


def _copies_carry_updaters(animation: Animation) -> bool:
    """Whether CE runs updaters on the animation's own copies (its starting
    mobject, its target and its copy), or will once it begins: copies take
    the updaters of what they copy.
    """
    try:
        copies: list[Any] = list(animation.get_all_mobjects_to_update())
    except AttributeError:
        # A Succession part not begun yet copies its mobject and target then.
        copies = [
            animation.mobject,
            getattr(animation, "target_mobject", None),
        ]
    return any(
        member.updaters
        for copy in copies
        if copy is not None
        for member in copy.get_family()
    )


def _updater_callbacks(
    scene: Scene, animations: Sequence[Animation]
) -> Iterator[Callback]:
    """The mobjects' updaters, in the order ``Scene.update_mobjects`` runs
    them, leaving out mobjects whose updating stays suspended all play.
    """
    resumed = _resumed_during_play(animations)
    seen: set[tuple[int, int]] = set()

    def visit(mob: Mobject) -> Iterator[Callback]:
        if mob.updating_suspended and id(mob) not in resumed:
            return
        for updater in mob.updaters:
            key = (id(mob), id(updater))
            if key not in seen:
                seen.add(key)
                yield _classify_updater(scene, mob, updater)
        for child in mob.submobjects:
            yield from visit(child)

    for mob in scene.mobjects:
        yield from visit(mob)


def _update_replaced(scene: Scene) -> str | None:
    """Why ``Scene.update_mobjects`` may not run the updaters the proofs
    read: a mobject whose ``update`` (which runs its updaters, then its
    children's) is a scene's code, on the instance or in its class.
    """
    classes: ByIdentity[type, bool] = ByIdentity()
    for mob in scene.mobjects:
        for member in mob.get_family():
            cls = type(member)
            if cls not in classes:
                classes[cls] = "update" in user_overrides(cls)
            if classes[cls] or callable(vars(member).get("update")):
                return (
                    f"a {type(member).__name__}'s update is the scene's code, so what"
                    " runs at a tick is not the updaters it holds"
                )
    return None


def _resumed_during_play(animations: Sequence[Animation]) -> set[int]:
    """The mobjects whose updating a Succession part may resume mid-play: the
    families of the parts it finishes before the play ends.
    """
    from arachne.animation.composition import Succession

    from . import trusted

    found: set[int] = set()
    for animation in trusted.iter_animations(animations):
        if isinstance(animation, Succession):
            for part in trusted.iter_animations(animation.animations):
                if part.mobject is not None:
                    found.update(id(m) for m in part.mobject.get_family())
    return found


def _redraw_code() -> Any:
    """The code of the updater ``always_redraw`` attaches."""
    from arachne.animation.updaters.mobject_update_utils import always_redraw

    return next(
        const
        for const in always_redraw.__code__.co_consts
        if inspect.iscode(const) and const.co_name == "<lambda>"
    )


def _classify_updater(scene: Scene, mob: Mobject, updater: Any) -> Callback:
    try:
        parameters = inspect.signature(updater).parameters
    except (TypeError, ValueError):
        return Callback(
            "updater", updater, mob, _line(updater), refusal="it has no signature"
        )
    if "dt" in parameters:
        return Callback(
            "dt_updater",
            updater,
            mob,
            _line(updater),
            refusal="it takes dt, so it integrates over the tick",
        )
    code = getattr(updater, "__code__", None)
    if code is not None and code is _redraw_code():
        cells = dict(
            zip(
                code.co_freevars,
                (c.cell_contents for c in updater.__closure__),
                strict=True,
            )
        )
        func, target = cells["func"], cells["mob"]

        def prove() -> Proof:
            proof = prove_builder(func, scene)
            proof.writes.append(target)
            return proof

        return _prove("redraw", func, target, prove)
    return _prove(
        "updater", updater, mob, lambda: prove_setter_body(updater, mob, scene)
    )


def _camera_callbacks(scene: Scene, camera: Any) -> Iterator[Callback]:
    for display in vars(camera).get("image_mobjects_from_cameras", ()):
        if display.camera is not camera:
            yield from _camera_callbacks(scene, display.camera)
    orientations = getattr(camera, "fixed_orientation_mobjects", None)
    if not orientations:
        return
    seen: set[int] = set()
    for mob, center in orientations.items():
        if id(center) in seen:
            continue
        seen.add(id(center))
        yield _prove("center", center, mob, partial(prove_readout, center, scene))


def _prove(
    kind: str, function: Any, owner: Any, prove: Callable[[], Proof]
) -> Callback:
    try:
        proof = prove()
    except Refused as refused:
        return Callback(kind, function, owner, _line(function), refusal=str(refused))
    return Callback(kind, function, owner, _line(function), proof=proof)


# Dependency order


@dataclass
class _Step:
    reads: set[int]
    writes: set[int]
    what: str


def _family_ids(mobjects: Sequence[Any]) -> set[int]:
    found: set[int] = set()
    for mob in mobjects:
        found.update(id(member) for member in mob.get_family())
    return found


def _dependency_refusal(
    scene: Scene, animations: Sequence[Animation], callbacks: Sequence[Callback]
) -> str | None:
    """Why a callback would read a value another writes after it in CE's
    update order (the animations, the Succession parts begun at events, the
    scene's mobjects' updaters), or ``None``.
    """
    from arachne.animation.composition import AnimationGroup, Succession

    from . import trusted

    by_function = {id(c.function): c for c in callbacks}
    steps: list[_Step] = []
    # Phase 1: the animations, in order; a callback animation reads and writes.
    for animation in trusted.iter_animations(animations):
        if isinstance(animation, AnimationGroup):
            continue
        if _is_callback_animation(animation):
            callback = by_function[id(animation.update_function)]  # type: ignore[attr-defined]
            assert callback.proof is not None
            steps.append(
                _Step(
                    _family_ids(callback.proof.reads),
                    _family_ids(callback.proof.writes),
                    f"the {callback.kind} at {callback.source_line}",
                )
            )
        else:
            steps.append(
                _Step(set(), trusted.writes(animation), type(animation).__name__)
            )
    # Phase 3: a Succession part begun at an event reads its mobjects there.
    for animation in trusted.iter_animations(animations):
        if isinstance(animation, Succession):
            for part in animation.animations[1:]:
                reads: set[int] = set()
                for leaf in trusted.iter_animations([part]):
                    for mob in (leaf.mobject, getattr(leaf, "target_mobject", None)):
                        if mob is not None:
                            reads |= _family_ids([mob])
                steps.append(
                    _Step(reads, set(), f"{type(part).__name__} begun at its event")
                )
    # Phase 4: the updaters, in update order.
    steps.extend(
        _Step(
            _family_ids(callback.proof.reads),
            _family_ids(callback.proof.writes),
            f"the {callback.kind} at {callback.source_line}",
        )
        for callback in callbacks
        if callback.kind in ("updater", "redraw") and callback.proof is not None
    )
    for index, step in enumerate(steps):
        for later in steps[index:]:
            if later is step and not step.reads:
                continue
            if step.reads & later.writes:
                if later is step:
                    return f"{step.what} reads what it writes"
                return (
                    f"{step.what} reads what {later.what} writes after it, so it would"
                    " see the previous sample's value"
                )
    return None
