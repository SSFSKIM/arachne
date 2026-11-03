"""What a stepping plan reads when it is made stands for the whole play only
when no code can change it: a plan resting on a value of a mutable kind
(what a list, a dict, an object, a class holds) ticks everything, whatever
changes the value; a generator's items run its code.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from . import plan_read_scenes
from .test_stepping import _same_as_full_ticks

PLAN_READ_SCENES = Path(plan_read_scenes.__file__)


@pytest.mark.parametrize(
    "name",
    [
        "StoredLambdaRead",
        "StoredLambdaItemRead",
        "SetterMethodRead",
        "ClassAttributeRead",
        "GlobalsItemRead",
        "OperatorItemRead",
        "EngineStoredItemRead",
        "AliasedAppendRead",
        "UnboundAppendRead",
        "PropertyRead",
    ],
)
def test_a_callable_stored_after_the_plan_is_made_is_called(session_of, name):
    """A function stored mid-play where another callback's call reads it
    (a list's or a dict's item, by a method, an alias, a function of the
    engine's or ``operator``; an object's, a class's, a property's; a module
    global): the frames equal those of ticking everything, the play ticking
    everything or both callbacks running at every tick.
    """
    play = _same_as_full_ticks(session_of, name, 4, PLAN_READ_SCENES).record.plays[-1]
    assert play.stepping_ticks == 0 or play.tick_plan.pulled == 2


@pytest.mark.parametrize(
    "name",
    [
        "LoopedGeneratorRead",
        "ComprehendedGeneratorRead",
        "SearchedGeneratorRead",
        "DefaultedGeneratorRead",
    ],
)
def test_iterating_a_generator_runs_its_code(session_of, name):
    """A generator moving ``shared`` with each item, run by ``for``, by a
    generator expression's ``for``, by ``in`` and by a ``for`` over a
    parameter's default: iterating is a call, and a generator is code the
    plan can't follow.
    """
    play = _same_as_full_ticks(session_of, name, 4, PLAN_READ_SCENES).record.plays[-1]
    assert play.stepping_ticks == 0 or play.tick_plan.pulled == 1


@pytest.mark.parametrize(
    "name",
    [
        "ReboundReceiverWrite",
        "ReboundItemWrite",
        "ReboundDefaultWrite",
        "ReboundCellWrite",
        "ReboundMethodWrite",
    ],
)
def test_a_receiver_rebound_after_the_plan_is_made_is_written(session_of, name):
    """The mobject a callback shifts, held by an object's attribute, a
    list's item, a function's default, a closure's cell or found by a
    class's method, rebound mid-play to the one a stepping reader reads:
    the frames equal those of ticking everything, the play ticking
    everything or both callbacks running at every tick.
    """
    play = _same_as_full_ticks(session_of, name, 4, PLAN_READ_SCENES).record.plays[-1]
    assert play.stepping_ticks == 0 or play.tick_plan.pulled == 2


@pytest.mark.parametrize(
    "name",
    [
        "OperatorHookRead",
        "TruthHookRead",
        "FormatHookRead",
        "PropertyHookRead",
        "MissingAttributeHookRead",
        "DefaultedOperatorHookRead",
        "DefaultedPropertyHookRead",
    ],
)
def test_what_python_runs_on_an_object_of_the_scenes_class_is_called(session_of, name):
    """An operator, a truth test, a formatting, a property and
    ``__getattr__`` of an object of the scene's class, held by a closure or
    a parameter's default, run its code, which moves ``shared``: the
    callback is not skipped.
    """
    play = _same_as_full_ticks(session_of, name, 4, PLAN_READ_SCENES).record.plays[-1]
    assert play.stepping_ticks == 0 or play.tick_plan.pulled == 1


def test_a_callback_writing_its_own_family_steps(session_of):
    """The control: a callback shifting only the mobject it updates rests
    on nothing mutable the plan read, and the play steps past it.
    """
    play = _same_as_full_ticks(
        session_of, "OwnFamilyWrite", 4, PLAN_READ_SCENES
    ).record.plays[-1]
    assert play.stepping_ticks > 0
    assert play.tick_plan.pulled == 0
