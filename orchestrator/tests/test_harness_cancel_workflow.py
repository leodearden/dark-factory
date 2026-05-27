"""Tests for ``Harness.cancel_workflow`` — Step 4 of zombie-escalation fix.

Verifies the registry-and-set behaviour without spinning up a real workflow:

* ``cancel_workflow`` returns ``False`` when the task has no slot.
* ``cancel_workflow`` sets the registered Event when the slot exists.
* ``is_workflow_active`` mirrors registry presence.
"""

from __future__ import annotations

import asyncio

import pytest
from _orch_helpers import _init_harness_state_for_test

from orchestrator.harness import Harness


@pytest.fixture
def harness() -> Harness:
    # cancel_workflow / is_workflow_active read only ``_workflow_cancel_events``.
    # Building a full Harness requires a real OrchestratorConfig; bypass __init__
    # and populate only the fields we exercise.
    h = Harness.__new__(Harness)
    _init_harness_state_for_test(h)  # task 1449: initialise digest counters
    h._workflow_cancel_events = {}
    # cancel_workflow stamps wall-clock for the reconcile sweep's grace
    # window (R3 race guard).  Initialise here so the bypass-__init__
    # fixture matches the real Harness attribute set.
    h._workflow_cancel_at = {}
    # hard_cancel_workflow reads _workflow_slot_tasks (task 1491, step-6).
    h._workflow_slot_tasks = {}
    return h


def test_cancel_workflow_returns_false_when_no_slot(harness: Harness):
    assert harness.cancel_workflow('does-not-exist') is False
    assert harness.is_workflow_active('does-not-exist') is False


def test_cancel_workflow_sets_event_when_slot_exists(harness: Harness):
    event = asyncio.Event()
    harness._workflow_cancel_events['42'] = event

    assert harness.is_workflow_active('42') is True
    assert event.is_set() is False

    assert harness.cancel_workflow('42') is True
    assert event.is_set() is True


def test_cancel_workflow_idempotent(harness: Harness):
    event = asyncio.Event()
    event.set()
    harness._workflow_cancel_events['42'] = event

    # Already-set event is fine; cancel still reports active=True.
    assert harness.cancel_workflow('42') is True
    assert event.is_set() is True


# ---------------------------------------------------------------------------
# hard_cancel_workflow — task 1491, ITEM 2
# ---------------------------------------------------------------------------


def test_hard_cancel_workflow_returns_false_when_no_slot(harness: Harness):
    """Returns False when no slot task is registered for the task_id."""
    assert harness.hard_cancel_workflow('does-not-exist') is False


@pytest.mark.asyncio
async def test_hard_cancel_workflow_cancels_registered_task(harness: Harness):
    """When a live asyncio.Task is registered, hard_cancel_workflow cancels it
    and returns True."""
    slot_task = asyncio.ensure_future(asyncio.sleep(3600))
    harness._workflow_slot_tasks['42'] = slot_task

    result = harness.hard_cancel_workflow('42')
    assert result is True

    # Allow the event loop to propagate the cancellation.
    with pytest.raises(asyncio.CancelledError):
        await slot_task
    assert slot_task.cancelled()


@pytest.mark.asyncio
async def test_hard_cancel_workflow_returns_false_for_done_task(harness: Harness):
    """Returns False when the registered task is already done."""
    slot_task = asyncio.ensure_future(asyncio.sleep(0))
    await slot_task  # let it complete

    harness._workflow_slot_tasks['42'] = slot_task
    assert slot_task.done()

    result = harness.hard_cancel_workflow('42')
    assert result is False
