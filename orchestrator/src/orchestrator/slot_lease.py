"""A workflow slot's lease, and the report on a slot that outlived its hard-cancel (task 6660).

A slot owns two things it must give back exactly once when it ends: its module
locks in the scheduler and its permit on the concurrency semaphore.
:class:`SlotLease` makes the release idempotent, so the two parties that may
release a slot cannot release it twice: ``Harness._run_slot``'s ``finally`` (the
normal path), and the terminal-status watcher's force-release of a slot whose
task row is already terminal but whose coroutine never ended after its
hard-cancel (``orchestrator/src/orchestrator/harness.py::Harness._force_release_wedged_slot``).
A second ``sem.release()`` would silently widen ``max_concurrent_tasks`` for the
rest of the run; a second ``scheduler.release`` after a re-dispatch would drop
the new dispatch's locks.

:func:`suspended_slot_tasks`, :func:`culprit` and :func:`wedged_slot_escalation`
build that force-release's loud signal: which await of which task holds the slot,
and the L1 a human reads.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Protocol

from escalation.categories import EscalationCategory
from escalation.models import Escalation

__all__ = [
    'SlotLease',
    'SuspendedTask',
    'await_chain',
    'culprit',
    'suspended_slot_tasks',
    'wedged_slot_escalation',
]


class _LockReleaser(Protocol):
    def release(self, task_id: str, *, requeued: bool = ...) -> None: ...


class SlotLease:
    """The release obligations of one ``_run_slot`` invocation."""

    def __init__(self, task_id: str, scheduler: _LockReleaser, sem: asyncio.Semaphore) -> None:
        self._task_id = task_id
        self._scheduler = scheduler
        self._sem = sem
        self._released = False

    @property
    def released(self) -> bool:
        return self._released

    def release(self, *, requeued: bool) -> bool:
        """Release the slot's module locks, then its semaphore permit; False on a repeat.

        *requeued* arms the scheduler's requeue cooldown, as
        ``scheduler.py::Scheduler.release`` documents. The permit is returned
        even if the lock release raises.
        """
        if self._released:
            return False
        self._released = True
        try:
            self._scheduler.release(self._task_id, requeued=requeued)
        finally:
            self._sem.release()
        return True


def await_chain(task: asyncio.Task) -> tuple[str, ...]:
    """``file:line in function`` for each coroutine *task* is suspended in, outermost first.

    ``Task.get_stack`` stops at the task's own coroutine frame; this follows
    ``cr_await`` down to the innermost suspended coroutine.
    """
    chain: list[str] = []
    awaitable: object = task.get_coro()
    while awaitable is not None:
        frame = getattr(awaitable, 'cr_frame', None) or getattr(awaitable, 'gi_frame', None)
        if frame is None:
            break
        code = frame.f_code
        chain.append(f'{code.co_filename}:{frame.f_lineno} in {code.co_name}')
        awaitable = getattr(awaitable, 'cr_await', None) or getattr(awaitable, 'gi_yieldfrom', None)
    return tuple(chain)


_ASYNCIO_DIR = os.path.dirname(asyncio.__file__) + os.sep


@dataclass(frozen=True)
class SuspendedTask:
    """One task of a wedged slot and the awaits it is suspended in, outermost first."""

    name: str
    awaits: tuple[str, ...]

    @property
    def culprit_frame(self) -> str | None:
        """The innermost await outside asyncio itself: the code that will not return."""
        for line in reversed(self.awaits):
            if not line.startswith(_ASYNCIO_DIR):
                return line
        return None


def suspended_slot_tasks(slot_task: asyncio.Task) -> tuple[SuspendedTask, ...]:
    """The slot task, then every live task named ``<slot task name>/...``, by name.

    ``CancellationScope`` runs a workflow's body and each terminal-cleanup
    entry as such child tasks (``<slot>/body``, ``<slot>/<entry>``), so the
    await that will not return is usually in a child, not in the slot task.
    """
    prefix = f'{slot_task.get_name()}/'
    children = sorted(
        (t for t in asyncio.all_tasks() if not t.done() and t.get_name().startswith(prefix)),
        key=lambda t: t.get_name(),
    )
    return tuple(SuspendedTask(t.get_name(), await_chain(t)) for t in (slot_task, *children))


def culprit(tasks: tuple[SuspendedTask, ...]) -> SuspendedTask:
    """The task whose await holds the slot: a cleanup entry, else the body, else the slot.

    With several cleanup entries alive (an earlier one outlived its bounded
    cancel) the first by name is chosen; the L1 lists every one.
    """
    slot, *children = tasks
    entries = [t for t in children if not t.name.endswith('/body')]
    return (entries or children or [slot])[0]


def wedged_slot_escalation(
    *,
    escalation_id: str,
    task_id: str,
    status: str,
    polls: int,
    modules: list[str],
    stuck: tuple[SuspendedTask, ...],
    filing_claimant_run_id: str | None,
) -> Escalation:
    """The L1 for a slot the watcher had to force-release."""
    blamed = culprit(stuck)
    listing = '\n'.join(
        f'{t.name}:\n' + ('\n'.join(f'  {line}' for line in t.awaits) or '  (no suspended coroutine frame)')
        for t in stuck
    )
    return Escalation(
        id=escalation_id,
        task_id=task_id,
        agent_role='orchestrator',
        severity='blocking',
        category=EscalationCategory.WORKFLOW_SLOT_WEDGED,
        summary=(
            f'Workflow slot for {status} task {task_id} never ended after its '
            f'hard-cancel; slot and {len(modules)} module lock(s) force-released'
        )[:200],
        detail=(
            f'Culprit: {blamed.name} at {blamed.culprit_frame}\n\n'
            f'The terminal-status watcher saw task {task_id} as {status!r} for '
            f'{polls} consecutive polls. The workflow ignored the soft cancel and '
            f'then the hard-cancel, so the harness released its workflow slot and '
            f'module locks itself. Nothing live was unlocked: the watcher acts only '
            f'on done/cancelled rows.\n\n'
            f'Modules held at release: {modules}\n\n'
            f'The Culprit line is the innermost await outside asyncio of '
            f'{blamed.name}, the task holding the slot. Every task of the slot, '
            f'with the awaits it is suspended in (outermost first):\n{listing}\n\n'
            f'The coroutines are left running and hold no slot. Find why the '
            f'Culprit await ignores cancellation.\n'
        ),
        suggested_action='investigate_wedged_workflow',
        level=1,
        filing_claimant_run_id=filing_claimant_run_id,
    )
