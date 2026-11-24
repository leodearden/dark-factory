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

A slot may also run on a borrowed permit: :meth:`SlotLease.lend` builds a lease
that takes no permit and names its lender. When the lender is released while
its borrower runs, the lender's permit passes to the borrower instead of going
back to the pool; a borrower that ends first returns nothing. The rules are
``plans/early-start-on-predecessor-branch-prd.md`` section 5.3 "Slot" and
invariant 4. :meth:`SlotLease.wait_released` is the slot-ended signal, set once
the locks are released and the permit is handed over or returned.

:func:`suspended_slot_tasks`, :func:`culprit` and :func:`wedged_slot_escalation`
build that force-release's loud signal: which await of which task holds the slot,
and the L1 a human reads.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from escalation.categories import EscalationCategory
from escalation.models import Escalation

__all__ = [
    'LendingRule',
    'SlotLease',
    'SlotLendingError',
    'SuspendedTask',
    'await_chain',
    'culprit',
    'suspended_slot_tasks',
    'wedged_slot_escalation',
]


logger = logging.getLogger(__name__)


class _LockReleaser(Protocol):
    def release(self, task_id: str, *, requeued: bool = ...) -> None: ...


class LendingRule(Enum):
    """The slot rule a refused lend breaks; each value says it in prose."""

    OWN_TASK = 'a lease cannot lend to its own task'
    LENDER_RELEASED = 'the lender is already released'
    ALREADY_LENDS = 'the lender already lends to a live borrower'
    NO_OWN_PERMIT = 'the lender holds no permit of its own; a borrower lends only after it inherits'
    NO_LIVE_SLOT = 'a borrower borrows from a live slot, and the lender has none'
    BORROWER_HAS_SLOT = 'the borrower already has a live slot of its own'


class SlotLendingError(RuntimeError):
    """A lend the slot rules refuse: which *rule* it breaks, and the two tasks."""

    def __init__(self, rule: LendingRule, *, lender_task_id: str, borrower_task_id: str) -> None:
        super().__init__(
            f'slot lending refused: task {lender_task_id} cannot lend to task {borrower_task_id}: {rule.value}',
        )
        self.rule = rule
        self.lender_task_id = lender_task_id
        self.borrower_task_id = borrower_task_id


class SlotLease:
    """The release obligations of one slot: its module locks, and the permit it holds or borrows."""

    def __init__(self, task_id: str, scheduler: _LockReleaser, sem: asyncio.Semaphore) -> None:
        self._bind(task_id, scheduler, permit=sem, lender=None)

    def _bind(
        self,
        task_id: str,
        scheduler: _LockReleaser,
        *,
        permit: asyncio.Semaphore | None,
        lender: SlotLease | None,
    ) -> None:
        self._task_id = task_id
        self._scheduler = scheduler
        self._permit = permit
        self._lender = lender
        self._borrower: SlotLease | None = None
        self._released = False
        self._released_event = asyncio.Event()

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def lender(self) -> SlotLease | None:
        """The lease this one borrowed from; still set after it inherits the lender's permit."""
        return self._lender

    @property
    def borrower(self) -> SlotLease | None:
        """The live lease borrowing this one's permit, if any."""
        return self._borrower

    @property
    def holds_permit(self) -> bool:
        return self._permit is not None

    @property
    def released(self) -> bool:
        return self._released

    def lend(self, task_id: str) -> SlotLease:
        """A lease for *task_id* that takes no permit and inherits this one's when this one is released."""
        rule = self._broken_lending_rule(task_id)
        if rule is not None:
            raise SlotLendingError(rule, lender_task_id=self._task_id, borrower_task_id=task_id)
        borrower = SlotLease.__new__(SlotLease)
        borrower._bind(task_id, self._scheduler, permit=None, lender=self)
        self._borrower = borrower
        return borrower

    def _broken_lending_rule(self, borrower_task_id: str) -> LendingRule | None:
        if borrower_task_id == self._task_id:
            return LendingRule.OWN_TASK
        if self._released:
            return LendingRule.LENDER_RELEASED
        if self._borrower is not None:
            return LendingRule.ALREADY_LENDS
        if self._permit is None:
            return LendingRule.NO_OWN_PERMIT
        return None

    async def wait_released(self) -> None:
        """Return once this lease is released: its locks freed and its permit handed over or returned."""
        await self._released_event.wait()

    def release(self, *, requeued: bool) -> bool:
        """Release the slot's module locks, settle its permit, then signal; False on a repeat.

        *requeued* arms the scheduler's requeue cooldown, as
        ``scheduler.py::Scheduler.release`` documents. A held permit passes to
        a live borrower, otherwise back to its pool; a borrower that never
        inherited gives nothing back. Each step runs even if the one before
        raises.
        """
        if self._released:
            return False
        self._released = True
        try:
            self._scheduler.release(self._task_id, requeued=requeued)
        finally:
            try:
                self._settle_permit()
            finally:
                self._released_event.set()
        return True

    def _settle_permit(self) -> None:
        permit, self._permit = self._permit, None
        borrower, self._borrower = self._borrower, None
        if permit is None:
            if self._lender is not None and self._lender._borrower is self:
                self._lender._borrower = None
        elif borrower is not None:
            borrower._permit = permit
            logger.info("task %s's slot ended; its permit passes to borrower %s", self._task_id, borrower._task_id)
        else:
            permit.release()


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
