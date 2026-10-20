"""A confirmed pre-existing main break always gets an owner: one open fix-main task per failure signature.

The signature is sha-independent
(``shared/src/shared/task_metadata.py::RetryLedger.compute_merge_outcome_signature``),
so commits landing on a red main never multiply the task.  The task store is the
dedupe ledger: the signature is stored as ``metadata.main_health_signature`` and
looked up through ``find_tasks_by_metadata``.  An open hit is adopted; a done or
cancelled one means the break came back, so a fresh task is filed.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from shared.mcp_envelope import parse_tool_result
from shared.task_metadata import RetryLedger
from shared.task_metadata_wire import coerce_task_metadata
from shared.task_statuses import ACTIVE, TERMINAL, OwnerLiveness, owner_liveness
from shared.task_submit import SubmitOutcome, SubmittedTask

from orchestrator.chronic_flake import SchedulerChronicFlakeTaskClient
from orchestrator.main_health_actions import MainHealthNextAction

logger = logging.getLogger(__name__)

SIGNATURE_KEY = 'main_health_signature'

_LOOKUP_TIMEOUT_SECS = 15.0

_FIX_BRIEF_TITLE_MAX = 100
_FIX_BRIEF_DETAIL_MAX = 800

REPRODUCE_FIRST = (
    'Before changing any code, reproduce the failure on a clean checkout of main. '
    'The pre-existing-on-main verdict can be a false positive under host load '
    '(esc-3514-2). If main is green there, close this task as not reproducible '
    'and repair the verify environment instead.'
)


class FixSpawnContext(StrEnum):
    """Who put the fix-main task on the board (``metadata.spawn_context``)."""

    AUTO_HEAL = 'main_health_auto_heal'
    FILED = 'main_health_fix_filed'


@dataclass(frozen=True)
class MainBreak:
    """One confirmed pre-existing break, as the classifier reported it."""

    category: str
    cause_hint: str
    reason: str

    @property
    def signature(self) -> str:
        return RetryLedger.compute_merge_outcome_signature(
            self.category, self.cause_hint, self.reason,
        )


def compose_fix_main_brief(
    category: str, cause_hint: str, detail: str = '',
) -> tuple[str, str]:
    """Return the ``(title, description)`` of a fix-main task.

    The title is ``'fix main: <cause_hint or category>'``, cut to
    ``_FIX_BRIEF_TITLE_MAX`` characters; the description carries category,
    cause and at most ``_FIX_BRIEF_DETAIL_MAX`` characters of *detail*.
    """
    label = cause_hint.strip() if cause_hint and cause_hint.strip() else category
    title = f'fix main: {label}'[:_FIX_BRIEF_TITLE_MAX]
    detail_snippet = (detail or '').strip()[:_FIX_BRIEF_DETAIL_MAX]
    description = (
        f'Automated main-health fix task.\n'
        f'Category: {category}\n'
        f'Cause: {cause_hint}\n'
    )
    if detail_snippet:
        description += f'\nDetail:\n{detail_snippet}'
    return title, description


def build_main_fix_task_arguments(
    brk: MainBreak,
    *,
    project_root: str,
    spawn_context: FixSpawnContext,
    escalation_id: str | None = None,
) -> dict:
    """The ``submit_task`` arguments of a HIGH-lane fix-main task for *brk*."""
    title, description = compose_fix_main_brief(brk.category, brk.cause_hint, brk.reason)
    metadata: dict = {
        'merge_lane': 'high',
        'spawn_context': spawn_context.value,
        SIGNATURE_KEY: brk.signature,
    }
    if escalation_id is not None:
        metadata['main_health_escalation_id'] = escalation_id
    return {
        'title': title,
        'description': f'{description}\n\n{REPRODUCE_FIRST}',
        'priority': 'high',
        'project_root': project_root,
        'metadata': metadata,
    }


@dataclass(frozen=True)
class FixOwner:
    """The task that owns the fix for one signature, or ``task_id=None`` when none could be found or filed."""

    signature: str
    task_id: str | None = None
    filed: bool = False

    def next_action(self, otherwise: MainHealthNextAction) -> MainHealthNextAction:
        return MainHealthNextAction.FIX_TASK_FILED if self.task_id else otherwise

    def escalation_detail(self, text: str) -> str:
        if self.task_id is None:
            return text
        how = 'filed' if self.filed else 'adopted'
        return (
            f'{text}\n\nFix owner: task {self.task_id} ({how}; '
            f'{SIGNATURE_KEY}={self.signature})'
        )


class MainFixTaskStore(Protocol):
    """The task-store seam; ``orchestrator/src/orchestrator/scheduler.py::Scheduler`` satisfies it."""

    async def dispatch_tool(
        self, name: str, arguments: dict, *, timeout: float = ...,
    ) -> dict: ...

    async def submit_task(
        self, arguments: dict, *, timeout: float = ...,
    ) -> SubmitOutcome: ...


class MainFixTaskFiler:
    """Ensures one open fix-main task per signature.

    One instance per orchestrator process (held by
    ``orchestrator/src/orchestrator/merge_lane/worker.py::SpeculativeMergeWorker``),
    so the per-signature lock serialises lookup-then-file across every
    workflow and the merge worker that meet the same break at once.
    """

    def __init__(self, store: MainFixTaskStore) -> None:
        self._store = store
        self._locks: dict[str, asyncio.Lock] = {}

    async def ensure(
        self,
        brk: MainBreak,
        *,
        project_root: str,
        spawn_context: FixSpawnContext = FixSpawnContext.FILED,
        escalation_id: str | None = None,
    ) -> FixOwner:
        """Adopt the open fix-main task for *brk*'s signature, or file one.

        Never raises.  A failed lookup files nothing: an unreadable store is
        not evidence that no owner exists, and the next break retries.
        """
        sig = brk.signature
        tasks = SchedulerChronicFlakeTaskClient(self._store, project_root)
        async with self._locks.setdefault(sig, asyncio.Lock()):
            try:
                hit = await self._find_open(sig, project_root)
                if hit is not None:
                    task_id, status = hit
                    await _finish_half_filed(tasks, task_id, status)
                    return FixOwner(sig, task_id)
                arguments = build_main_fix_task_arguments(
                    brk, project_root=project_root,
                    spawn_context=spawn_context, escalation_id=escalation_id,
                )
                return await _file(tasks, sig, arguments)
            except Exception:
                logger.warning(
                    'main-health fix owner: could not ensure a fix-main task for '
                    'signature %s', sig, exc_info=True,
                )
                return FixOwner(sig)

    async def _find_open(self, sig: str, project_root: str) -> tuple[str, str] | None:
        result = await self._store.dispatch_tool(
            'find_tasks_by_metadata',
            {
                'project_root': project_root,
                'key': SIGNATURE_KEY,
                'value': sig,
                'statuses': sorted(ACTIVE),
            },
            timeout=_LOOKUP_TIMEOUT_SECS,
        )
        rows, err = parse_tool_result(result, 'results', list)
        if err is not None:
            raise LookupError(f'find_tasks_by_metadata failed: {err}')
        assert rows is not None  # parse_tool_result invariant: (None, err) | (value, None)
        return (str(rows[0]['task_id']), str(rows[0]['status'])) if rows else None


async def _file(
    tasks: SchedulerChronicFlakeTaskClient, sig: str, arguments: dict,
) -> FixOwner:
    outcome = await tasks.submit_task({**arguments, 'planning_mode': True})
    match outcome:
        case SubmittedTask(task_id=task_id, combined=False):
            await tasks.commit_planning([task_id])
            return FixOwner(sig, task_id, filed=True)
        case SubmittedTask(task_id=task_id, status=status) if (
            status is not None and status not in TERMINAL
        ):
            return FixOwner(sig, task_id)
    logger.warning(
        'main-health fix owner: filing a fix-main task for signature %s did '
        'not yield an open task: %r', sig, outcome,
    )
    return FixOwner(sig)


async def _finish_half_filed(
    tasks: SchedulerChronicFlakeTaskClient, task_id: str, status: str,
) -> None:
    """Release an owner this filer created whose ``commit_planning`` never landed.

    The ``deferred`` arm of
    ``orchestrator/src/orchestrator/flake_ledger.py::_ensure_owner_task``: a
    ``planning_mode`` task left ``deferred`` would own its signature forever
    without ever being dispatched.  A deferral someone stamped
    (``metadata.deferral``, e.g. a hand carry) is theirs to release, and is
    left alone, as is a deferred task some other writer filed.
    """
    if owner_liveness(status) is not OwnerLiveness.DEFERRED:
        return
    task, read_error = await tasks.get_task(task_id)
    if task is None:
        logger.warning(
            'main-health fix owner: could not read deferred owner %s (%s); the '
            'next break retries', task_id, read_error,
        )
        return
    metadata = coerce_task_metadata(task.get('metadata')) or {}
    if metadata.get('spawn_context') in set(FixSpawnContext) and 'deferral' not in metadata:
        logger.info('main-health fix owner: completing the half-filed task %s', task_id)
        await tasks.commit_planning([task_id])


async def ensure_main_fix_owner(
    filer: MainFixTaskFiler | None,
    brk: MainBreak,
    *,
    project_root: str,
    spawn_context: FixSpawnContext = FixSpawnContext.FILED,
    escalation_id: str | None = None,
) -> FixOwner:
    """*filer*'s owner for *brk*, or an ownerless :class:`FixOwner` when no task store is wired."""
    if filer is None:
        return FixOwner(brk.signature)
    return await filer.ensure(
        brk, project_root=project_root,
        spawn_context=spawn_context, escalation_id=escalation_id,
    )
