"""The flake-debt DRAIN: files the de-flake owners an observation owes, after it.

plans/offline-lane-red-path-prd.md D6, §5.8 and §14 item 24 (task 6934, ε0).  An
observation writes its occurrence rows and one due item per suppressed test in ONE
local transaction (``orchestrator/src/orchestrator/flake_ledger.py::enqueue_debt``), so
nothing on the merge path waits on fused-memory.  This module turns each item into
today's :func:`~orchestrator.flake_ledger.open_debt` call — the lazy close, the
regression report and the owner filing of plans/flake-ledger-prd.md §5.9, unchanged —
plus the stale-head rule, and deletes the item once its debt is settled.

:func:`run_drain_pass` is one pass, and passes share no state beyond the ledger.  The
harness runs it every :data:`DRAIN_INTERVAL_SECS` as the ``flake-debt-drain``
background service.  A pass:

1. evaluates the stale-item escalation FIRST, before any network call, so a slow or
   hung pass cannot delay it (INV-4, INV-7);
2. reads at most 50 due items in drain order and settles each, a test's later items
   waiting behind its first failure;
3. stops at the first task-client failure, or after five minutes of wall clock.  Items
   it does not reach stay due: a rate, never a drop.

A failed attempt backs off 5 minutes, doubling to an hour.  Nothing caps how many owners
a burst files.  The drain never raises.

The drain's own ledger calls run off the event-loop thread (INV-8);
:func:`~orchestrator.flake_ledger.open_debt`'s internal sqlite stays on it, unchanged,
bounded by that function's connection count and by 50 calls per pass.

Import discipline: ``flake_ledger`` and ``shared`` only, and ``escalation.models``
lazily.  Never a lane module or ``git_ops``: the ancestry read arrives through
:class:`AncestryPort`.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, Protocol

from escalation.categories import EscalationCategory
from shared.task_statuses import OwnerLiveness, owner_liveness
from shared.task_submit import SubmitFailed, SubmitOutcome

from orchestrator.flake_ledger import (
    DUE_ITEM_STALE_AFTER,
    DUE_ITEM_STALE_MINUTES,
    NEVER_WIDEN_A_TIMEOUT,
    DebtRow,
    DueRow,
    DueSummary,
    FlakeLedgerTaskClient,
    delete_due_item,
    fix_contained,
    name_stale_due_items,
    open_debt,
    read_debt,
    read_due_items,
    record_failed_due_attempt,
    resolving_commit,
    summarize_due,
)

logger = logging.getLogger(__name__)

DRAIN_INTERVAL_SECS: Final = 30.0
STALE_DUE_ESCALATION_SENTINEL: Final = '__flake_debt_due_stale__'
STALE_DUE_ESCALATION_ROLE: Final = 'orchestrator-flake-debt-drain'

_ITEMS_PER_PASS = 50
_PASS_BUDGET = timedelta(minutes=5)
_FIRST_RETRY_DELAY = timedelta(minutes=5)
_MAX_RETRY_DELAY = timedelta(hours=1)
# Backstops above the adapter's own bounds (30s submit and commit, 15s dispatch).
_TASK_CLIENT_CALL_TIMEOUT_SECS = 60.0
_ANCESTRY_TIMEOUT_SECS = 30.0


class AncestryPort(Protocol):
    """The one git read the stale-head rule needs: whether *ancestor* is an ancestor of
    *descendant* — ``True``, ``False``, or ``None`` when git could not say."""

    async def is_ancestor_checked(self, ancestor: str, descendant: str) -> bool | None: ...


class DueFailure(StrEnum):
    """Why an attempt left its item due: the ``flake_debt_due.last_error`` vocabulary."""

    TASK_CLIENT_FAILED = 'task_client_failed'
    ANCESTRY_UNREADABLE = 'ancestry_unreadable'
    LEDGER_UNAVAILABLE = 'ledger_unavailable'
    NO_OWNER = 'no_owner'
    OWNER_DEFERRED = 'owner_deferred'
    OWNER_DONE = 'owner_done'
    OWNER_CLOSED = 'owner_closed'
    DRAIN_ERROR = 'drain_error'


class DrainStop(StrEnum):
    """Why a pass ended before the items it read."""

    TASK_CLIENT_FAILED = 'task_client_failed'
    BUDGET = 'budget'


@dataclass(frozen=True)
class DrainPassReport:
    attempted: int
    settled: int
    failed: int
    stop: DrainStop | None


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _retry_delay(attempt: int) -> timedelta:
    """The wait after the *attempt*-th failure: 5, 10, 20, 40, then 60 minutes."""
    return min(_FIRST_RETRY_DELAY * 2 ** (attempt - 1), _MAX_RETRY_DELAY)


class _WatchedTaskClient:
    """*inner*, with every failure it reports recorded in :attr:`failed`.

    The production adapter never raises: it answers failures in band.  A raise, a
    timeout, a ``SubmitFailed`` or an error half all count as "the task client errored",
    because telling a transport error from a semantic reject would mean parsing the
    ``SubmitFailed`` text.  Each method still hands :func:`open_debt` its own in-band
    failure shape, so the ledger's degrade paths run unchanged.  A cancellation — the
    ledger's own owner-read budget expiring, say — is recorded and re-raised.
    """

    def __init__(self, inner: FlakeLedgerTaskClient) -> None:
        self._inner = inner
        self.failed = False

    async def submit_task(self, arguments: dict) -> SubmitOutcome:
        try:
            outcome = await asyncio.wait_for(
                self._inner.submit_task(arguments), _TASK_CLIENT_CALL_TIMEOUT_SECS,
            )
        except asyncio.CancelledError:
            self.failed = True
            raise
        except Exception as exc:
            self.failed = True
            return SubmitFailed(f'submit_task raised {exc!r}')
        if isinstance(outcome, SubmitFailed):
            self.failed = True
        return outcome

    async def get_statuses(self, ids: list[str]) -> tuple[dict[str, str], Exception | None]:
        try:
            statuses, error = await asyncio.wait_for(
                self._inner.get_statuses(ids), _TASK_CLIENT_CALL_TIMEOUT_SECS,
            )
        except asyncio.CancelledError:
            self.failed = True
            raise
        except Exception as exc:
            self.failed = True
            return {}, exc
        if error is not None:
            self.failed = True
        return statuses, error

    async def get_task(self, task_id: str) -> tuple[dict | None, Exception | None]:
        try:
            task, error = await asyncio.wait_for(
                self._inner.get_task(task_id), _TASK_CLIENT_CALL_TIMEOUT_SECS,
            )
        except asyncio.CancelledError:
            self.failed = True
            raise
        except Exception as exc:
            self.failed = True
            return None, exc
        if error is not None:
            self.failed = True
        return task, error

    async def commit_planning(self, task_ids: list[str]) -> None:
        try:
            await asyncio.wait_for(
                self._inner.commit_planning(task_ids), _TASK_CLIENT_CALL_TIMEOUT_SECS,
            )
        except asyncio.CancelledError:
            self.failed = True
            raise
        except Exception:
            self.failed = True


class _HeadCheck:
    """One item's stale-head check: δ1's fix containment
    (``flake_ledger.py::fix_contained``) composed with δ2's ancestry read against the
    item's *head*.  :attr:`undecided` records that some read could not answer."""

    def __init__(self, git: AncestryPort, head: str) -> None:
        self._git = git
        self._head = head
        self.undecided = False

    async def ancestry(self, commit: str) -> bool | None:
        try:
            answer = await asyncio.wait_for(
                self._git.is_ancestor_checked(commit, self._head), _ANCESTRY_TIMEOUT_SECS,
            )
        except Exception:
            logger.warning(
                'flake_debt_drain: ancestry read %s..%s failed', commit, self._head,
                exc_info=True,
            )
            answer = None
        if answer is None:
            self.undecided = True
        return answer

    async def for_task(self, task: Mapping[str, Any]) -> bool | None:
        commit = resolving_commit(task)
        return fix_contained(task, None if commit is None else await self.ancestry(commit))


# ---------------------------------------------------------------------------
# The regression escalation (task η), filed when open_debt reports a re-entry
# ---------------------------------------------------------------------------

#: The escalation category that IS the ``regressed_after_resolution`` flag (task η): a
#: test re-entered flake debt after its de-flake task finished, so the fix did not hold.
REGRESSED_AFTER_RESOLUTION = EscalationCategory.REGRESSED_AFTER_RESOLUTION

#: ``make_id`` namespace for those escalations.  It names a counter FILE
#: (``esc-{key}.seq``), so it is fixed and filename-safe, never the raw pytest node-id;
#: each regression's identity lives in :func:`_regression_sentinel` instead.
_REGRESSION_ESCALATION_KEY = 'flake-regressed-after-resolution'

#: The ``orchestrator-`` prefix marks a harness sentinel, which the escalation server
#: never downgrades from ``critical``.
_REGRESSION_AGENT_ROLE = 'orchestrator-flake-ledger'


def _regression_sentinel(row: DebtRow) -> str:
    """The dedup task_id for *row*'s regression: one per (test, cycle).

    Not a single fixed sentinel, because each regression is a DISTINCT failed fix citing
    a distinct commit: one pending L2 must not swallow another test's regression, or a
    later cycle's.  It lives only in the stored ``task_id`` field, which ``get_by_task``
    filters on.
    """
    return f'{_REGRESSION_ESCALATION_KEY}:{row.test_id}:cycle-{row.open_count}'


def _cited_commit(row: DebtRow) -> str:
    """*row*'s prior resolving commit, VERBATIM, or an explicit statement that none was
    recorded — never the literal ``None``."""
    return row.prior_resolving_commit or '(none recorded)'


def _regression_detail(row: DebtRow, *, item: DueRow) -> str:
    """The escalation detail for *row*'s regression, observed through *item*: every fact a
    human needs to judge whether the prior fix was cosmetic (§5.5), with the fix's
    commit quoted VERBATIM or stated as not recorded."""
    return (
        f'{row.test_id} flaked again after its de-flake fix landed. The flake ledger '
        f'closed its debt cycle when the de-flake task was observed done, and a '
        f'merge-gate suppression has now RE-OPENED it: this is cycle {row.open_count} of '
        f'the same debt, and the fix below did not hold.\n'
        f'\n'
        f'  test:                    {row.test_id}\n'
        f'  project:                 {row.project_id}\n'
        f'  prior resolving commit:  {_cited_commit(row)}\n'
        f'  prior resolved_at:       {row.prior_resolved_at}  (when the ledger OBSERVED '
        f'the task done, not when the fix merged)\n'
        f'  opened_at:               {row.opened_at}\n'
        f'  open_count:              {row.open_count}\n'
        f'  observed at merge_sha:   {item.head or "(none)"}\n'
        f'  observing task_id:       {item.observing_task_id or "(none)"}\n'
        f'  observed_at:             {item.observed_at}\n'
        f'\n'
        f"The ledger files this cycle's de-flake task itself, straight after this "
        f'escalation; `orchestrator flake-ledger` shows its owner, `filing pending` while '
        f'the drain still owes the filing, or the unowned breach if no filing is due.\n'
        f'\n'
        f'{NEVER_WIDEN_A_TIMEOUT}'
    )


def _escalate_regressed_after_resolution(
    escalation_queue: Any, row: DebtRow, *, item: DueRow,
) -> None:
    """File the born-at-L2 ``regressed_after_resolution`` escalation for *row*, a test
    that re-entered flake debt after its de-flake task finished (task η).

    ``severity='critical'``, ``level=2`` and a harness-sentinel ``agent_role``, deduped
    on an open L2 through ``get_by_task``'s root-only ``'pending'`` fast path, never the
    archive-wide scan.  The dedup is belt-and-braces — ``open_debt`` reports each
    re-entry once — and it is per (test, cycle): see :func:`_regression_sentinel`.

    None-safe but LOUD: with no queue the regression is logged as a WARNING.  A failing
    submit raises, and ``open_debt``'s hook guard reports that loss.
    """
    if escalation_queue is None:
        logger.warning(
            'flake_debt_drain: %s — test_id=%s re-entered flake debt (cycle %d) after its '
            'fix landed (prior resolving commit %s), but no escalation queue is wired, '
            'so it is visible only in `orchestrator flake-ledger` (merge_sha=%s, '
            'task_id=%s)',
            REGRESSED_AFTER_RESOLUTION,
            row.test_id,
            row.open_count,
            _cited_commit(row),
            item.head,
            item.observing_task_id,
        )
        return

    from escalation.models import Escalation  # noqa: PLC0415 — local, escalation optional dep

    sentinel = _regression_sentinel(row)
    if escalation_queue.get_by_task(sentinel, status='pending', level=2):
        logger.info(
            'flake_debt_drain: an open %s escalation already covers test_id=%s cycle %d',
            REGRESSED_AFTER_RESOLUTION,
            row.test_id,
            row.open_count,
        )
        return

    commit = row.prior_resolving_commit
    if commit:
        inspect_the_fix = f'Inspect the fix that did not hold (git show {commit})'
    else:
        inspect_the_fix = (
            'No resolving commit was recorded for the fix that did not hold; find the '
            "prior de-flake task's change"
        )
    escalation_queue.submit(
        Escalation(
            id=escalation_queue.make_id(_REGRESSION_ESCALATION_KEY),
            task_id=sentinel,
            agent_role=_REGRESSION_AGENT_ROLE,
            severity='critical',
            level=2,
            category=REGRESSED_AFTER_RESOLUTION,
            summary=(
                f'Flake regressed after its fix: {row.test_id} re-entered flake debt '
                f'(cycle {row.open_count})'
            ),
            detail=_regression_detail(row, item=item),
            suggested_action=(
                f'{inspect_the_fix} and judge whether it was cosmetic — a widened '
                'timeout, a lengthened sleep, an added retry — before the new '
                "cycle's de-flake task repeats it."
            ),
            evidence=[
                {
                    'observation': (
                        f'{row.test_id} re-entered flake debt: open_count '
                        f'{row.open_count}, prior cycle resolved at {row.prior_resolved_at}'
                    ),
                    'measured_at': f'opened_at={row.opened_at}',
                    'ref': f'prior_resolving_commit={_cited_commit(row)}',
                },
            ],
        )
    )


# ---------------------------------------------------------------------------
# The stale-item escalation (INV-4 storm escape, INV-7 age surfacing)
# ---------------------------------------------------------------------------


def _last_error(item: DueRow) -> str:
    return item.last_error or 'not yet attempted'


def _stale_summary(summary: DueSummary) -> str:
    """The record's one-line framing.  No moving figure (age, attempts): ``amend``
    appends only when the framing changes, so it changes at the rate items do."""
    oldest = summary.stale_sample[0]
    return (
        f'{summary.stale} flake-debt item(s) waiting over {DUE_ITEM_STALE_MINUTES} min for a '
        f'de-flake filing; oldest {oldest.test_id} since {oldest.enqueued_at}: '
        f'{_last_error(oldest)}'
    )


def _stale_detail(summary: DueSummary) -> str:
    lines = [
        f'{summary.stale} item(s) on the flake ledger\'s due list (flake_debt_due) have '
        f'waited over {DUE_ITEM_STALE_MINUTES} minutes for their de-flake owner to be filed. '
        f'The oldest {len(summary.stale_sample)}, oldest first:',
        '',
    ]
    lines += [
        f'  {item.test_id}  enqueued {item.enqueued_at}  call_site={item.call_site}  '
        f'last_error={_last_error(item)}'
        for item in summary.stale_sample
    ]
    lines += [
        '',
        'last_error task_client_failed means the drain could not reach fused-memory; '
        'ancestry_unreadable means it could not tell whether the merge sha contains a '
        'finished de-flake task\'s fix; owner_deferred means a filed task has not left '
        'deferred. The drain retries every item on its own and resolves this record '
        'once no item is stale; nothing needs re-filing for task_client_failed. A '
        'persistent no_owner means the task store refused the filing, and the operator '
        'must act.',
    ]
    return '\n'.join(lines)


async def _surface_stale_items(
    db_path: Path, escalation_queue: Any, *, now: datetime, filing_claimant_run_id: str | None,
) -> None:
    """Open, amend or resolve the ONE stale-item escalation to match the due list.

    An item that a record has named keeps that record's id, so a record someone else
    resolved is never re-opened for the items it named; only newly stale items open a
    new one.  An item stops being stale only by leaving the list.
    """
    try:
        stale_before = now - DUE_ITEM_STALE_AFTER
        summary = await asyncio.to_thread(summarize_due, db_path, stale_before=stale_before)
        if summary is None:
            logger.warning('flake_debt_drain: could not read the due list to surface stale items')
            return
        if escalation_queue is None:
            if summary.stale:
                logger.warning(
                    'flake_debt_drain: %d due item(s) are stale but no escalation queue '
                    'is wired',
                    summary.stale,
                )
            return
        pending = escalation_queue.get_by_task(
            STALE_DUE_ESCALATION_SENTINEL, status='pending', agent_role=STALE_DUE_ESCALATION_ROLE,
        )
        if summary.stale == 0:
            for esc in pending:
                escalation_queue.resolve(
                    esc.id,
                    f'auto-resolved: no flake-debt item has waited over {DUE_ITEM_STALE_MINUTES} minutes',
                    resolved_by=STALE_DUE_ESCALATION_ROLE,
                )
            return
        if not pending and summary.unnamed_stale == 0:
            return
        title = _stale_summary(summary)[:200]
        detail = _stale_detail(summary)
        if pending:
            record_id = pending[0].id
        else:
            from escalation.models import Escalation

            esc = Escalation(
                id=escalation_queue.make_id(STALE_DUE_ESCALATION_SENTINEL),
                task_id=STALE_DUE_ESCALATION_SENTINEL,
                agent_role=STALE_DUE_ESCALATION_ROLE,
                severity='blocking',
                category='infra_issue',
                summary=title,
                detail=detail,
                suggested_action=(
                    'Restore fused-memory if last_error reads task_client_failed (the usual '
                    'cause); the drain then catches up and resolves this record itself. A '
                    'persistent no_owner means the task store refused the filing: escalate to '
                    "the operator. See OPERATIONS.md 'Reading the flake ledger'."
                ),
                level=1,
                filing_claimant_run_id=filing_claimant_run_id,
            )
            escalation_queue.submit(esc)
            record_id = esc.id
        await asyncio.to_thread(
            name_stale_due_items, db_path, record_id, stale_before=stale_before,
        )
        if pending:
            escalation_queue.amend(
                record_id, summary=title, detail=detail, agent_role=STALE_DUE_ESCALATION_ROLE,
            )
    except Exception:
        logger.warning('flake_debt_drain: the stale-item escalation failed', exc_info=True)


# ---------------------------------------------------------------------------
# The pass
# ---------------------------------------------------------------------------


async def _delete(db_path: Path, item: DueRow) -> DueFailure | None:
    deleted = await asyncio.to_thread(delete_due_item, db_path, item.id)
    return None if deleted else DueFailure.LEDGER_UNAVAILABLE


_FAILURE_BY_LIVENESS: Final = {
    OwnerLiveness.DEFERRED: DueFailure.OWNER_DEFERRED,
    OwnerLiveness.DONE: DueFailure.OWNER_DONE,
    OwnerLiveness.CLOSED: DueFailure.OWNER_CLOSED,
}


async def _settle(
    db_path: Path,
    project_id: str,
    item: DueRow,
    *,
    task_client: FlakeLedgerTaskClient,
    git: AncestryPort | None,
    escalation_queue: Any,
) -> DueFailure | None:
    """One attempt at *item*: ``None`` once its debt is settled and the item deleted,
    otherwise why it stays due."""
    check = None if git is None or item.head is None else _HeadCheck(git, item.head)
    if check is not None:
        # A CLOSED cycle's fix commit is on its row, so the stale-head rule for it needs
        # no task read, and must run here: open_debt's lazy close skips a resolved row,
        # and its upsert would re-enter the cycle.
        closed = await asyncio.to_thread(read_debt, db_path, item.test_id)
        if closed is not None and closed.resolved_at and closed.prior_resolving_commit:
            contained = await check.ancestry(closed.prior_resolving_commit)
            if contained is None:
                return DueFailure.ANCESTRY_UNREADABLE
            if not contained:
                logger.info(
                    'flake_debt_drain: evidence at %s predates the fix %s that closed cycle '
                    '%d of test_id=%s; dropping the item',
                    item.head,
                    closed.prior_resolving_commit,
                    closed.open_count,
                    item.test_id,
                )
                return await _delete(db_path, item)

    client = _WatchedTaskClient(task_client)
    row = await open_debt(
        db_path,
        project_id,
        item.test_id,
        task_client=client,
        on_regressed_after_resolution=functools.partial(
            _escalate_regressed_after_resolution, escalation_queue, item=item,
        ),
        head_contains_fix=None if check is None else check.for_task,
    )
    if client.failed:
        return DueFailure.TASK_CLIENT_FAILED
    if check is not None and check.undecided:
        return DueFailure.ANCESTRY_UNREADABLE
    if row is None:
        return DueFailure.LEDGER_UNAVAILABLE
    if row.resolved_at is not None:
        return await _delete(db_path, item)
    owner = row.owner_task_id
    if owner is None:
        return DueFailure.NO_OWNER
    # One live read: commit_planning never raises, so a pointer alone does not prove the
    # filing completed, and a deferred owner is never dispatched.
    statuses, error = await client.get_statuses([owner])
    if error is not None or client.failed:
        return DueFailure.TASK_CLIENT_FAILED
    failure = _FAILURE_BY_LIVENESS.get(owner_liveness(statuses.get(owner)))
    return failure if failure is not None else await _delete(db_path, item)


async def _attempt(
    db_path: Path,
    project_id: str,
    item: DueRow,
    *,
    task_client: FlakeLedgerTaskClient,
    git: AncestryPort | None,
    escalation_queue: Any,
    clock: Callable[[], datetime],
) -> DueFailure | None:
    """:func:`_settle` under a guard that turns a raise into one failed attempt, which it
    records with the item's backoff."""
    try:
        failure = await _settle(
            db_path, project_id, item,
            task_client=task_client, git=git, escalation_queue=escalation_queue,
        )
    except Exception:
        logger.warning(
            'flake_debt_drain: settling due item %s (test_id=%s) raised',
            item.id,
            item.test_id,
            exc_info=True,
        )
        failure = DueFailure.DRAIN_ERROR
    if failure is not None:
        await asyncio.to_thread(
            record_failed_due_attempt,
            db_path,
            item.id,
            reason=failure.value,
            next_attempt_at=clock() + _retry_delay(item.attempts + 1),
        )
    return failure


async def run_drain_pass(
    db_path: Path,
    project_id: str,
    *,
    task_client: FlakeLedgerTaskClient,
    git: AncestryPort | None,
    escalation_queue: Any,
    filing_claimant_run_id: str | None = None,
    clock: Callable[[], datetime] = _utc_now,
) -> DrainPassReport:
    """One pass of the drain over the ledger at *db_path* (see the module docstring).

    *git* ``None`` runs no stale-head check.  *clock* drives the stale bound, the due
    read, the backoff and the pass budget; :func:`open_debt` stamps with its own wall
    clock, so a retry can never read as a re-entry.  Never raises; a cancellation
    propagates.
    """
    attempted = settled = failed = 0
    stop: DrainStop | None = None
    try:
        started = clock()
        await _surface_stale_items(
            db_path, escalation_queue, now=started, filing_claimant_run_id=filing_claimant_run_id,
        )
        items = await asyncio.to_thread(
            read_due_items, db_path, now=clock(), limit=_ITEMS_PER_PASS,
        )
        if items is None:
            logger.warning('flake_debt_drain: could not read the due list; pass skipped')
            return DrainPassReport(0, 0, 0, None)
        waiting: set[str] = set()
        for item in items:
            if item.test_id in waiting:
                continue
            if clock() - started >= _PASS_BUDGET:
                stop = DrainStop.BUDGET
                break
            attempted += 1
            failure = await _attempt(
                db_path, project_id, item,
                task_client=task_client, git=git, escalation_queue=escalation_queue,
                clock=clock,
            )
            if failure is None:
                settled += 1
                continue
            failed += 1
            waiting.add(item.test_id)
            if failure is DueFailure.TASK_CLIENT_FAILED:
                stop = DrainStop.TASK_CLIENT_FAILED
                break
    except Exception:
        logger.warning('flake_debt_drain: the pass failed', exc_info=True)
    return DrainPassReport(attempted, settled, failed, stop)
