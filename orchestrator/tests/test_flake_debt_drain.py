"""Tests for the flake-debt drain (task 6934, ε0; plans/offline-lane-red-path-prd.md D6).

The drain turns each due item an observation left on the ledger's due list into one
``open_debt`` call, then settles or backs off the item.  Every assertion reads a public
surface: the ledger's readers, the pass report, a real ``EscalationQueue`` and the fakes'
own records.  The ledger is never faked.

This file imports no lane module and no ``git_ops`` — the ancestry port is faked here —
so the merge-lane ratchet does not sweep it.  Every async class carries
``@pytest.mark.asyncio`` (strict mode; see test_flake_ledger.py's module docstring).
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from escalation.queue import EscalationQueue
from shared.task_submit import SubmitFailed, SubmitOutcome, SubmittedTask

from orchestrator import flake_debt_drain
from orchestrator.flake_debt_drain import (
    REGRESSED_AFTER_RESOLUTION,
    STALE_DUE_ESCALATION_ROLE,
    STALE_DUE_ESCALATION_SENTINEL,
    AncestryPort,
    DrainPassReport,
    DrainStop,
    DueFailure,
    run_drain_pass,
)
from orchestrator.flake_ledger import (
    NEVER_WIDEN_A_TIMEOUT,
    DueItem,
    DueRow,
    FlakeLedgerTaskClient,
    enqueue_debt,
    open_debt,
    read_debt,
    read_due_items,
    resolve_debt,
)

T0 = datetime(2026, 10, 18, 12, 0, tzinfo=UTC)
_FAR = T0 + timedelta(days=3650)
PROJECT = 'dark_factory'
TASK_ID = '7001'
A = 'orchestrator/tests/test_x.py::test_a'
B = 'orchestrator/tests/test_x.py::test_b'
C = 'orchestrator/tests/test_x.py::test_c'
FIX = 'f1' * 20
STALE_HEAD = 'a' * 40
GOOD_HEAD = 'b' * 40
UNREAD_HEAD = 'c' * 40


class _FakeLedgerTaskClient:
    """The ``flake_ledger.FlakeLedgerTaskClient`` seam, recorded, minting a DISTINCT id
    per filing (``deflake-N``), so "each test got its own owner" is testable.

    An id handed out by ``submit_task`` is registered as *status_after_submit*;
    *statuses* pre-seeds or overrides that per id.  ``get_task`` reads the SAME status
    map, and a ``done`` task carries ``metadata.done_provenance`` naming its entry in
    *commits*.  An unknown id is a corroborated absence, ``(None, None)``.
    :meth:`finish` is the orchestrator completing a task: the live state changes and the
    ledger is never told.

    ``get_statuses`` returns the ``(statuses, error)`` PAIR the Protocol declares — a
    bare dict would make every corroboration raise into the ledger's degrade branch and
    the live-owner dedup branch would never run.

    *submit_raises* (toggleable) makes ``submit_task`` raise it.  *planning_commits*
    (toggleable): ``commit_planning`` releases a ``deferred`` task to
    ``pending``; ``False`` records the call and leaves it ``deferred``, the silent
    failure.  *unreachable* (toggleable): fused-memory is down — ``submit_task``
    answers ``SubmitFailed``, the reads answer an error half, ``commit_planning`` does
    nothing.  *submit_hangs* / *commit_hangs* (toggleable) never answer, setting
    :attr:`submit_entered` / :attr:`commit_entered` first.  *on_call* runs at the start
    of every method (a test's clock advance).  *folds_titles* (toggleable) is the task
    store's candidate key: a submission whose title matches an existing task that is not
    cancelled is answered as COMBINED into that task, a ``done`` one included.
    *get_task_hangs* (toggleable) makes ``get_task`` never answer.
    """

    def __init__(
        self,
        *,
        submit_raises: BaseException | None = None,
        status_after_submit: str = 'pending',
        statuses: dict[str, str] | None = None,
        commits: dict[str, str] | None = None,
        statuses_error: Exception | None = None,
        statuses_raises: BaseException | None = None,
        order: list[str] | None = None,
        planning_commits: bool = True,
        unreachable: bool = False,
        submit_hangs: bool = False,
        commit_hangs: bool = False,
        on_call: Callable[[], None] | None = None,
        folds_titles: bool = False,
        get_task_hangs: bool = False,
    ) -> None:
        self.folds_titles = folds_titles
        self.get_task_hangs = get_task_hangs
        self._titles: dict[str, str] = {}
        self.submit_calls: list[dict] = []
        self.statuses_calls: list[list[str]] = []
        self.task_calls: list[str] = []
        self.commit_calls: list[list[str]] = []
        self.filed: list[str] = []
        self.order: list[str] = order if order is not None else []
        self.planning_commits = planning_commits
        self.unreachable = unreachable
        self.submit_hangs = submit_hangs
        self.commit_hangs = commit_hangs
        self.submit_entered = asyncio.Event()
        self.commit_entered = asyncio.Event()
        self._on_call = on_call
        self.submit_raises = submit_raises
        self._status_after_submit = status_after_submit
        self._statuses: dict[str, str] = dict(statuses or {})
        self._commits: dict[str, str] = dict(commits or {})
        self._statuses_error = statuses_error
        self._statuses_raises = statuses_raises

    def _enter(self, name: str) -> None:
        self.order.append(name)
        if self._on_call is not None:
            self._on_call()

    async def submit_task(self, arguments: dict) -> SubmitOutcome:
        self._enter('submit_task')
        self.submit_calls.append(arguments)
        if self.submit_hangs:
            self.submit_entered.set()
            await asyncio.Event().wait()
        if self.submit_raises is not None:
            raise self.submit_raises
        if self.unreachable:
            return SubmitFailed('unreachable')
        if self.folds_titles:
            for task_id, title in self._titles.items():
                if title == arguments['title'] and self._statuses.get(task_id) != 'cancelled':
                    return SubmittedTask(task_id, self._statuses.get(task_id), combined=True)
        new_id = f'deflake-{len(self.filed) + 1}'
        self._titles[new_id] = arguments['title']
        self.filed.append(new_id)
        self._statuses.setdefault(new_id, self._status_after_submit)
        return SubmittedTask(new_id, self._status_after_submit)

    async def get_statuses(
        self, ids: list[str],
    ) -> tuple[dict[str, str], Exception | None]:
        self._enter('get_statuses')
        self.statuses_calls.append(list(ids))
        if self._statuses_raises is not None:
            raise self._statuses_raises
        if self.unreachable:
            return {}, ConnectionError('unreachable')
        return (
            {i: self._statuses[i] for i in ids if i in self._statuses},
            self._statuses_error,
        )

    async def get_task(self, task_id: str) -> tuple[dict | None, Exception | None]:
        self._enter('get_task')
        self.task_calls.append(task_id)
        if self.get_task_hangs:
            await asyncio.Event().wait()
        if self.unreachable:
            return None, ConnectionError('unreachable')
        if task_id not in self._statuses:
            return None, None
        status = self._statuses[task_id]
        metadata: dict = {}
        if status == 'done' and task_id in self._commits:
            metadata = {
                'done_provenance': {'kind': 'merged', 'commit': self._commits[task_id]},
            }
        return {'id': task_id, 'status': status, 'metadata': metadata}, None

    async def commit_planning(self, task_ids: list[str]) -> None:
        self._enter('commit_planning')
        self.commit_calls.append(list(task_ids))
        if self.commit_hangs:
            self.commit_entered.set()
            await asyncio.Event().wait()
        if self.unreachable or not self.planning_commits:
            return
        for task_id in task_ids:
            if self._statuses.get(task_id) == 'deferred':
                self._statuses[task_id] = 'pending'

    def finish(self, task_id: str, *, commit: str | None = None) -> None:
        """*task_id* goes ``done``, its fix merged at *commit* when one is recorded."""
        self._statuses[task_id] = 'done'
        if commit is not None:
            self._commits[task_id] = commit

    def status(self, task_id: str) -> str | None:
        return self._statuses.get(task_id)


class _RaisingTaskClient:
    """A client whose every method raises: a partial or broken adapter."""

    async def submit_task(self, arguments: dict) -> SubmitOutcome:
        raise RuntimeError('submit_task exploded')

    async def get_statuses(self, ids: list[str]) -> tuple[dict[str, str], Exception | None]:
        raise RuntimeError('get_statuses exploded')

    async def get_task(self, task_id: str) -> tuple[dict | None, Exception | None]:
        raise RuntimeError('get_task exploded')

    async def commit_planning(self, task_ids: list[str]) -> None:
        raise RuntimeError('commit_planning exploded')


class _FakeAncestry:
    """The drain's ancestry port: ``{(commit, head): True | False | None}``, *default* for
    any other pair, every call recorded."""

    def __init__(
        self, answers: dict[tuple[str, str], bool | None] | None = None, *, default: bool = True,
    ) -> None:
        self.answers = dict(answers or {})
        self.default = default
        self.calls: list[tuple[str, str]] = []

    async def is_ancestor_checked(self, ancestor: str, descendant: str) -> bool | None:
        self.calls.append((ancestor, descendant))
        return self.answers.get((ancestor, descendant), self.default)


class _Clock:
    def __init__(self, now: datetime = T0) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, delta: timedelta) -> None:
        self.now += delta


class _RecordingQueue(EscalationQueue):
    """The real queue, with every submitted record also kept in :attr:`submitted`."""

    def __init__(self, queue_dir: Path) -> None:
        super().__init__(queue_dir)
        self.submitted: list = []

    def submit(self, escalation) -> str:
        self.submitted.append(escalation)
        return super().submit(escalation)

    def regressions(self) -> list:
        return [e for e in self.submitted if e.category == REGRESSED_AFTER_RESOLUTION]

    def stale_records(self) -> list:
        return self.get_by_task(STALE_DUE_ESCALATION_SENTINEL, status='pending')


def _db(tmp_path: Path) -> Path:
    return tmp_path / 'runs.db'


def _observed(n: int) -> str:
    return (T0 - timedelta(hours=1) + timedelta(seconds=n)).isoformat()


def _enqueue(
    db: Path, test_id: str, n: int, *, at: datetime, head: str | None = GOOD_HEAD,
) -> None:
    enqueue_debt(
        db,
        PROJECT,
        [DueItem(test_id, _observed(n), 'merge_gate', head, TASK_ID)],
        now=at,
    )


def _items(db: Path) -> list[DueRow]:
    items = read_due_items(db, now=_FAR, limit=10_000)
    assert items is not None
    return items


def _item(db: Path, test_id: str) -> DueRow:
    (item,) = [i for i in _items(db) if i.test_id == test_id]
    return item


async def _pass(
    db: Path,
    client: FlakeLedgerTaskClient,
    clock: _Clock,
    *,
    git: AncestryPort | None = None,
    queue: EscalationQueue | None = None,
) -> DrainPassReport:
    return await run_drain_pass(
        db, PROJECT, task_client=client, git=git, escalation_queue=queue, clock=clock,
    )


async def _owned(db: Path, client: _FakeLedgerTaskClient, clock: _Clock, test_id: str, n: int) -> str:
    """Open *test_id*'s first cycle through the drain and return its owner."""
    _enqueue(db, test_id, n, at=clock.now)
    await _pass(db, client, clock)
    row = read_debt(db, test_id)
    assert row is not None and row.owner_task_id is not None, row
    return row.owner_task_id


@pytest.mark.asyncio
class TestFakeConformance:
    async def test_the_fakes_satisfy_their_protocols_statically(self) -> None:
        client: FlakeLedgerTaskClient = _FakeLedgerTaskClient()
        git: AncestryPort = _FakeAncestry()
        assert await client.get_statuses(['x']) == ({}, None)
        assert await git.is_ancestor_checked('a', 'b') is True


@pytest.mark.asyncio
class TestDrainFilesTheOwner:
    async def test_an_enqueued_item_gets_an_owner_and_leaves_the_list(self, tmp_path: Path) -> None:
        db, clock = _db(tmp_path), _Clock()
        client = _FakeLedgerTaskClient(status_after_submit='deferred')
        _enqueue(db, A, 0, at=clock.now)

        report = await _pass(db, client, clock)

        assert report == DrainPassReport(attempted=1, settled=1, failed=0, stop=None)
        row = read_debt(db, A)
        assert row is not None and row.owner_task_id == 'deflake-1'
        assert client.status('deflake-1') == 'pending'
        assert _items(db) == []

    async def test_a_deferred_owner_stays_due_and_the_retry_completes_it(
        self, tmp_path: Path,
    ) -> None:
        db, clock = _db(tmp_path), _Clock()
        client = _FakeLedgerTaskClient(status_after_submit='deferred', planning_commits=False)
        _enqueue(db, A, 0, at=clock.now)

        await _pass(db, client, clock)

        item = _item(db, A)
        assert (item.attempts, item.last_error) == (1, DueFailure.OWNER_DEFERRED.value)
        client.planning_commits = True
        clock.advance(timedelta(minutes=6))
        await _pass(db, client, clock)

        assert _items(db) == []
        assert client.filed == ['deflake-1'], 'the retry completes the owner, files no second'
        assert client.status('deflake-1') == 'pending'

    async def test_a_failed_filing_stays_due_until_the_backoff_and_files_once(
        self, tmp_path: Path,
    ) -> None:
        db, clock = _db(tmp_path), _Clock()
        client = _FakeLedgerTaskClient(unreachable=True)
        _enqueue(db, A, 0, at=clock.now)

        assert (await _pass(db, client, clock)).failed == 1
        client.unreachable = False
        clock.advance(timedelta(minutes=1))
        assert (await _pass(db, client, clock)).attempted == 0, 'not retried before the backoff'
        clock.advance(timedelta(minutes=5))
        assert (await _pass(db, client, clock)).settled == 1

        assert client.filed == ['deflake-1']
        row = read_debt(db, A)
        assert row is not None and row.owner_task_id == 'deflake-1'
        assert _items(db) == []

    async def test_a_drain_cut_off_after_the_owner_pointer_files_no_second_task(
        self, tmp_path: Path,
    ) -> None:
        db, clock = _db(tmp_path), _Clock()
        client = _FakeLedgerTaskClient(status_after_submit='deferred', commit_hangs=True)
        _enqueue(db, A, 0, at=clock.now)

        drain = asyncio.create_task(_pass(db, client, clock))
        await client.commit_entered.wait()
        drain.cancel()
        with pytest.raises(asyncio.CancelledError):
            await drain

        row = read_debt(db, A)
        assert row is not None and row.owner_task_id == 'deflake-1', 'the pointer is written'
        assert [i.attempts for i in _items(db)] == [0], 'the item is still listed'
        assert len(client.submit_calls) == 1

        client.commit_hangs = False
        await _pass(db, client, clock)

        assert len(client.submit_calls) == 1, 'the retry completes the owner, files no second'
        assert client.status('deflake-1') == 'pending'
        assert _items(db) == []

    async def test_more_than_a_pass_of_items_take_two_passes_and_lose_none(
        self, tmp_path: Path,
    ) -> None:
        db, clock = _db(tmp_path), _Clock()
        client = _FakeLedgerTaskClient()
        tests = [f'orchestrator/tests/test_many.py::test_{i:02d}' for i in range(60)]
        for i, test_id in enumerate(tests):
            _enqueue(db, test_id, i, at=clock.now)

        first = await _pass(db, client, clock)
        assert (first.attempted, first.settled) == (50, 50)
        assert len(_items(db)) == 10
        second = await _pass(db, client, clock)
        assert (second.attempted, second.settled) == (10, 10)

        assert _items(db) == []
        owners = {test_id: getattr(read_debt(db, test_id), 'owner_task_id', None) for test_id in tests}
        assert all(owners.values()), owners
        assert len(set(owners.values())) == 60

    async def test_an_item_with_no_head_runs_no_stale_head_check(self, tmp_path: Path) -> None:
        db, clock = _db(tmp_path), _Clock()
        client, git = _FakeLedgerTaskClient(), _FakeAncestry(default=False)
        queue = _RecordingQueue(tmp_path / 'queue')
        owner = await _owned(db, client, clock, A, 0)
        client.finish(owner, commit=FIX)
        _enqueue(db, A, 1, at=clock.now, head=None)

        await _pass(db, client, clock, git=git, queue=queue)

        assert git.calls == []
        row = read_debt(db, A)
        assert row is not None and row.open_count == 2, 'no head is no check: it re-enters'
        assert len(queue.regressions()) == 1


@pytest.mark.asyncio
class TestRegressionFromTheDrain:
    """Task η's ``regressed_after_resolution`` escalation, now raised from the drain:
    born at level 2, ``critical``, deduped per (test, cycle), the wording unchanged, the
    head and observing task taken from the due item."""

    async def _regress(
        self, db: Path, client: _FakeLedgerTaskClient, clock: _Clock, queue, test_id: str = A,
        *, n: int = 0, commit: str | None = FIX,
    ) -> None:
        """Cycle 1 for *test_id*, its owner finished with *commit*, then a re-entry."""
        owner = await _owned(db, client, clock, test_id, n)
        client.finish(owner, commit=commit)
        _enqueue(db, test_id, n + 1, at=clock.now)
        await _pass(db, client, clock, queue=queue)

    async def test_a_re_entry_raises_one_l2_citing_the_fix_and_the_item(
        self, tmp_path: Path,
    ) -> None:
        db, clock, client = _db(tmp_path), _Clock(), _FakeLedgerTaskClient()
        queue = _RecordingQueue(tmp_path / 'queue')

        await self._regress(db, client, clock, queue)

        row = read_debt(db, A)
        assert row is not None and row.open_count == 2 and row.owner_task_id == 'deflake-2'
        (esc,) = queue.regressions()
        assert (esc.level, esc.severity, esc.agent_role) == (
            2, 'critical', 'orchestrator-flake-ledger',
        )
        assert REGRESSED_AFTER_RESOLUTION == 'regressed_after_resolution'
        assert esc.summary == f'Flake regressed after its fix: {A} re-entered flake debt (cycle 2)'
        assert esc.detail.startswith(
            f'{A} flaked again after its de-flake fix landed. The flake ledger closed its debt '
            'cycle when the de-flake task was observed done, and a merge-gate suppression has '
            'now RE-OPENED it: this is cycle 2 of the same debt, and the fix below did not hold.'
        ), esc.detail
        for line in (
            f'  prior resolving commit:  {FIX}\n',
            f'  prior resolved_at:       {row.prior_resolved_at}  (when the ledger OBSERVED',
            f'  opened_at:               {row.opened_at}\n',
            '  open_count:              2\n',
            f'  observed at merge_sha:   {GOOD_HEAD}\n',
            f'  observing task_id:       {TASK_ID}\n',
            f'  observed_at:             {_observed(1)}\n',
            NEVER_WIDEN_A_TIMEOUT,
        ):
            assert line in esc.detail, (line, esc.detail)
        assert any(FIX in entry['ref'] for entry in esc.evidence), esc.evidence
        assert _items(db) == []

    async def test_a_failed_first_filing_and_a_hand_resolve_still_raise_one_escalation(
        self, tmp_path: Path,
    ) -> None:
        db, clock, client = _db(tmp_path), _Clock(), _FakeLedgerTaskClient()
        queue = _RecordingQueue(tmp_path / 'queue')
        owner = await _owned(db, client, clock, A, 0)
        client.finish(owner, commit=FIX)
        _enqueue(db, A, 1, at=clock.now)
        client.submit_raises = ConnectionError('fused-memory went away mid-filing')

        await _pass(db, client, clock, queue=queue)

        (esc,) = queue.regressions()
        assert _item(db, A).last_error == DueFailure.TASK_CLIENT_FAILED.value
        queue.resolve(esc.id, 'looked at it by hand', resolved_by='interactive')
        client.submit_raises = None
        clock.advance(timedelta(minutes=6))
        await _pass(db, client, clock, queue=queue)

        assert len(queue.regressions()) == 1, 'the retry is not a second re-entry'
        assert client.filed == [owner, 'deflake-2'], 'one task for the new cycle'
        row = read_debt(db, A)
        assert row is not None and row.open_count == 2 and row.owner_task_id == 'deflake-2'
        assert _items(db) == []

    async def test_each_test_and_cycle_dedups_on_its_own_sentinel(self, tmp_path: Path) -> None:
        clock = _Clock()
        queue = _RecordingQueue(tmp_path / 'queue')
        db, client = _db(tmp_path), _FakeLedgerTaskClient()
        await self._regress(db, client, clock, queue)
        replay_db = tmp_path / 'replay' / 'runs.db'
        await self._regress(replay_db, _FakeLedgerTaskClient(), clock, queue)
        client.finish('deflake-2', commit='e' * 40)
        _enqueue(db, A, 2, at=clock.now)
        await _pass(db, client, clock, queue=queue)
        await self._regress(db, client, clock, queue, B, n=10, commit='d' * 40)

        filed = [(e.task_id, e.summary, e.evidence[0]['ref']) for e in queue.regressions()]
        assert len(filed) == 3, filed
        for (task_id, summary, ref), (test_id, cycle, commit) in zip(
            filed, [(A, 2, FIX), (A, 3, 'e' * 40), (B, 2, 'd' * 40)], strict=True,
        ):
            assert test_id in task_id and f'cycle-{cycle}' in task_id, task_id
            assert f'(cycle {cycle})' in summary, summary
            assert commit in ref, ref

    async def test_a_commitless_resolution_is_named_as_such(self, tmp_path: Path) -> None:
        db, clock, client = _db(tmp_path), _Clock(), _FakeLedgerTaskClient()
        queue = _RecordingQueue(tmp_path / 'queue')

        await self._regress(db, client, clock, queue, commit=None)

        (esc,) = queue.regressions()
        assert 'none recorded' in esc.detail, esc.detail
        assert 'None' not in esc.detail, esc.detail
        assert all('None' not in entry['ref'] for entry in esc.evidence), esc.evidence

    async def test_no_queue_still_re_enters_and_is_loud(self, tmp_path: Path, caplog) -> None:
        db, clock, client = _db(tmp_path), _Clock(), _FakeLedgerTaskClient()

        with caplog.at_level(logging.WARNING, logger='orchestrator.flake_debt_drain'):
            await self._regress(db, client, clock, None)

        row = read_debt(db, A)
        assert row is not None and row.open_count == 2
        warned = [
            r.getMessage() for r in caplog.records
            if r.levelno == logging.WARNING and r.name == 'orchestrator.flake_debt_drain'
        ]
        assert len(warned) == 1, caplog.text
        assert A in warned[0] and REGRESSED_AFTER_RESOLUTION in warned[0]
        assert GOOD_HEAD in warned[0] and TASK_ID in warned[0]

    async def test_a_first_item_and_a_repeat_raise_no_regression(self, tmp_path: Path) -> None:
        db, clock, client = _db(tmp_path), _Clock(), _FakeLedgerTaskClient()
        queue = _RecordingQueue(tmp_path / 'queue')

        await _owned(db, client, clock, A, 0)
        _enqueue(db, A, 1, at=clock.now)
        await _pass(db, client, clock, queue=queue)

        row = read_debt(db, A)
        assert row is not None and row.open_count == 1
        assert queue.submitted == []
        assert client.filed == ['deflake-1']

    async def test_a_raising_queue_costs_only_the_regression_signal(
        self, tmp_path: Path, caplog,
    ) -> None:
        class _RaisingQueue(_RecordingQueue):
            def submit(self, escalation) -> str:
                raise RuntimeError('escalation queue unavailable')

        db, clock, client = _db(tmp_path), _Clock(), _FakeLedgerTaskClient()

        with caplog.at_level(logging.WARNING):
            await self._regress(db, client, clock, _RaisingQueue(tmp_path / 'queue'))

        row = read_debt(db, A)
        assert row is not None and row.open_count == 2 and row.owner_task_id == 'deflake-2'
        assert _items(db) == []
        assert 'escalation queue unavailable' in caplog.text


@pytest.mark.asyncio
class TestStaleHead:
    """PRD D6: evidence from a head that lacks a finished de-flake owner's fix closes
    that owner's cycle and never re-opens it as a regression (scenario 40)."""

    async def test_merge_gate_form_closes_the_cycle_and_a_fixed_head_re_enters_once(
        self, tmp_path: Path,
    ) -> None:
        db, clock, client = _db(tmp_path), _Clock(), _FakeLedgerTaskClient()
        git = _FakeAncestry({(FIX, STALE_HEAD): False, (FIX, GOOD_HEAD): True})
        queue = _RecordingQueue(tmp_path / 'queue')
        owner = await _owned(db, client, clock, A, 0)
        client.finish(owner, commit=FIX)
        _enqueue(db, A, 1, at=clock.now, head=STALE_HEAD)

        await _pass(db, client, clock, git=git, queue=queue)

        row = read_debt(db, A)
        assert row is not None
        assert row.resolved_at is not None and row.open_count == 1
        assert row.prior_resolving_commit == FIX
        assert queue.submitted == []
        assert _items(db) == []
        assert client.filed == [owner]

        _enqueue(db, A, 2, at=clock.now, head=GOOD_HEAD)
        await _pass(db, client, clock, git=git, queue=queue)

        row = read_debt(db, A)
        assert row is not None and row.open_count == 2 and row.resolved_at is None
        assert len(queue.regressions()) == 1
        assert client.filed == [owner, 'deflake-2']

    async def test_a_re_entry_files_under_a_title_the_store_does_not_fold_onto_the_done_owner(
        self, tmp_path: Path,
    ) -> None:
        db, clock, client = _db(tmp_path), _Clock(), _FakeLedgerTaskClient(folds_titles=True)
        git = _FakeAncestry({(FIX, GOOD_HEAD): True})
        owner = await _owned(db, client, clock, A, 0)
        client.finish(owner, commit=FIX)
        _enqueue(db, A, 1, at=clock.now, head=GOOD_HEAD)

        report = await _pass(db, client, clock, git=git, queue=_RecordingQueue(tmp_path / 'queue'))

        row = read_debt(db, A)
        assert row is not None and row.open_count == 2
        assert row.owner_task_id == 'deflake-2', 'the new cycle has its own owner'
        assert (report.failed, report.stop) == (0, None) and _items(db) == []

    async def test_drain_timing_form_an_owner_done_after_the_enqueue_closes_the_cycle(
        self, tmp_path: Path,
    ) -> None:
        db, clock, client = _db(tmp_path), _Clock(), _FakeLedgerTaskClient()
        git = _FakeAncestry({(FIX, STALE_HEAD): False})
        queue = _RecordingQueue(tmp_path / 'queue')
        owner = await _owned(db, client, clock, A, 0)
        _enqueue(db, A, 1, at=clock.now, head=STALE_HEAD)
        client.finish(owner, commit=FIX)

        await _pass(db, client, clock, git=git, queue=queue)

        row = read_debt(db, A)
        assert row is not None and row.resolved_at is not None and row.open_count == 1
        assert queue.submitted == []
        assert _items(db) == []

    async def test_a_second_stale_item_of_a_closed_cycle_is_dropped(self, tmp_path: Path) -> None:
        db, clock, client = _db(tmp_path), _Clock(), _FakeLedgerTaskClient()
        git = _FakeAncestry({(FIX, STALE_HEAD): False})
        queue = _RecordingQueue(tmp_path / 'queue')
        owner = await _owned(db, client, clock, A, 0)
        client.finish(owner, commit=FIX)
        _enqueue(db, A, 1, at=clock.now, head=STALE_HEAD)
        _enqueue(db, A, 2, at=clock.now, head=STALE_HEAD)

        report = await _pass(db, client, clock, git=git, queue=queue)

        assert (report.attempted, report.settled) == (2, 2)
        row = read_debt(db, A)
        assert row is not None and row.resolved_at is not None and row.open_count == 1
        assert queue.submitted == []
        assert _items(db) == []
        assert git.calls == [(FIX, STALE_HEAD), (FIX, STALE_HEAD)]

    async def test_a_closed_cycle_with_no_recorded_commit_re_enters(self, tmp_path: Path) -> None:
        db, clock, client = _db(tmp_path), _Clock(), _FakeLedgerTaskClient()
        git = _FakeAncestry(default=False)
        queue = _RecordingQueue(tmp_path / 'queue')
        owner = await _owned(db, client, clock, A, 0)
        client.finish(owner)
        assert await resolve_debt(db, PROJECT, A, task_client=client)
        _enqueue(db, A, 1, at=clock.now, head=STALE_HEAD)

        await _pass(db, client, clock, git=git, queue=queue)

        row = read_debt(db, A)
        assert row is not None and row.open_count == 2, 'no fix commit counts as contained'
        assert git.calls == []
        assert len(queue.regressions()) == 1

    @pytest.mark.parametrize('form', ['open_row', 'resolved_row'])
    async def test_an_unreadable_ancestry_leaves_the_item_due_and_the_row_unchanged(
        self, tmp_path: Path, form: str,
    ) -> None:
        db, clock, client = _db(tmp_path), _Clock(), _FakeLedgerTaskClient()
        git = _FakeAncestry({(FIX, UNREAD_HEAD): None})
        queue = _RecordingQueue(tmp_path / 'queue')
        owner = await _owned(db, client, clock, A, 0)
        client.finish(owner, commit=FIX)
        if form == 'resolved_row':
            assert await resolve_debt(db, PROJECT, A, task_client=client)
        _enqueue(db, A, 1, at=clock.now, head=UNREAD_HEAD)
        before = read_debt(db, A)

        await _pass(db, client, clock, git=git, queue=queue)

        assert read_debt(db, A) == before
        item = _item(db, A)
        assert (item.attempts, item.last_error) == (1, DueFailure.ANCESTRY_UNREADABLE.value)
        assert queue.submitted == []


@pytest.mark.asyncio
class TestPassBounds:
    async def test_a_tests_later_item_waits_while_its_earlier_one_fails(
        self, tmp_path: Path,
    ) -> None:
        db, clock, client = _db(tmp_path), _Clock(), _FakeLedgerTaskClient()
        git = _FakeAncestry({(FIX, UNREAD_HEAD): None})
        owner = await _owned(db, client, clock, A, 0)
        client.finish(owner, commit=FIX)
        assert await resolve_debt(db, PROJECT, A, task_client=client)
        _enqueue(db, A, 1, at=clock.now, head=UNREAD_HEAD)
        _enqueue(db, A, 2, at=clock.now, head=GOOD_HEAD)
        _enqueue(db, B, 3, at=clock.now, head=None)

        within = await _pass(db, client, clock, git=git)

        assert (within.attempted, within.settled, within.failed) == (2, 1, 1)
        assert [(i.observed_at, i.attempts) for i in _items(db)] == [
            (_observed(1), 1), (_observed(2), 0),
        ]
        clock.advance(timedelta(minutes=1))
        across = await _pass(db, client, clock, git=git)
        assert across.attempted == 0, 'the later item waits behind the backed-off one'
        assert [i.attempts for i in _items(db)] == [1, 0]

        git.answers[(FIX, UNREAD_HEAD)] = True
        clock.advance(timedelta(minutes=5))
        assert (await _pass(db, client, clock, git=git)).settled == 2
        assert _items(db) == []

    async def test_the_backoff_doubles_to_an_hour(self, tmp_path: Path) -> None:
        db, clock = _db(tmp_path), _Clock()
        client = _FakeLedgerTaskClient(unreachable=True)
        _enqueue(db, A, 0, at=clock.now)

        delays = []
        for _ in range(6):
            await _pass(db, client, clock)
            due = datetime.fromisoformat(_item(db, A).next_attempt_at)
            delays.append(due - clock.now)
            clock.now = due

        assert delays == [timedelta(minutes=m) for m in (5, 10, 20, 40, 60, 60)]
        assert _item(db, A).attempts == 6

    async def test_a_pass_stops_at_the_first_transport_failure(self, tmp_path: Path) -> None:
        db, clock = _db(tmp_path), _Clock()
        client = _FakeLedgerTaskClient(unreachable=True)
        for n, test_id in enumerate((A, B, C)):
            _enqueue(db, test_id, n, at=clock.now)

        report = await _pass(db, client, clock)

        assert report == DrainPassReport(
            attempted=1, settled=0, failed=1, stop=DrainStop.TASK_CLIENT_FAILED,
        )
        assert {i.test_id: i.attempts for i in _items(db)} == {A: 1, B: 0, C: 0}

    async def test_a_pass_stops_at_its_time_budget(self, tmp_path: Path) -> None:
        db, clock = _db(tmp_path), _Clock()
        client = _FakeLedgerTaskClient(on_call=lambda: clock.advance(timedelta(minutes=2)))
        for n, test_id in enumerate((A, B, C)):
            _enqueue(db, test_id, n, at=clock.now)

        report = await _pass(db, client, clock)

        assert report == DrainPassReport(attempted=1, settled=1, failed=0, stop=DrainStop.BUDGET)
        assert {i.test_id: i.attempts for i in _items(db)} == {B: 0, C: 0}

    async def test_a_pass_never_raises(self, tmp_path: Path) -> None:
        db, clock = _db(tmp_path), _Clock()
        _enqueue(db, A, 0, at=clock.now)

        broken: FlakeLedgerTaskClient = _RaisingTaskClient()
        raising = await _pass(db, broken, clock)

        assert (raising.failed, raising.stop) == (1, DrainStop.TASK_CLIENT_FAILED)
        assert _item(db, A).last_error == DueFailure.TASK_CLIENT_FAILED.value
        unopenable = tmp_path / 'not-a-db'
        unopenable.mkdir()
        assert await _pass(unopenable, _FakeLedgerTaskClient(), clock) == DrainPassReport(
            0, 0, 0, None,
        )


@pytest.mark.asyncio
class TestStaleItemEscalation:
    """Scenario 41, merge-gate half: fused-memory unreachable, then answering."""

    async def test_one_record_names_the_stale_items_and_the_drain_resolves_it(
        self, tmp_path: Path,
    ) -> None:
        db, clock = _db(tmp_path), _Clock()
        client = _FakeLedgerTaskClient(unreachable=True)
        queue = _RecordingQueue(tmp_path / 'queue')

        async def _at(minutes: int) -> None:
            clock.now = T0 + timedelta(minutes=minutes)
            await _pass(db, client, clock, queue=queue)

        _enqueue(db, A, 0, at=T0)
        await _at(0)
        await _at(29)
        assert queue.stale_records() == [], 'nothing escalates before 30 minutes'

        await _at(31)
        (first,) = queue.stale_records()
        assert (first.level, first.severity, first.category, first.agent_role) == (
            1, 'blocking', 'infra_issue', STALE_DUE_ESCALATION_ROLE,
        )
        assert first.summary.startswith('1 flake-debt item(s) waiting over 30 min'), first.summary
        assert A in first.summary and T0.isoformat() in first.summary
        assert DueFailure.TASK_CLIENT_FAILED.value in first.summary
        assert _item(db, A).escalation_id == first.id

        await _at(32)
        record = queue.get(first.id)
        assert record is not None and record.amendments == [], 'no change appends nothing'

        queue.resolve(first.id, 'looked at it by hand', resolved_by='interactive')
        await _at(33)
        assert queue.stale_records() == [], 'not re-opened for the items it named'

        _enqueue(db, B, 1, at=T0 + timedelta(minutes=33))
        await _at(65)
        (second,) = queue.stale_records()
        assert second.id != first.id
        assert second.summary.startswith('2 flake-debt item(s)'), second.summary
        assert _item(db, B).escalation_id == second.id
        await _at(66)
        record = queue.get(second.id)
        assert record is not None and record.amendments == []

        _enqueue(db, C, 2, at=T0 + timedelta(minutes=66))
        await _at(97)
        record = queue.get(second.id)
        assert record is not None and len(record.amendments) == 1, 'a changed summary amends'
        assert record.amendments[-1]['summary'].startswith('3 flake-debt item(s)')
        assert _item(db, C).escalation_id == second.id

        client.unreachable = False
        await _at(140)
        assert _items(db) == []
        for test_id in (A, B, C):
            row = read_debt(db, test_id)
            assert row is not None and row.owner_task_id is not None, row
        await _at(141)

        assert queue.stale_records() == []
        resolved = queue.get(second.id)
        assert resolved is not None and resolved.status == 'resolved'
        assert resolved.resolved_by == STALE_DUE_ESCALATION_ROLE
        assert len([e for e in queue.submitted if e.task_id == STALE_DUE_ESCALATION_SENTINEL]) == 2

    async def test_the_stale_record_is_filed_before_any_client_call(self, tmp_path: Path) -> None:
        db, clock = _db(tmp_path), _Clock()
        client = _FakeLedgerTaskClient(submit_hangs=True)
        queue = _RecordingQueue(tmp_path / 'queue')
        _enqueue(db, A, 0, at=T0 - timedelta(minutes=31))

        drain = asyncio.create_task(_pass(db, client, clock, queue=queue))
        await client.submit_entered.wait()
        try:
            assert len(queue.stale_records()) == 1
        finally:
            drain.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await drain


@pytest.mark.asyncio
class TestTaskClientBounds:
    """The drain's own backstops on the task client: a call that outlives
    ``_TASK_CLIENT_CALL_TIMEOUT_SECS`` fails the attempt, and a cancellation inside a
    call is recorded as a failure and still propagates."""

    async def test_a_hung_client_call_fails_the_attempt_and_stops_the_pass(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        monkeypatch.setattr(flake_debt_drain, '_TASK_CLIENT_CALL_TIMEOUT_SECS', 0.05)
        db, clock, client = _db(tmp_path), _Clock(), _FakeLedgerTaskClient(submit_hangs=True)
        _enqueue(db, A, 0, at=clock.now)
        _enqueue(db, B, 1, at=clock.now)

        report = await asyncio.wait_for(_pass(db, client, clock), timeout=10)

        assert report.stop is DrainStop.TASK_CLIENT_FAILED
        assert (report.attempted, report.failed) == (1, 1), 'the pass stops at the first'
        assert _item(db, A).last_error == DueFailure.TASK_CLIENT_FAILED.value
        assert len(client.submit_calls) == 1

    async def test_a_cancellation_inside_a_client_call_propagates(self, tmp_path: Path) -> None:
        db, clock = _db(tmp_path), _Clock()
        client = _FakeLedgerTaskClient(submit_raises=asyncio.CancelledError())
        _enqueue(db, A, 0, at=clock.now)

        with pytest.raises(asyncio.CancelledError):
            await _pass(db, client, clock)

        assert [i.attempts for i in _items(db)] == [0], 'the cancelled attempt is not recorded'

    async def test_a_read_the_ledger_cancels_at_its_budget_counts_as_a_client_failure(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """The ledger bounds its owner read and cancels the call when the budget expires;
        the drain must still read that as the task client failing, not as a clean read."""
        monkeypatch.setattr(
            flake_debt_drain, 'open_debt', functools.partial(open_debt, resolve_budget_secs=0.05),
        )
        db, clock, client = _db(tmp_path), _Clock(), _FakeLedgerTaskClient()
        owner = await _owned(db, client, clock, A, 0)
        client.finish(owner, commit=FIX)
        client.get_task_hangs = True
        _enqueue(db, A, 1, at=clock.now)

        report = await asyncio.wait_for(_pass(db, client, clock), timeout=10)

        assert report.stop is DrainStop.TASK_CLIENT_FAILED
        assert _item(db, A).last_error == DueFailure.TASK_CLIENT_FAILED.value
