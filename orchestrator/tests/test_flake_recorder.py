"""Tests for the dispatcher-side flake RECORDER (PRD task ε, `plans/flake-ledger-prd.md`).

`flake_recorder.record_merge_flake_suppression` is the one place the side-effects
of a merge-gate flake observation happen: the durable ``flake_occurrence`` ledger
row, the ``merge_flake_suppressed`` structured fact, the INV-4 storm-streak bump, and —
since ε0 (task 6934) — one due-list item per suppressed test, written in the rows' own
transaction.  The de-flake owner and the ``regressed_after_resolution`` escalation are
the drain's (``test_flake_debt_drain.py``).

It lives HERE, on the dispatcher, and not in ``verify.apply_merge_flake_suppression``
(which runs wherever the WORKTREE is) precisely because the producer's host may be a
remote runner with no event store, no escalation queue and its own module-global
streak counter — the defect ε exists to fix.  Every assertion below is therefore about
what the DISPATCHER does with an observation that merely rode ``VerifyResult`` to it,
including one case where the observation was genuinely wire-deserialized.
"""

from __future__ import annotations

import inspect
import json
import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
from _flake_golden import merge_gate_deflake_description
from escalation.queue import EscalationQueue
from test_flake_debt_drain import _FakeLedgerTaskClient, _RecordingQueue

from orchestrator import flake_debt_drain, flake_recorder
from orchestrator.flake_debt_drain import run_drain_pass
from orchestrator.flake_ledger import (
    UNKNOWN_TEST_ID,
    DebtContext,
    DebtTrigger,
    DueRow,
    EvidenceKind,
    FailureEvidence,
    FlakeCallSite,
    FlakeSuppression,
    FlakeVerdict,
    debt_context_from_json,
    delete_due_item,
    ledger_db_path,
    list_open_debt,
    read_debt,
    read_due_items,
    read_occurrences,
)
from orchestrator.verify import VerifyResult

_MERGE_SHA = 'd' * 40
_TASK_ID = '3789'
_PROJECT_ID = 'dark_factory'
_IDS = ('orchestrator/tests/test_x.py::test_y', 'orchestrator/tests/test_x.py::test_z')


class _FakeEventStore:
    """Records emit() calls without touching sqlite.

    Same shape as test_verify_merge_flake_suppression.py's — the recorder calls
    ``emit(event_type, task_id=..., data=...)`` exactly as the producer used to.
    """

    def __init__(self) -> None:
        self.emits: list[tuple] = []

    def emit(self, event_type, *, task_id=None, data=None, **kwargs) -> None:
        self.emits.append((event_type, task_id, data))


class _FakeEscalationQueue:
    """Records submit()/get_by_task()/make_id() for the storm-escalation tests.

    *open_l2* controls the dedup path: when truthy, get_by_task returns it so the
    filer treats an open L2 as already present and does NOT re-submit.

    *pending_from_submitted* answers from what was actually SUBMITTED instead, keyed on
    ``task_id`` (and the asked-for status and level) the way the real queue's lookup
    is.  A fixed *open_l2* gives the same answer for every key, so it can only show
    that the filer ASKS.  Keying on what was filed is what shows a pending escalation
    under one key leaves every other key free to file.
    """

    def __init__(self, open_l2=None, *, pending_from_submitted: bool = False) -> None:
        self.submitted: list = []
        self.get_by_task_calls: list = []
        self._open_l2 = open_l2
        self._pending_from_submitted = pending_from_submitted

    def make_id(self, task_id: str) -> str:
        return f'esc-{task_id}-1'

    def get_by_task(self, task_id, *, status=None, level=None):
        self.get_by_task_calls.append((task_id, status, level))
        if self._pending_from_submitted:
            return [
                esc for esc in self.submitted
                if esc.task_id == task_id
                and (status is None or esc.status == status)
                and (level is None or esc.level == level)
            ]
        return self._open_l2

    def submit(self, esc) -> None:
        self.submitted.append(esc)


@pytest.fixture(autouse=True)
def _reset_streak():
    """The streak is a MODULE-GLOBAL, so a test that bumps it and does not reset
    would make every later test in the session order-dependent.  Reset on BOTH
    sides so an xfail/error mid-test cannot leak either.
    """
    flake_recorder._merge_flake_suppression_streak = 0
    yield
    flake_recorder._merge_flake_suppression_streak = 0


def _suppression(
    verdict: FlakeVerdict = FlakeVerdict.passes_in_isolation,
    *,
    test_ids: tuple[str, ...] = _IDS,
    reason: str | None = None,
    runner: str = 'remote-lab-1',
    observed_at: str = '2026-08-22T12:00:00+00:00',
) -> FlakeSuppression:
    return FlakeSuppression(
        verdict=verdict,
        test_ids=test_ids,
        observed_at=observed_at,
        call_site=FlakeCallSite.merge_gate,
        runner=runner,
        psi_cpu_some10=41.5,
        unconfirmable_reason=reason,
    )


def _result(suppression: FlakeSuppression | None) -> VerifyResult:
    return VerifyResult(
        passed=suppression is not None
        and suppression.verdict == FlakeVerdict.passes_in_isolation,
        test_output='',
        lint_output='',
        type_output='',
        summary='ok',
        category='merge_flake_suppressed',
        flake_suppression=suppression,
    )


async def _record(result: VerifyResult, project_root: Path, **kwargs) -> None:
    kwargs.setdefault('merge_sha', _MERGE_SHA)
    kwargs.setdefault('task_id', _TASK_ID)
    await flake_recorder.record_merge_flake_suppression(
        result,
        project_root=project_root,
        project_id=_PROJECT_ID,
        **kwargs,
    )


def _occurrences(project_root: Path):
    return read_occurrences(ledger_db_path(project_root))


def _due(project_root: Path) -> list[DueRow]:
    items = read_due_items(
        ledger_db_path(project_root), now=datetime(2100, 1, 1, tzinfo=UTC), limit=10_000,
    )
    assert items is not None
    return items


@pytest.mark.asyncio
class TestRecordMergeFlakeSuppression:
    """The recorder's contract: ledger ALWAYS, event+streak only on a suppression."""

    # -- (a) B13: no observation carried -> nothing happens -------------------

    async def test_no_suppression_records_nothing(self, tmp_path: Path) -> None:
        """B13 (new dispatcher, OLD remote): the key is simply absent from the
        wire payload, so the field defaults to None.  That must be a silent
        no-op, not a crash and not a sentinel row — an old remote is a
        degradation, not an observation."""
        es, q = _FakeEventStore(), _FakeEscalationQueue()

        await _record(_result(None), tmp_path, event_store=es, escalation_queue=q)

        assert _occurrences(tmp_path) == []
        assert es.emits == []
        assert q.submitted == []
        assert flake_recorder._merge_flake_suppression_streak == 0

    # -- (b) B3: all three side-effects, asserted TOGETHER --------------------

    async def test_b3_suppression_writes_ledger_and_emits_and_bumps(self, tmp_path: Path) -> None:
        """B3 — the whole point of ε, asserted in ONE test.

        Two-out-of-three is the bug this task exists to fix (the remote path used
        to land ZERO of three), so splitting these into three tests would let a
        regression that drops exactly one of them stay green.
        """
        es, q = _FakeEventStore(), _FakeEscalationQueue()
        s = _suppression()

        await _record(_result(s), tmp_path, event_store=es, escalation_queue=q)

        # (1) the structured fact
        from orchestrator.event_store import EventType

        assert len(es.emits) == 1, es.emits
        event_type, task_id, data = es.emits[0]
        assert event_type == EventType.merge_flake_suppressed
        assert task_id == _TASK_ID
        assert sorted(data['node_ids']) == sorted(_IDS)
        assert data['merge_sha'] == _MERGE_SHA
        assert data['measured_at']

        # (2) the durable rows — one per named test
        rows = _occurrences(tmp_path)
        assert len(rows) == 2, rows
        assert {r.test_id for r in rows} == set(_IDS)
        for r in rows:
            assert r.verdict == 'passes_in_isolation'
            assert r.call_site == 'merge_gate'
            assert r.runner == 'remote-lab-1'
            assert r.merge_sha == _MERGE_SHA
            assert r.task_id == _TASK_ID
            assert r.project_id == _PROJECT_ID

        # (3) the INV-4 storm streak
        assert flake_recorder._merge_flake_suppression_streak == 1

    # -- (c) the SAME, for an observation that came off the wire --------------

    async def test_wire_deserialized_suppression_records_identically(self, tmp_path: Path) -> None:
        """The remote path, for real: the result is round-tripped through the
        runner codec before recording.

        This is what pins the ``==``-not-``is`` verdict comparison: a JSON
        round-trip can hand ``verdict`` back as a plain ``str``, and an identity
        test would silently skip the emit and the bump for EVERY remote
        suppression — precisely the host where the signal matters most.
        """
        from orchestrator.event_store import EventType
        from orchestrator.verify_runner import result_from_json, result_to_json

        es, q = _FakeEventStore(), _FakeEscalationQueue()
        wired = result_from_json(result_to_json(_result(_suppression())))

        await _record(wired, tmp_path, event_store=es, escalation_queue=q)

        assert [e[0] for e in es.emits] == [EventType.merge_flake_suppressed]
        rows = _occurrences(tmp_path)
        assert len(rows) == 2, rows
        assert {r.verdict for r in rows} == {'passes_in_isolation'}
        assert {r.runner for r in rows} == {'remote-lab-1'}
        assert flake_recorder._merge_flake_suppression_streak == 1

    # -- (d) non-suppressing verdicts: the ledger only ------------------------

    async def test_unconfirmable_records_the_row_only(self, tmp_path: Path) -> None:
        """§5.5 — record the OBSERVATION, not the remedy.  An unconfirmable
        observation changes no verdict, so it must NOT emit the suppression fact
        or bump the storm streak; but it IS counted, because θ's class-1 health
        check is an unconfirmable RATE and a dropped row makes "could not even
        determine which tests failed" invisible."""
        es, q = _FakeEventStore(), _FakeEscalationQueue()
        s = _suppression(
            FlakeVerdict.unconfirmable, test_ids=(), reason='no recoverable node-id',
        )

        await _record(_result(s), tmp_path, event_store=es, escalation_queue=q)

        rows = _occurrences(tmp_path)
        assert len(rows) == 1, rows
        assert rows[0].verdict == 'unconfirmable'
        assert rows[0].test_id == UNKNOWN_TEST_ID
        assert json.loads(rows[0].detail)['unconfirmable_reason'] == 'no recoverable node-id'
        assert es.emits == []
        assert q.submitted == []
        assert flake_recorder._merge_flake_suppression_streak == 0

    async def test_fails_in_isolation_records_the_row_only(self, tmp_path: Path) -> None:
        """A confirmed real red is still an observation worth counting — it is
        the DENOMINATOR θ's suppression rate divides by — but it is emphatically
        not a suppression, so no fact and no streak bump."""
        es, q = _FakeEventStore(), _FakeEscalationQueue()

        await _record(
            _result(_suppression(FlakeVerdict.fails_in_isolation)),
            tmp_path,
            event_store=es,
            escalation_queue=q,
        )

        rows = _occurrences(tmp_path)
        assert len(rows) == 2, rows
        assert {r.verdict for r in rows} == {'fails_in_isolation'}
        assert es.emits == []
        assert q.submitted == []
        assert flake_recorder._merge_flake_suppression_streak == 0

    # -- (e) INV-4: the storm escalation, now reachable from the dispatcher ---

    async def test_storm_threshold_files_one_l2_and_resets(self, tmp_path: Path) -> None:
        """INV-4's escape hatch, driven end-to-end through the recorder.

        Before ε this was unreachable on the remote path (the counter lived in
        the REMOTE process and reset on every process exit).  Recording on the
        dispatcher is what re-arms it.
        """
        es, q = _FakeEventStore(), _FakeEscalationQueue(open_l2=None)
        threshold = flake_recorder._MERGE_FLAKE_SUPPRESSION_STREAK_THRESHOLD

        for i in range(threshold):
            # Vary observed_at so §8.3's (test_id, observed_at, call_site) dedup
            # does not collapse these into one logical observation.
            s = _suppression(observed_at=f'2026-08-22T12:00:0{i}+00:00')
            await _record(_result(s), tmp_path, event_store=es, escalation_queue=q)

        assert len(q.submitted) == 1, q.submitted
        esc = q.submitted[0]
        assert esc.task_id == flake_recorder._MERGE_FLAKE_SUPPRESSION_STORM_SENTINEL
        assert esc.level == 2
        assert esc.severity == 'critical'
        assert esc.agent_role.startswith('orchestrator-')
        assert esc.category == 'merge_flake_suppression_storm'
        # The window resets so the next one makes an independent decision (B4).
        assert flake_recorder._merge_flake_suppression_streak == 0
        # ... and every suppression along the way still emitted + landed rows.
        assert len(es.emits) == threshold
        assert len(_occurrences(tmp_path)) == 2 * threshold

    async def test_a_bare_str_test_ids_warns_once_per_suppression(
        self, tmp_path: Path, caplog,
    ) -> None:
        """The rows and the due items both derive from ``test_ids``; the one upstream
        bug is reported once, not once per derivation."""
        s = _suppression(test_ids=cast(Any, _IDS[0]))

        with caplog.at_level(logging.WARNING):
            await _record(
                _result(s), tmp_path, event_store=_FakeEventStore(),
                escalation_queue=_FakeEscalationQueue(),
            )

        assert caplog.text.count('arrived as a bare str') == 1, caplog.text
        assert [d.test_id for d in _due(tmp_path)] == [_IDS[0]]

    # -- (f) B12: a broken ledger must never cost the other two signals -------

    async def test_unwritable_ledger_does_not_raise_and_still_emits(
        self, tmp_path: Path, caplog,
    ) -> None:
        """B12 — a ledger failure must never fail a verify or a merge, AND must
        not take the in-process signals down with it.

        ``data/orchestrator/runs.db`` is made a DIRECTORY, so every sqlite open
        against it fails.  The recorder swallows that loudly and carries on: the
        merge_flake_suppressed fact and the storm streak are independent
        evidence and losing all three to one broken disk is strictly worse than
        losing one.
        """
        (tmp_path / 'data' / 'orchestrator' / 'runs.db').mkdir(parents=True)
        es, q = _FakeEventStore(), _FakeEscalationQueue()

        with caplog.at_level(logging.WARNING):
            await _record(_result(_suppression()), tmp_path, event_store=es, escalation_queue=q)

        assert [r.levelname for r in caplog.records].count('WARNING') >= 1, caplog.text
        assert len(es.emits) == 1, es.emits
        assert flake_recorder._merge_flake_suppression_streak == 1

    async def test_a_raising_event_store_does_not_disarm_the_storm_detector(
        self, tmp_path: Path, caplog,
    ) -> None:
        """B12, the other ordering: a broken EVENT store must not take the INV-4
        streak down with it.

        The three side-effects share a never-raise contract, but with ONE shared
        ``try`` that contract was order-dependent: an ``emit`` that raised (a locked
        or closed sqlite store) jumped straight to the catch-all and the bump never
        ran — so the one signal whose entire job is to fire when something is going
        wrong was the first thing a failure switched off.  Each side-effect gets its
        own guard, so the ledger row and the streak both survive.
        """

        class _ExplodingEventStore:
            def emit(self, *a, **kw):
                raise RuntimeError('database is locked')

        q = _FakeEscalationQueue()
        with caplog.at_level(logging.WARNING):
            await _record(
                _result(_suppression()), tmp_path,
                event_store=_ExplodingEventStore(), escalation_queue=q,
            )

        assert len(_occurrences(tmp_path)) == 2, 'the ledger row is independent'
        assert flake_recorder._merge_flake_suppression_streak == 1, (
            'a broken event store must not disarm the storm detector'
        )
        assert 'database is locked' in caplog.text, 'the loss must be loud'

    # -- (g) the CLI / storeless caller ---------------------------------------

    async def test_none_stores_still_write_the_ledger_row(self, tmp_path: Path) -> None:
        """The ledger is the DURABLE evidence trail and does not depend on either
        in-process store, so a caller with neither still contributes to it."""
        await _record(_result(_suppression()), tmp_path)

        assert len(_occurrences(tmp_path)) == 2
        # None-safe all the way through: the streak still advances (it is a
        # module-global, not a store), and nothing raised.
        assert flake_recorder._merge_flake_suppression_streak == 1


class TestSuppressionStormStreak:
    """INV-4 storm detector: _bump_suppression_streak_and_maybe_escalate files
    exactly ONE born-at-L2 escalation once the module-global suppression streak
    reaches the threshold, then resets. Only suppressions bump it (task α, B4).

    Moved here verbatim from test_verify_merge_flake_suppression.py by task ε
    along with the helper itself: the storm detector is the RECORDER's, not the
    producer's, and a test left behind in the producer's file would keep asserting
    on a module that no longer owns the behaviour.
    """

    def _reset(self):
        """Belt to the autouse fixture's braces, and the handle the cases below
        use to reach the module — these drive the bump helper DIRECTLY, one unit
        below ``record_merge_flake_suppression``."""
        flake_recorder._merge_flake_suppression_streak = 0
        return flake_recorder

    def test_storm_files_one_l2_escalation_at_threshold_and_resets(self) -> None:
        vm = self._reset()
        threshold = vm._MERGE_FLAKE_SUPPRESSION_STREAK_THRESHOLD
        q = _FakeEscalationQueue(open_l2=None)

        for _ in range(threshold):
            vm._bump_suppression_streak_and_maybe_escalate(q, '2768', _MERGE_SHA)

        assert len(q.submitted) == 1, q.submitted
        esc = q.submitted[0]
        assert esc.level == 2
        assert esc.severity == 'critical'
        assert esc.agent_role.startswith('orchestrator-')
        assert esc.category == 'merge_flake_suppression_storm'
        # Counter cleared so the next window starts fresh (B4).
        assert vm._merge_flake_suppression_streak == 0

    def test_below_threshold_files_nothing(self) -> None:
        vm = self._reset()
        threshold = vm._MERGE_FLAKE_SUPPRESSION_STREAK_THRESHOLD
        q = _FakeEscalationQueue(open_l2=None)

        for _ in range(threshold - 1):
            vm._bump_suppression_streak_and_maybe_escalate(q, '2768', _MERGE_SHA)

        assert q.submitted == []
        assert vm._merge_flake_suppression_streak == threshold - 1

    def test_dedup_skips_submit_when_open_l2_exists(self) -> None:
        vm = self._reset()
        threshold = vm._MERGE_FLAKE_SUPPRESSION_STREAK_THRESHOLD
        q = _FakeEscalationQueue(open_l2=object())  # a truthy open L2

        for _ in range(threshold):
            vm._bump_suppression_streak_and_maybe_escalate(q, '2768', _MERGE_SHA)

        assert q.submitted == []
        # get_by_task consulted for dedup on the same sentinel, at level=2.
        assert q.get_by_task_calls, q.get_by_task_calls
        assert q.get_by_task_calls[-1][2] == 2
        assert vm._merge_flake_suppression_streak == 0

    def test_none_queue_is_tolerated(self) -> None:
        vm = self._reset()
        threshold = vm._MERGE_FLAKE_SUPPRESSION_STREAK_THRESHOLD

        for _ in range(threshold):
            vm._bump_suppression_streak_and_maybe_escalate(None, '2768', _MERGE_SHA)

        # No crash; window resets without filing (no queue to file into).
        assert vm._merge_flake_suppression_streak == 0


@pytest.mark.asyncio
class TestRecordEnqueuesDebt:
    """Since ε0 (task 6934) a suppression's debt is not filed here: each suppressed test
    gets an item on the ledger's due list, in the same transaction as its occurrence
    row, and ``flake_debt_drain`` files the owners afterwards.  No network call is left
    on this path, and the recorder takes no task client at all.  ASYNC-ONLY CLASS."""

    async def test_nine_suppressed_tests_enqueue_nine_items(self, tmp_path: Path) -> None:
        ids = tuple(f'orchestrator/tests/test_nine.py::test_{i}' for i in range(9))

        await _record(_result(_suppression(test_ids=ids)), tmp_path)

        items = _due(tmp_path)
        assert sorted(i.test_id for i in items) == sorted(ids)
        for item in items:
            assert (item.head, item.observing_task_id, item.call_site) == (
                _MERGE_SHA, _TASK_ID, 'merge_gate',
            )
            assert item.observed_at == '2026-08-22T12:00:00+00:00'
            assert (item.attempts, item.context) == (0, '{}')
        assert len(_occurrences(tmp_path)) == 9
        assert list_open_debt(ledger_db_path(tmp_path)) == [], 'the drain files, not this path'
        assert 'task_client' not in inspect.signature(
            flake_recorder.record_merge_flake_suppression,
        ).parameters

    async def test_the_rows_and_the_items_land_together_or_not_at_all(
        self, tmp_path: Path, caplog,
    ) -> None:
        """A due list whose shape refuses the item insert takes the occurrence rows down
        with it: one transaction, so a crash can never leave one without the other."""
        db_path = ledger_db_path(tmp_path)
        db_path.parent.mkdir(parents=True)
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute(
                'CREATE TABLE flake_debt_due '
                '(id INTEGER PRIMARY KEY, test_id TEXT, observed_at TEXT, call_site TEXT)'
            )
            conn.commit()
        finally:
            conn.close()

        with caplog.at_level(logging.WARNING):
            await _record(_result(_suppression()), tmp_path)

        assert _occurrences(tmp_path) == [], 'no row without its item'
        conn = sqlite3.connect(str(db_path))
        try:
            assert conn.execute('SELECT COUNT(*) FROM flake_debt_due').fetchone() == (0,)
        finally:
            conn.close()
        assert flake_recorder._merge_flake_suppression_streak == 1
        assert 'nothing was written' in caplog.text

    async def test_a_replay_lands_neither_a_second_row_nor_a_second_item(
        self, tmp_path: Path,
    ) -> None:
        s = _suppression()
        await _record(_result(s), tmp_path)
        await _record(_result(s), tmp_path)

        assert len(_occurrences(tmp_path)) == 2
        items = _due(tmp_path)
        assert len(items) == 2
        for item in items:
            assert delete_due_item(ledger_db_path(tmp_path), item.id)

        await _record(_result(s), tmp_path)

        assert len(_occurrences(tmp_path)) == 2
        assert _due(tmp_path) == [], 'a replay after the drain must not re-run open_debt'

    @pytest.mark.parametrize(
        ('verdict', 'ids', 'reason'),
        [
            (FlakeVerdict.fails_in_isolation, _IDS, None),
            (FlakeVerdict.unconfirmable, (), 'no recoverable node-id'),
        ],
    )
    async def test_non_suppressing_verdicts_write_rows_and_no_items(
        self, tmp_path: Path, verdict, ids, reason,
    ) -> None:
        """§5.5 — debt is owed for a SUPPRESSION, never for an observation.  The
        unconfirmable case is also the sentinel row: counted, never enqueued."""
        await _record(_result(_suppression(verdict, test_ids=ids, reason=reason)), tmp_path)

        assert len(_occurrences(tmp_path)) == (1 if not ids else len(ids))
        assert _due(tmp_path) == []

    async def test_a_literal_sentinel_on_a_suppression_is_never_enqueued(
        self, tmp_path: Path,
    ) -> None:
        await _record(_result(_suppression(test_ids=(UNKNOWN_TEST_ID,))), tmp_path)

        assert [r.test_id for r in _occurrences(tmp_path)] == [UNKNOWN_TEST_ID]
        assert _due(tmp_path) == []

    async def test_a_raising_ledger_write_still_bumps_the_storm_streak(
        self, tmp_path: Path, monkeypatch, caplog,
    ) -> None:
        def _raises(*args, **kwargs):
            raise RuntimeError('the ledger write blew up')

        monkeypatch.setattr(flake_recorder, 'enqueue_debt', _raises)
        es = _FakeEventStore()

        with caplog.at_level(logging.WARNING):
            await _record(_result(_suppression()), tmp_path, event_store=es)

        assert flake_recorder._merge_flake_suppression_streak == 1
        assert len(es.emits) == 1
        assert '[ledger]' in caplog.text

    async def test_exploding_test_ids_write_nothing_and_still_bump_the_streak(
        self, tmp_path: Path,
    ) -> None:
        """``FlakeSuppression`` rides the wire unvalidated, so reading ``s.test_ids`` can
        itself fail; B12 says that costs the merge nothing, and the storm escape stays
        armed."""
        class _UniterableIds(tuple[str, ...]):
            def __iter__(self):
                raise TypeError('test_ids is not iterable on this wire object')

        s = _suppression(test_ids=_UniterableIds(_IDS))
        await _record(_result(s), tmp_path)

        assert _occurrences(tmp_path) == []
        assert _due(tmp_path) == []
        assert flake_recorder._merge_flake_suppression_streak == 1

    async def test_many_distinct_tests_still_trip_the_storm_at_five(self, tmp_path: Path) -> None:
        """Distinct tests each owing a de-flake task is a systemic signal the per-test
        dedup cannot bound; the fixed-sentinel streak still trips across them."""
        q = _FakeEscalationQueue(open_l2=None)
        threshold = flake_recorder._MERGE_FLAKE_SUPPRESSION_STREAK_THRESHOLD

        for i in range(threshold):
            s = _suppression(
                test_ids=(f'orchestrator/tests/test_s.py::test_{i}',),
                observed_at=f'2026-08-22T12:00:0{i}+00:00',
            )
            await _record(_result(s), tmp_path, escalation_queue=q)

        assert len(q.submitted) == 1, q.submitted
        assert q.submitted[0].task_id == flake_recorder._MERGE_FLAKE_SUPPRESSION_STORM_SENTINEL
        assert q.submitted[0].level == 2
        assert len(_due(tmp_path)) == threshold


_STORM_SUMMARY = (
    'Merge-verify flake-suppression storm: the isolated-rerun-confirm gate has '
    'suppressed 5 merge-verify reds since the last reset'
)


def _merge_gate_title(test_id: str, cycle: int) -> str:
    return (
        f'Flake debt: {test_id} — passes_in_isolation after a merge-gate red was '
        f'suppressed, cycle {cycle}'
    )


def _metadata(row) -> dict:
    return {
        'spawn_context': 'flake_ledger_debt',
        'flake_debt_test': row.test_id,
        'flake_debt_open_count': row.open_count,
        'flake_debt_opened_at': row.opened_at,
    }


@pytest.mark.asyncio
class TestScenario32MergeGateUnchanged:
    """Scenario 32 (plans/offline-lane-red-path-prd.md): the merge gate through the one
    recorder body and the drain leaves today's rows, debt, storm escalation and task
    text, with the cycle now named in the de-flake title.  ASYNC-ONLY CLASS."""

    async def test_five_suppressions_and_one_drain_pass_leave_todays_effects(
        self, tmp_path: Path,
    ) -> None:
        queue = EscalationQueue(tmp_path / 'queue')
        for n in range(5):
            await _record(
                _result(_suppression(observed_at=f'2026-08-22T12:00:0{n}+00:00')),
                tmp_path,
                task_id=f'{_TASK_ID}-{n}',
                escalation_queue=queue,
            )
        client = _FakeLedgerTaskClient()

        await run_drain_pass(
            ledger_db_path(tmp_path), _PROJECT_ID,
            task_client=client, git=None, escalation_queue=queue,
        )

        rows = _occurrences(tmp_path)
        assert len(rows) == 10, rows
        assert {(r.test_id, r.task_id) for r in rows} == {
            (test_id, f'{_TASK_ID}-{n}') for test_id in _IDS for n in range(5)
        }
        for r in rows:
            assert (r.verdict, r.call_site, r.runner, r.merge_sha, r.detail) == (
                'passes_in_isolation', 'merge_gate', 'remote-lab-1', _MERGE_SHA, '{}',
            )
        assert _due(tmp_path) == []
        debt = list_open_debt(ledger_db_path(tmp_path))
        assert sorted(d.test_id for d in debt) == sorted(_IDS)
        assert all(d.owner_task_id for d in debt), debt

        (storm,) = queue.get_by_task('merge-flake-suppression-storm', status='pending', level=2)
        assert (storm.category, storm.severity, storm.summary) == (
            'merge_flake_suppression_storm', 'critical', _STORM_SUMMARY,
        )

        assert len(client.submit_calls) == 2, client.submit_calls
        for block in client.submit_calls:
            row = read_debt(ledger_db_path(tmp_path), block['metadata']['flake_debt_test'])
            assert row is not None
            assert block['title'] == _merge_gate_title(row.test_id, 1)
            assert block['description'] == merge_gate_deflake_description(row)
            assert block['metadata'] == _metadata(row)
            assert (block['priority'], block['planning_mode']) == ('medium', True)


_HEAD = 'e' * 40
_LANE_A = 'tests/test_lane.py::test_a'
_LANE_B = 'tests/test_lane.py::test_b'


def _lane_context(trigger: DebtTrigger = DebtTrigger.cleared, **overrides) -> DebtContext:
    kwargs: dict[str, Any] = {
        'call_site': FlakeCallSite.offline_lane,
        'trigger': trigger,
        'command': 'qdrant-integration',
        'heads': (_HEAD,),
        'pass_ids': ('p1',),
    }
    kwargs.update(overrides)
    return DebtContext(**kwargs)


def _lane_suppression(
    verdict: FlakeVerdict = FlakeVerdict.passes_in_isolation,
    *,
    test_ids: tuple[str, ...] = (_LANE_A,),
    observed_at: str = '2026-10-19T12:00:00+00:00',
    call_site: FlakeCallSite = FlakeCallSite.offline_lane,
    evidence: dict[str, FailureEvidence] | None = None,
) -> FlakeSuppression:
    return FlakeSuppression(
        verdict=verdict,
        test_ids=test_ids,
        observed_at=observed_at,
        call_site=call_site,
        runner='local',
        psi_cpu_some10=12.0,
        unconfirmable_reason=None,
        evidence=evidence,
    )


async def _observe(project_root: Path, suppressions, **kwargs) -> None:
    kwargs.setdefault('call_site', FlakeCallSite.offline_lane)
    kwargs.setdefault('head_sha', _HEAD)
    kwargs.setdefault('debt_eligible', {})
    await flake_recorder.record_call_site_observations(
        tuple(suppressions), project_root=project_root, project_id=_PROJECT_ID, **kwargs,
    )


def _malform_the_due_list(project_root: Path) -> None:
    """A due list whose shape refuses the item insert, so the whole write fails."""
    db_path = ledger_db_path(project_root)
    db_path.parent.mkdir(parents=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            'CREATE TABLE flake_debt_due '
            '(id INTEGER PRIMARY KEY, test_id TEXT, observed_at TEXT, call_site TEXT)'
        )
        conn.commit()
    finally:
        conn.close()


@pytest.mark.asyncio
class TestRecordCallSiteObservations:
    """The one recorder body every call site shares (offline-lane PRD D6, §5.8): rows and
    due items first, under one guard per suppression, then the call site's live signals.
    ASYNC-ONLY CLASS."""

    async def test_a_recovery_with_no_suppression_enqueues_its_items(self, tmp_path: Path) -> None:
        recovered = _lane_context(DebtTrigger.recovered)

        await _observe(
            tmp_path, (),
            observed_at='2026-10-19T12:00:00Z',
            debt_eligible={_LANE_A: recovered, _LANE_B: None},
        )

        assert _occurrences(tmp_path) == []
        items = {i.test_id: i for i in _due(tmp_path)}
        assert set(items) == {_LANE_A, _LANE_B}
        for item in items.values():
            assert (item.call_site, item.head, item.observing_task_id) == (
                'offline_lane', _HEAD, None,
            )
            assert item.observed_at == '2026-10-19T12:00:00+00:00'
        assert debt_context_from_json(items[_LANE_A].context) == recovered
        assert items[_LANE_B].context == '{}'

    async def test_an_unnamed_item_with_no_instant_is_skipped_loudly(
        self, tmp_path: Path, caplog,
    ) -> None:
        with caplog.at_level(logging.WARNING):
            await _observe(tmp_path, (), observed_at=None, debt_eligible={_LANE_A: None})

        assert _due(tmp_path) == []
        assert 'observed_at' in caplog.text

    async def test_the_lane_emits_nothing_and_bumps_nothing(self, tmp_path: Path) -> None:
        """Scenario 30's recorder half: the storm alarm stays the merge gate's."""
        es, q = _FakeEventStore(), _FakeEscalationQueue(open_l2=None)
        threshold = flake_recorder._MERGE_FLAKE_SUPPRESSION_STREAK_THRESHOLD

        for i in range(threshold + 1):
            test_id = f'tests/test_lane.py::test_{i}'
            await _observe(
                tmp_path,
                (_lane_suppression(test_ids=(test_id,), observed_at=f'2026-10-19T12:00:0{i}+00:00'),),
                debt_eligible={test_id: None},
                event_store=es, escalation_queue=q,
            )
            if i < threshold:
                await _record(
                    _result(_suppression(observed_at=f'2026-08-22T12:00:0{i}+00:00')),
                    tmp_path, task_id=f'merge-{i}', event_store=es, escalation_queue=q,
                )

        assert len(q.submitted) == 1, q.submitted
        assert [task_id for _type, task_id, _data in es.emits] == [
            f'merge-{i}' for i in range(threshold)
        ]
        assert flake_recorder._merge_flake_suppression_streak == 0

    async def test_every_eligible_test_lands_with_its_context_in_its_rows_transaction(
        self, tmp_path: Path,
    ) -> None:
        cleared = _lane_context()
        evidence = {
            _LANE_A: FailureEvidence(kind=EvidenceKind.failed, first_line='AssertionError: a'),
            _LANE_B: FailureEvidence(kind=EvidenceKind.error, first_line='ImportError: b'),
        }
        s = _lane_suppression(
            test_ids=(_LANE_A, _LANE_B), observed_at='2026-10-19T11:00:00Z', evidence=evidence,
        )

        await _observe(
            tmp_path, (s,), observed_at='2026-10-19T12:30:00+00:00', debt_eligible={_LANE_A: cleared},
        )

        rows = {r.test_id: r for r in _occurrences(tmp_path)}
        assert json.loads(rows[_LANE_A].detail)['evidence']['first_line'] == 'AssertionError: a'
        assert json.loads(rows[_LANE_B].detail)['evidence']['first_line'] == 'ImportError: b'
        (item,) = _due(tmp_path)
        assert item.test_id == _LANE_A
        assert item.observed_at == '2026-10-19T11:00:00+00:00', "the suppression's instant"
        assert debt_context_from_json(item.context) == cleared

    async def test_rows_and_items_land_together_or_not_at_all(self, tmp_path: Path) -> None:
        _malform_the_due_list(tmp_path)

        await _observe(tmp_path, (_lane_suppression(),), debt_eligible={_LANE_A: _lane_context()})

        assert _occurrences(tmp_path) == []
        conn = sqlite3.connect(str(ledger_db_path(tmp_path)))
        try:
            assert conn.execute('SELECT COUNT(*) FROM flake_debt_due').fetchone() == (0,)
        finally:
            conn.close()

    async def test_eligibility_is_the_callers_not_the_verdicts(self, tmp_path: Path) -> None:
        await _observe(
            tmp_path, (_lane_suppression(FlakeVerdict.fails_in_isolation),),
            debt_eligible={_LANE_A: None},
        )
        assert [i.test_id for i in _due(tmp_path)] == [_LANE_A]

        other = tmp_path / 'other'
        await _observe(other, (_lane_suppression(),), debt_eligible={})
        assert [r.test_id for r in _occurrences(other)] == [_LANE_A]
        assert _due(other) == []

    async def test_a_suppression_the_ledger_refuses_costs_only_its_own_rows(
        self, tmp_path: Path, caplog,
    ) -> None:
        bad = _lane_suppression(test_ids=(_LANE_A,), observed_at='not a time')
        good = _lane_suppression(test_ids=(_LANE_B,))

        with caplog.at_level(logging.WARNING):
            await _observe(tmp_path, (bad, good), debt_eligible={_LANE_A: None, _LANE_B: None})

        assert [r.test_id for r in _occurrences(tmp_path)] == [_LANE_B]
        assert [i.test_id for i in _due(tmp_path)] == [_LANE_B]
        assert 'nothing was written' in caplog.text

    async def test_a_raising_write_costs_only_its_own_suppression(
        self, tmp_path: Path, monkeypatch, caplog,
    ) -> None:
        """One guard per suppression: a write that RAISES (``enqueue_debt`` itself never
        does) is caught by the recorder and named ``[ledger]``."""
        first = _lane_suppression(test_ids=(_LANE_A,))
        real = flake_recorder.enqueue_debt

        def _first_raises(*args, **kwargs):
            if kwargs.get('occurrence') is first:
                raise RuntimeError('the first write blew up')
            return real(*args, **kwargs)

        monkeypatch.setattr(flake_recorder, 'enqueue_debt', _first_raises)
        with caplog.at_level(logging.WARNING):
            await _observe(
                tmp_path, (first, _lane_suppression(test_ids=(_LANE_B,))),
                debt_eligible={_LANE_B: None},
            )

        assert [r.test_id for r in _occurrences(tmp_path)] == [_LANE_B]
        assert [i.test_id for i in _due(tmp_path)] == [_LANE_B]
        assert '[ledger]' in caplog.text

    async def test_a_call_site_with_no_row_writes_rows_and_no_live_signal(
        self, tmp_path: Path,
    ) -> None:
        es, q = _FakeEventStore(), _FakeEscalationQueue(open_l2=None)
        threshold = flake_recorder._MERGE_FLAKE_SUPPRESSION_STREAK_THRESHOLD

        for i in range(threshold):
            s = _lane_suppression(
                call_site=FlakeCallSite.main_probe, observed_at=f'2026-10-19T12:00:0{i}+00:00',
            )
            await _observe(
                tmp_path, (s,), call_site=FlakeCallSite.main_probe,
                event_store=es, escalation_queue=q,
            )

        assert len(_occurrences(tmp_path)) == threshold
        assert es.emits == []
        assert q.submitted == []

    async def test_the_rows_exist_before_the_live_signals_fire(self, tmp_path: Path) -> None:
        seen: list[int] = []

        class _ReadingEventStore:
            def emit(self, *args, **kwargs) -> None:
                seen.append(len(_occurrences(tmp_path)))

        await _observe(
            tmp_path, (_lane_suppression(call_site=FlakeCallSite.merge_gate),),
            call_site=FlakeCallSite.merge_gate, observing_task_id=_TASK_ID,
            event_store=_ReadingEventStore(),
        )

        assert seen == [1]

    async def test_never_raises_with_no_collaborators(self, tmp_path: Path) -> None:
        await _observe(
            tmp_path, (_lane_suppression(call_site=FlakeCallSite.merge_gate),),
            call_site=FlakeCallSite.merge_gate, debt_eligible={_LANE_A: None},
            event_store=None, escalation_queue=None,
        )

        assert len(_occurrences(tmp_path)) == 1


_FIX = 'f1' * 20
_H1 = '1' * 40
_H2 = '2' * 40


@pytest.mark.asyncio
class TestLaneRegressionFromTheDrain:
    """Scenario 31's ledger half: a lane test that clears again after its de-flake task
    landed raises the ledger's regression escalation, worded for the lane, and files a
    second owner the title fold cannot swallow.  ASYNC-ONLY CLASS."""

    async def _first_cycle(self, tmp_path: Path, client: _FakeLedgerTaskClient) -> None:
        ctx1 = _lane_context(heads=(_H1,), pass_ids=('p1',))
        await _observe(
            tmp_path, (_lane_suppression(observed_at='2026-10-19T12:00:00+00:00'),),
            head_sha=_H1, debt_eligible={_LANE_A: ctx1},
        )
        await run_drain_pass(
            ledger_db_path(tmp_path), _PROJECT_ID, task_client=client, git=None,
            escalation_queue=None,
        )
        assert client.filed == ['deflake-1']
        assert client.submit_calls[0]['title'].endswith(', cycle 1')
        client.finish('deflake-1', commit=_FIX)

    async def _drain(self, tmp_path: Path, client: _FakeLedgerTaskClient) -> _RecordingQueue:
        queue = _RecordingQueue(tmp_path / 'queue')
        await run_drain_pass(
            ledger_db_path(tmp_path), _PROJECT_ID, task_client=client, git=None,
            escalation_queue=queue,
        )
        return queue

    async def test_a_clear_after_the_fix_raises_one_lane_worded_regression(
        self, tmp_path: Path,
    ) -> None:
        client = _FakeLedgerTaskClient(folds_titles=True)
        await self._first_cycle(tmp_path, client)
        ctx2 = _lane_context(heads=(_H2,), pass_ids=('p2',))
        await _observe(
            tmp_path, (_lane_suppression(observed_at='2026-10-19T13:00:00+00:00'),),
            head_sha=_H2, debt_eligible={_LANE_A: ctx2},
        )

        queue = await self._drain(tmp_path, client)

        (esc,) = queue.regressions()
        assert 'an offline-lane clear has now RE-OPENED it' in esc.detail, esc.detail
        for text in ('cleared', 'p2', _H2, 'qdrant-integration'):
            assert text in esc.detail, text
        assert 'merge-gate suppression' not in esc.detail
        assert 'observed at merge_sha' not in esc.detail
        assert (esc.level, esc.severity, esc.agent_role, esc.category) == (
            2, 'critical', 'orchestrator-flake-ledger', 'regressed_after_resolution',
        )
        assert flake_debt_drain.regression_call_site(esc.evidence) == 'offline_lane'
        assert client.filed == ['deflake-1', 'deflake-2'], 'the title fold swallowed cycle 2'
        assert client.submit_calls[-1]['title'].endswith(', cycle 2')

    async def test_a_recovery_after_the_fix_raises_a_recovery_worded_regression(
        self, tmp_path: Path,
    ) -> None:
        client = _FakeLedgerTaskClient(folds_titles=True)
        await self._first_cycle(tmp_path, client)
        recovered = _lane_context(DebtTrigger.recovered, heads=(_H2,), pass_ids=('p2',))
        await _observe(
            tmp_path, (), head_sha=_H2, observed_at='2026-10-19T13:00:00+00:00',
            debt_eligible={_LANE_A: recovered},
        )

        queue = await self._drain(tmp_path, client)

        (esc,) = queue.regressions()
        assert 'an offline-lane recovery has now RE-OPENED it' in esc.detail, esc.detail
        assert flake_debt_drain.regression_call_site(esc.evidence) == 'offline_lane'
