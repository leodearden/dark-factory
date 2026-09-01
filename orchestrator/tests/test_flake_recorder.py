"""Tests for the dispatcher-side flake RECORDER (PRD task ε, `plans/flake-ledger-prd.md`).

`flake_recorder.record_merge_flake_suppression` is the one place the three
side-effects of a merge-gate flake observation happen: the durable
``flake_occurrence`` ledger row, the ``merge_flake_suppressed`` structured fact,
and the INV-4 storm-streak bump.

It lives HERE, on the dispatcher, and not in ``verify.apply_merge_flake_suppression``
(which runs wherever the WORKTREE is) precisely because the producer's host may be a
remote runner with no event store, no escalation queue and its own module-global
streak counter — the defect ε exists to fix.  Every assertion below is therefore about
what the DISPATCHER does with an observation that merely rode ``VerifyResult`` to it,
including one case where the observation was genuinely wire-deserialized.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from orchestrator import flake_recorder
from orchestrator.flake_ledger import (
    UNKNOWN_TEST_ID,
    FlakeCallSite,
    FlakeSuppression,
    FlakeVerdict,
    ledger_db_path,
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
    """

    def __init__(self, open_l2=None) -> None:
        self.submitted: list = []
        self.get_by_task_calls: list = []
        self._open_l2 = open_l2

    def make_id(self, task_id: str) -> str:
        return f'esc-{task_id}-1'

    def get_by_task(self, task_id, *, status=None, level=None):
        self.get_by_task_calls.append((task_id, status, level))
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


def _record(result: VerifyResult, project_root: Path, **kwargs) -> None:
    kwargs.setdefault('merge_sha', _MERGE_SHA)
    kwargs.setdefault('task_id', _TASK_ID)
    flake_recorder.record_merge_flake_suppression(
        result,
        project_root=project_root,
        project_id=_PROJECT_ID,
        **kwargs,
    )


def _occurrences(project_root: Path):
    return read_occurrences(ledger_db_path(project_root))


class TestRecordMergeFlakeSuppression:
    """The recorder's contract: ledger ALWAYS, event+streak only on a suppression."""

    # -- (a) B13: no observation carried -> nothing happens -------------------

    def test_no_suppression_records_nothing(self, tmp_path: Path) -> None:
        """B13 (new dispatcher, OLD remote): the key is simply absent from the
        wire payload, so the field defaults to None.  That must be a silent
        no-op, not a crash and not a sentinel row — an old remote is a
        degradation, not an observation."""
        es, q = _FakeEventStore(), _FakeEscalationQueue()

        _record(_result(None), tmp_path, event_store=es, escalation_queue=q)

        assert _occurrences(tmp_path) == []
        assert es.emits == []
        assert q.submitted == []
        assert flake_recorder._merge_flake_suppression_streak == 0

    # -- (b) B3: all three side-effects, asserted TOGETHER --------------------

    def test_b3_suppression_writes_ledger_and_emits_and_bumps(self, tmp_path: Path) -> None:
        """B3 — the whole point of ε, asserted in ONE test.

        Two-out-of-three is the bug this task exists to fix (the remote path used
        to land ZERO of three), so splitting these into three tests would let a
        regression that drops exactly one of them stay green.
        """
        es, q = _FakeEventStore(), _FakeEscalationQueue()
        s = _suppression()

        _record(_result(s), tmp_path, event_store=es, escalation_queue=q)

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

    def test_wire_deserialized_suppression_records_identically(self, tmp_path: Path) -> None:
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

        _record(wired, tmp_path, event_store=es, escalation_queue=q)

        assert [e[0] for e in es.emits] == [EventType.merge_flake_suppressed]
        rows = _occurrences(tmp_path)
        assert len(rows) == 2, rows
        assert {r.verdict for r in rows} == {'passes_in_isolation'}
        assert {r.runner for r in rows} == {'remote-lab-1'}
        assert flake_recorder._merge_flake_suppression_streak == 1

    # -- (d) non-suppressing verdicts: the ledger only ------------------------

    def test_unconfirmable_records_the_row_only(self, tmp_path: Path) -> None:
        """§5.5 — record the OBSERVATION, not the remedy.  An unconfirmable
        observation changes no verdict, so it must NOT emit the suppression fact
        or bump the storm streak; but it IS counted, because θ's class-1 health
        check is an unconfirmable RATE and a dropped row makes "could not even
        determine which tests failed" invisible."""
        es, q = _FakeEventStore(), _FakeEscalationQueue()
        s = _suppression(
            FlakeVerdict.unconfirmable, test_ids=(), reason='no recoverable node-id',
        )

        _record(_result(s), tmp_path, event_store=es, escalation_queue=q)

        rows = _occurrences(tmp_path)
        assert len(rows) == 1, rows
        assert rows[0].verdict == 'unconfirmable'
        assert rows[0].test_id == UNKNOWN_TEST_ID
        assert json.loads(rows[0].detail)['unconfirmable_reason'] == 'no recoverable node-id'
        assert es.emits == []
        assert q.submitted == []
        assert flake_recorder._merge_flake_suppression_streak == 0

    def test_fails_in_isolation_records_the_row_only(self, tmp_path: Path) -> None:
        """A confirmed real red is still an observation worth counting — it is
        the DENOMINATOR θ's suppression rate divides by — but it is emphatically
        not a suppression, so no fact and no streak bump."""
        es, q = _FakeEventStore(), _FakeEscalationQueue()

        _record(
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

    def test_storm_threshold_files_one_l2_and_resets(self, tmp_path: Path) -> None:
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
            _record(_result(s), tmp_path, event_store=es, escalation_queue=q)

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

    # -- (f) B12: a broken ledger must never cost the other two signals -------

    def test_unwritable_ledger_does_not_raise_and_still_emits(
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
            _record(_result(_suppression()), tmp_path, event_store=es, escalation_queue=q)

        assert [r.levelname for r in caplog.records].count('WARNING') >= 1, caplog.text
        assert len(es.emits) == 1, es.emits
        assert flake_recorder._merge_flake_suppression_streak == 1

    def test_a_raising_event_store_does_not_disarm_the_storm_detector(
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
            _record(
                _result(_suppression()), tmp_path,
                event_store=_ExplodingEventStore(), escalation_queue=q,
            )

        assert len(_occurrences(tmp_path)) == 2, 'the ledger row is independent'
        assert flake_recorder._merge_flake_suppression_streak == 1, (
            'a broken event store must not disarm the storm detector'
        )
        assert 'database is locked' in caplog.text, 'the loss must be loud'

    # -- (g) the CLI / storeless caller ---------------------------------------

    def test_none_stores_still_write_the_ledger_row(self, tmp_path: Path) -> None:
        """The ledger is the DURABLE evidence trail and does not depend on either
        in-process store, so a caller with neither still contributes to it."""
        _record(_result(_suppression()), tmp_path)

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
