"""Tests for the flake ledger (plans/flake-ledger-prd.md, task α).

Schema assertions deliberately bypass the module under test and read the
on-disk truth with raw ``sqlite3`` (the ``test_event_store.py`` / ``test_run_store.py``
convention), so a reader mapping bug can never make a schema test pass.

STRUCTURAL CONSTRAINT — sync and async tests live in strictly separate classes.
``orchestrator/pyproject.toml`` sets no ``asyncio_mode`` (pytest-asyncio STRICT) and
promotes "marked with '@pytest.mark.asyncio' but it is not an async function" to an
ERROR-level filterwarning, so a sync ``def test_`` inside an ``@pytest.mark.asyncio``
class is a collection FAILURE, not a warning.  The ledger API is deliberately
mixed-colour (sync writer/readers, async debt functions); the split is structural.
"""

from __future__ import annotations

import gc
import logging
import os
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

# Column order is pinned, not merely presence: run_store.py:53-58 establishes
# physical-column-order parity between a freshly-created DB and an ALTER-migrated
# one as this repo's migration convention (an operator forensic `SELECT *` is
# positional).  These lists are PRD §5.3 verbatim.
_OCCURRENCE_COLUMNS = [
    'id',
    'observed_at',
    'test_id',
    'project_id',
    'verdict',
    'call_site',
    'runner',
    'merge_sha',
    'task_id',
    'psi_cpu_some10',
    'detail',
]

_DEBT_COLUMNS = [
    'test_id',
    'project_id',
    'opened_at',
    'resolved_at',
    'owner_task_id',
    'open_count',
    'prior_resolved_at',
    'prior_resolving_commit',
    'last_occurrence_at',
]


def _rows(db_path: Path, sql: str, params: tuple = ()) -> list[dict]:
    """Read rows with raw sqlite3 — never through the module under test."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def _table_names(db_path: Path) -> list[str]:
    return [r['name'] for r in _rows(db_path, "SELECT name FROM sqlite_master WHERE type='table'")]


def _column_names(db_path: Path, table: str) -> list[str]:
    return [r['name'] for r in _rows(db_path, f'PRAGMA table_info({table})')]


class TestEnsureSchema:
    """``ensure_schema`` creates both PRD §5.3 tables additively on an existing runs.db."""

    def test_creates_both_tables(self, tmp_path: Path) -> None:
        from orchestrator.flake_ledger import ensure_schema

        db_path = tmp_path / 'runs.db'
        ensure_schema(db_path)

        names = _table_names(db_path)
        assert 'flake_occurrence' in names
        assert 'flake_debt' in names

    def test_flake_occurrence_column_order(self, tmp_path: Path) -> None:
        from orchestrator.flake_ledger import ensure_schema

        db_path = tmp_path / 'runs.db'
        ensure_schema(db_path)

        assert _column_names(db_path, 'flake_occurrence') == _OCCURRENCE_COLUMNS

    def test_flake_debt_column_order(self, tmp_path: Path) -> None:
        from orchestrator.flake_ledger import ensure_schema

        db_path = tmp_path / 'runs.db'
        ensure_schema(db_path)

        assert _column_names(db_path, 'flake_debt') == _DEBT_COLUMNS

    def test_flake_debt_test_id_is_primary_key(self, tmp_path: Path) -> None:
        """§5.3: runs.db is per-project, so test_id ALONE is the key — which is what
        makes η's recurrence trigger a PK lookup rather than a scan."""
        from orchestrator.flake_ledger import ensure_schema

        db_path = tmp_path / 'runs.db'
        ensure_schema(db_path)

        pk_columns = [r['name'] for r in _rows(db_path, 'PRAGMA table_info(flake_debt)') if r['pk']]
        assert pk_columns == ['test_id']

    def test_flake_occurrence_id_is_autoincrement(self, tmp_path: Path) -> None:
        from orchestrator.flake_ledger import ensure_schema

        db_path = tmp_path / 'runs.db'
        ensure_schema(db_path)

        pk_columns = [
            r['name'] for r in _rows(db_path, 'PRAGMA table_info(flake_occurrence)') if r['pk']
        ]
        assert pk_columns == ['id']

        # AUTOINCREMENT (as opposed to plain INTEGER PRIMARY KEY) materialises
        # sqlite_sequence on first insert — monotonic ids, never reused.  This is the
        # BEHAVIOURAL proof and it is the whole assertion: an `'AUTOINCREMENT' in sql`
        # substring check would also match the word sitting in a comment.
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute(
                'INSERT INTO flake_occurrence '
                '(observed_at, test_id, project_id, verdict, call_site) '
                "VALUES ('2026-08-06T12:00:00+00:00', 't', 'p', 'unconfirmable', 'merge_gate')"
            )
            conn.commit()
        finally:
            conn.close()
        assert 'sqlite_sequence' in _table_names(db_path)

    def test_dedup_index_is_unique_on_the_idempotency_triple(self, tmp_path: Path) -> None:
        """§8.3's idempotency key ``(test_id, observed_at, call_site)`` is enforced
        declaratively by a UNIQUE index, not by swallowing an IntegrityError.

        Asserted STRUCTURALLY, via ``PRAGMA index_list``/``index_info``, not by grepping
        the DDL text: ``'unique' in sql.lower()`` would still pass if the keyword were
        dropped and the index merely NAMED ``..._unique``, and a substring check on the
        column names cannot tell the correct triple from a superset or a different order.
        The PRAGMAs report what SQLite actually built.
        """
        from orchestrator.flake_ledger import ensure_schema

        db_path = tmp_path / 'runs.db'
        ensure_schema(db_path)

        index_list = {r['name']: r for r in _rows(db_path, 'PRAGMA index_list(flake_occurrence)')}
        assert 'idx_flake_occurrence_dedup' in index_list
        assert index_list['idx_flake_occurrence_dedup']['unique'] == 1

        # Exact ORDERED column list, so a reordered or widened key is a failure.
        assert [
            r['name'] for r in _rows(db_path, 'PRAGMA index_info(idx_flake_occurrence_dedup)')
        ] == ['test_id', 'observed_at', 'call_site']

    def test_observed_at_index_serves_the_window_scan(self, tmp_path: Path) -> None:
        """``since``/``limit`` reads scan by ``observed_at``; the index is non-UNIQUE
        because many tests are legitimately observed at the same instant."""
        from orchestrator.flake_ledger import ensure_schema

        db_path = tmp_path / 'runs.db'
        ensure_schema(db_path)

        index_list = {r['name']: r for r in _rows(db_path, 'PRAGMA index_list(flake_occurrence)')}
        assert 'idx_flake_occurrence_observed' in index_list
        assert index_list['idx_flake_occurrence_observed']['unique'] == 0
        assert [
            r['name'] for r in _rows(db_path, 'PRAGMA index_info(idx_flake_occurrence_observed)')
        ] == ['observed_at']

    def test_column_defaults(self, tmp_path: Path) -> None:
        """``open_count`` NOT NULL DEFAULT 1 and ``detail`` DEFAULT '{}', verified by
        inserting a minimal row and reading the defaults back."""
        from orchestrator.flake_ledger import ensure_schema

        db_path = tmp_path / 'runs.db'
        ensure_schema(db_path)

        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute(
                'INSERT INTO flake_occurrence '
                '(observed_at, test_id, project_id, verdict, call_site) '
                "VALUES ('2026-08-06T12:00:00+00:00', 't', 'p', 'unconfirmable', 'merge_gate')"
            )
            conn.execute(
                'INSERT INTO flake_debt (test_id, project_id, opened_at, last_occurrence_at) '
                "VALUES ('t', 'p', '2026-08-06T12:00:00+00:00', '2026-08-06T12:00:00+00:00')"
            )
            conn.commit()
        finally:
            conn.close()

        assert _rows(db_path, 'SELECT detail FROM flake_occurrence')[0]['detail'] == '{}'
        assert _rows(db_path, 'SELECT open_count FROM flake_debt')[0]['open_count'] == 1

        # NOT NULL, so an explicit NULL is rejected rather than silently stored.
        conn = sqlite3.connect(str(db_path))
        try:
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    'INSERT INTO flake_debt '
                    '(test_id, project_id, opened_at, last_occurrence_at, open_count) '
                    "VALUES ('t2', 'p', '2026-08-06T12:00:00+00:00', "
                    "'2026-08-06T12:00:00+00:00', NULL)"
                )
        finally:
            conn.close()

    def test_ensure_schema_is_idempotent(self, tmp_path: Path) -> None:
        from orchestrator.flake_ledger import ensure_schema

        db_path = tmp_path / 'runs.db'
        ensure_schema(db_path)
        ensure_schema(db_path)

        names = _table_names(db_path)
        assert names.count('flake_occurrence') == 1
        assert names.count('flake_debt') == 1

    def test_additive_on_an_existing_runs_db(self, tmp_path: Path) -> None:
        """The ledger is a FIFTH owner of runs.db — its tables must appear on every
        existing DB on first write, with no migration to sequence.

        The "existing" DB is built by the REAL ``EventStore``/``RunStore`` writing real
        rows, not a hand-copied DDL replica, so additivity is proven against the actual
        other owners' schemas and cannot silently drift from them.
        """
        from orchestrator.event_store import EventStore, EventType
        from orchestrator.flake_ledger import ensure_schema
        from orchestrator.run_store import RunStore

        db_path = tmp_path / 'runs.db'
        EventStore(db_path, 'run-x').emit(EventType.task_started, task_id='42')
        RunStore(db_path).start_run('run-x', 'dark_factory', '2026-08-06T12:00:00+00:00')

        events_before = _rows(db_path, 'SELECT * FROM events ORDER BY id')
        runs_before = _rows(db_path, 'SELECT * FROM runs ORDER BY run_id')
        assert len(events_before) == 1
        assert len(runs_before) == 1

        ensure_schema(db_path)

        # The other owners' rows are untouched...
        assert _rows(db_path, 'SELECT * FROM events ORDER BY id') == events_before
        assert _rows(db_path, 'SELECT * FROM runs ORDER BY run_id') == runs_before
        # ...and the two new tables now exist alongside them.
        names = _table_names(db_path)
        assert 'flake_occurrence' in names
        assert 'flake_debt' in names
        assert 'events' in names
        assert 'runs' in names

    def test_ledger_db_path(self, tmp_path: Path) -> None:
        """One spelling of harness.py:2119's hand-built runs.db literal, for the four
        downstream consumers (β, ε, ζ, ι)."""
        from orchestrator.flake_ledger import ledger_db_path

        assert ledger_db_path(tmp_path) == tmp_path / 'data' / 'orchestrator' / 'runs.db'


class TestFlakeVerdict:
    """PRD §8's verdict vocabulary — produced by β, consumed by ε/ζ, persisted here.

    One discriminator serves both merge gates, so the vocabulary is the thing that
    keeps them from drifting into different notions of "passes in isolation" (INV-5).
    """

    def test_exact_member_set(self) -> None:
        """Exactly three verdicts.  Asserted as a SET equality so a fourth cannot be
        added silently — a new verdict changes what every consumer must handle."""
        from orchestrator.flake_ledger import FlakeVerdict

        assert {v.value for v in FlakeVerdict} == {
            'passes_in_isolation',
            'fails_in_isolation',
            'unconfirmable',
        }

    def test_is_str_comparable(self) -> None:
        from orchestrator.flake_ledger import FlakeVerdict

        assert FlakeVerdict.passes_in_isolation == 'passes_in_isolation'
        assert FlakeVerdict.fails_in_isolation == 'fails_in_isolation'
        assert FlakeVerdict.unconfirmable == 'unconfirmable'


class TestFlakeCallSite:
    """PRD §5.3's ``call_site`` vocabulary, an enum for the same reason the verdict is.

    Sharper edge than the verdict, in fact: ``call_site`` is one third of §8.3's dedup
    key, so a typo at one producer ('merge-gate') would both defeat idempotency and split
    θ's per-site rates into two half-populated buckets that read as a trend, not a bug.
    """

    def test_exact_member_set(self) -> None:
        from orchestrator.flake_ledger import FlakeCallSite

        assert {c.value for c in FlakeCallSite} == {
            'merge_gate',
            'main_probe',
            'chronic_marker',
        }

    def test_is_str_comparable(self) -> None:
        """String-comparability is what lets a member bind straight into a SQL
        parameter with no codec."""
        from orchestrator.flake_ledger import FlakeCallSite

        assert FlakeCallSite.merge_gate == 'merge_gate'
        assert FlakeCallSite.main_probe == 'main_probe'
        assert FlakeCallSite.chronic_marker == 'chronic_marker'


class TestFlakeSuppression:
    """PRD §8's discriminator output — produced wherever the worktree is (local or
    remote), consumed ONLY on the dispatcher, riding VerifyResult across the wire."""

    def _suppression(self, **overrides):
        from orchestrator.flake_ledger import FlakeSuppression, FlakeVerdict

        kwargs = {
            'verdict': FlakeVerdict.passes_in_isolation,
            'test_ids': ('tests/test_a.py::test_one',),
            'observed_at': '2026-08-06T12:00:00+00:00',
            'call_site': 'merge_gate',
            'runner': 'local',
            'psi_cpu_some10': 17.5,
            'unconfirmable_reason': None,
        }
        kwargs.update(overrides)
        return FlakeSuppression(**kwargs)

    def test_is_frozen(self) -> None:
        """Frozen: the discriminator is PURE (§8.1 invariant 5), so its output is a
        value, not a mutable buffer a later stage can quietly amend."""
        import dataclasses

        s = self._suppression()
        with pytest.raises(dataclasses.FrozenInstanceError):
            s.verdict = 'fails_in_isolation'  # type: ignore[misc]

    def test_holds_supplied_values(self) -> None:
        from orchestrator.flake_ledger import FlakeVerdict

        s = self._suppression(test_ids=('a::t1', 'b::t2'))
        assert s.verdict is FlakeVerdict.passes_in_isolation
        assert s.test_ids == ('a::t1', 'b::t2')
        assert s.observed_at == '2026-08-06T12:00:00+00:00'
        assert s.call_site == 'merge_gate'
        assert s.runner == 'local'
        assert s.psi_cpu_some10 == 17.5
        assert s.unconfirmable_reason is None

    def test_optional_fields_accept_none(self) -> None:
        """``psi_cpu_some10`` is None when ``PsiSample.read_ok`` is False — NEVER a
        fabricated 0.0, which would read as "the host was idle"."""
        s = self._suppression(psi_cpu_some10=None, unconfirmable_reason=None)
        assert s.psi_cpu_some10 is None
        assert s.unconfirmable_reason is None

    def test_unconfirmable_carries_a_reason_and_may_have_no_test_ids(self) -> None:
        """§8: EMPTY ``test_ids`` is legal only for ``unconfirmable``."""
        from orchestrator.flake_ledger import FlakeVerdict

        s = self._suppression(
            verdict=FlakeVerdict.unconfirmable,
            test_ids=(),
            unconfirmable_reason='node-ids mapped to no discovered subproject',
        )
        assert s.test_ids == ()
        assert s.unconfirmable_reason == 'node-ids mapped to no discovered subproject'


def _suppression(**overrides):
    """A ``FlakeSuppression`` with sane defaults; override any field by keyword."""
    from orchestrator.flake_ledger import FlakeSuppression, FlakeVerdict

    kwargs = {
        'verdict': FlakeVerdict.passes_in_isolation,
        'test_ids': ('tests/test_a.py::test_one',),
        'observed_at': '2026-08-06T12:00:00+00:00',
        'call_site': 'merge_gate',
        'runner': 'local',
        'psi_cpu_some10': 17.5,
        'unconfirmable_reason': None,
    }
    kwargs.update(overrides)
    return FlakeSuppression(**kwargs)


class TestRecordFlakeOccurrence:
    """The task's headline observable signal: an observation written through the API
    reads back identical, one row per examined test."""

    def test_writes_one_row_per_test_id(self, tmp_path: Path) -> None:
        from orchestrator.flake_ledger import read_occurrences, record_flake_occurrence

        db_path = tmp_path / 'runs.db'
        s = _suppression(test_ids=('tests/test_a.py::test_one', 'tests/test_b.py::test_two'))
        assert (
            record_flake_occurrence(db_path, 'dark_factory', s, merge_sha='abc123', task_id='3785')
            is None
        )

        rows = read_occurrences(db_path)
        assert [r.test_id for r in rows] == [
            'tests/test_a.py::test_one',
            'tests/test_b.py::test_two',
        ]

    def test_round_trips_every_field(self, tmp_path: Path) -> None:
        from orchestrator.flake_ledger import read_occurrences, record_flake_occurrence

        db_path = tmp_path / 'runs.db'
        s = _suppression()
        record_flake_occurrence(db_path, 'dark_factory', s, merge_sha='abc123', task_id='3785')

        (row,) = read_occurrences(db_path)
        # observed_at is the SUPPLIED string, not a write-time stamp — §8.3's
        # idempotency key is only stable if the discriminator owns this value.
        assert row.observed_at == '2026-08-06T12:00:00+00:00'
        assert row.test_id == 'tests/test_a.py::test_one'
        assert row.project_id == 'dark_factory'
        assert row.verdict == 'passes_in_isolation'
        assert row.call_site == 'merge_gate'
        assert row.runner == 'local'
        assert row.merge_sha == 'abc123'
        assert row.task_id == '3785'
        # SQLite REAL is an IEEE-754 double and so is a Python float, so this is
        # bit-exact — deliberately `==`, not pytest.approx.
        assert row.psi_cpu_some10 == 17.5
        assert row.detail == '{}'

    def test_on_disk_columns_match_independently_of_the_reader(self, tmp_path: Path) -> None:
        """Asserted with raw sqlite3 so a reader-mapping bug cannot make this pass."""
        from orchestrator.flake_ledger import record_flake_occurrence

        db_path = tmp_path / 'runs.db'
        record_flake_occurrence(
            db_path, 'dark_factory', _suppression(), merge_sha='abc123', task_id='3785'
        )

        (raw,) = _rows(db_path, 'SELECT * FROM flake_occurrence ORDER BY id')
        assert raw['observed_at'] == '2026-08-06T12:00:00+00:00'
        assert raw['test_id'] == 'tests/test_a.py::test_one'
        assert raw['project_id'] == 'dark_factory'
        assert raw['verdict'] == 'passes_in_isolation'
        assert raw['call_site'] == 'merge_gate'
        assert raw['runner'] == 'local'
        assert raw['merge_sha'] == 'abc123'
        assert raw['task_id'] == '3785'
        assert raw['psi_cpu_some10'] == 17.5
        assert raw['detail'] == '{}'

    def test_psi_none_stores_sql_null_not_zero(self, tmp_path: Path) -> None:
        """PSI honesty (§4/§5.3): a failed ``PsiSample.read_ok`` means "we do not know
        the host pressure".  Storing 0.0 would read as "the host was idle" — the exact
        fabrication that turns an overload into a mystery flake."""
        from orchestrator.flake_ledger import read_occurrences, record_flake_occurrence

        db_path = tmp_path / 'runs.db'
        record_flake_occurrence(
            db_path,
            'dark_factory',
            _suppression(psi_cpu_some10=None),
            merge_sha=None,
            task_id=None,
        )

        nulls = _rows(db_path, 'SELECT psi_cpu_some10 IS NULL AS is_null FROM flake_occurrence')
        assert nulls[0]['is_null'] == 1

        (row,) = read_occurrences(db_path)
        assert row.psi_cpu_some10 is None
        assert row.psi_cpu_some10 != 0.0
        assert row.merge_sha is None
        assert row.task_id is None

    def test_detail_is_empty_json_when_no_reason(self, tmp_path: Path) -> None:
        from orchestrator.flake_ledger import record_flake_occurrence

        db_path = tmp_path / 'runs.db'
        record_flake_occurrence(
            db_path, 'dark_factory', _suppression(), merge_sha=None, task_id=None
        )

        assert _rows(db_path, 'SELECT detail FROM flake_occurrence')[0]['detail'] == '{}'

    def test_detail_carries_the_unconfirmable_reason_as_json(self, tmp_path: Path) -> None:
        import json

        from orchestrator.flake_ledger import FlakeVerdict, record_flake_occurrence

        db_path = tmp_path / 'runs.db'
        record_flake_occurrence(
            db_path,
            'dark_factory',
            _suppression(
                verdict=FlakeVerdict.unconfirmable,
                unconfirmable_reason='node-ids mapped to no discovered subproject',
            ),
            merge_sha=None,
            task_id=None,
        )

        detail = _rows(db_path, 'SELECT detail FROM flake_occurrence')[0]['detail']
        assert json.loads(detail)['unconfirmable_reason'] == (
            'node-ids mapped to no discovered subproject'
        )

    def test_creates_the_schema_on_a_fresh_path(self, tmp_path: Path) -> None:
        """Callers hold a path, not a store object (§8.3), so the writer provisions the
        schema itself — no separate ``ensure_schema`` call is required of ε or ι."""
        from orchestrator.flake_ledger import record_flake_occurrence

        db_path = tmp_path / 'nested' / 'runs.db'
        assert not db_path.exists()

        record_flake_occurrence(
            db_path, 'dark_factory', _suppression(), merge_sha=None, task_id=None
        )

        assert db_path.exists()
        assert _rows(db_path, 'SELECT COUNT(*) AS n FROM flake_occurrence')[0]['n'] == 1


class TestRecordFlakeOccurrenceIdempotence:
    """§8.3: idempotent per ``(test_id, observed_at, call_site)``.

    A merge-path write is retried in the normal course of events, so a replay must be
    a silent no-op.  These assertions are written so that "let the catch-all swallow an
    IntegrityError" is NOT enough to pass — see the partial-batch and no-warning cases.
    """

    def _record(self, db_path: Path, **overrides) -> None:
        from orchestrator.flake_ledger import record_flake_occurrence

        record_flake_occurrence(
            db_path,
            'dark_factory',
            _suppression(**overrides),
            merge_sha='abc123',
            task_id='3785',
        )

    def test_replay_writes_one_row_per_test_id(self, tmp_path: Path) -> None:
        db_path = tmp_path / 'runs.db'
        self._record(db_path, test_ids=('A', 'B'))
        self._record(db_path, test_ids=('A', 'B'))

        assert [r['test_id'] for r in _rows(db_path, 'SELECT test_id FROM flake_occurrence')] == [
            'A',
            'B',
        ]

    def test_partial_batch_replay_keeps_the_new_rows(self, tmp_path: Path) -> None:
        """The assertion that forces declarative ``INSERT OR IGNORE`` rather than an
        exception swallow: a plain INSERT would abort the second batch on A's duplicate
        and silently LOSE B, which is a dropped observation, not a dedup."""
        db_path = tmp_path / 'runs.db'
        self._record(db_path, test_ids=('A',))
        self._record(db_path, test_ids=('A', 'B'))

        assert sorted(
            r['test_id'] for r in _rows(db_path, 'SELECT test_id FROM flake_occurrence')
        ) == ['A', 'B']

    def test_replay_logs_no_warning(self, tmp_path: Path, caplog) -> None:
        """A retried merge-path write is normal, not a fault.  Warning on every replay
        would train operators to ignore the very log line B12 depends on being
        meaningful."""
        import logging

        db_path = tmp_path / 'runs.db'
        self._record(db_path)
        with caplog.at_level(logging.WARNING, logger='orchestrator.flake_ledger'):
            self._record(db_path)

        assert [r for r in caplog.records if r.name == 'orchestrator.flake_ledger'] == []

    def test_a_different_observed_at_is_a_new_observation(self, tmp_path: Path) -> None:
        db_path = tmp_path / 'runs.db'
        self._record(db_path, observed_at='2026-08-06T12:00:00+00:00')
        self._record(db_path, observed_at='2026-08-06T13:00:00+00:00')

        assert _rows(db_path, 'SELECT COUNT(*) AS n FROM flake_occurrence')[0]['n'] == 2

    def test_a_different_call_site_is_a_new_observation(self, tmp_path: Path) -> None:
        """The dedup key is the TRIPLE, not the test_id: the same test judged by the
        merge gate and by the main probe at the same instant is two observations."""
        db_path = tmp_path / 'runs.db'
        self._record(db_path, call_site='merge_gate')
        self._record(db_path, call_site='main_probe')

        assert sorted(
            r['call_site'] for r in _rows(db_path, 'SELECT call_site FROM flake_occurrence')
        ) == ['main_probe', 'merge_gate']


class TestRecordFlakeOccurrenceEmptyTestIds:
    """§8: EMPTY ``test_ids`` is legal only for ``unconfirmable``.

    The gate-blind signal must not vanish when it fires.  θ's class-1 health check is
    an ``unconfirmable`` RATE, so an observation that could not even determine WHICH
    tests failed still has to be counted — dropping it would reproduce the exact
    blindness this PRD exists to end (6 unconfirmable lines sitting at INFO for a
    month).  The mirror case, a CONFIRMED verdict about zero tests, is meaningless and
    degrades loudly instead of becoming a sentinel that would corrupt θ's denominator.
    """

    def _record(self, db_path: Path, **overrides) -> None:
        from orchestrator.flake_ledger import record_flake_occurrence

        record_flake_occurrence(
            db_path,
            'dark_factory',
            _suppression(**overrides),
            merge_sha='abc123',
            task_id='3785',
        )

    def _unconfirmable(self, **overrides) -> dict:
        from orchestrator.flake_ledger import FlakeVerdict

        kwargs = {
            'verdict': FlakeVerdict.unconfirmable,
            'test_ids': (),
            'unconfirmable_reason': 'node-ids mapped to no discovered subproject',
        }
        kwargs.update(overrides)
        return kwargs

    def test_unconfirmable_with_no_test_ids_is_still_counted(self, tmp_path: Path) -> None:
        import json

        from orchestrator.flake_ledger import UNKNOWN_TEST_ID, read_occurrences

        db_path = tmp_path / 'runs.db'
        self._record(db_path, **self._unconfirmable())

        (row,) = read_occurrences(db_path)
        assert row.test_id == UNKNOWN_TEST_ID
        assert row.verdict == 'unconfirmable'
        assert row.project_id == 'dark_factory'
        assert row.call_site == 'merge_gate'
        assert row.runner == 'local'
        assert row.merge_sha == 'abc123'
        assert row.task_id == '3785'
        assert row.psi_cpu_some10 == 17.5
        assert json.loads(row.detail)['unconfirmable_reason'] == (
            'node-ids mapped to no discovered subproject'
        )

    def test_the_sentinel_row_still_dedups_on_the_triple(self, tmp_path: Path) -> None:
        db_path = tmp_path / 'runs.db'
        self._record(db_path, **self._unconfirmable())
        self._record(db_path, **self._unconfirmable())

        assert _rows(db_path, 'SELECT COUNT(*) AS n FROM flake_occurrence')[0]['n'] == 1

    @pytest.mark.parametrize('verdict_name', ['passes_in_isolation', 'fails_in_isolation'])
    def test_a_confirmed_verdict_about_zero_tests_degrades_loudly(
        self, tmp_path: Path, caplog, verdict_name: str
    ) -> None:
        """Honest degrade: writes NOTHING, does not raise, and WARNS.  A confirmed
        verdict about zero tests is meaningless; accepting it silently would be a
        fabricated observation, and turning it into a sentinel row would corrupt the
        unconfirmable denominator θ's class-1 check divides by."""
        import logging

        from orchestrator.flake_ledger import FlakeVerdict, ensure_schema

        # Provisioned up front — the realistic precondition (a live runs.db already
        # carries the tables), and it makes the count below a genuine "nothing was
        # written" assertion rather than an artifact of a missing table.
        db_path = tmp_path / 'runs.db'
        ensure_schema(db_path)

        with caplog.at_level(logging.WARNING, logger='orchestrator.flake_ledger'):
            self._record(db_path, verdict=FlakeVerdict(verdict_name), test_ids=())

        assert _rows(db_path, 'SELECT COUNT(*) AS n FROM flake_occurrence')[0]['n'] == 0
        warnings = [r for r in caplog.records if r.name == 'orchestrator.flake_ledger']
        assert len(warnings) == 1
        assert warnings[0].levelno == logging.WARNING
        assert verdict_name in warnings[0].getMessage()


class TestRecordFlakeOccurrenceWireDeserializedVerdict:
    """``FlakeSuppression`` "rides VerifyResult across the wire" (§8) and is a plain
    dataclass with NO runtime validation, so a JSON round-trip hands ``verdict`` back as
    a plain ``str`` rather than a ``FlakeVerdict``.

    Two defects this pins, both on the remote-runner path:

    (a) B12.  Triage used to sit OUTSIDE the try/except and reach ``s.verdict.value``,
        so a str verdict raised ``AttributeError`` straight out of a public entry point.
        The merge path has no ``VerifyInfraError`` handler, so that stalls the QUEUE.
    (b) Misrouting.  An ``is``-comparison against the enum member is False for an
        equal-valued str, so a wire-deserialized ``unconfirmable`` with no node-ids fell
        through to the silent-drop branch — deleting exactly the class-1 signal the
        sentinel exists to preserve, and under-counting θ's unconfirmable rate.
    """

    def _record(self, db_path: Path, **overrides) -> None:
        from orchestrator.flake_ledger import record_flake_occurrence

        record_flake_occurrence(
            db_path,
            'dark_factory',
            _suppression(**overrides),
            merge_sha='abc123',
            task_id='3785',
        )

    def test_str_unconfirmable_with_no_test_ids_still_lands_the_sentinel(
        self, tmp_path: Path
    ) -> None:
        """Defect (b): the class-1 signal survives the wire, verbatim str and all."""
        from orchestrator.flake_ledger import UNKNOWN_TEST_ID, read_occurrences

        db_path = tmp_path / 'runs.db'
        self._record(
            db_path,
            verdict='unconfirmable',  # a plain str, exactly as JSON hands it back
            test_ids=(),
            unconfirmable_reason='node-ids mapped to no discovered subproject',
        )

        (row,) = read_occurrences(db_path)
        assert row.test_id == UNKNOWN_TEST_ID
        assert row.verdict == 'unconfirmable'

    def test_str_verdict_with_test_ids_round_trips_identically(self, tmp_path: Path) -> None:
        """A str verdict must be coerced, not written verbatim — the column carries the
        §5.5 vocabulary, and one discriminator serves both merge gates."""
        from orchestrator.flake_ledger import read_occurrences

        db_path = tmp_path / 'runs.db'
        self._record(db_path, verdict='passes_in_isolation')

        (row,) = read_occurrences(db_path)
        assert row.verdict == 'passes_in_isolation'
        assert row.test_id == 'tests/test_a.py::test_one'

    @pytest.mark.parametrize('verdict_name', ['passes_in_isolation', 'fails_in_isolation'])
    def test_str_confirmed_verdict_with_no_test_ids_degrades_loudly(
        self, tmp_path: Path, caplog, verdict_name: str
    ) -> None:
        """Defect (a): the drop branch reaches ``.value``, so it must not see a raw str.
        Returns None, writes nothing, WARNS — never raises."""
        import logging

        from orchestrator.flake_ledger import ensure_schema

        db_path = tmp_path / 'runs.db'
        ensure_schema(db_path)

        with caplog.at_level(logging.WARNING, logger='orchestrator.flake_ledger'):
            self._record(db_path, verdict=verdict_name, test_ids=())

        assert _rows(db_path, 'SELECT COUNT(*) AS n FROM flake_occurrence')[0]['n'] == 0
        warnings = [r for r in caplog.records if r.name == 'orchestrator.flake_ledger']
        assert len(warnings) == 1
        assert verdict_name in warnings[0].getMessage()

    @pytest.mark.parametrize(
        'bogus', ['flaky_test', '', None, 0], ids=['unknown_word', 'empty', 'none', 'int']
    )
    def test_an_unrecognised_verdict_degrades_through_b12(
        self, tmp_path: Path, caplog, bogus
    ) -> None:
        """B12 applies to a MALFORMED payload too: an unrecognised verdict must not be
        written verbatim into the vocabulary column, and must not raise ``ValueError``
        out of the entry point.  ``flaky_test`` is the §5.5 anti-vocabulary — if it ever
        reaches here it is a bug upstream, and the ledger's job is to log it, not to
        fail the merge."""
        import logging

        from orchestrator.flake_ledger import ensure_schema

        db_path = tmp_path / 'runs.db'
        ensure_schema(db_path)

        with caplog.at_level(logging.WARNING, logger='orchestrator.flake_ledger'):
            assert self._record(db_path, verdict=bogus) is None

        assert _rows(db_path, 'SELECT COUNT(*) AS n FROM flake_occurrence')[0]['n'] == 0
        _assert_logged_loudly(caplog)


class TestRecordFlakeOccurrenceWireHardening:
    """The other three key-bearing fields get the same treatment ``verdict`` gets.

    ``FlakeSuppression`` rides the wire with NO runtime validation, and ``call_site`` and
    ``observed_at`` are each one third of §8.3's ``(test_id, observed_at, call_site)``
    idempotency key — so a malformed one does not merely look wrong, it silently DEFEATS
    dedup and skews θ's rates.  ``test_ids`` is not in the key but is the evidence trail
    itself.
    """

    def _record(self, db_path: Path, **overrides) -> None:
        from orchestrator.flake_ledger import record_flake_occurrence

        record_flake_occurrence(
            db_path,
            'dark_factory',
            _suppression(**overrides),
            merge_sha='abc123',
            task_id='3785',
        )

    def test_str_call_site_round_trips(self, tmp_path: Path) -> None:
        """A JSON round-trip hands ``call_site`` back as a plain str, exactly as it does
        ``verdict``; the coercion must accept it, not reject it."""
        db_path = tmp_path / 'runs.db'
        self._record(db_path, call_site='main_probe')

        assert _rows(db_path, 'SELECT call_site FROM flake_occurrence')[0]['call_site'] == (
            'main_probe'
        )

    @pytest.mark.parametrize(
        'bogus', ['merge-gate', 'MERGE_GATE', '', None], ids=['hyphen', 'upper', 'empty', 'none']
    )
    def test_an_unrecognised_call_site_degrades_through_b12(
        self, tmp_path: Path, caplog, bogus
    ) -> None:
        """'merge-gate' for 'merge_gate' is the whole reason this is an enum: persisted
        verbatim it would dedup against nothing and split θ's per-site rates in two."""
        from orchestrator.flake_ledger import ensure_schema

        db_path = tmp_path / 'runs.db'
        ensure_schema(db_path)

        with caplog.at_level(logging.WARNING, logger='orchestrator.flake_ledger'):
            self._record(db_path, call_site=bogus)

        assert _rows(db_path, 'SELECT COUNT(*) AS n FROM flake_occurrence')[0]['n'] == 0
        _assert_logged_loudly(caplog)

    def test_z_suffixed_and_offset_stamps_dedup_to_one_row(self, tmp_path: Path) -> None:
        """The defect this closes: ``'…T12:00:00Z'`` and ``'…T12:00:00+00:00'`` are the
        SAME instant in two valid spellings, so without normalisation one observation
        lands twice — idempotency defeated — and the two forms sort apart inside a
        ``since`` window, skewing θ's rates."""
        db_path = tmp_path / 'runs.db'
        self._record(db_path, observed_at='2026-08-06T12:00:00Z')
        self._record(db_path, observed_at='2026-08-06T12:00:00+00:00')

        rows = _rows(db_path, 'SELECT observed_at FROM flake_occurrence')
        assert len(rows) == 1
        # Canonicalised to the explicit-offset spelling, so the TEXT column's
        # lexicographic order is the chronological order `since` relies on.
        assert rows[0]['observed_at'] == '2026-08-06T12:00:00+00:00'

    def test_a_non_utc_offset_is_converted_not_stored_verbatim(self, tmp_path: Path) -> None:
        """13:00+01:00 IS 12:00Z.  Stored verbatim it would sort AFTER a 12:30Z
        observation that actually happened later."""
        db_path = tmp_path / 'runs.db'
        self._record(db_path, observed_at='2026-08-06T13:00:00+01:00')

        assert _rows(db_path, 'SELECT observed_at FROM flake_occurrence')[0]['observed_at'] == (
            '2026-08-06T12:00:00+00:00'
        )

    def test_a_naive_stamp_is_read_as_utc_not_as_host_local_time(self, tmp_path: Path) -> None:
        """The field is documented as UTC, so a missing offset is ATTACHED, never
        ``.astimezone()``-ed — that would apply the DISPATCHER's local offset and
        silently shift a remote host's observation."""
        db_path = tmp_path / 'runs.db'
        self._record(db_path, observed_at='2026-08-06T12:00:00')

        assert _rows(db_path, 'SELECT observed_at FROM flake_occurrence')[0]['observed_at'] == (
            '2026-08-06T12:00:00+00:00'
        )

    @pytest.mark.parametrize(
        'bogus', ['not-a-date', '', None, 0], ids=['garbage', 'empty', 'none', 'int']
    )
    def test_a_malformed_observed_at_degrades_through_b12(
        self, tmp_path: Path, caplog, bogus
    ) -> None:
        """Dropping the write is the lesser evil: a stamp that parses as nothing breaks
        BOTH the dedup key and the window scan for every later read."""
        from orchestrator.flake_ledger import ensure_schema

        db_path = tmp_path / 'runs.db'
        ensure_schema(db_path)

        with caplog.at_level(logging.WARNING, logger='orchestrator.flake_ledger'):
            self._record(db_path, observed_at=bogus)

        assert _rows(db_path, 'SELECT COUNT(*) AS n FROM flake_occurrence')[0]['n'] == 0
        _assert_logged_loudly(caplog)

    def test_a_bare_str_test_ids_does_not_fan_out_per_character(
        self, tmp_path: Path, caplog
    ) -> None:
        """One node-id passed unwrapped is an easy mistake at ε's call site, it is
        TRUTHY, and ``tuple('a::t')`` explodes it into one row PER CHARACTER — a dozen
        garbage test_ids written into the exact table this PRD exists to make
        trustworthy.  Wrapped, not dropped: a single node-id string has one honest
        reading, so the observation survives and the warning names the producer bug."""
        from orchestrator.flake_ledger import read_occurrences

        db_path = tmp_path / 'runs.db'
        with caplog.at_level(logging.WARNING, logger='orchestrator.flake_ledger'):
            self._record(db_path, test_ids='tests/test_a.py::test_one')

        assert [r.test_id for r in read_occurrences(db_path)] == ['tests/test_a.py::test_one']
        warnings = [r for r in caplog.records if r.name == 'orchestrator.flake_ledger']
        assert len(warnings) == 1
        assert 'test_ids' in warnings[0].getMessage()


class TestReadOccurrences:
    """The reader filters ι (per-test recurrence chains) and θ (windowed rates) need,
    so neither hand-rolls SQL against these tables — §5.2's API-only rule."""

    T_EARLY = '2026-08-06T11:00:00+00:00'
    T_MID = '2026-08-06T12:00:00+00:00'
    T_LATE = '2026-08-06T13:00:00+00:00'
    TEST_A = 'tests/test_a.py::test_one'
    TEST_B = 'tests/test_b.py::test_two'

    def _seed(self, db_path: Path) -> None:
        from orchestrator.flake_ledger import record_flake_occurrence

        # Seeded OUT of chronological order deliberately, so `ORDER BY id` alone
        # cannot satisfy the ordering assertions below.  This is realistic: a remote
        # observation is written on the dispatcher after the local one that followed
        # it, so insertion order genuinely diverges from observation order.
        for observed_at, test_id, call_site in [
            (self.T_MID, self.TEST_B, 'main_probe'),
            (self.T_LATE, self.TEST_B, 'merge_gate'),
            (self.T_EARLY, self.TEST_A, 'merge_gate'),
            (self.T_MID, self.TEST_A, 'merge_gate'),
        ]:
            record_flake_occurrence(
                db_path,
                'dark_factory',
                _suppression(test_ids=(test_id,), observed_at=observed_at, call_site=call_site),
                merge_sha=None,
                task_id=None,
            )

    def test_unfiltered_returns_all_in_a_pinned_order(self, tmp_path: Path) -> None:
        """Ordering is a CONTRACT, not incidental ``ORDER BY id``: ι prints these
        chains, so they must read chronologically even though the rows were inserted
        by separate calls."""
        from orchestrator.flake_ledger import read_occurrences

        db_path = tmp_path / 'runs.db'
        self._seed(db_path)

        # Chronological, and the two rows tied at T_MID break on `id` — which is what
        # makes the sequence fully determined rather than merely mostly-sorted.
        assert [(r.observed_at, r.test_id) for r in read_occurrences(db_path)] == [
            (self.T_EARLY, self.TEST_A),
            (self.T_MID, self.TEST_B),  # seeded first, so lower id
            (self.T_MID, self.TEST_A),
            (self.T_LATE, self.TEST_B),
        ]

    def test_filters_by_test_id(self, tmp_path: Path) -> None:
        from orchestrator.flake_ledger import read_occurrences

        db_path = tmp_path / 'runs.db'
        self._seed(db_path)

        rows = read_occurrences(db_path, test_id=self.TEST_A)
        assert [r.observed_at for r in rows] == [self.T_EARLY, self.T_MID]
        assert {r.test_id for r in rows} == {self.TEST_A}

    def test_since_is_inclusive_at_the_boundary(self, tmp_path: Path) -> None:
        """ISO-8601 UTC strings with a fixed offset sort lexicographically in the same
        order as chronologically, which is why a TEXT column suffices."""
        from orchestrator.flake_ledger import read_occurrences

        db_path = tmp_path / 'runs.db'
        self._seed(db_path)

        rows = read_occurrences(db_path, since=self.T_MID)
        # BOTH rows stamped exactly at the boundary are included, so an off-by-one in
        # either direction is visible.
        assert [(r.observed_at, r.test_id) for r in rows] == [
            (self.T_MID, self.TEST_B),
            (self.T_MID, self.TEST_A),
            (self.T_LATE, self.TEST_B),
        ]

    def test_filters_compose(self, tmp_path: Path) -> None:
        from orchestrator.flake_ledger import read_occurrences

        db_path = tmp_path / 'runs.db'
        self._seed(db_path)

        rows = read_occurrences(db_path, test_id=self.TEST_B, since=self.T_LATE)
        assert [(r.observed_at, r.test_id) for r in rows] == [(self.T_LATE, self.TEST_B)]

    def test_limit_returns_the_most_recent_rows_still_oldest_first(self, tmp_path: Path) -> None:
        """``flake_occurrence`` is append-only and UNPRUNED by design, so the reader has
        to be boundable — and the only bound worth having on an ever-growing table is its
        TAIL.  A bare ``LIMIT`` on the ascending contract order would hand a dashboard
        asking "what happened lately?" the OLDEST rows, which is worse than no bound
        because it looks like an answer."""
        from orchestrator.flake_ledger import read_occurrences

        db_path = tmp_path / 'runs.db'
        self._seed(db_path)

        assert [(r.observed_at, r.test_id) for r in read_occurrences(db_path, limit=2)] == [
            (self.T_MID, self.TEST_A),  # the later of the two T_MID rows (higher id)
            (self.T_LATE, self.TEST_B),
        ]

    def test_limit_larger_than_the_table_returns_everything(self, tmp_path: Path) -> None:
        from orchestrator.flake_ledger import read_occurrences

        db_path = tmp_path / 'runs.db'
        self._seed(db_path)

        assert read_occurrences(db_path, limit=99) == read_occurrences(db_path)

    def test_limit_composes_with_the_other_filters(self, tmp_path: Path) -> None:
        from orchestrator.flake_ledger import read_occurrences

        db_path = tmp_path / 'runs.db'
        self._seed(db_path)

        rows = read_occurrences(db_path, test_id=self.TEST_B, since=self.T_MID, limit=1)
        assert [(r.observed_at, r.test_id) for r in rows] == [(self.T_LATE, self.TEST_B)]

    @pytest.mark.parametrize('limit', [0, -1], ids=['zero', 'negative'])
    def test_a_non_positive_limit_yields_nothing(self, tmp_path: Path, limit: int) -> None:
        """SQLite reads a NEGATIVE ``LIMIT`` as "no limit at all" — the exact silent
        opposite of the request, on the one read path whose point is to stay bounded.
        Clamped, so ``limit=-1`` returns nothing rather than the whole table."""
        from orchestrator.flake_ledger import read_occurrences

        db_path = tmp_path / 'runs.db'
        self._seed(db_path)

        assert read_occurrences(db_path, limit=limit) == []

    def test_missing_db_returns_empty(self, tmp_path: Path) -> None:
        from orchestrator.flake_ledger import read_occurrences

        assert read_occurrences(tmp_path / 'absent' / 'runs.db') == []


class _FakeTaskClient:
    """Records every ``submit_task`` call, so a test can assert that α files NOTHING.

    Task ζ is what makes ``open_debt`` file a de-flake task; keeping the seam here and
    asserting it stays unused makes α's scope boundary machine-checked rather than
    trusted.
    """

    def __init__(self) -> None:
        self.submit_calls: list[dict] = []

    async def submit_task(self, arguments: dict) -> str:
        self.submit_calls.append(arguments)
        return 'ticket-fake-123'


@pytest.mark.asyncio
class TestOpenDebt:
    """``open_debt`` opens/advances the single ``flake_debt`` row for a test.

    ASYNC-ONLY CLASS — a sync ``def test_`` here is a collection ERROR (see the module
    docstring).  ``open_debt`` is declared async in α even though its body is pure sync
    SQLite, so ζ can add task filing without changing the function colour and churning
    every ``await`` at ε's merge-path call sites.
    """

    NOW = datetime(2026, 8, 6, 12, 0, tzinfo=UTC)
    LATER = datetime(2026, 8, 6, 13, 0, tzinfo=UTC)
    TEST_ID = 'tests/test_a.py::test_one'

    async def test_fresh_open_writes_one_row(self, tmp_path: Path) -> None:
        from orchestrator.flake_ledger import open_debt

        db_path = tmp_path / 'runs.db'
        row = await open_debt(db_path, 'dark_factory', self.TEST_ID, now=self.NOW)

        (raw,) = _rows(db_path, 'SELECT * FROM flake_debt')
        assert raw['test_id'] == self.TEST_ID
        assert raw['project_id'] == 'dark_factory'
        assert raw['opened_at'] == self.NOW.isoformat()
        assert raw['last_occurrence_at'] == self.NOW.isoformat()
        assert raw['resolved_at'] is None
        assert raw['open_count'] == 1
        # α STORES owner_task_id but never populates it — ζ files the task.
        assert raw['owner_task_id'] is None
        assert raw['prior_resolved_at'] is None
        assert raw['prior_resolving_commit'] is None

        assert row is not None
        assert row.test_id == self.TEST_ID
        assert row.project_id == 'dark_factory'
        assert row.opened_at == self.NOW.isoformat()
        assert row.last_occurrence_at == self.NOW.isoformat()
        assert row.resolved_at is None
        assert row.open_count == 1
        assert row.owner_task_id is None
        assert row.prior_resolved_at is None
        assert row.prior_resolving_commit is None

    async def test_returned_row_equals_read_debt(self, tmp_path: Path) -> None:
        from orchestrator.flake_ledger import open_debt, read_debt

        db_path = tmp_path / 'runs.db'
        row = await open_debt(db_path, 'dark_factory', self.TEST_ID, now=self.NOW)

        assert row == read_debt(db_path, self.TEST_ID)

    async def test_default_clock_is_tz_aware_utc(self, tmp_path: Path) -> None:
        """§5.4: tasks.db has no ``created_at``, so this column is θ's only clock for
        "how long has this debt been open?"."""
        from orchestrator.flake_ledger import open_debt

        db_path = tmp_path / 'runs.db'
        row = await open_debt(db_path, 'dark_factory', self.TEST_ID)

        assert row is not None
        assert row.opened_at
        parsed = datetime.fromisoformat(row.opened_at)
        assert parsed.tzinfo is not None
        assert parsed.utcoffset() == timedelta(0)

    async def test_reopen_while_still_open_is_not_a_re_entry(self, tmp_path: Path) -> None:
        """The debt never closed, so this is the SAME cycle: ``open_count`` and
        ``opened_at`` hold, and only ``last_occurrence_at`` advances.  Counting it as a
        re-entry would make η's recurrence trigger fire on an ordinary repeat."""
        from orchestrator.flake_ledger import open_debt

        db_path = tmp_path / 'runs.db'
        await open_debt(db_path, 'dark_factory', self.TEST_ID, now=self.NOW)
        row = await open_debt(db_path, 'dark_factory', self.TEST_ID, now=self.LATER)

        (raw,) = _rows(db_path, 'SELECT * FROM flake_debt')
        assert raw['open_count'] == 1
        assert raw['opened_at'] == self.NOW.isoformat()
        assert raw['last_occurrence_at'] == self.LATER.isoformat()
        assert raw['resolved_at'] is None

        assert row is not None
        assert row.open_count == 1
        assert row.opened_at == self.NOW.isoformat()
        assert row.last_occurrence_at == self.LATER.isoformat()

    async def test_read_debt_returns_none_for_an_unknown_test(self, tmp_path: Path) -> None:
        from orchestrator.flake_ledger import open_debt, read_debt

        db_path = tmp_path / 'runs.db'
        await open_debt(db_path, 'dark_factory', self.TEST_ID, now=self.NOW)

        assert read_debt(db_path, 'never-seen') is None

    async def test_read_debt_on_a_missing_db_returns_none(self, tmp_path: Path) -> None:
        from orchestrator.flake_ledger import read_debt

        assert read_debt(tmp_path / 'absent' / 'runs.db', self.TEST_ID) is None

    async def test_alpha_accepts_task_client_but_files_nothing(self, tmp_path: Path) -> None:
        """α's scope boundary, machine-checked: the seam exists so ζ can fill it, and
        α must not file a task through it."""
        from orchestrator.flake_ledger import open_debt

        db_path = tmp_path / 'runs.db'
        client = _FakeTaskClient()
        row = await open_debt(
            db_path, 'dark_factory', self.TEST_ID, task_client=client, now=self.NOW
        )

        assert client.submit_calls == []
        assert row is not None
        assert row.owner_task_id is None

    async def test_open_debt_refuses_the_unknown_sentinel(self, tmp_path: Path, caplog) -> None:
        """``UNKNOWN_TEST_ID`` names no test, so it can own no de-flake task — and this
        is REFUSED rather than merely documented, because ε/ζ plausibly iterate the
        test_ids of a recorded occurrence batch, which is exactly where the sentinel
        lives.  An accepted ``<unknown>`` row would show up in ι's report and, once ζ
        lands, file a de-flake task against a test that does not exist."""
        import logging

        from orchestrator.flake_ledger import UNKNOWN_TEST_ID, ensure_schema, open_debt

        db_path = tmp_path / 'runs.db'
        ensure_schema(db_path)

        with caplog.at_level(logging.WARNING, logger='orchestrator.flake_ledger'):
            assert await open_debt(db_path, 'dark_factory', UNKNOWN_TEST_ID, now=self.NOW) is None

        assert _rows(db_path, 'SELECT COUNT(*) AS n FROM flake_debt')[0]['n'] == 0
        warnings = [r for r in caplog.records if r.name == 'orchestrator.flake_ledger']
        assert len(warnings) == 1
        assert UNKNOWN_TEST_ID in warnings[0].getMessage()

    async def test_the_sentinel_still_records_as_an_occurrence(self, tmp_path: Path) -> None:
        """The refusal is scoped to DEBT, not to the evidence trail: θ's class-1 health
        check is an ``unconfirmable`` RATE, so the occurrence must still be counted."""
        from orchestrator.flake_ledger import (
            UNKNOWN_TEST_ID,
            FlakeVerdict,
            list_open_debt,
            read_occurrences,
            record_flake_occurrence,
        )

        db_path = tmp_path / 'runs.db'
        record_flake_occurrence(
            db_path,
            'dark_factory',
            _suppression(
                verdict=FlakeVerdict.unconfirmable,
                test_ids=(),
                unconfirmable_reason='node-ids mapped to no discovered subproject',
            ),
            merge_sha=None,
            task_id=None,
        )

        assert [r.test_id for r in read_occurrences(db_path)] == [UNKNOWN_TEST_ID]
        assert list_open_debt(db_path) == []


@pytest.mark.asyncio
class TestResolveDebt:
    """``resolve_debt`` closes the current cycle.  ASYNC-ONLY CLASS."""

    NOW = datetime(2026, 8, 6, 12, 0, tzinfo=UTC)
    LATER = datetime(2026, 8, 6, 13, 0, tzinfo=UTC)
    LATEST = datetime(2026, 8, 6, 14, 0, tzinfo=UTC)
    TEST_ID = 'tests/test_a.py::test_one'

    async def test_stamps_resolved_at_and_the_resolving_commit(self, tmp_path: Path) -> None:
        from orchestrator.flake_ledger import open_debt, resolve_debt

        db_path = tmp_path / 'runs.db'
        await open_debt(db_path, 'dark_factory', self.TEST_ID, now=self.NOW)
        assert (
            await resolve_debt(
                db_path,
                'dark_factory',
                self.TEST_ID,
                resolving_commit='deadbee',
                now=self.LATER,
            )
            is None
        )

        (raw,) = _rows(db_path, 'SELECT * FROM flake_debt')
        assert raw['resolved_at'] == self.LATER.isoformat()
        assert raw['prior_resolving_commit'] == 'deadbee'
        # The cycle's own clock is untouched — resolution ends a cycle, it does not
        # start one.
        assert raw['opened_at'] == self.NOW.isoformat()
        assert raw['open_count'] == 1

    async def test_the_resolved_row_is_retained(self, tmp_path: Path) -> None:
        """§5.2: resolved rows are kept DELIBERATELY because η's recurrence trigger
        reads them.  Deleting one here would silently disarm class 2(a)."""
        from orchestrator.flake_ledger import open_debt, read_debt, resolve_debt

        db_path = tmp_path / 'runs.db'
        await open_debt(db_path, 'dark_factory', self.TEST_ID, now=self.NOW)
        await resolve_debt(
            db_path, 'dark_factory', self.TEST_ID, resolving_commit='deadbee', now=self.LATER
        )

        assert _rows(db_path, 'SELECT COUNT(*) AS n FROM flake_debt')[0]['n'] == 1
        row = read_debt(db_path, self.TEST_ID)
        assert row is not None
        assert row.resolved_at == self.LATER.isoformat()

    async def test_resolution_removes_the_row_from_the_open_set(self, tmp_path: Path) -> None:
        """Resolution's observable effect: it disappears from ``list_open_debt`` while
        ``read_debt`` still finds it."""
        from orchestrator.flake_ledger import list_open_debt, open_debt, resolve_debt

        db_path = tmp_path / 'runs.db'
        await open_debt(db_path, 'dark_factory', self.TEST_ID, now=self.NOW)
        assert [r.test_id for r in list_open_debt(db_path)] == [self.TEST_ID]

        await resolve_debt(
            db_path, 'dark_factory', self.TEST_ID, resolving_commit='deadbee', now=self.LATER
        )
        assert list_open_debt(db_path) == []

    async def test_list_open_debt_is_deterministically_ordered(self, tmp_path: Path) -> None:
        """ι prints this list, so its order is a contract."""
        from orchestrator.flake_ledger import list_open_debt, open_debt

        db_path = tmp_path / 'runs.db'
        # Opened out of order; two share an opened_at so the test_id tiebreak shows.
        await open_debt(db_path, 'dark_factory', 'c::t', now=self.LATER)
        await open_debt(db_path, 'dark_factory', 'b::t', now=self.NOW)
        await open_debt(db_path, 'dark_factory', 'a::t', now=self.NOW)

        assert [r.test_id for r in list_open_debt(db_path)] == ['a::t', 'b::t', 'c::t']

    async def test_list_open_debt_on_a_missing_db_is_empty(self, tmp_path: Path) -> None:
        from orchestrator.flake_ledger import list_open_debt

        assert list_open_debt(tmp_path / 'absent' / 'runs.db') == []

    async def test_resolving_an_unknown_test_is_a_safe_no_op(self, tmp_path: Path) -> None:
        """A zero-rowcount UPDATE is legitimate, not an error: ζ's caller may resolve a
        test whose debt was never opened."""
        from orchestrator.flake_ledger import list_open_debt, resolve_debt

        db_path = tmp_path / 'runs.db'
        assert (
            await resolve_debt(
                db_path, 'dark_factory', 'never-seen', resolving_commit='deadbee', now=self.NOW
            )
            is None
        )

        assert _rows(db_path, 'SELECT COUNT(*) AS n FROM flake_debt')[0]['n'] == 0
        assert list_open_debt(db_path) == []

    async def test_resolving_twice_keeps_the_first_resolution(self, tmp_path: Path) -> None:
        """Genuinely idempotent, not last-write-wins.  A replayed "owning task went
        terminal" event must not walk ``resolved_at`` FORWARD or overwrite
        ``prior_resolving_commit`` on an already-closed cycle: those two fields are
        carried into the next re-open and cited verbatim in η's
        ``regressed_after_resolution`` L2, so last-write-wins would make them describe a
        phantom resolution that happened after the fact."""
        from orchestrator.flake_ledger import open_debt, resolve_debt

        db_path = tmp_path / 'runs.db'
        await open_debt(db_path, 'dark_factory', self.TEST_ID, now=self.NOW)
        await resolve_debt(
            db_path, 'dark_factory', self.TEST_ID, resolving_commit='deadbee', now=self.LATER
        )
        await resolve_debt(
            db_path, 'dark_factory', self.TEST_ID, resolving_commit='c0ffee', now=self.LATEST
        )

        (raw,) = _rows(db_path, 'SELECT * FROM flake_debt')
        assert raw['resolved_at'] == self.LATER.isoformat()
        assert raw['prior_resolving_commit'] == 'deadbee'
        # Resolving does not re-enter: only `open_debt` opens a cycle.
        assert raw['open_count'] == 1

    async def test_the_guard_is_per_cycle_not_permanent(self, tmp_path: Path) -> None:
        """Idempotence must not wedge the row shut: ``open_debt`` sets ``resolved_at``
        back to NULL on re-entry, so the NEXT cycle resolves normally.  Without this the
        first fix would be the only one the ledger could ever record."""
        from orchestrator.flake_ledger import open_debt, resolve_debt

        db_path = tmp_path / 'runs.db'
        await open_debt(db_path, 'dark_factory', self.TEST_ID, now=self.NOW)
        await resolve_debt(
            db_path, 'dark_factory', self.TEST_ID, resolving_commit='deadbee', now=self.LATER
        )
        await open_debt(db_path, 'dark_factory', self.TEST_ID, now=self.LATEST)
        await resolve_debt(
            db_path,
            'dark_factory',
            self.TEST_ID,
            resolving_commit='c0ffee',
            now=datetime(2026, 8, 6, 15, 0, tzinfo=UTC),
        )

        (raw,) = _rows(db_path, 'SELECT * FROM flake_debt')
        assert raw['resolved_at'] == datetime(2026, 8, 6, 15, 0, tzinfo=UTC).isoformat()
        assert raw['prior_resolving_commit'] == 'c0ffee'
        assert raw['open_count'] == 2


@pytest.mark.asyncio
class TestDebtReEntry:
    """α's single-row invariant: re-entry after a resolution UPDATEs the one row rather
    than inserting a second.  ASYNC-ONLY CLASS.

    This is what makes η's recurrence trigger a primary-key lookup rather than a scan,
    and what supplies its evidence (``prior_resolved_at``, ``prior_resolving_commit``).
    """

    T0 = datetime(2026, 8, 6, 10, 0, tzinfo=UTC)
    T1 = datetime(2026, 8, 6, 11, 0, tzinfo=UTC)
    T2 = datetime(2026, 8, 6, 12, 0, tzinfo=UTC)
    T3 = datetime(2026, 8, 6, 13, 0, tzinfo=UTC)
    T4 = datetime(2026, 8, 6, 14, 0, tzinfo=UTC)
    TEST_ID = 'tests/test_a.py::test_one'

    async def _cycle(self, db_path: Path, resolved_at, commit, reopened_at):
        from orchestrator.flake_ledger import open_debt, resolve_debt

        await resolve_debt(
            db_path, 'dark_factory', self.TEST_ID, resolving_commit=commit, now=resolved_at
        )
        return await open_debt(db_path, 'dark_factory', self.TEST_ID, now=reopened_at)

    async def test_second_cycle_updates_the_single_row(self, tmp_path: Path) -> None:
        from orchestrator.flake_ledger import open_debt

        db_path = tmp_path / 'runs.db'
        await open_debt(db_path, 'dark_factory', self.TEST_ID, now=self.T0)
        row = await self._cycle(db_path, self.T1, 'c0ffee', self.T2)

        assert _rows(db_path, 'SELECT COUNT(*) AS n FROM flake_debt')[0]['n'] == 1
        (raw,) = _rows(db_path, 'SELECT * FROM flake_debt')
        assert raw['open_count'] == 2
        assert raw['resolved_at'] is None  # open again
        # η's recurrence evidence: the previous cycle's close, carried forward.
        assert raw['prior_resolved_at'] == self.T1.isoformat()
        # Preserved UNTOUCHED across the re-open — §5.6 class 2(a) cites it VERBATIM
        # in the regressed_after_resolution L2.
        assert raw['prior_resolving_commit'] == 'c0ffee'
        # The NEW cycle's clock, so θ's age backstop measures the current cycle rather
        # than the original one.
        assert raw['opened_at'] == self.T2.isoformat()
        assert raw['last_occurrence_at'] == self.T2.isoformat()
        assert raw['test_id'] == self.TEST_ID
        assert raw['project_id'] == 'dark_factory'

        assert row is not None
        assert row.open_count == 2
        assert row.resolved_at is None
        assert row.prior_resolved_at == self.T1.isoformat()
        assert row.prior_resolving_commit == 'c0ffee'
        assert row.opened_at == self.T2.isoformat()

    async def test_third_cycle_tracks_the_most_recent_closed_cycle(self, tmp_path: Path) -> None:
        """The prior-cycle fields always describe the MOST RECENT closed cycle, not the
        first one."""
        from orchestrator.flake_ledger import open_debt

        db_path = tmp_path / 'runs.db'
        await open_debt(db_path, 'dark_factory', self.TEST_ID, now=self.T0)
        await self._cycle(db_path, self.T1, 'c0ffee', self.T2)
        row = await self._cycle(db_path, self.T3, 'f00d', self.T4)

        assert _rows(db_path, 'SELECT COUNT(*) AS n FROM flake_debt')[0]['n'] == 1
        (raw,) = _rows(db_path, 'SELECT * FROM flake_debt')
        assert raw['open_count'] == 3
        assert raw['resolved_at'] is None
        assert raw['prior_resolved_at'] == self.T3.isoformat()
        assert raw['prior_resolving_commit'] == 'f00d'
        assert raw['opened_at'] == self.T4.isoformat()

        assert row is not None
        assert row.open_count == 3
        assert row.prior_resolved_at == self.T3.isoformat()
        assert row.prior_resolving_commit == 'f00d'
        assert row.opened_at == self.T4.isoformat()


def _blocked_path(tmp_path: Path) -> Path:
    """A db_path whose parent DIRECTORY cannot be created — a FILE sits where the
    directory must go, so ``mkdir(parents=True)`` raises.  Portable: no chmod, no root
    dependency, works the same for an unprivileged CI user."""
    (tmp_path / 'blocker').write_text('x')
    return tmp_path / 'blocker' / 'runs.db'


def _corrupt_path(tmp_path: Path) -> Path:
    """A db_path that exists but is not a SQLite database — the sqlite3.DatabaseError
    path, so the contract is not tied to one exception type."""
    db_path = tmp_path / 'runs.db'
    db_path.write_bytes(b'not a database')
    return db_path


_FAULTS = [_blocked_path, _corrupt_path]


def _assert_logged_loudly(caplog) -> None:
    """B12's second half: the failure is LOUD, not a silent fail-soft.  ``exc_info`` is
    required — a bare message without the traceback is not a diagnosable report."""
    records = [r for r in caplog.records if r.name == 'orchestrator.flake_ledger']
    assert records, 'expected at least one warning on orchestrator.flake_ledger'
    assert any(r.levelno == logging.WARNING for r in records)
    assert all(r.exc_info is not None for r in records)


@pytest.mark.parametrize('make_path', _FAULTS, ids=['blocked_dir', 'corrupt_file'])
class TestNeverRaisesSync:
    """Boundary row B12, applied uniformly to every SYNC entry point: a ledger failure
    must never fail a verify or a merge.

    The merge path has no ``VerifyInfraError`` handler, so an uncaught raise here does
    not degrade one merge — it stalls the queue.  Each entry point must return its
    honest-degrade value AND log loudly.
    """

    def test_ensure_schema(self, tmp_path: Path, caplog, make_path) -> None:
        from orchestrator.flake_ledger import ensure_schema

        with caplog.at_level(logging.WARNING, logger='orchestrator.flake_ledger'):
            assert ensure_schema(make_path(tmp_path)) is None
        _assert_logged_loudly(caplog)

    def test_record_flake_occurrence(self, tmp_path: Path, caplog, make_path) -> None:
        from orchestrator.flake_ledger import record_flake_occurrence

        with caplog.at_level(logging.WARNING, logger='orchestrator.flake_ledger'):
            assert (
                record_flake_occurrence(
                    make_path(tmp_path),
                    'dark_factory',
                    _suppression(),
                    merge_sha=None,
                    task_id=None,
                )
                is None
            )
        _assert_logged_loudly(caplog)

    def test_read_occurrences(self, tmp_path: Path, caplog, make_path) -> None:
        from orchestrator.flake_ledger import read_occurrences

        with caplog.at_level(logging.WARNING, logger='orchestrator.flake_ledger'):
            assert read_occurrences(make_path(tmp_path)) == []
        _assert_logged_loudly(caplog)

    def test_read_debt(self, tmp_path: Path, caplog, make_path) -> None:
        from orchestrator.flake_ledger import read_debt

        with caplog.at_level(logging.WARNING, logger='orchestrator.flake_ledger'):
            assert read_debt(make_path(tmp_path), 'tests/test_a.py::test_one') is None
        _assert_logged_loudly(caplog)

    def test_list_open_debt(self, tmp_path: Path, caplog, make_path) -> None:
        from orchestrator.flake_ledger import list_open_debt

        with caplog.at_level(logging.WARNING, logger='orchestrator.flake_ledger'):
            assert list_open_debt(make_path(tmp_path)) == []
        _assert_logged_loudly(caplog)


@pytest.mark.asyncio
@pytest.mark.parametrize('make_path', _FAULTS, ids=['blocked_dir', 'corrupt_file'])
class TestNeverRaisesAsync:
    """Boundary row B12 for the ASYNC entry points.  ASYNC-ONLY CLASS."""

    async def test_open_debt(self, tmp_path: Path, caplog, make_path) -> None:
        """``open_debt`` degrading to ``None`` is precisely why α returns
        ``DebtRow | None`` rather than §8.3's literal ``DebtRow``: it cannot both
        refuse to raise and produce a row when the DB is unwritable."""
        from orchestrator.flake_ledger import open_debt

        with caplog.at_level(logging.WARNING, logger='orchestrator.flake_ledger'):
            assert await open_debt(make_path(tmp_path), 'dark_factory', 'a::t') is None
        _assert_logged_loudly(caplog)

    async def test_resolve_debt(self, tmp_path: Path, caplog, make_path) -> None:
        from orchestrator.flake_ledger import resolve_debt

        with caplog.at_level(logging.WARNING, logger='orchestrator.flake_ledger'):
            assert (
                await resolve_debt(
                    make_path(tmp_path), 'dark_factory', 'a::t', resolving_commit='deadbee'
                )
                is None
            )
        _assert_logged_loudly(caplog)


@pytest.mark.asyncio
class TestOneConnectionPerCall:
    """Every entry point opens exactly ONE sqlite3 connection.  ASYNC-ONLY CLASS.

    Not a micro-optimisation: each connection pays the full five-pragma durability triad,
    including a ``journal_mode=WAL`` switch and ``synchronous=FULL``, and these run on the
    merge path.  The schema DDL is still issued on every call (deliberately un-memoized —
    a per-path cache would go stale the moment the DB file is replaced), it just rides the
    connection the statement was going to open anyway.

    ``open_debt`` is the one that motivated this: it used to open FOUR — its own
    ``ensure_schema`` plus upsert, then ``read_debt``'s ``ensure_schema`` plus SELECT.
    """

    NOW = datetime(2026, 8, 6, 12, 0, tzinfo=UTC)
    TEST_ID = 'tests/test_a.py::test_one'

    @staticmethod
    def _count_connections(monkeypatch) -> list:
        opened: list = []
        real_connect = sqlite3.connect

        def _recording_connect(*args, **kwargs):
            conn = real_connect(*args, **kwargs)
            opened.append(conn)
            return conn

        monkeypatch.setattr(sqlite3, 'connect', _recording_connect)
        return opened

    async def test_open_debt_opens_one_connection(self, tmp_path: Path, monkeypatch) -> None:
        from orchestrator.flake_ledger import open_debt

        db_path = tmp_path / 'runs.db'
        opened = self._count_connections(monkeypatch)
        row = await open_debt(db_path, 'dark_factory', self.TEST_ID, now=self.NOW)

        assert len(opened) == 1, f'expected 1 connection, got {len(opened)}'
        # ...and it still returns the row, so the saving is not a dropped read-back.
        assert row is not None
        assert row.test_id == self.TEST_ID

    async def test_resolve_debt_opens_one_connection(self, tmp_path: Path, monkeypatch) -> None:
        from orchestrator.flake_ledger import open_debt, resolve_debt

        db_path = tmp_path / 'runs.db'
        await open_debt(db_path, 'dark_factory', self.TEST_ID, now=self.NOW)

        opened = self._count_connections(monkeypatch)
        await resolve_debt(
            db_path, 'dark_factory', self.TEST_ID, resolving_commit='deadbee', now=self.NOW
        )

        assert len(opened) == 1, f'expected 1 connection, got {len(opened)}'

    async def test_sync_entry_points_open_one_connection(self, tmp_path: Path, monkeypatch) -> None:
        from orchestrator.flake_ledger import (
            list_open_debt,
            read_debt,
            read_occurrences,
            record_flake_occurrence,
        )

        db_path = tmp_path / 'runs.db'
        opened = self._count_connections(monkeypatch)

        for call in (
            lambda: record_flake_occurrence(
                db_path, 'dark_factory', _suppression(), merge_sha=None, task_id=None
            ),
            lambda: read_occurrences(db_path),
            lambda: read_debt(db_path, self.TEST_ID),
            lambda: list_open_debt(db_path),
        ):
            opened.clear()
            call()
            assert len(opened) == 1, f'expected 1 connection, got {len(opened)}'


class TestConnectClosesOnPragmaFailure:
    """``_connect`` must not leak its sqlite3 connection when the pragma call raises.

    ``sqlite3.connect()`` succeeds LAZILY even against a non-DB file, so the first real
    statement — the ``PRAGMA journal_mode=WAL`` inside
    ``apply_full_durability_pragmas_sync`` — is where the failure surfaces, by which
    point ``conn`` already owns an open file descriptor.

    This matters MORE here than in the sibling stores: event_store.py:509 and
    run_store.py:87 share the same connect-then-pragma idiom but PROPAGATE the raise, so
    the process is dying anyway.  B12 makes this module deliberately SWALLOW the
    exception and return an honest-degrade value, so a long-lived orchestrator keeps
    calling on every merge — turning a one-shot leak into an unbounded one.

    Two layered assertions: the invariant itself (deterministic and portable), and the
    real-resource regression that the leak was actually measured as.
    """

    def test_connection_is_closed_when_the_pragma_call_raises(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """The invariant, immune to GC timing and to procfs availability."""
        from orchestrator import flake_ledger

        created: list[sqlite3.Connection] = []
        real_connect = sqlite3.connect

        def _recording_connect(*args, **kwargs):
            conn = real_connect(*args, **kwargs)
            created.append(conn)
            return conn

        def _boom(conn, **kwargs):
            raise RuntimeError('wal unsupported')

        # flake_ledger.py does `from shared.sqlite_sync_base import
        # apply_full_durability_pragmas_sync`, so the name is REBOUND in the
        # flake_ledger module namespace — that is the correct patch target.  Patching
        # `shared.sqlite_sync_base.apply_full_durability_pragmas_sync` would NOT take
        # effect, because the from-import already captured the original object.
        monkeypatch.setattr(flake_ledger, 'apply_full_durability_pragmas_sync', _boom)
        monkeypatch.setattr(sqlite3, 'connect', _recording_connect)

        with pytest.raises(RuntimeError, match='wal unsupported'):
            flake_ledger._connect(tmp_path / 'runs.db')

        assert len(created) == 1, 'expected exactly one sqlite3 connection to be opened'
        # 'Cannot operate on a closed database' is how a closed connection reports.
        with pytest.raises(sqlite3.ProgrammingError, match='closed database'):
            created[0].execute('SELECT 1')

    @pytest.mark.skipif(not Path('/proc/self/fd').exists(), reason='needs procfs')
    def test_repeated_calls_on_a_corrupt_db_leak_no_file_descriptors(self, tmp_path: Path) -> None:
        """The regression as measured: repeated B12-swallowed failures must not grow fds.

        ``gc.disable()`` IS LOAD-BEARING — do not "simplify" it away.  The leaked
        connection is held by a reference CYCLE (exception → traceback → frame → local
        ``conn``), so it is reclaimed by the generational collector, not by refcounting.
        With the cyclic GC left enabled the fd count moves NON-MONOTONICALLY (measured
        +100, then −54, then +65 across successive batches), which would make a plain
        "loop N times and assert the count does not grow" test flaky in BOTH directions:
        failing on fixed code when a batch straddles no collection, and passing on
        broken code when a collection happens to fire inside the window.

        Disabling the collector for the measurement window turns it into an exact
        signal: measured at HEAD 5430cc0c8b the delta was EXACTLY +60 for 30 calls
        (2 fds per call — back when ``ensure_schema`` opened its own connection beside
        the statement's), reproduced identically across 3 independent trials; against the
        fix it is EXACTLY 0.  No tuned tolerance.  Since the ``_open`` refactor an entry
        point opens ONE connection, so an unfixed ``_connect`` would leak 30 here rather
        than 60 — the RED signal is smaller but the assertion is unchanged, because the
        bound is zero, not a threshold.
        """
        from orchestrator.flake_ledger import read_occurrences

        db_path = _corrupt_path(tmp_path)
        # Warm imports and any lazy module state so the window measures per-call cost.
        assert read_occurrences(db_path) == []

        gc.collect()
        gc.disable()
        try:
            before = len(os.listdir('/proc/self/fd'))
            for _ in range(30):
                read_occurrences(db_path)
            after = len(os.listdir('/proc/self/fd'))
        finally:
            gc.enable()

        assert after - before == 0, (
            f'leaked {after - before} file descriptors across 30 calls '
            f'(before={before}, after={after})'
        )
