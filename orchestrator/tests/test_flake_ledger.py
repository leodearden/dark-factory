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

import logging
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


def _table_sql(db_path: Path, name: str) -> str:
    rows = _rows(db_path, 'SELECT sql FROM sqlite_master WHERE name = ?', (name,))
    assert rows, f'no sqlite_master entry for {name!r}'
    return rows[0]['sql'] or ''


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

        assert 'AUTOINCREMENT' in _table_sql(db_path, 'flake_occurrence').upper()

        pk_columns = [
            r['name'] for r in _rows(db_path, 'PRAGMA table_info(flake_occurrence)') if r['pk']
        ]
        assert pk_columns == ['id']

        # AUTOINCREMENT (as opposed to plain INTEGER PRIMARY KEY) materialises
        # sqlite_sequence on first insert — monotonic ids, never reused.
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
        declaratively by a UNIQUE index, not by swallowing an IntegrityError."""
        from orchestrator.flake_ledger import ensure_schema

        db_path = tmp_path / 'runs.db'
        ensure_schema(db_path)

        indexes = _rows(db_path, "SELECT name, sql FROM sqlite_master WHERE type='index'")
        dedup = [
            r
            for r in indexes
            if r['sql']
            and 'flake_occurrence' in r['sql']
            and 'test_id' in r['sql']
            and 'observed_at' in r['sql']
            and 'call_site' in r['sql']
        ]
        assert len(dedup) == 1, f'expected exactly one dedup index, got {indexes}'
        assert 'unique' in dedup[0]['sql'].lower()

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

    def test_value_equals_name(self) -> None:
        """The project convention (test_event_store.py:107-111) — what lets a member
        bind straight into SQL and travel the VerifyResult wire with no codec."""
        from orchestrator.flake_ledger import FlakeVerdict

        for v in FlakeVerdict:
            assert isinstance(v.value, str)
            assert v.value == v.name

    def test_is_str_comparable(self) -> None:
        from orchestrator.flake_ledger import FlakeVerdict

        assert FlakeVerdict.passes_in_isolation == 'passes_in_isolation'
        assert FlakeVerdict.fails_in_isolation == 'fails_in_isolation'
        assert FlakeVerdict.unconfirmable == 'unconfirmable'

    def test_vocabulary_names_the_observation_not_the_remedy(self) -> None:
        """§5.5: the verdict is ``passes_in_isolation``, never ``flaky_test: true``.
        A verdict records what was OBSERVED; calling a test "flaky" prejudges the
        remedy and is the vocabulary this PRD exists to replace."""
        from orchestrator.flake_ledger import FlakeVerdict

        assert not any('flaky' in v.value for v in FlakeVerdict)


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

    def test_field_order(self) -> None:
        """Declaration order is the contract — ε constructs these and the codec
        round-trips them."""
        import dataclasses

        from orchestrator.flake_ledger import FlakeSuppression

        assert [f.name for f in dataclasses.fields(FlakeSuppression)] == [
            'verdict',
            'test_ids',
            'observed_at',
            'call_site',
            'runner',
            'psi_cpu_some10',
            'unconfirmable_reason',
        ]

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

    async def test_resolving_twice_is_idempotent(self, tmp_path: Path) -> None:
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
        assert raw['resolved_at'] == self.LATEST.isoformat()
        assert raw['prior_resolving_commit'] == 'c0ffee'
        # Resolving does not re-enter: only `open_debt` opens a cycle.
        assert raw['open_count'] == 1


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
