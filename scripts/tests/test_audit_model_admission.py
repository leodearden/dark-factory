"""Tests for scripts/audit_model_admission.py.

Hermetic: every test runs against a synthetic runs.db materialised into
tmp_path, never against the 181 MB live store the script defaults to.

The schema below is a VERBATIM copy of the three tables the audit reads,
captured with

    sqlite3 data/orchestrator/runs.db ".schema events invocations account_events"

Copied rather than imported because scripts/tests/ is collected by
`uv run --project shared pytest` and imports NO first-party package
(dark-factory-orchestrator.yaml:111-112) — so the orchestrator's own event
store, which owns this DDL, is out of reach here. Re-capture with that command
rather than hand-editing if the writer's schema moves.
"""
import json
import sqlite3
from datetime import UTC, datetime, timedelta

import audit_model_admission
import pytest

RUNS_DB_SCHEMA = """
CREATE TABLE events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    run_id      TEXT    NOT NULL,
    task_id     TEXT,
    event_type  TEXT    NOT NULL,
    phase       TEXT,
    role        TEXT,
    data        TEXT    DEFAULT '{}',
    cost_usd    REAL,
    duration_ms INTEGER
);
CREATE TABLE invocations (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              TEXT NOT NULL,
    task_id             TEXT,
    project_id          TEXT NOT NULL,
    account_name        TEXT NOT NULL,
    model               TEXT NOT NULL,
    role                TEXT NOT NULL,
    cost_usd            REAL NOT NULL DEFAULT 0.0,
    input_tokens        INTEGER,
    output_tokens       INTEGER,
    cache_read_tokens   INTEGER,
    cache_create_tokens INTEGER,
    duration_ms         INTEGER NOT NULL DEFAULT 0,
    capped              INTEGER NOT NULL DEFAULT 0,
    started_at          TEXT NOT NULL,
    completed_at        TEXT NOT NULL
);
CREATE TABLE account_events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    account_name TEXT NOT NULL,
    event_type   TEXT NOT NULL,
    project_id   TEXT,
    run_id       TEXT,
    details      TEXT,
    created_at   TEXT NOT NULL
);
"""


@pytest.fixture
def runs_db_path(tmp_path):
    """Path to a fresh, empty runs.db carrying :data:`RUNS_DB_SCHEMA`."""
    path = tmp_path / 'runs.db'
    conn = sqlite3.connect(path)
    try:
        conn.executescript(RUNS_DB_SCHEMA)
        conn.commit()
    finally:
        conn.close()
    return path


@pytest.fixture
def runs_db(runs_db_path):
    """A WRITABLE connection on :func:`runs_db_path`, for seeding scenarios.

    The audit's scan functions take an open connection, so a test normally
    seeds through this fixture and hands the same connection straight to the
    function under test. Tests that exercise the read-only connection factory
    or the CLI take ``runs_db_path`` instead — both name the same file.
    """
    conn = sqlite3.connect(runs_db_path)
    try:
        yield conn
    finally:
        conn.close()


def _payload(value):
    """JSON-encode a dict/list payload; pass a str or None through VERBATIM.

    The pass-through is what lets a test seed a deliberately malformed payload
    — the live store holds an ``account_events.details`` of the bare string
    ``'Escalation watcher (auto)'`` — so the audit's tolerant-parse paths are
    exercised against the real shape rather than a hypothetical one. Same
    convention as ``make_tasks_db``'s ``metadata`` handling in conftest.py.
    """
    if value is None or isinstance(value, str):
        return value
    return json.dumps(value)


def _event(
    conn,
    timestamp,
    event_type,
    *,
    run_id='run-1',
    task_id=None,
    phase=None,
    role=None,
    data=None,
    cost_usd=None,
    duration_ms=None,
):
    """Insert one `events` row, stating its payload as a dict rather than JSON text."""
    conn.execute(
        'INSERT INTO events (timestamp, run_id, task_id, event_type, phase, role, '
        'data, cost_usd, duration_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
        (
            timestamp,
            run_id,
            task_id,
            event_type,
            phase,
            role,
            _payload({} if data is None else data),
            cost_usd,
            duration_ms,
        ),
    )
    conn.commit()


def _invocation(
    conn,
    *,
    model,
    role,
    started_at,
    completed_at,
    run_id='run-1',
    task_id=None,
    project_id='dark_factory',
    account_name='max-a',
    cost_usd=0.0,
    input_tokens=None,
    output_tokens=None,
    cache_read_tokens=None,
    cache_create_tokens=None,
    duration_ms=0,
    capped=0,
):
    """Insert one `invocations` row."""
    conn.execute(
        'INSERT INTO invocations (run_id, task_id, project_id, account_name, model, role, '
        'cost_usd, input_tokens, output_tokens, cache_read_tokens, cache_create_tokens, '
        'duration_ms, capped, started_at, completed_at) '
        'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
        (
            run_id,
            task_id,
            project_id,
            account_name,
            model,
            role,
            cost_usd,
            input_tokens,
            output_tokens,
            cache_read_tokens,
            cache_create_tokens,
            duration_ms,
            capped,
            started_at,
            completed_at,
        ),
    )
    conn.commit()


def _account_event(
    conn,
    *,
    account_name,
    event_type,
    created_at,
    details=None,
    project_id='dark_factory',
    run_id='run-1',
):
    """Insert one `account_events` row; *details* follows :func:`_payload`."""
    conn.execute(
        'INSERT INTO account_events (account_name, event_type, project_id, run_id, '
        'details, created_at) VALUES (?, ?, ?, ?, ?, ?)',
        (account_name, event_type, project_id, run_id, _payload(details), created_at),
    )
    conn.commit()


# --- scan_routing_decisions: the resolver-decision surface (checks 1 and 3) ---

FABLE = 'claude-fable-5-1'
APPLY = datetime(2026, 9, 12, 6, 43, 16, tzinfo=UTC)


def _at(**offset):
    """An ISO-8601 timestamp *offset* from the D6 apply time, spelled as the store does."""
    return (APPLY + timedelta(**offset)).isoformat()


def _routing_payload(
    *,
    role,
    model,
    source_layer='config',
    rule_id=None,
    rejected=(),
    routing_tier=0,
):
    """The exact 11-key `routing_decision` payload the producer emits.

    Key set taken from orchestrator/src/orchestrator/workflow.py::
    _record_routing_decision and corroborated against a live row, so a
    producer-side key rename shows up here as a failing test rather than as a
    silently empty audit section.
    """
    return {
        'role': role,
        'model': model,
        'effort': 'max',
        'budget_usd': 8.0,
        'max_turns': 100,
        'source_layer': source_layer,
        'rule_id': rule_id,
        'rejected': list(rejected),
        'routing_tier': routing_tier,
        'decided_at': _at(),
        'inputs_digest': 'a86343567ef2a2d8',
    }


def test_config_layer_selection_is_reported_with_its_source_layer(runs_db):
    _event(
        runs_db, _at(hours=2), 'routing_decision', task_id='4377', role='merger',
        data=_routing_payload(role='merger', model=FABLE),
    )

    scan = audit_model_admission.scan_routing_decisions(runs_db, model=FABLE, since=APPLY)

    assert len(scan.selections) == 1
    selected = scan.selections[0]
    assert selected.role == 'merger'
    assert selected.task_id == '4377'
    assert selected.source_layer == 'config'
    assert selected.rule_id is None
    assert selected.routing_tier == 0
    assert scan.rejections == ()


def test_policy_rule_selection_carries_the_rule_id_and_routing_tier(runs_db):
    """Check 3: "did rule steward-retry-fable match?" is answerable only from these two."""
    _event(
        runs_db, _at(days=1), 'routing_decision', task_id='4211', role='steward',
        data=_routing_payload(
            role='steward', model=FABLE, source_layer='policy_rule',
            rule_id='steward-retry-fable', routing_tier=1,
        ),
    )

    scan = audit_model_admission.scan_routing_decisions(runs_db, model=FABLE, since=APPLY)

    assert [(s.role, s.rule_id, s.routing_tier) for s in scan.selections] == [
        ('steward', 'steward-retry-fable', 1)
    ]


def test_a_rejection_on_a_decision_that_resolved_elsewhere_is_still_reported(runs_db):
    """A rejection is recorded on a decision that resolved to a DIFFERENT model.

    Filtering to rows where the target was selected would therefore miss every
    rejection there is — which is the whole of check 1.
    """
    _event(
        runs_db, _at(hours=3), 'routing_decision', task_id='4500', role='implementer',
        data=_routing_payload(
            role='implementer', model='opus', source_layer='role_default',
            rejected=['config:model-not-in-allowlist'],
        ),
    )

    scan = audit_model_admission.scan_routing_decisions(runs_db, model=FABLE, since=APPLY)

    assert scan.selections == ()
    assert len(scan.rejections) == 1
    rejection = scan.rejections[0]
    assert rejection.role == 'implementer'
    assert rejection.resolved_model == 'opus'
    assert rejection.reasons == ('config:model-not-in-allowlist',)


@pytest.mark.parametrize(
    'entry',
    [
        'config:model-not-in-allowlist',
        'policy_rule:model-ceiling-exhausted',
        'metadata_override:model-capacity-exhausted',
    ],
)
def test_each_known_rejection_reason_is_recognised_behind_its_layer_prefix(runs_db, entry):
    _event(
        runs_db, _at(hours=4), 'routing_decision', task_id='4501', role='merger',
        data=_routing_payload(role='merger', model='opus', rejected=[entry]),
    )

    scan = audit_model_admission.scan_routing_decisions(runs_db, model=FABLE, since=APPLY)

    assert [r.reasons for r in scan.rejections] == [(entry,)]


def test_model_not_in_ladder_is_not_a_model_rejection(runs_db):
    """`policy_rule:model-not-in-ladder` means no candidate was ever formed.

    routing.py::resolve_route appends it when a '+N' spec cannot be resolved
    against the ladder, BEFORE any model is validated — so it is not evidence
    that a model was attempted and refused.
    """
    _event(
        runs_db, _at(hours=5), 'routing_decision', task_id='4502', role='debugger',
        data=_routing_payload(
            role='debugger', model='opus', rejected=['policy_rule:model-not-in-ladder'],
        ),
    )

    scan = audit_model_admission.scan_routing_decisions(runs_db, model=FABLE, since=APPLY)

    assert scan.rejections == ()


def test_decisions_before_since_are_excluded_from_both_halves(runs_db):
    _event(
        runs_db, _at(hours=-1), 'routing_decision', task_id='4300', role='merger',
        data=_routing_payload(role='merger', model=FABLE),
    )
    _event(
        runs_db, _at(hours=-1), 'routing_decision', task_id='4301', role='merger',
        data=_routing_payload(
            role='merger', model='opus', rejected=['config:model-not-in-allowlist'],
        ),
    )

    scan = audit_model_admission.scan_routing_decisions(runs_db, model=FABLE, since=APPLY)

    assert scan.selections == ()
    assert scan.rejections == ()


def test_an_empty_table_yields_empty_halves_rather_than_raising(runs_db):
    scan = audit_model_admission.scan_routing_decisions(runs_db, model=FABLE, since=APPLY)

    assert scan.selections == ()
    assert scan.rejections == ()
    assert scan.skipped_rows == 0


def test_a_malformed_payload_is_skipped_and_counted_rather_than_raised(runs_db):
    """A 181 MB live store must not be abortable by one bad row — but a
    dropped row has to stay visible, so it is counted rather than swallowed."""
    _event(runs_db, _at(hours=6), 'routing_decision', task_id='4503', data='not json{')
    _event(
        runs_db, _at(hours=7), 'routing_decision', task_id='4504', role='merger',
        data=_routing_payload(role='merger', model=FABLE),
    )

    scan = audit_model_admission.scan_routing_decisions(runs_db, model=FABLE, since=APPLY)

    assert scan.skipped_rows == 1
    assert [s.task_id for s in scan.selections] == ['4504']
