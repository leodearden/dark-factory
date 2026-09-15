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
import argparse
import json
import sqlite3
import time
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


# --- scan_invocations: what actually RAN, and how it ended (check 2) ---


def _invocation_end_payload(*, turns, model, success=True, account_name='max-b'):
    """The `invocation_end` payload keys this audit reads, as the producer writes them."""
    return {
        'turns': turns,
        'success': success,
        'subtype': 'success' if success else 'error',
        'model': model,
        'account_name': account_name,
        'input_tokens': 44,
        'output_tokens': 17317,
        'cache_read_tokens': 1785372,
        'cache_create_tokens': 84011,
        'transcript_turns': turns + 18,
        'timed_out': False,
        'ended_awaiting_background': False,
    }


def _merge_finalized_payload(*, branch, state, merge_sha=None, reason=None, generation=1):
    """The `merge_finalized` payload keys this audit reads.

    NOTE the producer leaves the event row's `role` column EMPTY on these, so
    they can only be joined to an invocation by task_id.
    """
    return {
        'request_id': f'mr-{branch}',
        'branch': branch,
        'state': state,
        'snapshot_tip': None,
        'merge_sha': merge_sha,
        'superseded_by': None,
        'generation': generation,
        'reason': reason,
        'landed_via_chain': None,
    }


def _fable_merger_run(conn, *, task_id, turns=45, duration_ms=120_000, cost_usd=6.08):
    """Seed one Fable merger invocation plus its matching invocation_end."""
    _invocation(
        conn, model=FABLE, role='merger', task_id=task_id, account_name='max-b',
        cost_usd=cost_usd, duration_ms=duration_ms,
        started_at=_at(hours=2), completed_at=_at(hours=2, minutes=2),
    )
    _event(
        conn, _at(hours=2, minutes=2), 'invocation_end', task_id=task_id, role='merger',
        data=_invocation_end_payload(turns=turns, model=FABLE),
    )


def test_a_fable_merger_run_reports_its_turns_and_the_merge_it_resolved(runs_db):
    _fable_merger_run(runs_db, task_id='4377')
    _event(
        runs_db, _at(hours=2, minutes=30), 'merge_finalized', task_id='4377',
        data=_merge_finalized_payload(branch='4377', state='done', merge_sha='d411f107'),
    )

    rows = audit_model_admission.scan_invocations(runs_db, model=FABLE, since=APPLY)

    assert len(rows) == 1
    run = rows[0]
    assert run.role == 'merger'
    assert run.turns == 45
    assert run.succeeded is True
    assert run.end_event_model == FABLE
    assert run.cost_usd == 6.08
    assert run.merge_outcome is not None
    assert run.merge_outcome.state == 'done'
    assert run.merge_outcome.merge_sha == 'd411f107'


def test_the_last_merge_finalized_wins_not_the_first(runs_db):
    """A post-merge VERIFICATION failure blocks the task and is later retried to
    done. Reading the first merge_finalized would report the Fable merger as
    having failed to resolve the merge, which is a different claim entirely."""
    _fable_merger_run(runs_db, task_id='4377')
    _event(
        runs_db, _at(hours=3), 'merge_finalized', task_id='4377',
        data=_merge_finalized_payload(
            branch='4377', state='blocked',
            reason='Post-merge verification failed: pytest exited 1', generation=1,
        ),
    )
    _event(
        runs_db, _at(hours=20), 'merge_finalized', task_id='4377',
        data=_merge_finalized_payload(
            branch='4377', state='done', merge_sha='d411f107', generation=3,
        ),
    )

    rows = audit_model_admission.scan_invocations(runs_db, model=FABLE, since=APPLY)

    outcome = rows[0].merge_outcome
    assert outcome is not None
    assert outcome.state == 'done'
    assert outcome.merge_sha == 'd411f107'


def test_a_later_merger_on_another_model_does_not_lend_its_success_to_this_run(runs_db):
    """The one error this audit must never make: reporting a failure by the
    model under audit as a success.

    A merge can be attempted by more than one merger run — the audited model's
    merger leaves it blocked, a merger on a DIFFERENT model later retries it to
    done. Joined on task_id and bounded only below, the audited row would render
    `done (<sha>)`, and the report's central check-2 claim ("no blocked merge
    appears as a final outcome for any task this model's merger touched") would
    be unsupportable. The window closes at the next merger run's start.
    """
    _fable_merger_run(runs_db, task_id='4377')
    _event(
        runs_db, _at(hours=3), 'merge_finalized', task_id='4377',
        data=_merge_finalized_payload(
            branch='4377', state='blocked', reason='merge conflict', generation=1,
        ),
    )
    _invocation(  # a merger on another model picks the task up and lands it
        runs_db, model='opus', role='merger', task_id='4377',
        started_at=_at(hours=4), completed_at=_at(hours=4, minutes=30),
    )
    _event(
        runs_db, _at(hours=4, minutes=20), 'merge_finalized', task_id='4377',
        data=_merge_finalized_payload(
            branch='4377', state='done', merge_sha='d411f107', generation=2,
        ),
    )

    rows = audit_model_admission.scan_invocations(runs_db, model=FABLE, since=APPLY)

    assert len(rows) == 1  # only the audited model's run is in scope
    outcome = rows[0].merge_outcome
    assert outcome is not None
    assert outcome.state == 'blocked'
    assert outcome.merge_sha is None


def test_a_merge_finalized_before_the_run_started_is_not_attributed_to_it(runs_db):
    """The window is bounded below as well: an earlier merger run's outcome
    belongs to that run, not to the next one to touch the same task."""
    _event(
        runs_db, _at(hours=1), 'merge_finalized', task_id='4377',
        data=_merge_finalized_payload(branch='4377', state='blocked', reason='earlier'),
    )
    _fable_merger_run(runs_db, task_id='4377')  # starts at +2h

    rows = audit_model_admission.scan_invocations(runs_db, model=FABLE, since=APPLY)

    assert rows[0].merge_outcome is None


def test_an_invocation_with_no_matching_end_event_reports_turns_none(runs_db):
    _invocation(
        runs_db, model=FABLE, role='merger', task_id='4900', duration_ms=5_000,
        started_at=_at(hours=4), completed_at=_at(hours=4, minutes=1),
    )

    rows = audit_model_admission.scan_invocations(runs_db, model=FABLE, since=APPLY)

    assert len(rows) == 1
    assert rows[0].turns is None
    assert rows[0].succeeded is None
    assert rows[0].end_event_model is None


@pytest.mark.parametrize(
    ('role', 'duration_ms', 'expected'),
    [
        ('merger', 600_000, True),    # AT the limit counts as over — at-or-above
        ('merger', 599_999, False),
        ('steward', 900_000, None),   # no configured limit: unknown, not "under"
    ],
)
def test_the_flat_ceiling_flag_is_at_or_above_and_unknown_without_a_limit(
    runs_db, role, duration_ms, expected
):
    _invocation(
        runs_db, model=FABLE, role=role, task_id='4901', duration_ms=duration_ms,
        started_at=_at(hours=5), completed_at=_at(hours=5, minutes=10),
    )

    rows = audit_model_admission.scan_invocations(
        runs_db, model=FABLE, since=APPLY, role_ceilings_secs={'merger': 600},
    )

    assert rows[0].at_or_over_flat_role_ceiling is expected


def test_the_killed_at_timeout_signal_is_read_from_the_end_event_not_inferred(runs_db):
    """Exceeding the flat per-role ceiling does NOT mean a run was killed.

    Once a transcript proves liveness the watchdog stops enforcing the flat
    ceiling and enforces max(working_idle_secs, ceiling) as an IDLE bound
    instead (workflow.py, task 2360). A healthy merger that keeps producing
    turns therefore runs well past 600 s by design — so the audit reports the
    producer's own `timed_out` verdict rather than inferring one from duration.
    """
    _fable_merger_run(runs_db, task_id='4377', duration_ms=1_149_252)

    rows = audit_model_admission.scan_invocations(runs_db, model=FABLE, since=APPLY)

    assert rows[0].at_or_over_flat_role_ceiling is True
    assert rows[0].timed_out is False
    assert rows[0].succeeded is True


def test_a_zero_cost_row_survives_to_the_output(runs_db):
    """The milestone task names "$0 cost rows for Fable" as an escalation
    trigger, so a falsy cost must not be filtered out anywhere on the path."""
    _invocation(
        runs_db, model=FABLE, role='merger', task_id='4902', cost_usd=0.0,
        duration_ms=1_000, started_at=_at(hours=6), completed_at=_at(hours=6, minutes=1),
    )

    rows = audit_model_admission.scan_invocations(runs_db, model=FABLE, since=APPLY)

    assert [(r.task_id, r.cost_usd) for r in rows] == [('4902', 0.0)]


def test_rows_before_since_and_rows_on_other_models_are_excluded(runs_db):
    _invocation(
        runs_db, model=FABLE, role='merger', task_id='4300',
        started_at=_at(hours=-3), completed_at=_at(hours=-2),
    )
    _invocation(
        runs_db, model='opus', role='merger', task_id='4903',
        started_at=_at(hours=7), completed_at=_at(hours=8),
    )
    _invocation(
        runs_db, model=FABLE, role='merger', task_id='4904',
        started_at=_at(hours=7), completed_at=_at(hours=8),
    )

    rows = audit_model_admission.scan_invocations(runs_db, model=FABLE, since=APPLY)

    assert [r.task_id for r in rows] == ['4904']


# --- scan_scoped_cap: the scoped-cap posture (check 4) ---


def test_a_cap_hit_scoped_to_the_model_is_reported_with_its_account_and_reason(runs_db):
    _account_event(
        runs_db, account_name='max-b', event_type='cap_hit', created_at=_at(hours=8),
        details={'reason': "You've hit your limit", 'scope': FABLE},
    )

    scan = audit_model_admission.scan_scoped_cap(runs_db, model=FABLE, since=APPLY)

    assert len(scan.scoped_hits) == 1
    assert scan.scoped_hits[0].account_name == 'max-b'
    assert scan.scoped_hits[0].reason == "You've hit your limit"
    assert scan.unscoped_cap_hit_count == 0


def test_a_cap_hit_scoped_to_another_model_is_neither_scoped_nor_unscoped_here(runs_db):
    _account_event(
        runs_db, account_name='max-b', event_type='cap_hit', created_at=_at(hours=8),
        details={'reason': 'limit', 'scope': 'claude-fable-5'},
    )

    scan = audit_model_admission.scan_scoped_cap(runs_db, model=FABLE, since=APPLY)

    assert scan.scoped_hits == ()
    assert scan.unscoped_cap_hit_count == 0


def test_an_account_level_cap_hit_is_counted_rather_than_dropped(runs_db):
    """The failure mode check 4 exists for is a Fable cap that marked the WHOLE
    account because the restart-tier scoped_cap_models leaf had not taken effect
    yet. Dropping scope-less rows would make exactly that case invisible."""
    _account_event(
        runs_db, account_name='max-a', event_type='cap_hit', created_at=_at(hours=9),
        details={'reason': "You're out of extra usage"},
    )

    scan = audit_model_admission.scan_scoped_cap(runs_db, model=FABLE, since=APPLY)

    assert scan.scoped_hits == ()
    assert scan.unscoped_cap_hit_count == 1


def test_a_bare_non_json_details_string_is_tolerated_as_unscoped(runs_db):
    """The live table holds details='Escalation watcher (auto)'."""
    _account_event(
        runs_db, account_name='max-a', event_type='cap_hit', created_at=_at(hours=10),
        details='Escalation watcher (auto)',
    )

    scan = audit_model_admission.scan_scoped_cap(runs_db, model=FABLE, since=APPLY)

    assert scan.scoped_hits == ()
    assert scan.unscoped_cap_hit_count == 1


def test_cap_hits_before_since_are_excluded(runs_db):
    _account_event(
        runs_db, account_name='max-b', event_type='cap_hit', created_at=_at(hours=-5),
        details={'reason': 'limit', 'scope': FABLE},
    )
    _account_event(
        runs_db, account_name='max-a', event_type='cap_hit', created_at=_at(hours=-5),
        details={'reason': 'limit'},
    )

    scan = audit_model_admission.scan_scoped_cap(runs_db, model=FABLE, since=APPLY)

    assert scan.scoped_hits == ()
    assert scan.unscoped_cap_hit_count == 0


def test_restarts_are_windowed_ordered_and_name_the_service_that_restarted(runs_db):
    """WHICH service restarted is the whole question: usage_cap.scoped_cap_models
    is restart-tier on the ORCHESTRATOR, so a dashboard or fused-memory restart
    is not evidence that the leaf took effect."""
    _event(
        runs_db, _at(hours=-2), 'service_restart',
        data={'service': 'dashboard', 'reason': 'post_merge_dashboard_code_change'},
    )
    _event(
        runs_db, _at(hours=12), 'service_restart',
        data={'service': 'orchestrator', 'reason': 'fleet_redeploy'},
    )
    _event(
        runs_db, _at(hours=5), 'service_restart',
        data={'service': 'fused-memory', 'reason': 'post_merge_fused_memory_code_change'},
    )

    scan = audit_model_admission.scan_scoped_cap(runs_db, model=FABLE, since=APPLY)

    assert [r.service for r in scan.restarts] == ['fused-memory', 'orchestrator']
    assert scan.restarts[1].reason == 'fleet_redeploy'


def test_zero_scoped_hits_is_an_empty_tuple_not_none(runs_db):
    scan = audit_model_admission.scan_scoped_cap(runs_db, model=FABLE, since=APPLY)

    assert scan.scoped_hits == ()
    assert scan.restarts == ()
    assert scan.unscoped_cap_hit_count == 0


def test_restarts_are_found_even_though_live_rows_carry_a_task_id(runs_db):
    """Every service_restart row in the live store has a task_id — the merge
    that triggered it. Reading these grouped by task and taking the untagged
    bucket reports zero restarts against real data while passing against a
    fixture that leaves task_id NULL."""
    _event(
        runs_db, _at(hours=3), 'service_restart', task_id='4319',
        data={'service': 'fused-memory', 'reason': 'post_merge_fused_memory_code_change'},
    )

    scan = audit_model_admission.scan_scoped_cap(runs_db, model=FABLE, since=APPLY)

    assert [r.service for r in scan.restarts] == ['fused-memory']


# --- spend_in_window (check 5) and roles_on_model (check 6) ---


def _spend(runs_db, *, model=FABLE, ceiling_usd=150.0):
    return audit_model_admission.spend_in_window(
        runs_db, model=model, window_start=APPLY,
        window_end=APPLY + timedelta(hours=24), ceiling_usd=ceiling_usd,
    )


def test_the_spend_window_is_half_open_at_the_start_and_the_end(runs_db):
    _invocation(
        runs_db, model=FABLE, role='merger', task_id='a', cost_usd=6.08,
        started_at=_at(), completed_at=_at(),  # exactly at window_start: INCLUDED
    )
    _invocation(
        runs_db, model=FABLE, role='steward', task_id='b', cost_usd=3.86,
        started_at=_at(hours=12), completed_at=_at(hours=12),
    )
    _invocation(
        runs_db, model=FABLE, role='merger', task_id='c', cost_usd=99.0,
        started_at=_at(hours=24), completed_at=_at(hours=24),  # at window_end: EXCLUDED
    )

    spend = _spend(runs_db)

    assert spend.invocation_count == 2
    assert spend.total_usd == pytest.approx(9.94)


def test_an_empty_window_totals_zero_rather_than_none(runs_db):
    spend = _spend(runs_db)

    assert spend.total_usd == 0.0
    assert spend.invocation_count == 0
    assert spend.headroom_usd == 150.0
    assert spend.at_or_over_ceiling is False


def test_an_absent_ceiling_reads_as_unknown_not_as_a_zero_ceiling(runs_db):
    """A 0.0 default would make ANY spend at all — even zero — compute as
    at/over ceiling with negative headroom, which renders identically to a real
    breach on the one column a reader is meant to treat as an alarm."""
    _invocation(
        runs_db, model=FABLE, role='merger', task_id='a', cost_usd=6.08,
        started_at=_at(hours=1), completed_at=_at(hours=1),
    )

    spend = _spend(runs_db, ceiling_usd=None)

    assert spend.total_usd == pytest.approx(6.08)
    assert spend.ceiling_usd is None
    assert spend.headroom_usd is None
    assert spend.at_or_over_ceiling is None


@pytest.mark.parametrize(
    ('cost_usd', 'expected'),
    [(150.0, True), (149.99, False)],
)
def test_the_ceiling_flag_is_at_or_above_not_strictly_above(runs_db, cost_usd, expected):
    _invocation(
        runs_db, model=FABLE, role='merger', task_id='a', cost_usd=cost_usd,
        started_at=_at(hours=1), completed_at=_at(hours=1),
    )

    spend = _spend(runs_db)

    assert spend.at_or_over_ceiling is expected
    assert spend.headroom_usd == pytest.approx(150.0 - cost_usd)


def test_observed_roles_within_the_allowlist_leave_unexpected_empty(runs_db):
    _invocation(
        runs_db, model=FABLE, role='merger', task_id='a', cost_usd=6.08,
        started_at=_at(hours=1), completed_at=_at(hours=1),
    )
    _invocation(
        runs_db, model=FABLE, role='steward', task_id='b', cost_usd=1.93,
        started_at=_at(hours=2), completed_at=_at(hours=2),
    )
    _invocation(
        runs_db, model=FABLE, role='steward', task_id='c', cost_usd=1.93,
        started_at=_at(hours=3), completed_at=_at(hours=3),
    )

    containment = audit_model_admission.roles_on_model(
        runs_db, model=FABLE, since=APPLY, expected_roles=('merger', 'steward'),
    )

    assert containment.unexpected_roles == ()
    assert {r.role: (r.count, r.total_usd) for r in containment.by_role} == {
        'merger': (1, pytest.approx(6.08)),
        'steward': (2, pytest.approx(3.86)),
    }


def test_a_role_outside_the_allowlist_is_named_in_unexpected_roles(runs_db):
    """The ladder-containment regression dark-factory-orchestrator.yaml's
    L1241-1244 deviation exists to prevent: the retry ladder was left unchanged
    so a "+1" retry-tier-up cannot route an implementer to Fable."""
    _invocation(
        runs_db, model=FABLE, role='implementer', task_id='a', cost_usd=4.0,
        started_at=_at(hours=1), completed_at=_at(hours=1),
    )

    containment = audit_model_admission.roles_on_model(
        runs_db, model=FABLE, since=APPLY, expected_roles=('merger', 'steward'),
    )

    assert containment.unexpected_roles == ('implementer',)


def test_an_expected_role_with_no_runs_is_a_visible_zero_not_a_missing_line(runs_db):
    """"The merger never ran on Fable at all" is the loudest possible check-6
    finding, and an omitted row would render it as silence."""
    _invocation(
        runs_db, model=FABLE, role='steward', task_id='b', cost_usd=1.93,
        started_at=_at(hours=2), completed_at=_at(hours=2),
    )

    containment = audit_model_admission.roles_on_model(
        runs_db, model=FABLE, since=APPLY, expected_roles=('merger', 'steward'),
    )

    assert [(r.role, r.count) for r in containment.by_role] == [('merger', 0), ('steward', 1)]


# --- audit / render_markdown / render_json / main (report assembly and CLI) ---

WINDOW = (APPLY, APPLY + timedelta(hours=24))


@pytest.fixture
def live_shaped_db(runs_db):
    """A fixture seeded to mirror the shape actually observed post-D6-apply.

    One config-layer Fable merger dispatch that ran 45 turns and resolved a
    merge, and one policy-rule Fable steward dispatch at routing tier 1.
    """
    _event(
        runs_db, _at(hours=2), 'routing_decision', task_id='4377', role='merger',
        data=_routing_payload(role='merger', model=FABLE),
    )
    _event(
        runs_db, _at(days=1), 'routing_decision', task_id='4211', role='steward',
        data=_routing_payload(
            role='steward', model=FABLE, source_layer='policy_rule',
            rule_id='steward-retry-fable', routing_tier=1,
        ),
    )
    _fable_merger_run(runs_db, task_id='4377')
    _event(
        runs_db, _at(hours=2, minutes=30), 'merge_finalized', task_id='4377',
        data=_merge_finalized_payload(branch='4377', state='done', merge_sha='d411f107'),
    )
    _invocation(
        runs_db, model=FABLE, role='steward', task_id='4211', cost_usd=3.86,
        duration_ms=90_000, started_at=_at(hours=19), completed_at=_at(hours=20),
    )
    _event(
        runs_db, _at(hours=3), 'service_restart', task_id='4319',
        data={'service': 'orchestrator', 'reason': 'fleet_redeploy'},
    )
    return runs_db


def _audit(conn, **overrides):
    kwargs = {
        'model': FABLE, 'since': APPLY, 'expected_roles': ('merger', 'steward'),
        'window': WINDOW, 'ceiling_usd': 150.0,
    }
    return audit_model_admission.audit(conn, **{**kwargs, **overrides})


def test_the_markdown_body_carries_the_measured_values_not_a_summary(live_shaped_db):
    body = audit_model_admission.render_markdown(_audit(live_shaped_db))

    for token in (
        FABLE,                # the target model, named in the sections
        'config',             # the merger's source_layer
        'steward-retry-fable',  # the rule that matched
        '45',                 # turns, joined from invocation_end
        'done',               # the merge state resolved
        'd411f107',           # the merge sha
        '9.94',               # 6.08 + 3.86 spend in window
        '150',                # the ceiling it is measured against
    ):
        assert token in body, f'{token!r} missing from the rendered report'


def test_an_absent_ceiling_renders_as_a_dash_rather_than_as_a_breach(live_shaped_db):
    body = audit_model_admission.render_markdown(_audit(live_shaped_db, ceiling_usd=None))

    section = body.split('### 5.')[1].split('### 6.')[0]
    assert '| 2 | 9.94 | - | - | - |' in section


def test_render_json_round_trips_to_one_key_per_section(live_shaped_db):
    payload = json.loads(audit_model_admission.render_json(_audit(live_shaped_db)))

    assert set(payload) == {
        'meta', 'routing_decisions', 'invocations', 'steward_tier_escalation',
        'scoped_cap', 'spend', 'role_containment',
    }
    assert payload['meta']['model'] == FABLE
    assert payload['invocations'][0]['turns'] == 45


def _tier_section(result):
    return json.loads(audit_model_admission.render_json(result))['steward_tier_escalation']


def test_a_tier_escalation_that_never_happened_does_not_render_as_one(runs_db):
    """Absence is not failure — but the two must not render identically, or a
    reader cannot tell "the rule never got a chance" from "the rule matched".

    Both audits read the SAME store, taken before and after the tier-1 dispatch
    is seeded: audit() is a pure read, so ordering the seeding is what keeps the
    two cases independent without needing two databases.
    """
    _event(
        runs_db, _at(hours=2), 'routing_decision', task_id='4377', role='merger',
        data=_routing_payload(role='merger', model=FABLE),
    )
    never = _audit(runs_db)

    _event(
        runs_db, _at(days=1), 'routing_decision', task_id='4211', role='steward',
        data=_routing_payload(
            role='steward', model=FABLE, source_layer='policy_rule',
            rule_id='steward-retry-fable', routing_tier=1,
        ),
    )
    exercised = _audit(runs_db)

    assert exercised.tier_escalations and not never.tier_escalations
    assert _tier_section(exercised)['exercised'] is True
    assert _tier_section(never)['exercised'] is False
    assert 'steward-retry-fable' in audit_model_admission.render_markdown(exercised)
    assert 'steward-retry-fable' not in audit_model_admission.render_markdown(never)


def test_main_emits_parseable_json_and_exits_zero(runs_db_path, live_shaped_db, capsys):
    exit_code = audit_model_admission.main([
        '--model', FABLE, '--expect-roles', 'merger,steward',
        '--since', APPLY.isoformat(), '--window', '24h', '--ceiling', '150',
        '--runs-db', str(runs_db_path), '--format', 'json',
    ])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload['meta']['model'] == FABLE
    assert [r['role'] for r in payload['invocations']] == ['merger', 'steward']


def test_main_without_a_ceiling_reports_no_ceiling_rather_than_a_zero_one(
    runs_db_path, live_shaped_db, capsys
):
    """--ceiling is the flag most likely to be forgotten on task 5441's re-run."""
    audit_model_admission.main([
        '--model', FABLE, '--expect-roles', 'merger,steward',
        '--since', APPLY.isoformat(), '--runs-db', str(runs_db_path), '--format', 'json',
    ])

    spend = json.loads(capsys.readouterr().out)['spend']
    assert spend['ceiling_usd'] is None
    assert spend['headroom_usd'] is None
    assert spend['at_or_over_ceiling'] is None


@pytest.mark.parametrize(
    ('spec', 'expected'),
    [('24h', timedelta(hours=24)), ('1h', timedelta(hours=1)), ('14d', timedelta(days=14))],
)
def test_a_window_spec_parses_to_the_span_it_names(spec, expected):
    """24h is this report's window and 14d is task 5441's — both hand-typed."""
    assert audit_model_admission._parse_window(spec) == expected


@pytest.mark.parametrize(
    'spec', ['24', '24x', '24hours', '24h ', 'h', '', '0h', '-1h', '1.5d', '24 h', '24H'],
)
def test_a_malformed_window_spec_is_rejected_rather_than_truncated(spec):
    """'24' silently read as 24 hours would move the window every spend figure
    is measured over without saying so; the anchors are the only thing
    preventing it, and nothing else in the suite exercises them."""
    with pytest.raises(argparse.ArgumentTypeError):
        audit_model_admission._parse_window(spec)


@pytest.fixture
def host_tz_is_not_utc(monkeypatch):
    """Pin the process's local zone to a non-UTC one for the duration of a test.

    Without this, "naive means UTC" is unfalsifiable on a UTC host: reading a
    naive bound as local time would be indistinguishable from reading it as
    UTC, and the assertion would pass against either implementation.
    """
    monkeypatch.setenv('TZ', 'America/New_York')
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def test_a_naive_since_is_read_as_utc_not_as_host_local_time(host_tz_is_not_utc):
    """The store is UTC throughout and these values are used directly as SQL
    comparands, so shifting a hand-typed bound by the host's offset would move
    the whole window silently."""
    parsed = audit_model_admission._parse_moment('2026-09-12T06:43:16')

    assert parsed.isoformat() == '2026-09-12T06:43:16+00:00'
    assert parsed == APPLY


def test_an_offset_bearing_since_is_converted_to_the_same_utc_instant():
    """The apply commit is stamped +01:00; the report's --since is its UTC
    equivalent, and both spellings must resolve to one instant."""
    parsed = audit_model_admission._parse_moment('2026-09-12T07:43:16+01:00')

    assert parsed.isoformat() == '2026-09-12T06:43:16+00:00'
    assert parsed.utcoffset() == timedelta(0)


@pytest.mark.parametrize('spec', ['yesterday', '2026-13-01T00:00:00', ''])
def test_a_junk_since_is_rejected_by_the_argument_parser(spec):
    with pytest.raises(argparse.ArgumentTypeError):
        audit_model_admission._parse_moment(spec)


@pytest.mark.parametrize(
    ('spec', 'expected'),
    [('merger=600', ('merger', 600)), ('reviewer=1200', ('reviewer', 1200))],
)
def test_a_role_ceiling_spec_parses_to_a_role_and_its_seconds(spec, expected):
    assert audit_model_admission._parse_role_ceiling(spec) == expected


@pytest.mark.parametrize(
    'spec', ['merger', 'merger=', '=600', 'merger=0', 'merger=-1', 'merger=10m'],
)
def test_a_malformed_role_ceiling_spec_is_rejected_at_the_boundary(spec):
    """`merger=0` in particular: accepted, it would flag every merger run as
    at-or-over its ceiling, since the comparison is at-or-above."""
    with pytest.raises(argparse.ArgumentTypeError):
        audit_model_admission._parse_role_ceiling(spec)


def test_the_role_ceiling_flag_binds_and_leaves_the_other_defaults_standing(
    runs_db_path, live_shaped_db, capsys
):
    """The flat-ceiling mapping is a real CLI dimension, not a test-only seam:
    a future admission on a role whose timeouts.<role> differs is auditable
    without editing the module."""
    audit_model_admission.main([
        '--model', FABLE, '--expect-roles', 'merger,steward',
        '--since', APPLY.isoformat(), '--runs-db', str(runs_db_path),
        '--role-ceiling', 'steward=60', '--format', 'json',
    ])

    rows = {r['role']: r for r in json.loads(capsys.readouterr().out)['invocations']}
    assert rows['steward']['at_or_over_flat_role_ceiling'] is True   # 90 s run, 60 s limit
    assert rows['merger']['at_or_over_flat_role_ceiling'] is False   # default 600 s stands


def test_main_leaves_the_store_byte_for_byte_unchanged(runs_db_path, live_shaped_db, capsys):
    before = (runs_db_path.read_bytes(), runs_db_path.stat().st_mtime_ns)

    audit_model_admission.main([
        '--model', FABLE, '--expect-roles', 'merger,steward',
        '--since', APPLY.isoformat(), '--runs-db', str(runs_db_path),
    ])
    capsys.readouterr()

    assert (runs_db_path.read_bytes(), runs_db_path.stat().st_mtime_ns) == before


def test_the_connection_factory_refuses_a_write(runs_db_path):
    conn = audit_model_admission._connect_ro(runs_db_path)
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute(
                "INSERT INTO events (timestamp, run_id, event_type) VALUES ('t', 'r', 'e')"
            )
    finally:
        conn.close()
