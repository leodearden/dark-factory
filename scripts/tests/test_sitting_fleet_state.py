"""Tests for scripts/sitting/fleet_state.py — stamped fleet measurements for the return brief (task 5376)."""
from __future__ import annotations

import dataclasses
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from escalation.models import Escalation
from escalation.shadow_ruling import ShadowRuling, agreement_report
from orchestrator.session_registry import DecisionRecord, normalize_escalations_dir
from sitting import fleet_state as mod
from sitting import inventory, payloads
from sitting.payloads import AgreedMarker, PreparedMarker

from orchestrator import digest

NOW = datetime(2026, 9, 26, 5, 30, tzinfo=UTC)
WINDOW = mod.Window(NOW - timedelta(days=1), NOW)
IN_WINDOW = '2026-09-25T18:00:00+00:00'
BEFORE_WINDOW = '2026-09-20T18:00:00+00:00'

_RUNS_SCHEMA = """
CREATE TABLE events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, run_id TEXT NOT NULL, task_id TEXT,
    event_type TEXT NOT NULL, phase TEXT, role TEXT, data TEXT DEFAULT '{}', cost_usd REAL, duration_ms INTEGER
);
CREATE TABLE invocations (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, task_id TEXT, project_id TEXT NOT NULL,
    account_name TEXT NOT NULL, model TEXT NOT NULL, role TEXT NOT NULL, cost_usd REAL NOT NULL DEFAULT 0.0,
    duration_ms INTEGER NOT NULL DEFAULT 0, capped INTEGER NOT NULL DEFAULT 0,
    started_at TEXT NOT NULL, completed_at TEXT NOT NULL
);
CREATE TABLE task_results (
    run_id TEXT NOT NULL, task_id TEXT NOT NULL, project_id TEXT NOT NULL, title TEXT,
    outcome TEXT NOT NULL, completed_at TEXT, PRIMARY KEY (run_id, task_id)
);
"""


def _completed(conn: sqlite3.Connection, task_id: str, outcome: str, at: str) -> None:
    conn.execute(
        "INSERT INTO events (timestamp, run_id, task_id, event_type, data) VALUES (?, 'r1', ?, 'task_completed', ?)",
        (at, task_id, json.dumps({'outcome': outcome})),
    )


def _invocation(conn: sqlite3.Connection, task_id: str | None, model: str, role: str, cost: float, capped: int,
                at: str = IN_WINDOW) -> None:
    conn.execute(
        'INSERT INTO invocations (run_id, task_id, project_id, account_name, model, role, cost_usd, capped, '
        "started_at, completed_at) VALUES ('r1', ?, 'dark_factory', 'acct', ?, ?, ?, ?, ?, ?)",
        (task_id, model, role, cost, capped, at, at),
    )


def _build_runs_db(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(_RUNS_SCHEMA)
        for task_id in ('1', '2', '3'):
            _completed(conn, task_id, 'done', IN_WINDOW)
        _completed(conn, '4', 'blocked', IN_WINDOW)
        _completed(conn, '5', 'done', BEFORE_WINDOW)
        _invocation(conn, '1', 'claude-opus', 'implementer', 1.5, 0)
        _invocation(conn, '1', 'claude-opus', 'implementer', 2.5, 1)
        _invocation(conn, '4', 'claude-sonnet', 'reviewer', 0.5, 0)
        _invocation(conn, '9', 'claude-opus', 'implementer', 9.0, 1, at=BEFORE_WINDOW)
        conn.execute("INSERT INTO task_results VALUES ('r1', '1', 'dark_factory', 't1', 'done', ?)", (IN_WINDOW,))
        conn.execute("INSERT INTO task_results VALUES ('r1', '4', 'dark_factory', 't4', 'blocked', ?)", (IN_WINDOW,))
        conn.commit()
    finally:
        conn.close()
    return path


def _write_escalation(queue: Path, *, subdir: str = '', **fields) -> Escalation:
    fields.setdefault('task_id', fields['id'].split('-')[1])
    fields.setdefault('agent_role', 'escalation-watcher-auto')
    fields.setdefault('severity', 'blocking')
    fields.setdefault('category', 'design_concern')
    fields.setdefault('summary', f"summary of {fields['id']}")
    fields.setdefault('timestamp', '2026-09-24T05:30:00+00:00')
    esc = Escalation(**fields)
    target = queue / subdir if subdir else queue
    target.mkdir(parents=True, exist_ok=True)
    (target / f'{esc.id}.json').write_text(esc.to_json())
    return esc


def _resolved(queue: Path, esc_id: str, *, resolved_at: str = IN_WINDOW, **fields) -> Escalation:
    fields.setdefault('level', 2)
    fields.setdefault('resolved_by', 'interactive')
    return _write_escalation(queue, id=esc_id, status='resolved', resolved_at=resolved_at, **fields)


def _write_decision(fleet: Path, **fields) -> DecisionRecord:
    fields.setdefault('project', 'dark_factory')
    fields.setdefault('text', f"question {fields['id']}")
    fields.setdefault('filed_at', IN_WINDOW)
    record = DecisionRecord(**fields)
    decisions = fleet / 'decisions'
    decisions.mkdir(parents=True, exist_ok=True)
    (decisions / f'{record.id}.json').write_text(record.to_json())
    return record


def _prepared(recommendation: str = 'A', no_lean_reason: str = '') -> str:
    return payloads.render_prepared_marker(PreparedMarker(
        recommendation=recommendation, no_lean_reason=no_lean_reason, sitting_id='s1', prepared_at=IN_WINDOW,
    ))


def _agreed(answer: str, agreed: bool | None) -> str:
    return payloads.render_agreed_marker(AgreedMarker(answer=answer, agreed=agreed, answer_rounds=1, at=IN_WINDOW))


def _note(*lines: str) -> str:
    return payloads.append_markers('Leo ruled; triage note prose', *lines)


def _snapshot(root: Path) -> dict[Path, tuple[bytes, int]]:
    return {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in root.rglob('*') if path.is_file()}


@pytest.fixture
def df(tmp_path, make_tasks_db, project_root_with_tasks_db) -> Path:
    root = tmp_path / 'src' / 'dark-factory'
    make_tasks_db([
        {'id': 10, 'status': 'blocked', 'title': 'blocked behind an open escalation'},
        {'id': 11, 'status': 'blocked', 'title': 'blocked with nothing open'},
        {'id': 12, 'status': 'pending'},
        {'id': 13, 'status': 'done'},
    ], directory=project_root_with_tasks_db(root).parent)
    _build_runs_db(mod.runs_db_path(root))
    queue = root / 'data' / 'escalations'
    _write_escalation(queue, id='esc-10-1', level=1, category='infra_issue')
    _resolved(queue, 'esc-10-2')
    _write_escalation(queue, id='esc-12-1', level=2)
    return root


def _index(*queues: Path, tmp_path: Path):
    return inventory.collect_open_items(queue_dirs=[str(q) for q in queues], decisions_root=tmp_path / 'fleet',
                                        now=NOW).escalation_index


class TestMeasurement:
    def test_is_frozen_and_stamped(self):
        measurement = mod.Measurement(3, NOW.isoformat(), 'runs.db', 'ok')

        assert (measurement.value, measurement.measured_at, measurement.source, measurement.status) == (
            3, NOW.isoformat(), 'runs.db', 'ok')
        assert measurement.ok
        with pytest.raises(dataclasses.FrozenInstanceError):
            measurement.value = 4  # type: ignore[misc]

    @pytest.mark.parametrize('status', ['', 'OK', 'missing', 'degraded'])
    def test_rejects_a_status_outside_the_vocabulary(self, status):
        with pytest.raises(ValueError, match='status'):
            mod.Measurement(0, NOW.isoformat(), 'runs.db', status)  # type: ignore[arg-type]

    @pytest.mark.parametrize('field', ['measured_at', 'source'])
    def test_an_unstamped_measurement_is_unrepresentable(self, field):
        stamped = {'value': 0, 'measured_at': NOW.isoformat(), 'source': 'runs.db', 'status': 'ok'}

        with pytest.raises(ValueError, match=field):
            mod.Measurement(**{**stamped, field: ''})

    @pytest.mark.parametrize('status', ['source_missing', 'unreadable'])
    def test_a_flagged_measurement_must_state_its_shortfall(self, status):
        with pytest.raises(ValueError, match='shortfall'):
            mod.Measurement(0, NOW.isoformat(), 'runs.db', status)

        flagged = mod.Measurement(0, NOW.isoformat(), 'runs.db', status,
                                  shortfalls=(inventory.Shortfall('runs_db', 'runs.db', 'absent'),))
        assert not flagged.ok

    def test_window_is_aware_and_ordered(self):
        assert mod.Window.trailing(NOW, days=1) == WINDOW
        with pytest.raises(ValueError, match='aware'):
            mod.Window(datetime(2026, 9, 25), datetime(2026, 9, 26))
        with pytest.raises(ValueError, match='start'):
            mod.Window(NOW, NOW - timedelta(days=1))


class TestLanded:
    def test_equals_the_digest_count_and_an_independent_fixture_count(self, df):
        runs_db = mod.runs_db_path(df)

        measurement = mod.landed(runs_db, WINDOW, now=NOW)

        assert measurement.status == 'ok'
        assert measurement.value == digest.count_done_in_window(
            runs_db, WINDOW.start.isoformat(), WINDOW.end.isoformat()) == 3
        assert measurement.measured_at == NOW.isoformat()
        assert measurement.source == str(runs_db)

    def test_a_missing_runs_db_is_a_flagged_zero(self, tmp_path):
        runs_db = tmp_path / 'absent' / 'runs.db'

        measurement = mod.landed(runs_db, WINDOW, now=NOW)

        assert (measurement.value, measurement.status) == (0, 'source_missing')
        assert str(runs_db) in measurement.shortfalls[0].path
        assert not runs_db.parent.exists()

    def test_a_file_that_is_not_a_database_is_unreadable(self, tmp_path):
        runs_db = tmp_path / 'runs.db'
        runs_db.write_bytes(b'this is not sqlite at all, not even close' * 4)

        measurement = mod.landed(runs_db, WINDOW, now=NOW)

        assert (measurement.value, measurement.status) == (0, 'unreadable')

    def test_a_database_without_the_events_table_is_unreadable_not_zero(self, tmp_path):
        runs_db = tmp_path / 'runs.db'
        conn = sqlite3.connect(runs_db)
        conn.execute('CREATE TABLE unrelated (x INTEGER)')
        conn.close()

        measurement = mod.landed(runs_db, WINDOW, now=NOW)

        assert measurement.status == 'unreadable'
        assert 'events' in measurement.shortfalls[0].reason


class TestStuck:
    def test_one_row_per_blocked_task_with_its_open_escalation_or_an_explicit_none(self, df, tmp_path):
        index = _index(df / 'data' / 'escalations', tmp_path=tmp_path)

        measurement = mod.stuck(df, index, now=NOW)

        assert measurement.status == 'ok'
        assert measurement.source == str(df / '.taskmaster' / 'tasks' / 'tasks.db')
        rows = {row.task_id: row for row in measurement.value}
        assert set(rows) == {'10', '11'}
        assert rows['10'].open_escalations is not None
        (open_esc,) = rows['10'].open_escalations
        assert (open_esc.escalation_id, open_esc.category, open_esc.level) == ('esc-10-1', 'infra_issue', 1)
        assert open_esc.age_days == pytest.approx(2.0)
        assert open_esc.queue_dir == normalize_escalations_dir(df / 'data' / 'escalations')
        assert 'esc-10-1' in rows['10'].reason and 'infra_issue' in rows['10'].reason
        assert 'data/escalations' in rows['10'].reason
        assert rows['11'].open_escalations == ()
        assert rows['11'].reason == mod.NO_OPEN_ESCALATION == 'blocked with no open escalation'
        assert rows['10'].title == 'blocked behind an open escalation'

    def test_rows_are_frozen(self, df, tmp_path):
        row = mod.stuck(df, _index(df / 'data' / 'escalations', tmp_path=tmp_path), now=NOW).value[0]

        with pytest.raises(dataclasses.FrozenInstanceError):
            row.title = 'changed'  # type: ignore[misc]

    def test_an_unindexed_queue_makes_the_reason_unknown_never_none_open(self, df):
        measurement = mod.stuck(df, {}, now=NOW)

        assert {row.open_escalations for row in measurement.value} == {None}
        assert all(row.reason != mod.NO_OPEN_ESCALATION for row in measurement.value)
        assert measurement.shortfalls

    @pytest.mark.parametrize(('contents', 'status'), [(None, 'source_missing'), (b'garbage' * 20, 'unreadable')])
    def test_an_unreadable_tasks_db_is_a_stated_shortfall(self, tmp_path, contents, status):
        root = tmp_path / 'proj'
        if contents is not None:
            db = root / '.taskmaster' / 'tasks' / 'tasks.db'
            db.parent.mkdir(parents=True)
            db.write_bytes(contents)

        measurement = mod.stuck(root, {}, now=NOW)

        assert (measurement.value, measurement.status) == ((), status)
        assert 'tasks.db' in measurement.shortfalls[0].path


class TestSpendAndCapHits:
    def test_equals_the_digest_model_role_rollup_cells(self, df):
        runs_db = mod.runs_db_path(df)

        measurement = mod.spend_and_cap_hits(runs_db, WINDOW, now=NOW)

        expected = digest.model_role_rollup(runs_db, WINDOW.start.isoformat(), WINDOW.end.isoformat())
        assert measurement.status == 'ok'
        assert measurement.value == tuple(expected.rows)
        cells = {(row.model, row.role): row for row in measurement.value}
        opus = cells[('claude-opus', 'implementer')]
        assert (opus.total_cost_usd, opus.cap_hit_rate, opus.cost_per_done) == (4.0, 0.5, 4.0)
        assert cells[('claude-sonnet', 'reviewer')].cost_per_done is None

    def test_a_missing_runs_db_is_a_flagged_empty_rollup(self, tmp_path):
        measurement = mod.spend_and_cap_hits(tmp_path / 'runs.db', WINDOW, now=NOW)

        assert (measurement.value, measurement.status) == ((), 'source_missing')


class TestAutonomousCloses:
    """Windowed on when a close HAPPENED (``closed_at``), never on when its decision was filed."""

    @pytest.fixture
    def fleet(self, tmp_path) -> Path:
        fleet = tmp_path / 'fleet'
        evidence = 'Leo 2026-09-25: "take option A"\nquoted from esc-5580-4'
        _write_decision(fleet, id='answered-in', state='answered', escalation_id='esc-1-1', closing_evidence=evidence,
                        closed_at='2026-09-25T20:00:00+00:00')
        _write_decision(fleet, id='dropped-in', state='dropped', closing_evidence='superseded by task 7',
                        closed_at='2026-09-25T19:00:00+00:00')
        _write_decision(fleet, id='filed-before-closed-in', state='answered', filed_at=BEFORE_WINDOW,
                        closing_evidence='a long-open gate closed last night', closed_at='2026-09-25T21:00:00+00:00')
        _write_decision(fleet, id='closed-before', state='answered', filed_at=BEFORE_WINDOW,
                        closing_evidence='an older close', closed_at='2026-09-21T00:00:00+00:00')
        _write_decision(fleet, id='evidence-undated', state='answered', closing_evidence='closed before closed_at')
        _write_decision(fleet, id='evidence-bad-stamp', state='dropped', closing_evidence='hand-edited',
                        closed_at='last tuesday')
        _write_decision(fleet, id='answered-bare', state='answered')
        _write_decision(fleet, id='open-with-evidence', closing_evidence='never closed')
        return fleet

    def test_window_closes_carry_evidence_verbatim_and_a_lifetime_count(self, fleet):
        measurement = mod.autonomous_closes(fleet, WINDOW, now=NOW)

        assert measurement.status == 'ok'
        closes = {close.decision_id: close for close in measurement.value.in_window}
        assert closes['answered-in'].evidence == 'Leo 2026-09-25: "take option A"\nquoted from esc-5580-4'
        assert (closes['answered-in'].state, closes['answered-in'].escalation_id) == ('answered', 'esc-1-1')
        assert measurement.value.lifetime == 6

    def test_a_close_filed_before_the_window_counts_when_it_closed_inside_it(self, fleet):
        closes = mod.autonomous_closes(fleet, WINDOW, now=NOW).value

        assert [close.decision_id for close in closes.in_window] == [
            'dropped-in', 'answered-in', 'filed-before-closed-in',
        ]
        late = closes.in_window[-1]
        assert (late.filed_at, late.closed_at) == (BEFORE_WINDOW, '2026-09-25T21:00:00+00:00')

    def test_an_undated_close_is_shown_never_dropped(self, fleet):
        closes = mod.autonomous_closes(fleet, WINDOW, now=NOW).value

        assert [(close.decision_id, close.closed_at) for close in closes.undated] == [
            ('evidence-bad-stamp', 'last tuesday'), ('evidence-undated', ''),
        ]
        assert not {'evidence-bad-stamp', 'evidence-undated', 'closed-before'} & {
            close.decision_id for close in closes.in_window
        }

    def test_an_unreadable_record_is_a_shortfall_not_a_silent_skip(self, fleet):
        (fleet / 'decisions' / 'corrupt.json').write_text('{not json')

        measurement = mod.autonomous_closes(fleet, WINDOW, now=NOW)

        assert measurement.status == 'ok'
        assert [shortfall.path for shortfall in measurement.shortfalls] == [str(fleet / 'decisions' / 'corrupt.json')]

    def test_a_missing_registry_is_flagged(self, tmp_path):
        measurement = mod.autonomous_closes(tmp_path / 'no-fleet', WINDOW, now=NOW)

        assert measurement.status == 'source_missing'
        assert (measurement.value.in_window, measurement.value.undated, measurement.value.lifetime) == ((), (), 0)


def _shadow(ruling_class: str, action: str) -> str:
    return ShadowRuling(ruling_class=ruling_class, proposed_action=action, evidence='probe cleared',
                        confidence=0.9).to_note_line()


class TestStandingPolicyRulings:
    def test_folds_the_per_queue_agreement_reports(self, tmp_path):
        first, second = tmp_path / 'a' / 'data' / 'escalations', tmp_path / 'b' / 'data' / 'escalations'
        cls = 'infra_issue_transient_self_cleared'
        _resolved(first, 'esc-1-1', triage_note=_shadow(cls, 'close_only'), resolution_action='close_only')
        _resolved(first, 'esc-1-2', triage_note=_shadow(cls, 'close_only'), resolution_action='resume')
        _resolved(second, 'esc-2-1', triage_note=_shadow(cls, 'close_only'), resolution_action='close_only')
        _write_escalation(second, id='esc-2-2', triage_note=_shadow(cls, 'resume'))

        measurement = mod.standing_policy_rulings([str(first), str(second)], WINDOW, now=NOW)

        per_queue = [agreement_report(q, since=WINDOW.start, until=WINDOW.end) for q in (first, second)]
        queue_rows = [row for report in per_queue if (row := report.for_class(cls)) is not None]
        folded = measurement.value.report.for_class(cls)
        assert measurement.status == 'ok'
        assert folded is not None
        assert (folded.agreed, folded.diverged) == (2, 1) == (
            sum(row.agreed for row in queue_rows), sum(row.diverged for row in queue_rows))
        assert measurement.value.report.unresolved_lifetime == 1
        assert (measurement.value.report.since, measurement.value.report.until) == (WINDOW.start, WINDOW.end)

    def test_with_no_adopted_class_it_states_shadow_only_rather_than_rendering_empty(self, df):
        measurement = mod.standing_policy_rulings([str(df / 'data' / 'escalations')], WINDOW, now=NOW)

        assert measurement.status == 'ok'
        assert measurement.value.report.classes == ()
        assert 'shadow-only' in measurement.value.statement
        assert 'measurement' in measurement.value.statement

    def test_no_existing_queue_is_flagged(self, tmp_path):
        measurement = mod.standing_policy_rulings([str(tmp_path / 'nowhere')], WINDOW, now=NOW)

        assert measurement.status == 'source_missing'
        assert measurement.shortfalls[0].path == str(tmp_path / 'nowhere')


class TestPreparerTrial:
    @pytest.fixture
    def queue(self, tmp_path) -> Path:
        queue = tmp_path / 'data' / 'escalations'
        _resolved(queue, 'esc-1-1', triage_note=_note(_prepared('A'), _agreed('A', True)), resolution_turns=1)
        _resolved(queue, 'esc-2-1', triage_note=_note(_prepared('A'), _agreed('B', False)), resolution_turns=3)
        _resolved(queue, 'esc-3-1', triage_note=_note(_prepared('', 'two live readings'), _agreed('B', None)),
                  resolution_turns=2)
        _resolved(queue, 'esc-4-1', subdir='archive/2026-09-25',
                  triage_note=_note(_prepared('A'), _agreed('B', False), _agreed('A', True)))
        _resolved(queue, 'esc-5-1', triage_note=_note('x_prepared: {not json', _agreed('A', True)))
        _resolved(queue, 'esc-6-1', triage_note=_note(_prepared('A')))
        _resolved(queue, 'esc-7-1', triage_note=_note(_prepared('A'), 'x_agreed: {"answer": "A"}'))
        _resolved(queue, 'esc-8-1', resolved_at=BEFORE_WINDOW, triage_note=_note(_prepared('A'), _agreed('A', True)),
                  resolution_turns=9)
        _write_escalation(queue, id='esc-9-1', level=2, triage_note=_note(_prepared('A'), _agreed('A', True)))
        _resolved(queue, 'esc-10-1', resolution_turns=5)
        _resolved(queue, 'esc-11-1', level=1, resolution_turns=7)
        return queue

    def test_agreement_counts_the_last_agreed_marker_and_excludes_rejected_ones(self, queue):
        measurement = mod.preparer_trial([str(queue)], WINDOW, now=NOW)

        trial = measurement.value
        assert measurement.status == 'ok'
        assert (trial.n, trial.denominator, trial.numerator) == (5, 3, 2)
        assert trial.rate == pytest.approx(2 / 3)
        assert (trial.no_lean, trial.unanswered, trial.rejected) == (1, 1, 2)

    def test_resolution_turns_distribution_over_resolved_l2s_in_the_window(self, queue):
        turns = mod.preparer_trial([str(queue)], WINDOW, now=NOW).value.resolution_turns

        assert (turns.resolved, turns.with_turns, turns.median, turns.total) == (8, 4, 2.5, 11)

    def test_an_empty_window_has_no_rate_rather_than_zero(self, tmp_path):
        queue = tmp_path / 'data' / 'escalations'
        queue.mkdir(parents=True)

        trial = mod.preparer_trial([str(queue)], WINDOW, now=NOW).value

        assert (trial.n, trial.denominator, trial.rate) == (0, 0, None)
        assert (trial.resolution_turns.with_turns, trial.resolution_turns.median) == (0, None)

    def test_no_existing_queue_is_flagged(self, tmp_path):
        measurement = mod.preparer_trial([str(tmp_path / 'nowhere')], WINDOW, now=NOW)

        assert measurement.status == 'source_missing'


class TestCrossProject:
    def test_keyed_by_canonical_token_and_an_absent_root_is_stated_not_omitted(self, df, tmp_path):
        bare = tmp_path / 'src' / 'reify'
        bare.mkdir(parents=True)
        index = _index(df / 'data' / 'escalations', tmp_path=tmp_path)

        states = mod.measure_projects([df, bare], index, window=WINDOW, now=NOW)

        assert set(states) == {'dark_factory', 'reify'}
        assert states['dark_factory'].landed.value == 3
        assert states['dark_factory'].root == str(df)
        reify = states['reify']
        for measurement in (reify.landed, reify.stuck, reify.spend, reify.standing_policy, reify.trial):
            assert measurement.status == 'source_missing'
            assert measurement.shortfalls
            assert measurement.measured_at == NOW.isoformat()

    def test_a_projects_queues_are_its_own(self, df, tmp_path):
        recon = df / 'data' / 'reconciliation' / 'escalations'
        _write_escalation(recon, id='esc-11-1', level=0, category='risk_identified')
        index = _index(df / 'data' / 'escalations', recon, tmp_path=tmp_path)

        state = mod.measure_projects([df], index, window=WINDOW, now=NOW)['dark_factory']

        rows = {row.task_id: row for row in state.stuck.value}
        assert rows['11'].open_escalations is not None
        assert [(e.escalation_id, e.queue_dir) for e in rows['11'].open_escalations] == [
            ('esc-11-1', normalize_escalations_dir(recon))]
        assert 'data/reconciliation/escalations' in rows['11'].reason
        assert state.standing_policy.source == ', '.join(
            normalize_escalations_dir(q) for q in (df / 'data' / 'escalations', recon))


class TestReadOnly:
    def test_no_gatherer_writes_anything(self, df, tmp_path):
        fleet = tmp_path / 'fleet'
        _write_decision(fleet, id='answered-in', state='answered', closing_evidence='quoted')
        queue = df / 'data' / 'escalations'
        before = _snapshot(tmp_path)
        dirs_before = sorted(path for path in tmp_path.rglob('*') if path.is_dir())

        index = _index(queue, tmp_path=tmp_path)
        mod.landed(mod.runs_db_path(df), WINDOW, now=NOW)
        mod.spend_and_cap_hits(mod.runs_db_path(df), WINDOW, now=NOW)
        mod.stuck(df, index, now=NOW)
        mod.autonomous_closes(fleet, WINDOW, now=NOW)
        mod.standing_policy_rulings([str(queue), str(tmp_path / 'nowhere')], WINDOW, now=NOW)
        mod.preparer_trial([str(queue)], WINDOW, now=NOW)
        mod.measure_projects([df, tmp_path / 'src' / 'absent'], index, window=WINDOW, now=NOW)

        assert _snapshot(tmp_path) == before
        assert sorted(path for path in tmp_path.rglob('*') if path.is_dir()) == dirs_before
        assert not list(tmp_path.rglob('*.lock'))
