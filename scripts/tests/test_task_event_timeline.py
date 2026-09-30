"""Tests for scripts/task_event_timeline.py — one task's runs.db events, counted.

Every test drives the CLI through ``main(argv)`` against a synthetic runs.db
seeded by ``scripts/tests/conftest.py``'s ``runs_db`` / ``runs_db_path``
fixtures, never against a live event log.

The known-answer corpus reproduces reify esc-5495-3: four reviewer cap-kills
and a separate fifth invocation must tally as 4 and 1, never as one composite.

To run just this suite:

    uv run --project shared pytest scripts/tests/test_task_event_timeline.py -q
"""
from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta

import pytest
from task_event_timeline import main

TASK = '5495'
_BASE_TIME = datetime(2026, 9, 26, 23, 0, tzinfo=UTC)
_REVIEWER = 'reviewer_comprehensive'
_CAP_KILL = {'subtype': 'error_max_budget_usd'}


def _cap_kill_then_reroute(cost_usd):
    return [
        {'event_type': 'invocation_end', 'run_id': 'run-813a', 'task_id': TASK,
         'phase': 'review', 'role': _REVIEWER, 'data': _CAP_KILL,
         'cost_usd': cost_usd, 'duration_ms': 1_200_000},
        {'event_type': 'routing_decision', 'run_id': 'run-813a', 'task_id': TASK,
         'phase': 'review', 'role': _REVIEWER, 'data': {'model': 'claude-opus'}},
    ]


ESC_5495_3 = [
    {'event_type': 'orchestrator_start', 'run_id': 'run-0000', 'task_id': None},
    {'event_type': 'phase_enter', 'run_id': 'run-813a', 'task_id': TASK, 'phase': 'review'},
    *_cap_kill_then_reroute(5.12),
    {'event_type': 'invocation_end', 'run_id': 'run-813a', 'task_id': '5496',
     'phase': 'review', 'role': _REVIEWER, 'data': _CAP_KILL, 'cost_usd': 5.0},
    *_cap_kill_then_reroute(5.08),
    *_cap_kill_then_reroute(5.10),
    {'event_type': 'phase_enter', 'run_id': 'run-813a', 'task_id': '5496', 'phase': 'verify'},
    *_cap_kill_then_reroute(5.14),
    {'event_type': 'invocation_end', 'run_id': 'run-813a', 'task_id': TASK,
     'phase': 'review', 'role': _REVIEWER, 'data': {'subtype': 'success'}, 'cost_usd': 3.4},
    {'event_type': 'escalation_created', 'run_id': 'run-813a', 'task_id': TASK,
     'data': {'escalation_id': 'esc-5495-1'}},
    {'event_type': 'orchestrator_start', 'run_id': 'run-9f00', 'task_id': None},
    {'event_type': 'invocation_end', 'run_id': 'run-9f00', 'task_id': '5496',
     'role': 'unblock_auto', 'data': {}},
    {'event_type': 'invocation_end', 'run_id': 'run-9f00', 'task_id': TASK,
     'role': 'unblock_auto', 'data': {}},
]


def _timestamp(position):
    return (_BASE_TIME + timedelta(minutes=position)).isoformat()


def seed(conn, corpus):
    """Seed *corpus* (plain dicts of seed_event kwargs) with increasing timestamps.

    The fixture db is fresh and events.id is AUTOINCREMENT, so a row's
    1-based corpus position IS its id.
    """
    for position, row in enumerate(corpus, start=1):
        kwargs = {key: value for key, value in row.items() if key != 'event_type'}
        conn.seed_event(_timestamp(position), row['event_type'], **kwargs)


def _task_rows(corpus, task_id=TASK):
    return [(event_id, row) for event_id, row in enumerate(corpus, start=1)
            if row.get('task_id') == task_id]


def _run_json(capsys, db, *extra):
    exit_code = main(['--db', str(db), TASK, '--json', *extra])
    return exit_code, json.loads(capsys.readouterr().out)


def _run_text(capsys, db, *extra):
    exit_code = main(['--db', str(db), TASK, *extra])
    return exit_code, capsys.readouterr().out.splitlines()


def _tokens(line):
    return set(re.findall(r'[\w.-]+', line))


def _event_lines(lines):
    return [line for line in lines if line.startswith('#')]


def _filter_args(event_types):
    return [arg for event_type in event_types for arg in ('--event-type', event_type)]


def test_json_lists_every_event_of_the_task_across_runs_in_id_order(runs_db, runs_db_path, capsys):
    seed(runs_db, ESC_5495_3)

    exit_code, timeline = _run_json(capsys, runs_db_path)

    assert exit_code == 0
    assert timeline['db'] == str(runs_db_path.resolve())
    assert timeline['task_id'] == TASK
    assert timeline['runs'] == ['run-813a', 'run-9f00']
    expected = [
        {'id': event_id, 'timestamp': _timestamp(event_id), 'run_id': row['run_id'],
         'event_type': row['event_type'], 'phase': row.get('phase'), 'role': row.get('role'),
         'cost_usd': row.get('cost_usd'), 'duration_ms': row.get('duration_ms'),
         'data': row.get('data', {})}
        for event_id, row in _task_rows(ESC_5495_3)
    ]
    assert [event['id'] for event in timeline['events']] == [e['id'] for e in expected]
    for event, wanted in zip(timeline['events'], expected, strict=True):
        assert {key: event[key] for key in wanted} == wanted


def test_json_tallies_invocation_outcomes_by_role_and_subtype(runs_db, runs_db_path, capsys):
    seed(runs_db, ESC_5495_3)

    _, timeline = _run_json(capsys, runs_db_path)

    assert timeline['invocation_outcomes'] == [
        {'role': _REVIEWER, 'subtype': 'error_max_budget_usd', 'count': 4},
        {'role': _REVIEWER, 'subtype': 'success', 'count': 1},
        {'role': 'unblock_auto', 'subtype': None, 'count': 1},
    ]


def test_a_malformed_payload_is_listed_and_counted_with_empty_data(runs_db, runs_db_path, capsys):
    corpus = [*ESC_5495_3, {'event_type': 'invocation_end', 'run_id': 'run-9f00',
                            'task_id': TASK, 'role': 'verifier', 'data': '{not json'}]
    seed(runs_db, corpus)

    exit_code, timeline = _run_json(capsys, runs_db_path)

    assert exit_code == 0
    malformed_id = len(corpus)
    assert timeline['events'][-1]['id'] == malformed_id
    assert timeline['events'][-1]['data'] == {}
    assert len(timeline['events']) == len(_task_rows(corpus))
    assert timeline['invocation_outcomes'][-1] == {'role': 'verifier', 'subtype': None, 'count': 1}


def test_text_names_the_log_then_one_line_per_event_in_id_order(runs_db, runs_db_path, capsys):
    seed(runs_db, ESC_5495_3)

    exit_code, lines = _run_text(capsys, runs_db_path)

    assert exit_code == 0
    assert lines[0] == str(runs_db_path.resolve())
    rows = _task_rows(ESC_5495_3)
    event_lines = _event_lines(lines[1:])
    assert [line.split()[0] for line in event_lines] == [f'#{event_id}' for event_id, _ in rows]
    for line, (_, row) in zip(event_lines, rows, strict=True):
        assert {row['run_id'], row['event_type']} <= _tokens(line)
        subtype = row.get('data', {}).get('subtype')
        if subtype is not None:
            assert subtype in _tokens(line)
    assert sum('error_max_budget_usd' in _tokens(line) for line in event_lines) == 4
    assert sum('success' in _tokens(line) for line in event_lines) == 1


def test_text_summary_counts_events_runs_and_invocation_outcomes(runs_db, runs_db_path, capsys):
    seed(runs_db, ESC_5495_3)

    _, lines = _run_text(capsys, runs_db_path)

    last_event_line = lines.index(_event_lines(lines)[-1])
    summary, *outcome_block = [line for line in lines[last_event_line + 1:] if line.strip()]
    assert {str(len(_task_rows(ESC_5495_3))), '2'} <= _tokens(summary)
    roles = {_REVIEWER, 'unblock_auto'}
    outcome_lines = [line for line in outcome_block if roles & _tokens(line)]
    expected = [
        {_REVIEWER, 'error_max_budget_usd', '4'},
        {_REVIEWER, 'success', '1'},
        {'unblock_auto', '1'},
    ]
    assert len(outcome_lines) == len(expected)
    for line, wanted in zip(outcome_lines, expected, strict=True):
        assert wanted <= _tokens(line)


FILTERS = [
    pytest.param(['invocation_end'], 6, id='one-type'),
    pytest.param(['invocation_end', 'escalation_created'], 7, id='repeated-flag'),
]


@pytest.mark.parametrize(('event_types', 'expected_count'), FILTERS)
def test_event_type_filter_narrows_the_json_listing(
    runs_db, runs_db_path, capsys, event_types, expected_count,
):
    seed(runs_db, ESC_5495_3)

    exit_code, timeline = _run_json(capsys, runs_db_path, *_filter_args(event_types))

    assert exit_code == 0
    assert len(timeline['events']) == expected_count
    assert {event['event_type'] for event in timeline['events']} == set(event_types)


@pytest.mark.parametrize(('event_types', 'expected_count'), FILTERS)
def test_event_type_filter_narrows_the_text_listing(
    runs_db, runs_db_path, capsys, event_types, expected_count,
):
    seed(runs_db, ESC_5495_3)

    exit_code, lines = _run_text(capsys, runs_db_path, *_filter_args(event_types))

    assert exit_code == 0
    event_lines = _event_lines(lines)
    assert len(event_lines) == expected_count
    for line in event_lines:
        assert set(event_types) & _tokens(line)
