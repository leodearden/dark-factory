"""An /escalations task card carries only the task fields the Escalations drawer reads.

The drawer (``tab_escalations.jsx::EscalationSidebar``) reads a card's id,
title and status; it fetches the task's prose on demand from
``api/task_prose.py``. Everything else a task row holds is payload nobody reads.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
from _canned_mcp import CannedMCP, _raw_row

from dashboard.config import DashboardConfig
from dashboard.data import redux_api
from dashboard.data.datum import Datum, DatumState, unknown_datum
from dashboard.data.escalation_corpus import EscalationView

_WIRE_KEYS = {'value', 'as_of', 'state', 'reason', 'freshness_bound_seconds'}


@pytest.fixture(autouse=True)
def _escalation_caches():
    """Clear the corpus, the analytics memo, the snapshot units and the per-id cache."""
    from dashboard.api import escalations as escalation_routes
    from dashboard.data import escalation_corpus, task_lookup, task_snapshot

    def _clear() -> None:
        escalation_corpus._corpus_cache_clear()
        escalation_routes._analytics_memo_clear()
        task_snapshot.clear_snapshot_cache()
        task_lookup._lookup_cache_clear()

    _clear()
    yield
    _clear()


def _config(root) -> DashboardConfig:
    """A config over *root* whose every queue dir exists, so a clean walk is fresh."""
    config = DashboardConfig(project_root=root, known_project_roots=[], escalation_urls={})
    config.escalations_dir.mkdir(parents=True, exist_ok=True)
    config.reconciliation_escalations_dir.mkdir(parents=True, exist_ok=True)
    return config


def _write_escalation(config: DashboardConfig, esc_id: str, *, task_id: str) -> None:
    from escalation.models import Escalation

    record = Escalation(
        id=esc_id, task_id=task_id, agent_role='implementer', severity='blocking',
        category='design_concern', summary=f'summary of {esc_id}',
        timestamp='2026-09-01T00:00:00+00:00', status='pending',
    ).to_dict()
    (config.escalations_dir / f'{esc_id}.json').write_text(json.dumps(record))


def _task(body: dict, esc_id: str) -> dict:
    return next(
        row['task']
        for sub in body['ESCALATIONS']['subsections']
        for row in sub['escalations']
        if row['id'] == esc_id
    )


def test_a_card_carries_only_id_title_and_status(client, tmp_path):
    """A terminal card (one get_task) and an active card (the snapshot) both ship three fields."""
    config = _config(tmp_path)
    _write_escalation(config, 'esc-3-1', task_id='3')
    _write_escalation(config, 'esc-11-1', task_id='11')
    client.app.state.config = config
    canned = CannedMCP(
        rows=[
            _raw_row(
                3, 'done',
                description='d' * 500, details='x', priority='high', dependencies=['11'],
                metadata={
                    'dry_run_proposals': [{'proposal_text': 'p' * 500}],
                    'memory_hints': {'queries': ['q']},
                    'files': ['a.py'],
                },
            ),
            _raw_row(11, 'in-progress', priority='medium', metadata={'files': ['b.py']}),
        ],
        status_map={3: 'done', 11: 'in-progress'},
        status_page_size=2000,
    )

    with patch('dashboard.data.tasks.mcp_tool_call', new=canned):
        resp = client.get('/api/v2/dashboard/escalations')

    assert resp.status_code == 200
    body = resp.json()
    for esc_id, card in (
        ('esc-3-1', {'id': 3, 'title': 'task 3', 'status': 'done'}),
        ('esc-11-1', {'id': 11, 'title': 'task 11', 'status': 'in-progress'}),
    ):
        task = _task(body, esc_id)
        assert set(task) == _WIRE_KEYS, esc_id
        assert task['state'] == 'fresh', esc_id
        assert task['value'] == card, esc_id


_MEASURED_AT = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
_SUMMARY = {
    'by_level': {0: 0, 1: 0, 2: 0},
    'by_status': {'pending': 0, 'resolved': 0, 'dismissed': 0},
    'skipped_count': 0,
}


def _views() -> dict:
    return {view: Datum(0, _MEASURED_AT, DatumState.FRESH, None, 120) for view in EscalationView}


def _esc_row(esc_id: str) -> dict:
    return {
        'id': esc_id, 'task_id': '1', 'level': 0, 'status': 'pending',
        'summary': 'oops', 'project': 'projA', 'project_root': '/p/projA',
    }


def test_a_projected_card_keeps_its_provenance():
    """A stale card is projected and stays stale; an unknown card stays unknown, saying why."""
    queues = {
        'subsections': [{
            'id': '/p/projA', 'label': 'projA', 'kind': 'orchestrator',
            'escalations': [_esc_row('esc-1-1'), _esc_row('esc-2-1')], 'skipped': [],
            'summary': dict(_SUMMARY), 'views': _views(),
        }],
        'summary': dict(_SUMMARY),
        'views': _views(),
    }
    whole_row = {
        'id': 1, 'title': 'one', 'status': 'pending', 'priority': 'high',
        'dependencies': [2], 'updated_at': '2026-10-01T11:00:00+00:00',
        'metadata': {'memory_hints': {'queries': ['q']}, 'files': ['a.py']},
        'claimant_run_id': None, 'heartbeat_at': None,
        'created_at': None, 'created_at_source': None,
    }
    cards = {
        ('/p/projA', 'esc-1-1'): Datum(whole_row, _MEASURED_AT, DatumState.FRESH, None, 10),
        ('/p/projA', 'esc-2-1'): unknown_datum('no task id', 1200),
    }

    body = redux_api.shape_escalations(
        queues, cards, served_at=_MEASURED_AT + timedelta(seconds=60),
    )
    stale, unknown = (row['task'] for row in body['ESCALATIONS']['subsections'][0]['escalations'])

    assert stale['state'] == 'stale'
    assert stale['reason']
    assert stale['value'] == {'id': 1, 'title': 'one', 'status': 'pending'}
    assert (unknown['value'], unknown['state'], unknown['reason']) == (None, 'unknown', 'no task id')
