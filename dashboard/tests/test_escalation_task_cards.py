"""An /escalations task card carries only the task fields the Escalations drawer reads.

The drawer (``tab_escalations.jsx::EscalationSidebar``) reads a card's id,
title and status; it fetches the task's prose on demand from
``api/task_prose.py``. Everything else a task row holds is payload nobody reads.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
from _canned_mcp import CannedMCP, _raw_row
from _served_reads import forget_served_reads
from escalation.models import Escalation
from escalation.queue import EscalationQueue

from dashboard.config import DashboardConfig
from dashboard.data import redux_api, task_lookup
from dashboard.data.datum import (
    Datum,
    DatumContractError,
    DatumInvariant,
    DatumState,
    unknown_datum,
)
from dashboard.data.escalation_corpus import EscalationView


@pytest.fixture(autouse=True)
def _forget_route_reads():
    """Forget the stores the route serves from: the suite's served reads, then the per-id lookup."""
    def _forget() -> None:
        forget_served_reads()
        task_lookup._lookup_cache_clear()

    _forget()
    yield
    _forget()


def _config_filing(root, task_ids: dict[str, str]) -> DashboardConfig:
    """A config over *root* whose queue holds one pending escalation per ``esc_id -> task_id``."""
    config = DashboardConfig(project_root=root, known_project_roots=[], escalation_urls={})
    queue = EscalationQueue(config.escalations_dir)
    for esc_id, task_id in task_ids.items():
        queue.submit(Escalation(
            id=esc_id, task_id=task_id, agent_role='implementer', severity='blocking',
            category='design_concern', summary=f'summary of {esc_id}',
            timestamp='2026-09-01T00:00:00+00:00', status='pending',
        ))
    config.reconciliation_escalations_dir.mkdir(parents=True, exist_ok=True)
    return config


def _task(body: dict, esc_id: str) -> dict:
    return next(
        row['task']
        for sub in body['ESCALATIONS']['subsections']
        for row in sub['escalations']
        if row['id'] == esc_id
    )


def test_a_card_carries_only_id_title_and_status(client, tmp_path):
    """A terminal card (one get_task) and an active card (the snapshot) both ship three fields."""
    client.app.state.config = _config_filing(tmp_path, {'esc-3-1': '3', 'esc-11-1': '11'})
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


def _queues(*esc_ids: str) -> dict:
    """``build_escalation_queues`` output: one projA subsection holding a pending row per id."""
    rows = [
        {
            'id': esc_id, 'task_id': '1', 'level': 0, 'status': 'pending',
            'summary': 'oops', 'project': 'projA', 'project_root': '/p/projA',
        }
        for esc_id in esc_ids
    ]
    return {
        'subsections': [{
            'id': '/p/projA', 'label': 'projA', 'kind': 'orchestrator',
            'escalations': rows, 'skipped': [],
            'summary': dict(_SUMMARY), 'views': _views(),
        }],
        'summary': dict(_SUMMARY),
        'views': _views(),
    }


def test_a_projected_card_keeps_its_provenance():
    """A stale card is projected and stays stale; an unknown card stays unknown, saying why."""
    queues = _queues('esc-1-1', 'esc-2-1')
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


def test_a_card_lacking_a_drawer_field_names_its_row():
    """A card value without ``title`` is a wiring bug, reported against the row it broke."""
    queues = _queues('esc-1-1')
    cards = {
        ('/p/projA', 'esc-1-1'): Datum(
            {'id': 1, 'status': 'pending'}, _MEASURED_AT, DatumState.FRESH, None, 10,
        ),
    }

    with pytest.raises(DatumContractError) as raised:
        redux_api.shape_escalations(queues, cards, served_at=_MEASURED_AT)

    assert raised.value.invariant is DatumInvariant.VALUE_CONTRACT
    assert 'ESCALATIONS projA esc-1-1 task' in str(raised.value)
    assert 'title' in str(raised.value)
