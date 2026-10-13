"""GET /api/v2/dashboard/invariants end to end, over real per-root tasks.db files.

Boundary tests B1-B4 of docs/prds/claimant-invariant-detection.md, and the
recurrence chain section's B5 (read side) and B8 of
docs/prds/recurring-deterministic-tasks.md, read through the real route, the
lifespan's real DbPool, real escalation files and a tmp-root roster.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from _tasks_db_fixture import task_row, write_tasks_db
from escalation.models import Escalation

from dashboard.config import DashboardConfig
from dashboard.data import escalation_corpus

_URL = '/api/v2/dashboard/invariants'
_E1_KEYS = {
    'root',
    'project',
    'status',
    'error',
    'rows_scanned',
    'measured_at',
    'terminal_tier',
    'hygiene_tier',
    'chains',
}
_COUNT_KEYS = {'count', 'unledgered_count', 'stale_count'}
_CLAIMANT = 'run-a/s/pid=1'
_STALE = (datetime.now(UTC) - timedelta(hours=6)).isoformat()


@pytest.fixture(autouse=True)
def _fresh_escalation_corpus() -> Iterator[None]:
    escalation_corpus._corpus_cache_clear()
    yield
    escalation_corpus._corpus_cache_clear()


def _real_db(root: Path) -> Path:
    return root / '.taskmaster' / 'tasks' / 'tasks.db'


def _roots(tmp_path: Path) -> tuple[Path, Path, Path]:
    a, b, c = (tmp_path / name for name in ('proj-a', 'proj-b', 'proj-c'))
    for root in (a, b, c):
        root.mkdir()
    return a.resolve(), b.resolve(), c.resolve()


def _get(client, primary: Path, known: list[Path]) -> dict[str, Any]:
    client.app.state.config = DashboardConfig(project_root=primary, known_project_roots=known)
    resp = client.get(_URL)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _by_root(body: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {record['root']: record for record in body['INVARIANTS']}


def _keys_anywhere(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _keys_anywhere(child)
    elif isinstance(value, list):
        for child in value:
            yield from _keys_anywhere(child)


def test_b1_roster_is_primary_first_deduplicated_and_reports_the_violation(client, tmp_path):
    a, b, c = _roots(tmp_path)
    write_tasks_db(
        _real_db(a),
        [task_row(1, 'done', _CLAIMANT, _STALE), task_row(2, 'done'), task_row(3, 'pending')],
    )
    body = _get(client, a, [a, b, c])

    assert [record['root'] for record in body['INVARIANTS']] == [str(a), str(b), str(c)]
    for record in body['INVARIANTS']:
        assert set(record) >= _E1_KEYS
        assert record['measured_at'] == body['served_at']
    datetime.fromisoformat(body['served_at'])

    record_a = _by_root(body)[str(a)]
    assert record_a['status'] == 'ok'
    assert record_a['project'] == a.name
    assert record_a['rows_scanned'] > 0
    [violation] = record_a['terminal_tier']['violations']
    assert violation['task_id'] == 1
    assert violation['ledgered'] is False


def test_b2_bare_taskmaster_decoy_is_never_read(client, tmp_path):
    a, b, _ = _roots(tmp_path)
    write_tasks_db(_real_db(a), [task_row(1, 'pending')])
    write_tasks_db(b / '.taskmaster' / 'tasks.db', [task_row(1, 'done', _CLAIMANT, _STALE)])
    records = _by_root(_get(client, a, [b]))

    record_b = records[str(b)]
    assert record_b['status'] == 'blind'
    assert str(_real_db(b)) in record_b['error']
    assert record_b['terminal_tier'] is None
    assert records[str(a)]['status'] == 'ok'


@pytest.mark.skipif(os.geteuid() == 0, reason='root ignores file mode bits')
def test_b2_unreadable_db_is_blind(client, tmp_path):
    a, _, c = _roots(tmp_path)
    write_tasks_db(_real_db(a), [task_row(1, 'pending')])
    db_c = write_tasks_db(_real_db(c), [task_row(1, 'done', _CLAIMANT, _STALE)])
    db_c.chmod(0o000)
    try:
        records = _by_root(_get(client, a, [c]))
    finally:
        db_c.chmod(0o644)

    record_c = records[str(c)]
    assert record_c['status'] == 'blind'
    assert str(db_c) in record_c['error']
    assert record_c['terminal_tier'] is None
    assert record_c['hygiene_tier'] is None
    assert records[str(a)]['status'] == 'ok'


def test_b3_b4_tiers_reach_the_wire(client, tmp_path):
    a, _, _ = _roots(tmp_path)
    fresh = datetime.now(UTC).isoformat()
    write_tasks_db(
        _real_db(a),
        [task_row(1, 'pending', _CLAIMANT, _STALE), task_row(2, 'done', None, fresh)],
    )
    record_a = _by_root(_get(client, a, []))[str(a)]

    assert record_a['hygiene_tier']['stale_count'] == 1
    assert 2 not in [v['task_id'] for v in record_a['terminal_tier']['violations']]


def test_an_all_blind_roster_makes_no_count_claim_anywhere(client, tmp_path):
    a, b, c = _roots(tmp_path)
    body = _get(client, a, [b, c])

    assert [record['status'] for record in body['INVARIANTS']] == ['blind'] * 3
    assert _COUNT_KEYS.isdisjoint(_keys_anywhere(body))
    assert all(record['chains'] is None for record in body['INVARIANTS'])


def _link(link_id: int, status: str, key: str, at: datetime) -> dict[str, Any]:
    metadata = {
        'recurrence': {'key': key, 'interval_secs': 3600},
        'milestone': {'mode': 'dated', 'at': at.isoformat()},
    }
    return task_row(link_id, status, metadata=metadata)


def _write_escalation(root: Path, esc_id: str, task_id: str) -> None:
    record = Escalation(
        id=esc_id,
        task_id=task_id,
        agent_role='implementer',
        severity='blocking',
        category='design_concern',
        summary=f'summary of {esc_id}',
        timestamp='2026-10-12T00:00:00+00:00',
    ).to_dict()
    queue_dir = root / 'data' / 'escalations'
    queue_dir.mkdir(parents=True, exist_ok=True)
    (queue_dir / f'{esc_id}.json').write_text(json.dumps(record))


def _seed_every_chain_state(root: Path) -> None:
    now = datetime.now(UTC)
    write_tasks_db(
        _real_db(root),
        [
            _link(1, 'pending', 'future-job', now + timedelta(hours=2)),
            _link(2, 'pending', 'late-job', now - timedelta(hours=3)),
            _link(3, 'done', 'crashed-mint', now - timedelta(hours=30)),
            _link(4, 'done', 'renewed-job', now - timedelta(hours=30)),
            _link(5, 'pending', 'renewed-job', now + timedelta(hours=2)),
            _link(6, 'cancelled', 'stopped-job', now - timedelta(hours=30)),
            _link(7, 'blocked', 'failing-job', now - timedelta(hours=3)),
            task_row(8, 'pending'),
        ],
    )


def _chains_by_key(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    chains = {chain['key']: chain for chain in record['chains']}
    assert len(chains) == len(record['chains']), 'a chain key appears more than once'
    return chains


def test_b5_b8_every_chain_reaches_the_wire_in_its_state(client, tmp_path):
    a, b, _ = _roots(tmp_path)
    _seed_every_chain_state(a)
    _write_escalation(a, 'esc-7-1', task_id='7')
    body = _get(client, a, [b])

    record_a = _by_root(body)[str(a)]
    assert record_a['measured_at'] == body['served_at']
    chains = _chains_by_key(record_a)
    assert {key: (chain['state'], chain['link_id']) for key, chain in chains.items()} == {
        'crashed-mint': ('broken', 3),
        'failing-job': ('paused', 7),
        'future-job': ('scheduled', 1),
        'late-job': ('overdue', 2),
        'renewed-job': ('scheduled', 5),
        'stopped-job': ('ended', 6),
    }
    assert chains['failing-job']['pending_escalation_ids'] == ['esc-7-1']
    assert chains['future-job']['pending_escalation_ids'] == []
    assert _by_root(body)[str(b)]['chains'] is None


def test_a_partially_read_escalation_queue_leaves_the_pinning_escalation_unknown(
    client, tmp_path
):
    a, _, _ = _roots(tmp_path)
    _seed_every_chain_state(a)
    _write_escalation(a, 'esc-7-1', task_id='7')
    (a / 'data' / 'escalations' / 'esc-7-2.json').write_text('{not json')
    record_a = _by_root(_get(client, a, []))[str(a)]

    chains = _chains_by_key(record_a)
    assert chains['failing-job']['state'] == 'paused'
    assert chains['failing-job']['pending_escalation_ids'] is None


def test_a_failed_escalation_corpus_walk_costs_no_root_its_record(client, tmp_path, monkeypatch):
    a, b, _ = _roots(tmp_path)
    _seed_every_chain_state(a)
    _write_escalation(a, 'esc-7-1', task_id='7')
    write_tasks_db(_real_db(b), [task_row(1, 'done', _CLAIMANT, _STALE)])

    def _denied(queues):
        raise PermissionError(13, 'Permission denied', str(a / 'data' / 'escalations'))

    monkeypatch.setattr(escalation_corpus, 'walk_corpus', _denied)
    records = _by_root(_get(client, a, [b]))

    assert [records[str(root)]['status'] for root in (a, b)] == ['ok', 'ok']
    [violation] = records[str(b)]['terminal_tier']['violations']
    assert violation['task_id'] == 1
    chains = _chains_by_key(records[str(a)])
    assert chains['failing-job']['state'] == 'paused'
    assert all(chain['pending_escalation_ids'] is None for chain in chains.values())
