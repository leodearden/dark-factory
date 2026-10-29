"""l1_to_l2_promotion measures each L2 member from the time it joined the cluster.

A member present when the L2 was created joined at the L2's own timestamp. A
member a later promote_to_l2 folded in joined at its ``member_folded_at`` stamp.
A member filed after its L2 with no stamp is a fold from before the stamp
existed: its join time is unknown, so it is counted, not measured or logged.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from escalation.models import Escalation
from escalation.queue import EscalationQueue

from dashboard.data.escalation_analytics import build_escalation_analytics
from dashboard.data.escalation_corpus import QueueKind, QueueRef, measure_corpus

_LOGGER = 'dashboard.data.escalation_analytics'
_NOW = datetime(2026, 10, 28, 12, 0, tzinfo=UTC)
_L2_FILED_AT = _NOW - timedelta(days=1)


def _promotion(esc_dir: Path, runs_db: Path, *, now: datetime) -> dict:
    queue = QueueRef(
        id=str(esc_dir), label='p', kind=QueueKind.ORCHESTRATOR, directory=esc_dir,
        runs_db=runs_db,
    )
    payload = build_escalation_analytics(measure_corpus((queue,), now=now))
    return payload['per_project'][0]['lifespan']['l1_to_l2_promotion']


def _write_pending(esc_dir: Path, record: dict) -> None:
    esc_dir.mkdir(parents=True, exist_ok=True)
    (esc_dir / f"{record['id']}.json").write_text(json.dumps(record))


def _record(esc_id: str, *, filed_at: datetime, level: int, **fields) -> dict:
    return {
        'id': esc_id, 'task_id': esc_id.split('-')[1], 'agent_role': 'steward',
        'severity': 'blocking', 'category': 'design_concern', 'summary': esc_id,
        'timestamp': filed_at.isoformat(), 'status': 'pending', 'level': level,
        **fields,
    }


def _write_cluster(esc_dir: Path, *, folded_member_filed_at: datetime, **l2_fields) -> None:
    _write_pending(esc_dir, _record(
        'esc-300-1', filed_at=_L2_FILED_AT, level=2,
        members=['esc-300-0', 'esc-301-1'], **l2_fields,
    ))
    _write_pending(esc_dir, _record(
        'esc-300-0', filed_at=_L2_FILED_AT - timedelta(hours=1), level=1,
    ))
    _write_pending(esc_dir, _record(
        'esc-301-1', filed_at=folded_member_filed_at, level=1,
    ))


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage() for r in caplog.records
        if r.name == _LOGGER and r.levelno >= logging.WARNING
    ]


def test_folded_member_is_measured_from_its_fold_time(tmp_path, caplog):
    esc_dir = tmp_path / 'esc'
    filed_at = _L2_FILED_AT + timedelta(hours=2)
    _write_cluster(
        esc_dir, folded_member_filed_at=filed_at,
        member_folded_at={'esc-301-1': (filed_at + timedelta(minutes=30)).isoformat()},
    )

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        promotion = _promotion(esc_dir, tmp_path / 'runs.db', now=_NOW)

    assert promotion == {
        'count': 2, 'p50_secs': 2700.0, 'p90_secs': 3420.0, 'fold_time_unknown': 0,
    }
    assert _warnings(caplog) == []


def test_unstamped_member_filed_after_its_l2_is_counted_not_warned(tmp_path, caplog):
    esc_dir = tmp_path / 'esc'
    _write_cluster(esc_dir, folded_member_filed_at=_L2_FILED_AT + timedelta(hours=2))

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        promotion = _promotion(esc_dir, tmp_path / 'runs.db', now=_NOW)

    assert promotion == {
        'count': 1, 'p50_secs': 3600.0, 'p90_secs': 3600.0, 'fold_time_unknown': 1,
    }
    assert _warnings(caplog) == []


@pytest.mark.parametrize('corrupt', [['esc-301-1'], 'esc-301-1'], ids=['list', 'string'])
def test_a_non_mapping_fold_stamp_skips_its_cluster_with_a_warning(tmp_path, caplog, corrupt):
    esc_dir = tmp_path / 'esc'
    _write_cluster(
        esc_dir, folded_member_filed_at=_L2_FILED_AT + timedelta(hours=2),
        member_folded_at=corrupt,
    )

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        promotion = _promotion(esc_dir, tmp_path / 'runs.db', now=_NOW)

    assert promotion == {
        'count': 0, 'p50_secs': None, 'p90_secs': None, 'fold_time_unknown': 0,
    }
    assert len(_warnings(caplog)) == 1
    assert 'esc-300-1' in _warnings(caplog)[0]


def _l1(queue: EscalationQueue, task_id: str) -> Escalation:
    esc = Escalation(
        id=queue.make_id(task_id), task_id=task_id, agent_role='steward',
        severity='blocking', category='design_concern', summary=f'member {task_id}',
        level=1,
    )
    queue.submit(esc)
    return esc


def test_the_fold_writer_feeds_the_reader(tmp_path, caplog):
    esc_dir = tmp_path / 'esc'
    queue = EscalationQueue(esc_dir)
    created_with = _l1(queue, '400')
    l2 = Escalation(
        id=queue.make_id('401'), task_id='401', agent_role='escalation-watcher-auto',
        severity='blocking', category='design_concern', summary='L2 cluster', level=2,
        root_cause='Bad merge strategy', members=[created_with.id],
    )
    queue.submit(l2)
    folded_later = _l1(queue, '402')
    assert queue.add_members_to_l2(l2.id, [folded_later.id]) is not None

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        promotion = _promotion(esc_dir, tmp_path / 'runs.db', now=datetime.now(UTC))

    assert promotion['count'] == 2
    assert promotion['fold_time_unknown'] == 0
    assert promotion['p50_secs'] >= 0
    assert _warnings(caplog) == []
