"""A silent promote of an all-info cluster is refused, not minted (task 5222).

PRD ``plans/info-l0-disposition-router-prd.md`` D4 — "All-info clusters never
enter L2" — and its boundary rows #9 (an explicit severity still promotes) and
#10 (a silent all-info promote returns ``info_only_cluster`` and writes no
record).  The server judges nothing about the cluster; it only enforces the
contract that an L2 the members themselves call informational is never put in
front of a human by DEFAULT.

The regression this closes, end to end: esc-3037-1 (L0, info, architect) ->
esc-3037-3 (L1, info, harness-orphan-reaper, severity faithfully preserved) ->
esc-3037-4 (L2, *blocking*, escalation-watcher-auto, members=['esc-3037-3'],
later closed as benign).  Task 3976 stopped that inflation by making an omitted
severity INHERIT max(member severities), which turned the human-paging blocking
L2 into an info L2 — still a human-queue record for a cluster nobody needed to
read.  D4 removes the record itself: the auto-watcher closes the members at L1
with a digest instead, or re-issues with an explicit severity when its RCA
judges the cluster collectively blocking.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from _filing_tools import call_promote

from escalation.models import Escalation
from escalation.queue import EscalationQueue
from escalation.server import create_server

ROOT_CAUSE = 'informational architect notes about one module'
EVAL_FIXTURE_TASK_ID = 'df_task_2430_adv_plan'


def _promote_args(member_ids: list[str], **overrides: Any) -> dict[str, Any]:
    return {
        'task_id': '5222',
        'agent_role': 'escalation-watcher-auto',
        'member_ids': member_ids,
        'root_cause': ROOT_CAUSE,
        'evidence': 'every member is an informational note',
        'options': ['A: close the members at L1', 'B: convert to a task'],
        'summary': 'info-only cluster',
        **overrides,
    }


def _seed_member(
    queue: EscalationQueue, esc_id: str, *, severity: str, task_id: str = '5222',
) -> Escalation:
    member = Escalation(
        id=esc_id,
        task_id=task_id,
        agent_role='harness-orphan-reaper',
        severity=severity,
        category='design_concern',
        summary=f'cluster member {esc_id}',
        level=1,
    )
    queue.submit(member)
    return member


def _level_two_records(queue: EscalationQueue) -> list[Escalation]:
    records = (
        Escalation.from_json(path.read_text())
        for path in queue.queue_dir.rglob('esc-*.json')
    )
    return [record for record in records if record.level == 2]


@pytest.fixture
def queue(tmp_path: Path) -> EscalationQueue:
    return EscalationQueue(tmp_path / 'esc')


@pytest.fixture
def server(queue: EscalationQueue):
    return create_server(queue, startup_sweep=False)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'member_ids',
    [['esc-info-1'], ['esc-info-1', 'esc-info-2'], ['esc-info-1', 'esc-info-1']],
    ids=['one-member', 'two-members', 'duplicated-id'],
)
async def test_silent_all_info_promote_is_refused(queue, server, member_ids):
    distinct = list(dict.fromkeys(member_ids))
    for esc_id in distinct:
        _seed_member(queue, esc_id, severity='info')

    result = await call_promote(server, **_promote_args(member_ids))

    assert result == {
        'status': 'refused',
        'code': 'info_only_cluster',
        'members': distinct,
        'derived_severity': 'info',
    }
    assert _level_two_records(queue) == []
    for esc_id in distinct:
        member = queue.get(esc_id)
        assert member is not None
        assert (member.status, member.level, member.severity) == ('pending', 1, 'info')


@pytest.mark.asyncio
async def test_refusal_never_folds_into_an_existing_pending_l2(queue, server):
    _seed_member(queue, 'esc-blk-1', severity='blocking')
    created = await call_promote(server, **_promote_args(['esc-blk-1']))
    assert created['status'] == 'created', created
    before = queue.get(created['id'])
    assert before is not None
    _seed_member(queue, 'esc-info-1', severity='info')

    result = await call_promote(server, **_promote_args(['esc-info-1']))

    assert result['status'] == 'refused', result
    assert result['code'] == 'info_only_cluster'
    after = queue.get(created['id'])
    assert after is not None
    assert after.members == before.members
    assert after.severity == before.severity
    assert after.amendments == before.amendments
    assert after.updated_at == before.updated_at
    assert [record.id for record in _level_two_records(queue)] == [created['id']]


@pytest.mark.asyncio
@pytest.mark.parametrize('explicit', ['blocking', 'critical'])
async def test_explicit_severity_still_promotes_an_all_info_cluster(
    queue, server, explicit,
):
    _seed_member(queue, 'esc-blk-1', severity='blocking')
    created = await call_promote(server, **_promote_args(['esc-blk-1']))
    assert created['status'] == 'created', created
    _seed_member(queue, 'esc-info-1', severity='info')

    result = await call_promote(
        server, **_promote_args(['esc-info-1'], severity=explicit),
    )

    assert result['status'] == 'updated', result
    assert result['id'] == created['id']
    assert 'esc-info-1' in result['members']


@pytest.mark.asyncio
async def test_explicit_info_severity_still_mints_an_info_l2(queue, server):
    """The server enforces only the SILENT case.

    Re-issuing with an explicit ``severity='info'`` to get past the refusal is
    forbidden by the auto-watcher skill, not by the server: an explicit value is
    the caller's authority at mint time, in both directions.
    """
    _seed_member(queue, 'esc-info-1', severity='info')

    result = await call_promote(
        server, **_promote_args(['esc-info-1'], severity='info'),
    )

    assert result['status'] == 'created', result
    record = queue.get(result['id'])
    assert record is not None
    assert (record.level, record.severity) == (2, 'info')


@pytest.mark.asyncio
async def test_eval_lane_refusal_takes_precedence(queue, server):
    _seed_member(queue, 'esc-eval-1', severity='info', task_id=EVAL_FIXTURE_TASK_ID)
    _seed_member(queue, 'esc-eval-2', severity='info', task_id=EVAL_FIXTURE_TASK_ID)

    result = await call_promote(
        server, **_promote_args(['esc-eval-1', 'esc-eval-2']),
    )

    assert result.get('code') == 'eval_lane_contained', result
    assert _level_two_records(queue) == []
