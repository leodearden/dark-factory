"""Boundary rows 1-8 of ``plans/decision-stream-prd.md`` §5.5, over real HTTP.

Task 8422. Each test owns a fresh queue directory, and with it that queue's
``policy-table-state.json``, served by conftest's ``serve_escalation_mcp``.
Identities are sent through ``_escalation_http.escalation_http_call`` (the one
place the header literals live) and taken from ``escalation.authority``.
Class switches are set through ``policy_state.set_class_state``, and every
read-back goes through ``queue.get``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from _escalation_http import escalation_http_call
from _escalation_seed import seed_escalation
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

from escalation.authority import COORDINATOR_IDENTITY, EXECUTOR_IDENTITY, ClassSwitch
from escalation.classify import classify_resolver_tier
from escalation.models import Escalation
from escalation.policy_state import policy_state_path, set_class_state
from escalation.queue import EscalationQueue
from escalation.store_identity import StoreIdentity

pytestmark = pytest.mark.asyncio

_TASK = '8422'
_LANDING = 'landing_provenance'
_LANDING_EVIDENCE = (
    'found_on_main at 1a2b3c4d; git merge-base --is-ancestor 1a2b3c4d main rc=0'
)
_CLAIM_FIELDS = ('policy_class', 'claimed_at', 'claimed_by')


@pytest.fixture
def served(tmp_path: Path, serve_escalation_mcp) -> tuple[str, EscalationQueue]:
    base_url, _port, queue = serve_escalation_mcp(tmp_path / 'escalations')
    return base_url, queue


def _landing_record(queue: EscalationQueue, **kw: Any) -> Escalation:
    kw.setdefault('category', 'provenance_unattributed')
    return seed_escalation(queue, level=2, task_id=_TASK, **kw)


def _executor_claim(age: timedelta, policy_class: str = _LANDING) -> dict[str, str]:
    return {
        'policy_class': policy_class,
        'claimed_by': EXECUTOR_IDENTITY,
        'claimed_at': (datetime.now(UTC) - age).isoformat(),
    }


async def _claim(
    base_url: str, escalation_id: str, policy_class: str = _LANDING, *,
    identity: str | None = EXECUTOR_IDENTITY, clean: bool = False,
) -> dict[str, Any]:
    return await escalation_http_call(
        base_url, 'claim_policy_class', identity=identity,
        escalation_id=escalation_id, policy_class=policy_class, clean=clean,
    )


async def _resolve(
    base_url: str, escalation_id: str, *, identity: str | None,
    resolution: str = _LANDING_EVIDENCE, resolution_class: str | None = 'actionable',
    action: str = 'close_only', **kw: Any,
) -> dict[str, Any]:
    return await escalation_http_call(
        base_url, 'resolve_issue', identity=identity,
        escalation_id=escalation_id, resolution=resolution, action=action,
        resolution_class=resolution_class, **kw,
    )


def _claim_of(queue: EscalationQueue, escalation_id: str) -> tuple[Any, ...]:
    record = queue.get(escalation_id)
    assert record is not None
    return tuple(getattr(record, field) for field in _CLAIM_FIELDS)


def _status(queue: EscalationQueue, escalation_id: str) -> str:
    record = queue.get(escalation_id)
    assert record is not None
    return record.status


async def _drain(base_url: str, tool_name: str, **tool_kwargs: Any) -> list[dict[str, Any]]:
    """A header-less list read's rows, from ``structured_content``.

    ``result.data`` is no use here: fastmcp turns each ``dict[str, Any]`` row
    of a list-returning tool into an opaque object with no fields.
    """
    async with Client(StreamableHttpTransport(f'{base_url}/mcp/')) as client:
        result = await client.call_tool(tool_name, tool_kwargs)
    assert result.structured_content is not None
    return result.structured_content['result']


class TestRow1ExecutorClosesALiveClassRecord:
    async def test_claim_then_close_stamps_the_executor_and_the_class(self, served):
        base_url, queue = served
        record = _landing_record(queue)
        set_class_state(queue.queue_dir, _LANDING, ClassSwitch.LIVE)

        claimed = await _claim(base_url, record.id)
        assert claimed.get('policy_class') == _LANDING, claimed
        assert claimed.get('claimed_by') == EXECUTOR_IDENTITY

        pending = await _drain(base_url, 'get_pending_escalations', compact=True)
        by_task = await _drain(base_url, 'get_task_escalations', task_id=_TASK, compact=True)
        for drain in (pending, by_task):
            (row,) = [r for r in drain if r['id'] == record.id]
            assert {field: row[field] for field in _CLAIM_FIELDS} == {
                'policy_class': _LANDING,
                'claimed_at': claimed['claimed_at'],
                'claimed_by': EXECUTOR_IDENTITY,
            }

        result = await _resolve(base_url, record.id, identity=EXECUTOR_IDENTITY)
        assert 'error' not in result, result

        closed = queue.get(record.id)
        assert closed is not None
        assert closed.status == 'dismissed'
        assert closed.resolved_by == EXECUTOR_IDENTITY
        assert closed.policy_class == _LANDING
        assert closed.resolution_class == 'actionable'

    async def test_an_executor_close_requires_a_resolution_class(self, served):
        base_url, queue = served
        record = _landing_record(queue)
        set_class_state(queue.queue_dir, _LANDING, ClassSwitch.LIVE)
        await _claim(base_url, record.id)

        result = await _resolve(
            base_url, record.id, identity=EXECUTOR_IDENTITY, resolution_class=None,
        )

        assert result.get('code') == 'resolution_class_required', result
        assert _status(queue, record.id) == 'pending'


class TestRow2ShadowClassIsRefused:
    async def test_a_fresh_claim_on_a_shadow_class_is_level_forbidden(self, served):
        base_url, queue = served
        record = _landing_record(queue)
        claimed = await _claim(base_url, record.id)
        assert claimed.get('policy_class') == _LANDING, claimed

        result = await _resolve(base_url, record.id, identity=EXECUTOR_IDENTITY)

        assert result.get('code') == 'level_forbidden', result
        assert result.get('policy_refusal') == 'class_not_live'
        assert _status(queue, record.id) == 'pending'


class TestRow3DenyRoleRecord:
    async def test_no_claim_and_no_close_on_a_watchdog_record(self, served):
        base_url, queue = served
        record = _landing_record(queue, agent_role='orchestrator-watchdog')
        set_class_state(queue.queue_dir, _LANDING, ClassSwitch.LIVE)

        claimed = await _claim(base_url, record.id)
        assert claimed.get('code') == 'level_forbidden', claimed
        assert _claim_of(queue, record.id) == (None, None, None)

        result = await _resolve(base_url, record.id, identity=EXECUTOR_IDENTITY)
        assert result.get('code') == 'level_forbidden', result
        assert _status(queue, record.id) == 'pending'


class TestRows4To6CoordinatorAgainstAClaim:
    async def test_row4_a_fresh_claim_refuses_the_coordinator(self, served):
        base_url, queue = served
        claim = _executor_claim(timedelta(minutes=10))
        record = _landing_record(queue, **claim)

        result = await _resolve(base_url, record.id, identity=COORDINATOR_IDENTITY)

        assert result.get('code') == 'claimed_by_executor', result
        assert result.get('policy_class') == _LANDING
        assert result.get('claimed_at') == claim['claimed_at']
        assert _status(queue, record.id) == 'pending'

    async def test_row5_a_record_the_executor_closed_is_left_as_it_closed_it(self, served):
        base_url, queue = served
        record = _landing_record(queue)
        set_class_state(queue.queue_dir, _LANDING, ClassSwitch.LIVE)
        await _claim(base_url, record.id)
        closed = await _resolve(base_url, record.id, identity=EXECUTOR_IDENTITY)
        assert 'error' not in closed, closed

        await _resolve(base_url, record.id, identity=COORDINATOR_IDENTITY)

        after = queue.get(record.id)
        assert after is not None
        assert after.status == 'dismissed'
        assert after.resolved_by == EXECUTOR_IDENTITY
        assert after.late_resolutions == [], 'row 5: nothing is written'

    async def test_row6_an_expired_claim_lets_the_coordinator_apply(self, served):
        base_url, queue = served
        record = _landing_record(queue, **_executor_claim(timedelta(hours=7)))

        result = await _resolve(base_url, record.id, identity=COORDINATOR_IDENTITY)
        assert 'error' not in result, result

        closed = queue.get(record.id)
        assert closed is not None
        assert closed.status == 'dismissed'
        assert closed.resolved_by == COORDINATOR_IDENTITY
        assert classify_resolver_tier(closed.resolved_by) == 'human'


class TestRow7HeaderLessConnection:
    async def test_a_header_less_resolve_ignores_a_fresh_claim(self, served):
        base_url, queue = served
        record = _landing_record(queue, **_executor_claim(timedelta(minutes=10)))

        result = await _resolve(
            base_url, record.id, identity=None, resolved_by='interactive',
        )
        assert 'error' not in result, result
        assert _status(queue, record.id) == 'dismissed'

    async def test_a_header_less_connection_cannot_claim(self, served):
        base_url, queue = served
        record = _landing_record(queue)

        claimed = await _claim(base_url, record.id, identity=None)

        assert claimed.get('code') == 'level_forbidden', claimed
        assert _claim_of(queue, record.id) == (None, None, None)


class TestRow8ClassBackToShadow:
    async def test_the_next_executor_close_after_a_flip_to_shadow_is_refused(self, served):
        base_url, queue = served
        record = _landing_record(queue)
        set_class_state(queue.queue_dir, _LANDING, ClassSwitch.LIVE)
        await _claim(base_url, record.id)
        set_class_state(queue.queue_dir, _LANDING, ClassSwitch.SHADOW)

        result = await _resolve(base_url, record.id, identity=EXECUTOR_IDENTITY)

        assert result.get('code') == 'level_forbidden', result
        assert result.get('policy_refusal') == 'class_not_live'
        assert _status(queue, record.id) == 'pending'


class TestClaimRefusals:
    async def test_a_second_claim_names_the_first(self, served):
        base_url, queue = served
        record = _landing_record(queue)
        first = await _claim(base_url, record.id)

        second = await _claim(base_url, record.id)

        assert second.get('code') == 'claim_refused', second
        assert second.get('policy_refusal') == 'already_claimed'
        assert second.get('policy_class') == _LANDING
        assert second.get('claimed_at') == first['claimed_at']

    async def test_an_off_class_cannot_be_claimed(self, served):
        base_url, queue = served
        record = _landing_record(queue)
        policy_state_path(queue.queue_dir).write_text('{"classes": {"landing_provenance": "off"}}')

        claimed = await _claim(base_url, record.id)

        assert claimed.get('code') == 'claim_refused', claimed
        assert claimed.get('policy_refusal') == 'class_off'
        assert _claim_of(queue, record.id) == (None, None, None)

    async def test_an_unknown_class_cannot_be_claimed(self, served):
        base_url, queue = served
        record = _landing_record(queue)

        claimed = await _claim(base_url, record.id, 'no_such_class')

        assert claimed.get('code') == 'claim_refused', claimed
        assert claimed.get('policy_refusal') == 'unknown_class'

    async def test_a_file_writing_class_needs_a_clean_project_root(self, served):
        base_url, queue = served
        record = seed_escalation(queue, level=2, task_id=_TASK, category='task_failure')

        dirty = await _claim(base_url, record.id, 'b3_low_risk_unblock')
        assert dirty.get('code') == 'claim_refused', dirty
        assert dirty.get('policy_refusal') == 'project_root_dirty'

        clean = await _claim(base_url, record.id, 'b3_low_risk_unblock', clean=True)
        assert clean.get('policy_class') == 'b3_low_risk_unblock', clean

    async def test_a_non_executor_identity_cannot_claim(self, served):
        base_url, queue = served
        record = _landing_record(queue)

        claimed = await _claim(base_url, record.id, identity=COORDINATOR_IDENTITY)

        assert claimed.get('code') == 'level_forbidden', claimed
        assert _claim_of(queue, record.id) == (None, None, None)


class TestStateFileAndPromotion:
    async def test_an_unreadable_state_file_refuses_the_executor(self, served):
        base_url, queue = served
        record = _landing_record(queue, **_executor_claim(timedelta(minutes=10)))
        policy_state_path(queue.queue_dir).write_text('not json')

        result = await _resolve(base_url, record.id, identity=EXECUTOR_IDENTITY)

        assert result.get('code') == 'policy_state_unreadable', result
        assert _status(queue, record.id) == 'pending'

    @pytest.mark.parametrize('identity', [EXECUTOR_IDENTITY, COORDINATOR_IDENTITY])
    async def test_promote_to_l2_admits_the_executor_and_the_coordinator(
        self, served, identity: str,
    ):
        base_url, queue = served
        member = seed_escalation(queue, level=1, task_id=_TASK)

        result = await escalation_http_call(
            base_url, 'promote_to_l2', identity=identity,
            task_id=_TASK, agent_role='escalation-executor', member_ids=[member.id],
            root_cause=f'rc-8422-{identity}', evidence='e', options=['A', 'B'],
            summary='cluster',
        )

        assert result.get('code') != 'level_forbidden', result
        assert result.get('status') in {'created', 'updated'}, result


@pytest.fixture
def served_recon(tmp_path: Path, serve_escalation_mcp) -> tuple[str, EscalationQueue]:
    recon_dir = tmp_path / 'recon-escalations'
    base_url, _port, queue = serve_escalation_mcp(
        recon_dir,
        store_identity=StoreIdentity(
            kind='reconciliation', queue_dir=recon_dir, project_id=None, project_root=None,
        ),
    )
    return base_url, queue


class TestReconQueue:
    async def test_a_recon_row_is_claimable_only_on_a_recon_queue(
        self, served, served_recon,
    ):
        project_url, project_queue = served
        recon_url, recon_queue = served_recon
        on_recon = seed_escalation(recon_queue, level=2, task_id=_TASK, category='recon_stale_run')
        on_project = seed_escalation(
            project_queue, level=2, task_id=_TASK, category='recon_stale_run',
        )

        claimed = await _claim(recon_url, on_recon.id, 'recon_run_self_cleared')
        refused = await _claim(project_url, on_project.id, 'recon_run_self_cleared')

        assert claimed.get('policy_class') == 'recon_run_self_cleared', claimed
        assert refused.get('code') == 'claim_refused', refused
        assert refused.get('policy_refusal') == 'row_mismatch'
