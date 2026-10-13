"""``resolution_action`` is written by ``EscalationQueue.resolve``, inside its lock.

The stamp lands in the same per-id critical section as the status
check-and-set, so it is atomic with the transition it describes.  No caller
pre-writes a snapshot of the record to carry it there.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from escalation.models import Escalation
from escalation.queue import EscalationQueue, ResolveOutcome
from escalation.server import create_server

SWEEP_TEXT = 'Auto-dismissed: steward interrupted (attempt cap)'
LATE_TEXT = "the steward's real finding: verify never ran"
STEWARD = 'claude-task-3902-steward'


def _seed_pending(
    queue: EscalationQueue, esc_id: str = 'esc-3902-1', **fields: Any,
) -> str:
    queue.submit(Escalation(**{
        'id': esc_id,
        'task_id': '3902',
        'agent_role': 'implementer',
        'severity': 'blocking',
        'category': 'task_failure',
        'summary': 'a pending L0',
        **fields,
    }))
    return esc_id


def _interleave_after_first_read(
    monkeypatch: pytest.MonkeyPatch,
    queue: EscalationQueue,
    target_id: str,
    concurrent_op: Callable[[], object],
) -> None:
    """Run *concurrent_op* right after the first ``queue.get(target_id)``.

    That first read still returns the record as it was BEFORE the op, which is
    the gate snapshot ``resolve_issue`` takes going stale under a concurrent
    writer.  Every later read, the op's own included, passes straight through.
    """
    real_get = queue.get
    armed = True

    def get_then_race(escalation_id: str) -> Escalation | None:
        nonlocal armed
        snapshot = real_get(escalation_id)
        if armed and escalation_id == target_id:
            armed = False
            concurrent_op()
        return snapshot

    monkeypatch.setattr(queue, 'get', get_then_race)


async def _resolve_issue(queue: EscalationQueue, **kwargs: Any) -> dict[str, Any]:
    tool = await create_server(queue).get_tool('resolve_issue')
    return tool.fn(**kwargs)


def _archive_copies(queue: EscalationQueue, esc_id: str) -> list[Path]:
    return list((queue.queue_dir / 'archive').rglob(f'{esc_id}.json'))


def _blank_outcome() -> ResolveOutcome:
    return {
        'applied': False,
        'prior_status': None,
        'prior_resolved_by': None,
        'late_resolution_captured': False,
        'resolution_class_corrected': None,
    }


class TestQueueResolveStampsResolutionAction:

    def test_applied_resolve_stamps_the_action_on_the_archived_record_and_the_callback(
        self, tmp_path: Path,
    ):
        queue = EscalationQueue(tmp_path / 'esc')
        esc_id = _seed_pending(queue)
        seen: list[Escalation] = []
        queue.set_resolve_callback(seen.append)

        esc = queue.resolve(esc_id, 'abandoning', dismiss=True, resolution_action='abandon')

        assert esc is not None and esc.resolution_action == 'abandon'
        archived = _archive_copies(queue, esc_id)
        assert len(archived) == 1, f'expected exactly one archive copy: {archived}'
        assert json.loads(archived[0].read_text())['resolution_action'] == 'abandon'
        assert not (queue.queue_dir / f'{esc_id}.json').exists()
        assert len(seen) == 1 and seen[0].resolution_action == 'abandon', (
            'the harness reads the action off the resolve-callback argument'
        )

    def test_resolve_without_resolution_action_leaves_it_unset(self, tmp_path: Path):
        queue = EscalationQueue(tmp_path / 'esc')
        esc_id = _seed_pending(queue)

        queue.resolve(esc_id, 'ok')

        stored = queue.get(esc_id)
        assert stored is not None and stored.resolution_action is None

    def test_already_terminal_resolve_does_not_stamp_the_action(self, tmp_path: Path):
        queue = EscalationQueue(tmp_path / 'esc')
        esc_id = _seed_pending(queue)
        queue.resolve(esc_id, SWEEP_TEXT, dismiss=True, resolved_by='auto-dismissed')
        outcome = _blank_outcome()

        queue.resolve(
            esc_id, LATE_TEXT, resolved_by=STEWARD,
            resolution_action='resume', outcome=outcome,
        )

        assert outcome['applied'] is False
        assert outcome['late_resolution_captured'] is True
        stored = queue.get(esc_id)
        assert stored is not None
        assert stored.resolution_action is None, (
            'the terminal state belongs to whoever closed the record first'
        )
        assert stored.status == 'dismissed'
        assert stored.resolved_by == 'auto-dismissed'
        assert not (queue.queue_dir / f'{esc_id}.json').exists()


class TestResolveIssueDoesNotRewriteAStaleSnapshot:
    """Regression for fk-e9e943d0cd2c: no write-back of the gate snapshot."""

    @pytest.mark.asyncio
    async def test_a_sweep_archiving_between_snapshot_and_resolve_is_not_resurrected_or_re_resolved(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        queue = EscalationQueue(tmp_path / 'esc')
        esc_id = _seed_pending(queue)
        fired: list[tuple[str, str, str | None]] = []
        queue.set_resolve_callback(lambda e: fired.append((e.id, e.status, e.resolved_by)))
        _interleave_after_first_read(
            monkeypatch, queue, esc_id,
            lambda: queue.resolve(
                esc_id, SWEEP_TEXT, dismiss=True, resolved_by='auto-dismissed',
            ),
        )

        result = await _resolve_issue(
            queue, escalation_id=esc_id, resolution=LATE_TEXT,
            action='resume', resolved_by=STEWARD,
        )

        assert fired == [(esc_id, 'dismissed', 'auto-dismissed')], (
            'only the sweep may close the record; a second callback means the '
            'stale snapshot resurrected it and resolved it again'
        )
        assert not (queue.queue_dir / f'{esc_id}.json').exists()
        archived = _archive_copies(queue, esc_id)
        assert len(archived) == 1, f'expected exactly one archive copy: {archived}'
        stored = queue.get(esc_id)
        assert stored is not None
        assert stored.status == 'dismissed'
        assert stored.resolved_by == 'auto-dismissed'
        assert stored.resolution_action is None
        assert result['late_resolution_captured'] is True, (
            f'the late text must reach the task-4495 capture: {result}'
        )

    @pytest.mark.asyncio
    async def test_a_member_added_between_snapshot_and_resolve_is_kept_and_cascaded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        queue = EscalationQueue(tmp_path / 'esc')
        l1 = {
            'agent_role': 'steward', 'category': 'design_concern',
            'summary': 'an L1 member', 'level': 1,
        }
        m1 = _seed_pending(queue, 'esc-m1', **l1)
        m2 = _seed_pending(queue, 'esc-m2', **l1)
        l2 = _seed_pending(
            queue, 'esc-l2', agent_role='escalation-watcher-auto',
            category='design_concern', summary='an L2 cluster', level=2,
            members=[m1],
        )
        _interleave_after_first_read(
            monkeypatch, queue, l2, lambda: queue.add_members_to_l2(l2, [m2]),
        )

        await _resolve_issue(
            queue, escalation_id=l2, resolution='closing the cluster',
            action='close_only',
        )

        stored_l2 = queue.get(l2)
        assert stored_l2 is not None
        assert stored_l2.status == 'dismissed'
        assert stored_l2.members == [m1, m2], (
            'the stale snapshot clobbered the concurrent member append'
        )
        assert stored_l2.resolution_action == 'close_only'
        stored_m2 = queue.get(m2)
        assert stored_m2 is not None
        assert stored_m2.status == 'dismissed'
        assert stored_m2.resolved_by == f'l2-cascade:{l2}'
