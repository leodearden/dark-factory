"""The escalation-category contract, tested through its public names only.

``escalation.categories.EscalationCategory`` is the one registered vocabulary
every filed ``Escalation.category`` must belong to, and
``require_registered_category`` is the one check every enforcement point calls.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from escalation.categories import (
    EscalationCategory,
    UnregisteredEscalationCategory,
    require_registered_category,
)
from escalation.models import Escalation
from escalation.queue import EscalationQueue

_CATEGORIES_THIS_TASK_REGISTERS = (
    'cascade_status_rejection',
    'already_landed',
    'reconciliation_standing_decision_storm',
    'regressed_after_resolution',
    'milestone_check_stalled',
    'review_due',
)

_RETIRED_PROSE_LIST = (
    'scope_violation',
    'design_concern',
    'cleanup_needed',
    'dependency_discovered',
    'risk_identified',
    'infra_issue',
    'reconciliation_stale_human_operator',
    'reconciliation_stale_gate_backlog',
    'recon_missing_index',
)

_AGENT_INSTRUCTED = (
    'missing_premise',
    'planning_stalled',
    'out_of_scope',
    'merge_scope_mismatch',
    'workflow_exit_contract',
    'integration_skew',
)


class TestRegistry:
    @pytest.mark.parametrize('value', _CATEGORIES_THIS_TASK_REGISTERS)
    def test_outstanding_categories_are_registered(self, value: str):
        assert value in EscalationCategory

    @pytest.mark.parametrize('value', _RETIRED_PROSE_LIST)
    def test_retired_prose_list_stays_registered(self, value: str):
        assert value in EscalationCategory

    @pytest.mark.parametrize('value', _AGENT_INSTRUCTED)
    def test_agent_instructed_categories_are_registered(self, value: str):
        assert value in EscalationCategory

    def test_member_value_is_its_lowercased_name(self):
        mismatched = [m for m in EscalationCategory if m.value != m.name.lower()]
        assert mismatched == []

    def test_values_are_unique(self):
        assert len({m.value for m in EscalationCategory}) == len(EscalationCategory)


class TestRequireRegisteredCategory:
    def test_registered_plain_string_passes(self):
        assert require_registered_category('infra_issue') is None

    def test_member_passes(self):
        assert require_registered_category(EscalationCategory.INFRA_ISSUE) is None

    def test_typo_raises_informatively(self):
        with pytest.raises(UnregisteredEscalationCategory) as excinfo:
            require_registered_category('infra_isue')
        message = str(excinfo.value)
        assert repr('infra_isue') in message
        assert 'escalation.categories.EscalationCategory' in message
        assert 'infra_issue' in message

    def test_unregistered_is_a_value_error(self):
        assert issubclass(UnregisteredEscalationCategory, ValueError)

    @pytest.mark.parametrize('value', ['INFRA_ISSUE', ' infra_issue'])
    def test_membership_is_exact(self, value: str):
        with pytest.raises(UnregisteredEscalationCategory):
            require_registered_category(value)

    @pytest.mark.parametrize('value', [None, ''])
    def test_missing_category_raises(self, value: object):
        with pytest.raises(UnregisteredEscalationCategory):
            require_registered_category(value)


def _esc(queue: EscalationQueue, category: str) -> Escalation:
    return Escalation(
        id=queue.make_id('t1'),
        task_id='t1',
        agent_role='implementer',
        severity='blocking',
        category=category,
        summary='s',
    )


def _json_files(queue_dir: Path) -> list[Path]:
    return sorted(queue_dir.rglob('*.json'))


class TestQueueChokepoint:
    @pytest.fixture
    def queue(self, tmp_path: Path) -> EscalationQueue:
        return EscalationQueue(tmp_path / 'esc')

    def test_submit_rejects_unregistered_before_any_write(self, queue: EscalationQueue):
        notified: list[Escalation] = []
        queue.set_notify_callback(notified.append)

        with pytest.raises(UnregisteredEscalationCategory):
            queue.submit(_esc(queue, 'infra_isue'))

        assert queue.get_pending() == []
        assert _json_files(queue.queue_dir) == []
        assert notified == []

    def test_submit_accepts_registered_string(self, queue: EscalationQueue):
        esc_id = queue.submit(_esc(queue, 'infra_issue'))

        stored = queue.get(esc_id)
        assert stored is not None
        assert stored.category == 'infra_issue'

    def test_submit_accepts_member_and_stores_plain_string(self, queue: EscalationQueue):
        esc_id = queue.submit(_esc(queue, EscalationCategory.REVIEW_DUE))

        on_disk = json.loads((queue.queue_dir / f'{esc_id}.json').read_text())
        assert on_disk['category'] == 'review_due'
        stored = queue.get(esc_id)
        assert stored is not None
        assert stored.category == 'review_due'

    def test_submit_resolved_rejects_before_mutating(self, queue: EscalationQueue):
        resolved: list[Escalation] = []
        queue.set_resolve_callback(resolved.append)
        esc = _esc(queue, 'nope')

        with pytest.raises(UnregisteredEscalationCategory):
            queue.submit_resolved(esc, 'r')

        assert esc.status == 'pending'
        assert esc.resolved_at is None
        assert _json_files(queue.queue_dir) == []
        assert resolved == []

    def test_has_open_l1_rejects_unregistered_category(self, queue: EscalationQueue):
        with pytest.raises(UnregisteredEscalationCategory):
            queue.has_open_l1('t1', category='infra_isue')

    def test_has_open_l1_unfiltered_and_registered_unchanged(self, queue: EscalationQueue):
        assert queue.has_open_l1('t1') is False
        assert queue.has_open_l1('t1', category='infra_issue') is False

    def test_find_terminal_by_citation_rejects_unregistered(self, queue: EscalationQueue):
        with pytest.raises(UnregisteredEscalationCategory):
            queue.find_terminal_by_citation('t1', 'nope', 'a' * 40)

    def test_find_terminal_by_citation_registered_unchanged(self, queue: EscalationQueue):
        assert queue.find_terminal_by_citation('t1', 'infra_issue', 'a' * 40) is None

    def test_legacy_unregistered_record_loads_lists_and_resolves(
        self, queue: EscalationQueue,
    ):
        legacy = Escalation(
            id='esc-legacy-1',
            task_id='legacy',
            agent_role='implementer',
            severity='blocking',
            category='validation_lane_red',
            summary='a record filed before the registry existed',
            level=1,
        )
        legacy_path = queue.queue_dir / 'esc-legacy-1.json'
        legacy_path.write_text(json.dumps(legacy.to_dict()))

        loaded = queue.get('esc-legacy-1')
        assert loaded is not None
        assert loaded.category == 'validation_lane_red'
        assert 'esc-legacy-1' in [e.id for e in queue.get_pending()]

        closed = queue.resolve('esc-legacy-1', 'closed')

        assert closed is not None
        assert closed.status == 'resolved'
        assert not legacy_path.exists()
        archived = queue.get('esc-legacy-1')
        assert archived is not None
        assert archived.status == 'resolved'
