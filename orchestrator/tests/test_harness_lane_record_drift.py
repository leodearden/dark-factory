"""Tests for the born-at-L2 lane-record-drift filer + pool drift wiring.

Task 2986 (PRD `plans/warm-lane-exhaustion-hardening-prd.md` leaf γ / W2b) —
single-writer warm-lane assignment store with LOUD drift.

When the WarmLanePool's durable-record mirror write fails ``drift_l2_threshold``
consecutive times, the pool fires its opaque ``_on_lane_record_drift`` callback.
The Harness installs ``_file_lane_record_drift_l2`` there — a born-at-L2 human
escalation, deduped to a single pending L2 via ``find_pending_l2_by_root_cause``
(root_cause literal ``lane_record_drift``).  This mirrors the
``_file_watcher_outage_l2`` / ``_file_reblock_guard_l2`` filer template and the
``_on_pool_storage_absent`` install-in-harness pattern.

Grouped by facet (all RED until step-10 lands the harness filer + wiring):
  TestFileLaneRecordDriftL2  — filer: one L2, dedup, no-op when queue absent
  TestPoolDriftWiring        — pool callback + lifecycle + threshold wired
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest
from escalation.models import Escalation

from orchestrator.config import GitConfig
from orchestrator.harness import Harness

# ---------------------------------------------------------------------------
# Fixture (mirrors test_harness_reblock_signature_guard.py::harness)
# ---------------------------------------------------------------------------


@pytest.fixture
def harness(mock_orch_config) -> Harness:
    """Harness with mocked internals for lane-record-drift filer unit testing.

    Pool is disabled by default (GitConfig().warm_lane_pool is False) so the
    filer tests exercise the escalation path in isolation; TestPoolDriftWiring
    builds its own pool-enabled config.
    """
    with (
        patch('orchestrator.harness.McpLifecycle'),
        patch('orchestrator.harness.OverrideStore'),
        patch('orchestrator.harness.Scheduler'),
        patch('orchestrator.harness.BriefingAssembler'),
    ):
        h = Harness(mock_orch_config)

    # No escalation queue by default (bare-harness style)
    h._escalation_queue = None

    return h


# ---------------------------------------------------------------------------
# Helper (mirrors test_harness_reblock_signature_guard.py::_make_mock_queue)
# ---------------------------------------------------------------------------


def _make_mock_queue(*, pending_l2_root_cause: str | None = None):
    """Build a minimal mock EscalationQueue for the drift filer tests.

    Exposes make_id, submit, find_pending_l2_by_root_cause, get_pending, get.
    ``_submitted`` is the list of submitted Escalation objects.
    """
    submitted: list = []

    def make_id(task_id: str) -> str:
        return f'esc-{task_id}-L2-1'

    def submit(esc):
        submitted.append(esc)

    def find_pending_l2_by_root_cause(root_cause: str) -> str | None:
        return pending_l2_root_cause

    q = MagicMock()
    q.make_id = make_id
    q.submit = submit
    q.find_pending_l2_by_root_cause = find_pending_l2_by_root_cause
    q.get_pending = MagicMock(return_value=[])
    q._submitted = submitted
    q.get = MagicMock(return_value=None)
    return q


# ---------------------------------------------------------------------------
# Filer: born-at-L2 lane_record_drift escalation (dedup + fail-safe)
# ---------------------------------------------------------------------------


class TestFileLaneRecordDriftL2:
    """_file_lane_record_drift_l2 files exactly one deduped born-at-L2."""

    def test_fresh_queue_files_one_l2_with_contract_fields(
        self, harness: Harness, caplog
    ):
        """(1) Fresh queue → exactly one Escalation with the contract fields.

        level==2, severity in {'urgent','critical'} (exempt from the
        agent-role downgrade gate → routes straight to a human),
        root_cause=='lane_record_drift' (the bare fixed dedup literal),
        agent_role an orchestrator/harness sentinel, category set.
        """
        q = _make_mock_queue()
        harness._escalation_queue = q
        harness.event_store = MagicMock()

        with caplog.at_level(logging.WARNING):
            harness._file_lane_record_drift_l2(3)

        # Exactly one L2 filed
        assert len(q._submitted) == 1, (
            f'Expected 1 L2 submitted, got {len(q._submitted)}'
        )
        filed = q._submitted[0]
        assert isinstance(filed, Escalation)
        assert filed.level == 2
        assert filed.severity in {'urgent', 'critical'}, (
            f'severity must be human-routed urgent/critical, got {filed.severity!r}'
        )
        assert filed.root_cause == 'lane_record_drift', (
            f'root_cause must be the bare fixed dedup literal, got {filed.root_cause!r}'
        )
        assert filed.agent_role == 'orchestrator-lane-record-drift', (
            f'agent_role must be an orchestrator/harness sentinel, got {filed.agent_role!r}'
        )
        assert filed.category in {'infra_issue', 'risk_identified'}, (
            f'category must be set to a valid category, got {filed.category!r}'
        )
        # The drift count must surface in the operator-facing text.
        assert '3' in (filed.summary or '') + (filed.detail or ''), (
            'drift count should appear in summary/detail'
        )

        # WARNING logged (loudness)
        assert any(r.levelno >= logging.WARNING for r in caplog.records), (
            'Expected a WARNING record when filing the drift L2'
        )

    def test_dedup_second_call_files_nothing(self, harness: Harness):
        """(2) A pending L2 with the same root_cause exists → no second submit."""
        q = _make_mock_queue(
            pending_l2_root_cause='esc-__lane_record_drift__-L2-1'
        )
        harness._escalation_queue = q
        harness.event_store = MagicMock()

        harness._file_lane_record_drift_l2(3)

        assert len(q._submitted) == 0, (
            f'Expected 0 new L2 submissions (dedup), got {len(q._submitted)}'
        )

    def test_noop_when_queue_absent(self, harness: Harness):
        """(3) No escalation queue → no-op (bare harness stays green, never raises)."""
        harness._escalation_queue = None

        # Must not raise; must not require a queue.
        harness._file_lane_record_drift_l2(3)


# ---------------------------------------------------------------------------
# Wiring: the warm-lane pool has drift loudness installed post-construction
# ---------------------------------------------------------------------------


class TestPoolDriftWiring:
    """After harness construction the warm-lane pool has drift loudness wired.

    - ``_on_lane_record_drift`` is installed (the harness filer) at the
      pool-callback install point (beside ``_on_pool_storage_absent``).
    - ``_lane_lifecycle`` is wired by the GitOps constructor so write-through
      holds in normal operation (not only crash recovery).
    - ``drift_l2_threshold`` is plumbed from ``config.git`` (a NON-default value
      is used so the assertion proves the value is threaded, not merely the
      shared default 3).
    """

    def test_pool_callback_lifecycle_and_threshold_wired(self, mock_orch_config):
        config = mock_orch_config
        # Enable the pool with a NON-default drift threshold (7) so the
        # threshold assertion below only passes if the config value is actually
        # threaded into the pool constructor (default would coincidentally be 3).
        config.git = GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
            warm_lane_pool=True,
            warm_lane_drift_l2_threshold=7,
        )
        config.max_concurrent_tasks = 2

        with (
            patch('orchestrator.harness.McpLifecycle'),
            patch('orchestrator.harness.OverrideStore'),
            patch('orchestrator.harness.Scheduler'),
            patch('orchestrator.harness.BriefingAssembler'),
        ):
            h = Harness(config)

        pool = h.git_ops.warm_lane_pool
        assert pool is not None, (
            'pool must exist when warm_lane_pool=True and size>0'
        )

        # (a) callback installed by the harness install point
        assert pool._on_lane_record_drift is not None, (
            '_on_lane_record_drift must be installed by the harness'
        )
        assert pool._on_lane_record_drift == h._file_lane_record_drift_l2, (
            'the installed callback must be the harness drift filer'
        )

        # (b) lifecycle wired by the GitOps constructor (write-through active)
        assert pool._lane_lifecycle is not None, (
            'pool lane_lifecycle must be wired for durable write-through'
        )
        assert pool._lane_lifecycle is h.git_ops._lane_lifecycle, (
            'pool must share the GitOps LaneLifecycle instance'
        )

        # (c) threshold plumbed from config (NON-default proves the plumbing)
        assert pool.drift_l2_threshold == config.git.warm_lane_drift_l2_threshold, (
            'pool.drift_l2_threshold must be plumbed from config.git'
        )
        assert pool.drift_l2_threshold == 7
