"""Tests for Harness._get_ground_truth() — the lazy TaskGroundTruth wiring
seam (task 2243, W10-θ2).

``TaskGroundTruth`` (θ1, orchestrator/src/orchestrator/task_ground_truth.py)
captures its ``escalation_queue`` collaborator at CONSTRUCTION time.
``Harness._escalation_queue`` is ``None`` at ``__init__`` and is only
populated later, during ``run()`` startup (harness.py:7115). A resolver
built eagerly in ``__init__`` — or memoized the first time it's asked for,
with no refresh — would freeze that startup-time ``None`` forever, silently
breaking every open-escalation-aware recovery row (`_RECOVERY` rows f/g/h)
for the rest of the process lifetime. ``_get_ground_truth()`` must therefore
build (or rebuild) lazily, always capturing the LIVE queue. This module pins
that "frozen-None trap".
"""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import MagicMock, patch

import pytest

from orchestrator.harness import Harness
from orchestrator.task_ground_truth import TaskGroundTruth

# ---------------------------------------------------------------------------
# Harness fixture (Pattern A — mirrors test_reconcile_stranded.py's `harness`
# fixture, minus the git_ops/scheduler behavioural rewiring that only the
# reconcile-sweep tests need).
# ---------------------------------------------------------------------------


@pytest.fixture
def harness(mock_orch_config):
    """Create a bare Harness with mocked internals for unit testing wiring."""
    with patch('orchestrator.harness.McpLifecycle'), \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        h = Harness(mock_orch_config)
    return h


class TestGetGroundTruthWiring:
    def test_escalation_queue_is_none_at_construction(self, harness: Harness):
        """Precondition this whole module pins against: __init__ leaves
        _escalation_queue at None; it is only set later, during run()."""
        assert harness._escalation_queue is None

    def test_returns_a_task_ground_truth_instance(self, harness: Harness):
        harness._escalation_queue = MagicMock(name='live_escalation_queue')

        resolver = harness._get_ground_truth()

        assert isinstance(resolver, TaskGroundTruth)

    def test_captures_live_escalation_queue_not_init_time_none(self, harness: Harness):
        """The frozen-None trap, direct pin: a resolver built while
        _escalation_queue is still None (e.g. a call that races startup)
        must NOT permanently freeze that None. Once run() populates the
        live queue, the next call must pick it up."""
        stale = harness._get_ground_truth()
        assert stale.escalation_queue is None  # built before startup populated it

        live_queue = MagicMock(name='live_escalation_queue')
        harness._escalation_queue = live_queue

        fresh = harness._get_ground_truth()

        assert fresh.escalation_queue is live_queue

    def test_captures_git_ops_scheduler_and_worktree_resolver(self, harness: Harness):
        harness._escalation_queue = MagicMock(name='live_escalation_queue')

        resolver = harness._get_ground_truth()

        assert resolver.git_ops is harness.git_ops
        assert resolver.scheduler is harness.scheduler
        assert resolver.worktree_resolver == harness._resolve_task_worktree

    def test_heartbeat_ttl_set_from_config(self, harness: Harness):
        harness._escalation_queue = MagicMock(name='live_escalation_queue')

        resolver = harness._get_ground_truth()

        assert resolver.heartbeat_ttl == timedelta(minutes=10)

    def test_memoizes_across_calls_once_queue_is_stable(self, harness: Harness):
        """Once _escalation_queue is stable (the steady-state post-startup
        case), repeated calls must return the SAME resolver instance rather
        than rebuilding on every reconcile check."""
        harness._escalation_queue = MagicMock(name='live_escalation_queue')

        first = harness._get_ground_truth()
        second = harness._get_ground_truth()

        assert first is second
