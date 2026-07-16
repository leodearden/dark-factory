"""Tests for escalation.classify — resolver→tier classification, the
effective-benign predicate, and the per-path benign default helper
(plans/escalation-lifecycle-dashboard-prd.md Contract Seam 1).
"""

from __future__ import annotations

import pytest

from escalation.classify import (
    classify_resolver_tier,
    default_resolution_class_for_resolver,
    effective_benign,
)
from escalation.models import Escalation


def _make_escalation(
    esc_id: str = 'esc-1-1',
    *,
    status: str = 'pending',
    resolution_class: str | None = None,
) -> Escalation:
    esc = Escalation(
        id=esc_id,
        task_id='1',
        agent_role='orchestrator',
        severity='blocking',
        category='task_failure',
        summary='Something failed',
    )
    esc.status = status
    esc.resolution_class = resolution_class
    return esc


class TestClassifyResolverTierHuman:
    """classify_resolver_tier() maps human resolvers to tier 'human'."""

    def test_interactive_is_human(self):
        """resolved_by='interactive' classifies as 'human'."""
        assert classify_resolver_tier('interactive') == 'human'

    def test_escalation_watcher_is_human(self):
        """resolved_by='escalation-watcher' (exact) classifies as 'human'."""
        assert classify_resolver_tier('escalation-watcher') == 'human'


class TestClassifyResolverTierCascade:
    """classify_resolver_tier() maps l2-cascade:* resolvers to tier 'cascade'."""

    def test_l2_cascade_prefix_is_cascade(self):
        """resolved_by='l2-cascade:esc-5-1' classifies as 'cascade'."""
        assert classify_resolver_tier('l2-cascade:esc-5-1') == 'cascade'


class TestClassifyResolverTierAutoWatcher:
    """classify_resolver_tier() maps auto-watcher resolvers to tier 'auto-watcher'."""

    def test_escalation_watcher_auto_is_auto_watcher(self):
        """resolved_by='escalation-watcher-auto' classifies as 'auto-watcher'."""
        assert classify_resolver_tier('escalation-watcher-auto') == 'auto-watcher'

    def test_orchestrator_escalation_watcher_auto_is_auto_watcher(self):
        """resolved_by='orchestrator-escalation-watcher-auto' classifies as 'auto-watcher'."""
        assert classify_resolver_tier('orchestrator-escalation-watcher-auto') == 'auto-watcher'

    def test_escalation_watcher_vs_escalation_watcher_auto_disambiguation(self):
        """Exact 'escalation-watcher' is 'human'; 'escalation-watcher-auto' is 'auto-watcher' — not conflated."""
        assert classify_resolver_tier('escalation-watcher') == 'human'
        assert classify_resolver_tier('escalation-watcher-auto') == 'auto-watcher'


class TestClassifyResolverTierSteward:
    """classify_resolver_tier() maps claude-task-*-steward resolvers to tier 'steward'."""

    def test_claude_task_steward_is_steward(self):
        """resolved_by='claude-task-2656-steward' classifies as 'steward'."""
        assert classify_resolver_tier('claude-task-2656-steward') == 'steward'


class TestClassifyResolverTierReaperSweep:
    """classify_resolver_tier() maps automated-sweep resolvers to tier 'reaper-sweep'."""

    def test_harness_orphan_reaper_is_reaper_sweep(self):
        """resolved_by='harness-orphan-reaper' classifies as 'reaper-sweep'."""
        assert classify_resolver_tier('harness-orphan-reaper') == 'reaper-sweep'

    def test_auto_dismissed_is_reaper_sweep(self):
        """resolved_by='auto-dismissed' classifies as 'reaper-sweep'."""
        assert classify_resolver_tier('auto-dismissed') == 'reaper-sweep'

    def test_harness_escalation_revalidation_sweep_is_reaper_sweep(self):
        """resolved_by='harness-escalation-revalidation-sweep' classifies as 'reaper-sweep'."""
        assert classify_resolver_tier('harness-escalation-revalidation-sweep') == 'reaper-sweep'

    def test_orchestrator_starvation_watchdog_is_reaper_sweep(self):
        """resolved_by='orchestrator-starvation-watchdog' classifies as 'reaper-sweep'."""
        assert classify_resolver_tier('orchestrator-starvation-watchdog') == 'reaper-sweep'


class TestClassifyResolverTierUnknownAndOther:
    """classify_resolver_tier() maps None to 'unknown' and unrecognised values to 'other-auto'."""

    def test_none_is_unknown(self):
        """resolved_by=None classifies as 'unknown'."""
        assert classify_resolver_tier(None) == 'unknown'

    def test_random_role_is_other_auto(self):
        """resolved_by='random-role' (unrecognised) classifies as 'other-auto'."""
        assert classify_resolver_tier('random-role') == 'other-auto'


class TestEffectiveBenign:
    """effective_benign(record) -> (class, provenance): stamp-first, proxy-fallback."""

    def test_stamped_benign_dismissed(self):
        """Stamped resolution_class='benign' on a dismissed record -> ('benign', 'stamped')."""
        record = _make_escalation(status='dismissed', resolution_class='benign')
        assert effective_benign(record) == ('benign', 'stamped')

    def test_stamped_actionable_resolved(self):
        """Stamped resolution_class='actionable' on a resolved record -> ('actionable', 'stamped')."""
        record = _make_escalation(status='resolved', resolution_class='actionable')
        assert effective_benign(record) == ('actionable', 'stamped')

    def test_unstamped_dismissed_infers_benign(self):
        """Unstamped (resolution_class=None) dismissed record -> ('benign', 'inferred')."""
        record = _make_escalation(status='dismissed', resolution_class=None)
        assert effective_benign(record) == ('benign', 'inferred')

    def test_unstamped_resolved_infers_actionable(self):
        """Unstamped (resolution_class=None) resolved record -> ('actionable', 'inferred')."""
        record = _make_escalation(status='resolved', resolution_class=None)
        assert effective_benign(record) == ('actionable', 'inferred')

    def test_pending_is_excluded(self):
        """Pending record (resolution_class=None, status='pending') -> (None, 'excluded')."""
        record = _make_escalation(status='pending', resolution_class=None)
        assert effective_benign(record) == (None, 'excluded')

    def test_unmodeled_status_raises(self):
        """An unstamped record with a status outside {pending,resolved,dismissed}
        must raise, not silently resolve to ('excluded') — no-silent-fail-soft."""
        record = _make_escalation(status='cancelled', resolution_class=None)
        with pytest.raises(ValueError, match='cancelled'):
            effective_benign(record)


class TestDefaultResolutionClassForResolver:
    """default_resolution_class_for_resolver(resolved_by): 'benign' iff reaper-sweep tier, else None."""

    def test_auto_dismissed_defaults_benign(self):
        """resolved_by='auto-dismissed' defaults to 'benign'."""
        assert default_resolution_class_for_resolver('auto-dismissed') == 'benign'

    def test_harness_orphan_reaper_defaults_benign(self):
        """resolved_by='harness-orphan-reaper' defaults to 'benign'."""
        assert default_resolution_class_for_resolver('harness-orphan-reaper') == 'benign'

    def test_orchestrator_starvation_watchdog_defaults_benign(self):
        """resolved_by='orchestrator-starvation-watchdog' defaults to 'benign'."""
        assert default_resolution_class_for_resolver('orchestrator-starvation-watchdog') == 'benign'

    def test_harness_escalation_revalidation_sweep_defaults_benign(self):
        """resolved_by='harness-escalation-revalidation-sweep' defaults to 'benign'."""
        assert default_resolution_class_for_resolver('harness-escalation-revalidation-sweep') == 'benign'

    def test_interactive_defaults_none(self):
        """resolved_by='interactive' (human tier) defaults to None — no auto-benign for humans."""
        assert default_resolution_class_for_resolver('interactive') is None

    def test_escalation_watcher_defaults_none(self):
        """resolved_by='escalation-watcher' (human tier) defaults to None."""
        assert default_resolution_class_for_resolver('escalation-watcher') is None

    def test_escalation_watcher_auto_defaults_none(self):
        """resolved_by='escalation-watcher-auto' (auto-watcher tier) defaults to None — β wires it explicitly."""
        assert default_resolution_class_for_resolver('escalation-watcher-auto') is None

    def test_steward_defaults_none(self):
        """resolved_by='claude-task-2656-steward' (steward tier) defaults to None."""
        assert default_resolution_class_for_resolver('claude-task-2656-steward') is None

    def test_l2_cascade_defaults_none(self):
        """resolved_by='l2-cascade:esc-1-1' (cascade tier) defaults to None — cascade inherits the parent's class instead."""
        assert default_resolution_class_for_resolver('l2-cascade:esc-1-1') is None

    def test_none_defaults_none(self):
        """resolved_by=None (unknown tier) defaults to None."""
        assert default_resolution_class_for_resolver(None) is None

    def test_random_role_defaults_none(self):
        """resolved_by='random-role' (other-auto tier) defaults to None."""
        assert default_resolution_class_for_resolver('random-role') is None
