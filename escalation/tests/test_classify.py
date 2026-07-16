"""Tests for escalation.classify — resolver→tier classification, the
effective-benign predicate, and the per-path benign default helper
(plans/escalation-lifecycle-dashboard-prd.md Contract Seam 1).
"""

from __future__ import annotations

from escalation.classify import classify_resolver_tier


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
