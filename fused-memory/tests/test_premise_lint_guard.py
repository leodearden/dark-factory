"""Unit tests for fused_memory.middleware.premise_lint_guard.

Step 1 (RED -> step-2 GREEN): invariant matrix for premise_lint_error.

Mirrors test_execution_class_guard.py's invariant-matrix layout: this guard
(W5-xi) is the exact sibling of execution_class_guard (W5-eta) — same
submit_task boundary, same recon-stage agent_id scoping, same structured
ValidationError shape — but it enforces recon_self_model.premise_lint's
false-premise scan over the task description instead of the execution_class
declaration. Guards against the 2083/2092/2093 false-premise batch that
mis-modeled recon's run_id and marker-deletion mechanics.
"""

from __future__ import annotations

from fused_memory.middleware.premise_lint_guard import premise_lint_error

# ---------------------------------------------------------------------------
# Step-1: invariant matrix tests
# ---------------------------------------------------------------------------


class TestPremiseLintErrorInvariantMatrix:
    """recon-stage-only enforcement of the false-premise lint over
    task descriptions."""

    def test_recon_stage_run_id_persistence_premise_rejects(self):
        """recon-stage agent_id + a description asserting run_id persists
        across cycles -> reject, naming the run_id_is_fresh_per_run
        invariant and carrying the Violation detail."""
        result = premise_lint_error(
            'Fix the bug where run_id persists across cycles in the ledger.',
            'recon-stage-task_knowledge_sync',
            '/proj',
        )
        assert result is not None
        assert result.get('error_type') == 'ValidationError'
        assert 'run_id_is_fresh_per_run' in result['error']
        assert 'never persisted across cycles' in result['error']

    def test_recon_stage_marker_deletion_premise_rejects(self):
        """recon-stage agent_id + a description asserting Stage 3
        remediation deletes the flag marker -> reject, naming the
        markers_deleted_only_by_gc invariant and carrying the Violation
        detail."""
        result = premise_lint_error(
            'Stage 3 remediation deletes the flag marker once resolved.',
            'recon-stage-task_knowledge_sync',
            '/proj',
        )
        assert result is not None
        assert result.get('error_type') == 'ValidationError'
        assert 'markers_deleted_only_by_gc' in result['error']
        assert 'deleted only by GC' in result['error']

    def test_recon_stage_clean_description_accepts(self):
        """recon-stage agent_id + a clean description (no known false
        premise) -> accept (None); premise-lint does not block clean recon
        submits."""
        result = premise_lint_error(
            'Reconcile task 7 status against the knowledge graph.',
            'recon-stage-task_knowledge_sync',
            '/proj',
        )
        assert result is None

    def test_non_recon_agent_false_premise_accepts(self):
        """NON-recon agent_id with a false-premise description -> accept
        (None); enforcement is scoped to recon-stage callers only."""
        result = premise_lint_error(
            'run_id persists across cycles, so we can key off it.',
            'claude-interactive',
            '/proj',
        )
        assert result is None

    def test_agent_id_none_false_premise_accepts(self):
        """agent_id=None + a false-premise description -> accept (None);
        no enforcement without a recon-stage caller identity."""
        result = premise_lint_error(
            'run_id persists across cycles, so we can key off it.',
            None,
            '/proj',
        )
        assert result is None
