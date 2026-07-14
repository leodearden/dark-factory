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
        invariant and carrying the Violation detail.

        NOTE (reviewer_comprehensive: false_positive_design): this
        description is phrased as a "fix the bug where..." remediation
        task. The guard intentionally does NOT exempt this framing — see
        premise_lint_guard.py's "Known limitation: remediation framing is
        not exempted" docstring section. Since the premise is false by
        construction, a task claiming to fix it is restating the exact
        confusion the guard exists to reject; this rejection is the
        intended behavior, not a false-positive bug in the guard.
        """
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

    def test_recon_stage_false_premise_in_prompt_only_rejects(self):
        """A false premise stated in `prompt` (submit_task's own docstring
        names this its primary "Task description for AI generation" field)
        is caught even when `description` is entirely clean — description
        and prompt are both linted, so a false premise cannot dodge the
        guard by choosing the other channel."""
        result = premise_lint_error(
            'Reconcile task 7 status against the knowledge graph.',
            'recon-stage-task_knowledge_sync',
            '/proj',
            prompt='Fix the bug where run_id persists across cycles in the ledger.',
        )
        assert result is not None
        assert result.get('error_type') == 'ValidationError'
        assert 'run_id_is_fresh_per_run' in result['error']

    def test_recon_stage_both_fields_clean_accepts(self):
        """A clean `description` AND a clean `prompt` -> accept (None)."""
        result = premise_lint_error(
            'Reconcile task 7 status against the knowledge graph.',
            'recon-stage-task_knowledge_sync',
            '/proj',
            prompt='Reconcile task 7',
        )
        assert result is None

    def test_recon_stage_run_id_reuse_paraphrase_rejects(self):
        """A paraphrase of run_id persistence ("reuse the run_id from the
        previous cycle") is also caught — the lint is a denylist over
        common phrasings of a false premise, not a single rigid template."""
        result = premise_lint_error(
            'Just reuse the run_id from the previous cycle to save a mint.',
            'recon-stage-task_knowledge_sync',
            '/proj',
        )
        assert result is not None
        assert 'run_id_is_fresh_per_run' in result['error']

    def test_recon_stage_run_id_carries_over_paraphrase_rejects(self):
        """Another paraphrase ("the run_id carries over between cycles")
        is caught by the broadened verb/preposition alternation."""
        result = premise_lint_error(
            'Note that the run_id carries over between cycles.',
            'recon-stage-task_knowledge_sync',
            '/proj',
        )
        assert result is not None
        assert 'run_id_is_fresh_per_run' in result['error']

    def test_recon_stage_marker_removal_paraphrase_rejects(self):
        """A paraphrase of marker deletion using "removes" rather than
        "deletes" is also caught by the broadened verb alternation."""
        result = premise_lint_error(
            'Stage 3 remediation removes the flag marker once resolved.',
            'recon-stage-task_knowledge_sync',
            '/proj',
        )
        assert result is not None
        assert 'markers_deleted_only_by_gc' in result['error']

    def test_recon_stage_marker_clears_paraphrase_rejects(self):
        """"clears" is likewise caught (removes/clears/purges are all
        paraphrases of the same false marker-deletion premise)."""
        result = premise_lint_error(
            'Stage 3 remediation clears the stage1_flag_marker once resolved.',
            'recon-stage-task_knowledge_sync',
            '/proj',
        )
        assert result is not None
        assert 'markers_deleted_only_by_gc' in result['error']

    def test_recon_stage_false_premise_in_details_only_rejects(self):
        """A false premise stated ONLY in `details` (submit_task's own
        docstring names this "Task details / implementation notes",
        commonly carrying substantial mechanism prose for recon-authored
        remediation tasks) is caught even when `description` and `prompt`
        are both clean — `details` is linted on equal footing with the
        other three free-text fields, closing the coverage gap where a
        false premise could otherwise dodge the guard by choosing that
        channel."""
        result = premise_lint_error(
            'Reconcile task 7 status against the knowledge graph.',
            'recon-stage-task_knowledge_sync',
            '/proj',
            prompt='Reconcile task 7',
            details='Implementation note: run_id persists across cycles in this subsystem.',
        )
        assert result is not None
        assert result.get('error_type') == 'ValidationError'
        assert 'run_id_is_fresh_per_run' in result['error']

    def test_recon_stage_false_premise_in_title_only_rejects(self):
        """A false premise stated ONLY in `title` is likewise caught."""
        result = premise_lint_error(
            'Reconcile task 7 status against the knowledge graph.',
            'recon-stage-task_knowledge_sync',
            '/proj',
            prompt='Reconcile task 7',
            title='Bug: run_id persists across cycles',
        )
        assert result is not None
        assert result.get('error_type') == 'ValidationError'
        assert 'run_id_is_fresh_per_run' in result['error']

    def test_recon_stage_all_four_fields_clean_accepts(self):
        """A clean description, prompt, title, AND details together ->
        accept (None)."""
        result = premise_lint_error(
            'Reconcile task 7 status against the knowledge graph.',
            'recon-stage-task_knowledge_sync',
            '/proj',
            prompt='Reconcile task 7',
            title='Reconcile task 7',
            details='No special implementation notes.',
        )
        assert result is None

    def test_recon_stage_marker_purges_paraphrase_rejects(self):
        """"purges" is likewise caught — it was added to the marker-deletion
        verb alternation alongside removes/clears but had no dedicated
        test."""
        result = premise_lint_error(
            'Stage 3 remediation purges the flag_for_stage2 marker once resolved.',
            'recon-stage-task_knowledge_sync',
            '/proj',
        )
        assert result is not None
        assert 'markers_deleted_only_by_gc' in result['error']

    def test_recon_stage_marker_deletion_cross_clause_false_positive_accepts(self):
        """A benign description that mentions "Stage 3" in one clause and a
        correct "GC ... deletes the marker" in a separate clause — split by
        a semicolon — must NOT be flagged. Regression guard for the
        tempered-dot gap (`_GAP_NO_NEGATION`) spanning a clause terminator
        and combining an unrelated "Stage 3" mention with a later, correct
        "GC deletes" clause into a spurious subject/verb/object match; the
        sentence never asserts that Stage 3 deletes the marker."""
        result = premise_lint_error(
            'Stage 3 processes the flag_for_stage2 marker; '
            'GC later deletes the marker on terminal task.',
            'recon-stage-task_knowledge_sync',
            '/proj',
        )
        assert result is None

    def test_recon_stage_two_distinct_invariants_across_fields_both_named(self):
        """A run_id-persistence premise in `description` plus a
        marker-deletion premise in `details` trigger TWO distinct
        invariants in one submission. The error must name both — sorted
        and deduped via `invariants = ', '.join(sorted({...}))` — and
        carry both violations' detail text; this aggregation path was
        previously only exercised with a single violation."""
        result = premise_lint_error(
            'Fix the bug where run_id persists across cycles in the ledger.',
            'recon-stage-task_knowledge_sync',
            '/proj',
            prompt='Reconcile task 7',
            details='Stage 3 remediation deletes the flag marker once resolved.',
        )
        assert result is not None
        assert result.get('error_type') == 'ValidationError'
        error = result['error']
        # Sorted alphabetically: markers_deleted_only_by_gc < run_id_is_fresh_per_run.
        assert 'invariant(s): markers_deleted_only_by_gc, run_id_is_fresh_per_run' in error
        assert 'never persisted across cycles' in error
        assert 'deleted only by GC' in error

    def test_recon_stage_duplicate_violation_across_fields_deduped(self):
        """The SAME false premise stated in two different fields
        (`description` and `title`) produces two Violations sharing one
        invariant AND one identical detail text. The combined error must
        not repeat that detail text twice — `detail_texts` dedup by exact
        text is a distinct code path from the `invariants` set dedup, and
        was previously untested."""
        result = premise_lint_error(
            'Fix the bug where run_id persists across cycles in the ledger.',
            'recon-stage-task_knowledge_sync',
            '/proj',
            prompt='Reconcile task 7',
            title='Bug: run_id persists across cycles',
        )
        assert result is not None
        error = result['error']
        assert error.count('never persisted across cycles') == 1

    def test_recon_stage_premise_split_across_fields_does_not_combine(self):
        """A false premise assembled only by JOINING two clean fields must
        NOT be flagged: each field is linted independently, so a match
        cannot straddle the boundary between an otherwise-clean `prompt`
        and an otherwise-clean `description`. Regression guard for the
        false-positive risk in the prior join-then-lint implementation
        (`f'{prompt}\\n{description}'`), where `prompt` ending "...the
        run_id" and `description` beginning "persists across cycles..."
        would spuriously combine into a match neither field makes alone."""
        result = premise_lint_error(
            'persists across cycles when read after a crash.',
            'recon-stage-task_knowledge_sync',
            '/proj',
            prompt='Investigate why the ledger references the run_id',
        )
        assert result is None
