"""Unit tests for the completion-claim verification gate (task 3142, PRD leaf pi).

The gate is the code-level enforcement of the "Terminal-State Pre-Check
Discipline" that until now existed only as prompt text (reconciliation/prompts/
stage1.py). It extracts completion claims that NAME something concrete (a task
id, a commit sha, a tkt_ id), checks each against its live authority, and — on
mismatch OR unresolvable — has the episode ingested TAGGED rather than
rejected.

These tests are pure: extraction is textual, verification runs against injected
tri-state probes, so no Taskmaster, no ticket DB and no git are needed (the one
exception is the make_commit_probe suite, which builds a throwaway repo).
"""

from __future__ import annotations

import pytest

from fused_memory.services.completion_claim_gate import (
    CompletionClaim,
    extract_completion_claims,
)

_KNOWN = frozenset({'reify', 'dark_factory'})


def _extract(text: str, default_project_id: str = 'reify') -> list[CompletionClaim]:
    return extract_completion_claims(
        text,
        default_project_id=default_project_id,
        known_project_ids=_KNOWN,
    )


class TestAppliedWorkExtraction:
    """`applied_work` phrasing (applied / landed / merged / shipped / patched)
    anchored to an explicit task reference in the same clause."""

    def test_has_been_applied_yields_one_task_claim(self):
        text = "task 5422's de-flake fix has been applied"
        claims = _extract(text)

        assert len(claims) == 1
        claim = claims[0]
        assert claim.kind == 'applied_work'
        assert claim.subject == 'task'
        assert claim.ref == '5422'
        assert claim.project_id == 'reify'
        # The span points back into the original text so the flag can quote it.
        start, end = claim.span
        assert 0 <= start < end <= len(text)
        assert text[start:end].strip()

    @pytest.mark.parametrize(
        'text',
        [
            "task 5422's de-flake fix has been applied",
            'the fix for task 5422 landed',
            'df 5422 was merged',
            'task 5422 has shipped',
            "task 5422's flake was patched",
        ],
    )
    def test_applied_work_family_each_yields_one_claim(self, text):
        claims = _extract(text)

        assert len(claims) == 1, f'{text!r} -> {claims!r}'
        assert claims[0].kind == 'applied_work'
        assert claims[0].subject == 'task'
        assert claims[0].ref == '5422'

    @pytest.mark.parametrize(
        'text',
        [
            # Negated terminal outcome — a consistent NON-completion statement.
            "task 5422's fix has NOT yet landed",
            "task 5422's fix has not been applied",
            # Future/aspirational framing — describes work that has NOT completed.
            "task 5422's fix will land tomorrow",
            'task 5422 is going to be merged once review clears',
        ],
    )
    def test_negated_and_aspirational_framing_yields_nothing(self, text):
        assert _extract(text) == []

    def test_claim_without_a_named_ref_yields_nothing(self):
        """Volume control: an unanchored claim is not actionable and not tagged."""
        assert _extract('the de-flake fix has been applied') == []

    def test_ref_without_completion_phrasing_yields_nothing(self):
        assert _extract('task 5422 is pending review') == []

    def test_clause_scoping_keeps_ref_and_phrasing_together(self):
        """A ref in one clause and phrasing in another is not a claim."""
        assert _extract('task 5422 is under review. the other fix has been applied') == []
