"""Tests for the recon claim-verification guard (task 2438).

Hardens Stage 2 recon task-filing against fabricated commit/diff/stamp
claims attributed to a completed task/commit/ACTION — mirrors the shape of
tests/test_recon_code_fix_premise_guard.py, but the target here is
retrospective FALSE-FACT claims embedded in a filed task's own description,
not a self-referential recon code-fix premise.

Motivating incident: task 2433 asserted, as verified fact, that "task 2372
added an ACTION #5 stamp (metadata.done_provenance_invalidated=true) on task
reopen." No such stamp/token has ever existed in tree or history — a
fabrication. See recon_claim_verification_guard.py's module docstring.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# task-2438 step-01 RED: TestExtractAttributedClaims
# ──────────────────────────────────────────────────────────────────────────────


class TestExtractAttributedClaims:
    """Tests for extract_attributed_claims() + AttributedClaim from
    fused_memory.middleware.recon_claim_verification_guard.
    """

    def test_incident_sentence_yields_one_claim_with_attribution(self):
        """(a) The verbatim task-2433 sentence yields one AttributedClaim with
        token=='done_provenance_invalidated' and an attribution naming both
        task 2372 and ACTION #5."""
        from fused_memory.middleware.recon_claim_verification_guard import (
            extract_attributed_claims,
        )

        text = (
            "This is the same site that stamps "
            "metadata.done_provenance_invalidated=true per task 2372 ACTION #5 "
            "on task reopen."
        )

        claims = extract_attributed_claims(text)

        assert len(claims) == 1
        claim = claims[0]
        assert claim.token == "done_provenance_invalidated"
        assert "task 2372" in claim.attribution
        assert "ACTION #5" in claim.attribution

    def test_prospective_field_without_anchor_yields_no_claims(self):
        """(b) A prospective feature description that names a
        `metadata.foo_bar` field with NO task/commit/ACTION anchor yields []
        — this is a new-work spec, not a retrospective fact claim."""
        from fused_memory.middleware.recon_claim_verification_guard import (
            extract_attributed_claims,
        )

        text = (
            "We should add a new `metadata.foo_bar` field so operators can "
            "override the default routing behaviour going forward."
        )

        assert extract_attributed_claims(text) == []

    def test_empty_and_plain_prose_yield_no_claims(self):
        """(c) Empty/whitespace/plain-prose text yields []."""
        from fused_memory.middleware.recon_claim_verification_guard import (
            extract_attributed_claims,
        )

        assert extract_attributed_claims("") == []
        assert extract_attributed_claims("   \n\t  ") == []
        assert extract_attributed_claims(
            "The scheduler dispatches tasks once their dependencies are done."
        ) == []

    def test_multiple_distinct_tokens_dedupe_in_first_seen_order(self):
        """(d) Multiple distinct tokens dedupe in first-seen order; a repeated
        token keeps its FIRST occurrence's attribution. The filler paragraphs
        are wide enough that the bounded co-occurrence window cannot bridge
        across them, so this test does not depend on the exact window size."""
        from fused_memory.middleware.recon_claim_verification_guard import (
            extract_attributed_claims,
        )

        filler = (
            "This paragraph exists purely to separate the two mentions of "
            "the same token by a wide margin so the bounded co-occurrence "
            "window used for attribution matching cannot bridge across it, "
            "keeping this test independent of the exact window size chosen. "
        ) * 2

        text = (
            "The guard stamps metadata.alpha_flag=true per task 100 during "
            "the first pass. "
            + filler
            + "metadata.beta_flag=true per task 200 was added afterwards. "
            + filler
            + "Finally metadata.alpha_flag=true per task 300 repeats the "
            "earlier stamp."
        )

        claims = extract_attributed_claims(text)

        assert [c.token for c in claims] == ["alpha_flag", "beta_flag"]
        assert "task 100" in claims[0].attribution
        assert "task 300" not in claims[0].attribution
        assert "task 200" in claims[1].attribution
