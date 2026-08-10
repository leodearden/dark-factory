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


# The verbatim text from esc-3085-1 instance (2): a reify-authored claim that
# a task was re-filed into ANOTHER project's tree as a ticket that did not
# exist. Neither the phrasing family nor the ticket subject was covered before.
_INSTANCE_2 = (
    'reify task 5638 was reported unactionable and re-filed into '
    "dark_factory's task tree as ticket tkt_0RRRC5AASJ9Z630VP4PCN9H376"
)


class TestFilingDispatchExtraction:
    """The esc-3085-1 scope extension: filing/dispatch phrasing, and the
    ticket / commit subjects alongside tasks."""

    def test_instance_2_extracts_as_a_ticket_filing_claim(self):
        claims = _extract(_INSTANCE_2)

        assert len(claims) == 1, claims
        claim = claims[0]
        assert claim.kind == 'filing_dispatch'
        # Ticket beats task: the tkt_ id is the more specific authority, and it
        # is the one that was actually false in the incident.
        assert claim.subject == 'ticket'
        assert claim.ref == 'tkt_0RRRC5AASJ9Z630VP4PCN9H376'

    @pytest.mark.parametrize(
        'phrasing',
        [
            'was filed as',
            'was re-filed as',
            'was refiled as',
            'was submitted as',
            'was queued as',
            'was dispatched as',
            'was cancelled as',
            'was closed as duplicate of',
        ],
    )
    def test_filing_dispatch_family_each_yields_a_ticket_claim(self, phrasing):
        text = f'the follow-up {phrasing} ticket tkt_0RRRC5AASJ9Z630VP4PCN9H376'
        claims = _extract(text)

        assert len(claims) == 1, f'{text!r} -> {claims!r}'
        assert claims[0].kind == 'filing_dispatch'
        assert claims[0].subject == 'ticket'
        assert claims[0].ref == 'tkt_0RRRC5AASJ9Z630VP4PCN9H376'

    def test_commit_sha_claim_resolves_to_the_commit_subject(self):
        claims = _extract('the de-flake fix landed in commit 7bbcd5d815')

        assert len(claims) == 1, claims
        assert claims[0].kind == 'applied_work'
        assert claims[0].subject == 'commit'
        assert claims[0].ref == '7bbcd5d815'

    def test_commit_beats_task_but_ticket_beats_commit(self):
        """Subject precedence is ticket > commit > task, per clause."""
        task_and_commit = _extract('task 5422 was merged as commit 7bbcd5d815')
        assert [(c.subject, c.ref) for c in task_and_commit] == [('commit', '7bbcd5d815')]

    @pytest.mark.parametrize(
        'text',
        [
            # Hex-looking word with no commit/sha/merge cue anchoring it.
            'the deadbeef fixture was applied',
            # A `tkt_` prefix with no body is not a ticket id.
            'the follow-up was filed as ticket tkt_',
            # Filing phrasing with no ref at all.
            'the follow-up was filed as a ticket',
        ],
    )
    def test_non_refs_yield_nothing(self, text):
        assert _extract(text) == []

    @pytest.mark.parametrize(
        'text',
        [
            'the follow-up will be filed as ticket tkt_0RRRC5AASJ9Z630VP4PCN9H376',
            'the follow-up is supposed to be filed as ticket tkt_0RRRC5AASJ9Z630VP4PCN9H376',
            'the follow-up has not been filed as ticket tkt_0RRRC5AASJ9Z630VP4PCN9H376',
        ],
    )
    def test_filing_negation_and_aspiration_yields_nothing(self, text):
        """The imported strippers have no arm for the filing vocabulary, so the
        supplementary ones must cover it — otherwise the negation hole the
        strippers exist to close reopens for exactly the new family."""
        assert _extract(text) == []


class TestCrossProjectRefResolution:
    """Which project's registry adjudicates a task ref.

    esc-3085-1: the incident claim was written by a reify agent ABOUT a
    dark_factory artefact, so resolving every ref against the writer's project
    would have produced a false verdict in the other direction.
    """

    @pytest.mark.parametrize(
        'text',
        [
            'dark_factory task 3142 has landed',
            'dark_factory:3142 was merged',
            "dark_factory's task 3142 has landed",
        ],
    )
    def test_recognised_qualifier_overrides_the_writers_project(self, text):
        claims = _extract(text, default_project_id='reify')

        assert len(claims) == 1, f'{text!r} -> {claims!r}'
        assert claims[0].subject == 'task'
        assert claims[0].ref == '3142'
        assert claims[0].project_id == 'dark_factory'

    def test_unqualified_ref_inherits_the_writers_project(self):
        claims = _extract('task 3142 has landed', default_project_id='reify')

        assert len(claims) == 1
        assert claims[0].project_id == 'reify'

    def test_arbitrary_preceding_word_is_not_a_qualifier(self):
        """'the merge task 3142' must not make 'merge' a project name."""
        claims = _extract('the merge task 3142 has landed', default_project_id='reify')

        assert len(claims) == 1
        assert claims[0].project_id == 'reify'

    def test_unknown_project_qualifier_falls_back_to_the_writer(self):
        claims = _extract('someproject task 3142 has landed', default_project_id='reify')

        assert len(claims) == 1
        assert claims[0].project_id == 'reify'

    def test_ticket_claim_carries_no_project(self):
        """A tkt_ id is a globally unique PK — it needs no project to resolve,
        and pinning one would reintroduce the instance-(2) false verdict."""
        claims = _extract(_INSTANCE_2, default_project_id='reify')

        assert len(claims) == 1
        assert claims[0].subject == 'ticket'
        assert claims[0].project_id is None
