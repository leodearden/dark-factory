"""Tests for orchestrator.landing_evidence (task 2678).

Unit tests for the shared, module-level ``validate_landing_evidence`` helper
that both harness.py (x4 sites) and merge_queue.py (x1 site) delegate to for
already-landed evidence attribution (FIX 2 citation-lineage, task 2500/2675)
and post-hoc effect-present validation (FIX 1', task 2500/2675) — extracted
here so all five re-derivation sites share ONE copy (INV-5) rather than five
independently-drifting inline blocks.

Covers:
  step-01 (RED)  DISCOVERY mode (candidate_sha=None): citation discovery +
                 FIX2 lineage (both directions) + FIX1' effect-present.
  step-03 (RED)  CANDIDATE mode (candidate_sha given): effect-present only,
                 no citation discovery.

Mirrors test_harness_already_landed_gate_wiring.py's ``_wired_ancestry_harness``
git_ops shape (find_task_citation_commit / is_ancestor / commit_effect_present_in_main
as AsyncMocks on a bare MagicMock git_ops) so the same sub-method mocking
idiom keeps exercising the real (now-shared) helper logic.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.landing_evidence import validate_landing_evidence


def _git_ops(*, citation, is_ancestor_map, effect_present) -> MagicMock:
    """Build a bare MagicMock git_ops with the three sub-methods the helper calls.

    ``is_ancestor_map`` maps ``(ancestor, descendant)`` arg-pairs to their
    return value; an unexpected pair raises AssertionError so an errant call
    the test didn't anticipate fails loudly instead of returning a stray
    MagicMock truthy default.
    """
    git_ops = MagicMock()
    git_ops.find_task_citation_commit = AsyncMock(return_value=citation)

    async def _is_ancestor(a, b):
        try:
            return is_ancestor_map[(a, b)]
        except KeyError as err:
            raise AssertionError(f'unexpected is_ancestor call: {a!r}, {b!r}') from err

    git_ops.is_ancestor = AsyncMock(side_effect=_is_ancestor)
    git_ops.commit_effect_present_in_main = AsyncMock(return_value=effect_present)
    return git_ops


@pytest.mark.asyncio
class TestValidateLandingEvidenceDiscoveryMode:
    """DISCOVERY mode (candidate_sha=None): discover a citation, apply the
    FIX2 lineage guard, then the FIX1' effect-present guard.
    """

    async def test_citation_on_branch_accepted_checks_branch_tip_effect(self) -> None:
        """(a) citation found + is_ancestor(citation, branch) True -> accepted,
        evidence_sha==citation, and commit_effect_present_in_main awaited
        with branch_tip_sha (the in-branch work-commit citation may be a
        stale intermediate commit, so the branch's actual final state —
        its tip — is what gets checked, task 2500 amendment).
        """
        branch = 'task/42'
        branch_tip_sha = 'f' * 40
        citation_sha = 'a' * 40
        git_ops = _git_ops(
            citation=citation_sha,
            is_ancestor_map={(citation_sha, branch): True},
            effect_present=True,
        )

        verdict = await validate_landing_evidence(
            git_ops, '42', branch, branch_tip_sha=branch_tip_sha,
        )

        assert verdict.accepted is True
        assert verdict.evidence_sha == citation_sha
        assert verdict.reason == 'ok'
        git_ops.commit_effect_present_in_main.assert_awaited_once_with(branch_tip_sha)

    async def test_own_merge_commit_citation_accepted_checks_citation_effect(
        self,
    ) -> None:
        """(b) citation is this branch's own no-ff merge commit
        (is_ancestor(citation, branch) False, is_ancestor(branch, citation)
        True) -> accepted, evidence_sha==citation, and the effect check is
        anchored on the citation itself (its diff-tree is empty, so this is
        a deliberate no-op check rather than a redundant branch-tip check).
        """
        branch = 'task/42'
        citation_sha = 'a' * 40
        git_ops = _git_ops(
            citation=citation_sha,
            is_ancestor_map={
                (citation_sha, branch): False,
                (branch, citation_sha): True,
            },
            effect_present=True,
        )

        verdict = await validate_landing_evidence(
            git_ops, '42', branch, branch_tip_sha='f' * 40,
        )

        assert verdict.accepted is True
        assert verdict.evidence_sha == citation_sha
        git_ops.commit_effect_present_in_main.assert_awaited_once_with(citation_sha)

    async def test_no_citation_rejected(self) -> None:
        """(c) find_task_citation_commit -> None -> rejected, reason ==
        'no_citation', evidence_sha is None.
        """
        git_ops = _git_ops(citation=None, is_ancestor_map={}, effect_present=True)

        verdict = await validate_landing_evidence(
            git_ops, '42', 'task/42', branch_tip_sha='f' * 40,
        )

        assert verdict.accepted is False
        assert verdict.reason == 'no_citation'
        assert verdict.evidence_sha is None

    async def test_foreign_citation_neither_direction_rejected(self) -> None:
        """(d) the citation is an ancestor in NEITHER direction (an
        unrelated task's commit that merely matched the citation grep
        pattern — the task 2624 incident shape) -> rejected, reason ==
        'lineage_mismatch'.
        """
        branch = 'task/42'
        citation_sha = 'a' * 40
        git_ops = _git_ops(
            citation=citation_sha,
            is_ancestor_map={
                (citation_sha, branch): False,
                (branch, citation_sha): False,
            },
            effect_present=True,
        )

        verdict = await validate_landing_evidence(
            git_ops, '42', branch, branch_tip_sha='f' * 40,
        )

        assert verdict.accepted is False
        assert verdict.reason == 'lineage_mismatch'
        assert verdict.evidence_sha is None

    async def test_effect_absent_rejected(self) -> None:
        """(e) commit_effect_present_in_main -> False -> rejected, reason ==
        'effect_absent', evidence_sha is None (a later commit on main
        reverted the citation's effect — the found_on_main post-hoc-revert
        blind spot, task 2500/2675/1175).
        """
        branch = 'task/42'
        citation_sha = 'a' * 40
        git_ops = _git_ops(
            citation=citation_sha,
            is_ancestor_map={(citation_sha, branch): True},
            effect_present=False,
        )

        verdict = await validate_landing_evidence(
            git_ops, '42', branch, branch_tip_sha='f' * 40,
        )

        assert verdict.accepted is False
        assert verdict.reason == 'effect_absent'
        assert verdict.evidence_sha is None

    async def test_probe_carries_structured_fields(self) -> None:
        """(f) verdict.probe is a dict carrying task_id, branch,
        branch_tip_sha, citation, effect_check_sha, reason — so a caller can
        build a structured-facts escalation without prose-parsing.
        """
        branch = 'task/42'
        branch_tip_sha = 'f' * 40
        citation_sha = 'a' * 40
        git_ops = _git_ops(
            citation=citation_sha,
            is_ancestor_map={(citation_sha, branch): True},
            effect_present=True,
        )

        verdict = await validate_landing_evidence(
            git_ops, '42', branch, branch_tip_sha=branch_tip_sha,
        )

        assert verdict.probe['task_id'] == '42'
        assert verdict.probe['branch'] == branch
        assert verdict.probe['branch_tip_sha'] == branch_tip_sha
        assert verdict.probe['citation'] == citation_sha
        assert verdict.probe['effect_check_sha'] == branch_tip_sha
        assert verdict.probe['reason'] == 'ok'

    async def test_citation_on_branch_with_no_branch_tip_falls_back_to_citation(
        self,
    ) -> None:
        """(g) Defensive branch: citation_on_branch True but branch_tip_sha
        is None (a DISCOVERY caller omitted it despite the citation being an
        in-branch work commit) -> the guard ``citation_on_branch and
        branch_tip_sha is not None`` does not hold, so effect_check_sha
        falls back to the citation itself rather than passing None through
        to commit_effect_present_in_main. Pins the one branch where the
        helper's guard diverges from the original harness inline shape
        (``branch_tip_sha if citation_on_branch else citation``), which
        would have passed None straight through (review finding, task 2678
        amendment).
        """
        branch = 'task/42'
        citation_sha = 'a' * 40
        git_ops = _git_ops(
            citation=citation_sha,
            is_ancestor_map={(citation_sha, branch): True},
            effect_present=True,
        )

        verdict = await validate_landing_evidence(
            git_ops, '42', branch, branch_tip_sha=None,
        )

        assert verdict.accepted is True
        assert verdict.evidence_sha == citation_sha
        git_ops.commit_effect_present_in_main.assert_awaited_once_with(citation_sha)
        assert verdict.probe['effect_check_sha'] == citation_sha


@pytest.mark.asyncio
class TestValidateLandingEvidenceCandidateMode:
    """CANDIDATE mode (candidate_sha given): a pre-attributed sha (marker
    subject / ground-truth report) skips citation discovery and the FIX2
    lineage guard entirely — only the FIX1' effect-present guard applies.
    """

    async def test_effect_present_accepted_skips_discovery(self) -> None:
        """(a) commit_effect_present_in_main(candidate)=True -> accepted,
        evidence_sha==candidate, reason=='ok', and find_task_citation_commit
        + is_ancestor are NOT called — attribution was already established
        by the caller (find_merge_marker subject / ground-truth report).
        """
        candidate_sha = 'b' * 40
        git_ops = _git_ops(
            citation='should-not-be-used',
            is_ancestor_map={},
            effect_present=True,
        )

        verdict = await validate_landing_evidence(
            git_ops, '42', 'task/42',
            branch_tip_sha=None,
            candidate_sha=candidate_sha,
        )

        assert verdict.accepted is True
        assert verdict.evidence_sha == candidate_sha
        assert verdict.reason == 'ok'
        git_ops.find_task_citation_commit.assert_not_called()
        git_ops.is_ancestor.assert_not_called()
        git_ops.commit_effect_present_in_main.assert_awaited_once_with(candidate_sha)

    async def test_effect_absent_rejected(self) -> None:
        """(b) commit_effect_present_in_main(candidate)=False -> rejected,
        reason=='effect_absent', evidence_sha is None.
        """
        candidate_sha = 'b' * 40
        git_ops = _git_ops(citation=None, is_ancestor_map={}, effect_present=False)

        verdict = await validate_landing_evidence(
            git_ops, '42', 'task/42',
            branch_tip_sha=None,
            candidate_sha=candidate_sha,
        )

        assert verdict.accepted is False
        assert verdict.reason == 'effect_absent'
        assert verdict.evidence_sha is None

    async def test_probe_records_candidate_as_evidence_and_effect_check_sha(
        self,
    ) -> None:
        """(c) verdict.probe records the candidate as both 'citation' and
        'effect_check_sha' (candidate mode has no separately-discovered
        citation — the candidate IS the evidence under test).
        """
        candidate_sha = 'b' * 40
        git_ops = _git_ops(citation=None, is_ancestor_map={}, effect_present=True)

        verdict = await validate_landing_evidence(
            git_ops, '42', 'task/42',
            branch_tip_sha=None,
            candidate_sha=candidate_sha,
        )

        assert verdict.probe['citation'] == candidate_sha
        assert verdict.probe['effect_check_sha'] == candidate_sha
