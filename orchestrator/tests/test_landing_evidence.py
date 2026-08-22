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

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.delivered_checks import DeliveredCheckResult
from orchestrator.git_ops import CommitEffectProbe
from orchestrator.landing_evidence import (
    LandingEvidenceVerdict,
    file_unattributed_landing_escalation,
    format_unattributed_landing_detail,
    validate_landing_evidence,
)


def _git_ops(
    *, citation, is_ancestor_map, effect_present, effect_probe=None,
) -> MagicMock:
    """Build a bare MagicMock git_ops with the three sub-methods the helper calls.

    ``is_ancestor_map`` maps ``(ancestor, descendant)`` arg-pairs to their
    return value; an unexpected pair raises AssertionError so an errant call
    the test didn't anticipate fails loudly instead of returning a stray
    MagicMock truthy default.

    ``effect_probe`` (task 3116) stubs ``describe_commit_effect_in_main``, the
    reject-path-only diagnostic enrichment.  It is left UNSTUBBED when the
    kwarg is omitted, so every pre-3116 test keeps its exact call shape —
    which is also the live shape in the seven OTHER test files that stub only
    ``commit_effect_present_in_main`` on a bare MagicMock.
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
    if effect_probe is not None:
        git_ops.describe_commit_effect_in_main = AsyncMock(return_value=effect_probe)
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

    async def test_divergent_citation_accepted_anchors_effect_on_citation(
        self,
    ) -> None:
        """(d) FIX-A (task 2870 / esc-5252-9): a genuine on-main citation
        whose branch ref has diverged/realigned so it is an ancestor in
        NEITHER direction is NO LONGER rejected as 'lineage_mismatch'. Every
        DISCOVERY caller pre-establishes the branch's content is on main, and
        find_task_citation_commit greps ``git log main``, so branch-ref
        ancestry is not the landing authority — the citation is accepted and
        the FIX-1' effect-present guard is anchored on the CITATION itself
        (the divergent branch tip is not authoritative). Only the
        (citation, branch) is_ancestor call is made now — the second,
        bidirectional reject arm is gone (an unmapped (branch, citation)
        call would raise).
        """
        branch = 'task/42'
        citation_sha = 'a' * 40
        git_ops = _git_ops(
            citation=citation_sha,
            is_ancestor_map={(citation_sha, branch): False},
            effect_present=True,
        )

        verdict = await validate_landing_evidence(
            git_ops, '42', branch, branch_tip_sha='f' * 40,
        )

        assert verdict.accepted is True
        assert verdict.evidence_sha == citation_sha
        assert verdict.reason == 'ok'
        git_ops.commit_effect_present_in_main.assert_awaited_once_with(citation_sha)
        assert verdict.probe['effect_check_sha'] == citation_sha

    async def test_divergent_citation_effect_absent_still_rejected(self) -> None:
        """(d') FIX-1' still gates after the FIX-A relaxation: the same
        divergent-citation setup but with the citation's effect reverted at
        main HEAD (effect_present False) is still rejected as 'effect_absent',
        evidence_sha None — relaxing the lineage guard does not weaken the
        real gate against a reverted (task-1175) landing.
        """
        branch = 'task/42'
        citation_sha = 'a' * 40
        git_ops = _git_ops(
            citation=citation_sha,
            is_ancestor_map={(citation_sha, branch): False},
            effect_present=False,
        )

        verdict = await validate_landing_evidence(
            git_ops, '42', branch, branch_tip_sha='f' * 40,
        )

        assert verdict.accepted is False
        assert verdict.reason == 'effect_absent'
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


@pytest.mark.asyncio
class TestValidateLandingEvidenceEffectDivergenceProbe:
    """Reject-path-only diagnostic enrichment (task 3116).

    The gate DECISION still comes from the boolean
    ``commit_effect_present_in_main``; ``describe_commit_effect_in_main`` is
    consulted ONLY where that bool already rejected, purely to thread WHICH
    paths diverged into ``LandingEvidenceVerdict.probe``.  That split is what
    keeps the seven other test files' bare-MagicMock git_ops driving the
    decision exactly as before, and it makes it structurally impossible for a
    defect in the (much larger) diagnostic path to cement or withhold a
    completion.
    """

    async def test_discovery_reject_threads_diverged_paths_into_probe(self) -> None:
        """(a) DISCOVERY effect_absent: the verdict is unchanged, and probe
        now names the diverged path — the one line that would have resolved
        both reported instances.  The enrichment MUST probe the same sha the
        decision ran against; probing a different one would name paths from
        the wrong commit, which is worse than naming none.
        """
        citation = 'a' * 40
        probe_result = CommitEffectProbe(
            present=False,
            diverged_paths=('tests/infra/harness-layout-baseline.manifest',),
            anchor_sha='c' * 40,
            failure='paths_diverged',
        )
        git_ops = _git_ops(
            citation=citation,
            is_ancestor_map={(citation, 'task/42'): False},
            effect_present=False,
            effect_probe=probe_result,
        )

        verdict = await validate_landing_evidence(
            git_ops, '42', 'task/42', branch_tip_sha=None,
        )

        assert verdict.accepted is False
        assert verdict.reason == 'effect_absent'
        assert verdict.evidence_sha is None
        assert verdict.probe['diverged_paths'] == [
            'tests/infra/harness-layout-baseline.manifest'
        ]
        assert verdict.probe['effect_failure'] == 'paths_diverged'
        assert verdict.probe['effect_anchor_sha'] == 'c' * 40
        git_ops.describe_commit_effect_in_main.assert_awaited_once_with(
            verdict.probe['effect_check_sha'],
        )

    async def test_probe_carries_the_survival_facts_that_decided_it(self) -> None:
        """The part-(b) measurement must be threaded from CommitEffectProbe
        into ``verdict.probe``, not left in the dataclass.

        Without this the escalation can name the diverged paths — which part
        (b) demoted to a diagnostic — while omitting the aggregate/per-file
        numbers that ARE the decision, so the reader sees everything except
        the basis of the rejection.
        """
        citation = 'a' * 40
        git_ops = _git_ops(
            citation=citation,
            is_ancestor_map={(citation, 'task/42'): False},
            effect_present=False,
            effect_probe=CommitEffectProbe(
                present=False,
                diverged_paths=('src/core.py',),
                anchor_sha='c' * 40,
                failure='effect_not_survived',
                aggregate_survival=0.5,
                added_lines_total=60,
                worst_guarded_path='src/core.py',
                worst_guarded_survival=0.45,
                aggregate_threshold=0.98,
                per_file_threshold=0.9,
                per_file_min_added_lines=25,
                vacuous_paths=('gone.py',),
            ),
        )

        verdict = await validate_landing_evidence(
            git_ops, '42', 'task/42', branch_tip_sha=None,
        )

        assert verdict.probe['aggregate_survival'] == 0.5
        assert verdict.probe['added_lines_total'] == 60
        assert verdict.probe['worst_guarded_path'] == 'src/core.py'
        assert verdict.probe['worst_guarded_survival'] == 0.45
        assert verdict.probe['aggregate_threshold'] == 0.98
        assert verdict.probe['per_file_threshold'] == 0.9
        assert verdict.probe['per_file_min_added_lines'] == 25
        assert verdict.probe['vacuous_paths'] == ['gone.py']

        # End to end: the facts reach the escalation body, not just the dict.
        _, detail = format_unattributed_landing_detail('42', 'task/42', verdict)
        assert '0.5000' in detail
        assert '60' in detail

    async def test_candidate_reject_threads_diverged_paths_into_probe(self) -> None:
        """(b) CANDIDATE mode carries the same enrichment, anchored on the
        candidate sha the decision used.
        """
        candidate_sha = 'b' * 40
        probe_result = CommitEffectProbe(
            present=False,
            diverged_paths=('tests/infra/harness-layout-baseline.manifest',),
            anchor_sha='c' * 40,
            failure='paths_diverged',
        )
        git_ops = _git_ops(
            citation=None, is_ancestor_map={}, effect_present=False,
            effect_probe=probe_result,
        )

        verdict = await validate_landing_evidence(
            git_ops, '42', 'task/42',
            branch_tip_sha=None,
            candidate_sha=candidate_sha,
        )

        assert verdict.accepted is False
        assert verdict.reason == 'effect_absent'
        assert verdict.probe['diverged_paths'] == [
            'tests/infra/harness-layout-baseline.manifest'
        ]
        assert verdict.probe['effect_failure'] == 'paths_diverged'
        assert verdict.probe['effect_anchor_sha'] == 'c' * 40
        git_ops.describe_commit_effect_in_main.assert_awaited_once_with(candidate_sha)

    async def test_accept_path_never_probes(self) -> None:
        """(c) The enrichment is reject-path-ONLY: an accepted verdict costs
        no extra git work and carries no divergence keys at all.
        """
        candidate_sha = 'b' * 40
        git_ops = _git_ops(
            citation=None, is_ancestor_map={}, effect_present=True,
            effect_probe=CommitEffectProbe(present=True),
        )

        verdict = await validate_landing_evidence(
            git_ops, '42', 'task/42',
            branch_tip_sha=None,
            candidate_sha=candidate_sha,
        )

        assert verdict.accepted is True
        git_ops.describe_commit_effect_in_main.assert_not_awaited()
        assert 'diverged_paths' not in verdict.probe

    async def test_enrichment_can_never_change_the_verdict(self) -> None:
        """(d) THE TOCTOU RACE, made visible rather than hidden: the decision
        said absent, then main HEAD advanced and the re-probe disagrees.  The
        verdict is STILL effect_absent — a diagnostic must never overturn a
        gate decision — and the probe records the disagreement explicitly
        (empty diverged_paths, no failure code) so the formatter can render
        the race instead of silently contradicting itself.
        """
        candidate_sha = 'b' * 40
        git_ops = _git_ops(
            citation=None, is_ancestor_map={}, effect_present=False,
            effect_probe=CommitEffectProbe(
                present=True, diverged_paths=(), anchor_sha='c' * 40, failure=None,
            ),
        )

        verdict = await validate_landing_evidence(
            git_ops, '42', 'task/42',
            branch_tip_sha=None,
            candidate_sha=candidate_sha,
        )

        assert verdict.accepted is False
        assert verdict.reason == 'effect_absent'
        assert verdict.probe['diverged_paths'] == []
        assert verdict.probe['effect_failure'] is None

    async def test_structural_failure_code_passes_through(self) -> None:
        """(e) A structural failure (no path divergence to report) threads its
        code through, so the escalation names the real cause instead of
        implying paths diverged when none did.
        """
        candidate_sha = 'b' * 40
        git_ops = _git_ops(
            citation=None, is_ancestor_map={}, effect_present=False,
            effect_probe=CommitEffectProbe(
                present=False, diverged_paths=(), anchor_sha=None,
                failure='empty_branch_merge',
            ),
        )

        verdict = await validate_landing_evidence(
            git_ops, '42', 'task/42',
            branch_tip_sha=None,
            candidate_sha=candidate_sha,
        )

        assert verdict.probe['effect_failure'] == 'empty_branch_merge'
        assert verdict.probe['diverged_paths'] == []

    async def test_unprobeable_git_ops_records_the_error_loudly(self) -> None:
        """(f) LOUD, NOT SILENT.  A duck-typed stand-in predating the new
        method — the live shape in seven other test files — must not break the
        gate, but its failure is RECORDED into the probe (which is rendered
        verbatim into the escalation a human reads), never swallowed.

        ``diverged_paths`` is None here, deliberately distinct from ``[]``:
        None means "could not be determined", ``[]`` means "determined, and
        empty".  Collapsing the two would let an unprobeable stand-in render
        as a clean no-divergence result.
        """
        candidate_sha = 'b' * 40
        git_ops = _git_ops(
            citation=None, is_ancestor_map={}, effect_present=False,
        )
        git_ops.describe_commit_effect_in_main = AsyncMock(
            side_effect=AttributeError('describe_commit_effect_in_main'),
        )

        verdict = await validate_landing_evidence(
            git_ops, '42', 'task/42',
            branch_tip_sha=None,
            candidate_sha=candidate_sha,
        )

        assert verdict.accepted is False
        assert verdict.reason == 'effect_absent'
        assert verdict.probe['diverged_paths'] is None
        assert isinstance(verdict.probe['effect_probe_error'], str)
        assert verdict.probe['effect_probe_error']
        assert 'describe_commit_effect_in_main' in verdict.probe['effect_probe_error']


def _verdict(reason: str, **probe_extra) -> LandingEvidenceVerdict:
    """A rejected LandingEvidenceVerdict with a realistic probe (task 3116)."""
    probe = {
        'task_id': '42',
        'branch': 'task/42',
        'branch_tip_sha': None,
        'citation': 'a' * 40,
        'effect_check_sha': 'a' * 40,
        'reason': reason,
    }
    probe.update(probe_extra)
    return LandingEvidenceVerdict(
        accepted=False, evidence_sha=None, reason=reason, probe=probe,
    )


class TestFormatUnattributedLandingDetail:
    """The escalation body a human actually reads (task 3116).

    These assert WHICH BLOCK the renderer selects and what runtime DATA it
    carries into that block — a real branch in ``_render_effect_divergence``
    or ``_render_delivered_checks_differential``, keyed off the verdict's
    probe.  They deliberately do NOT assert the wording of any constant
    paragraph.

    That standard is the one ``test_provenance_gate_integration.py`` states
    for itself ("the operator PROSE around them is deliberately not
    asserted"), and it is adopted here for the same reason: a wording pin on
    a constant f-string exercises no branch, so it cannot catch a defect —
    it can only BLOCK the corrective edit when the semantics it describes
    move.  That is not hypothetical; the retired byte-identity framing had to
    be reworded once part (b) shipped survival semantics, and a substring pin
    on 'cannot distinguish' stood in the way.

    The prose still matters — the original 'reverted exactly the paths it
    touched' claim asserted a revert as FACT and sent two investigations
    chasing a revert that never happened (~5.80 USD across tasks
    3653/3640/3717, plus a spurious task_failure escalation and four days
    blocked).  It is kept honest by review of the source strings, which is
    where a reword can be judged against the semantics, rather than by tests
    that only notice the text changed.
    """

    def test_survival_measurement_is_rendered_not_just_carried(self) -> None:
        """The part-(b) numbers that DECIDED the verdict must reach the human.

        ``diverged_paths`` is explicitly demoted by part (b) to a diagnostic
        that "no longer decides anything".  An escalation that printed only
        the paths would show the reader everything except the basis of the
        rejection — and invite precisely the "it says diverged, so it was
        reverted" leap this whole task exists to stop.

        Asserted on RUNTIME DATA (the ratio, its denominator and the
        threshold all come from the probe), not on wording.
        """
        _, detail = format_unattributed_landing_detail(
            '42', 'task/42',
            _verdict(
                'effect_absent',
                diverged_paths=['src/core.py'],
                effect_failure='effect_not_survived',
                aggregate_survival=0.5,
                added_lines_total=60,
                aggregate_threshold=0.98,
            ),
        )

        assert '0.5000' in detail, 'the measured ratio must appear'
        assert '60' in detail, 'the denominator must appear beside the ratio'
        assert '0.98' in detail, 'the threshold applied must appear'
        assert 'BELOW threshold' in detail

    def test_worst_guarded_file_is_named_with_its_floor(self) -> None:
        """The per-file guard is what names a reverted deliverable hidden
        behind a healthy aggregate, so the escalation must say WHICH file and
        at what floor the guard even applied.
        """
        _, detail = format_unattributed_landing_detail(
            '42', 'task/42',
            _verdict(
                'effect_absent',
                diverged_paths=['src/core.py'],
                aggregate_survival=0.99,
                added_lines_total=2000,
                aggregate_threshold=0.98,
                worst_guarded_path='src/deliverable.py',
                worst_guarded_survival=0.10,
                per_file_threshold=0.9,
                per_file_min_added_lines=25,
            ),
        )

        assert 'src/deliverable.py' in detail
        assert '0.1000' in detail
        assert '25' in detail, 'the added-lines floor the guard applies at'

    def test_unmeasured_survival_says_so_rather_than_rendering_none(self) -> None:
        """An all-vacuous branch has aggregate_survival None.  Silence would
        read as "measured and fine"; a bare "None" reads as a bug.
        """
        _, detail = format_unattributed_landing_detail(
            '42', 'task/42',
            _verdict(
                'effect_absent',
                diverged_paths=['obsolete.py'],
                effect_failure='vacuous_effect_absent',
                aggregate_survival=None,
                added_lines_total=0,
                vacuous_paths=['obsolete.py'],
            ),
        )

        assert 'not measured' in detail.lower()
        assert 'survival: None' not in detail
        assert 'vacuous' in detail.lower()

    def test_probe_without_survival_keys_renders_cleanly(self) -> None:
        """A legacy probe carrying only the part-(a) keys must not sprout a
        half-empty survival block — the seven other gate-wiring test files
        construct exactly that shape.

        Asserted against the BLOCK MARKERS, not the bare word 'survival':
        the reason prose legitimately says "did not SURVIVE" and "survival
        semantics", so a substring check on 'survival' would fail here for
        reasons having nothing to do with the block.
        """
        _, detail = format_unattributed_landing_detail(
            '42', 'task/42',
            _verdict('effect_absent', diverged_paths=['a.py']),
        )

        assert 'survival (aggregate)' not in detail
        assert 'worst guarded file' not in detail
        assert 'not measured' not in detail
        assert 'a.py' in detail

    def test_diverged_paths_render_in_a_labelled_block(self) -> None:
        """(b) The path must appear under its own LABELLED header, not merely
        somewhere in the raw ``probe: {...}`` dict repr — which already
        contains it today, so a path-only assertion would pass without the
        fix.
        """
        path = 'tests/infra/harness-layout-baseline.manifest'
        _, detail = format_unattributed_landing_detail(
            '42', 'task/42',
            _verdict(
                'effect_absent',
                diverged_paths=[path],
                effect_failure='paths_diverged',
                effect_anchor_sha='c' * 40,
            ),
        )

        assert 'diverged paths' in detail.lower()
        assert path in detail
        assert 'c' * 40 in detail

    def test_summary_names_the_first_path_and_stays_clamped(self) -> None:
        """(c) The summary feeds Escalation.summary, so the divergence
        fragment must survive the [:200] clamp — pinned with 20 x 150-char
        paths, far past the limit.
        """
        paths = [f'{"p" * 140}/{i:03d}.py' for i in range(20)]
        summary, _ = format_unattributed_landing_detail(
            '42', 'task/42',
            _verdict('effect_absent', diverged_paths=paths,
                     effect_failure='paths_diverged'),
        )

        assert 'diverged:' in summary.lower()
        assert len(summary) <= 200

    def test_structural_failure_does_not_claim_paths_diverged(self) -> None:
        """(d) A structural failure has no path divergence to report; the
        detail must name the failure CODE and must not imply paths diverged.
        """
        _, detail = format_unattributed_landing_detail(
            '42', 'task/42',
            _verdict('effect_absent', diverged_paths=[],
                     effect_failure='empty_branch_merge'),
        )

        assert 'empty_branch_merge' in detail
        assert 'diverged paths (' not in detail.lower()

    def test_race_case_renders_an_explicit_note(self) -> None:
        """(e) The re-probe found the effect PRESENT — main HEAD advanced
        between the decision and the probe.  Render the race explicitly
        rather than silently contradicting the verdict.
        """
        _, detail = format_unattributed_landing_detail(
            '42', 'task/42',
            _verdict('effect_absent', diverged_paths=[], effect_failure=None),
        )
        lowered = detail.lower()

        assert 'main head may have advanced' in lowered
        assert 'found the effect present' in lowered

    def test_probe_error_says_paths_could_not_be_determined(self) -> None:
        """(f) An unprobeable git_ops must read as "unknown", never as "no
        divergence", and the error text itself must reach the human.
        """
        _, detail = format_unattributed_landing_detail(
            '42', 'task/42',
            _verdict(
                'effect_absent',
                diverged_paths=None,
                effect_probe_error="AttributeError('describe_commit_effect_in_main')",
            ),
        )
        lowered = detail.lower()

        assert 'could not be determined' in lowered
        assert 'describe_commit_effect_in_main' in detail

    def test_no_citation_verdict_renders_unchanged(self) -> None:
        """(g) The divergence block is effect_absent-SCOPED — a no_citation
        verdict gets neither the block nor the summary fragment.
        """
        summary, detail = format_unattributed_landing_detail(
            '42', 'task/42', _verdict('no_citation'),
        )

        assert 'diverged paths' not in detail.lower()
        assert 'diverged:' not in summary.lower()

    def test_legacy_empty_probe_renders_without_raising(self) -> None:
        """(h) Several existing call-site tests construct ``probe={}``; the
        block is skipped when the keys are absent rather than raising.
        """
        verdict = LandingEvidenceVerdict(
            accepted=False, evidence_sha=None, reason='effect_absent', probe={},
        )

        summary, detail = format_unattributed_landing_detail('42', 'task/42', verdict)

        assert 'effect_absent' in detail
        assert len(summary) <= 200

    def test_detail_names_the_unwired_state_explicitly(self) -> None:
        """(e)/step-16.7 — the escalation must SAY when the second accept path
        could not run because this call site supplies no checks.  Silently
        omitting it would hide exactly the degradation the amendment warns
        about: a permanently unwired site looks like a task with nothing to
        check.
        """
        verdict = LandingEvidenceVerdict(
            accepted=False, evidence_sha=None, reason='effect_absent',
            probe={
                'task_id': '42', 'diverged_paths': ['shared/manifest.py'],
                'delivered_checks_state': 'unwired',
            },
        )

        _summary, detail = format_unattributed_landing_detail('42', 'task/42', verdict)

        assert 'delivered-checks differential' in detail
        assert 'unwired' in detail

    def test_detail_distinguishes_none_declared_from_unwired(self) -> None:
        """A task that declares no checks is a TASK-AUTHORING gap; an unwired
        call site is an ORCHESTRATOR gap.  They are fixed by different people,
        so the escalation must not render them identically.
        """
        def _detail(state: str, **extra: object) -> str:
            probe = {'task_id': '42', 'diverged_paths': [], 'delivered_checks_state': state}
            probe.update(extra)
            return format_unattributed_landing_detail(
                '42', 'task/42',
                LandingEvidenceVerdict(
                    accepted=False, evidence_sha=None, reason='effect_absent',
                    probe=probe,
                ),
            )[1]

        unwired = _detail('unwired')
        none_declared = _detail('none_declared')
        evaluated = _detail(
            'evaluated',
            delivered_checks_outcome='no_signal',
            delivered_checks_legs=[{
                'name': 'cap-x', 'verdict': 'no_signal',
                'parent': 'failed', 'citation': 'failed', 'main': 'failed',
            }],
        )

        assert unwired != none_declared
        assert 'no delivered_checks' in none_declared
        assert 'cap-x' in evaluated


class TestFileUnattributedLandingEscalationDedup:
    """Category-scoped L1 dedup (task 3116).

    ``has_open_l1(task_id)`` with the default ``category=None`` matches ANY
    open L1 on the task, which is a TWO-WAY BLINDFOLD: an unrelated open L1
    (a task_failure, say) silently suppresses a provenance_unattributed
    filing, so a provenance defect hides behind an escalation that has
    nothing to do with it.  Observed live on task 4105.

    Narrowing the dedup to ``category='provenance_unattributed'`` slightly
    RAISES L1 volume — a task can now hold one open L1 per category rather
    than one overall.  That is INTENDED per the task amendment: not hiding
    provenance defects behind unrelated escalations is worth the extra
    volume.  Do not "fix" this back.
    """

    def test_dedup_call_is_category_scoped(self) -> None:
        """(a) The dedup probe must name the category, not pass a bare id."""
        queue = MagicMock()
        queue.has_open_l1.return_value = False

        file_unattributed_landing_escalation(
            queue, '42', 'task/42', _verdict('effect_absent'),
            agent_role='harness-reconcile',
        )

        queue.has_open_l1.assert_called_once_with(
            '42', category='provenance_unattributed',
        )

    def test_open_same_category_l1_still_suppresses(self) -> None:
        """(b) Dedup WITHIN the category is preserved: an already-open
        provenance_unattributed L1 must still suppress a duplicate filing, so
        repeated ticks re-observing the same evidence don't stack L1s.
        """
        queue = MagicMock()
        queue.has_open_l1.side_effect = (
            lambda task_id, *, category=None: category == 'provenance_unattributed'
        )

        file_unattributed_landing_escalation(
            queue, '42', 'task/42', _verdict('effect_absent'),
            agent_role='harness-reconcile',
        )

        queue.submit.assert_not_called()

    def test_open_other_category_l1_no_longer_suppresses(self) -> None:
        """(c) THE BUG, and the whole point: an open L1 of a DIFFERENT
        category (e.g. task_failure) must no longer hide a provenance
        defect.  Today's bare call suppresses this filing entirely.
        """
        queue = MagicMock()
        queue.has_open_l1.side_effect = (
            lambda task_id, *, category=None: category is None
        )
        queue.make_id.return_value = 'esc-42-1'

        file_unattributed_landing_escalation(
            queue, '42', 'task/42', _verdict('effect_absent'),
            agent_role='harness-reconcile',
        )

        queue.submit.assert_called_once()
        esc = queue.submit.call_args.args[0]
        assert esc.category == 'provenance_unattributed'

    def test_none_queue_is_a_noop(self) -> None:
        """(d) Best-effort: a bare-harness/bare-worker unit test passes None."""
        file_unattributed_landing_escalation(
            None, '42', 'task/42', _verdict('effect_absent'),
            agent_role='harness-reconcile',
        )

    def test_raising_queue_is_contained(self) -> None:
        """(e) The existing try/except still contains a queue that raises —
        an escalation filer must never break its caller.
        """
        queue = MagicMock()
        queue.has_open_l1.side_effect = RuntimeError('queue exploded')

        file_unattributed_landing_escalation(
            queue, '42', 'task/42', _verdict('effect_absent'),
            agent_role='harness-reconcile',
        )

        queue.submit.assert_not_called()


def _grep_check(
    name: str = 'cap-x', *, pattern: str = 'def new_capability', expect: str = 'present',
) -> dict[str, object]:
    """A minimal, VALID ``metadata.delivered_checks`` grep entry.

    Kept schema-valid (``shared.capability_manifest.DeliveredCheckMeta``)
    rather than a bare stub: the differential hands these straight to
    ``run_delivered_check``, which re-validates, so an invalid fixture would
    degrade to ERRORED and silently test nothing.
    """
    return {'name': name, 'kind': 'grep', 'pattern': pattern, 'expect': expect}


def _script_check(name: str = 'cap-script') -> dict[str, object]:
    """A VALID script-kind entry — the 2-of-458 carve-out in the live store."""
    return {'name': name, 'kind': 'script', 'script': 'scripts/check.sh', 'timeout_secs': 5}


def _check_runner(outcomes):
    """Return a (stub_run_delivered_check, calls) pair.

    *outcomes* maps ``ref`` -> :class:`DeliveredCheckResult`, or is a callable
    ``(check, ref) -> DeliveredCheckResult`` for per-check control.  ``calls``
    records ``(check name, ref)`` in order, which is what lets a test assert
    the differential probed the PARENT, the CITATION and MAIN — a static pass
    at main alone proves nothing, so the sequence IS the signal.
    """
    calls: list[tuple[str, str]] = []

    async def _stub(check, *, project_root, ref='main', runner=None):
        calls.append((check['name'], ref))
        if callable(outcomes):
            return outcomes(check, ref)
        return outcomes[ref]

    return _stub, calls


@pytest.mark.asyncio
class TestValidateLandingEvidenceDeliveredChecksDifferential:
    """The delivered-checks DIFFERENTIAL — the SECOND accept path (task 3116).

    Threshold line survival recovers most of the false-positive class, but it
    is still a heuristic over line sets.  The differential is orthogonal
    positive evidence: run the task's own declared capability check at the
    citation's PRE-MERGE PARENT, at the citation, and at current main.  A
    check that was FALSE before this commit, TRUE at it, and TRUE now proves
    THIS commit made the capability true — which no amount of line-set decay
    can argue with.

    THE THREE-PROBE SEQUENCE IS THE SIGNAL.  A check that merely passes at
    main proves nothing: the capability might have arrived by any other route,
    or have been true all along.  Every test here asserts the refs.

    THE CRITICAL RULE IS UPGRADE-ONLY.  The live store's checks are not
    trustworthy input: they rot (a path is renamed and the check fails
    forever), they are written too broad (already true before the merge), and
    2 of 458 are script-kind, which takes no ref at all.  So a failing,
    erroring or nonsensical differential must degrade to NO SIGNAL and leave
    the survival verdict exactly as it found it.  That is enforced
    STRUCTURALLY — the differential is only ever consulted inside the reject
    branch — not by care.
    """

    @staticmethod
    def _rejecting_git_ops(candidate_sha: str) -> MagicMock:
        """A git_ops whose survival check REJECTS — the only path the
        differential is ever consulted on.
        """
        return _git_ops(
            citation=None,
            is_ancestor_map={},
            effect_present=False,
            effect_probe=CommitEffectProbe(
                present=False,
                diverged_paths=('shared/manifest.py',),
                anchor_sha=candidate_sha,
                failure='effect_not_survived',
                aggregate_survival=0.42,
            ),
        )

    async def test_differential_upgrades_a_survival_rejection(self) -> None:
        """(a) FAIL at the parent, PASS at the citation, PASS at main ->
        a verdict survival REJECTED is ACCEPTED.

        The three refs are asserted in order because establishing that THIS
        merge made the capability true is the entire signal.
        """
        sha = 'c' * 40
        git_ops = self._rejecting_git_ops(sha)
        stub, calls = _check_runner({
            f'{sha}^1': DeliveredCheckResult.FAILED,
            sha: DeliveredCheckResult.DELIVERED,
            'main': DeliveredCheckResult.DELIVERED,
        })

        with patch('orchestrator.delivered_checks.run_delivered_check', stub):
            verdict = await validate_landing_evidence(
                git_ops, '42', 'task/42', branch_tip_sha=None, candidate_sha=sha,
                delivered_checks=[_grep_check()],
            )

        assert verdict.accepted is True
        assert verdict.reason == 'ok'
        assert verdict.evidence_sha == sha
        # The survival check ran and rejected FIRST — this is a rescue of its
        # verdict, not a bypass of it.
        git_ops.commit_effect_present_in_main.assert_awaited_once_with(sha)
        assert calls == [
            ('cap-x', f'{sha}^1'), ('cap-x', sha), ('cap-x', 'main'),
        ], f'the differential must probe parent, citation and main: {calls}'
        assert verdict.probe['delivered_checks_state'] == 'evaluated'
        assert verdict.probe['delivered_checks_outcome'] == 'confirmed'

    async def test_differential_is_never_consulted_when_survival_accepts(self) -> None:
        """(b) UPGRADE-ONLY, enforced structurally: when survival ACCEPTS, the
        differential does not run at all, so no rotten check can downgrade it.

        Asserting zero runner calls (not merely an accepted verdict) is what
        pins the structure — a differential wired as an additional conjunct
        would still call the runner here.
        """
        sha = 'c' * 40
        git_ops = _git_ops(citation=None, is_ancestor_map={}, effect_present=True)
        stub, calls = _check_runner({
            f'{sha}^1': DeliveredCheckResult.DELIVERED,
            sha: DeliveredCheckResult.FAILED,
            'main': DeliveredCheckResult.FAILED,
        })

        with patch('orchestrator.delivered_checks.run_delivered_check', stub):
            verdict = await validate_landing_evidence(
                git_ops, '42', 'task/42', branch_tip_sha=None, candidate_sha=sha,
                delivered_checks=[_grep_check()],
            )

        assert verdict.accepted is True
        assert calls == [], (
            f'a check that fails at main must not be able to reach an accepted '
            f'verdict at all; saw {calls}'
        )

    @pytest.mark.parametrize(
        ('rot', 'outcomes'),
        [
            (
                'fails at main (the capability is genuinely gone, or the check rotted)',
                {
                    'parent': DeliveredCheckResult.FAILED,
                    'citation': DeliveredCheckResult.DELIVERED,
                    'main': DeliveredCheckResult.FAILED,
                },
            ),
            (
                'names a path that no longer exists — errors forever',
                {
                    'parent': DeliveredCheckResult.ERRORED,
                    'citation': DeliveredCheckResult.ERRORED,
                    'main': DeliveredCheckResult.ERRORED,
                },
            ),
            (
                'pattern so broad it was already true before the merge',
                {
                    'parent': DeliveredCheckResult.DELIVERED,
                    'citation': DeliveredCheckResult.DELIVERED,
                    'main': DeliveredCheckResult.DELIVERED,
                },
            ),
            (
                'never true on any ref',
                {
                    'parent': DeliveredCheckResult.FAILED,
                    'citation': DeliveredCheckResult.FAILED,
                    'main': DeliveredCheckResult.FAILED,
                },
            ),
        ],
    )
    async def test_rotten_differentials_degrade_to_no_signal(
        self, rot: str, outcomes: dict[str, DeliveredCheckResult],
    ) -> None:
        """(b) Every rotten shape the live store actually contains leaves the
        survival REJECTION exactly as it was — never a harder rejection, never
        an exception, always an explicit NO SIGNAL in the probe.
        """
        sha = 'c' * 40
        git_ops = self._rejecting_git_ops(sha)
        by_ref = {
            f'{sha}^1': outcomes['parent'],
            sha: outcomes['citation'],
            'main': outcomes['main'],
        }
        stub, calls = _check_runner(by_ref)

        with patch('orchestrator.delivered_checks.run_delivered_check', stub):
            verdict = await validate_landing_evidence(
                git_ops, '42', 'task/42', branch_tip_sha=None, candidate_sha=sha,
                delivered_checks=[_grep_check()],
            )

        assert verdict.accepted is False, f'rot ({rot}) must not flip the verdict'
        assert verdict.reason == 'effect_absent', 'the reason must be unchanged'
        assert verdict.probe['delivered_checks_outcome'] == 'no_signal'
        assert calls, 'the differential must actually have been attempted'

    async def test_a_raising_differential_is_contained_as_no_signal(self) -> None:
        """(b) The differential is an OPTIONAL second opinion, so it must never
        break the gate — but it must not be swallowed either.  The failure is
        recorded into the probe that gets rendered into the escalation.
        """
        sha = 'c' * 40
        git_ops = self._rejecting_git_ops(sha)

        async def _boom(check, *, project_root, ref='main', runner=None):
            raise RuntimeError('delivered-check runner exploded')

        with patch('orchestrator.delivered_checks.run_delivered_check', _boom):
            verdict = await validate_landing_evidence(
                git_ops, '42', 'task/42', branch_tip_sha=None, candidate_sha=sha,
                delivered_checks=[_grep_check()],
            )

        assert verdict.accepted is False
        assert verdict.reason == 'effect_absent'
        assert verdict.probe['delivered_checks_outcome'] == 'no_signal'
        assert 'delivered-check runner exploded' in str(
            verdict.probe.get('delivered_checks_error'),
        )

    async def test_expect_absent_check_still_upgrades(self) -> None:
        """(c) 43 of the live store's 458 checks are ``expect='absent'`` — a
        capability whose delivery is a REMOVAL.  These are exactly the
        deletion-shaped deliverables the vacuous arm (b3) also special-cases;
        getting both wrong at once is the failure this pins.

        ``run_delivered_check`` already resolves ``expect`` into
        DELIVERED/FAILED (for ``absent`` it is NO-match that means delivered),
        so the differential's legs read uniformly and this layer must NOT
        invert them a second time.
        """
        sha = 'c' * 40
        git_ops = self._rejecting_git_ops(sha)
        stub, calls = _check_runner({
            f'{sha}^1': DeliveredCheckResult.FAILED,
            sha: DeliveredCheckResult.DELIVERED,
            'main': DeliveredCheckResult.DELIVERED,
        })

        with patch('orchestrator.delivered_checks.run_delivered_check', stub):
            verdict = await validate_landing_evidence(
                git_ops, '42', 'task/42', branch_tip_sha=None, candidate_sha=sha,
                delivered_checks=[_grep_check('cap-removed', expect='absent')],
            )

        assert verdict.accepted is True
        assert verdict.probe['delivered_checks_outcome'] == 'confirmed'
        assert [ref for _name, ref in calls] == [f'{sha}^1', sha, 'main']

    async def test_expect_absent_is_not_inverted_twice(self) -> None:
        """(c), the other side: the INVERSE leg pattern (delivered at the
        parent, failing at the citation and at main) is a capability that was
        REMOVED by a later commit, not delivered by this one.

        A second inversion applied here would read it as proof and upgrade —
        the precise double-inversion bug this pins against.
        """
        sha = 'c' * 40
        git_ops = self._rejecting_git_ops(sha)
        stub, _calls = _check_runner({
            f'{sha}^1': DeliveredCheckResult.DELIVERED,
            sha: DeliveredCheckResult.FAILED,
            'main': DeliveredCheckResult.FAILED,
        })

        with patch('orchestrator.delivered_checks.run_delivered_check', stub):
            verdict = await validate_landing_evidence(
                git_ops, '42', 'task/42', branch_tip_sha=None, candidate_sha=sha,
                delivered_checks=[_grep_check('cap-removed', expect='absent')],
            )

        assert verdict.accepted is False
        assert verdict.probe['delivered_checks_outcome'] == 'no_signal'

    async def test_script_kind_contributes_no_signal_and_never_raises(self) -> None:
        """(d) A script check execs against the LIVE CHECKOUT and takes no ref,
        so it cannot express a differential at all.  It is skipped explicitly
        — recorded as a carve-out, never run three times against a checkout
        that would answer identically each time.
        """
        sha = 'c' * 40
        git_ops = self._rejecting_git_ops(sha)
        stub, calls = _check_runner({})

        with patch('orchestrator.delivered_checks.run_delivered_check', stub):
            verdict = await validate_landing_evidence(
                git_ops, '42', 'task/42', branch_tip_sha=None, candidate_sha=sha,
                delivered_checks=[_script_check()],
            )

        assert verdict.accepted is False
        assert calls == [], 'a script check must not be run against a ref'
        assert verdict.probe['delivered_checks_outcome'] == 'no_signal'
        legs = verdict.probe['delivered_checks_legs']
        assert len(legs) == 1
        assert legs[0]['name'] == 'cap-script'
        assert legs[0]['verdict'] == 'script_kind_no_signal'

    async def test_a_script_check_does_not_suppress_a_greppable_one(self) -> None:
        """(d) The carve-out is per CHECK, not per task: a task carrying both
        kinds still gets the differential from the greppable one.
        """
        sha = 'c' * 40
        git_ops = self._rejecting_git_ops(sha)
        stub, calls = _check_runner({
            f'{sha}^1': DeliveredCheckResult.FAILED,
            sha: DeliveredCheckResult.DELIVERED,
            'main': DeliveredCheckResult.DELIVERED,
        })

        with patch('orchestrator.delivered_checks.run_delivered_check', stub):
            verdict = await validate_landing_evidence(
                git_ops, '42', 'task/42', branch_tip_sha=None, candidate_sha=sha,
                delivered_checks=[_script_check(), _grep_check()],
            )

        assert verdict.accepted is True
        assert {name for name, _ref in calls} == {'cap-x'}

    async def test_three_supply_states_are_distinguishable_in_probe(self) -> None:
        """(e) THE UNWIRED HAZARD.  ``unwired`` (the parameter was never
        passed) and ``none_declared`` (the task declares no checks) must not
        collapse into one value: without the distinction a permanently
        unwired call site looks exactly like a task with nothing to check, and
        the capstone task that flips this parameter to required has no signal
        to act on.
        """
        sha = 'c' * 40
        stub, _calls = _check_runner({
            f'{sha}^1': DeliveredCheckResult.FAILED,
            sha: DeliveredCheckResult.DELIVERED,
            'main': DeliveredCheckResult.DELIVERED,
        })

        unwired = await validate_landing_evidence(
            self._rejecting_git_ops(sha), '42', 'task/42',
            branch_tip_sha=None, candidate_sha=sha,
        )
        none_declared = await validate_landing_evidence(
            self._rejecting_git_ops(sha), '42', 'task/42',
            branch_tip_sha=None, candidate_sha=sha, delivered_checks=[],
        )
        with patch('orchestrator.delivered_checks.run_delivered_check', stub):
            evaluated = await validate_landing_evidence(
                self._rejecting_git_ops(sha), '42', 'task/42',
                branch_tip_sha=None, candidate_sha=sha,
                delivered_checks=[_grep_check()],
            )

        assert unwired.probe['delivered_checks_state'] == 'unwired'
        assert none_declared.probe['delivered_checks_state'] == 'none_declared'
        assert evaluated.probe['delivered_checks_state'] == 'evaluated'
        assert len({
            unwired.probe['delivered_checks_state'],
            none_declared.probe['delivered_checks_state'],
            evaluated.probe['delivered_checks_state'],
        }) == 3

    async def test_supply_state_is_recorded_on_accepted_verdicts_too(self) -> None:
        """(e) The wiring state is a property of the CALL SITE, not of the
        outcome, so it is recorded on accepts as well — otherwise the only way
        to learn a site is unwired would be to wait for it to reject.
        """
        sha = 'c' * 40
        git_ops = _git_ops(citation=None, is_ancestor_map={}, effect_present=True)

        verdict = await validate_landing_evidence(
            git_ops, '42', 'task/42', branch_tip_sha=None, candidate_sha=sha,
        )

        assert verdict.accepted is True
        assert verdict.probe['delivered_checks_state'] == 'unwired'

    async def test_discovery_mode_differential_anchors_on_the_effect_check_sha(
        self,
    ) -> None:
        """DISCOVERY mode rescues identically, anchored on the sha the
        SURVIVAL check actually ran against — the branch tip for an in-branch
        work-commit citation — so both modes ask the same question.
        """
        branch = 'task/42'
        branch_tip_sha = 'f' * 40
        citation_sha = 'a' * 40
        git_ops = _git_ops(
            citation=citation_sha,
            is_ancestor_map={(citation_sha, branch): True},
            effect_present=False,
            effect_probe=CommitEffectProbe(present=False, failure='effect_not_survived'),
        )
        stub, calls = _check_runner({
            f'{branch_tip_sha}^1': DeliveredCheckResult.FAILED,
            branch_tip_sha: DeliveredCheckResult.DELIVERED,
            'main': DeliveredCheckResult.DELIVERED,
        })

        with patch('orchestrator.delivered_checks.run_delivered_check', stub):
            verdict = await validate_landing_evidence(
                git_ops, '42', branch, branch_tip_sha=branch_tip_sha,
                delivered_checks=[_grep_check()],
            )

        assert verdict.accepted is True
        assert verdict.evidence_sha == citation_sha
        assert [ref for _name, ref in calls] == [
            f'{branch_tip_sha}^1', branch_tip_sha, 'main',
        ]

    async def test_no_citation_reject_never_runs_the_differential(self) -> None:
        """A missing citation is an ATTRIBUTION failure, not a decayed-effect
        one: there is no commit to anchor a differential on, so the rescue
        must not fire (and must not fabricate one from thin air).
        """
        git_ops = _git_ops(citation=None, is_ancestor_map={}, effect_present=True)
        stub, calls = _check_runner({})

        with patch('orchestrator.delivered_checks.run_delivered_check', stub):
            verdict = await validate_landing_evidence(
                git_ops, '42', 'task/42', branch_tip_sha=None,
                delivered_checks=[_grep_check()],
            )

        assert verdict.accepted is False
        assert verdict.reason == 'no_citation'
        assert calls == []

    async def test_omitting_the_parameter_preserves_the_pre_task_shape(self) -> None:
        """(f) b6 — THE INTERFACE STAYS BINARY.  Every existing caller passes
        no ``delivered_checks``; each must behave exactly as before, so the
        four call-site sibling tasks can wire their own sites concurrently
        against an unchanged contract.
        """
        sha = 'c' * 40
        stub, calls = _check_runner({})

        with patch('orchestrator.delivered_checks.run_delivered_check', stub):
            accepted = await validate_landing_evidence(
                _git_ops(citation=None, is_ancestor_map={}, effect_present=True),
                '42', 'task/42', branch_tip_sha=None, candidate_sha=sha,
            )
            rejected = await validate_landing_evidence(
                self._rejecting_git_ops(sha), '42', 'task/42',
                branch_tip_sha=None, candidate_sha=sha,
            )

        assert isinstance(accepted, LandingEvidenceVerdict)
        assert (accepted.accepted, accepted.reason, accepted.evidence_sha) == (
            True, 'ok', sha,
        )
        assert (rejected.accepted, rejected.reason, rejected.evidence_sha) == (
            False, 'effect_absent', None,
        )
        assert calls == [], 'an unwired call site must issue no check subprocesses'
        # The effect_absent diagnostics from Part A are untouched by Part B.
        assert rejected.probe['diverged_paths'] == ['shared/manifest.py']
