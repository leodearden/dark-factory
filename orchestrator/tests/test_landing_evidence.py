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

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.delivered_checks import DeliveredCheckResult
from orchestrator.git_ops import CommitEffectProbe
from orchestrator.landing_evidence import (
    LandingEvidenceVerdict,
    LandingMethod,
    LandingReason,
    LandingVerdict,
    file_unattributed_landing_escalation,
    format_unattributed_landing_detail,
    validate_landing_evidence,
)


def _git_ops(
    *,
    citation,
    is_ancestor_map,
    effect_present,
    effect_probe=None,
    main_branch='main',
    delivered_checks_enabled=True,
    fork_point=None,
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

    ``config`` is a REAL object, not a MagicMock attribute (amendment pass):
    the differential now reads ``config.main_branch`` for its third leg and
    ``config.delivered_checks.enabled`` for the kill switch, and an
    auto-created MagicMock attribute would sail through both as a truthy
    non-string.  ``fork_point`` stubs ``merge_base_with_main``; left UNSTUBBED
    when omitted so the ``^1`` fallback stays exercised on the shape the other
    gate-wiring files construct.
    """
    git_ops = MagicMock()
    git_ops.config = SimpleNamespace(
        main_branch=main_branch,
        delivered_checks=SimpleNamespace(enabled=delivered_checks_enabled),
    )
    if fork_point is not None:
        git_ops.merge_base_with_main = AsyncMock(return_value=fork_point)
    else:
        del git_ops.merge_base_with_main
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


#: "argument not supplied", distinct from an explicitly-passed ``None`` — for
#: helpers whose default must not collide with a MEANINGFUL None (task 4499).
#: Annotated ``Any`` so it can stand in as the default of a narrower parameter,
#: exactly as the sibling sentinel in escalation/tests/test_queue.py does.
_UNSET: Any = object()


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
        # Explicit (task 4499): the filer's auto-dismiss guard reads a TERMINAL
        # record on this citation, so "nothing was previously adjudicated" must
        # be STATED, not inherited from MagicMock's truthy auto-child — which
        # would suppress the filing and silently undo task 3116's fix.
        queue.find_terminal_by_citation.return_value = None
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
        # Explicit (task 4499): the filer's auto-dismiss guard reads a TERMINAL
        # record on this citation, so "nothing was previously adjudicated" must
        # be STATED, not inherited from MagicMock's truthy auto-child — which
        # would suppress the filing and silently undo task 3116's fix.
        queue.find_terminal_by_citation.return_value = None
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
        # Explicit (task 4499): the filer's auto-dismiss guard reads a TERMINAL
        # record on this citation, so "nothing was previously adjudicated" must
        # be STATED, not inherited from MagicMock's truthy auto-child — which
        # would suppress the filing and silently undo task 3116's fix.
        queue.find_terminal_by_citation.return_value = None
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
        # Explicit (task 4499): stated so the ABSENCE of a filing below is
        # unambiguously the raising has_open_l1 being contained, and not the
        # auto-dismiss guard suppressing on MagicMock's truthy auto-child.
        # (has_open_l1 raises first, so this is never reached — which is the
        # point: the precondition is stated, never inherited.)
        queue.find_terminal_by_citation.return_value = None
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

        This git_ops carries NO ``merge_base_with_main`` (the shape the other
        gate-wiring test files construct), so the parent leg falls back to
        ``tip^1``.  The fallback is pinned deliberately: a stand-in without
        the method must degrade to the weaker baseline, never raise and never
        skip the differential.
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

    async def test_branch_tip_anchor_uses_the_fork_point_not_the_tips_parent(
        self,
    ) -> None:
        """THE MULTI-COMMIT BRANCH — a branch tip's ``^1`` is NOT a baseline.

        DISCOVERY mode anchors on the BRANCH TIP whenever the citation is an
        in-branch work commit, and ``tip^1`` is then the branch's own previous
        work commit.  For a two-commit branch whose deliverable landed in the
        FIRST commit, the capability is already true at ``tip^1``: the parent
        leg reads DELIVERED, the FAIL/PASS/PASS sequence never forms, and the
        second accept path silently declines — for precisely the multi-commit
        shape most likely to have tripped the survival heuristic in the first
        place.

        The branch's FORK POINT from main is the honest "before this landing"
        ref, and this fixture is arranged so the two answers DISAGREE: FAILED
        at the fork point, DELIVERED at ``tip^1``.  Anchoring on ``^1`` gives
        no_signal; anchoring on the fork point rescues the landing.
        """
        branch = 'task/42'
        branch_tip_sha = 'f' * 40
        citation_sha = 'a' * 40
        fork_point = '9' * 40
        git_ops = _git_ops(
            citation=citation_sha,
            is_ancestor_map={(citation_sha, branch): True},
            effect_present=False,
            effect_probe=CommitEffectProbe(present=False, failure='effect_not_survived'),
            fork_point=fork_point,
        )
        stub, calls = _check_runner({
            fork_point: DeliveredCheckResult.FAILED,
            # The deliverable landed in the branch's FIRST commit, so it is
            # already true one commit back from the tip.
            f'{branch_tip_sha}^1': DeliveredCheckResult.DELIVERED,
            branch_tip_sha: DeliveredCheckResult.DELIVERED,
            'main': DeliveredCheckResult.DELIVERED,
        })

        with patch('orchestrator.delivered_checks.run_delivered_check', stub):
            verdict = await validate_landing_evidence(
                git_ops, '42', branch, branch_tip_sha=branch_tip_sha,
                delivered_checks=[_grep_check()],
            )

        git_ops.merge_base_with_main.assert_awaited_once_with(branch_tip_sha)
        assert [ref for _name, ref in calls] == [
            fork_point, branch_tip_sha, 'main',
        ], 'the parent leg must be the fork point, not the tip\'s first parent'
        assert verdict.accepted is True, (
            'anchored on tip^1 this landing reads as no_signal and stays '
            'rejected — the fork point is what makes the branch the subject '
            'of the question'
        )
        assert verdict.probe['delivered_checks_parent_ref'] == fork_point

    async def test_candidate_mode_keeps_the_first_parent_as_the_baseline(
        self,
    ) -> None:
        """CANDIDATE mode anchors on a MERGE COMMIT, whose ``^1`` IS
        main-before-the-merge — the correct pre-landing baseline.

        So the fork-point substitution must NOT happen here even when
        ``merge_base_with_main`` is available: merge-base(main, <a commit
        already on main>) is that commit itself, which would collapse the
        parent and citation legs into the same ref and make the differential
        structurally incapable of ever confirming.
        """
        sha = 'c' * 40
        git_ops = self._rejecting_git_ops(sha)
        git_ops.merge_base_with_main = AsyncMock(return_value=sha)
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
        git_ops.merge_base_with_main.assert_not_awaited()
        assert [ref for _name, ref in calls] == [f'{sha}^1', sha, 'main']

    async def test_the_main_leg_honours_a_configured_main_branch(self) -> None:
        """The third leg names ``config.main_branch``, not the literal 'main'.

        ``main_branch`` is a configurable Pydantic field.  On a project that
        sets it to anything else, a hardcoded 'main' is an unresolvable ref:
        ``run_delivered_check`` returns ERRORED, that degrades to no_signal,
        and the second accept path is PERMANENTLY dead there with no
        diagnostic saying why.  Everything else in this check threads the
        configured branch through, so this was the one place that would
        silently mis-target.
        """
        sha = 'c' * 40
        git_ops = self._rejecting_git_ops(sha)
        git_ops.config = SimpleNamespace(
            main_branch='trunk',
            delivered_checks=SimpleNamespace(enabled=True),
        )
        stub, calls = _check_runner({
            f'{sha}^1': DeliveredCheckResult.FAILED,
            sha: DeliveredCheckResult.DELIVERED,
            'trunk': DeliveredCheckResult.DELIVERED,
            # A hardcoded 'main' would land here instead.
            'main': DeliveredCheckResult.ERRORED,
        })

        with patch('orchestrator.delivered_checks.run_delivered_check', stub):
            verdict = await validate_landing_evidence(
                git_ops, '42', 'task/42', branch_tip_sha=None, candidate_sha=sha,
                delivered_checks=[_grep_check()],
            )

        assert [ref for _name, ref in calls] == [f'{sha}^1', sha, 'trunk']
        assert verdict.accepted is True

    async def test_the_kill_switch_disables_the_second_accept_path(self) -> None:
        """``delivered_checks.enabled=False`` switches this consumer off too.

        The mark-done delivered-check gate honours that flag
        (``Harness._delivered_checks_withhold``); a second consumer that
        ignored it would mean disabling the feature did not actually disable
        it.  The rejection is left exactly as survival found it, no check is
        executed, and the state is RECORDED rather than looking like ordinary
        bad luck — an operator must be able to see in the escalation that the
        path was switched off, not merely unlucky.
        """
        sha = 'c' * 40
        git_ops = self._rejecting_git_ops(sha)
        git_ops.config = SimpleNamespace(
            main_branch='main',
            delivered_checks=SimpleNamespace(enabled=False),
        )
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

        assert calls == [], 'no declared check may run while the feature is off'
        assert verdict.accepted is False
        assert verdict.reason == 'effect_absent'
        assert verdict.probe['delivered_checks_outcome'] == 'disabled'
        _summary, detail = format_unattributed_landing_detail('42', 'task/42', verdict)
        assert 'delivered_checks.enabled is false' in detail

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


# ---------------------------------------------------------------------------
# task 4647 — validate_landing_evidence re-expressed as a MODE over the shared
# producer family.
#
# The PRD's epsilon bullet asks for ONE producer family with an explicit
# mode/policy discriminator, not two functions that must be kept in step
# forever.  These tests pin the two halves of that: the discriminator is
# readable (`method`), and NOTHING ELSE observable moved.  The second half is
# the load-bearing one — this function has seven incumbent production call
# sites and four test files pinning its reason strings, so an unchanged public
# surface is what lets it be re-expressed at all.
# ---------------------------------------------------------------------------


class TestValidateLandingEvidencePublicSurface:
    """The re-expression kept ONE verdict type, under both names.

    Task 4500 is the capstone that adds a parameter to
    ``validate_landing_evidence``; its precondition is that this task did not
    move the incumbent shape underneath it.  That is preserved by the call
    sites, not mirrored here: the whole surrounding suite invokes this function
    by keyword in its incumbent shape, so a reordering, a renamed keyword or a
    defaulted ``branch_tip_sha`` fails loudly and behaviourally.  A copy of the
    signature living in this file would only add a pin on keyword-only
    parameter ORDER, which no caller can observe — and if 4500 wants a
    tripwire, it belongs on the call 4500 itself makes.
    """

    def test_the_two_verdict_names_are_one_type(self) -> None:
        """ONE authority, not two — an alias, never a second dataclass.

        Both producers return this; a consumer must never have to know which
        one answered in order to read the verdict.
        """
        assert LandingEvidenceVerdict is LandingVerdict


@pytest.mark.asyncio
class TestValidateLandingEvidenceModeDiscriminator:
    """``method`` is the explicit mode/policy discriminator.

    It is what makes one producer family legible: a consumer can read whether
    a verdict came from the NON-DECAYING patch-id contract or from the legacy
    effect-present policy, without inferring it from the reason code or from
    which function it happened to call.  Leaf epsilon retires the legacy
    policy at the landing-detection sites and needs exactly this to tell them
    apart.
    """

    @staticmethod
    def _rejecting_git_ops(sha: str) -> MagicMock:
        return _git_ops(
            citation=sha,
            is_ancestor_map={},
            effect_present=False,
            effect_probe=CommitEffectProbe(
                present=False, diverged_paths=('pkg/a.py',), failure=None,
                anchor_sha=sha,
            ),
        )

    async def test_candidate_mode_reports_the_merge_marker_method(self) -> None:
        sha = 'a' * 40
        verdict = await validate_landing_evidence(
            _git_ops(citation=None, is_ancestor_map={}, effect_present=True),
            '42', 'task/42', branch_tip_sha=None, candidate_sha=sha,
        )
        assert verdict.accepted is True
        assert verdict.method is LandingMethod.merge_marker

    async def test_candidate_mode_reports_it_on_a_REJECT_too(self) -> None:
        """WHICH policy answered is a property of the CALL, not of the outcome.

        A consumer reading a rejected verdict is exactly the one that needs to
        know which policy produced it — that is the reader deciding whether to
        re-dispatch, escalate, or distrust the answer.
        """
        sha = 'b' * 40
        verdict = await validate_landing_evidence(
            self._rejecting_git_ops(sha), '42', 'task/42',
            branch_tip_sha=None, candidate_sha=sha,
        )
        assert verdict.accepted is False
        assert verdict.method is LandingMethod.merge_marker

    async def test_discovery_mode_reports_the_citation_method(self) -> None:
        branch, tip, citation = 'task/42', 'f' * 40, 'a' * 40
        verdict = await validate_landing_evidence(
            _git_ops(
                citation=citation,
                is_ancestor_map={(citation, branch): True},
                effect_present=True,
            ),
            '42', branch, branch_tip_sha=tip,
        )
        assert verdict.accepted is True
        assert verdict.method is LandingMethod.citation

    async def test_discovery_mode_reports_it_on_a_no_citation_miss(self) -> None:
        verdict = await validate_landing_evidence(
            _git_ops(citation=None, is_ancestor_map={}, effect_present=True),
            '42', 'task/42', branch_tip_sha='f' * 40,
        )
        assert verdict.reason == 'no_citation'
        assert verdict.method is LandingMethod.citation

    async def test_discovery_mode_reports_it_on_an_effect_absent_reject(self) -> None:
        branch, citation = 'task/42', 'a' * 40
        verdict = await validate_landing_evidence(
            _git_ops(
                citation=citation,
                is_ancestor_map={(citation, branch): False},
                effect_present=False,
            ),
            '42', branch, branch_tip_sha=None,
        )
        assert verdict.reason == 'effect_absent'
        assert verdict.method is LandingMethod.citation

    async def test_it_is_NEVER_the_patch_id_contract(self) -> None:
        """The whole point of the discriminator.

        ``patch_id`` marks the non-decaying producer.  If this legacy policy
        ever claimed it, a consumer would trust a DECAYING answer as if it
        could not decay — which is strictly worse than having no discriminator
        at all, because it converts an unknown into a wrong certainty.
        """
        branch, tip, citation = 'task/42', 'f' * 40, 'a' * 40
        verdicts = [
            await validate_landing_evidence(
                _git_ops(
                    citation=citation,
                    is_ancestor_map={(citation, branch): True},
                    effect_present=True,
                ),
                '42', branch, branch_tip_sha=tip,
            ),
            await validate_landing_evidence(
                _git_ops(citation=None, is_ancestor_map={}, effect_present=True),
                '42', branch, branch_tip_sha=tip,
            ),
            await validate_landing_evidence(
                _git_ops(citation=None, is_ancestor_map={}, effect_present=True),
                '42', branch, branch_tip_sha=None, candidate_sha=citation,
            ),
        ]
        for verdict in verdicts:
            assert verdict.method is not LandingMethod.patch_id
            assert verdict.method is not LandingMethod.unspecified, (
                'a verdict this module produced always names the path that '
                'produced it; unspecified means hand-constructed'
            )


@pytest.mark.asyncio
class TestValidateLandingEvidenceBehaviourPreservation:
    """NOTHING OBSERVABLE MOVED — the half that makes the change safe.

    Seven production call sites (harness.py x4, merge_queue.py x1,
    escalation/server.py x2) and five test files read this function's reason
    codes.  The reason values become :class:`LandingReason` MEMBERS, which are
    genuine ``str`` subclasses, so every incumbent comparison against a plain
    string holds unchanged and no consumer edit is needed.  The rename to the
    Contract's ``no_attribution`` spelling is REGISTERED (so no escalation can
    render 'Unrecognized reason code') but not yet EMITTED here; leaf epsilon
    flips the emitted value when it repoints the consumers.
    """

    async def test_an_accept_still_reports_ok(self) -> None:
        sha = 'a' * 40
        verdict = await validate_landing_evidence(
            _git_ops(citation=None, is_ancestor_map={}, effect_present=True),
            '42', 'task/42', branch_tip_sha=None, candidate_sha=sha,
        )
        assert verdict.reason == 'ok'
        assert verdict.reason is LandingReason.ok
        assert verdict.probe['reason'] == 'ok'
        # A StrEnum member must survive the round trips a consumer subjects it
        # to: an f-string, a dict key, and JSON.
        assert f'{verdict.reason}' == 'ok'
        # Annotated rather than inferred: the `is LandingReason.ok` assert
        # above narrows `verdict.reason` to the member type, which would make
        # the plain-string lookup a type error while still passing at runtime.
        # A member IS a str, so `dict[str, int]` is the honest key type — and
        # spelling it out is precisely the round trip being claimed.
        keyed_by_member: dict[str, int] = {verdict.reason: 1}
        assert keyed_by_member['ok'] == 1
        assert json.loads(json.dumps({'reason': verdict.reason}))['reason'] == 'ok'

    async def test_a_discovery_miss_still_reports_no_citation(self) -> None:
        verdict = await validate_landing_evidence(
            _git_ops(citation=None, is_ancestor_map={}, effect_present=True),
            '42', 'task/42', branch_tip_sha='f' * 40,
        )
        assert verdict.reason == 'no_citation'
        assert verdict.reason is LandingReason.no_citation
        assert verdict.reason != LandingReason.no_attribution, (
            'the Contract rename is REGISTERED here, not yet EMITTED — '
            'epsilon flips it when it repoints the consumers'
        )

    async def test_a_survival_reject_still_reports_effect_absent(self) -> None:
        branch, citation = 'task/42', 'a' * 40
        verdict = await validate_landing_evidence(
            _git_ops(
                citation=citation,
                is_ancestor_map={(citation, branch): False},
                effect_present=False,
            ),
            '42', branch, branch_tip_sha=None,
        )
        assert verdict.reason == 'effect_absent'
        assert verdict.reason is LandingReason.effect_absent

    @pytest.mark.parametrize(
        ('delivered_checks', 'expected'),
        [
            (None, 'unwired'),
            ([], 'none_declared'),
            ([{'name': 'x', 'kind': 'grep', 'pattern': 'y', 'files': ['z']}], 'evaluated'),
        ],
    )
    async def test_delivered_checks_state_is_seeded_exactly_as_before(
        self, delivered_checks, expected,
    ) -> None:
        """Recorded UNCONDITIONALLY, on accepts too.

        Wiring is a property of the CALL SITE, not of the outcome: if only
        rejections carried it, the sole way to learn a site is unwired would
        be to wait for it to reject.  Task 4500 reads exactly this key to tell
        a genuinely-unwired site from a site that declared nothing.
        """
        sha = 'a' * 40
        verdict = await validate_landing_evidence(
            _git_ops(citation=None, is_ancestor_map={}, effect_present=True),
            '42', 'task/42', branch_tip_sha=None, candidate_sha=sha,
            delivered_checks=delivered_checks,
        )
        assert verdict.accepted is True, 'seeded on the ACCEPT path too'
        assert verdict.probe['delivered_checks_state'] == expected

    async def test_the_probe_keys_are_unchanged(self) -> None:
        """Every incumbent key still present, spelled and valued as before."""
        branch, tip, citation = 'task/42', 'f' * 40, 'a' * 40
        verdict = await validate_landing_evidence(
            _git_ops(
                citation=citation,
                is_ancestor_map={(citation, branch): True},
                effect_present=True,
            ),
            '42', branch, branch_tip_sha=tip,
        )
        probe = verdict.probe
        assert probe['task_id'] == '42'
        assert probe['branch'] == branch
        assert probe['branch_tip_sha'] == tip
        assert probe['citation'] == citation
        assert probe['effect_check_sha'] == tip
        assert probe['delivered_checks_state'] == 'unwired'
        assert probe['reason'] == 'ok'

    async def test_the_effect_absent_divergence_keys_survive(self) -> None:
        branch, citation = 'task/42', 'a' * 40
        verdict = await validate_landing_evidence(
            _git_ops(
                citation=citation,
                is_ancestor_map={(citation, branch): False},
                effect_present=False,
                effect_probe=CommitEffectProbe(
                    present=False, diverged_paths=('pkg/a.py',),
                    failure=None, anchor_sha=citation,
                ),
            ),
            '42', branch, branch_tip_sha=None,
        )
        assert verdict.probe['diverged_paths'] == ['pkg/a.py']
        assert verdict.probe['effect_failure'] is None
        assert verdict.probe['effect_anchor_sha'] == citation


@pytest.mark.asyncio
class TestEffectDivergenceGateSurvivesTheStrEnum:
    """(f) ``_render_effect_divergence`` gates on ``reason != 'effect_absent'``.

    That comparison is the ONLY literal reason comparison left in production
    code, and it now runs against a :class:`LandingReason` member rather than
    a plain string.  A StrEnum compares equal to its spelling, so it still
    holds — but "it should still hold" is exactly the class of assumption that
    silently stops holding, and this block's disappearance would be invisible:
    the escalation would still render, just without the one section that
    answers "is this a revert or just skew?".
    """

    async def test_the_divergence_block_renders_for_an_effect_absent_verdict(
        self,
    ) -> None:
        branch, citation = 'task/42', 'a' * 40
        verdict = await validate_landing_evidence(
            _git_ops(
                citation=citation,
                is_ancestor_map={(citation, branch): False},
                effect_present=False,
                effect_probe=CommitEffectProbe(
                    present=False, diverged_paths=('pkg/a.py',),
                    failure=None, anchor_sha=citation,
                ),
            ),
            '42', branch, branch_tip_sha=None,
        )
        _summary, detail = format_unattributed_landing_detail('42', branch, verdict)
        assert 'diverged paths' in detail
        assert 'pkg/a.py' in detail
        assert 'Unrecognized reason code' not in detail

    async def test_the_block_stays_absent_for_every_other_reason(self) -> None:
        verdict = await validate_landing_evidence(
            _git_ops(citation=None, is_ancestor_map={}, effect_present=True),
            '42', 'task/42', branch_tip_sha='f' * 40,
        )
        _summary, detail = format_unattributed_landing_detail('42', 'task/42', verdict)
        assert 'diverged paths' not in detail
        assert 'Unrecognized reason code' not in detail


class TestFileUnattributedLandingEscalationStampsCitationSha:
    """The filed record carries the evidence identity it could not attribute (task 4499).

    ``citation_sha`` is written at FILING time so that, once the record is
    resolved, the archived resolution still answers "which evidence was this?".
    That read-after-resolve is the identity half of the
    ``(task_id, category, citation_sha)`` triple that suppresses an identical
    refile — without the stamp there is nothing to match on, and the
    close-then-refile ping-pong cannot be closed.

    Driven against a REAL ``EscalationQueue`` and asserted on the PERSISTED
    record, not the in-memory object: an unserialised field would look correct
    here and match nothing in production.
    """

    def _queue(self, tmp_path):
        from escalation.queue import EscalationQueue  # noqa: PLC0415

        return EscalationQueue(tmp_path / 'queue')

    def _only_pending(self, queue, task_id: str = '42'):
        pending = queue.get_by_task(task_id, status='pending')
        assert len(pending) == 1, f'expected exactly one pending record; got {[e.id for e in pending]}'
        return pending[0]

    def test_discovery_mode_citation_is_stamped(self, tmp_path) -> None:
        """(1) A DISCOVERY-mode reject stamps the discovered citation sha."""
        queue = self._queue(tmp_path)

        file_unattributed_landing_escalation(
            queue, '42', 'task/42', _verdict('effect_absent'),
            agent_role='harness-reconcile',
        )

        assert self._only_pending(queue).citation_sha == 'a' * 40, (
            'the discovered citation must survive onto the persisted record'
        )

    def test_candidate_mode_citation_is_stamped(self, tmp_path) -> None:
        """(2) CANDIDATE mode — citation == effect_check_sha == the candidate sha."""
        queue = self._queue(tmp_path)
        candidate = 'd' * 40

        file_unattributed_landing_escalation(
            queue, '42', 'task/42',
            _verdict('effect_absent', citation=candidate, effect_check_sha=candidate),
            agent_role='orchestrator-merge-worker',
        )

        assert self._only_pending(queue).citation_sha == candidate

    def test_no_citation_verdict_stamps_none(self, tmp_path) -> None:
        """(3) A no_citation reject has no evidence identity — and must never gain one.

        ``_reject('no_citation')`` returns BEFORE the citation is assigned, so
        this arm is None by construction and can never be suppressed.
        """
        queue = self._queue(tmp_path)

        file_unattributed_landing_escalation(
            queue, '42', 'task/42',
            _verdict('no_citation', citation=None, effect_check_sha=None),
            agent_role='harness-reconcile',
        )

        assert self._only_pending(queue).citation_sha is None, (
            'a no_citation reject must carry no evidence identity'
        )

    def test_the_rest_of_the_record_is_unchanged(self, tmp_path) -> None:
        """(4) The stamp is additive — every other field files exactly as before."""
        queue = self._queue(tmp_path)

        file_unattributed_landing_escalation(
            queue, '42', 'task/42', _verdict('effect_absent'),
            agent_role='orchestrator-merge-worker',
        )

        esc = self._only_pending(queue)
        assert esc.level == 1
        assert esc.severity == 'blocking'
        assert esc.category == 'provenance_unattributed'
        assert esc.suggested_action == 'investigate_unattributed_landing_evidence'
        assert esc.agent_role == 'orchestrator-merge-worker'


class TestResolvedCitationSuppressesIdenticalRefile:
    """THE LOOP PIN — close-then-refile is closed, but new evidence still gets through.

    The defect (task 4499): ``provenance_unattributed``'s reject condition is
    ABSORBING — main only moves forward, so evidence that stopped surviving
    stays gone.  The only dedup guard reads PENDING records, so the moment the
    auto-watcher resolves the L1 the guard goes False and the next tick refiles
    the identical finding.  Ping-pong, forever, one fresh sequence number per
    round.

    BOTH HALVES ARE PINNED HERE, deliberately.  A test covering only the
    suppression half would pass while the guard silently swallowed genuine new
    evidence — trading an escalation storm for an escalation blackout, which is
    strictly the worse failure.  Every escape route (a different sha, no sha, a
    different task, a different category) is asserted alongside.

    Driven end-to-end against a REAL ``EscalationQueue``, exactly as the
    revalidation reproduction was.
    """

    TASK = '42'
    SHA_A = 'b' * 40
    SHA_B = 'c' * 40

    def _queue(self, tmp_path):
        from escalation.queue import EscalationQueue  # noqa: PLC0415

        return EscalationQueue(tmp_path / 'queue')

    def _file(
        self,
        queue,
        sha: str | None,
        *,
        task_id: str | None = None,
        reason: str = 'effect_absent',
        effect_check_sha: str | None = _UNSET,
    ):
        """File one L1 for *sha*.

        *effect_check_sha* defaults to the ``_UNSET`` sentinel — not to
        ``None`` — so it mirrors the citation unless a caller states otherwise,
        while still letting a caller pass an explicit ``None``.
        """
        task = self.TASK if task_id is None else task_id
        anchor = sha if effect_check_sha is _UNSET else effect_check_sha
        file_unattributed_landing_escalation(
            queue, task, f'task/{task}',
            _verdict(reason, citation=sha, effect_check_sha=anchor),
            agent_role='harness-reconcile',
        )

    def _pending(self, queue, task_id: str | None = None):
        return queue.get_by_task(self.TASK if task_id is None else task_id, status='pending')

    def _arrange_resolved(self, tmp_path, sha: str | None = None):
        """File one L1 on *sha*, resolve it, and prove the PENDING guard is off."""
        queue = self._queue(tmp_path)
        self._file(queue, self.SHA_A if sha is None else sha)
        pending = self._pending(queue)
        assert [e.id for e in pending] == [f'esc-{self.TASK}-1'], (
            f'Pre-condition: expected one filed L1; got {[e.id for e in pending]}'
        )
        queue.resolve(
            pending[0].id, 'confirmed benign', resolved_by='escalation-watcher-auto',
        )
        # The whole premise: with the L1 closed, the PENDING guard (including
        # task 3116's category scoping) is genuinely off — so anything observed
        # below is the NEW terminal-record guard, not the old one.
        assert queue.has_open_l1(self.TASK, category='provenance_unattributed') is False, (
            'Pre-condition: the pending guard must be OFF, or this test proves nothing'
        )
        return queue, pending[0].id

    def _seq(self, queue) -> str:
        from escalation.queue import SEQ_COUNTER_SUFFIX  # noqa: PLC0415

        return (queue.queue_dir / f'esc-{self.TASK}{SEQ_COUNTER_SUFFIX}').read_text().strip()

    def test_identical_refile_after_resolution_is_suppressed(self, tmp_path) -> None:
        """(1) SUPPRESSION HALF — the same evidence never files a second time."""
        queue, first_id = self._arrange_resolved(tmp_path)

        self._file(queue, self.SHA_A)

        assert self._pending(queue) == [], (
            f'PING-PONG: an identical refile produced {[e.id for e in self._pending(queue)]}'
        )

    def test_suppressed_refile_mints_no_id_at_all(self, tmp_path) -> None:
        """(1b) Suppression happens BEFORE make_id — no record, no sequence burned.

        A suppression that still minted an id would leak the per-task counter
        one number per tick, which is the same unbounded growth in a quieter
        costume.
        """
        queue, first_id = self._arrange_resolved(tmp_path)
        assert self._seq(queue) == '1', f'Pre-condition: seq should be 1; got {self._seq(queue)}'

        self._file(queue, self.SHA_A)

        assert {e.id for e in queue.get_by_task(self.TASK)} == {first_id}, (
            'a suppressed refile must leave the full record set untouched'
        )
        assert self._seq(queue) == '1', (
            f'a suppressed refile burned a sequence number: seq is now {self._seq(queue)}'
        )

    def test_a_different_citation_sha_still_files(self, tmp_path) -> None:
        """(2) GENUINE-NEW-EVIDENCE HALF — the guard must not become a blackout."""
        queue, _ = self._arrange_resolved(tmp_path)

        self._file(queue, self.SHA_B)

        pending = self._pending(queue)
        assert len(pending) == 1, (
            f'new evidence was swallowed; pending = {[e.id for e in pending]}'
        )
        assert pending[0].citation_sha == self.SHA_B, (
            f'the new record carries the wrong identity: {pending[0].citation_sha!r}'
        )

    def test_suppression_accumulates_per_sha_rather_than_being_one_shot(self, tmp_path) -> None:
        """(3) Each adjudicated sha stays suppressed — suppression is per-evidence."""
        queue, _ = self._arrange_resolved(tmp_path)
        self._file(queue, self.SHA_B)
        second = self._pending(queue)[0]
        queue.resolve(second.id, 'also benign', resolved_by='escalation-watcher-auto')

        self._file(queue, self.SHA_B)
        self._file(queue, self.SHA_A)

        assert self._pending(queue) == [], (
            'suppression is one-shot — a previously adjudicated sha refiled: '
            f'{[e.id for e in self._pending(queue)]}'
        )

    def test_no_citation_verdict_always_escapes(self, tmp_path) -> None:
        """(4) A falsy identity is not an identity — a no_citation reject always files."""
        queue = self._queue(tmp_path)
        self._file(queue, None, reason='no_citation')
        first = self._pending(queue)[0]
        assert first.citation_sha is None
        queue.resolve(first.id, 'confirmed benign', resolved_by='escalation-watcher-auto')

        self._file(queue, None, reason='no_citation')

        assert len(self._pending(queue)) == 1, (
            'a no_citation reject was suppressed against a citation-less resolution'
        )

    def test_same_citation_under_a_different_effect_anchor_is_still_suppressed(
        self, tmp_path,
    ) -> None:
        """(3b) THE ACCEPTED TRADE-OFF — the citation is the identity, not the anchor.

        In DISCOVERY mode the survival check does not always run against the
        citation: for an in-branch work commit ``validate_landing_evidence``
        anchors on ``branch_tip_sha``, so ``probe['effect_check_sha']`` is the
        sha actually measured.  A refile carrying the SAME citation under a
        DIFFERENT anchor is therefore a genuinely different measurement, and it
        is suppressed anyway.  That is intended, not an oversight: the citation
        moves only when the task lands again, while the branch tip moves on
        every commit added to the branch, so keying on the anchor would re-open
        the ping-pong at each tip advance — the storm this guard exists to
        close.  Pinned so the trade-off is a decision on the record rather than
        something a future reader silently "fixes"; the reasoning, and the
        bound on the residue, live in ``file_unattributed_landing_escalation``'s
        docstring.
        """
        queue, first_id = self._arrange_resolved(tmp_path)

        # Same citation, but the branch tip advanced between ticks.
        self._file(queue, self.SHA_A, effect_check_sha='d' * 40)

        assert self._pending(queue) == [], (
            'suppression became anchor-sensitive: a same-citation refile under a '
            'moved branch tip filed again, which re-opens the ping-pong on every '
            f'tip advance — pending = {[e.id for e in self._pending(queue)]}'
        )
        absorbed = queue.get(first_id)
        assert absorbed is not None and absorbed.refiles_suppressed == 1, (
            'the suppression must still be counted on the record that absorbed it'
        )

    def test_another_tasks_resolution_never_suppresses(self, tmp_path) -> None:
        """(5a) CROSS-TASK ESCAPE — the same sha on a different task is a different finding."""
        queue, _ = self._arrange_resolved(tmp_path)

        self._file(queue, self.SHA_A, task_id='99')

        assert len(self._pending(queue, task_id='99')) == 1, (
            "task 42's resolution suppressed a filing for task 99"
        )

    def test_another_categorys_resolution_never_suppresses(self, tmp_path) -> None:
        """(5b) CROSS-CATEGORY ESCAPE — a different root cause must get through.

        Mirrors task 3116's ruling on the pending guard: a new root cause must
        never hide behind an unrelated adjudication.
        """
        from escalation.models import Escalation  # noqa: PLC0415

        queue = self._queue(tmp_path)
        unrelated = Escalation(
            id=queue.make_id(self.TASK), task_id=self.TASK, agent_role='orchestrator',
            severity='blocking', category='task_failure', summary='something else failed',
            level=1, citation_sha=self.SHA_A,
        )
        queue.submit(unrelated)
        queue.resolve(unrelated.id, 'unrelated, handled', resolved_by='steward')

        self._file(queue, self.SHA_A)

        assert len(self._pending(queue)) == 1, (
            'a resolved task_failure suppressed a provenance_unattributed filing'
        )


class TestSuppressedRefileIsCounted:
    """The suppression must be a DURABLE structured fact, not log-only (task 4499).

    INV-2 ``structured-facts-at-failure`` and INV-4 ``storm-escape-required``:
    every suppression path carries a counter, and the fact lands on the record
    rather than only in a log line.  Without it, "this adjudication has quietly
    absorbed 900 refiles" is invisible from the queue — the storm is suppressed
    AND unobservable, which is how a suppression turns into a blindfold.
    """

    TASK = '42'
    SHA_A = 'b' * 40
    SHA_B = 'c' * 40

    def _queue(self, tmp_path):
        from escalation.queue import EscalationQueue  # noqa: PLC0415

        return EscalationQueue(tmp_path / 'queue')

    def _file(self, queue, sha: str):
        file_unattributed_landing_escalation(
            queue, self.TASK, f'task/{self.TASK}',
            _verdict('effect_absent', citation=sha, effect_check_sha=sha),
            agent_role='harness-reconcile',
        )

    def _file_and_resolve(self, queue, sha: str) -> str:
        self._file(queue, sha)
        pending = queue.get_by_task(self.TASK, status='pending')
        assert len(pending) == 1, f'expected one filing; got {[e.id for e in pending]}'
        queue.resolve(pending[0].id, 'confirmed benign', resolved_by='escalation-watcher-auto')
        return pending[0].id

    def test_three_suppressed_refiles_are_counted_on_the_prior_record(self, tmp_path) -> None:
        """(1) The storm is countable from the record the refiles were absorbed by."""
        queue = self._queue(tmp_path)
        esc_id = self._file_and_resolve(queue, self.SHA_A)

        for _ in range(3):
            self._file(queue, self.SHA_A)

        record = queue.get(esc_id)
        assert record is not None
        assert record.refiles_suppressed == 3, (
            'the suppression is LOG-ONLY — the record shows '
            f'{record.refiles_suppressed!r} absorbed refiles, expected 3'
        )

    def test_the_prior_record_is_otherwise_untouched(self, tmp_path) -> None:
        """(2) Only the counter moves; the adjudication itself is immutable."""
        queue = self._queue(tmp_path)
        esc_id = self._file_and_resolve(queue, self.SHA_A)
        before = queue.get(esc_id)
        assert before is not None

        self._file(queue, self.SHA_A)

        after = queue.get(esc_id)
        assert after is not None
        assert after.status == 'resolved', f'status changed: {after.status!r}'
        assert after.resolved_by == 'escalation-watcher-auto', (
            f'resolved_by changed: {after.resolved_by!r}'
        )
        assert after.resolution == before.resolution
        assert after.resolved_at == before.resolved_at
        assert after.citation_sha == self.SHA_A, (
            f'the identity being matched on drifted: {after.citation_sha!r}'
        )

    def test_the_record_stays_archived(self, tmp_path) -> None:
        """(3) Counting must not resurrect the resolution into the queue root.

        A resurrected record would read as a fresh pending escalation to every
        consumer that scans the root — turning the storm counter into a storm.
        """
        queue = self._queue(tmp_path)
        esc_id = self._file_and_resolve(queue, self.SHA_A)

        self._file(queue, self.SHA_A)

        assert not (queue.queue_dir / f'{esc_id}.json').exists(), (
            'RESURRECTION: the counted record was written back into the queue root'
        )
        assert queue.get_by_task(self.TASK, status='pending') == [], (
            'counting a suppressed refile produced a pending record'
        )

    def test_the_counter_is_per_record(self, tmp_path) -> None:
        """(4) Each adjudicated sha counts its OWN absorbed refiles.

        A shared counter would make two independent findings look like one
        storm, and would misattribute which evidence is actually recurring.
        """
        queue = self._queue(tmp_path)
        first = self._file_and_resolve(queue, self.SHA_A)
        second = self._file_and_resolve(queue, self.SHA_B)

        self._file(queue, self.SHA_A)
        self._file(queue, self.SHA_A)

        first_record, second_record = queue.get(first), queue.get(second)
        assert first_record is not None and second_record is not None
        assert first_record.refiles_suppressed == 2, (
            f'expected 2 on the recurring record; got {first_record.refiles_suppressed!r}'
        )
        assert second_record.refiles_suppressed == 0, (
            'the counter leaked across records; the quiet one reads '
            f'{second_record.refiles_suppressed!r}'
        )


class TestSuppressionFailsOpen:
    """The new guard must FAIL OPEN — never swallow an escalation (task 4499).

    Suppression is the ONLY outcome of this filer that LOSES an escalation, so
    the failure directions are not symmetric.  A guard that errs toward FILING
    costs a duplicate record someone closes; a guard that errs toward
    SUPPRESSING costs a provenance defect nobody ever sees.  Every uncertain
    path must therefore resolve toward filing — the same policy
    ``find_dedupe_parent``'s falsy-key short-circuit and
    ``gate_backlog_fingerprint_key``'s fail-toward-duplicates rule encode.

    Note the deliberate CONTRAST with ``has_open_l1``, pinned by
    ``test_raising_queue_is_contained`` above: a raising ``has_open_l1`` DOES
    drop the filing (it reaches the outer blanket except), because its failure
    direction is "file a duplicate on the next tick".  A raising
    ``find_terminal_by_citation`` must not, because its failure direction would
    be "lose this escalation now".
    """

    TASK = '42'
    SHA = 'b' * 40

    def _queue(self, tmp_path):
        from escalation.queue import EscalationQueue  # noqa: PLC0415

        return EscalationQueue(tmp_path / 'queue')

    def _file(self, queue, *, reason: str = 'effect_absent'):
        file_unattributed_landing_escalation(
            queue, self.TASK, f'task/{self.TASK}',
            _verdict(reason, citation=self.SHA, effect_check_sha=self.SHA),
            agent_role='harness-reconcile',
        )

    def test_raising_lookup_still_files_and_warns(self, tmp_path, caplog) -> None:
        """(1) A lookup that explodes must not reach the outer blanket except.

        Patched on a REAL queue so everything downstream of the lookup stays
        honest — the record really has to be minted, written and re-readable.
        """
        queue = self._queue(tmp_path)

        with patch.object(
            queue, 'find_terminal_by_citation', side_effect=RuntimeError('index corrupt'),
        ), caplog.at_level('WARNING'):
            self._file(queue)

        assert len(queue.get_by_task(self.TASK, status='pending')) == 1, (
            'a failing suppression lookup DROPPED the escalation — the one '
            'failure direction that loses a provenance defect'
        )
        assert any('index corrupt' in r.message or r.exc_info for r in caplog.records), (
            'the lookup failure must be loud, not silently swallowed'
        )

    def test_queue_without_the_method_still_files(self, tmp_path) -> None:
        """(2) A duck-typed / older stand-in lacking the method files as before."""
        legacy = MagicMock(spec=['has_open_l1', 'make_id', 'submit'])
        legacy.has_open_l1.return_value = False
        legacy.make_id.return_value = f'esc-{self.TASK}-1'
        assert not hasattr(legacy, 'find_terminal_by_citation'), (
            'Pre-condition: the stand-in must genuinely lack the attribute'
        )

        self._file(legacy)

        legacy.submit.assert_called_once()

    def test_a_dismissed_prior_suppresses_end_to_end(self, tmp_path) -> None:
        """(3) A dismissal adjudicates the evidence just as a resolution does.

        Pinned at the queue level in the find_terminal_by_citation tests; this
        pins the WIRING — that the filer actually honours a dismissal.
        """
        queue = self._queue(tmp_path)
        self._file(queue)
        first = queue.get_by_task(self.TASK, status='pending')[0]
        queue.resolve(first.id, 'not a real defect', dismiss=True, resolved_by='steward')

        self._file(queue)

        assert queue.get_by_task(self.TASK, status='pending') == [], (
            'a DISMISSED adjudication failed to suppress an identical refile'
        )

    def test_a_raising_counter_still_suppresses(self, tmp_path, caplog) -> None:
        """(4) A bookkeeping failure must not re-open the storm.

        The counter is observability, not the decision.  Letting its failure
        propagate would file the escalation the guard just decided to suppress
        — turning a metrics fault into the storm it exists to stop.
        """
        queue = self._queue(tmp_path)
        self._file(queue)
        first = queue.get_by_task(self.TASK, status='pending')[0]
        queue.resolve(first.id, 'confirmed benign', resolved_by='escalation-watcher-auto')

        with patch.object(
            queue, 'note_suppressed_refile', side_effect=RuntimeError('write failed'),
        ), caplog.at_level('WARNING'):
            self._file(queue)

        assert queue.get_by_task(self.TASK, status='pending') == [], (
            'a failing counter re-opened the refile storm'
        )
        assert any(r.exc_info for r in caplog.records), (
            'the counter failure must be loud, not silently swallowed'
        )

    def test_none_queue_is_still_a_silent_noop(self, tmp_path) -> None:
        """(5a) Pre-existing contract: a bare-harness caller passes None."""
        self._file(None)

    def test_raising_has_open_l1_is_still_contained_without_filing(self, tmp_path) -> None:
        """(5b) Pre-existing contract, unchanged: has_open_l1 keeps the OLD direction.

        Its failure means "file a duplicate on the next tick", which is
        recoverable — so it stays contained by the outer blanket except and
        this step must not change it.
        """
        queue = MagicMock()
        queue.has_open_l1.side_effect = RuntimeError('queue exploded')

        self._file(queue)

        queue.submit.assert_not_called()
