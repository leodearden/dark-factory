"""Tests for the harness's already-landed pre-dispatch gate (task 2313).

Architecturally parallel to task 2156's landed-outbox dispatch gate
(test_harness_landed_dispatch_gate_wiring.py), but this gate consults LIVE
GIT STATE (ancestry + content-equivalence) rather than the durable
LandedOutbox, so it also catches OUT-OF-BAND landings that never went
through this orchestrator's own merge queue: a sibling direct-merge, a prior
orchestrator run, or a squash/rebase/manual landing.

Covers:
  step-3  (RED)  Ancestry happy-path: is_ancestor True + citation present +
                 not degenerate -> flips to done, returns True.
  step-5  (RED)  Ancestry-path false-positive guards: open-L1 veto,
                 degenerate-branch veto, missing-citation veto.
  step-7  (RED)  Branch-deleted merge-marker path: marker found (and not
                 stale) -> flips to done; stale marker (ancestor of
                 branch_base_sha) -> vetoes the flip.

Mirrors test_harness_landed_dispatch_gate_wiring.py's ``_build_harness``
bare-harness construction helper exactly.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.config import DeliveredChecksConfig
from orchestrator.delivered_checks import DeliveredChecksBlock
from orchestrator.harness import Harness
from orchestrator.landing_evidence import LandingEvidenceVerdict

_HARNESS_SRC_PATH = Path(__file__).parent.parent / 'src' / 'orchestrator' / 'harness.py'


def _build_harness(mock_orch_config) -> Harness:
    """Construct a Harness with heavy constructors patched out.

    Mirrors test_harness_landed_dispatch_gate_wiring.py's ``_build_harness``.
    """
    mock_orch_config.max_concurrent_tasks = 2
    mock_orch_config.fused_memory.project_id = 'test'

    with patch('orchestrator.harness.McpLifecycle'), \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        return Harness(mock_orch_config)


@pytest.mark.asyncio
class TestAlreadyLandedDispatchGateAncestryHappyPath:
    """Ancestry happy-path: is_ancestor True + citation present + not degenerate."""

    async def test_ancestry_with_citation_flips_to_done(
        self, mock_orch_config,
    ) -> None:
        """RED until step-4 adds Harness._already_landed_dispatch_gate.

        is_ancestor True, a citation commit is present on main, and the
        branch is not degenerate -> the gate drives the task to done via
        _mark_in_progress_done (anchored on the citation sha) and returns
        True so the scheduler withholds dispatch this tick.
        """
        h = _build_harness(mock_orch_config)

        citation_sha = 'a' * 40
        h.git_ops = MagicMock()
        h.git_ops.resolve_branch_sha = AsyncMock(return_value='f' * 40)
        h.git_ops.is_ancestor = AsyncMock(return_value=True)
        h.git_ops.find_task_citation_commit = AsyncMock(return_value=citation_sha)
        h.git_ops.commit_effect_present_in_main = AsyncMock(return_value=True)
        h.git_ops.config.branch_prefix = 'task/'
        h.git_ops.config.main_branch = 'main'

        h.scheduler.get_task = AsyncMock(
            return_value={'id': '42', 'metadata': {}},
        )
        h._branch_is_degenerate = AsyncMock(return_value=False)
        h._mark_in_progress_done = AsyncMock()
        h._escalation_queue = None

        result = await h._already_landed_dispatch_gate('42')

        assert result is True
        cast(AsyncMock, h._mark_in_progress_done).assert_awaited_once()
        call_args = cast(AsyncMock, h._mark_in_progress_done).await_args
        assert call_args is not None
        assert call_args.args[0] == '42'
        assert call_args.args[1] == citation_sha
        assert call_args.args[3] == 'dispatch-gate-already-on-main'


def _wired_ancestry_harness(mock_orch_config) -> Harness:
    """Bare harness pre-wired so the ancestry path would flip on its own —
    each guard test overrides exactly one attribute to trip its veto.
    """
    h = _build_harness(mock_orch_config)
    h.git_ops = MagicMock()
    h.git_ops.resolve_branch_sha = AsyncMock(return_value='f' * 40)
    h.git_ops.is_ancestor = AsyncMock(return_value=True)
    h.git_ops.find_task_citation_commit = AsyncMock(return_value='a' * 40)
    h.git_ops.commit_effect_present_in_main = AsyncMock(return_value=True)
    h.git_ops.config.branch_prefix = 'task/'
    h.git_ops.config.main_branch = 'main'

    h.scheduler.get_task = AsyncMock(return_value={'id': '42', 'metadata': {}})
    h._branch_is_degenerate = AsyncMock(return_value=False)
    h._mark_in_progress_done = AsyncMock()
    h._escalation_queue = None
    return h


@pytest.mark.asyncio
class TestAlreadyLandedDispatchGateAncestryGuards:
    """Ancestry-path false-positive guards must veto the flip (RED until step-6).

    Each sub-case starts from an otherwise-flipping ancestry setup
    (is_ancestor True, citation present, not degenerate, no open L1) and
    trips exactly one guard — the gate must return False and must NOT
    call _mark_in_progress_done.
    """

    async def test_open_l1_vetoes_flip(self, mock_orch_config) -> None:
        """An open L1 escalation is a deliberate human handoff — never
        second-guessed, even though is_ancestor is True.
        """
        h = _wired_ancestry_harness(mock_orch_config)
        h._escalation_queue = MagicMock()
        h._escalation_queue.has_open_l1 = MagicMock(return_value=True)

        result = await h._already_landed_dispatch_gate('42')

        assert result is False
        cast(AsyncMock, h._mark_in_progress_done).assert_not_awaited()

    async def test_degenerate_branch_vetoes_flip(self, mock_orch_config) -> None:
        """A degenerate branch (tip == branch_base_sha) carries zero task
        work — is_ancestor==True is a false 'already on main' signal.
        """
        h = _wired_ancestry_harness(mock_orch_config)
        h._branch_is_degenerate = AsyncMock(return_value=True)

        result = await h._already_landed_dispatch_gate('42')

        assert result is False
        cast(AsyncMock, h._mark_in_progress_done).assert_not_awaited()

    async def test_missing_citation_vetoes_flip(self, mock_orch_config) -> None:
        """No commit on main cites this task — reject the zero-commit-branch
        shape where is_ancestor returns True trivially but no work landed.
        """
        h = _wired_ancestry_harness(mock_orch_config)
        h.git_ops.find_task_citation_commit = AsyncMock(return_value=None)

        result = await h._already_landed_dispatch_gate('42')

        assert result is False
        cast(AsyncMock, h._mark_in_progress_done).assert_not_awaited()

    async def test_divergent_citation_flips_to_done(
        self, mock_orch_config,
    ) -> None:
        """FIX-A (task 2870 / esc-5252-9): the relaxed citation-lineage guard
        now ACCEPTS a genuine on-main citation whose branch ref has
        diverged/realigned. is_ancestor(branch, 'main') is True (the ancestry
        evidence is real) and is_ancestor(citation, branch) is False (the
        citation is not an in-branch work commit — the divergent shape), and
        the bidirectional (branch, citation) reject arm is gone (an unmapped
        call would raise). The effect is present, so the gate flips the task
        to done anchored on the CITATION (the divergent branch tip is not
        authoritative), rather than leaving it PENDING to close↔refile
        ping-pong every reconcile tick.
        """
        h = _wired_ancestry_harness(mock_orch_config)
        branch = 'task/42'
        citation_sha = 'a' * 40  # matches _wired_ancestry_harness's citation default

        async def _is_ancestor(a, b):
            if (a, b) == (branch, 'main'):
                return True
            if (a, b) == (citation_sha, branch):
                return False  # citation NOT a work commit on this branch (divergent)
            raise AssertionError(f'unexpected is_ancestor call: {a!r}, {b!r}')
        h.git_ops.is_ancestor = AsyncMock(side_effect=_is_ancestor)

        result = await h._already_landed_dispatch_gate('42')

        assert result is True
        cast(AsyncMock, h._mark_in_progress_done).assert_awaited_once()
        call_args = cast(AsyncMock, h._mark_in_progress_done).await_args
        assert call_args is not None
        assert call_args.args[0] == '42'
        assert call_args.args[1] == citation_sha
        assert call_args.args[3] == 'dispatch-gate-already-on-main'

    async def test_no_ff_merge_commit_citation_flips_to_done(
        self, mock_orch_config,
    ) -> None:
        """FIX 2 (task 2500) citation-lineage guard must ACCEPT this branch's
        own no-ff merge commit, not just work commits on the branch.

        Regression for esc-2500-2: git_ops.DEFAULT_COMMIT_CITATION_PATTERN
        deliberately also matches the ``^Merge task/{tid} into`` no-ff merge
        subject, and find_task_citation_commit returns the MOST RECENT match
        on main — so for a legitimate no-ff landing whose branch ref still
        exists (a prior orchestrator run that merged but crashed before
        deleting the branch / marking done, or a manual
        ``git merge --no-ff task/42``) the citation IS the merge commit. A
        merge commit is a DESCENDANT of the branch tip, so
        is_ancestor(citation, branch) is False; the guard must fall back to
        is_ancestor(branch, citation) (True — the branch tip is a parent of
        its own merge commit) and still flip the task to done rather than
        re-dispatch it as duplicate work.
        """
        h = _wired_ancestry_harness(mock_orch_config)
        branch = 'task/42'
        merge_commit_sha = 'a' * 40  # citation default; here it's the merge commit

        async def _is_ancestor(a, b):
            if (a, b) == (branch, 'main'):
                return True
            if (a, b) == (merge_commit_sha, branch):
                return False  # merge commit is a DESCENDANT of the branch tip
            if (a, b) == (branch, merge_commit_sha):
                return True  # branch tip is a parent of its own merge commit
            raise AssertionError(f'unexpected is_ancestor call: {a!r}, {b!r}')
        h.git_ops.is_ancestor = AsyncMock(side_effect=_is_ancestor)

        result = await h._already_landed_dispatch_gate('42')

        assert result is True
        cast(AsyncMock, h._mark_in_progress_done).assert_awaited_once()
        call_args = cast(AsyncMock, h._mark_in_progress_done).await_args
        assert call_args is not None
        assert call_args.args[0] == '42'
        assert call_args.args[1] == merge_commit_sha
        assert call_args.args[3] == 'dispatch-gate-already-on-main'

    async def test_reverted_citation_effect_vetoes_flip(
        self, mock_orch_config,
    ) -> None:
        """FIX 1 (task 2500) effect-present guard: the citation is present
        AND in-lineage (is_ancestor(citation, branch) True, from
        _wired_ancestry_harness's blanket is_ancestor=True default) but its
        effect was reverted at current main HEAD
        (commit_effect_present_in_main returns False) — a later commit on
        main undid the citation's changes, so the citation's ancestry is
        real but stale. The gate must reject it, not flip.

        RED: the ancestor path has no effect-present check yet, so it
        would flip regardless of commit_effect_present_in_main.
        """
        h = _wired_ancestry_harness(mock_orch_config)
        h.git_ops.commit_effect_present_in_main = AsyncMock(return_value=False)

        result = await h._already_landed_dispatch_gate('42')

        assert result is False
        cast(AsyncMock, h._mark_in_progress_done).assert_not_awaited()

    async def test_intermediate_work_commit_citation_checks_branch_tip_effect(
        self, mock_orch_config,
    ) -> None:
        """FIX 1 effect-present guard (task 2500 amendment, review finding):
        when the citation is an in-branch WORK commit
        (citation_on_branch True — shape (a)), it may be an INTERMEDIATE
        commit rather than the branch's final state. A LATER commit on
        this SAME branch can legitimately re-touch the citation's own
        touched paths again on the way to the branch's final content, so
        checking the citation's stale snapshot against main would
        false-reject a genuine multi-commit landing. The guard must
        anchor commit_effect_present_in_main on the branch TIP sha
        (resolve_branch_sha's return value) instead of the possibly-stale
        citation sha, and still flip to done — anchoring
        _mark_in_progress_done's provenance on the citation, as before.
        """
        h = _wired_ancestry_harness(mock_orch_config)
        branch_tip_sha = 'f' * 40  # matches _wired_ancestry_harness's resolve_branch_sha
        citation_sha = 'a' * 40  # matches _wired_ancestry_harness's citation default

        async def _effect_present(sha):
            # Only the branch tip reflects the landing's actual final
            # state at HEAD — the intermediate citation's own snapshot is
            # stale (a later on-branch commit re-touched its paths).
            return sha == branch_tip_sha
        h.git_ops.commit_effect_present_in_main = AsyncMock(side_effect=_effect_present)

        result = await h._already_landed_dispatch_gate('42')

        assert result is True
        cast(
            AsyncMock, h.git_ops.commit_effect_present_in_main,
        ).assert_awaited_once_with(branch_tip_sha)
        cast(AsyncMock, h._mark_in_progress_done).assert_awaited_once()
        call_args = cast(AsyncMock, h._mark_in_progress_done).await_args
        assert call_args is not None
        assert call_args.args[0] == '42'
        assert call_args.args[1] == citation_sha
        assert call_args.args[3] == 'dispatch-gate-already-on-main'


@pytest.mark.asyncio
class TestAlreadyLandedDispatchGateAncestryDelegatesToHelper:
    """The ancestry path DELEGATES to the shared validate_landing_evidence
    helper (task 2678) instead of inlining its own FIX2 lineage + FIX1'
    effect-present logic. RED until step-06.

    The open-L1 veto, the degenerate-branch veto, and the is_ancestor(branch,
    main) gate all still short-circuit BEFORE the helper is ever consulted —
    only pinned indirectly here by _wired_ancestry_harness's setup; the
    dedicated veto tests live in TestAlreadyLandedDispatchGateAncestryGuards
    and must remain green (unchanged) after step-06's refactor.
    """

    async def test_accepted_verdict_marks_done_and_returns_true(
        self, mock_orch_config,
    ) -> None:
        h = _wired_ancestry_harness(mock_orch_config)
        branch_tip_sha = 'f' * 40  # matches _wired_ancestry_harness's resolve_branch_sha
        verdict = LandingEvidenceVerdict(
            accepted=True, evidence_sha='a' * 40, reason='ok', probe={},
        )

        with patch(
            'orchestrator.harness.validate_landing_evidence',
            AsyncMock(return_value=verdict),
        ) as mock_validate:
            result = await h._already_landed_dispatch_gate('42')

        assert result is True
        mock_validate.assert_awaited_once()
        call = mock_validate.await_args
        assert call is not None
        assert call.args[0] is h.git_ops
        assert call.args[1] == '42'
        assert call.args[2] == 'task/42'
        assert call.kwargs['branch_tip_sha'] == branch_tip_sha

        cast(AsyncMock, h._mark_in_progress_done).assert_awaited_once()
        mark_call = cast(AsyncMock, h._mark_in_progress_done).await_args
        assert mark_call is not None
        assert mark_call.args[0] == '42'
        assert mark_call.args[1] == 'a' * 40
        assert mark_call.args[3] == 'dispatch-gate-already-on-main'

    async def test_rejected_verdict_returns_false_no_mark_done(
        self, mock_orch_config,
    ) -> None:
        h = _wired_ancestry_harness(mock_orch_config)
        verdict = LandingEvidenceVerdict(
            accepted=False, evidence_sha=None, reason='effect_absent', probe={},
        )

        with patch(
            'orchestrator.harness.validate_landing_evidence',
            AsyncMock(return_value=verdict),
        ):
            result = await h._already_landed_dispatch_gate('42')

        assert result is False
        cast(AsyncMock, h._mark_in_progress_done).assert_not_awaited()


def _wired_marker_harness(
    mock_orch_config, *, marker_sha, branch_base_sha, marker_is_ancestor_of_base,
) -> Harness:
    """Bare harness with is_ancestor(branch, main) False, so the ancestry
    path never engages and the marker path is what's under test.

    ``is_ancestor`` is mocked with a side_effect function because it is
    called with two DIFFERENT argument pairs in this path: once for
    ``(branch, main_branch)`` (must be False to reach the marker path) and
    once for ``(marker_sha, branch_base_sha)`` (the stale-marker check).

    ``commit_effect_present_in_main`` defaults to True (a healthy marker
    landing) and ``_escalation_queue`` is a wired MagicMock with
    ``has_open_l1`` defaulting False — callers flip either to exercise the
    CANDIDATE-mode reject-and-escalate path (task 2678).
    """
    h = _build_harness(mock_orch_config)
    h.git_ops = MagicMock()
    h.git_ops.config.branch_prefix = 'task/'
    h.git_ops.config.main_branch = 'main'

    # Branch ref does not exist (deleted post-merge) — this is what routes
    # the gate past the ancestry check and into the marker path at all.
    h.git_ops.resolve_branch_sha = AsyncMock(return_value=None)

    async def _is_ancestor(a, b):
        if a == marker_sha and b == branch_base_sha:
            return marker_is_ancestor_of_base
        raise AssertionError(f'unexpected is_ancestor call: {a!r}, {b!r}')

    h.git_ops.is_ancestor = AsyncMock(side_effect=_is_ancestor)
    h.git_ops.find_merge_marker = AsyncMock(return_value=marker_sha)
    h.git_ops.find_task_citation_commit = AsyncMock(return_value=None)
    h.git_ops.commit_effect_present_in_main = AsyncMock(return_value=True)

    h.scheduler.get_task = AsyncMock(
        return_value={'id': '42', 'metadata': {'branch_base_sha': branch_base_sha}},
    )
    h._branch_is_degenerate = AsyncMock(return_value=False)
    h._mark_in_progress_done = AsyncMock()
    h._escalation_queue = MagicMock()
    h._escalation_queue.has_open_l1 = MagicMock(return_value=False)
    h._escalation_queue.make_id = MagicMock(return_value='esc-42-1')
    return h


@pytest.mark.asyncio
class TestAlreadyLandedDispatchGateMarkerPath:
    """Branch-deleted merge-marker path (RED until step-8 for the
    effect-absent/escalation sub-case; the flip and stale-marker sub-cases
    are already green from step-8's predecessor and are extended here with
    escalation-queue assertions).

    resolve_branch_sha(branch) is None in all sub-cases (the branch ref is
    gone), so the ancestry path never engages — only the marker path can
    produce a result.
    """

    async def test_marker_found_and_not_stale_flips_to_done(
        self, mock_orch_config,
    ) -> None:
        """A merge marker on main, not an ancestor of branch_base_sha (i.e.
        it postdates this incarnation's creation point), with its effect
        PRESENT at main HEAD (the CANDIDATE-mode helper accepts) -> flips
        to done, anchored on the marker sha, with no escalation filed.

        Also pins the citation-laziness fix (task 2313 review): the marker
        path doesn't consume a citation, so find_task_citation_commit must
        never be called here.
        """
        marker_sha = 'b' * 40
        branch_base_sha = 'e' * 40
        h = _wired_marker_harness(
            mock_orch_config,
            marker_sha=marker_sha,
            branch_base_sha=branch_base_sha,
            marker_is_ancestor_of_base=False,
        )

        result = await h._already_landed_dispatch_gate('42')

        assert result is True
        cast(AsyncMock, h._mark_in_progress_done).assert_awaited_once()
        call_args = cast(AsyncMock, h._mark_in_progress_done).await_args
        assert call_args is not None
        assert call_args.args[0] == '42'
        assert call_args.args[1] == marker_sha
        assert call_args.args[3] == 'dispatch-gate-marker-found'
        cast(AsyncMock, h.git_ops.find_task_citation_commit).assert_not_called()
        cast(MagicMock, h._escalation_queue).submit.assert_not_called()

    async def test_marker_found_not_stale_effect_absent_escalates_no_mark_done(
        self, mock_orch_config,
    ) -> None:
        """The task-1175 shape: a merge marker on main, not stale, but its
        effect has been REVERTED at current main HEAD
        (commit_effect_present_in_main False) — a later commit on main
        undid exactly the paths the marker's landing touched. The
        CANDIDATE-mode FIX 1' guard (task 2678, closing the 1175 clobber in
        this path) must reject it: no _mark_in_progress_done call, the gate
        returns False, and exactly one dedup-guarded 'provenance_unattributed'
        L1 escalation is filed carrying the branch, the marker sha, and the
        'effect_absent' reason. has_open_l1('42') is consulted for dedup
        before the escalation is filed.

        RED: the marker path has no effect check / escalation yet, so it
        would flip to done regardless of commit_effect_present_in_main.
        """
        marker_sha = 'b' * 40
        branch_base_sha = 'e' * 40
        h = _wired_marker_harness(
            mock_orch_config,
            marker_sha=marker_sha,
            branch_base_sha=branch_base_sha,
            marker_is_ancestor_of_base=False,
        )
        h.git_ops.commit_effect_present_in_main = AsyncMock(return_value=False)

        result = await h._already_landed_dispatch_gate('42')

        assert result is False
        cast(AsyncMock, h._mark_in_progress_done).assert_not_awaited()

        # has_open_l1 is also consulted by the gate's pre-existing top-of-
        # method open-L1 veto (unrelated to this dedup check) — assert the
        # dedup call happened (most recent call), not an exact call count.
        cast(MagicMock, h._escalation_queue).has_open_l1.assert_called_with('42')
        cast(MagicMock, h._escalation_queue).submit.assert_called_once()
        esc = cast(MagicMock, h._escalation_queue).submit.call_args[0][0]
        assert esc.category == 'provenance_unattributed'
        assert esc.task_id == '42'
        assert 'task/42' in esc.detail
        assert marker_sha in esc.detail
        assert 'effect_absent' in esc.detail

    async def test_stale_marker_vetoes_flip(self, mock_orch_config) -> None:
        """A marker that IS an ancestor of branch_base_sha predates this
        incarnation (branch was deleted + re-created under the same task
        id) -> vetoes the flip, no _mark_in_progress_done call, and no
        escalation (this veto predates the incarnation entirely — it is
        not an unattributed-landing signal, so it must not be treated as
        one).
        """
        marker_sha = 'b' * 40
        branch_base_sha = 'e' * 40
        h = _wired_marker_harness(
            mock_orch_config,
            marker_sha=marker_sha,
            branch_base_sha=branch_base_sha,
            marker_is_ancestor_of_base=True,
        )

        result = await h._already_landed_dispatch_gate('42')

        assert result is False
        cast(AsyncMock, h._mark_in_progress_done).assert_not_awaited()
        cast(MagicMock, h._escalation_queue).submit.assert_not_called()


def _wired_content_harness(
    mock_orch_config, *, citation_sha, content_in_main, effect_present=True,
) -> Harness:
    """Bare harness with the branch existing but is_ancestor(branch, main)
    False and find_merge_marker None, so neither the ancestry path nor the
    marker path can produce a result — only the content-equivalence
    fallback is under test.

    The branch must "exist" (resolve_branch_sha truthy) for this fallback
    to even be reached — the cheap pre-filter (task 2313 review) routes a
    nonexistent branch straight to the marker path instead.

    ``is_ancestor`` is a side_effect function: the DISCOVERY-mode helper
    (task 2678) calls it once for ``(branch, main)`` to bypass the ancestry
    path (always False here) and, only once a citation is found, again for
    the FIX 2 citation-lineage guard ``(citation, branch)`` — modeled here
    as an in-branch work commit (True) so the interesting
    content-equivalence behavior (accept/reject on effect-present) isn't
    entangled with the lineage guard, which has its own dedicated coverage
    in TestAlreadyLandedDispatchGateAncestryGuards and
    test_landing_evidence.py. ``get_main_sha`` is wired but must NEVER be
    called — the silent ``citation or get_main_sha()`` fallback (task 2678)
    is deleted.
    """
    h = _build_harness(mock_orch_config)
    h.git_ops = MagicMock()
    h.git_ops.config.branch_prefix = 'task/'
    h.git_ops.config.main_branch = 'main'
    branch = 'task/42'

    async def _is_ancestor(a, b):
        if (a, b) == (branch, 'main'):
            return False  # bypass the ancestry path
        if citation_sha is not None and (a, b) == (citation_sha, branch):
            return True  # citation modeled as an in-branch work commit
        raise AssertionError(f'unexpected is_ancestor call: {a!r}, {b!r}')

    h.git_ops.resolve_branch_sha = AsyncMock(return_value='f' * 40)
    h.git_ops.is_ancestor = AsyncMock(side_effect=_is_ancestor)
    h.git_ops.find_merge_marker = AsyncMock(return_value=None)
    h.git_ops.find_task_citation_commit = AsyncMock(return_value=citation_sha)
    h.git_ops.branch_content_in_main = AsyncMock(return_value=content_in_main)
    h.git_ops.get_main_sha = AsyncMock(return_value='c' * 40)
    h.git_ops.commit_effect_present_in_main = AsyncMock(return_value=effect_present)

    h.scheduler.get_task = AsyncMock(return_value={'id': '42', 'metadata': {}})
    h._branch_is_degenerate = AsyncMock(return_value=False)
    h._mark_in_progress_done = AsyncMock()
    h._escalation_queue = MagicMock()
    h._escalation_queue.has_open_l1 = MagicMock(return_value=False)
    h._escalation_queue.make_id = MagicMock(return_value='esc-42-1')
    return h


@pytest.mark.asyncio
class TestAlreadyLandedDispatchGateContentEquivalence:
    """Content-equivalence fallback (RED until step-10 for the no-citation
    and effect-absent escalation sub-cases; the effect-present accept
    sub-case is extended here with lineage/effect wiring and a
    no-escalation assertion).

    is_ancestor(branch, main) is False and find_merge_marker returns None
    in every sub-case, so only the content-equivalence fallback can
    produce a result.
    """

    async def test_content_equivalent_no_citation_escalates_no_mark_done(
        self, mock_orch_config,
    ) -> None:
        """branch_content_in_main True, no citation on main -> there is no
        positive evidence to attribute a landing to (DISCOVERY mode
        rejects with reason 'no_citation'). The silent
        ``citation or get_main_sha()`` fallback (task 2678) is deleted: the
        gate must NOT fabricate an anchor from main HEAD, must NOT mark
        done, and must file exactly one 'provenance_unattributed'
        escalation carrying the branch. get_main_sha must never be called.
        """
        h = _wired_content_harness(
            mock_orch_config,
            citation_sha=None,
            content_in_main=True,
        )

        result = await h._already_landed_dispatch_gate('42')

        assert result is False
        cast(AsyncMock, h._mark_in_progress_done).assert_not_awaited()
        cast(AsyncMock, h.git_ops.get_main_sha).assert_not_called()

        cast(MagicMock, h._escalation_queue).submit.assert_called_once()
        esc = cast(MagicMock, h._escalation_queue).submit.call_args[0][0]
        assert esc.category == 'provenance_unattributed'
        assert esc.task_id == '42'
        assert 'task/42' in esc.detail
        assert 'no_citation' in esc.detail

    async def test_content_equivalent_with_citation_and_effect_present_marks_done(
        self, mock_orch_config,
    ) -> None:
        """branch_content_in_main True, a citation commit is present on
        main AND its effect is present at main HEAD -> the citation sha
        anchors the flip, not main HEAD; no escalation filed.
        """
        citation_sha = 'd' * 40
        h = _wired_content_harness(
            mock_orch_config,
            citation_sha=citation_sha,
            content_in_main=True,
            effect_present=True,
        )

        result = await h._already_landed_dispatch_gate('42')

        assert result is True
        cast(AsyncMock, h._mark_in_progress_done).assert_awaited_once()
        call_args = cast(AsyncMock, h._mark_in_progress_done).await_args
        assert call_args is not None
        assert call_args.args[0] == '42'
        assert call_args.args[1] == citation_sha
        assert call_args.args[3] == 'dispatch-gate-content-equivalent'
        cast(MagicMock, h._escalation_queue).submit.assert_not_called()

    async def test_content_equivalent_with_citation_and_effect_absent_escalates(
        self, mock_orch_config,
    ) -> None:
        """branch_content_in_main True, a citation is present on main, but
        its effect was reverted at current main HEAD
        (commit_effect_present_in_main False) -> reject (FIX 1'), no
        mark_done, and exactly one 'provenance_unattributed' escalation
        (reason 'effect_absent').
        """
        citation_sha = 'd' * 40
        h = _wired_content_harness(
            mock_orch_config,
            citation_sha=citation_sha,
            content_in_main=True,
            effect_present=False,
        )

        result = await h._already_landed_dispatch_gate('42')

        assert result is False
        cast(AsyncMock, h._mark_in_progress_done).assert_not_awaited()

        cast(MagicMock, h._escalation_queue).submit.assert_called_once()
        esc = cast(MagicMock, h._escalation_queue).submit.call_args[0][0]
        assert esc.category == 'provenance_unattributed'
        assert esc.task_id == '42'
        assert 'effect_absent' in esc.detail

    async def test_content_not_equivalent_dispatches_normally(
        self, mock_orch_config,
    ) -> None:
        """branch_content_in_main False -> no evidence at all; the gate
        returns False so the task dispatches normally this tick, with no
        escalation and no mark_done.
        """
        h = _wired_content_harness(
            mock_orch_config,
            citation_sha=None,
            content_in_main=False,
        )

        result = await h._already_landed_dispatch_gate('42')

        assert result is False
        cast(AsyncMock, h._mark_in_progress_done).assert_not_awaited()
        cast(MagicMock, h._escalation_queue).submit.assert_not_called()


class TestAlreadyLandedDispatchGateGetMainShaFallbackGrepGuard:
    """Source-level guard (task 2678): the silent
    ``citation or await self.git_ops.get_main_sha()`` fallback must be
    fully deleted from the content-equivalence path, not merely made
    unreachable — a future edit must not resurrect it as dead code.
    """

    def test_get_main_sha_fallback_expression_absent(self) -> None:
        content = _HARNESS_SRC_PATH.read_text()
        assert 'or await self.git_ops.get_main_sha()' not in content, (
            'harness.py still contains the silent get_main_sha() fallback '
            'expression; task 2678 replaces it with '
            'validate_landing_evidence + escalate-instead-of-stamp.'
        )


# ---------------------------------------------------------------------------
# task 2677 step-7/step-8 — a done_evidence_stale rejection surfacing from
# _mark_in_progress_done's scheduler.mark_done call must gate the dispatch
# (the contested task must never dispatch) on the tick it is first observed,
# and a later tick at the SAME reopen_at must NOT re-attempt the doomed
# write nor file a duplicate escalation. Unlike every other test in this
# file, _mark_in_progress_done is NOT replaced by a bare AsyncMock here —
# these tests exercise the real StaleEvidenceRejection catch (step-6) plus
# the gate's own should_skip pre-check (step-8), so scheduler.mark_done
# itself is the stale-rejecting double and _mark_in_progress_done runs for
# real.
# ---------------------------------------------------------------------------

_STALE_REOPEN_AT = '2026-07-15T00:00:00+00:00'


def _wire_stale_evidence_mark_done(h: Harness, *, evidence_commit: str) -> None:
    """Replace scheduler.mark_done with one that always rejects as stale.

    Mirrors test_reconcile_stranded.py's identically-named helper (kept
    local rather than imported since that module is a test file, not a
    shared fixture library).
    """
    from orchestrator.scheduler import StaleEvidenceRejection

    async def _stale_reject(tid, *, kind, sha, note=None):  # noqa: ARG001
        raise StaleEvidenceRejection(
            task_id=tid,
            evidence_commit=evidence_commit,
            evidence_committed_at='2026-07-10T00:00:00+00:00',
            reopen_at=_STALE_REOPEN_AT,
            agent_id='claude-recon-x',
            raw="success=False payload={'error': 'done_evidence_stale'}",
        )
    h.scheduler.mark_done = AsyncMock(side_effect=_stale_reject)


def _let_mark_in_progress_done_run_for_real(h: Harness, tmp_path) -> None:
    """Undo the blanket ``h._mark_in_progress_done = AsyncMock()`` the
    ``_wired_*_harness`` builders apply, and give ``_resolve_task_worktree``
    a real (nonexistent) tmp path so the worktree-cleanup branch is a clean
    no-op (``worktree_path.exists()`` is False) rather than tripping over
    ``h.git_ops``'s auto-mocked ``warm_lane_pool``/``cleanup_worktree``.
    """
    del h._mark_in_progress_done
    h.git_ops.warm_lane_pool = None
    h.git_ops.worktree_base = tmp_path / 'worktrees'


@pytest.mark.asyncio
class TestAlreadyLandedDispatchGateStaleEvidenceConflict:
    """RED until step-8 adds the should_skip pre-check to the gate."""

    async def test_ancestry_path_gates_and_does_not_reattempt(
        self, mock_orch_config, tmp_path,
    ) -> None:
        from escalation.queue import EscalationQueue

        from orchestrator.provenance_conflict import ProvenanceConflictSink

        h = _wired_ancestry_harness(mock_orch_config)
        _let_mark_in_progress_done_run_for_real(h, tmp_path)
        h._escalation_queue = EscalationQueue(tmp_path / 'esc')
        h._provenance_conflict_sink = ProvenanceConflictSink(
            escalation_queue=h._escalation_queue,
        )
        h.scheduler.get_task = AsyncMock(
            return_value={'id': '42', 'metadata': {'reopen_at': _STALE_REOPEN_AT}},
        )
        _wire_stale_evidence_mark_done(h, evidence_commit='a' * 40)

        result_1 = await h._already_landed_dispatch_gate('42')

        assert result_1 is True, 'a contested task must not dispatch'
        assert cast(AsyncMock, h.scheduler.mark_done).await_count == 1
        conflicts = [
            e for e in h._escalation_queue.get_by_task('42', status='pending')
            if e.category == 'provenance_conflict'
        ]
        assert len(conflicts) == 1, f'expected exactly one pending L2, got {len(conflicts)}'
        assert conflicts[0].level == 2
        assert conflicts[0].severity == 'urgent'

        result_2 = await h._already_landed_dispatch_gate('42')

        assert result_2 is True, 'must stay gated on a repeat tick'
        assert cast(AsyncMock, h.scheduler.mark_done).await_count == 1, (
            'a repeat tick at the same reopen_at must not re-attempt the '
            'already-rejected write'
        )
        conflicts_after = [
            e for e in h._escalation_queue.get_by_task('42', status='pending')
            if e.category == 'provenance_conflict'
        ]
        assert len(conflicts_after) == 1, 'must not file a duplicate escalation'

    async def test_content_equivalence_path_gates_and_does_not_reattempt(
        self, mock_orch_config, tmp_path,
    ) -> None:
        from escalation.queue import EscalationQueue

        from orchestrator.provenance_conflict import ProvenanceConflictSink

        # A citation is required for task 2678's DISCOVERY-mode gate to accept
        # the landing and proceed to the done-write — only then can the
        # stale-evidence rejection this test exercises fire from mark_done.
        h = _wired_content_harness(
            mock_orch_config, citation_sha='c' * 40, content_in_main=True,
        )
        _let_mark_in_progress_done_run_for_real(h, tmp_path)
        h._escalation_queue = EscalationQueue(tmp_path / 'esc')
        h._provenance_conflict_sink = ProvenanceConflictSink(
            escalation_queue=h._escalation_queue,
        )
        h.scheduler.get_task = AsyncMock(
            return_value={'id': '42', 'metadata': {'reopen_at': _STALE_REOPEN_AT}},
        )
        _wire_stale_evidence_mark_done(h, evidence_commit='c' * 40)

        result_1 = await h._already_landed_dispatch_gate('42')

        assert result_1 is True, 'a contested task must not dispatch'
        assert cast(AsyncMock, h.scheduler.mark_done).await_count == 1
        conflicts = [
            e for e in h._escalation_queue.get_by_task('42', status='pending')
            if e.category == 'provenance_conflict'
        ]
        assert len(conflicts) == 1, f'expected exactly one pending L2, got {len(conflicts)}'

        result_2 = await h._already_landed_dispatch_gate('42')

        assert result_2 is True, 'must stay gated on a repeat tick'
        assert cast(AsyncMock, h.scheduler.mark_done).await_count == 1, (
            'a repeat tick at the same reopen_at must not re-attempt the '
            'already-rejected write'
        )
        conflicts_after = [
            e for e in h._escalation_queue.get_by_task('42', status='pending')
            if e.category == 'provenance_conflict'
        ]
        assert len(conflicts_after) == 1, 'must not file a duplicate escalation'


def _wired_absent_branch_harness(mock_orch_config) -> Harness:
    """Bare harness with resolve_branch_sha -> None (branch ref does not
    exist, e.g. a fresh task never dispatched).  is_ancestor and
    branch_content_in_main are wired but must NEVER be called — the cheap
    pre-filter should route straight to (and stop at) the marker check.
    """
    h = _build_harness(mock_orch_config)
    h.git_ops = MagicMock()
    h.git_ops.config.branch_prefix = 'task/'
    h.git_ops.config.main_branch = 'main'

    h.git_ops.resolve_branch_sha = AsyncMock(return_value=None)
    h.git_ops.is_ancestor = AsyncMock(return_value=False)
    h.git_ops.branch_content_in_main = AsyncMock(return_value=False)
    h.git_ops.find_merge_marker = AsyncMock(return_value=None)
    h.git_ops.find_task_citation_commit = AsyncMock(return_value=None)

    h.scheduler.get_task = AsyncMock(return_value={'id': '42', 'metadata': {}})
    h._branch_is_degenerate = AsyncMock(return_value=False)
    h._mark_in_progress_done = AsyncMock()
    h._escalation_queue = None
    return h


@pytest.mark.asyncio
class TestAlreadyLandedDispatchGateBranchExistencePreFilter:
    """Cheap pre-filter (task 2313 review): resolve_branch_sha is resolved
    ONCE up front instead of letting is_ancestor / branch_content_in_main
    each spawn a subprocess that would only fail through to False.  These
    tests pin the reduced-I/O shape of both the branch-absent and the
    branch-exists-but-not-landed common cases.
    """

    async def test_missing_branch_skips_ancestor_and_content_checks(
        self, mock_orch_config,
    ) -> None:
        """A branch that doesn't exist yet must never reach is_ancestor or
        branch_content_in_main — both would just fail through to False at
        the cost of a wasted subprocess.  Only the marker search (the one
        check meaningful without a live ref) runs.
        """
        h = _wired_absent_branch_harness(mock_orch_config)

        result = await h._already_landed_dispatch_gate('42')

        assert result is False
        cast(AsyncMock, h.git_ops.is_ancestor).assert_not_called()
        cast(AsyncMock, h.git_ops.branch_content_in_main).assert_not_called()
        cast(AsyncMock, h.git_ops.find_merge_marker).assert_awaited_once()
        cast(AsyncMock, h._mark_in_progress_done).assert_not_awaited()

    async def test_existing_not_landed_branch_skips_marker_and_citation(
        self, mock_orch_config,
    ) -> None:
        """A branch that exists but isn't landed (not an ancestor, content
        not equivalent) must never call find_merge_marker — its own
        internal gate would just return None anyway since the branch ref
        still exists — nor find_task_citation_commit, which only the
        ancestry and content-equivalence paths consume.
        """
        h = _build_harness(mock_orch_config)
        h.git_ops = MagicMock()
        h.git_ops.config.branch_prefix = 'task/'
        h.git_ops.config.main_branch = 'main'

        h.git_ops.resolve_branch_sha = AsyncMock(return_value='f' * 40)
        h.git_ops.is_ancestor = AsyncMock(return_value=False)
        h.git_ops.branch_content_in_main = AsyncMock(return_value=False)
        h.git_ops.find_merge_marker = AsyncMock(return_value=None)
        h.git_ops.find_task_citation_commit = AsyncMock(return_value=None)

        h.scheduler.get_task = AsyncMock(return_value={'id': '42', 'metadata': {}})
        h._branch_is_degenerate = AsyncMock(return_value=False)
        h._mark_in_progress_done = AsyncMock()
        h._escalation_queue = None

        result = await h._already_landed_dispatch_gate('42')

        assert result is False
        cast(AsyncMock, h.git_ops.find_merge_marker).assert_not_called()
        cast(AsyncMock, h.git_ops.find_task_citation_commit).assert_not_called()
        cast(AsyncMock, h._mark_in_progress_done).assert_not_awaited()


@pytest.mark.asyncio
class TestAlreadyLandedDispatchGateInstall:
    """Harness.__init__ wires scheduler._already_landed_gate to the bound method."""

    async def test_scheduler_attribute_wired_to_bound_method(
        self, mock_orch_config,
    ) -> None:
        """RED until step-12 installs the wiring in Harness.__init__.

        Same bound-method equality idiom as
        TestHarnessLandedDispatchGateInstall (task 2156) — a freshly-accessed
        bound method is a new wrapper object each time, so ``==`` (not
        ``is``) is the correct comparison; MagicMock retains the exact
        object assigned during __init__, so this is a genuine RED
        (unset/auto-vivified Mock != bound method) before Harness.__init__
        wires the callable.
        """
        h = _build_harness(mock_orch_config)

        assert h.scheduler._already_landed_gate == h._already_landed_dispatch_gate, (
            'Harness must wire scheduler._already_landed_gate = '
            'harness._already_landed_dispatch_gate after construction'
        )


# ---------------------------------------------------------------------------
# TestAlreadyLandedGateDeliveredChecksGuard (task 3057 — step-3 RED / step-4 GREEN)
#
# Seam 2 of the eleven attribution-shaped mark-done seams. This gate's three
# evidence arms (git ancestry / merge marker / content equivalence) each prove
# only that SOMETHING of this branch reached main — never that THIS task's
# declared capability survived to it. All three are covered, in one
# parametrized matrix, so no single arm can regress unguarded.
# ---------------------------------------------------------------------------

_GATE_TARGET = 'orchestrator.harness.gate_mark_done_on_delivered_checks'

#: (arm id, reason/site label, expected evidence sha) for the three stamp arms.
_ARMS = [
    pytest.param('ancestry', id='ancestry-arm'),
    pytest.param('marker', id='marker-arm'),
    pytest.param('content', id='content-equivalent-arm'),
]

_ARM_REASON = {
    'ancestry': 'dispatch-gate-already-on-main',
    'marker': 'dispatch-gate-marker-found',
    'content': 'dispatch-gate-content-equivalent',
}

_CITATION_SHA = 'a' * 40
_MARKER_SHA = 'b' * 40
_BRANCH_BASE_SHA = '9' * 40

_ARM_EVIDENCE_SHA = {
    'ancestry': _CITATION_SHA,
    'marker': _MARKER_SHA,
    'content': _CITATION_SHA,
}

_DC_CHECK = {'name': 'cap-x', 'kind': 'grep', 'pattern': 'SomePattern', 'expect': 'present'}


def _arm_harness(mock_orch_config, arm: str, *, metadata: dict | None = None) -> Harness:
    """Build a harness wired so *arm* is the arm that reaches a stamp.

    Reuses this module's existing per-arm wiring helpers verbatim, then
    overlays the task metadata (each arm's helper hard-codes its own) and a
    real ``delivered_checks`` config section so the guard has live knobs to
    forward.
    """
    if arm == 'ancestry':
        h = _wired_ancestry_harness(mock_orch_config)
        base_meta: dict = {}
    elif arm == 'marker':
        h = _wired_marker_harness(
            mock_orch_config,
            marker_sha=_MARKER_SHA,
            branch_base_sha=_BRANCH_BASE_SHA,
            marker_is_ancestor_of_base=False,
        )
        base_meta = {'branch_base_sha': _BRANCH_BASE_SHA}
    elif arm == 'content':
        h = _wired_content_harness(
            mock_orch_config, citation_sha=_CITATION_SHA, content_in_main=True,
        )
        base_meta = {}
    else:  # pragma: no cover - guard against a typo in the parametrization
        raise AssertionError(f'unknown arm {arm!r}')

    meta = dict(base_meta)
    meta.update(metadata if metadata is not None else {'delivered_checks': [_DC_CHECK]})
    h.scheduler.get_task = AsyncMock(return_value={'id': '42', 'metadata': meta})
    h.config.delivered_checks = DeliveredChecksConfig(
        enabled=True, check_timeout_secs=11.0,
    )
    return h, meta


@pytest.mark.asyncio
class TestAlreadyLandedGateDeliveredChecksGuard:
    """The delivered-capability guard on the pre-dispatch already-landed gate.

    Task 2794's six-row acceptance matrix, applied to seam 2 and replicated
    across ALL THREE evidence arms. The recovery on every block is this
    gate's OWN existing "no landing evidence" path — ``return False``, i.e.
    deliberately the pre-2313 behavior (dispatch normally) rather than gating.
    A ``False`` return can never wedge a task the way a permanent gate could,
    and the invariant being defended is "never stamp a hollow done".
    """

    # --- row 1: hollow-done regression / FAILED ---------------------------

    @pytest.mark.parametrize('arm', _ARMS)
    async def test_failed_block_withholds_stamp_and_dispatches(
        self, mock_orch_config, arm,
    ) -> None:
        """FAILED: the branch reached main but the declared capability did
        NOT. No stamp, and the gate returns False so the task DISPATCHES and
        an agent actually delivers it."""
        h, meta = _arm_harness(mock_orch_config, arm)
        guard = AsyncMock(return_value=DeliveredChecksBlock(
            reason='failed', main_sha='m' * 40, failed_check=_DC_CHECK,
        ))

        with patch(_GATE_TARGET, guard):
            result = await h._already_landed_dispatch_gate('42')

        assert result is False
        cast(AsyncMock, h._mark_in_progress_done).assert_not_awaited()

        guard.assert_awaited_once()
        assert guard.await_args.args[0] == '42'
        assert guard.await_args.args[1] == meta
        kwargs = guard.await_args.kwargs
        assert kwargs['project_root'] == str(h.config.project_root)
        assert kwargs['check_timeout_secs'] == 11.0
        assert kwargs['enabled'] is True
        assert kwargs['site'] == _ARM_REASON[arm]

    # --- row 2: all_delivered -> byte-identical stamp ----------------------

    @pytest.mark.parametrize('arm', _ARMS)
    async def test_all_delivered_stamps_exactly_as_today(
        self, mock_orch_config, arm,
    ) -> None:
        """The capability IS on main -> the arm's existing stamp fires with
        its exact evidence sha / note / reason, and the gate returns True."""
        h, _meta = _arm_harness(mock_orch_config, arm)
        guard = AsyncMock(return_value=None)

        with patch(_GATE_TARGET, guard):
            result = await h._already_landed_dispatch_gate('42')

        assert result is True
        mark = cast(AsyncMock, h._mark_in_progress_done)
        mark.assert_awaited_once()
        assert mark.await_args.args[0] == '42'
        assert mark.await_args.args[1] == _ARM_EVIDENCE_SHA[arm]
        assert mark.await_args.args[3] == _ARM_REASON[arm]

    # --- row 3: no delivered_checks -> unchanged, but still DELEGATED -----

    @pytest.mark.parametrize('arm', _ARMS)
    async def test_check_less_task_delegates_and_stamps(
        self, mock_orch_config, arm,
    ) -> None:
        """A check-less task must not gain a new requirement.

        The harness DELEGATES unconditionally (passing the metadata through)
        rather than short-circuiting itself — the inertness lives in the
        helper, pinned at source in test_delivered_check_gate.py. Duplicating
        it here would be a second place for the kill-switch/inertness rule to
        drift.
        """
        h, meta = _arm_harness(mock_orch_config, arm, metadata={})
        guard = AsyncMock(return_value=None)

        with patch(_GATE_TARGET, guard):
            result = await h._already_landed_dispatch_gate('42')

        assert result is True
        cast(AsyncMock, h._mark_in_progress_done).assert_awaited_once()
        guard.assert_awaited_once()
        assert 'delivered_checks' not in guard.await_args.args[1]

    # --- rows 4 & 5: fail-safe blocks are handled UNIFORMLY with FAILED ----

    @pytest.mark.parametrize('arm', _ARMS)
    @pytest.mark.parametrize('reason', ['errored', 'main_sha_unresolved'])
    async def test_fail_safe_blocks_also_withhold_and_dispatch(
        self, mock_orch_config, arm, reason,
    ) -> None:
        """ERRORED / main_sha_unresolved take the SAME recovery as FAILED.

        Deliberate: a malformed delivered_checks descriptor ERRORs forever, so
        a "wait and retry" degradation could WEDGE the task permanently. An
        extra dispatch of an already-landed task is the strictly better
        failure — it always terminates.
        """
        h, _meta = _arm_harness(mock_orch_config, arm)
        guard = AsyncMock(return_value=DeliveredChecksBlock(reason=reason))

        with patch(_GATE_TARGET, guard):
            result = await h._already_landed_dispatch_gate('42')

        assert result is False
        cast(AsyncMock, h._mark_in_progress_done).assert_not_awaited()

    # --- row 6: kill switch is FORWARDED, never re-implemented -------------

    @pytest.mark.parametrize('arm', _ARMS)
    async def test_kill_switch_is_forwarded_not_short_circuited(
        self, mock_orch_config, arm,
    ) -> None:
        """``delivered_checks.enabled=False`` is forwarded to the helper, which
        owns the kill switch. The harness must NOT branch on it locally, or
        one hot reload would stop disarming all eleven seams at once."""
        h, _meta = _arm_harness(mock_orch_config, arm)
        h.config.delivered_checks = DeliveredChecksConfig(
            enabled=False, check_timeout_secs=11.0,
        )
        guard = AsyncMock(return_value=None)

        with patch(_GATE_TARGET, guard):
            result = await h._already_landed_dispatch_gate('42')

        assert result is True
        guard.assert_awaited_once()
        assert guard.await_args.kwargs['enabled'] is False

    @pytest.mark.parametrize('arm', _ARMS)
    async def test_kill_switch_with_real_helper_stamps_as_today(
        self, mock_orch_config, arm,
    ) -> None:
        """End-to-end with the REAL helper: disabled -> the stamp is
        byte-identical to today, with zero check work."""
        h, _meta = _arm_harness(mock_orch_config, arm)
        h.config.delivered_checks = DeliveredChecksConfig(
            enabled=False, check_timeout_secs=11.0,
        )

        result = await h._already_landed_dispatch_gate('42')

        assert result is True
        mark = cast(AsyncMock, h._mark_in_progress_done)
        mark.assert_awaited_once()
        assert mark.await_args.args[3] == _ARM_REASON[arm]

    # --- ordering: no wasted check work on landings already refused -------

    @pytest.mark.parametrize('arm', _ARMS)
    async def test_rejected_landing_evidence_never_reaches_the_guard(
        self, mock_orch_config, arm,
    ) -> None:
        """A REJECTED validate_landing_evidence verdict already means "no
        usable landing evidence" — the guard must not pay for a git-grep on a
        landing that is being refused anyway."""
        h, _meta = _arm_harness(mock_orch_config, arm)
        guard = AsyncMock(return_value=None)
        rejected = LandingEvidenceVerdict(
            accepted=False, evidence_sha=None, reason='no_citation', probe={},
        )

        with patch(_GATE_TARGET, guard), \
                patch(
                    'orchestrator.harness.validate_landing_evidence',
                    AsyncMock(return_value=rejected),
                ):
            result = await h._already_landed_dispatch_gate('42')

        assert result is False
        guard.assert_not_awaited()
        cast(AsyncMock, h._mark_in_progress_done).assert_not_awaited()

    async def test_degenerate_branch_never_reaches_the_guard(
        self, mock_orch_config,
    ) -> None:
        """The ancestry arm's degenerate-branch veto returns False BEFORE any
        evidence is derived — the guard must sit downstream of it."""
        h, _meta = _arm_harness(mock_orch_config, 'ancestry')
        h._branch_is_degenerate = AsyncMock(return_value=True)
        guard = AsyncMock(return_value=None)

        with patch(_GATE_TARGET, guard):
            result = await h._already_landed_dispatch_gate('42')

        assert result is False
        guard.assert_not_awaited()

    @pytest.mark.parametrize('arm', _ARMS)
    async def test_stale_conflict_skip_never_reaches_the_guard(
        self, mock_orch_config, arm,
    ) -> None:
        """The task-2677 provenance-conflict memo short-circuits BEFORE the
        git work; the guard must not resurrect that cost."""
        h, _meta = _arm_harness(mock_orch_config, arm)
        h._provenance_conflict_sink = MagicMock()
        h._provenance_conflict_sink.should_skip = MagicMock(return_value=True)
        guard = AsyncMock(return_value=None)

        with patch(_GATE_TARGET, guard):
            result = await h._already_landed_dispatch_gate('42')

        assert result is True
        guard.assert_not_awaited()
        cast(AsyncMock, h._mark_in_progress_done).assert_not_awaited()
