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

from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.harness import Harness


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


def _wired_marker_harness(
    mock_orch_config, *, marker_sha, branch_base_sha, marker_is_ancestor_of_base,
) -> Harness:
    """Bare harness with is_ancestor(branch, main) False, so the ancestry
    path never engages and the marker path is what's under test.

    ``is_ancestor`` is mocked with a side_effect function because it is
    called with two DIFFERENT argument pairs in this path: once for
    ``(branch, main_branch)`` (must be False to reach the marker path) and
    once for ``(marker_sha, branch_base_sha)`` (the stale-marker check).
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

    h.scheduler.get_task = AsyncMock(
        return_value={'id': '42', 'metadata': {'branch_base_sha': branch_base_sha}},
    )
    h._branch_is_degenerate = AsyncMock(return_value=False)
    h._mark_in_progress_done = AsyncMock()
    h._escalation_queue = None
    return h


@pytest.mark.asyncio
class TestAlreadyLandedDispatchGateMarkerPath:
    """Branch-deleted merge-marker path (RED until step-8).

    resolve_branch_sha(branch) is None in both sub-cases (the branch ref is
    gone), so the ancestry path never engages — only the marker path can
    produce a result.
    """

    async def test_marker_found_and_not_stale_flips_to_done(
        self, mock_orch_config,
    ) -> None:
        """A merge marker on main, not an ancestor of branch_base_sha (i.e.
        it postdates this incarnation's creation point) -> flips to done,
        anchored on the marker sha.

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

    async def test_stale_marker_vetoes_flip(self, mock_orch_config) -> None:
        """A marker that IS an ancestor of branch_base_sha predates this
        incarnation (branch was deleted + re-created under the same task
        id) -> vetoes the flip, no _mark_in_progress_done call.
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


def _wired_content_harness(
    mock_orch_config, *, citation_sha, main_sha, content_in_main,
) -> Harness:
    """Bare harness with the branch existing but is_ancestor False and
    find_merge_marker None, so neither the ancestry path nor the marker
    path can produce a result — only the content-equivalence fallback is
    under test.

    The branch must "exist" (resolve_branch_sha truthy) for this fallback
    to even be reached — the cheap pre-filter (task 2313 review) routes a
    nonexistent branch straight to the marker path instead.
    """
    h = _build_harness(mock_orch_config)
    h.git_ops = MagicMock()
    h.git_ops.config.branch_prefix = 'task/'
    h.git_ops.config.main_branch = 'main'

    h.git_ops.resolve_branch_sha = AsyncMock(return_value='f' * 40)
    h.git_ops.is_ancestor = AsyncMock(return_value=False)
    h.git_ops.find_merge_marker = AsyncMock(return_value=None)
    h.git_ops.find_task_citation_commit = AsyncMock(return_value=citation_sha)
    h.git_ops.branch_content_in_main = AsyncMock(return_value=content_in_main)
    h.git_ops.get_main_sha = AsyncMock(return_value=main_sha)

    h.scheduler.get_task = AsyncMock(return_value={'id': '42', 'metadata': {}})
    h._branch_is_degenerate = AsyncMock(return_value=False)
    h._mark_in_progress_done = AsyncMock()
    h._escalation_queue = None
    return h


@pytest.mark.asyncio
class TestAlreadyLandedDispatchGateContentEquivalence:
    """Content-equivalence fallback (RED until step-10).

    is_ancestor(branch, main) is False and find_merge_marker returns None
    in all three sub-cases, so only the content-equivalence fallback can
    produce a result.
    """

    async def test_content_equivalent_no_citation_anchors_on_main_head(
        self, mock_orch_config,
    ) -> None:
        """branch_content_in_main True, no citation on main -> flips to
        done, anchored on main HEAD (get_main_sha) since there is no
        citation commit to prefer.
        """
        main_sha = 'c' * 40
        h = _wired_content_harness(
            mock_orch_config,
            citation_sha=None,
            main_sha=main_sha,
            content_in_main=True,
        )

        result = await h._already_landed_dispatch_gate('42')

        assert result is True
        cast(AsyncMock, h._mark_in_progress_done).assert_awaited_once()
        call_args = cast(AsyncMock, h._mark_in_progress_done).await_args
        assert call_args is not None
        assert call_args.args[0] == '42'
        assert call_args.args[1] == main_sha
        assert call_args.args[3] == 'dispatch-gate-content-equivalent'

    async def test_content_equivalent_with_citation_prefers_citation_anchor(
        self, mock_orch_config,
    ) -> None:
        """branch_content_in_main True AND a citation commit is present on
        main -> the citation sha anchors the flip, not main HEAD.
        """
        citation_sha = 'd' * 40
        h = _wired_content_harness(
            mock_orch_config,
            citation_sha=citation_sha,
            main_sha='c' * 40,
            content_in_main=True,
        )

        result = await h._already_landed_dispatch_gate('42')

        assert result is True
        cast(AsyncMock, h._mark_in_progress_done).assert_awaited_once()
        call_args = cast(AsyncMock, h._mark_in_progress_done).await_args
        assert call_args is not None
        assert call_args.args[0] == '42'
        assert call_args.args[1] == citation_sha
        assert call_args.args[3] == 'dispatch-gate-content-equivalent'

    async def test_content_not_equivalent_dispatches_normally(
        self, mock_orch_config,
    ) -> None:
        """branch_content_in_main False -> no evidence at all; the gate
        returns False so the task dispatches normally this tick.
        """
        h = _wired_content_harness(
            mock_orch_config,
            citation_sha=None,
            main_sha='c' * 40,
            content_in_main=False,
        )

        result = await h._already_landed_dispatch_gate('42')

        assert result is False
        cast(AsyncMock, h._mark_in_progress_done).assert_not_awaited()


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
