"""Tests for the journal-first already-merged guard collapse (task 2245 / PRD α).

``MergeProvenance.lookup(task_id)`` (task 2153 / W1 α) is the single, authoritative
source consulted by every already-merged guard before falling back to the legacy
``_has_prior_implementation`` heuristic. See ``plans/workflow-state-machine-prd.md``
Contract §8 (MP-1: journal-first; MP-2: no recovery-DONE without a provenance basis).

``MergeProvenance._outbox`` is a process-global set via ``MergeProvenance.bind`` —
the autouse ``_reset_merge_provenance`` fixture resets it before and after every
test in this module so a bound outbox never leaks into another test.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _workflow_helpers import (  # noqa: F401  _Fixture: re-export, see test_workflow_helpers.py
    _bind_landed_row,
    _Fixture,
    _make,
)

from orchestrator.config import DeliveredChecksConfig
from orchestrator.delivered_checks import DeliveredChecksBlock
from orchestrator.landed_outbox import LandedOutbox, LandedRow, MergeProvenance
from orchestrator.merge_queue import reconcile_landed_outbox
from orchestrator.scheduler import SetTaskStatusRejected
from orchestrator.workflow import (
    WorkflowOutcome,
    WorkflowState,
    _PriorImplStatus,
)


@pytest.fixture(autouse=True)
def _reset_merge_provenance():
    """MergeProvenance._outbox is a process-global — never leak a bound outbox."""
    MergeProvenance._outbox = None
    yield
    MergeProvenance._outbox = None


# ---------------------------------------------------------------------------
# Tests: TaskWorkflow._finalise_recovery_done
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFinaliseRecoveryDone:
    """Unit tests for TaskWorkflow._finalise_recovery_done() (PRD α, MP-2 chokepoint).

    The sole writer of ``self._merge_recovery_basis`` — every already-merged
    guard's only route to a recovery-DONE goes through this method.
    """

    async def test_journal_basis_sets_marker_and_marks_done(self, tmp_path: Path):
        f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')

        outcome = await f.wf._finalise_recovery_done(
            basis='journal', sha='advancedsha123', kind='merged',
            note='landed-outbox journal hit (pre-PLAN recovery)',
        )

        assert outcome == WorkflowOutcome.DONE
        assert f.wf._merge_recovery_basis == 'journal'
        assert f.wf.state == WorkflowState.DONE
        f.mark_done.assert_awaited_once_with(
            f.wf.task_id, kind='merged', sha='advancedsha123',
            note='landed-outbox journal hit (pre-PLAN recovery)',
        )
        f.set_task_status.assert_awaited_once()
        args, kwargs = f.set_task_status.await_args
        assert args[0] == f.wf.task_id
        assert args[1] == 'done'
        assert kwargs['done_provenance']['kind'] == 'merged'
        assert kwargs['done_provenance']['commit'] == 'advancedsha123'

    async def test_fallback_basis_sets_marker_and_marks_done(self, tmp_path: Path):
        f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')

        outcome = await f.wf._finalise_recovery_done(
            basis='fallback', sha='mainsha123', kind='found_on_main',
            note='branch already on main at workflow start (pre-PLAN recovery)',
        )

        assert outcome == WorkflowOutcome.DONE
        assert f.wf._merge_recovery_basis == 'fallback'
        assert f.wf.state == WorkflowState.DONE
        f.mark_done.assert_awaited_once_with(
            f.wf.task_id, kind='found_on_main', sha='mainsha123',
            note='branch already on main at workflow start (pre-PLAN recovery)',
        )

    async def test_mark_done_rejection_routes_to_mark_blocked_not_phantom_done(
        self, tmp_path: Path,
    ):
        """A rejected mark_done must route to _mark_blocked, not report DONE."""
        f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')
        rejection = SetTaskStatusRejected(
            task_id=f.wf.task_id, error_code='conflict', raw='row already terminal',
        )
        f.wf.scheduler.mark_done = AsyncMock(side_effect=rejection)
        mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
        f.wf._mark_blocked = mark_blocked  # type: ignore[method-assign]

        outcome = await f.wf._finalise_recovery_done(
            basis='fallback', sha='mainsha123', kind='found_on_main',
            note='fallback note',
        )

        assert outcome == WorkflowOutcome.BLOCKED
        mark_blocked.assert_awaited_once()
        assert mark_blocked.await_args is not None
        _args, kwargs = mark_blocked.await_args
        assert kwargs.get('escalate_to_human') is True


# ---------------------------------------------------------------------------
# Tests: TaskWorkflow._recover_if_already_merged (Guard 1, rewired journal-first)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRecoverIfAlreadyMerged:
    """Unit tests for the pre-PLAN guard rewired onto the journal-first lookup.

    ``_check_branch_on_main`` is stubbed directly (mirrors
    ``test_workflow.py``'s ``wf._check_branch_on_main = AsyncMock(...)``
    convention) so these stay unit tests of the guard's decision logic —
    no real git subprocess involved.
    """

    async def test_journal_hit_returns_done_without_consulting_git_or_fallback(
        self, tmp_path: Path,
    ):
        """A journal hit is authoritative — short-circuits before the
        git-layer probe and the legacy heuristic are ever consulted [row 1]."""
        f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')
        _bind_landed_row(tmp_path, task_id=f.wf.task_id, advanced_sha='advancedsha123')
        f.wf._check_branch_on_main = AsyncMock(  # type: ignore[method-assign]
            side_effect=AssertionError(
                '_check_branch_on_main must not be called on a journal hit',
            ),
        )
        f.wf._has_prior_implementation = MagicMock(  # type: ignore[method-assign]
            side_effect=AssertionError(
                '_has_prior_implementation must not be called on a journal hit',
            ),
        )

        outcome = await f.wf._recover_if_already_merged()

        assert outcome == WorkflowOutcome.DONE
        assert f.wf._merge_recovery_basis == 'journal'
        f.mark_done.assert_awaited_once_with(
            f.wf.task_id, kind='merged', sha='advancedsha123',
            note='landed-outbox journal hit (pre-PLAN recovery)',
        )

    async def test_journal_miss_on_main_with_prior_work_falls_back_to_found_on_main(
        self, tmp_path: Path,
    ):
        """Journal miss + on-main + prior implementation work → fallback DONE."""
        f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')
        f.wf._check_branch_on_main = AsyncMock(  # type: ignore[method-assign]
            return_value=('wthead123', 'mainsha123'),
        )
        f.wf._has_prior_implementation = MagicMock(  # type: ignore[method-assign]
            return_value=_PriorImplStatus(has_work=True, entries=[], base_commit=None),
        )

        outcome = await f.wf._recover_if_already_merged()

        assert outcome == WorkflowOutcome.DONE
        assert f.wf._merge_recovery_basis == 'fallback'
        f.mark_done.assert_awaited_once_with(
            f.wf.task_id, kind='found_on_main', sha='mainsha123',
            note='branch already on main at workflow start (pre-PLAN recovery)',
        )
        f.wf._has_prior_implementation.assert_called_once_with(wt_head='wthead123')

    async def test_journal_miss_on_main_with_no_prior_work_returns_none(
        self, tmp_path: Path,
    ):
        """Journal miss + on-main + no prior work → no recovery (rows 2-3:
        inherited .task/ contamination / rebased-HEAD==base must not false-done)."""
        f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')
        f.wf._check_branch_on_main = AsyncMock(  # type: ignore[method-assign]
            return_value=('wthead123', 'mainsha123'),
        )
        f.wf._has_prior_implementation = MagicMock(  # type: ignore[method-assign]
            return_value=_PriorImplStatus(has_work=False, entries=[], base_commit=None),
        )

        outcome = await f.wf._recover_if_already_merged()

        assert outcome is None
        assert f.wf._merge_recovery_basis is None
        f.mark_done.assert_not_awaited()

    async def test_journal_miss_not_on_main_returns_none(self, tmp_path: Path):
        """Journal miss + branch not on main → no recovery, no fallback probe."""
        f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')
        f.wf._check_branch_on_main = AsyncMock(return_value=None)  # type: ignore[method-assign]
        f.wf._has_prior_implementation = MagicMock(  # type: ignore[method-assign]
            side_effect=AssertionError(
                '_has_prior_implementation must not be called when not on main',
            ),
        )

        outcome = await f.wf._recover_if_already_merged()

        assert outcome is None
        assert f.wf._merge_recovery_basis is None
        f.mark_done.assert_not_awaited()


# ---------------------------------------------------------------------------
# Tests: TaskWorkflow._has_prior_implementation per-entry classification
# (Layer A, task 2372 — recurrence class 2125/2315/2340)
# ---------------------------------------------------------------------------

# Each case: (iteration-log entry, expected has_work). Covers the entry
# shapes actually written by the codebase (see workflow.py:4443 main
# implementer, :4559 judge early_exit, :5218 debugger, :5443 amendment
# implementer) plus the exact task-2125 poison shape (zero-work implementer
# entry with no 'source' key).
_CLASSIFICATION_CASES = [
    pytest.param(
        {
            'agent': 'implementer', 'source': 'orchestrator',
            'steps_attempted': [], 'steps_completed': [],
            'commit': 'oldbase',
        },
        False,
        id='zero_work_main_implementer',
    ),
    pytest.param(
        {
            'iteration': 1, 'agent': 'implementer',
            'steps_attempted': [], 'steps_completed': [],
            'commit': '3bbdf8a1', 'summary': 'No new steps completed',
        },
        False,
        id='task_2125_poison_no_source_key',
    ),
    pytest.param(
        {'agent': 'debugger', 'steps_completed': []},
        True,
        id='debugger_hardcoded_empty_steps_completed',
    ),
    pytest.param(
        {'agent': 'implementer', 'source': 'amendment', 'amendment_round': 1},
        True,
        id='amendment_implementer_omits_steps_completed',
    ),
    pytest.param(
        {'agent': 'judge', 'event': 'early_exit', 'substantive_work': True},
        True,
        id='judge_early_exit_substantive',
    ),
    pytest.param(
        {'agent': 'judge', 'event': 'early_exit', 'substantive_work': False},
        False,
        id='judge_early_exit_not_substantive',
    ),
    pytest.param(
        {
            'agent': 'implementer', 'source': 'orchestrator',
            'steps_attempted': ['s1'], 'steps_completed': ['s1'],
            'commit': 'newhead',
        },
        True,
        id='main_implementer_nonempty_steps_completed',
    ),
]


@pytest.mark.asyncio
class TestHasPriorImplementationClassification:
    """Per-entry classification for the iteration-log fallback (task 2372,
    Layer A). The classifier must exclude ONLY the narrow zero-work
    implementer signature (``agent=='implementer' AND source!='amendment'
    AND not steps_completed``) while still counting debugger entries
    (which hard-code ``steps_completed:[]``), amendment-implementer entries
    (which omit ``steps_completed`` entirely), and judge ``early_exit``
    entries with ``substantive_work=True``.

    Each case is checked via both call shapes of
    ``_has_prior_implementation``: the bare fallback (no ``wt_head`` — used
    by the pre-EXECUTE and pre-MERGE guards) and the SHA-primary path
    (``wt_head`` supplied and diverging from ``base_commit='oldbase'`` —
    isolates the iteration-log term of the ``sha_diverges AND
    has_iter_log_work`` conjunction, since sha_diverges=True here makes
    has_work reduce to exactly has_iter_log_work).
    """

    @pytest.mark.parametrize('entry, expected_has_work', _CLASSIFICATION_CASES)
    async def test_bare_fallback_classification(
        self, tmp_path: Path, entry: dict, expected_has_work: bool,
    ):
        f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')
        f.artifacts.append_iteration_log(dict(entry))

        status = f.wf._has_prior_implementation()

        assert status.has_work is expected_has_work

    @pytest.mark.parametrize('entry, expected_has_work', _CLASSIFICATION_CASES)
    async def test_sha_primary_classification_isolates_iter_log_term(
        self, tmp_path: Path, entry: dict, expected_has_work: bool,
    ):
        f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')
        f.artifacts.append_iteration_log(dict(entry))

        status = f.wf._has_prior_implementation(wt_head='newhead-diverged')

        assert status.has_work is expected_has_work


# ---------------------------------------------------------------------------
# Tests: TaskWorkflow._recover_before_execute (Guard 2, extracted pre-EXECUTE)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRecoverBeforeExecute:
    """Unit tests for the extracted pre-EXECUTE ghost-loop guard (PRD α).

    Mirrors ``TestRecoverIfAlreadyMerged`` but for the guard that runs after
    PLAN and before the execute/verify/review loop.  External (eval-mode)
    worktrees never ghost-recover — they always run the execute loop, checked
    before the journal or any git probe.
    """

    async def test_external_worktree_never_recovers(self, tmp_path: Path):
        """worktree_external short-circuits to None before the journal or
        git layer are ever consulted — even when a journal hit exists."""
        f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')
        f.wf._worktree_external = True
        _bind_landed_row(tmp_path, task_id=f.wf.task_id, advanced_sha='advancedsha123')
        f.wf._check_branch_on_main = AsyncMock(  # type: ignore[method-assign]
            side_effect=AssertionError(
                '_check_branch_on_main must not be called for an external worktree',
            ),
        )

        outcome = await f.wf._recover_before_execute()

        assert outcome is None
        assert f.wf._merge_recovery_basis is None
        f.mark_done.assert_not_awaited()

    async def test_journal_hit_returns_done_without_consulting_git_or_fallback(
        self, tmp_path: Path,
    ):
        """A journal hit is authoritative — short-circuits before the
        git-layer probe and the legacy heuristic are ever consulted [row 1]."""
        f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')
        _bind_landed_row(tmp_path, task_id=f.wf.task_id, advanced_sha='advancedsha123')
        f.wf._check_branch_on_main = AsyncMock(  # type: ignore[method-assign]
            side_effect=AssertionError(
                '_check_branch_on_main must not be called on a journal hit',
            ),
        )
        f.wf._has_prior_implementation = MagicMock(  # type: ignore[method-assign]
            side_effect=AssertionError(
                '_has_prior_implementation must not be called on a journal hit',
            ),
        )

        outcome = await f.wf._recover_before_execute()

        assert outcome == WorkflowOutcome.DONE
        assert f.wf._merge_recovery_basis == 'journal'
        f.mark_done.assert_awaited_once_with(
            f.wf.task_id, kind='merged', sha='advancedsha123',
            note='landed-outbox journal hit (pre-EXECUTE recovery)',
        )

    async def test_journal_miss_on_main_with_prior_work_falls_back_to_found_on_main(
        self, tmp_path: Path,
    ):
        """Journal miss + on-main + non-empty branch-content diff (Layer C,
        task 2372: the ground-truth `git diff base..wt_head` gate) → fallback
        DONE."""
        f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')
        f.wf._check_branch_on_main = AsyncMock(  # type: ignore[method-assign]
            return_value=('wthead123', 'mainsha123'),
        )
        f.wf.git_ops.get_merge_diff_files = AsyncMock(return_value=(['a.py'], None))

        outcome = await f.wf._recover_before_execute()

        assert outcome == WorkflowOutcome.DONE
        assert f.wf._merge_recovery_basis == 'fallback'
        f.mark_done.assert_awaited_once_with(
            f.wf.task_id, kind='found_on_main', sha='mainsha123',
            note='branch already on main at workflow start (pre-EXECUTE recovery)',
        )

    async def test_journal_miss_on_main_with_no_prior_work_returns_none(
        self, tmp_path: Path,
    ):
        """Journal miss + on-main + EMPTY branch-content diff (Layer C, task
        2372) → no recovery, proceed normally — a fresh/re-dispatched
        worktree (wt_head trivially an ancestor of main) or a stale branch
        point must never false-DONE regardless of iteration-log contents."""
        f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')
        f.wf._check_branch_on_main = AsyncMock(  # type: ignore[method-assign]
            return_value=('wthead123', 'mainsha123'),
        )
        f.wf.git_ops.get_merge_diff_files = AsyncMock(return_value=([], None))

        outcome = await f.wf._recover_before_execute()

        assert outcome is None
        assert f.wf._merge_recovery_basis is None
        f.mark_done.assert_not_awaited()

    async def test_found_on_main_recovery_stamps_nonempty_metadata_files(
        self, tmp_path: Path,
    ):
        """ACTION #2: a pre-EXECUTE found_on_main recovery-DONE must stamp
        metadata.files with the real branch-diff files, never an empty list.

        Task 2125's phantom-done stamped ``metadata.files=[]`` — precisely
        what let it slip the reconciliation gate undetected.  Deriving files
        from the same branch-diff gate that authorizes the recovery (Layer
        C) closes that hole structurally: an empty diff now means NO
        recovery at all (see the prior test), so a found_on_main DONE from
        this guard always carries non-empty, real evidence.
        """
        f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')
        f.wf._check_branch_on_main = AsyncMock(  # type: ignore[method-assign]
            return_value=('wthead123', 'mainsha123'),
        )
        f.wf.git_ops.get_merge_diff_files = AsyncMock(
            return_value=(['pkg/a.py', 'pkg/b.py'], None),
        )

        outcome = await f.wf._recover_before_execute()

        assert outcome == WorkflowOutcome.DONE
        f.update_task.assert_awaited_once()
        args, _kwargs = f.update_task.await_args
        assert args[0] == f.wf.task_id
        metadata = args[1]
        assert metadata['files'], f'Expected non-empty files, got {metadata["files"]!r}'
        assert set(metadata['files']) == {'pkg/a.py', 'pkg/b.py'}

    async def test_journal_miss_not_on_main_returns_none(self, tmp_path: Path):
        """Journal miss + branch not on main → no recovery, no fallback probe."""
        f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')
        f.wf._check_branch_on_main = AsyncMock(return_value=None)  # type: ignore[method-assign]
        f.wf._has_prior_implementation = MagicMock(  # type: ignore[method-assign]
            side_effect=AssertionError(
                '_has_prior_implementation must not be called when not on main',
            ),
        )

        outcome = await f.wf._recover_before_execute()

        assert outcome is None
        assert f.wf._merge_recovery_basis is None
        f.mark_done.assert_not_awaited()


# ---------------------------------------------------------------------------
# Tests: TaskWorkflow._recover_before_merge (Guard 3, extracted merge-phase)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRecoverBeforeMerge:
    """Unit tests for the extracted merge-phase ghost-loop guard (PRD α).

    Mirrors ``TestRecoverBeforeExecute`` but for the guard that runs inside
    ``_run_merge_phase`` immediately after computing ``branch_head``/``main_sha``
    (which the caller passes in explicitly — this helper does no git I/O of
    its own beyond the ``is_ancestor`` ghost-loop check).
    """

    async def test_journal_hit_returns_done_without_consulting_git_or_fallback(
        self, tmp_path: Path,
    ):
        """A journal hit is authoritative — short-circuits before is_ancestor
        and the legacy heuristic are ever consulted [row 1]."""
        f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')
        _bind_landed_row(tmp_path, task_id=f.wf.task_id, advanced_sha='advancedsha123')
        f.is_ancestor.side_effect = AssertionError(
            'is_ancestor must not be called on a journal hit',
        )
        f.wf._has_prior_implementation = MagicMock(  # type: ignore[method-assign]
            side_effect=AssertionError(
                '_has_prior_implementation must not be called on a journal hit',
            ),
        )

        outcome = await f.wf._recover_before_merge('branchhead123', 'mainsha123')

        assert outcome == WorkflowOutcome.DONE
        assert f.wf._merge_recovery_basis == 'journal'
        f.mark_done.assert_awaited_once_with(
            f.wf.task_id, kind='merged', sha='advancedsha123',
            note='landed-outbox journal hit (pre-MERGE recovery)',
        )

    async def test_journal_miss_ancestor_with_prior_work_falls_back_to_found_on_main(
        self, tmp_path: Path,
    ):
        """Journal miss + branch is ancestor of main + prior implementation
        work → fallback DONE (provenance sha is main_sha, not branch_head)."""
        f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj', branch_on_main=True)
        f.wf._has_prior_implementation = MagicMock(  # type: ignore[method-assign]
            return_value=_PriorImplStatus(has_work=True, entries=[], base_commit=None),
        )

        outcome = await f.wf._recover_before_merge('branchhead123', 'mainsha123')

        assert outcome == WorkflowOutcome.DONE
        assert f.wf._merge_recovery_basis == 'fallback'
        f.mark_done.assert_awaited_once_with(
            f.wf.task_id, kind='found_on_main', sha='mainsha123',
            note='branch already on main at merge phase (pre-MERGE recovery)',
        )
        f.is_ancestor.assert_awaited_once_with('branchhead123', 'mainsha123')

    async def test_journal_miss_ancestor_with_no_prior_work_returns_none(
        self, tmp_path: Path,
    ):
        """Journal miss + ancestor (spurious merge signal) + no prior work →
        None, proceed with the real merge (task 2911-style guard)."""
        f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj', branch_on_main=True)
        f.wf._has_prior_implementation = MagicMock(  # type: ignore[method-assign]
            return_value=_PriorImplStatus(has_work=False, entries=[], base_commit=None),
        )

        outcome = await f.wf._recover_before_merge('branchhead123', 'mainsha123')

        assert outcome is None
        assert f.wf._merge_recovery_basis is None
        f.mark_done.assert_not_awaited()

    async def test_journal_miss_not_ancestor_returns_none(self, tmp_path: Path):
        """Journal miss + branch NOT an ancestor of main → None, fallback
        heuristic never consulted (a real merge is still needed)."""
        f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj', branch_on_main=False)
        f.wf._has_prior_implementation = MagicMock(  # type: ignore[method-assign]
            side_effect=AssertionError(
                '_has_prior_implementation must not be called when not an ancestor',
            ),
        )

        outcome = await f.wf._recover_before_merge('branchhead123', 'mainsha123')

        assert outcome is None
        assert f.wf._merge_recovery_basis is None
        f.mark_done.assert_not_awaited()

    async def test_warning_scoped_to_ancestor_without_work_only(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ):
        """The 'branch appears merged ... but has no implementation
        entries' breadcrumb (task 2504 subtlety A) must fire ONLY on the
        ancestor-but-no-work spurious-merge-signal sub-case — never on a
        normal divergent (not-ancestor) merge, which must stay silent."""
        breadcrumb = 'but has no implementation entries'

        # (a) Ancestor + no prior work → spurious merge signal → breadcrumb.
        f_ancestor = _make(
            worktree=tmp_path / 'wt-ancestor', project_root=tmp_path / 'proj-ancestor',
            branch_on_main=True,
        )
        f_ancestor.wf._has_prior_implementation = MagicMock(  # type: ignore[method-assign]
            return_value=_PriorImplStatus(has_work=False, entries=[], base_commit=None),
        )
        with caplog.at_level(logging.WARNING):
            outcome_ancestor = await f_ancestor.wf._recover_before_merge(
                'branchhead123', 'mainsha123',
            )
        assert outcome_ancestor is None
        assert breadcrumb in caplog.text

        caplog.clear()

        # (b) NOT an ancestor → real divergent merge → must stay silent.
        f_not_ancestor = _make(
            worktree=tmp_path / 'wt-not-ancestor', project_root=tmp_path / 'proj-not-ancestor',
            branch_on_main=False,
        )
        with caplog.at_level(logging.WARNING):
            outcome_not_ancestor = await f_not_ancestor.wf._recover_before_merge(
                'branchhead123', 'mainsha123',
            )
        assert outcome_not_ancestor is None
        assert breadcrumb not in caplog.text


# ---------------------------------------------------------------------------
# Tests: TaskWorkflow._branch_work_landed_on_main (task 2504)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestBranchWorkLandedOnMain:
    """Unit tests for the shared is_ancestor+has_work predicate (task 2504).

    ``_branch_work_landed_on_main`` extracts the ``is_ancestor(...) AND
    _has_prior_implementation(wt_head=...).has_work`` check that
    ``_recover_before_merge`` already implemented correctly, so it can be
    reused (with the correct ``wt_head`` mode per call site) at the
    ESCALATED-resume site in ``_drive()``, which previously did a raw
    ``is_ancestor`` check with no has-work guard and false-positived on an
    empty branch (base is trivially an ancestor of main).
    """

    async def test_ancestor_with_real_prior_work_returns_true(self, tmp_path: Path):
        """Ancestor + genuinely-implemented branch (SHA diverged from base
        AND a work-classified iteration-log entry) → True."""
        f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj', branch_on_main=True)
        f.artifacts.append_iteration_log({
            'agent': 'implementer', 'source': 'orchestrator',
            'steps_attempted': ['s1'], 'steps_completed': ['s1'],
            'commit': 'newhead',
        })

        result = await f.wf._branch_work_landed_on_main(
            'newhead-diverged', 'mainsha123', wt_head='newhead-diverged',
        )

        assert result is True

    async def test_ancestor_with_empty_branch_returns_false(self, tmp_path: Path):
        """Ancestor + empty branch (wt_head == base_commit == 'oldbase') →
        False — the empty-branch false positive this helper exists to
        prevent, even with a work-shaped iteration-log entry present (SHA
        non-divergence vetoes the log signal, per
        ``_has_prior_implementation``'s defense-in-depth contract)."""
        f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj', branch_on_main=True)
        f.artifacts.append_iteration_log({
            'agent': 'implementer', 'source': 'orchestrator',
            'steps_attempted': ['s1'], 'steps_completed': ['s1'],
            'commit': 'newhead',
        })

        result = await f.wf._branch_work_landed_on_main(
            'oldbase', 'mainsha123', wt_head='oldbase',
        )

        assert result is False

    async def test_not_ancestor_returns_false_without_consulting_has_work(
        self, tmp_path: Path,
    ):
        """Not an ancestor of main → False immediately; _has_prior_implementation
        must not even be consulted (mirrors the sibling guards' short-circuit
        contract)."""
        f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj', branch_on_main=False)
        f.wf._has_prior_implementation = MagicMock(  # type: ignore[method-assign]
            side_effect=AssertionError('must not be called when not an ancestor'),
        )

        result = await f.wf._branch_work_landed_on_main(
            'branchhead123', 'mainsha123', wt_head='branchhead123',
        )

        assert result is False


# ---------------------------------------------------------------------------
# Tests: MP-2 enforcement (PRD §9 boundary row 4) — no recovery-DONE without
# an explicit provenance basis
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFinaliseRecoveryDoneRefusesUnprovenancedDone:
    """`_finalise_recovery_done` is the sole chokepoint for a recovery-DONE
    (PRD α, MP-2). It must refuse to proceed without a valid provenance
    basis and a truthy sha, raising BEFORE any status mutation — no marker
    write, no phase transition, no scheduler call.
    """

    async def test_refuses_none_basis(self, tmp_path: Path):
        f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')

        with pytest.raises((AssertionError, ValueError)):
            await f.wf._finalise_recovery_done(
                basis=None, sha='somesha', kind='merged', note='n',  # type: ignore[arg-type]
            )

        assert f.wf._merge_recovery_basis is None
        assert f.wf.state == WorkflowState.PLAN
        f.mark_done.assert_not_awaited()

    async def test_refuses_invalid_basis(self, tmp_path: Path):
        f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')

        with pytest.raises((AssertionError, ValueError)):
            await f.wf._finalise_recovery_done(
                basis='hunch', sha='somesha', kind='merged', note='n',
            )

        assert f.wf._merge_recovery_basis is None
        assert f.wf.state == WorkflowState.PLAN
        f.mark_done.assert_not_awaited()

    async def test_refuses_empty_sha(self, tmp_path: Path):
        f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')

        with pytest.raises((AssertionError, ValueError)):
            await f.wf._finalise_recovery_done(
                basis='journal', sha='', kind='merged', note='n',
            )

        assert f.wf._merge_recovery_basis is None
        assert f.wf.state == WorkflowState.PLAN
        f.mark_done.assert_not_awaited()

    async def test_refuses_none_sha(self, tmp_path: Path):
        f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')

        with pytest.raises((AssertionError, ValueError)):
            await f.wf._finalise_recovery_done(
                basis='fallback', sha=None, kind='found_on_main', note='n',  # type: ignore[arg-type]
            )

        assert f.wf._merge_recovery_basis is None
        assert f.wf.state == WorkflowState.PLAN
        f.mark_done.assert_not_awaited()


@pytest.mark.asyncio
class TestNoPhantomDoneProperty:
    """Boundary row 4 (PRD §9): across all three already-merged guards, a
    journal-miss + no-prior-work shape must never produce a recovery-DONE
    (the task must remain re-dispatchable — ``_merge_recovery_basis`` stays
    ``None``), and whenever a guard DOES return ``WorkflowOutcome.DONE`` the
    marker must be a valid provenance basis (``'journal'`` or ``'fallback'``).
    """

    async def test_journal_miss_no_work_leaves_task_re_dispatchable_all_guards(
        self, tmp_path: Path,
    ):
        no_work = _PriorImplStatus(has_work=False, entries=[], base_commit=None)

        # Guard 1: _recover_if_already_merged — on-main, no prior work.
        f1 = _make(worktree=tmp_path / 'wt1', project_root=tmp_path / 'proj1')
        f1.wf._check_branch_on_main = AsyncMock(  # type: ignore[method-assign]
            return_value=('wthead123', 'mainsha123'),
        )
        f1.wf._has_prior_implementation = MagicMock(return_value=no_work)  # type: ignore[method-assign]
        outcome1 = await f1.wf._recover_if_already_merged()
        assert outcome1 is None
        assert f1.wf._merge_recovery_basis is None
        f1.mark_done.assert_not_awaited()

        # Guard 2: _recover_before_execute — on-main, EMPTY branch-content
        # diff (Layer C, task 2372 — this guard no longer consults
        # _has_prior_implementation at all).
        f2 = _make(worktree=tmp_path / 'wt2', project_root=tmp_path / 'proj2')
        f2.wf._check_branch_on_main = AsyncMock(  # type: ignore[method-assign]
            return_value=('wthead123', 'mainsha123'),
        )
        f2.wf.git_ops.get_merge_diff_files = AsyncMock(return_value=([], None))
        outcome2 = await f2.wf._recover_before_execute()
        assert outcome2 is None
        assert f2.wf._merge_recovery_basis is None
        f2.mark_done.assert_not_awaited()

        # Guard 3: _recover_before_merge — ancestor True (spurious merge
        # signal), no prior work.
        f3 = _make(
            worktree=tmp_path / 'wt3', project_root=tmp_path / 'proj3',
            branch_on_main=True,
        )
        f3.wf._has_prior_implementation = MagicMock(return_value=no_work)  # type: ignore[method-assign]
        outcome3 = await f3.wf._recover_before_merge('branchhead123', 'mainsha123')
        assert outcome3 is None
        assert f3.wf._merge_recovery_basis is None
        f3.mark_done.assert_not_awaited()

    async def test_journal_hit_done_always_carries_a_valid_basis_all_guards(
        self, tmp_path: Path,
    ):
        # Guard 1
        f1 = _make(worktree=tmp_path / 'wt1', project_root=tmp_path / 'proj1')
        _bind_landed_row(tmp_path, task_id=f1.wf.task_id, advanced_sha='sha1')
        outcome1 = await f1.wf._recover_if_already_merged()
        assert outcome1 == WorkflowOutcome.DONE
        assert f1.wf._merge_recovery_basis in ('journal', 'fallback')

        # Guard 2
        f2 = _make(worktree=tmp_path / 'wt2', project_root=tmp_path / 'proj2')
        _bind_landed_row(tmp_path, task_id=f2.wf.task_id, advanced_sha='sha2')
        outcome2 = await f2.wf._recover_before_execute()
        assert outcome2 == WorkflowOutcome.DONE
        assert f2.wf._merge_recovery_basis in ('journal', 'fallback')

        # Guard 3
        f3 = _make(worktree=tmp_path / 'wt3', project_root=tmp_path / 'proj3')
        _bind_landed_row(tmp_path, task_id=f3.wf.task_id, advanced_sha='sha3')
        outcome3 = await f3.wf._recover_before_merge('branchhead123', 'mainsha123')
        assert outcome3 == WorkflowOutcome.DONE
        assert f3.wf._merge_recovery_basis in ('journal', 'fallback')


# ---------------------------------------------------------------------------
# Tests: TaskWorkflow._finalise_merged_done consumes the write-ahead LandedRow
# on the single-branch happy path (task 2681/ζ — close the RC-3 stale-row
# window that reconcile_landed_outbox documented as a KNOWN LIMITATION).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFinaliseMergedDoneConsumesLandedRow:
    """The single-branch happy-path completion (``_finalise_merged_done``)
    consumes its write-ahead LandedRow via ``MergeProvenance.consume`` the
    moment ``mark_done`` succeeds — so a successfully-landed task no longer
    leaves a stale row that survives to the next orchestrator startup (the
    task-2155 KNOWN LIMITATION). The consume is scoped to the SUCCESS path
    only: a rejected done-write must leave the row for the reconciler to retry.
    """

    def _bind_outbox_with_row(
        self, tmp_path: Path, task_id: str, advanced_sha: str,
    ) -> LandedOutbox:
        """Construct + bind a real LandedOutbox holding a row for *task_id*,
        returning the handle so the caller can assert on it after the fact
        (unlike ``_bind_landed_row``, which discards the outbox reference)."""
        outbox = LandedOutbox(tmp_path / 'landed_outbox.json')
        outbox.record(LandedRow(
            task_id=task_id, branch_tip_sha='branchtip',
            advanced_sha=advanced_sha, landed_at=1.0,
        ))
        MergeProvenance.bind(outbox)
        return outbox

    async def test_happy_path_consumes_row_after_mark_done(self, tmp_path: Path):
        f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')
        outbox = self._bind_outbox_with_row(tmp_path, f.wf.task_id, 'advsha')
        f.wf._merge_sha = 'advsha'
        f.wf._reconcile_metadata_files_for_done = AsyncMock()  # type: ignore[method-assign]

        # Row present BEFORE completion.
        assert outbox.lookup(f.wf.task_id) is not None

        outcome = await f.wf._finalise_merged_done()

        assert outcome == WorkflowOutcome.DONE
        f.mark_done.assert_awaited_once_with(
            f.wf.task_id, kind='merged', sha='advsha',
        )
        # Row CONSUMED on completion — no restart / startup reconcile needed.
        assert outbox.lookup(f.wf.task_id) is None

    async def test_rc3_prunes_nothing_after_happy_path_completion(
        self, tmp_path: Path,
    ):
        """End-to-end: after a happy-path completion, driving the real startup
        RC-3 reconcile over the same bound outbox finds nothing to prune — the
        row was already consumed on completion. An UNconsumed 'done' row WOULD
        trip RC-3's ``already_done_pruned`` branch (get_status → 'done',
        advanced_sha an ancestor of main), so a zero prune count is a genuine
        RED→GREEN signal that the happy-path consume fired.
        """
        f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')
        outbox = self._bind_outbox_with_row(tmp_path, f.wf.task_id, 'advsha')
        f.wf._merge_sha = 'advsha'
        f.wf._reconcile_metadata_files_for_done = AsyncMock()  # type: ignore[method-assign]
        # get_status → 'done' so an unconsumed row WOULD be RC-3 pruned;
        # is_ancestor/get_main_sha are already AsyncMocks from _make
        # (is_ancestor → True, get_main_sha → 'mainsha123').
        f.wf.scheduler.get_status = AsyncMock(return_value='done')

        outcome = await f.wf._finalise_merged_done()
        assert outcome == WorkflowOutcome.DONE

        report = await reconcile_landed_outbox(
            outbox, f.wf.git_ops, f.wf.scheduler,
        )

        assert report['already_done_pruned'] == 0, (
            f'Expected zero RC-3 prunes for a happy-path-completed task; '
            f'got {report!r}'
        )
        assert report['marked_done'] == 0

    async def test_rejected_done_write_leaves_row_for_reconciler(
        self, tmp_path: Path,
    ):
        """Consume-only-on-success: a rejected ``mark_done`` routes to
        ``_mark_blocked`` and must NOT consume the write-ahead row — the task
        is not done, so the row must survive for the startup/dispatch
        reconciler to retry (RC-1/RC-2)."""
        f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')
        outbox = self._bind_outbox_with_row(tmp_path, f.wf.task_id, 'advsha')
        f.wf._merge_sha = 'advsha'
        f.wf._reconcile_metadata_files_for_done = AsyncMock()  # type: ignore[method-assign]
        rejection = SetTaskStatusRejected(
            task_id=f.wf.task_id, error_code='conflict', raw='row already terminal',
        )
        f.wf.scheduler.mark_done = AsyncMock(side_effect=rejection)
        mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
        f.wf._mark_blocked = mark_blocked  # type: ignore[method-assign]

        outcome = await f.wf._finalise_merged_done()

        assert outcome == WorkflowOutcome.BLOCKED
        mark_blocked.assert_awaited_once()
        # The write-ahead row must STILL be present — a rejected done-write
        # must not consume it.
        assert outbox.lookup(f.wf.task_id) is not None


# ---------------------------------------------------------------------------
# TestFinaliseRecoveryDoneDeliveredChecksGuard (task 3057 — step-5 RED /
# step-6 GREEN)
#
# Seams 4+7 of the eleven attribution-shaped mark-done seams, guarded at the
# ONE chokepoint they all funnel through. `_finalise_recovery_done` is the
# sole route to a recovery-DONE for all six recovery arms (3x journal
# kind='merged' + 3x fallback kind='found_on_main'), so the guard sits here
# rather than at the six call sites — a future seventh recovery arm cannot be
# added unguarded.
# ---------------------------------------------------------------------------

_WF_GATE_TARGET = 'orchestrator.workflow.gate_mark_done_on_delivered_checks'

#: (basis, kind) for the two recovery bases, parametrized everywhere so
#: neither the journal arm nor the fallback arm can regress independently.
_RECOVERY_ARMS = [
    pytest.param('journal', 'merged', id='journal-merged'),
    pytest.param('fallback', 'found_on_main', id='fallback-found-on-main'),
]

_WF_DC_CHECK = {
    'name': 'cap-x', 'kind': 'grep', 'pattern': 'SomePattern', 'expect': 'present',
}


def _arm_recovery_fixture(
    tmp_path: Path, *, metadata: dict | None = None, enabled: bool = True,
) -> _Fixture:
    """`_make` plus a live ``delivered_checks`` config and task metadata.

    The stock ``_make`` config is a ``MagicMock(spec_set=...)``, so
    ``config.delivered_checks`` would hand the guard MagicMocks rather than
    the real knobs it must forward. A real ``DeliveredChecksConfig`` keeps the
    forwarding assertions honest.
    """
    f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')
    f.wf.task['metadata'] = (
        {'delivered_checks': [_WF_DC_CHECK]} if metadata is None else metadata
    )
    f.wf.config.delivered_checks = DeliveredChecksConfig(
        enabled=enabled, check_timeout_secs=7.5,
    )
    f.wf._reconcile_metadata_files_for_done = AsyncMock()  # type: ignore[method-assign]
    return f


def _spy_phases(f: _Fixture) -> list[WorkflowState]:
    """Record every ``_enter_phase`` target while preserving real behaviour."""
    seen: list[WorkflowState] = []
    real = f.wf._enter_phase

    def _spy(new_state: WorkflowState) -> None:
        seen.append(new_state)
        real(new_state)

    f.wf._enter_phase = _spy  # type: ignore[method-assign]
    return seen


@pytest.mark.asyncio
class TestFinaliseRecoveryDoneDeliveredChecksGuard:
    """The delivered-capability guard on the recovery-DONE chokepoint.

    A journal row or a branch-on-main probe proves only that SOMETHING of this
    branch reached main — never that THIS task's declared capability survived
    to it. On any block the chokepoint returns ``None``, which every caller
    already reads as "no recovery, proceed with the phase", so the workflow
    goes on to actually deliver the task.
    """

    # --- row 1: hollow-done regression / FAILED -> ZERO side effects -------

    @pytest.mark.parametrize(('basis', 'kind'), _RECOVERY_ARMS)
    async def test_failed_block_withholds_recovery_with_zero_side_effects(
        self, tmp_path: Path, basis: str, kind: str,
    ):
        """A withheld recovery must leave NOTHING behind: no stamp, no DONE
        phase, no metadata-files reconcile, and no ``_merge_recovery_basis``
        marker (that marker is this chokepoint's own provenance record — a
        recovery that did not happen must not claim one)."""
        f = _arm_recovery_fixture(tmp_path)
        phases = _spy_phases(f)
        guard = AsyncMock(return_value=DeliveredChecksBlock(
            reason='failed', main_sha='m' * 40, failed_check=_WF_DC_CHECK,
        ))

        with patch(_WF_GATE_TARGET, guard):
            outcome = await f.wf._finalise_recovery_done(
                basis=basis, sha='advancedsha123', kind=kind, note='n',
            )

        assert outcome is None
        f.mark_done.assert_not_awaited()
        assert WorkflowState.DONE not in phases
        assert f.wf.state != WorkflowState.DONE
        cast(AsyncMock, f.wf._reconcile_metadata_files_for_done).assert_not_awaited()
        assert f.wf._merge_recovery_basis is None

    # --- row 2: all_delivered -> byte-identical recovery -------------------

    @pytest.mark.parametrize(('basis', 'kind'), _RECOVERY_ARMS)
    async def test_all_delivered_finalises_exactly_as_today(
        self, tmp_path: Path, basis: str, kind: str,
    ):
        """Capability verifiably on main -> today's exact stamp, basis marker,
        DONE phase and metadata-files reconcile, in today's order."""
        f = _arm_recovery_fixture(tmp_path)
        guard = AsyncMock(return_value=None)

        with patch(_WF_GATE_TARGET, guard):
            outcome = await f.wf._finalise_recovery_done(
                basis=basis, sha='advancedsha123', kind=kind, note='n',
                files=['a.py'],
            )

        assert outcome == WorkflowOutcome.DONE
        assert f.wf._merge_recovery_basis == basis
        assert f.wf.state == WorkflowState.DONE
        f.mark_done.assert_awaited_once_with(
            f.wf.task_id, kind=kind, sha='advancedsha123', note='n',
        )
        cast(
            AsyncMock, f.wf._reconcile_metadata_files_for_done,
        ).assert_awaited_once_with(override_files=['a.py'])

    # --- row 3: no delivered_checks -> unchanged, but still DELEGATED ------

    @pytest.mark.parametrize(('basis', 'kind'), _RECOVERY_ARMS)
    async def test_check_less_task_delegates_and_finalises(
        self, tmp_path: Path, basis: str, kind: str,
    ):
        """A check-less task must not gain a new requirement. The workflow
        DELEGATES unconditionally (forwarding the task's metadata) rather than
        short-circuiting itself — inertness lives in the helper alone."""
        f = _arm_recovery_fixture(tmp_path, metadata={})
        guard = AsyncMock(return_value=None)

        with patch(_WF_GATE_TARGET, guard):
            outcome = await f.wf._finalise_recovery_done(
                basis=basis, sha='advancedsha123', kind=kind, note='n',
            )

        assert outcome == WorkflowOutcome.DONE
        f.mark_done.assert_awaited_once()
        guard.assert_awaited_once()
        assert guard.await_args is not None
        assert guard.await_args.args[1] == f.wf.task['metadata']

    # --- rows 4 & 5: fail-safe blocks are handled UNIFORMLY with FAILED ----

    @pytest.mark.parametrize(('basis', 'kind'), _RECOVERY_ARMS)
    @pytest.mark.parametrize('reason', ['errored', 'main_sha_unresolved'])
    async def test_fail_safe_blocks_also_withhold_the_recovery(
        self, tmp_path: Path, basis: str, kind: str, reason: str,
    ):
        """ERRORED / main_sha_unresolved take the SAME path as FAILED: no
        claim either way means no recovery-DONE. Fail-safe here is cheap —
        the workflow simply proceeds with the phase and re-delivers."""
        f = _arm_recovery_fixture(tmp_path)
        guard = AsyncMock(return_value=DeliveredChecksBlock(
            reason=reason,  # type: ignore[arg-type]
        ))

        with patch(_WF_GATE_TARGET, guard):
            outcome = await f.wf._finalise_recovery_done(
                basis=basis, sha='advancedsha123', kind=kind, note='n',
            )

        assert outcome is None
        f.mark_done.assert_not_awaited()
        assert f.wf.state != WorkflowState.DONE
        assert f.wf._merge_recovery_basis is None

    # --- row 6: kill switch is FORWARDED, never re-implemented -------------

    @pytest.mark.parametrize(('basis', 'kind'), _RECOVERY_ARMS)
    async def test_kill_switch_is_forwarded_not_short_circuited(
        self, tmp_path: Path, basis: str, kind: str,
    ):
        f = _arm_recovery_fixture(tmp_path, enabled=False)
        guard = AsyncMock(return_value=None)

        with patch(_WF_GATE_TARGET, guard):
            outcome = await f.wf._finalise_recovery_done(
                basis=basis, sha='advancedsha123', kind=kind, note='n',
            )

        assert outcome == WorkflowOutcome.DONE
        guard.assert_awaited_once()
        assert guard.await_args is not None
        assert guard.await_args.kwargs['enabled'] is False

    @pytest.mark.parametrize(('basis', 'kind'), _RECOVERY_ARMS)
    async def test_kill_switch_with_real_helper_finalises_as_today(
        self, tmp_path: Path, basis: str, kind: str,
    ):
        """End-to-end with the REAL helper: disabled -> byte-identical
        recovery, with zero check work."""
        f = _arm_recovery_fixture(tmp_path, enabled=False)

        outcome = await f.wf._finalise_recovery_done(
            basis=basis, sha='advancedsha123', kind=kind, note='n',
        )

        assert outcome == WorkflowOutcome.DONE
        f.mark_done.assert_awaited_once()
        cast(AsyncMock, f.wf.git_ops.get_main_sha).assert_not_awaited()

    # --- plumbing: the MAIN checkout, never the worktree -------------------

    @pytest.mark.parametrize(('basis', 'kind'), _RECOVERY_ARMS)
    async def test_forwards_main_checkout_config_knobs(
        self, tmp_path: Path, basis: str, kind: str,
    ):
        """``project_root`` must be the MAIN checkout, NOT ``self.worktree``.

        The grep kind evaluates the COMMITTED tree at ``ref=main_sha``, so a
        worktree path here would silently audit the wrong tree — and, worse,
        would usually still pass, making the guard look armed while proving
        nothing about main.
        """
        f = _arm_recovery_fixture(tmp_path)
        guard = AsyncMock(return_value=None)

        with patch(_WF_GATE_TARGET, guard):
            await f.wf._finalise_recovery_done(
                basis=basis, sha='advancedsha123', kind=kind, note='n',
            )

        assert guard.await_args is not None
        kwargs = guard.await_args.kwargs
        assert kwargs['project_root'] == str(f.wf.config.project_root)
        assert kwargs['project_root'] != str(f.wf.worktree)
        assert kwargs['check_timeout_secs'] == 7.5
        assert kwargs['enabled'] is True
        assert guard.await_args.args[0] == f.wf.task_id
        assert basis in kwargs['site']
        # The git_ops HANDLE, not merely a truthy sentinel: without it the
        # guard cannot resolve a main SHA and every check-carrying task
        # collapses to `main_sha_unresolved`, i.e. a fleet-wide silent
        # disarm-into-wedge in which EVERY recovery stamp is withheld
        # forever. Pinning identity here is what makes
        # `git_ops=self.git_ops` -> `git_ops=None` a failing mutation.
        assert kwargs['git_ops'] is f.wf.git_ops

    # --- ordering: the MP-2 structural guard still fires FIRST -------------

    @pytest.mark.parametrize('bad', [
        pytest.param({'basis': 'nonsense', 'sha': 'advancedsha123'}, id='bad-basis'),
        pytest.param({'basis': 'journal', 'sha': ''}, id='falsy-sha'),
    ])
    async def test_mp2_assertion_precedes_the_guard(
        self, tmp_path: Path, bad: dict,
    ):
        """A malformed caller must fail LOUDLY, never be masked by a
        capability withholding — the MP-2 AssertionError (no recovery-DONE
        without a provenance basis) still raises before any check work."""
        f = _arm_recovery_fixture(tmp_path)
        guard = AsyncMock(return_value=None)

        with patch(_WF_GATE_TARGET, guard), pytest.raises(AssertionError):
            await f.wf._finalise_recovery_done(kind='merged', note='n', **bad)

        guard.assert_not_awaited()

    # --- integration: the REAL callers honour the widened return type ------

    async def test_recover_before_execute_journal_arm_honours_a_block(
        self, tmp_path: Path,
    ):
        """Guard 2's journal arm, driven end-to-end: a blocked chokepoint must
        surface as ``None`` (no recovery — proceed with EXECUTE), not as a
        crash or a phantom DONE."""
        f = _arm_recovery_fixture(tmp_path)
        _bind_landed_row(tmp_path, task_id=f.wf.task_id, advanced_sha='advancedsha123')
        guard = AsyncMock(return_value=DeliveredChecksBlock(
            reason='failed', main_sha='m' * 40, failed_check=_WF_DC_CHECK,
        ))

        with patch(_WF_GATE_TARGET, guard):
            outcome = await f.wf._recover_before_execute()

        assert outcome is None
        f.mark_done.assert_not_awaited()
        assert f.wf._merge_recovery_basis is None

    async def test_recover_if_already_merged_fallback_arm_honours_a_block(
        self, tmp_path: Path,
    ):
        """Guard 1's fallback (``found_on_main``) arm, driven end-to-end —
        the arm whose stamps the acceptance predicate actually audits."""
        f = _arm_recovery_fixture(tmp_path)
        f.wf._check_branch_on_main = AsyncMock(  # type: ignore[method-assign]
            return_value=('wthead123', 'mainsha123'),
        )
        f.wf._has_prior_implementation = MagicMock(  # type: ignore[method-assign]
            return_value=_PriorImplStatus(has_work=True, entries=[], base_commit=None),
        )
        guard = AsyncMock(return_value=DeliveredChecksBlock(
            reason='failed', main_sha='m' * 40, failed_check=_WF_DC_CHECK,
        ))

        with patch(_WF_GATE_TARGET, guard):
            outcome = await f.wf._recover_if_already_merged()

        assert outcome is None
        f.mark_done.assert_not_awaited()
        assert f.wf._merge_recovery_basis is None
