"""Tests for orchestrator.merge_gates: extracted post-merge gates, finalize,
and reason-prefix constants (MQ-refactor task β).

These tests encode the behavior-preserving contracts of the module split,
mirroring task α's test_merge_types.py:

1. Module-existence — ``orchestrator.merge_gates`` exists and exports the
   full closure of moved symbols (reason prefixes, types, sentinel,
   functions).
2. Logger-name — the module logs under the ``orchestrator.merge_queue``
   logger name (not ``orchestrator.merge_gates``) so existing ``caplog``
   assertions filtered to the merge_queue logger keep capturing the moved
   gates' fail-open/fail-closed warnings.
3. Reach-back / string-path monkeypatch routing — the existing test suite
   monkeypatches merge-gate dependencies by STRING PATH
   ``orchestrator.merge_queue.<name>``.  A moved function must resolve a
   monkeypatched-or-staying sibling via a function-local deferred import so
   those patches stay effective even though the function body now lives in
   this module.  Each test below patches BOTH namespaces with CONTRASTING
   return values — the merge_gates-local (naive) patch would steer the
   outcome one way, the merge_queue (reach-back target) patch the other —
   so the assertion is unambiguous about which one governed.
4. Shim re-export identity + reason-prefix byte-identity (added in a later
   step, once merge_queue.py's shim swap lands).
"""

# NOTE: byte-identical reason-prefix literal checks and shim re-export
# identity checks are added in a later step (once merge_queue.py's shim
# swap lands) — see the module docstring above.

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def test_merge_gates_exports_moved_public_symbols() -> None:
    from orchestrator.merge_gates import (
        DROPPED_PLAN_TARGETS_REASON_PREFIX,
        PLAN_FILES_NOT_TOUCHED_REASON_PREFIX,
        POST_MERGE_EQUIVALENCE_FAILED_REASON_PREFIX,
        POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX,
        DropGuardResult,
        PlanFilesTouchedResult,
        PostMergePyrightResult,
        _check_plan_files_touched_in_branch,
        _check_plan_targets_in_tree,
        _check_post_merge_equivalence,
        _check_post_merge_pyright,
        _commit_is_linear,
        _finalize_advanced_merge,
        _GenerationChainContext,
        _map_advance_failure,
        _normalize_plan_path,
        _OVERLAP_GIT_ERROR_SENTINEL,
        _rebase_delta_touched_overlap,
        _resolve_second_parent,
        _reverify_rebased_tree,
    )

    for name, obj in {
        'DROPPED_PLAN_TARGETS_REASON_PREFIX': DROPPED_PLAN_TARGETS_REASON_PREFIX,
        'PLAN_FILES_NOT_TOUCHED_REASON_PREFIX': PLAN_FILES_NOT_TOUCHED_REASON_PREFIX,
        'POST_MERGE_EQUIVALENCE_FAILED_REASON_PREFIX': POST_MERGE_EQUIVALENCE_FAILED_REASON_PREFIX,
        'POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX': POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX,
        'DropGuardResult': DropGuardResult,
        'PlanFilesTouchedResult': PlanFilesTouchedResult,
        'PostMergePyrightResult': PostMergePyrightResult,
        '_GenerationChainContext': _GenerationChainContext,
        '_OVERLAP_GIT_ERROR_SENTINEL': _OVERLAP_GIT_ERROR_SENTINEL,
        '_check_plan_targets_in_tree': _check_plan_targets_in_tree,
        '_normalize_plan_path': _normalize_plan_path,
        '_check_plan_files_touched_in_branch': _check_plan_files_touched_in_branch,
        '_check_post_merge_equivalence': _check_post_merge_equivalence,
        '_rebase_delta_touched_overlap': _rebase_delta_touched_overlap,
        '_reverify_rebased_tree': _reverify_rebased_tree,
        '_check_post_merge_pyright': _check_post_merge_pyright,
        '_resolve_second_parent': _resolve_second_parent,
        '_commit_is_linear': _commit_is_linear,
        '_finalize_advanced_merge': _finalize_advanced_merge,
        '_map_advance_failure': _map_advance_failure,
    }.items():
        assert obj is not None, f'{name} must not be None'


def test_merge_gates_logger_name_is_merge_queue() -> None:
    """merge_gates emits under the 'orchestrator.merge_queue' logger name.

    RED (pre-module): ``orchestrator.merge_gates`` does not exist yet.

    Required so existing ``caplog.at_level(..., logger='orchestrator.merge_queue')``
    assertions in test_merge_queue_equivalence.py / test_merge_queue.py keep
    capturing the moved gates' WARNING-level fail-open/fail-closed messages
    after the functions relocate to this module.
    """
    import orchestrator.merge_gates as merge_gates

    assert merge_gates.logger.name == 'orchestrator.merge_queue'


@pytest.mark.asyncio
class TestReachBackRouting:
    """Reach-back / string-path monkeypatch routing contract.

    Each test patches the SAME logical dependency in both namespaces with
    CONTRASTING values: the merge_gates-local (naive bare-global) patch
    steers the outcome one way, the merge_queue (reach-back target) patch
    the other.  Asserting on the merge_queue-steered outcome proves the
    call went through the deferred import rather than the co-located
    merge_gates sibling.
    """

    async def test_reverify_rebased_tree_reachback_to_rebase_delta_overlap(self) -> None:
        """(a) _reverify_rebased_tree must resolve _rebase_delta_touched_overlap
        via orchestrator.merge_queue, not the co-located merge_gates copy."""
        from orchestrator.merge_gates import _reverify_rebased_tree

        git_ops = MagicMock()
        req = MagicMock()
        req.task_id = 'task-rvrt-reachback'
        req.worktree = MagicMock()
        merge_wt = MagicMock()
        sentinel_outcome = MagicMock(name='sentinel-verify-outcome')

        with (
            # Naive-resolution target: disjoint (empty) → would return None
            # WITHOUT ever calling _run_post_merge_verify.
            patch(
                'orchestrator.merge_gates._rebase_delta_touched_overlap',
                AsyncMock(return_value=[]),
            ),
            # Reach-back target: overlapping → must delegate to
            # _run_post_merge_verify (itself already reach-back, per step-2).
            patch(
                'orchestrator.merge_queue._rebase_delta_touched_overlap',
                AsyncMock(return_value=['overlap.py']),
            ),
            patch(
                'orchestrator.merge_queue._run_post_merge_verify',
                AsyncMock(return_value=sentinel_outcome),
            ),
        ):
            result = await _reverify_rebased_tree(
                git_ops, req, merge_wt,
                rebased_from='from-sha',
                rebased_onto='onto-sha',
                timeouts={},
                enospc_retries={},
                max_timeouts=3,
                max_enospc=1,
            )

        assert result is sentinel_outcome, (
            f'expected the orchestrator.merge_queue-patched overlap to govern '
            f'the re-verify decision and return its sentinel outcome, got {result!r}'
        )

    async def test_finalize_advanced_merge_reachback_to_equivalence_and_pyright(self) -> None:
        """(b) _finalize_advanced_merge must resolve _check_post_merge_equivalence
        and _check_post_merge_pyright via orchestrator.merge_queue, not the
        co-located merge_gates copies."""
        from orchestrator.merge_gates import _finalize_advanced_merge

        git_ops = MagicMock()
        git_ops.push_main = AsyncMock(return_value='pushed')
        git_ops.cleanup_merge_worktree = AsyncMock()
        git_ops._last_advanced_sha = 'abc123def'
        req = MagicMock()
        req.task_id = 'task-finalize-reachback'
        req.branch = 'br-finalize-reachback'
        req.worktree = MagicMock()
        req.config = MagicMock()
        req.module_configs = []
        cas_retries = {req.task_id: 1}
        timeouts = {req.task_id: 1}
        enospc_retries = {req.task_id: 1}

        naive_broken_pyright = MagicMock(broken=True, failing_subprojects=['naive-pkg'], detail='naive-detail')
        reachback_clean_pyright = MagicMock(broken=False, failing_subprojects=[], detail='')

        with (
            # Naive-resolution targets: equivalence diverged + pyright broken →
            # would return 'blocked' before ever reaching push_main.
            patch(
                'orchestrator.merge_gates._check_post_merge_equivalence',
                AsyncMock(return_value=['naive-diverged.py']),
            ),
            patch(
                'orchestrator.merge_gates._check_post_merge_pyright',
                AsyncMock(return_value=naive_broken_pyright),
            ),
            # Reach-back targets: equivalence clean + pyright clean → must
            # reach the 'done' path and call push_main.
            patch(
                'orchestrator.merge_queue._check_post_merge_equivalence',
                AsyncMock(return_value=[]),
            ),
            patch(
                'orchestrator.merge_queue._check_post_merge_pyright',
                AsyncMock(return_value=reachback_clean_pyright),
            ),
        ):
            outcome = await _finalize_advanced_merge(
                git_ops, req, None,
                merge_commit_fallback='fallback-sha',
                base_sha='base-sha',
                started_monotonic=0.0,
                cas_retries=cas_retries,
                timeouts=timeouts,
                enospc_retries=enospc_retries,
                merged_branch_tip='trusted-tip',
            )

        assert outcome.status == 'done', (
            f'expected the orchestrator.merge_queue-patched equivalence/pyright '
            f'results to govern the outcome (done), got {outcome.status}: {outcome.reason!r}'
        )
        git_ops.push_main.assert_awaited_once()

    async def test_check_post_merge_pyright_reachback_to_run_unscoped_typechecks(self) -> None:
        """(c) _check_post_merge_pyright must resolve _run_unscoped_typechecks
        via orchestrator.merge_queue (it has no merge_gates-local copy at all —
        this reach-back was added directly in step-2, not deferred)."""
        from orchestrator.config import ModuleConfig, OrchestratorConfig
        from orchestrator.merge_gates import PostMergePyrightResult, _check_post_merge_pyright

        git_ops = MagicMock()
        git_ops._create_merge_worktree = AsyncMock(return_value=('fake-merge-wt', None))
        git_ops.cleanup_merge_worktree = AsyncMock()
        module_configs = [ModuleConfig(prefix='pkg', type_check_command='pyright src/')]
        patched_result = PostMergePyrightResult(
            failing_subprojects=['pkg'], detail='patched-detail',
        )

        with patch(
            'orchestrator.merge_queue._run_unscoped_typechecks',
            AsyncMock(return_value=patched_result),
        ):
            result = await _check_post_merge_pyright(
                'deadbeef', git_ops, OrchestratorConfig(), module_configs,
                task_id='task-pyright-reachback',
            )

        assert result is patched_result, (
            f'expected the orchestrator.merge_queue-patched _run_unscoped_typechecks '
            f'result to be returned unchanged, got {result!r}'
        )
        git_ops.cleanup_merge_worktree.assert_awaited_once()
