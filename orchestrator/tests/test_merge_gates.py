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
3. Reach-back / string-path monkeypatch routing (added in a later step).
4. Shim re-export identity + reason-prefix byte-identity (added in a later
   step, once merge_queue.py's shim swap lands).
"""

# NOTE: byte-identical reason-prefix literal checks and shim re-export
# identity checks are added in a later step (once merge_queue.py's shim
# swap lands) — see the module docstring above.

from __future__ import annotations


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
