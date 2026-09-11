"""The merge lane's façade.

PRD ``plans/merge-lane-quality-prd.md`` task ζ1. This package exports
exactly the lane's public surface (PRD § Contract → *Public surface of the
façade*): the worker as ``MergeLane``, the request entry points, the value
types, the reason constants, and the ports.

Until task ζ2 moves the lane in, everything but the ports and value types
still lives in ``orchestrator.merge_queue`` and ``orchestrator.merge_gates``,
and ``orchestrator.merge_queue`` imports this package's ports at module
level. So each export is resolved on first access, once both modules have
finished importing, rather than bound here while ``orchestrator.merge_queue``
is still half-built. ζ2 reverses the direction and replaces the table with
plain imports.
"""
from __future__ import annotations

from typing import Any

import orchestrator.merge_gates as _gates
import orchestrator.merge_queue as _lane
from orchestrator.merge_lane import ports as _ports
from orchestrator.merge_lane import types as _types


class _ExportTable(dict[str, tuple[Any, str]]):
    """Export name → (module, attribute).

    A name outside the façade is an ``AttributeError``, which is what module
    ``__getattr__`` must raise for ``hasattr`` and ``from … import`` to
    behave.
    """

    def __missing__(self, name: str) -> tuple[Any, str]:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


_EXPORTS = _ExportTable({
    'MergeLane': (_lane, 'SpeculativeMergeWorker'),
    'SpeculativeMergeWorker': (_lane, 'SpeculativeMergeWorker'),
    'coalesce_or_enqueue_merge_request': (_lane, 'coalesce_or_enqueue_merge_request'),
    'enqueue_merge_request': (_lane, 'enqueue_merge_request'),
    'retire_cancelled_merge_request': (_lane, 'retire_cancelled_merge_request'),
    'resolve_dispatch_time_merge_base': (_lane, '_resolve_dispatch_time_merge_base'),
    '_resolve_dispatch_time_merge_base': (_lane, '_resolve_dispatch_time_merge_base'),
    'patch_content_contained': (_lane, 'patch_content_contained'),
    'MergeRequest': (_lane, 'MergeRequest'),
    'GroupMergeRequest': (_lane, 'GroupMergeRequest'),
    'MergeOutcome': (_lane, 'MergeOutcome'),
    'QueuedBranch': (_lane, 'QueuedBranch'),
    'WaiterRecord': (_lane, 'WaiterRecord'),
    'InFlightMergeRegistry': (_lane, 'InFlightMergeRegistry'),
    'ABANDONED_REASON_PREFIX': (_lane, 'ABANDONED_REASON_PREFIX'),
    'ALREADY_LANDED_REASON_PREFIX': (_gates, 'ALREADY_LANDED_REASON_PREFIX'),
    'CROSS_REPO_DELIVERABLE_REASON_PREFIX': (_gates, 'CROSS_REPO_DELIVERABLE_REASON_PREFIX'),
    'DROPPED_PLAN_TARGETS_REASON_PREFIX': (_lane, 'DROPPED_PLAN_TARGETS_REASON_PREFIX'),
    'MAIN_HEALTH_RED_REASON_PREFIX': (_lane, 'MAIN_HEALTH_RED_REASON_PREFIX'),
    'MERGE_WORKER_SHUTDOWN_REASON': (_lane, 'MERGE_WORKER_SHUTDOWN_REASON'),
    'NEEDS_REBASE_REASON_PREFIX': (_lane, 'NEEDS_REBASE_REASON_PREFIX'),
    'PLAN_FILES_NOT_TOUCHED_REASON_PREFIX': (_lane, 'PLAN_FILES_NOT_TOUCHED_REASON_PREFIX'),
    'POST_MERGE_EQUIVALENCE_FAILED_REASON_PREFIX': (
        _lane, 'POST_MERGE_EQUIVALENCE_FAILED_REASON_PREFIX',
    ),
    'POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX': (_lane, 'POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX'),
    'TRAIN_INCOMPLETE_REASON_PREFIX': (_lane, 'TRAIN_INCOMPLETE_REASON_PREFIX'),
    'TRAIN_PARTIAL_FLIP_REASON_PREFIX': (_lane, 'TRAIN_PARTIAL_FLIP_REASON_PREFIX'),
    'TRAIN_REBASE_CONFLICT_REASON_PREFIX': (_lane, 'TRAIN_REBASE_CONFLICT_REASON_PREFIX'),
    'TRAIN_VERIFY_FAILED_REASON_PREFIX': (_lane, 'TRAIN_VERIFY_FAILED_REASON_PREFIX'),
    'TRANSIENT_INFRA_REASON_PREFIX': (_lane, 'TRANSIENT_INFRA_REASON_PREFIX'),
    'TRIVIAL_PASS_MAIN_RED_REASON_PREFIX': (_lane, 'TRIVIAL_PASS_MAIN_RED_REASON_PREFIX'),
    'WORKTREE_MISSING_REASON_PREFIX': (_lane, 'WORKTREE_MISSING_REASON_PREFIX'),
    'VerifyPort': (_ports, 'VerifyPort'),
    'ClockPort': (_ports, 'ClockPort'),
    'EscalationPort': (_ports, 'EscalationPort'),
    'ProductionVerifier': (_ports, 'ProductionVerifier'),
    'ProductionClock': (_ports, 'ProductionClock'),
    'ProductionEscalations': (_ports, 'ProductionEscalations'),
    'DiscardingEscalations': (_ports, 'DiscardingEscalations'),
    'DiskGuardOutcome': (_types, 'DiskGuardOutcome'),
    'EscalationRecord': (_types, 'EscalationRecord'),
})

__all__ = sorted(_EXPORTS)  # pyright: ignore[reportUnsupportedDunderAll]


def __getattr__(name: str) -> Any:
    module, attribute = _EXPORTS[name]
    return getattr(module, attribute)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))
