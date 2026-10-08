"""The merge lane's façade.

PRD ``plans/merge-lane-quality-prd.md`` tasks ζ1 and ζ2. This package
exports exactly the lane's public surface (PRD § Contract → *Public surface
of the façade*): the worker as ``MergeLane``, the request entry points, the
value types, the reason constants, and the ports. Every submodule is private
to the package.

Each export is resolved through the table below on first access, by
importing its submodule then, so importing any one submodule never drags in
the worker; and a table rather than ``from .worker import …`` lines, because
an import-and-never-reference binding is exactly what the ratchet's
``reexport_names`` measure counts (``scripts/merge_lane_metrics.py``).
"""
from __future__ import annotations

import importlib
from typing import Any

_WORKER = 'orchestrator.merge_lane.worker'
_GATES = 'orchestrator.merge_lane.gates'
_LANDING_EVIDENCE = 'orchestrator.merge_lane.landing_evidence'
_PORTS = 'orchestrator.merge_lane.ports'
_TYPES = 'orchestrator.merge_lane.types'


class _ExportTable(dict[str, tuple[str, str]]):
    """Export name → (submodule, attribute).

    A name outside the façade is an ``AttributeError``, which is what module
    ``__getattr__`` must raise for ``hasattr`` and ``from … import`` to
    behave.
    """

    def __missing__(self, name: str) -> tuple[str, str]:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


_EXPORTS = _ExportTable({
    'MergeLane': (_WORKER, 'SpeculativeMergeWorker'),
    'SpeculativeMergeWorker': (_WORKER, 'SpeculativeMergeWorker'),
    'coalesce_or_enqueue_merge_request': (_WORKER, 'coalesce_or_enqueue_merge_request'),
    'enqueue_merge_request': (_WORKER, 'enqueue_merge_request'),
    'retire_cancelled_merge_request': (_WORKER, 'retire_cancelled_merge_request'),
    'resolve_dispatch_time_merge_base': (_WORKER, '_resolve_dispatch_time_merge_base'),
    '_resolve_dispatch_time_merge_base': (_WORKER, '_resolve_dispatch_time_merge_base'),
    'patch_content_contained': (_LANDING_EVIDENCE, 'patch_content_contained'),
    'MergeRequest': (_TYPES, 'MergeRequest'),
    'GroupMergeRequest': (_TYPES, 'GroupMergeRequest'),
    'MergeOutcome': (_TYPES, 'MergeOutcome'),
    'QueuedBranch': (_TYPES, 'QueuedBranch'),
    'WaiterRecord': (_TYPES, 'WaiterRecord'),
    'InFlightMergeRegistry': (_TYPES, 'InFlightMergeRegistry'),
    'ABANDONED_REASON_PREFIX': (_WORKER, 'ABANDONED_REASON_PREFIX'),
    'ALREADY_LANDED_REASON_PREFIX': (_GATES, 'ALREADY_LANDED_REASON_PREFIX'),
    'CROSS_REPO_DELIVERABLE_REASON_PREFIX': (_GATES, 'CROSS_REPO_DELIVERABLE_REASON_PREFIX'),
    'DROPPED_PLAN_TARGETS_REASON_PREFIX': (_GATES, 'DROPPED_PLAN_TARGETS_REASON_PREFIX'),
    'MAIN_HEALTH_RED_REASON_PREFIX': (_WORKER, 'MAIN_HEALTH_RED_REASON_PREFIX'),
    'MERGE_WORKER_SHUTDOWN_REASON': (_WORKER, 'MERGE_WORKER_SHUTDOWN_REASON'),
    'NEEDS_REBASE_REASON_PREFIX': (_WORKER, 'NEEDS_REBASE_REASON_PREFIX'),
    'PLAN_FILES_NOT_TOUCHED_REASON_PREFIX': (_GATES, 'PLAN_FILES_NOT_TOUCHED_REASON_PREFIX'),
    'POST_MERGE_EQUIVALENCE_FAILED_REASON_PREFIX': (
        _GATES, 'POST_MERGE_EQUIVALENCE_FAILED_REASON_PREFIX',
    ),
    'POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX': (_GATES, 'POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX'),
    'TRAIN_INCOMPLETE_REASON_PREFIX': (_WORKER, 'TRAIN_INCOMPLETE_REASON_PREFIX'),
    'TRAIN_PARTIAL_FLIP_REASON_PREFIX': (_WORKER, 'TRAIN_PARTIAL_FLIP_REASON_PREFIX'),
    'TRAIN_REBASE_CONFLICT_REASON_PREFIX': (_WORKER, 'TRAIN_REBASE_CONFLICT_REASON_PREFIX'),
    'TRAIN_VERIFY_FAILED_REASON_PREFIX': (_WORKER, 'TRAIN_VERIFY_FAILED_REASON_PREFIX'),
    'TRANSIENT_INFRA_REASON_PREFIX': (_WORKER, 'TRANSIENT_INFRA_REASON_PREFIX'),
    'TRIVIAL_PASS_MAIN_RED_REASON_PREFIX': (_WORKER, 'TRIVIAL_PASS_MAIN_RED_REASON_PREFIX'),
    'WORKTREE_MISSING_REASON_PREFIX': (_WORKER, 'WORKTREE_MISSING_REASON_PREFIX'),
    'VerifyPort': (_PORTS, 'VerifyPort'),
    'ClockPort': (_PORTS, 'ClockPort'),
    'EscalationPort': (_PORTS, 'EscalationPort'),
    'ProductionVerifier': (_PORTS, 'ProductionVerifier'),
    'ProductionClock': (_PORTS, 'ProductionClock'),
    'ProductionEscalations': (_PORTS, 'ProductionEscalations'),
    'DiscardingEscalations': (_PORTS, 'DiscardingEscalations'),
    'DiskGuardOutcome': (_TYPES, 'DiskGuardOutcome'),
    'EscalationRecord': (_TYPES, 'EscalationRecord'),
})

__all__ = sorted(_EXPORTS)  # pyright: ignore[reportUnsupportedDunderAll]


def __getattr__(name: str) -> Any:
    module, attribute = _EXPORTS[name]
    return getattr(importlib.import_module(module), attribute)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))
