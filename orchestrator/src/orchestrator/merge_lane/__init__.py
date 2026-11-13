"""The merge lane's façade.

PRD ``plans/merge-lane-quality-prd.md`` tasks ζ1 and ζ2. This package
exports exactly the lane's public surface (PRD § Contract → *Public surface
of the façade*): the worker as ``MergeLane``, the request entry points, the
value types, the reason constants, and the ports. Every submodule is private
to the package.

Each export is resolved through the table below on first access, by
importing its submodule then, so importing any one submodule never drags in
the worker. Pyright cannot see through a module ``__getattr__``, so the
``TYPE_CHECKING`` block declares each export's static type, and ``__all__``
is a literal because ruff reads only a literal as marking those imports
re-exports. The table, the block and ``__all__`` are pinned equal by
``orchestrator/tests/test_merge_lane_package.py``.
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from orchestrator.merge_lane.gates import (
        ALREADY_LANDED_REASON_PREFIX,
        CROSS_REPO_DELIVERABLE_REASON_PREFIX,
        DROPPED_PLAN_TARGETS_REASON_PREFIX,
        PLAN_FILES_NOT_TOUCHED_REASON_PREFIX,
        POST_MERGE_EQUIVALENCE_FAILED_REASON_PREFIX,
        POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX,
    )
    from orchestrator.merge_lane.intake import (
        coalesce_or_enqueue_merge_request,
        enqueue_merge_request,
    )
    from orchestrator.merge_lane.intake_buffers import (
        MERGE_WORKER_SHUTDOWN_REASON,
    )
    from orchestrator.merge_lane.landing_evidence import (
        patch_content_contained,
    )
    from orchestrator.merge_lane.member_redrive import (
        MemberRedriveOutcome,
    )
    from orchestrator.merge_lane.ports import (
        ClockPort,
        DiscardingEscalations,
        EscalationPort,
        ProductionClock,
        ProductionEscalations,
        ProductionVerifier,
        VerifyPort,
    )
    from orchestrator.merge_lane.types import (
        DiskGuardOutcome,
        EscalationRecord,
        GroupMergeRequest,
        InFlightMergeRegistry,
        MergeCancelRefusal,
        MergeCancelVerdict,
        MergeOutcome,
        MergeRequest,
        OperatorMergeCancel,
        QueuedBranch,
        WaiterRecord,
    )
    from orchestrator.merge_lane.worker import (
        ABANDONED_REASON_PREFIX,
        MAIN_HEALTH_RED_REASON_PREFIX,
        NEEDS_REBASE_REASON_PREFIX,
        TRAIN_INCOMPLETE_REASON_PREFIX,
        TRAIN_PARTIAL_FLIP_REASON_PREFIX,
        TRAIN_REBASE_CONFLICT_REASON_PREFIX,
        TRAIN_VERIFY_FAILED_REASON_PREFIX,
        TRANSIENT_INFRA_REASON_PREFIX,
        TRIVIAL_PASS_MAIN_RED_REASON_PREFIX,
        WORKTREE_MISSING_REASON_PREFIX,
        resolve_dispatch_time_merge_base,
        retire_cancelled_merge_request,
    )
    from orchestrator.merge_lane.worker import (
        SpeculativeMergeWorker as MergeLane,
    )

_WORKER = 'orchestrator.merge_lane.worker'
_GATES = 'orchestrator.merge_lane.gates'
_INTAKE = 'orchestrator.merge_lane.intake'
_INTAKE_BUFFERS = 'orchestrator.merge_lane.intake_buffers'
_LANDING_EVIDENCE = 'orchestrator.merge_lane.landing_evidence'
_MEMBER_REDRIVE = 'orchestrator.merge_lane.member_redrive'
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
    'coalesce_or_enqueue_merge_request': (_INTAKE, 'coalesce_or_enqueue_merge_request'),
    'enqueue_merge_request': (_INTAKE, 'enqueue_merge_request'),
    'retire_cancelled_merge_request': (_WORKER, 'retire_cancelled_merge_request'),
    'resolve_dispatch_time_merge_base': (_WORKER, 'resolve_dispatch_time_merge_base'),
    'patch_content_contained': (_LANDING_EVIDENCE, 'patch_content_contained'),
    'MemberRedriveOutcome': (_MEMBER_REDRIVE, 'MemberRedriveOutcome'),
    'MergeRequest': (_TYPES, 'MergeRequest'),
    'GroupMergeRequest': (_TYPES, 'GroupMergeRequest'),
    'MergeOutcome': (_TYPES, 'MergeOutcome'),
    'MergeCancelRefusal': (_TYPES, 'MergeCancelRefusal'),
    'MergeCancelVerdict': (_TYPES, 'MergeCancelVerdict'),
    'OperatorMergeCancel': (_TYPES, 'OperatorMergeCancel'),
    'QueuedBranch': (_TYPES, 'QueuedBranch'),
    'WaiterRecord': (_TYPES, 'WaiterRecord'),
    'InFlightMergeRegistry': (_TYPES, 'InFlightMergeRegistry'),
    'ABANDONED_REASON_PREFIX': (_WORKER, 'ABANDONED_REASON_PREFIX'),
    'ALREADY_LANDED_REASON_PREFIX': (_GATES, 'ALREADY_LANDED_REASON_PREFIX'),
    'CROSS_REPO_DELIVERABLE_REASON_PREFIX': (_GATES, 'CROSS_REPO_DELIVERABLE_REASON_PREFIX'),
    'DROPPED_PLAN_TARGETS_REASON_PREFIX': (_GATES, 'DROPPED_PLAN_TARGETS_REASON_PREFIX'),
    'MAIN_HEALTH_RED_REASON_PREFIX': (_WORKER, 'MAIN_HEALTH_RED_REASON_PREFIX'),
    'MERGE_WORKER_SHUTDOWN_REASON': (_INTAKE_BUFFERS, 'MERGE_WORKER_SHUTDOWN_REASON'),
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

__all__ = [
    'ABANDONED_REASON_PREFIX',
    'ALREADY_LANDED_REASON_PREFIX',
    'CROSS_REPO_DELIVERABLE_REASON_PREFIX',
    'ClockPort',
    'DROPPED_PLAN_TARGETS_REASON_PREFIX',
    'DiscardingEscalations',
    'DiskGuardOutcome',
    'EscalationPort',
    'EscalationRecord',
    'GroupMergeRequest',
    'InFlightMergeRegistry',
    'MAIN_HEALTH_RED_REASON_PREFIX',
    'MERGE_WORKER_SHUTDOWN_REASON',
    'MemberRedriveOutcome',
    'MergeCancelRefusal',
    'MergeCancelVerdict',
    'MergeLane',
    'MergeOutcome',
    'MergeRequest',
    'NEEDS_REBASE_REASON_PREFIX',
    'OperatorMergeCancel',
    'PLAN_FILES_NOT_TOUCHED_REASON_PREFIX',
    'POST_MERGE_EQUIVALENCE_FAILED_REASON_PREFIX',
    'POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX',
    'ProductionClock',
    'ProductionEscalations',
    'ProductionVerifier',
    'QueuedBranch',
    'TRAIN_INCOMPLETE_REASON_PREFIX',
    'TRAIN_PARTIAL_FLIP_REASON_PREFIX',
    'TRAIN_REBASE_CONFLICT_REASON_PREFIX',
    'TRAIN_VERIFY_FAILED_REASON_PREFIX',
    'TRANSIENT_INFRA_REASON_PREFIX',
    'TRIVIAL_PASS_MAIN_RED_REASON_PREFIX',
    'VerifyPort',
    'WORKTREE_MISSING_REASON_PREFIX',
    'WaiterRecord',
    'coalesce_or_enqueue_merge_request',
    'enqueue_merge_request',
    'patch_content_contained',
    'resolve_dispatch_time_merge_base',
    'retire_cancelled_merge_request',
]


def __getattr__(name: str) -> object:
    module, attribute = _EXPORTS[name]
    return getattr(importlib.import_module(module), attribute)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))
