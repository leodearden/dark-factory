"""Tests for orchestrator.merge_types: extracted request/outcome/item/entry

types + registries (MQ-refactor task α).

These tests encode the two behavior-preserving contracts of the module
split:

1. Module-existence — ``orchestrator.merge_types`` exists and exports the
   full closure of moved public (and internal-but-referenced) symbols.
2. Shim identity — ``orchestrator.merge_queue`` re-exports the *same*
   objects (not copies) so every existing importer keeps working
   unchanged.
"""

from __future__ import annotations

from pathlib import Path

from _orch_helpers import make_placeholder_future


def test_merge_types_exports_moved_public_symbols() -> None:
    from orchestrator.merge_types import (
        GroupMergeRequest,
        InflightEntry,
        InflightVerifyResult,
        InFlightMergeRegistry,
        MainHealthAutoHealRegistry,
        MergeBounceRegistry,
        MergeDispatchResult,
        MergeOutcome,
        MergeRequest,
        SoloVerifyResult,
        SpeculativeItem,
        TerminalOutcomeRecord,
        TerminalOutcomeRetention,
        TrainCallbackFactory,
        TrainCallbacks,
        WaiterRecord,
    )

    for name, obj in {
        "GroupMergeRequest": GroupMergeRequest,
        "InflightEntry": InflightEntry,
        "InflightVerifyResult": InflightVerifyResult,
        "InFlightMergeRegistry": InFlightMergeRegistry,
        "MainHealthAutoHealRegistry": MainHealthAutoHealRegistry,
        "MergeBounceRegistry": MergeBounceRegistry,
        "MergeDispatchResult": MergeDispatchResult,
        "MergeOutcome": MergeOutcome,
        "MergeRequest": MergeRequest,
        "SoloVerifyResult": SoloVerifyResult,
        "SpeculativeItem": SpeculativeItem,
        "TerminalOutcomeRecord": TerminalOutcomeRecord,
        "TerminalOutcomeRetention": TerminalOutcomeRetention,
        "TrainCallbackFactory": TrainCallbackFactory,
        "TrainCallbacks": TrainCallbacks,
        "WaiterRecord": WaiterRecord,
    }.items():
        assert obj is not None, f"{name} must not be None"


def test_merge_queue_reexports_identical_objects() -> None:
    """merge_queue re-exports the SAME objects from merge_types (shim identity).

    Covers every moved name, including the private/alias ones that staying
    worker code in merge_queue.py still references by bare name
    (``_InFlightEntry``, ``_HostUnavailability``, ``MergeReadyPredicate``,
    ``_INFLIGHT_MERGE_ETA_ESTIMATE_SECS``).

    RED (pre-shim): merge_queue.py still defines its own independent copies
    of these types (the duplicate definitions left in place by the EXPAND
    step), so ``getattr(merge_queue, name) is getattr(merge_types, name)``
    fails for every name — two distinct objects that merely share a name.
    """
    import orchestrator.merge_queue as merge_queue
    import orchestrator.merge_types as merge_types

    moved_names = [
        "MainHealthAutoHealRegistry",
        "MergeBounceRegistry",
        "TerminalOutcomeRecord",
        "TerminalOutcomeRetention",
        "_InFlightEntry",
        "InFlightMergeRegistry",
        "MergeDispatchResult",
        "WaiterRecord",
        "MergeRequest",
        "GroupMergeRequest",
        "MergeOutcome",
        "SoloVerifyResult",
        "SpeculativeItem",
        "InflightEntry",
        "_HostUnavailability",
        "InflightVerifyResult",
        "TrainCallbacks",
        "TrainCallbackFactory",
        "MergeReadyPredicate",
        "_INFLIGHT_MERGE_ETA_ESTIMATE_SECS",
    ]

    for name in moved_names:
        mq_obj = getattr(merge_queue, name)
        mt_obj = getattr(merge_types, name)
        assert mq_obj is mt_obj, (
            f"{name}: orchestrator.merge_queue.{name} and "
            f"orchestrator.merge_types.{name} must be the identical object"
        )

    assert issubclass(merge_queue.GroupMergeRequest, merge_queue.MergeRequest)

    outcome = merge_queue.MergeOutcome(status='done')
    assert outcome.status == 'done'

    request = merge_queue.MergeRequest(
        task_id='t1',
        branch='591',
        worktree=Path('/tmp/wt'),
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=None,
        result=make_placeholder_future(),
    )
    assert request.task_id == 't1'
    assert request.branch == '591'
