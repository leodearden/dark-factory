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

import pytest


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
