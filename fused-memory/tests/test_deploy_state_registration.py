"""Fused-memory-process side of observable 3 (task 2239, PRD §5.2 DS-1..4).

Proves that importing fused_memory.middleware.task_interceptor — the
boot-time write-interception middleware, not shared.deploy_state directly —
is itself sufficient to register DeployState into shared.task_metadata's
per-process _SUBMODEL_REGISTRY. The orchestrator-side registration property
is covered by orchestrator/tests/test_deploy_state.py; this test proves the
fused-memory process (which cannot import `orchestrator`) picks up the same
registration via its own side-effect import.

Deliberately does NOT import shared.deploy_state at module level: the
registry/parse_metadata observations below are captured before this file
ever references shared.deploy_state itself, so a self-import can't mask a
missing side-effect import in task_interceptor.py.
"""

from __future__ import annotations

import shared.task_metadata as task_metadata_module

import fused_memory.middleware.task_interceptor  # noqa: F401


def test_task_interceptor_import_registers_deploy_state_with_zero_warnings() -> None:
    registered_before_reference_import = task_metadata_module._SUBMODEL_REGISTRY.get(
        'deploy_state'
    )
    _model, warnings = task_metadata_module.parse_metadata(
        {'deploy_state': {'phase': 'ran'}}, direction='write'
    )

    import shared.deploy_state as deploy_state_module  # only now, for reference

    assert registered_before_reference_import is deploy_state_module.DeployState
    assert warnings == []
