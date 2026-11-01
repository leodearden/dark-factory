"""Per-tool stage scoping and hygiene of the fused-memory tool classification.

Recon stage gating is DENY-LIST ONLY: ``--disallowed-tools`` is fed from the
``STAGE*_DISALLOWED`` lists in ``reconciliation/cli_stage_runner.py`` and there
is no allow-list, so a newly registered tool is callable from every stage —
including Stage 3, read-only by contract — until someone names it in a list.
``update_memory`` (task 3088, esc-3623-3) shipped exactly that way.

The COMPLETENESS guard — every registered tool is denied to Stage 3 or reviewed
as read-only — lives beside its escalation twin, at
``tests/test_stages.py::TestDisallowedToolLists::test_every_fused_memory_tool_is_classified``.
This file keeps the pins on individual tools' stage scoping and on the
classification's own hygiene (no stale denial, disjoint buckets, no mutator
parked in the reviewed-safe set). The live surface and the reviewed-safe set
come from ``tests/_fused_memory_tool_surface.py``.
"""

from _fused_memory_tool_surface import (
    FUSED_MEMORY_TOOL_PREFIX,
    REVIEWED_FUSED_MEMORY_STAGE_SAFE,
    fused_memory_names,
    registered_fused_memory_tools,
)
from _tool_surface_classification import assert_pairwise_disjoint

from fused_memory.reconciliation.cli_stage_runner import (
    DISALLOW_CONSOLIDATION_PROPOSALS,
    DISALLOW_DEFECT_RECORD_WRITES,
    DISALLOW_FUSED_MEMORY_CONTROL_PLANE_WRITES,
    DISALLOW_MEMORY_WRITES,
    DISALLOW_TASK_WRITES,
    STAGE1_DISALLOWED,
    STAGE2_DISALLOWED,
    STAGE3_DISALLOWED,
)

_MUTATING_VERB_PREFIXES = (
    'add_',
    'set_',
    'delete_',
    'update_',
    'remove_',
    'merge_',
    'rename_',
    'replay_',
    'rebuild_',
    'trigger_',
    'reorder_',
    'cancel_',
    'submit_',
    'resolve_',
    'commit_',
    'clear_',
    'request_',
    'unhalt_',
    'reassign_',
    'redact_',
    'refresh_',
    'consolidate_',
    'ensure_',
    'reload_',
    'propose_',
)

_FUSED_MEMORY_DENY_BUCKETS = {
    'DISALLOW_MEMORY_WRITES': DISALLOW_MEMORY_WRITES,
    'DISALLOW_TASK_WRITES': DISALLOW_TASK_WRITES,
    'DISALLOW_FUSED_MEMORY_CONTROL_PLANE_WRITES': DISALLOW_FUSED_MEMORY_CONTROL_PLANE_WRITES,
    'DISALLOW_CONSOLIDATION_PROPOSALS': DISALLOW_CONSOLIDATION_PROPOSALS,
    'DISALLOW_DEFECT_RECORD_WRITES': DISALLOW_DEFECT_RECORD_WRITES,
}

_STAGE_LISTS = {
    'STAGE1_DISALLOWED': STAGE1_DISALLOWED,
    'STAGE2_DISALLOWED': STAGE2_DISALLOWED,
    'STAGE3_DISALLOWED': STAGE3_DISALLOWED,
}


def test_no_reviewed_safe_tool_is_named_for_a_mutation():
    """A cheap name-prefix backstop against parking a mutator in the reviewed-safe set.

    It catches only mutators named for their verb; one with a neutral name
    passes. Read-only-ness rests on the human review that
    ``REVIEWED_FUSED_MEMORY_STAGE_SAFE`` records, not on this test.
    """
    suspicious = sorted(
        name
        for name in REVIEWED_FUSED_MEMORY_STAGE_SAFE
        if name.removeprefix(FUSED_MEMORY_TOOL_PREFIX).startswith(_MUTATING_VERB_PREFIXES)
    )
    assert not suspicious, (
        f'REVIEWED_FUSED_MEMORY_STAGE_SAFE holds tools named for a mutation: '
        f'{suspicious}. Classify each into a DISALLOW_* bucket instead.'
    )


def test_every_denied_fused_memory_tool_is_registered():
    """A denial naming a tool the server does not register is decoration."""
    denied = fused_memory_names(
        name for stage_list in _STAGE_LISTS.values() for name in stage_list
    )
    stale = denied - registered_fused_memory_tools().keys()
    assert not stale, (
        f'The stage disallow lists name fused-memory tools the server no longer '
        f'registers: {sorted(stale)}. Remove them, or fix the rename.'
    )


def test_fused_memory_deny_buckets_are_pairwise_disjoint():
    """Each tool sits in exactly one bucket; two would be contradictory decisions."""
    assert_pairwise_disjoint(_FUSED_MEMORY_DENY_BUCKETS)


def test_stage_machinery_mutators_are_denied_in_every_stage():
    """No stage may re-enter reconciliation or reconfigure the server running it."""
    for tool in (f'{FUSED_MEMORY_TOOL_PREFIX}trigger_reconciliation', f'{FUSED_MEMORY_TOOL_PREFIX}reload_config'):
        for stage_name, stage_list in _STAGE_LISTS.items():
            assert tool in stage_list, f'{stage_name} must deny {tool}'


def test_add_system_record_is_denied_in_stage3_only():
    """Parity with ``add_memory``: Stages 1 and 2 write memory, Stage 3 does not."""
    tool = f'{FUSED_MEMORY_TOOL_PREFIX}add_system_record'
    assert tool in STAGE3_DISALLOWED
    assert tool not in STAGE1_DISALLOWED
    assert tool not in STAGE2_DISALLOWED


def test_commit_planning_is_kept_by_stage2_and_denied_in_stage1_and_stage3():
    """Anti-over-denial: Stage 2 files dependent batches through planning mode.

    ``prompts/stage2.py`` drives ``planning_mode -> add_dependency ->
    commit_planning``; denying the commit there strands every planned batch.
    Stages 1 and 3 file no tasks, like ``submit_task`` / ``resolve_ticket``.
    """
    tool = f'{FUSED_MEMORY_TOOL_PREFIX}commit_planning'
    assert tool not in STAGE2_DISALLOWED, 'Stage 2 must keep its batch-filing path'
    assert tool in STAGE1_DISALLOWED
    assert tool in STAGE3_DISALLOWED


def test_defect_record_writers_are_denied_in_every_stage():
    """No recon stage writes defect records (plans/defect-tests-prd.md decision 7)."""
    for tool in (
        f'{FUSED_MEMORY_TOOL_PREFIX}declare_defect',
        f'{FUSED_MEMORY_TOOL_PREFIX}record_defect_outcome',
    ):
        for stage_name, stage_list in _STAGE_LISTS.items():
            assert tool in stage_list, f'{stage_name} must deny {tool}'
