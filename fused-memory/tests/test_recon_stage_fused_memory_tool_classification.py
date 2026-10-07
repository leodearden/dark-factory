"""Every tool the fused-memory MCP server registers is classified for recon stages.

Recon stage gating is DENY-LIST ONLY: ``--disallowed-tools`` is fed from the
``STAGE*_DISALLOWED`` lists in ``reconciliation/cli_stage_runner.py`` and there
is no allow-list, so a newly registered tool is callable from every stage —
including Stage 3, read-only by contract — until someone names it in a list.

That is not hypothetical. ``update_memory`` (task 3088, esc-3623-3) shipped in
no disallow list and left an in-place silent-rewrite primitive callable from
Stage 3. Reviewer suggestion #3 (``reviewer_comprehensive``) on task 3623,
anchored in ``tests/test_recon_amend_tool_advertisement.py``, asked for a guard
that makes the next one impossible to ship unclassified. This is it: the
fused-memory twin of
``tests/test_stages.py::TestDisallowedToolLists::test_every_escalation_server_tool_is_classified``.

The live tool surface is enumerated, never a hand-kept list: a constant-only
test pins what the list says, not what the server exposes.
"""

import asyncio
import itertools
from unittest.mock import AsyncMock

from fused_memory.reconciliation.cli_stage_runner import (
    DISALLOW_FUSED_MEMORY_CONTROL_PLANE_WRITES,
    DISALLOW_MEMORY_WRITES,
    DISALLOW_TASK_WRITES,
    STAGE1_DISALLOWED,
    STAGE2_DISALLOWED,
    STAGE3_DISALLOWED,
)
from fused_memory.server.tools import create_mcp_server

_PREFIX = 'mcp__fused-memory__'

# The reviewed-safe half of the classification: tools that only READ, so a
# stage — Stage 3 included — may hold them. Adding a name here is a decision
# that the tool mutates nothing; it is not a formality.
_REVIEWED_FUSED_MEMORY_STAGE_SAFE = frozenset(
    f'{_PREFIX}{name}'
    for name in (
        'search',
        'count_memories_by_metadata',
        'get_cycle_summary_presence',
        'scan_memory_content',
        'get_memories_by_metadata',
        'get_memory_by_id',
        'get_entity',
        'get_entity_by_uuid',
        'get_episodes',
        'get_status',
        'get_queue_stats',
        'get_curator_state',
        'get_wal_status',
        'get_dead_letters',
        'get_tasks',
        'get_statuses',
        'get_external_statuses',
        'get_task',
        'list_tickets',
        'search_tasks',
        'get_pin_queue',
        'get_scheduler_state',
        'get_scheduler_events',
    )
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
)

_FUSED_MEMORY_DENY_BUCKETS = {
    'DISALLOW_MEMORY_WRITES': DISALLOW_MEMORY_WRITES,
    'DISALLOW_TASK_WRITES': DISALLOW_TASK_WRITES,
    'DISALLOW_FUSED_MEMORY_CONTROL_PLANE_WRITES': DISALLOW_FUSED_MEMORY_CONTROL_PLANE_WRITES,
}

_STAGE_LISTS = {
    'STAGE1_DISALLOWED': STAGE1_DISALLOWED,
    'STAGE2_DISALLOWED': STAGE2_DISALLOWED,
    'STAGE3_DISALLOWED': STAGE3_DISALLOWED,
}


def _registered_tool_names() -> set[str]:
    svc = AsyncMock()
    svc.durable_queue = None
    server = create_mcp_server(svc)
    return {f'{_PREFIX}{tool.name}' for tool in asyncio.run(server.list_tools())}


def test_every_fused_memory_server_tool_is_classified():
    """Every registered tool is denied to Stage 3 or reviewed as read-only.

    The deny side is the composed ``STAGE3_DISALLOWED``, not a union of the
    ``DISALLOW_*`` constants: Stage 3 is the stage that must hold no mutator,
    and every fused-memory deny bucket folds into it, so this is both the
    classification check and a direct enforcement of Stage 3's contract. It
    also stays right when a future bucket is added.
    """
    unclassified = (
        _registered_tool_names() - set(STAGE3_DISALLOWED) - _REVIEWED_FUSED_MEMORY_STAGE_SAFE
    )
    assert not unclassified, (
        f'These fused-memory tools are reachable from every recon stage, the '
        f'read-only Stage 3 included, and have not been classified: '
        f'{sorted(unclassified)}. Stage gating is deny-list only, so an unlisted '
        f'mutator is silently callable — the update_memory incident. Put each '
        f'mutating tool in exactly one bucket in cli_stage_runner.py: '
        f'DISALLOW_MEMORY_WRITES (denied in Stage 3 only), DISALLOW_TASK_WRITES '
        f'(denied in Stage 1 and Stage 3; Stage 2 files tasks), or '
        f'DISALLOW_FUSED_MEMORY_CONTROL_PLANE_WRITES (denied in every stage). A '
        f'tool that only reads goes in _REVIEWED_FUSED_MEMORY_STAGE_SAFE here — '
        f'a decision that it mutates nothing, not a formality.'
    )


def test_no_reviewed_safe_tool_is_named_for_a_mutation():
    """The reviewed-safe set must not become the path of least resistance.

    A name that leads with a mutating verb is almost certainly a mutator, and
    parking it here to silence the guard above would re-open the hole.
    """
    suspicious = sorted(
        name
        for name in _REVIEWED_FUSED_MEMORY_STAGE_SAFE
        if name.removeprefix(_PREFIX).startswith(_MUTATING_VERB_PREFIXES)
    )
    assert not suspicious, (
        f'_REVIEWED_FUSED_MEMORY_STAGE_SAFE holds tools named for a mutation: '
        f'{suspicious}. Classify each into a DISALLOW_* bucket instead.'
    )


def test_every_denied_fused_memory_tool_is_registered():
    """A denial naming a tool the server does not register is decoration."""
    denied = {
        name
        for stage_list in _STAGE_LISTS.values()
        for name in stage_list
        if name.startswith(_PREFIX)
    }
    stale = denied - _registered_tool_names()
    assert not stale, (
        f'The stage disallow lists name fused-memory tools the server no longer '
        f'registers: {sorted(stale)}. Remove them, or fix the rename.'
    )


def test_every_reviewed_safe_tool_is_registered():
    """A stale reviewed-safe entry would hide a rename behind a passing guard."""
    stale = _REVIEWED_FUSED_MEMORY_STAGE_SAFE - _registered_tool_names()
    assert not stale, (
        f'_REVIEWED_FUSED_MEMORY_STAGE_SAFE names tools the server no longer '
        f'registers: {sorted(stale)}. Remove them, or fix the rename.'
    )


def test_reviewed_safe_tools_are_not_denied_to_stage3():
    """A tool cannot be safe for the read-only stage and denied to it."""
    contradictory = _REVIEWED_FUSED_MEMORY_STAGE_SAFE & set(STAGE3_DISALLOWED)
    assert not contradictory, (
        f'Both reviewed-safe and denied to Stage 3: {sorted(contradictory)}'
    )


def test_fused_memory_deny_buckets_are_pairwise_disjoint():
    """Each tool sits in exactly one bucket; two would be contradictory decisions."""
    for (name_a, a), (name_b, b) in itertools.combinations(
        _FUSED_MEMORY_DENY_BUCKETS.items(), 2
    ):
        shared = set(a) & set(b)
        assert not shared, f'{name_a} and {name_b} both classify {sorted(shared)}'


def test_stage_machinery_mutators_are_denied_in_every_stage():
    """No stage may re-enter reconciliation or reconfigure the server running it."""
    for tool in (f'{_PREFIX}trigger_reconciliation', f'{_PREFIX}reload_config'):
        for stage_name, stage_list in _STAGE_LISTS.items():
            assert tool in stage_list, f'{stage_name} must deny {tool}'


def test_add_system_record_is_denied_in_stage3_only():
    """Parity with ``add_memory``: Stages 1 and 2 write memory, Stage 3 does not."""
    tool = f'{_PREFIX}add_system_record'
    assert tool in STAGE3_DISALLOWED
    assert tool not in STAGE1_DISALLOWED
    assert tool not in STAGE2_DISALLOWED


def test_commit_planning_is_kept_by_stage2_and_denied_in_stage1_and_stage3():
    """Anti-over-denial: Stage 2 files dependent batches through planning mode.

    ``prompts/stage2.py`` drives ``planning_mode -> add_dependency ->
    commit_planning``; denying the commit there strands every planned batch.
    Stages 1 and 3 file no tasks, like ``submit_task`` / ``resolve_ticket``.
    """
    tool = f'{_PREFIX}commit_planning'
    assert tool not in STAGE2_DISALLOWED, 'Stage 2 must keep its batch-filing path'
    assert tool in STAGE1_DISALLOWED
    assert tool in STAGE3_DISALLOWED
