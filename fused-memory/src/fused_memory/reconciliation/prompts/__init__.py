"""Stage and judge system prompts."""

import logging

logger = logging.getLogger(__name__)

# Base template — use {{project_id}} so it survives .format(tools=...) as {project_id}.
# Each caller then formats with .format(project_id=self.project_id) at runtime.
_PROJECT_ID_GUIDELINE = (
    'Always pass project_id="{{project_id}}" when calling fused-memory MCP tools '
    '({tools}).'
)

# Stage 1: Memory Consolidator — memory read/write, no task tools
_STAGE1_PROJECT_ID_GUIDELINE = _PROJECT_ID_GUIDELINE.format(
    tools=(
        'search, get_entity, get_episodes, get_status, add_memory, delete_memory, '
        'update_edge'
    )
)

# Stage 2: Task-Knowledge Sync — full memory + task access. Bulk-creation
# tools (expand_task / parse_prd) were retired with the Taskmaster cutover;
# task decomposition now happens via planning_mode + curator only.
_STAGE2_PROJECT_ID_GUIDELINE = _PROJECT_ID_GUIDELINE.format(
    tools=(
        'search, get_entity, get_episodes, add_memory, delete_memory, update_edge, '
        'get_tasks, get_task, set_task_status, submit_task, resolve_ticket, '
        'update_task, remove_task, add_dependency, remove_dependency'
    )
)

# Stage 3: Integrity Check — read-only, no writes
_STAGE3_PROJECT_ID_GUIDELINE = _PROJECT_ID_GUIDELINE.format(
    tools='search, get_entity, get_episodes, get_status, get_tasks, get_task'
)

# Shared guidance about the memory_ids=[] + stores=['graphiti'] → graphiti_writes_queued
# invariant.  Both stages need to teach the LLM not to count async-enqueued Graphiti
# writes under their `memories_*` stats; only the stat-key tokens differ between stages.
_GRAPHITI_QUEUED_GUIDANCE_TEMPLATE = (
    "Graphiti-only async-enqueued writes show `stores: ['graphiti']` in the response but "
    "return `memory_ids: []` because the write is queued rather than persisted inline. These "
    "must NOT be counted under {stat_keys_phrase}. Report them instead under "
    "a separate `graphiti_writes_queued` stat. The stats verifier enforces this split "
    "independently and will override any inflated {primary_stat_key} count, but you should "
    "report it correctly from the start to avoid divergence."
)

_STAGE1_GRAPHITI_QUEUED_GUIDANCE = _GRAPHITI_QUEUED_GUIDANCE_TEMPLATE.format(
    stat_keys_phrase="`memories_added` / `memories_written`",
    primary_stat_key="`memories_added`",
)

_STAGE2_GRAPHITI_QUEUED_GUIDANCE = _GRAPHITI_QUEUED_GUIDANCE_TEMPLATE.format(
    stat_keys_phrase="`memories_written`",
    primary_stat_key="`memories_written`",
)

# ---------------------------------------------------------------------------
# Shared recon_report tool-usage guidance (PRD γ §9)
# ---------------------------------------------------------------------------
# Extracted to prevent prompt drift across Stage 1 / 2 / 3.  Any UUID-format
# tweak, field-name change, or dedup-anchor update belongs here; it propagates
# automatically to all stage prompts that interpolate this constant.
#
# Stage-specific deltas stay *inline* in each stage module:
#   Stage 2 — "(including cross_project_routing findings above)" in the intro.
#   Stage 3 — "## Report Channel" section header + read-only NOTE inserted
#             between the cite-tool list and the stats line.
#
# Dedup anchor (reviewer finding dedup_correctness, PRD §9.3; corrected task-1594):
#   _derive_affected_ids reads cited_tasks (not the top-level task_id field of
#   add_finding) when building the fingerprint identity for compute_content_fingerprint.
#   Always call cite_task for the primary subject task so the fingerprint is stable.
#   For multi-task findings, the cited_tasks signature shifts as citations grow or
#   shrink — pass task_id=<primary> at the top level of add_finding as a supplementary
#   stable anchor when one primary subject exists.
#   Exception: cross_project findings use task_id=None (operator routing); cite_task
#   is the sole dedup anchor there (see ## Cross-Project Routing in stage2.py).
#
# Call shapes below are GENERATED from live FastMCP tool signatures (task-2559
# root-cause fix for run_id-omission drift that survived two reviewer rounds) —
# see render_recon_report_tool_guidance() — rather than hand-transcribed, so a
# rendered example can never silently omit a required kwarg again.
_RECON_REPORT_PLACEHOLDERS = {
    'run_id': '<from Reconciliation Context>',
    'finding_id': '<finding_id from add_finding response>',
    'severity': '<severity>',
    'category': '<category>',
    'description': '<description>',
    'suggested_action': '<suggested_action>',
    'actionable': '<actionable>',
    'task_id': '<task_id>',
    'flag_type': '<flag_type>',
    'key': '<key>',
    'value': '<value>',
    'delta': '<delta>',
    'summary': '<brief human-readable summary>',
    'name': '<canonical entity name>',
    'edge_uuid': '<full 36-char UUID>',
    'project_id': '<project_id>',
    'memory_id': '<uuid>',
    'store': "<'mem0'|'graphiti'>",
}


def render_recon_report_tool_guidance() -> str:
    """Render _RECON_REPORT_TOOL_GUIDANCE's call shapes from live tool signatures.

    Introspects each agent-called report tool's live signature (via
    :func:`fused_memory.server.recon_report.get_recon_report_tool_signatures`,
    which owns the one place this package reaches into FastMCP's tool-manager
    internals) so every rendered call always carries every parameter the live
    tool requires. This is the root-cause fix for run_id-omission drift: a
    hand-transcribed example can silently go stale when a signature changes;
    a generated one cannot (task-2559). A param with no entry in
    _RECON_REPORT_PLACEHOLDERS falls back to a generic ``<param_name>``
    placeholder, so even a newly-added required kwarg is guaranteed to render.

    start_report is harness-called (agents never call it themselves) and is
    intentionally excluded from generation — its mention below stays prose.

    Raises whatever :func:`get_recon_report_tool_signatures` raises (e.g. if
    FastMCP's internals have changed shape) — the module-level call site below
    catches this and falls back to a frozen static string rather than letting
    it become an ImportError for every consumer of this package.
    """
    from fused_memory.server.recon_report import get_recon_report_tool_signatures

    signatures = get_recon_report_tool_signatures()

    def render_call(tool_name: str) -> str:
        args = ', '.join(
            f'{param_name}={_RECON_REPORT_PLACEHOLDERS.get(param_name, f"<{param_name}>")}'
            for param_name in signatures[tool_name].parameters
        )
        return f'mcp__recon-report__{tool_name}({args})'

    add_finding_call = render_call('add_finding')
    cite_entity_call = render_call('cite_entity')
    cite_edge_call = render_call('cite_edge')
    cite_task_call = render_call('cite_task')
    cite_memory_call = render_call('cite_memory')
    set_stat_call = render_call('set_stat')
    inc_stat_call = render_call('inc_stat')
    complete_call = render_call('complete')

    return (
        'The harness calls `mcp__recon-report__start_report` for you before the stage begins'
        f' — do NOT call it yourself. For each finding, call `{add_finding_call}`'
        ' and capture the `finding_id` from the response. Then attach typed citations:\n'
        f'- `{cite_entity_call}` —'
        ' pass the ENTITY NAME (not a UUID); the server resolves the UUID internally.\n'
        f'- `{cite_edge_call}` —'
        ' copy the UUID verbatim from the `id` field of a fresh tool result'
        ' (`xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`). Never truncate or construct edge UUIDs.\n'
        f'- `{cite_task_call}`'
        ' — both project_id and task_id are required. **Dedup anchor**:'
        ' `_derive_affected_ids` reads `cited_tasks` (not the top-level `task_id` field of'
        ' `add_finding`) when building the fingerprint for `compute_content_fingerprint`.'
        ' Always call `cite_task` for the primary subject task so the fingerprint is stable.'
        ' For multi-task findings, the cited_tasks signature shifts as citations grow or'
        ' shrink — also pass `task_id=<primary>` at the top level of `add_finding` as a'
        ' supplementary stable anchor when one clear primary subject exists. Exception:'
        ' cross_project findings use `task_id=None` (operator routing); `cite_task` is the'
        ' sole dedup anchor there.\n'
        f'- `{cite_memory_call}` — `memory_id` must be the full 36-char UUID from the'
        ' `id` field of a fresh tool result.\n'
        f'For stats counters use `{set_stat_call}` or'
        f' `{inc_stat_call}`. When all findings are recorded'
        ' and all work is done, call'
        f' `{complete_call}` as your'
        ' terminal action — do NOT produce a structured JSON response; the assembled'
        ' recon_report state is the authoritative output channel for this stage.'
    )


# Last-resort fallback if render_recon_report_tool_guidance() raises at import
# time (e.g. a FastMCP upgrade changes the tool-manager internals guarded by
# get_recon_report_tool_signatures(), or recon_report's server construction
# regresses). This is a FROZEN, hand-written snapshot of a known-good render
# — it is not exercised on the normal path and is not re-verified against the
# live signatures, so treat it as a crash-avoidance safety net, not a source
# of truth: it can go stale exactly like the hand-transcribed text this task
# replaced. Every call shape below still carries run_id, so even a stale
# fallback cannot regress the original run_id-omission bug this task fixed.
_RECON_REPORT_TOOL_GUIDANCE_FALLBACK = (
    'The harness calls `mcp__recon-report__start_report` for you before the stage begins'
    ' — do NOT call it yourself. For each finding, call'
    ' `mcp__recon-report__add_finding(run_id=<from Reconciliation Context>,'
    ' severity=<severity>, category=<category>, description=<description>,'
    ' suggested_action=<suggested_action>, actionable=<actionable>, task_id=<task_id>,'
    ' flag_type=<flag_type>)` and capture the `finding_id` from the response. Then attach'
    ' typed citations:\n'
    '- `mcp__recon-report__cite_entity(run_id=<from Reconciliation Context>,'
    ' finding_id=<finding_id from add_finding response>, name=<canonical entity name>)`'
    ' — pass the ENTITY NAME (not a UUID); the server resolves the UUID internally.\n'
    '- `mcp__recon-report__cite_edge(run_id=<from Reconciliation Context>,'
    ' finding_id=<finding_id from add_finding response>, edge_uuid=<full 36-char UUID>)`'
    ' — copy the UUID verbatim from the `id` field of a fresh tool result'
    ' (`xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`). Never truncate or construct edge UUIDs.\n'
    '- `mcp__recon-report__cite_task(run_id=<from Reconciliation Context>,'
    ' finding_id=<finding_id from add_finding response>, project_id=<project_id>,'
    ' task_id=<task_id>)` — both project_id and task_id are required. **Dedup anchor**:'
    ' `_derive_affected_ids` reads `cited_tasks` (not the top-level `task_id` field of'
    ' `add_finding`) when building the fingerprint for `compute_content_fingerprint`.'
    ' Always call `cite_task` for the primary subject task so the fingerprint is stable.'
    ' For multi-task findings, the cited_tasks signature shifts as citations grow or'
    ' shrink — also pass `task_id=<primary>` at the top level of `add_finding` as a'
    ' supplementary stable anchor when one clear primary subject exists. Exception:'
    ' cross_project findings use `task_id=None` (operator routing); `cite_task` is the'
    ' sole dedup anchor there.\n'
    '- `mcp__recon-report__cite_memory(run_id=<from Reconciliation Context>,'
    ' finding_id=<finding_id from add_finding response>, memory_id=<uuid>,'
    " store=<'mem0'|'graphiti'>)` — `memory_id` must be the full 36-char UUID from the"
    ' `id` field of a fresh tool result.\n'
    'For stats counters use `mcp__recon-report__set_stat(run_id=<from Reconciliation'
    ' Context>, key=<key>, value=<value>)` or `mcp__recon-report__inc_stat(run_id=<from'
    ' Reconciliation Context>, key=<key>, delta=<delta>)`. When all findings are recorded'
    ' and all work is done, call `mcp__recon-report__complete(run_id=<from Reconciliation'
    ' Context>, summary=<brief human-readable summary>)` as your terminal action — do NOT'
    ' produce a structured JSON response; the assembled recon_report state is the'
    ' authoritative output channel for this stage.'
)

try:
    _RECON_REPORT_TOOL_GUIDANCE = render_recon_report_tool_guidance()
except Exception:
    logger.exception(
        'render_recon_report_tool_guidance() failed at import time; falling back to '
        'the frozen _RECON_REPORT_TOOL_GUIDANCE_FALLBACK static string. Recon-report '
        'tool-call guidance may be stale until the underlying introspection failure '
        '(see fused_memory.server.recon_report.get_recon_report_tool_signatures) is '
        'fixed — this self-heals once that succeeds again.'
    )
    _RECON_REPORT_TOOL_GUIDANCE = _RECON_REPORT_TOOL_GUIDANCE_FALLBACK
