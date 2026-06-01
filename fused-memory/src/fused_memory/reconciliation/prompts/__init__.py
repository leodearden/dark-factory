"""Stage and judge system prompts."""

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
_RECON_REPORT_TOOL_GUIDANCE = (
    'The harness calls `mcp__recon-report__start_report` for you before the stage begins'
    ' — do NOT call it yourself. For each finding, call `mcp__recon-report__add_finding(...)`'
    ' and capture the `finding_id` from the response. Then attach typed citations:\n'
    '- `mcp__recon-report__cite_entity(finding_id=..., name=<canonical entity name>)` —'
    ' pass the ENTITY NAME (not a UUID); the server resolves the UUID internally.\n'
    '- `mcp__recon-report__cite_edge(finding_id=..., edge_uuid=<full 36-char UUID>)` —'
    ' copy the UUID verbatim from the `id` field of a fresh tool result'
    ' (`xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`). Never truncate or construct edge UUIDs.\n'
    '- `mcp__recon-report__cite_task(finding_id=..., project_id=<project_id>,'
    ' task_id=<task_id>)` — both fields are required. **Dedup anchor**:'
    ' `_derive_affected_ids` reads `cited_tasks` (not the top-level `task_id` field of'
    ' `add_finding`) when building the fingerprint for `compute_content_fingerprint`.'
    ' Always call `cite_task` for the primary subject task so the fingerprint is stable.'
    ' For multi-task findings, the cited_tasks signature shifts as citations grow or'
    ' shrink — also pass `task_id=<primary>` at the top level of `add_finding` as a'
    ' supplementary stable anchor when one clear primary subject exists. Exception:'
    ' cross_project findings use `task_id=None` (operator routing); `cite_task` is the'
    ' sole dedup anchor there.\n'
    "- `mcp__recon-report__cite_memory(finding_id=..., memory_id=<uuid>,"
    " store=<'mem0'|'graphiti'>)` — `memory_id` must be the full 36-char UUID from the"
    ' `id` field of a fresh tool result.\n'
    'For stats counters use `mcp__recon-report__set_stat(key=..., value=...)` or'
    ' `mcp__recon-report__inc_stat(key=..., delta=...)`. When all findings are recorded'
    ' and all work is done, call'
    ' `mcp__recon-report__complete(summary=<brief human-readable summary>)` as your'
    ' terminal action — do NOT produce a structured JSON response; the assembled'
    ' recon_report state is the authoritative output channel for this stage.'
)
