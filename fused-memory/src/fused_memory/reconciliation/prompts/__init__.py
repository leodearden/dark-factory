"""Stage and judge system prompts."""

import functools
import inspect
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

# ---------------------------------------------------------------------------
# Shared escalation-store boundary (task 3163, plans/escalation-store-ambiguity-prd.md)
# ---------------------------------------------------------------------------
# The BODY below is identical for all three stages — store identity, the missing
# read tools, and the "an empty result is not evidence of absence" rule hold
# everywhere — but the SANCTIONED-ACTION clause varies by stage capability,
# which is exactly the case the recon_self_model render_*_section() precedent
# exists for. So the note is RENDERED via render_escalation_boundary_note()
# rather than pasted: only Stage 2 holds the FIX D escalate_blocker path.
#
# The companion mechanism is DISALLOW_ESCALATION_READS in cli_stage_runner.py,
# which denies the escalation READ tools in every stage. This paragraph is not
# decorative: `--disallowed-tools` OMITS a denied tool from the agent's listing
# rather than rejecting the call, so without it the agent gets no explanation
# for the missing tool and can conclude the escalation surface does not exist.
#
# MUST NOT contain the string '## Available Tools' — build_stage2_system_prompt
# raises RuntimeError unless that sentinel appears exactly once in
# STAGE2_SYSTEM_PROMPT. Not an f-string: it is interpolated INTO f-strings, and
# braces inside an interpolated value are not re-parsed by the enclosing
# f-string, so its own text needs no {{/}} escaping.
ESCALATION_BOUNDARY_NOTE = """\
## Escalation Store Boundary
Your `mcp__escalation__*` connection serves the RECONCILIATION escalation store \
only — never a project's escalation queue. A question of the form "was an \
escalation ever filed for task X?" CANNOT be answered from this stage: the \
escalation READ tools are not available to you here, and any empty or missing \
escalation result you encounter is NOT evidence that no escalation exists. It \
means only that you are looking in the wrong store. Never report, infer, or \
flag "no escalation was filed for task X" on the strength of this connection; \
if a finding depends on that question, record what you actually observed and \
hand the question on rather than asserting an absence.\
"""

# Per-stage sanctioned-action clauses appended to ESCALATION_BOUNDARY_NOTE.
# Keep these two mutually exclusive: a stage is either told about the one write
# path it may take, or told plainly that it has none. Never both, never neither.
_ESCALATION_BOUNDARY_SANCTIONED_WRITE = (
    'The one sanctioned escalation action in this stage is the write path '
    '(`mcp__escalation__escalate_blocker`) for the Stale Flag Escalation (FIX D) case.'
)

_ESCALATION_BOUNDARY_NO_ACTION = (
    'No escalation action is sanctioned in this stage: do not file escalations '
    'here. Record what you actually observed through the recon_report channel '
    '(`mcp__recon-report__add_finding`) and let the reconciliation harness route '
    'it from there.'
)


def render_escalation_boundary_note(*, can_escalate: bool) -> str:
    """Render the escalation-store boundary note for one stage.

    Follows the :func:`render_source_completion_section` precedent
    (reconciliation/recon_self_model.py): one shared stage-agnostic body plus a
    single clause selected by what the stage is actually sanctioned to do, with
    a keyword-only capability flag.

    The clause is parameterized because the FIX D Stale Flag Escalation is a
    **Stage 2 mechanism only** — its prompt text lives in prompts/stage2.py and
    its flag-relay driver in stages/task_knowledge_sync.py, inside
    ``class TaskKnowledgeSync``, not ``class IntegrityCheck``. Stage 1 and
    Stage 3 have no FIX D path and no other sanctioned escalation write, so
    they get the no-action variant.

    This matters beyond tidiness: ``escalate_blocker`` is deliberately NOT in
    any disallow list (see DISALLOW_ESCALATION_READS in cli_stage_runner.py,
    which denies only the READ tools), so the tool really is callable from every
    stage. Naming it in a stage's prompt is therefore a live positive license,
    not inert prose — and Stage 3 declares itself read-only two lines earlier.
    Never tell a stage about an action it is not sanctioned to take
    (loud-over-silent), and never license a durable write from a read-only
    stage.

    Args:
        can_escalate: True for Stage 2, which holds the FIX D escalate_blocker
            path; False for Stage 1 and Stage 3, which hold no sanctioned
            escalation action.

    Returns:
        ESCALATION_BOUNDARY_NOTE followed by the matching sanctioned-action
        clause. The shared core appears exactly once either way (INV-5).
    """
    clause = (
        _ESCALATION_BOUNDARY_SANCTIONED_WRITE
        if can_escalate
        else _ESCALATION_BOUNDARY_NO_ACTION
    )
    return f'{ESCALATION_BOUNDARY_NOTE} {clause}'

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
# Shared deterministic-gate closure guidance (task 3023)
# ---------------------------------------------------------------------------
# Stage 1 emits the flag and Stage 2 files the remediation task, so this
# suppression policy is only sound if BOTH stages hold the identical rule.
# Kept as ONE constant interpolated into both stage prompts — hand-duplicated
# prompt prose in this module has already drifted twice.
#
# NOTE: this is a PLAIN (non-f) string, so its literal braces need no doubling.
# Do NOT inline this prose into the stage f-strings — that would require
# escaping every `{`/`}` below.  Keep it free of bare recon-report tool call
# examples (see test_recon_report_guidance_drift.py, which scans the assembled
# stage prompts and requires every such example to carry `run_id=`).
_GATE_CLOSURE_ARCHIVE_GUIDANCE = """\
## Deterministic-Gate Closure: pending-only probes are not proof of absence
Scope: a `done` task with `task_kind='deterministic'`, `metadata.gate_escalated_at` \
set, and a `done_provenance.kind` of `deterministic-gate` or `deterministic-milestone`.

`mcp__escalation__get_pending_escalations(task_id=...)` is PENDING/ROOT-ONLY BY \
DESIGN. When a human resolves a born-at-L2 deterministic gate, that record is MOVED \
to `data/escalations/archive/<date>/` and correctly disappears from that query. An \
empty result there is the EXPECTED outcome for a properly-closed gate — it is NOT \
evidence that the escalation record was never written.

**You cannot query the orchestrator's escalation queue at all, so you can NEVER \
establish that a gate escalation record was never written.** The `escalation` MCP \
server in YOUR config is backed by the RECONCILIATION escalation queue \
(`data/reconciliation/escalations`, port 8103 — see \
`ReconciliationConfig.escalation_queue_dir`). Orchestrator deterministic-gate records \
are written to a DIFFERENT store (`data/escalations`, port 8102 — see \
`dark-factory-orchestrator.yaml`), which NO MCP server in your config is connected to. \
An empty result from ANY `mcp__escalation__*` query available to you is therefore \
UNINFORMATIVE about orchestrator gate records — not evidence of absence. Do not treat \
it as such, and do not cite it as evidence in a finding, a memory, or a filed task. \
(An archive-inclusive `get_task_escalations` lookup does exist on the ORCHESTRATOR-side \
escalation server, port 8102 — but you are not connected to that server and must not \
claim its results.)

**Establish closure from the TASK RECORD instead — this rule is primary and \
sufficient.** Call `mcp__fused-memory__get_task(task_id)` (a READ; allowed in both \
Stage 1 and Stage 2). `status == 'done'` PLUS `metadata.gate_escalated_at` set PLUS a \
`done_provenance.kind` of `deterministic-gate` or `deterministic-milestone` IS \
LEGITIMATE closure, on its own — as is a resolved/archived record if you have one. Do \
not flag it, do not file remediation work, and do not write a memory asserting the gap. \
Absent positive contrary evidence you can actually obtain, DO NOT ASSERT ABSENCE AT ALL.

Incident lineage, so you recognise this pattern rather than re-deriving it: \
dark_factory tasks 2841, 2842, 2844, 2846, 2919, 2954, 2955, 2958, 2999, 3005, 3006 \
and reify tasks 5330, 5341, 5349, 5352, 5353 were all filed on this same false \
premise. The orchestrator side (`deterministic_runner.py` / `harness.py`) already \
queries the archive inclusively and is NOT the defect — do not file work against it.\
"""

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
    'cited_run_id': '<full 36-char run UUID>',
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

    Every parameter still renders — dropping optional ones would reopen the
    drift hole this task closed — but a parameter carrying a default value
    (genuinely optional) is wrapped in square brackets, e.g.
    ``[task_id=<task_id>]``, so the example does not read as though every
    kwarg must always be supplied. A parameter with no default (required)
    renders bare, mirroring common CLI usage-string conventions
    (``cmd required [optional]``).

    start_report is harness-called (agents never call it themselves) and is
    intentionally excluded from generation — its mention below stays prose.

    Raises whatever :func:`get_recon_report_tool_signatures` raises (e.g. if
    FastMCP's internals have changed shape) — :func:`get_recon_report_tool_guidance`
    catches this and falls back to a frozen static string rather than letting
    it become an ImportError for every consumer of this package.
    """
    from fused_memory.server.recon_report import get_recon_report_tool_signatures

    signatures = get_recon_report_tool_signatures()

    def render_call(tool_name: str) -> str:
        parts = []
        for param_name, param in signatures[tool_name].parameters.items():
            placeholder = _RECON_REPORT_PLACEHOLDERS.get(param_name, f'<{param_name}>')
            kwarg = f'{param_name}={placeholder}'
            parts.append(kwarg if param.default is inspect.Parameter.empty else f'[{kwarg}]')
        return f'mcp__recon-report__{tool_name}({", ".join(parts)})'

    add_finding_call = render_call('add_finding')
    cite_entity_call = render_call('cite_entity')
    cite_edge_call = render_call('cite_edge')
    cite_task_call = render_call('cite_task')
    cite_memory_call = render_call('cite_memory')
    cite_run_call = render_call('cite_run')
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
        f'- `{cite_run_call}` — whenever a finding\'s description or'
        ' suggested_action references another reconciliation run\'s run_id, call this'
        ' to confirm it exists and attach it. Copy `cited_run_id` verbatim from the'
        ' `run_id` or `metadata.run_id` field of a fresh tool result — never re-type'
        ' or paraphrase a run_id from memory.\n'
        f'For stats counters use `{set_stat_call}` or'
        f' `{inc_stat_call}`. When all findings are recorded'
        ' and all work is done, call'
        f' `{complete_call}` as your'
        ' terminal action — do NOT produce a structured JSON response; the assembled'
        ' recon_report state is the authoritative output channel for this stage.'
    )


# Last-resort fallback if render_recon_report_tool_guidance() raises when first
# called (e.g. a FastMCP upgrade changes the tool-manager internals guarded by
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
    '- `mcp__recon-report__cite_run(run_id=<from Reconciliation Context>,'
    ' finding_id=<finding_id from add_finding response>,'
    " cited_run_id=<full 36-char run UUID>)` — whenever a finding's description or"
    " suggested_action references another reconciliation run's run_id, call this to"
    ' confirm it exists and attach it. Copy `cited_run_id` verbatim from the `run_id`'
    ' or `metadata.run_id` field of a fresh tool result — never re-type or paraphrase'
    ' a run_id from memory.\n'
    'For stats counters use `mcp__recon-report__set_stat(run_id=<from Reconciliation'
    ' Context>, key=<key>, value=<value>)` or `mcp__recon-report__inc_stat(run_id=<from'
    ' Reconciliation Context>, key=<key>, delta=<delta>)`. When all findings are recorded'
    ' and all work is done, call `mcp__recon-report__complete(run_id=<from Reconciliation'
    ' Context>, summary=<brief human-readable summary>)` as your terminal action — do NOT'
    ' produce a structured JSON response; the assembled recon_report state is the'
    ' authoritative output channel for this stage.'
)


@functools.lru_cache(maxsize=1)
def _cached_recon_report_tool_guidance() -> str:
    """Cache wrapper around :func:`render_recon_report_tool_guidance`.

    Split out from :func:`get_recon_report_tool_guidance` (task-2559
    amendment) so that ``lru_cache`` only ever memoizes a *successful*
    render: ``functools.lru_cache`` does not cache a call that raises (the
    exception propagates on every call until one succeeds), so a transient
    introspection failure here is retried on the very next call instead of
    being pinned forever. See :func:`get_recon_report_tool_guidance` for the
    fallback handling built on top of this.
    """
    return render_recon_report_tool_guidance()


def get_recon_report_tool_guidance() -> str:
    """Return the recon-report tool-call guidance text, computed once and cached.

    Deliberately lazy (task-2559 amendment): computing this at
    ``prompts/__init__.py`` import time meant EVERY import of the ``prompts``
    package — including sibling submodules unrelated to recon-report guidance,
    e.g. ``prompts.judge`` (imported by ``reconciliation.judge``, which never
    touches recon-report tool calls) — paid the cost of constructing a
    throwaway FastMCP recon_report server. Deferring to first call means only
    a consumer that actually needs the guidance (Stage 1/2/3, which interpolate
    it into their system prompts at their own module import) triggers the
    build, and the :func:`_cached_recon_report_tool_guidance` helper ensures a
    successful build happens at most once per process.

    Falls back to the frozen :data:`_RECON_REPORT_TOOL_GUIDANCE_FALLBACK`
    static string if :func:`render_recon_report_tool_guidance` raises (e.g. a
    FastMCP upgrade changes the tool-manager internals guarded by
    ``get_recon_report_tool_signatures``) rather than letting it become an
    ImportError for every consumer of this package. Unlike the successful
    path, the fallback is deliberately NOT cached — only
    :func:`_cached_recon_report_tool_guidance`'s ``lru_cache`` is consulted,
    and it never stores a raised call, so a transient failure (e.g. a
    momentary hiccup constructing the throwaway FastMCP server) does not
    permanently pin every later call in this same process to the frozen
    fallback; the very next call retries the live render and self-heals as
    soon as it succeeds.
    """
    try:
        return _cached_recon_report_tool_guidance()
    except Exception:
        logger.exception(
            'render_recon_report_tool_guidance() failed; falling back to '
            'the frozen _RECON_REPORT_TOOL_GUIDANCE_FALLBACK static string. Recon-report '
            'tool-call guidance may be stale until the underlying introspection failure '
            '(see fused_memory.server.recon_report.get_recon_report_tool_signatures) is '
            'fixed — the next call retries automatically, in this process or a fresh one.'
        )
        return _RECON_REPORT_TOOL_GUIDANCE_FALLBACK
