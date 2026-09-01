"""Stage and judge system prompts."""

import functools
import inspect
import logging
from collections.abc import Mapping

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
        'update_edge, update_memory, redact_episode_content, delete_episode'
    )
)

# Stage 2: Task-Knowledge Sync — full memory + task access. Bulk-creation
# tools (expand_task / parse_prd) were retired with the Taskmaster cutover;
# task decomposition now happens via planning_mode + curator only.
_STAGE2_PROJECT_ID_GUIDELINE = _PROJECT_ID_GUIDELINE.format(
    tools=(
        'search, get_entity, get_episodes, add_memory, delete_memory, update_edge, '
        'update_memory, redact_episode_content, delete_episode, '
        'get_tasks, get_task, set_task_status, submit_task, resolve_ticket, '
        'update_task, remove_task, add_dependency, remove_dependency'
    )
)

# Stage 3: Integrity Check — read-only, no writes
_STAGE3_PROJECT_ID_GUIDELINE = _PROJECT_ID_GUIDELINE.format(
    tools='search, get_entity, get_episodes, get_status, get_tasks, get_task'
)

# ---------------------------------------------------------------------------
# Shared memory-amend / episode-mutation tool listing (esc-3391-1)
# ---------------------------------------------------------------------------
# Stage 1 and Stage 2 can already call all three of these tools today — neither
# STAGE1_DISALLOWED nor STAGE2_DISALLOWED folds DISALLOW_MEMORY_WRITES
# (cli_stage_runner.py) — but until now neither stage's "## Available Tools"
# block named them, so the agent never learned they exist. `--disallowed-tools`
# OMITS a denied tool rather than rejecting a call (cli_stage_runner.py:96-104),
# so an unadvertised-but-held tool is just as invisible to the agent by the
# mirror-image failure: nothing in its prompt ever mentions it.
#
# MUST NOT contain the literal '## Available Tools' — build_stage2_system_prompt
# raises RuntimeError unless that sentinel appears exactly once in
# STAGE2_SYSTEM_PROMPT. Not an f-string: it is interpolated INTO f-strings, and
# braces inside an interpolated value are not re-parsed by the enclosing
# f-string, so its own text needs no {{/}} escaping.
AMEND_AND_EPISODE_TOOLS_BLOCK = """\
- `mcp__fused-memory__update_memory` — in-place Mem0 content amend / metadata patch. \
The Qdrant point id and created_at are PRESERVED, so nothing citing the record dangles. \
PREFER THIS over delete + re-add when consolidating a cluster: delete + re-add mints a \
new UUID and strands any task metadata citing the old one. Call with `store="mem0"` \
(`store="graphiti"` is rejected — use `update_edge` instead) and a non-empty `reason` \
whenever you pass `content`; the record payload goes in `metadata_patch` — `metadata` is \
the causation envelope and is never stored on the record.
- `mcp__fused-memory__redact_episode_content` — in-place replacement of ONE episode's raw \
content. NON-destructive: the episode's extracted edges are deliberately left untouched.
- `mcp__fused-memory__delete_episode` — IRREVERSIBLE. `cascade=True` (the default) also \
destroys entities and edges exclusively sourced from that episode.\
"""

# ---------------------------------------------------------------------------
# Shared stale/wrong-knowledge annotation norm (esc-3391-1 ruling)
# ---------------------------------------------------------------------------
# States the precedence order for annotating superseded, wrong, or corrupted
# knowledge across both stores — lightest-touch first, last-resort last — plus
# the ruling's option (b) cross-check rule against trusting episode prose at
# face value. Shared between Stage 1 and Stage 2 because both hold every tool
# named below (neither STAGE1_DISALLOWED nor STAGE2_DISALLOWED folds
# DISALLOW_MEMORY_WRITES) and every sentence is true verbatim in both stages.
# Placed in the prompt rather than CLAUDE.md because the recon stages are the
# only consumer of raw episode prose.
#
# MUST NOT contain the literal '## Available Tools' — build_stage2_system_prompt
# raises RuntimeError unless that sentinel appears exactly once in
# STAGE2_SYSTEM_PROMPT. MUST NOT contain a bare `mcp__recon-report__` call
# example — test_recon_report_guidance_drift.py scans the assembled prompts
# and requires every such example to carry `run_id=`; this section introduces
# none. MUST NOT reference "## UUID Resolution Discipline" by heading name —
# stage2.py has no such section; refer to `replacement_memory_id` by
# parameter name instead, a tool-level fact true in both stages. Not an
# f-string: it is interpolated INTO f-strings, and braces inside an
# interpolated value are not re-parsed by the enclosing f-string, so its own
# text needs no {{/}} escaping (there are none below regardless).
STALE_KNOWLEDGE_ANNOTATION_NORM = """\
## Annotating Stale or Wrong Knowledge
When you find stale, superseded, or wrong knowledge, use the LIGHTEST-touch tool that \
correctly resolves it — in this order:

(a) **Stale or superseded FACT**: write a superseding edge, or call `update_edge` with \
`invalid_at` set on the stale edge. This is the house pattern and is already in live use. \
`search` returns EDGES, not raw episode prose — a superseding edge is the annotation that \
is actually RETRIEVABLE; an annotation buried in episode text is not.
(b) **Episode text wrong ON ITS FACE**: use `mcp__fused-memory__redact_episode_content` \
ONLY for this case. In BOTH sub-cases below, first call `get_episodes` and copy the \
episode's CURRENT content verbatim — `new_content` REPLACES the stored text in full and \
the original is not recoverable; if you cannot retrieve the full current text, do not \
redact. Leave the surrounding narrative untouched. The two sub-cases need DIFFERENT \
edits, and confusing them gets your call refused:
  - **Wrong claim** (a premature past-tense completion claim, a superseded assertion): \
keep the text as-is and prefix a dated `[SUPERSEDED: ...]` marker. The wrong prose stays \
readable as a record of what was believed at the time.
  - **Corrupted or leaked fragment** (e.g. leaked serialized tool-call XML): you must \
REPLACE the corrupted span ITSELF with a redaction marker such as \
`[REDACTED: leaked tool-call fragment]`, not merely prefix a marker to the front. The \
server re-runs its leak detector over your `new_content` and RAISES on a call that still \
carries the fragment ("a redaction that re-introduces the leak is not a redaction"), so a \
verbatim copy with a prefix added is rejected every time. A refusal here does NOT mean \
you should fall through to (c) — it means your replacement text is not yet leak-free; \
excise the fragment and retry.
(c) **Last resort**: `mcp__fused-memory__delete_episode` — see the tool listing above \
for what `cascade=True` destroys. Use this only when (a) and (b) cannot resolve the \
problem. A `redact_episode_content` REFUSAL is not such a case: `redact_episode_content` \
exists precisely to avoid the cascade, so a rejected redaction means fix the \
`new_content` and retry (b), never escalate to (c).
(d) **Mem0 cluster consolidation**: amend the SURVIVOR in place via \
`mcp__fused-memory__update_memory` — see the tool listing above for why — and only THEN \
delete the redundant siblings, naming the survivor via `replacement_memory_id`. Never \
delete and re-add the survivor.

**Episode prose is a point-in-time narration, not current truth.** An episode's content \
may assert work as complete that is still in progress by the time you read it. Before \
acting on an episode's claim, cross-check current entity/edge state (`search` / \
`get_entity`) or live task status (`get_task`) rather than trusting the prose at face \
value.\
"""

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

# The one concrete case the boundary note above exists for (task 3023), rendered
# as a SUBSECTION of it rather than as a second top-level block elsewhere in the
# prompt. Everything general — store identity, the missing escalation read tools,
# "an empty result is not evidence of absence", "never flag 'no escalation was
# filed for task X'" — is stated ONCE, in ESCALATION_BOUNDARY_NOTE (INV-5); this
# constant adds only what is NOT already there: the POSITIVE closure rule read
# off the task record, and the incident lineage.
#
# Do NOT restate the pending-only semantics of `get_pending_escalations` here, or
# present any `mcp__escalation__*` read as a probe the stage ran or could run: the
# escalation READ tools are denied to every stage (DISALLOW_ESCALATION_READS in
# cli_stage_runner.py) and the denial OMITS them from the tool listing, so
# describing them as available contradicts both the note above and the stage's
# actual tool surface. The bare token `get_task_escalations` is retained
# deliberately — registry source_assertion #2 of
# `deterministic_gate_escalation_record_archived_not_missing`
# (config/recon_code_fix_premise_registry.yaml) reads this file's text — and is
# described accurately, as an ORCHESTRATOR-side lookup no stage can reach.
#
# NOTE: a PLAIN (non-f) string interpolated into the stage f-strings, so its
# literal braces need no doubling. Keep it free of bare recon-report tool call
# examples (see test_recon_report_guidance_drift.py, which scans the assembled
# stage prompts and requires every such example to carry `run_id=`).
_GATE_CLOSURE_ARCHIVE_GUIDANCE = """\
### Deterministic-gate closure is established from the TASK RECORD
Scope: a `done` task with `task_kind='deterministic'`, `metadata.gate_escalated_at` \
set, and a `done_provenance.kind` of `deterministic-gate` or `deterministic-milestone`.

Call `mcp__fused-memory__get_task(task_id)` — a READ you do hold. `status == 'done'` \
PLUS `metadata.gate_escalated_at` set PLUS one of those `done_provenance.kind` values \
IS LEGITIMATE CLOSURE on its own; this rule is primary and sufficient. Do not flag such \
a task, do not file remediation work against it, and do not write a memory asserting a \
missing escalation record. An archive-inclusive `get_task_escalations` lookup that would \
confirm the record directly does exist, but only on the ORCHESTRATOR-side escalation \
server, which is not part of your tool surface — that changes nothing about the rule \
here, which needs no escalation lookup at all.

Incident lineage, so you recognise this pattern rather than re-deriving it: dark_factory \
tasks 2841, 2842, 2844, 2846, 2919, 2954, 2955, 2958, 2999, 3005, 3006 and reify tasks \
5330, 5341, 5349, 5352, 5353 were all filed on the false premise that these gates closed \
without an escalation record. The orchestrator side (`deterministic_runner.py` / \
`harness.py`) already queries the archive inclusively and is NOT the defect — do not file \
work against it.\
"""

# Single-sourced per INV-5 `no-lockstep-duplication` (docs/legibility/design-invariants.md):
# this paragraph originated as an inline bullet in stage1.py and now lives here ONLY, with
# all three stage prompts interpolating it. Every stage can be handed a `duplicate_finding`
# error — add_finding returns it on a colliding (task_id, flag_type) signature AND on a
# repeated description for a null-null finding (the _run_desc_index path, which every stage
# reaches whenever it files informational / cross_project findings), and cite_task returns
# it from BOTH in-run cited-task folds (task-2425 project-scoped, task-2432 entity-scoped,
# the latter with no stage carve-out) — so the salvage protocol has to reach all of them.
# Do NOT re-inline a per-stage copy; reword HERE. Keep the enumeration open-ended rather
# than counting the paths: an exact count that a real response can contradict tells an
# agent the shape it just received cannot occur, which is exactly what invites the
# fabricate-a-finding_id behaviour this paragraph exists to prevent.
#
# NOTE: a PLAIN (non-f) string interpolated into the stage f-strings, so it must stay free
# of literal braces. Rendered as a markdown BULLET so it drops cleanly into stage1's
# error-response list and stage2's guidance list alike. Keep it free of bare recon-report
# tool call examples (see test_recon_report_guidance_drift.py, which requires every such
# example in an assembled prompt to carry `run_id=`) — name the tools, never call them.
DUPLICATE_FINDING_SALVAGE_GUIDANCE = """\
- `duplicate_finding` — NOT a new successful filing: never invent or count a \
`finding_id` for it. Several different paths return it. `add_finding` returns it when \
this run already holds a finding with the same (task_id, flag_type) signature — or, for \
a finding carrying neither, the same description — filed by this stage or by an earlier \
one. `cite_task` returns it when the task you just cited is already the in-run anchor \
for an equivalent finding — and on THAT path your own finding is purged wholesale, \
taking its description, its suggested action, and any citations you had already attached \
to it. Whichever path returned it, the response's `existing_finding_id` is the canonical \
id for this finding: re-issue your citations against that id rather than fabricating a \
new one, and carry over any detail your finding carried that the surviving one does not. \
Once purged there is nothing left to recover it from, so do not simply drop the extra \
detail as redundant.\
"""


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
        ESCALATION_BOUNDARY_NOTE, the matching sanctioned-action clause, and the
        deterministic-gate closure subsection (:data:`_GATE_CLOSURE_ARCHIVE_GUIDANCE`,
        task 3023) — the one concrete case this boundary exists for, rendered here
        so the shared core still appears exactly once (INV-5) instead of being
        restated by a second block elsewhere in the prompt.
    """
    clause = (
        _ESCALATION_BOUNDARY_SANCTIONED_WRITE
        if can_escalate
        else _ESCALATION_BOUNDARY_NO_ACTION
    )
    return f'{ESCALATION_BOUNDARY_NOTE} {clause}\n\n{_GATE_CLOSURE_ARCHIVE_GUIDANCE}'

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
# rendered example can never silently omit a required KWARG again.
#
# Which TOOLS get a call shape at all is derived too (task 3878): the renderer
# iterates the keys of the signature mapping it is handed, so it can no longer
# silently omit a whole tool the way its predecessor — nine hard-coded
# render_call('...') invocations that ignored the mapping's other keys — did.
# The mapping is the live tool set minus the harness-called and stage-gated
# tools (start_report keeps its PROSE mention below — the sentence that tells
# the agent not to call it — but no call shape).
#
# Classification rules and rationale: see the "Tool classification" block in
# server/recon_report.py, which is the single canonical statement of them and
# sits where an author registering a new @mcp.tool() is already looking.
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



# The curated NARRATIVE order the guidance renders KNOWN tools in — deliberately
# NOT registration order (``signatures.keys()`` yields start_report /
# add_finding / set_stat / inc_stat / complete / delete_finding / cite_*),
# because the prose tells a story: file a finding -> attach typed citations ->
# record stats -> terminal complete.
#
# This tuple governs PLACEMENT and prose richness ONLY — never PRESENCE (task
# 3878). :func:`_render_recon_report_tool_guidance` renders every key of the
# mapping it is handed, appending a generic call-shape bullet for any tool not
# named here, so a newly-registered tool cannot be silently omitted from the
# guidance; the worst that can happen is that it lands in the trailing "Also
# registered on this server" list with no curated annotation until someone
# writes one. A name here that is ABSENT from the mapping is skipped rather
# than raising, so this tuple going stale in the other direction (a tool
# deregistered upstream) cannot break rendering either.
#
# This tuple and _GUIDANCE_TOOL_PROSE below must stay in sync — every name here
# except the _GUIDANCE_STATS_GROUP pair needs a prose entry. That is pinned by
# tests/test_recon_report_guidance_drift.py::TestCuratedGuidanceTablesStayInSync,
# and a slip degrades to a generic bullet rather than raising (see
# _render_recon_report_tool_guidance's is_annotated()).
_GUIDANCE_TOOL_ORDER: tuple[str, ...] = (
    'add_finding',
    'delete_finding',
    'cite_entity',
    'cite_edge',
    'cite_task',
    'cite_memory',
    'cite_run',
    'set_stat',
    'inc_stat',
    'complete',
)

# The citation tools share ONE lead-in sentence introducing them as a list.
# It is emitted immediately before whichever of them renders FIRST (rather
# than being welded onto add_finding's paragraph, where it used to live), so
# inserting delete_finding between filing and citing does not strand a
# colon-terminated 'Then attach typed citations:' in front of unrelated prose
# — and so the lead-in disappears entirely if no citation tool is present.
_GUIDANCE_CITATION_GROUP: tuple[str, ...] = (
    'cite_entity',
    'cite_edge',
    'cite_task',
    'cite_memory',
    'cite_run',
)
_GUIDANCE_CITATIONS_LEAD_IN = 'Then attach typed citations:\n'

# set_stat and inc_stat share ONE sentence because they are ALTERNATIVES ("use
# X or Y"), not successive steps like the rest of the narrative. They render as
# a group, at the position of whichever of them comes first in
# _GUIDANCE_TOOL_ORDER, with whichever are present joined by ' or ' — splitting
# them into two independent sentences would change the shipped wording.
_GUIDANCE_STATS_GROUP: tuple[str, ...] = ('set_stat', 'inc_stat')

# Opening prose. Unconditional, and deliberately NOT derived from *signatures*:
# start_report's mention here is the sentence that tells the agent the harness
# already called it, so it must survive even though no start_report CALL SHAPE
# is rendered (the caller filters it out — see HARNESS_CALLED_REPORT_TOOLS in
# server/recon_report.py).
_GUIDANCE_PREAMBLE = (
    'The harness calls `mcp__recon-report__start_report` for you before the stage begins'
    ' — do NOT call it yourself. '
)

# Curated per-tool prose. The `{call}` marker is substituted with the rendered
# call shape via ``str.replace`` (NOT ``str.format``), so prose containing
# literal braces needs no escaping.
#
# Keys must cover _GUIDANCE_TOOL_ORDER minus _GUIDANCE_STATS_GROUP (whose two
# tools share one sentence rendered from this table's peer above). A missing or
# renamed key does NOT raise — the tool degrades to an uncurated "Also
# registered on this server" bullet — so the sync is pinned by a test rather
# than by a crash; see _render_recon_report_tool_guidance's is_annotated().
#
# This prose is LOAD-BEARING, not decoration: the cite_task dedup-anchor
# paragraph, the cite_edge/cite_memory verbatim-UUID rules, cite_run's
# never-paraphrase rule and complete's terminal-action rule are all behavioural
# contracts the stages depend on. Edit the wording here and it propagates to
# every stage prompt at once; delete it and a stage silently loses the rule.
_GUIDANCE_TOOL_PROSE: dict[str, str] = {
    'add_finding': (
        'For each finding, call `{call}` and capture the `finding_id` from the'
        ' response. '
    ),
    'delete_finding': (
        'To retract a finding you filed in error, call `{call}` — IRREVERSIBLE, and'
        ' scoped by run_id + finding_id. It is rejected once that finding\'s OWNING'
        ' stage entry has been completed, so retraction is for in-progress stages'
        ' only: because every stage calls `complete` as its terminal action, that'
        ' guard means you are retracting a finding your own still-running stage'
        ' filed, not overriding a verdict a finished stage already closed.'
        ' Structured errors: run_id_unknown / finding_unknown /'
        ' report_already_completed. Retract and re-file rather than filing a'
        ' correction alongside a finding you know to be wrong.\n'
    ),
    'cite_entity': (
        '- `{call}` — pass the ENTITY NAME (not a UUID); the server resolves the UUID'
        ' internally.\n'
    ),
    'cite_edge': (
        '- `{call}` — copy the UUID verbatim from the `id` field of a fresh tool result'
        ' (`xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`). Never truncate or construct edge UUIDs.\n'
    ),
    'cite_task': (
        '- `{call}` — both project_id and task_id are required. **Dedup anchor**:'
        ' `_derive_affected_ids` reads `cited_tasks` (not the top-level `task_id` field of'
        ' `add_finding`) when building the fingerprint for `compute_content_fingerprint`.'
        ' Always call `cite_task` for the primary subject task so the fingerprint is stable.'
        ' For multi-task findings, the cited_tasks signature shifts as citations grow or'
        ' shrink — also pass `task_id=<primary>` at the top level of `add_finding` as a'
        ' supplementary stable anchor when one clear primary subject exists. Exception:'
        ' cross_project findings use `task_id=None` (operator routing); `cite_task` is the'
        ' sole dedup anchor there.\n'
    ),
    'cite_memory': (
        '- `{call}` — `memory_id` must be the full 36-char UUID from the `id` field of a'
        ' fresh tool result.\n'
    ),
    'cite_run': (
        '- `{call}` — whenever a finding\'s description or suggested_action references'
        " another reconciliation run's run_id, call this to confirm it exists and attach"
        ' it. Copy `cited_run_id` verbatim from the `run_id` or `metadata.run_id` field of'
        ' a fresh tool result — never re-type or paraphrase a run_id from memory.\n'
    ),
    'complete': (
        'When all findings are recorded and all work is done, call `{call}` as your'
        ' terminal action — do NOT produce a structured JSON response; the assembled'
        ' recon_report state is the authoritative output channel for this stage.'
    ),
}


def _render_recon_report_tool_guidance(
    signatures: Mapping[str, tuple[tuple[str, bool], ...]],
) -> str:
    """Render the recon-report tool-usage guidance prose from *signatures*.

    Shared by :func:`render_recon_report_tool_guidance` (live introspection)
    and the frozen :func:`_frozen_recon_report_tool_guidance` (rendered from
    :data:`_FROZEN_RECON_REPORT_SIGNATURE_SPECS`) — both flow through this one
    prose template and this one ``render_call``, so they cannot drift apart in
    WORDING. The only surface that can still go stale is the frozen snapshot's
    parameter data, which
    ``tests/test_recon_report_guidance_drift.py::TestFallbackIsDerivedFromTheSameRenderer``
    guards directly against the live signatures.

    *signatures* maps each tool name to an ordered ``(param_name, required)``
    tuple sequence — deliberately not ``inspect.Signature`` objects — because
    that pair (a name to render, a bool to decide bracketing) is all
    ``render_call`` below ever reads. This is also the NATIVE shape of the
    frozen snapshot, so :func:`_frozen_recon_report_tool_guidance` passes it
    straight through with no conversion, while
    :func:`render_recon_report_tool_guidance` converts the live
    ``inspect.Signature`` objects into this same shape before calling here.
    Sharing one input shape (rather than teaching this function to branch on
    two) is what makes "the same render_call() code path" a literal fact
    rather than a convention to maintain, and it retires a previous design
    that fabricated synthetic ``inspect.Signature``/``inspect.Parameter``
    objects for the frozen path purely to satisfy this function, including a
    ``default=None`` that was never a real tool default and could mislead a
    future reader who assumed it was.

    Every parameter still renders — dropping optional ones would reopen the
    drift hole task-2559 closed — but a parameter carrying a default value
    (genuinely optional) is wrapped in square brackets, e.g.
    ``[task_id=<task_id>]``, so the example does not read as though every
    kwarg must always be supplied. A parameter with no default (required)
    renders bare, mirroring common CLI usage-string conventions
    (``cmd required [optional]``).

    This function excludes NOTHING. start_report is absent from the rendered
    call shapes because the CALLER does not hand it over; the exclusion is data
    applied by :func:`render_recon_report_tool_guidance`, not a property of
    this function, which renders whatever mapping it is given. Filtering which
    tools an agent should be told about at all is the CALLER's job (rules and
    rationale: the "Tool classification" block in ``server/recon_report.py``).
    The start_report PROSE mention in :data:`_GUIDANCE_PREAMBLE` is
    unconditional and deliberately not derived from *signatures* — it is the
    sentence that tells the agent the harness already called it.

    PRESENCE is derived, PLACEMENT is curated (task 3878). Every key of
    *signatures* is rendered: :data:`_GUIDANCE_TOOL_ORDER` +
    :data:`_GUIDANCE_TOOL_PROSE` decide only WHERE a known tool appears and how
    richly it is annotated, and any tool NOT named there still renders as a
    generic call-shape bullet under a trailing "Also registered on this server"
    lead-in. So a newly-registered tool can never be silently absent from the
    guidance — which is what the previous design, nine hard-coded
    ``render_call('...')`` calls that ignored the mapping's other keys, allowed.

    Neither curated table can turn a slip into a crash. A curated name MISSING
    from *signatures* is skipped, and a name in :data:`_GUIDANCE_TOOL_ORDER`
    with no matching :data:`_GUIDANCE_TOOL_PROSE` entry degrades to the same
    generic bullet an unknown tool gets (reviewer robustness finding) rather
    than raising ``KeyError``. That matters because this renderer is shared
    with the frozen fallback: a raise here would take out BOTH paths at once
    and surface as an ImportError for every consumer of the ``prompts``
    package, the exact blast radius the lazy-render design exists to contain.
    So this function renders whatever it is handed, and never depends on the
    caller supplying a particular tool or on the two curated tables agreeing.
    """

    def render_call(tool_name: str) -> str:
        parts = []
        for param_name, required in signatures[tool_name]:
            placeholder = _RECON_REPORT_PLACEHOLDERS.get(param_name, f'<{param_name}>')
            kwarg = f'{param_name}={placeholder}'
            parts.append(kwarg if required else f'[{kwarg}]')
        return f'mcp__recon-report__{tool_name}({", ".join(parts)})'

    def is_annotated(tool_name: str) -> bool:
        """Does *tool_name* have curated prose (or a curated group sentence) to render?

        _GUIDANCE_TOOL_ORDER and _GUIDANCE_TOOL_PROSE are two tables that must
        stay in sync, and nothing structurally forces them to. A name in the
        ORDER with no matching PROSE entry — added to one table only, or a key
        renamed/typo'd in the other — must NOT raise ``KeyError`` here: this
        function is shared by the live path AND the frozen fallback, so a raise
        would take out both and become an ImportError for every consumer of the
        ``prompts`` package (see :func:`_frozen_recon_report_tool_guidance`).
        Instead the tool degrades to the same generic call-shape bullet an
        unknown tool gets, losing one annotation rather than every stage prompt.
        """
        return tool_name in _GUIDANCE_STATS_GROUP or tool_name in _GUIDANCE_TOOL_PROSE

    stats_present = [tool for tool in _GUIDANCE_STATS_GROUP if tool in signatures]
    citations_lead_in_pending = _GUIDANCE_CITATIONS_LEAD_IN
    sections = [_GUIDANCE_PREAMBLE]
    for tool in _GUIDANCE_TOOL_ORDER:
        if tool not in signatures or not is_annotated(tool):
            # Either a curated name that is no longer registered, or one with no
            # curated prose to render. Skip it rather than raising: a
            # deregistered tool degrades to a missing paragraph, an unannotated
            # one falls through to the generic bullet list below. The
            # annotation check happens HERE, before the citation lead-in is
            # emitted, so an unannotated cite_* tool cannot strand a
            # colon-terminated 'Then attach typed citations:' in front of
            # unrelated prose.
            continue
        if tool in _GUIDANCE_CITATION_GROUP and citations_lead_in_pending:
            sections.append(citations_lead_in_pending)
            citations_lead_in_pending = ''
        if tool in _GUIDANCE_STATS_GROUP:
            if tool != stats_present[0]:
                continue  # already covered by the shared stats sentence below
            joined = ' or '.join(f'`{render_call(name)}`' for name in stats_present)
            sections.append(f'For stats counters use {joined}. ')
            continue
        sections.append(_GUIDANCE_TOOL_PROSE[tool].replace('{call}', render_call(tool)))

    extra = [
        tool
        for tool in signatures
        if tool not in _GUIDANCE_TOOL_ORDER or not is_annotated(tool)
    ]
    if extra:
        bullets = '\n'.join(f'- `{render_call(tool)}`' for tool in extra)
        sections.append(
            '\nAlso registered on this server (see the server instructions for full'
            f' semantics):\n{bullets}'
        )

    return ''.join(sections)


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

    Filters the live tool set down to the SHARED-GUIDANCE tools before
    rendering: harness-called tools (start_report) and stage-gated tools
    (denied in some stages via DISALLOW_RECON_REPORT_LEDGER_WRITES) are
    dropped, per the classification constants that live next to the
    ``@mcp.tool()`` registrations in ``server/recon_report.py``. That filtering
    is this function's job precisely so that
    :func:`_render_recon_report_tool_guidance` can stay a pure "render every
    key I was handed" function shared with the frozen fallback. start_report
    still gets a prose mention in the rendered text — just no call shape.

    Raises whatever :func:`get_recon_report_tool_signatures` raises (e.g. if
    FastMCP's internals have changed shape) — :func:`get_recon_report_tool_guidance`
    catches this and falls back to the frozen fallback rather than letting
    it become an ImportError for every consumer of this package.
    """
    from fused_memory.server.recon_report import (
        HARNESS_CALLED_REPORT_TOOLS,
        STAGE_GATED_REPORT_TOOLS,
        get_recon_report_tool_signatures,
    )

    signatures = get_recon_report_tool_signatures()
    # WHICH tools an agent is told about is decided by the classification that
    # lives next to the @mcp.tool() registrations — see the "Tool
    # classification" block in server/recon_report.py for the rules and why
    # they exist. Filtering HERE rather than inside
    # _render_recon_report_tool_guidance() is what lets that renderer stay a
    # pure 'render everything I was handed' function shared with the frozen
    # fallback, so the two paths cannot drift in wording.
    specs = {
        tool: tuple(
            (name, param.default is inspect.Parameter.empty)
            for name, param in sig.parameters.items()
        )
        for tool, sig in signatures.items()
        if tool not in HARNESS_CALLED_REPORT_TOOLS and tool not in STAGE_GATED_REPORT_TOOLS
    }
    return _render_recon_report_tool_guidance(specs)


# Last-resort fallback if render_recon_report_tool_guidance() raises when first
# called (e.g. a FastMCP upgrade changes the tool-manager internals guarded by
# get_recon_report_tool_signatures(), or recon_report's server construction
# regresses) -- a crash-avoidance safety net, not a source of truth in its own
# right. It is RENDERED from _FROZEN_RECON_REPORT_SIGNATURE_SPECS below through
# the *same* _render_recon_report_tool_guidance() template and render_call()
# the live guidance uses, so its WORDING cannot drift from the generated
# guidance -- there is only one prose template and one renderer. The one
# remaining hand-maintained surface is the snapshot's parameter data (names,
# order, required-ness) and its tool-set COVERAGE: this dict must hold exactly
# the shared-guidance tools, so a newly registered agent-callable tool needs a
# key here as well as a classification in server/recon_report.py.
# tests/test_recon_report_guidance_drift.py::TestFallbackIsDerivedFromTheSameRenderer
# fails loudly until it is, comparing this dict against the DERIVED
# shared-guidance set rather than against a second hand-maintained list.
# Every call shape below still carries run_id (true by construction: run_id is
# required in the snapshot for all 10 tools), so even a stale snapshot cannot
# regress the original run_id-omission bug task-2559 fixed.
#
# Key ORDER here mirrors _GUIDANCE_TOOL_ORDER purely so this dict reads in the
# same sequence as the text it renders; the renderer keys off the curated order
# tuple, not off this insertion order.
_FROZEN_RECON_REPORT_SIGNATURE_SPECS: dict[str, tuple[tuple[str, bool], ...]] = {
    'add_finding': (
        ('run_id', True),
        ('severity', True),
        ('category', True),
        ('description', True),
        ('suggested_action', True),
        ('actionable', False),
        ('task_id', False),
        ('flag_type', False),
    ),
    'delete_finding': (('run_id', True), ('finding_id', True)),
    'cite_entity': (('run_id', True), ('finding_id', True), ('name', True)),
    'cite_edge': (('run_id', True), ('finding_id', True), ('edge_uuid', True)),
    'cite_task': (
        ('run_id', True),
        ('finding_id', True),
        ('project_id', True),
        ('task_id', True),
    ),
    'cite_memory': (
        ('run_id', True),
        ('finding_id', True),
        ('memory_id', True),
        ('store', True),
    ),
    'cite_run': (('run_id', True), ('finding_id', True), ('cited_run_id', True)),
    'set_stat': (('run_id', True), ('key', True), ('value', True)),
    'inc_stat': (('run_id', True), ('key', True), ('delta', False)),
    'complete': (('run_id', True), ('summary', True)),
}


@functools.lru_cache(maxsize=1)
def _frozen_recon_report_tool_guidance() -> str:
    """Render the frozen fallback guidance, lazily, from the frozen snapshot.

    Deliberately NOT rendered at module-import time (reviewer robustness
    finding). :data:`_FROZEN_RECON_REPORT_SIGNATURE_SPECS` is hand-maintained,
    and rendering it can still fail or silently degrade. A dropped, renamed or
    typo'd key no longer raises ``KeyError`` — since task 3878 the renderer
    iterates the mapping it is handed and SKIPS a curated name that is absent —
    but the failure modes that remain are real: a typo'd key renders as an
    uncurated "Also registered on this server" bullet instead of its annotated
    paragraph, a dropped key loses that tool's guidance (and its behavioural
    rules) entirely, and a malformed VALUE — anything that is not an iterable of
    ``(name, required)`` pairs — still raises from ``render_call``. Computing
    this eagerly at import time turned any such raise into an ImportError for
    every consumer of the ``prompts`` package, including
    sibling submodules such as ``prompts.judge`` that never touch
    recon-report guidance at all — exactly the blast radius
    :func:`get_recon_report_tool_guidance`'s try/except exists to contain.
    Deferring the render to first call means a broken snapshot edit only ever
    surfaces here, when and if the live path has ALSO already failed and this
    safety net is actually needed; see :func:`get_recon_report_tool_guidance`,
    which logs loudly if this raises too rather than silently swallowing a
    broken fallback.

    Cached like :func:`_cached_recon_report_tool_guidance`: the input
    (:data:`_FROZEN_RECON_REPORT_SIGNATURE_SPECS`) is a module-level constant
    that cannot change within a process, so memoizing a successful render is
    safe and avoids re-rendering on every call during a sustained outage.
    ``lru_cache`` never memoizes a raised call, so a broken snapshot is
    retried (and re-raises, and re-logs) on every fallback attempt rather
    than being permanently pinned to failure.
    """
    return _render_recon_report_tool_guidance(_FROZEN_RECON_REPORT_SIGNATURE_SPECS)


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

    Falls back to :func:`_frozen_recon_report_tool_guidance` if
    :func:`render_recon_report_tool_guidance` raises (e.g. a FastMCP upgrade
    changes the tool-manager internals guarded by
    ``get_recon_report_tool_signatures``) rather than letting it become an
    ImportError for every consumer of this package. The fallback branch here
    is deliberately NOT what is memoized by
    :func:`_cached_recon_report_tool_guidance`'s ``lru_cache`` — only a
    successful live render is, and it never stores a raised call, so a
    transient failure (e.g. a momentary hiccup constructing the throwaway
    FastMCP server) does not permanently pin every later call in this same
    process to the fallback; the very next call retries the live render and
    self-heals as soon as it succeeds. (:func:`_frozen_recon_report_tool_guidance`
    has its own, separate cache over the frozen snapshot, which never changes
    within a process — see its docstring.)

    If the fallback ITSELF raises (a broken edit to
    :data:`_FROZEN_RECON_REPORT_SIGNATURE_SPECS`, on top of a live
    introspection failure), that is logged loudly and re-raised rather than
    silently swallowed — a double failure like that means recon-report
    guidance is genuinely unavailable, which callers must see rather than
    silently receive stale or wrong text for.
    """
    try:
        return _cached_recon_report_tool_guidance()
    except Exception:
        logger.exception(
            'render_recon_report_tool_guidance() failed; falling back to '
            'the frozen recon-report tool guidance snapshot (see '
            '_frozen_recon_report_tool_guidance). Recon-report '
            'tool-call guidance may be stale until the underlying introspection failure '
            '(see fused_memory.server.recon_report.get_recon_report_tool_signatures) is '
            'fixed — the next call retries automatically, in this process or a fresh one.'
        )
        try:
            return _frozen_recon_report_tool_guidance()
        except Exception:
            logger.exception(
                'the frozen recon-report tool guidance fallback ALSO failed to render -- '
                'check _FROZEN_RECON_REPORT_SIGNATURE_SPECS in '
                'reconciliation/prompts/__init__.py for a dropped, renamed, or invalid '
                'tool/parameter name. Both the live and the frozen recon-report guidance '
                'paths are now broken.'
            )
            raise
