"""Prompt assembly — builds full prompts for each agent invocation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from orchestrator.agents.roles import WAIT_PATTERN_REMINDER
from orchestrator.config import OrchestratorConfig
from orchestrator.mcp_lifecycle import mcp_call

logger = logging.getLogger(__name__)

COMMIT_BULLET_LIMIT = 40
"""Max commit bullets rendered in one briefing section (task 3033 amendment).

Bounds the architect's already-committed-work section, whose source
(``TaskWorkflow._detect_committed_branch_work``) returns ALL of
``base_commit..HEAD`` — unbounded on a long-lived branch that accumulated WIP
safety-commits across many requeues/rebases plus normal step commits. Without a
cap the section grows with branch length and its ``git show <sha>`` protocol
drives one architect tool call per commit. Its sibling
``_detect_tip_wip_commits`` needs no cap: it is naturally bounded by the
contiguous WIP run at HEAD.
"""


def _format_commit_bullets(commits: list[dict], limit: int | None = None) -> str:
    """Render ``[{'sha': ..., 'subject': ...}]`` as HEAD-first markdown bullets.

    Single source of the 12-char abbreviation convention, shared by
    :meth:`BriefingAssembler.build_architect_prompt`'s committed-work section
    and :meth:`BriefingAssembler.build_implementer_prompt`'s WIP section — both
    consume the same shape from the two sibling detectors, so the format must
    not drift between them.

    ``limit`` caps the rendered bullets (HEAD-first, i.e. most recent) and
    appends an explicit "…and N more" line: a truncation must be VISIBLE in the
    prompt, never silent, so the agent knows the list is partial and can widen
    it itself. ``None`` (the default) renders everything.
    """
    shown = commits if limit is None else commits[:limit]
    lines = [f"- `{c['sha'][:12]}` — {c['subject']}" for c in shown]
    hidden = len(commits) - len(shown)
    if hidden > 0:
        lines.append(
            f'- …and {hidden} more commit(s) on this branch (not shown — run '
            f'`git log --oneline` to see the rest)'
        )
    return '\n'.join(lines)


FOREIGN_PROJECT_TAG_KEYS = ('src_project', 'project_id', 'group_id', 'project')
"""Metadata keys, in precedence order, that name a memory result's owning project.

``src_project`` is FIRST: the task-2273 CGL-eta rehome
(``fused-memory/src/fused_memory/maintenance/rehome_scope_tag.py``, kind
``cgl_eta_cross_target_rehome``) wrote Mem0 entries that physically live in
``dst_project``'s collection but reference ``src_project``'s task numbers —
``src_project`` is the authoritative origin project, so it must win over any
co-present ``project_id``/``group_id`` on the same entry. ``dst_project`` is
deliberately ABSENT from this tuple: consulting it would falsely certify a
rehomed foreign fact as local, since it names where the fact was relocated
TO, not where it came from.
"""


def _canonical_project(value: str) -> str:
    """Canonicalise a project identifier for comparison.

    Mirrors fused-memory's ``canonicalize_project_id`` semantics
    (``fused_memory/utils/validation.py``; see the divergent-spelling
    contract in ``plans/cross-graph-entity-leak-prd.md`` decision 1 / S1) —
    strip, lowercase, ``'-'`` -> ``'_'`` — so a tag spelled ``dark-factory``
    is not mistaken for a project distinct from ``dark_factory``.

    Re-implemented locally rather than imported: orchestrator declares no
    runtime dependency on fused-memory (it appears only in
    ``orchestrator/pyproject.toml``'s ``[tool.pyright] extraPaths``, a
    type-checking-only reference), so importing it here would risk an
    ``ImportError`` in any deployment where fused-memory is not co-installed.
    """
    return value.strip().lower().replace('-', '_')


def _result_project(entry: dict) -> tuple[str, str] | None:
    """Read a result's owning-project tag from its metadata, if any.

    Walks :data:`FOREIGN_PROJECT_TAG_KEYS` in precedence order and returns
    the ``(key, value)`` pair of the first present, non-empty **string**
    value found in ``entry['metadata']``. A non-string value (e.g. an int)
    is treated as absent rather than crashing the comparison. Returns
    ``None`` — i.e. untagged — when ``entry['metadata']`` is missing, not a
    dict, or carries none of the recognised keys.

    The matched key is returned alongside the value (not just the value) so
    a drop can be logged with enough context — which key fired, and what it
    said — to diagnose a false-positive filter from the logs alone.
    """
    metadata = entry.get('metadata')
    if not isinstance(metadata, dict):
        return None
    for key in FOREIGN_PROJECT_TAG_KEYS:
        tag = metadata.get(key)
        if isinstance(tag, str) and tag.strip():
            return key, tag
    return None


def filter_foreign_project_results(payload_text: str, project_id: str) -> tuple[str, int]:
    """Drop cross-project results from a fused-memory ``search`` JSON payload.

    ``payload_text`` is the JSON-serialised ``{'results': [...]}`` dict that
    FastMCP returns as the ``search`` tool's text block (see
    :meth:`BriefingAssembler._mcp_search`). Each result's project tag is read
    via :func:`_result_project` (see :data:`FOREIGN_PROJECT_TAG_KEYS`); the
    entry is dropped when a tag is present and, after
    :func:`_canonical_project` normalisation, differs from ``project_id``,
    and kept otherwise — including when ``metadata`` is missing, empty, or
    not a dict.

    Untagged results are deliberately kept rather than dropped: every
    Graphiti-sourced result has ``metadata == {}`` today (verified at
    ``fused-memory/src/fused_memory/services/memory_service.py:3332-3407``,
    ``_search_graphiti``, which only ever adds a ``planned`` key), so
    dropping untagged results would empty the ``# Context`` block for most
    queries. Only Mem0-sourced results can carry a project tag today.

    Returns the re-serialised payload — sibling top-level keys such as
    ``degraded``/``failed_stores``/``failed_store_diagnostics`` are preserved
    verbatim — and the number of results dropped. Returns ``('', dropped)``
    when nothing survives, so the existing ``if section:`` guards in
    ``_get_memory_context`` skip an all-foreign section the same way they
    skip an empty one.

    When nothing is dropped (the common case — see above), ``payload_text``
    is returned unchanged rather than re-serialised: this preserves the
    upstream formatting byte-for-byte and avoids the cost of a needless
    round-trip. When something IS dropped, the re-serialisation uses
    ``ensure_ascii=False`` so non-ASCII content (dark-factory memory text is
    dense with em dashes and accented characters) is not escaped into
    ``\\uXXXX`` sequences in the rendered ``# Context`` block.

    Fails OPEN on a malformed payload — non-JSON text, JSON that is not an
    object, or a missing/non-list ``results`` — returning ``(payload_text,
    0)`` unchanged and logging a WARNING. Blanking the ``# Context`` block on
    a serialisation surprise would be a silent capability loss across every
    prompt builder; preserving today's (unfiltered) behaviour with a loud
    warning is the safer failure direction. A stray non-dict entry, or a
    non-dict ``metadata`` on an otherwise-well-formed entry, is kept rather
    than raising — treated the same as an untagged result.
    """
    try:
        payload = json.loads(payload_text)
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        logger.warning(f'filter_foreign_project_results: payload is not valid JSON ({e}); keeping unfiltered')
        return payload_text, 0

    if not isinstance(payload, dict):
        logger.warning(
            f'filter_foreign_project_results: payload is a {type(payload).__name__}, '
            'not a JSON object; keeping unfiltered'
        )
        return payload_text, 0

    results = payload.get('results')
    if not isinstance(results, list):
        logger.warning(
            f"filter_foreign_project_results: payload['results'] is a "
            f'{type(results).__name__}, not a list; keeping unfiltered'
        )
        return payload_text, 0

    target = _canonical_project(project_id)
    kept = []
    dropped = 0
    for entry in results:
        if not isinstance(entry, dict):
            kept.append(entry)
            continue
        match = _result_project(entry)
        if match is not None:
            key, tag = match
            if _canonical_project(tag) != target:
                dropped += 1
                logger.debug(
                    f'filter_foreign_project_results: dropped {entry.get("id")!r} '
                    f'({key}={tag!r})'
                )
                continue
        kept.append(entry)

    if not kept:
        return '', dropped

    if dropped == 0:
        # No-op: nothing was filtered, so avoid re-serialising a payload
        # that is byte-for-byte unchanged — this is the overwhelmingly
        # common case, since every Graphiti-sourced result is untagged
        # today and the filter never fires on it.
        return payload_text, 0

    payload = dict(payload)
    payload['results'] = kept
    return json.dumps(payload, indent=2, ensure_ascii=False), dropped


MEMORY_CONTEXT_CAVEAT = (
    "_This context was recalled from the `{project_id}` project's memory — "
    'it is NOT a description of this worktree. It may name tasks, repos, '
    'crates, or file paths that do not exist here. Do not assume a recalled '
    'path is real: verify it exists before reading it or `cd`-ing into it._'
)
"""Standing provenance caveat rendered right after the ``# Context`` heading.

Covers the leak channel :func:`filter_foreign_project_results` cannot reach:
every Graphiti-sourced memory result has ``metadata == {}`` today (see that
function's docstring), so an untagged foreign fact — e.g. a path belonging
to a different project's repo — survives the filter unclassified and
renders verbatim. The tag filter is the permanent chokepoint for taggable
(Mem0) results; this caveat is what actually converts "agent `cd`'s/reads
into a recalled foreign path" into "agent verifies the path first" for the
untagged majority. Interpolated with ``self.project_id`` via ``.format()``.
"""


@dataclass
class CompletionJudgeVerdict:
    """Structured verdict returned by the completion judge agent.

    Distinct from ``evals.judge.JudgeVerdict`` (the Elo pairwise comparison
    judge) — this verdict exits the implementer loop early when the judge
    decides the substantive work is complete, regardless of plan.json
    bookkeeping state.
    """

    complete: bool
    reasoning: str
    uncovered_plan_steps: list[str]
    substantive_work: bool


COMPLETION_JUDGE_SCHEMA = {
    'type': 'object',
    'properties': {
        'complete': {'type': 'boolean'},
        'reasoning': {'type': 'string'},
        'uncovered_plan_steps': {
            'type': 'array', 'items': {'type': 'string'},
        },
        'substantive_work': {'type': 'boolean'},
    },
    'required': ['complete', 'reasoning', 'uncovered_plan_steps', 'substantive_work'],
    'additionalProperties': False,
}


class BriefingAssembler:
    """Builds prompts for agent invocations."""

    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.memory_url = config.fused_memory.url
        self.project_id = config.fused_memory.project_id

    def _agent_identity(self, task_id: str | None, role: str) -> str:
        agent_id = f'claude-task-{task_id}-{role}' if task_id else f'claude-{role}'
        return (
            f'## Agent Identity\n\n'
            f'- **agent_id:** `{agent_id}`\n'
            f'- **project_id:** `{self.project_id}`\n'
        )

    async def build_architect_prompt(
        self,
        task: dict,
        worktree: Path | None = None,
        context: str | None = None,
        *,
        include_prior_proposals: bool = False,
        committed_work: list[dict] | None = None,
    ) -> str:
        """Build prompt for the architect agent.

        Args:
            include_prior_proposals: When True, surface the task's most
                recent ``dry_run_proposals`` entry (if any) as a prior
                block-time investigation. Defaults to False so a truly-fresh
                first dispatch stays proposal-free (C-A1 anti-anchoring) —
                only the re-plan path (an existing plan fell through to
                architect) should pass True.
            committed_work: HEAD-first ``[{'sha': ..., 'subject': ...}]`` for
                every commit this branch already carries beyond its base, as
                produced by ``TaskWorkflow._detect_committed_branch_work``
                (task 3033 / PRD §A1). When truthy, renders the
                already-committed-work section that teaches the architect to
                pre-satisfy already-implemented steps via
                ``mark_step_committed`` instead of leaving them pending. When
                None or empty — the truly-fresh first dispatch — the prompt is
                byte-identical to the pre-γ baseline, so nothing is added on
                the common path.
        """
        if context is None:
            context = await self._get_memory_context(task.get('id'))

        task_block = self._format_task(task, include_files=False)
        identity = self._agent_identity(task.get('id'), 'architect')

        prior_proposal_section = ''
        if include_prior_proposals:
            prior_proposal_section = self._format_prior_proposal(task)

        # Mirrors build_implementer_prompt's wip_section structurally (same
        # bullet format, same corroborate-then-attribute numbered protocol) so
        # both briefings teach one consistent habit; the substantive differences
        # are the tool named (mark_step_committed, an authoring-time authority,
        # vs mark_step_done) and the explicit VERIFY-is-the-gate warning.
        committed_section = ''
        if committed_work:
            commits_list = _format_commit_bullets(
                committed_work, limit=COMMIT_BULLET_LIMIT,
            )
            committed_section = f"""
## Already-Committed Work On This Branch — Pre-Satisfy, Don't Re-Plan-As-Pending

This branch already carries the commit(s) below beyond its base. A prior
dispatch may have implemented some (or all) of this task before the plan was
lost, so part of the work you are about to plan may already be done, committed,
and green on this very branch.

{commits_list}

Protocol:

1. Run `git show <sha>` for each commit above to see exactly what it contains.
2. Author the plan's steps as normal — do NOT drop or merge steps just because
   the work exists. The full TDD structure and its provenance must be preserved.
3. For every step the committed work ALREADY satisfies — confirmed first-hand by
   running that step's tests on this branch and seeing them pass — call
   `mark_step_committed(step_id, <sha>)` with the commit that carries it,
   instead of leaving the step pending. That marks the step done and tags its
   description `[COMMITTED <sha>]` so the provenance is durable.
4. Leave genuinely-unsatisfied steps pending, so the implementer does only the
   remaining work.
5. If EVERY step is pre-satisfied, the EXECUTE loop is skipped entirely and the
   branch flows PLAN → VERIFY → REVIEW → MERGE with zero implementer turns.

VERIFY is the semantic gate, not `mark_step_committed`. A falsely pre-satisfied
step surfaces as a VERIFY failure and BLOCKS the task — so never pre-satisfy a
step you have not actually seen pass on this branch. When in doubt, leave it
pending: an unnecessary implementer turn is cheap, a false green is not.
"""

        return f"""\
{context}

{identity}

# Task

{task_block}

{prior_proposal_section}{committed_section}
# Action

1. Explore the codebase thoroughly — read relevant files, understand existing patterns and utilities.
2. Produce a TDD implementation plan using the plan-tools MCP tools:
   a. Call `create_plan(task_id, title, analysis, files)` with your analysis.
   b. Call `add_prerequisite(prereq_id, description)` for any setup work needed before TDD steps.
   c. Call `add_plan_step(step_id, step_type, description)` for each TDD step, in order. Alternate test/impl.
   d. Call `add_design_decision(decision, rationale)` for non-obvious choices.
   e. Call `add_reuse_item(what, where, how)` for existing code/patterns being reused.
   f. Call `confirm_plan()` as your FINAL action, once every step is added, to mark the plan complete. Without it the plan is treated as incomplete and will not advance.
3. List ALL files (or directory paths) you expect to create or modify in the `files` parameter of `create_plan` — this drives concurrency locks and the phantom-done gate, so be exhaustive and precise.
"""

    async def build_revalidation_prompt(
        self,
        task: dict,
        existing_plan: dict,
        changed_files: list[str],
        worktree: Path | None = None,
        context: str | None = None,
    ) -> str:
        """Build prompt for the architect to revalidate a plan after blast-radius requeue.

        The task was planned in a prior session but requeued because module
        locks were contended.  Main has since advanced (the contending task
        merged).  The architect reviews the existing plan against the changes
        and either confirms, updates, or recreates it.
        """
        if context is None:
            context = await self._get_memory_context(task.get('id'))

        task_block = self._format_task(task)
        identity = self._agent_identity(task.get('id'), 'architect')
        prior_proposal_section = self._format_prior_proposal(task)

        plan_files = set(existing_plan.get('files', []))
        overlapping = [f for f in changed_files if f in plan_files]
        non_overlapping = [f for f in changed_files if f not in plan_files]

        overlap_section = ''
        if overlapping:
            overlap_list = '\n'.join(f'- `{f}`' for f in overlapping)
            overlap_section = (
                f'### Overlapping with your plan (REVIEW THESE):\n{overlap_list}'
            )

        other_section = ''
        if non_overlapping:
            other_list = '\n'.join(f'- `{f}`' for f in non_overlapping[:30])
            suffix = ''
            if len(non_overlapping) > 30:
                suffix = f'\n- ... and {len(non_overlapping) - 30} more'
            other_section = (
                f'### Other changes on main:\n{other_list}{suffix}'
            )

        if not changed_files:
            files_section = '_No files changed on main since your plan was created._'
        else:
            files_section = f'{overlap_section}\n\n{other_section}'.strip()

        plan_json = json.dumps(existing_plan, indent=2)

        return f"""\
{context}

{identity}

# Task

{task_block}

{prior_proposal_section}
# Plan Revalidation

You created a plan for this task in a prior session, but the task was requeued
because another task held locks on modules you need. That task has since merged
to main, so the files it touched may have changed.

Your job: review your previous plan against the current state of the codebase
and either confirm it, update it, or recreate it from scratch.

## Your Previous Plan

```json
{plan_json}
```

## Files Changed on Main Since Your Plan

{files_section}

## Action

1. Read the overlapping files (if any) to understand what changed.
2. Decide whether your plan is still valid.
3. Take ONE of these actions:
   a. **Plan still valid**: call `confirm_plan()`.
   b. **Plan needs updates**: use `update_plan_metadata(files=...)`,
      `remove_plan_step(step_id)`, `replace_plan_step(step_id, step_type, description)`,
      and/or `add_plan_step(step_id, step_type, description)` to adjust the plan in place.
   c. **Plan is invalid**: call `create_plan(...)` to start fresh, then add steps as usual.
4. Ensure the `files` list is exhaustive — it drives concurrency locks.
5. Do NOT remove or replace steps with status "done".
"""

    async def build_plan_completion_prompt(
        self,
        task: dict,
        partial_plan: dict,
        worktree: Path | None = None,  # noqa: ARG002 — interface symmetry
        context: str | None = None,
    ) -> str:
        """Build prompt to finish a partial plan left by an interrupted session.

        A prior planning session wrote some steps but never called
        ``confirm_plan`` (it ran out of budget/turns, or crashed). Rather than
        discard that work and re-plan from scratch, hand the partial plan back
        to the architect: verify the existing steps against current main, add
        whatever is missing, then finalize. The architect may also discard the
        partial entirely (``create_plan`` overwrites it) if it reflects a
        flawed approach.
        """
        if context is None:
            context = await self._get_memory_context(task.get('id'))

        task_block = self._format_task(task)
        identity = self._agent_identity(task.get('id'), 'architect')

        existing_steps = [
            s for s in partial_plan.get('steps', []) if isinstance(s, dict)
        ]
        plan_json = json.dumps(partial_plan, indent=2)

        return f"""\
{context}

{identity}

# Task

{task_block}

# Plan Completion — Finish an Interrupted Plan

A previous planning session for this task was cut short (it ran out of budget
or turns) before it finalized the plan. It left a PARTIAL plan on disk with
{len(existing_steps)} step(s) already recorded. Your job is to finish it — not
to start over from nothing.

## Partial Plan (already on disk)

```json
{plan_json}
```

## Action

1. Read the relevant files to re-ground yourself in the task and verify the
   existing steps are still correct against the CURRENT state of the codebase
   (main may have advanced since the partial was written).
2. Then take ONE of these paths:
   a. **Partial is sound, just unfinished** — keep the existing steps and add
      the remaining ones with `add_plan_step(...)` (continue the existing TDD
      ordering and step-id sequence), adjusting `files` via
      `update_plan_metadata(files=...)` if the scope grew. Fix individual steps
      with `replace_plan_step` / `remove_plan_step` as needed.
   b. **Partial reflects a flawed approach** — call `create_plan(...)` to
      replace it wholesale, then add steps as usual.
   c. **The task should not be planned at all** — use the rejection exits
      (`report_blocking_dependency` / `report_task_already_done` /
      `report_ready_to_merge` / `report_unactionable_task`) exactly as in
      fresh planning.

   If any existing or newly-added step is already satisfied by a commit this
   branch carries — confirmed by running that commit's tests and seeing them
   pass — call `mark_step_committed(step_id, <sha>)` for it instead of
   leaving it `pending`; do NOT drop or merge steps just because the work
   exists.
3. If you produced a plan (paths a or b), call `confirm_plan()` as your FINAL
   action to mark it complete. Without it the plan stays incomplete and will
   not advance.
4. Keep the `files` list exhaustive — it drives concurrency locks. Do NOT
   remove or replace steps with status "done".
"""

    async def build_plan_tightening_prompt(
        self,
        task: dict,
        plan: dict,
        not_touched: list[str],
        worktree: Path | None = None,  # noqa: ARG002 — interface symmetry
        context: str | None = None,
    ) -> str:
        """Architect narrowing pass after the pre-merge plan-files gate.

        The merge gate flagged ``not_touched``: plan-declared files that
        no commit on the branch actually touched.  Give the architect
        ONE bounded chance to drop genuinely-unneeded entries via
        ``update_plan_metadata(files=[narrowed list])``.

        Lenient semantics — the architect may keep some flagged entries
        (treating them as genuinely needed; the gate's re-check is then
        the source of truth).  The only hard constraint is no NEW files:
        a post-pass subset check rejects any plan that added entries
        beyond the current ``plan.files`` set.
        """
        if context is None:
            context = await self._get_memory_context(task.get('id'))

        task_block = self._format_task(task)
        identity = self._agent_identity(task.get('id'), 'architect')

        current_files = list(plan.get('files', []))
        files_list = '\n'.join(f'- `{f}`' for f in current_files) or '_(empty)_'
        not_touched_list = '\n'.join(f'- `{f}`' for f in not_touched) or '_(none)_'

        return f"""\
{context}

{identity}

# Task

{task_block}

# Plan Tightening — Architect Narrowing Pass

The pre-merge gate flagged that some files you declared in this task's
plan were never touched by any commit on the branch.  Before the merge
can proceed, please narrow the plan against current branch state — drop
entries that turned out not to be needed, or confirm the plan if the
work is genuinely incomplete.

## Current plan files

{files_list}

## Flagged — declared but not touched on the branch

{not_touched_list}

## Action — choose exactly ONE

a. **Drop genuinely-unneeded entries**: call
   `update_plan_metadata(files=[<narrowed list>])` with a subset of the
   current plan files.  You may keep some flagged entries if you judge
   them genuinely needed; the gate's re-check is the source of truth.
b. **Plan is honest as-is**: call `confirm_plan()` unchanged.  The
   workflow will then file a level-1 escalation (auto-watcher triages; promotes to L2 if a human is needed) — choose this only when the
   work is genuinely incomplete and the flagged files really do need
   edits.

## Forbidden for this pass

Do not call `create_plan`, `add_plan_step`, or `replace_plan_step`.
This narrowing pass is scoped strictly to the `files` list; step
rewrites belong to a separate revalidation flow.

You must NOT add new files to the plan: the post-pass verifier rejects
any plan whose `files` list contains entries beyond the current set
above.  If the work needs new files, call `confirm_plan()` instead and
let a human triage the scope change.
"""

    async def build_simple_task_prompt(
        self,
        task: dict,
        worktree: Path | None = None,
        context: str | None = None,
    ) -> str:
        """Build prompt for the SIMPLE_TASK agent (Lever C).

        The agent will explore briefly, register a single-step plan via the
        plan-tools MCP server, edit the listed files, commit, then call
        ``mark_step_done`` and stop. The orchestrator then advances to VERIFY
        without invoking the implementer.
        """
        if context is None:
            context = await self._get_memory_context(task.get('id'))

        task_block = self._format_task(task)
        identity = self._agent_identity(task.get('id'), 'simple_task')

        files = (task.get('metadata') or {}).get('files') or []
        if files:
            files_list = '\n'.join(f'- `{f}`' for f in files)
            files_section = f'## Listed files\n\n{files_list}'
        else:
            files_section = '_No files listed in task metadata — explore briefly to identify the target file(s)._'

        # The stop-criterion prose below ("no new abstraction, no cross-module design,
        # substantial architectural thought") mirrors _COMPLEXITY_RUBRIC in roles.py.
        # Update both in lockstep if the rubric changes.
        return f"""\
{context}

{identity}

# Task

{task_block}

# Action

This task was routed to the SIMPLE_TASK path because its author declared
complexity:simple. A simple task may be high-priority and may span several
files/modules — the declaration means the *change* is mechanically simple,
not that the task is trivial. Your job is to do the change end-to-end in a
single session:

1. **Read the listed files** below. Confirm the change is mechanically
   simple — no new abstraction, no cross-module design required.
2. **Register a plan via plan-tools MCP** — call
   `mcp__plan-tools__create_plan(task_id, title, analysis, files)`.
   - For doc/comment-only or behaviour-preserving refactors: add a single
     `impl` step via `add_plan_step`.
   - If a new test belongs here: add `test` then `impl`.
3. **Implement** — edit the file(s), run any tests touching the module,
   commit (excluding `.task/`), then call
   `mcp__plan-tools__mark_step_done(step_id, commit_sha)`.
4. **Stop after marking done.** Do not loop further.

If the change turns out to need cross-module design, a new abstraction, or
substantial architectural thought, STOP without calling `create_plan` — do
NOT stop merely because the change spans several files/modules. The
orchestrator will route to the full architect path on the next dispatch.

If the task spec itself is broken or unworkable, call
`mcp__plan-tools__report_unactionable_task(reason, evidence)` and stop.

If instead the work is already complete and green on THIS BRANCH and only
the merge to main is missing — a clean fast-forward (main is an ancestor of
the branch tip and the tip is not already contained in main), verify already
PASSED on this exact tip, and review already returned PASS or
suggestions-only on this exact tree — call
`mcp__plan-tools__report_ready_to_merge(commit, evidence)` instead of
`report_unactionable_task` and stop.

{files_section}
"""

    async def build_implementer_prompt(
        self,
        plan: dict,
        iteration_log: list[dict],
        context: str | None = None,
        rebase_notice: dict | None = None,
        task_id: str | None = None,
        wip_notice: list[dict] | None = None,
    ) -> str:
        """Build prompt for the implementer agent."""
        effective_tid = task_id or plan.get('task_id')
        if context is None:
            context = await self._get_memory_context(effective_tid)

        identity = self._agent_identity(effective_tid, 'implementer')

        completed = [s for s in plan.get('steps', []) if isinstance(s, dict) and s.get('status') == 'done']
        pending = [s for s in plan.get('steps', []) if isinstance(s, dict) and s.get('status') == 'pending']
        pre_completed = [s for s in plan.get('prerequisites', []) if isinstance(s, dict) and s.get('status') == 'done']
        pre_pending = [s for s in plan.get('prerequisites', []) if isinstance(s, dict) and s.get('status') == 'pending']

        log_summary = ''
        if iteration_log:
            recent = iteration_log[-3:]
            log_lines = []
            for entry in recent:
                log_lines.append(
                    f"- Iteration {entry.get('iteration', '?')}: "
                    f"completed {entry.get('steps_completed', [])}, "
                    f"summary: {entry.get('summary', 'N/A')}"
                )
            log_summary = "## Recent Iterations\n\n" + '\n'.join(log_lines)

        rebase_section = ''
        if rebase_notice:
            files_list = '\n'.join(
                f'- `{f}`' for f in rebase_notice['changed_files'][:30]
            )
            rebase_section = f"""
## Rebase Notice

Your worktree was rebased onto the latest main branch.

- **Previous base:** `{rebase_notice['old_base'][:12]}`
- **New base:** `{rebase_notice['new_base'][:12]}`
- **Files changed on main since last base:**

{files_list}

Review any overlap with your plan steps before continuing — file contents may have changed.
"""

        wip_section = ''
        if wip_notice:
            # No limit: _detect_tip_wip_commits is bounded by the contiguous
            # WIP run at HEAD, unlike the architect's whole-branch detector.
            commits_list = _format_commit_bullets(wip_notice)
            wip_section = f"""
## Already-Committed WIP — Verify Before Re-Implementing

The harness auto-commits uncommitted work as a safety net before a
rebase/requeue/reclaim. The commit(s) below landed at branch HEAD this way,
which means the next pending step's implementation may already be sitting
there, complete, waiting on `mark_step_done`.

{commits_list}

Before writing any new code:

1. Run `git show <sha>` for each commit above to see what it contains.
2. Run the next pending step's tests.
3. If they already pass, call `mark_step_done(step_id, commit_sha)` with the
   WIP commit's SHA instead of re-implementing the step.
4. Only write new code if the step is genuinely unsatisfied by this commit.
"""

        return f"""\
{context}

{identity}

# Plan Overview

**Task:** {plan.get('title', 'Unknown')}
**Analysis:** {plan.get('analysis', 'N/A')}

## Progress

- Prerequisites: {len(pre_completed)} done, {len(pre_pending)} pending
- Steps: {len(completed)} done, {len(pending)} pending

{log_summary}
{rebase_section}
{wip_section}
# Session Startup Protocol

1. Read `.task/plan.json` to see the full plan with current status — it is a
   symlink into the durable
   `<worktree_base>/.task-meta/<worktree-name>/plan.json` (the
   worktree-lane-lifecycle W11 relocation keeps plan/iteration state outside
   the worktree for pooled lanes, and the symlink survives worktree resets),
   so reading either path resolves to the same plan.
2. Read `<worktree_base>/.task-meta/<worktree-name>/iterations.jsonl` (NOT
   symlinked into the lane) to see prior iteration details.
3. Run `git log --oneline -10` to see recent commits.
4. Identify the next pending step (prerequisites first, then steps).
5. **Mandatory pre-flight — before writing any code for that step:** run
   `git status` and `git diff HEAD` on the step's target files. A prior
   iteration may have already written — or fully implemented — this
   step's code and then crashed or ran out of context before staging,
   committing, or calling `mark_step_done`. That work shows up here as an
   uncommitted diff in the working tree, not as a WIP safety-commit (that
   narrower case is covered separately above when detected). If the diff
   already satisfies the step's spec, run its tests; if they pass, commit
   the existing work and call `mark_step_done(step_id, commit_sha)`
   instead of re-implementing from scratch. Only write new code if the
   step is genuinely unsatisfied by what's already on disk.

# Action

Execute the next pending steps in TDD order. Commit after each step. Call `mark_step_done(step_id, commit_sha)` to record progress. Stop at a logical boundary.
"""

    async def build_amender_prompt(
        self,
        plan: dict,
        iteration_log: list[dict],
        suggestions: list[dict],
        locked_modules: list[str],
        context: str | None = None,
        task_id: str | None = None,
    ) -> str:
        """Build prompt for an amendment pass — implementer applies in-scope
        review suggestions without re-planning.

        ``suggestions`` is the pre-filtered in-scope list (already restricted
        to files inside ``locked_modules``). ``locked_modules`` is listed in
        the prompt so the agent can self-check scope before editing.
        """
        effective_tid = task_id or plan.get('task_id')
        if context is None:
            context = await self._get_memory_context(effective_tid)

        identity = self._agent_identity(effective_tid, 'implementer')

        log_summary = ''
        if iteration_log:
            recent = iteration_log[-3:]
            log_lines = []
            for entry in recent:
                log_lines.append(
                    f"- Iteration {entry.get('iteration', '?')} "
                    f"[{entry.get('agent', '?')}]: "
                    f"{entry.get('summary', 'N/A')}"
                )
            log_summary = "## Recent Iterations\n\n" + '\n'.join(log_lines)

        modules_list = '\n'.join(f'- `{m}`' for m in sorted(locked_modules))

        suggestion_blocks = []
        for i, s in enumerate(suggestions, 1):
            reviewer = s.get('reviewer', 'unknown')
            category = s.get('category', '')
            location = s.get('location', '')
            description = s.get('description', '')
            fix = s.get('suggested_fix', '')
            block = [
                f'### {i}. [{reviewer}] {category}',
            ]
            if location:
                block.append(f'**Location:** `{location}`')
            block.append(f'**Issue:** {description}')
            if fix:
                block.append(f'**Suggested fix:** {fix}')
            suggestion_blocks.append('\n'.join(block))
        suggestions_body = '\n\n'.join(suggestion_blocks)

        return f"""\
{context}

{identity}

# Amendment Pass

The implementation for this task is complete and verification has passed.
A code reviewer surfaced the suggestions below, all scoped to modules this
task already holds locks for. Your job is to apply them as focused
amendments — small edits that address each point without re-planning or
expanding the task's concurrency footprint.

## Plan Overview

**Task:** {plan.get('title', 'Unknown')}
**Analysis:** {plan.get('analysis', 'N/A')}

{log_summary}

## Scope Discipline

This task holds locks for the following modules:

{modules_list}

1. Work ONLY inside these locked modules. Creating new files inside them is
   allowed; editing files outside them is NOT.
2. Do NOT modify the plan — it is durable at
   `<worktree_base>/.task-meta/<worktree-name>/plan.json` (the lane
   `.task/plan.json` is a symlink into it) and frozen for this pass.
3. If a suggestion requires touching a file outside the locked modules,
   skip it and note the reason in your commit message — it will be
   re-surfaced by the next review cycle or escalated as a follow-up task.
4. Prefix amendment commit messages with `amend:` so they're distinguishable
   from the main plan commits.

## Suggestions to Address

{suggestions_body}

# Action

1. Read `.task/plan.json` (a symlink into the durable
   `<worktree_base>/.task-meta/<worktree-name>/plan.json`, which survives
   worktree resets) and
   `<worktree_base>/.task-meta/<worktree-name>/iterations.jsonl` (NOT
   symlinked into the lane) to refresh context on what was already done.
2. Run `git log --oneline -10` to see recent commits.
3. Apply each in-scope suggestion above. Commit amendments with `amend:`
   prefixes, grouping related fixes when sensible.
4. Run verification for the touched files before finishing.
{WAIT_PATTERN_REMINDER}
"""

    async def build_debugger_prompt(
        self, failures: str, plan: dict, context: str | None = None,
        task_id: str | None = None,
    ) -> str:
        """Build prompt for the debugger agent."""
        effective_tid = task_id or plan.get('task_id')
        if context is None:
            context = await self._get_memory_context(effective_tid)

        identity = self._agent_identity(effective_tid, 'debugger')

        return f"""\
{context}

{identity}

# Task Context

**Task:** {plan.get('title', 'Unknown')}
**Analysis:** {plan.get('analysis', 'N/A')}

# Failures

```
{failures}
```

# Action

1. Analyze the root cause of each failure.
2. Make minimal, targeted fixes.
3. Run the verification commands to confirm fixes.
4. Commit your fixes.
"""

    async def build_reviewer_prompt(
        self, reviewer_type: str, diff: str, context: str | None = None,
        *, amendment_suggestions: list[dict] | None = None,
    ) -> str:
        """Build prompt for a reviewer agent.

        When *amendment_suggestions* is provided, this review immediately
        follows an in-workflow amendment round; an advisory "# Amendment
        Re-Review Scope" section is appended constraining the reviewer to
        verify those prior suggestions were addressed and to report only new
        findings within the amendment delta (task 2750).  When it is ``None``
        the returned prompt is byte-identical to the non-amendment path.  The
        advisory section is best-effort — the deterministic
        ``partition_suggestions_by_delta`` filter is the enforceable guarantee.
        """
        if context is None:
            context = await self._get_memory_context()

        # Truncate very large diffs to avoid blowing the context
        if len(diff) > 50000:
            diff = diff[:50000] + '\n\n... [diff truncated] ...'

        prompt = f"""\
{context}

# Code Diff to Review

```diff
{diff}
```

# Action

1. Review the diff according to your specialization. Explore the codebase as needed for context.
2. Call `submit_review_verdict(reviewer="{reviewer_type}", verdict=..., issues=..., summary=...)` with your findings.

Your verdict is read from the `submit_review_verdict` tool call, not from your prose output — you MUST call it before finishing.
"""

        if amendment_suggestions:
            listed = '\n'.join(
                f"- {s.get('location') or '(no location)'} — "
                f"{s.get('description') or ''}"
                for s in amendment_suggestions
            )
            prompt += f"""
# Amendment Re-Review Scope

This diff was just amended to address the prior review's in-scope suggestions,
listed below. Do NOT hunt for fresh, unrelated nits across the full diff:
first confirm the listed prior suggestions were addressed, then report ONLY
new findings introduced within the amendment delta — the lines this amendment
actually changed. Blocking regressions anywhere in the diff remain in scope.

Prior suggestions the amendment was asked to address:
{listed}
"""

        return prompt

    async def build_completion_judge_prompt(
        self,
        plan: dict,
        iteration_log: list[dict],
        diff: str,
        task_id: str | None = None,
        context: str | None = None,
    ) -> str:
        """Build prompt for the completion judge agent."""
        effective_tid = task_id or plan.get('task_id')
        if context is None:
            context = await self._get_memory_context(effective_tid)

        identity = self._agent_identity(effective_tid, 'judge')

        # Truncate diff (same cap as reviewer)
        if len(diff) > 50000:
            diff = diff[:50000] + '\n\n... [diff truncated] ...'

        # Last 5 iteration log entries (reviewer uses 3; judge benefits from
        # seeing more of the arc of work)
        log_section = ''
        if iteration_log:
            recent = iteration_log[-5:]
            lines = [
                f"- iter {e.get('iteration', '?')} [{e.get('agent', '?')}]: "
                f"{e.get('summary', 'N/A')}"
                for e in recent
            ]
            log_section = "## Recent Iterations\n\n" + '\n'.join(lines)

        # Serialize only the plan fields the judge needs
        plan_json = json.dumps({
            'task_id': plan.get('task_id'),
            'title': plan.get('title'),
            'analysis': plan.get('analysis'),
            'prerequisites': plan.get('prerequisites', []),
            'steps': plan.get('steps', []),
        }, indent=2)

        return f"""\
{context}

{identity}

# Plan

```json
{plan_json}
```

{log_section}

# Code Diff (worktree vs pre-task base)

```diff
{diff}
```

# Action

Read the code in the worktree as needed to verify behavior. Then return
your verdict as JSON matching the schema. Follow the safety rules: if the
diff is empty or trivial, `substantive_work=false` and `complete=false`.
"""

    async def build_merger_prompt(
        self, conflicts: str, task_intent: str, context: str | None = None
    ) -> str:
        """Build prompt for the merger agent."""
        if context is None:
            context = await self._get_memory_context()

        return f"""\
{context}

# Task Intent

{task_intent}

# Merge Conflicts

{conflicts}

# Action

1. Read both sides of each conflict carefully.
2. Understand the intent of each change.
3. Resolve conflicts conservatively — preserve both sides' intent.
4. Run tests to verify the resolution.
5. If you cannot confidently resolve, call `submit_merge_disposition(blocked=true, reason="<why>")` and stop.
6. Once you have resolved, tested, and committed successfully, call `submit_merge_disposition(blocked=false, reason="")`.

Your disposition is read from the `submit_merge_disposition` tool call, not from your prose output — you MUST call it before finishing.
"""

    async def build_resume_prompt(
        self,
        task: dict,
        plan: dict,
        escalation_summary: str,
        resolution: str,
        worktree: Path | None = None,
    ) -> str:
        """Build prompt for resuming after an escalation resolution."""
        context = await self._get_memory_context(task.get('id'))
        prior_proposal_section = self._format_prior_proposal(task)

        return f"""\
{context}

# Resuming After Escalation

This task was paused because an agent escalated a blocking issue.

## The Issue
{escalation_summary}

## Handler's Resolution
{resolution}

{prior_proposal_section}
## Action
Resume the task applying the handler's resolution. The prior agent's work
is preserved in the worktree. Read `.task/plan.json` (a symlink into the
durable `<worktree_base>/.task-meta/<worktree-name>/plan.json`, which survives
worktree resets) and `<worktree_base>/.task-meta/<worktree-name>/iterations.jsonl`
(NOT symlinked into the lane) to understand current progress, then continue
from where the previous agent left off.
"""

    async def build_steward_initial_prompt(
        self,
        task: dict,
        escalation: dict,
        pending_escalations: list[dict],
        worktree: Path,
    ) -> str:
        """Build full briefing prompt for the steward's first invocation.

        Includes memory context, task details, escalation info, and action
        instructions.  Used for the initial session and after cap-hit resets.
        """
        context = await self._get_memory_context(task.get('id'))
        identity = self._agent_identity(task.get('id'), 'steward')
        task_block = self._format_task(task)
        esc_block = self._format_escalation(escalation)

        pending_block = ''
        other_pending = [e for e in pending_escalations if e.get('id') != escalation.get('id')]
        if other_pending:
            items = '\n'.join(
                f'- `{e.get("id")}` [{e.get("category")}]: {e.get("summary")}'
                for e in other_pending
            )
            pending_block = f'\n## Other Pending Escalations\n\n{items}\n'

        return f"""\
{context}

{identity}

# Task

{task_block}

# Escalation

{esc_block}
{pending_block}
# Parameters

- **project_id:** `{self.project_id}`
- **project_root:** `{self.config.project_root}`
- **worktree:** `{worktree}`

# Action

1. Understand the escalation and the task context.
2. Check whether this task's branch is already merged to main (`git merge-base --is-ancestor HEAD main` from the worktree, or `git log --oneline main | head -20`). If the branch is already on main, set the task status to `done` via fused-memory's `set_task_status` tool — **`done_provenance` must always include `kind`**: pass `done_provenance={{"kind": "found_on_main", "commit": "<landing-sha-on-main>", "note": "<one-sentence explanation>"}}` (both `commit` and `note` are required for this kind — there is no commit-less fallback), e.g. note "covered by sibling task" or "already merged prior to this session". If there is no distinct merge commit to cite (e.g. a fast-forward merge), use the branch's own tip commit SHA as `commit` — after a fast-forward that SHA becomes `main`'s HEAD directly — and say so in `note` (e.g. "fast-forward merge, no separate merge commit"). If the landing commit came from this session calling `merge_request`, use `done_provenance={{"kind": "merged", "commit": "<merge-sha>"}}` instead. Then call `resolve_issue` explaining the task was already merged. Do NOT attempt to fix code or re-merge.
3. Read the relevant code.
4. Handle the escalation — fix the issue, or triage suggestions.
5. Run tests to verify any code changes.
6. Call `resolve_issue` with a summary of what you did.
"""

    async def build_steward_continuation_prompt(
        self,
        task: dict,
        escalation: dict,
    ) -> str:
        """Build a minimal prompt for resuming the steward session.

        The session already has full context from the initial briefing,
        so this just provides the new escalation details.
        """
        esc_block = self._format_escalation(escalation)

        return f"""\
# New Escalation for Task {task.get('id', '?')} — {task.get('title', '')}

{esc_block}

Handle this escalation, then call `resolve_issue` with a summary.
"""

    @staticmethod
    def _format_escalation(escalation: dict) -> str:
        """Format an escalation dict (blocking or suggestion) into markdown."""
        lines = [
            f'- **ID:** `{escalation.get("id", "?")}`',
            f'- **Category:** {escalation.get("category", "unknown")}',
            f'- **Severity:** {escalation.get("severity", "unknown")}',
            f'- **Summary:** {escalation.get("summary", "N/A")}',
        ]
        if escalation.get('detail'):
            lines.append(f'- **Detail:** {escalation["detail"]}')
        if escalation.get('suggested_action'):
            lines.append(f'- **Suggested action:** {escalation["suggested_action"]}')
        return chr(10).join(lines)

    async def _get_memory_context(self, task_id: str | None = None) -> str:
        """Call fused-memory search for project context."""
        recalled_sections: list[str] = []
        foreign_dropped = 0
        queries_fired = 0
        memory_unavailable = False

        try:
            # Project overview
            overview, dropped = await self._scoped_search('project overview architecture goals')
            foreign_dropped += dropped
            queries_fired += 1
            if overview:
                recalled_sections.append(f'## Project Context\n\n{overview}')

            # Conventions
            conventions, dropped = await self._scoped_search('coding conventions and project norms')
            foreign_dropped += dropped
            queries_fired += 1
            if conventions:
                recalled_sections.append(f'## Conventions\n\n{conventions}')

            # Recent decisions
            decisions, dropped = await self._scoped_search('recent decisions and rationale')
            foreign_dropped += dropped
            queries_fired += 1
            if decisions:
                recalled_sections.append(f'## Recent Decisions\n\n{decisions}')

            # Task-specific context
            if task_id:
                task_ctx, dropped = await self._scoped_search(
                    f'task {task_id} context and related decisions'
                )
                foreign_dropped += dropped
                queries_fired += 1
                if task_ctx:
                    recalled_sections.append(f'## Task Context\n\n{task_ctx}')

        except Exception as e:
            logger.warning(f'Failed to fetch memory context: {e}')
            memory_unavailable = True

        # Compute (and log) the filtered-result summary BEFORE any early
        # return below: an all-foreign result set and a partial failure are
        # both "no facts survived" outcomes, and the fact that a leak was
        # caught and blocked must never be discarded along with them — see
        # filter_foreign_project_results' loud-over-silent fail-open stance.
        # foreign_dropped sums per-query drops over the SAME corpus (four
        # queries can all match one distinct foreign memory), so the note
        # names both numbers rather than implying `foreign_dropped` distinct
        # facts were found.
        drop_note = ''
        if foreign_dropped > 0:
            query_word = 'query' if queries_fired == 1 else 'queries'
            drop_note = (
                f'{foreign_dropped} memory result slot(s) across {queries_fired} '
                f'{query_word} were tagged to another project and filtered out'
            )
            logger.info(
                f'_get_memory_context: {drop_note} of the context assembled for '
                f'{self.project_id!r}'
            )

        if not recalled_sections:
            if memory_unavailable:
                if drop_note:
                    return (
                        '# Context\n\n_Memory unavailable — proceed with codebase '
                        f'exploration. Note: {drop_note} before the failure._'
                    )
                return '# Context\n\n_Memory unavailable — proceed with codebase exploration._'
            if drop_note:
                return f'# Context\n\n_No memory context available ({drop_note})._'
            return '# Context\n\n_No memory context available._'

        # recalled_sections is non-empty: gate the provenance caveat on that
        # fact alone, NOT on memory_unavailable — a later query failing must
        # not suppress the caveat (the untagged-leak mitigation) for
        # sections that were already genuinely recalled. The failure, if
        # any, is appended afterwards as its own section rather than
        # silently dropped.
        caveat = MEMORY_CONTEXT_CAVEAT.format(project_id=self.project_id)
        if drop_note:
            caveat += f'\n\n_In total, {drop_note}._'

        rendered_sections = list(recalled_sections)
        if memory_unavailable:
            rendered_sections.append(
                '_Memory unavailable for the remaining queries — proceed with '
                'codebase exploration for anything not covered above._'
            )

        return '# Context\n\n' + caveat + '\n\n' + '\n\n---\n\n'.join(rendered_sections)

    async def _scoped_search(self, query: str) -> tuple[str | None, int]:
        """Search fused-memory and drop cross-project results from the reply.

        Thin wrapper over the UNCHANGED :meth:`_mcp_search` — never touches
        which queries fire or their ``limit`` (task 3253 owns that
        adjudication) — that applies :func:`filter_foreign_project_results`
        to the raw text before it reaches :meth:`_get_memory_context`.
        Returns ``(None, 0)`` when the underlying search itself returned
        nothing (nothing to filter).

        Assumes :meth:`_mcp_search` answers with a single JSON document: it
        joins every MCP response text block with ``'\\n'`` before returning
        (unchanged by this task, to keep its silent-fallthrough allowlist
        entry valid). If the search tool ever replies with more than one
        text block, the joined text is not valid JSON and the filter fails
        open (unfiltered, WARNING logged) for that query — see
        ``test_briefing_project_scope.py``'s ``TestScopedSearch`` for the
        pinned limitation.
        """
        raw = await self._mcp_search(query)
        if not raw:
            return None, 0
        return filter_foreign_project_results(raw, self.project_id)

    async def _mcp_search(self, query: str) -> str | None:
        """Search fused-memory via its MCP HTTP endpoint."""
        try:
            result = await mcp_call(
                f'{self.memory_url}/mcp',
                'tools/call',
                {
                    'name': 'search',
                    'arguments': {
                        'query': query,
                        'project_id': self.project_id,
                        'limit': 5,
                    },
                },
                timeout=10,
            )
            content = result.get('result', {}).get('content', [])
            texts = []
            for block in content:
                if isinstance(block, dict) and block.get('type') == 'text':
                    texts.append(block['text'])
            return '\n'.join(texts) if texts else None

        except Exception as e:
            logger.debug(f'MCP search failed for "{query}": {e}')
            return None

    def _format_prior_proposal(self, task: dict) -> str:
        """Format the most recent dry-run block-time proposal, if any.

        Reads ``task.metadata.dry_run_proposals[-1]`` defensively (None-safe
        at every level) and renders it as a markdown block carrying an
        explicit provenance/verification line — the reader must not assume a
        persisted proposal still holds against the current tree. Returns ''
        when there is no proposal, the latest entry is not a dict, or the
        proposal predates the task's last block transition
        (``metadata.last_blocked_at``).

        The staleness comparison fails OPEN (i.e. includes the proposal)
        whenever either timestamp is absent or fails to parse via
        ``datetime.fromisoformat`` — including a naive-vs-aware mismatch,
        which raises ``TypeError`` on comparison — so a formatting hiccup
        never silently drops persisted analysis.

        Called only from retry/resume prompt builders
        (``build_revalidation_prompt``, ``build_resume_prompt``) and, behind
        ``include_prior_proposals=True``, from ``build_architect_prompt``'s
        re-plan path — NEVER unconditionally from ``_format_task``, which
        would leak proposals into the first-dispatch anti-anchoring path
        (C-A1).
        """
        proposals = (task.get('metadata') or {}).get('dry_run_proposals') or []
        if not proposals:
            return ''
        proposal = proposals[-1]
        if not isinstance(proposal, dict):
            return ''

        proposal_text = proposal.get('proposal_text', '')
        risk_label = proposal.get('risk_label', '')
        files_referenced = proposal.get('files_referenced') or []
        created_at = proposal.get('timestamp') or proposal.get('investigated_at') or ''

        last_blocked_at = (task.get('metadata') or {}).get('last_blocked_at')
        if created_at and last_blocked_at:
            try:
                is_stale = datetime.fromisoformat(created_at) < datetime.fromisoformat(last_blocked_at)
            except (ValueError, TypeError):
                pass  # fail open — never silently drop persisted analysis
            else:
                if is_stale:
                    return ''

        # Coerce defensively: files_referenced is persisted, untyped data, so
        # a stray non-string element must never crash the prompt build — the
        # same fail-open spirit as the timestamp comparison above. Deferred
        # until after the staleness early-return so a stale/omitted proposal
        # skips the work entirely.
        files_str = ', '.join(str(f) for f in files_referenced)

        return f"""\
## Prior Block-Time Investigation

A prior block-time investigation concluded the following; verify against the current tree before reusing — do NOT assume it still holds:

**Proposal:** {proposal_text}
**Risk:** {risk_label}
**Files referenced:** {files_str}
**Investigated at:** {created_at}
"""

    def _format_task(self, task: dict, *, include_files: bool = True) -> str:
        """Format a task dict as readable text.

        Args:
            task: Task dict with id, title, description, metadata, etc.
            include_files: When False, the ``metadata.files`` line is omitted
                from the output.  Defaults to True so all existing callers are
                unaffected.  Pass ``include_files=False`` from
                ``build_architect_prompt`` to anti-anchor the first plan
                derivation (C-A1): the architect must derive its own file
                footprint rather than echoing the queue-time metadata guess.
        """
        lines = []
        if task.get('id'):
            lines.append(f'**ID:** {task["id"]}')
        if task.get('title'):
            lines.append(f'**Title:** {task["title"]}')
        if task.get('description'):
            lines.append(f'**Description:** {task["description"]}')
        if task.get('details'):
            lines.append(f'**Details:** {task["details"]}')
        if include_files and task.get('metadata', {}).get('files'):
            lines.append(f'**Files:** {", ".join(task["metadata"]["files"])}')
        deps = task.get('dependencies', [])
        if deps:
            dep_ids = [str(d.get('id', d)) if isinstance(d, dict) else str(d) for d in deps]
            lines.append(f'**Dependencies:** {", ".join(dep_ids)}')
        return '\n'.join(lines) if lines else json.dumps(task, indent=2)
