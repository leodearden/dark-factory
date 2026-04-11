"""Prompt assembly — builds full prompts for each agent invocation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from orchestrator.config import OrchestratorConfig
from orchestrator.mcp_lifecycle import mcp_call

logger = logging.getLogger(__name__)


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
        self, task: dict, worktree: Path | None = None, context: str | None = None
    ) -> str:
        """Build prompt for the architect agent."""
        if context is None:
            context = await self._get_memory_context(task.get('id'))

        task_block = self._format_task(task)
        identity = self._agent_identity(task.get('id'), 'architect')

        # Use absolute path so the agent's Write tool targets the correct location
        plan_path = str(worktree / '.task' / 'plan.json') if worktree else '.task/plan.json'

        return f"""\
{context}

{identity}

# Task

{task_block}

# Action

1. Explore the codebase thoroughly — read relevant files, understand existing patterns and utilities.
2. Produce a TDD implementation plan.
3. List ALL files you expect to create or modify in the `files` field — this drives concurrency locks, so be exhaustive and precise.
4. Ensure the `modules` field accurately lists ALL code directories this task will touch.
5. Write the plan using the Write tool to `{plan_path}`.
"""

    async def build_implementer_prompt(
        self,
        plan: dict,
        iteration_log: list[dict],
        context: str | None = None,
        rebase_notice: dict | None = None,
        task_id: str | None = None,
    ) -> str:
        """Build prompt for the implementer agent."""
        effective_tid = task_id or plan.get('task_id')
        if context is None:
            context = await self._get_memory_context(effective_tid)

        identity = self._agent_identity(effective_tid, 'implementer')

        completed = [s for s in plan.get('steps', []) if s.get('status') == 'done']
        pending = [s for s in plan.get('steps', []) if s.get('status') == 'pending']
        pre_completed = [s for s in plan.get('prerequisites', []) if s.get('status') == 'done']
        pre_pending = [s for s in plan.get('prerequisites', []) if s.get('status') == 'pending']

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
# Session Startup Protocol

1. Read `.task/plan.json` to see the full plan with current status.
2. Read `.task/iterations.jsonl` to see prior iteration details.
3. Run `git log --oneline -10` to see recent commits.
4. Identify the next pending step (prerequisites first, then steps).

# Action

Execute the next pending steps in TDD order. Commit after each step. Update plan.json status fields. Stop at a logical boundary.
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
2. Do NOT modify `.task/plan.json`. The plan is frozen for this pass.
3. If a suggestion requires touching a file outside the locked modules,
   skip it and note the reason in your commit message — it will be
   re-surfaced by the next review cycle or escalated as a follow-up task.
4. Prefix amendment commit messages with `amend:` so they're distinguishable
   from the main plan commits.

## Suggestions to Address

{suggestions_body}

# Action

1. Read `.task/plan.json` and `.task/iterations.jsonl` to refresh context on
   what was already done.
2. Run `git log --oneline -10` to see recent commits.
3. Apply each in-scope suggestion above. Commit amendments with `amend:`
   prefixes, grouping related fixes when sensible.
4. Run verification for the touched files before finishing.
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
        self, reviewer_type: str, diff: str, context: str | None = None
    ) -> str:
        """Build prompt for a reviewer agent."""
        if context is None:
            context = await self._get_memory_context()

        # Truncate very large diffs to avoid blowing the context
        if len(diff) > 50000:
            diff = diff[:50000] + '\n\n... [diff truncated] ...'

        return f"""\
{context}

# Code Diff to Review

```diff
{diff}
```

# Action

Review the diff according to your specialization. Explore the codebase as needed for context. Output your review as pure JSON.
"""

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
5. If you cannot confidently resolve, output "BLOCKED: <reason>" and stop.
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

        return f"""\
{context}

# Resuming After Escalation

This task was paused because an agent escalated a blocking issue.

## The Issue
{escalation_summary}

## Handler's Resolution
{resolution}

## Action
Resume the task applying the handler's resolution. The prior agent's work
is preserved in the worktree. Read .task/plan.json and .task/iterations.jsonl
to understand current progress, then continue from where the previous agent left off.
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
2. Check whether this task's branch is already merged to main (`git merge-base --is-ancestor HEAD main` from the worktree, or `git log --oneline main | head -20`). If the branch is already on main, set the task status to `done` via fused-memory's `set_task_status` tool, then call `resolve_issue` explaining the task was already merged. Do NOT attempt to fix code or re-merge.
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
        sections = []

        try:
            # Project overview
            overview = await self._mcp_search('project overview architecture goals')
            if overview:
                sections.append(f'## Project Context\n\n{overview}')

            # Conventions
            conventions = await self._mcp_search('coding conventions and project norms')
            if conventions:
                sections.append(f'## Conventions\n\n{conventions}')

            # Recent decisions
            decisions = await self._mcp_search('recent decisions and rationale')
            if decisions:
                sections.append(f'## Recent Decisions\n\n{decisions}')

            # Task-specific context
            if task_id:
                task_ctx = await self._mcp_search(
                    f'task {task_id} context and related decisions'
                )
                if task_ctx:
                    sections.append(f'## Task Context\n\n{task_ctx}')

        except Exception as e:
            logger.warning(f'Failed to fetch memory context: {e}')
            sections.append('## Context\n\n_Memory unavailable — proceed with codebase exploration._')

        if not sections:
            return '# Context\n\n_No memory context available._'

        return '# Context\n\n' + '\n\n---\n\n'.join(sections)

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

    def _format_task(self, task: dict) -> str:
        """Format a task dict as readable text."""
        lines = []
        if task.get('id'):
            lines.append(f'**ID:** {task["id"]}')
        if task.get('title'):
            lines.append(f'**Title:** {task["title"]}')
        if task.get('description'):
            lines.append(f'**Description:** {task["description"]}')
        if task.get('details'):
            lines.append(f'**Details:** {task["details"]}')
        if task.get('metadata', {}).get('modules'):
            lines.append(f'**Modules:** {", ".join(task["metadata"]["modules"])}')
        deps = task.get('dependencies', [])
        if deps:
            dep_ids = [str(d.get('id', d)) if isinstance(d, dict) else str(d) for d in deps]
            lines.append(f'**Dependencies:** {", ".join(dep_ids)}')
        return '\n'.join(lines) if lines else json.dumps(task, indent=2)
