"""Prompt assembly — builds full prompts for each agent invocation."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from orchestrator.config import OrchestratorConfig
from orchestrator.mcp_lifecycle import mcp_call

logger = logging.getLogger(__name__)


class BriefingAssembler:
    """Builds prompts for agent invocations."""

    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.memory_url = config.fused_memory.url
        self.project_id = config.fused_memory.project_id

    async def build_architect_prompt(
        self, task: dict, worktree: Path | None = None, context: str | None = None
    ) -> str:
        """Build prompt for the architect agent."""
        if context is None:
            context = await self._get_memory_context(task.get('id'))

        task_block = self._format_task(task)

        # Use absolute path so the agent's Write tool targets the correct location
        if worktree:
            plan_path = str(worktree / '.task' / 'plan.json')
        else:
            plan_path = '.task/plan.json'

        return f"""\
{context}

# Task

{task_block}

# Action

1. Explore the codebase thoroughly — read relevant files, understand existing patterns and utilities.
2. Produce a TDD implementation plan.
3. Ensure the `modules` field accurately lists ALL code directories this task will touch.
4. Write the plan using the Write tool to `{plan_path}`.
"""

    async def build_implementer_prompt(
        self, plan: dict, iteration_log: list[dict], context: str | None = None
    ) -> str:
        """Build prompt for the implementer agent."""
        if context is None:
            context = await self._get_memory_context(plan.get('task_id'))

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

        return f"""\
{context}

# Plan Overview

**Task:** {plan.get('title', 'Unknown')}
**Analysis:** {plan.get('analysis', 'N/A')}

## Progress

- Prerequisites: {len(pre_completed)} done, {len(pre_pending)} pending
- Steps: {len(completed)} done, {len(pending)} pending

{log_summary}

# Session Startup Protocol

1. Read `.task/plan.json` to see the full plan with current status.
2. Read `.task/iterations.jsonl` to see prior iteration details.
3. Run `git log --oneline -10` to see recent commits.
4. Identify the next pending step (prerequisites first, then steps).

# Action

Execute the next pending steps in TDD order. Commit after each step. Update plan.json status fields. Stop at a logical boundary and log your iteration.
"""

    async def build_debugger_prompt(
        self, failures: str, plan: dict, context: str | None = None
    ) -> str:
        """Build prompt for the debugger agent."""
        if context is None:
            context = await self._get_memory_context(plan.get('task_id'))

        return f"""\
{context}

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
5. Log your work in `.task/iterations.jsonl`.
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
