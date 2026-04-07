"""Stage 2: Task-Knowledge Sync — reconcile task state against memory state."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from fused_memory.models.reconciliation import (
    ReconciliationEvent,
    StageReport,
    Watermark,
)
from fused_memory.reconciliation.agent_loop import ToolDefinition
from fused_memory.reconciliation.prompts.stage2 import STAGE2_SYSTEM_PROMPT
from fused_memory.reconciliation.stages.base import BaseStage

if TYPE_CHECKING:
    from fused_memory.reconciliation.verify import CodebaseVerifier


class TaskKnowledgeSync(BaseStage):
    """Stage 2: Reconcile tasks against memory, attach hints, fix inconsistencies."""

    verifier: CodebaseVerifier | None = None

    def get_system_prompt(self) -> str:
        return STAGE2_SYSTEM_PROMPT

    def get_tools(self) -> dict[str, ToolDefinition]:
        tools = {}
        tools.update(self._memory_read_tools())
        tools.update(self._memory_write_tools())
        tools.update(self._task_read_tools())
        tools.update(self._task_write_tools())

        if self.verifier:
            tools['verify_against_codebase'] = self._verify_tool()

        return tools

    def _verify_tool(self) -> ToolDefinition:
        verifier = self.verifier
        assert verifier is not None  # only called when verifier is set

        async def verify(claim: str, context: str = '', scope_hints: list[str] | None = None):
            result = await verifier.verify(
                claim=claim, context=context, scope_hints=scope_hints, project_id=self.project_id
            )
            return result.model_dump()

        return ToolDefinition(
            name='verify_against_codebase',
            description='Verify a factual claim against the codebase via read-only explore agent.',
            parameters={
                'type': 'object',
                'properties': {
                    'claim': {'type': 'string'},
                    'context': {'type': 'string'},
                    'scope_hints': {'type': 'array', 'items': {'type': 'string'}},
                },
                'required': ['claim'],
            },
            function=verify,
        )

    async def assemble_payload(
        self,
        events: list[ReconciliationEvent],
        watermark: Watermark,
        prior_reports: list[StageReport],
    ) -> str:
        stage1_report = prior_reports[0] if prior_reports else None

        # Get task tree
        tasks_data: dict = {}
        if self.taskmaster:
            try:
                tasks_data = await self.taskmaster.get_tasks(project_root=self.project_id)
            except Exception:
                tasks_data = {}

        all_tasks = tasks_data.get('tasks', [])
        if not isinstance(all_tasks, list):
            all_tasks = []

        active_tasks = [
            t for t in all_tasks
            if isinstance(t, dict) and t.get('status') in ('pending', 'in-progress', 'review')
        ]
        done_tasks = [
            t for t in all_tasks if isinstance(t, dict) and t.get('status') == 'done'
        ]

        return f"""## Stage 2: Task-Knowledge Sync
## Project: {self.project_id}

### Stage 1 Report Summary
{_format_report(stage1_report)}

### Stage 1 Flagged Items (Task-Relevant)
{_format_flagged(stage1_report.items_flagged if stage1_report else [])}

### Active Task Tree ({len(active_tasks)} active, {len(done_tasks)} done, {len(all_tasks)} total)
{_format_tasks(active_tasks[:30])}

### Recently Completed Tasks
{_format_tasks(done_tasks[:20])}

## Your Task
Reconcile task state against memory:
1. For completed tasks: verify knowledge was captured. If sparse, use verify_against_codebase \
to check repo state, then write appropriate memories.
2. For tasks whose assumptions were invalidated by Stage 1 findings: modify, re-scope, or \
delete tasks. Update dependent tasks.
3. For AI-generated tasks: cross-reference against knowledge graph for factual consistency.
4. Attach memory_hints to tasks that would benefit from knowledge context at execution time. \
Use entity references + semantic queries, NOT inline content.
5. Check if any knowledge implies new tasks should be created or existing tasks unblocked.
6. Hints on completed tasks are static — don't update them.
7. Call stage_complete with your report when done.
"""


class IntegrityCheck(BaseStage):
    """Stage 3: Read-only cross-system consistency verification."""

    # Defined here alongside Stage 2 for import convenience; re-exported from stages/__init__

    verifier: CodebaseVerifier | None = None

    def get_system_prompt(self) -> str:
        from fused_memory.reconciliation.prompts.stage3 import STAGE3_SYSTEM_PROMPT
        return STAGE3_SYSTEM_PROMPT

    def get_tools(self) -> dict[str, ToolDefinition]:
        tools = {}
        tools.update(self._memory_read_tools())
        tools.update(self._task_read_tools())

        if self.verifier:
            tools['verify_against_codebase'] = self._verify_tool()

        return tools

    def _verify_tool(self) -> ToolDefinition:
        verifier = self.verifier
        assert verifier is not None  # only called when verifier is set

        async def verify(claim: str, context: str = '', scope_hints: list[str] | None = None):
            result = await verifier.verify(
                claim=claim, context=context, scope_hints=scope_hints, project_id=self.project_id
            )
            return result.model_dump()

        return ToolDefinition(
            name='verify_against_codebase',
            description='Verify a factual claim against the codebase via read-only explore agent.',
            parameters={
                'type': 'object',
                'properties': {
                    'claim': {'type': 'string'},
                    'context': {'type': 'string'},
                    'scope_hints': {'type': 'array', 'items': {'type': 'string'}},
                },
                'required': ['claim'],
            },
            function=verify,
        )

    async def assemble_payload(
        self,
        events: list[ReconciliationEvent],
        watermark: Watermark,
        prior_reports: list[StageReport],
    ) -> str:
        stage1_report = prior_reports[0] if len(prior_reports) > 0 else None
        stage2_report = prior_reports[1] if len(prior_reports) > 1 else None

        flagged = []
        if stage1_report:
            flagged.extend(stage1_report.items_flagged)
        if stage2_report:
            flagged.extend(stage2_report.items_flagged)

        return f"""## Stage 3: Cross-System Integrity Check
## Project: {self.project_id}

### Stage 1 Report
{_format_report(stage1_report)}

### Stage 2 Report
{_format_report(stage2_report)}

### Items Flagged for Cross-System Verification ({len(flagged)})
{_format_flagged(flagged)}

## Your Task
Verify consistency across all three systems:
1. Spot-check: do recently modified tasks align with current memory state?
2. Spot-check: do recently written memories align with task state?
3. For flagged items: investigate and classify as consistent/inconsistent.
4. Use verify_against_codebase for any factual disputes.
5. Report all findings. Inconsistencies found here will be addressed in the next cycle's \
Stage 1 and Stage 2.
6. Call stage_complete with your report.
"""


def _format_report(report: StageReport | None) -> str:
    if report is None:
        return 'No report available.'
    duration = (report.completed_at - report.started_at).total_seconds()
    return (
        f'Stage: {report.stage.value}\n'
        f'Duration: {duration:.1f}s | LLM calls: {report.llm_calls} | '
        f'Tokens: {report.tokens_used}\n'
        f'Actions taken: {len(report.actions_taken)}\n'
        f'Stats: {json.dumps(report.stats, default=str)}\n'
        f'Items flagged: {len(report.items_flagged)}'
    )


def _format_flagged(items: list[dict]) -> str:
    if not items:
        return 'No flagged items.'
    lines = []
    for item in items[:20]:
        lines.append(f'- {json.dumps(item, default=str)}')
    if len(items) > 20:
        lines.append(f'... and {len(items) - 20} more')
    return '\n'.join(lines)


def _format_tasks(tasks: list[dict]) -> str:
    if not tasks:
        return 'No tasks.'
    lines = []
    for t in tasks:
        tid = t.get('id', '?')
        title = t.get('title', '?')
        status = t.get('status', '?')
        deps = t.get('dependencies', [])
        lines.append(f'- [{tid}] ({status}) {title} deps={deps}')
    return '\n'.join(lines)
