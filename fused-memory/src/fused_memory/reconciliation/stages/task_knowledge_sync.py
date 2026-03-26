"""Stage 2: Task-Knowledge Sync — reconcile task state against memory state.
Stage 3: Cross-System Integrity Check — read-only verification."""

from __future__ import annotations

import json

from fused_memory.models.reconciliation import (
    ReconciliationEvent,
    StageReport,
    Watermark,
)
from fused_memory.reconciliation.cli_stage_runner import (
    STAGE2_DISALLOWED,
    STAGE3_DISALLOWED,
    STAGE3_REPORT_SCHEMA,
)
from fused_memory.reconciliation.prompts.stage2 import STAGE2_SYSTEM_PROMPT
from fused_memory.reconciliation.stages.base import BaseStage


class TaskKnowledgeSync(BaseStage):
    """Stage 2: Reconcile tasks against memory, attach hints, fix inconsistencies."""

    # Remediation support — set by harness for second pass
    remediation_mode: bool = False

    def get_system_prompt(self) -> str:
        return STAGE2_SYSTEM_PROMPT

    def get_disallowed_tools(self) -> list[str]:
        return STAGE2_DISALLOWED

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
                tasks_data = await self.taskmaster.get_tasks(project_root=self.project_root)
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

        remediation_note = ''
        if self.remediation_mode:
            remediation_note = (
                '### Remediation Mode\n'
                'This is a focused remediation run. Address remaining task-level issues '
                'from Stage 1. Do not perform general task-knowledge sync.\n\n'
            )

        return f"""## Stage 2: Task-Knowledge Sync
## Project: {self.project_id}

{remediation_note}### Stage 1 Report Summary
{_format_report(stage1_report)}

### Stage 1 Flagged Items (Task-Relevant)
{_format_flagged(stage1_report.items_flagged if stage1_report else [])}

### Active Task Tree
{_format_tasks(active_tasks[:50])}

### Recently Completed Tasks
{_format_tasks(done_tasks[:30])}

## Your Task
Reconcile task state against memory:
1. For completed tasks: verify knowledge was captured. If sparse, search for related memories \
to check context, then write appropriate memories.
2. For tasks whose assumptions were invalidated by Stage 1 findings: modify, re-scope, or \
delete tasks. Update dependent tasks.
3. For AI-generated tasks: cross-reference against knowledge graph for factual consistency.
4. Attach memory_hints to tasks that would benefit from knowledge context at execution time. \
Use entity references + semantic queries, NOT inline content.
5. Check if any knowledge implies new tasks should be created or existing tasks unblocked.
6. Hints on completed tasks are static — don't update them.
7. Do NOT write task counts, status distributions, or task tree size data as memories. \
These numbers are transient context — never persist them.
8. When you have completed your work, produce your final structured report as your response.

Always pass project_id="{self.project_id}" when calling fused-memory MCP tools.
Use project_root="/home/leo/src/dark-factory" for all task operations.
"""


class IntegrityCheck(BaseStage):
    """Stage 3: Read-only cross-system consistency verification."""

    def get_system_prompt(self) -> str:
        from fused_memory.reconciliation.prompts.stage3 import STAGE3_SYSTEM_PROMPT
        return STAGE3_SYSTEM_PROMPT

    def get_disallowed_tools(self) -> list[str]:
        return STAGE3_DISALLOWED

    def get_report_schema(self) -> dict:
        return STAGE3_REPORT_SCHEMA

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
4. Report all findings. Inconsistencies found here will be addressed in the next cycle's \
Stage 1 and Stage 2.
5. When you have completed your work, produce your final structured report as your response.

Always pass project_id="{self.project_id}" when calling fused-memory MCP tools.
"""


def _format_report(report: StageReport | None) -> str:
    if report is None:
        return 'No report available.'
    duration = (report.completed_at - report.started_at).total_seconds()
    return (
        f'Stage: {report.stage.value}\n'
        f'Duration: {duration:.1f}s | LLM calls: {report.llm_calls} | '
        f'Tokens: {report.tokens_used}\n'
        f'Stats: {json.dumps(report.stats, default=str)}\n'
        f'Items flagged: {len(report.items_flagged)}'
    )


def _format_flagged(items: list[dict]) -> str:
    if not items:
        return 'No flagged items.'
    lines = []
    for item in items[:50]:
        lines.append(f'- {json.dumps(item, default=str)}')
    if len(items) > 50:
        lines.append(f'... and {len(items) - 50} more')
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
