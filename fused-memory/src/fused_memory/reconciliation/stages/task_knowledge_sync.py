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
from fused_memory.reconciliation.prompts import (
    _STAGE2_PROJECT_ID_GUIDELINE,
    _STAGE3_PROJECT_ID_GUIDELINE,
)
from fused_memory.reconciliation.prompts.stage2 import STAGE2_SYSTEM_PROMPT
from fused_memory.reconciliation.stages.base import BaseStage
from fused_memory.reconciliation.task_filter import (
    filter_task_tree,
    format_filtered_task_tree,
    format_task_list,
)


class TaskKnowledgeSync(BaseStage):
    """Stage 2: Reconcile tasks against memory, attach hints, fix inconsistencies."""

    # Remediation support — set by harness for second pass
    remediation_mode: bool = False

    # Minimum number of tasks to proactively spot-check each run
    MIN_TASK_SAMPLE: int = 5

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

        # Delegate all filtering, partitioning, and sorting to task_filter
        filtered = filter_task_tree(tasks_data)
        all_tasks = tasks_data.get('tasks', [])
        if not isinstance(all_tasks, list):
            all_tasks = []

        remediation_note = ''
        if self.remediation_mode:
            remediation_note = (
                '### Remediation Mode\n'
                'This is a focused remediation run. Address remaining task-level issues '
                'from Stage 1. Do not perform general task-knowledge sync.\n\n'
            )

        proactive_sample_section = ''
        if not self.remediation_mode:
            sample = _select_proactive_sample(all_tasks, self.MIN_TASK_SAMPLE)
            proactive_sample_section = (
                f'\n### Proactive Task Sample ({len(sample)} tasks)\n'
                f'{format_task_list(sample)}\n'
            )

        return f"""## Stage 2: Task-Knowledge Sync
## Project: {self.project_id}

{remediation_note}### Stage 1 Report Summary
{_format_report(stage1_report)}

### Stage 1 Flagged Items (Task-Relevant)
{_format_flagged(stage1_report.items_flagged if stage1_report else [])}

{format_filtered_task_tree(filtered)}

### Recently Completed Tasks
{format_task_list(filtered.done_tasks[:30])}{proactive_sample_section}

## Your Task
Reconcile task state against memory:
1. For completed tasks: verify knowledge was captured. If sparse, search for related memories \
to check context, then write appropriate memories.
2. For tasks whose assumptions were invalidated by Stage 1 findings: modify, re-scope, or \
delete tasks. Update dependent tasks.
3. For AI-generated tasks: cross-reference against knowledge graph for factual consistency.
4. Attach memory_hints to tasks that would benefit from knowledge context at execution time. \
Use entity references + semantic queries, NOT inline content.
5. Proactively review the **Proactive Task Sample** regardless of Stage 1 findings: check \
in-progress tasks for completion knowledge to capture, blocked tasks for unblock conditions \
that may now be met, and done tasks for missing knowledge capture.
6. Check if any knowledge implies new tasks should be created or existing tasks unblocked.
7. Hints on completed tasks are static — don't update them.
8. When you have completed your work, produce your final structured report as your response.

{_STAGE2_PROJECT_ID_GUIDELINE.format(project_id=self.project_id)}
Use project_root="{self.project_root}" for all task operations.
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

{_STAGE3_PROJECT_ID_GUIDELINE.format(project_id=self.project_id)}
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


def _select_proactive_sample(all_tasks: list[dict], n: int) -> list[dict]:
    """Select the top-N tasks for proactive spot-checking.

    Sorted by status priority (in-progress > blocked > review > pending > done),
    then by task ID descending (proxy for recency — higher ID = more recently created).
    Returns at most n tasks; fewer if the task list is smaller than n.

    Non-dict elements in all_tasks are filtered out defensively, matching the
    isinstance(t, dict) guard pattern used by active_tasks/done_tasks derivations.
    """
    # Filter out non-dict elements before sorting so sort_key never crashes
    tasks = [t for t in all_tasks if isinstance(t, dict)]

    # Import from task_filter — the single source of truth for status priority
    from fused_memory.reconciliation.task_filter import _STATUS_PRIORITY  # noqa: PLC0415

    def sort_key(t: dict) -> tuple[int, int]:
        status = t.get('status', 'pending')
        priority = _STATUS_PRIORITY.get(status, len(_STATUS_PRIORITY))
        # Negate ID so that higher IDs sort first within the same priority
        tid = t.get('id', 0)
        try:
            tid_int = int(tid)
        except (TypeError, ValueError):
            tid_int = 0
        return (priority, -tid_int)

    sorted_tasks = sorted(tasks, key=sort_key)
    return sorted_tasks[:n]
