"""Stage 2: Task-Knowledge Sync — reconcile task state against memory state.
Stage 3: Cross-System Integrity Check — read-only verification."""

from __future__ import annotations

import asyncio
import heapq
import itertools
import json
import logging
from collections.abc import Iterable

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
    FilteredTaskTree,
    filter_task_tree,
    format_filtered_task_tree,
    format_task_list,
)

logger = logging.getLogger(__name__)


class TaskKnowledgeSync(BaseStage):
    """Stage 2: Reconcile tasks against memory, attach hints, fix inconsistencies."""

    # Remediation support — set by harness for second pass
    remediation_mode: bool = False

    # Active task tree — set by harness before run() (task 455)
    filtered_task_tree: FilteredTaskTree | None = None

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

        # Dual-path: use harness-injected tree or self-fetch (task 455)
        if self.filtered_task_tree is not None:
            # Harness path: tree already fetched and filtered before run()
            filtered = self.filtered_task_tree
        else:
            # Fallback path: self-fetch via taskmaster
            tasks_data: dict = {}
            if self.taskmaster:
                try:
                    tasks_data = await self.taskmaster.get_tasks(project_root=self.project_root)
                except Exception:
                    tasks_data = {}
            filtered = filter_task_tree(tasks_data)

        # Defensive invariant check (task-782): see _check_filtered_tree_invariant.
        self._check_filtered_tree_invariant(filtered)

        # Render "Recently Completed Tasks" section.
        # Invariant: filter_task_tree() always appends to done_tasks when it increments
        # done_count (capped at MAX_DONE_TASKS_RETAINED=30), so done_tasks is guaranteed
        # non-empty whenever done_count > 0.  No fallback summary branch is needed.
        # _check_filtered_tree_invariant() warns for externally-constructed trees that
        # violate this.
        if filtered.done_tasks:
            recently_completed_text = format_task_list(filtered.done_tasks)
        else:
            recently_completed_text = format_task_list([])  # 'No tasks.'

        # Done-task provenance section — feeds verified evidence to the agent
        # so 'shipped via X' edges come from commit diffs instead of being
        # fabricated from metadata.modules. Empty string when no done tasks
        # carry done_provenance (legacy tree, warn-only rollout).
        provenance_section = await _render_done_provenance_section(
            filtered.done_tasks, self.project_root,
        )

        remediation_note = ''
        if self.remediation_mode:
            remediation_note = (
                '### Remediation Mode\n'
                'This is a focused remediation run. Address remaining task-level issues '
                'from Stage 1. Do not perform general task-knowledge sync.\n\n'
            )

        proactive_sample_section = ''
        if not self.remediation_mode:
            # Pool intentionally excludes unknown-status tasks (dropped by
            # filter_task_tree) and caps done_tasks at MAX_DONE_TASKS_RETAINED.
            sample = _select_proactive_sample(
                itertools.chain(filtered.active_tasks, filtered.done_tasks, filtered.cancelled_tasks),
                self.MIN_TASK_SAMPLE,
            )
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
{recently_completed_text}
{provenance_section}{proactive_sample_section}

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

    @staticmethod
    def _warn_if_count_tasks_mismatch(
        count: int,
        tasks: list,
        count_label: str,
        tasks_label: str,
        section_label: str,
        task_ref: str,
    ) -> None:
        """Emit a WARNING when a count>0/tasks-empty invariant is violated.

        Extracted to avoid repeating the same guard pattern for each
        count↔tasks pair (done, cancelled, and any future additions).
        """
        if count > 0 and not tasks:
            logger.warning(
                'FilteredTaskTree invariant violation: %s=%d but %s is '
                'empty. Externally-constructed tree bypassed filter_task_tree() guarantee. '
                '%s section will render as empty. (%s defensive check)',
                count_label,
                count,
                tasks_label,
                section_label,
                task_ref,
            )

    @staticmethod
    def _check_filtered_tree_invariant(filtered: FilteredTaskTree) -> None:
        """Emit a WARNING for each violated done/cancelled count↔tasks invariant.

        filter_task_tree() always appends to done_tasks when it increments done_count
        (capped at MAX_DONE_TASKS_RETAINED=30), and always appends to cancelled_tasks
        when it increments cancelled_count (capped at MAX_CANCELLED_TASKS_RETAINED=15),
        so both invariants are impossible to violate via the normal code path.
        Externally-constructed FilteredTaskTree instances that bypass filter_task_tree()
        could violate either; these checks catch them at the callsite rather than silently
        dropping data from the "Recently Completed" or "Recently Cancelled" sections.
        """
        TaskKnowledgeSync._warn_if_count_tasks_mismatch(
            filtered.done_count,
            filtered.done_tasks,
            'done_count',
            'done_tasks',
            'Recently Completed',
            'task-782',
        )
        TaskKnowledgeSync._warn_if_count_tasks_mismatch(
            filtered.cancelled_count,
            filtered.cancelled_tasks,
            'cancelled_count',
            'cancelled_tasks',
            'Recently Cancelled',
            'task-828',
        )


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


async def _render_done_provenance_section(
    done_tasks: list[dict],
    project_root: str | None,
    *,
    max_files_per_task: int = 50,
    max_chars_per_task: int = 2000,
) -> str:
    """Render a '### Done-task Provenance' block from task metadata.done_provenance.

    For each done task:
    - With ``commit``: emits the resolved SHA + a bounded file list produced by
      ``git show --name-only --format=%H%n%ai%n%s <sha>``. Capped at
      ``max_files_per_task`` files and ``max_chars_per_task`` characters per task
      so a single runaway commit can't blow the prompt budget.
    - With ``note``: emits the note text verbatim (no git call).
    - Without either: emits ``provenance: unknown (legacy)``.

    Returns an empty string when ``done_tasks`` is empty — no section is injected
    in that case, keeping the prompt tight when no new completions exist.
    """
    if not done_tasks:
        return ''

    lines: list[str] = ['### Done-task Provenance']
    for task in done_tasks:
        if not isinstance(task, dict):
            continue
        tid = task.get('id', '?')
        title = task.get('title', '')
        metadata = task.get('metadata') if isinstance(task.get('metadata'), dict) else {}
        prov = metadata.get('done_provenance') if isinstance(metadata, dict) else None
        header = f'- [{tid}] {title}'

        if not isinstance(prov, dict):
            lines.append(f'{header} — provenance: unknown (legacy)')
            continue

        commit = prov.get('commit') if isinstance(prov.get('commit'), str) else None
        note = prov.get('note') if isinstance(prov.get('note'), str) else None

        if commit and project_root:
            diff_block = await _git_show_name_only(
                project_root, commit,
                max_files=max_files_per_task,
                max_chars=max_chars_per_task,
            )
            lines.append(f'{header}\n  commit: {commit}')
            if diff_block:
                indented = '\n'.join('    ' + ln for ln in diff_block.splitlines())
                lines.append(indented)
        if note:
            lines.append(f'  note: {note}')
        if not commit and not note:
            lines.append(f'{header} — provenance: unknown (legacy)')

    return '\n'.join(lines) + '\n'


async def _git_show_name_only(
    project_root: str, commit: str,
    *, max_files: int, max_chars: int,
) -> str:
    """Run ``git show --name-only --format=%H%n%ai%n%s <commit>`` and truncate.

    Returns a short text block:

        <sha>
        <iso date>
        <subject>
        files:
          path/to/file1
          path/to/file2
          ... (N more)

    Returns an empty string on subprocess failure — the caller still emits the
    commit SHA header, just without the file list. We deliberately don't raise
    so one broken ref doesn't abort the whole Stage-2 briefing.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            'git', '-C', project_root, 'show', '--name-only',
            '--format=%H%n%ai%n%s', '--no-color', commit,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
        except TimeoutError:
            proc.kill()
            return ''
    except FileNotFoundError:
        return ''
    except Exception as e:
        logger.warning('git show failed for %s in %s: %s', commit, project_root, e)
        return ''
    if proc.returncode != 0:
        return ''

    raw = stdout.decode('utf-8', errors='replace')
    lines = raw.splitlines()
    if len(lines) < 3:
        return raw[:max_chars]

    header = lines[:3]
    file_lines = [ln for ln in lines[3:] if ln.strip()]
    total = len(file_lines)
    shown = file_lines[:max_files]
    more = total - len(shown)

    block = '\n'.join(header) + '\nfiles:'
    for f in shown:
        block += f'\n  {f}'
    if more > 0:
        block += f'\n  ... ({more} more)'

    if len(block) > max_chars:
        block = block[:max_chars] + '\n  ... (truncated)'
    return block


def _select_proactive_sample(tasks: Iterable[dict], n: int) -> list[dict]:
    """Select the top-N tasks for proactive spot-checking.

    Sorted by status priority (in-progress > blocked > review > pending > done),
    then by task ID descending (proxy for recency — higher ID = more recently created).
    Returns at most n tasks; fewer if the task list is smaller than n.

    Input must contain only dict elements; callers should pass
    FilteredTaskTree.active_tasks/done_tasks/cancelled_tasks fields, which
    filter_task_tree already pre-validates to be dict-only.
    """
    # Import from task_filter — the single source of truth for status priority and id parsing
    from fused_memory.reconciliation.task_filter import _STATUS_PRIORITY, _id_key  # noqa: PLC0415

    def sort_key(t: dict) -> tuple[int, int]:
        status = t.get('status', 'pending')
        priority = _STATUS_PRIORITY.get(status, len(_STATUS_PRIORITY))
        return (priority, -_id_key(t))

    return heapq.nsmallest(n, tasks, key=sort_key)
