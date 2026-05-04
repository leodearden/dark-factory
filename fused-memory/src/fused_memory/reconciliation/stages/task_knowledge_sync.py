"""Stage 2: Task-Knowledge Sync — reconcile task state against memory state.
Stage 3: Cross-System Integrity Check — read-only verification."""

from __future__ import annotations

import asyncio
import heapq
import itertools
import json
import logging
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fused_memory.backends.task_backend_protocol import TaskBackendProtocol

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

# Projects allowed to use the briefing-refresh hook.  This is a reify-specific
# feature; gating on project_id prevents accidental triggering by other projects
# that happen to have the same file layout.  Extend when needed.
_BRIEFING_REFRESH_PROJECT_ALLOWLIST: frozenset[str] = frozenset({'reify'})


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

    async def run(
        self,
        events: list[ReconciliationEvent],
        watermark: Watermark,
        prior_reports: list[StageReport],
        run_id: str,
        model: str | None = None,
    ) -> StageReport:
        """Run the briefing-refresh hook then delegate to BaseStage.run()."""
        await self._maybe_queue_briefing_refresh_tasks(run_id=run_id)
        return await super().run(events, watermark, prior_reports, run_id, model=model)

    async def _maybe_queue_briefing_refresh_tasks(self, run_id: str = '') -> None:
        """Best-effort: queue 'Refresh briefing' tasks for each briefing-known-gaps mismatch.

        Silently skips if project_root or taskmaster is absent, or if project_id
        is not in ``_BRIEFING_REFRESH_PROJECT_ALLOWLIST`` (reify-specific feature).
        Any exception is caught and logged as a WARNING so a broken script can
        never abort Stage 2.
        """
        if not self.project_root or not self.taskmaster:
            return
        if self.project_id not in _BRIEFING_REFRESH_PROJECT_ALLOWLIST:
            return
        try:
            mismatches = await _run_briefing_known_gaps_script(self.project_root)
            if not mismatches:
                return
            # Avoid a redundant get_tasks round-trip when the harness has
            # already injected the full task tree into self.filtered_task_tree.
            existing_tasks: list[dict] | None = None
            if self.filtered_task_tree is not None:
                existing_tasks = list(itertools.chain(
                    self.filtered_task_tree.active_tasks,
                    self.filtered_task_tree.done_tasks,
                    self.filtered_task_tree.cancelled_tasks,
                ))
            summary = await _queue_briefing_refresh_tasks(
                self.taskmaster, self.project_root, mismatches,
                existing_tasks=existing_tasks,
                run_id=run_id,
            )
            # Extract values inside the try/except so a contract violation
            # (e.g. summary being None due to a future refactor) is caught
            # here rather than propagating out of this method.
            created_ids = summary.get('created', [])
            skipped_ids = summary.get('skipped', [])
            failed_ids = summary.get('failed', [])
        except Exception:
            logger.warning(
                'briefing_refresh_hook_failed',
                exc_info=True,
                extra={'project_root': self.project_root},
            )
            return
        # Logging is intentionally outside the try/except above so a logging
        # bug (e.g. a reserved-name collision in `extra`) surfaces as a real
        # error rather than being swallowed as a misleading
        # 'briefing_refresh_hook_failed' WARNING. Note: 'created' is a
        # reserved LogRecord attribute (the timestamp), so we use
        # 'created_ids'/'skipped_ids'/'failed_ids' here.
        logger.info(
            'briefing_refresh_tasks_queued',
            extra={
                'project_root': self.project_root,
                'created_ids': created_ids,
                'skipped_ids': skipped_ids,
                'failed_ids': failed_ids,
            },
        )

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
            # filter_task_tree accepts `object`; either GetTasksResult or {} is fine.
            tasks_data: object = {}
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

        flagged_text = _format_flagged(
            stage1_report.items_flagged if stage1_report else [],
            run_stage='stage2',
        )

        known_projects_section = self._format_known_projects_section()

        return f"""## Stage 2: Task-Knowledge Sync
## Project: {self.project_id}

{remediation_note}### Stage 1 Report Summary
{_format_report(stage1_report)}

### Stage 1 Flagged Items (Task-Relevant)
{flagged_text}
{known_projects_section}
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
Use project_root="{self.project_root}" for tasks scoped to this project.
For cross-project routing see "Known Projects" above.
"""

    def _format_known_projects_section(self) -> str:
        """Render the cross-project routing context for the Stage 2 LLM.

        Emits a ``### Known Projects`` markdown section listing every
        configured project_id and its project_root, marking the current
        one.  Returns the empty string when fewer than two projects are
        known — there is no "cross-project" dimension to surface in that
        case, and the section would only add noise.
        """
        known = self.known_projects
        if len(known) < 2:
            return ''
        # Stable ordering: current project first, then alphabetical.
        ordered = [(self.project_id, known[self.project_id])] if self.project_id in known else []
        for pid in sorted(p for p in known if p != self.project_id):
            ordered.append((pid, known[pid]))
        # Pad the project_id column to a consistent width for readability.
        width = max(len(pid) for pid, _ in ordered)
        lines = []
        for pid, root in ordered:
            marker = '  (current)' if pid == self.project_id else ''
            lines.append(f'- {pid:<{width}}  → {root}{marker}')
        return (
            '\n### Known Projects (for cross-project routing)\n'
            + '\n'.join(lines)
            + '\n'
        )

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

        flagged_text = _format_flagged(flagged, run_stage='stage3')

        return f"""## Stage 3: Cross-System Integrity Check
## Project: {self.project_id}

### Stage 1 Report
{_format_report(stage1_report)}

### Stage 2 Report
{_format_report(stage2_report)}

### Items Flagged for Cross-System Verification ({len(flagged)})
{flagged_text}

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


_FLAGGED_ITEMS_CHAR_BUDGET = 40_000


def _format_flagged(
    items: list[dict],
    *,
    budget_chars: int = _FLAGGED_ITEMS_CHAR_BUDGET,
    run_stage: str | None = None,
) -> str:
    """Render flagged items as a bullet list, capped by *budget_chars*.

    Returns the rendered bullet-list text.  The per-render breakdown
    (``rendered``, ``dropped``, ``first_item_fragmented``) is emitted in the
    structured warning's ``extra`` when truncation fires — callers that need
    telemetry should read it from the warning record, not the return value.

    When *run_stage* is provided it is embedded in the warning's ``extra`` dict
    so ops can correlate the drop to its call site without a separate
    stage-specific shortfall warning.

    Edge case: if the very first item's JSON alone exceeds *budget_chars*, a
    truncated fragment is always rendered (with a ``… [item truncated]`` marker)
    so the LLM receives at least some signal rather than an opaque footer-only
    body.  ``rendered`` stays 0 (the fragment is not a full render) and the
    warning's ``extra`` includes ``first_item_fragmented=True`` so
    callers/telemetry can distinguish fragmented-first-item from all-dropped.
    """
    if not items:
        return 'No flagged items.'
    lines: list[str] = []
    running_chars = 0
    first_item_fragmented = False
    for idx, item in enumerate(items):
        json_str = json.dumps(item, default=str)
        line = f'- {json_str}'
        # +1 for the '\n' separator between lines
        separator = 1 if lines else 0
        if running_chars + separator + len(line) > budget_chars:
            # Budget exceeded — stop and emit a truncation footer + warning.
            if idx == 0:
                # First item alone exceeds the budget.  Always show at least a
                # truncated fragment so the LLM has some signal rather than an
                # opaque footer-only body.
                marker = '… [item truncated]'
                available = budget_chars - len('- ') - len(marker)
                if available > 0:
                    lines.append(f'- {json_str[:available]}{marker}')
                    first_item_fragmented = True
            dropped = len(items) - idx
            # Footer shows only items that are completely absent (not the
            # fragmented first item, which already appears as a truncated line).
            completely_missing = dropped - (1 if first_item_fragmented else 0)
            if completely_missing > 0:
                lines.append(f'... and {completely_missing} more (truncated: char budget)')
            extra: dict = {
                'total': len(items),
                'rendered': idx,
                'dropped': dropped,
                'budget_chars': budget_chars,
                'first_item_fragmented': first_item_fragmented,
            }
            if run_stage is not None:
                extra['run_stage'] = run_stage
            logger.warning('reconciliation.flagged_items_truncated', extra=extra)
            return '\n'.join(lines)
        lines.append(line)
        running_chars += separator + len(line)
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


async def _run_briefing_known_gaps_script(project_root: str) -> list[dict] | None:
    """Run reify's refresh_briefing_known_gaps.py in --json mode.

    Returns a list of mismatch dicts when mismatches are present, an empty list
    when none are found, or None when the script is absent (non-reify project),
    the briefing file is missing, or the subprocess fails.

    Exit codes:
    - 0: no mismatches → return []
    - 1: mismatches present → return parsed JSON list
    - 2+: script error → log WARNING, return None
    """
    script_path = Path(project_root) / 'scripts' / 'refresh_briefing_known_gaps.py'
    if not script_path.exists():
        return None

    briefing_path = Path(project_root) / 'review' / 'briefing.yaml'
    if not briefing_path.exists():
        return None

    tasks_path = Path(project_root) / '.taskmaster' / 'tasks' / 'tasks.json'

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(script_path),
            '--briefing', str(briefing_path),
            '--tasks', str(tasks_path),
            '--json',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10.0)
        except TimeoutError:
            proc.kill()
            logger.warning(
                'briefing_known_gaps_script_timeout',
                extra={'project_root': project_root},
            )
            return None
    except FileNotFoundError:
        logger.warning(
            'briefing_known_gaps_script_not_found',
            extra={'project_root': project_root},
        )
        return None
    except Exception as exc:
        logger.warning(
            'briefing_known_gaps_script_error',
            extra={'project_root': project_root, 'error': str(exc)},
        )
        return None

    if proc.returncode not in (0, 1):
        logger.warning(
            'briefing_known_gaps_script_failed',
            extra={
                'project_root': project_root,
                'returncode': proc.returncode,
                'stderr': stderr.decode('utf-8', errors='replace')[:500],
            },
        )
        return None

    try:
        return json.loads(stdout.decode('utf-8', errors='replace'))
    except json.JSONDecodeError as exc:
        logger.warning(
            'briefing_known_gaps_script_bad_json',
            extra={'project_root': project_root, 'error': str(exc)},
        )
        return None


async def _queue_briefing_refresh_tasks(
    taskmaster: 'TaskBackendProtocol',
    project_root: str,
    mismatches: list[dict],
    existing_tasks: list[dict] | None = None,
    run_id: str = '',
) -> dict:
    """Create 'Refresh briefing: remove task N from known_gaps' tasks for each mismatch.

    Skips creation when a task with status 'pending' and the same canonical title
    already exists (exact-title de-dup, case-sensitive string equality).

    ``existing_tasks`` may be pre-supplied by the caller (e.g. derived from
    the harness-injected ``filtered_task_tree``) to avoid a redundant
    ``get_tasks`` round-trip.  When ``None``, the function fetches the tree
    itself.

    ``run_id`` is written into task metadata as ``_causation_id`` so the
    created tasks are traceable back to the reconciliation run that filed them.

    Returns ``{"created": [task_ids], "skipped": [task_ids], "failed": [task_ids]}``.
    """
    if existing_tasks is None:
        existing_raw = await taskmaster.get_tasks(project_root=project_root)
        existing_tasks = existing_raw.get('tasks', []) if isinstance(existing_raw, dict) else []
    pending_titles = {
        t.get('title', '')
        for t in existing_tasks
        if t.get('status') == 'pending'
    }

    created: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []

    task_metadata: str | None = (
        json.dumps({'_causation_id': run_id, 'agent_id': 'recon-stage-task_knowledge_sync'})
        if run_id else None
    )

    for mismatch in mismatches:
        task_id = str(mismatch.get('task_id', ''))
        title = f'Refresh briefing: remove task {task_id} from known_gaps'

        if title in pending_titles:
            skipped.append(task_id)
            continue

        subproject = mismatch.get('subproject', '')
        task_title = mismatch.get('title', '')
        what = mismatch.get('what', '')
        description = (
            f'Subproject: {subproject}\n'
            f'Task title: {task_title}\n'
            f'Gap: {what}'
        )
        try:
            result = await taskmaster.add_task(
                project_root=project_root,
                title=title,
                description=description,
                metadata=task_metadata,
            )
            if isinstance(result, dict) and result.get('id'):
                created.append(str(result['id']))
            else:
                logger.warning(
                    'briefing_refresh_add_task_unexpected_shape',
                    extra={
                        'project_root': project_root,
                        'task_id': task_id,
                        'result_type': type(result).__name__,
                    },
                )
                failed.append(task_id)
        except Exception:
            logger.warning(
                'briefing_refresh_add_task_failed',
                exc_info=True,
                extra={'project_root': project_root, 'task_id': task_id},
            )
            failed.append(task_id)

    return {'created': created, 'skipped': skipped, 'failed': failed}
