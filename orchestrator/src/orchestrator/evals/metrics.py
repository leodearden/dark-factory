"""Metric collection and composite scoring for eval runs."""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from orchestrator.workflow import TaskWorkflow

logger = logging.getLogger(__name__)


@dataclass
class EvalMetrics:
    """Metrics collected from a single eval run."""

    # Correctness (pass/fail gate)
    tests_pass: bool = False
    lint_clean: bool = False
    typecheck_clean: bool = False
    plan_completion_pct: float = 0.0
    plan_steps: int = 0

    # Efficiency
    cost_usd: float = 0.0
    workflow_duration_ms: int = 0
    turns_used: int = 0
    iterations: int = 0          # implementer re-invocations
    debug_cycles: int = 0        # debugger invocations

    # Token usage
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_create_tokens: int = 0

    # Quality signals
    review_blocking_issues: int = 0
    review_suggestions: int = 0
    lines_changed: int = 0
    files_changed: int = 0

    # Derived
    composite_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_composite(m: EvalMetrics) -> float:
    """Quality-weighted score normalised by task complexity.

    - Fails tests → score 0
    - blocking_rate = blocking_issues / plan_steps (larger tasks tolerate more issues)
    - debug_cycles get a light penalty (the system self-correcting is good)
    - Final score = quality × plan_completion_pct
    """
    if not m.tests_pass:
        return 0.0
    steps = max(m.plan_steps, 1)
    blocking_rate = m.review_blocking_issues / steps
    quality = 1.0 - (blocking_rate * 2.0) - (m.debug_cycles * 0.05)
    quality = max(quality, 0.0)
    return round(quality * m.plan_completion_pct, 4)


async def collect_metrics(
    workflow: TaskWorkflow,
    worktree: Path,
    task: dict,
) -> EvalMetrics:
    """Collect metrics from a completed workflow run."""
    wf_metrics = workflow.metrics

    # Plan completion
    plan = workflow.artifacts.read_plan() if workflow.artifacts else {}
    steps = plan.get('steps', [])
    done_count = sum(1 for s in steps if s.get('status') == 'done')
    total_steps = len(steps) if steps else 1
    plan_completion = done_count / total_steps

    # Verification results (re-read from last run)
    from orchestrator.verify import run_verification
    verify = await run_verification(worktree, workflow.config)

    # Review stats from artifacts
    reviews = workflow.artifacts.aggregate_reviews() if workflow.artifacts else None
    blocking_issues = len(reviews.blocking_issues) if reviews else 0
    suggestions = len(reviews.suggestions) if reviews else 0

    # Git stats (diff against pre-task commit to capture all workflow changes)
    lines_changed, files_changed = await _git_diff_stats(
        worktree, task['pre_task_commit'],
    )

    m = EvalMetrics(
        tests_pass=verify.passed if verify else False,
        lint_clean=(not verify.lint_output) if verify else False,
        typecheck_clean=(not verify.type_output) if verify else False,
        plan_completion_pct=plan_completion,
        plan_steps=total_steps,
        cost_usd=wf_metrics.total_cost_usd,
        workflow_duration_ms=wf_metrics.total_duration_ms,
        turns_used=wf_metrics.total_turns,
        iterations=wf_metrics.execute_iterations,
        debug_cycles=wf_metrics.verify_attempts,
        input_tokens=wf_metrics.total_input_tokens,
        output_tokens=wf_metrics.total_output_tokens,
        cache_read_tokens=wf_metrics.total_cache_read_tokens,
        cache_create_tokens=wf_metrics.total_cache_create_tokens,
        review_blocking_issues=blocking_issues,
        review_suggestions=suggestions,
        lines_changed=lines_changed,
        files_changed=files_changed,
    )
    m.composite_score = compute_composite(m)
    return m


async def _git_diff_stats(worktree: Path, base_commit: str) -> tuple[int, int]:
    """Get lines changed and files changed vs the pre-task baseline commit."""
    try:
        proc = await asyncio.create_subprocess_exec(
            'git', 'diff', '--stat', f'{base_commit}..HEAD',
            cwd=str(worktree),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        output = stdout.decode().strip()
        if not output:
            return 0, 0

        # Last line: " X files changed, Y insertions(+), Z deletions(-)"
        summary = output.split('\n')[-1]
        files_changed = 0
        lines_changed = 0

        m = re.search(r'(\d+) files? changed', summary)
        if m:
            files_changed = int(m.group(1))
        for m in re.finditer(r'(\d+) (?:insertions?|deletions?)', summary):
            lines_changed += int(m.group(1))

        return lines_changed, files_changed
    except Exception:
        return 0, 0
