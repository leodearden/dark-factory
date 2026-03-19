"""Metric collection and composite scoring for eval runs."""

from __future__ import annotations

import asyncio
import logging
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

    # Efficiency
    cost_usd: float = 0.0
    wall_clock_ms: int = 0
    turns_used: int = 0
    iterations: int = 0          # implementer re-invocations
    debug_cycles: int = 0        # debugger invocations

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
    """Quality-weighted score. Higher is better.

    - Fails tests → score 0
    - Otherwise: quality factor (penalized by blocking issues and debug cycles)
      times plan completion percentage
    """
    if not m.tests_pass:
        return 0.0
    quality = 1.0 - (m.review_blocking_issues * 0.2) - (m.debug_cycles * 0.1)
    quality = max(quality, 0.0)
    return quality * m.plan_completion_pct


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

    # Git stats
    lines_changed, files_changed = await _git_diff_stats(worktree)

    m = EvalMetrics(
        tests_pass=verify.passed if verify else False,
        lint_clean=(not verify.lint_output) if verify else False,
        typecheck_clean=(not verify.type_output) if verify else False,
        plan_completion_pct=plan_completion,
        cost_usd=wf_metrics.total_cost_usd,
        wall_clock_ms=wf_metrics.total_duration_ms,
        turns_used=0,  # aggregated per-agent, not tracked at workflow level
        iterations=wf_metrics.execute_iterations,
        debug_cycles=wf_metrics.verify_attempts,
        review_blocking_issues=blocking_issues,
        review_suggestions=suggestions,
        lines_changed=lines_changed,
        files_changed=files_changed,
    )
    m.composite_score = compute_composite(m)
    return m


async def _git_diff_stats(worktree: Path) -> tuple[int, int]:
    """Get lines changed and files changed from git diff --stat."""
    try:
        proc = await asyncio.create_subprocess_exec(
            'git', 'diff', '--stat', 'HEAD',
            cwd=str(worktree),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        lines = stdout.decode().strip().split('\n')
        if not lines:
            return 0, 0

        # Last line: " X files changed, Y insertions(+), Z deletions(-)"
        summary = lines[-1]
        files_changed = 0
        lines_changed = 0

        import re
        m = re.search(r'(\d+) files? changed', summary)
        if m:
            files_changed = int(m.group(1))
        for m in re.finditer(r'(\d+) (?:insertions?|deletions?)', summary):
            lines_changed += int(m.group(1))

        return lines_changed, files_changed
    except Exception:
        return 0, 0
