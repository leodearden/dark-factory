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

    # Correctness (pass/fail gate). ``None`` means "unknown / suspicious" —
    # see the false-green guard at the bottom of ``collect_metrics``.
    tests_pass: bool | None = False
    lint_clean: bool | None = False
    typecheck_clean: bool | None = False
    plan_completion_pct: float = 0.0
    plan_steps: int = 0

    # Efficiency
    cost_usd: float = 0.0
    workflow_duration_ms: int = 0
    turns_used: int = 0
    iterations: int = 0          # implementer re-invocations
    debug_cycles: int = 0        # debugger invocations

    # Completion judge (ζ) — judge_cost_usd is a SUBSET of cost_usd, not
    # disjoint. Report generators should not double-count.
    judge_invocations: int = 0
    judge_cost_usd: float = 0.0
    judge_early_exits: int = 0

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

    # Inference speed
    tokens_per_second: float = 0.0       # output_tokens / generation_seconds
    is_local_model: bool = False          # True when env_overrides has ANTHROPIC_BASE_URL
    hardware_time_seconds: float = 0.0    # wall-clock for hardware cost imputation

    # Derived
    composite_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_false_green(m: EvalMetrics, max_iterations: int) -> bool:
    """The 404-bug signature: iteration cap hit with zero work but T/T/T.

    When every agent subprocess errors at the network layer (e.g. the vLLM
    bridge 404 bug, 2026-04-08), the workflow burns through its iteration
    budget making no code changes, verify then runs against the untouched
    baseline, and reports clean gates for whatever the pre-task tree already
    passed. Cost lands at $0 because the CLI never completed a usage-tracked
    turn. See ``docs/vllm-eval-status.md`` (2026-04-08 afternoon).
    """
    return (
        bool(m.tests_pass)
        and m.lines_changed == 0
        and m.files_changed == 0
        and m.iterations >= max_iterations
        and m.cost_usd == 0.0
    )


def compute_composite(m: EvalMetrics) -> float:
    """Pure quality score bounded to 0..1.

    - Fails tests (or ``tests_pass=None`` from the false-green guard) → score 0
    - blocking_rate = blocking_issues / plan_steps (larger tasks tolerate more issues)
    - debug_cycles get a light penalty (the system self-correcting is good)
    - Final score = quality, clamped to [0, 1]

    ``plan_completion_pct`` is still collected as a diagnostic signal (visible
    in result JSON) but no longer gates the composite. It was previously a
    multiplier, but local models that write correct code without updating
    plan.json status fields would get score 0 despite T/T/T gates.
    """
    if not m.tests_pass:
        return 0.0
    steps = max(m.plan_steps, 1)
    blocking_rate = m.review_blocking_issues / steps
    quality = 1.0 - (blocking_rate * 2.0) - (m.debug_cycles * 0.05)
    quality = max(quality, 0.0)
    quality = min(quality, 1.0)
    return round(quality, 4)


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

    # Inference speed metrics
    duration_secs = wf_metrics.total_duration_ms / 1000 if wf_metrics.total_duration_ms else 0.0
    tps = wf_metrics.total_output_tokens / duration_secs if duration_secs > 0 else 0.0
    is_local = bool(workflow.config.env_overrides.get('ANTHROPIC_BASE_URL'))

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
        judge_invocations=wf_metrics.judge_invocations,
        judge_cost_usd=wf_metrics.judge_cost_usd,
        judge_early_exits=wf_metrics.judge_early_exits,
        input_tokens=wf_metrics.total_input_tokens,
        output_tokens=wf_metrics.total_output_tokens,
        cache_read_tokens=wf_metrics.total_cache_read_tokens,
        cache_create_tokens=wf_metrics.total_cache_create_tokens,
        review_blocking_issues=blocking_issues,
        review_suggestions=suggestions,
        lines_changed=lines_changed,
        files_changed=files_changed,
        tokens_per_second=round(tps, 2),
        is_local_model=is_local,
        hardware_time_seconds=round(duration_secs, 3),
    )
    # False-green guard — catches the 404-bug signature so the same class of
    # silent failure doesn't need manual quarantine in future runs.
    if _is_false_green(m, workflow.config.max_execute_iterations):
        logger.warning(
            'False-green signature for task %s: %d iters @ cap, '
            '$0 cost, 0 lines / 0 files changed, T/T/T — '
            'nulling gate fields so score is 0',
            task.get('id', '?'), m.iterations,
        )
        m.tests_pass = None
        m.lint_clean = None
        m.typecheck_clean = None

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
