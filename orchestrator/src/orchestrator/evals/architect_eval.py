"""One architect eval cell as steps: invoke the architect, classify the attempt, assemble the cell.

Rationale: docs/plan-scoring-and-judge.md, "The architect cell: invoke,
classify, assemble (task 6737)".
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from shared.cli_invoke import AgentResult

from orchestrator.agents.invoke import invoke_agent
from orchestrator.artifacts import TaskArtifacts
from orchestrator.workflow import _meta_root_for_worktree

from .configs import EvalConfig
from .eval_gate import EVAL_CAP_WAIT_SANITY_SECS, GateFactory, build_eval_usage_gate
from .judge import PlanQualityVerdict, judge_plan_quality, score_plan_structure
from .measurement import (
    MeasurementStage,
    Scored,
    Unmeasured,
    UnmeasuredCause,
    UnmeasuredRecord,
    exception_reason,
)
from .metrics import EvalMetrics, compose_cost_source, resolve_terminal_kind
from .plan_predicates import is_scorable_plan
from .replay_frame import REPLAY_FRAME_ID
from .results import CellName, ResultStore
from .snapshots import (
    cleanup_eval_worktree,
    create_eval_worktree,
    get_diff_between_commits,
)

logger = logging.getLogger(__name__)

CreateWorktree = Callable[..., Awaitable[tuple[Path, str]]]
CleanupWorktree = Callable[[Path, Path], Awaitable[None]]
OpenArtifacts = Callable[[Path], TaskArtifacts]
InvokeAgent = Callable[..., Awaitable[AgentResult]]
JudgePlan = Callable[[dict, str, dict], Awaitable[Scored[PlanQualityVerdict] | Unmeasured]]
ReadDiff = Callable[[Path, str, str], Awaitable[str]]

CANDIDATE_ATTRIBUTABLE_CAUSES = frozenset({UnmeasuredCause.TIMEOUT})
"""Architect causes that still asked the candidate, so they never taint a cell."""


def open_eval_artifacts(worktree: Path) -> TaskArtifacts:
    """The artifacts at the relocated meta root the plan-tools server writes to."""
    return TaskArtifacts(worktree, meta_root=_meta_root_for_worktree(worktree))


def cell_label(task: dict, config: EvalConfig) -> str:
    return f"{task['id']} × {config.name}"


@dataclass(frozen=True)
class ArchitectEvalDeps:
    """Every effect boundary of one architect cell; the defaults are production."""

    store: ResultStore
    create_worktree: CreateWorktree = create_eval_worktree
    cleanup_worktree: CleanupWorktree = cleanup_eval_worktree
    open_artifacts: OpenArtifacts = open_eval_artifacts
    invoke: InvokeAgent = invoke_agent
    judge: JudgePlan = judge_plan_quality
    read_reference_diff: ReadDiff = get_diff_between_commits
    gate_factory: GateFactory = build_eval_usage_gate
    cap_wait_secs: float = EVAL_CAP_WAIT_SANITY_SECS


@dataclass(frozen=True)
class ArchitectAttempt:
    """What invoking the architect left behind, before any scoring."""

    address: CellName
    worktree: Path
    outcome: str
    plan: dict
    declines: Mapping[str, dict | None]
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_create_tokens: int = 0
    turns: int = 0
    duration_ms: int = 0
    cost_usd: float = 0.0
    cost_source: str = 'cli'
    is_local_model: bool = False
    unmeasured: Unmeasured | None = None


class PlanScoring(StrEnum):
    TAINTED = 'tainted'
    FLOORED = 'floored'
    JUDGE = 'judge'


def classify_attempt(attempt: ArchitectAttempt) -> PlanScoring:
    """How the attempt's plan is scored: excluded, structural floor, or LLM judge."""
    never_asked = (
        attempt.unmeasured is not None
        and attempt.unmeasured.cause not in CANDIDATE_ATTRIBUTABLE_CAUSES
    )
    scorable = is_scorable_plan(attempt.plan)
    if never_asked and not scorable:
        return PlanScoring.TAINTED
    if never_asked or not scorable:
        return PlanScoring.FLOORED
    return PlanScoring.JUDGE


@dataclass(frozen=True)
class PlanJudgement:
    """The plan judge's outcome, and whether it judged without a reference diff."""

    outcome: Scored[PlanQualityVerdict] | Unmeasured
    blind: bool


async def read_reference_diff(task: dict, deps: ArchitectEvalDeps) -> str:
    """The fixture's landed diff, or ``''`` when there is none to read."""
    post = (task.get('reference') or {}).get('post_task_commit')
    if not post:
        logger.warning(
            f"Architect eval {task['id']}: fixture carries no "
            f'reference.post_task_commit, so NO reference diff is available'
        )
        return ''
    try:
        return await deps.read_reference_diff(
            Path(task['project_root']), task['pre_task_commit'], post,
        )
    except Exception as e:
        logger.warning(f"reference diff failed for {task['id']}: {e}")
        return ''


async def judge_plan(
    plan: dict, reference_diff: str, task: dict, deps: ArchitectEvalDeps,
) -> PlanJudgement:
    """Ask the plan judge; a raise becomes an ``Unmeasured`` with unknown spend."""
    blind = not reference_diff
    if blind:
        logger.warning(
            f"Architect eval {task['id']}: invoking the plan judge with an EMPTY "
            f'reference diff — no ground truth to grade against'
        )
    try:
        outcome = await deps.judge(plan, reference_diff, task)
    except Exception as e:
        logger.warning(
            f"plan judge raised for {task['id']}; degrading to structural floor",
            exc_info=True,
        )
        outcome = Unmeasured(
            MeasurementStage.JUDGE, UnmeasuredCause.RAISED,
            exception_reason(e), cost_usd=0.0,
        )
    return PlanJudgement(outcome, blind=blind)


@dataclass(frozen=True)
class _JudgeShare:
    plan_quality: float | None = None
    unmeasured: Unmeasured | None = None
    invocations: int = 0
    cost_usd: float = 0.0
    without_reference: bool = False


def _judge_share(judgement: PlanJudgement) -> _JudgeShare:
    outcome = judgement.outcome
    if isinstance(outcome, Scored):
        return _JudgeShare(
            plan_quality=outcome.value.plan_quality, invocations=1,
            cost_usd=outcome.cost_usd, without_reference=judgement.blind,
        )
    if outcome.cause is UnmeasuredCause.RAISED:
        return _JudgeShare(unmeasured=outcome)
    return _JudgeShare(unmeasured=outcome, invocations=1, cost_usd=outcome.cost_usd)


def _plan_quality(
    plan: dict, scoring: PlanScoring, share: _JudgeShare,
) -> float | None:
    if scoring is PlanScoring.TAINTED:
        return None
    if share.plan_quality is not None:
        return share.plan_quality
    return score_plan_structure(plan)


def assemble_metrics(
    attempt: ArchitectAttempt,
    scoring: PlanScoring,
    judgement: PlanJudgement | None,
) -> EvalMetrics:
    """The persisted metrics of one architect cell; pure."""
    if (scoring is PlanScoring.JUDGE) != (judgement is not None):
        raise ValueError(
            f'a {scoring.value} architect cell takes a judgement exactly when it '
            f'was judged; got {judgement!r}'
        )
    share = _judge_share(judgement) if judgement is not None else _JudgeShare()
    plan = attempt.plan
    duration_secs = attempt.duration_ms / 1000
    records = [
        u.to_record() for u in (attempt.unmeasured, share.unmeasured) if u is not None
    ]
    return EvalMetrics(
        plan_quality=_plan_quality(plan, scoring, share),
        role_under_test='architect',
        terminal_kind=resolve_terminal_kind(plan, attempt.declines),
        replay_frame=REPLAY_FRAME_ID,
        tests_pass=None,
        plan_steps=len(plan.get('steps') or []),
        cost_usd=attempt.cost_usd + share.cost_usd,
        cost_source=compose_cost_source(
            attempt.cost_source, secondary_cost_usd=share.cost_usd,
        ),
        input_tokens=attempt.input_tokens,
        output_tokens=attempt.output_tokens,
        is_local_model=attempt.is_local_model,
        cache_read_tokens=attempt.cache_read_tokens,
        cache_create_tokens=attempt.cache_create_tokens,
        turns_used=attempt.turns,
        tokens_per_second=(
            round(attempt.output_tokens / duration_secs, 2) if duration_secs > 0 else 0.0
        ),
        workflow_duration_ms=attempt.duration_ms,
        invocation_errors=records or None,
        cap_tainted=scoring is PlanScoring.TAINTED,
        judge_cost_usd=share.cost_usd,
        judge_invocations=share.invocations,
        judged_without_reference=share.without_reference,
    )


def log_architect_cell(
    label: str, scoring: PlanScoring, metrics: EvalMetrics, wall_clock_ms: int,
) -> None:
    """The scoring route's WARNING, if any, then the cell's completion line."""
    records: list[UnmeasuredRecord] = metrics.invocation_errors or []
    if scoring is PlanScoring.TAINTED:
        logger.warning(
            f'Architect eval {label}: invocation refused with no scorable plan '
            f'artifact ({records[0]}) — plan judge skipped, plan_quality=None, '
            f'cell marked cap_tainted (NOT scored 0.0)'
        )
    elif scoring is PlanScoring.FLOORED and metrics.plan_steps:
        logger.warning(
            f'Architect eval {label}: invocation refused ({records[0]}) but a plan '
            f'artifact exists — LLM judge skipped, scored on the structural floor '
            f'({metrics.plan_quality}), NOT tainted'
        )
    elif scoring is PlanScoring.FLOORED:
        logger.warning(
            f'Architect eval {label}: architect ran successfully but produced no '
            f'scorable plan — plan judge skipped, scored on the structural floor '
            f'({metrics.plan_quality}), NOT tainted'
        )
    errors_tag = 'cap_tainted' if metrics.cap_tainted else 'invocation_errors'
    logger.info(
        f'Architect eval complete: {label} → plan_quality={metrics.plan_quality} '
        f'({wall_clock_ms / 1000:.1f}s, ${metrics.cost_usd:.2f} incl. '
        f'${metrics.judge_cost_usd:.2f} judge)'
        + (f' [{errors_tag}: {metrics.invocation_errors}]' if records else '')
        + (' [judged_without_reference]' if metrics.judged_without_reference else '')
        + f' [terminal_kind={metrics.terminal_kind}]'
    )
