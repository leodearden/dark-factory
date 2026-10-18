"""One architect eval cell as steps: invoke the architect, classify the attempt, assemble the cell.

Rationale: docs/plan-scoring-and-judge.md, "The architect cell: invoke,
classify, assemble (task 6737)".
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from shared.cli_invoke import AgentResult, AllAccountsCappedException, invoke_with_cap_retry

from orchestrator.agents.invoke import invoke_agent
from orchestrator.agents.roles import ARCHITECT
from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import OrchestratorConfig
from orchestrator.workflow import _inject_plan_tools_mcp, _meta_root_for_worktree

from .configs import EvalConfig
from .eval_gate import (
    EVAL_CAP_WAIT_SANITY_SECS,
    GateFactory,
    InjectedGate,
    build_eval_usage_gate,
    owned_usage_gate,
)
from .judge import PlanQualityVerdict, judge_plan_quality, score_plan_structure
from .measurement import (
    MeasurementStage,
    Scored,
    Unmeasured,
    UnmeasuredCause,
    UnmeasuredRecord,
    classify_unmeasured,
    clip_reason,
    coerce_cost_usd,
    exception_reason,
)
from .metrics import (
    EvalMetrics,
    compose_cost_source,
    is_proxied_endpoint,
    read_decline_artifacts,
    resolve_cost_usd,
    resolve_terminal_kind,
)
from .plan_predicates import is_scorable_plan
from .replay_frame import REPLAY_FRAME_ID, ReplayFramedBriefingAssembler
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


def architect_eval_mcp_scope(worktree: Path) -> dict[str, Any]:
    """The ONLY MCP an eval architect may reach: plan-tools, strictly scoped.

    The PAIR is the invariant (task 4757). A plan-tools config without the
    strict flag still ambient-merges the worktree's tracked ``.mcp.json``, i.e.
    the live escalation (:8102) and fused-memory (:8002) servers.
    """
    return {'mcp_config': _inject_plan_tools_mcp(None, worktree), 'strict_mcp_config': True}


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


@dataclass(frozen=True)
class _Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_create_tokens: int = 0
    turns: int = 0
    duration_ms: int = 0

    @classmethod
    def of(cls, result: AgentResult) -> _Usage:
        """The last attempt's usage; a provider that did not report one counts 0."""
        return cls(
            input_tokens=result.input_tokens or 0,
            output_tokens=result.output_tokens or 0,
            cache_read_tokens=result.cache_read_tokens or 0,
            cache_create_tokens=result.cache_create_tokens or 0,
            turns=result.turns or 0,
            duration_ms=result.duration_ms,
        )


@dataclass(frozen=True)
class _Exit:
    """How the architect invoke ended, before declines and cost are attached."""

    outcome: str
    plan: dict
    usage: _Usage = dataclasses.field(default_factory=_Usage)
    unmeasured: Unmeasured | None = None


async def invoke_architect(
    task: dict,
    config: EvalConfig,
    deps: ArchitectEvalDeps,
    *,
    build_orch_config: Callable[[], OrchestratorConfig],
    usage_gate: InjectedGate,
    timeout_minutes: float,
) -> ArchitectAttempt:
    """Invoke the architect once in a fresh worktree and record what it left behind."""
    project_root = Path(task['project_root'])
    worktree, run_id = await deps.create_worktree(
        project_root, task['id'], task['pre_task_commit'],
        setup_commands=task.get('setup_commands'),
    )
    address = CellName(task['id'], config.name, run_id)
    call = _ArchitectCall(
        task, config, deps, worktree, build_orch_config, usage_gate, timeout_minutes,
    )
    try:
        ended = await call.run()
    finally:
        declines = _read_declines(call.artifacts, call.label)
        await deps.cleanup_worktree(project_root, worktree)
    is_local_model = is_proxied_endpoint(config.env_overrides)
    cost_usd, cost_source = _architect_cost(
        ended.usage, model=config.model, prices=call.prices,
        spent_usd=call.spent_usd, is_local_model=is_local_model,
    )
    usage = ended.usage
    return ArchitectAttempt(
        address=address,
        worktree=worktree,
        outcome=ended.outcome,
        plan=ended.plan,
        declines=declines,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=usage.cache_read_tokens,
        cache_create_tokens=usage.cache_create_tokens,
        turns=usage.turns,
        duration_ms=usage.duration_ms,
        cost_usd=cost_usd,
        cost_source=cost_source,
        is_local_model=is_local_model,
        unmeasured=ended.unmeasured,
    )


@dataclass
class _ArchitectCall:
    """One architect invocation in an existing worktree, and how far it got."""

    task: dict
    config: EvalConfig
    deps: ArchitectEvalDeps
    worktree: Path
    build_orch_config: Callable[[], OrchestratorConfig]
    usage_gate: InjectedGate
    timeout_minutes: float
    artifacts: TaskArtifacts | None = None
    prices: dict[str, Any] | None = None
    spent_usd: float = 0.0

    @property
    def label(self) -> str:
        return cell_label(self.task, self.config)

    async def run(self) -> _Exit:
        try:
            result, artifacts = await self._ask()
        except TimeoutError:
            return _timed_out(self.label, self.timeout_minutes, self.spent_usd)
        except AllAccountsCappedException as e:
            return _cap_exhausted(self.label, e, self.artifacts, self.spent_usd)
        except Exception as e:
            return _harness_error(self.label, e, self.spent_usd)
        return _read_the_answer(self.label, result, artifacts, self.config, self.spent_usd)

    async def _ask(self) -> tuple[AgentResult, TaskArtifacts]:
        orch_config = self.build_orch_config()
        self.prices = orch_config.prices
        artifacts = self.artifacts = self.deps.open_artifacts(self.worktree)
        pre = self.task['pre_task_commit']
        task_def = self.task.get('task_definition', {})
        artifacts.init(
            self.task['id'], task_def.get('title', ''), task_def.get('description', ''),
            base_commit=pre,
        )
        briefing = ReplayFramedBriefingAssembler(orch_config, base_commit=pre)
        prompt = await briefing.build_architect_prompt(task_def, worktree=self.worktree)
        async with owned_usage_gate(
            self.usage_gate, orch_config, factory=self.deps.gate_factory, label=self.label,
        ) as gate:
            result = await invoke_with_cap_retry(
                gate,
                f'Architect eval {self.label}',
                cap_wait_sanity_secs=self.deps.cap_wait_secs,
                invoke_fn=self._timed_attempt,
                backend=self.config.backend,
                **_architect_invocation(self.task, self.config, prompt, self.worktree),
            )
        return result, artifacts

    async def _timed_attempt(self, **kwargs: Any) -> AgentResult:
        """One attempt bounded by the operator's timeout, its spend banked as it returns."""
        result = await asyncio.wait_for(
            self.deps.invoke(**kwargs), timeout=self.timeout_minutes * 60,
        )
        self.spent_usd += coerce_cost_usd(result.cost_usd)
        return result


def _architect_invocation(
    task: dict, config: EvalConfig, prompt: str, worktree: Path,
) -> dict[str, Any]:
    """This candidate's architect invocation, scoped to plan-tools alone."""
    return {
        'prompt': prompt,
        'system_prompt': ARCHITECT.system_prompt,
        'cwd': worktree,
        'model': config.model,
        'max_turns': task.get('max_architect_turns', 50),
        'max_budget_usd': config.max_budget_usd,
        'allowed_tools': ARCHITECT.allowed_tools or None,
        'disallowed_tools': ARCHITECT.disallowed_tools or None,
        'effort': config.effort or 'high',
        'env_overrides': config.env_overrides or None,
        **architect_eval_mcp_scope(worktree),
    }


def _read_the_answer(
    label: str,
    result: AgentResult,
    artifacts: TaskArtifacts,
    config: EvalConfig,
    spent_usd: float,
) -> _Exit:
    usage = _Usage.of(result)
    refusal = _refusal(label, result, config, spent_usd)
    try:
        plan = artifacts.read_plan() or {}
    except Exception as e:
        logger.error(f'Architect eval {label} failed: {e}')
        return _Exit('blocked', {}, usage, refusal or _harness_marker(e, spent_usd))
    return _Exit('done' if result.success else 'blocked', plan, usage, refusal)


def _refusal(
    label: str, result: AgentResult, config: EvalConfig, spent_usd: float,
) -> Unmeasured | None:
    """The transport refusal behind *result*, carrying every attempt's spend."""
    try:
        refused = classify_unmeasured(
            result, stage=MeasurementStage.ARCHITECT, backend=config.backend,
        )
    except Exception:
        logger.warning(
            f'invocation-error classification raised for {label}; leaving the cell unmarked',
            exc_info=True,
        )
        return None
    return dataclasses.replace(refused, cost_usd=spent_usd) if refused is not None else None


def _timed_out(label: str, timeout_minutes: float, spent_usd: float) -> _Exit:
    logger.error(f'Architect eval {label} timed out after {timeout_minutes}m')
    marker = Unmeasured(
        MeasurementStage.ARCHITECT, UnmeasuredCause.TIMEOUT,
        f'no answer within {timeout_minutes}m', cost_usd=spent_usd,
    )
    return _Exit('timeout', {}, unmeasured=marker)


def _cap_exhausted(
    label: str,
    exc: AllAccountsCappedException,
    artifacts: TaskArtifacts | None,
    spent_usd: float,
) -> _Exit:
    logger.error(
        f'Architect eval {label}: all accounts capped after {exc.retries} retries '
        f'({exc.elapsed_secs:.1f}s) — giving up on this cell rather than stalling '
        f'the campaign'
    )
    marker = Unmeasured(
        MeasurementStage.ARCHITECT, UnmeasuredCause.CAP_EXHAUSTED,
        clip_reason(str(exc)), cost_usd=spent_usd,
    )
    return _Exit('blocked', _reread_plan(label, artifacts), unmeasured=marker)


def _reread_plan(label: str, artifacts: TaskArtifacts | None) -> dict:
    """A plan written before the pool ran dry, or ``{}``; never a raise."""
    if artifacts is None:
        return {}
    try:
        return artifacts.read_plan() or {}
    except Exception:
        logger.warning(
            f'plan re-read after cap exhaustion failed for {label}; cell stays unmeasurable',
            exc_info=True,
        )
        return {}


def _harness_error(label: str, exc: Exception, spent_usd: float) -> _Exit:
    logger.error(f'Architect eval {label} failed: {exc}')
    return _Exit('blocked', {}, unmeasured=_harness_marker(exc, spent_usd))


def _harness_marker(exc: Exception, spent_usd: float) -> Unmeasured:
    return Unmeasured(
        MeasurementStage.ARCHITECT, UnmeasuredCause.HARNESS_ERROR,
        exception_reason(exc), cost_usd=spent_usd,
    )


def _read_declines(
    artifacts: TaskArtifacts | None, label: str,
) -> Mapping[str, dict | None]:
    """The decline artifacts, read while the meta root still exists; never a raise."""
    try:
        return read_decline_artifacts(artifacts)
    except Exception:
        logger.warning(
            f'decline-artifact read failed for {label}; the cell keeps its '
            f'plan-derived terminal kind',
            exc_info=True,
        )
        return {}


def _architect_cost(
    usage: _Usage,
    *,
    model: str,
    prices: dict[str, Any] | None,
    spent_usd: float,
    is_local_model: bool,
) -> tuple[float, str]:
    """The architect component's ``(cost_usd, cost_source)`` per Invariant P5."""
    if not (usage.input_tokens or usage.output_tokens or spent_usd):
        return 0.0, 'cli'
    return resolve_cost_usd(
        usage.input_tokens, usage.output_tokens, model=model, prices=prices,
        cli_cost_usd=spent_usd, is_local_model=is_local_model,
    )


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
