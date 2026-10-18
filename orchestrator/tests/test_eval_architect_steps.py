"""The architect eval cell's steps, driven one at a time (task 6737).

``orchestrator.evals.architect_eval`` splits one architect cell into
invoke → classify → assemble. The pure steps (``classify_attempt``,
``assemble_metrics``) are driven here on hand-built ``ArchitectAttempt`` values;
the judge, reference-diff and invoke steps, and ``runner.run_architect_eval``'s
composition of them, through an injected ``ArchitectEvalDeps``. No collaborator
is patched by dotted path. The taint table and the rest of the rationale live in
docs/plan-scoring-and-judge.md, "The architect cell: invoke, classify, assemble
(task 6737)".
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _campaign_gate_helpers import eval_base_config
from _orch_helpers import build_usage_gate
from shared.cli_invoke import AgentResult, AllAccountsCappedException
from shared.config_models import AccountConfig
from shared.testing import make_gate_mock

from orchestrator.agents.briefing import BriefingAssembler
from orchestrator.agents.invoke import invoke_agent
from orchestrator.config import OrchestratorConfig
from orchestrator.evals import eval_gate, runner, snapshots
from orchestrator.evals.architect_eval import (
    ArchitectAttempt,
    ArchitectEvalDeps,
    PlanJudgement,
    PlanScoring,
    assemble_metrics,
    classify_attempt,
    invoke_architect,
    judge_plan,
    open_eval_artifacts,
    read_reference_diff,
)
from orchestrator.evals.configs import EvalConfig
from orchestrator.evals.judge import (
    PlanQualityVerdict,
    judge_plan_quality,
    score_plan_structure,
)
from orchestrator.evals.measurement import (
    MeasurementStage,
    Scored,
    Unmeasured,
    UnmeasuredCause,
)
from orchestrator.evals.metrics import _DECLINE_READERS, DECLINE_KINDS
from orchestrator.evals.replay_frame import REPLAY_FRAME_ID
from orchestrator.evals.results import CellName, ResultStore

_STEPS_LOGGER = 'orchestrator.evals.architect_eval'


def _scorable_plan() -> dict:
    return {
        'steps': [
            {'id': 'step-1', 'type': 'test', 'description': 'RED test for X'},
            {'id': 'step-2', 'type': 'impl', 'description': 'GREEN implement X'},
        ],
        'files': ['pkg/mod.py'],
    }


def _create_plan_stub() -> dict:
    return {
        'task_id': 't', 'title': 'x', 'analysis': 'a', 'files': [],
        'prerequisites': [], 'steps': [], 'design_decisions': [], 'reuse': [],
    }


def _attempt(**over) -> ArchitectAttempt:
    fields = {
        'address': CellName('df_task_2605', 'architect-sonnet-high', 'run-abc'),
        'worktree': Path('/fake/wt'),
        'outcome': 'done',
        'plan': {},
        'declines': dict.fromkeys(DECLINE_KINDS),
    }
    return ArchitectAttempt(**(fields | over))


def _architect_unmeasured(cause: UnmeasuredCause, cost_usd: float = 0.0) -> Unmeasured:
    return Unmeasured(MeasurementStage.ARCHITECT, cause, f'{cause.value} happened', cost_usd)


def _scored(plan_quality: float = 0.77, cost_usd: float = 0.4) -> Scored:
    return Scored(
        PlanQualityVerdict(plan_quality=plan_quality, per_criterion={}, reasoning='ok'),
        cost_usd=cost_usd,
    )


_TRANSPORT_CAUSES = [
    UnmeasuredCause.CAP_HIT, UnmeasuredCause.AUTH_FAILED,
    UnmeasuredCause.MODEL_NOT_FOUND, UnmeasuredCause.API_ERROR,
    UnmeasuredCause.SERVER_ERROR, UnmeasuredCause.WEDGE,
]


class TestClassifyAttempt:
    """The taint table, one row per exit."""

    def test_a_healthy_attempt_with_a_scorable_plan_is_judged(self):
        assert classify_attempt(_attempt(plan=_scorable_plan())) is PlanScoring.JUDGE

    @pytest.mark.parametrize('plan', [
        {}, {'steps': []}, {'steps': None}, _create_plan_stub(),
    ], ids=['empty', 'steps-empty', 'steps-none', 'create-plan-stub'])
    def test_a_healthy_attempt_with_a_stepless_plan_is_floored(self, plan):
        assert classify_attempt(_attempt(plan=plan)) is PlanScoring.FLOORED

    @pytest.mark.parametrize('cause', _TRANSPORT_CAUSES, ids=str)
    def test_a_transport_refusal_with_no_plan_is_tainted(self, cause):
        attempt = _attempt(outcome='blocked', unmeasured=_architect_unmeasured(cause))

        assert classify_attempt(attempt) is PlanScoring.TAINTED

    @pytest.mark.parametrize('cause', _TRANSPORT_CAUSES, ids=str)
    def test_a_transport_refusal_that_left_a_scorable_plan_is_floored(self, cause):
        attempt = _attempt(
            outcome='blocked', plan=_scorable_plan(),
            unmeasured=_architect_unmeasured(cause),
        )

        assert classify_attempt(attempt) is PlanScoring.FLOORED

    @pytest.mark.parametrize(('plan', 'expected'), [
        ({}, PlanScoring.TAINTED),
        (_scorable_plan(), PlanScoring.FLOORED),
    ], ids=['no-plan', 'scorable-plan'])
    def test_cap_exhaustion_follows_the_transport_rows(self, plan, expected):
        attempt = _attempt(
            outcome='blocked', plan=plan,
            unmeasured=_architect_unmeasured(UnmeasuredCause.CAP_EXHAUSTED),
        )

        assert classify_attempt(attempt) is expected

    def test_a_harness_error_with_no_plan_is_tainted(self):
        attempt = _attempt(
            outcome='blocked',
            unmeasured=_architect_unmeasured(UnmeasuredCause.HARNESS_ERROR),
        )

        assert classify_attempt(attempt) is PlanScoring.TAINTED

    def test_a_timeout_with_no_plan_is_floored_not_tainted(self):
        attempt = _attempt(
            outcome='timeout',
            unmeasured=_architect_unmeasured(UnmeasuredCause.TIMEOUT),
        )

        assert classify_attempt(attempt) is PlanScoring.FLOORED

    def test_a_timeout_with_a_scorable_plan_is_judged_on_its_content(self):
        attempt = _attempt(
            outcome='timeout', plan=_scorable_plan(),
            unmeasured=_architect_unmeasured(UnmeasuredCause.TIMEOUT),
        )

        assert classify_attempt(attempt) is PlanScoring.JUDGE

    @pytest.mark.parametrize(
        'cause', [c for c in UnmeasuredCause if c is not UnmeasuredCause.TIMEOUT], ids=str,
    )
    def test_every_cause_but_timeout_taints_a_planless_attempt(self, cause):
        """A new cause lands on the tainted row unless it is deliberately exempted."""
        attempt = _attempt(outcome='blocked', unmeasured=_architect_unmeasured(cause))

        assert classify_attempt(attempt) is PlanScoring.TAINTED


class TestAssembleMetrics:
    def test_a_tainted_cell_has_no_score_and_says_why(self):
        refusal = _architect_unmeasured(UnmeasuredCause.CAP_HIT)
        attempt = _attempt(outcome='blocked', unmeasured=refusal)

        metrics = assemble_metrics(attempt, PlanScoring.TAINTED, None)

        assert metrics.plan_quality is None
        assert metrics.cap_tainted is True
        assert metrics.judge_invocations == 0
        assert metrics.judge_cost_usd == 0.0
        assert metrics.invocation_errors == [refusal.to_record()]

    def test_a_floored_cell_scores_the_structural_floor(self):
        plan = _scorable_plan()
        attempt = _attempt(
            outcome='blocked', plan=plan,
            unmeasured=_architect_unmeasured(UnmeasuredCause.CAP_HIT),
        )

        metrics = assemble_metrics(attempt, PlanScoring.FLOORED, None)

        assert metrics.plan_quality == score_plan_structure(plan)
        assert metrics.cap_tainted is False

    @pytest.mark.parametrize('blind', [False, True])
    def test_a_scored_judgement_is_kept_and_its_spend_added(self, blind):
        attempt = _attempt(plan=_scorable_plan(), cost_usd=1.23)

        metrics = assemble_metrics(
            attempt, PlanScoring.JUDGE, PlanJudgement(_scored(0.77, 0.4), blind=blind),
        )

        assert metrics.plan_quality == 0.77
        assert metrics.judge_invocations == 1
        assert metrics.judge_cost_usd == pytest.approx(0.4)
        assert metrics.cost_usd == pytest.approx(1.23 + 0.4)
        assert metrics.judged_without_reference is blind
        assert metrics.invocation_errors is None

    @pytest.mark.parametrize('blind', [False, True])
    def test_an_unparseable_judgement_floors_and_records_after_the_architect(self, blind):
        plan = _scorable_plan()
        timeout = _architect_unmeasured(UnmeasuredCause.TIMEOUT)
        unparseable = Unmeasured(
            MeasurementStage.JUDGE, UnmeasuredCause.UNPARSEABLE, 'bad json', cost_usd=0.6,
        )
        attempt = _attempt(outcome='timeout', plan=plan, unmeasured=timeout)

        metrics = assemble_metrics(
            attempt, PlanScoring.JUDGE, PlanJudgement(unparseable, blind=blind),
        )

        assert metrics.plan_quality == score_plan_structure(plan)
        assert metrics.invocation_errors == [timeout.to_record(), unparseable.to_record()]
        assert metrics.judge_invocations == 1
        assert metrics.judge_cost_usd == pytest.approx(0.6)
        assert metrics.judged_without_reference is False

    def test_a_raised_judgement_floors_and_credits_no_call(self):
        plan = _scorable_plan()
        raised = Unmeasured(
            MeasurementStage.JUDGE, UnmeasuredCause.RAISED, 'RuntimeError: x', cost_usd=0.0,
        )

        metrics = assemble_metrics(
            _attempt(plan=plan, cost_usd=1.23), PlanScoring.JUDGE,
            PlanJudgement(raised, blind=False),
        )

        assert metrics.plan_quality == score_plan_structure(plan)
        assert metrics.judge_invocations == 0
        assert metrics.judge_cost_usd == 0.0
        assert metrics.cost_usd == pytest.approx(1.23)
        assert metrics.invocation_errors == [raised.to_record()]

    @pytest.mark.parametrize(('attempt_source', 'judge_cost', 'expected'), [
        ('price_table', 0.4, 'mixed'),
        ('cli', 0.4, 'cli'),
        ('price_table', 0.0, 'price_table'),
        ('unpriced_proxy', 0.0, 'unpriced_proxy'),
    ])
    def test_cost_source_composes_the_two_components(
        self, attempt_source, judge_cost, expected,
    ):
        attempt = _attempt(plan=_scorable_plan(), cost_source=attempt_source)

        metrics = assemble_metrics(
            attempt, PlanScoring.JUDGE,
            PlanJudgement(_scored(cost_usd=judge_cost), blind=False),
        )

        assert metrics.cost_source == expected

    def test_terminal_kind_comes_from_the_declines(self):
        declines = dict.fromkeys(DECLINE_KINDS) | {
            'false_premise': {'reason': 'r', 'reported_at': '2026-07-19T12:00:00+00:00'},
        }

        metrics = assemble_metrics(
            _attempt(declines=declines), PlanScoring.FLOORED, None,
        )

        assert metrics.terminal_kind == 'false_premise'

    def test_fixed_architect_cell_stamps(self):
        metrics = assemble_metrics(_attempt(), PlanScoring.FLOORED, None)

        assert metrics.role_under_test == 'architect'
        assert metrics.replay_frame == REPLAY_FRAME_ID
        assert metrics.tests_pass is None

    def test_plan_steps_tolerates_a_null_steps_list(self):
        metrics = assemble_metrics(
            _attempt(plan={'steps': None}), PlanScoring.FLOORED, None,
        )

        assert metrics.plan_steps == 0

    def test_usage_is_copied_from_the_attempt(self):
        attempt = _attempt(
            plan=_scorable_plan(), input_tokens=12_000, output_tokens=3_000,
            cache_read_tokens=48_000, cache_create_tokens=6_000, turns=9,
            duration_ms=4567, is_local_model=True,
        )

        metrics = assemble_metrics(
            attempt, PlanScoring.JUDGE, PlanJudgement(_scored(), blind=False),
        )

        assert metrics.input_tokens == 12_000
        assert metrics.output_tokens == 3_000
        assert metrics.cache_read_tokens == 48_000
        assert metrics.cache_create_tokens == 6_000
        assert metrics.turns_used == 9
        assert metrics.workflow_duration_ms == 4567
        assert metrics.is_local_model is True
        assert metrics.tokens_per_second == round(3_000 / (4567 / 1000), 2)

    def test_a_zero_duration_attempt_reports_zero_tokens_per_second(self):
        metrics = assemble_metrics(
            _attempt(output_tokens=3_000, duration_ms=0), PlanScoring.FLOORED, None,
        )

        assert metrics.tokens_per_second == 0.0


def _arch_task(**over) -> dict:
    task = {
        'id': 'df_task_2605',
        'project_root': '/fake/project',
        'pre_task_commit': 'basecommit123',
        'reference': {'post_task_commit': 'postcommit456'},
        'task_definition': {'title': 'Widget', 'description': 'implement the widget'},
    }
    return task | over


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


@pytest.mark.asyncio
class TestJudgePlan:
    async def test_the_judge_sees_the_plan_the_diff_and_the_task(self, tmp_path: Path):
        verdict = _scored()
        judge = AsyncMock(return_value=verdict)
        deps = ArchitectEvalDeps(store=ResultStore(tmp_path), judge=judge)
        plan, task = _scorable_plan(), _arch_task()

        judgement = await judge_plan(plan, '--- a\n+++ b\n', task, deps)

        judge.assert_awaited_once_with(plan, '--- a\n+++ b\n', task)
        assert judgement == PlanJudgement(verdict, blind=False)

    async def test_an_empty_reference_judges_blind_and_says_so_once(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ):
        caplog.set_level(logging.WARNING, logger=_STEPS_LOGGER)
        deps = ArchitectEvalDeps(
            store=ResultStore(tmp_path), judge=AsyncMock(return_value=_scored()),
        )

        judgement = await judge_plan(_scorable_plan(), '', _arch_task(), deps)

        assert judgement.blind is True
        empty_lines = [m for m in _warnings(caplog) if 'EMPTY reference diff' in m]
        assert len(empty_lines) == 1
        assert 'df_task_2605' in empty_lines[0]

    async def test_a_raising_judge_becomes_an_unmeasured_raise(self, tmp_path: Path):
        deps = ArchitectEvalDeps(
            store=ResultStore(tmp_path),
            judge=AsyncMock(side_effect=RuntimeError('judge exploded')),
        )

        judgement = await judge_plan(_scorable_plan(), 'diff', _arch_task(), deps)

        assert judgement.outcome == Unmeasured(
            MeasurementStage.JUDGE, UnmeasuredCause.RAISED,
            'RuntimeError: judge exploded', cost_usd=0.0,
        )


@pytest.mark.asyncio
class TestReadReferenceDiff:
    async def test_the_landed_diff_is_read_between_the_fixture_commits(
        self, tmp_path: Path,
    ):
        reader = AsyncMock(return_value='the landed diff')
        deps = ArchitectEvalDeps(store=ResultStore(tmp_path), read_reference_diff=reader)

        diff = await read_reference_diff(_arch_task(), deps)

        assert diff == 'the landed diff'
        reader.assert_awaited_once_with(
            Path('/fake/project'), 'basecommit123', 'postcommit456',
        )

    async def test_a_failing_read_degrades_to_an_empty_diff(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ):
        caplog.set_level(logging.WARNING, logger=_STEPS_LOGGER)
        deps = ArchitectEvalDeps(
            store=ResultStore(tmp_path),
            read_reference_diff=AsyncMock(side_effect=RuntimeError('git exploded')),
        )

        diff = await read_reference_diff(_arch_task(), deps)

        assert diff == ''
        assert any('git exploded' in m for m in _warnings(caplog))

    async def test_a_fixture_without_a_reference_says_so_once(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ):
        caplog.set_level(logging.WARNING, logger=_STEPS_LOGGER)
        reader = AsyncMock()
        deps = ArchitectEvalDeps(store=ResultStore(tmp_path), read_reference_diff=reader)
        task = _arch_task()
        del task['reference']

        diff = await read_reference_diff(task, deps)

        assert diff == ''
        reader.assert_not_awaited()
        lines = [m for m in _warnings(caplog) if 'reference.post_task_commit' in m]
        assert len(lines) == 1
        assert 'df_task_2605' in lines[0]


class TestArchitectEvalDeps:
    def test_the_defaults_are_the_production_collaborators(self, tmp_path: Path):
        deps = ArchitectEvalDeps(store=ResultStore(tmp_path))

        assert deps.create_worktree is snapshots.create_eval_worktree
        assert deps.cleanup_worktree is snapshots.cleanup_eval_worktree
        assert deps.open_artifacts is open_eval_artifacts
        assert deps.invoke is invoke_agent
        assert deps.judge is judge_plan_quality
        assert deps.read_reference_diff is snapshots.get_diff_between_commits
        assert deps.gate_factory is eval_gate.build_eval_usage_gate
        assert deps.cap_wait_secs == eval_gate.EVAL_CAP_WAIT_SANITY_SECS

    def test_every_dep_is_frozen(self, tmp_path: Path):
        deps = ArchitectEvalDeps(store=ResultStore(tmp_path))

        for field in dataclasses.fields(deps):
            with pytest.raises(dataclasses.FrozenInstanceError):
                setattr(deps, field.name, None)


_REAL_SLEEP = asyncio.sleep
_DECLINE_READS = frozenset(_DECLINE_READERS.values())
_CAP_TEXT = "You've hit your session limit · resets 8pm"
_CFG = EvalConfig('architect-sonnet-high', 'claude', 'sonnet', 'high', role='architect')


def _answered(**over) -> AgentResult:
    fields = {
        'success': True, 'output': 'wrote the plan', 'cost_usd': 1.23,
        'duration_ms': 4567, 'turns': 9, 'input_tokens': 12_000,
        'output_tokens': 3_000, 'cache_read_tokens': 48_000,
        'cache_create_tokens': 6_000,
    }
    return AgentResult(**(fields | over))


def _capped(**over) -> AgentResult:
    fields = {
        'success': False, 'output': _CAP_TEXT, 'cost_usd': 0.9,
        'duration_ms': 1200, 'turns': 2, 'input_tokens': 500,
        'output_tokens': 40, 'subtype': 'error', 'api_error_status': 429,
    }
    return AgentResult(**(fields | over))


def _one_account_gate():
    return build_usage_gate(
        [AccountConfig(name='max-only', oauth_token_env='CLAUDE_OAUTH_ONLY')],
        ['tok-only'],
    )


class _FakeArtifacts:
    """A TaskArtifacts double that appends every read and write to ``events``."""

    def __init__(
        self,
        events: list[str],
        *,
        plan: dict | None,
        plan_error: Exception | None,
        declines: Mapping[str, dict],
    ):
        self._events = events
        self._plan = plan
        self._plan_error = plan_error
        self.initialised_with: tuple | None = None
        for kind, reader in _DECLINE_READERS.items():
            setattr(self, reader, self._recording_reader(reader, declines.get(kind)))

    def _recording_reader(
        self, name: str, artifact: dict | None,
    ) -> Callable[[], dict | None]:
        def read() -> dict | None:
            self._events.append(name)
            return artifact
        return read

    def init(
        self, task_id: str, task_title: str, task_description: str,
        base_commit: str | None = None,
    ) -> None:
        self._events.append('init')
        self.initialised_with = (task_id, task_title, task_description, base_commit)

    def read_plan(self) -> dict | None:
        self._events.append('read_plan')
        if self._plan_error is not None:
            raise self._plan_error
        return self._plan


@dataclasses.dataclass
class _Cell:
    """One architect cell's doubles, the deps that carry them, and what they saw."""

    events: list[str]
    artifacts: _FakeArtifacts
    project_root: Path
    worktree: Path
    open_artifacts: MagicMock
    cleanup_worktree: AsyncMock
    invoke: AsyncMock
    gate_factory: AsyncMock
    deps: ArchitectEvalDeps

    @property
    def task(self) -> dict:
        return _arch_task(project_root=str(self.project_root))

    def orch_config(self) -> OrchestratorConfig:
        return runner.build_eval_orch_config(
            _CFG, self.task, eval_base_config(self.project_root),
        )


def _cell(
    tmp_path: Path,
    *,
    plan: dict | None = None,
    plan_error: Exception | None = None,
    declines: Mapping[str, dict] | None = None,
    invoke: AsyncMock | None = None,
    gate_factory: AsyncMock | None = None,
    cap_wait_secs: float = eval_gate.EVAL_CAP_WAIT_SANITY_SECS,
) -> _Cell:
    events: list[str] = []
    artifacts = _FakeArtifacts(
        events, plan=plan, plan_error=plan_error, declines=declines or {},
    )
    worktree = tmp_path / 'wt'

    async def create_worktree(*_args, **_kwargs) -> tuple[Path, str]:
        events.append('create_worktree')
        return worktree, 'run-abc'

    async def cleanup_worktree(*_args) -> None:
        events.append('cleanup_worktree')

    open_artifacts = MagicMock(return_value=artifacts)
    cleanup = AsyncMock(side_effect=cleanup_worktree)
    invoke = invoke if invoke is not None else AsyncMock(return_value=_answered())
    gate_factory = (
        gate_factory if gate_factory is not None else AsyncMock(return_value=None)
    )
    deps = ArchitectEvalDeps(
        store=ResultStore(tmp_path / 'results'),
        create_worktree=AsyncMock(side_effect=create_worktree),
        cleanup_worktree=cleanup,
        open_artifacts=open_artifacts,
        invoke=invoke,
        judge=AsyncMock(return_value=_scored()),
        read_reference_diff=AsyncMock(return_value='--- a\n+++ b\n'),
        gate_factory=gate_factory,
        cap_wait_secs=cap_wait_secs,
    )
    return _Cell(
        events, artifacts, tmp_path, worktree, open_artifacts, cleanup, invoke,
        gate_factory, deps,
    )


async def _invoke(
    cell: _Cell,
    *,
    build_orch_config: Callable[[], OrchestratorConfig] | None = None,
    timeout_minutes: float = 60,
    usage_gate: eval_gate.InjectedGate = eval_gate.GATE_UNSET,
) -> ArchitectAttempt:
    return await invoke_architect(
        cell.task, _CFG, cell.deps,
        build_orch_config=build_orch_config or cell.orch_config,
        usage_gate=usage_gate,
        timeout_minutes=timeout_minutes,
    )


def _raise_config_error() -> OrchestratorConfig:
    raise RuntimeError('config exploded')


def _usage(attempt: ArchitectAttempt) -> tuple[int, ...]:
    return (
        attempt.input_tokens, attempt.output_tokens, attempt.cache_read_tokens,
        attempt.cache_create_tokens, attempt.turns, attempt.duration_ms,
    )


def _marker(attempt: ArchitectAttempt) -> Unmeasured:
    assert attempt.unmeasured is not None, 'the attempt carries no unmeasured marker'
    assert attempt.unmeasured.stage is MeasurementStage.ARCHITECT
    return attempt.unmeasured


@pytest.fixture
def production_briefing(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Stubs the production briefing; the replay-frame subclass still appends its block."""
    prompt = AsyncMock(return_value='ARCH PROMPT')
    monkeypatch.setattr(BriefingAssembler, 'build_architect_prompt', prompt)
    return prompt


@pytest.mark.asyncio
@pytest.mark.usefixtures('production_briefing')
class TestInvokeArchitectExits:
    """invoke_architect's exit table, one case per row."""

    async def test_an_answer_is_read_off_the_result_and_the_plan(self, tmp_path: Path):
        cell = _cell(tmp_path, plan=_scorable_plan())

        attempt = await _invoke(cell)

        assert attempt.outcome == 'done'
        assert attempt.unmeasured is None
        assert attempt.plan == _scorable_plan()
        assert cell.events.count('read_plan') == 1
        assert _usage(attempt) == (12_000, 3_000, 48_000, 6_000, 9, 4567)
        assert (attempt.cost_usd, attempt.cost_source) == (pytest.approx(1.23), 'cli')
        assert attempt.is_local_model is False
        assert attempt.address == CellName('df_task_2605', _CFG.name, 'run-abc')
        assert attempt.worktree == cell.worktree

    async def test_the_artifacts_are_opened_at_the_worktree_for_the_task(
        self, tmp_path: Path,
    ):
        cell = _cell(tmp_path, plan=_scorable_plan())

        await _invoke(cell)

        cell.open_artifacts.assert_called_once_with(cell.worktree)
        assert cell.artifacts.initialised_with == (
            'df_task_2605', 'Widget', 'implement the widget', 'basecommit123',
        )

    async def test_an_unanswered_result_is_blocked(self, tmp_path: Path):
        cell = _cell(
            tmp_path, plan=_scorable_plan(),
            invoke=AsyncMock(return_value=_answered(success=False, output='gave up')),
        )

        attempt = await _invoke(cell)

        assert attempt.outcome == 'blocked'
        assert attempt.plan == _scorable_plan()

    async def test_a_refusal_is_marked_with_the_banked_spend(self, tmp_path: Path):
        cell = _cell(tmp_path, invoke=AsyncMock(return_value=_capped(cost_usd=0.9)))

        attempt = await _invoke(cell)

        assert attempt.outcome == 'blocked'
        marker = _marker(attempt)
        assert marker.cause is UnmeasuredCause.CAP_HIT
        assert marker.cost_usd == pytest.approx(0.9)
        assert attempt.cost_usd == pytest.approx(0.9)

    @pytest.mark.parametrize(('result', 'cause'), [
        (_capped(), UnmeasuredCause.CAP_HIT),
        (_answered(), UnmeasuredCause.HARNESS_ERROR),
    ], ids=['after-a-refusal', 'after-an-answer'])
    async def test_an_unreadable_plan_keeps_the_result_and_the_more_specific_marker(
        self, tmp_path: Path, result: AgentResult, cause: UnmeasuredCause,
    ):
        cell = _cell(
            tmp_path, plan_error=OSError('plan.json unreadable'),
            invoke=AsyncMock(return_value=result),
        )

        attempt = await _invoke(cell)

        assert attempt.outcome == 'blocked'
        assert attempt.plan == {}
        marker = _marker(attempt)
        assert marker.cause is cause
        assert marker.cost_usd == pytest.approx(result.cost_usd)
        assert attempt.input_tokens == result.input_tokens
        assert attempt.output_tokens == result.output_tokens
        assert attempt.turns == result.turns
        assert attempt.duration_ms == result.duration_ms

    async def test_a_timeout_reads_no_plan_and_spends_nothing(self, tmp_path: Path):
        cell = _cell(
            tmp_path, plan=_scorable_plan(),
            invoke=AsyncMock(side_effect=TimeoutError()),
        )

        attempt = await _invoke(cell, timeout_minutes=7)

        assert attempt.outcome == 'timeout'
        assert attempt.unmeasured == Unmeasured(
            MeasurementStage.ARCHITECT, UnmeasuredCause.TIMEOUT,
            'no answer within 7m', cost_usd=0.0,
        )
        assert 'read_plan' not in cell.events
        assert attempt.plan == {}
        assert _usage(attempt) == (0, 0, 0, 0, 0, 0)
        assert (attempt.cost_usd, attempt.cost_source) == (0.0, 'cli')

    @pytest.mark.parametrize(('plan_error', 'plan'), [
        (None, _scorable_plan()),
        (OSError('plan.json unreadable'), {}),
    ], ids=['re-read-succeeds', 're-read-raises'])
    async def test_cap_exhaustion_re_reads_the_plan_once(
        self, tmp_path: Path, plan_error: Exception | None, plan: dict,
    ):
        cell = _cell(
            tmp_path, plan=_scorable_plan(), plan_error=plan_error,
            invoke=AsyncMock(side_effect=AllAccountsCappedException(3, 10.0, 'x')),
        )

        attempt = await _invoke(cell)

        assert attempt.outcome == 'blocked'
        assert _marker(attempt).cause is UnmeasuredCause.CAP_EXHAUSTED
        assert cell.events.count('read_plan') == 1
        assert attempt.plan == plan

    async def test_a_harness_error_before_the_artifacts_opens_nothing(
        self, tmp_path: Path,
    ):
        cell = _cell(tmp_path, plan=_scorable_plan())

        attempt = await _invoke(cell, build_orch_config=_raise_config_error)

        assert attempt.outcome == 'blocked'
        assert attempt.unmeasured == Unmeasured(
            MeasurementStage.ARCHITECT, UnmeasuredCause.HARNESS_ERROR,
            'RuntimeError: config exploded', cost_usd=0.0,
        )
        cell.open_artifacts.assert_not_called()
        cell.invoke.assert_not_awaited()
        assert attempt.declines == dict.fromkeys(DECLINE_KINDS)
        assert (attempt.cost_usd, attempt.cost_source) == (0.0, 'cli')
        assert attempt.address.run_id == 'run-abc'

    async def test_an_invoke_raise_is_a_harness_error(self, tmp_path: Path):
        cell = _cell(
            tmp_path, plan=_scorable_plan(),
            invoke=AsyncMock(side_effect=RuntimeError('architect exploded')),
        )

        attempt = await _invoke(cell)

        assert attempt.outcome == 'blocked'
        assert attempt.plan == {}
        assert _marker(attempt).cause is UnmeasuredCause.HARNESS_ERROR

    async def test_the_declines_are_what_the_artifacts_reported(self, tmp_path: Path):
        artifact = {'reason': 'r', 'reported_at': '2026-07-19T12:00:00+00:00'}
        cell = _cell(tmp_path, declines={'false_premise': artifact})

        attempt = await _invoke(cell)

        assert attempt.declines == dict.fromkeys(DECLINE_KINDS) | {
            'false_premise': artifact,
        }


_EXIT_ROWS: dict[str, Callable[[Path], _Cell]] = {
    'answered': lambda tmp_path: _cell(tmp_path, plan=_scorable_plan()),
    'refused': lambda tmp_path: _cell(
        tmp_path, invoke=AsyncMock(return_value=_capped()),
    ),
    'plan-unreadable': lambda tmp_path: _cell(tmp_path, plan_error=OSError('x')),
    'timeout': lambda tmp_path: _cell(
        tmp_path, invoke=AsyncMock(side_effect=TimeoutError()),
    ),
    'cap-exhausted': lambda tmp_path: _cell(
        tmp_path,
        invoke=AsyncMock(side_effect=AllAccountsCappedException(3, 10.0, 'x')),
    ),
    'invoke-raised': lambda tmp_path: _cell(
        tmp_path, invoke=AsyncMock(side_effect=RuntimeError('x')),
    ),
}


@pytest.mark.asyncio
@pytest.mark.usefixtures('production_briefing')
class TestInvokeArchitectCleansUpLast:
    """The decline artifacts live under the meta root cleanup removes."""

    @pytest.mark.parametrize('row', list(_EXIT_ROWS))
    async def test_every_exit_reads_every_decline_then_cleans_up_once(
        self, tmp_path: Path, row: str,
    ):
        cell = _EXIT_ROWS[row](tmp_path)

        await _invoke(cell)

        cell.cleanup_worktree.assert_awaited_once_with(
            Path(cell.task['project_root']), cell.worktree,
        )
        cleanup_at = cell.events.index('cleanup_worktree')
        assert set(cell.events[:cleanup_at]) >= _DECLINE_READS
        assert not _DECLINE_READS & set(cell.events[cleanup_at:])

    async def test_a_harness_error_before_the_artifacts_still_cleans_up(
        self, tmp_path: Path,
    ):
        cell = _cell(tmp_path)

        await _invoke(cell, build_orch_config=_raise_config_error)

        cell.cleanup_worktree.assert_awaited_once_with(
            Path(cell.task['project_root']), cell.worktree,
        )
        assert not _DECLINE_READS & set(cell.events)


@pytest.mark.asyncio
@pytest.mark.usefixtures('production_briefing')
class TestInvokeArchitectBounds:
    async def test_the_operator_timeout_bounds_an_attempt_not_the_cap_wait(
        self, tmp_path: Path,
    ):
        async def slow_slot(**_kwargs) -> str:
            await _REAL_SLEEP(0.2)
            return 'tok'

        gate = make_gate_mock(before_invoke=AsyncMock(side_effect=slow_slot))
        cell = _cell(tmp_path, plan=_scorable_plan())

        attempt = await _invoke(cell, usage_gate=gate, timeout_minutes=0.0005)

        assert attempt.outcome == 'done'
        assert attempt.unmeasured is None

    async def test_a_fully_capped_pool_gives_up_at_the_deps_patience(
        self, tmp_path: Path,
    ):
        cell = _cell(
            tmp_path, invoke=AsyncMock(return_value=_capped()), cap_wait_secs=0.0,
            gate_factory=AsyncMock(return_value=_one_account_gate()),
        )

        attempt = await asyncio.wait_for(_invoke(cell), timeout=30)

        assert attempt.outcome == 'blocked'
        assert _marker(attempt).cause is UnmeasuredCause.CAP_EXHAUSTED


@pytest.mark.asyncio
@pytest.mark.usefixtures('production_briefing')
class TestInvokeArchitectGateOwnership:
    async def test_an_omitted_gate_is_built_ridden_and_shut_down(self, tmp_path: Path):
        gate = make_gate_mock()
        cell = _cell(tmp_path, plan=_scorable_plan(),
                     gate_factory=AsyncMock(return_value=gate))
        orch = cell.orch_config()

        await _invoke(cell, build_orch_config=lambda: orch)

        cell.gate_factory.assert_awaited_once_with(orch)
        gate.before_invoke.assert_awaited()
        gate.shutdown.assert_awaited_once()

    async def test_a_built_gate_is_shut_down_when_the_invoke_raises(
        self, tmp_path: Path,
    ):
        gate = make_gate_mock()
        cell = _cell(
            tmp_path, gate_factory=AsyncMock(return_value=gate),
            invoke=AsyncMock(side_effect=RuntimeError('architect exploded')),
        )

        attempt = await _invoke(cell)

        assert _marker(attempt).cause is UnmeasuredCause.HARNESS_ERROR
        gate.shutdown.assert_awaited_once()

    async def test_an_injected_gate_is_ridden_and_never_shut_down(self, tmp_path: Path):
        gate = make_gate_mock()
        cell = _cell(tmp_path, plan=_scorable_plan())

        await _invoke(cell, usage_gate=gate)

        cell.gate_factory.assert_not_awaited()
        gate.before_invoke.assert_awaited()
        gate.shutdown.assert_not_awaited()

    async def test_an_injected_none_is_one_ungated_invoke(self, tmp_path: Path):
        cell = _cell(tmp_path, plan=_scorable_plan())

        await _invoke(cell, usage_gate=None)

        cell.gate_factory.assert_not_awaited()
        cell.invoke.assert_awaited_once()


def _write_fixture(tmp_path: Path) -> Path:
    fixture = tmp_path / 'df_task_2605.json'
    fixture.write_text(json.dumps(_arch_task(project_root=str(tmp_path))))
    return fixture


@pytest.mark.asyncio
@pytest.mark.usefixtures('production_briefing')
class TestRunArchitectEvalComposesTheSteps:
    async def test_a_decline_read_before_cleanup_reaches_the_persisted_cell(
        self, tmp_path: Path,
    ):
        decline = {'reason': 'premise is false', 'reported_at': '2026-07-19T12:00:00+00:00'}
        cell = _cell(tmp_path, declines={'false_premise': decline})

        result = await runner.run_architect_eval(
            _write_fixture(tmp_path), _CFG, eval_base_config(tmp_path), deps=cell.deps,
        )

        [persisted] = cell.deps.store.load_all()
        assert persisted.run_id == result.run_id == 'run-abc'
        assert persisted.metrics['terminal_kind'] == 'false_premise'
        assert (
            cell.events.index('read_false_premise')
            < cell.events.index('cleanup_worktree')
        )

    async def test_store_and_deps_together_are_refused(self, tmp_path: Path):
        cell = _cell(tmp_path)

        with pytest.raises(ValueError) as refused:
            await runner.run_architect_eval(
                _write_fixture(tmp_path), _CFG, eval_base_config(tmp_path),
                store=ResultStore(tmp_path / 'other'), deps=cell.deps,
            )

        assert 'store' in str(refused.value)
        assert 'deps' in str(refused.value)
        cell.invoke.assert_not_awaited()
