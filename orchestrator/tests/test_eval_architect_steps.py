"""The architect eval cell's steps, driven one at a time (task 6737).

``orchestrator.evals.architect_eval`` splits one architect cell into
invoke → classify → assemble. The pure steps (``classify_attempt``,
``assemble_metrics``) are driven here on hand-built ``ArchitectAttempt`` values,
and the judge and reference-diff steps through an injected
``ArchitectEvalDeps``; nothing is patched by dotted path. The taint table and
the rest of the rationale live in docs/plan-scoring-and-judge.md, "The
architect cell: invoke, classify, assemble (task 6737)".
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from orchestrator.agents.invoke import invoke_agent
from orchestrator.evals import eval_gate, snapshots
from orchestrator.evals.architect_eval import (
    ArchitectAttempt,
    ArchitectEvalDeps,
    PlanJudgement,
    PlanScoring,
    assemble_metrics,
    classify_attempt,
    judge_plan,
    open_eval_artifacts,
    read_reference_diff,
)
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
from orchestrator.evals.metrics import DECLINE_KINDS
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
