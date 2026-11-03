"""Verify events carry the plan hash, per-unit outcomes and task-leg duration (task 5651)."""

from __future__ import annotations

import dataclasses
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator import verify_evidence
from orchestrator.agents.invoke import AgentResult
from orchestrator.artifacts import ReviewAggregation, TaskArtifacts
from orchestrator.config import GitConfig, ModuleConfig, OrchestratorConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.git_ops import GitOps
from orchestrator.scheduler import TaskAssignment
from orchestrator.verify import VerifyResult, run_scoped_verification, run_verification
from orchestrator.verify_admission_record import AdmissionLeg
from orchestrator.verify_evidence import (
    CacheMode,
    UnitOutcome,
    UnitTier,
    VerifyUnit,
    module_leg_unit,
)
from orchestrator.verify_plan import ScopeKind, VerifyPlan, derive_verify_plan
from orchestrator.verify_runner import (
    UNSCOPED_TYPECHECK_FAILED_CATEGORY,
    LocalRunner,
    VerifyRunnerPool,
    build_merge_verify_spec,
    result_from_dict,
    result_to_dict,
)
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome

_ALPHA = ModuleConfig(
    prefix='alpha',
    test_command='uv run --project alpha pytest alpha/tests',
    lint_command='uv run --project alpha ruff check alpha',
    type_check_command='uv run --project alpha pyright alpha',
)
_ALPHA_SOURCE = 'alpha/src/alpha/core.py'
_TREE = {_ALPHA_SOURCE: 'x = 1\n'}


def _alpha_plan(
    role: Literal['merge', 'task'] = 'task', config: OrchestratorConfig | None = None,
) -> VerifyPlan:
    return derive_verify_plan([_ALPHA_SOURCE], [_ALPHA], config, _TREE.get, role=role)


def _lint_run_scope(plan: VerifyPlan) -> ScopeKind:
    return next(run.scope_kind for run in plan.runs if run.reason.startswith('lint:'))


def _result(*, passed: bool = True, summary: str = '', **overrides: Any) -> VerifyResult:
    return VerifyResult(
        passed=passed, test_output='', lint_output='', type_output='', summary=summary, **overrides,
    )


class TestPlanHash:
    def test_same_inputs_hash_equal_and_wire_stable(self):
        first = _alpha_plan().to_dict()
        second = _alpha_plan().to_dict()

        digest = verify_evidence.plan_hash(first)

        assert digest == verify_evidence.plan_hash(second)
        assert digest is not None
        assert len(digest) == 64
        assert digest == digest.lower()
        int(digest, 16)
        assert verify_evidence.plan_hash(json.loads(json.dumps(first))) == digest

    def test_file_scoped_and_full_suite_plans_hash_differently(self):
        file_scoped = _alpha_plan(role='task')
        full_suite = _alpha_plan(
            role='merge', config=OrchestratorConfig(merge_verify_breadth='full'),
        )
        assert _lint_run_scope(file_scoped) is ScopeKind.FILE_SCOPED
        assert _lint_run_scope(full_suite) is ScopeKind.FULL_SUITE

        assert verify_evidence.plan_hash(file_scoped.to_dict()) != verify_evidence.plan_hash(
            full_suite.to_dict(),
        )

    def test_prose_reason_does_not_change_the_hash(self):
        plan = _alpha_plan()
        reworded = dataclasses.replace(
            plan,
            runs=(dataclasses.replace(plan.runs[0], reason='reworded prose'), *plan.runs[1:]),
        )
        assert reworded.runs[0].reason != plan.runs[0].reason

        assert verify_evidence.plan_hash(reworded.to_dict()) == verify_evidence.plan_hash(
            plan.to_dict(),
        )

    def test_no_plan_has_no_hash(self):
        assert verify_evidence.plan_hash(None) is None

    @pytest.mark.parametrize('has_plan', [True, False])
    def test_verify_result_exposes_the_hash_of_its_plan(self, has_plan):
        plan = _alpha_plan().to_dict() if has_plan else None

        result = _result(plan=plan)

        assert result.plan_hash == verify_evidence.plan_hash(plan)
        assert (result.plan_hash is None) is not has_plan


def _module_unit(module: str, leg: str, outcome: UnitOutcome) -> VerifyUnit:
    return VerifyUnit(
        unit=f'module:{module}:{leg}', tier=UnitTier.MODULE, key=None,
        outcome=outcome, evidence_event_id=None,
    )


class TestUnitsOnResult:
    @staticmethod
    async def _green_test_red_lint(worktree: Path) -> VerifyResult:
        module = ModuleConfig(
            prefix='alpha', test_command='true', lint_command='false', type_check_command=None,
        )
        return await run_verification(worktree, OrchestratorConfig(), module, role='merge')

    @pytest.mark.asyncio
    async def test_each_leg_that_ran_is_a_unit_with_its_outcome(self, tmp_path):
        result = await self._green_test_red_lint(tmp_path)

        assert result.units == (
            _module_unit('alpha', 'test', UnitOutcome.RAN_GREEN),
            _module_unit('alpha', 'lint', UnitOutcome.RAN_RED),
        )
        assert result.cache_mode is CacheMode.OFF

    @pytest.mark.asyncio
    async def test_units_round_trip_the_wire_codec(self, tmp_path):
        result = await self._green_test_red_lint(tmp_path)

        rebuilt = result_from_dict(json.loads(json.dumps(result_to_dict(result))))

        assert rebuilt == result
        assert isinstance(rebuilt.units, tuple)
        assert all(isinstance(unit, VerifyUnit) for unit in rebuilt.units)
        assert rebuilt.cache_mode is CacheMode.OFF

    def test_an_older_remote_without_units_rebuilds_as_unrecorded(self):
        wire = result_to_dict(_result())
        del wire['units'], wire['cache_mode']

        rebuilt = result_from_dict(wire)

        assert rebuilt.units is None
        assert rebuilt.cache_mode is CacheMode.OFF

    def test_an_unknown_outcome_is_refused_by_name(self):
        wire = result_to_dict(_result())
        wire['units'] = [{
            'unit': 'module:alpha:test', 'tier': 'module', 'key': None,
            'outcome': 'bogus', 'evidence_event_id': None,
        }]

        with pytest.raises(ValueError, match='bogus'):
            result_from_dict(wire)

    @pytest.mark.asyncio
    async def test_a_trivial_pass_records_that_nothing_ran(self, tmp_path):
        result = await run_scoped_verification(
            tmp_path, OrchestratorConfig(), [], task_files=['docs/x.md'], role='task',
        )

        assert result.trivial
        assert result.units == ()


def _two_module_repo(root: Path) -> str:
    for module in ('alpha', 'beta'):
        (root / module).mkdir(parents=True)
        (root / module / 'core.py').write_text('x = 1\n')
    for argv in (
        ['git', 'init', '-q', '-b', 'main'],
        ['git', '-c', 'user.email=t@t', '-c', 'user.name=t', 'add', '-A'],
        ['git', '-c', 'user.email=t@t', '-c', 'user.name=t', 'commit', '-q', '-m', 'init'],
    ):
        subprocess.run(argv, cwd=root, check=True)
    return subprocess.run(
        ['git', 'rev-parse', 'HEAD'], cwd=root, check=True, capture_output=True, text=True,
    ).stdout.strip()


def _module(prefix: str, *, test: str = 'true') -> ModuleConfig:
    return ModuleConfig(prefix=prefix, test_command=test, lint_command='true', type_check_command='true')


def _unscoped_gate(*, broken: bool = False, failing: tuple[str, ...] = ()):
    async def _run_unscoped(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(
            broken=broken, failing_subprojects=list(failing), timed_out_subprojects=[],
        )
    return _run_unscoped


def _merge_pool(
    repo: Path, modules: list[ModuleConfig], store: EventStore, *, run_unscoped: Any,
) -> tuple[VerifyRunnerPool, OrchestratorConfig]:
    config = OrchestratorConfig(
        project_root=repo, merge_verify_workspace=True, merge_verify_breadth='full',
        verify_cold_preprovision_command='',
    )
    local = LocalRunner(
        merge_wt=repo, config=config, module_configs=modules, task_files=None,
        run_scoped=run_scoped_verification, run_unscoped=run_unscoped,
    )
    return VerifyRunnerPool([local], event_store=store, task_id='5651'), config


class TestMergeVerifyEventCarriesUnits:
    @pytest.mark.asyncio
    async def test_the_row_carries_every_unit_of_both_modules(self, tmp_path):
        repo = tmp_path / 'repo'
        merge_sha = _two_module_repo(repo)
        modules = [_module('alpha'), _module('beta', test='false')]
        store = EventStore(tmp_path / 'runs.db', run_id='r')
        pool, config = _merge_pool(repo, modules, store, run_unscoped=_unscoped_gate())

        result = await pool.dispatch(
            merge_sha, build_merge_verify_spec(config, modules, None),
            attempt=1, depth=2, speculative=True, chain_items=3, chain_build_ms=40,
        )

        (row,) = store.fetch_events_by_type(EventType.merge_verify)
        data = row['data']
        assert sorted(data['units'], key=lambda unit: unit['unit']) == [
            {
                'unit': f'module:{module}:{leg}', 'tier': 'module', 'key': None,
                'outcome': 'ran_red' if (module, leg) == ('beta', 'test') else 'ran_green',
                'evidence_event_id': None,
            }
            for module in ('alpha', 'beta') for leg in ('lint', 'test', 'type')
        ]
        assert data['plan_hash'] is not None
        assert data['plan_hash'] == result.plan_hash
        assert data['cache_mode'] == 'off'
        assert isinstance(data['duration_ms'], int)
        assert {key: data[key] for key in (
            'runner', 'merge_sha', 'passed', 'attempt', 'depth', 'speculative',
            'chain_items', 'chain_build_ms', 'retry_scope', 'retry_subset_sizes',
        )} == {
            'runner': 'local', 'merge_sha': merge_sha, 'passed': False, 'attempt': 1,
            'depth': 2, 'speculative': True, 'chain_items': 3, 'chain_build_ms': 40,
            'retry_scope': None, 'retry_subset_sizes': None,
        }

    @pytest.mark.asyncio
    async def test_a_broken_unscoped_gate_keeps_the_scoped_plan_and_units(self, tmp_path):
        repo = tmp_path / 'repo'
        merge_sha = _two_module_repo(repo)
        modules = [_module('alpha'), _module('beta')]
        store = EventStore(tmp_path / 'runs.db', run_id='r')
        broken_gate = _unscoped_gate(broken=True, failing=('alpha',))
        pool, config = _merge_pool(repo, modules, store, run_unscoped=broken_gate)

        result = await pool.dispatch(merge_sha, build_merge_verify_spec(config, modules, None))

        assert result.category == UNSCOPED_TYPECHECK_FAILED_CATEGORY
        assert result.plan_hash is not None
        assert result.units is not None
        assert {unit.unit for unit in result.units} == {
            f'module:{module}:{leg}' for module in ('alpha', 'beta') for leg in ('test', 'lint', 'type')
        }
        assert {unit.outcome for unit in result.units} == {UnitOutcome.RAN_GREEN}


class TestAggregatedUnits:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(('beta_units', 'expected'), [
        pytest.param(None, None, id='one-unrecorded-module-leaves-the-aggregate-unrecorded'),
        pytest.param((), (module_leg_unit('alpha', 'test', 0),), id='recorded-modules-concatenate'),
    ])
    async def test_a_module_verdict_without_units_fails_closed(
        self, tmp_path, monkeypatch, beta_units, expected,
    ):
        repo = tmp_path / 'repo'
        _two_module_repo(repo)
        by_module = {
            'alpha': _result(units=(module_leg_unit('alpha', 'test', 0),)),
            'beta': _result(units=beta_units),
        }

        async def _module_verdict(
            _worktree: Path, _config: OrchestratorConfig, module: ModuleConfig, **_kwargs: Any,
        ) -> VerifyResult:
            return by_module[module.prefix]

        monkeypatch.setattr('orchestrator.verify.run_verification', _module_verdict)

        result = await run_scoped_verification(
            repo, OrchestratorConfig(), [_module('alpha'), _module('beta')],
            task_files=['alpha/core.py', 'beta/core.py'], role='task',
        )

        assert result.units == expected


# ---------------------------------------------------------------------------
# A TaskWorkflow over a real git repo and a real EventStore, with its agent and
# scheduler collaborators doubled (the test_rebase_verify_cost.py harness).

_TASK_ID = '42'


def _git(cwd: Path, *argv: str) -> str:
    return subprocess.run(
        ['git', '-c', 'user.email=t@t', '-c', 'user.name=t', *argv],
        cwd=cwd, check=True, capture_output=True, text=True,
    ).stdout.strip()


def _commit(cwd: Path, name: str) -> None:
    (cwd / name).write_text(f'{name}\n')
    _git(cwd, 'add', '-A')
    _git(cwd, 'commit', '-q', '-m', name)


@pytest.fixture
def config(tmp_path: Path) -> OrchestratorConfig:
    repo = tmp_path / 'repo'
    repo.mkdir()
    _git(repo, 'init', '-q', '-b', 'main')
    _commit(repo, 'lib.py')
    return OrchestratorConfig(
        project_root=repo,
        max_concurrent_tasks=1,
        max_verify_attempts=2,
        escalate_preexisting_main_break=False,
        git=GitConfig(
            main_branch='main', branch_prefix='task/', remote='origin', worktree_dir='.worktrees',
        ),
    )


@pytest.fixture
def git_ops(config: OrchestratorConfig) -> GitOps:
    return GitOps(config.git, config.project_root)


async def _task_workflow(
    config: OrchestratorConfig, git_ops: GitOps, store: EventStore, monkeypatch: pytest.MonkeyPatch,
) -> TaskWorkflow:
    worktree = await git_ops.create_worktree(_TASK_ID)
    workflow = TaskWorkflow(
        assignment=TaskAssignment(
            task_id=_TASK_ID,
            task={
                'id': _TASK_ID, 'title': 'X', 'description': '', 'status': 'pending',
                'metadata': {'files': ['lib']}, 'dependencies': [],
            },
            modules=['lib'],
        ),
        config=config, git_ops=git_ops,
        scheduler=MagicMock(), briefing=MagicMock(), mcp=MagicMock(),
        event_store=store,
    )
    workflow.worktree = worktree.path
    workflow._base_commit = worktree.base_commit
    workflow.artifacts = TaskArtifacts(worktree.path)
    workflow.artifacts.init(_TASK_ID, 'X', 'desc', base_commit=worktree.base_commit)
    workflow.plan = {'task_id': _TASK_ID, 'steps': []}
    workflow.briefing.build_debugger_prompt = AsyncMock(return_value='debug')
    monkeypatch.setattr(workflow, '_check_escalations', MagicMock(return_value=[]))
    monkeypatch.setattr(workflow, '_invoke', AsyncMock(return_value=AgentResult(success=True, output='')))
    return workflow


def _admission_leg(*, waited_secs: float, flock_wait_secs: float) -> AdmissionLeg:
    return AdmissionLeg(
        plan_id='p', module='alpha', scope_kind='full_suite', order='arrival',
        arrived_at='2026-10-26T00:00:00+00:00', granted_at='2026-10-26T00:00:03+00:00',
        waited_secs=waited_secs, depth_at_arrival=1, slot_held=True,
        flock_wait_secs=flock_wait_secs, held_secs=9.0, rc=0, timed_out=False,
    )


def _workflow_verify_rows(store: EventStore) -> list[dict[str, Any]]:
    return [row['data'] for row in store.fetch_events_by_type(EventType.workflow_verify)]


class TestImplementSiteWorkflowVerify:
    @pytest.mark.asyncio
    async def test_the_review_edge_row_carries_the_passing_verdict(
        self, config, git_ops, tmp_path, monkeypatch,
    ):
        store = EventStore(tmp_path / 'events.db', run_id='r')
        workflow = await _task_workflow(config, git_ops, store, monkeypatch)
        plan = _alpha_plan().to_dict()
        failing = _result(
            passed=False, summary='tests failed', duration_secs=1.0,
            units=(module_leg_unit('alpha', 'test', 1),),
        )
        passing = _result(
            plan=plan, duration_secs=12.5,
            units=(module_leg_unit('alpha', 'test', 0), module_leg_unit('alpha', 'lint', 0)),
            admission_legs=(_admission_leg(waited_secs=3.0, flock_wait_secs=1.5),),
        )
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification', AsyncMock(side_effect=[failing, passing]),
        )
        monkeypatch.setattr(workflow, '_review', AsyncMock(return_value=ReviewAggregation(
            has_blocking_issues=False, blocking_issues=[], suggestions=[], reviews={'analyst': {}},
        )))

        assert await workflow._execute_verify_review_loop() == WorkflowOutcome.DONE

        assert workflow.worktree is not None
        assert _workflow_verify_rows(store) == [{
            'passed': True,
            'tip_sha': _git(workflow.worktree, 'rev-parse', 'HEAD'),
            'base_sha': _git(config.project_root, 'rev-parse', 'main'),
            'branch': f'task/{_TASK_ID}',
            'duration_ms': 12500,
            'slot_wait_secs': 4.5,
            'site': 'implement',
            'plan_hash': verify_evidence.plan_hash(plan),
            'cache_mode': 'off',
            'units': [unit.to_dict() for unit in passing.units or ()],
        }]


async def _forked_task_workflow(
    config: OrchestratorConfig, git_ops: GitOps, store: EventStore, monkeypatch: pytest.MonkeyPatch,
) -> TaskWorkflow:
    """A task branch with one commit of its own, forked one commit behind main's tip."""
    workflow = await _task_workflow(config, git_ops, store, monkeypatch)
    assert workflow.worktree is not None
    _commit(workflow.worktree, 'task_change.py')
    _commit(config.project_root, 'main_change.py')
    return workflow


class TestPreMergeSiteWorkflowVerify:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(('passed', 'timed_out'), [(True, False), (False, False), (False, True)])
    async def test_each_post_rebase_reverify_leaves_one_row(
        self, config, git_ops, tmp_path, monkeypatch, passed, timed_out,
    ):
        config.pre_merge_reverify = True
        store = EventStore(tmp_path / 'events.db', run_id='r')
        workflow = await _forked_task_workflow(config, git_ops, store, monkeypatch)
        plan = _alpha_plan().to_dict()
        verdict = _result(
            passed=passed, timed_out=timed_out, summary='' if passed else 'tests failed',
            plan=plan, duration_secs=7.0, units=(module_leg_unit('alpha', 'test', 0 if passed else 1),),
        )
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification', AsyncMock(return_value=verdict),
        )

        assert await workflow._rebase_before_enqueue() is passed

        assert workflow.worktree is not None
        tip = _git(workflow.worktree, 'rev-parse', 'HEAD')
        main_tip = _git(config.project_root, 'rev-parse', 'main')
        assert _git(config.project_root, 'merge-base', 'main', tip) == main_tip
        assert _workflow_verify_rows(store) == [{
            'passed': passed,
            'tip_sha': tip,
            'base_sha': main_tip,
            'branch': f'task/{_TASK_ID}',
            'duration_ms': 7000,
            'slot_wait_secs': None,
            'site': 'pre_merge',
            'plan_hash': verify_evidence.plan_hash(plan),
            'cache_mode': 'off',
            'units': [unit.to_dict() for unit in verdict.units or ()],
        }]
        wastes = [row['data'] for row in store.fetch_events_by_type(EventType.waste_detected)]
        expect_waste = not passed and not timed_out
        assert [w['waste_type'] for w in wastes] == (['post_rebase_verify_fail'] if expect_waste else [])

    @pytest.mark.asyncio
    async def test_without_the_rollback_switch_no_reverify_runs_and_no_row_is_left(
        self, config, git_ops, tmp_path, monkeypatch,
    ):
        assert config.pre_merge_reverify is False
        store = EventStore(tmp_path / 'events.db', run_id='r')
        workflow = await _forked_task_workflow(config, git_ops, store, monkeypatch)
        verify = AsyncMock(return_value=_result())
        monkeypatch.setattr('orchestrator.workflow.run_scoped_verification', verify)

        assert await workflow._rebase_before_enqueue() is True

        verify.assert_not_awaited()
        assert _workflow_verify_rows(store) == []
