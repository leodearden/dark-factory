"""Verify events carry the plan hash, per-unit outcomes and task-leg duration (task 5651)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Literal

import pytest

from orchestrator import verify_evidence
from orchestrator.config import ModuleConfig, OrchestratorConfig
from orchestrator.verify import VerifyResult, run_scoped_verification, run_verification
from orchestrator.verify_evidence import CacheMode, UnitOutcome, UnitTier, VerifyUnit
from orchestrator.verify_plan import ScopeKind, VerifyPlan, derive_verify_plan
from orchestrator.verify_runner import result_from_dict, result_to_dict

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


def _result(*, passed: bool = True, **overrides: Any) -> VerifyResult:
    return VerifyResult(
        passed=passed, test_output='', lint_output='', type_output='', summary='', **overrides,
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
