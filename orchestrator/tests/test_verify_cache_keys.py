"""Verify-cache unit keys and the plan/evidence seams they build on (task 5652)."""

from __future__ import annotations

import hashlib

from orchestrator import verify_evidence, verify_plan
from orchestrator.verify import VerifyResult
from orchestrator.verify_cache import modes
from orchestrator.verify_cmd import parse_config_command
from orchestrator.verify_evidence import Leg
from orchestrator.verify_plan import PlannedRun, ScopeKind
from orchestrator.verify_spec_shared import canonical_json

_TEST_CMD = parse_config_command('uv run --project alpha pytest alpha/tests -q')
_LINT_CMD = parse_config_command('uv run --project alpha ruff check alpha')
_TYPE_CMD = parse_config_command('uv run --project alpha pyright alpha')


def _full_suite_test_run(reason: str = 'pytest: conftest touched') -> PlannedRun:
    return PlannedRun('alpha', _TEST_CMD, ScopeKind.FULL_SUITE, reason)


class TestRunSpecHash:
    def test_identical_runs_hash_equal(self):
        assert verify_evidence.run_spec_hash(_full_suite_test_run().to_dict()) == (
            verify_evidence.run_spec_hash(_full_suite_test_run().to_dict())
        )

    def test_prose_reason_is_not_identity(self):
        first = _full_suite_test_run('pytest: conftest touched').to_dict()
        second = _full_suite_test_run('pytest: structural change').to_dict()

        assert verify_evidence.run_spec_hash(first) == verify_evidence.run_spec_hash(second)

    def test_scope_kind_is_identity(self):
        full = _full_suite_test_run().to_dict()
        scoped = PlannedRun(
            'alpha', _TEST_CMD, ScopeKind.FILE_SCOPED, 'pytest: scoped',
            ('alpha/tests/test_one.py',),
        ).to_dict()

        assert verify_evidence.run_spec_hash(full) != verify_evidence.run_spec_hash(scoped)

    def test_plan_hash_keeps_its_alpha_definition(self):
        run = _full_suite_test_run().to_dict()
        execution_spec = {key: value for key, value in run.items() if key != 'reason'}

        assert verify_evidence.plan_hash({'runs': [run]}) == (
            hashlib.sha256(canonical_json([execution_spec]).encode()).hexdigest()
        )


class TestPlannedRunLeg:
    def test_each_reason_prefix_names_its_leg(self):
        lint = PlannedRun('alpha', _LINT_CMD, ScopeKind.FULL_SUITE, 'lint: structural')
        type_check = PlannedRun('alpha', _TYPE_CMD, ScopeKind.FULL_SUITE, 'pyright: structural')
        test = _full_suite_test_run()

        assert verify_plan.planned_run_leg(lint) is Leg.LINT
        assert verify_plan.planned_run_leg(type_check) is Leg.TYPE
        assert verify_plan.planned_run_leg(test) is Leg.TEST

    def test_a_reason_without_a_tool_prefix_has_no_leg(self):
        skipped = PlannedRun('__fallback__', None, ScopeKind.SKIPPED, 'no .py files touched')

        assert verify_plan.planned_run_leg(skipped) is None

    def test_leg_values_are_the_check_run_labels(self):
        assert {str(leg) for leg in Leg} == {'test', 'lint', 'type'}
        assert verify_evidence.module_leg_name('alpha', Leg.TYPE) == 'module:alpha:type'

    def test_module_of_a_registered_module_run_is_its_prefix(self):
        assert verify_plan.module_of(_full_suite_test_run()) == 'alpha'

    def test_module_of_the_sentinel_prefixes_is_none(self):
        fallback = PlannedRun('__fallback__', _TEST_CMD, ScopeKind.FULL_SUITE, 'pytest: conftest')
        trivial = PlannedRun('', None, ScopeKind.TRIVIAL, 'No source files changed')

        assert verify_plan.module_of(fallback) is None
        assert verify_plan.module_of(trivial) is None

    def test_verify_result_cache_mode_is_the_modes_enum(self):
        result = VerifyResult(passed=True, test_output='', lint_output='', type_output='', summary='')

        assert result.cache_mode is modes.CacheMode.OFF
