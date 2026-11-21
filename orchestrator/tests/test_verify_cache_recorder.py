"""What a verify result leaves for the verify-result cache, and writing it (task 5653)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from orchestrator.verify_cache.recorder import CacheRecord, LegRun

from orchestrator.config import ModuleConfig, OrchestratorConfig, VerifyCacheConfig
from orchestrator.verify import (
    VerifyResult,
    pytest_reported_when_complete,
    run_scoped_verification,
    run_verification,
)
from orchestrator.verify_cache.modes import CacheMode
from orchestrator.verify_cache.policy import LegResult
from orchestrator.verify_evidence import Leg
from orchestrator.verify_runner import result_from_dict, result_to_dict

_PROGRESS_COLUMN = 73


def _progress(statuses: str, percent: int) -> str:
    return f'{statuses.ljust(_PROGRESS_COLUMN)}[{percent:3d}%]'


def _leg(leg: Leg = Leg.TEST, *, rc: int = 0) -> LegResult:
    return LegResult(
        leg=leg, rc=rc, timed_out=False, trivial=False, flake_suppressed=False,
        junit_tests=None, collected=None, never_cache=False,
    )


def _record(*legs: LegRun) -> CacheRecord:
    return CacheRecord(role='merge', modes=VerifyCacheConfig(merge_mode=CacheMode.LOG), legs=legs)


def _result(**overrides: Any) -> VerifyResult:
    return VerifyResult(passed=True, test_output='', lint_output='', type_output='', summary='', **overrides)


class TestPytestReportedWhenComplete:
    def test_one_quiet_session_counts_its_tally(self):
        output = '\n'.join([
            _progress('...............', 100),
            '12 passed, 2 skipped, 1 xfailed in 3.1s',
        ])

        assert pytest_reported_when_complete(output) == 15

    def test_segmented_sessions_sum(self):
        output = '\n'.join([
            '===== segment 1/2 [orchestrator] cwd=/w/orchestrator rc=0 status=passed (3.1s) =====',
            'bringing up nodes...',
            _progress('........', 100),
            '8 passed in 1.20s',
            '===== segment 2/2 [shared] cwd=/w/shared rc=0 status=passed (2.0s) =====',
            'bringing up nodes...',
            _progress('...s', 100),
            '3 passed, 1 skipped in 0.90s',
        ])

        assert pytest_reported_when_complete(output) == 12

    def test_an_and_chain_of_quiet_sessions_sums(self):
        output = '\n'.join([
            'bringing up nodes...',
            _progress('...', 100),
            '3 passed in 0.5s',
            'bringing up nodes...',
            _progress('.....', 100),
            '5 passed, 16 warnings in 366.84s (0:06:06)',
        ])

        assert pytest_reported_when_complete(output) == 8

    def test_a_final_progress_line_below_full_is_no_count(self):
        output = '\n'.join([
            _progress('..........', 45),
            '10 passed in 2.0s',
        ])

        assert pytest_reported_when_complete(output) is None

    def test_a_truncated_session_after_a_complete_one_is_no_count(self):
        output = '\n'.join([
            _progress('...', 100),
            '3 passed in 0.5s',
            _progress('..', 40),
        ])

        assert pytest_reported_when_complete(output) is None

    def test_a_tally_without_a_progress_line_is_no_count(self):
        assert pytest_reported_when_complete('12 passed in 3.1s') is None

    def test_output_without_a_pytest_session_is_no_count(self):
        assert pytest_reported_when_complete('All checks passed\n') is None

    def test_deselected_and_warnings_are_not_counted(self):
        output = '\n'.join([
            _progress('...', 100),
            '3 passed, 4 deselected, 2 warnings in 0.3s',
        ])

        assert pytest_reported_when_complete(output) == 3

    def test_every_outcome_of_a_collected_test_is_counted(self):
        output = '\n'.join([
            _progress('F.EEsxX', 100),
            '1 failed, 1 passed, 1 skipped, 1 xfailed, 1 xpassed, 2 errors in 0.5s',
        ])

        assert pytest_reported_when_complete(output) == 7


class TestLegFacts:
    @pytest.mark.asyncio
    async def test_each_check_that_ran_leaves_one_leg(self, tmp_path):
        module = ModuleConfig(
            prefix='alpha', test_command='true', lint_command='false', type_check_command=None,
            verify_cache='never',
        )
        config = OrchestratorConfig()

        result = await run_verification(tmp_path, config, module, role='merge')

        record = result.cache_record
        assert record is not None
        assert (record.role, record.modes) == ('merge', config.verify_cache)
        assert record.legs == (
            LegRun('module:alpha:test', LegResult(
                leg=Leg.TEST, rc=0, timed_out=False, trivial=False, flake_suppressed=False,
                junit_tests=None, collected=None, never_cache=True,
            )),
            LegRun('module:alpha:lint', LegResult(
                leg=Leg.LINT, rc=1, timed_out=False, trivial=False, flake_suppressed=False,
                junit_tests=None, collected=None, never_cache=True,
            )),
        )

    @pytest.mark.asyncio
    async def test_a_full_merge_pytest_leg_counts_its_junit_and_its_collection(self, tmp_path):
        suite = tmp_path / 'suite'
        suite.mkdir()
        (suite / 'test_two.py').write_text('def test_a():\n    pass\n\n\ndef test_b():\n    pass\n')
        module = ModuleConfig(
            prefix='alpha', test_command=f'{sys.executable} -m pytest {suite} -q',
            lint_command=None, type_check_command=None,
        )
        config = OrchestratorConfig(merge_verify_breadth='full')

        result = await run_verification(tmp_path, config, module, role='merge')

        assert result.passed, result.test_output
        assert result.cache_record is not None
        (test_leg,) = result.cache_record.legs
        assert (test_leg.leg.junit_tests, test_leg.leg.collected) == (2, 2)

    @pytest.mark.asyncio
    @pytest.mark.parametrize('beta_recorded', [True, False])
    async def test_aggregation_concatenates_legs_and_fails_closed(
        self, tmp_path, monkeypatch, beta_recorded,
    ):
        alpha_leg = LegRun('module:alpha:test', _leg())
        beta_leg = LegRun('module:beta:test', _leg(rc=1))
        by_module = {
            'alpha': _result(cache_record=_record(alpha_leg)),
            'beta': _result(cache_record=_record(beta_leg) if beta_recorded else None),
        }

        async def _module_verdict(
            _worktree: Path, _config: OrchestratorConfig, module: ModuleConfig, **_kwargs: Any,
        ) -> VerifyResult:
            return by_module[module.prefix]

        monkeypatch.setattr('orchestrator.verify.run_verification', _module_verdict)
        repo = _two_module_repo(tmp_path / 'repo')

        result = await run_scoped_verification(
            repo, OrchestratorConfig(), [_module('alpha'), _module('beta')],
            task_files=['alpha/core.py', 'beta/core.py'], role='task',
        )

        if beta_recorded:
            assert result.cache_record is not None
            assert result.cache_record.legs == (alpha_leg, beta_leg)
        else:
            assert result.cache_record is None

    def test_the_record_never_crosses_the_wire(self):
        result = _result(cache_record=_record(LegRun('module:alpha:test', _leg())))

        wire = result_to_dict(result)
        rebuilt = result_from_dict(json.loads(json.dumps(wire)))

        assert 'cache_record' not in wire
        assert rebuilt.cache_record is None
        assert rebuilt == result


def _module(prefix: str) -> ModuleConfig:
    return ModuleConfig(prefix=prefix, test_command='true', lint_command='true', type_check_command='true')


def _two_module_repo(root: Path) -> Path:
    for module in ('alpha', 'beta'):
        (root / module).mkdir(parents=True)
        (root / module / 'core.py').write_text('x = 1\n')
    for argv in (
        ['git', 'init', '-q', '-b', 'main'],
        ['git', '-c', 'user.email=t@t', '-c', 'user.name=t', 'add', '-A'],
        ['git', '-c', 'user.email=t@t', '-c', 'user.name=t', 'commit', '-q', '-m', 'init'],
    ):
        subprocess.run(argv, cwd=root, check=True)
    return root
