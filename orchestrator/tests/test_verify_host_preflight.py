"""verify_host_preflight: the full merge gate on one named verify host, through the production remote path (task 11731)."""
from __future__ import annotations

import dataclasses
import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from _gate_fixtures import TEST_DEPLOYED_SHA
from _git_fixtures import seed_repo
from _verify_artifact_fixtures import registry_config
from _verify_host_drive import runner_double
from _verify_result_doubles import real_verify_result

from orchestrator import verify_host_preflight as preflight
from orchestrator.config import GitConfig, ModuleConfig, OrchestratorConfig, VerifyRunnerConfig
from orchestrator.verify import VerifyResult
from orchestrator.verify_runner import RunnerUnavailable

ENSURE_SECS = 7.0
VERIFY_SECS = 100.0


class FakeClock:
    """A monotonic clock that moves only when a runner double's call advances it."""

    def __init__(self) -> None:
        self.readings = [0.0]

    def __call__(self) -> float:
        return self.readings[-1]

    def advance(self, secs: float) -> None:
        self.readings.append(self.readings[-1] + secs)


def declared_runner(name: str = 'box', **fields: Any) -> VerifyRunnerConfig:
    """A declared, DISABLED verify runner with gate paths."""
    values: dict[str, Any] = {
        'name': name, 'ssh_host': name, 'git_remote': name,
        'df_checkout_path': '/remote/dark-factory', 'enabled': False,
    }
    values.update(fields)
    return VerifyRunnerConfig(**values)


def preflight_config(
    repo: Path,
    *,
    runners: list[VerifyRunnerConfig] | None = None,
    persistent_merge_worktree: bool = False,
) -> OrchestratorConfig:
    return registry_config(
        [ModuleConfig(prefix='alpha', test_command='pytest'),
         ModuleConfig(prefix='beta', test_command='pytest')],
        project_root=repo,
        git=GitConfig(main_branch='main', persistent_merge_worktree=persistent_merge_worktree),
        merge_verify_breadth='scoped',
        verify_env_probes=[],
        verify_runners=runners if runners is not None else [declared_runner()],
    )


def green_result(**fields: Any) -> VerifyResult:
    values: dict[str, Any] = {
        'failing_test_ids_by_module': {'alpha': [], 'beta': []},
        'skipped_test_ids': ['tests.test_s::test_skipped'],
    }
    values.update(fields)
    return dataclasses.replace(real_verify_result(True), **values)


def _advancing(clock: FakeClock, secs: float, outcome: object) -> Callable[..., object]:
    def effect(*_args: object, **_kwargs: object) -> object:
        clock.advance(secs)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome
    return effect


def timed_double(clock: FakeClock, name: str = 'box', **kwargs: Any) -> MagicMock:
    """runner_double whose ensure_gate and verify advance *clock* by ENSURE_SECS and VERIFY_SECS."""
    kwargs.setdefault('result', green_result())
    double = runner_double(name, is_local=False, **kwargs)
    for attr, secs in (('ensure_gate', ENSURE_SECS), ('run_merge_verify', VERIFY_SECS)):
        mock = getattr(double, attr)
        outcome = mock.side_effect if mock.side_effect is not None else mock.return_value
        mock.side_effect = _advancing(clock, secs, outcome)
    return double


def rev_parse(repo: Path, rev: str) -> str:
    return subprocess.run(
        ['git', '-C', str(repo), 'rev-parse', rev], check=True, capture_output=True, text=True,
    ).stdout.strip()


def preflight_dir(repo: Path) -> Path:
    return repo / 'data' / 'verify-logs' / 'preflight'


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return seed_repo(tmp_path / 'repo')


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


async def run_box(
    config: OrchestratorConfig,
    double: MagicMock,
    clock: FakeClock,
    *,
    sha: str | None = None,
) -> preflight.PreflightOutcome:
    return await preflight.run_preflight(
        config, runner='box', sha=sha, reference=None, deployed_sha=TEST_DEPLOYED_SHA,
        build_runner=lambda _rc: double, clock=clock,
    )


@pytest.mark.asyncio
class TestDispatchAndRecord:
    async def test_dispatches_the_full_gate_for_the_main_tip_after_ensuring_the_gate(
        self, repo: Path, clock: FakeClock,
    ) -> None:
        double = timed_double(clock)
        order = MagicMock()
        order.attach_mock(double.ensure_gate, 'ensure_gate')
        order.attach_mock(double.run_merge_verify, 'run_merge_verify')

        await run_box(preflight_config(repo), double, clock)

        assert [c[0] for c in order.mock_calls] == ['ensure_gate', 'run_merge_verify']
        double.run_merge_verify.assert_awaited_once()
        (merge_sha, spec), kwargs = double.run_merge_verify.await_args
        assert merge_sha == rev_parse(repo, 'main')
        assert spec.merge_verify_breadth == 'full'
        assert spec.task_files is None
        assert spec.verify_cache is None
        assert tuple(vc.prefix for vc in spec.verify_commands) == ('alpha', 'beta')
        assert kwargs['archive_root'] == repo / 'data' / 'verify-logs'
        assert kwargs['task_id'] == preflight.PREFLIGHT_ARCHIVE_KEY

    async def test_resolves_a_revision_to_the_full_sha_it_dispatches_and_records(
        self, repo: Path, clock: FakeClock,
    ) -> None:
        double = timed_double(clock)

        outcome = await run_box(preflight_config(repo), double, clock, sha='HEAD')

        full_sha = rev_parse(repo, 'HEAD')
        assert len(full_sha) == 40
        assert double.run_merge_verify.await_args.args[0] == full_sha
        assert outcome.primary.sha == full_sha
        assert outcome.record_paths == (preflight_dir(repo) / f'box-{full_sha}.json',)

    async def test_a_green_verdict_writes_its_record(self, repo: Path, clock: FakeClock) -> None:
        double = timed_double(clock)

        outcome = await run_box(preflight_config(repo), double, clock)

        sha = rev_parse(repo, 'main')
        path = preflight_dir(repo) / f'box-{sha}.json'
        record = json.loads(path.read_text())
        assert record['runner'] == 'box'
        assert record['sha'] == sha
        assert record['passed'] is True
        assert record['ensure_gate_secs'] == ENSURE_SECS
        assert record['wall_secs'] == ENSURE_SECS + VERIFY_SECS
        assert record['gate']['sha'] == TEST_DEPLOYED_SHA.sha
        assert record['modules'] == {
            'alpha': {'failing_test_ids': [], 'failing_count': 0},
            'beta': {'failing_test_ids': [], 'failing_count': 0},
        }
        assert record['skipped_test_ids'] == ['tests.test_s::test_skipped']
        assert outcome.ok is True
        assert outcome.record_paths == (path,)

    async def test_accepts_a_declared_runner_that_is_not_enabled(
        self, repo: Path, clock: FakeClock,
    ) -> None:
        config = preflight_config(repo)
        assert config.enabled_verify_runners == []
        built: list[VerifyRunnerConfig] = []
        double = timed_double(clock)

        def build(rc: VerifyRunnerConfig) -> MagicMock:
            built.append(rc)
            return double

        outcome = await preflight.run_preflight(
            config, runner='box', sha=None, reference=None, deployed_sha=TEST_DEPLOYED_SHA,
            build_runner=build, clock=clock,
        )

        assert [rc.name for rc in built] == ['box']
        assert outcome.primary.passed is True


def record_of(repo: Path, outcome: preflight.PreflightOutcome) -> dict[str, Any]:
    return json.loads(record_path_for(repo, outcome.primary.sha).read_text())


def record_path_for(repo: Path, sha: str, runner: str = 'box') -> Path:
    return preflight_dir(repo) / f'{runner}-{sha}.json'


@pytest.mark.asyncio
class TestVerdictMapping:
    async def test_a_red_verdict_records_its_failing_ids_per_module(
        self, repo: Path, clock: FakeClock,
    ) -> None:
        red = dataclasses.replace(
            real_verify_result(False),
            failing_test_ids_by_module={'alpha': ['tests.test_a::test_x']},
            failing_test_ids=['tests.test_a::test_x'],
        )

        outcome = await run_box(preflight_config(repo), timed_double(clock, result=red), clock)

        record = record_of(repo, outcome)
        assert record['passed'] is False
        assert record['modules']['alpha'] == {
            'failing_test_ids': ['tests.test_a::test_x'], 'failing_count': 1,
        }
        assert record['modules']['beta'] == {'failing_test_ids': [], 'failing_count': 0}
        assert outcome.ok is False

    async def test_a_trivial_pass_is_not_recorded_as_passed(
        self, repo: Path, clock: FakeClock,
    ) -> None:
        vacuous = green_result(trivial=True)

        outcome = await run_box(preflight_config(repo), timed_double(clock, result=vacuous), clock)

        record = record_of(repo, outcome)
        assert record['passed'] is False
        assert record['trivial'] is True
        assert outcome.ok is False

    async def test_an_unavailable_verify_still_writes_a_no_verdict_record(
        self, repo: Path, clock: FakeClock,
    ) -> None:
        double = timed_double(clock, error=RunnerUnavailable('host down', kind='transport'))

        outcome = await run_box(preflight_config(repo), double, clock)

        record = record_of(repo, outcome)
        assert record['passed'] is False
        assert record['no_verdict'] == {'kind': 'transport', 'message': 'host down'}
        assert outcome.ok is False

    async def test_an_unavailable_gate_records_its_kind_and_never_dispatches_the_verify(
        self, repo: Path, clock: FakeClock,
    ) -> None:
        double = timed_double(
            clock, ensure_error=RunnerUnavailable('skewed', kind='wire_shape'),
        )

        outcome = await run_box(preflight_config(repo), double, clock)

        record = record_of(repo, outcome)
        assert record['no_verdict']['kind'] == 'wire_shape'
        assert record['passed'] is False
        double.run_merge_verify.assert_not_awaited()


@pytest.mark.asyncio
class TestRefusals:
    @staticmethod
    async def assert_refused(
        config: OrchestratorConfig,
        *,
        runner: str = 'box',
        sha: str | None = None,
        reference: str | None = None,
    ) -> None:
        double = runner_double('box', is_local=False)
        built: list[VerifyRunnerConfig] = []

        def build(rc: VerifyRunnerConfig) -> MagicMock:
            built.append(rc)
            return double

        with pytest.raises(preflight.PreflightRefused):
            await preflight.run_preflight(
                config, runner=runner, sha=sha, reference=reference,
                deployed_sha=TEST_DEPLOYED_SHA, build_runner=build,
            )

        assert built == []
        double.ensure_gate.assert_not_awaited()
        double.run_merge_verify.assert_not_awaited()
        assert not preflight_dir(config.project_root).exists()

    async def test_refuses_an_undeclared_runner(self, repo: Path) -> None:
        await self.assert_refused(preflight_config(repo), runner='nope')

    async def test_refuses_a_project_with_a_persistent_merge_worktree(self, repo: Path) -> None:
        await self.assert_refused(preflight_config(repo, persistent_merge_worktree=True))

    async def test_refuses_a_revision_that_names_no_commit(self, repo: Path) -> None:
        await self.assert_refused(preflight_config(repo), sha='no-such-rev')

    async def test_refuses_a_reference_equal_to_the_runner(self, repo: Path) -> None:
        await self.assert_refused(preflight_config(repo), reference='box')

    async def test_refuses_a_runner_without_gate_paths(self, repo: Path) -> None:
        config = preflight_config(repo, runners=[declared_runner(df_checkout_path=None)])

        await self.assert_refused(config)

    @pytest.mark.parametrize('name', ['a/b', '..', '.'])
    async def test_refuses_a_runner_name_that_is_not_one_path_segment(
        self, repo: Path, name: str,
    ) -> None:
        config = preflight_config(repo, runners=[declared_runner(name)])

        await self.assert_refused(config, runner=name)
