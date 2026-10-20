"""TaskVerifyDispatcher: one routing point for a task-role verify, local or remote.

The value layer (policy, eligibility, the decision record) is pure.  The
dispatcher is driven over a real committed repo, a real HostAllocator/HostPool
and a real RemoteRunner whose public transport seams (``run=``, ``ssh_run=``,
``id_factory=``) are scripted by :class:`_Transport`.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import subprocess
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from _git_fixtures import RepoSeed, seed_repo

from orchestrator import task_verify_dispatch
from orchestrator.config import GitConfig, ModuleConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps
from orchestrator.task_verify_dispatch import (
    CrossCheck,
    DispatchDecision,
    RedConfirmation,
    TaskVerifyPolicy,
    eligible,
)
from orchestrator.verify import VerifyResult
from orchestrator.verify_hosts import HostPool
from orchestrator.verify_runner import HostAllocator, RemoteRunner

_SIDECAR_KEYS = {
    'decision', 'runner', 'reason', 'tip_sha', 'base_sha',
    'remote_wall_secs', 'lease_held_secs', 'cross_check', 'red_confirmed_locally',
}


def _decision(reason, **overrides) -> DispatchDecision:
    fields = {
        'runner': 'local',
        'reason': reason,
        'tip_sha': 'a' * 40,
        'base_sha': 'b' * 40,
        'remote_wall_secs': None,
        'lease_held_secs': None,
        'cross_check': None,
        'red_confirmed_locally': None,
    }
    fields.update(overrides)
    return DispatchDecision(**fields)


class TestTaskVerifyPolicy:
    def test_defaults_route_everything_locally(self) -> None:
        policy = TaskVerifyPolicy()

        assert policy.host_policy == 'local'
        assert policy.remote_modules == ()
        assert policy.cross_check_every_n == 0
        assert policy.red_confirms_locally is False

    def test_is_frozen(self) -> None:
        pytest.raises(
            dataclasses.FrozenInstanceError, setattr, TaskVerifyPolicy(), 'host_policy', 'remote_all',
        )

    def test_remote_modules_policy_needs_modules(self) -> None:
        with pytest.raises(ValueError, match='host_policy') as excinfo:
            TaskVerifyPolicy(host_policy='remote_modules', remote_modules=())

        assert 'remote_modules' in str(excinfo.value)

    def test_negative_cross_check_cadence_is_refused(self) -> None:
        with pytest.raises(ValueError, match='cross_check_every_n'):
            TaskVerifyPolicy(cross_check_every_n=-1)

    @pytest.mark.parametrize('host_policy', ['remote', 'LOCAL', ''])
    def test_unknown_host_policy_is_refused(self, host_policy) -> None:
        with pytest.raises(ValueError, match='host_policy'):
            TaskVerifyPolicy(host_policy=host_policy)

    @pytest.mark.parametrize('remote_modules', [['orchestrator'], 'orchestrator'])
    def test_remote_modules_must_be_a_tuple(self, remote_modules) -> None:
        with pytest.raises(ValueError, match='remote_modules'):
            TaskVerifyPolicy(host_policy='remote_modules', remote_modules=remote_modules)


class TestEligible:
    _POLICIES = [
        TaskVerifyPolicy(),
        TaskVerifyPolicy(host_policy='remote_all'),
        TaskVerifyPolicy(host_policy='remote_modules', remote_modules=('orchestrator',)),
    ]

    @pytest.mark.parametrize('executed', [set(), {'orchestrator'}, {'fused_memory'}])
    def test_local_policy_is_never_eligible(self, executed: set[str]) -> None:
        assert eligible(
            TaskVerifyPolicy(), executed, force_workspace=False, per_module_overrides_present=False,
        ) is False

    @pytest.mark.parametrize('executed', [set(), {'orchestrator'}, {'fused_memory', 'shared'}])
    def test_remote_all_is_eligible_for_any_executed_set(self, executed: set[str]) -> None:
        assert eligible(
            TaskVerifyPolicy(host_policy='remote_all'), executed,
            force_workspace=False, per_module_overrides_present=False,
        ) is True

    @pytest.mark.parametrize(('executed', 'expected'), [
        ({'orchestrator'}, True),
        ({'orchestrator', 'fused_memory'}, True),
        ({'fused_memory'}, False),
        (set(), False),
    ])
    def test_remote_modules_needs_a_listed_module_executed(
        self, executed: set[str], expected: bool,
    ) -> None:
        policy = TaskVerifyPolicy(host_policy='remote_modules', remote_modules=('orchestrator',))

        assert eligible(
            policy, executed, force_workspace=False, per_module_overrides_present=False,
        ) is expected

    @pytest.mark.parametrize('policy', _POLICIES)
    def test_force_workspace_is_never_eligible(self, policy: TaskVerifyPolicy) -> None:
        assert eligible(
            policy, {'orchestrator'}, force_workspace=True, per_module_overrides_present=False,
        ) is False

    @pytest.mark.parametrize('policy', _POLICIES)
    def test_per_module_overrides_are_never_eligible(self, policy: TaskVerifyPolicy) -> None:
        assert eligible(
            policy, {'orchestrator'}, force_workspace=False, per_module_overrides_present=True,
        ) is False


class TestDispatchDecision:
    @pytest.mark.parametrize('reason', ['local', 'REMOTE', 'busy'])
    def test_reason_outside_the_vocabulary_is_refused(self, reason) -> None:
        with pytest.raises(ValueError, match=f"'{reason}'"):
            _decision(reason)

    @pytest.mark.parametrize(('reason', 'expected'), [
        ('remote', 'remote'),
        ('not_eligible', 'local'),
        ('merge_demand', 'local'),
        ('cross_check', 'local'),
        ('red_confirm', 'local'),
        ('dispatcher_error', 'local'),
    ])
    def test_decision_is_derived_from_the_reason(self, reason, expected: str) -> None:
        assert _decision(reason).decision == expected

    def test_to_dict_has_exactly_the_sidecar_keys(self) -> None:
        rendered = _decision('remote', runner='laptop', remote_wall_secs=1.5).to_dict()

        assert set(rendered) == _SIDECAR_KEYS
        assert rendered['decision'] == 'remote'
        assert rendered['runner'] == 'laptop'
        assert rendered['reason'] == 'remote'
        assert rendered['tip_sha'] == 'a' * 40
        assert rendered['base_sha'] == 'b' * 40
        assert rendered['remote_wall_secs'] == 1.5
        assert rendered['cross_check'] is None
        assert rendered['red_confirmed_locally'] is None

    @pytest.mark.parametrize(('remote_passed', 'local_passed', 'agree'), [
        (True, True, True),
        (False, True, False),
        (True, False, False),
        (False, False, True),
    ])
    def test_cross_check_renders_its_agreement(
        self, remote_passed: bool, local_passed: bool, agree: bool,
    ) -> None:
        cross = CrossCheck(remote_passed=remote_passed, local_passed=local_passed)

        assert cross.agree is agree
        assert _decision('cross_check', cross_check=cross).to_dict()['cross_check'] == {
            'remote_passed': remote_passed, 'local_passed': local_passed, 'agree': agree,
        }

    @pytest.mark.parametrize('local_passed', [True, False])
    def test_red_confirmation_renders_the_remote_red(self, local_passed: bool) -> None:
        confirmation = RedConfirmation(local_passed=local_passed)

        rendered = _decision('red_confirm', red_confirmed_locally=confirmation).to_dict()

        assert rendered['red_confirmed_locally'] == {
            'remote_passed': False, 'local_passed': local_passed,
        }



# ---------------------------------------------------------------------------
# Dispatcher harness
# ---------------------------------------------------------------------------

_A_PY = 'orchestrator/src/orchestrator/a.py'
_B_PY = 'fused_memory/src/fused_memory/b.py'
_SEED = RepoSeed(files=((_A_PY, 'A = 1\n'), (_B_PY, 'B = 1\n')), message='initial commit')
_DISPATCH_LOGGER = 'orchestrator.task_verify_dispatch'

Answer = tuple[int, str, str]
VerifyAnswer = Callable[[list[str]], Awaitable[Answer]]


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ['git', '-C', str(repo), *args], check=True, capture_output=True, text=True,
    ).stdout.strip()


def _commit_on_task_branch(repo: Path, path: str, text: str) -> str:
    (repo / path).write_text(text)
    _git(repo, 'commit', '-q', '-am', f'touch {path}')
    return _git(repo, 'rev-parse', 'HEAD')


def _result(passed: bool, *, category: str = '', summary: str = '') -> VerifyResult:
    return VerifyResult(
        passed=passed, test_output='' if passed else 'FAILED test_x', lint_output='',
        type_output='', summary=summary or ('verify green' if passed else 'tests failed'),
        category=category,
    )


async def _never_called(argv: list[str]) -> Answer:
    raise AssertionError(f'unexpected remote verify: {argv}')


@dataclass
class _Transport:
    """Scripted answers for every argv a RemoteRunner issues, logged in order."""

    verify: VerifyAnswer = _never_called
    cancel_rc: int = 0
    pgrep_rc: int = 1
    local_df_head: str = 'a' * 40
    remote_df_head: str = 'a' * 40
    upstream_rc: int = 1
    pull_rc: int = 0
    log: list[str] = field(default_factory=list)
    argvs: list[list[str]] = field(default_factory=list)

    async def run(self, argv: list[str], *, cwd: Any = None) -> Answer:
        label, answer = self.answer(argv)
        self.log.append(label)
        self.argvs.append(list(argv))
        return answer

    async def ssh_run(self, argv: list[str], *, cwd: Any = None) -> Answer:
        self.log.append('verify_task')
        self.argvs.append(list(argv))
        return await self.verify(argv)

    def answer(self, argv: list[str]) -> tuple[str, Answer]:
        if argv[:2] == ['git', 'push']:
            return ('delete_ref' if '--delete' in argv else 'push'), (0, '', '')
        if argv == ['git', 'rev-parse', 'HEAD']:
            return 'local_df_head', (0, self.local_df_head, '')
        if argv == ['git', 'rev-parse', '@{upstream}']:
            return 'upstream', (self.upstream_rc, '', '')
        if argv[0] == 'ssh':
            return self.ssh_answer(argv[-1])
        return 'unscripted', (127, '', 'unscripted')

    def ssh_answer(self, remote_cmd: str) -> tuple[str, Answer]:
        if 'cancel-verify' in remote_cmd:
            return 'cancel_verify', (self.cancel_rc, '', '')
        if remote_cmd.startswith('pgrep'):
            return 'pgrep', (self.pgrep_rc, '', '')
        if 'pull --ff-only' in remote_cmd:
            return 'pull', (self.pull_rc, '', 'pull refused')
        if 'rev-parse HEAD' in remote_cmd:
            return 'remote_df_head', (0, self.remote_df_head, '')
        return 'unscripted', (127, '', 'unscripted')


class _Hooks:
    def __init__(self, *, demand: bool = False) -> None:
        self.demand = demand
        self.demand_calls = 0
        self.unavailable: list[tuple[str, str]] = []
        self.recovered: list[str] = []

    def merge_demand(self) -> bool:
        self.demand_calls += 1
        return self.demand

    def on_unavailable(self, name: str, reason: str) -> None:
        self.unavailable.append((name, reason))

    def on_recovered(self, name: str) -> None:
        self.recovered.append(name)


class _LocalRun:
    """A recording stand-in for run_scoped_verification."""

    def __init__(self, observe: Callable[[], object]) -> None:
        self.result = _result(True, summary='local verdict')
        self.raises: Exception | None = None
        self.observe = observe
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.observed: list[object] = []

    async def __call__(self, *args: Any, **kwargs: Any) -> VerifyResult:
        self.calls.append((args, kwargs))
        self.observed.append(self.observe())
        if self.raises is not None:
            raise self.raises
        return self.result


async def _no_sleep(_seconds: float) -> None:
    return None


@dataclass
class _Harness:
    repo: Path
    base: str
    tip: str
    config: OrchestratorConfig
    modules: list[ModuleConfig]
    allocator: HostAllocator
    pool: HostPool
    hooks: _Hooks
    transport: _Transport
    local: _LocalRun
    archive_root: Path
    policy: TaskVerifyPolicy
    dispatcher: Any = None

    def kwargs(self, **overrides: Any) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            'task_files': [_A_PY], 'attempt_id': 1, 'task_id': '7',
            'archive_root': self.archive_root, 'force_workspace': False,
            'role': 'task', 'event_store': None,
        }
        kwargs.update(overrides)
        return kwargs

    async def call(self, **overrides: Any) -> VerifyResult:
        return await self.dispatcher(self.repo, self.config, self.modules, **self.kwargs(**overrides))

    def expected_call(self, **overrides: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
        return (self.repo, self.config, self.modules), self.kwargs(**overrides)

    def sidecars(self, task_id: str = '7', attempt_id: int = 1) -> list[dict[str, Any]]:
        paths = sorted((self.archive_root / task_id).glob(f'attempt-{attempt_id}.dispatch-*.json'))
        return [json.loads(path.read_text()) for path in paths]

    def only_sidecar(self, task_id: str = '7', attempt_id: int = 1) -> dict[str, Any]:
        sidecars = self.sidecars(task_id, attempt_id)
        assert len(sidecars) == 1, sidecars
        return sidecars[0]


def _harness(
    tmp_path: Path,
    policy: TaskVerifyPolicy,
    *,
    touch: str = _A_PY,
    remotes: tuple[str, ...] = ('laptop',),
    demand: bool = False,
    transport: _Transport | None = None,
    modules: list[ModuleConfig] | None = None,
    runner_kwargs: dict[str, Any] | None = None,
) -> _Harness:
    repo = seed_repo(tmp_path / 'repo', _SEED)
    base = _git(repo, 'rev-parse', 'HEAD')
    _git(repo, 'checkout', '-q', '-b', 'task')
    tip = _commit_on_task_branch(repo, touch, 'CHANGED = 2\n')
    transport = transport if transport is not None else _Transport()
    runners = [
        RemoteRunner(
            name, ssh_host=f'{name}.local', git_remote=f'{name}-remote', cwd=repo,
            run=transport.run, ssh_run=transport.ssh_run, id_factory=lambda: 'req1',
            **(runner_kwargs or {}),
        )
        for name in remotes
    ]
    allocator = HostAllocator(runners)
    hooks = _Hooks(demand=demand)
    pool = HostPool(
        allocator, merge_demand=hooks.merge_demand, on_unavailable=hooks.on_unavailable,
        on_recovered=hooks.on_recovered, probe_sleep=_no_sleep,
    )
    harness = _Harness(
        repo=repo, base=base, tip=tip,
        config=OrchestratorConfig(git=GitConfig(main_branch='main'), verify_env_probes=[]),
        modules=modules if modules is not None else [
            ModuleConfig('orchestrator', test_command='true'),
            ModuleConfig('fused_memory', test_command='true'),
        ],
        allocator=allocator, pool=pool, hooks=hooks, transport=transport,
        local=_LocalRun(lambda: allocator.is_busy('laptop')),
        archive_root=tmp_path / 'verify-logs', policy=policy,
    )
    harness.dispatcher = task_verify_dispatch.TaskVerifyDispatcher(
        pool, GitOps(GitConfig(main_branch='main'), repo),
        policy_of=lambda _config: harness.policy, run_local=harness.local,
    )
    return harness


_ORCH_ONLY = TaskVerifyPolicy(host_policy='remote_modules', remote_modules=('orchestrator',))


# ---------------------------------------------------------------------------
# Routing that never reaches a remote
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestLocalRoutes:
    async def test_local_policy_runs_locally_and_records_nothing(self, tmp_path: Path) -> None:
        h = _harness(tmp_path, TaskVerifyPolicy())

        result = await h.call()

        assert result is h.local.result
        assert h.local.calls == [h.expected_call()]
        assert not h.archive_root.exists()
        assert h.hooks.demand_calls == 0
        assert h.transport.log == []
        assert h.dispatcher.decision_for('7', 1) is None

    @pytest.mark.parametrize(('policy', 'touch', 'overrides'), [
        pytest.param(
            TaskVerifyPolicy(host_policy='remote_modules', remote_modules=('fused_memory',)),
            _A_PY, {}, id='module-not-listed',
        ),
        pytest.param(
            TaskVerifyPolicy(host_policy='remote_all'), _A_PY, {'force_workspace': True},
            id='force-workspace',
        ),
        pytest.param(_ORCH_ONLY, _B_PY, {'task_files': None}, id='git-derived-files'),
    ])
    async def test_an_ineligible_verify_runs_locally_with_a_sidecar(
        self, tmp_path: Path, policy: TaskVerifyPolicy, touch: str, overrides: dict[str, Any],
    ) -> None:
        h = _harness(tmp_path, policy, touch=touch)

        result = await h.call(**overrides)

        assert result is h.local.result
        assert h.local.calls == [h.expected_call(**overrides)]
        sidecar = h.only_sidecar()
        assert sidecar['reason'] == 'not_eligible'
        assert sidecar['decision'] == 'local'
        assert sidecar['runner'] == 'local'
        assert sidecar['tip_sha'] == h.tip
        assert sidecar['base_sha'] == h.base
        assert h.hooks.demand_calls == 0
        assert h.local.observed == [False]
        assert h.allocator.is_busy('laptop') is False
        assert h.transport.log == []

    async def test_a_collapsed_module_override_is_ineligible_and_warned_once(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        modules = [
            ModuleConfig('orchestrator', test_command='true'),
            ModuleConfig('fused_memory', test_command='true', verify_env={'X': '1'}),
        ]
        h = _harness(tmp_path, TaskVerifyPolicy(host_policy='remote_all'), modules=modules)
        caplog.set_level(logging.WARNING, logger=_DISPATCH_LOGGER)

        await h.call(attempt_id=1)
        await h.call(attempt_id=2)

        assert h.only_sidecar(attempt_id=1)['reason'] == 'not_eligible'
        assert h.only_sidecar(attempt_id=2)['reason'] == 'not_eligible'
        warnings = [
            r for r in caplog.records
            if r.name == _DISPATCH_LOGGER and r.levelno == logging.WARNING
        ]
        assert len(warnings) == 1
        assert 'fused_memory' in warnings[0].getMessage()
        assert 'verify_env' in warnings[0].getMessage()

    async def test_merge_demand_runs_locally_without_taking_the_slot(self, tmp_path: Path) -> None:
        h = _harness(tmp_path, _ORCH_ONLY, demand=True)

        result = await h.call()

        assert result is h.local.result
        assert h.local.calls == [h.expected_call()]
        assert h.only_sidecar()['reason'] == 'merge_demand'
        assert h.local.observed == [False]
        assert h.allocator.is_busy('laptop') is False
        assert h.transport.log == []

    async def test_no_remote_host_runs_locally(self, tmp_path: Path) -> None:
        h = _harness(tmp_path, TaskVerifyPolicy(host_policy='remote_all'), remotes=())

        result = await h.call()

        assert result is h.local.result
        assert h.only_sidecar()['reason'] == 'no_host'

    async def test_decisions_are_kept_per_task_attempt(self, tmp_path: Path) -> None:
        h = _harness(
            tmp_path, TaskVerifyPolicy(host_policy='remote_modules', remote_modules=('fused_memory',)),
            demand=True,
        )
        await h.call(task_id='7', attempt_id=1)
        second_tip = _commit_on_task_branch(h.repo, _A_PY, 'CHANGED = 3\n')
        h.policy = TaskVerifyPolicy(host_policy='remote_all')

        await h.call(task_id='8', attempt_id=2)

        first = h.dispatcher.decision_for('7', 1)
        second = h.dispatcher.decision_for('8', 2)
        assert (first.reason, first.tip_sha) == ('not_eligible', h.tip)
        assert (second.reason, second.tip_sha) == ('merge_demand', second_tip)
        assert h.dispatcher.decision_for('7', 2) is None
        assert h.dispatcher.decision_for('9', 1) is None

    async def test_no_archive_root_writes_no_sidecar(self, tmp_path: Path) -> None:
        h = _harness(tmp_path, _ORCH_ONLY, demand=True)

        await h.call(archive_root=None)

        assert list(tmp_path.rglob('*.dispatch-*.json')) == []
        decision = h.dispatcher.decision_for('7', 1)
        assert decision is not None
        assert decision.reason == 'merge_demand'
