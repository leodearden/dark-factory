"""TaskVerifyDispatcher: one routing point for a task-role verify, local or remote.

The value layer (policy, eligibility, the decision record) is pure.  The
dispatcher is driven over a real committed repo, a real HostAllocator/HostPool
and a real RemoteRunner built by ``_gate_fixtures.gated_remote_runner``, which
answers the gate's publish and ensure calls; every other call through its public
transport seams (``run=``, ``ssh_run=``, ``id_factory=``) is scripted by
:class:`_Transport`.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import shlex
import subprocess
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from _gate_fixtures import TEST_DEPLOYED_SHA, gated_remote_runner
from _git_fixtures import RepoSeed, seed_repo
from _orch_helpers import wait_responsive

from orchestrator import task_verify_dispatch
from orchestrator.config import GitConfig, ModuleConfig, OrchestratorConfig
from orchestrator.event_store import EventType
from orchestrator.git_ops import GitOps
from orchestrator.task_verify_dispatch import (
    CrossCheck,
    DispatchDecision,
    RedConfirmation,
    TaskVerifyPolicy,
    eligible,
)
from orchestrator.task_verify_spec import TaskVerifySpec, task_spec_from_json
from orchestrator.verify import VerifyInfraError, VerifyResult
from orchestrator.verify_hosts import HostPool
from orchestrator.verify_runner import HostAllocator, RunnerUnavailable, result_to_json

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


def _commit_unrooted_tip(repo: Path) -> str:
    _git(repo, 'checkout', '-q', '--orphan', 'unrooted')
    _git(repo, 'commit', '-q', '-m', 'a tip with no merge base with main')
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
    log: list[str] = field(default_factory=list)
    argvs: list[list[str]] = field(default_factory=list)
    on_delete_ref: Callable[[], object] | None = None
    at_delete_ref: list[object] = field(default_factory=list)
    cancel_raises: Exception | None = None
    hang_on: str | None = None
    reached: asyncio.Event = field(default_factory=asyncio.Event)

    async def run(self, argv: list[str], *, cwd: Any = None) -> Answer:
        label, answer = self.answer(argv)
        self.log.append(label)
        self.argvs.append(list(argv))
        if label == 'delete_ref' and self.on_delete_ref is not None:
            self.at_delete_ref.append(self.on_delete_ref())
        if label == 'cancel_verify' and self.cancel_raises is not None:
            raise self.cancel_raises
        if label == self.hang_on:
            self.reached.set()
            await asyncio.Event().wait()
        return answer

    async def ssh_run(self, argv: list[str], *, cwd: Any = None) -> Answer:
        self.log.append('verify_task')
        self.argvs.append(list(argv))
        return await self.verify(argv)

    def answer(self, argv: list[str]) -> tuple[str, Answer]:
        if argv[:2] == ['git', 'push']:
            return ('delete_ref' if '--delete' in argv else 'push'), (0, '', '')
        if argv[0] == 'ssh':
            return self.ssh_answer(argv[-1])
        return 'unscripted', (127, '', 'unscripted')

    def ssh_answer(self, remote_cmd: str) -> tuple[str, Answer]:
        if 'cancel-verify' in remote_cmd:
            return 'cancel_verify', (self.cancel_rc, '', '')
        if remote_cmd.startswith('pgrep'):
            return 'pgrep', (self.pgrep_rc, '', '')
        return 'unscripted', (127, '', 'unscripted')


class _Hooks:
    def __init__(self, *, demand: bool = False) -> None:
        self.demand = demand
        self.demand_calls = 0
        self.unavailable: list[tuple[str, RunnerUnavailable]] = []
        self.recovered: list[str] = []

    def merge_demand(self) -> bool:
        self.demand_calls += 1
        return self.demand

    def on_unavailable(self, name: str, exc: RunnerUnavailable) -> None:
        self.unavailable.append((name, exc))

    def unavailable_kinds(self) -> list[tuple[str, str]]:
        return [(name, exc.kind) for name, exc in self.unavailable]

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
        gated_remote_runner(
            name, f'{name}.local', f'{name}-remote', repo,
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

    async def test_an_underivable_file_scope_runs_locally_under_remote_all(
        self, tmp_path: Path,
    ) -> None:
        h = _harness(tmp_path, TaskVerifyPolicy(host_policy='remote_all'))
        tip = _commit_unrooted_tip(h.repo)

        result = await h.call(task_files=None)

        assert result is h.local.result
        assert h.local.calls == [h.expected_call(task_files=None)]
        sidecar = h.only_sidecar()
        assert (sidecar['reason'], sidecar['decision']) == ('not_eligible', 'local')
        assert (sidecar['tip_sha'], sidecar['base_sha']) == (tip, None)
        assert h.hooks.demand_calls == 0
        assert h.transport.log == []

    async def test_a_scope_derivation_failure_runs_locally_as_a_dispatcher_error(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        h = _harness(tmp_path, TaskVerifyPolicy(host_policy='remote_all'))
        _git(h.repo, 'symbolic-ref', 'HEAD', 'refs/heads/unborn')
        caplog.set_level(logging.WARNING, logger=_DISPATCH_LOGGER)

        result = await h.call()

        assert result is h.local.result
        assert h.local.calls == [h.expected_call()]
        sidecar = h.only_sidecar()
        assert (sidecar['reason'], sidecar['decision'], sidecar['runner']) == (
            'dispatcher_error', 'local', 'local',
        )
        assert sidecar['tip_sha'] is None
        assert h.hooks.demand_calls == 0
        assert h.transport.log == []
        [warning] = [r for r in caplog.records if r.name == _DISPATCH_LOGGER]
        assert warning.levelno == logging.WARNING
        assert warning.exc_info is not None

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



# ---------------------------------------------------------------------------
# The remote leg, through a real RemoteRunner
# ---------------------------------------------------------------------------


def _answers(result: VerifyResult) -> VerifyAnswer:
    async def answer(argv: list[str]) -> Answer:
        return 0, result_to_json(result), ''
    return answer


def _exits(rc: int) -> VerifyAnswer:
    async def answer(argv: list[str]) -> Answer:
        return rc, '', 'ssh: connect to host laptop.local port 22: No route to host'
    return answer


def _raises(exc: Exception) -> VerifyAnswer:
    async def answer(argv: list[str]) -> Answer:
        raise exc
    return answer


def _sent_spec(transport: _Transport) -> TaskVerifySpec:
    remote_cmd = transport.argvs[transport.log.index('verify_task')][-1]
    words = shlex.split(remote_cmd)
    return task_spec_from_json(words[words.index('--spec') + 1])


class _Events:
    """A recording event store: every emit, in order."""

    def __init__(self) -> None:
        self.emitted: list[tuple[Any, str | None, dict[str, Any]]] = []

    def emit(self, event_type: Any, *, task_id: str | None = None, data: Any = None, **_: Any) -> None:
        self.emitted.append((event_type, task_id, data or {}))


def _assert_slot_free(h: _Harness) -> None:
    assert h.allocator.is_busy('laptop') is False
    assert h.pool.holders() == {}


@pytest.mark.asyncio
class TestRemoteLeg:
    async def test_a_remote_green_is_returned_and_archived(self, tmp_path: Path) -> None:
        transport = _Transport(verify=_answers(_result(True, summary='remote verdict')))
        h = _harness(tmp_path, _ORCH_ONLY, transport=transport)

        result = await h.call()

        assert result.passed is True
        assert result.summary == 'remote verdict'
        assert h.local.calls == []
        task_dir = h.archive_root / '7'
        [pass_summary] = list(task_dir.glob('attempt-1.remote-laptop.pass-summary-*.json'))
        [sidecar_path] = list(task_dir.glob('attempt-1.dispatch-*.json'))
        assert result.archive_log_paths == [str(pass_summary)]
        assert sorted(task_dir.iterdir()) == sorted([pass_summary, sidecar_path])
        sidecar = json.loads(sidecar_path.read_text())
        assert sidecar['decision'] == 'remote'
        assert sidecar['runner'] == 'laptop'
        assert sidecar['reason'] == 'remote'
        assert (sidecar['tip_sha'], sidecar['base_sha']) == (h.tip, h.base)
        assert sidecar['remote_wall_secs'] is not None
        assert sidecar['lease_held_secs'] is not None
        _assert_slot_free(h)
        assert h.hooks.recovered == ['laptop']
        spec = _sent_spec(transport)
        assert spec.task_files == (_A_PY,)
        assert spec.attempt_id == 1
        assert spec.tip_sha == h.tip

    async def test_a_dirty_tree_runs_locally_before_any_push(self, tmp_path: Path) -> None:
        transport = _Transport(verify=_answers(_result(True)))
        h = _harness(tmp_path, _ORCH_ONLY, transport=transport)
        (h.repo / 'scratch.txt').write_text('untracked\n')

        result = await h.call()

        assert result is h.local.result
        assert h.local.calls == [h.expected_call()]
        assert h.only_sidecar()['reason'] == 'dirty_tree'
        assert 'push' not in h.transport.log
        _assert_slot_free(h)

    async def test_a_gate_answering_another_deploy_runs_locally_before_any_verify(
        self, tmp_path: Path,
    ) -> None:
        other_deploy = dataclasses.replace(TEST_DEPLOYED_SHA, sha='e' * 40)
        h = _harness(
            tmp_path, _ORCH_ONLY, transport=_Transport(verify=_answers(_result(True))),
            runner_kwargs={'deployed_sha': other_deploy},
        )

        result = await h.call()

        assert result is h.local.result
        assert h.local.calls == [h.expected_call()]
        assert h.only_sidecar()['reason'] == 'runner_unavailable'
        [(name, skew)] = h.hooks.unavailable
        assert (name, skew.kind) == ('laptop', 'wire_shape')
        assert skew.identity is not None
        assert skew.identity.local.code_sha == other_deploy.sha
        assert skew.identity.remote.code_sha == TEST_DEPLOYED_SHA.sha
        assert h.hooks.recovered == []
        assert h.allocator.is_quarantined('laptop') is False
        assert h.transport.log == []
        _assert_slot_free(h)

    async def test_a_gate_built_for_the_dispatch_is_announced_to_its_event_store(
        self, tmp_path: Path,
    ) -> None:
        h = _harness(
            tmp_path, _ORCH_ONLY, transport=_Transport(verify=_answers(_result(True))),
            runner_kwargs={'gate_built': True},
        )
        events = _Events()

        result = await h.call(event_store=events)

        assert result.passed is True
        assert [(event, task) for event, task, _ in events.emitted] == [
            (EventType.runner_stale, '7'), (EventType.runner_synced, '7'),
        ]
        assert h.only_sidecar()['reason'] == 'remote'

    async def test_an_unreachable_runner_runs_locally(self, tmp_path: Path) -> None:
        h = _harness(tmp_path, _ORCH_ONLY, transport=_Transport(verify=_exits(255)))

        result = await h.call()

        assert result is h.local.result
        assert h.local.calls == [h.expected_call()]
        assert h.only_sidecar()['reason'] == 'runner_unavailable'
        assert h.hooks.unavailable_kinds() == [('laptop', 'transport')]
        assert h.hooks.recovered == []
        _assert_slot_free(h)

    async def test_an_infra_transient_remote_red_reruns_locally_in_the_same_call(
        self, tmp_path: Path,
    ) -> None:
        remote_red = _result(False, category='env_transient', summary='uv cache locked')
        h = _harness(tmp_path, _ORCH_ONLY, transport=_Transport(verify=_answers(remote_red)))

        result = await h.call()

        assert result is h.local.result
        assert h.local.calls == [h.expected_call()]
        assert h.only_sidecar()['reason'] == 'infra_transient'
        _assert_slot_free(h)

    async def test_an_unexpected_transport_error_is_a_dispatcher_error(self, tmp_path: Path) -> None:
        h = _harness(
            tmp_path, _ORCH_ONLY, transport=_Transport(verify=_raises(RuntimeError('boom'))),
        )

        with pytest.raises(RuntimeError, match='boom'):
            await h.call()

        assert h.local.calls == []
        assert h.only_sidecar()['reason'] == 'dispatcher_error'
        _assert_slot_free(h)



# ---------------------------------------------------------------------------
# Cancellation while the remote verify is in flight
# ---------------------------------------------------------------------------


def _hangs(started: asyncio.Event) -> VerifyAnswer:
    async def answer(argv: list[str]) -> Answer:
        started.set()
        await asyncio.Event().wait()
        raise AssertionError('a never-set event was set')
    return answer


def _hangs_then_raises(started: asyncio.Event, exc: Exception) -> VerifyAnswer:
    async def answer(argv: list[str]) -> Answer:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise exc from None
        raise AssertionError('a never-set event was set')
    return answer


def _pending_tasks() -> set[asyncio.Task[Any]]:
    current = asyncio.current_task()
    return {t for t in asyncio.all_tasks() if t is not current and not t.done()}


async def _cancel_mid_verify(h: _Harness, started: asyncio.Event) -> None:
    before = _pending_tasks()
    call = asyncio.create_task(h.call())
    await wait_responsive(started.wait(), label='dispatch reached its hang point')

    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call

    assert _pending_tasks() == before


@pytest.mark.asyncio
class TestCancellation:
    async def test_cancel_verify_is_issued_while_the_dispatch_is_in_flight(
        self, tmp_path: Path,
    ) -> None:
        started = asyncio.Event()
        h = _harness(tmp_path, _ORCH_ONLY, transport=_Transport(verify=_hangs(started)))

        await _cancel_mid_verify(h, started)

        log = h.transport.log
        assert 'cancel_verify' in log
        assert log.index('verify_task') < log.index('cancel_verify') < log.index('delete_ref')
        cancel_argv = h.transport.argvs[log.index('cancel_verify')]
        assert 'orchestrator cancel-verify --request-id req1' in cancel_argv[-1]
        _assert_slot_free(h)
        assert h.sidecars() == []
        assert h.dispatcher.decision_for('7', 1) is None
        assert h.local.calls == []

    async def test_the_slot_stays_leased_until_the_cancelled_dispatch_unwinds(
        self, tmp_path: Path,
    ) -> None:
        started = asyncio.Event()
        h = _harness(tmp_path, _ORCH_ONLY, transport=_Transport(verify=_hangs(started)))
        h.transport.on_delete_ref = lambda: (h.allocator.is_busy('laptop'), dict(h.pool.holders()))

        await _cancel_mid_verify(h, started)

        assert h.transport.at_delete_ref == [(True, {'laptop': 'task:7'})]
        log = h.transport.log
        assert log.index('verify_task') < log.index('cancel_verify') < log.index('delete_ref')
        _assert_slot_free(h)

    async def test_a_failed_cancel_parks_the_host_for_the_next_call(self, tmp_path: Path) -> None:
        started = asyncio.Event()
        transport = _Transport(verify=_hangs(started), cancel_rc=1, pgrep_rc=0)
        h = _harness(tmp_path, _ORCH_ONLY, transport=transport)
        transport.on_delete_ref = lambda: (
            h.allocator.is_busy('laptop'), h.allocator.is_parked('laptop'), dict(h.pool.holders()),
        )

        await _cancel_mid_verify(h, started)

        assert transport.at_delete_ref == [(True, False, {'laptop': 'task:7'})]
        log = transport.log
        assert log.index('cancel_verify') < log.index('delete_ref') < log.index('pgrep')
        assert h.allocator.is_parked('laptop') is True
        assert h.pool.holders() == {}
        result = await h.call(attempt_id=2)
        assert result is h.local.result
        assert h.only_sidecar(attempt_id=2)['reason'] == 'host_parked'

    async def test_a_cancel_while_the_tip_is_pushed_frees_the_slot(self, tmp_path: Path) -> None:
        transport = _Transport(hang_on='push')
        h = _harness(tmp_path, _ORCH_ONLY, transport=transport)

        await _cancel_mid_verify(h, transport.reached)

        assert transport.log == ['push', 'cancel_verify']
        assert transport.argvs[0][-1] == f'{h.tip}:refs/merge-verify/req1'
        _assert_slot_free(h)
        assert h.sidecars() == []
        assert h.local.calls == []

    async def test_a_remote_leg_that_fails_while_being_reaped_is_logged(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        started = asyncio.Event()
        verify = _hangs_then_raises(started, RuntimeError('torn down'))
        h = _harness(tmp_path, _ORCH_ONLY, transport=_Transport(verify=verify))
        caplog.set_level(logging.DEBUG, logger=_DISPATCH_LOGGER)

        await _cancel_mid_verify(h, started)

        [record] = [r for r in caplog.records if r.name == _DISPATCH_LOGGER]
        assert 'torn down' in record.getMessage()
        _assert_slot_free(h)

    async def test_a_cancel_verify_that_raises_still_frees_the_slot(self, tmp_path: Path) -> None:
        started = asyncio.Event()
        transport = _Transport(verify=_hangs(started), cancel_raises=RuntimeError('ssh died'))
        h = _harness(tmp_path, _ORCH_ONLY, transport=transport)

        await _cancel_mid_verify(h, started)

        log = transport.log
        assert log.index('cancel_verify') < log.index('delete_ref') < log.index('pgrep')
        _assert_slot_free(h)
        assert h.allocator.is_parked('laptop') is False



# ---------------------------------------------------------------------------
# Parity knobs, read per call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestParity:
    async def test_every_nth_remote_verdict_is_cross_checked_locally(self, tmp_path: Path) -> None:
        policy = dataclasses.replace(_ORCH_ONLY, cross_check_every_n=2)
        transport = _Transport(verify=_answers(_result(True, summary='remote verdict')))
        h = _harness(tmp_path, policy, transport=transport)

        first = await h.call(attempt_id=1)
        assert first.summary == 'remote verdict'
        assert h.local.calls == []
        assert h.only_sidecar(attempt_id=1)['reason'] == 'remote'
        assert h.only_sidecar(attempt_id=1)['cross_check'] is None

        second = await h.call(attempt_id=2)
        assert second is h.local.result
        assert h.local.calls == [h.expected_call(attempt_id=2)]
        assert h.transport.log.count('verify_task') == 2
        sidecar = h.only_sidecar(attempt_id=2)
        assert (sidecar['reason'], sidecar['decision'], sidecar['runner']) == (
            'cross_check', 'local', 'local',
        )
        assert sidecar['cross_check'] == {'remote_passed': True, 'local_passed': True, 'agree': True}
        assert sidecar['remote_wall_secs'] is not None

        transport.verify = _answers(_result(False, category='test_failure'))
        third = await h.call(attempt_id=3)
        assert third.passed is False
        assert h.only_sidecar(attempt_id=3)['reason'] == 'remote'

        fourth = await h.call(attempt_id=4)
        assert fourth is h.local.result
        assert h.only_sidecar(attempt_id=4)['cross_check'] == {
            'remote_passed': False, 'local_passed': True, 'agree': False,
        }
        _assert_slot_free(h)

    async def test_a_refusal_does_not_consume_a_cross_check_turn(self, tmp_path: Path) -> None:
        policy = dataclasses.replace(_ORCH_ONLY, cross_check_every_n=2)
        h = _harness(tmp_path, policy, transport=_Transport(verify=_answers(_result(True))))

        await h.call(attempt_id=1)
        h.hooks.demand = True
        await h.call(attempt_id=2)
        h.hooks.demand = False
        await h.call(attempt_id=3)

        assert h.only_sidecar(attempt_id=2)['reason'] == 'merge_demand'
        assert h.only_sidecar(attempt_id=3)['reason'] == 'cross_check'

    async def test_a_remote_red_is_confirmed_locally(self, tmp_path: Path) -> None:
        policy = dataclasses.replace(_ORCH_ONLY, red_confirms_locally=True)
        remote_red = _result(False, category='test_failure')
        h = _harness(tmp_path, policy, transport=_Transport(verify=_answers(remote_red)))

        result = await h.call()

        assert result is h.local.result
        assert h.local.calls == [h.expected_call()]
        sidecar = h.only_sidecar()
        assert (sidecar['reason'], sidecar['decision'], sidecar['runner']) == (
            'red_confirm', 'local', 'local',
        )
        assert sidecar['red_confirmed_locally'] == {'remote_passed': False, 'local_passed': True}
        assert sidecar['cross_check'] is None
        _assert_slot_free(h)

    async def test_a_remote_green_is_not_confirmed_locally(self, tmp_path: Path) -> None:
        policy = dataclasses.replace(_ORCH_ONLY, red_confirms_locally=True)
        h = _harness(tmp_path, policy, transport=_Transport(verify=_answers(_result(True))))

        result = await h.call()

        assert result.passed is True
        assert h.local.calls == []
        assert h.only_sidecar()['reason'] == 'remote'

    async def test_a_policy_reloaded_to_local_short_circuits(self, tmp_path: Path) -> None:
        h = _harness(tmp_path, _ORCH_ONLY, transport=_Transport(verify=_answers(_result(True))))
        await h.call(attempt_id=1)

        h.policy = TaskVerifyPolicy()
        result = await h.call(attempt_id=2)

        assert result is h.local.result
        assert h.sidecars(attempt_id=2) == []
        assert h.dispatcher.decision_for('7', 2) is None
        assert h.only_sidecar(attempt_id=1)['reason'] == 'remote'

    async def test_a_local_run_failure_keeps_the_route_reason(self, tmp_path: Path) -> None:
        h = _harness(tmp_path, _ORCH_ONLY, demand=True)
        h.local.raises = VerifyInfraError('test', 28, 'No space left on device')

        with pytest.raises(VerifyInfraError):
            await h.call()

        assert h.only_sidecar()['reason'] == 'merge_demand'
