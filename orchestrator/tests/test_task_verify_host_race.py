"""TaskVerifyDispatcher's host race against the real local verify queue.

PRD ``plans/remote-task-leg-verify-prd.md`` decision 14 (an eligible verify
queues on every host at once and the first turn wins) and decision 16 (a task
slot is a gate turn on every host); rows 5, 6, 8a, 14, 21, 26-45 and 49-51.

The local run is the real ``run_scoped_verification`` (wrapped only to record
the plan it was handed and how often it was cancelled) on a scratch repo,
behind the real local ``AdmissionGate`` (``gate_for`` the test's slots dir).
Each remote host is the pool's own gate, with a fresh real ``RemoteRunner``
per turn over a scripted transport.  Every local leg is a shell body that
logs its pid and then waits for its ``open`` file, so a test decides when a
leg finishes.  A test holds a turn on any gate with :func:`_queue`.  Every
config points ``verify_admission_slots_dir`` under ``tmp_path``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import itertools
import json
import logging
import os
import shlex
import subprocess
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeVar

import pytest
from _gate_fixtures import gated_remote_runner
from _task_verify_fixtures import VERIFY_MODULE, seed_task_branch_repo
from shared.verify_admission import acquire_task_slot

from orchestrator.config import GitConfig, ModuleConfig, OrchestratorConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.git_ops import GitOps
from orchestrator.task_verify_dispatch import TaskVerifyDispatcher, TaskVerifyPolicy
from orchestrator.task_verify_spec import TaskVerifySpec, task_spec_from_json
from orchestrator.verify import (
    VerifyResult,
    admission_active,
    admission_policy,
    run_scoped_verification,
)
from orchestrator.verify_admission_gate import (
    AdmissionGate,
    AdmissionPlan,
    AdmissionPolicy,
    gate_for,
)
from orchestrator.verify_categories import INFRA_TRANSIENT_CATEGORIES
from orchestrator.verify_hosts import HostPool
from orchestrator.verify_runner import HostAllocator, RunnerUnavailable, result_to_json
from orchestrator.workflow_types import CancellationScope, WorkflowCancelled

pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures('code_default_config')]

_REAL = pytest.mark.real_verify_admission
_DISPATCH_LOGGER = 'orchestrator.task_verify_dispatch'
_REMOTE_ALL = TaskVerifyPolicy(host_policy='remote_all')
_SECOND_MODULE = 'mod2'
_TEST, _LINT, _TYPE = (f'{kind}-{VERIFY_MODULE}' for kind in ('test', 'lint', 'type'))
_PROVISION = 'provision'
_SSH_DOWN = (255, '', 'ssh: connect to host laptop.local port 22: No route to host')
_T = TypeVar('_T')


def _result(passed: bool, *, category: str = '', summary: str = '') -> VerifyResult:
    return VerifyResult(
        passed=passed, test_output='' if passed else 'FAILED test_x', lint_output='',
        type_output='', summary=summary or ('remote verdict' if passed else 'remote red'),
        category=category,
    )


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ['git', '-C', str(repo), *args], check=True, capture_output=True, text=True,
    ).stdout.strip()


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _pending_tasks() -> set[asyncio.Task[Any]]:
    current = asyncio.current_task()
    return {t for t in asyncio.all_tasks() if t is not current and not t.done()}


async def _bounded(awaitable: Awaitable[_T], timeout: float = 30.0) -> _T:
    async with asyncio.timeout(timeout):
        return await awaitable


async def _until(
    predicate: Callable[[], bool], *, watching: asyncio.Task[Any] | None = None,
    timeout: float = 15.0,
) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            if watching is not None and watching.done():
                watching.result()
                raise AssertionError('the call finished before reaching the awaited state')
            await asyncio.sleep(0.01)


async def _no_sleep(_seconds: float) -> None:
    return None


class _Legs:
    """Local leg commands: each appends its pid to ``<label>.spawns``, then waits for ``<label>.open``."""

    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir()

    def command(self, label: str, *, ignore_term: bool = False) -> str:
        self.release(label)
        trap = "trap '' TERM; " if ignore_term else ''
        return f'{trap}echo $$ >> {self._quoted("spawns", label)}; {self._wait(label)}'

    def times_out_first(self, label: str) -> str:
        """Waits (until its leg times out) on its first run only."""
        spawns = self._quoted('spawns', label)
        return f'echo $$ >> {spawns}; [ "$(wc -l < {spawns})" -gt 1 ] || {self._wait(label)}'

    def block(self, *labels: str) -> None:
        for label in labels:
            self._path('open', label).unlink(missing_ok=True)

    def release(self, *labels: str) -> None:
        for label in labels:
            self._path('open', label).touch()

    def pids(self, label: str) -> list[int]:
        path = self._path('spawns', label)
        return [int(pid) for pid in path.read_text().split()] if path.exists() else []

    def spawned(self, label: str) -> bool:
        return bool(self.pids(label))

    def all_pids(self) -> list[int]:
        return [
            int(pid) for path in sorted(self.root.glob('*.spawns'))
            for pid in path.read_text().split()
        ]

    def _wait(self, label: str) -> str:
        return f'while [ ! -e {self._quoted("open", label)} ]; do sleep 0.02; done'

    def _path(self, kind: str, label: str) -> Path:
        return self.root / f'{label}.{kind}'

    def _quoted(self, kind: str, label: str) -> str:
        return shlex.quote(str(self._path(kind, label)))


def _spec_of(argv: list[str]) -> TaskVerifySpec:
    words = shlex.split(argv[-1])
    return task_spec_from_json(words[words.index('--spec') + 1])


@dataclass
class _Remote:
    """One host's scripted transport: every argv is logged; the verify answers *verdict*
    (or *exit*, or raises *raises*), after *released* when *hold* is set."""

    verdict: VerifyResult = field(default_factory=lambda: _result(True))
    hold: bool = False
    exit: tuple[int, str, str] | None = None
    raises: Exception | None = None
    cancel_rc: int = 0
    pgrep_rc: int = 1
    on_pgrep: Callable[[], object] | None = None
    at_pgrep: list[object] = field(default_factory=list)
    log: list[str] = field(default_factory=list)
    argvs: list[list[str]] = field(default_factory=list)
    started: asyncio.Event = field(default_factory=asyncio.Event)
    released: asyncio.Event = field(default_factory=asyncio.Event)

    async def run(self, argv: list[str], *, cwd: Any = None) -> tuple[int, str, str]:
        label = self._label(argv)
        self.log.append(label)
        self.argvs.append(list(argv))
        if label == 'pgrep' and self.on_pgrep is not None:
            self.at_pgrep.append(self.on_pgrep())
        answers = {
            'push': (0, '', ''), 'delete_ref': (0, '', ''),
            'cancel_verify': (self.cancel_rc, '', ''), 'pgrep': (self.pgrep_rc, '', ''),
        }
        return answers.get(label, (127, '', 'unscripted'))

    async def ssh_run(self, argv: list[str], *, cwd: Any = None) -> tuple[int, str, str]:
        self.log.append('verify_task')
        self.argvs.append(list(argv))
        self.started.set()
        if self.hold:
            await self.released.wait()
        if self.raises is not None:
            raise self.raises
        if self.exit is not None:
            return self.exit
        return 0, result_to_json(self.verdict), ''

    def specs(self) -> list[TaskVerifySpec]:
        return [
            _spec_of(argv)
            for argv, label in zip(self.argvs, self.log, strict=True) if label == 'verify_task'
        ]

    @staticmethod
    def _label(argv: list[str]) -> str:
        if argv[:2] == ['git', 'push']:
            return 'delete_ref' if '--delete' in argv else 'push'
        if argv[0] == 'ssh' and 'cancel-verify' in argv[-1]:
            return 'cancel_verify'
        if argv[0] == 'ssh' and argv[-1].startswith('pgrep'):
            return 'pgrep'
        return 'unscripted'


@dataclass
class _Turn:
    """A turn the test holds on one gate, from outside any verify."""

    task: asyncio.Task[None]
    granted: asyncio.Event
    release: asyncio.Event

    async def leave(self) -> None:
        self.release.set()
        await _bounded(self.task)


def _queue(
    gate: AdmissionGate, policy: Callable[[], AdmissionPolicy], *, task_id: str, arrival: float | None = None,
) -> _Turn:
    """Queue a task-role plan of *task_id* on *gate*, holding its turn until released."""
    plan = AdmissionPlan.begin(role='task', task_id=task_id)
    if arrival is not None:
        plan = dataclasses.replace(plan, arrival=arrival)
    granted, release = asyncio.Event(), asyncio.Event()

    async def hold() -> None:
        async with gate.admit(plan.request(module=None, scope_kind=None), policy):
            granted.set()
            await release.wait()

    return _Turn(asyncio.create_task(hold()), granted, release)


def _waiters(gate: AdmissionGate) -> list[str | None]:
    return [w.request.task_id for w in gate.snapshot().waiters]


def _holding(gate: AdmissionGate) -> list[str | None]:
    return [h.request.task_id for h in gate.snapshot().holders]


def _waiting(gate: AdmissionGate, task_id: str) -> bool:
    return task_id in _waiters(gate)


class _RealLocal:
    """``run_scoped_verification`` itself, recording each call's plan and its cancel count."""

    def __init__(self) -> None:
        self.plans: list[AdmissionPlan | None] = []
        self.tasks: list[asyncio.Task[Any] | None] = []
        self.cancels: list[int] = []
        self.on_start: Callable[[], object] | None = None
        self.at_start: list[object] = []
        self.on_return: Callable[[], object] | None = None

    async def __call__(self, *args: Any, **kwargs: Any) -> VerifyResult:
        self.plans.append(kwargs.get('admission_plan'))
        self.tasks.append(asyncio.current_task())
        if self.on_start is not None:
            self.at_start.append(self.on_start())
        try:
            result = await run_scoped_verification(*args, **kwargs)
        except asyncio.CancelledError:
            task = asyncio.current_task()
            self.cancels.append(task.cancelling() if task is not None else -1)
            raise
        if self.on_return is not None:
            self.on_return()
        return result


class _PausableGitOps(GitOps):
    """GitOps whose fork-point read waits on *pause* once it is set: a scope derivation held open."""

    def __init__(self, config: GitConfig, repo: Path) -> None:
        super().__init__(config, repo)
        self.pause: asyncio.Event | None = None
        self.paused = asyncio.Event()

    async def merge_base_with_main(self, ref: str) -> str | None:
        if self.pause is not None:
            self.paused.set()
            await self.pause.wait()
        return await super().merge_base_with_main(ref)


@dataclass
class _Rig:
    repo: Path
    tip: str
    config: OrchestratorConfig
    legs: _Legs
    modules: list[ModuleConfig]
    task_files: list[str]
    quarantine: set[str]
    unavailable: list[tuple[str, RunnerUnavailable]]
    pool: HostPool
    remotes: dict[str, _Remote]
    local: _RealLocal
    git_ops: _PausableGitOps
    events: EventStore
    archive_root: Path
    policy: TaskVerifyPolicy
    bench_hooks: list[Callable[[str], object]]
    policy_raises: Exception | None = None
    dispatcher: TaskVerifyDispatcher = field(init=False)

    def policy_of(self, _config: OrchestratorConfig) -> TaskVerifyPolicy:
        if self.policy_raises is not None:
            exc, self.policy_raises = self.policy_raises, None
            raise exc
        return self.policy

    async def call(self, *, task_id: str = '7', attempt_id: int = 1) -> VerifyResult:
        return await self.dispatcher(
            self.repo, self.config, self.modules, task_files=list(self.task_files),
            attempt_id=attempt_id, task_id=task_id, archive_root=self.archive_root,
            event_store=self.events,
        )

    def start(self, **kwargs: Any) -> asyncio.Task[VerifyResult]:
        return asyncio.create_task(self.call(**kwargs))

    @property
    def local_gate(self) -> AdmissionGate:
        return gate_for(self.config.verify_admission_slots_dir)

    @property
    def laptop(self) -> AdmissionGate:
        return self.pool.gate('laptop')

    def hold(self, host: str = 'laptop', *, task_id: str = 'holder') -> _Turn:
        return _queue(
            self.pool.gate(host), lambda: self.pool.policy(host, self.config), task_id=task_id,
        )

    def hold_local(self, *, task_id: str = 'local-holder', arrival: float | None = None) -> _Turn:
        return _queue(
            self.local_gate, lambda: admission_policy(self.config), task_id=task_id,
            arrival=arrival,
        )

    async def hold_both(self) -> tuple[_Turn, _Turn]:
        """The laptop's turn and the local turn, both held by the test."""
        laptop, local = self.hold(), self.hold_local()
        await _bounded(laptop.granted.wait())
        await _bounded(local.granted.wait())
        return laptop, local

    async def queued_on_both(self, call: asyncio.Task[Any], task_id: str = '7') -> None:
        await _until(
            lambda: _waiting(self.laptop, task_id) and _waiting(self.local_gate, task_id),
            watching=call,
        )

    def sidecar(self, task_id: str = '7', attempt_id: int = 1) -> dict[str, Any]:
        [path] = sorted((self.archive_root / task_id).glob(f'attempt-{attempt_id}.dispatch-*.json'))
        return json.loads(path.read_text())

    def plan_events(self) -> list[dict[str, Any]]:
        return [
            row['data'] for row in self.events.fetch_events_by_type(EventType.verify_admission_plan)
        ]

    def unavailable_kinds(self) -> list[tuple[str, str]]:
        return [(host, exc.kind) for host, exc in self.unavailable]


TestLeg = Literal['waits', 'none', 'times_out_first']
Provision = Literal['none', 'waits', 'ignores_term']


def _module(
    legs: _Legs, prefix: str, *, test_leg: TestLeg, timeout: float | None,
    lint_ignores_term: bool,
) -> ModuleConfig:
    test_label = f'test-{prefix}'
    test_command = {
        'waits': lambda: legs.command(test_label),
        'none': lambda: None,
        'times_out_first': lambda: legs.times_out_first(test_label),
    }[test_leg]()
    return ModuleConfig(
        prefix,
        test_command=test_command,
        lint_command=legs.command(f'lint-{prefix}', ignore_term=lint_ignores_term),
        type_check_command=legs.command(f'type-{prefix}'),
        verify_command_timeout_secs=timeout,
    )


def _config(tmp_path: Path, *, concurrent: bool, **overrides: Any) -> OrchestratorConfig:
    fields: dict[str, Any] = {
        'project_root': tmp_path,
        'git': GitConfig(main_branch='main'),
        'verify_env_probes': [],
        'verify_admission_enabled': True,
        'verify_admission_slots_dir': str(tmp_path / 'slots'),
        'verify_admission_order': 'arrival',
        'verify_admission_task_slots': 1,
        'verify_clock_stop_enabled': False,
        'concurrent_verify': concurrent,
    }
    fields.update(overrides)
    return OrchestratorConfig(**fields)


def _rig(
    tmp_path: Path,
    *,
    hosts: tuple[str, ...] = ('laptop',),
    policy: TaskVerifyPolicy = _REMOTE_ALL,
    concurrent: bool = True,
    modules: tuple[str, ...] = (VERIFY_MODULE,),
    test_leg: TestLeg = 'waits',
    module_timeout: float | None = None,
    provision: Provision = 'none',
    lints_ignoring_term: tuple[str, ...] = (),
) -> _Rig:
    """*provision* other than ``'none'`` makes the worktree cold (it gains ``.task/``)
    and runs the :data:`_PROVISION` leg as its cold pre-provision.  The lint leg of
    each module in *lints_ignoring_term* traps SIGTERM, so only a SIGKILL ends it."""
    repo, _base, tip = seed_task_branch_repo(tmp_path, modules=modules)
    legs = _Legs(tmp_path / 'legs')
    config_overrides: dict[str, Any] = {}
    if provision != 'none':
        (repo / '.task').mkdir()
        config_overrides['verify_cold_preprovision_command'] = legs.command(
            _PROVISION, ignore_term=provision == 'ignores_term',
        )
    module_configs = [
        _module(
            legs, prefix, test_leg=test_leg, timeout=module_timeout,
            lint_ignores_term=prefix in lints_ignoring_term,
        )
        for prefix in modules
    ]
    remotes = {host: _Remote() for host in hosts}
    request_ids = itertools.count(1)
    quarantine: set[str] = set()
    unavailable: list[tuple[str, RunnerUnavailable]] = []
    bench_hooks: list[Callable[[str], object]] = []

    def runner_for(host: str):
        remote = remotes[host]
        return gated_remote_runner(
            host, f'{host}.local', f'{host}-remote', repo,
            run=remote.run, ssh_run=remote.ssh_run, id_factory=lambda: f'req{next(request_ids)}',
        )

    def on_unavailable(host: str, exc: RunnerUnavailable) -> None:
        unavailable.append((host, exc))
        quarantine.add(host)
        for hook in bench_hooks:
            hook(host)

    pool = HostPool(
        hosts, runner_for=runner_for, quarantine=quarantine,
        on_unavailable=on_unavailable, probe_sleep=_no_sleep,
    )
    git_ops = _PausableGitOps(GitConfig(main_branch='main'), repo)
    rig = _Rig(
        repo=repo, tip=tip,
        config=_config(tmp_path, concurrent=concurrent, **config_overrides),
        legs=legs, modules=module_configs,
        task_files=[f'{prefix}/test_x.py' for prefix in modules],
        quarantine=quarantine, unavailable=unavailable, pool=pool, remotes=remotes,
        local=_RealLocal(), git_ops=git_ops,
        events=EventStore(tmp_path / 'runs.db', run_id='race'),
        archive_root=tmp_path / 'verify-logs', policy=policy, bench_hooks=bench_hooks,
    )
    rig.dispatcher = TaskVerifyDispatcher(
        pool, git_ops, policy_of=rig.policy_of, run_local=rig.local,
    )
    return rig


async def _assert_cancelled_call_unwound(rig: _Rig, before: set[asyncio.Task[Any]]) -> None:
    """The local task is done when the call raises, cancelled once.  A cancel landing
    while the call awaits its shielded reap raises before that reap has ended, so the
    tasks the reap still runs are awaited only until they end."""
    assert rig.local.tasks
    assert all(task is not None and task.done() for task in rig.local.tasks)
    assert rig.local.cancels == [1]
    await _until(lambda: _pending_tasks() - before == set())


def _assert_unwound(rig: _Rig, *, laptop_holders: list[str], local_holders: list[str]) -> None:
    assert rig.pool.holders() == {}
    assert _waiters(rig.laptop) == []
    assert _waiters(rig.local_gate) == []
    assert _holding(rig.laptop) == laptop_holders
    assert _holding(rig.local_gate) == local_holders


# ---------------------------------------------------------------------------
# Rows 5, 6, 8a, 14, 21: one call, one host
# ---------------------------------------------------------------------------


@_REAL
class TestOneHost:
    async def test_row5_a_free_turn_runs_remote_and_never_starts_local(self, tmp_path: Path) -> None:
        rig = _rig(tmp_path)

        result = await _bounded(rig.call())

        assert result.summary == 'remote verdict'
        assert rig.local.plans == []
        assert rig.legs.all_pids() == []
        sidecar = rig.sidecar()
        assert (sidecar['decision'], sidecar['runner'], sidecar['reason']) == (
            'remote', 'laptop', 'remote',
        )
        assert sidecar['remote_wait_secs'] >= 0
        _assert_unwound(rig, laptop_holders=[], local_holders=[])

    async def test_row6_a_busy_merge_lease_on_the_host_leaves_its_task_turn_free(
        self, tmp_path: Path,
    ) -> None:
        rig = _rig(tmp_path)
        merge_transport = _Remote()
        merge_runner = gated_remote_runner(
            'laptop', 'laptop.local', 'laptop-remote', rig.repo,
            run=merge_transport.run, ssh_run=merge_transport.ssh_run, id_factory=lambda: 'merge1',
        )
        allocator = HostAllocator([merge_runner])
        lease = allocator.acquire_remote()
        assert lease is not None and allocator.is_busy('laptop')

        result = await _bounded(rig.call())

        assert result.summary == 'remote verdict'
        assert rig.sidecar()['reason'] == 'remote'
        assert rig.remotes['laptop'].log.count('verify_task') == 1
        assert allocator.is_busy('laptop')
        assert allocator.acquire_remote() is None
        assert merge_transport.log == []
        _assert_unwound(rig, laptop_holders=[], local_holders=[])

    async def test_row8a_no_remote_host_runs_local_inline_with_its_own_plan(
        self, tmp_path: Path,
    ) -> None:
        rig = _rig(tmp_path, hosts=())

        result = await _bounded(rig.call())

        assert result.passed
        assert rig.local.plans == [None]
        [event] = rig.plan_events()
        assert (event['task_id'], event['role'], event['outcome']) == ('7', 'task', 'returned')
        sidecar = rig.sidecar()
        assert (sidecar['reason'], sidecar['runner'], sidecar['remote_wait_secs']) == (
            'no_host', 'local', None,
        )

    async def test_row14_a_failed_cancel_holds_the_turn_through_the_probe_then_benches(
        self, tmp_path: Path,
    ) -> None:
        rig = _rig(tmp_path)
        remote = rig.remotes['laptop']
        remote.hold, remote.cancel_rc, remote.pgrep_rc = True, 1, 0
        remote.on_pgrep = lambda: (_holding(rig.laptop), dict(rig.pool.holders()))
        call = rig.start()
        await _bounded(remote.started.wait())

        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await _bounded(call)

        assert remote.at_pgrep
        assert all(seen == (['7'], {'laptop': 'task:7'}) for seen in remote.at_pgrep)
        assert rig.unavailable_kinds() == [('laptop', 'transport')]
        _assert_unwound(rig, laptop_holders=[], local_holders=[])

        result = await _bounded(rig.call(attempt_id=2))

        assert result.passed
        sidecar = rig.sidecar(attempt_id=2)
        assert (sidecar['reason'], sidecar['runner']) == ('local_granted_first', 'local')
        assert remote.log.count('verify_task') == 1
        _assert_unwound(rig, laptop_holders=[], local_holders=[])

    async def test_row21_a_remote_verify_that_raises_leaves_the_turn_and_propagates(
        self, tmp_path: Path,
    ) -> None:
        rig = _rig(tmp_path)
        rig.remotes['laptop'].raises = RuntimeError('boom')

        with pytest.raises(RuntimeError, match='boom'):
            await _bounded(rig.call())

        assert rig.sidecar()['reason'] == 'dispatcher_error'
        assert rig.local.plans == []
        _assert_unwound(rig, laptop_holders=[], local_holders=[])


# ---------------------------------------------------------------------------
# Rows 26-30, 32, 33, 35, 36, 41, 42, 45: who gets the turn
# ---------------------------------------------------------------------------


@_REAL
class TestTheRace:
    async def test_row26_a_laptop_turn_freed_while_queued_withdraws_the_local_run(
        self, tmp_path: Path,
    ) -> None:
        rig = _rig(tmp_path, concurrent=True)
        rig.legs.block(_LINT)
        laptop_holder, local_holder = await rig.hold_both()
        call = rig.start()
        await _until(
            lambda: rig.legs.spawned(_LINT) and _waiting(rig.local_gate, '7'), watching=call,
        )
        [lint_pid] = rig.legs.pids(_LINT)

        await laptop_holder.leave()
        result = await _bounded(call)

        assert result.summary == 'remote verdict'
        assert not rig.legs.spawned(_TEST)
        assert not _alive(lint_pid)
        assert rig.local.cancels == [1]
        assert rig.plan_events() == []
        sidecar = rig.sidecar()
        assert (sidecar['reason'], sidecar['runner']) == ('remote', 'laptop')
        assert sidecar['remote_wait_secs'] > 0
        _assert_unwound(rig, laptop_holders=[], local_holders=['local-holder'])
        await local_holder.leave()

    async def test_row27_a_local_turn_granted_first_keeps_the_verify_local(
        self, tmp_path: Path,
    ) -> None:
        rig = _rig(tmp_path)
        rig.legs.block(_TEST)
        laptop_holder = rig.hold()
        await _bounded(laptop_holder.granted.wait())
        call = rig.start()
        await _until(lambda: rig.legs.spawned(_TEST), watching=call)
        next_waiter = rig.hold(task_id='next')
        await _until(lambda: _waiters(rig.laptop) == ['7', 'next'])

        await laptop_holder.leave()
        await _bounded(next_waiter.granted.wait())

        assert _waiters(rig.laptop) == []
        assert not call.done()
        rig.legs.release(_TEST)
        result = await _bounded(call)
        assert result.passed
        assert rig.sidecar()['reason'] == 'local_granted_first'
        assert rig.local.cancels == []
        assert rig.remotes['laptop'].log == []
        await next_waiter.leave()

    async def test_row29_the_older_arrival_gets_the_one_laptop_turn(self, tmp_path: Path) -> None:
        rig = _rig(tmp_path)
        remote = rig.remotes['laptop']
        remote.hold = True
        laptop_holder, local_holder = await rig.hold_both()
        older = rig.start(task_id='7')
        await rig.queued_on_both(older, '7')
        younger = rig.start(task_id='8')
        await rig.queued_on_both(younger, '8')

        await laptop_holder.leave()
        await _bounded(remote.started.wait())

        assert rig.pool.holders() == {'laptop': 'task:7'}
        assert [spec.task_id for spec in remote.specs()] == ['7']
        assert _waiters(rig.laptop) == ['8']
        remote.released.set()
        await _bounded(older)
        await _bounded(younger)
        assert [spec.task_id for spec in remote.specs()] == ['7', '8']
        assert rig.sidecar('7')['reason'] == rig.sidecar('8')['reason'] == 'remote'
        await local_holder.leave()

    async def test_row30_runner_unavailable_after_withdrawal_reenters_with_the_original_arrival(
        self, tmp_path: Path,
    ) -> None:
        rig = _rig(tmp_path)
        rig.legs.block(_TEST)
        remote = rig.remotes['laptop']
        remote.exit = _SSH_DOWN
        laptop_holder, local_holder = await rig.hold_both()
        call = rig.start()
        await rig.queued_on_both(call)
        younger = rig.hold_local(task_id='younger')
        await _until(lambda: _waiting(rig.local_gate, 'younger'))

        await laptop_holder.leave()
        await _until(lambda: len(rig.local.plans) == 2 and _waiting(rig.local_gate, '7'), watching=call)
        await local_holder.leave()
        await _until(lambda: rig.legs.spawned(_TEST) or younger.granted.is_set(), watching=call)

        assert rig.legs.spawned(_TEST)
        assert not younger.granted.is_set()
        rig.legs.release(_TEST)
        result = await _bounded(call)
        assert result.passed
        [plan, reentry] = rig.local.plans
        assert reentry is plan
        assert rig.local.cancels == [1]
        assert rig.unavailable_kinds() == [('laptop', 'transport')]
        assert remote.log.count('verify_task') == 1
        sidecar = rig.sidecar()
        assert (sidecar['reason'], sidecar['runner'], sidecar['remote_wait_secs']) == (
            'runner_unavailable', 'local', None,
        )
        assert rig.pool.holders() == {}
        assert _holding(rig.laptop) == _waiters(rig.laptop) == []
        await younger.leave()

    async def test_row32_a_readmitted_host_offers_its_oldest_waiter_a_turn(
        self, tmp_path: Path,
    ) -> None:
        rig = _rig(tmp_path)
        rig.quarantine.add('laptop')
        local_holder = rig.hold_local()
        await _bounded(local_holder.granted.wait())
        call = rig.start()
        await rig.queued_on_both(call)
        younger = rig.hold(task_id='younger')
        await _until(lambda: _waiting(rig.laptop, 'younger'))
        await _until(lambda: rig.legs.spawned(_LINT), watching=call)

        rig.quarantine.discard('laptop')
        await asyncio.sleep(0.05)
        assert _waiters(rig.laptop) == ['7', 'younger']
        assert _holding(rig.laptop) == []
        rig.pool.note_readmitted('laptop', rig.config)
        result = await _bounded(call)

        assert result.summary == 'remote verdict'
        assert rig.sidecar()['reason'] == 'remote'
        assert rig.local.cancels == [1]
        await _bounded(younger.granted.wait())
        await younger.leave()
        await local_holder.leave()

    async def test_row33_a_reload_to_local_mid_wait_leaves_the_later_grant(
        self, tmp_path: Path,
    ) -> None:
        rig = _rig(tmp_path)
        laptop_holder, local_holder = await rig.hold_both()
        call = rig.start()
        await rig.queued_on_both(call)
        next_waiter = rig.hold(task_id='next')
        await _until(lambda: _waiting(rig.laptop, 'next'))

        rig.policy = TaskVerifyPolicy()
        await laptop_holder.leave()
        await _bounded(next_waiter.granted.wait())

        assert _waiters(rig.laptop) == []
        assert not call.done()
        await local_holder.leave()
        result = await _bounded(call)
        assert result.passed
        assert rig.sidecar()['reason'] == 'local_granted_first'
        assert rig.local.cancels == []
        assert rig.remotes['laptop'].log == []

        rig.policy = _REMOTE_ALL
        await next_waiter.leave()
        again = await _bounded(rig.call(attempt_id=2))
        assert again.summary == 'remote verdict'
        assert rig.sidecar(attempt_id=2)['reason'] == 'remote'

    async def test_row35_a_withdrawal_mid_gather_leaves_no_process_and_a_clean_tree(
        self, tmp_path: Path,
    ) -> None:
        rig = _rig(tmp_path, concurrent=True)
        rig.legs.block(_LINT)
        laptop_holder, local_holder = await rig.hold_both()
        call = rig.start()
        await _until(
            lambda: rig.legs.spawned(_LINT) and rig.legs.spawned(_TYPE)
            and _waiting(rig.local_gate, '7'),
            watching=call,
        )
        [type_pid] = rig.legs.pids(_TYPE)
        await _until(lambda: not _alive(type_pid))

        await laptop_holder.leave()
        result = await _bounded(call)

        assert result.summary == 'remote verdict'
        assert not rig.legs.spawned(_TEST)
        assert [pid for pid in rig.legs.all_pids() if _alive(pid)] == []
        assert _git(rig.repo, 'status', '--porcelain') == ''
        await local_holder.leave()

    @pytest.mark.parametrize(
        'modules', [(VERIFY_MODULE,), (VERIFY_MODULE, _SECOND_MODULE)], ids=['one', 'two'],
    )
    async def test_row35_the_verdict_waits_for_a_leg_that_only_sigkill_ends(
        self, tmp_path: Path, modules: tuple[str, ...],
    ) -> None:
        rig = _rig(tmp_path, concurrent=True, modules=modules, lints_ignoring_term=(VERIFY_MODULE,))
        lints = [f'lint-{prefix}' for prefix in modules]
        types = [f'type-{prefix}' for prefix in modules]
        rig.legs.block(*lints)
        laptop_holder, local_holder = await rig.hold_both()
        call = rig.start()
        await _until(
            lambda: all(rig.legs.spawned(leg) for leg in (*lints, *types))
            and _waiting(rig.local_gate, '7'),
            watching=call,
        )
        await _until(lambda: not any(_alive(pid) for leg in types for pid in rig.legs.pids(leg)))
        [stubborn_lint] = rig.legs.pids(_LINT)
        alive_at_return: list[bool] = []
        call.add_done_callback(lambda _call: alive_at_return.append(_alive(stubborn_lint)))

        await laptop_holder.leave()
        result = await _bounded(call)

        assert result.summary == 'remote verdict'
        assert alive_at_return == [False]
        assert [pid for pid in rig.legs.all_pids() if _alive(pid)] == []
        await local_holder.leave()

    async def test_row36_a_reentry_reruns_the_cold_provision_and_keeps_its_arrival(
        self, tmp_path: Path,
    ) -> None:
        rig = _rig(tmp_path, provision='waits')
        rig.legs.block(_PROVISION)
        rig.remotes['laptop'].exit = _SSH_DOWN

        def provisioned() -> int:
            return len(rig.legs.pids(_PROVISION))

        laptop_holder, local_holder = await rig.hold_both()
        call = rig.start()
        await _until(lambda: provisioned() == 1 and _waiting(rig.laptop, '7'), watching=call)

        await laptop_holder.leave()
        await _until(lambda: provisioned() == 2, watching=call)
        [plan, reentry] = rig.local.plans
        assert plan is not None and reentry is plan
        rig.legs.release(_PROVISION)
        await _until(lambda: _waiting(rig.local_gate, '7'), watching=call)

        [waiter] = [w for w in rig.local_gate.snapshot().waiters if w.request.task_id == '7']
        assert (waiter.request.plan_id, waiter.request.plan_arrival) == (plan.plan_id, plan.arrival)
        await local_holder.leave()
        result = await _bounded(call)
        assert result.passed
        assert rig.sidecar()['reason'] == 'runner_unavailable'
        assert rig.local.cancels == [1]

    async def test_row41_a_cancelled_oldest_call_hands_the_turn_to_the_next_call(
        self, tmp_path: Path,
    ) -> None:
        rig = _rig(tmp_path)
        laptop_holder, local_holder = await rig.hold_both()
        oldest = rig.start(task_id='7')
        await rig.queued_on_both(oldest, '7')
        following = rig.start(task_id='8')
        await rig.queued_on_both(following, '8')

        oldest.cancel()
        with pytest.raises(asyncio.CancelledError):
            await _bounded(oldest)
        assert _waiters(rig.laptop) == ['8']
        await laptop_holder.leave()
        result = await _bounded(following)

        assert result.summary == 'remote verdict'
        assert [spec.task_id for spec in rig.remotes['laptop'].specs()] == ['8']
        await local_holder.leave()

    async def test_row42_a_local_verdict_while_queued_on_the_laptop_is_returned(
        self, tmp_path: Path,
    ) -> None:
        rig = _rig(tmp_path)
        laptop_holder = rig.hold()
        await _bounded(laptop_holder.granted.wait())

        result = await _bounded(rig.call())

        assert result.passed
        assert rig.sidecar()['reason'] == 'local_granted_first'
        assert _waiters(rig.laptop) == []
        assert _holding(rig.laptop) == ['holder']
        assert rig.remotes['laptop'].log == []
        await laptop_holder.leave()

    async def test_row45_the_spec_carries_the_head_at_the_turn(self, tmp_path: Path) -> None:
        rig = _rig(tmp_path)
        laptop_holder, local_holder = await rig.hold_both()
        call = rig.start()
        await rig.queued_on_both(call)
        (rig.repo / VERIFY_MODULE / 'test_x.py').write_text('# moved after arrival\n')
        _git(rig.repo, 'commit', '-q', '-am', 'move HEAD after arrival')
        new_head = _git(rig.repo, 'rev-parse', 'HEAD')

        await laptop_holder.leave()
        result = await _bounded(call)

        assert result.summary == 'remote verdict'
        [spec] = rig.remotes['laptop'].specs()
        assert spec.tip_sha == new_head != rig.tip
        assert rig.sidecar()['tip_sha'] == new_head
        await local_holder.leave()


# ---------------------------------------------------------------------------
# Row 31: cancellation at every stage of the race
# ---------------------------------------------------------------------------


class _Workflow:
    """One dispatcher call as a workflow body under the real ``CancellationScope``,
    cancelled soft (its cancel event) or hard (its task cancelled)."""

    def __init__(self, rig: _Rig, kind: Literal['soft', 'hard']) -> None:
        self._kind = kind
        self._cancel_event = asyncio.Event()
        scope = CancellationScope(self._cancel_event, (), cancel_grace_secs=5.0)
        self.task = asyncio.create_task(scope.supervise(rig.call()))

    def cancel(self) -> None:
        if self._kind == 'soft':
            self._cancel_event.set()
        else:
            self.task.cancel()

    async def raised(self) -> None:
        with pytest.raises(WorkflowCancelled) as cancelled:
            await _bounded(self.task)
        assert cancelled.value.kind == self._kind


@_REAL
@pytest.mark.parametrize('kind', ['soft', 'hard'])
class TestCancellation:
    async def test_row31_cancelled_while_queued_on_both_hosts(
        self, tmp_path: Path, kind: Literal['soft', 'hard'],
    ) -> None:
        rig = _rig(tmp_path)
        laptop_holder, local_holder = await rig.hold_both()
        before = _pending_tasks()
        call = _Workflow(rig, kind)
        await rig.queued_on_both(call.task)

        call.cancel()
        await call.raised()

        _assert_unwound(rig, laptop_holders=['holder'], local_holders=['local-holder'])
        await _assert_cancelled_call_unwound(rig, before)
        await laptop_holder.leave()
        await local_holder.leave()

    async def test_row31_cancelled_while_the_local_run_is_being_withdrawn(
        self, tmp_path: Path, kind: Literal['soft', 'hard'],
    ) -> None:
        rig = _rig(tmp_path, provision='ignores_term')
        rig.legs.block(_PROVISION)
        laptop_holder = rig.hold()
        await _bounded(laptop_holder.granted.wait())
        before = _pending_tasks()
        call = _Workflow(rig, kind)
        await _until(
            lambda: rig.legs.spawned(_PROVISION) and _waiting(rig.laptop, '7'), watching=call.task,
        )
        [provision_pid] = rig.legs.pids(_PROVISION)
        await laptop_holder.leave()
        await _until(lambda: rig.pool.holders() == {'laptop': 'task:7'}, watching=call.task)
        assert not any(task is not None and task.done() for task in rig.local.tasks)

        call.cancel()
        await asyncio.sleep(0.1)
        rig.legs.release(_PROVISION)
        await call.raised()

        _assert_unwound(rig, laptop_holders=[], local_holders=[])
        await _assert_cancelled_call_unwound(rig, before)
        assert not _alive(provision_pid)
        assert rig.remotes['laptop'].log == []

    async def test_row31_cancelled_while_the_scope_is_rederived(
        self, tmp_path: Path, kind: Literal['soft', 'hard'],
    ) -> None:
        rig = _rig(tmp_path)
        laptop_holder, local_holder = await rig.hold_both()
        before = _pending_tasks()
        call = _Workflow(rig, kind)
        await rig.queued_on_both(call.task)
        rig.git_ops.pause = asyncio.Event()
        await laptop_holder.leave()
        await _bounded(rig.git_ops.paused.wait())
        assert rig.pool.holders() == {'laptop': 'task:7'}

        call.cancel()
        await call.raised()

        _assert_unwound(rig, laptop_holders=[], local_holders=['local-holder'])
        await _assert_cancelled_call_unwound(rig, before)
        assert rig.remotes['laptop'].log == []
        await local_holder.leave()


# ---------------------------------------------------------------------------
# Row 34: admission inactive (the conftest seam neutralises it; no marker)
# ---------------------------------------------------------------------------


class TestAdmissionInactive:
    async def test_row34_an_ungated_test_leg_latches_the_plan_so_the_attempt_leaves(
        self, tmp_path: Path,
    ) -> None:
        rig = _rig(tmp_path)
        assert admission_active(rig.config) is False
        rig.legs.block(_TEST)
        laptop_holder = rig.hold()
        await _bounded(laptop_holder.granted.wait())
        call = rig.start()
        await _until(lambda: rig.legs.spawned(_TEST), watching=call)
        [plan] = rig.local.plans
        assert plan is not None and plan.has_been_granted()

        await laptop_holder.leave()
        await _until(lambda: _holding(rig.laptop) == _waiters(rig.laptop) == [])

        [test_pid] = rig.legs.pids(_TEST)
        assert _alive(test_pid)
        rig.legs.release(_TEST)
        result = await _bounded(call)
        assert result.passed
        assert rig.sidecar()['reason'] == 'local_granted_first'
        assert rig.local.cancels == []
        assert rig.remotes['laptop'].log == []


# ---------------------------------------------------------------------------
# Rows 37, 38, 44, 49: a plan already granted (or already done) stays local
# ---------------------------------------------------------------------------


@_REAL
class TestAGrantedPlanStaysLocal:
    async def test_row37_a_granted_module_keeps_the_plan_local_while_another_queues(
        self, tmp_path: Path,
    ) -> None:
        rig = _rig(tmp_path, modules=(VERIFY_MODULE, _SECOND_MODULE))
        tests = [f'test-{prefix}' for prefix in (VERIFY_MODULE, _SECOND_MODULE)]
        rig.legs.block(*tests)
        laptop_holder = rig.hold()
        await _bounded(laptop_holder.granted.wait())
        call = rig.start()
        await _until(lambda: any(rig.legs.spawned(t) for t in tests), watching=call)
        [first] = [t for t in tests if rig.legs.spawned(t)]
        [second] = [t for t in tests if t != first]
        older = rig.hold_local(task_id='older', arrival=0.0)
        await _until(lambda: _waiting(rig.local_gate, 'older'))
        rig.legs.release(first)
        await _bounded(older.granted.wait())
        await _until(lambda: _waiting(rig.local_gate, '7'), watching=call)

        await laptop_holder.leave()
        await _until(lambda: _holding(rig.laptop) == _waiters(rig.laptop) == [])

        assert not call.done()
        assert not rig.legs.spawned(second)
        rig.legs.release(second)
        await older.leave()
        result = await _bounded(call)
        assert result.passed
        assert rig.sidecar()['reason'] == 'local_granted_first'
        assert rig.local.cancels == []
        assert rig.remotes['laptop'].log == []

    async def test_row37_a_timeout_retry_between_turns_keeps_the_plan_local(
        self, tmp_path: Path,
    ) -> None:
        rig = _rig(tmp_path, test_leg='times_out_first', module_timeout=1.0)
        laptop_holder = rig.hold()
        await _bounded(laptop_holder.granted.wait())
        call = rig.start()
        await _until(lambda: rig.legs.spawned(_TEST), watching=call)
        older = rig.hold_local(task_id='older', arrival=0.0)
        await _bounded(older.granted.wait())
        await _until(lambda: _waiting(rig.local_gate, '7'), watching=call)
        assert len(rig.legs.pids(_TEST)) == 1

        await laptop_holder.leave()
        await _until(lambda: _holding(rig.laptop) == _waiters(rig.laptop) == [])

        assert not call.done()
        await older.leave()
        result = await _bounded(call)
        assert result.passed
        assert len(rig.legs.pids(_TEST)) == 2
        assert rig.sidecar()['reason'] == 'local_granted_first'
        assert rig.local.cancels == []
        assert rig.remotes['laptop'].log == []

    async def test_row38_a_leg_granted_but_waiting_for_its_flock_keeps_the_plan_local(
        self, tmp_path: Path,
    ) -> None:
        rig = _rig(tmp_path)
        slots_dir = Path(rig.config.verify_admission_slots_dir)
        slots_dir.mkdir()
        laptop_holder = rig.hold()
        await _bounded(laptop_holder.granted.wait())

        with acquire_task_slot('task', slots_dir=slots_dir, n=1, wait=False) as held:
            assert held is True
            call = rig.start()
            await _until(lambda: _holding(rig.local_gate) == ['7'], watching=call)
            await laptop_holder.leave()
            await _until(lambda: _holding(rig.laptop) == _waiters(rig.laptop) == [])
            await asyncio.sleep(0.2)
            assert not rig.legs.spawned(_TEST)
            assert not call.done()

        result = await _bounded(call)
        assert result.passed
        assert rig.legs.spawned(_TEST)
        assert rig.sidecar()['reason'] == 'local_granted_first'
        assert rig.local.cancels == []
        assert rig.remotes['laptop'].log == []

    async def test_row44_admission_off_at_the_test_leg_and_on_at_the_turn_keeps_the_leg(
        self, tmp_path: Path,
    ) -> None:
        rig = _rig(tmp_path)
        rig.legs.block(_TEST)
        rig.config.verify_admission_enabled = False
        laptop_holder = rig.hold()
        await _bounded(laptop_holder.granted.wait())
        call = rig.start()
        await _until(lambda: rig.legs.spawned(_TEST), watching=call)
        rig.config.verify_admission_enabled = True
        assert admission_active(rig.config) is True
        [plan] = rig.local.plans
        assert plan is not None and plan.has_been_granted()

        await laptop_holder.leave()
        await _until(lambda: _holding(rig.laptop) == _waiters(rig.laptop) == [])

        [test_pid] = rig.legs.pids(_TEST)
        assert _alive(test_pid)
        assert _holding(rig.local_gate) == []
        rig.legs.release(_TEST)
        result = await _bounded(call)
        assert result.passed
        assert rig.sidecar()['reason'] == 'local_granted_first'
        assert rig.local.cancels == []
        assert rig.remotes['laptop'].log == []

    async def test_row49_a_plan_with_no_gated_leg_done_in_the_grant_tick_returns_its_verdict(
        self, tmp_path: Path,
    ) -> None:
        rig = _rig(tmp_path, test_leg='none')
        rig.quarantine.add('laptop')

        def readmit_as_the_local_run_returns() -> None:
            assert _waiters(rig.laptop) == ['7']
            rig.quarantine.discard('laptop')
            rig.pool.note_readmitted('laptop', rig.config)
            assert _holding(rig.laptop) == ['7']

        rig.local.on_return = readmit_as_the_local_run_returns
        result = await _bounded(rig.call())

        assert result.passed
        assert rig.sidecar()['reason'] == 'local_granted_first'
        assert rig.local.cancels == []
        assert rig.remotes['laptop'].log == []
        _assert_unwound(rig, laptop_holders=[], local_holders=[])


# ---------------------------------------------------------------------------
# Rows 39, 40, 43, 50, 51: after the claim, and around it
# ---------------------------------------------------------------------------


@_REAL
class TestAroundTheClaim:
    @pytest.mark.parametrize(('knobs', 'verdict', 'reason', 'parity_field'), [
        pytest.param(
            {'cross_check_every_n': 1}, _result(True), 'cross_check', 'cross_check',
            id='cross_check',
        ),
        pytest.param(
            {'red_confirms_locally': True}, _result(False, category='test_failure'),
            'red_confirm', 'red_confirmed_locally', id='red_confirm',
        ),
    ])
    async def test_row39_a_parity_run_takes_a_fresh_plan_behind_a_younger_waiter(
        self, tmp_path: Path, knobs: dict[str, Any], verdict: VerifyResult, reason: str,
        parity_field: str,
    ) -> None:
        rig = _rig(tmp_path, policy=dataclasses.replace(_REMOTE_ALL, **knobs))
        remote = rig.remotes['laptop']
        remote.verdict, remote.hold = verdict, True
        local_holder = rig.hold_local()
        await _bounded(local_holder.granted.wait())
        call = rig.start()
        await _bounded(remote.started.wait())
        younger = rig.hold_local(task_id='younger')
        await _until(lambda: _waiting(rig.local_gate, 'younger'))

        remote.released.set()
        await _until(lambda: _waiters(rig.local_gate) == ['younger', '7'], watching=call)
        await local_holder.leave()
        await _bounded(younger.granted.wait())
        await asyncio.sleep(0.1)

        assert not rig.legs.spawned(_TEST)
        assert _waiters(rig.local_gate) == ['7']
        await younger.leave()
        result = await _bounded(call)
        assert result.passed
        assert rig.local.plans == [None]
        sidecar = rig.sidecar()
        assert sidecar['reason'] == reason
        assert sidecar[parity_field] is not None
        assert sidecar['remote_wait_secs'] is not None

    async def test_row40_an_attempt_that_raises_before_withdrawal_leaves_its_place(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.WARNING, logger=_DISPATCH_LOGGER)
        rig = _rig(tmp_path)
        laptop_holder, local_holder = await rig.hold_both()
        call = rig.start()
        await rig.queued_on_both(call)
        next_waiter = rig.hold(task_id='next')
        await _until(lambda: _waiting(rig.laptop, 'next'))

        def failures() -> list[logging.LogRecord]:
            return [
                r for r in caplog.records
                if r.name == _DISPATCH_LOGGER and r.exc_info is not None
                and 'policy unreadable' in str(r.exc_info[1])
            ]

        rig.policy_raises = RuntimeError('policy unreadable')
        await laptop_holder.leave()
        await _bounded(next_waiter.granted.wait())
        await _until(lambda: bool(failures()), watching=call)

        [record] = failures()
        assert record.levelno == logging.WARNING
        assert _waiters(rig.laptop) == []
        assert _holding(rig.laptop) == ['next']
        assert not call.done()
        await local_holder.leave()
        result = await _bounded(call)
        assert result.passed
        assert rig.sidecar()['reason'] == 'local_granted_first'
        assert rig.local.cancels == []
        await next_waiter.leave()

    async def test_row40_an_attempt_failing_beside_the_claimers_return_is_a_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The desk's grant lands in the same step that ends the laptop's claimed
        turn, so both attempts are done when the race next looks at them."""
        caplog.set_level(logging.DEBUG, logger=_DISPATCH_LOGGER)
        rig = _rig(tmp_path, hosts=('laptop', 'desk'))
        rig.quarantine.add('desk')
        rig.remotes['laptop'].verdict = _result(
            False, category=sorted(INFRA_TRANSIENT_CATEGORIES)[0], summary='host out of space',
        )

        def readmit_the_desk_to_an_unreadable_policy(benched: str) -> None:
            assert benched == 'laptop'
            rig.quarantine.discard('desk')
            rig.policy_raises = RuntimeError('policy unreadable')
            rig.pool.note_readmitted('desk', rig.config)

        rig.bench_hooks.append(readmit_the_desk_to_an_unreadable_policy)
        result = await _bounded(rig.call())

        assert result.passed
        assert rig.sidecar()['reason'] == 'infra_transient'
        [record] = [
            r for r in caplog.records
            if r.name == _DISPATCH_LOGGER and r.exc_info is not None
            and 'policy unreadable' in str(r.exc_info[1])
        ]
        assert (record.levelno, record.args) == (logging.WARNING, ('desk',))
        assert rig.remotes['desk'].log == []
        _assert_unwound(rig, laptop_holders=[], local_holders=[])

    @pytest.mark.parametrize(('arrival_knobs', 'grant_knobs', 'reason'), [
        pytest.param({'cross_check_every_n': 1}, {}, 'remote', id='reloaded_off'),
        pytest.param({}, {'cross_check_every_n': 1}, 'cross_check', id='reloaded_on'),
    ])
    async def test_row39_the_claiming_grant_reads_the_parity_knobs_that_settle_the_verdict(
        self, tmp_path: Path, arrival_knobs: dict[str, Any], grant_knobs: dict[str, Any],
        reason: str,
    ) -> None:
        rig = _rig(tmp_path, policy=dataclasses.replace(_REMOTE_ALL, **arrival_knobs))
        laptop_holder, local_holder = await rig.hold_both()
        call = rig.start()
        await rig.queued_on_both(call)

        rig.policy = dataclasses.replace(_REMOTE_ALL, **grant_knobs)
        await laptop_holder.leave()
        await _until(lambda: rig.pool.holders() == {'laptop': 'task:7'}, watching=call)
        await local_holder.leave()
        result = await _bounded(call)

        assert result.passed
        sidecar = rig.sidecar()
        assert sidecar['reason'] == reason
        assert (sidecar['cross_check'] is not None) == (reason == 'cross_check')
        assert rig.remotes['laptop'].log.count('verify_task') == 1

    @pytest.mark.parametrize(('fallback', 'kinds', 'verifies'), [
        pytest.param('dirty_tree', [], 0, id='dirty_tree'),
        pytest.param(
            'infra_transient', [('laptop', 'no_verdict')], 1, id='infra_transient',
        ),
    ])
    async def test_row43_a_fallback_after_withdrawal_reenters_with_the_original_plan(
        self, tmp_path: Path, fallback: str, kinds: list[tuple[str, str]], verifies: int,
    ) -> None:
        rig = _rig(tmp_path)
        remote = rig.remotes['laptop']
        remote.verdict = _result(
            False, category=sorted(INFRA_TRANSIENT_CATEGORIES)[0], summary='host out of space',
        )
        laptop_holder, local_holder = await rig.hold_both()
        call = rig.start()
        await rig.queued_on_both(call)
        if fallback == 'dirty_tree':
            (rig.repo / 'scratch.txt').write_text('untracked\n')

        await laptop_holder.leave()
        await _until(lambda: len(rig.local.plans) == 2, watching=call)
        await local_holder.leave()
        result = await _bounded(call)

        assert result.passed
        [plan, reentry] = rig.local.plans
        assert reentry is plan
        assert rig.local.cancels == [1]
        assert rig.sidecar()['reason'] == fallback
        assert rig.unavailable_kinds() == kinds
        assert remote.log.count('verify_task') == verifies
        assert rig.pool.holders() == {}
        assert _waiters(rig.laptop) == []

    async def test_row43_a_non_runner_unavailable_error_leaves_the_turn_and_reraises(
        self, tmp_path: Path,
    ) -> None:
        rig = _rig(tmp_path)
        rig.remotes['laptop'].raises = RuntimeError('boom')
        laptop_holder, local_holder = await rig.hold_both()
        call = rig.start()
        await rig.queued_on_both(call)
        next_waiter = rig.hold(task_id='next')
        await _until(lambda: _waiting(rig.laptop, 'next'))

        await laptop_holder.leave()
        with pytest.raises(RuntimeError, match='boom'):
            await _bounded(call)

        assert next_waiter.granted.is_set()
        assert rig.pool.holders() == {}
        assert rig.sidecar()['reason'] == 'dispatcher_error'
        assert len(rig.local.plans) == 1
        assert rig.local.cancels == [1]
        assert _waiters(rig.local_gate) == []
        await next_waiter.leave()
        await local_holder.leave()

    async def test_row50_a_red_confirm_starts_after_the_turn_passed_on(
        self, tmp_path: Path,
    ) -> None:
        rig = _rig(tmp_path, policy=dataclasses.replace(_REMOTE_ALL, red_confirms_locally=True))
        remote = rig.remotes['laptop']
        remote.verdict, remote.hold = _result(False, category='test_failure'), True
        rig.local.on_start = lambda: (
            dict(rig.pool.holders()), _holding(rig.laptop), rig.legs.all_pids(),
        )
        call = rig.start()
        await _bounded(remote.started.wait())
        next_waiter = rig.hold(task_id='next')
        await _until(lambda: _waiting(rig.laptop, 'next'))

        remote.released.set()
        result = await _bounded(call)

        assert result.passed
        assert rig.local.at_start == [({}, ['next'], [])]
        assert rig.sidecar()['reason'] == 'red_confirm'
        await next_waiter.leave()

    async def test_row51_the_first_of_two_free_hosts_claims_and_the_other_leaves(
        self, tmp_path: Path,
    ) -> None:
        rig = _rig(tmp_path, hosts=('laptop', 'desktop'))

        result = await _bounded(rig.call())

        assert result.summary == 'remote verdict'
        assert rig.sidecar()['runner'] == 'laptop'
        assert rig.remotes['desktop'].log == []
        assert rig.pool.holders() == {}
        for host in ('laptop', 'desktop'):
            assert _holding(rig.pool.gate(host)) == _waiters(rig.pool.gate(host)) == []

    async def test_row51_an_attempt_still_queued_on_another_host_is_reaped_with_the_call(
        self, tmp_path: Path,
    ) -> None:
        rig = _rig(tmp_path, hosts=('laptop', 'desktop'))
        desktop_holder = rig.hold('desktop')
        await _bounded(desktop_holder.granted.wait())

        result = await _bounded(rig.call())

        assert result.summary == 'remote verdict'
        assert rig.sidecar()['runner'] == 'laptop'
        assert _waiters(rig.pool.gate('desktop')) == []
        assert _holding(rig.pool.gate('desktop')) == ['holder']
        assert rig.remotes['desktop'].log == []
        await desktop_holder.leave()
