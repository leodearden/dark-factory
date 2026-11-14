"""A real Harness whose remote verify host ``laptop`` is the real verify-task CLI on a fixture repo.

The integration gate (task 6672, ``test_task_verify_integration.py``) needs
every seam joined: the Harness's own HostPool, TaskVerifyDispatcher and merge
worker, over a remote host that really runs ``orchestrator verify-task``.

* :class:`LaptopHost` is the one transport behind every RemoteRunner.  It
  answers the gate publish and ensure, the verify-ref pushes, ``ssh true``,
  pgrep and cancel-verify itself, and routes each verify by an
  :data:`Answer` per subcommand.
* :class:`InProcessCli` runs the real CLI in this process; :class:`ChildCli`
  runs it as a real child process, for the one row that kills it.
* :func:`open_scene` builds the Harness over a seeded repo and a task
  worktree, with ``orchestrator.verify_hosts.RemoteRunner`` (the single
  construction site of both the pool's and the worker's runners) replaced by
  a recording constructor that wires in the transport.
* :func:`open_workflow_scene` builds the same Harness over the workflow
  fixture repo and gives it a real TaskWorkflow (:class:`WorkflowScene`),
  built as ``Harness._run_slot`` builds one.

The laptop runs on the same repository as the workstation, under its own
``git.worktree_dir`` (:data:`LAPTOP_WORKTREE_DIR`), so a leg can tell from its
cwd which host ran it (:class:`Legs`).

This module imports no merge-lane module (``scripts/merge_lane_metrics.py::
imports_lane_module``): it reaches GitOps and the merge worker only through
Harness attributes and the worker's public methods.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol
from unittest.mock import MagicMock, patch

import pytest
import yaml
from _gate_fixtures import TEST_DEPLOYED_SHA, ensure_reply, is_ensure, is_gate_publish
from _git_fixtures import RepoSeed, seed_repo
from _merge_lane_fakes import branch_merge_request
from _orch_helpers import CANCEL_SCOPE_BARRIER_TIMEOUT, wait_responsive
from _workflow_helpers import E2E_REPO_SEED, PLAN, AgentStub, FakeBriefing, FakeMcp, FakeScheduler
from click.testing import CliRunner
from escalation.queue import EscalationQueue

import orchestrator.cli as cli_module
from orchestrator.config import (
    GitConfig,
    ModuleConfig,
    OrchestratorConfig,
    SandboxConfig,
    VerifyRunnerConfig,
    apply_reload,
)
from orchestrator.event_store import EventStore, EventType
from orchestrator.harness import Harness
from orchestrator.remote_gate import GateIdentity
from orchestrator.scheduler import TaskAssignment
from orchestrator.task_verify_dispatch import TaskVerifyDispatcher
from orchestrator.task_verify_spec import TaskVerifySpec, build_task_verify_spec
from orchestrator.verify import VerifyResult, admission_policy
from orchestrator.verify_admission_gate import (
    AdmissionGate,
    AdmissionPlan,
    AdmissionPolicy,
    gate_for,
)
from orchestrator.verify_cancel import kill_process_tree, read_worktree_path
from orchestrator.verify_hosts import HostPool
from orchestrator.verify_runner import RemoteRunner, result_to_json
from orchestrator.workflow import TaskWorkflow, build_workflow
from orchestrator.workflow_types import TerminalReport

LAPTOP = 'laptop'
LAPTOP_WORKTREE_DIR = '.laptop-worktrees'
MODULE = 'mod'
SECOND_MODULE = 'mod2'
TASK_ID = '7'
GREEN = VerifyResult(
    passed=True, test_output='', lint_output='', type_output='', summary='laptop verdict',
)
PGID_SENTINEL = 2 ** 22
"""What an in-process verify-task records as its pgid: pid_max, so no process group bears it."""

_GIT = GitConfig(
    main_branch='main', branch_prefix='task/', remote='origin',
    worktree_dir='.worktrees', push_after_advance=False,
)
SSH_CONNECT_FAILURE = (
    'ssh: connect to host laptop.invalid port 22: Connection timed out\n'
    'fatal: Could not read from remote repository.\n'
)


# ---------------------------------------------------------------------------
# What the laptop answers to a verify
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cli:
    """Run the real CLI in this process."""


@dataclass(frozen=True)
class Subprocess:
    """Run the real CLI as a child process with its own process group."""


@dataclass(frozen=True)
class Exit:
    """Exit *rc* with these streams, as the ssh would relay them."""

    rc: int
    stdout: str = ''
    stderr: str = ''


@dataclass(frozen=True)
class Verdict:
    """Print *result* and exit 0."""

    result: VerifyResult


@dataclass(frozen=True)
class Raises:
    """The local ssh call itself raises *error*."""

    error: Exception


Settled = Cli | Exit | Verdict | Raises


@dataclass(frozen=True)
class Hold:
    """Wait until released (then answer *then*) or until the verify is unwound.

    However the hold ends, the transport logs ``hold_ended`` for its request.
    """

    then: Settled = Verdict(GREEN)


Answer = Settled | Subprocess | Hold


@dataclass(frozen=True)
class Ensure:
    """ensure-gate's answer: rc 0 with a reply naming *identity* (default: this process's)."""

    rc: int = 0
    built: bool = False
    identity: GateIdentity | None = None

    def reply(self) -> tuple[int, str, str]:
        if self.rc != 0:
            return self.rc, '', f'ensure-gate failed with rc {self.rc}'
        return 0, ensure_reply(built=self.built, identity=self.identity), ''


@dataclass(frozen=True)
class Call:
    """One transport call: its label, the request it names (if any), and its argv."""

    label: str
    request_id: str | None
    argv: tuple[str, ...]


# ---------------------------------------------------------------------------
# The CLI hosts
# ---------------------------------------------------------------------------

_CLI_LOCK = threading.Lock()


def patch_cli_seams(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep an in-process verify-task out of setsid and off this process's stdin."""
    monkeypatch.setattr(cli_module, 'start_own_process_group', lambda: PGID_SENTINEL)
    monkeypatch.setattr(cli_module, 'start_stdin_watchdog', lambda *a, **k: MagicMock())


class InProcessCli:
    """The real ``orchestrator`` CLI, invoked in this process on a worker thread.

    One invocation at a time: for its whole run CliRunner swaps ``sys.std*``
    and ``os.environ`` process-wide.  The swapped environment names the laptop
    yaml (``ORCH_CONFIG_PATH``, which any OrchestratorConfig built meanwhile
    also reads) and drops the variables that would override it; both are
    restored afterwards.

    An in-process run cannot be cancelled.  The real cancel-verify must never
    meet one: its pgid file holds :data:`PGID_SENTINEL` (with
    :func:`patch_cli_seams`), and a real cancel would SIGKILL whatever process
    group bore that number.  Nor can the transport answer for it, since the
    worker thread would run on holding the swap; :class:`LaptopHost` refuses
    such a cancel instead.
    """

    def __init__(self, laptop_yaml: Path) -> None:
        self._env: dict[str, str | None] = {
            'ORCH_CONFIG_PATH': str(laptop_yaml),
            'ORCH_PROJECT_ROOT': None,
            'ORCH_VERIFY_COLD_PREPROVISION_COMMAND': None,
        }

    async def invoke(self, args: Sequence[str]) -> tuple[int, str, str]:
        return await asyncio.to_thread(self._invoke, list(args))

    def _invoke(self, args: list[str]) -> tuple[int, str, str]:
        with _CLI_LOCK:
            result = CliRunner().invoke(cli_module.main, args, env=self._env)
        stderr = result.stderr
        if result.exception is not None and not isinstance(result.exception, SystemExit):
            stderr += f'\n{result.exception!r}'
        return result.exit_code, result.stdout, stderr


def child_env() -> dict[str, str]:
    """A verify-task child's env: this worktree's source, a hermetic cold verify, a slow watchdog."""
    env = dict(os.environ)
    src = str(Path(__file__).resolve().parent.parent / 'src')
    existing = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = f'{src}:{existing}' if existing else src
    env.pop('ORCH_PROJECT_ROOT', None)
    env.pop('ORCH_VERIFY_COLD_PREPROVISION_COMMAND', None)
    env['ORCH_WATCHDOG_HEARTBEAT_TIMEOUT_SECS'] = '600'
    return env


class ChildCli:
    """The real CLI as a child process whose stdin stays open; every child is recorded."""

    def __init__(self) -> None:
        self.children: list[asyncio.subprocess.Process] = []

    async def invoke(self, args: Sequence[str]) -> tuple[int, str, str]:
        child = await asyncio.create_subprocess_exec(
            sys.executable, '-c', 'from orchestrator.cli import main; main()', *args,
            env=child_env(), stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        self.children.append(child)
        assert child.stdout is not None and child.stderr is not None
        stdout, stderr = await asyncio.gather(child.stdout.read(), child.stderr.read())
        return await child.wait(), stdout.decode(), stderr.decode()

    async def close(self) -> None:
        """Kill only the recorded children, by their own pid (each one's pgid after setsid)."""
        for child in self.children:
            if child.returncode is None:
                kill_process_tree(child.pid, backstop_pgid=child.pid)
            if child.stdin is not None:
                child.stdin.close()
            with contextlib.suppress(ProcessLookupError, TimeoutError):
                async with asyncio.timeout(10):
                    await child.wait()


# ---------------------------------------------------------------------------
# The transport
# ---------------------------------------------------------------------------


def _orchestrator_args(remote_cmd: str) -> list[str]:
    """The CLI arguments of a remote command sent through the gate binary."""
    words = shlex.split(remote_cmd)
    if words[:1] != ['env'] or not words[1].startswith('ORCH_GATE_BIN=') or words[2] != 'orchestrator':
        raise AssertionError(f'not a gate command: {remote_cmd!r}')
    return words[3:]


def _option(args: Sequence[str], name: str) -> str | None:
    return args[args.index(name) + 1] if name in args else None


def _push_label(argv: Sequence[str]) -> tuple[str, str | None]:
    if '--delete' in argv:
        ref = argv[-1]
        return 'delete_ref', ref.rsplit('/', 1)[-1]
    refspec = argv[-1]
    if ':refs/merge-verify/' in refspec:
        return 'push', refspec.rsplit('/', 1)[-1]
    return 'main_push', None


class LaptopHost:
    """The laptop as every RemoteRunner sees it, through ``run`` and ``ssh_run``.

    Every call is logged as a :class:`Call`; a verify's raw exit code and
    stdout are kept by request id.  ``answers`` maps each verify subcommand to
    its :data:`Answer`.
    """

    def __init__(self, cli: InProcessCli, children: ChildCli) -> None:
        self._cli = cli
        self._children = children
        self.answers: dict[str, Answer] = {'verify-task': Cli(), 'verify-merge': Verdict(GREEN)}
        self.ensure = Ensure()
        self.publish_connect_fails = False
        self.healthy = True
        self.cancel_rc = 0
        self.pgrep_rc = 1
        self.on_pgrep: Callable[[], object] | None = None
        self.pgrep_seen: list[object] = []
        self.calls: list[Call] = []
        self._next_id = 1
        self._outcomes: dict[str, tuple[int, str]] = {}
        self._held: dict[str, asyncio.Future[Settled]] = {}
        self._in_child: set[str] = set()
        self._in_process: set[str] = set()

    # -- the runner's seams -------------------------------------------------

    def next_request_id(self) -> str:
        request_id = self.upcoming_request_id
        self._next_id += 1
        return request_id

    async def run(self, argv: list[str], *, cwd: Any = None) -> tuple[int, str, str]:
        if is_gate_publish(argv):
            self._log('publish', None, argv)
            return (128, '', SSH_CONNECT_FAILURE) if self.publish_connect_fails else (0, '', '')
        if is_ensure(argv):
            self._log('ensure', None, argv)
            return self.ensure.reply()
        if argv[:1] == ['git']:
            return await self._git(argv, cwd)
        return await self._ssh(argv)

    async def ssh_run(self, argv: list[str], *, cwd: Any = None) -> tuple[int, str, str]:
        args = _orchestrator_args(argv[-1])
        subcommand, request_id = args[0], _option(args, '--request-id')
        assert request_id is not None, args
        self._log(subcommand.replace('-', '_'), request_id, argv)
        rc, stdout, stderr = await self._answer(self.answers[subcommand], args, request_id)
        self._outcomes[request_id] = (rc, stdout)
        return rc, stdout, stderr

    # -- what a test reads and drives ----------------------------------------

    @property
    def upcoming_request_id(self) -> str:
        """The id the next runner dispatch will be given."""
        return f'req{self._next_id}'

    def labels(self) -> list[str]:
        return [call.label for call in self.calls]

    def requests(self, label: str) -> list[str | None]:
        return [call.request_id for call in self.calls if call.label == label]

    def stdout_of(self, request_id: str) -> str:
        return self._outcomes[request_id][1]

    def exit_code_of(self, request_id: str) -> int:
        return self._outcomes[request_id][0]

    async def until_verifying(self, subcommand: str, *, nth: int = 1) -> str:
        """The request id of the *nth* *subcommand* verify, once its ssh has started."""
        label = subcommand.replace('-', '_')
        await until(lambda: len(self.requests(label)) >= nth, what=f'{subcommand} #{nth} to start')
        request_id = self.requests(label)[nth - 1]
        assert request_id is not None
        return request_id

    def release(self, request_id: str, answer: Settled | None = None) -> None:
        """End *request_id*'s hold with *answer*, else with its :class:`Hold`'s own."""
        held = self._held.get(request_id)
        if held is not None and not held.done():
            held.set_result(_RELEASED if answer is None else answer)

    # -- internals -------------------------------------------------------------

    def _log(self, label: str, request_id: str | None, argv: Sequence[str]) -> None:
        self.calls.append(Call(label, request_id, tuple(argv)))

    async def _git(self, argv: list[str], cwd: Any) -> tuple[int, str, str]:
        if 'rev-parse' in argv:
            self._log('rev_parse', None, argv)
            done = subprocess.run(argv, cwd=cwd, capture_output=True, text=True)
            return done.returncode, done.stdout, done.stderr
        if 'push' in argv:
            label, request_id = _push_label(argv)
            self._log(label, request_id, argv)
            return 0, '', ''
        self._log('unscripted', None, argv)
        return 127, '', 'unscripted git call'

    async def _ssh(self, argv: list[str]) -> tuple[int, str, str]:
        remote_cmd = argv[-1]
        if remote_cmd == 'true':
            self._log('health', None, argv)
            return (0, '', '') if self.healthy else (255, '', SSH_CONNECT_FAILURE)
        if remote_cmd.startswith('pgrep'):
            self._log('pgrep', None, argv)
            if self.on_pgrep is not None:
                self.pgrep_seen.append(self.on_pgrep())
            return self.pgrep_rc, '', ''
        if 'reap-gates' in remote_cmd:
            self._log('reap', None, argv)
            return 0, '', ''
        if 'cancel-verify' in remote_cmd:
            return await self._cancel(argv, _orchestrator_args(remote_cmd))
        self._log('unscripted', None, argv)
        return 127, '', 'unscripted ssh call'

    async def _cancel(self, argv: list[str], args: list[str]) -> tuple[int, str, str]:
        request_id = _option(args, '--request-id')
        self._log('cancel_verify', request_id, argv)
        if request_id in self._in_process:
            raise AssertionError(
                f'in-process verify-task {request_id} cannot be cancelled; '
                'answer it with Hold() or Subprocess()',
            )
        if request_id in self._in_child:
            return await self._cli.invoke(args)
        if self.cancel_rc == 0 and request_id is not None:
            self.release(request_id, Exit(137, '', 'Killed'))
        return self.cancel_rc, '', ''

    async def _answer(self, answer: Answer, args: list[str], request_id: str) -> tuple[int, str, str]:
        match answer:
            case Cli():
                # Not cleared when the await is cancelled: the worker thread runs on.
                self._in_process.add(request_id)
                settled = await self._cli.invoke(args)
                self._in_process.discard(request_id)
                return settled
            case Subprocess():
                self._in_child.add(request_id)
                return await self._children.invoke(args)
            case Exit(rc=rc, stdout=stdout, stderr=stderr):
                return rc, stdout, stderr
            case Verdict(result=result):
                return 0, result_to_json(result), ''
            case Raises(error=error):
                raise error
            case Hold(then=then):
                return await self._answer(await self._hold(request_id, then), args, request_id)

    async def _hold(self, request_id: str, then: Settled) -> Settled:
        held: asyncio.Future[Settled] = asyncio.get_running_loop().create_future()
        self._held[request_id] = held
        try:
            released = await held
        finally:
            self._log('hold_ended', request_id, ())
        return then if released is _RELEASED else released


_RELEASED: Settled = Exit(-1)
"""The release that keeps the :class:`Hold`'s own answer."""


class RecordingRunners:
    """A RemoteRunner constructor that wires in one :class:`LaptopHost` and records each runner."""

    def __init__(self, host: LaptopHost) -> None:
        self._host = host
        self.built: list[RemoteRunner] = []

    def __call__(self, *args: Any, **kwargs: Any) -> RemoteRunner:
        runner = RemoteRunner(
            *args, **kwargs,
            run=self._host.run, ssh_run=self._host.ssh_run, id_factory=self._host.next_request_id,
        )
        self.built.append(runner)
        return runner


# ---------------------------------------------------------------------------
# Legs: commands that log where they ran
# ---------------------------------------------------------------------------

Where = Literal['local', 'laptop', 'other']


class Legs:
    """Leg commands logging their pid under ``(label, where)``, ``where`` read from the cwd.

    ``local`` is the task checkout, ``laptop`` the laptop's worktree base, and
    ``other`` anything else.  A leg waits while its ``(label, where)`` is
    blocked and fails while it is failing; by default it passes at once.
    """

    def __init__(self, root: Path, *, task_checkout: Path, laptop_base: Path) -> None:
        self.root = root
        root.mkdir()
        self._task = str(task_checkout.resolve())
        self._laptop = str(laptop_base.resolve())

    def command(self, label: str, *, ignore_term: bool = False) -> str:
        trap = "trap '' TERM; " if ignore_term else ''
        task, laptop = shlex.quote(self._task), shlex.quote(self._laptop)
        base = shlex.quote(str(self.root / label))
        return (
            f'{trap}case "$(pwd -P)" in {task}|{task}/*) w=local;; {laptop}/*) w=laptop;; '
            f'*) w=other;; esac; echo $$ >> {base}."$w".spawns; '
            f'while [ -e {base}."$w".blocked ]; do sleep 0.02; done; '
            f'[ ! -e {base}."$w".fails ]'
        )

    def block(self, label: str, where: Where = 'local') -> None:
        self._path(label, where, 'blocked').touch()

    def release(self, label: str, where: Where = 'local') -> None:
        self._path(label, where, 'blocked').unlink(missing_ok=True)

    def fail(self, label: str, where: Where) -> None:
        self._path(label, where, 'fails').touch()

    def pids(self, label: str, where: Where = 'local') -> list[int]:
        path = self._path(label, where, 'spawns')
        return [int(pid) for pid in path.read_text().split()] if path.exists() else []

    def spawned(self, label: str, where: Where = 'local') -> bool:
        return bool(self.pids(label, where))

    def spawned_at(self, where: Where) -> list[str]:
        """The labels that logged a pid at *where* (a leg killed in its first step logs none)."""
        labels = (p.name.rsplit('.', 2)[0] for p in self.root.glob(f'*.{where}.spawns'))
        return sorted(label for label in labels if self.spawned(label, where))

    def _path(self, label: str, where: Where, kind: str) -> Path:
        return self.root / f'{label}.{where}.{kind}'


# ---------------------------------------------------------------------------
# The scene
# ---------------------------------------------------------------------------


def laptop_yaml(tmp_path: Path, repo: Path, provision: str) -> Path:
    """The laptop's own config: the fixture repo, its own worktree base, admission off."""
    path = tmp_path / 'laptop.yaml'
    path.write_text(yaml.safe_dump({
        'project_root': str(repo),
        'git': {'main_branch': 'main', 'worktree_dir': LAPTOP_WORKTREE_DIR},
        'verify_cold_preprovision_command': provision,
        'verify_admission_enabled': False,
        'verify_env_probes': [],
        'verify_clock_stop_enabled': False,
        'verify_use_cgroup_scope': False,
        'verify_runners': [],
    }))
    return path


def _workstation_config(
    tmp_path: Path, repo: Path, laptop: Path, legs: Legs, provision: str,
    overrides: dict[str, Any],
) -> OrchestratorConfig:
    """The workstation's config; its workspace-wide commands are the legs ``test``, ``lint``, ``type``."""
    fields: dict[str, Any] = {
        'project_root': repo,
        'git': _GIT,
        'test_command': legs.command('test'),
        'lint_command': legs.command('lint'),
        'type_check_command': legs.command('type'),
        'verify_runners': [VerifyRunnerConfig(
            name=LAPTOP, ssh_host='laptop.invalid', git_remote='laptop-remote',
            df_checkout_path='/home/laptop/dark-factory', config_path=str(laptop),
        )],
        'verify_cold_preprovision_command': provision,
        'verify_env': {'IOTA_GATE': '1'},
        'verify_env_probes': [],
        'verify_clock_stop_enabled': False,
        'verify_admission_slots_dir': str(tmp_path / 'slots'),
        'verify_admission_order': 'arrival',
        'verify_admission_task_slots': 1,
        'verify_host_unreachable_escalate_after_n': 1,
        'verify_host_reprobe_interval_s': 1.0,
        'verify_host_policy': 'prefer_remote',
        'escalate_preexisting_main_break': False,
    }
    return OrchestratorConfig(**{**fields, **overrides})


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ['git', '-C', str(repo), *args], check=True, capture_output=True, text=True,
    ).stdout.strip()


async def until(predicate: Callable[[], bool], *, what: str) -> None:
    """Poll *predicate* until it holds, failing by *what* on the responsive-time deadline."""
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.02)

    await wait_responsive(poll(), label=what)


@dataclass
class Turn:
    """A turn the test holds on one gate, from outside any verify."""

    task: asyncio.Task[None]
    granted: asyncio.Event
    release: asyncio.Event

    async def leave(self) -> None:
        self.release.set()
        async with asyncio.timeout(30):
            await self.task



class PendingMerge(Protocol):
    """What ``Scene.end_merge`` reads off an enqueued merge request."""

    @property
    def request_id(self) -> str: ...

    @property
    def result(self) -> asyncio.Future[Any]: ...


def _queue(
    gate: AdmissionGate, policy: Callable[[], AdmissionPolicy], *, task_id: str, arrival: float | None = None,
) -> Turn:
    plan = AdmissionPlan.begin(role='task', task_id=task_id)
    if arrival is not None:
        plan = dataclasses.replace(plan, arrival=arrival)
    granted, release = asyncio.Event(), asyncio.Event()

    async def hold() -> None:
        async with gate.admit(plan.request(module=None, scope_kind=None), policy):
            granted.set()
            await release.wait()

    return Turn(asyncio.create_task(hold()), granted, release)


def task_ids(views: Sequence[Any]) -> list[str | None]:
    return [view.request.task_id for view in views]


@dataclass
class Scene:
    """The Harness, its task worktree, the laptop transport and the legs of one test."""

    tmp_path: Path
    harness: Harness
    repo: Path
    worktree: Path
    laptop: LaptopHost
    runners: RecordingRunners
    children: ChildCli
    legs: Legs
    archive_root: Path
    laptop_worktree_base: Path
    task_id: str = TASK_ID
    task_files: tuple[str, ...] = (f'{MODULE}/test_x.py',)
    _worker_started: bool = field(default=False, init=False)

    @property
    def config(self) -> OrchestratorConfig:
        return self.harness.config

    @property
    def tip(self) -> str:
        """The task worktree's HEAD."""
        return _git(self.worktree, 'rev-parse', 'HEAD')

    @property
    def pool(self) -> HostPool:
        return self.harness._host_pool

    @property
    def dispatcher(self) -> TaskVerifyDispatcher:
        return self.harness._task_verify_dispatcher

    @property
    def worker(self):  # return type inferred: no lane type name imported
        assert self.harness._merge_worker is not None, 'start_worker() first'
        return self.harness._merge_worker

    @property
    def laptop_gate(self) -> AdmissionGate:
        return self.pool.gate(LAPTOP)

    @property
    def local_gate(self) -> AdmissionGate:
        return gate_for(self.config.verify_admission_slots_dir)

    def module(
        self, prefix: str = MODULE, *, test: str | None = None,
        verify_env: dict[str, str] | None = None,
    ) -> ModuleConfig:
        """*prefix*'s module: *test* as its only command, else the legs' test, lint and type."""
        if test is not None:
            return ModuleConfig(prefix, test_command=test, verify_env=verify_env)
        return ModuleConfig(
            prefix,
            test_command=self.legs.command(f'test-{prefix}'),
            lint_command=self.legs.command(f'lint-{prefix}'),
            type_check_command=self.legs.command(f'type-{prefix}'),
            verify_env=verify_env,
        )

    def spec(self, modules: list[ModuleConfig], *, attempt_id: int = 1) -> TaskVerifySpec:
        return build_task_verify_spec(
            self.config, modules, self.task_files, force_workspace=False,
            attempt_id=attempt_id, task_id=self.task_id, tip_sha=self.tip,
        )

    async def call(
        self, modules: list[ModuleConfig], *, attempt_id: int = 1, force_workspace: bool = False,
    ) -> VerifyResult:
        """The harness dispatcher's verify of the task worktree."""
        return await self.dispatcher(
            self.worktree, self.config, modules, task_files=list(self.task_files),
            attempt_id=attempt_id, task_id=self.task_id, archive_root=self.archive_root,
            force_workspace=force_workspace, event_store=self.harness.event_store,
        )

    def start(self, modules: list[ModuleConfig], **kwargs: Any) -> asyncio.Task[VerifyResult]:
        return asyncio.create_task(self.call(modules, **kwargs))

    def sidecars(self, attempt_id: int = 1) -> list[dict[str, Any]]:
        paths = sorted(
            (self.archive_root / self.task_id).glob(f'attempt-{attempt_id}.dispatch-*.json'),
        )
        return [json.loads(path.read_text()) for path in paths]

    def sidecar(self, attempt_id: int = 1) -> dict[str, Any]:
        [only] = self.sidecars(attempt_id)
        return only

    def archived_files(self) -> list[str]:
        root = self.archive_root / self.task_id
        return sorted(str(p.relative_to(root)) for p in root.rglob('*') if p.is_file())

    def hold(self, *, task_id: str = 'holder') -> Turn:
        return _queue(
            self.laptop_gate, lambda: self.pool.policy(LAPTOP, self.config), task_id=task_id,
        )

    def hold_local(self, *, task_id: str = 'local-holder', arrival: float | None = None) -> Turn:
        return _queue(
            self.local_gate, lambda: admission_policy(self.config), task_id=task_id,
            arrival=arrival,
        )

    def turns(self) -> dict[str, Any]:
        """Who holds and who waits on each gate, and the pool's hold records."""
        laptop, local = self.laptop_gate.snapshot(), self.local_gate.snapshot()
        return {
            'pool_holders': self.pool.holders(),
            'laptop_waiters': task_ids(laptop.waiters),
            'laptop_holders': task_ids(laptop.holders),
            'local_waiters': task_ids(local.waiters),
            'local_holders': task_ids(local.holders),
        }

    def unwound(self, *, laptop_holders: Sequence[str] = (), local_holders: Sequence[str] = ()) -> bool:
        """No call holds or waits for a turn; only the test's own turns are held."""
        return self.turns() == {
            'pool_holders': {},
            'laptop_waiters': [],
            'laptop_holders': list(laptop_holders),
            'local_waiters': [],
            'local_holders': list(local_holders),
        }

    def laptop_merge_entries(self) -> list[str]:
        base = self.laptop_worktree_base
        return sorted(p.name for p in base.iterdir() if p.name.startswith('_merge-')) if base.exists() else []

    async def until(self, predicate: Callable[[], bool], *, what: str) -> None:
        await until(predicate, what=what)

    async def until_recorded_cold_worktree(
        self, pgf: Path, *, watching: asyncio.Task[Any], timeout: float = 60.0,
    ) -> Path:
        """The worktree a verify-task recorded beside *pgf*, once it holds its cold ``.task/``."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            recorded = read_worktree_path(pgf)
            if recorded is not None and (recorded / '.task').is_dir():
                return recorded
            if watching.done():
                watching.result()
                raise AssertionError('the call ended before verify-task recorded its worktree')
            await asyncio.sleep(0.05)
        raise AssertionError(f'verify-task never recorded a cold worktree beside {pgf}')

    def reload(self, **keys: Any) -> dict[str, Any]:
        """Hot-reload *keys* into the live harness config, as ``reload_config`` does; its result."""
        outcome = apply_reload(self.config, self.config.model_copy(update=keys))
        assert outcome['reloaded'], outcome
        return outcome

    async def start_worker(self) -> None:
        """Start the harness's own merge worker, with the escalation queue run() would give it."""
        self.harness._escalation_queue = EscalationQueue(self.tmp_path / 'escalations')
        await self.harness._start_merge_worker()
        await self.harness._launch_merge_worker_loop()
        self._worker_started = True

    async def stop_worker(self) -> None:
        if self._worker_started:
            self._worker_started = False
            await self.harness._stop_merge_worker()

    def laptop_row(self) -> dict[str, Any]:
        """The laptop's row of the merge worker's snapshot ``hosts`` block."""
        return {row['name']: row for row in self.worker.snapshot()['hosts']}[LAPTOP]

    async def merge(self, task_id: str):  # return type inferred: no lane type name imported
        """Enqueue, on the harness's merge queue, a merge of task *task_id*'s new branch."""
        request = await branch_merge_request(
            self.harness.git_ops, self.config, task_id, filename=f'{task_id}.txt',
        )
        await self.harness._merge_queue.put(request)
        return request

    async def end_merge(self, request: PendingMerge, *, settle_secs: float = 30.0) -> str:
        """The merge's outcome status once it lands or fails, else ``'cancelled'`` by the worker."""
        try:
            outcome = await asyncio.wait_for(asyncio.shield(request.result), settle_secs)
        except (TimeoutError, asyncio.CancelledError):
            if not request.result.done():
                self.worker.cancel_request(request.request_id)
            await asyncio.wait({request.result}, timeout=settle_secs)
            return 'cancelled'
        return outcome.status

    def escalations(self, category: str) -> list[Any]:
        """The pending escalations of *category* on the harness's queue."""
        queue = self.harness._escalation_queue
        return [] if queue is None else [e for e in queue.get_pending() if e.category == category]

    def events(self, event_type: EventType) -> list[dict[str, Any]]:
        store = self.harness.event_store
        assert store is not None
        return store.fetch_events_by_type(event_type)


_IGNORED = ('.gitignore', f'.task/\n{LAPTOP_WORKTREE_DIR}/\n')
_MODULE_FILES = tuple((f'{prefix}/test_x.py', '# placeholder\n') for prefix in (MODULE, SECOND_MODULE))


def _repo_path(tmp_path: Path) -> Path:
    return tmp_path / 'repo'


def _harness(config: OrchestratorConfig) -> Harness:
    with patch('orchestrator.harness.McpLifecycle'), \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        return Harness(config, deployed_sha=TEST_DEPLOYED_SHA)


@contextlib.asynccontextmanager
async def _open_harness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    files: Sequence[tuple[str, str]],
    task_id: str,
    archive_root: Path,
    provision: Callable[[Legs], str],
    laptop_provision: str,
    config_overrides: dict[str, Any],
) -> AsyncIterator[Scene]:
    """A Harness over a repo seeded with *files*; task *task_id*'s worktree is not created yet.

    *provision* gives the workstation's cold pre-provision command, which may be a leg.
    """
    monkeypatch.setenv('ORCH_FLEET_DIR', str(tmp_path / 'fleet'))
    patch_cli_seams(monkeypatch)
    repo = seed_repo(_repo_path(tmp_path), RepoSeed(files=(*files, _IGNORED), message='initial commit'))
    task_checkout, laptop_base = repo / _GIT.worktree_dir / task_id, repo / LAPTOP_WORKTREE_DIR
    laptop = laptop_yaml(tmp_path, repo, laptop_provision)
    legs = Legs(tmp_path / 'legs', task_checkout=task_checkout, laptop_base=laptop_base)
    children = ChildCli()
    host = LaptopHost(InProcessCli(laptop), children)
    runners = RecordingRunners(host)
    monkeypatch.setattr('orchestrator.verify_hosts.RemoteRunner', runners)
    harness = _harness(
        _workstation_config(tmp_path, repo, laptop, legs, provision(legs), config_overrides),
    )
    harness.event_store = EventStore(tmp_path / 'runs.db', run_id='iota')
    scene = Scene(
        tmp_path=tmp_path, harness=harness, repo=repo, worktree=task_checkout, laptop=host,
        runners=runners, children=children, legs=legs, archive_root=archive_root,
        laptop_worktree_base=laptop_base, task_id=task_id,
    )
    async with contextlib.AsyncExitStack() as teardown:
        teardown.push_async_callback(children.close)
        teardown.push_async_callback(scene.stop_worker)
        yield scene


@contextlib.asynccontextmanager
async def open_scene(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    provision: str = '',
    laptop_provision: str = '',
    **config_overrides: Any,
) -> AsyncIterator[Scene]:
    """A Harness over a seeded repo whose task ``7`` has one commit on ``mod/test_x.py``.

    *provision* is the workstation's cold pre-provision command (what a spec
    carries); *laptop_provision* the laptop yaml's own.
    """
    async with _open_harness(
        tmp_path, monkeypatch, files=_MODULE_FILES, task_id=TASK_ID,
        archive_root=tmp_path / 'verify-logs', provision=lambda _legs: provision,
        laptop_provision=laptop_provision, config_overrides=config_overrides,
    ) as scene:
        worktree = (await scene.harness.git_ops.create_worktree(TASK_ID)).path
        assert worktree == scene.worktree, worktree
        (worktree / MODULE / 'test_x.py').write_text('# changed on the task branch\n')
        await scene.harness.git_ops.commit(worktree, 'task change')
        yield scene


# ---------------------------------------------------------------------------
# A real TaskWorkflow over the scene
# ---------------------------------------------------------------------------

WORKFLOW_TASK_ID: str = PLAN['task_id']
PROVISION_LEG = 'provision'
_WORKFLOW_LOOP: dict[str, Any] = {
    'max_concurrent_tasks': 1,
    'max_execute_iterations': 5,
    'max_verify_attempts': 3,
    'max_review_cycles': 2,
    'judge_after_each_iteration': False,
    'max_amendment_rounds': 1,
    'sandbox': SandboxConfig(enabled=False),
}


def _assignment() -> TaskAssignment:
    return TaskAssignment(
        task_id=WORKFLOW_TASK_ID,
        task={
            'id': WORKFLOW_TASK_ID, 'title': PLAN['title'],
            'description': 'Add a farewell(name) function to lib.py with tests',
            'status': 'pending', 'metadata': {'files': ['lib']}, 'dependencies': [],
        },
        modules=['lib'],
    )


def pending_tasks() -> set[asyncio.Task[Any]]:
    current = asyncio.current_task()
    return {t for t in asyncio.all_tasks() if t is not current and not t.done()}


class PausingMergeBase:
    """GitOps' ``merge_base_with_main``; once armed, each call waits to be resumed first."""

    def __init__(self, real: Callable[[str], Awaitable[str | None]]) -> None:
        self._real = real
        self._armed = False
        self._resumed = asyncio.Event()
        self.paused = asyncio.Event()

    def arm(self) -> None:
        self._armed = True

    def resume(self) -> None:
        self._resumed.set()

    async def __call__(self, ref: str) -> str | None:
        if self._armed:
            self.paused.set()
            await self._resumed.wait()
        return await self._real(ref)


@dataclass
class WorkflowScene:
    """A real TaskWorkflow on *scene*'s Harness, built as ``Harness._run_slot`` builds it.

    Its merge queue is never drained, so a green run parks at MERGE until the
    scene's cancel event (soft) or a cancel of its task (hard) ends it.
    """

    scene: Scene
    agents: AgentStub
    merge_base: PausingMergeBase
    scheduler: FakeScheduler = field(default_factory=FakeScheduler)
    merge_queue: asyncio.Queue[Any] = field(default_factory=asyncio.Queue)
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    workflow: TaskWorkflow | None = field(default=None, init=False)
    baseline: frozenset[asyncio.Task[Any]] = field(default=frozenset(), init=False)
    _run: asyncio.Task[TerminalReport] | None = field(default=None, init=False)

    def start(self, *, with_dispatcher: bool = True) -> asyncio.Task[TerminalReport]:
        """Build the workflow (given the harness's dispatcher, unless not) and start its run()."""
        harness = self.scene.harness
        self.workflow = build_workflow(
            assignment=_assignment(), config=harness.config, git_ops=harness.git_ops,
            scheduler=self.scheduler, briefing=FakeBriefing(), mcp=FakeMcp(),
            merge_queue=self.merge_queue, event_store=harness.event_store,
            cancel_event=self.cancel_event,
            task_verify=harness._task_verify_dispatcher if with_dispatcher else None,
        )
        self.baseline = frozenset(pending_tasks())
        self._run = asyncio.create_task(self.workflow.run())
        return self._run

    @property
    def run(self) -> asyncio.Task[TerminalReport]:
        assert self._run is not None, 'start() first'
        return self._run

    async def until(self, predicate: Callable[[], bool], *, what: str) -> None:
        """Poll *predicate* while run() is still running; its early end fails *what*."""
        run = self.run

        def ready() -> bool:
            if run.done():
                ended = 'cancelled' if run.cancelled() else repr(run.result())
                raise AssertionError(f'run() ended ({ended}) before {what}')
            return predicate()

        await until(ready, what=what)

    async def until_queued_on_both(self) -> None:
        turns = self.scene.turns
        await self.until(
            lambda: self.scene.task_id in turns()['laptop_waiters']
            and self.scene.task_id in turns()['local_waiters'],
            what='the verify to queue on both gates',
        )

    async def until_parked_at_merge(self) -> None:
        await self.until(lambda: not self.merge_queue.empty(), what='run() to park at MERGE')

    def verify_rows(self) -> list[dict[str, Any]]:
        return [row['data'] for row in self.scene.events(EventType.workflow_verify)]

    def soft_cancel(self) -> None:
        self.cancel_event.set()

    def hard_cancel(self) -> None:
        self.run.cancel()

    async def report(self) -> TerminalReport:
        """run()'s TerminalReport; a CancelledError escaping run() fails here."""
        async with asyncio.timeout(CANCEL_SCOPE_BARRIER_TIMEOUT):
            return await self.run

    async def until_no_task_outlives_the_run(self) -> None:
        """Every task started since :meth:`start` has ended (the poll runs in a task of its own)."""
        caller = asyncio.current_task()
        await until(
            lambda: not pending_tasks() - self.baseline - {caller}, what='every run task to end',
        )

    async def close(self) -> None:
        self.merge_base.resume()
        if self._run is not None and not self._run.done():
            self._run.cancel()
            await asyncio.wait({self._run}, timeout=CANCEL_SCOPE_BARRIER_TIMEOUT)


@contextlib.asynccontextmanager
async def open_workflow_scene(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, term_ignoring_provision: bool = False,
    **config_overrides: Any,
) -> AsyncIterator[WorkflowScene]:
    """A WorkflowScene over a repo seeded with ``E2E_REPO_SEED``; the workflow makes task 42's worktree.

    Its agents are :class:`AgentStub` and its verify-logs root is where the
    workflow puts it, ``<project_root>/data/verify-logs``.  The worktree is
    cold (it holds ``.task/``), so *term_ignoring_provision* makes every cold
    verify run the :data:`PROVISION_LEG` leg, which traps SIGTERM.
    """
    def provision(legs: Legs) -> str:
        return legs.command(PROVISION_LEG, ignore_term=True) if term_ignoring_provision else ''

    async with _open_harness(
        tmp_path, monkeypatch, files=E2E_REPO_SEED.files, task_id=WORKFLOW_TASK_ID,
        archive_root=_repo_path(tmp_path) / 'data' / 'verify-logs', provision=provision,
        laptop_provision='', config_overrides={**_WORKFLOW_LOOP, **config_overrides},
    ) as scene:
        agents = AgentStub()
        monkeypatch.setattr('orchestrator.workflow.invoke_agent', agents.invoke_agent)
        git_ops = scene.harness.git_ops
        merge_base = PausingMergeBase(git_ops.merge_base_with_main)
        monkeypatch.setattr(git_ops, 'merge_base_with_main', merge_base)
        flow = WorkflowScene(scene, agents, merge_base)
        try:
            yield flow
        finally:
            await flow.close()
