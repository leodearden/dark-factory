"""Builds and steps one real offline-lane worker in a test.

``RedPathLane`` owns a real ``OfflineLaneWorker`` over a seeded tmp repo whose
``lane/`` fixture project is the command's cwd, a stub ``pytest`` on PATH that
plays back one scripted ``Run`` per launch, the production typed task client over a
stateful stand-in for fused-memory's task tools, a real escalation queue and an
injected clock.  The worker is stepped through its public surface only: ``on_post_merge``
then ``run`` as a background task, which ends when the worker reads main's head
once more than the drive allowed (``MeteredGitOps``).
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import functools
import json
import logging
import math
import os
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from _git_fixtures import RepoSeed, seed_repo
from _lane_episode_fakes import Clock
from _orch_helpers import (
    MEASURED_SPAWN_LATENCY_SECS,
    VERIFY_CLI_PER_TEST_TIMEOUT,
    git_env_with_ceiling,
    mcp_tool_envelope,
    required_timeout_secs,
)
from escalation.queue import EscalationQueue
from shared.task_statuses import TaskStatus

from orchestrator.chronic_flake import SchedulerChronicFlakeTaskClient
from orchestrator.config import GitConfig, LaneCommand, OrchestratorConfig
from orchestrator.git_ops import GitOps
from orchestrator.offline_lane import OfflineLaneWorker
from orchestrator.offline_lane_episode_state import CommandRecord, Episode, EpisodeStore
from orchestrator.offline_lane_filing import SPAWN_CONTEXT
from orchestrator.offline_lane_pass import PassOutcomeHook
from orchestrator.offline_lane_records import PassRecord, RunRecordStore
from orchestrator.scheduler import Scheduler

A = 'tests/test_a.py::test_one'
B = 'tests/test_b.py::test_two'
C = 'tests/test_c.py::test_three'
W = 'tests/test_a.py::test_param[x y]'
NINE = tuple(f'tests/test_n.py::test_{index}' for index in range(9))

FIXTURE_DIR = 'lane'
LANE_COMMAND = LaneCommand(name='lane-fixture', command='pytest -m integration tests', cwd=FIXTURE_DIR)
#: ``lane-suite`` is no tool the command parser knows, so the lane cannot narrow it and intersects instead.
OPAQUE_COMMAND = LaneCommand(name='lane-opaque', command='lane-suite tests', cwd=FIXTURE_DIR)

#: The junit resolver maps a report's classnames to node ids only through module files that exist under the cwd.
LANE_SEED = RepoSeed(
    files=(
        ('README.md', '# offline-lane red-path fixture\n'),
        *((f'{FIXTURE_DIR}/tests/test_{name}.py', '# a module the lane stub reports on\n') for name in 'abcn'),
    ),
    message='seed the lane fixture project',
)

ADVANCE_FILE = 'advance.txt'
POLL_INTERVAL_SECS = 0.01
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)

#: Launches one drive can make: the head read, the worktree list and its add or reset and
#: clean, the detection, the outage probe, the confirm and the engine's ancestry read.
SPAWNS_PER_DRIVE = 8
DRIVE_BOUND_SECS = math.ceil(SPAWNS_PER_DRIVE * MEASURED_SPAWN_LATENCY_SECS)
#: Launches before the first drive: the seed's template copy and its index refresh.
SETUP_SPAWNS = 2
#: Launches of the main commit an advance makes: add, commit, rev-parse.
COMMIT_SPAWNS = 3


def lane_test_timeout(drives: int, extra_spawns: int = 0) -> int:
    """A scenario's ``pytest.mark.timeout``: its drives' bounds and its out-of-bound launches, on the 60 s grid.

    Never below the verify budget, which keeps every marker out of the inversion band.
    """
    bill = required_timeout_secs(
        drives * DRIVE_BOUND_SECS, SETUP_SPAWNS + COMMIT_SPAWNS * drives + extra_spawns,
    )
    return max(VERIFY_CLI_PER_TEST_TIMEOUT, math.ceil(bill / 60) * 60)


@dataclass(frozen=True)
class Run:
    """One scripted sub-run of the stub: what it prints, reports in junit, and how it ends.

    ``exit`` defaults to 1 when any test failed or errored, else 0; ``signal`` kills
    the stub with that signal after its output is flushed.
    """

    failed: Mapping[str, str] = field(default_factory=dict)
    errors: Mapping[str, str] = field(default_factory=dict)
    passed: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()
    printed: tuple[str, ...] = ()
    junit: bool = True
    complete: bool = True
    exit: int | None = None
    signal: int | None = None

    def to_json(self) -> dict[str, Any]:
        red = bool(self.failed or self.errors)
        return {
            'failed': [list(item) for item in self.failed.items()],
            'errors': [list(item) for item in self.errors.items()],
            'passed': list(self.passed),
            'skipped': list(self.skipped),
            'printed': list(self.printed),
            'junit': self.junit,
            'complete': self.complete,
            'exit': int(red) if self.exit is None else self.exit,
            'signal': None if self.signal is None else int(self.signal),
        }


_STUB_BODY = r'''
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

RUNS = Path(CTL, 'runs.json')
INVOCATIONS = Path(CTL, 'invocations.jsonl')
GREEN = {
    'failed': [], 'errors': [], 'passed': [], 'skipped': [], 'printed': [],
    'junit': True, 'complete': True, 'exit': 0, 'signal': None,
}


def pop_run():
    queue = json.loads(RUNS.read_text()) if RUNS.exists() else []
    if not queue:
        return GREEN
    RUNS.write_text(json.dumps(queue[1:]))
    return queue[0]


def junit_path(argv):
    for index, arg in enumerate(argv):
        if arg.startswith('--junitxml='):
            return arg.partition('=')[2]
        if arg == '--junitxml' and index + 1 < len(argv):
            return argv[index + 1]
    return None


def testcase(suite, node):
    module, _, rest = node.partition('::')
    owner, _, name = rest.rpartition('::')
    classname = module.removesuffix('.py').replace('/', '.')
    if owner:
        classname += '.' + owner.replace('::', '.')
    return ET.SubElement(suite, 'testcase', classname=classname, name=name)


def write_junit(path, run):
    root = ET.Element('testsuites')
    suite = ET.SubElement(root, 'testsuite', name='pytest')
    for node, line in run['failed']:
        ET.SubElement(testcase(suite, node), 'failure', message=line)
    for node, line in run['errors']:
        ET.SubElement(testcase(suite, node), 'error', message=line)
    for node in run['passed']:
        testcase(suite, node)
    for node in run['skipped']:
        ET.SubElement(testcase(suite, node), 'skipped', type='pytest.skip', message='skipped')
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)


def summary(run):
    counts = (
        (len(run['failed']), 'failed', 'failed'),
        (len(run['passed']), 'passed', 'passed'),
        (len(run['skipped']), 'skipped', 'skipped'),
        (len(run['errors']), 'error', 'errors'),
    )
    parts = ', '.join(f'{n} {one if n == 1 else many}' for n, one, many in counts if n)
    return f'=== {parts or "no tests ran"} in 0.01s ==='


def reported(word, node, line):
    return f'{word} {node} - {line}' if line else f'{word} {node}'


def main():
    argv = sys.argv[1:]
    with INVOCATIONS.open('a') as log:
        log.write(json.dumps({'program': Path(sys.argv[0]).name, 'argv': argv, 'cwd': os.getcwd()}) + '\n')
    run = pop_run()
    for line in run['printed']:
        print(line)
    for node, line in run['failed']:
        print(reported('FAILED', node, line))
    for node, line in run['errors']:
        print(reported('ERROR', node, line))
    if run['complete']:
        print(summary(run))
    path = junit_path(argv)
    if run['junit'] and path:
        write_junit(path, run)
    sys.stdout.flush()
    if run['signal'] is not None:
        os.kill(os.getpid(), run['signal'])
    sys.exit(run['exit'])


main()
'''


class LaneStub:
    """The lane's test runner: one stdlib program on a bin dir, playing back scripted ``Run``s in order.

    Every launch logs its argv and cwd and pops the next run; with none left it is a
    green session that ran no tests.
    """

    def __init__(self, root: Path) -> None:
        self.bin_dir = root / 'bin'
        self.ctl = root / 'ctl'
        self.bin_dir.mkdir(parents=True)
        self.ctl.mkdir(parents=True)
        self.runs_path = self.ctl / 'runs.json'
        self.invocations_path = self.ctl / 'invocations.jsonl'
        for name in ('pytest', 'lane-suite'):
            self.install(name)

    def install(self, name: str) -> None:
        program = self.bin_dir / name
        program.write_text(f'#!{sys.executable} -I\nCTL = {str(self.ctl)!r}\n{_STUB_BODY}')
        program.chmod(0o755)

    def put_on_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Every lane launch passes ``os.environ``, so the stub shadows any real ``pytest``."""
        monkeypatch.setenv('PATH', f'{self.bin_dir}{os.pathsep}{os.environ.get("PATH", "")}')

    def script(self, *runs: Run) -> None:
        self.runs_path.write_text(json.dumps([*self.unconsumed(), *(run.to_json() for run in runs)]))

    def unconsumed(self) -> list[dict[str, Any]]:
        return json.loads(self.runs_path.read_text()) if self.runs_path.exists() else []

    def invocations(self) -> list[dict[str, Any]]:
        if not self.invocations_path.exists():
            return []
        return [json.loads(line) for line in self.invocations_path.read_text().splitlines()]


def effect_of(record: PassRecord) -> dict[str, Any]:
    """The effect block every completed pass record carries."""
    assert isinstance(record.effect, dict), f'pass {record.pass_id} has no effect block'
    return record.effect


async def git(repo: Path, *args: str) -> str:
    """Run git in *repo* with discovery fenced at *repo*; its stdout, stripped."""
    proc = await asyncio.create_subprocess_exec(
        'git', *args, cwd=repo, env=git_env_with_ceiling(repo),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f'git {" ".join(args)} failed in {repo}: {stderr.decode(errors="replace").strip()}')
    return stdout.decode().strip()


async def head(repo: Path) -> str:
    return await git(repo, 'rev-parse', 'HEAD')


async def commit(repo: Path, files: Mapping[str, str], message: str) -> str:
    """Commit *files* to *repo*'s checked-out main, staging only them, and return the new head.

    Never ``add -A``: the lane's own worktree lives untracked under ``.worktrees``.
    """
    for relpath, text in files.items():
        target = repo / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    await git(repo, 'add', '--', *files)
    await git(repo, 'commit', '-m', message)
    return await head(repo)


class MeteredGitOps(GitOps):
    """A ``GitOps`` whose head reads are metered: past the allowance, a read parks until cancelled.

    The worker reads main's head when a pass starts and again at every poll tick, so
    parking marks the point at which every read the drive allowed has been acted on.
    """

    def __init__(self, config: GitConfig, project_root: Path) -> None:
        super().__init__(config, project_root)
        self.reads_left = 0
        self.parked = asyncio.Event()

    def allow(self, reads: int) -> None:
        self.reads_left = reads
        self.parked = asyncio.Event()

    async def get_main_sha(self) -> str:
        if self.reads_left <= 0:
            self.parked.set()
            await asyncio.get_running_loop().create_future()
        self.reads_left -= 1
        return await super().get_main_sha()


class RunLoopFailures(logging.Handler):
    """Collects what the worker's run loop logs at ERROR, which it does only when a pass raised."""

    LOGGER = 'orchestrator.offline_lane'

    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.messages: list[str] = []
        self.raised = asyncio.Event()
        self.loop = asyncio.get_running_loop()

    def emit(self, record: logging.LogRecord) -> None:
        if record.name == self.LOGGER:
            self.messages.append(record.getMessage())
            self.loop.call_soon_threadsafe(self.raised.set)

    @contextlib.contextmanager
    def attached(self) -> Iterator[RunLoopFailures]:
        logger = logging.getLogger(self.LOGGER)
        logger.addHandler(self)
        try:
            yield self
        finally:
            logger.removeHandler(self)


async def first_set(events: Sequence[asyncio.Event], timeout: float, what: str) -> None:
    waiters = [asyncio.create_task(event.wait()) for event in events]
    try:
        done, _pending = await asyncio.wait(waiters, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for waiter in waiters:
            waiter.cancel()
    if not done:
        raise AssertionError(f'{what}: nothing happened within {timeout}s')


SUBMIT_RESPONSES_PATH = Path(__file__).resolve().parents[2] / 'shared' / 'tests' / 'fixtures' / 'submit_task_responses.json'
DEFLAKE_SPAWN_CONTEXT = 'flake_ledger_debt'


@functools.cache
def submit_responses() -> Mapping[str, Any]:
    """The scenario-24 cases: the submit_task payloads fused-memory's own tests assert it returns."""
    return json.loads(SUBMIT_RESPONSES_PATH.read_text())['cases']


def submit_payload(case: str, **fields: Any) -> dict[str, Any]:
    return {**submit_responses()[case]['payload'], **fields}


def metadata_holds(held: object, value: object) -> bool:
    return held == value or (isinstance(held, list) and value in held)


class FakeTaskStore:
    """fused-memory's task tools as the production typed client reaches them, over one in-memory task table.

    Every answer is the JSON-RPC body of a ``tools/call``; every ``submit_task`` answer is
    built from the scenario-24 fixture's payloads.  A submission collides with any
    non-cancelled task of the same title, as fused-memory's dedup does.
    """

    def __init__(self) -> None:
        self.tasks: dict[str, dict[str, Any]] = {}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.issued = 0

    async def call_tool(self, name: str, arguments: dict, *, timeout: float = 30) -> dict:
        self.calls.append((name, copy.deepcopy(arguments)))
        return mcp_tool_envelope(self.answer(name, arguments))

    def answer(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        match name:
            case 'submit_task':
                return self.submitted(arguments)
            case 'get_task':
                return self.got(str(arguments['id']))
            case 'get_statuses':
                return {'statuses': {i: self.tasks[i]['status'] for i in map(str, arguments['ids']) if i in self.tasks}}
            case 'find_tasks_by_metadata':
                return self.found(arguments)
            case 'commit_planning':
                return self.committed(str(arguments['task_ids']).split(','), arguments['target_status'])
            case _:
                raise NotImplementedError(f'the fake task store answers no {name!r}')

    def submitted(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if not arguments.get('planning_mode'):
            raise NotImplementedError('the fake task store files planning-mode submissions only')
        title = arguments['title']
        collided = next(
            (task for task in self.tasks.values() if task['title'] == title and task['status'] != TaskStatus.CANCELLED),
            None,
        )
        if collided is not None:
            return submit_payload('planning_collision_terminal', task_id=collided['id'], status=collided['status'])
        self.issued += 1
        task_id = str(self.issued)
        self.tasks[task_id] = {
            'id': task_id,
            'title': title,
            'description': arguments.get('description', ''),
            'details': arguments.get('details', ''),
            'priority': arguments.get('priority'),
            'status': TaskStatus.DEFERRED.value,
            'metadata': copy.deepcopy(arguments.get('metadata') or {}),
        }
        return submit_payload('planning_create', task_id=task_id, status=TaskStatus.DEFERRED.value)

    def got(self, task_id: str) -> dict[str, Any]:
        if task_id not in self.tasks:
            return {'error': f'Task {task_id} not found', 'error_type': 'TaskNotFoundError'}
        return {'data': copy.deepcopy(self.tasks[task_id])}

    def found(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        statuses = arguments.get('statuses')
        matches = [
            task for task in self.tasks.values()
            if (statuses is None or task['status'] in statuses)
            and metadata_holds(task['metadata'].get(arguments['key']), arguments.get('value'))
        ]
        offset, limit = arguments.get('offset', 0), arguments.get('limit', 100)
        page = matches[offset:offset + limit]
        return {
            'results': [{'task_id': int(task['id']), 'status': task['status']} for task in page],
            'pagination': {
                'offset': offset, 'limit': limit, 'returned': len(page), 'has_more': offset + len(page) < len(matches),
            },
        }

    def committed(self, task_ids: Sequence[str], target_status: str) -> dict[str, Any]:
        for task_id in task_ids:
            task = self.tasks.get(task_id)
            if task is not None and task['status'] == TaskStatus.DEFERRED:
                task['status'] = target_status
        return {'success': True}

    def task(self, task_id: str) -> dict[str, Any]:
        return copy.deepcopy(self.tasks[task_id])

    def spawned(self, spawn_context: str) -> list[dict[str, Any]]:
        return [
            copy.deepcopy(task) for task in self.tasks.values() if task['metadata'].get('spawn_context') == spawn_context
        ]

    def lane_tasks(self) -> list[dict[str, Any]]:
        return self.spawned(SPAWN_CONTEXT)

    def deflake_tasks(self) -> list[dict[str, Any]]:
        return self.spawned(DEFLAKE_SPAWN_CONTEXT)

    def calls_since(self, mark: int) -> list[tuple[str, dict[str, Any]]]:
        return self.calls[mark:]


def typed_task_client(config: OrchestratorConfig, store: FakeTaskStore) -> SchedulerChronicFlakeTaskClient:
    """The lane's task client exactly as ``Harness._start_offline_lane`` wires it, over *store*."""
    return SchedulerChronicFlakeTaskClient(Scheduler(config, mcp_session=store), config.project_root)


class RedPathLane:
    """One real offline-lane worker over a seeded repo, a stub test runner, a fake task store, a real queue and a fake clock."""

    def __init__(
        self,
        *,
        config: OrchestratorConfig,
        git_ops: MeteredGitOps,
        queue: EscalationQueue,
        store: FakeTaskStore,
        stub: LaneStub,
        clock: Clock,
        command: LaneCommand,
        lock_path: Path,
        hooks: Sequence[PassOutcomeHook],
    ) -> None:
        self.config = config
        self.git = git_ops
        self.queue = queue
        self.store = store
        self.stub = stub
        self.clock = clock
        self.command = command
        self.lock_path = lock_path
        self.hooks = tuple(hooks)
        self.advances = 0
        self.worker = self.build_worker()

    @classmethod
    async def open(
        cls,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        command: LaneCommand = LANE_COMMAND,
        git_overrides: Mapping[str, Any] | None = None,
        hooks: Sequence[PassOutcomeHook] = (),
    ) -> RedPathLane:
        repo = seed_repo(tmp_path / 'repo', LANE_SEED)
        stub = LaneStub(tmp_path / 'stub')
        stub.put_on_path(monkeypatch)
        git_config = GitConfig(
            offline_lane_legacy_numeric_enabled=False,
            offline_lane_commands=[command],
            offline_lane_poll_interval_secs=POLL_INTERVAL_SECS,
            **(git_overrides or {}),
        )
        return cls(
            config=OrchestratorConfig(project_root=repo, git=git_config),
            git_ops=MeteredGitOps(git_config, repo),
            queue=EscalationQueue(tmp_path / 'escalations'),
            store=FakeTaskStore(),
            stub=stub,
            clock=Clock(start=datetime.now(UTC).replace(second=0, microsecond=0)),
            command=command,
            lock_path=tmp_path / 'offline_lane.lock',
            hooks=hooks,
        )

    @property
    def project_root(self) -> Path:
        return self.config.project_root

    @property
    def command_name(self) -> str:
        return self.command.name

    def build_worker(self) -> OfflineLaneWorker:
        return OfflineLaneWorker(
            self.git,
            self.config,
            lock_path=self.lock_path,
            task_client=typed_task_client(self.config, self.store),
            escalation_queue=self.queue,
            pass_outcome_hooks=self.hooks,
            now=self.clock,
        )

    async def head(self) -> str:
        return await head(self.project_root)

    async def drive(self, reads: int = 1, until: asyncio.Event | None = None) -> None:
        """Run the worker until it parks on the read after *reads* head reads, or until *until* is set."""
        self.git.allow(reads)
        stop = self.git.parked if until is None else until
        failures = RunLoopFailures()
        with failures.attached():
            task = asyncio.create_task(self.worker.run())
            try:
                await first_set((stop, failures.raised), DRIVE_BOUND_SECS, 'offline-lane drive')
            finally:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        assert failures.messages == [], f'the worker run loop failed a pass: {failures.messages}'

    async def advance(self, *runs: Run, minutes: float = 30) -> PassRecord:
        """Advance the clock and main, then drive the pass the merge trigger starts; *runs* are its sub-runs."""
        self.clock.advance(minutes=minutes)
        self.advances += 1
        await commit(self.project_root, {ADVANCE_FILE: f'{self.advances}\n'}, f'advance {self.advances}')
        return await self.triggered_pass(runs)

    async def rerun(self, *runs: Run, minutes: float = 30) -> PassRecord:
        """Advance the clock only, then drive a triggered pass at the same head."""
        self.clock.advance(minutes=minutes)
        return await self.triggered_pass(runs)

    async def triggered_pass(self, runs: Sequence[Run]) -> PassRecord:
        self.stub.script(*runs)
        started = self.clock()
        await self.worker.on_post_merge('lane-fixture-merge', '', '')
        await self.drive()
        assert self.stub.unconsumed() == [], 'the pass left scripted sub-runs unconsumed'
        started_passes = self.passes(started)
        assert started_passes, 'the triggered drive started no pass'
        return started_passes[-1]

    async def tick(self, *runs: Run, minutes: float = 0) -> list[PassRecord]:
        """Advance the clock, then drive one idle poll tick; the passes that started during it."""
        self.clock.advance(minutes=minutes)
        self.stub.script(*runs)
        started = self.clock()
        await self.drive()
        assert self.stub.unconsumed() == [], 'the tick left scripted sub-runs unconsumed'
        return self.passes(started)

    def passes(self, since: datetime = EPOCH) -> list[PassRecord]:
        store = RunRecordStore(self.project_root, clock=self.clock)
        return list(store.passes_since(self.command_name, since))

    def record(self) -> CommandRecord:
        return EpisodeStore(self.project_root, clock=self.clock).load(self.command_name)

    def episode(self, node: str) -> Episode | None:
        return self.record().episodes.get(node)
