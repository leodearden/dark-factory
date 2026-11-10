"""Tests for orchestrator.spawn_t3_lane, driven through its verbs against a fake T3.

``launch``, ``provider``, ``await-started`` and ``wait`` talk to
``_fake_t3_server.FakeT3Server`` over ``/mcp``; the ``t3`` fixture seeds this
host's credential for it. Its tool replies are the module-level ``*_JSON``
constants below, in T3 Code v0.1.6's shapes (``~/src/t3code-v0.1.6``):
``packages/contracts/src/project.ts::Project``,
``apps/server/src/mcp/toolkits/project/tools.ts`` (``t3_project_list``,
``t3_thread_launch``) and ``packages/contracts/src/orchestratorMcp.ts`` (the
thread list, read and wait results).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import ANY

import pytest
from _fake_t3_server import (
    DEAD_ORIGIN,
    ENVIRONMENT_ID,
    HANG_UP,
    FakeT3Server,
    ToolAnswer,
    ToolFailure,
    ToolResponder,
    ToolSuccess,
    put_fake_t3_cli_on_path,
    replies_in_turn,
)
from _process_identity_variants import own_identity
from _process_tree_fakes import claude_process_tree
from shared.process_identity import ProcessIdentity

from orchestrator import session_hooks, t3_mcp_client
from orchestrator import session_registry as sr
from orchestrator import spawn_t3_lane as lane
from orchestrator.session_ownership_probes import ProcessTree

_PKG_SRC = Path(lane.__file__).resolve().parents[1]

THREAD_ID = 'mcp:launch:' + '6f0c2a3e74894c119a510d6c1f3b2a10' * 2
PROJECT_ID = '5a05e1d3-5bd0-40e5-8ce7-24b2cd160300'
RUN_ID = 'b2c4e6f8-0a1c-4e3a-8b5d-7f9e1a2b3c4d'
SESSION_ID = '15fbcf7c-12af-481c-a641-9b00c03a741e'

PROJECT_JSON: dict[str, Any] = {
    'id': PROJECT_ID,
    'title': 'proj',
    'workspaceRoot': '/replaced/by/project',
    'repositoryIdentity': None,
    'defaultModelSelection': None,
    'scripts': [],
    'archivedAt': None,
    'createdAt': '2026-10-20T08:00:00.000Z',
    'updatedAt': '2026-10-20T08:00:00.000Z',
    'deletedAt': None,
}
"""One ``Project`` of ``t3_project_list``'s ``{projects, nextCursor}`` (abridged);
:func:`project` sets its ``workspaceRoot``."""

THREAD_LAUNCH_JSON: dict[str, Any] = {
    'threadId': THREAD_ID,
    'projectId': PROJECT_ID,
    'modelSelection': {'instanceId': 'claudeAgent', 'model': 'claude-opus-5-5'},
    'runId': RUN_ID,
    'status': 'starting',
}
"""``t3_thread_launch``'s success; ``runId`` and ``status`` may be null."""

THREAD_WAIT_JSON: dict[str, Any] = {
    'threadId': THREAD_ID,
    'runId': RUN_ID,
    'status': 'running',
    'timedOut': True,
}
"""``t3_thread_wait``'s ``OrchestratorMcpThreadWaitResult`` for a slice that ran out; :func:`waited` varies it."""

THREAD_LIST_JSON: dict[str, Any] = {
    'projectId': PROJECT_ID,
    'currentThreadId': None,
    'threads': [],
    'nextCursor': None,
    'total': 0,
}
"""``t3_thread_list``'s ``OrchestratorMcpThreadListResult`` for a project with no active thread."""

THREAD_READ_JSON: dict[str, Any] = {
    'thread': {
        'threadId': THREAD_ID,
        'projectId': PROJECT_ID,
        'title': 'unblock:df#7489 t3-lane',
        'status': 'running',
        'latestRunId': RUN_ID,
        'activeRunId': RUN_ID,
        'providerInstanceId': 'claudeAgent',
        'model': 'claude-opus-5-5',
        'runtimeMode': 'full-access',
        'worktreePath': None,
        'runCount': 1,
        'itemCount': 3,
        'pendingRequestCount': 0,
        'archived': False,
        'settled': False,
        'settledAt': None,
    },
    'recentRuns': [],
    'items': [],
    'nextPosition': None,
    'hasMore': False,
}
"""``t3_thread_read``'s ``OrchestratorMcpThreadReadResult``
(``packages/contracts/src/orchestratorMcp.ts``, the detail abridged); :func:`thread_read` varies it."""

SERVER_RUNTIME_JSON: dict[str, Any] = {
    'version': 1,
    'pid': 881190,
    'port': 1,
    'origin': DEAD_ORIGIN,
    'startedAt': '2026-10-23T09:09:43.139Z',
    'serviceManaged': True,
    'serverVersion': '0.1.6',
}
"""``$T3CODE_HOME/userdata/server-runtime.json`` as the live server writes it, but
naming an origin nothing answers: a test that forgets :func:`point_runtime_at`
fails as descriptor-unreachable instead of reaching the workstation's real T3 on :3773."""

def waited(*, timed_out: bool, status: str = 'running') -> ToolSuccess:
    return ToolSuccess({**THREAD_WAIT_JSON, 'status': status, 'timedOut': timed_out})


def thread_read(**thread: Any) -> ToolSuccess:
    """``t3_thread_read``'s reply with the thread's ``status``, ``archived`` or ``latestRunId`` overridden."""
    return ToolSuccess({**THREAD_READ_JSON, 'thread': {**THREAD_READ_JSON['thread'], **thread}})


def project(root: Path | str, **overrides: Any) -> dict[str, Any]:
    return {**PROJECT_JSON, 'workspaceRoot': str(root), **overrides}


def projects_page(*projects: dict[str, Any], next_cursor: int | None = None) -> ToolSuccess:
    return ToolSuccess({'projects': list(projects), 'nextCursor': next_cursor})


@dataclass
class Env:
    project: Path
    t3_home: Path
    fleet: Path
    projects_dir: Path


def path_without_t3() -> str:
    """PATH minus every directory holding a ``t3``, so a self-pair never reaches the workstation's real one."""
    dirs = os.environ.get('PATH', '').split(os.pathsep)
    return os.pathsep.join(d for d in dirs if d and not (Path(d) / 't3').exists())


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Env:
    project_dir = tmp_path / 'proj'
    (project_dir / 'sub').mkdir(parents=True)
    t3_home = tmp_path / 't3home'
    userdata = t3_home / 'userdata'
    userdata.mkdir(parents=True)
    (userdata / 'server-runtime.json').write_text(json.dumps(SERVER_RUNTIME_JSON))
    fleet = tmp_path / 'fleet'
    projects_dir = tmp_path / 'claude-projects'
    monkeypatch.setenv('T3CODE_HOME', str(t3_home))
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(fleet))
    monkeypatch.setenv('CLAUDE_PROJECTS_DIR', str(projects_dir))
    monkeypatch.setenv('PATH', path_without_t3())
    return Env(project_dir, t3_home, fleet, projects_dir)


@pytest.fixture
def t3_server() -> Iterator[FakeT3Server]:
    with FakeT3Server.running() as server:
        yield server


def point_runtime_at(env: Env, server: FakeT3Server) -> None:
    """Make server-runtime.json name the fake server, never the host's real :3773."""
    write_runtime_origin(env, server.origin)


def write_runtime_origin(env: Env, origin: object) -> None:
    runtime = {**SERVER_RUNTIME_JSON, 'origin': origin}
    (env.t3_home / 'userdata' / 'server-runtime.json').write_text(json.dumps(runtime))


def remove_runtime(env: Env) -> None:
    (env.t3_home / 'userdata' / 'server-runtime.json').unlink()


def stored_credential(env: Env) -> t3_mcp_client.Credential:
    credential = t3_mcp_client.read_credential_status(ENVIRONMENT_ID, root=env.fleet).credential
    assert credential is not None
    return credential


@pytest.fixture
def t3(env: Env, t3_server: FakeT3Server) -> FakeT3Server:
    """The fake T3 the runtime names, this host paired with it, one live project at ``env.project``."""
    point_runtime_at(env, t3_server)
    t3_server.write_paired_credential(env.fleet)
    t3_server.tool_replies |= {
        't3_project_list': projects_page(project(env.project)),
        't3_thread_launch': ToolSuccess(THREAD_LAUNCH_JSON),
        't3_thread_list': ToolSuccess(THREAD_LIST_JSON),
    }
    return t3_server


STALL_CAP_SECS = 10.0
"""The longest a stalled reply is held: a call the lane leaves unbounded fails a test's elapsed bound instead of hanging it."""

StallThen = Callable[[ToolAnswer], ToolResponder]


@pytest.fixture
def stalling(t3: FakeT3Server) -> Iterator[StallThen]:
    """``stall_then(answer)``: a responder holding each call until released or capped, then answering ``answer``."""
    released = threading.Event()

    def stall_then(answer: ToolAnswer) -> ToolResponder:
        def respond(arguments: Mapping[str, Any]) -> ToolAnswer:
            released.wait(STALL_CAP_SECS)
            return answer

        return respond

    yield stall_then
    released.set()


def run_verb(capsys: pytest.CaptureFixture[str], *argv: str) -> str:
    rc = lane.main(list(argv))
    out = capsys.readouterr().out
    assert rc == 0, out
    assert out.count('\n') == 1, out
    return out.strip()


def spawn_record(env: Env, slug: str = 'unblock-df-7489-4242') -> str:
    """A spawn's launching record, with the result file ``_run_launching`` gives every spawn."""
    record_dir = sr.record_path_for_slug(slug, root=env.fleet).parent
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug,
            status=sr.Status.LAUNCHING,
            cwd=str(env.project),
            launcher_pid=4242,
            start_ts=datetime.now(UTC).isoformat(),
            result_file=str(record_dir / sr.RESULT_FILENAME),
        ),
        root=env.fleet,
    )
    return str(record_dir)


def stale_t3_record(
    env: Env,
    *,
    environment_id: str | None = None,
    slug: str = 'sibling-df-1-1',
    thread_id: str = 'th-vanished',
) -> str:
    """A live, indexed t3 record of ``env.project`` whose thread no listing names."""
    record_dir = spawn_record(env, slug)
    record = sr.read_record(slug, root=env.fleet)
    record.display = sr.Display(
        kind='t3', t3_thread_id=thread_id, t3_environment_id=environment_id
    )
    sr.write_record(record, root=env.fleet)
    sr.write_t3_thread_pointer(thread_id, slug, root=env.fleet)
    time.sleep(0.01)
    return record_dir


def sweep_line_levels(caplog: pytest.LogCaptureFixture) -> list[int]:
    """The level of each index-size line ``mark_t3_sessions_exited`` logged."""
    return [
        record.levelno
        for record in caplog.records
        if record.name == 'orchestrator.session_registry'
        and record.getMessage().startswith('mark_t3_sessions_exited: marked')
    ]


def prompt_file(tmp_path: Path, text: str) -> str:
    path = tmp_path / 'prompt.txt'
    path.write_text(text)
    return str(path)


def launch_line(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, cwd: Path | str, *flags: str,
    record: str = '', title: str = 't', prompt: str = 'p',
) -> str:
    return run_verb(
        capsys, 'launch', '--cwd', str(cwd), '--title', title,
        '--prompt-file', prompt_file(tmp_path, prompt), '--record', record, *flags,
    )


def sent_launch(server: FakeT3Server) -> dict[str, Any]:
    [arguments] = server.tool_calls('t3_thread_launch')
    return arguments


PROXY_SETTINGS: dict[str, Any] = {
    'defaultModelSelection': {'instanceId': 'claude-proxy', 'model': 'claude-opus-5-5'},
    'providerInstances': {
        'claudeAgent': {'driver': 'claudeAgent', 'enabled': False},
        'claude-proxy': {'driver': 'claudeAgent', 'enabled': True},
    },
}
"""``$T3CODE_HOME/userdata/settings.json`` on a host whose only enabled Claude instance
is ``claude-proxy`` (v0.1.6 shapes, abridged)."""

PROJECT_DISABLES_PROXY: dict[str, Any] = {
    **PROXY_SETTINGS,
    'projectSettingsOverrides': {PROJECT_ID: {'providerInstances': {'claude-proxy': {'enabled': False}}}},
}


def write_settings(env: Env, settings: dict[str, Any]) -> None:
    (env.t3_home / 'userdata' / 'settings.json').write_text(json.dumps(settings))


# ---------------------------------------------------------------------------
# provider
# ---------------------------------------------------------------------------


class TestProvider:
    def test_the_projects_live_instance_is_named_with_its_source(
        self, env: Env, t3: FakeT3Server, capsys: pytest.CaptureFixture[str]
    ) -> None:
        write_settings(env, PROXY_SETTINGS)

        line = run_verb(capsys, 'provider', '--cwd', str(env.project))

        assert line == 'instance instance_id=claude-proxy source=project'

    def test_a_disabled_explicit_instance_is_refused(
        self, env: Env, t3: FakeT3Server, capsys: pytest.CaptureFixture[str]
    ) -> None:
        write_settings(env, PROXY_SETTINGS)

        line = run_verb(
            capsys, 'provider', '--cwd', str(env.project), '--provider', 'claudeAgent'
        )

        assert line == 'refused reason=provider-disabled instance_id=claudeAgent source=explicit'

    def test_a_project_disable_is_read_through_the_project_list(
        self, env: Env, t3: FakeT3Server, capsys: pytest.CaptureFixture[str]
    ) -> None:
        write_settings(env, PROJECT_DISABLES_PROXY)

        line = run_verb(capsys, 'provider', '--cwd', str(env.project))

        assert line == 'refused reason=provider-disabled instance_id=claudeAgent source=fallback'
        assert t3.tool_calls('t3_project_list') == [{'limit': 100}]

    def test_without_a_credential_enablement_is_checked_environment_wide(
        self, env: Env, t3_server: FakeT3Server, capsys: pytest.CaptureFixture[str]
    ) -> None:
        point_runtime_at(env, t3_server)
        write_settings(env, PROJECT_DISABLES_PROXY)

        line = run_verb(capsys, 'provider', '--cwd', str(env.project))

        assert line == 'instance instance_id=claude-proxy source=project'

    def test_no_settings_falls_back_to_claude_agent_unverified(
        self, env: Env, t3: FakeT3Server, capsys: pytest.CaptureFixture[str]
    ) -> None:
        line = run_verb(capsys, 'provider', '--cwd', str(env.project))

        assert line == 'instance instance_id=claudeAgent source=fallback'


# ---------------------------------------------------------------------------
# detect
# ---------------------------------------------------------------------------


def _start_epoch(pid: int) -> float:
    stat = lane.read_proc_stat(pid, Path('/proc'))
    assert stat is not None
    btime = next(
        int(line.split()[1])
        for line in Path('/proc/stat').read_text().splitlines()
        if line.startswith('btime ')
    )
    return btime + stat.start_ticks / os.sysconf('SC_CLK_TCK')


def _write_runtime(t3_home: Path, pid: int, started: float) -> None:
    userdata = t3_home / 'userdata'
    userdata.mkdir(parents=True, exist_ok=True)
    started_at = datetime.fromtimestamp(started, tz=UTC).isoformat().replace('+00:00', 'Z')
    (userdata / 'server-runtime.json').write_text(
        json.dumps({**SERVER_RUNTIME_JSON, 'pid': pid, 'startedAt': started_at})
    )


def _detect_in_new_session(t3_home: Path) -> str:
    """Run `detect` by path as spawn-claude.sh does, detached from any controlling tty."""
    completed = subprocess.run(
        [sys.executable, str(Path(lane.__file__)), 'detect'],
        env={**os.environ, 'PYTHONPATH': str(_PKG_SRC), 'T3CODE_HOME': str(t3_home)},
        capture_output=True,
        text=True,
        start_new_session=True,
        timeout=60,
        check=True,
    )
    return completed.stdout.strip()


@pytest.mark.skipif(not Path('/proc/self/stat').exists(), reason='needs /proc')
class TestDetect:
    def test_an_ancestor_server_with_a_matching_start_time_and_no_tty_is_t3(
        self, tmp_path: Path
    ) -> None:
        _write_runtime(tmp_path, os.getpid(), _start_epoch(os.getpid()))

        assert _detect_in_new_session(tmp_path) == 't3'

    def test_a_server_pid_that_is_not_an_ancestor_is_not_t3(self, tmp_path: Path) -> None:
        stranger = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
        try:
            _write_runtime(tmp_path, stranger.pid, _start_epoch(stranger.pid))
            assert _detect_in_new_session(tmp_path) == 'not-t3'
        finally:
            stranger.kill()
            stranger.wait()

    def test_a_reused_server_pid_with_another_start_time_is_not_t3(self, tmp_path: Path) -> None:
        _write_runtime(tmp_path, os.getpid(), _start_epoch(os.getpid()) - 3600)

        assert _detect_in_new_session(tmp_path) == 'not-t3'

    def test_no_server_runtime_file_is_not_t3(self, tmp_path: Path) -> None:
        assert _detect_in_new_session(tmp_path) == 'not-t3'


def _fake_proc(root: Path, *, own_tty: int, server_pid: int = 777) -> Path:
    """A /proc with this process -> server_pid, the server started 100 s after boot."""
    proc = root / 'proc'
    for pid, ppid, tty, start_ticks in (
        (os.getpid(), server_pid, own_tty, 50_000),
        (server_pid, 1, 0, 100 * os.sysconf('SC_CLK_TCK')),
    ):
        (proc / str(pid)).mkdir(parents=True)
        tail = ['S', str(ppid), '0', '0', str(tty)] + ['0'] * 14 + [str(start_ticks)] + ['0'] * 10
        (proc / str(pid) / 'stat').write_text(f'{pid} (proc name) {" ".join(tail)}\n')
    (proc / 'stat').write_text('cpu  1 2 3\nbtime 1761379200\n')
    return proc


class TestDetectWithAFakeProc:
    def _env(self, tmp_path: Path) -> dict[str, str]:
        _write_runtime(tmp_path / 't3', 777, 1761379200 + 100)
        return {'T3CODE_HOME': str(tmp_path / 't3')}

    def test_the_fake_tree_without_a_tty_is_t3(self, tmp_path: Path) -> None:
        proc = _fake_proc(tmp_path, own_tty=0)

        assert lane.is_t3_hosted(self._env(tmp_path), proc_root=proc)

    def test_a_controlling_tty_is_not_t3(self, tmp_path: Path) -> None:
        proc = _fake_proc(tmp_path, own_tty=34816)

        assert not lane.is_t3_hosted(self._env(tmp_path), proc_root=proc)

    def test_no_proc_is_not_t3(self, tmp_path: Path) -> None:
        assert not lane.is_t3_hosted(self._env(tmp_path), proc_root=tmp_path / 'absent')


# ---------------------------------------------------------------------------
# launch
# ---------------------------------------------------------------------------


class TestLaunch:
    def test_started_sends_the_launch_request_and_stamps_the_record(
        self, env: Env, t3: FakeT3Server, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        record_dir = spawn_record(env)

        line = launch_line(
            capsys, tmp_path, env.project, '--skip-permissions', '--model', 'opus',
            record=record_dir, title='unblock:df#7489 x', prompt='do the thing',
        )

        assert line == f'started thread_id={THREAD_ID} project_id={PROJECT_ID} run_id={RUN_ID}'
        assert t3.tool_calls('t3_thread_launch') == [
            {
                'projectId': PROJECT_ID,
                'title': 'unblock:df#7489 x',
                'message': 'do the thing',
                'modelSelection': {'instanceId': 'claudeAgent', 'model': 'opus'},
                'runtimeMode': 'full-access',
                'workspaceStrategy': {'type': 'root'},
                'clientRequestId': ANY,
            }
        ]
        record = sr.read_record(Path(record_dir).name, root=env.fleet)
        assert record.status == sr.Status.RUNNING
        assert record.display == sr.Display(
            kind='t3', wm_title='unblock:df#7489 x', t3_thread_id=THREAD_ID,
            t3_environment_id=ENVIRONMENT_ID,
        )

    @pytest.mark.parametrize(
        ('flags', 'runtime_mode'),
        [(['--skip-permissions'], 'full-access'), ([], 'approval-required')],
        ids=['skip-permissions', 'approvals'],
    )
    @pytest.mark.parametrize('in_worktree', [False, True], ids=['root', 'worktree'])
    def test_the_runtime_mode_follows_the_flag_and_the_workspace_follows_the_cwd(
        self, env: Env, t3: FakeT3Server, tmp_path: Path, capsys: pytest.CaptureFixture[str],
        flags: list[str], runtime_mode: str, in_worktree: bool,
    ) -> None:
        worktree = env.project / '.worktrees' / '8123'
        worktree.mkdir(parents=True)

        line = launch_line(capsys, tmp_path, worktree if in_worktree else env.project, *flags)

        assert line.startswith('started ')
        strategy = {'type': 'existing_worktree', 'worktreePath': str(worktree)} if in_worktree else {'type': 'root'}
        sent = sent_launch(t3)
        assert (sent['runtimeMode'], sent['workspaceStrategy']) == (runtime_mode, strategy)

    def test_no_model_sends_no_model_selection(
        self, env: Env, t3: FakeT3Server, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        launch_line(capsys, tmp_path, env.project)

        assert 'modelSelection' not in sent_launch(t3)

    def test_a_slug_reused_by_a_later_record_does_not_replay_the_earlier_thread(
        self, env: Env, t3: FakeT3Server, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        launch_line(capsys, tmp_path, env.project, record=spawn_record(env))
        launch_line(capsys, tmp_path, env.project, record=spawn_record(env))

        request_ids = [call['clientRequestId'] for call in t3.tool_calls('t3_thread_launch')]
        assert len(set(request_ids)) == 2

    def test_without_a_record_each_launch_gets_its_own_request_id(
        self, env: Env, t3: FakeT3Server, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        t3.tool_replies['t3_thread_launch'] = ToolSuccess({**THREAD_LAUNCH_JSON, 'runId': None})

        first = launch_line(capsys, tmp_path, env.project)
        launch_line(capsys, tmp_path, env.project)

        assert first == f'started thread_id={THREAD_ID} project_id={PROJECT_ID} run_id=none'
        request_ids = [call['clientRequestId'] for call in t3.tool_calls('t3_thread_launch')]
        assert all(isinstance(key, str) and key for key in request_ids)
        assert len(set(request_ids)) == 2

    @pytest.mark.parametrize(
        ('parent_args', 'expected'),
        [(['--parent', 'watcher-df-9'], 'watcher-df-9'), ([], None), (['--parent', '../x'], None)],
    )
    def test_parent_is_stamped_only_when_given_as_a_record_key(
        self, env: Env, t3: FakeT3Server, tmp_path: Path, capsys: pytest.CaptureFixture[str],
        parent_args: list[str], expected: str | None,
    ) -> None:
        record_dir = spawn_record(env)

        line = launch_line(capsys, tmp_path, env.project, *parent_args, record=record_dir)

        assert line.startswith('started ')
        record = sr.read_record(Path(record_dir).name, root=env.fleet)
        assert (record.parent_session_id, record.status) == (expected, sr.Status.RUNNING)
        assert record.display is not None and record.display.t3_thread_id == THREAD_ID

    def test_a_model_runs_on_the_projects_live_claude_instance(
        self, env: Env, t3: FakeT3Server, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        write_settings(env, PROXY_SETTINGS)

        line = launch_line(capsys, tmp_path, env.project, '--model', 'fable')

        assert line.startswith('started ')
        assert sent_launch(t3)['modelSelection'] == {'instanceId': 'claude-proxy', 'model': 'fable'}

    def test_a_disabled_explicit_instance_is_refused_before_any_start(
        self, env: Env, t3: FakeT3Server, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        write_settings(env, PROXY_SETTINGS)
        record_dir = spawn_record(env)

        line = launch_line(
            capsys, tmp_path, env.project, '--model', 'fable', '--provider', 'claudeAgent', record=record_dir
        )

        assert line == 'refused reason=provider-disabled instance_id=claudeAgent source=explicit'
        assert t3.tool_calls('t3_thread_launch') == [] and t3.tool_calls('t3_thread_list') == []
        assert sr.read_record(Path(record_dir).name, root=env.fleet).display is None

    def test_launch_first_sweeps_the_environments_records_whose_thread_is_gone(
        self, env: Env, t3: FakeT3Server, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        stale_dir = stale_t3_record(env)
        t3.tool_replies['t3_thread_read'] = ToolFailure('thread_not_found', 'Thread th-vanished was not found.')

        launch_line(capsys, tmp_path, env.project)

        stale = sr.read_record(Path(stale_dir).name, root=env.fleet)
        assert (stale.status, stale.exit_code) == (sr.Status.EXITED, sr.ORPHAN_EXIT_CODE)
        assert t3.tool_calls('t3_thread_list') == [{'allProjects': True, 'limit': 100}]

    def test_an_unlisted_thread_whose_read_archives_it_is_swept(
        self, env: Env, t3: FakeT3Server, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        stale_dir = stale_t3_record(env)
        t3.tool_replies['t3_thread_read'] = thread_read(threadId='th-vanished', archived=True)

        launch_line(capsys, tmp_path, env.project)

        stale = sr.read_record(Path(stale_dir).name, root=env.fleet)
        assert (stale.status, stale.exit_code) == (sr.Status.EXITED, sr.ORPHAN_EXIT_CODE)

    def test_an_unlisted_thread_that_still_reads_present_is_not_swept(
        self, env: Env, t3: FakeT3Server, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """T3 v0.1.6 lists no side chat (``ThreadManagementService.ts::listProjectThreads``), yet it is live."""
        stale_dir = stale_t3_record(env)
        t3.tool_replies['t3_thread_read'] = thread_read(threadId='th-vanished')

        launch_line(capsys, tmp_path, env.project)

        assert sr.read_record(Path(stale_dir).name, root=env.fleet).status == sr.Status.LAUNCHING
        assert sr.resolve_session_slug_for_t3_thread('th-vanished', root=env.fleet) == (
            Path(stale_dir).name
        )
        assert t3.tool_calls('t3_thread_read') == [{'threadId': 'th-vanished', 'limit': 1}]

    def test_an_unlisted_thread_whose_read_fails_is_not_swept(
        self, env: Env, t3: FakeT3Server, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        stale_dir = stale_t3_record(env)
        t3.tool_replies['t3_thread_read'] = ToolFailure('orchestration_error', 'busy')

        launch_line(capsys, tmp_path, env.project)

        assert sr.read_record(Path(stale_dir).name, root=env.fleet).status == sr.Status.LAUNCHING

    def test_after_one_unreachable_read_the_sweep_asks_about_no_further_thread(
        self, env: Env, t3: FakeT3Server, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        first = stale_t3_record(env, slug='sibling-df-1-1', thread_id='th-a')
        second = stale_t3_record(env, slug='sibling-df-1-2', thread_id='th-b')
        t3.tool_replies['t3_thread_read'] = ToolFailure('orchestration_error', 'busy')

        launch_line(capsys, tmp_path, env.project)

        assert t3.tool_calls('t3_thread_read') == [{'threadId': 'th-a', 'limit': 1}]
        for record_dir in (first, second):
            assert sr.read_record(Path(record_dir).name, root=env.fleet).status == sr.Status.LAUNCHING

    def test_launch_logs_its_sweeps_index_line_at_info(
        self,
        env: Env,
        t3: FakeT3Server,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.DEBUG, logger='orchestrator.session_registry'):
            launch_line(capsys, tmp_path, env.project)

        assert sweep_line_levels(caplog) == [logging.INFO]

    def test_a_record_of_another_environment_is_left_to_its_own_sweep(
        self, env: Env, t3: FakeT3Server, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        stale_dir = stale_t3_record(env, environment_id='env-elsewhere')

        launch_line(capsys, tmp_path, env.project)

        assert sr.read_record(Path(stale_dir).name, root=env.fleet).status == sr.Status.LAUNCHING
        assert sr.resolve_session_slug_for_t3_thread('th-vanished', root=env.fleet) == (
            Path(stale_dir).name
        )
        assert t3.tool_calls('t3_thread_read') == []

    def test_an_incomplete_thread_listing_sweeps_nothing(
        self, env: Env, t3: FakeT3Server, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        stale_dir = stale_t3_record(env)
        t3.tool_replies['t3_thread_list'] = replies_in_turn(
            ToolSuccess({**THREAD_LIST_JSON, 'nextCursor': 1}), ToolFailure('orchestration_error', 'page lost')
        )

        line = launch_line(capsys, tmp_path, env.project)

        assert line.startswith('started ')
        assert len(t3.tool_calls('t3_thread_list')) == 2
        assert sr.read_record(Path(stale_dir).name, root=env.fleet).status == sr.Status.LAUNCHING

    def test_a_thread_listing_whose_total_moves_between_pages_sweeps_nothing(
        self, env: Env, t3: FakeT3Server, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """T3 pages by offset: a thread archived between the reads shifts the record's live thread past page 2."""
        record_dir = stale_t3_record(env)
        t3.tool_replies['t3_thread_list'] = replies_in_turn(
            ToolSuccess({**THREAD_LIST_JSON, 'threads': [{'threadId': 'th-archived'}], 'nextCursor': 1, 'total': 2}),
            ToolSuccess({**THREAD_LIST_JSON, 'threads': [], 'total': 1}),
        )

        line = launch_line(capsys, tmp_path, env.project)

        assert line.startswith('started ')
        assert sr.read_record(Path(record_dir).name, root=env.fleet).status == sr.Status.LAUNCHING

    def test_a_prompt_over_the_ceiling_is_spilled_into_the_record_dir(
        self, env: Env, t3: FakeT3Server, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        record_dir = spawn_record(env)
        big = 'x' * (lane.MESSAGE_CHAR_LIMIT + 1)

        launch_line(capsys, tmp_path, env.project, record=record_dir, prompt=big)

        spill = Path(record_dir) / 'prompt.md'
        assert spill.read_text() == big
        assert sent_launch(t3)['message'] == (
            f'Read {spill} and follow it as your brief; that file holds your whole task.'
        )

    def test_a_prompt_at_the_ceiling_is_sent_whole(
        self, env: Env, t3: FakeT3Server, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        exact = 'y' * lane.MESSAGE_CHAR_LIMIT

        launch_line(capsys, tmp_path, env.project, prompt=exact)

        assert sent_launch(t3)['message'] == exact

    def test_a_spill_without_a_record_goes_to_a_temp_file(
        self, env: Env, t3: FakeT3Server, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        big = 'z' * (lane.MESSAGE_CHAR_LIMIT + 1)

        launch_line(capsys, tmp_path, env.project, prompt=big)

        message = sent_launch(t3)['message']
        spill = Path(message.removeprefix('Read ').split(' and follow')[0])
        try:
            assert spill.read_text() == big
        finally:
            spill.unlink()


@dataclass
class PagedProjects:
    worktree: Path
    archived: Path
    pages: dict[int | None, ToolAnswer]
    """``t3_project_list``'s reply by the call's ``cursor``."""


@pytest.fixture
def paged(env: Env, t3: FakeT3Server, tmp_path: Path) -> PagedProjects:
    """Three pages: a deleted project at the worktree, an archived one elsewhere, then the live one."""
    worktree = env.project / '.worktrees' / '8123'
    worktree.mkdir(parents=True)
    archived = tmp_path / 'archived'
    (archived / 'sub').mkdir(parents=True)
    pages: dict[int | None, ToolAnswer] = {
        None: projects_page(
            project(worktree, id='deleted-project', deletedAt='2026-10-30T00:00:00.000Z'), next_cursor=1
        ),
        1: projects_page(
            project(archived, id='archived-project', archivedAt='2026-10-30T00:00:00.000Z'), next_cursor=2
        ),
        2: projects_page(project(env.project)),
    }
    t3.tool_replies['t3_project_list'] = lambda arguments: pages[arguments.get('cursor')]
    return PagedProjects(worktree, archived, pages)


class TestProjectResolution:
    def test_every_page_is_read_and_the_live_project_takes_its_worktree(
        self, t3: FakeT3Server, paged: PagedProjects, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        line = launch_line(capsys, tmp_path, paged.worktree)

        assert line == f'started thread_id={THREAD_ID} project_id={PROJECT_ID} run_id={RUN_ID}'
        assert t3.tool_calls('t3_project_list') == [
            {'limit': 100}, {'limit': 100, 'cursor': 1}, {'limit': 100, 'cursor': 2}
        ]
        sent = sent_launch(t3)
        assert (sent['projectId'], sent['workspaceStrategy']) == (
            PROJECT_ID, {'type': 'existing_worktree', 'worktreePath': str(paged.worktree)}
        )

    def test_a_cwd_only_an_archived_project_contains_is_refused_naming_its_root(
        self, env: Env, t3: FakeT3Server, paged: PagedProjects, tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        record_dir = spawn_record(env)

        line = launch_line(capsys, tmp_path, paged.archived / 'sub', record=record_dir)

        assert line == f'refused reason=project-archived detail={os.path.realpath(paged.archived)}'
        assert t3.tool_calls('t3_thread_launch') == []
        assert sr.read_record(Path(record_dir).name, root=env.fleet).display is None

    def test_a_cwd_outside_every_project_is_refused(
        self, t3: FakeT3Server, paged: PagedProjects, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        line = launch_line(capsys, tmp_path, tmp_path)

        assert line == 'refused reason=cwd-outside-projects'
        assert t3.tool_calls('t3_thread_launch') == []

    @pytest.mark.parametrize(
        'lost_page', [ToolFailure('orchestration_error', 'page lost'), HANG_UP], ids=['tool-failure', 'hang-up']
    )
    def test_a_project_page_that_fails_is_launch_failed(
        self, t3: FakeT3Server, paged: PagedProjects, tmp_path: Path, capsys: pytest.CaptureFixture[str],
        lost_page: ToolAnswer,
    ) -> None:
        paged.pages[1] = lost_page

        line = launch_line(capsys, tmp_path, paged.worktree)

        assert line.split()[0] == 'launch-failed'
        assert t3.tool_calls('t3_thread_launch') == []

    def test_a_stalled_project_page_is_launch_failed_without_a_launch(
        self, env: Env, t3: FakeT3Server, stalling: StallThen, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(lane, 'CALL_TIMEOUT_SECS', 1.0)
        t3.tool_replies['t3_project_list'] = stalling(projects_page(project(env.project)))

        started = time.monotonic()
        line = launch_line(capsys, tmp_path, env.project)

        assert line.split()[0] == 'launch-failed'
        assert t3.tool_calls('t3_thread_launch') == []
        assert time.monotonic() - started < STALL_CAP_SECS / 2


def forget_credential(env: Env) -> None:
    t3_mcp_client.credential_path(ENVIRONMENT_ID, root=env.fleet).unlink()


class TestLaunchFailures:
    def test_t3_refusing_a_worktree_launch_as_invalid_is_refused_with_its_message(
        self, env: Env, t3: FakeT3Server, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        worktree = env.project / '.worktrees' / '8123'
        worktree.mkdir(parents=True)
        t3.tool_replies['t3_thread_launch'] = ToolFailure('invalid_request', 'The request was rejected.')
        record_dir = spawn_record(env)

        line = launch_line(capsys, tmp_path, worktree, record=record_dir)

        assert line == 'refused reason=cwd-not-project-worktree detail=The%20request%20was%20rejected.'
        record = sr.read_record(Path(record_dir).name, root=env.fleet)
        assert (record.display, record.status) == (None, sr.Status.LAUNCHING)
        assert len(t3.tool_calls('t3_thread_launch')) == 1

    def test_an_invalid_root_launch_is_launch_failed_whatever_its_message_says(
        self, env: Env, t3: FakeT3Server, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        t3.tool_replies['t3_thread_launch'] = ToolFailure('invalid_request', "Path '/x' is not a worktree of this project.")
        record_dir = spawn_record(env)

        line = launch_line(capsys, tmp_path, env.project, record=record_dir)

        assert line == (
            'launch-failed detail=invalid_request:%20Path%20%27/x%27%20is%20not%20a%20worktree%20of%20this%20project.'
        )
        assert sr.read_record(Path(record_dir).name, root=env.fleet).display is None

    def test_another_failure_of_a_worktree_launch_is_launch_failed(
        self, env: Env, t3: FakeT3Server, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        worktree = env.project / '.worktrees' / '8123'
        worktree.mkdir(parents=True)
        t3.tool_replies['t3_thread_launch'] = ToolFailure('orchestration_error', 'the provider is down')

        line = launch_line(capsys, tmp_path, worktree)

        assert line == 'launch-failed detail=orchestration_error:%20the%20provider%20is%20down'

    def test_a_lost_reply_is_resent_once_with_the_same_request_id(
        self, env: Env, t3: FakeT3Server, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        t3.tool_replies['t3_thread_launch'] = replies_in_turn(HANG_UP, ToolSuccess(THREAD_LAUNCH_JSON))
        record_dir = spawn_record(env)

        line = launch_line(capsys, tmp_path, env.project, record=record_dir)

        assert line == f'started thread_id={THREAD_ID} project_id={PROJECT_ID} run_id={RUN_ID}'
        request_ids = [call['clientRequestId'] for call in t3.tool_calls('t3_thread_launch')]
        assert len(request_ids) == 2 and len(set(request_ids)) == 1

    def test_a_reply_lost_twice_is_launch_failed(
        self, env: Env, t3: FakeT3Server, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        t3.tool_replies['t3_thread_launch'] = replies_in_turn(HANG_UP)

        line = launch_line(capsys, tmp_path, env.project)

        assert line.split()[0] == 'launch-failed'
        assert len(t3.tool_calls('t3_thread_launch')) == 2

    def test_a_stalled_launch_reply_is_resent_once_with_the_same_request_id(
        self, env: Env, t3: FakeT3Server, stalling: StallThen, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(lane, 'CALL_TIMEOUT_SECS', 1.0)
        t3.tool_replies['t3_thread_launch'] = replies_in_turn(
            stalling(ToolSuccess(THREAD_LAUNCH_JSON)), ToolSuccess(THREAD_LAUNCH_JSON)
        )
        record_dir = spawn_record(env)

        started = time.monotonic()
        line = launch_line(capsys, tmp_path, env.project, record=record_dir)

        assert line == f'started thread_id={THREAD_ID} project_id={PROJECT_ID} run_id={RUN_ID}'
        request_ids = [call['clientRequestId'] for call in t3.tool_calls('t3_thread_launch')]
        assert len(request_ids) == 2 and len(set(request_ids)) == 1
        assert time.monotonic() - started < STALL_CAP_SECS / 2

    def test_no_credential_and_no_t3_to_self_pair_is_no_launcher(
        self, env: Env, t3_server: FakeT3Server, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        point_runtime_at(env, t3_server)

        line = launch_line(capsys, tmp_path, env.project)

        assert line.startswith('no-launcher reason=t3-mcp-pair-failed detail=')
        assert t3_server.mcp_requests() == []

    def test_no_credential_self_pairs_on_the_t3_host(
        self, env: Env, t3: FakeT3Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        forget_credential(env)
        put_fake_t3_cli_on_path(t3, tmp_path, monkeypatch, version_line='t3 v0.1.6')

        line = launch_line(capsys, tmp_path, env.project)

        assert line.startswith('started ')
        assert stored_credential(env).paired_by is t3_mcp_client.PairedBy.SELF_PAIR

    def test_a_bearer_t3_refuses_with_no_t3_to_re_pair_is_credential_expired(
        self, env: Env, t3: FakeT3Server, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        t3.write_paired_credential(env.fleet, access_token='revoked')

        line = launch_line(capsys, tmp_path, env.project)

        assert line.startswith('no-launcher reason=t3-mcp-credential-expired detail=')
        assert t3.tool_calls('t3_thread_launch') == []

    def test_no_server_runtime_is_no_launcher(
        self, env: Env, t3: FakeT3Server, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        remove_runtime(env)

        line = launch_line(capsys, tmp_path, env.project)

        assert line == 'no-launcher reason=server-runtime-missing'
        assert t3.mcp_requests() == []

    def test_a_runtime_origin_nothing_answers_is_no_launcher(
        self, env: Env, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        write_runtime_origin(env, DEAD_ORIGIN)

        line = launch_line(capsys, tmp_path, env.project)

        assert line.startswith('no-launcher reason=descriptor-unreachable detail=')


# ---------------------------------------------------------------------------
# Transcripts, await-started, binding
# ---------------------------------------------------------------------------


def write_transcript(
    env: Env, session_id: str, first_user_text: str, *, mtime: float | None = None,
    later_text: str = '',
) -> Path:
    directory = env.projects_dir / sr.encode_cwd(str(env.project))
    directory.mkdir(parents=True, exist_ok=True)
    entries = [
        {'type': 'queue-operation', 'operation': 'enqueue', 'content': later_text},
        {'type': 'user', 'isMeta': True, 'message': {'role': 'user', 'content': later_text}},
        {
            'type': 'user',
            'entrypoint': 'sdk-ts',
            'sessionId': session_id,
            'message': {'role': 'user', 'content': [{'type': 'text', 'text': first_user_text}]},
        },
        {'type': 'user', 'message': {'role': 'user', 'content': later_text}},
    ]
    path = directory / f'{session_id}.jsonl'
    path.write_text(''.join(json.dumps(entry) + '\n' for entry in entries))
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def hand_record(
    env: Env,
    session_id: str,
    owner_pid: int | None,
    *,
    owner_identity: ProcessIdentity | None = None,
) -> str:
    """A hand-launched record plus its session-id pointer, as the hooks write them."""
    slug = f'session-proj-{session_id}'
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug,
            status=sr.Status.RUNNING,
            cwd=str(env.project),
            claude_session_id=session_id,
            claude_owner_pid=owner_pid,
            claude_owner_identity=owner_identity,
        ),
        root=env.fleet,
    )
    sr.write_claude_session_pointer(session_id, slug, root=env.fleet)
    return slug


def trailer(record_dir: str) -> str:
    """The tail of the launch message, as spawn-claude.sh's result trailer words it."""
    return (
        'Do it.\n\n---\nBefore you end this session (whether you finish, hand off, or get\n'
        f'blocked), write your outcome to: {record_dir}/result.md\n'
    )


class TestAwaitStarted:
    def _await(self, capsys: pytest.CaptureFixture[str], env: Env, record_dir: str,
               timeout: str = '5', since: float = 0.0) -> str:
        return run_verb(
            capsys, 'await-started', '--thread-id', THREAD_ID, '--cwd', str(env.project),
            '--record', record_dir, '--timeout', timeout, '--since', str(since),
        )

    def test_started_binds_the_transcript_and_folds_the_hand_launched_record(
        self, env: Env, t3: FakeT3Server, capsys: pytest.CaptureFixture[str]
    ) -> None:
        t3.tool_replies['t3_thread_read'] = replies_in_turn(thread_read(status='preparing'), thread_read(status='running'))
        record_dir = spawn_record(env)
        write_transcript(env, SESSION_ID, trailer(record_dir))
        hand_slug = hand_record(env, SESSION_ID, owner_pid=424242)

        line = self._await(capsys, env, record_dir, since=time.time() - 60)

        assert line == f'started sid={SESSION_ID} hand_slug={hand_slug} bound=yes via=transcript'
        slug = Path(record_dir).name
        record = sr.read_record(slug, root=env.fleet)
        assert (record.claude_session_id, record.claude_launch_session_id) == (SESSION_ID, SESSION_ID)
        assert record.claude_owner_pid == 424242
        assert sr.resolve_session_slug_for_claude_session_id(SESSION_ID, root=env.fleet) == slug
        assert sr.resolve_session_slug_for_pid(424242, root=env.fleet) == slug
        folded = sr.read_record(hand_slug, root=env.fleet)
        assert (folded.status, folded.exit_code) == (sr.Status.EXITED, sr.FOLDED_EXIT_CODE)

    def test_started_copies_the_hand_launched_owner_identity_with_its_pid(
        self, env: Env, t3: FakeT3Server, capsys: pytest.CaptureFixture[str]
    ) -> None:
        t3.tool_replies['t3_thread_read'] = thread_read(status='running')
        record_dir = spawn_record(env)
        write_transcript(env, SESSION_ID, trailer(record_dir))
        owner = own_identity()
        hand_record(env, SESSION_ID, owner_pid=os.getpid(), owner_identity=owner)

        self._await(capsys, env, record_dir, since=time.time() - 60)

        record = sr.read_record(Path(record_dir).name, root=env.fleet)
        assert record.claude_owner_pid == os.getpid()
        assert record.claude_owner_identity == owner

    def test_started_binds_without_a_pid_when_no_hand_launched_record_exists(
        self, env: Env, t3: FakeT3Server, capsys: pytest.CaptureFixture[str]
    ) -> None:
        t3.tool_replies['t3_thread_read'] = thread_read(status='waiting')
        record_dir = spawn_record(env)
        write_transcript(env, SESSION_ID, trailer(record_dir))

        line = self._await(capsys, env, record_dir)

        assert line == f'started sid={SESSION_ID} hand_slug=none bound=yes via=transcript'
        assert sr.read_record(Path(record_dir).name, root=env.fleet).claude_owner_pid is None

    def test_a_session_already_pointing_at_the_spawn_record_folds_nothing(
        self, env: Env, t3: FakeT3Server, capsys: pytest.CaptureFixture[str]
    ) -> None:
        t3.tool_replies['t3_thread_read'] = thread_read(status='running')
        record_dir = spawn_record(env)
        slug = Path(record_dir).name
        record = sr.read_record(slug, root=env.fleet)
        record.claude_session_id = SESSION_ID
        sr.write_record(record, root=env.fleet)
        sr.write_claude_session_pointer(SESSION_ID, slug, root=env.fleet)
        write_transcript(env, SESSION_ID, trailer(record_dir))

        line = self._await(capsys, env, record_dir)

        assert line == f'started sid={SESSION_ID} hand_slug=none bound=yes via=transcript'
        assert sr.read_record(slug, root=env.fleet).status == sr.Status.LAUNCHING

    def test_started_with_no_transcript_naming_the_record_is_unbound(
        self, env: Env, t3: FakeT3Server, capsys: pytest.CaptureFixture[str]
    ) -> None:
        t3.tool_replies['t3_thread_read'] = thread_read(status='running')
        record_dir = spawn_record(env)
        write_transcript(env, 'not-in-first-message', 'hello', later_text=trailer(record_dir))
        write_transcript(env, 'too-old', trailer(record_dir), mtime=time.time() - 3600)
        write_transcript(env, 'prefix-only', trailer(record_dir + '1'))
        write_transcript(env, 'mention-only', f'Read {record_dir}/result.md and summarise it.')

        line = self._await(capsys, env, record_dir, timeout='1', since=time.time() - 60)

        assert line == 'started sid=none hand_slug=none bound=no via=none'
        assert sr.read_record(Path(record_dir).name, root=env.fleet).claude_session_id is None

    def test_started_without_a_record_skips_discovery(
        self, env: Env, t3: FakeT3Server, capsys: pytest.CaptureFixture[str]
    ) -> None:
        t3.tool_replies['t3_thread_read'] = thread_read(status='completed')

        assert self._await(capsys, env, '') == 'started sid=none hand_slug=none bound=no via=none'

    @pytest.mark.parametrize('status', sorted(lane.STARTUP_FAILURE_STATUSES))
    def test_a_failure_status_before_any_transcript_is_failed_to_start(
        self, env: Env, t3: FakeT3Server, capsys: pytest.CaptureFixture[str], status: str
    ) -> None:
        t3.tool_replies['t3_thread_read'] = thread_read(status=status)

        assert self._await(capsys, env, spawn_record(env)) == 'failed-to-start'
        assert self._await(capsys, env, '') == 'failed-to-start'

    def test_a_failure_status_after_the_transcript_appeared_is_started(
        self, env: Env, t3: FakeT3Server, capsys: pytest.CaptureFixture[str]
    ) -> None:
        t3.tool_replies['t3_thread_read'] = thread_read(status='failed')
        record_dir = spawn_record(env)
        write_transcript(env, SESSION_ID, trailer(record_dir))

        assert self._await(capsys, env, record_dir) == (
            f'started sid={SESSION_ID} hand_slug=none bound=yes via=transcript'
        )

    def test_an_interrupted_run_is_started(
        self, env: Env, t3: FakeT3Server, capsys: pytest.CaptureFixture[str]
    ) -> None:
        t3.tool_replies['t3_thread_read'] = thread_read(status='interrupted')

        assert self._await(capsys, env, '') == 'started sid=none hand_slug=none bound=no via=none'

    def test_a_thread_that_never_leaves_preparation_times_out(
        self, env: Env, t3: FakeT3Server, capsys: pytest.CaptureFixture[str]
    ) -> None:
        t3.tool_replies['t3_thread_read'] = thread_read(status='queued')

        assert self._await(capsys, env, spawn_record(env), timeout='0.3') == 'timeout'

    def test_a_thread_t3_does_not_know_times_out(
        self, env: Env, t3: FakeT3Server, capsys: pytest.CaptureFixture[str]
    ) -> None:
        t3.tool_replies['t3_thread_read'] = ToolFailure('thread_not_found', f'Thread {THREAD_ID} was not found.')

        assert self._await(capsys, env, '', timeout='0.3') == 'timeout'

    def test_a_server_hanging_up_times_out(
        self, env: Env, t3: FakeT3Server, capsys: pytest.CaptureFixture[str]
    ) -> None:
        t3.hangs_up = True

        assert self._await(capsys, env, '', timeout='0.3') == 'timeout'

    def test_a_stalled_thread_read_times_out_within_the_verbs_bound(
        self, env: Env, t3: FakeT3Server, stalling: StallThen, capsys: pytest.CaptureFixture[str]
    ) -> None:
        t3.tool_replies['t3_thread_read'] = stalling(thread_read(status='queued'))

        started = time.monotonic()

        assert self._await(capsys, env, '', timeout='1.0') == 'timeout'
        assert time.monotonic() - started < STALL_CAP_SECS / 2

    def test_started_folds_a_hand_record_that_claimed_the_thread_after_the_stamp(
        self, env: Env, t3: FakeT3Server, capsys: pytest.CaptureFixture[str]
    ) -> None:
        t3.tool_replies['t3_thread_read'] = thread_read(status='running')
        record_dir = spawn_record(env)
        slug = Path(record_dir).name
        sr.update_display(slug, root=env.fleet, kind='t3', t3_thread_id=THREAD_ID)
        hand = t3_session_start(env)

        line = self._await(capsys, env, record_dir)

        assert line == (
            f'started sid={SESSION_ID} hand_slug={hand.session_slug} bound=yes via=thread-pointer'
        )
        assert_folded_into(env, slug, hand)

    def test_started_reports_a_record_its_hook_bound_through_the_thread_pointer(
        self, env: Env, t3: FakeT3Server, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        t3.tool_replies['t3_thread_read'] = thread_read(status='running')
        record_dir = spawn_record(env)
        launch_line(capsys, tmp_path, env.project, record=record_dir)
        adopted = t3_session_start(env)

        line = self._await(capsys, env, record_dir)

        assert adopted.session_slug == Path(record_dir).name
        assert line == f'started sid={SESSION_ID} hand_slug=none bound=yes via=thread-pointer'

    def test_the_read_names_the_thread(
        self, env: Env, t3: FakeT3Server, capsys: pytest.CaptureFixture[str]
    ) -> None:
        t3.tool_replies['t3_thread_read'] = thread_read(status='running')

        self._await(capsys, env, '')

        reads = t3.tool_calls('t3_thread_read')
        assert reads and all(read['threadId'] == THREAD_ID for read in reads)


T3_CLAUDE_PID = 424242


def t3_session_start(env: Env) -> sr.SessionRecord:
    """The real SessionStart hook of claude ``T3_CLAUDE_PID`` in thread ``THREAD_ID``, as T3 starts it."""
    return session_hooks.run_session_start(
        {'session_id': SESSION_ID, 'cwd': str(env.project)},
        {'T3CODE_THREAD_ID': THREAD_ID},
        root=env.fleet,
        process_tree=claude_process_tree(T3_CLAUDE_PID),
    )


def assert_folded_into(env: Env, slug: str, hand: sr.SessionRecord) -> None:
    """The spawn record *slug* carries *hand*'s session and owner, every pointer names it, and *hand* is folded."""
    record = sr.read_record(slug, root=env.fleet)
    assert (record.claude_session_id, record.claude_launch_session_id) == (SESSION_ID, SESSION_ID)
    assert (record.claude_owner_pid, record.claude_owner_identity) == (
        T3_CLAUDE_PID,
        hand.claude_owner_identity,
    )
    folded = sr.read_record(hand.session_slug, root=env.fleet)
    assert (folded.status, folded.exit_code) == (sr.Status.EXITED, sr.FOLDED_EXIT_CODE)
    assert sr.resolve_session_slug_for_t3_thread(THREAD_ID, root=env.fleet) == slug
    assert sr.resolve_session_slug_for_claude_session_id(SESSION_ID, root=env.fleet) == slug
    assert sr.resolve_session_slug_for_pid(T3_CLAUDE_PID, root=env.fleet) == slug


class TestThreadPointerReconcile:
    def test_launch_folds_a_hand_record_whose_session_start_claimed_the_thread_first(
        self, env: Env, t3: FakeT3Server, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        hands: list[sr.SessionRecord] = []

        def launch_while_the_child_starts(arguments: Mapping[str, Any]) -> ToolSuccess:
            hands.append(t3_session_start(env))
            return ToolSuccess(THREAD_LAUNCH_JSON)

        t3.tool_replies['t3_thread_launch'] = launch_while_the_child_starts
        record_dir = spawn_record(env)

        launch_line(capsys, tmp_path, env.project, record=record_dir)

        [hand] = hands
        assert sr.is_t3_record(hand)
        assert_folded_into(env, Path(record_dir).name, hand)

    def test_a_thread_pointer_another_spawn_holds_is_left_alone(self, env: Env) -> None:
        record_dir = spawn_record(env)
        other_dir = spawn_record(env, 'review-df-7000-777')
        other = sr.read_record('review-df-7000-777', root=env.fleet)
        other.display = sr.Display(kind='t3', t3_thread_id=THREAD_ID)
        other.claude_session_id, other.status = SESSION_ID, sr.Status.RUNNING
        sr.write_record(other, root=env.fleet)
        sr.write_t3_thread_pointer(THREAD_ID, 'review-df-7000-777', root=env.fleet)

        binding = lane.bind_thread_pointer(record_dir, THREAD_ID)

        assert (binding.bound, binding.via) == (False, lane.BindingSource.NONE)
        assert sr.read_record(Path(other_dir).name, root=env.fleet).status == sr.Status.RUNNING
        assert sr.read_record(Path(record_dir).name, root=env.fleet).claude_session_id is None
        assert sr.resolve_session_slug_for_t3_thread(THREAD_ID, root=env.fleet) == (
            'review-df-7000-777'
        )

    def test_the_transcript_path_folds_a_hand_launched_t3_record(self, env: Env) -> None:
        record_dir = spawn_record(env)
        hand = t3_session_start(env)

        binding = lane.bind_session(record_dir, lane.Transcript(SESSION_ID, Path('unused'), 0.0))

        assert (binding.bound, binding.hand_slug, binding.via) == (
            True,
            hand.session_slug,
            lane.BindingSource.TRANSCRIPT,
        )
        folded = sr.read_record(hand.session_slug, root=env.fleet)
        assert (folded.status, folded.exit_code) == (sr.Status.EXITED, sr.FOLDED_EXIT_CODE)


def test_a_spilled_launch_message_is_discovered(env: Env) -> None:
    record_dir = spawn_record(env)
    pointer = f'Read {record_dir}/prompt.md and follow it as your brief; that file holds your whole task.'
    write_transcript(env, SESSION_ID, pointer)

    found = lane.discover_transcript(str(env.project), record_dir, 0.0)

    assert found is not None and found.session_id == SESSION_ID


def test_a_session_another_spawn_owns_is_neither_bound_nor_folded(env: Env) -> None:
    record_dir = spawn_record(env)
    other_dir = spawn_record(env, 'review-df-7000-777')
    other = sr.read_record('review-df-7000-777', root=env.fleet)
    other.result_file = f'{other_dir}/result.md'
    other.claude_session_id, other.status = SESSION_ID, sr.Status.RUNNING
    sr.write_record(other, root=env.fleet)
    sr.write_claude_session_pointer(SESSION_ID, 'review-df-7000-777', root=env.fleet)

    binding = lane.bind_session(record_dir, lane.Transcript(SESSION_ID, Path('unused'), 0.0))

    assert (binding.bound, binding.hand_slug) == (False, None)
    assert sr.read_record('review-df-7000-777', root=env.fleet).status == sr.Status.RUNNING
    assert sr.read_record(Path(record_dir).name, root=env.fleet).claude_session_id is None
    assert sr.resolve_session_slug_for_claude_session_id(SESSION_ID, root=env.fleet) == (
        'review-df-7000-777'
    )


class TestRebinding:
    def test_a_newer_transcript_naming_the_record_rebinds_it(self, env: Env) -> None:
        record_dir = spawn_record(env)
        now = time.time()
        write_transcript(env, 'first-session', trailer(record_dir), mtime=now - 600)
        lane.bind_session(record_dir, lane.Transcript('first-session', Path('unused'), now - 600))
        write_transcript(env, 'handoff-session', trailer(record_dir), mtime=now)
        hand_record(env, 'handoff-session', owner_pid=515151)

        lane.maintain_binding(record_dir, str(env.project))

        record = sr.read_record(Path(record_dir).name, root=env.fleet)
        assert (record.claude_session_id, record.claude_owner_pid) == ('handoff-session', 515151)
        assert record.claude_launch_session_id == 'first-session'

    def test_a_remint_the_hooks_bound_is_not_undone(self, env: Env) -> None:
        record_dir = spawn_record(env)
        slug = Path(record_dir).name
        now = time.time()
        write_transcript(env, 'launch-session', trailer(record_dir), mtime=now - 600)
        write_transcript(env, 'cleared-session', '/clear', mtime=now)
        record = sr.read_record(slug, root=env.fleet)
        record.claude_session_id, record.claude_owner_pid = 'cleared-session', 616161
        sr.write_record(record, root=env.fleet)

        lane.maintain_binding(record_dir, str(env.project))

        assert sr.read_record(slug, root=env.fleet).claude_session_id == 'cleared-session'

    def test_a_remint_whose_transcript_is_not_written_yet_is_not_undone(self, env: Env) -> None:
        record_dir = spawn_record(env)
        slug = Path(record_dir).name
        write_transcript(env, 'launch-session', trailer(record_dir))
        record = sr.read_record(slug, root=env.fleet)
        record.claude_session_id = 'cleared-session-unwritten'
        sr.write_record(record, root=env.fleet)

        lane.maintain_binding(record_dir, str(env.project))

        assert sr.read_record(slug, root=env.fleet).claude_session_id == 'cleared-session-unwritten'

    def test_a_transcript_under_claude_config_dir_is_discovered(
        self, env: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config_dir = tmp_path / 'claude-home'
        monkeypatch.setenv('CLAUDE_CONFIG_DIR', str(config_dir))
        record_dir = spawn_record(env)
        directory = config_dir / 'projects' / sr.encode_cwd(str(env.project))
        directory.mkdir(parents=True)
        entry = {'type': 'user', 'message': {'role': 'user', 'content': trailer(record_dir)}}
        (directory / f'{SESSION_ID}.jsonl').write_text(json.dumps(entry) + '\n')

        found = lane.discover_transcript(str(env.project), record_dir, 0.0)

        assert found is not None and found.session_id == SESSION_ID


# ---------------------------------------------------------------------------
# wait
# ---------------------------------------------------------------------------


Present = lane.ThreadObservation(lane.ThreadState.PRESENT, 'running', RUN_ID)


def _ended(run_id: str = RUN_ID, status: str = 'failed') -> lane.ThreadObservation:
    return lane.ThreadObservation(lane.ThreadState.PRESENT, status, run_id)


class TestJoinTracker:
    def test_a_final_result_finishes_zero_even_while_running(self) -> None:
        assert lane.JoinTracker().verdict(Present, result_final=True, now=0) == 0

    def test_an_archived_thread_finishes_zero(self) -> None:
        archived = lane.ThreadObservation(lane.ThreadState.ARCHIVED)
        assert lane.JoinTracker().verdict(archived, result_final=False, now=0) == 0

    def test_a_gone_thread_finishes_127(self) -> None:
        gone = lane.ThreadObservation(lane.ThreadState.GONE)
        assert lane.JoinTracker().verdict(gone, result_final=False, now=0) == 127

    @pytest.mark.parametrize('status', ['idle', 'waiting', 'running', 'completed', 'queued'])
    def test_live_or_idle_keeps_waiting(self, status: str) -> None:
        tracker = lane.JoinTracker()
        observation = lane.ThreadObservation(lane.ThreadState.PRESENT, status, RUN_ID)
        assert tracker.verdict(observation, result_final=False, now=0) is None
        assert tracker.verdict(observation, result_final=False, now=10_000) is None

    @pytest.mark.parametrize('status', sorted(lane.ENDED_RUN_STATUSES))
    def test_an_ended_run_finishes_one_after_the_grace(self, status: str) -> None:
        tracker = lane.JoinTracker()
        assert tracker.verdict(_ended(status=status), result_final=False, now=0) is None
        assert tracker.verdict(_ended(status=status), result_final=False, now=119) is None
        assert tracker.verdict(_ended(status=status), result_final=False, now=120) == 1

    def test_a_newer_run_within_the_grace_resets_it(self) -> None:
        tracker = lane.JoinTracker()
        tracker.verdict(_ended('run-1'), result_final=False, now=0)
        assert tracker.verdict(Present, result_final=False, now=60) is None
        assert tracker.verdict(_ended('run-2'), result_final=False, now=130) is None
        assert tracker.verdict(_ended('run-2'), result_final=False, now=249) is None
        assert tracker.verdict(_ended('run-2'), result_final=False, now=250) == 1

    def test_a_final_result_beats_an_ended_run(self) -> None:
        tracker = lane.JoinTracker()
        tracker.verdict(_ended(), result_final=False, now=0)
        assert tracker.verdict(_ended(), result_final=True, now=500) == 0

    def test_unreachable_past_the_grace_gives_up_126(self) -> None:
        tracker = lane.JoinTracker()
        down = lane.ThreadObservation(lane.ThreadState.UNREACHABLE)
        assert tracker.verdict(down, result_final=False, now=0) is None
        assert tracker.verdict(down, result_final=False, now=600) is None
        assert tracker.verdict(down, result_final=False, now=601) == 126

    def test_a_successful_listing_resets_the_unreachable_clock(self) -> None:
        tracker = lane.JoinTracker()
        down = lane.ThreadObservation(lane.ThreadState.UNREACHABLE)
        tracker.verdict(down, result_final=False, now=0)
        tracker.verdict(Present, result_final=False, now=500)
        assert tracker.verdict(down, result_final=False, now=700) is None
        assert tracker.verdict(down, result_final=False, now=1300) is None
        assert tracker.verdict(down, result_final=False, now=1301) == 126


@dataclass
class FakeClock:
    now: float = 0.0
    sleeps: list[float] = field(default_factory=list)

    def monotonic(self) -> float:
        return self.now

    def sleep(self, secs: float) -> None:
        self.sleeps.append(secs)
        self.now += secs


def wait_for(env: Env, record_dir: str, clock: FakeClock, **options: Any) -> str:
    return lane.wait(
        THREAD_ID, str(env.project), record_dir, os.environ, clock=clock.monotonic, sleep=clock.sleep, **options
    )


class TestWait:
    def test_a_slice_that_times_out_is_followed_at_once_by_the_next(self, env: Env, t3: FakeT3Server) -> None:
        record_dir = spawn_record(env)

        def the_child_finishes(arguments: Mapping[str, Any]) -> ToolSuccess:
            (Path(record_dir) / 'result.md').write_text('---\noutcome: done\n---\n')
            return waited(timed_out=False, status='completed')

        t3.tool_replies |= {
            't3_thread_read': thread_read(status='running'),
            't3_thread_wait': replies_in_turn(waited(timed_out=True), waited(timed_out=True), the_child_finishes),
        }
        clock = FakeClock()

        assert wait_for(env, record_dir, clock) == 'finished code=0'
        assert t3.tool_calls('t3_thread_wait') == [{'threadId': THREAD_ID, 'timeoutMs': 600_000}] * 3
        assert clock.sleeps == []
        assert len(t3.tool_calls('t3_thread_list')) == 4

    def test_an_ended_run_is_rechecked_after_a_pause_until_its_grace_passes(
        self, env: Env, t3: FakeT3Server
    ) -> None:
        t3.tool_replies |= {
            't3_thread_read': thread_read(status='failed', latestRunId=RUN_ID),
            't3_thread_wait': waited(timed_out=False, status='failed'),
        }
        clock = FakeClock()

        assert wait_for(env, spawn_record(env), clock) == 'finished code=1'
        assert clock.sleeps and set(clock.sleeps) == {lane.SETTLED_RECHECK_SECS}
        assert clock.now >= lane.ENDED_RUN_GRACE_SECS

    def test_a_server_down_past_its_grace_is_126(self, env: Env, t3: FakeT3Server) -> None:
        t3.hangs_up = True
        clock = FakeClock()

        assert wait_for(env, '', clock) == 'finished code=126'
        assert clock.now > lane.UNREACHABLE_GRACE_SECS
        assert set(clock.sleeps) == {lane.SETTLED_RECHECK_SECS}

    def test_a_self_pair_that_fails_is_not_retried_on_later_passes(
        self, env: Env, t3: FakeT3Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        forget_credential(env)
        put_fake_t3_cli_on_path(t3, tmp_path, monkeypatch, version_line='t3 v0.1.6')
        t3.decision_state_override = 'forged'

        assert wait_for(env, '', FakeClock()) == 'finished code=126'
        assert len(t3.issued_pairing_codes()) == 1

    def test_a_stalled_thread_read_counts_as_unreachable_for_that_pass(
        self, env: Env, t3: FakeT3Server, stalling: StallThen, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(lane, 'CALL_TIMEOUT_SECS', 1.0)
        t3.tool_replies |= {
            't3_thread_read': replies_in_turn(stalling(thread_read(archived=True)), thread_read(archived=True)),
            't3_thread_wait': waited(timed_out=True),
        }

        started = time.monotonic()

        assert wait_for(env, spawn_record(env), FakeClock()) == 'finished code=0'
        assert time.monotonic() - started < STALL_CAP_SECS / 2

    def test_wait_stops_at_a_slice_boundary_once_its_caller_is_gone(self, env: Env, t3: FakeT3Server) -> None:
        t3.tool_replies |= {'t3_thread_read': thread_read(), 't3_thread_wait': waited(timed_out=True)}
        parents = iter([4242, 4242, 4242, 1])

        with pytest.raises(lane.CallerGone):
            wait_for(env, '', FakeClock(), parent_pid=lambda: next(parents))

        assert len(t3.tool_calls('t3_thread_wait')) == 2

    def test_a_thread_t3_does_not_know_finishes_127(self, env: Env, t3: FakeT3Server) -> None:
        t3.tool_replies['t3_thread_read'] = ToolFailure('thread_not_found', f'Thread {THREAD_ID} was not found.')

        assert wait_for(env, '', FakeClock()) == 'finished code=127'

    def test_an_archived_thread_finishes_zero(
        self, env: Env, t3: FakeT3Server, capsys: pytest.CaptureFixture[str]
    ) -> None:
        t3.tool_replies['t3_thread_read'] = thread_read(archived=True)

        line = run_verb(capsys, 'wait', '--thread-id', THREAD_ID, '--cwd', str(env.project), '--record', '')

        assert line == 'finished code=0'

    def test_a_settled_thread_is_rechecked_after_a_pause_until_archived(self, env: Env, t3: FakeT3Server) -> None:
        settled = thread_read(status='idle', settled=True, settledAt='2026-10-25T09:00:00.000Z')
        t3.tool_replies |= {
            't3_thread_read': replies_in_turn(settled, settled, thread_read(archived=True)),
            't3_thread_wait': waited(timed_out=False, status='completed'),
        }
        record_dir = spawn_record(env)
        (Path(record_dir) / 'result.md').write_text('outcome: in-progress\n')
        clock = FakeClock()

        assert wait_for(env, record_dir, clock) == 'finished code=0'
        assert clock.sleeps == [lane.SETTLED_RECHECK_SECS] * 2

    def test_each_pass_sweeps_the_environment_wide_listing(self, env: Env, t3: FakeT3Server) -> None:
        stale_dir = stale_t3_record(env)
        t3.tool_replies['t3_thread_read'] = thread_read(archived=True, projectId=None)

        assert wait_for(env, '', FakeClock()) == 'finished code=0'
        stale = sr.read_record(Path(stale_dir).name, root=env.fleet)
        assert (stale.status, stale.exit_code) == (sr.Status.EXITED, sr.ORPHAN_EXIT_CODE)
        assert t3.tool_calls('t3_thread_list') == [{'allProjects': True, 'limit': 100}]

    def test_a_pass_whose_sweep_marks_nothing_logs_its_line_at_debug(
        self, env: Env, t3: FakeT3Server, caplog: pytest.LogCaptureFixture
    ) -> None:
        t3.tool_replies['t3_thread_read'] = thread_read(archived=True)

        with caplog.at_level(logging.DEBUG, logger='orchestrator.session_registry'):
            assert wait_for(env, '', FakeClock()) == 'finished code=0'

        assert sweep_line_levels(caplog) == [logging.DEBUG]

    def test_a_pass_leaves_an_unlisted_thread_that_still_reads_present(self, env: Env, t3: FakeT3Server) -> None:
        stale_dir = stale_t3_record(env)

        def read_per_thread(arguments: Mapping[str, Any]) -> ToolSuccess:
            thread_id = arguments['threadId']
            return thread_read(threadId=thread_id, archived=thread_id == THREAD_ID)

        t3.tool_replies['t3_thread_read'] = read_per_thread

        assert wait_for(env, '', FakeClock()) == 'finished code=0'
        assert sr.read_record(Path(stale_dir).name, root=env.fleet).status == sr.Status.LAUNCHING

    def test_an_incomplete_listing_sweeps_nothing_while_waiting(self, env: Env, t3: FakeT3Server) -> None:
        stale_dir = stale_t3_record(env)
        t3.tool_replies |= {
            't3_thread_read': thread_read(archived=True),
            't3_thread_list': replies_in_turn(
                ToolSuccess({**THREAD_LIST_JSON, 'nextCursor': 1}), ToolFailure('orchestration_error', 'page lost')
            ),
        }

        assert wait_for(env, '', FakeClock()) == 'finished code=0'
        assert sr.read_record(Path(stale_dir).name, root=env.fleet).status == sr.Status.LAUNCHING

    def test_an_unbound_record_is_bound_while_waiting(self, env: Env, t3: FakeT3Server) -> None:
        record_dir = spawn_record(env)
        (Path(record_dir) / 'result.md').write_text('outcome: done\n')
        write_transcript(env, SESSION_ID, trailer(record_dir))
        t3.tool_replies['t3_thread_read'] = thread_read()

        assert wait_for(env, record_dir, FakeClock()) == 'finished code=0'
        assert sr.read_record(Path(record_dir).name, root=env.fleet).claude_session_id == SESSION_ID

    def test_main_reports_a_gone_caller_on_stderr_only(
        self, env: Env, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def gone(*_args: object, **_kwargs: object) -> str:
            raise lane.CallerGone('caller 4242 is gone')

        monkeypatch.setattr(lane, 'wait', gone)

        rc = lane.main(['wait', '--thread-id', THREAD_ID, '--cwd', str(env.project), '--record', ''])

        captured = capsys.readouterr()
        assert (rc, captured.out) == (lane.INTERNAL_FAULT_EXIT, '')
        assert captured.err == 'spawn_t3_lane: caller 4242 is gone\n'


# ---------------------------------------------------------------------------
# self-record and the CLI contract
# ---------------------------------------------------------------------------


def _tree(parents: dict[int, int], comms: dict[int, str]) -> ProcessTree:
    lookup_parent: Callable[[int], int | None] = parents.get
    lookup_comm: Callable[[int], str | None] = comms.get
    return ProcessTree(
        hook_parent_pid=lambda: 300, parent_pid_of=lookup_parent, comm_of=lookup_comm
    )


class TestSelfRecord:
    def test_resolves_the_nearest_claude_ancestors_record_and_its_parent(self, env: Env) -> None:
        sr.write_record(
            sr.SessionRecord(
                session_slug='session-df-abc', status=sr.Status.RUNNING,
                claude_owner_pid=100, parent_session_id='watcher-df-9',
            ),
            root=env.fleet,
        )
        sr.write_session_pointer(100, 'session-df-abc', root=env.fleet)
        tree = _tree({300: 200, 200: 100, 100: 1}, {300: 'bash', 200: 'bash', 100: 'claude'})

        assert lane.self_record(tree) == 'slug=session-df-abc parent=watcher-df-9'

    def test_no_claude_ancestor_is_none(self, env: Env) -> None:
        tree = _tree({300: 200, 200: 1}, {300: 'bash', 200: 'sshd'})

        assert lane.self_record(tree) == 'slug=none parent=none'

    def test_an_unvouched_pointer_is_none(self, env: Env) -> None:
        sr.write_session_pointer(100, 'session-df-abc', root=env.fleet)
        tree = _tree({300: 100}, {300: 'bash', 100: 'claude'})

        assert lane.self_record(tree) == 'slug=none parent=none'


def test_a_path_with_a_space_is_percent_escaped(
    t3: FakeT3Server, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spaced = tmp_path / 'with space'
    spaced.mkdir()

    line = launch_line(capsys, tmp_path, spaced)

    assert line == 'refused reason=cwd-outside-projects'
    assert lane._verdict('x', detail='/a b') == 'x detail=/a%20b'


def test_an_internal_fault_exits_3_with_nothing_on_stdout(
    env: Env, t3: FakeT3Server, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = lane.main([
        'launch', '--cwd', str(env.project), '--title', 't',
        '--prompt-file', str(tmp_path / 'missing'), '--record', '',
    ])

    captured = capsys.readouterr()
    assert (rc, captured.out) == (lane.INTERNAL_FAULT_EXIT, '')
    assert 'Traceback' in captured.err


def test_the_module_runs_by_path_with_only_pythonpath(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(Path(lane.__file__)), 'detect'],
        env={'PATH': '/usr/bin:/bin', 'PYTHONPATH': str(_PKG_SRC), 'HOME': str(tmp_path),
             'T3CODE_HOME': str(tmp_path / 'none')},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert (completed.returncode, completed.stdout) == (0, 'not-t3\n'), completed.stderr


class TestPairAndStatus:
    def test_pair_self_pairs_with_the_runtime_origin(
        self,
        env: Env,
        t3_server: FakeT3Server,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        point_runtime_at(env, t3_server)
        put_fake_t3_cli_on_path(t3_server, tmp_path, monkeypatch, version_line='t3 v0.1.6')

        line = run_verb(capsys, 'pair')

        credential = stored_credential(env)
        assert line == (
            f'paired environment_id={ENVIRONMENT_ID} expires_at={credential.expires_at.isoformat()} access=full-access'
        )
        path = env.fleet / 't3-mcp' / f'{ENVIRONMENT_ID}.json'
        assert path.stat().st_mode & 0o777 == 0o600
        assert json.loads(path.read_text())['paired_by'] == 'self-pair'

    def test_pair_with_a_cli_of_another_version_refuses_and_writes_nothing(
        self,
        env: Env,
        t3_server: FakeT3Server,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        point_runtime_at(env, t3_server)
        put_fake_t3_cli_on_path(t3_server, tmp_path, monkeypatch, version_line='t3 v0.1.4')

        line = run_verb(capsys, 'pair')

        assert line == 'pair-failed reason=cli-version-mismatch cli=0.1.4 server=0.1.6'
        assert not (env.fleet / 't3-mcp').exists()

    def test_pair_with_an_operator_code_needs_neither_runtime_nor_cli(
        self,
        env: Env,
        t3_server: FakeT3Server,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        remove_runtime(env)
        empty = tmp_path / 'empty-bin'
        empty.mkdir()
        monkeypatch.setenv('PATH', str(empty))

        paired = run_verb(capsys, 'pair', '--code', t3_server.add_pairing_code(), '--origin', t3_server.origin)
        rejected = run_verb(capsys, 'pair', '--code', 'wrong', '--origin', t3_server.origin)

        assert paired.startswith(f'paired environment_id={ENVIRONMENT_ID} ')
        assert stored_credential(env).paired_by is t3_mcp_client.PairedBy.OPERATOR
        assert rejected.startswith('pair-failed reason=pairing_rejected')

    def test_pair_with_no_origin_anywhere_says_so(self, env: Env, capsys: pytest.CaptureFixture[str]) -> None:
        remove_runtime(env)

        assert run_verb(capsys, 'pair') == 'pair-failed reason=server-runtime-missing'

    def test_pair_refuses_plain_http_off_loopback_at_the_argument(
        self, env: Env, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as exited:
            lane.main(['pair', '--origin', 'http://192.168.1.105:3773'])

        assert exited.value.code == 2
        captured = capsys.readouterr()
        assert captured.out == ''
        assert 'plain http is allowed only to a loopback host' in captured.err

    @pytest.mark.parametrize(
        ('origin', 'token'),
        [('http://192.168.1.105:3773', 'http://192.168.1.105:3773'), (None, 'none')],
        ids=['off-loopback-http', 'null-origin'],
    )
    def test_a_runtime_origin_present_but_unusable_is_named_not_called_missing(
        self, env: Env, capsys: pytest.CaptureFixture[str], origin: str | None, token: str
    ) -> None:
        write_runtime_origin(env, origin)

        pair = run_verb(capsys, 'pair')
        status = run_verb(capsys, 'status')

        assert pair.startswith(f'pair-failed reason=server-runtime-origin-unusable origin={token} detail=')
        assert status.startswith(f'status-failed reason=server-runtime-origin-unusable origin={token} detail=')
        assert not pair.endswith('detail=none')
        assert not (env.fleet / 't3-mcp').exists()

    def test_status_reports_paired_unpaired_and_expired(
        self,
        env: Env,
        t3_server: FakeT3Server,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        point_runtime_at(env, t3_server)
        put_fake_t3_cli_on_path(t3_server, tmp_path, monkeypatch, version_line='t3 v0.1.6')
        run_verb(capsys, 'pair')
        path = env.fleet / 't3-mcp' / f'{ENVIRONMENT_ID}.json'
        expires_at = stored_credential(env).expires_at.isoformat()

        paired = run_verb(capsys, 'status')
        stored = json.loads(path.read_text())
        path.unlink()
        unpaired = run_verb(capsys, 'status')
        past = (datetime.now(UTC) - timedelta(days=1)).replace(microsecond=0).isoformat()
        path.write_text(json.dumps({**stored, 'expires_at': past}))
        expired = run_verb(capsys, 'status')

        origin = t3_server.origin
        assert paired == f'paired environment_id={ENVIRONMENT_ID} expires_at={expires_at} origin={origin}'
        assert unpaired == f'unpaired environment_id={ENVIRONMENT_ID} expires_at=none origin={origin}'
        assert expired == f'expired environment_id={ENVIRONMENT_ID} expires_at={past} origin={origin}'

    def test_status_without_a_descriptor_cannot_select_a_file(
        self, env: Env, capsys: pytest.CaptureFixture[str]
    ) -> None:
        origin = DEAD_ORIGIN

        unreachable = run_verb(capsys, 'status', '--origin', origin)
        remove_runtime(env)
        no_runtime = run_verb(capsys, 'status')

        assert unreachable.startswith(f'status-failed reason=descriptor-unreachable origin={origin} detail=')
        assert not unreachable.endswith('detail=none')
        assert no_runtime == 'status-failed reason=server-runtime-missing origin=none'

    def test_status_of_a_server_answering_an_unusable_descriptor(
        self, env: Env, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with FakeT3Server.running(serves_descriptor=False) as server:
            line = run_verb(capsys, 'status', '--origin', server.origin)

        assert line.startswith(f'status-failed reason=descriptor-incompatible origin={server.origin} detail=')

    def test_pair_against_an_unusable_descriptor_writes_nothing(
        self, env: Env, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with FakeT3Server.running(serves_descriptor=False) as server:
            line = run_verb(capsys, 'pair', '--code', 'any', '--origin', server.origin)

        assert line.startswith('pair-failed reason=descriptor-incompatible detail=')
        assert not (env.fleet / 't3-mcp').exists()
