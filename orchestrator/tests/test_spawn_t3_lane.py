"""Tests for orchestrator.spawn_t3_lane, driven through its verbs with a fake ``t3`` on PATH.

The fake's JSON shapes are the module-level ``*_JSON`` constants below, copied
from ``~/src/t3code`` v0.1.2: ``apps/server/src/cli/thread.ts`` (what each
command prints) and ``packages/contracts/src/orchestratorMcp.ts`` (the item
schemas). ``FAKE_T3_SOURCE`` reads its scripted responses from
``$FAKE_T3_SCRIPT`` and logs every call to ``$FAKE_T3_LOG``.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import textwrap
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from orchestrator import session_registry as sr
from orchestrator import spawn_t3_lane as lane
from orchestrator.session_ownership_probes import ProcessTree

_PKG_SRC = Path(lane.__file__).resolve().parents[1]

THREAD_ID = '6f0c2a3e-7489-4c11-9a51-0d6c1f3b2a10'
PROJECT_ID = '5a05e1d3-5bd0-40e5-8ce7-24b2cd160300'
RUN_ID = 'b2c4e6f8-0a1c-4e3a-8b5d-7f9e1a2b3c4d'
SESSION_ID = '15fbcf7c-12af-481c-a641-9b00c03a741e'

THREAD_START_JSON: dict[str, Any] = {'threadId': THREAD_ID, 'projectId': PROJECT_ID, 'runId': RUN_ID}
"""``t3 thread start --json`` without ``--wait``: thread.ts ``ThreadLaunchResult``; runId may be null."""

THREAD_LIST_ITEM_JSON: dict[str, Any] = {
    'threadId': THREAD_ID,
    'title': 'unblock:df#7489 t3-lane',
    'createdBy': 'user',
    'creationSource': 'mcp',
    'status': 'running',
    'latestRunId': RUN_ID,
    'providerInstanceId': 'claudeAgent',
    'model': 'claude-opus-5-5',
    'runtimeMode': 'full-access',
    'interactionMode': 'default',
    'linkedPullRequest': None,
    'pullRequests': [],
    'settled': False,
    'settledAt': None,
    'snoozed': False,
    'snoozedUntil': None,
    'parentThreadId': None,
    'relationshipToParent': None,
    'itemCount': 3,
    'createdAt': '2026-10-25T08:04:00.000Z',
    'updatedAt': '2026-10-25T08:05:43.000Z',
}
"""One element of ``t3 thread list --json``, which prints a bare JSON ARRAY of these
(``OrchestratorMcpThreadListItem``); archived threads are never listed. ``status`` is
``idle`` or a run status: preparing|queued|starting|running|waiting|completed|
interrupted|failed|cancelled|rolled_back."""

THREAD_SHOW_JSON: dict[str, Any] = {
    'thread': {
        'threadId': THREAD_ID,
        'projectId': PROJECT_ID,
        'title': 'unblock:df#7489 t3-lane',
        'status': 'completed',
        'latestRunId': RUN_ID,
        'activeRunId': None,
        'runCount': 1,
        'pendingRequestCount': 0,
        'archived': True,
        'settled': False,
    },
    'messages': [],
}
"""``t3 thread show <id> --json``: ``{thread: OrchestratorMcpThreadDetail, messages}``
(detail abridged to the fields the lane reads, ``archived`` among them)."""

PROJECTS_TABLE_SQL = (
    'CREATE TABLE projection_projects (project_id TEXT PRIMARY KEY, title TEXT NOT NULL, '
    'workspace_root TEXT NOT NULL, deleted_at TEXT)'
)
"""The columns of ``$T3CODE_HOME/userdata/statev2.sqlite`` table ``projection_projects``
that ``launch`` reads for the registered project roots."""

SERVER_RUNTIME_JSON: dict[str, Any] = {
    'version': 1,
    'pid': 881190,
    'port': 3773,
    'origin': 'http://127.0.0.1:3773',
    'startedAt': '2026-10-23T09:09:43.139Z',
    'serviceManaged': True,
    'serverVersion': '0.1.2',
}
"""``$T3CODE_HOME/userdata/server-runtime.json`` as the live server writes it."""

FAKE_T3_SOURCE = textwrap.dedent(
    '''\
    #!/usr/bin/env python3
    """Fake t3: answers `thread start|list|show` from $FAKE_T3_SCRIPT, logs to $FAKE_T3_LOG.

    The script maps "start" / "list" / "show" to a list of responses
    {"exit": int, "stdout": <json or null>, "stderr": str}; each call takes the
    next one and the last repeats. A response with "add_thread" appends that
    list item to every later "list" answer, as a real launch would.
    """
    import json, os, sys
    script_path = os.environ["FAKE_T3_SCRIPT"]
    script = json.load(open(script_path))
    verb = sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "thread" else ""
    stdin = sys.stdin.read() if "-" in sys.argv[3:] else None
    with open(os.environ["FAKE_T3_LOG"], "a") as log:
        log.write(json.dumps({"argv": sys.argv[1:], "stdin": stdin}) + "\\n")
    counts = script.setdefault("_counts", {})
    responses = script.get(verb) or [{"exit": 2, "stdout": None, "stderr": "unscripted"}]
    response = responses[min(counts.get(verb, 0), len(responses) - 1)]
    counts[verb] = counts.get(verb, 0) + 1
    stdout = response.get("stdout")
    if verb == "list" and stdout is not None:
        stdout = stdout + script.get("_added", [])
    if "add_thread" in response:
        script.setdefault("_added", []).append(response["add_thread"])
    json.dump(script, open(script_path, "w"))
    if stdout is not None:
        print(json.dumps(stdout))
    if response.get("stderr"):
        print(response["stderr"], file=sys.stderr)
    sys.exit(response.get("exit", 0))
    '''
)


def ok(stdout: Any) -> dict[str, Any]:
    return {'exit': 0, 'stdout': stdout}


def unreachable() -> dict[str, Any]:
    return {'exit': 1, 'stdout': None, 'stderr': 'Could not reach the T3 Code server.'}


def listed(**overrides: Any) -> dict[str, Any]:
    return {**THREAD_LIST_ITEM_JSON, **overrides}


@dataclass
class FakeT3:
    script_path: Path
    log_path: Path

    def script(self, **responses: list[dict[str, Any]]) -> None:
        self.script_path.write_text(json.dumps(responses))

    def calls(self, verb: str) -> list[dict[str, Any]]:
        if not self.log_path.exists():
            return []
        entries = [json.loads(line) for line in self.log_path.read_text().splitlines()]
        return [entry for entry in entries if entry['argv'][1:2] == [verb]]


@pytest.fixture
def fake_t3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeT3:
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    executable = bin_dir / 't3'
    executable.write_text(FAKE_T3_SOURCE.replace('/usr/bin/env python3', sys.executable, 1))
    executable.chmod(0o755)
    fake = FakeT3(tmp_path / 'fake-t3.json', tmp_path / 'fake-t3.log')
    fake.script()
    monkeypatch.setenv('PATH', f'{bin_dir}{os.pathsep}{os.environ["PATH"]}')
    monkeypatch.setenv('FAKE_T3_SCRIPT', str(fake.script_path))
    monkeypatch.setenv('FAKE_T3_LOG', str(fake.log_path))
    return fake


@dataclass
class Env:
    project: Path
    t3_home: Path
    fleet: Path
    projects_dir: Path


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Env:
    project = tmp_path / 'proj'
    (project / 'sub').mkdir(parents=True)
    t3_home = tmp_path / 't3home'
    userdata = t3_home / 'userdata'
    userdata.mkdir(parents=True)
    (userdata / 'server-runtime.json').write_text(json.dumps(SERVER_RUNTIME_JSON))
    with sqlite3.connect(userdata / 'statev2.sqlite') as connection:
        connection.execute(PROJECTS_TABLE_SQL)
        connection.execute(
            'INSERT INTO projection_projects VALUES (?, ?, ?, NULL)',
            (PROJECT_ID, 'proj', str(project)),
        )
    connection.close()
    fleet = tmp_path / 'fleet'
    projects_dir = tmp_path / 'claude-projects'
    monkeypatch.setenv('T3CODE_HOME', str(t3_home))
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(fleet))
    monkeypatch.setenv('CLAUDE_PROJECTS_DIR', str(projects_dir))
    monkeypatch.setenv('SPAWN_T3_POLL_SECS', '0')
    return Env(project, t3_home, fleet, projects_dir)


def run_verb(capsys: pytest.CaptureFixture[str], *argv: str) -> str:
    rc = lane.main(list(argv))
    out = capsys.readouterr().out
    assert rc == 0, out
    assert out.count('\n') == 1, out
    return out.strip()


def spawn_record(env: Env, slug: str = 'unblock-df-7489-4242') -> str:
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug,
            status=sr.Status.LAUNCHING,
            cwd=str(env.project),
            launcher_pid=4242,
            start_ts=datetime.now(UTC).isoformat(),
        ),
        root=env.fleet,
    )
    return str(sr.record_path_for_slug(slug, root=env.fleet).parent)


def prompt_file(tmp_path: Path, text: str) -> str:
    path = tmp_path / 'prompt.txt'
    path.write_text(text)
    return str(path)


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
    def test_started_sends_the_prompt_and_stamps_the_record(
        self, env: Env, fake_t3: FakeT3, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fake_t3.script(start=[ok(THREAD_START_JSON)])
        record_dir = spawn_record(env)

        line = run_verb(
            capsys, 'launch', '--cwd', str(env.project), '--title', 'unblock:df#7489 x',
            '--prompt-file', prompt_file(tmp_path, 'do the thing'), '--record', record_dir,
            '--model', 'opus',
        )

        assert line == f'started thread_id={THREAD_ID} project_id={PROJECT_ID} run_id={RUN_ID}'
        [call] = fake_t3.calls('start')
        assert call['argv'] == [
            'thread', 'start', '--json', '--title', 'unblock:df#7489 x',
            '--provider', 'claudeAgent', '--model', 'opus', str(env.project), '-',
        ]
        assert call['stdin'] == 'do the thing'
        record = sr.read_record(Path(record_dir).name, root=env.fleet)
        assert record.status == sr.Status.RUNNING
        assert record.display == sr.Display(
            kind='t3', wm_title='unblock:df#7489 x', t3_thread_id=THREAD_ID
        )

    @pytest.mark.parametrize(
        ('parent_args', 'expected'),
        [(['--parent', 'watcher-df-9'], 'watcher-df-9'), ([], None), (['--parent', '../x'], None)],
    )
    def test_parent_is_stamped_only_when_given_as_a_record_key(
        self, env: Env, fake_t3: FakeT3, tmp_path: Path, capsys: pytest.CaptureFixture[str],
        parent_args: list[str], expected: str | None,
    ) -> None:
        fake_t3.script(start=[ok(THREAD_START_JSON)])
        record_dir = spawn_record(env)

        line = run_verb(
            capsys, 'launch', '--cwd', str(env.project), '--title', 't',
            '--prompt-file', prompt_file(tmp_path, 'p'), '--record', record_dir, *parent_args,
        )

        assert line.startswith('started ')
        record = sr.read_record(Path(record_dir).name, root=env.fleet)
        assert (record.parent_session_id, record.status) == (expected, sr.Status.RUNNING)
        assert record.display is not None and record.display.t3_thread_id == THREAD_ID

    def test_a_null_run_id_and_no_record(
        self, env: Env, fake_t3: FakeT3, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fake_t3.script(start=[ok({**THREAD_START_JSON, 'runId': None})])

        line = run_verb(
            capsys, 'launch', '--cwd', str(env.project), '--title', 't',
            '--prompt-file', prompt_file(tmp_path, 'p'), '--record', '',
        )

        assert line == f'started thread_id={THREAD_ID} project_id={PROJECT_ID} run_id=none'
        assert '--model' not in fake_t3.calls('start')[0]['argv']

    def test_no_t3_on_path_is_no_launcher(
        self, env: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv('PATH', str(tmp_path / 'empty'))

        line = run_verb(
            capsys, 'launch', '--cwd', str(env.project), '--title', 't',
            '--prompt-file', prompt_file(tmp_path, 'p'), '--record', '',
        )

        assert line == 'no-launcher reason=t3-missing'

    def test_no_server_runtime_is_no_launcher(
        self, env: Env, fake_t3: FakeT3, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (env.t3_home / 'userdata' / 'server-runtime.json').unlink()

        line = run_verb(
            capsys, 'launch', '--cwd', str(env.project), '--title', 't',
            '--prompt-file', prompt_file(tmp_path, 'p'), '--record', '',
        )

        assert line == 'no-launcher reason=server-runtime-missing'
        assert fake_t3.calls('start') == []

    def test_a_cwd_inside_a_project_is_refused_naming_its_root(
        self, env: Env, fake_t3: FakeT3, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        line = run_verb(
            capsys, 'launch', '--cwd', str(env.project / 'sub'), '--title', 't',
            '--prompt-file', prompt_file(tmp_path, 'p'), '--record', '',
        )

        assert line == f'refused reason=cwd-not-project-root detail={env.project.resolve()}'
        assert fake_t3.calls('start') == []

    def test_a_cwd_outside_every_project_is_refused(
        self, env: Env, fake_t3: FakeT3, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        line = run_verb(
            capsys, 'launch', '--cwd', str(tmp_path), '--title', 't',
            '--prompt-file', prompt_file(tmp_path, 'p'), '--record', '',
        )

        assert line == 'refused reason=cwd-not-project-root detail=none'

    def test_an_unreadable_project_store_is_refused_as_unverifiable(
        self, env: Env, fake_t3: FakeT3, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (env.t3_home / 'userdata' / 'statev2.sqlite').unlink()

        line = run_verb(
            capsys, 'launch', '--cwd', str(env.project), '--title', 't',
            '--prompt-file', prompt_file(tmp_path, 'p'), '--record', '',
        )

        assert line == 'refused reason=cwd-not-project-root detail=unverifiable'

    def test_a_failed_start_with_no_thread_afterwards_is_launch_failed(
        self, env: Env, fake_t3: FakeT3, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        stale = listed(title='t', createdAt='2020-01-01T00:00:00.000Z')
        fake_t3.script(start=[unreachable()], list=[ok([stale])])

        line = run_verb(
            capsys, 'launch', '--cwd', str(env.project), '--title', 't',
            '--prompt-file', prompt_file(tmp_path, 'p'), '--record', '',
        )

        assert line == 'launch-failed'
        assert len(fake_t3.calls('list')) == 2

    def test_a_malformed_listing_skips_the_sweep_but_not_the_launch(
        self, env: Env, fake_t3: FakeT3, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fake_t3.script(start=[ok(THREAD_START_JSON)], list=[ok([{'title': 'no thread id'}])])

        line = run_verb(
            capsys, 'launch', '--cwd', str(env.project), '--title', 't',
            '--prompt-file', prompt_file(tmp_path, 'p'), '--record', '',
        )

        assert line.startswith('started ')
        with pytest.raises(lane.T3CommandFailed):
            lane.list_threads(str(env.project))

    def test_two_threads_with_the_title_since_the_launch_is_launch_failed(
        self, env: Env, fake_t3: FakeT3, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        just_now = (datetime.now(UTC) + timedelta(seconds=5)).isoformat()
        twins = [listed(title='t', createdAt=just_now), listed(threadId='twin', title='t', createdAt=just_now)]
        fake_t3.script(start=[unreachable()], list=[ok(twins)])

        line = run_verb(
            capsys, 'launch', '--cwd', str(env.project), '--title', 't',
            '--prompt-file', prompt_file(tmp_path, 'p'), '--record', '',
        )

        assert line == 'launch-failed'

    def test_launch_first_sweeps_the_projects_records_whose_thread_is_gone(
        self, env: Env, fake_t3: FakeT3, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        stale_dir = spawn_record(env, 'sibling-df-1-1')
        sr.update_display('sibling-df-1-1', root=env.fleet, kind='t3', t3_thread_id='th-vanished')
        time.sleep(0.01)
        fake_t3.script(start=[ok(THREAD_START_JSON)], list=[ok([listed()])])

        run_verb(
            capsys, 'launch', '--cwd', str(env.project), '--title', 't',
            '--prompt-file', prompt_file(tmp_path, 'p'), '--record', '',
        )

        stale = sr.read_record(Path(stale_dir).name, root=env.fleet)
        assert (stale.status, stale.exit_code) == (sr.Status.EXITED, sr.ORPHAN_EXIT_CODE)

    def test_a_failure_after_the_send_finds_the_launched_thread(
        self, env: Env, fake_t3: FakeT3, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        just_now = (datetime.now(UTC) + timedelta(seconds=5)).isoformat()
        fake_t3.script(
            start=[unreachable()], list=[ok([listed(title='other'), listed(title='t', createdAt=just_now)])]
        )
        record_dir = spawn_record(env)

        line = run_verb(
            capsys, 'launch', '--cwd', str(env.project), '--title', 't',
            '--prompt-file', prompt_file(tmp_path, 'p'), '--record', record_dir,
        )

        assert line == f'started thread_id={THREAD_ID} project_id={PROJECT_ID} run_id={RUN_ID}'
        record = sr.read_record(Path(record_dir).name, root=env.fleet)
        assert record.display is not None and record.display.t3_thread_id == THREAD_ID

    def test_a_prompt_over_the_ceiling_is_spilled_into_the_record_dir(
        self, env: Env, fake_t3: FakeT3, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fake_t3.script(start=[ok(THREAD_START_JSON)])
        record_dir = spawn_record(env)
        big = 'x' * (lane.MESSAGE_CHAR_LIMIT + 1)

        run_verb(
            capsys, 'launch', '--cwd', str(env.project), '--title', 't',
            '--prompt-file', prompt_file(tmp_path, big), '--record', record_dir,
        )

        spill = Path(record_dir) / 'prompt.md'
        assert spill.read_text() == big
        assert fake_t3.calls('start')[0]['stdin'] == (
            f'Read {spill} and follow it as your brief; that file holds your whole task.'
        )

    def test_a_prompt_at_the_ceiling_is_sent_whole(
        self, env: Env, fake_t3: FakeT3, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fake_t3.script(start=[ok(THREAD_START_JSON)])
        exact = 'y' * lane.MESSAGE_CHAR_LIMIT

        run_verb(
            capsys, 'launch', '--cwd', str(env.project), '--title', 't',
            '--prompt-file', prompt_file(tmp_path, exact), '--record', '',
        )

        assert fake_t3.calls('start')[0]['stdin'] == exact

    def test_a_spill_without_a_record_goes_to_a_temp_file(
        self, env: Env, fake_t3: FakeT3, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fake_t3.script(start=[ok(THREAD_START_JSON)])
        big = 'z' * (lane.MESSAGE_CHAR_LIMIT + 1)

        run_verb(
            capsys, 'launch', '--cwd', str(env.project), '--title', 't',
            '--prompt-file', prompt_file(tmp_path, big), '--record', '',
        )

        message = fake_t3.calls('start')[0]['stdin']
        spill = Path(message.removeprefix('Read ').split(' and follow')[0])
        try:
            assert spill.read_text() == big
        finally:
            spill.unlink()


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


def hand_record(env: Env, session_id: str, owner_pid: int | None) -> str:
    """A hand-launched record plus its session-id pointer, as the hooks write them."""
    slug = f'session-proj-{session_id}'
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug,
            status=sr.Status.RUNNING,
            cwd=str(env.project),
            claude_session_id=session_id,
            claude_owner_pid=owner_pid,
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
        self, env: Env, fake_t3: FakeT3, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fake_t3.script(list=[ok([listed(status='preparing')]), ok([listed(status='running')])])
        record_dir = spawn_record(env)
        write_transcript(env, SESSION_ID, trailer(record_dir))
        hand_slug = hand_record(env, SESSION_ID, owner_pid=424242)

        line = self._await(capsys, env, record_dir, since=time.time() - 60)

        assert line == f'started sid={SESSION_ID} hand_slug={hand_slug} bound=yes'
        slug = Path(record_dir).name
        record = sr.read_record(slug, root=env.fleet)
        assert (record.claude_session_id, record.claude_launch_session_id) == (SESSION_ID, SESSION_ID)
        assert record.claude_owner_pid == 424242
        assert sr.resolve_session_slug_for_claude_session_id(SESSION_ID, root=env.fleet) == slug
        assert sr.resolve_session_slug_for_pid(424242, root=env.fleet) == slug
        folded = sr.read_record(hand_slug, root=env.fleet)
        assert (folded.status, folded.exit_code) == (sr.Status.EXITED, sr.FOLDED_EXIT_CODE)

    def test_started_binds_without_a_pid_when_no_hand_launched_record_exists(
        self, env: Env, fake_t3: FakeT3, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fake_t3.script(list=[ok([listed(status='waiting')])])
        record_dir = spawn_record(env)
        write_transcript(env, SESSION_ID, trailer(record_dir))

        line = self._await(capsys, env, record_dir)

        assert line == f'started sid={SESSION_ID} hand_slug=none bound=yes'
        assert sr.read_record(Path(record_dir).name, root=env.fleet).claude_owner_pid is None

    def test_a_session_already_pointing_at_the_spawn_record_folds_nothing(
        self, env: Env, fake_t3: FakeT3, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fake_t3.script(list=[ok([listed(status='running')])])
        record_dir = spawn_record(env)
        slug = Path(record_dir).name
        record = sr.read_record(slug, root=env.fleet)
        record.claude_session_id = SESSION_ID
        sr.write_record(record, root=env.fleet)
        sr.write_claude_session_pointer(SESSION_ID, slug, root=env.fleet)
        write_transcript(env, SESSION_ID, trailer(record_dir))

        line = self._await(capsys, env, record_dir)

        assert line == f'started sid={SESSION_ID} hand_slug=none bound=yes'
        assert sr.read_record(slug, root=env.fleet).status == sr.Status.LAUNCHING

    def test_started_with_no_transcript_naming_the_record_is_unbound(
        self, env: Env, fake_t3: FakeT3, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fake_t3.script(list=[ok([listed(status='running')])])
        record_dir = spawn_record(env)
        write_transcript(env, 'not-in-first-message', 'hello', later_text=trailer(record_dir))
        write_transcript(env, 'too-old', trailer(record_dir), mtime=time.time() - 3600)
        write_transcript(env, 'prefix-only', trailer(record_dir + '1'))
        write_transcript(env, 'mention-only', f'Read {record_dir}/result.md and summarise it.')

        line = self._await(capsys, env, record_dir, timeout='1', since=time.time() - 60)

        assert line == 'started sid=none hand_slug=none bound=no'
        assert sr.read_record(Path(record_dir).name, root=env.fleet).claude_session_id is None

    def test_started_without_a_record_skips_discovery(
        self, env: Env, fake_t3: FakeT3, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fake_t3.script(list=[ok([listed(status='completed')])])

        assert self._await(capsys, env, '') == 'started sid=none hand_slug=none bound=no'

    @pytest.mark.parametrize('status', sorted(lane.STARTUP_FAILURE_STATUSES))
    def test_a_failure_status_before_any_transcript_is_failed_to_start(
        self, env: Env, fake_t3: FakeT3, capsys: pytest.CaptureFixture[str], status: str
    ) -> None:
        fake_t3.script(list=[ok([listed(status=status)])])

        assert self._await(capsys, env, spawn_record(env)) == 'failed-to-start'
        assert self._await(capsys, env, '') == 'failed-to-start'

    def test_a_failure_status_after_the_transcript_appeared_is_started(
        self, env: Env, fake_t3: FakeT3, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fake_t3.script(list=[ok([listed(status='failed')])])
        record_dir = spawn_record(env)
        write_transcript(env, SESSION_ID, trailer(record_dir))

        assert self._await(capsys, env, record_dir) == (
            f'started sid={SESSION_ID} hand_slug=none bound=yes'
        )

    def test_an_interrupted_run_is_started(
        self, env: Env, fake_t3: FakeT3, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fake_t3.script(list=[ok([listed(status='interrupted')])])

        assert self._await(capsys, env, '') == 'started sid=none hand_slug=none bound=no'

    def test_a_thread_that_never_leaves_preparation_times_out(
        self, env: Env, fake_t3: FakeT3, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fake_t3.script(list=[ok([listed(status='queued')])])

        assert self._await(capsys, env, spawn_record(env), timeout='0.3') == 'timeout'

    def test_an_unreachable_server_times_out(
        self, env: Env, fake_t3: FakeT3, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fake_t3.script(list=[unreachable()])

        assert self._await(capsys, env, '', timeout='0.3') == 'timeout'


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


Listed = lane.ThreadObservation(lane.ThreadState.LISTED, 'running', RUN_ID)


def _ended(run_id: str = RUN_ID, status: str = 'failed') -> lane.ThreadObservation:
    return lane.ThreadObservation(lane.ThreadState.LISTED, status, run_id)


class TestJoinTracker:
    def test_a_final_result_finishes_zero_even_while_running(self) -> None:
        assert lane.JoinTracker().verdict(Listed, result_final=True, now=0) == 0

    def test_an_archived_thread_finishes_zero(self) -> None:
        archived = lane.ThreadObservation(lane.ThreadState.ARCHIVED)
        assert lane.JoinTracker().verdict(archived, result_final=False, now=0) == 0

    def test_a_gone_thread_finishes_127(self) -> None:
        gone = lane.ThreadObservation(lane.ThreadState.GONE)
        assert lane.JoinTracker().verdict(gone, result_final=False, now=0) == 127

    @pytest.mark.parametrize('status', ['idle', 'waiting', 'running', 'completed', 'queued'])
    def test_live_or_idle_keeps_waiting(self, status: str) -> None:
        tracker = lane.JoinTracker()
        observation = lane.ThreadObservation(lane.ThreadState.LISTED, status, RUN_ID)
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
        assert tracker.verdict(Listed, result_final=False, now=60) is None
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
        tracker.verdict(Listed, result_final=False, now=500)
        assert tracker.verdict(down, result_final=False, now=700) is None
        assert tracker.verdict(down, result_final=False, now=1300) is None
        assert tracker.verdict(down, result_final=False, now=1301) == 126


class TestWait:
    def _wait(self, capsys: pytest.CaptureFixture[str], env: Env, record_dir: str) -> str:
        return run_verb(
            capsys, 'wait', '--thread-id', THREAD_ID, '--cwd', str(env.project),
            '--record', record_dir,
        )

    def test_a_final_result_md_finishes_zero(
        self, env: Env, fake_t3: FakeT3, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fake_t3.script(list=[ok([listed(status='running')])])
        record_dir = spawn_record(env)
        (Path(record_dir) / 'result.md').write_text('---\noutcome: done\n---\nAll good.\n')

        assert self._wait(capsys, env, record_dir) == 'finished code=0'

    def test_settled_and_in_progress_are_not_finished_until_archived(
        self, env: Env, fake_t3: FakeT3, capsys: pytest.CaptureFixture[str]
    ) -> None:
        settled = listed(status='idle', settled=True, settledAt='2026-10-25T09:00:00.000Z')
        fake_t3.script(list=[ok([settled]), ok([settled]), ok([])], show=[ok(THREAD_SHOW_JSON)])
        record_dir = spawn_record(env)
        (Path(record_dir) / 'result.md').write_text('outcome: in-progress\n')

        assert self._wait(capsys, env, record_dir) == 'finished code=0'
        assert len(fake_t3.calls('list')) == 3

    def test_a_successful_show_without_the_thread_finishes_127(
        self, env: Env, fake_t3: FakeT3, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fake_t3.script(list=[ok([])], show=[ok({'messages': []})])

        assert self._wait(capsys, env, '') == 'finished code=127'

    def test_a_failed_show_is_unreachable_not_gone(self, env: Env, fake_t3: FakeT3) -> None:
        fake_t3.script(list=[ok([])], show=[{'exit': 1, 'stdout': None, 'stderr': 'gone'}])

        observation, known = lane.observe_thread(THREAD_ID, str(env.project))

        assert observation.state is lane.ThreadState.UNREACHABLE
        assert known == {}

    def test_wait_stops_when_its_caller_is_gone(self, env: Env, fake_t3: FakeT3) -> None:
        fake_t3.script(list=[ok([listed(status='running')])])
        parents = iter([4242, 4242, 4242, 1])

        with pytest.raises(lane.CallerGone):
            lane.wait(THREAD_ID, str(env.project), '', {'SPAWN_T3_POLL_SECS': '0'},
                      parent_pid=lambda: next(parents))

        assert len(fake_t3.calls('list')) == 2

    def test_main_reports_a_gone_caller_on_stderr_only(
        self, env: Env, fake_t3: FakeT3, capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def gone(*_args: object, **_kwargs: object) -> str:
            raise lane.CallerGone('caller 4242 is gone')

        monkeypatch.setattr(lane, 'wait', gone)

        rc = lane.main(['wait', '--thread-id', THREAD_ID, '--cwd', str(env.project), '--record', ''])

        captured = capsys.readouterr()
        assert (rc, captured.out) == (lane.INTERNAL_FAULT_EXIT, '')
        assert captured.err == 'spawn_t3_lane: caller 4242 is gone\n'

    def test_a_failed_show_during_an_outage_is_not_mistaken_for_deletion(
        self, env: Env, fake_t3: FakeT3, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fake_t3.script(
            list=[ok([]), unreachable(), ok([])],
            show=[unreachable(), ok(THREAD_SHOW_JSON)],
        )

        assert self._wait(capsys, env, '') == 'finished code=0'

    def test_each_poll_sweeps_the_projects_t3_records(
        self, env: Env, fake_t3: FakeT3, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fake_t3.script(list=[ok([])], show=[ok(THREAD_SHOW_JSON)])
        record_dir = spawn_record(env, 'sibling-df-1-1')
        sr.update_display('sibling-df-1-1', root=env.fleet, kind='t3', t3_thread_id='th-vanished')
        time.sleep(0.01)

        assert self._wait(capsys, env, '') == 'finished code=0'
        record = sr.read_record(Path(record_dir).name, root=env.fleet)
        assert (record.status, record.exit_code) == (sr.Status.EXITED, sr.ORPHAN_EXIT_CODE)

    def test_an_unbound_record_is_bound_while_waiting(
        self, env: Env, fake_t3: FakeT3, capsys: pytest.CaptureFixture[str]
    ) -> None:
        record_dir = spawn_record(env)
        (Path(record_dir) / 'result.md').write_text('outcome: done\n')
        write_transcript(env, SESSION_ID, trailer(record_dir))
        fake_t3.script(list=[ok([listed()])])

        assert self._wait(capsys, env, record_dir) == 'finished code=0'
        assert sr.read_record(Path(record_dir).name, root=env.fleet).claude_session_id == SESSION_ID


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
    env: Env, fake_t3: FakeT3, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spaced = env.project / 'with space'
    spaced.mkdir()

    line = run_verb(
        capsys, 'launch', '--cwd', str(spaced), '--title', 't',
        '--prompt-file', prompt_file(tmp_path, 'p'), '--record', '',
    )

    assert line == f'refused reason=cwd-not-project-root detail={env.project.resolve()}'
    assert lane._verdict('x', detail='/a b') == 'x detail=/a%20b'


def test_an_internal_fault_exits_3_with_nothing_on_stdout(
    env: Env, fake_t3: FakeT3, tmp_path: Path, capsys: pytest.CaptureFixture[str]
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
