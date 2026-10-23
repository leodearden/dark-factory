"""B15: the Stop hook's opt-in self-exit, driven through ``session_hooks.run_stop``.

Contract: plans/review-all-program-tooling-prd.md §4 H5. Every pid here is
above the host pid_max (4194304), so the real /proc-backed shared-host rule
reads "not a host" and no recorded signal could ever have reached a live
process.
"""

from __future__ import annotations

import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from _process_tree_fakes import claude_process_tree, counting_walks

from orchestrator import session_hooks as sh
from orchestrator import session_registry as sr
from orchestrator.session_ownership_probes import ProcessTree
from orchestrator.session_self_exit import SelfExit, SelfExitRefusal

OWNER = 7_193_001
OWNER_PARENT = 7_193_000
LAUNCHER = 7_193_050
_SLUG = 'prd-df-7193-7193050'
_TERMINAL_RESULT = 'outcome: done\nchanged: x\naction_needed: none\n'


@pytest.fixture(autouse=True)
def _isolate_fleet_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))


def _record_dir(root: Path, slug: str = _SLUG) -> Path:
    return sr.record_path_for_slug(slug, root=root).parent


def _marker(root: Path, slug: str = _SLUG) -> Path:
    return _record_dir(root, slug) / sr.EXIT_REQUESTED_FILENAME


@dataclass
class _KillRecorder:
    """A ``kill`` that records each signal and what the disk held when it was sent."""

    root: Path
    slug: str = _SLUG
    calls: list[tuple[int, int]] = field(default_factory=list)
    seen: list[tuple[sr.SessionRecord, bool]] = field(default_factory=list)

    def __call__(self, pid: int, sig: int) -> None:
        self.calls.append((pid, sig))
        self.seen.append((sr.read_record(self.slug, root=self.root), _marker(self.root, self.slug).exists()))


@dataclass(frozen=True)
class _Armed:
    env: dict[str, str]
    tree: ProcessTree
    hook_input: dict[str, object]


def _armed(
    root: Path, *, result: str | None = _TERMINAL_RESULT, marker: str | None = 'uuid-owner', **record_fields: object,
) -> _Armed:
    """A self-exit session whose every precondition holds; *record_fields* break one on the record."""
    record_dir = _record_dir(root)
    fields: dict = {
        'status': sr.Status.RUNNING,
        'self_exit': True,
        'claude_session_id': 'uuid-owner',
        'claude_owner_pid': OWNER,
        'launcher_pid': LAUNCHER,
        'result_file': str(record_dir / sr.RESULT_FILENAME),
    }
    fields.update(record_fields)
    sr.write_record(sr.SessionRecord(session_slug=_SLUG, **fields), root=root)
    if result is not None:
        (record_dir / sr.RESULT_FILENAME).write_text(result, encoding='utf-8')
    if marker is not None:
        _marker(root).write_text(marker, encoding='utf-8')
    return _Armed(
        env={'CLAUDE_SPAWN_SESSION_ID': _SLUG, 'CLAUDE_SPAWN_OWNER_PPID': str(OWNER_PARENT)},
        tree=claude_process_tree(OWNER, OWNER_PARENT),
        hook_input={'session_id': 'uuid-owner', 'cwd': str(root), 'background_tasks': [], 'session_crons': []},
    )


def _stop(
    root: Path,
    armed: _Armed,
    kill: Callable[[int, int], None],
    *,
    hook_input: Mapping[str, object] | None = None,
    env: Mapping[str, str] | None = None,
    tree: ProcessTree | None = None,
) -> str:
    return sh.run_stop(
        armed.hook_input if hook_input is None else hook_input,
        armed.env if env is None else env,
        root=root,
        process_tree=armed.tree if tree is None else tree,
        self_exit=SelfExit(kill=kill),
    )


def _read(root: Path, slug: str = _SLUG) -> sr.SessionRecord:
    return sr.read_record(slug, root=root)


class TestArmedOwnerStop:
    def test_signals_only_the_owning_claude(self, tmp_path):
        armed = _armed(tmp_path)
        kill = _KillRecorder(tmp_path)

        _stop(tmp_path, armed, kill)

        assert kill.calls == [(OWNER, signal.SIGINT)]

    def test_stamps_attempted_and_deletes_the_marker(self, tmp_path):
        armed = _armed(tmp_path)

        _stop(tmp_path, armed, _KillRecorder(tmp_path))

        record = _read(tmp_path)
        request = record.exit_request
        assert record.status is sr.Status.IDLE
        assert request is not None
        assert request.state is sr.ExitRequestState.ATTEMPTED
        assert request.reason is None
        assert request.pid == OWNER
        assert request.marker_session_id == 'uuid-owner'
        assert record.exit_requested_at == request.at
        datetime.fromisoformat(request.at)
        assert not _marker(tmp_path).exists()

    def test_makes_one_ancestor_walk(self, tmp_path):
        armed = _armed(tmp_path)
        tree, walks = counting_walks(armed.tree)

        _stop(tmp_path, armed, _KillRecorder(tmp_path), tree=tree)

        assert len(walks) == 1

    def test_the_stamp_is_on_disk_before_the_signal_and_the_marker_goes_after(self, tmp_path):
        armed = _armed(tmp_path)
        kill = _KillRecorder(tmp_path)

        _stop(tmp_path, armed, kill)

        [(at_kill, marker_present)] = kill.seen
        assert at_kill.status is sr.Status.IDLE
        assert at_kill.exit_request is not None
        assert at_kill.exit_request.state is sr.ExitRequestState.ATTEMPTED
        assert at_kill.exit_requested_at == at_kill.exit_request.at
        assert marker_present
        assert not _marker(tmp_path).exists()

    def test_a_marker_from_before_a_session_id_re_mint_still_exits(self, tmp_path):
        armed = _armed(tmp_path, marker='uuid-before-compact')
        kill = _KillRecorder(tmp_path)

        _stop(tmp_path, armed, kill)

        assert kill.calls == [(OWNER, signal.SIGINT)]
        request = _read(tmp_path).exit_request
        assert request is not None
        assert request.marker_session_id == 'uuid-before-compact'


@dataclass(frozen=True)
class _Break:
    """One precondition broken: on the record (*armed*), the hook input, or the event's claude pid."""

    reason: SelfExitRefusal
    armed: Mapping[str, Any] = field(default_factory=dict)
    hook: Mapping[str, object] = field(default_factory=dict)
    drop: tuple[str, ...] = ()
    claude: int | None = OWNER


_BREAKS = {
    'self-exit-off': _Break(SelfExitRefusal.SELF_EXIT_OFF, armed={'self_exit': False}),
    'no-result': _Break(SelfExitRefusal.RESULT_UNREADABLE, armed={'result': None}),
    'checkpoint': _Break(SelfExitRefusal.OUTCOME_NOT_TERMINAL, armed={'result': 'outcome: in-progress\n'}),
    'no-outcome-line': _Break(SelfExitRefusal.OUTCOME_NOT_TERMINAL, armed={'result': 'I finished the work.\n'}),
    'no-claude-ancestor': _Break(SelfExitRefusal.OWNER_PID_UNRESOLVED, claude=None),
    'no-bound-pid': _Break(SelfExitRefusal.OWNER_PID_UNRESOLVED, armed={'claude_owner_pid': None}),
    'other-claude': _Break(SelfExitRefusal.OWNER_PID_MISMATCH, claude=7_193_009),
    'background-absent': _Break(SelfExitRefusal.BACKGROUND_TASKS_UNREPORTED, drop=('background_tasks',)),
    'background-null': _Break(SelfExitRefusal.BACKGROUND_TASKS_UNREPORTED, hook={'background_tasks': None}),
    'background-running': _Break(SelfExitRefusal.BACKGROUND_TASKS_RUNNING, hook={'background_tasks': [{'id': 'b1'}]}),
    'crons-absent': _Break(SelfExitRefusal.SESSION_CRONS_UNREPORTED, drop=('session_crons',)),
    'crons-scheduled': _Break(
        SelfExitRefusal.SESSION_CRONS_SCHEDULED, hook={'session_crons': [{'cron': '*/5 * * * *'}]},
    ),
}


class TestRefusals:
    @pytest.mark.parametrize('broken', _BREAKS.values(), ids=_BREAKS.keys())
    def test_one_broken_precondition_refuses_with_its_reason(self, tmp_path, broken):
        armed = _armed(tmp_path, **broken.armed)
        hook_input = {k: v for k, v in {**armed.hook_input, **broken.hook}.items() if k not in broken.drop}
        tree = claude_process_tree(None) if broken.claude is None else claude_process_tree(broken.claude, OWNER_PARENT)
        kill = _KillRecorder(tmp_path)

        _stop(tmp_path, armed, kill, hook_input=hook_input, tree=tree)

        record = _read(tmp_path)
        request = record.exit_request
        assert kill.calls == []
        assert request is not None
        assert request.state is sr.ExitRequestState.REFUSED
        assert request.reason == broken.reason
        assert request.pid == broken.claude
        assert record.exit_requested_at is None
        assert record.status is sr.Status.IDLE
        assert _marker(tmp_path).exists()

    def test_a_refusal_is_not_a_deadlock(self, tmp_path):
        armed = _armed(tmp_path)
        kill = _KillRecorder(tmp_path)

        _stop(tmp_path, armed, kill, hook_input={**armed.hook_input, 'background_tasks': [{'id': 'b1'}]})
        refused = _read(tmp_path).exit_request
        _stop(tmp_path, armed, kill)

        assert refused is not None
        assert refused.state is sr.ExitRequestState.REFUSED
        assert kill.calls == [(OWNER, signal.SIGINT)]
        request = _read(tmp_path).exit_request
        assert request is not None
        assert request.state is sr.ExitRequestState.ATTEMPTED
        assert not _marker(tmp_path).exists()


@pytest.fixture
def shared_host_pid(tmp_path_factory: pytest.TempPathFactory) -> Iterator[int]:
    """A live process whose argv is ``<python> agents --inherit-permission-mode ...``."""
    host_dir = tmp_path_factory.mktemp('agents-host')
    (host_dir / 'agents').write_text('import time\ntime.sleep(600)\n', encoding='utf-8')
    proc = subprocess.Popen(
        [sys.executable, 'agents', '--inherit-permission-mode', 'bypassPermissions'], cwd=host_dir,
    )
    try:
        _await_host_argv(proc.pid)
        yield proc.pid
    finally:
        proc.kill()
        proc.wait(timeout=10)


def _await_host_argv(pid: int) -> None:
    """Wait until ``/proc/<pid>/cmdline`` shows the host argv; exec publishes it after Popen returns."""
    cmdline = Path(f'/proc/{pid}/cmdline')
    deadline = time.monotonic() + 10
    while cmdline.read_bytes().split(b'\0')[1:2] != [b'agents']:
        if time.monotonic() > deadline:
            pytest.fail(f'{cmdline} never showed the `agents` argv')
        time.sleep(0.01)


@pytest.mark.skipif(not Path('/proc/self/cmdline').exists(), reason='host detection reads /proc')
def test_a_shared_session_host_is_never_signalled(tmp_path, shared_host_pid):
    armed = _armed(tmp_path, claude_owner_pid=shared_host_pid)
    kill = _KillRecorder(tmp_path)

    _stop(tmp_path, armed, kill, tree=claude_process_tree(shared_host_pid, OWNER_PARENT))

    request = _read(tmp_path).exit_request
    assert kill.calls == []
    assert request is not None
    assert request.state is sr.ExitRequestState.REFUSED
    assert request.reason == SelfExitRefusal.SHARED_SESSION_HOST
    assert request.pid == shared_host_pid


class TestNoOwnerNoStamp:
    def test_a_withheld_launch_window_event_stamps_nothing(self, tmp_path):
        armed = _armed(
            tmp_path,
            status=sr.Status.LAUNCHING,
            claude_session_id=None,
            claude_owner_pid=None,
            start_ts=datetime.now(UTC).isoformat(),
        )
        kill = _KillRecorder(tmp_path)

        _stop(tmp_path, armed, kill, env={'CLAUDE_SPAWN_SESSION_ID': _SLUG})

        record = _read(tmp_path)
        assert record.exit_request is None
        assert record.status is sr.Status.LAUNCHING
        assert kill.calls == []
        assert _marker(tmp_path).exists()

    def test_an_unproven_event_stamps_nothing(self, tmp_path):
        armed = _armed(tmp_path, claude_session_id=None, claude_owner_pid=None)
        kill = _KillRecorder(tmp_path)

        _stop(tmp_path, armed, kill, env={'CLAUDE_SPAWN_SESSION_ID': _SLUG})

        assert _read(tmp_path).exit_request is None
        assert kill.calls == []
        assert _marker(tmp_path).exists()

    def test_a_nested_inheritor_never_kills(self, tmp_path):
        armed = _armed(tmp_path, claude_session_id='uuid-parent')
        nested = {**armed.hook_input, 'session_id': 'uuid-nested'}
        tree = claude_process_tree(7_193_101, 7_193_100)
        kill = _KillRecorder(tmp_path)

        _stop(tmp_path, armed, kill, hook_input=nested, tree=tree)

        assert kill.calls == []
        assert _read(tmp_path).exit_request is None
        assert _marker(tmp_path).exists()
        forked_slug = sh.hook_session_slug(nested, armed.env, root=tmp_path, process_tree=tree)
        assert forked_slug != _SLUG
        assert _read(tmp_path, forked_slug).exit_request is None

    def test_no_marker_stays_silent(self, tmp_path):
        armed = _armed(tmp_path, marker=None)
        kill = _KillRecorder(tmp_path)

        _stop(tmp_path, armed, kill)

        record = _read(tmp_path)
        assert record.exit_request is None
        assert record.exit_requested_at is None
        assert kill.calls == []
