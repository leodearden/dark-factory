"""Lease identity and liveness when ``$CLAUDE_PID`` names a shared ``claude agents`` host.

Every session hosted by one ``claude agents`` process sees that host's pid as
``$CLAUDE_PID``, and the host outlives each of them. These tests drive the
public lease seams with a REAL process whose argv has the host's shape.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest  # pyright: ignore[reportMissingImports]

from orchestrator import session_registry as sr

pytestmark = pytest.mark.skipif(
    not Path('/proc/self/cmdline').exists(), reason='host detection reads /proc'
)

_NAME = 'recon-watcher-df'
_PAST_TTL_SECS = sr.LEASE_HEARTBEAT_TTL.total_seconds() + 60


@pytest.fixture
def shared_host_pid(tmp_path_factory: pytest.TempPathFactory) -> Iterator[int]:
    """A live process whose argv is ``<python> agents --inherit-permission-mode ...``."""
    host_dir = tmp_path_factory.mktemp('agents-host')
    (host_dir / 'agents').write_text('import time\ntime.sleep(600)\n', encoding='utf-8')
    proc = subprocess.Popen(
        [sys.executable, 'agents', '--inherit-permission-mode', 'bypassPermissions'],
        cwd=host_dir,
    )
    try:
        yield proc.pid
    finally:
        proc.kill()
        proc.wait(timeout=10)


def _age(path: Path, seconds: float) -> None:
    t = time.time() - seconds
    os.utime(path, (t, t))


def _hosted_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, pid: int, session_id: str | None
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    monkeypatch.setenv('CLAUDE_PID', str(pid))
    if session_id is None:
        monkeypatch.delenv('CLAUDE_CODE_SESSION_ID', raising=False)
    else:
        monkeypatch.setenv('CLAUDE_CODE_SESSION_ID', session_id)


def _body(tmp_path: Path, name: str = _NAME) -> sr.LeaseHolder:
    return sr.LeaseHolder.from_json(sr.lease_path_for_name(name, root=tmp_path).read_text())


def _cli(capsys: pytest.CaptureFixture[str], *argv: str) -> list[str]:
    capsys.readouterr()
    assert sr.main(list(argv)) == 0
    return capsys.readouterr().out.splitlines()


def _warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelno >= logging.WARNING]


# --- resolution layer --------------------------------------------------------


def test_a_shared_host_pid_degrades_the_liveness_pid_to_zero_loudly(
    shared_host_pid: int, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING)
    env = {'CLAUDE_PID': str(shared_host_pid), 'CLAUDE_CODE_SESSION_ID': 'sess-a'}

    assert sr.resolve_session_pid(env) == 0

    messages = [r.getMessage() for r in _warnings(caplog)]
    assert any('degraded' in m.lower() and str(shared_host_pid) in m for m in messages)


def test_hosted_sessions_derive_distinct_session_scoped_slugs(shared_host_pid: int) -> None:
    host = shared_host_pid

    slug_a = sr.default_lease_slug(
        _NAME, {'CLAUDE_PID': str(host), 'CLAUDE_CODE_SESSION_ID': 'sess-a'}
    )
    slug_b = sr.default_lease_slug(
        _NAME, {'CLAUDE_PID': str(host), 'CLAUDE_CODE_SESSION_ID': 'sess-b'}
    )

    assert slug_a == f'{_NAME}-{host}-sess-a'
    assert slug_b == f'{_NAME}-{host}-sess-b'
    assert slug_a != slug_b
    assert f'{_NAME}-{host}' not in (slug_a, slug_b)


@pytest.mark.parametrize(
    'session_id',
    [None, '', '   ', 'has space', 'semi;colon'],
    ids=['absent', 'empty', 'blank', 'space', 'semicolon'],
)
def test_a_hosted_slug_is_underivable_without_a_usable_session_id(
    shared_host_pid: int, session_id: str | None
) -> None:
    env = {'CLAUDE_PID': str(shared_host_pid)}
    if session_id is not None:
        env['CLAUDE_CODE_SESSION_ID'] = session_id

    assert sr.default_lease_slug(_NAME, env) is None


def test_a_non_host_session_is_unchanged(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING)
    own = os.getpid()
    env = {'CLAUDE_PID': str(own), 'CLAUDE_CODE_SESSION_ID': 'sess-a'}

    assert sr.resolve_session_pid(env) == own
    assert _warnings(caplog) == []
    assert sr.default_lease_slug(_NAME, env) == f'{_NAME}-{own}'


# --- lease verbs through the CLI ---------------------------------------------


def test_a_hosted_claim_records_pid_zero_and_a_session_scoped_slug(
    shared_host_pid: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    host = shared_host_pid
    sr.write_record(
        sr.SessionRecord(
            session_slug='sibling-hosted-record',
            status=sr.Status.RUNNING,
            claude_session_id='claude-session-uuid',
            claude_owner_pid=host,
        ),
        root=tmp_path,
    )
    sr.write_session_pointer(host, 'sibling-hosted-record', root=tmp_path)
    assert sr.resolve_session_slug_for_pid(host, root=tmp_path) == 'sibling-hosted-record'
    _hosted_env(monkeypatch, tmp_path, pid=host, session_id='sess-a')

    lines = _cli(capsys, 'lease-claim', '--name', _NAME)

    assert lines[0] == 'decision=acquired'
    assert f'slug={_NAME}-{host}-sess-a' in lines
    assert lines[-1] == 'holder_record=unlinked'
    body = _body(tmp_path)
    assert body.pid == 0
    assert body.session_slug == f'{_NAME}-{host}-sess-a'
    assert body.record_slug == ''


def test_a_sibling_hosted_session_cannot_claim_heartbeat_or_release(
    shared_host_pid: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    host = shared_host_pid
    _hosted_env(monkeypatch, tmp_path, pid=host, session_id='sess-a')
    _cli(capsys, 'lease-claim', '--name', _NAME)
    lease = sr.lease_path_for_name(_NAME, root=tmp_path)
    claimed_body = lease.read_text()
    _hosted_env(monkeypatch, tmp_path, pid=host, session_id='sess-b')

    claim_lines = _cli(capsys, 'lease-claim', '--name', _NAME)

    assert claim_lines[0] == 'decision=stand-down'
    assert 'holder_liveness=orphaned' in claim_lines
    assert lease.read_text() == claimed_body

    _age(lease, 600)
    aged_mtime = lease.stat().st_mtime

    assert _cli(capsys, 'lease-heartbeat', '--name', _NAME)[0] == 'result=refused'
    assert lease.stat().st_mtime == aged_mtime
    assert _cli(capsys, 'lease-release', '--name', _NAME)[0] == 'result=refused'
    assert lease.is_file()


def test_the_owning_hosted_session_heartbeats_quietly_and_releases(
    shared_host_pid: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    _hosted_env(monkeypatch, tmp_path, pid=shared_host_pid, session_id='sess-a')
    _cli(capsys, 'lease-claim', '--name', _NAME)
    lease = sr.lease_path_for_name(_NAME, root=tmp_path)
    _age(lease, 600)
    aged_mtime = lease.stat().st_mtime
    caplog.clear()
    caplog.set_level(logging.WARNING)

    assert _cli(capsys, 'lease-heartbeat', '--name', _NAME)[0] == 'result=applied'

    assert lease.stat().st_mtime > aged_mtime
    assert _warnings(caplog) == []
    assert _cli(capsys, 'lease-release', '--name', _NAME)[0] == 'result=applied'
    assert not lease.exists()


def test_a_hosted_lease_is_reaped_after_its_session_stops_heartbeating_while_the_host_lives(
    shared_host_pid: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    host = shared_host_pid
    _hosted_env(monkeypatch, tmp_path, pid=host, session_id='sess-a')
    _cli(capsys, 'lease-claim', '--name', _NAME)
    lease = sr.lease_path_for_name(_NAME, root=tmp_path)
    _age(lease, _PAST_TTL_SECS)
    assert Path(f'/proc/{host}').exists()

    _cli(capsys, 'lease-reap')

    assert not lease.exists()

    _cli(capsys, 'lease-claim', '--name', _NAME)
    _age(lease, _PAST_TTL_SECS)
    _hosted_env(monkeypatch, tmp_path, pid=host, session_id='sess-c')

    assert _cli(capsys, 'lease-claim', '--name', _NAME)[0] == 'decision=acquired'
    assert _body(tmp_path).session_slug.endswith('-sess-c')


@pytest.mark.parametrize('verb', ['lease-claim', 'lease-heartbeat', 'lease-release'])
def test_a_hosted_claim_without_a_session_id_exits_2_naming_it(
    shared_host_pid: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    verb: str,
) -> None:
    _hosted_env(monkeypatch, tmp_path, pid=shared_host_pid, session_id=None)

    with pytest.raises(SystemExit) as excinfo:
        sr.main([verb, '--name', _NAME])

    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert 'CLAUDE_CODE_SESSION_ID' in err
    assert '--slug' in err
    assert not sr.lease_path_for_name(_NAME, root=tmp_path).exists()
