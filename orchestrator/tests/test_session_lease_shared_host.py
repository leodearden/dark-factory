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

import pytest

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
        _await_host_argv(proc.pid)
        yield proc.pid
    finally:
        proc.kill()
        proc.wait(timeout=10)


def _await_host_argv(pid: int) -> None:
    """Wait for ``/proc/<pid>/cmdline`` to show the host argv.

    Popen returns once the child's exec has closed its CLOEXEC pipe, which the
    kernel does before it publishes the new argv, so a loaded host can read an
    empty cmdline for a moment after Popen returns.
    """
    cmdline = Path(f'/proc/{pid}/cmdline')
    deadline = time.monotonic() + 10
    while cmdline.read_bytes().split(b'\0')[1:2] != [b'agents']:
        if time.monotonic() > deadline:
            pytest.fail(f'{cmdline} never showed the `agents` argv')
        time.sleep(0.01)


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


def _seed_sibling_record(tmp_path: Path, host: int) -> None:
    """Point the host pid's sessions-by-pid entry at a record the host owns."""
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
    _seed_sibling_record(tmp_path, host)
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


# --- lease bodies that still carry a running host pid ------------------------
#
# Claimed before hosted sessions recorded pid 0, or via an explicit `--pid`.


def _seed_lease(tmp_path: Path, pid: int, name: str = _NAME) -> Path:
    holder = sr.LeaseHolder(
        session_slug=f'{name}-{pid}', pid=pid, start_ts='2026-10-01T00:00:00+00:00'
    )
    sr.claim_lease(name, holder=holder, root=tmp_path)
    return sr.lease_path_for_name(name, root=tmp_path)


def test_lease_status_reports_a_legacy_host_pid_lease_reclaimable_past_ttl_only(
    shared_host_pid: int, tmp_path: Path
) -> None:
    host = shared_host_pid
    lease = _seed_lease(tmp_path, host)
    _age(lease, _PAST_TTL_SECS)

    aged = sr.lease_status(_NAME, root=tmp_path)

    assert aged.holder_pid == host
    assert aged.holder_pid_alive is True
    assert aged.reclaimable is True

    _age(lease, 60)

    assert sr.lease_status(_NAME, root=tmp_path).reclaimable is False


def test_reap_stale_leases_removes_an_aged_legacy_host_pid_lease(
    shared_host_pid: int, tmp_path: Path
) -> None:
    aged_lease = _seed_lease(tmp_path, shared_host_pid)
    fresh_lease = _seed_lease(tmp_path, shared_host_pid, name='watcher-df')
    _age(aged_lease, _PAST_TTL_SECS)
    _age(fresh_lease, 60)

    assert [r.reason for r in sr.reap_stale_leases(root=tmp_path)] == ['stale_pid']
    assert not aged_lease.exists()
    assert fresh_lease.is_file()


def test_claim_lease_reclaims_an_aged_legacy_host_pid_lease(
    shared_host_pid: int, tmp_path: Path
) -> None:
    host = shared_host_pid
    lease = _seed_lease(tmp_path, host)
    held_body = lease.read_text()
    contender = sr.LeaseHolder(
        session_slug='contender', pid=os.getpid(), start_ts='2026-10-13T00:00:00+00:00'
    )
    _age(lease, 60)

    fresh = sr.claim_lease(_NAME, holder=contender, root=tmp_path)

    assert fresh.decision is sr.LeaseDecision.STAND_DOWN
    assert f'pid {host} alive' in fresh.message
    assert lease.read_text() == held_body

    _age(lease, _PAST_TTL_SECS)

    aged = sr.claim_lease(_NAME, holder=contender, root=tmp_path)

    assert aged.decision is sr.LeaseDecision.ACQUIRED
    assert _body(tmp_path) == contender


def test_an_operator_pid_override_naming_a_host_still_ages_out(
    shared_host_pid: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    host = shared_host_pid
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    _cli(capsys, 'lease-claim', '--name', _NAME, '--slug', 'op-token', '--pid', str(host))
    lease = sr.lease_path_for_name(_NAME, root=tmp_path)
    assert _body(tmp_path).pid == host
    _age(lease, _PAST_TTL_SECS)

    _cli(capsys, 'lease-reap')

    assert not lease.exists()


def test_a_running_non_host_holder_is_never_stale(tmp_path: Path) -> None:
    lease = _seed_lease(tmp_path, os.getpid())
    _age(lease, 10 * sr.LEASE_HEARTBEAT_TTL.total_seconds())

    assert sr.lease_status(_NAME, root=tmp_path).reclaimable is False
    assert sr.reap_stale_leases(root=tmp_path) == []
    assert lease.is_file()


def test_resolve_own_record_slug_is_none_for_a_shared_host(
    shared_host_pid: int, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The host's pointer is last-writer-wins across every hosted session.

    So it may name a SIBLING's record, and a DecisionRecord must read unlinked
    rather than be mis-attributed.
    """
    _seed_sibling_record(tmp_path, shared_host_pid)
    caplog.set_level(logging.DEBUG)
    env = {'CLAUDE_PID': str(shared_host_pid), 'CLAUDE_CODE_SESSION_ID': 'sess-a'}

    assert sr.resolve_own_record_slug(env, root=tmp_path) is None
    assert _warnings(caplog) == []
