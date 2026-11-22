"""Tests for orchestrator.session_reaper: the liveness sweeps, the stale-record and pointer reaps, and run_reap.

The ``reap`` CLI verb is covered in test_session_registry.py, the window-id
contract in test_session_windows.py.
"""

from __future__ import annotations

import fcntl
import logging
import os
import re
import subprocess
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from _process_identity_variants import IdentityVariant, own_identity
from _session_record_fixtures import make_record, set_mtime, write_t3_record
from shared.process_identity import ProcessIdentity

from orchestrator import fleet_files, session_reaper, session_records, session_windows

# A pid virtually guaranteed dead on any host (matches the convention in
# shared/tests/test_proc_liveness.py: 2**31 - 1).
_DEAD_PID = 2**31 - 1

_NOW = datetime(2026, 7, 7, 12, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Step-7: TTL / pid stale-record reaper matrix
# ---------------------------------------------------------------------------


def test_ttl_constants_are_module_level_timedeltas() -> None:
    assert timedelta(hours=24) == session_reaper.TERMINAL_TTL
    assert timedelta(hours=1) == session_reaper.NON_TERMINAL_HEARTBEAT_TTL


def test_reap_removes_terminal_record_past_ttl(tmp_path: Path) -> None:
    r = make_record(status=session_records.Status.EXITED, launcher_pid=os.getpid())
    session_records.write_record(r, root=tmp_path)
    record_path = session_records.record_path_for_slug(r.session_slug, root=tmp_path)
    set_mtime(record_path, _NOW, session_reaper.TERMINAL_TTL + timedelta(hours=1))

    reaped = session_reaper.reap_stale_records(root=tmp_path, now=_NOW)

    assert len(reaped) == 1
    assert reaped[0].session_slug == r.session_slug
    assert reaped[0].reason == 'terminal_ttl'
    assert reaped[0].path == record_path.parent
    assert not record_path.parent.exists()


def test_reap_keeps_terminal_record_within_ttl(tmp_path: Path) -> None:
    r = make_record(status=session_records.Status.EXITED, launcher_pid=os.getpid())
    session_records.write_record(r, root=tmp_path)
    record_path = session_records.record_path_for_slug(r.session_slug, root=tmp_path)
    set_mtime(record_path, _NOW, session_reaper.TERMINAL_TTL - timedelta(hours=1))

    reaped = session_reaper.reap_stale_records(root=tmp_path, now=_NOW)

    assert reaped == []
    assert record_path.is_file()


def test_reap_removes_non_terminal_dead_pid_past_heartbeat_ttl(tmp_path: Path) -> None:
    r = make_record(status=session_records.Status.LAUNCHING, launcher_pid=_DEAD_PID)
    session_records.write_record(r, root=tmp_path)
    record_path = session_records.record_path_for_slug(r.session_slug, root=tmp_path)
    set_mtime(record_path, _NOW, session_reaper.NON_TERMINAL_HEARTBEAT_TTL + timedelta(minutes=1))

    reaped = session_reaper.reap_stale_records(root=tmp_path, now=_NOW)

    assert len(reaped) == 1
    assert reaped[0].session_slug == r.session_slug
    assert reaped[0].reason == 'stale_pid'
    assert not record_path.parent.exists()


def test_reap_keeps_non_terminal_live_pid_regardless_of_age(tmp_path: Path) -> None:
    r = make_record(status=session_records.Status.RUNNING, launcher_pid=os.getpid())
    session_records.write_record(r, root=tmp_path)
    record_path = session_records.record_path_for_slug(r.session_slug, root=tmp_path)
    set_mtime(record_path, _NOW, timedelta(days=30))  # past both TTLs

    reaped = session_reaper.reap_stale_records(root=tmp_path, now=_NOW)

    assert reaped == []
    assert record_path.is_file()


def test_reap_keeps_non_terminal_dead_pid_within_heartbeat_ttl(tmp_path: Path) -> None:
    r = make_record(status=session_records.Status.LAUNCHING, launcher_pid=_DEAD_PID)
    session_records.write_record(r, root=tmp_path)
    record_path = session_records.record_path_for_slug(r.session_slug, root=tmp_path)
    set_mtime(record_path, _NOW, session_reaper.NON_TERMINAL_HEARTBEAT_TTL - timedelta(minutes=1))

    reaped = session_reaper.reap_stale_records(root=tmp_path, now=_NOW)

    assert reaped == []
    assert record_path.is_file()


def test_reap_removes_corrupt_body_past_heartbeat_ttl(tmp_path: Path) -> None:
    r = make_record(status=session_records.Status.LAUNCHING, launcher_pid=os.getpid())
    session_records.write_record(r, root=tmp_path)
    record_path = session_records.record_path_for_slug(r.session_slug, root=tmp_path)
    record_path.write_text('{not valid json')
    set_mtime(record_path, _NOW, session_reaper.NON_TERMINAL_HEARTBEAT_TTL + timedelta(minutes=1))

    reaped = session_reaper.reap_stale_records(root=tmp_path, now=_NOW)

    assert len(reaped) == 1
    assert reaped[0].session_slug == r.session_slug
    assert reaped[0].reason == 'corrupt'
    assert not record_path.parent.exists()


def test_reap_removes_missing_body_past_heartbeat_ttl(tmp_path: Path) -> None:
    """A slug dir with no record.json at all is still reaped by path identity."""
    r = make_record(status=session_records.Status.LAUNCHING, launcher_pid=os.getpid())
    session_records.write_record(r, root=tmp_path)
    record_path = session_records.record_path_for_slug(r.session_slug, root=tmp_path)
    record_path.unlink()
    set_mtime(record_path.parent, _NOW, session_reaper.NON_TERMINAL_HEARTBEAT_TTL + timedelta(minutes=1))

    reaped = session_reaper.reap_stale_records(root=tmp_path, now=_NOW)

    assert len(reaped) == 1
    assert reaped[0].session_slug == r.session_slug
    assert reaped[0].reason == 'corrupt'
    assert not record_path.parent.exists()


def test_reap_handles_mixed_population_in_one_sweep(tmp_path: Path) -> None:
    """The reaper's whole point is sweeping many records in one pass; pin that
    a kept/terminal_ttl/stale_pid/corrupt mix in a single sessions dir reaps
    exactly the expected subset in one call and leaves survivors untouched.
    """
    kept_running = make_record(
        session_slug='kept-running', status=session_records.Status.RUNNING, launcher_pid=os.getpid()
    )
    kept_recent_terminal = make_record(
        session_slug='kept-recent-terminal', status=session_records.Status.EXITED, launcher_pid=os.getpid()
    )
    reap_terminal = make_record(
        session_slug='reap-terminal', status=session_records.Status.EXITED, launcher_pid=os.getpid()
    )
    reap_stale = make_record(
        session_slug='reap-stale-pid', status=session_records.Status.LAUNCHING, launcher_pid=_DEAD_PID
    )

    for r in (kept_running, kept_recent_terminal, reap_terminal, reap_stale):
        session_records.write_record(r, root=tmp_path)

    set_mtime(
        session_records.record_path_for_slug(kept_running.session_slug, root=tmp_path),
        _NOW,
        timedelta(days=30),  # long-lived but a live pid -> kept regardless of age
    )
    set_mtime(
        session_records.record_path_for_slug(kept_recent_terminal.session_slug, root=tmp_path),
        _NOW,
        session_reaper.TERMINAL_TTL - timedelta(hours=1),
    )
    set_mtime(
        session_records.record_path_for_slug(reap_terminal.session_slug, root=tmp_path),
        _NOW,
        session_reaper.TERMINAL_TTL + timedelta(hours=1),
    )
    set_mtime(
        session_records.record_path_for_slug(reap_stale.session_slug, root=tmp_path),
        _NOW,
        session_reaper.NON_TERMINAL_HEARTBEAT_TTL + timedelta(minutes=1),
    )

    # A corrupt-body record dir, aged past the heartbeat TTL.
    corrupt_dir = session_records.sessions_dir(root=tmp_path) / 'reap-corrupt'
    corrupt_dir.mkdir(parents=True)
    corrupt_record = corrupt_dir / 'record.json'
    corrupt_record.write_text('{not valid json')
    set_mtime(corrupt_record, _NOW, session_reaper.NON_TERMINAL_HEARTBEAT_TTL + timedelta(minutes=1))

    reaped = session_reaper.reap_stale_records(root=tmp_path, now=_NOW)

    reaped_by_slug = {r.session_slug: r.reason for r in reaped}
    assert reaped_by_slug == {
        'reap-terminal': 'terminal_ttl',
        'reap-stale-pid': 'stale_pid',
        'reap-corrupt': 'corrupt',
    }

    remaining = {p.name for p in session_records.sessions_dir(root=tmp_path).iterdir()}
    assert remaining == {'kept-running', 'kept-recent-terminal'}


def test_reap_continues_sweep_when_one_directory_fails_to_remove(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A single unreapable directory (permission error, a file held open, an
    ENOTEMPTY race with a concurrent writer) must not abort the sweep -- a
    later stale directory in the same sessions dir is still reaped.
    """
    poison = make_record(session_slug='poison', status=session_records.Status.EXITED, launcher_pid=os.getpid())
    victim = make_record(session_slug='victim', status=session_records.Status.EXITED, launcher_pid=os.getpid())
    for r in (poison, victim):
        session_records.write_record(r, root=tmp_path)
        set_mtime(
            session_records.record_path_for_slug(r.session_slug, root=tmp_path),
            _NOW,
            session_reaper.TERMINAL_TTL + timedelta(hours=1),
        )

    poison_dir = session_records.record_path_for_slug(poison.session_slug, root=tmp_path).parent
    real_rmtree = session_reaper.shutil.rmtree

    def _flaky_rmtree(path: str | os.PathLike[str], *args: Any, **kwargs: Any) -> None:
        if Path(path) == poison_dir:
            raise OSError('simulated ENOTEMPTY race')
        real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(session_reaper.shutil, 'rmtree', _flaky_rmtree)

    with caplog.at_level(logging.ERROR):
        reaped = session_reaper.reap_stale_records(root=tmp_path, now=_NOW)

    assert {r.session_slug for r in reaped} == {'victim'}
    assert poison_dir.is_dir()  # left behind, not silently claimed as reaped
    assert any(r.levelno >= logging.ERROR for r in caplog.records)


# ---------------------------------------------------------------------------
# Task 2511 step-5: mark_orphaned_sessions_exited (liveness sweep, Part 2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'status',
    [session_records.Status.LAUNCHING, session_records.Status.RUNNING, session_records.Status.IDLE, session_records.Status.AWAITING_INPUT],
)
def test_mark_orphaned_marks_non_terminal_dead_pid_past_heartbeat_ttl(
    status: session_records.Status, tmp_path: Path
) -> None:
    r = make_record(session_slug='orphan-1', status=status, launcher_pid=_DEAD_PID, exit_code=None)
    session_records.write_record(r, root=tmp_path)
    record_path = session_records.record_path_for_slug(r.session_slug, root=tmp_path)
    set_mtime(record_path, _NOW, session_reaper.NON_TERMINAL_HEARTBEAT_TTL + timedelta(minutes=1))

    marked = session_reaper.mark_orphaned_sessions_exited(root=tmp_path, now=_NOW)

    assert {m.session_slug for m in marked} == {'orphan-1'}
    assert record_path.parent.is_dir()  # marked, NOT deleted (reap_stale_records' job)
    assert record_path.is_file()
    reloaded = session_records.read_record('orphan-1', root=tmp_path)
    assert reloaded.status == session_records.Status.EXITED
    assert reloaded.exit_code == session_reaper.ORPHAN_EXIT_CODE


def test_mark_orphaned_keeps_non_terminal_live_pid_regardless_of_age(tmp_path: Path) -> None:
    r = make_record(session_slug='kept-live', status=session_records.Status.RUNNING, launcher_pid=os.getpid())
    session_records.write_record(r, root=tmp_path)
    record_path = session_records.record_path_for_slug(r.session_slug, root=tmp_path)
    set_mtime(record_path, _NOW, timedelta(days=30))  # past both TTLs

    marked = session_reaper.mark_orphaned_sessions_exited(root=tmp_path, now=_NOW)

    assert marked == []
    assert session_records.read_record('kept-live', root=tmp_path).status == session_records.Status.RUNNING


def test_mark_orphaned_keeps_non_terminal_dead_pid_within_heartbeat_ttl(tmp_path: Path) -> None:
    r = make_record(session_slug='kept-recent', status=session_records.Status.LAUNCHING, launcher_pid=_DEAD_PID)
    session_records.write_record(r, root=tmp_path)
    record_path = session_records.record_path_for_slug(r.session_slug, root=tmp_path)
    set_mtime(record_path, _NOW, session_reaper.NON_TERMINAL_HEARTBEAT_TTL - timedelta(minutes=1))

    marked = session_reaper.mark_orphaned_sessions_exited(root=tmp_path, now=_NOW)

    assert marked == []
    assert session_records.read_record('kept-recent', root=tmp_path).status == session_records.Status.LAUNCHING


def test_mark_orphaned_leaves_already_terminal_record_untouched(tmp_path: Path) -> None:
    r = make_record(
        session_slug='already-exited', status=session_records.Status.EXITED, exit_code=0, launcher_pid=_DEAD_PID
    )
    session_records.write_record(r, root=tmp_path)
    record_path = session_records.record_path_for_slug(r.session_slug, root=tmp_path)
    set_mtime(record_path, _NOW, session_reaper.NON_TERMINAL_HEARTBEAT_TTL + timedelta(days=2))

    marked = session_reaper.mark_orphaned_sessions_exited(root=tmp_path, now=_NOW)

    assert marked == []
    reloaded = session_records.read_record('already-exited', root=tmp_path)
    assert reloaded.status == session_records.Status.EXITED
    assert reloaded.exit_code == 0  # NOT re-stamped with ORPHAN_EXIT_CODE


def test_mark_orphaned_preserves_all_other_fields(tmp_path: Path) -> None:
    r = make_record(
        session_slug='orphan-fields',
        status=session_records.Status.RUNNING,
        launcher_pid=_DEAD_PID,
        role='unblock',
        project='df',
        task_id='2085',
        result_file='/tmp/whatever/result.md',
        start_ts='2026-07-07T00:00:00+00:00',
    )
    session_records.write_record(r, root=tmp_path)
    record_path = session_records.record_path_for_slug(r.session_slug, root=tmp_path)
    set_mtime(record_path, _NOW, session_reaper.NON_TERMINAL_HEARTBEAT_TTL + timedelta(minutes=1))

    session_reaper.mark_orphaned_sessions_exited(root=tmp_path, now=_NOW)

    reloaded = session_records.read_record('orphan-fields', root=tmp_path)
    assert reloaded.status == session_records.Status.EXITED
    assert reloaded.exit_code == session_reaper.ORPHAN_EXIT_CODE
    assert reloaded.role == 'unblock'
    assert reloaded.project == 'df'
    assert reloaded.task_id == '2085'
    assert reloaded.result_file == '/tmp/whatever/result.md'
    assert reloaded.launcher_pid == _DEAD_PID
    assert reloaded.start_ts == '2026-07-07T00:00:00+00:00'


def test_mark_orphaned_handles_mixed_population_in_one_sweep(tmp_path: Path) -> None:
    marked_orphan = make_record(
        session_slug='will-be-marked', status=session_records.Status.RUNNING, launcher_pid=_DEAD_PID
    )
    kept_live = make_record(
        session_slug='kept-live-mix', status=session_records.Status.RUNNING, launcher_pid=os.getpid()
    )
    kept_recent = make_record(
        session_slug='kept-recent-mix', status=session_records.Status.LAUNCHING, launcher_pid=_DEAD_PID
    )
    kept_terminal = make_record(
        session_slug='kept-terminal-mix',
        status=session_records.Status.EXITED,
        exit_code=0,
        launcher_pid=_DEAD_PID,
    )
    for r in (marked_orphan, kept_live, kept_recent, kept_terminal):
        session_records.write_record(r, root=tmp_path)

    set_mtime(
        session_records.record_path_for_slug('will-be-marked', root=tmp_path),
        _NOW,
        session_reaper.NON_TERMINAL_HEARTBEAT_TTL + timedelta(minutes=1),
    )
    set_mtime(
        session_records.record_path_for_slug('kept-live-mix', root=tmp_path),
        _NOW,
        timedelta(days=30),
    )
    set_mtime(
        session_records.record_path_for_slug('kept-recent-mix', root=tmp_path),
        _NOW,
        session_reaper.NON_TERMINAL_HEARTBEAT_TTL - timedelta(minutes=1),
    )
    set_mtime(
        session_records.record_path_for_slug('kept-terminal-mix', root=tmp_path),
        _NOW,
        session_reaper.NON_TERMINAL_HEARTBEAT_TTL + timedelta(days=2),
    )

    marked = session_reaper.mark_orphaned_sessions_exited(root=tmp_path, now=_NOW)

    assert {m.session_slug for m in marked} == {'will-be-marked'}
    assert session_records.read_record('will-be-marked', root=tmp_path).status == session_records.Status.EXITED
    assert session_records.read_record('kept-live-mix', root=tmp_path).status == session_records.Status.RUNNING
    assert session_records.read_record('kept-recent-mix', root=tmp_path).status == session_records.Status.LAUNCHING
    reloaded_terminal = session_records.read_record('kept-terminal-mix', root=tmp_path)
    assert reloaded_terminal.status == session_records.Status.EXITED
    assert reloaded_terminal.exit_code == 0


# ---------------------------------------------------------------------------
# Task 8125: the pid sweeps judge the launcher by its identity
# ---------------------------------------------------------------------------

def _launcher_slug(variant: IdentityVariant) -> str:
    return f'launcher-{variant}'


_UNQUALIFIED = 'launcher-unqualified'
_ELSEWHERE = _launcher_slug(IdentityVariant.OTHER_HOST)
_JUDGED_DEAD = {_launcher_slug(v) for v in IdentityVariant if v.judged_dead}
_JUDGED_KEPT = {_launcher_slug(v) for v in IdentityVariant if not v.judged_dead} | {_UNQUALIFIED}


def _launcher_identities() -> dict[str, ProcessIdentity | None]:
    """One record slug per identity, all naming this live test process: only the identity can make one dead."""
    return {_launcher_slug(v): own_identity(v) for v in IdentityVariant} | {_UNQUALIFIED: None}


def _seed_stale_launchers(root: Path) -> None:
    for slug, identity in _launcher_identities().items():
        record = make_record(
            session_slug=slug,
            status=session_records.Status.RUNNING,
            launcher_pid=os.getpid(),
            launcher_identity=identity,
        )
        session_records.write_record(record, root=root)
        set_mtime(
            session_records.record_path_for_slug(slug, root=root),
            _NOW,
            session_reaper.NON_TERMINAL_HEARTBEAT_TTL + timedelta(minutes=1),
        )


def _logged_other_host_skip(caplog: pytest.LogCaptureFixture, sweep: str, subject: str) -> bool:
    return any(
        sweep in r.getMessage() and subject in r.getMessage() and 'reason=other_host' in r.getMessage()
        for r in caplog.records
        if r.name == session_reaper.__name__ and r.levelno == logging.INFO
    )


class TestPidSweepsJudgeTheLauncherByIdentity:
    def test_mark_orphaned_marks_exactly_the_launchers_judged_dead(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        _seed_stale_launchers(tmp_path)

        with caplog.at_level(logging.INFO, logger=session_reaper.__name__):
            marked = session_reaper.mark_orphaned_sessions_exited(root=tmp_path, now=_NOW)

        assert {m.session_slug for m in marked} == _JUDGED_DEAD
        for slug in _JUDGED_KEPT:
            assert session_records.read_record(slug, root=tmp_path).status == session_records.Status.RUNNING
        assert _logged_other_host_skip(caplog, 'mark_orphaned_sessions_exited', _ELSEWHERE)

    def test_reap_stale_records_reaps_exactly_the_launchers_judged_dead(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        _seed_stale_launchers(tmp_path)

        with caplog.at_level(logging.INFO, logger=session_reaper.__name__):
            reaped = session_reaper.reap_stale_records(root=tmp_path, now=_NOW)

        assert {r.session_slug: r.reason for r in reaped} == dict.fromkeys(
            _JUDGED_DEAD, 'stale_pid'
        )
        for slug in _JUDGED_KEPT:
            assert session_records.record_path_for_slug(slug, root=tmp_path).is_file()
        assert _logged_other_host_skip(caplog, 'reap_stale_records', _ELSEWHERE)


# ---------------------------------------------------------------------------
# mark_windowless_wm_sessions_exited reads windows through a WindowProbe (PRD B9)
# ---------------------------------------------------------------------------


def _fixed_window_probe(live: frozenset[str] | None) -> session_windows.WindowProbe:
    """A probe whose every listing answers *live*. The sweep never resolves a title."""

    def _no_title_lookup(title: str) -> str | None:
        pytest.fail(f'the window sweep resolved a title: {title!r}')

    return session_windows.WindowProbe(live_window_ids=lambda: live, resolve_by_title=_no_title_lookup)


def _decimal_captured_wm_record(tmp_path: Path) -> None:
    session_records.write_record(
        make_record(
            session_slug='decimal-captured',
            status=session_records.Status.RUNNING,
            display=session_records.Display(kind='wm', wm_title='t', wm_window_id='52428807'),
        ),
        root=tmp_path,
    )


def test_a_decimal_captured_window_the_probe_lists_in_canonical_form_is_kept(tmp_path: Path) -> None:
    _decimal_captured_wm_record(tmp_path)

    marked = session_reaper.mark_windowless_wm_sessions_exited(
        root=tmp_path, windows=_fixed_window_probe(frozenset({'0x03200007'}))
    )

    assert marked == []
    assert session_records.read_record('decimal-captured', root=tmp_path).status == session_records.Status.RUNNING


def test_a_decimal_captured_window_the_probe_does_not_list_is_marked_exited(tmp_path: Path) -> None:
    _decimal_captured_wm_record(tmp_path)

    marked = session_reaper.mark_windowless_wm_sessions_exited(
        root=tmp_path, windows=_fixed_window_probe(frozenset())
    )

    assert [m.session_slug for m in marked] == ['decimal-captured']
    reloaded = session_records.read_record('decimal-captured', root=tmp_path)
    assert (reloaded.status, reloaded.exit_code) == (session_records.Status.EXITED, session_reaper.ORPHAN_EXIT_CODE)


def test_a_probe_that_cannot_list_windows_marks_nothing(tmp_path: Path) -> None:
    _decimal_captured_wm_record(tmp_path)

    marked = session_reaper.mark_windowless_wm_sessions_exited(root=tmp_path, windows=_fixed_window_probe(None))

    assert marked == []
    assert session_records.read_record('decimal-captured', root=tmp_path).status == session_records.Status.RUNNING


def _fake_wmctrl_run(
    stdout_lines: list[str], returncode: int = 0
) -> Callable[[list[str]], subprocess.CompletedProcess[str]]:
    """Build a fake ``run`` callable returning a canned ``wmctrl -l`` result."""

    def _run(argv: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            argv, returncode=returncode, stdout='\n'.join(stdout_lines)
        )

    return _run


# ---------------------------------------------------------------------------
# Task 2934 step-5/6: mark_windowless_wm_sessions_exited (happy path)
# ---------------------------------------------------------------------------


def test_mark_windowless_marks_wm_record_whose_window_id_is_absent(tmp_path: Path) -> None:
    r = make_record(
        session_slug='windowless-1',
        status=session_records.Status.AWAITING_INPUT,
        display=session_records.Display(kind='wm', wm_title='some title', wm_window_id='0x1a'),
    )
    session_records.write_record(r, root=tmp_path)
    run = _fake_wmctrl_run(['0x0000ff99  0 host  some other window'])

    marked = session_reaper.mark_windowless_wm_sessions_exited(
        root=tmp_path, windows=session_windows.wmctrl_window_probe(run=run)
    )

    assert {m.session_slug for m in marked} == {'windowless-1'}
    record_path = session_records.record_path_for_slug('windowless-1', root=tmp_path)
    assert record_path.parent.is_dir()  # marked, NOT deleted
    assert record_path.is_file()
    reloaded = session_records.read_record('windowless-1', root=tmp_path)
    assert reloaded.status == session_records.Status.EXITED
    assert reloaded.exit_code == session_reaper.ORPHAN_EXIT_CODE


def test_mark_windowless_keeps_decimal_captured_id_matching_padded_hex_live_window(
    tmp_path: Path,
) -> None:
    # DECIMAL-vs-hex KEEP: '26' (decimal) is the SAME window as the live,
    # zero-padded-hex '0x0000001a' wmctrl reports -- must NOT be reaped.
    r = make_record(
        session_slug='decimal-kept',
        status=session_records.Status.RUNNING,
        display=session_records.Display(kind='wm', wm_title='t', wm_window_id='26'),
    )
    session_records.write_record(r, root=tmp_path)
    run = _fake_wmctrl_run(['0x0000001a  0 host  live window'])

    marked = session_reaper.mark_windowless_wm_sessions_exited(
        root=tmp_path, windows=session_windows.wmctrl_window_probe(run=run)
    )

    assert marked == []
    assert session_records.read_record('decimal-kept', root=tmp_path).status == session_records.Status.RUNNING


def test_mark_windowless_preserves_all_other_fields(tmp_path: Path) -> None:
    r = make_record(
        session_slug='windowless-fields',
        status=session_records.Status.IDLE,
        role='unblock',
        project='df',
        task_id='2085',
        launcher_pid=os.getpid(),
        start_ts='2026-07-07T00:00:00+00:00',
        display=session_records.Display(kind='wm', wm_title='t', wm_window_id='0x1a'),
    )
    session_records.write_record(r, root=tmp_path)
    run = _fake_wmctrl_run(['0x99  0 host  other window'])

    session_reaper.mark_windowless_wm_sessions_exited(
        root=tmp_path, windows=session_windows.wmctrl_window_probe(run=run)
    )

    reloaded = session_records.read_record('windowless-fields', root=tmp_path)
    assert reloaded.status == session_records.Status.EXITED
    assert reloaded.exit_code == session_reaper.ORPHAN_EXIT_CODE
    assert reloaded.role == 'unblock'
    assert reloaded.project == 'df'
    assert reloaded.task_id == '2085'
    assert reloaded.launcher_pid == os.getpid()
    assert reloaded.start_ts == '2026-07-07T00:00:00+00:00'


# ---------------------------------------------------------------------------
# Task 2934 step-7/8: mark_windowless_wm_sessions_exited fail-soft, scope,
# and pid/age-independence
# ---------------------------------------------------------------------------


def test_mark_windowless_fail_soft_when_wmctrl_unavailable(tmp_path: Path) -> None:
    r = make_record(
        session_slug='no-wmctrl',
        status=session_records.Status.AWAITING_INPUT,
        display=session_records.Display(kind='wm', wm_title='t', wm_window_id='0x1a'),
    )
    session_records.write_record(r, root=tmp_path)
    run = _fake_wmctrl_run([], returncode=127)  # missing binary -- no window evidence

    marked = session_reaper.mark_windowless_wm_sessions_exited(
        root=tmp_path, windows=session_windows.wmctrl_window_probe(run=run)
    )

    assert marked == []
    # A headless run reaps NOTHING, even though this record's window would
    # (if we could prove it) be gone -- "no evidence" != "window gone".
    assert session_records.read_record('no-wmctrl', root=tmp_path).status == session_records.Status.AWAITING_INPUT


def test_mark_windowless_never_touches_tmux_display(tmp_path: Path) -> None:
    r = make_record(
        session_slug='tmux-session',
        status=session_records.Status.RUNNING,
        display=session_records.Display(kind='tmux', wm_title='', wm_window_id=None, tmux_target='fleet-df:2'),
    )
    session_records.write_record(r, root=tmp_path)
    run = _fake_wmctrl_run(['0x99  0 host  unrelated'])

    marked = session_reaper.mark_windowless_wm_sessions_exited(
        root=tmp_path, windows=session_windows.wmctrl_window_probe(run=run)
    )

    assert marked == []
    assert session_records.read_record('tmux-session', root=tmp_path).status == session_records.Status.RUNNING


def test_mark_windowless_never_touches_displayless_record(tmp_path: Path) -> None:
    r = make_record(session_slug='headless', status=session_records.Status.RUNNING, display=None)
    session_records.write_record(r, root=tmp_path)
    run = _fake_wmctrl_run(['0x99  0 host  unrelated'])

    marked = session_reaper.mark_windowless_wm_sessions_exited(
        root=tmp_path, windows=session_windows.wmctrl_window_probe(run=run)
    )

    assert marked == []
    assert session_records.read_record('headless', root=tmp_path).status == session_records.Status.RUNNING


def test_mark_windowless_leaves_already_terminal_wm_record_untouched(tmp_path: Path) -> None:
    r = make_record(
        session_slug='already-exited-wm',
        status=session_records.Status.EXITED,
        exit_code=0,
        display=session_records.Display(kind='wm', wm_title='t', wm_window_id='0x1a'),
    )
    session_records.write_record(r, root=tmp_path)
    run = _fake_wmctrl_run(['0x99  0 host  unrelated'])  # window gone, but already terminal

    marked = session_reaper.mark_windowless_wm_sessions_exited(
        root=tmp_path, windows=session_windows.wmctrl_window_probe(run=run)
    )

    assert marked == []
    reloaded = session_records.read_record('already-exited-wm', root=tmp_path)
    assert reloaded.status == session_records.Status.EXITED
    assert reloaded.exit_code == 0  # NOT re-stamped with ORPHAN_EXIT_CODE


@pytest.mark.parametrize(
    'wm_window_id', [None, '0xZZ', 'nope'], ids=['none', 'unparseable-hex', 'unparseable-decimal']
)
def test_mark_windowless_skips_unparseable_or_missing_window_id(
    wm_window_id: str | None, tmp_path: Path
) -> None:
    r = make_record(
        session_slug='no-id',
        status=session_records.Status.RUNNING,
        display=session_records.Display(kind='wm', wm_title='t', wm_window_id=wm_window_id),
    )
    session_records.write_record(r, root=tmp_path)
    run = _fake_wmctrl_run(['0x99  0 host  unrelated'])

    marked = session_reaper.mark_windowless_wm_sessions_exited(
        root=tmp_path, windows=session_windows.wmctrl_window_probe(run=run)
    )

    assert marked == []
    assert session_records.read_record('no-id', root=tmp_path).status == session_records.Status.RUNNING


def test_mark_windowless_reaps_regardless_of_live_pid_and_fresh_age(tmp_path: Path) -> None:
    """THE CRUX: window-gone is definitive death evidence, independent of
    launcher_pid liveness or record age -- proving this sweep reaps exactly
    the zombies mark_orphaned_sessions_exited's pid/TTL guards currently keep.
    """
    r = make_record(
        session_slug='live-pid-fresh-but-windowless',
        status=session_records.Status.AWAITING_INPUT,
        launcher_pid=os.getpid(),  # ALIVE
        display=session_records.Display(kind='wm', wm_title='t', wm_window_id='0x1a'),
    )
    session_records.write_record(r, root=tmp_path)
    record_path = session_records.record_path_for_slug('live-pid-fresh-but-windowless', root=tmp_path)
    set_mtime(record_path, _NOW, timedelta(minutes=1))  # far under NON_TERMINAL_HEARTBEAT_TTL
    run = _fake_wmctrl_run(['0x99  0 host  unrelated'])

    marked = session_reaper.mark_windowless_wm_sessions_exited(
        root=tmp_path, windows=session_windows.wmctrl_window_probe(run=run)
    )

    assert {m.session_slug for m in marked} == {'live-pid-fresh-but-windowless'}
    reloaded = session_records.read_record('live-pid-fresh-but-windowless', root=tmp_path)
    assert reloaded.status == session_records.Status.EXITED
    assert reloaded.exit_code == session_reaper.ORPHAN_EXIT_CODE


def test_mark_windowless_handles_mixed_population_in_one_sweep(tmp_path: Path) -> None:
    windowless_wm = make_record(
        session_slug='mix-windowless',
        status=session_records.Status.RUNNING,
        display=session_records.Display(kind='wm', wm_title='t', wm_window_id='0x1a'),
    )
    present_window_wm = make_record(
        session_slug='mix-present',
        status=session_records.Status.RUNNING,
        display=session_records.Display(kind='wm', wm_title='t2', wm_window_id='0x99'),
    )
    tmux_session = make_record(
        session_slug='mix-tmux',
        status=session_records.Status.RUNNING,
        display=session_records.Display(kind='tmux', wm_title='', wm_window_id=None, tmux_target='fleet-df:2'),
    )
    headless = make_record(session_slug='mix-headless', status=session_records.Status.RUNNING, display=None)
    terminal_wm = make_record(
        session_slug='mix-terminal',
        status=session_records.Status.EXITED,
        exit_code=0,
        display=session_records.Display(kind='wm', wm_title='t3', wm_window_id='0x1a'),
    )
    for r in (windowless_wm, present_window_wm, tmux_session, headless, terminal_wm):
        session_records.write_record(r, root=tmp_path)

    run = _fake_wmctrl_run(['0x99  0 host  still-open'])

    marked = session_reaper.mark_windowless_wm_sessions_exited(
        root=tmp_path, windows=session_windows.wmctrl_window_probe(run=run)
    )

    assert {m.session_slug for m in marked} == {'mix-windowless'}
    assert session_records.read_record('mix-windowless', root=tmp_path).status == session_records.Status.EXITED
    assert session_records.read_record('mix-present', root=tmp_path).status == session_records.Status.RUNNING
    assert session_records.read_record('mix-tmux', root=tmp_path).status == session_records.Status.RUNNING
    assert session_records.read_record('mix-headless', root=tmp_path).status == session_records.Status.RUNNING
    reloaded_terminal = session_records.read_record('mix-terminal', root=tmp_path)
    assert reloaded_terminal.status == session_records.Status.EXITED
    assert reloaded_terminal.exit_code == 0


def _call_indexed_wmctrl_run(
    *results: tuple[int, str],
    on_call: Mapping[int, Callable[[], None]] | None = None,
) -> Callable[[list[str]], subprocess.CompletedProcess[str]]:
    """A fake ``run`` answering call N with ``results[N]``, repeating the last thereafter.

    ``on_call[N]`` runs inside call N, before it answers -- the hook a test
    uses to land a record write just after one of the sweep's probes.
    """
    calls = 0

    def _run(argv: list[str]) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        if on_call is not None and calls in on_call:
            on_call[calls]()
        returncode, stdout = results[min(calls, len(results) - 1)]
        calls += 1
        return subprocess.CompletedProcess(argv, returncode=returncode, stdout=stdout)

    return _run


def test_mark_windowless_keeps_a_window_that_appears_after_the_first_probe(
    tmp_path: Path,
) -> None:
    def _set_display_lands() -> None:
        session_records.write_record(
            make_record(
                session_slug='late-window',
                status=session_records.Status.RUNNING,
                display=session_records.Display(kind='wm', wm_title='t', wm_window_id='0x1a'),
            ),
            root=tmp_path,
        )

    run = _call_indexed_wmctrl_run(
        (0, '0x99  0 host  other\n'),
        (0, '0x1a  0 host  late\n0x99  0 host  other\n'),
        on_call={0: _set_display_lands},
    )

    marked = session_reaper.mark_windowless_wm_sessions_exited(
        root=tmp_path, windows=session_windows.wmctrl_window_probe(run=run)
    )

    assert 'late-window' not in {m.session_slug for m in marked}
    assert session_records.read_record('late-window', root=tmp_path).status == session_records.Status.RUNNING


def test_mark_windowless_marks_nothing_when_the_confirming_probe_fails(tmp_path: Path) -> None:
    session_records.write_record(
        make_record(
            session_slug='unconfirmed',
            status=session_records.Status.RUNNING,
            display=session_records.Display(kind='wm', wm_title='t', wm_window_id='0x1a'),
        ),
        root=tmp_path,
    )
    run = _call_indexed_wmctrl_run((0, '0x99  0 host  other\n'), (127, ''))

    marked = session_reaper.mark_windowless_wm_sessions_exited(
        root=tmp_path, windows=session_windows.wmctrl_window_probe(run=run)
    )

    assert marked == []
    reloaded = session_records.read_record('unconfirmed', root=tmp_path)
    assert reloaded.status == session_records.Status.RUNNING
    assert reloaded.exit_code is None


def test_mark_windowless_leaves_a_record_rebound_to_a_live_window_after_the_pass(
    tmp_path: Path,
) -> None:
    def _write_wm_record(window_id: str) -> None:
        session_records.write_record(
            make_record(
                session_slug='rebound',
                status=session_records.Status.RUNNING,
                display=session_records.Display(kind='wm', wm_title='t', wm_window_id=window_id),
            ),
            root=tmp_path,
        )

    _write_wm_record('0x1a')
    run = _call_indexed_wmctrl_run(
        (0, '0x99  0 host  other\n'),
        (0, '0x2b  0 host  resumed\n0x99  0 host  other\n'),
        on_call={1: lambda: _write_wm_record('0x2b')},
    )

    marked = session_reaper.mark_windowless_wm_sessions_exited(
        root=tmp_path, windows=session_windows.wmctrl_window_probe(run=run)
    )

    assert marked == []
    reloaded = session_records.read_record('rebound', root=tmp_path)
    assert reloaded.status == session_records.Status.RUNNING
    assert reloaded.display is not None
    assert reloaded.display.wm_window_id == '0x2b'


# ---------------------------------------------------------------------------
# The reap lock (task 7944)
# ---------------------------------------------------------------------------


def test_reap_lock_is_a_stable_sidecar_under_the_fleet_root(tmp_path: Path) -> None:
    lock_path = session_reaper.reap_lock_path(tmp_path)
    with session_reaper.reap_lock(tmp_path):
        assert lock_path.is_file()
        assert lock_path.parent == tmp_path
    assert lock_path.is_file()


def test_reap_lock_excludes_another_holder_until_released(tmp_path: Path) -> None:
    session_reaper.reap_lock_path(tmp_path).touch()
    fd = os.open(session_reaper.reap_lock_path(tmp_path), os.O_RDWR)
    try:
        with session_reaper.reap_lock(tmp_path), pytest.raises(BlockingIOError):
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# Task 7489: the pointer reap and the T3 lane's sweeps
# ---------------------------------------------------------------------------

_SID = '0b6a3c1e-1111-4222-8333-444455556666'
_THREAD = '7d1f0c2a-5b3e-4c8d-9a6f-1e2d3c4b5a69'
_OTHER_THREAD = 'mcp:launch:' + '0f9e8d7c6b5a49388271605f4e3d2c1b' * 2


class TestReapStaleSessionPointers:
    def test_reap_removes_pointers_no_record_vouches_for(self, tmp_path: Path) -> None:
        session_records.write_record(make_record(session_slug='vouched', claude_session_id=_SID), root=tmp_path)
        session_records.write_record(
            make_record(session_slug='rebound', claude_session_id='newer-session'), root=tmp_path
        )
        session_records.write_claude_session_pointer(_SID, 'vouched', root=tmp_path)
        session_records.write_claude_session_pointer('old-session', 'rebound', root=tmp_path)
        session_records.write_claude_session_pointer('orphan-session', 'gone-record', root=tmp_path)
        pointers = session_records.claude_session_pointers_dir(tmp_path)
        (pointers / 'bad name').write_text('vouched')
        fresh_temp = pointers / f'inflight{fleet_files.ATOMIC_WRITE_TEMP_SUFFIX}'
        fresh_temp.write_text('vouched')

        removed = session_reaper.reap_stale_session_pointers(root=tmp_path)

        assert {path.name for path in removed} == {'old-session', 'orphan-session', 'bad name'}
        assert sorted(path.name for path in pointers.iterdir()) == sorted([_SID, fresh_temp.name])

    def test_reap_removes_pointers_no_live_t3_record_vouches_for(self, tmp_path: Path) -> None:
        ended, gone, moved = (
            '1a2b3c4d-0000-4000-8000-000000000001',
            '1a2b3c4d-0000-4000-8000-000000000002',
            '1a2b3c4d-0000-4000-8000-000000000003',
        )
        write_t3_record(tmp_path, 'live', _THREAD, cwd=str(tmp_path))
        write_t3_record(tmp_path, 'ended', ended, cwd=str(tmp_path), status=session_records.Status.EXITED)
        write_t3_record(tmp_path, 'moved', _OTHER_THREAD, cwd=str(tmp_path))
        session_records.write_t3_thread_pointer(_THREAD, 'live', root=tmp_path)
        session_records.write_t3_thread_pointer(ended, 'ended', root=tmp_path)
        session_records.write_t3_thread_pointer(gone, 'gone-record', root=tmp_path)
        session_records.write_t3_thread_pointer(moved, 'moved', root=tmp_path)
        pointers = session_records.t3_thread_pointers_dir(tmp_path)
        (pointers / 'bad name').write_text('live')
        fresh_temp = pointers / f'inflight{fleet_files.ATOMIC_WRITE_TEMP_SUFFIX}'
        fresh_temp.write_text('live')

        removed = session_reaper.reap_stale_session_pointers(root=tmp_path)

        assert {path.name for path in removed} == {ended, gone, moved, 'bad name'}
        assert sorted(path.name for path in pointers.iterdir()) == sorted(
            [_THREAD, fresh_temp.name]
        )

    def _write_orphan_pointers(self, tmp_path: Path, *names: str) -> list[Path]:
        for name in names:
            session_records.write_claude_session_pointer(name, 'gone-record', root=tmp_path)
        pointers = session_records.claude_session_pointers_dir(tmp_path)
        return [pointers / name for name in names]

    def test_entry_removed_by_a_concurrent_sweep_is_skipped_silently(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        raced, kept = self._write_orphan_pointers(tmp_path, 'raced-session', 'orphan-session')
        original_unlink = Path.unlink

        def unlink_after_rival(self: Path, missing_ok: bool = False) -> None:
            if self == raced:
                original_unlink(self)
            original_unlink(self, missing_ok=missing_ok)

        monkeypatch.setattr(Path, 'unlink', unlink_after_rival)

        with caplog.at_level(logging.WARNING, logger=session_reaper.__name__):
            removed = session_reaper.reap_stale_session_pointers(root=tmp_path)

        assert removed == [kept]
        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []

    def test_entry_that_cannot_be_removed_still_warns(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stuck, other = self._write_orphan_pointers(tmp_path, 'stuck-session', 'orphan-session')
        original_unlink = Path.unlink

        def unlink_refusing_stuck(self: Path, missing_ok: bool = False) -> None:
            if self == stuck:
                raise PermissionError(13, 'Permission denied', str(self))
            original_unlink(self, missing_ok=missing_ok)

        monkeypatch.setattr(Path, 'unlink', unlink_refusing_stuck)

        with caplog.at_level(logging.WARNING, logger=session_reaper.__name__):
            removed = session_reaper.reap_stale_session_pointers(root=tmp_path)

        assert removed == [other]
        [warning] = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert warning.startswith(f'reap_stale_session_pointers: failed to remove {stuck}: ')


_ENVIRONMENT = '3f9a1c2e-5b7d-4e8f-9a0b-1c2d3e4f5a6b'
_GONE_THREAD = '2c4e6a8b-0d1f-4a3c-9e5b-7d9f1b3d5e70'


class _ThreadEnded:
    """A ``thread_ended`` predicate answering *answer* for every thread, recording each thread asked."""

    def __init__(self, answer: bool = True) -> None:
        self.answer = answer
        self.calls: list[str] = []

    def __call__(self, thread_id: str) -> bool:
        self.calls.append(thread_id)
        return self.answer


class TestMarkT3SessionsExited:
    def test_marks_only_indexed_records_of_this_environment_whose_thread_is_gone(
        self, tmp_path: Path
    ) -> None:
        foreign_thread, unindexed_thread = (
            '4e6a8c0d-2f1b-4c5e-8a7d-9b1d3f5a7c92',
            '6a8c0e2f-4b3d-4e7a-8c9f-1d3f5b7d9e14',
        )
        cwd = str(tmp_path)
        write_t3_record(tmp_path, 'listed', _THREAD, cwd=cwd, environment_id=_ENVIRONMENT, pointer=True)
        write_t3_record(tmp_path, 'gone', _GONE_THREAD, cwd=cwd, environment_id=_ENVIRONMENT, pointer=True)
        write_t3_record(
            tmp_path, 'foreign', foreign_thread, cwd=cwd, environment_id='env-elsewhere', pointer=True
        )
        write_t3_record(tmp_path, 'unindexed', unindexed_thread, cwd=cwd, environment_id=_ENVIRONMENT)

        marked = session_reaper.mark_t3_sessions_exited(
            _ENVIRONMENT, frozenset({_THREAD}), root=tmp_path, thread_ended=_ThreadEnded()
        )

        assert marked == ['gone']
        gone = session_records.read_record('gone', root=tmp_path)
        assert (gone.status, gone.exit_code) == (session_records.Status.EXITED, session_reaper.ORPHAN_EXIT_CODE)
        for slug in ('listed', 'foreign', 'unindexed'):
            assert session_records.read_record(slug, root=tmp_path).status == session_records.Status.RUNNING
        assert sorted(path.name for path in session_records.t3_thread_pointers_dir(tmp_path).iterdir()) == (
            sorted([_THREAD, foreign_thread])
        )

    def test_a_record_with_no_environment_is_this_environments(self, tmp_path: Path) -> None:
        write_t3_record(tmp_path, 'legacy', _GONE_THREAD, cwd=str(tmp_path), pointer=True)

        assert session_reaper.mark_t3_sessions_exited(
            _ENVIRONMENT, frozenset(), root=tmp_path, thread_ended=_ThreadEnded()
        ) == ['legacy']

    def test_a_worktree_bound_record_is_marked_whatever_its_cwd(self, tmp_path: Path) -> None:
        worktree = tmp_path / 'proj' / '.worktrees' / '8124'
        write_t3_record(
            tmp_path,
            'worktree-bound',
            _GONE_THREAD,
            cwd=str(worktree),
            environment_id=_ENVIRONMENT,
            pointer=True,
        )

        assert session_reaper.mark_t3_sessions_exited(
            _ENVIRONMENT, frozenset(), root=tmp_path, thread_ended=_ThreadEnded()
        ) == ['worktree-bound']

    def test_a_record_written_after_the_listing_is_skipped_unasked(self, tmp_path: Path) -> None:
        write_t3_record(tmp_path, 'just-launched', _GONE_THREAD, cwd=str(tmp_path), pointer=True)
        listed_at = datetime.now(UTC) - timedelta(minutes=1)
        thread_ended = _ThreadEnded()

        assert (
            session_reaper.mark_t3_sessions_exited(
                _ENVIRONMENT,
                frozenset(),
                root=tmp_path,
                listed_at=listed_at,
                thread_ended=thread_ended,
            )
            == []
        )
        assert thread_ended.calls == []

    def test_an_unlisted_thread_not_confirmed_ended_keeps_its_record(self, tmp_path: Path) -> None:
        write_t3_record(
            tmp_path, 'side-chat', _GONE_THREAD, cwd=str(tmp_path), environment_id=_ENVIRONMENT, pointer=True
        )

        marked = session_reaper.mark_t3_sessions_exited(
            _ENVIRONMENT, frozenset(), root=tmp_path, thread_ended=_ThreadEnded(answer=False)
        )

        assert marked == []
        assert session_records.read_record('side-chat', root=tmp_path).status == session_records.Status.RUNNING
        assert session_records.resolve_session_slug_for_t3_thread(_GONE_THREAD, root=tmp_path) == 'side-chat'

    def test_only_this_environments_unlisted_indexed_threads_are_confirmed(
        self, tmp_path: Path
    ) -> None:
        foreign_thread, unindexed_thread = (
            '4e6a8c0d-2f1b-4c5e-8a7d-9b1d3f5a7c92',
            '6a8c0e2f-4b3d-4e7a-8c9f-1d3f5b7d9e14',
        )
        cwd = str(tmp_path)
        write_t3_record(tmp_path, 'listed', _THREAD, cwd=cwd, environment_id=_ENVIRONMENT, pointer=True)
        write_t3_record(tmp_path, 'gone', _GONE_THREAD, cwd=cwd, environment_id=_ENVIRONMENT, pointer=True)
        write_t3_record(
            tmp_path, 'foreign', foreign_thread, cwd=cwd, environment_id='env-elsewhere', pointer=True
        )
        write_t3_record(tmp_path, 'unindexed', unindexed_thread, cwd=cwd, environment_id=_ENVIRONMENT)
        thread_ended = _ThreadEnded()

        session_reaper.mark_t3_sessions_exited(
            _ENVIRONMENT, frozenset({_THREAD}), root=tmp_path, thread_ended=thread_ended
        )

        assert thread_ended.calls == [_GONE_THREAD]

    def test_one_info_line_reports_the_index_entries_scanned(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        for slug, thread in (
            ('listed', _THREAD),
            ('also-listed', _OTHER_THREAD),
            ('gone', _GONE_THREAD),
        ):
            write_t3_record(tmp_path, slug, thread, cwd=str(tmp_path), pointer=True)

        with caplog.at_level(logging.INFO, logger='orchestrator.session_reaper'):
            session_reaper.mark_t3_sessions_exited(
                _ENVIRONMENT,
                frozenset({_THREAD, _OTHER_THREAD}),
                root=tmp_path,
                thread_ended=_ThreadEnded(),
            )

        [line] = [
            record.getMessage().replace(str(tmp_path), '<root>')
            for record in caplog.records
            if record.name == 'orchestrator.session_reaper'
            and record.levelno == logging.INFO
            and 'mark_t3_sessions_exited' in record.getMessage()
        ]
        assert re.search(r'\b3\b', line), line

    @pytest.mark.parametrize(
        ('routine', 'ended', 'level'),
        [
            pytest.param(False, False, logging.INFO, id='launch-sweep-marking-nothing'),
            pytest.param(True, True, logging.INFO, id='routine-sweep-marking-one'),
            pytest.param(True, False, logging.DEBUG, id='routine-sweep-marking-nothing'),
        ],
    )
    def test_only_a_routine_sweep_that_marks_nothing_logs_its_line_at_debug(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        routine: bool,
        ended: bool,
        level: int,
    ) -> None:
        write_t3_record(tmp_path, 'gone', _GONE_THREAD, cwd=str(tmp_path), pointer=True)

        with caplog.at_level(logging.DEBUG, logger='orchestrator.session_reaper'):
            session_reaper.mark_t3_sessions_exited(
                _ENVIRONMENT,
                frozenset(),
                root=tmp_path,
                thread_ended=_ThreadEnded(answer=ended),
                routine=routine,
            )

        assert [
            record.levelno
            for record in caplog.records
            if record.name == 'orchestrator.session_reaper'
            and record.getMessage().startswith('mark_t3_sessions_exited: marked')
        ] == [level]


class TestT3RecordsAreExemptFromPidSweeps:
    def _stale_t3_record(self, tmp_path: Path, status: session_records.Status) -> Path:
        write_t3_record(tmp_path, 't3-quiet', 'th-1', cwd=str(tmp_path), status=status)
        record = session_records.read_record('t3-quiet', root=tmp_path)
        record.launcher_pid = _DEAD_PID
        session_records.write_record(record, root=tmp_path)
        path = session_records.record_path_for_slug('t3-quiet', root=tmp_path)
        set_mtime(path, _NOW, session_reaper.TERMINAL_TTL + timedelta(hours=1))
        return path

    def test_mark_orphaned_skips_a_quiet_t3_record_with_a_dead_launcher(
        self, tmp_path: Path
    ) -> None:
        self._stale_t3_record(tmp_path, session_records.Status.AWAITING_INPUT)

        assert session_reaper.mark_orphaned_sessions_exited(root=tmp_path, now=_NOW) == []
        assert session_records.read_record('t3-quiet', root=tmp_path).status == session_records.Status.AWAITING_INPUT

    def test_reap_keeps_a_non_terminal_t3_record_with_a_dead_launcher(self, tmp_path: Path) -> None:
        path = self._stale_t3_record(tmp_path, session_records.Status.RUNNING)

        assert session_reaper.reap_stale_records(root=tmp_path, now=_NOW) == []
        assert path.is_file()
        assert not session_records.t3_thread_pointers_dir(tmp_path).exists()

    def test_the_reap_indexes_a_t3_record_whose_thread_no_pointer_holds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._stale_t3_record(tmp_path, session_records.Status.RUNNING)
        monkeypatch.setenv('PATH', str(tmp_path / 'no-bin'))

        assert session_reaper.run_reap(tmp_path) is not None

        assert session_records.resolve_session_slug_for_t3_thread('th-1', root=tmp_path) == 't3-quiet'
        assert session_records.read_record('t3-quiet', root=tmp_path).status == session_records.Status.RUNNING

    def test_indexing_never_moves_a_held_pointer_or_indexes_a_terminal_record(
        self, tmp_path: Path
    ) -> None:
        self._stale_t3_record(tmp_path, session_records.Status.RUNNING)
        write_t3_record(tmp_path, 'holder', 'th-1', cwd=str(tmp_path), pointer=True)
        write_t3_record(tmp_path, 'ended', 'th-2', cwd=str(tmp_path), status=session_records.Status.EXITED)

        assert session_reaper.index_unpointed_t3_records(root=tmp_path) == []

        assert session_records.resolve_session_slug_for_t3_thread('th-1', root=tmp_path) == 'holder'
        assert not session_records.t3_thread_pointer_path('th-2', tmp_path).exists()

    def test_a_t3_record_displaced_from_its_thread_pointer_is_pid_swept(
        self, tmp_path: Path
    ) -> None:
        self._stale_t3_record(tmp_path, session_records.Status.RUNNING)
        write_t3_record(tmp_path, 'holder', 'th-1', cwd=str(tmp_path), pointer=True)

        reaped = session_reaper.reap_stale_records(root=tmp_path, now=_NOW)

        assert [(r.session_slug, r.reason) for r in reaped] == [('t3-quiet', 'stale_pid')]
        assert session_records.resolve_session_slug_for_t3_thread('th-1', root=tmp_path) == 'holder'

    def test_mark_orphaned_marks_a_displaced_t3_record_with_a_dead_launcher(
        self, tmp_path: Path
    ) -> None:
        self._stale_t3_record(tmp_path, session_records.Status.AWAITING_INPUT)
        write_t3_record(tmp_path, 'holder', 'th-1', cwd=str(tmp_path), pointer=True)

        marked = session_reaper.mark_orphaned_sessions_exited(root=tmp_path, now=_NOW)

        assert [r.session_slug for r in marked] == ['t3-quiet']

    def test_a_failed_index_write_leaves_the_t3_record_exempt(self, tmp_path: Path) -> None:
        path = self._stale_t3_record(tmp_path, session_records.Status.RUNNING)
        session_records.t3_thread_pointers_dir(tmp_path).parent.mkdir(parents=True, exist_ok=True)
        session_records.t3_thread_pointers_dir(tmp_path).write_text('not a dir')

        assert session_reaper.index_unpointed_t3_records(root=tmp_path) == []
        assert session_reaper.reap_stale_records(root=tmp_path, now=_NOW) == []
        assert session_reaper.mark_orphaned_sessions_exited(root=tmp_path, now=_NOW) == []
        assert path.is_file()
        assert session_records.read_record('t3-quiet', root=tmp_path).status == session_records.Status.RUNNING

    def test_reap_still_applies_terminal_ttl_to_a_t3_record(self, tmp_path: Path) -> None:
        self._stale_t3_record(tmp_path, session_records.Status.EXITED)

        reaped = session_reaper.reap_stale_records(root=tmp_path, now=_NOW)

        assert [(r.session_slug, r.reason) for r in reaped] == [('t3-quiet', 'terminal_ttl')]


def test_mark_session_folded_stamps_the_fold_code_once(tmp_path: Path) -> None:
    session_records.write_record(make_record(session_slug='hand', status=session_records.Status.RUNNING), root=tmp_path)

    folded = session_reaper.mark_session_folded('hand', root=tmp_path)

    assert folded is not None
    assert (folded.status, folded.exit_code) == (session_records.Status.EXITED, session_reaper.FOLDED_EXIT_CODE)
    assert session_reaper.mark_session_folded('hand', root=tmp_path) is None


def test_folding_a_record_drops_the_pointer_naming_it(tmp_path: Path) -> None:
    write_t3_record(tmp_path, 'hand', _THREAD, cwd=str(tmp_path))
    session_records.write_t3_thread_pointer(_THREAD, 'hand', root=tmp_path)

    assert session_reaper.mark_session_folded('hand', root=tmp_path) is not None

    assert not session_records.t3_thread_pointer_path(_THREAD, tmp_path).exists()
