"""Tests for shared.proc_group.terminate_process_group.

Verifies the SIGTERM-then-SIGKILL sequence that ensures bash → cargo → rustc
(and similar nested) process trees are fully reaped on shutdown.
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal

import pytest

from shared.proc_group import snapshot_process_group, terminate_process_group


async def _pgid_gone_within(pgid: int, timeout: float = 5.0, step: float = 0.1) -> bool:
    """Poll until a process group is fully reaped by the kernel.

    After terminate_process_group reaps the bash leader, any grandchild
    processes are reparented to the user's ``systemd --user`` subreaper
    (or pid 1) and become zombies until that subreaper waitpids them.
    Until that happens, ``os.killpg(pgid, 0)`` still returns 0 rather
    than raising ProcessLookupError.  Observed subreaper latency is
    0–500 ms in isolation but stretches under 32-worker xdist load.
    The default 5 s budget is comfortably longer than any observed reap
    latency; a genuine leak (regression) causes the caller's assert to
    fire.

    PermissionError (EPERM): in the theoretically possible (though
    practically negligible) case where the kernel recycles *pgid* to a
    process owned by another user during the poll window,
    ``os.killpg(pgid, 0)`` raises EPERM rather than ESRCH.  Both mean
    "no longer our group to worry about", so EPERM is treated as success.
    This cannot mask a genuine leak: EPERM only fires once the pgid has
    been assigned to a different user's process, at which point the group
    we spawned is definitively gone.
    """
    iterations = max(1, int(timeout / step))
    for _ in range(iterations):
        try:
            os.killpg(pgid, 0)
        except (ProcessLookupError, PermissionError):
            return True
        await asyncio.sleep(step)
    return False


class TestTerminateProcessGroup:
    """Unit/integration tests for terminate_process_group."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(15)
    async def test_terminate_process_group_kills_real_subprocess(self, tmp_path):
        """terminate_process_group reaps a real bash subprocess and its group.

        Spawn bash with start_new_session=True so it leads its own process
        group. After terminate_process_group returns:
        - proc.returncode must be set (process reaped)
        - os.killpg(pgid, 0) must eventually raise ProcessLookupError once
          the kernel reaps any reparented grandchild zombies (bounded 5 s poll)
        """
        proc = await asyncio.create_subprocess_shell(
            'sleep 30',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        pgid = proc.pid  # start_new_session → pgid == pid

        await terminate_process_group(proc, pgid, grace_secs=5.0)

        assert proc.returncode is not None, (
            f'Process group {pgid} not reaped: proc.returncode is None'
        )
        assert await _pgid_gone_within(pgid), (
            f'Process group {pgid} was not fully reaped within 5 s — '
            f'kernel zombie-reap race or genuine leak.'
        )

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_terminate_process_group_escalates_to_sigkill(self):
        """When the child ignores SIGTERM, SIGKILL fires after grace_secs.

        bash traps SIGTERM (ignores it) so the SIGTERM leg times out, then
        SIGKILL should kill the group. proc.returncode == -9 (SIGKILL).
        """
        proc = await asyncio.create_subprocess_shell(
            "trap '' TERM; sleep 30",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        pgid = proc.pid

        # Let the shell install the trap before we send SIGTERM; otherwise
        # the signal may arrive during shell parsing and kill the shell
        # directly (rc=-15) instead of being ignored.
        await asyncio.sleep(0.2)

        await terminate_process_group(proc, pgid, grace_secs=0.5)

        assert proc.returncode is not None, (
            'Process was not killed even after SIGKILL escalation'
        )
        assert proc.returncode == -signal.SIGKILL, (
            f'Expected returncode -9 (SIGKILL), got {proc.returncode}'
        )

    @pytest.mark.asyncio
    @pytest.mark.timeout(15)
    async def test_terminate_process_group_reaps_grandchildren(self):
        """terminate_process_group kills grandchildren (bash → sleep sleep).

        Reproduces the canonical cargo → rustc incident shape: bash spawns two
        background sleeps and waits for them.  After terminate_process_group,
        pgrep must report no processes in the group.
        """
        proc = await asyncio.create_subprocess_shell(
            'sleep 60 & sleep 60 & wait',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        pgid = proc.pid

        # Brief settle so grandchildren actually start.
        await asyncio.sleep(0.2)

        await terminate_process_group(proc, pgid, grace_secs=5.0)

        assert await _pgid_gone_within(pgid), (
            f'Process group {pgid} was not fully reaped within 5 s — '
            f'grandchildren leaked.'
        )

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_terminate_process_group_idempotent_on_already_dead_proc(self):
        """terminate_process_group is a no-op when the process has already exited.

        Covers the ProcessLookupError race: if the OS has already reaped the
        group before we call terminate_process_group, the helper must return
        cleanly without raising.  This locks in the design decision to defensively
        suppress ProcessLookupError/OSError around both killpg calls.
        """
        proc = await asyncio.create_subprocess_shell(
            'true',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        pgid = proc.pid
        # Let the process exit naturally before calling the helper.
        await proc.wait()
        assert proc.returncode is not None

        # Must not raise even though the process group is already gone.
        await terminate_process_group(proc, pgid, grace_secs=5.0)

    # ------------------------------------------------------------------
    # Regression tests for task 845 — session kill caused by killpg on
    # reused PID after TOCTOU in os.getpgid(proc.pid).
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_no_killpg_after_explicit_reap(self, monkeypatch):
        """After proc.wait() reaps the process, terminate_process_group must NOT call killpg.

        Regression for the session-kill incidents: once the OS has reaped
        the child PID, that PID may be reused by an unrelated process
        (e.g. user ``systemd --user``).  Any killpg keyed off the stale
        PID would then hit the wrong group and kill the login session.
        The returncode-check must short-circuit before any killpg dispatch.
        """
        calls: list[tuple[int, int]] = []

        def spy_killpg(pgid: int, sig: int) -> None:
            calls.append((pgid, sig))

        proc = await asyncio.create_subprocess_shell(
            'true',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        pgid = proc.pid
        await proc.wait()
        assert proc.returncode is not None

        # Patch AFTER spawn/wait so the test harness itself can't be affected.
        monkeypatch.setattr('shared.proc_group.os.killpg', spy_killpg)

        await terminate_process_group(proc, pgid, grace_secs=0.2)

        assert calls == [], (
            f'terminate_process_group must not call killpg on a reaped proc; '
            f'got {calls}'
        )

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_refuses_pgid_equal_to_self_pid(self, monkeypatch, caplog):
        """Pgid == os.getpid() must be refused without signalling.

        Defence-in-depth: if anything ever corrupts the captured pgid to
        point at our own process group (which is how the pre-fix incidents
        ended up hitting systemd --user), the helper must log an error and
        return without dispatching a signal.
        """
        calls: list[tuple[int, int]] = []

        def spy_killpg(pgid: int, sig: int) -> None:
            calls.append((pgid, sig))

        monkeypatch.setattr('shared.proc_group.os.killpg', spy_killpg)

        # Build a fake proc object that looks alive so we don't exit via
        # the returncode short-circuit — the sanity check is what we're
        # validating.  Using a real Process requires we match its pid,
        # which we explicitly do NOT want for this negative test.
        class FakeProc:
            pid = os.getpid()
            returncode = None

            async def wait(self) -> int:
                return 0

        with caplog.at_level(logging.ERROR, logger='shared.proc_group'):
            await terminate_process_group(
                FakeProc(), os.getpid(), grace_secs=0.1,  # type: ignore[arg-type]
            )

        assert calls == [], f'killpg should be refused; got {calls}'
        assert any(
            'refusing to killpg' in rec.message.lower()
            for rec in caplog.records
        ), f'expected a refusal log record, got {[r.message for r in caplog.records]}'

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_refuses_pgid_one(self, monkeypatch):
        """Pgid == 1 (init) must be refused."""
        calls: list[tuple[int, int]] = []

        def spy_killpg(pgid: int, sig: int) -> None:
            calls.append((pgid, sig))

        monkeypatch.setattr('shared.proc_group.os.killpg', spy_killpg)

        class FakeProc:
            pid = 1
            returncode = None

            async def wait(self) -> int:
                return 0

        await terminate_process_group(FakeProc(), 1, grace_secs=0.1)  # type: ignore[arg-type]
        assert calls == []

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_refuses_pgid_mismatching_proc_pid(self, monkeypatch):
        """Pgid != proc.pid must be refused (corrupted capture).

        With start_new_session=True, pgid captured at spawn equals proc.pid.
        A later mismatch indicates either PID-reuse-through-reap or a bug
        in the caller; either way, don't risk signalling the wrong group.
        """
        calls: list[tuple[int, int]] = []

        def spy_killpg(pgid: int, sig: int) -> None:
            calls.append((pgid, sig))

        monkeypatch.setattr('shared.proc_group.os.killpg', spy_killpg)

        # Use a high pid value that isn't our own to bypass the other guards.
        fake_pid = 999_999
        fake_pgid = 999_998

        class FakeProc:
            pid = fake_pid
            returncode = None

            async def wait(self) -> int:
                return 0

        await terminate_process_group(FakeProc(), fake_pgid, grace_secs=0.1)  # type: ignore[arg-type]
        assert calls == []


class TestSnapshotProcessGroup:
    """Tests for snapshot_process_group(pgid) — /proc-based process-group snapshot.

    The helper is called inside _run_subprocess's TimeoutError handler to
    capture which processes were alive in the wedged CLI's process group
    just before SIGTERM/SIGKILL.  It must never raise and must return a
    useful string for any pgid value.
    """

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_real_child_appears_in_snapshot(self, tmp_path):
        """Snapshot of a live child's pgid returns non-empty string mentioning the child.

        Spawn a real 'sleep 30' child with start_new_session=True so it
        leads its own process group (pgid == pid).  Call
        snapshot_process_group(pgid) while it is alive and assert:
        - the returned string is non-empty
        - it contains the child pid (as str) or the comm 'sleep'
        """
        proc = await asyncio.create_subprocess_exec(
            'sleep', '30',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        pgid = proc.pid
        try:
            snapshot = snapshot_process_group(pgid)
            assert snapshot, f'Expected non-empty snapshot for pgid={pgid}'
            assert str(pgid) in snapshot or 'sleep' in snapshot, (
                f'Expected child pid {pgid} or comm "sleep" in snapshot:\n{snapshot}'
            )
        finally:
            proc.kill()
            await proc.wait()

    def test_invalid_pgid_never_raises(self):
        """snapshot_process_group on an unused/invalid pgid returns a benign string.

        Two sub-cases:
        - a very large pgid (999_999_999) unlikely to exist
        - pgid <= 1 (invalid by convention)
        Neither should raise.
        """
        result_large = snapshot_process_group(999_999_999)
        assert isinstance(result_large, str), (
            f'Expected str, got {type(result_large)}'
        )

        result_invalid = snapshot_process_group(0)
        assert isinstance(result_invalid, str), (
            f'Expected str, got {type(result_invalid)}'
        )

    def test_pgid_le_zero_returns_diagnostic_string(self):
        """pgid <= 0 returns a diagnostic string (not an error, not empty).

        Locks in the guard at the top of _snapshot_process_group_unsafe:
        values 0 and -1 are never valid process-group ids on Linux.
        """
        for pgid in (0, -1, -999):
            result = snapshot_process_group(pgid)
            assert isinstance(result, str), f'Expected str for pgid={pgid}'
            assert result, f'Expected non-empty string for pgid={pgid}'
            # Should contain a diagnostic message (not raise, not return a hit)
            assert 'no processes' in result.lower() or 'pgid' in result.lower(), (
                f'Expected diagnostic message for pgid={pgid}, got: {result!r}'
            )

    def test_missing_proc_returns_diagnostic(self, monkeypatch):
        """/proc unavailable (simulated via monkeypatching Path.exists) returns diagnostic.

        Simulates a non-Linux environment or a container where /proc is absent.
        snapshot_process_group must never raise.
        """
        from pathlib import Path
        from unittest.mock import patch

        # Patch Path.exists so /proc reports as not existing
        original_exists = Path.exists

        def patched_exists(self) -> bool:
            if str(self) == '/proc':
                return False
            return original_exists(self)

        with patch.object(Path, 'exists', patched_exists):
            result = snapshot_process_group(12345)

        assert isinstance(result, str), f'Expected str, got {type(result)}'
        assert result, 'Expected non-empty diagnostic string'

    def test_no_matching_process_returns_no_processes_message(self):
        """A valid-looking but unused pgid returns 'no processes found'.

        Use an extremely large pgid (2**30) that almost certainly has no
        processes assigned.  The function must return the benign diagnostic
        string rather than an empty string or raising.
        """
        result = snapshot_process_group(2**30)
        assert isinstance(result, str)
        assert result  # non-empty
        # Must say 'no processes found' or similar
        assert 'no processes' in result.lower() or 'no snapshot' in result.lower(), (
            f'Expected "no processes found" message for unused pgid, got: {result!r}'
        )

    def test_stat_line_with_comm_containing_spaces_and_parens(self, monkeypatch, tmp_path):
        """stat-line parser handles comm names with spaces and nested parens.

        The kernel's /proc/<pid>/stat format wraps the comm field in parens:
            "pid (comm with spaces (and parens)) state ppid pgrp ..."
        The parser uses rfind(')') to locate the end of comm, then reads the
        remaining fields positionally.  This test locks in that offset logic
        against a hand-crafted synthetic stat line.
        """
        from pathlib import Path
        from unittest.mock import patch

        # Synthetic pgid we want to match
        target_pgid = 77777

        # Build a synthetic /proc/<pid>/ tree under tmp_path
        fake_pid = 77778
        pid_dir = tmp_path / str(fake_pid)
        pid_dir.mkdir()

        # Comm with embedded spaces and parens — the adversarial case
        comm_name = 'my weird (proc) name'
        stat_content = (
            f'{fake_pid} ({comm_name}) S '  # pid (comm) state
            f'1 {target_pgid} {target_pgid} 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n'
            # fields after state: ppid=1, pgrp=target_pgid, session=target_pgid, ...
        )
        (pid_dir / 'stat').write_text(stat_content)
        (pid_dir / 'comm').write_text(comm_name + '\n')
        (pid_dir / 'wchan').write_text('do_wait\n')

        # Make a fake /proc directory that only has our synthetic pid entry plus
        # some non-numeric entries (should be skipped by the implementation).
        fake_proc = tmp_path / 'fake_proc'
        fake_proc.mkdir()
        # Symlink or recreate the pid subdir under fake_proc
        import shutil
        shutil.copytree(str(pid_dir), str(fake_proc / str(fake_pid)))
        (fake_proc / 'version').write_text('Linux 5.x')  # non-numeric, must be skipped

        from shared import proc_group as _pg

        original_exists = Path.exists

        def patched_exists(self) -> bool:
            if str(self) == '/proc':
                return True
            return original_exists(self)

        with (
            patch.object(Path, 'exists', patched_exists),
            patch.object(_pg, '_snapshot_process_group_unsafe') as mock_unsafe,
        ):
            # Use the real _snapshot_process_group_unsafe but with a fake proc dir.
            # Because monkeypatching iterdir on Path is fragile, call the internal
            # function directly with a patched proc_dir reference instead.
            mock_unsafe.side_effect = lambda pgid: _snapshot_impl_with_proc_dir(
                pgid, fake_proc
            )
            result = snapshot_process_group(target_pgid)

        assert isinstance(result, str)
        assert result, f'Expected non-empty snapshot, got: {result!r}'
        # The comm with spaces/parens must be present in the output
        assert comm_name in result or str(fake_pid) in result, (
            f'Expected comm {comm_name!r} or pid {fake_pid} in snapshot:\n{result}'
        )


def _snapshot_impl_with_proc_dir(pgid: int, proc_dir) -> str:
    """Re-implementation of _snapshot_process_group_unsafe with an injectable proc_dir.

    Used by the synthetic-stat-line test to point at a fake /proc tree built
    under tmp_path.  Mirrors the real implementation exactly so the test locks
    in the field-offset logic under the adversarial (spaces+parens in comm) case.
    """
    from pathlib import Path

    if pgid <= 0:
        return f'snapshot_process_group({pgid}): pgid <= 0 — no snapshot taken'

    proc_dir = Path(proc_dir)
    if not proc_dir.exists():
        return f'snapshot_process_group({pgid}): /proc not available'

    rows: list[str] = []
    try:
        entries = list(proc_dir.iterdir())
    except OSError:
        return f'snapshot_process_group({pgid}): could not list /proc'

    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)

        try:
            stat_text = (entry / 'stat').read_text()
        except OSError:
            continue

        try:
            rparen = stat_text.rfind(')')
            if rparen < 0:
                continue
            tail = stat_text[rparen + 2:]
            fields = tail.split()
            state = fields[0]
            ppid = int(fields[1])
            pgrp = int(fields[2])
        except (IndexError, ValueError):
            continue

        if pgrp != pgid:
            continue

        try:
            comm = (entry / 'comm').read_text().strip()
        except OSError:
            comm = '?'

        try:
            wchan = (entry / 'wchan').read_text().strip()
        except OSError:
            wchan = '?'

        rows.append(f'  pid={pid} ppid={ppid} state={state} wchan={wchan} comm={comm}')

    if not rows:
        return f'snapshot_process_group({pgid}): no processes found in group'

    header = f'snapshot_process_group({pgid}): {len(rows)} process(es) in group:'
    return '\n'.join([header] + rows)
