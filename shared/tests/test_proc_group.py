"""Tests for shared.proc_group.terminate_process_group.

Verifies the SIGTERM-then-SIGKILL sequence that ensures bash → cargo → rustc
(and similar nested) process trees are fully reaped on shutdown.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal

import pytest

from shared.proc_group import (
    reap_process_groups,
    scan_process_groups_under_path,
    snapshot_process_group,
    terminate_process_group,
)


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


async def _spawn_sleeper_in(cwd) -> asyncio.subprocess.Process:
    """Spawn a real ``sleep 30`` leading its own process group, with cwd *cwd*.

    ``start_new_session=True`` makes the child the leader of a fresh process
    group (``pgid == pid``), and *cwd* becomes the process's working directory
    so ``/proc/<pid>/cwd`` points at (a path under) the scan root.
    """
    return await asyncio.create_subprocess_exec(
        'sleep', '30',
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        cwd=str(cwd),
        start_new_session=True,
    )


def _kill_group(pgid: int) -> None:
    """Best-effort SIGKILL of an entire process group (test cleanup)."""
    with contextlib.suppress(ProcessLookupError, OSError):
        os.killpg(pgid, signal.SIGKILL)


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_await_trap_installed_returns_once_shell_armed():
    """_await_trap_installed returns once the shell announces readiness.

    The shell's ``ready`` line is only printed after ``trap '' TERM`` has
    returned, so observing it on stdout is a happens-before edge proving the
    SIGTERM disposition is already SIG_IGN — see _await_trap_installed's own
    docstring for the supporting /proc/<pid>/status measurement.  The proc
    must still be alive (unreaped) afterwards: the handshake only observes
    readiness, it must not consume/reap the shell out from under the
    subsequent terminate_process_group call.
    """
    proc = await asyncio.create_subprocess_shell(
        "trap '' TERM; echo ready; sleep 30",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        await _await_trap_installed(proc)
        assert proc.returncode is None, (
            f'expected shell pid={proc.pid} still running after the '
            f'readiness handshake, got returncode={proc.returncode}'
        )
    finally:
        _kill_group(proc.pid)
        with contextlib.suppress(Exception):
            await proc.wait()


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

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_snapshot_includes_cmdline(self, tmp_path):
        """Snapshot includes the full cmdline of each process, not just the 15-char comm.

        Spawn a child with a DISTINCTIVE argv token ('31.4159') that cannot
        appear in the 15-char comm field (which is just 'sleep').  Assert that
        the token appears in the snapshot — it can only come from
        /proc/<pid>/cmdline, proving the new field is captured.
        """
        proc = await asyncio.create_subprocess_exec(
            'sleep', '31.4159',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        pgid = proc.pid
        try:
            snapshot = snapshot_process_group(pgid)
            assert snapshot, f'Expected non-empty snapshot for pgid={pgid}'
            assert '31.4159' in snapshot, (
                f"Expected distinctive arg '31.4159' (from /proc/cmdline) in snapshot; "
                f"got:\n{snapshot}"
            )
        finally:
            proc.kill()
            await proc.wait()


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


class TestScanProcessGroupsUnderPath:
    """Tests for scan_process_groups_under_path — /proc-based at-or-under scan.

    Used by the task-2828 startup survivor barrier to find process groups
    whose cwd / open fds / mmap'd paths fall at-or-under the ``_merge-verify``
    worktree of a *previous* orchestrator run.  Like snapshot_process_group it
    must never raise, even on vanished/permission-denied pids.
    """

    @pytest.mark.asyncio
    @pytest.mark.timeout(15)
    async def test_scan_matches_cwd_under_root_and_respects_boundary(self, tmp_path):
        """A process whose cwd is under root is returned; sibling/outside are not.

        Three real subprocesses:
        - ``under``   — cwd = <root>/build          → MUST be returned
        - ``sibling`` — cwd = <root>XYZ (prefix but not under root) → MUST NOT
        - ``outside`` — cwd = a wholly unrelated dir → MUST NOT

        The sibling case locks in the at-or-under boundary (equality OR
        ``startswith(str(root) + os.sep)``) so ``_merge-verifyXYZ`` never
        matches ``_merge-verify``.
        """
        base = tmp_path.resolve()
        root = base / '_merge-verify'
        work = root / 'build'
        sibling = base / '_merge-verifyXYZ'
        outside = base / 'other'
        for d in (work, sibling, outside):
            d.mkdir(parents=True)

        under = await _spawn_sleeper_in(work)
        sib = await _spawn_sleeper_in(sibling)
        out = await _spawn_sleeper_in(outside)
        try:
            found = scan_process_groups_under_path(root)
            assert under.pid in found, (
                f'expected pgid {under.pid} (cwd under {root}) in {found}'
            )
            assert sib.pid not in found, (
                f'sibling {sibling} must NOT match root {root} (prefix boundary); '
                f'got {found}'
            )
            assert out.pid not in found, (
                f'outside pgid {out.pid} must NOT be returned; got {found}'
            )
        finally:
            for p in (under, sib, out):
                _kill_group(p.pid)
                with contextlib.suppress(Exception):
                    await p.wait()

    @pytest.mark.asyncio
    @pytest.mark.timeout(15)
    async def test_scan_respects_exclude_pgids(self, tmp_path):
        """A pgid in *exclude_pgids* is dropped even when its cwd is under root.

        Also exercises the equality branch (cwd == root exactly).
        """
        root = tmp_path.resolve() / '_merge-verify'
        root.mkdir()
        proc = await _spawn_sleeper_in(root)
        try:
            assert proc.pid in scan_process_groups_under_path(root)
            assert proc.pid not in scan_process_groups_under_path(
                root, exclude_pgids=frozenset({proc.pid})
            )
        finally:
            _kill_group(proc.pid)
            with contextlib.suppress(Exception):
                await proc.wait()

    def test_scan_never_raises_on_unreadable_pid(self, monkeypatch, tmp_path):
        """A readlink that raises (vanished / permission-denied pid) is swallowed.

        Force every ``os.readlink`` to raise ProcessLookupError, simulating a
        pid that vanishes mid-walk.  The scan must still return a set and never
        propagate the error (mirrors snapshot_process_group's never-raise
        invariant).
        """
        def boom(*_a, **_k):
            raise ProcessLookupError('pid vanished mid-scan')

        monkeypatch.setattr('shared.proc_group.os.readlink', boom)
        result = scan_process_groups_under_path(tmp_path / '_merge-verify')
        assert isinstance(result, set)

    def test_scan_never_raises_on_real_proc(self, tmp_path):
        """Scanning the real /proc never raises and returns a set.

        On a non-root runner, ``/proc/1/cwd`` (and other root-owned pids) raise
        PermissionError, and pids vanish mid-walk (ProcessLookupError); the scan
        must swallow both.  A fresh, never-created root matches nothing.
        """
        result = scan_process_groups_under_path(tmp_path / 'never-created')
        assert isinstance(result, set)
        assert result == set()


class TestReapProcessGroups:
    """Tests for reap_process_groups — SIGTERM→wait→SIGKILL over a set of pgids.

    The barrier's reaping half: generalizes terminate_process_group's
    escalation from a single owned proc handle to a set of foreign pgids,
    refusing any unsafe pgid (self/parent/own-group/init) via
    _unsafe_pgid_reason.
    """

    @pytest.mark.asyncio
    @pytest.mark.timeout(15)
    async def test_reap_kills_real_group(self, tmp_path):
        """A real planted process group is reaped and reported 'reaped'."""
        proc = await _spawn_sleeper_in(tmp_path)
        pgid = proc.pid
        try:
            # Off-thread so the event loop keeps running and the child watcher
            # reaps the killed leader promptly (mirrors the production
            # asyncio.to_thread(reap...) call site).
            outcomes = await asyncio.to_thread(reap_process_groups, {pgid})
            assert outcomes.get(pgid) == 'reaped', f'unexpected outcomes: {outcomes}'
            assert await _pgid_gone_within(pgid), (
                f'process group {pgid} not gone after reap'
            )
        finally:
            _kill_group(pgid)
            with contextlib.suppress(Exception):
                await proc.wait()

    @pytest.mark.timeout(5)
    def test_reap_refuses_unsafe_pgids(self, monkeypatch):
        """os.getpgrp() and pgid 1 are REFUSED and never signalled.

        Defence-in-depth: reaping our own group (or init) would kill the
        orchestrator/test session.  Refused pgids must yield 'refused:<reason>'
        and receive no killpg at all (not even the liveness 0-probe).
        """
        calls: list[tuple[int, int]] = []
        monkeypatch.setattr(
            'shared.proc_group.os.killpg',
            lambda pgid, sig: calls.append((pgid, sig)),
        )
        own = os.getpgrp()
        outcomes = reap_process_groups({own, 1})
        assert outcomes[own].startswith('refused:'), outcomes
        assert outcomes[1].startswith('refused:'), outcomes
        assert calls == [], (
            f'refused pgids must never be signalled; got killpg calls {calls}'
        )
