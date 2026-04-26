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

from shared.proc_group import terminate_process_group


class TestTerminateProcessGroup:
    """Unit/integration tests for terminate_process_group."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(15)
    async def test_terminate_process_group_kills_real_subprocess(self, tmp_path):
        """terminate_process_group reaps a real bash subprocess and its group.

        Spawn bash with start_new_session=True so it leads its own process
        group. After terminate_process_group returns:
        - proc.returncode must be set (process reaped)
        - os.killpg(pgid, 0) must raise ProcessLookupError (group gone)
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
        with pytest.raises(ProcessLookupError):
            os.killpg(pgid, 0)

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

        # Poll: terminate_process_group's contract is best-effort — it
        # signals the group and waits for the *leader* (bash) to be reaped
        # via proc.wait(), but does NOT explicitly waitpid grandchildren.
        # When bash exits, the OS reparents its background sleeps to the
        # user's systemd --user subreaper (or pid 1), which reaps them on
        # its own schedule — a non-deterministic 0–500 ms window.  5 s is
        # comfortably longer than any observed subreaper latency on Linux;
        # if the group genuinely leaks (regression), the loop falls through
        # and the assert fires.
        #
        # We also catch PermissionError (EPERM): in the theoretically
        # possible (though practically negligible) case where the kernel
        # recycles pgid to a process owned by another user during the poll
        # window, os.killpg(pgid, 0) raises EPERM rather than ESRCH.  Both
        # mean "no longer our group to worry about".
        group_gone = False
        for _ in range(50):  # 50 × 0.1 s = 5 s budget
            try:
                os.killpg(pgid, 0)
            except (ProcessLookupError, PermissionError):
                group_gone = True
                break
            await asyncio.sleep(0.1)
        assert group_gone, (
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
