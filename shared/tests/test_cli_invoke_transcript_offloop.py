"""Transcript reads inside ``_run_subprocess`` must never run on the event loop (task 3925).

THE INVARIANT UNDER TEST
────────────────────────
Every on-disk transcript read reachable from ``_run_subprocess`` —
``count_transcript_turns`` in the startup-regime poll, in the working-regime
progress-extension poll, and in the post-kill ``except TimeoutError:`` re-read,
plus ``read_transcript_records`` on the normal-exit path — must execute on a
worker thread (``asyncio.to_thread``), NOT on the thread running the event loop.

WHY IT MATTERS.  The orchestrator runs every role of every concurrent task on
ONE event loop (``orchestrator/src/orchestrator/cli.py``, ``asyncio.run(_main())``).
Each of these reads globs ``<config_dir>/projects/*/<session_id>.jsonl``, opens
the whole file and ``json.loads`` every line — a 1.0-1.3 MB JSONL for a mature
agent session.  Executed inline on the loop by up to ``max_concurrent_tasks``
agents, that blocking work starves every other coroutine in the process.  The
sibling rationale is stated at ``orchestrator/src/orchestrator/harness.py:7965``
(``run_substrate_recheck``), and the invariant itself is restated in the source at
the top of ``_run_subprocess``'s watchdog poll loop.

HOW IT IS TESTED — thread identity, not wall clock
──────────────────────────────────────────────────
The primary assertion in every class below is categorical: the patched
transcript function records ``threading.get_ident()`` on each call, and the test
asserts none of the recorded idents equals the ident of the thread running the
event loop.  This is load-independent and encodes the property directly.

That choice is deliberate.  ``orchestrator/tests/test_liveness_boundary_gate.py:352-358``
records a documented history of load-flakiness in exactly this code — "6.98s wall
for a kill the watchdog itself measured at 0.1s — correct behaviour reported red
by scheduling noise outside the code under test" — naming seven prior tasks
(1836/1851/2320/2840/2921/2959/3491) burned by wall-clock proxies.  Exactly ONE
supplementary responsiveness test lives here, and its bound is categorical rather
than tuned; see its docstring.

PATCH-TARGET DISCIPLINE.  Every patch below uses the module-global string target
``'shared.cli_invoke.count_transcript_turns'`` / ``'...read_transcript_records'``.
``asyncio.to_thread(count_transcript_turns, ...)`` resolves that global at CALL
time, so the patches keep working.  A ``functools.partial`` or module-scope alias
built at import time in ``cli_invoke.py`` would capture the REAL function and
silently defeat these patches (and the 10 pre-existing ones in
``test_cli_invoke.py``) — the tests would do real filesystem reads against empty
tmp_path dirs instead of failing loudly.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from shared.cli_invoke import _run_subprocess

# ── Fake process builders ───────────────────────────────────────────────────
# Copied in spirit from TestRunSubprocessWatchdog._make_hanging_proc
# (shared/tests/test_cli_invoke.py:3783) so this file stands alone, matching how
# test_cli_invoke_background.py / test_cli_invoke_sandbox_wrap.py each carry
# their own fixtures rather than importing across test modules.


def _make_hanging_proc():
    """Return (proc, call_count) whose communicate() hangs on call 1 and raises
    TimeoutError on call 2 (the post-SIGTERM retry) → SIGKILL branch."""
    call_count = [0]

    async def communicate_side_effect(input=None):  # noqa: A002
        call_count[0] += 1
        if call_count[0] == 1:
            # Hang until the watchdog cancels comm_task — CancelledError lands here.
            await asyncio.Event().wait()
        raise TimeoutError

    proc = MagicMock()
    proc.communicate = communicate_side_effect
    proc.terminate = MagicMock()
    proc.kill = MagicMock()
    proc.wait = AsyncMock()
    proc.returncode = None
    proc.pid = 12345
    return proc, call_count


def _make_delayed_success_proc(delay_secs, stdout_bytes=b'{"type":"result","subtype":"success"}'):
    """Return a proc whose communicate() sleeps *delay_secs* then succeeds once.

    Mirrors ``TestRunSubprocessWorkingRegimeProgressExtension._make_delayed_success_proc``
    (test_cli_invoke.py:4056).  Used where a test must observe the watchdog POLL
    reads in isolation: the run leaves the loop via ``comm_task in done`` and
    never enters the ``except TimeoutError:`` handler, so the handler's own
    one-shot re-read cannot land in the recorded-ident list.
    """

    async def communicate_side_effect(input=None):  # noqa: A002
        await asyncio.sleep(delay_secs)
        return (stdout_bytes, b'')

    proc = MagicMock()
    proc.communicate = communicate_side_effect
    proc.terminate = MagicMock()
    proc.kill = MagicMock()
    proc.wait = AsyncMock()
    proc.returncode = 0
    proc.pid = 12345
    return proc


def _fake_exec_returning(proc):
    """An ``asyncio.create_subprocess_exec`` stand-in that yields *proc*."""

    async def fake_exec(*args, **kwargs):
        return proc

    return fake_exec


def _ident_recorder_growing(recorded: list[int]):
    """A sync ``count_transcript_turns`` stand-in appending the calling thread's
    ident and returning a MONOTONICALLY GROWING count (1, 2, 3, ...).

    Mirrors ``TestRunSubprocessWorkingRegimeProgressExtension._always_growing_turns``
    (test_cli_invoke.py:4094).  Growing counts latch ``seen_turn`` on call 1 and
    keep refreshing ``last_progress_monotonic`` on every later call, so every
    call after the first takes the ``elif extension_engaged`` branch and the run
    is never idle-killed.
    """
    counter = [0]

    def _side_effect(config_dir, session_id):
        recorded.append(threading.get_ident())
        counter[0] += 1
        return counter[0]

    return _side_effect


def _ident_recorder(recorded: list[int], return_value):
    """A sync ``count_transcript_turns`` stand-in appending the CALLING thread's
    ident to *recorded* and returning *return_value*.

    Same closure shape as the existing side-effect idiom at
    ``test_cli_invoke.py:4094`` (``_always_growing_turns``).  Under
    ``asyncio.to_thread`` these list mutations happen on a worker thread; that
    stays correct because at most one read is in flight per ``_run_subprocess``
    and the ``await`` establishes happens-before on both sides.  A future test
    firing CONCURRENT reads would need real synchronisation.
    """

    def _side_effect(config_dir, session_id):
        recorded.append(threading.get_ident())
        return return_value

    return _side_effect


class TestStartupRegimePollOffLoop:
    """The startup-regime watchdog poll (``cli_invoke.py:2842``, inside
    ``if not seen_turn and config_dir and session_id:``) must read off the loop."""

    async def test_startup_poll_reads_transcript_off_the_loop_thread(self, tmp_path):
        """Every startup-regime ``count_transcript_turns`` call runs on a worker thread.

        The fake returns None, so ``seen_turn`` never latches and every poll
        takes the ``if not seen_turn`` branch under test; None also means the
        startup-wedge kill can never fire (B7 conservative degrade), so the run
        polls at the patched 0.05s cadence for the process's whole 0.3s life.

        The process COMPLETES rather than hanging, deliberately.  The run then
        leaves the watchdog loop via ``comm_task in done`` and never enters the
        ``except TimeoutError:`` handler, whose own one-shot re-read
        (cli_invoke.py:2970) is a DIFFERENT site owned by step-5/step-6 — with a
        hanging proc that handler's still-on-loop read lands in ``recorded`` and
        this test would be asserting two sites at once.  The normal-exit path
        reads via ``read_transcript_records``, a different module global that is
        not patched here, so it cannot pollute ``recorded`` either.
        """
        loop_ident = threading.get_ident()
        sid = str(uuid.uuid4())
        cfg_dir = tmp_path / 'cfg'
        cfg_dir.mkdir()

        proc = _make_delayed_success_proc(0.3)
        recorded: list[int] = []

        with (
            patch(
                'shared.cli_invoke.asyncio.create_subprocess_exec',
                side_effect=_fake_exec_returning(proc),
            ),
            patch('shared.cli_invoke.terminate_process_group', AsyncMock()),
            patch(
                'shared.cli_invoke.count_transcript_turns',
                side_effect=_ident_recorder(recorded, None),
            ),
            patch('shared.cli_invoke._WATCHDOG_POLL_SECS', 0.05),
        ):
            result = await _run_subprocess(
                ['fake'], cwd=tmp_path, env={}, model='opus',
                timeout_seconds=5.0, startup_grace_secs=0.05,
                session_id=sid, config_dir=cfg_dir,
            )

        assert result.timed_out is False, 'Expected a normal exit, not a kill'
        assert recorded, 'Expected the startup-regime poll to read the transcript at least once'
        assert loop_ident not in recorded, (
            f'count_transcript_turns ran on the event-loop thread ({loop_ident}) — '
            f'the startup-regime poll at cli_invoke.py:2842 blocks the shared loop. '
            f'Recorded idents: {sorted(set(recorded))}'
        )

    async def test_startup_poll_does_not_block_the_event_loop(self, tmp_path):
        """Supplementary responsiveness check: the loop keeps running coroutines
        while a transcript read is in flight.

        A ticker coroutine increments a counter every ``await asyncio.sleep(0.005)``.
        The fake ``count_transcript_turns`` snapshots that counter, does a single
        blocking ``time.sleep(0.5)`` on its FIRST call only, snapshots again, and
        records the delta.

        WHY ``>= 1`` IS THE RIGHT BOUND — DO NOT "TIGHTEN" IT.  This is a
        categorical on-loop/off-loop discriminator, not a tuned wall-clock
        threshold.  If the read runs ON the loop, the ticker provably gets
        EXACTLY ZERO ticks during the 0.5s sleep — the loop cannot schedule
        anything.  If it runs off-loop, ~100 ticks are expected at a 5ms cadence
        over 0.5s, so ``>= 1`` carries roughly 50x headroom and is immune to the
        scheduling noise that made the wall-clock proxies at
        ``test_liveness_boundary_gate.py:407-415`` load-fragile.  Raising this
        bound would trade that structural guarantee for a load-sensitive one.
        """
        sid = str(uuid.uuid4())
        cfg_dir = tmp_path / 'cfg'
        cfg_dir.mkdir()

        proc, _ = _make_hanging_proc()
        ticks = [0]
        deltas: list[int] = []
        slept = [False]

        async def ticker():
            while True:
                await asyncio.sleep(0.005)
                ticks[0] += 1

        def blocking_read(config_dir, session_id):
            if slept[0]:
                return None
            slept[0] = True
            before = ticks[0]
            time.sleep(0.5)
            deltas.append(ticks[0] - before)
            return None

        ticker_task = asyncio.ensure_future(ticker())
        try:
            with (
                patch(
                    'shared.cli_invoke.asyncio.create_subprocess_exec',
                    side_effect=_fake_exec_returning(proc),
                ),
                patch('shared.cli_invoke.terminate_process_group', AsyncMock()),
                patch('shared.cli_invoke.count_transcript_turns', side_effect=blocking_read),
                patch('shared.cli_invoke._WATCHDOG_POLL_SECS', 0.05),
            ):
                await _run_subprocess(
                    ['fake'], cwd=tmp_path, env={}, model='opus',
                    timeout_seconds=1.0, startup_grace_secs=0.05,
                    session_id=sid, config_dir=cfg_dir,
                )
        finally:
            ticker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await ticker_task

        assert deltas, 'Expected the blocking transcript read to have run at least once'
        assert deltas[0] >= 1, (
            f'The event loop made ZERO progress during a 0.5s transcript read '
            f'(ticker delta={deltas[0]}) — the read is executing inline on the loop. '
            f'Off-loop, ~100 ticks are expected at a 5ms cadence.'
        )


class TestWorkingRegimeExtensionPollOffLoop:
    """The working-regime progress-extension poll (``cli_invoke.py:2850``, inside
    ``elif extension_engaged and config_dir and session_id:``) must read off the loop.

    THIS IS THE HIGHEST-VALUE SITE IN PRODUCTION.
    ``orchestrator/src/orchestrator/workflow.py`` passes BOTH ``working_idle_secs``
    and ``absolute_cap_secs`` for every role, so ``extension_engaged`` latches for
    every agent and this branch fires every ``_WATCHDOG_WORKING_POLL_SECS`` (60s)
    for the entire multi-minute-to-multi-hour working lifetime of every concurrent
    agent — unlike the startup-regime branch, which stops firing the moment
    ``seen_turn`` latches.
    """

    async def test_extension_poll_reads_transcript_off_the_loop_thread(self, tmp_path):
        """Every progress-extension ``count_transcript_turns`` call runs on a worker thread.

        Turn counts grow monotonically, so ``seen_turn`` latches on call 1 and
        every later call takes the ``elif extension_engaged`` branch under test.
        Call index 0 is therefore the pre-latch ``if not seen_turn`` branch
        (already covered by TestStartupRegimePollOffLoop); indices >= 1 are the
        branch this test exists for, which is why ``len(recorded) >= 2`` is
        asserted rather than merely ``recorded``.

        BOTH ``working_idle_secs`` and ``absolute_cap_secs`` are passed — the live
        orchestrator configuration, and the precondition for ``extension_engaged``.
        That the run survives well past ``timeout_seconds=0.1`` and exits normally
        is itself proof the extension regime engaged: without it, the flat 0.1s
        ceiling would have killed the 0.3s process.

        The process COMPLETES rather than hanging, for the same reason as the
        startup-poll test: a cap/idle kill would route through the
        ``except TimeoutError:`` handler whose own re-read (cli_invoke.py:2970) is
        step-5/step-6's site, and its ident would land in ``recorded`` and make
        this test assert two sites at once.
        """
        loop_ident = threading.get_ident()
        sid = str(uuid.uuid4())
        cfg_dir = tmp_path / 'cfg'
        cfg_dir.mkdir()

        proc = _make_delayed_success_proc(0.3)
        recorded: list[int] = []

        with (
            patch(
                'shared.cli_invoke.asyncio.create_subprocess_exec',
                side_effect=_fake_exec_returning(proc),
            ),
            patch('shared.cli_invoke.terminate_process_group', AsyncMock()),
            patch(
                'shared.cli_invoke.count_transcript_turns',
                side_effect=_ident_recorder_growing(recorded),
            ),
            patch('shared.cli_invoke._WATCHDOG_POLL_SECS', 0.02),
            patch('shared.cli_invoke._WATCHDOG_WORKING_POLL_SECS', 0.02),
        ):
            result = await _run_subprocess(
                ['fake'], cwd=tmp_path, env={}, model='opus',
                timeout_seconds=0.1, startup_grace_secs=0.02,
                session_id=sid, config_dir=cfg_dir,
                working_idle_secs=10.0, absolute_cap_secs=5.0,
            )

        assert result.timed_out is False, (
            'Expected the productive run to survive the 0.1s flat ceiling and exit '
            'normally — if it was killed, extension_engaged never latched and the '
            'elif branch under test never ran.'
        )
        assert result.duration_ms >= 100, (
            f'Expected the run to outlive the old 0.1s ceiling (proving the extension '
            f'regime engaged), got duration_ms={result.duration_ms}'
        )
        assert len(recorded) >= 2, (
            f'Expected at least one read through the elif extension branch (call index '
            f'>= 1); only {len(recorded)} read(s) happened, so index 0 (the pre-latch '
            f'if-branch) may be all that ran.'
        )
        assert loop_ident not in recorded, (
            f'count_transcript_turns ran on the event-loop thread ({loop_ident}) — '
            f'the working-regime progress-extension poll at cli_invoke.py:2850 blocks '
            f'the shared loop for every agent, for its entire working lifetime. '
            f'Recorded idents: {sorted(set(recorded))}'
        )


class TestTimeoutPathRereadOffLoop:
    """The one-shot post-kill re-read (``cli_invoke.py:2970``, inside the
    ``except TimeoutError:`` handler) must read off the loop.

    This read stamps ``transcript_turns`` onto the returned ``_SubprocessResult``,
    which ``classify_agent_failure`` surfaces in its ``diagnostic_detail`` — so
    the value must still arrive intact across the thread hop, not just off-loop.
    """

    async def test_timeout_path_transcript_reread_runs_off_the_loop_thread(self, tmp_path):
        """The post-kill re-read runs on a worker thread, and its value still lands.

        The fake returns a fixed sentinel 7 on every call.  7 >= 1 latches
        ``seen_turn`` on the first poll, so the run is killed by the flat
        ``timeout_seconds`` ceiling rather than the startup-wedge branch — either
        route reaches the same ``except TimeoutError:`` handler and its re-read.

        Because the extension params are NOT passed, ``extension_engaged`` stays
        False and no further poll reads happen after the latch: ``recorded`` is
        the (off-loop, post-step-2) latching poll plus the handler's re-read, so
        any loop-thread ident in it can only have come from the handler.
        """
        loop_ident = threading.get_ident()
        sid = str(uuid.uuid4())
        cfg_dir = tmp_path / 'cfg'
        cfg_dir.mkdir()

        proc, _ = _make_hanging_proc()
        recorded: list[int] = []

        with (
            patch(
                'shared.cli_invoke.asyncio.create_subprocess_exec',
                side_effect=_fake_exec_returning(proc),
            ),
            patch('shared.cli_invoke.terminate_process_group', AsyncMock()),
            patch(
                'shared.cli_invoke.count_transcript_turns',
                side_effect=_ident_recorder(recorded, 7),
            ),
            patch('shared.cli_invoke._WATCHDOG_POLL_SECS', 0.05),
        ):
            result = await _run_subprocess(
                ['fake'], cwd=tmp_path, env={}, model='opus',
                timeout_seconds=0.4, startup_grace_secs=0.05,
                session_id=sid, config_dir=cfg_dir,
            )

        assert result.timed_out is True, 'Expected the ceiling kill to fire'
        assert result.transcript_turns == 7, (
            f'The one-shot re-read must still execute and its value still reach the '
            f'_SubprocessResult stamp across the thread hop; got '
            f'transcript_turns={result.transcript_turns}'
        )
        assert loop_ident not in recorded, (
            f'count_transcript_turns ran on the event-loop thread ({loop_ident}) — the '
            f'post-kill re-read at cli_invoke.py:2970 blocks the shared loop. '
            f'Recorded idents: {sorted(set(recorded))}'
        )

    async def test_timeout_path_skips_read_when_session_id_missing(self, tmp_path):
        """PRESERVATION GUARD (expected to pass before and after the change).

        With ``session_id=None`` the ``if (config_dir and session_id)`` guard
        short-circuits: no read happens at all, so no thread hop is paid when
        there is nothing to read, and ``transcript_turns`` stays None.  Mirrors
        ``test_run_subprocess_transcript_turns_none_when_no_session_id``
        (test_cli_invoke.py:1842).
        """
        cfg_dir = tmp_path / 'cfg'
        cfg_dir.mkdir()

        proc, _ = _make_hanging_proc()
        recorded: list[int] = []

        with (
            patch(
                'shared.cli_invoke.asyncio.create_subprocess_exec',
                side_effect=_fake_exec_returning(proc),
            ),
            patch('shared.cli_invoke.terminate_process_group', AsyncMock()),
            patch(
                'shared.cli_invoke.count_transcript_turns',
                side_effect=_ident_recorder(recorded, 7),
            ),
            patch('shared.cli_invoke._WATCHDOG_POLL_SECS', 0.05),
        ):
            result = await _run_subprocess(
                ['fake'], cwd=tmp_path, env={}, model='opus',
                timeout_seconds=0.4, startup_grace_secs=0.05,
                session_id=None, config_dir=cfg_dir,
            )

        assert result.timed_out is True, 'Expected the ceiling kill to fire'
        assert recorded == [], (
            f'No transcript read may be attempted without a session_id — the guard '
            f'must short-circuit before any to_thread hop. Recorded {len(recorded)} call(s).'
        )
        assert result.transcript_turns is None, (
            f'Expected transcript_turns=None with no session_id, got {result.transcript_turns}'
        )


# ── Normal-exit fixture ─────────────────────────────────────────────────────
# Two assistant records whose tail is an unreaped background launch.  Borrowed
# from TestDetectEndedAwaitingBackground::test_single_abandoned_launch_is_true
# (test_cli_invoke_background.py:99) so the expected ended_awaiting_background
# value is one the detector's own suite already pins, not one invented here.
#   → transcript_turns == 2 (two type='assistant' records)
#   → ended_awaiting_background is True (launch never followed by a reap)
_ABANDONED_LAUNCH_RECORDS = [
    {
        'type': 'assistant',
        'message': {
            'role': 'assistant',
            'content': [{'type': 'text', 'text': 'kick off the long build'}],
        },
    },
    {
        'type': 'assistant',
        'message': {
            'role': 'assistant',
            'content': [{
                'type': 'tool_use',
                'name': 'Bash',
                'input': {'command': './occt-test.sh', 'run_in_background': True},
            }],
        },
    },
]


class TestNormalExitReadOffLoop:
    """The normal-exit read (``cli_invoke.py:3033``, ``read_transcript_records``)
    must read off the loop.

    This site had NO direct assertion anywhere in the repo before task 3925 — no
    test patched ``shared.cli_invoke.read_transcript_records``, and the only
    tests reaching it did so against an empty cfg_dir (records → None).  These
    tests add that coverage.

    NOTE ON ISOLATION.  ``count_transcript_turns`` delegates to the module-global
    ``read_transcript_records``, so a watchdog poll firing during these tests
    would append to ``recorded`` through that indirection.  Both tests therefore
    patch ``count_transcript_turns`` to a no-op returning None, which guarantees
    every entry in ``recorded`` came from the normal-exit site — without which
    ``len(recorded) == 1`` would not mean what it claims.
    """

    async def test_normal_exit_transcript_read_runs_off_the_loop_thread(self, tmp_path):
        """The normal-exit read runs on a worker thread, once, and both derived
        signals still fall out of the thread-returned list.

        ``len(recorded) == 1`` pins the task-2761 no-double-I/O property: ONE
        read feeds BOTH ``transcript_turns`` and ``ended_awaiting_background``.
        That property must survive the thread hop — a refactor that offloaded
        each derivation separately would read the file twice and still pass every
        other assertion here.
        """
        loop_ident = threading.get_ident()
        sid = str(uuid.uuid4())
        cfg_dir = tmp_path / 'cfg'
        cfg_dir.mkdir()

        proc = _make_delayed_success_proc(0.02)
        recorded: list[int] = []

        def read_recorder(config_dir, session_id):
            recorded.append(threading.get_ident())
            return list(_ABANDONED_LAUNCH_RECORDS)

        with (
            patch(
                'shared.cli_invoke.asyncio.create_subprocess_exec',
                side_effect=_fake_exec_returning(proc),
            ),
            patch('shared.cli_invoke.count_transcript_turns', return_value=None),
            patch('shared.cli_invoke.read_transcript_records', side_effect=read_recorder),
        ):
            result = await _run_subprocess(
                ['fake'], cwd=tmp_path, env={}, model='opus',
                timeout_seconds=5.0,
                session_id=sid, config_dir=cfg_dir,
            )

        assert result.timed_out is False, 'Expected a normal exit, not a kill'
        assert len(recorded) == 1, (
            f'Expected exactly ONE normal-exit transcript read feeding both derived '
            f'signals (task 2761 no-double-I/O); got {len(recorded)} read(s).'
        )
        assert loop_ident not in recorded, (
            f'read_transcript_records ran on the event-loop thread ({loop_ident}) — the '
            f'normal-exit read at cli_invoke.py:3033 blocks the shared loop. '
            f'Recorded idents: {sorted(set(recorded))}'
        )
        assert result.transcript_turns == 2, (
            f'Expected the assistant-turn count derived from the thread-returned list, '
            f'got {result.transcript_turns}'
        )
        assert result.ended_awaiting_background is True, (
            'Expected the abandoned-background-launch tail to be detected from the '
            'thread-returned list — both post-read derivation passes must still work.'
        )

    async def test_normal_exit_skips_read_when_config_dir_missing(self, tmp_path):
        """PRESERVATION GUARD (expected to pass before and after the change).

        With ``config_dir=None`` the ``if (config_dir and session_id)`` guard
        short-circuits: no read, no thread hop, and both signals take their
        documented fail-safe defaults (``transcript_turns=None``,
        ``ended_awaiting_background=False``).
        """
        sid = str(uuid.uuid4())
        proc = _make_delayed_success_proc(0.02)
        recorded: list[int] = []

        def read_recorder(config_dir, session_id):
            recorded.append(threading.get_ident())
            return list(_ABANDONED_LAUNCH_RECORDS)

        with (
            patch(
                'shared.cli_invoke.asyncio.create_subprocess_exec',
                side_effect=_fake_exec_returning(proc),
            ),
            patch('shared.cli_invoke.count_transcript_turns', return_value=None),
            patch('shared.cli_invoke.read_transcript_records', side_effect=read_recorder),
        ):
            result = await _run_subprocess(
                ['fake'], cwd=tmp_path, env={}, model='opus',
                timeout_seconds=5.0,
                session_id=sid, config_dir=None,
            )

        assert result.timed_out is False, 'Expected a normal exit, not a kill'
        assert recorded == [], (
            f'No transcript read may be attempted without a config_dir — the guard must '
            f'short-circuit before any to_thread hop. Recorded {len(recorded)} call(s).'
        )
        assert result.transcript_turns is None, (
            f'Expected the fail-safe default transcript_turns=None, got {result.transcript_turns}'
        )
        assert result.ended_awaiting_background is False, (
            'Expected the fail-safe default ended_awaiting_background=False'
        )
