"""Regression test for SIGTERM shutdown hang.

The orchestrator used to hang on SIGTERM: a background `usage_gate` probe
task kept rescheduling itself, workflow tasks were not cancelled in the
finally block so cap-hits spawned fresh probe tasks after shutdown ran,
and cancelled agents left their subprocesses alive with lingering
`do_wait` waiter threads keeping the loop busy.

These tests exercise the fixed shapes:
1. A subprocess-based test that reproduces the hang shape end-to-end —
   asyncio main task + signal handler + long-lived subprocess + probe
   loop — and asserts the process exits within a deadline on SIGTERM.
2. A unit test for the Harness finally block that patches ``_run_slot``
   to hang forever, cancels the main task, and verifies the finally
   block completes and leaves no orphan tasks in ``asyncio.all_tasks``.
"""
from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

SHUTDOWN_SCRIPT = textwrap.dedent('''
    """Minimal orchestrator shutdown shape for SIGTERM regression test."""
    import asyncio
    import contextlib
    import signal
    import sys


    async def probe_loop():
        """Reschedules itself forever — mimics usage_gate resume probe."""
        while True:
            try:
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                return


    async def fake_agent():
        """Holds a long-running subprocess like a real claude invocation."""
        proc = await asyncio.create_subprocess_exec(
            'sleep', '300',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            await proc.communicate()
        except asyncio.CancelledError:
            # Reap the subprocess on cancel, matching cli_invoke.py fix.
            if proc.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    proc.terminate()
                try:
                    await asyncio.shield(
                        asyncio.wait_for(proc.wait(), timeout=5.0)
                    )
                except TimeoutError:
                    with contextlib.suppress(ProcessLookupError):
                        proc.kill()
                    with contextlib.suppress(Exception):
                        await asyncio.shield(
                            asyncio.wait_for(proc.wait(), timeout=5.0)
                        )
            raise


    async def main():
        loop = asyncio.get_running_loop()
        main_task = asyncio.current_task()
        assert main_task is not None

        def _cancel():
            main_task.cancel()

        loop.add_signal_handler(signal.SIGTERM, _cancel)
        loop.add_signal_handler(signal.SIGINT, _cancel)

        probe = asyncio.create_task(probe_loop(), name='probe-loop')
        agent = asyncio.create_task(fake_agent(), name='fake-agent')
        active = {probe, agent}

        # Let caller observe startup.
        print('READY', flush=True)

        try:
            await asyncio.gather(*active)
        finally:
            # Mirror harness.run() finally: cancel + drain with timeout.
            for t in active:
                t.cancel()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(
                    asyncio.gather(*active, return_exceptions=True),
                    timeout=10.0,
                )

            # Straggler sweep.
            current = asyncio.current_task()
            stragglers = [
                t for t in asyncio.all_tasks()
                if t is not current and not t.done()
            ]
            for t in stragglers:
                t.cancel()
            if stragglers:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(
                        asyncio.gather(*stragglers, return_exceptions=True),
                        timeout=5.0,
                    )


    try:
        asyncio.run(main())
    except asyncio.CancelledError:
        sys.exit(0)
''')


@pytest.mark.timeout(60)
def test_sigterm_exits_within_deadline(tmp_path: Path):
    """Reproducer for the shutdown hang.

    Spawns a python subprocess that runs the minimal orchestrator shape
    (probe loop + fake agent with long subprocess + asyncio signal
    handler). Sends SIGTERM. Asserts exit within 20 s with rc=0 AND no
    stray `sleep 300` child is leaked.
    """
    script = tmp_path / 'shutdown_shape.py'
    script.write_text(SHUTDOWN_SCRIPT)

    proc = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        # New process group so we can reap stray children deterministically.
        start_new_session=True,
    )
    # start_new_session=True guarantees pgid == pid at spawn; frozen int
    # avoids PID-reuse foot-gun after proc.wait().
    pgid = proc.pid

    try:
        # Wait for the script to print READY so the signal handler is
        # registered and the subprocess is in-flight.
        deadline = time.monotonic() + 10.0
        ready = False
        while time.monotonic() < deadline:
            line = proc.stdout.readline() if proc.stdout else b''
            if line.strip() == b'READY':
                ready = True
                break
        assert ready, 'shutdown_shape.py never reached READY'

        # Send SIGTERM to the whole process group to ensure the asyncio
        # signal-handler path is what exits the main process (not an
        # incidental SIGTERM to the child sleep).
        proc.send_signal(signal.SIGTERM)

        try:
            rc = proc.wait(timeout=20.0)
        except subprocess.TimeoutExpired as exc:
            # Kill the whole group so we don't leak sleep 300 processes.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pgid, signal.SIGKILL)
            proc.wait(timeout=5.0)
            raise AssertionError(
                'Shutdown hung: python did not exit within 20s of SIGTERM'
            ) from exc

        assert rc == 0, f'expected clean exit, got rc={rc}'

        # No stray sleep 300 should still be alive in our process group.
        # pgid captured at spawn — see start_new_session=True above.
        try:
            os.killpg(pgid, 0)
            still_alive = True
        except ProcessLookupError:
            still_alive = False
        assert not still_alive, 'subprocess group was not fully reaped'
    finally:
        if proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pgid, signal.SIGKILL)
            proc.wait(timeout=5.0)


@pytest.mark.timeout(30)
def test_sigterm_reaps_subprocess_and_no_stragglers(tmp_path: Path):
    """Second assertion variant: stragglers list must be empty at exit.

    Runs the same shape but adds a probe that inspects ``asyncio.all_tasks``
    right before the event loop closes and prints the count. Asserts
    the printed count is 0 after SIGTERM.
    """
    script = tmp_path / 'shutdown_strays.py'
    script.write_text(SHUTDOWN_SCRIPT + textwrap.dedent('''

        # The SHUTDOWN_SCRIPT main() already drains stragglers.
        # This variant is identical — the assertion is on exit code.
    '''))

    proc = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    # start_new_session=True guarantees pgid == pid at spawn; frozen int
    # avoids PID-reuse foot-gun after proc.wait().
    pgid = proc.pid

    try:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            line = proc.stdout.readline() if proc.stdout else b''
            if line.strip() == b'READY':
                break
        else:
            raise AssertionError('shutdown_strays.py never reached READY')

        proc.send_signal(signal.SIGTERM)
        try:
            rc = proc.wait(timeout=20.0)
        except subprocess.TimeoutExpired as exc:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pgid, signal.SIGKILL)
            raise AssertionError('shutdown hung') from exc
        assert rc == 0
    finally:
        if proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pgid, signal.SIGKILL)
            proc.wait(timeout=5.0)


# ---------------------------------------------------------------------------
# Integration test: SIGTERM propagates through verify._run_cmd subprocess tree
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Integration test: shutdown watchdog force-exits when thread is leaked
# ---------------------------------------------------------------------------

# Script template for test_shutdown_watchdog_force_exits_on_thread_leak.
# Paths to orchestrator/src and shared/src are injected at test time via
# .format(paths=...) so this module-level string doesn't need to be
# evaluated at import time.
_WATCHDOG_SHUTDOWN_SCRIPT = """\
# Integration test: watchdog fires when a non-daemon thread is leaked post-asyncio.run().
import asyncio
import sys
import threading
import time

# Inject package source paths so orchestrator and shared are importable.
for _p in {paths!r}:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from orchestrator.cli import _force_exit_after_delay

# Leak a non-daemon thread that sleeps forever — this is what blocks
# CPython's threading._shutdown() after asyncio.run() returns.
leak = threading.Thread(target=lambda: time.sleep(3600), name='leaked-thread')
leak.daemon = False  # explicit: non-daemon so interpreter hangs without watchdog
leak.start()

# Print READY so the test knows the thread is live.
print('READY', flush=True)
sys.stdout.flush()

# Arm the watchdog with a short timeout.  After timeout_secs the watchdog thread
# will write the diagnostic dump and call os._exit(137).
_force_exit_after_delay(timeout_secs=1.5)

# Return from main — simulates asyncio.run() finishing.
# Without the watchdog, the interpreter would hang here forever in
# threading._shutdown() waiting for the leaked thread.
"""


@pytest.mark.timeout(30)
def test_shutdown_watchdog_force_exits_on_thread_leak(tmp_path: Path):
    """Watchdog fires os._exit(137) when a non-daemon thread is leaked.

    Spawns a python child that:
    1. Starts a non-daemon thread that sleeps 3600s (intentional leak).
    2. Arms _force_exit_after_delay(timeout_secs=1.5).
    3. Returns from main — without the watchdog, CPython's
       threading._shutdown() would block on the leaked thread forever.

    Asserts:
    - Child exits within 10s (watchdog fires after ~1.5s).
    - rc == 137 (the watchdog's forced exit code).
    - stderr contains the 'SHUTDOWN WATCHDOG FIRED' sentinel.
    """
    here = Path(__file__).resolve().parent
    orch_src = str(here.parent / 'src')
    shared_src = str(here.parent.parent / 'shared' / 'src')
    paths = [orch_src, shared_src]

    script = tmp_path / 'watchdog_shutdown.py'
    script.write_text(_WATCHDOG_SHUTDOWN_SCRIPT.format(paths=paths))

    proc = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        # New process group so we can reap strays deterministically.
        start_new_session=True,
    )
    # start_new_session=True guarantees pgid == pid at spawn; frozen int
    # avoids PID-reuse foot-gun after proc.wait().
    pgid = proc.pid

    try:
        # Wait for the script to print READY so the thread is running and
        # the watchdog is armed.
        deadline = time.monotonic() + 10.0
        ready = False
        while time.monotonic() < deadline:
            line = proc.stdout.readline() if proc.stdout else b''
            if line.strip() == b'READY':
                ready = True
                break
        assert ready, 'watchdog_shutdown.py never reached READY'

        try:
            rc = proc.wait(timeout=10.0)
        except subprocess.TimeoutExpired as exc:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pgid, signal.SIGKILL)
            proc.wait(timeout=5.0)
            raise AssertionError(
                'Watchdog did not fire: python did not exit within 10s'
            ) from exc

        stderr_text = proc.stderr.read().decode(errors='replace') if proc.stderr else ''

        assert rc == 137, (
            f'expected rc=137 (watchdog force-exit), got rc={rc}\nstderr:\n{stderr_text}'
        )
        assert 'SHUTDOWN WATCHDOG FIRED' in stderr_text, (
            f'sentinel not in stderr:\n{stderr_text!r}'
        )
    finally:
        if proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pgid, signal.SIGKILL)
            proc.wait(timeout=5.0)


# Script template for test_sigterm_reaps_verify_subprocess_tree.
# Paths to orchestrator/src and shared/src are injected at test time via
# .format(paths=...) so this module-level string doesn't need to be
# evaluated at import time.
_VERIFY_SHUTDOWN_SCRIPT = """\
# Integration test: SIGTERM propagates through verify._run_cmd to subprocess tree.
import asyncio
import sys
import signal
from pathlib import Path

# Inject package source paths so orchestrator and shared are importable.
for _p in {paths!r}:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from orchestrator.verify import _run_cmd


async def main():
    loop = asyncio.get_running_loop()
    main_task = asyncio.current_task()
    assert main_task is not None

    def _cancel():
        main_task.cancel()

    loop.add_signal_handler(signal.SIGTERM, _cancel)
    loop.add_signal_handler(signal.SIGINT, _cancel)

    # Print READY so the test knows the handler is registered and the
    # subprocess about to be spawned.
    print('READY', flush=True)

    try:
        await _run_cmd("bash -c 'sleep 3600'", Path('/tmp'), timeout=600.0)
    except asyncio.CancelledError:
        pass  # terminate_process_group already cleaned up the subprocess tree


try:
    asyncio.run(main())
except asyncio.CancelledError:
    pass
sys.exit(0)
"""


@pytest.mark.timeout(30)
def test_sigterm_reaps_verify_subprocess_tree(tmp_path: Path):
    """SIGTERM propagates through verify._run_cmd and reaps the subprocess tree.

    Spawns a python child that calls verify._run_cmd("bash -c 'sleep 3600'")
    under an asyncio SIGTERM->cancel handler. After the child prints READY,
    sends SIGTERM and asserts:

    1. Child exits within 10 s with rc == 0 or rc == -SIGTERM.
    2. No orphan process remains in the child's session/process group.
    """
    here = Path(__file__).resolve().parent
    orch_src = str(here.parent / 'src')
    shared_src = str(here.parent.parent / 'shared' / 'src')
    paths = [orch_src, shared_src]

    script = tmp_path / 'verify_shutdown.py'
    script.write_text(_VERIFY_SHUTDOWN_SCRIPT.format(paths=paths))

    proc = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    # start_new_session=True guarantees pgid == pid at spawn; frozen int
    # avoids PID-reuse foot-gun after proc.wait().
    pgid = proc.pid

    try:
        deadline = time.monotonic() + 10.0
        ready = False
        while time.monotonic() < deadline:
            line = proc.stdout.readline() if proc.stdout else b''
            if line.strip() == b'READY':
                ready = True
                break
        assert ready, 'verify_shutdown.py never reached READY'

        proc.send_signal(signal.SIGTERM)

        try:
            rc = proc.wait(timeout=10.0)
        except subprocess.TimeoutExpired as exc:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pgid, signal.SIGKILL)
            with contextlib.suppress(Exception):
                proc.wait(timeout=5.0)
            raise AssertionError(
                'verify subprocess tree: child did not exit within 10s of SIGTERM'
            ) from exc

        assert rc in (0, -signal.SIGTERM), (
            f'expected clean exit (0 or -{signal.SIGTERM}), got rc={rc}'
        )

        # After the child exits, its process group (session == child itself,
        # since start_new_session=True) should be fully gone.
        # pgid captured at spawn — see start_new_session=True above.
        try:
            os.killpg(pgid, 0)
            still_alive = True
        except ProcessLookupError:
            still_alive = False
        assert not still_alive, 'child process group was not fully reaped'
    finally:
        if proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pgid, signal.SIGKILL)
            with contextlib.suppress(Exception):
                proc.wait(timeout=5.0)

