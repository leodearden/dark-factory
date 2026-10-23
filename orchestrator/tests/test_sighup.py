"""Regression test for SIGHUP-kills-orchestrator.

`UsageGate.__init__` calls `register_signal_handlers()`, but the orchestrator
constructs the gate from `Harness(config)` *before* `asyncio.run(_main())`.
At construction time there is no running loop, so `add_signal_handler` is
deferred — and SIGHUP keeps its default disposition (terminate) until the
harness explicitly re-installs it inside its own coroutine.

This test reproduces the exact shape: build a gate outside the loop,
`asyncio.run()` a coroutine that calls `register_signal_handlers()`, and
verify that SIGHUP no longer kills the process.

Mirrors the subprocess pattern in ``test_shutdown.py``.
"""

from __future__ import annotations

import contextlib
import os
import select
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest
from _orch_helpers import VERIFY_CLI_PER_TEST_TIMEOUT

# Every test in this module spawns a python child.
pytestmark = pytest.mark.timeout(VERIFY_CLI_PER_TEST_TIMEOUT)

SIGHUP_SCRIPT = textwrap.dedent('''
    """Minimal shape: build UsageGate outside loop, register inside loop."""
    import asyncio
    import os
    import signal
    import sys

    from shared.config_models import AccountConfig, UsageCapConfig
    from shared.usage_gate import UsageGate

    os.environ.setdefault('TEST_SIGHUP_TOKEN', 'fake-token')
    config = UsageCapConfig(
        accounts=[AccountConfig(name='a', oauth_token_env='TEST_SIGHUP_TOKEN')],
        wait_for_reset=False,
        auth_reprobe_secs=3600,
    )

    # Build the gate BEFORE the event loop exists — mirrors orchestrator
    # cli.run() / Harness(config) sequence. __init__ defers handler install.
    gate = UsageGate(config)
    assert gate._sighup_handler_installed is False, (
        'pre-loop construction must defer SIGHUP install'
    )


    async def main():
        # Install inside the loop, like harness.run() now does.
        gate.register_signal_handlers()
        assert gate._sighup_handler_installed is True, (
            'in-loop registration must succeed'
        )
        print('READY', flush=True)

        # Park forever; SIGTERM will cancel us, SIGHUP must NOT terminate.
        loop = asyncio.get_running_loop()
        main_task = asyncio.current_task()
        assert main_task is not None
        loop.add_signal_handler(signal.SIGTERM, main_task.cancel)
        loop.add_signal_handler(signal.SIGINT, main_task.cancel)

        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            print('CANCELLED', flush=True)
            return


    try:
        asyncio.run(main())
    except asyncio.CancelledError:
        sys.exit(0)
''')


def _await_ready(proc: subprocess.Popen[bytes], *, within_secs: float) -> bytes:
    """Return the child's stdout through its READY line.

    Fails with AssertionError on deadline or EOF rather than blocking.
    """
    assert proc.stdout is not None
    fd = proc.stdout.fileno()
    deadline = time.monotonic() + within_secs
    seen = b''
    while b'READY' not in seen.splitlines():
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
            raise AssertionError(
                f'no READY line within {within_secs}s '
                f'(rc={proc.poll()}, stdout={seen!r})'
            )
        chunk = os.read(fd, 4096)
        if not chunk:
            raise AssertionError(
                f'stdout closed before READY (rc={proc.poll()}, stdout={seen!r})'
            )
        seen += chunk
    return seen


@pytest.mark.timeout(30)
def test_sighup_does_not_kill_process(tmp_path: Path):
    """Send SIGHUP and verify the process is still alive ~1s later.

    This is the regression test that would have caught the original bug
    (`kill -HUP <pid>` terminating the orchestrator within 2s).
    """
    script = tmp_path / 'sighup_shape.py'
    script.write_text(SIGHUP_SCRIPT)

    proc = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    pgid = proc.pid

    try:
        # Wait for READY.
        deadline = time.monotonic() + 10.0
        ready = False
        while time.monotonic() < deadline:
            line = proc.stdout.readline() if proc.stdout else b''
            if line.strip() == b'READY':
                ready = True
                break
        assert ready, 'sighup_shape.py never reached READY'

        # Send SIGHUP — bug shape: this used to terminate the process.
        proc.send_signal(signal.SIGHUP)

        # Sleep, then assert still alive.
        time.sleep(1.0)
        assert proc.poll() is None, (
            f'process exited on SIGHUP (rc={proc.returncode}); '
            'register_signal_handlers() did not install the handler'
        )

        # Clean up via SIGTERM.
        proc.send_signal(signal.SIGTERM)
        try:
            rc = proc.wait(timeout=10.0)
        except subprocess.TimeoutExpired as exc:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pgid, signal.SIGKILL)
            proc.wait(timeout=5.0)
            raise AssertionError('process did not exit on SIGTERM') from exc

        assert rc == 0, f'expected clean exit on SIGTERM, got rc={rc}'
    finally:
        if proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pgid, signal.SIGKILL)
            proc.wait(timeout=5.0)


@pytest.mark.parametrize(
    'code',
    [
        pytest.param(
            "import sys, time; sys.stdout.write('REA'); sys.stdout.flush(); "
            'time.sleep(600)',
            id='alive-mid-line',
        ),
        pytest.param(
            "import sys; sys.stdout.write('not ready\\n')",
            id='exits-without-ready',
        ),
    ],
)
def test_ready_wait_fails_instead_of_blocking(code: str):
    proc = subprocess.Popen(
        [sys.executable, '-c', code],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    pgid = proc.pid
    try:
        with pytest.raises(AssertionError, match='READY'):
            _await_ready(proc, within_secs=2.0)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pgid, signal.SIGKILL)
        proc.wait(timeout=5.0)
