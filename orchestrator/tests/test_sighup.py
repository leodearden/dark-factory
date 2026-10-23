"""Regression test for SIGHUP-kills-orchestrator.

`UsageGate.__init__` calls `register_signal_handlers()`, but the orchestrator
constructs the gate from `Harness(config)` *before* `asyncio.run(_main())`.
At construction time there is no running loop, so `add_signal_handler` is
deferred — and SIGHUP keeps its default disposition (terminate) until the
harness explicitly re-installs it inside its own coroutine.

These tests reproduce the exact shape: build a gate outside the loop,
`asyncio.run()` a coroutine that calls `register_signal_handlers()`, then
send SIGHUP and SIGTERM back-to-back. Under the default disposition SIGHUP
is fatal when it is sent, before SIGTERM can be delivered, so the child's
exit status alone says whether SIGHUP killed it; no wall-clock wait is
involved. A control test runs the same sequence without the in-loop
registration and expects death by SIGHUP.
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

    # Start from the orchestrator's real default disposition; an inherited
    # SIG_IGN survives exec and would mask the bug.
    signal.signal(signal.SIGHUP, signal.SIG_DFL)

    REGISTER_IN_LOOP = '--skip-in-loop-registration' not in sys.argv[1:]

    os.environ.setdefault('TEST_SIGHUP_TOKEN', 'fake-token')
    config = UsageCapConfig(
        accounts=[AccountConfig(name='a', oauth_token_env='TEST_SIGHUP_TOKEN')],
        wait_for_reset=False,
        auth_reprobe_secs=3600,
    )

    # Build the gate BEFORE the event loop exists — mirrors orchestrator
    # cli.run() / Harness(config) sequence. __init__ defers handler install.
    gate = UsageGate(config)
    assert signal.getsignal(signal.SIGHUP) is signal.SIG_DFL, (
        'pre-loop construction must defer SIGHUP install'
    )


    async def main():
        if REGISTER_IN_LOOP:
            # Install inside the loop, like harness.run() now does.
            gate.register_signal_handlers()
            assert signal.getsignal(signal.SIGHUP) is not signal.SIG_DFL, (
                'in-loop registration must succeed'
            )

        # Park forever; SIGTERM will cancel us, SIGHUP must NOT terminate.
        loop = asyncio.get_running_loop()
        main_task = asyncio.current_task()
        assert main_task is not None
        loop.add_signal_handler(signal.SIGTERM, main_task.cancel)
        loop.add_signal_handler(signal.SIGINT, main_task.cancel)
        print('READY', flush=True)

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

# These only bound a genuinely hung child; with the 5s reap they stay well
# inside VERIFY_CLI_PER_TEST_TIMEOUT, so a hang fails as an assertion.
READY_WITHIN_SECS = 60.0
EXIT_WITHIN_SECS = 60.0


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


def _exit_after_hup_then_term(
    tmp_path: Path, *, register_in_loop: bool,
) -> subprocess.CompletedProcess[bytes]:
    """Run the child to READY, send SIGHUP then SIGTERM, and collect its exit.

    SIGHUP is sent first and is fatal at send time under SIG_DFL, so the exit
    status alone says whether SIGHUP terminated the child. The empty PATH
    keeps the gate's SIGHUP re-probe from spawning a real `claude`.
    """
    script = tmp_path / 'sighup_shape.py'
    script.write_text(SIGHUP_SCRIPT)
    no_claude_on_path = tmp_path / 'no-claude-on-path'
    no_claude_on_path.mkdir()
    argv = [sys.executable, str(script)]
    if not register_in_loop:
        argv.append('--skip-in-loop-registration')

    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, 'PATH': str(no_claude_on_path)},
        start_new_session=True,
    )
    pgid = proc.pid
    try:
        ready = _await_ready(proc, within_secs=READY_WITHIN_SECS)
        proc.send_signal(signal.SIGHUP)
        proc.send_signal(signal.SIGTERM)
        try:
            out, err = proc.communicate(timeout=EXIT_WITHIN_SECS)
        except subprocess.TimeoutExpired as exc:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pgid, signal.SIGKILL)
            _, err = proc.communicate(timeout=5.0)
            raise AssertionError(
                f'process did not exit within {EXIT_WITHIN_SECS}s of SIGTERM; '
                f'stderr={err!r}'
            ) from exc
        return subprocess.CompletedProcess(argv, proc.returncode, ready + out, err)
    finally:
        if proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pgid, signal.SIGKILL)
            proc.wait(timeout=5.0)


def test_sighup_does_not_kill_process(tmp_path: Path):
    """SIGHUP then SIGTERM must end in a clean cancellation, not death by SIGHUP.

    This is the regression test that would have caught the original bug
    (`kill -HUP <pid>` terminating the orchestrator within 2s).
    """
    result = _exit_after_hup_then_term(tmp_path, register_in_loop=True)
    assert result.returncode != -signal.SIGHUP, (
        'SIGHUP terminated the process; register_signal_handlers() inside '
        'the loop did not install the handler'
    )
    assert result.returncode == 0 and b'CANCELLED' in result.stdout, (
        f'expected a clean cancellation on SIGTERM, got rc={result.returncode}; '
        f'stdout={result.stdout!r}; stderr={result.stderr!r}'
    )


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


def test_without_in_loop_registration_sighup_kills_the_process(tmp_path: Path):
    """The original bug shape must die by SIGHUP under the exact signal
    sequence the regression test uses, so a pass there means the handler is
    installed, not that the check is blind."""
    result = _exit_after_hup_then_term(tmp_path, register_in_loop=False)
    assert result.returncode == -signal.SIGHUP, (
        f'expected death by SIGHUP without in-loop registration, '
        f'got rc={result.returncode}; stderr={result.stderr!r}'
    )
