"""Self-tests for ``_leader_stderr``: the stderr drain and the trigger assertion."""
from __future__ import annotations

import contextlib
import select
import subprocess
import sys

import pytest
from _leader_stderr import (
    assert_watchdog_fire_trigger,
    drain_exited_stderr,
    watchdog_fire_trigger_report,
)

from orchestrator.verify_cancel import WatchdogTrigger


def _close_pipes(proc: subprocess.Popen) -> None:
    for pipe in (proc.stdin, proc.stderr):
        if pipe is not None:
            with contextlib.suppress(OSError):
                pipe.close()


def test_drain_exited_stderr_returns_everything_an_exited_process_wrote():
    """An exited writer's whole stderr comes back, first line to last.

    The payload stays far below one 64 KiB pipe buffer, so the writer can
    never block on a full pipe before this test reads it.
    """
    script = (
        'import sys\n'
        'sys.stderr.write("FIRST-LINE\\n")\n'
        'for i in range(200):\n'
        '    sys.stderr.write(f"filler-{i}\\n")\n'
        'sys.stderr.write("LAST-LINE\\n")\n'
    )
    proc = subprocess.Popen([sys.executable, '-c', script], stderr=subprocess.PIPE)
    try:
        proc.wait()

        drained = drain_exited_stderr(proc)

        assert drained.startswith(b'FIRST-LINE\n'), drained[:200]
        assert drained.endswith(b'LAST-LINE\n'), drained[-200:]
    finally:
        _close_pipes(proc)


def test_drain_exited_stderr_returns_while_a_survivor_still_holds_the_pipe():
    """The ceiling, not EOF, ends the drain when a survivor holds the write end.

    The child leaves behind a grandchild with DEFAULT stdio, so the
    grandchild inherits the stderr write end and keeps it open after the
    child exits.  The grandchild blocks reading the stdin pipe this test
    holds, so it outlives the drain and exits only when the test closes that
    pipe.  A drain that returns while its pipe still shows neither data nor
    EOF therefore stopped at its ceiling.  The proof is structural, with no
    elapsed-time assertion: an unbounded read never returns at all, and the
    test dies at pytest-timeout instead.
    """
    script = (
        'import subprocess, sys\n'
        'subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"])\n'
        'sys.stderr.write("LAST-LINE\\n")\n'
    )
    proc = subprocess.Popen(
        [sys.executable, '-c', script], stdin=subprocess.PIPE, stderr=subprocess.PIPE
    )
    try:
        proc.wait()

        drained = drain_exited_stderr(proc, ceiling_secs=0.5)

        assert b'LAST-LINE' in drained, drained[-2000:]
        assert not select.select([proc.stderr], [], [], 0)[0], (
            'the stderr pipe already shows EOF, so the drain may have ended at '
            'EOF rather than at its ceiling -- this test proves nothing about the '
            'bound unless the write end is still held when the drain returns'
        )
    finally:
        _close_pipes(proc)


def test_drain_exited_stderr_refuses_a_live_process():
    """A still-running writer is refused, naming its pid.

    Any snapshot of a live writer's stderr is racy, so a token assertion on
    it would be a flaky false negative rather than a finding.
    """
    proc = subprocess.Popen(
        [sys.executable, '-c', 'import time; time.sleep(300)'],
        stderr=subprocess.PIPE,
    )
    try:
        with pytest.raises(AssertionError) as exc_info:
            drain_exited_stderr(proc)

        message = str(exc_info.value)
        assert str(proc.pid) in message, message
        assert 'has not exited' in message, message
    finally:
        proc.kill()
        proc.wait()
        _close_pipes(proc)


def test_assert_watchdog_fire_trigger_discriminates_the_branch():
    """The trigger assertion passes only on exactly the expected token.

    This is the non-vacuity guard for every row that asserts a trigger: a
    helper that always passed would make all of those rows vacuous.  Each
    failing case strips the helper's echo of the (short) text from the
    message before checking it, so the reported and expected values must be
    named in the diagnosis itself, not just in the echoed input.
    """
    noise = 'INFO verify: building mod\nINFO verify: still running\n'

    def token_line(trigger: WatchdogTrigger) -> str:
        return watchdog_fire_trigger_report(trigger) + '\n'

    def diagnosis(text: str, expected: WatchdogTrigger, source: str) -> str:
        with pytest.raises(AssertionError) as exc_info:
            assert_watchdog_fire_trigger(text, expected, source=source)
        message = str(exc_info.value)
        assert repr(text) in message, f'the text is not echoed: {message!r}'
        return message.replace(repr(text), '')

    assert assert_watchdog_fire_trigger(
        noise + token_line(WatchdogTrigger.EOF) + noise, WatchdogTrigger.EOF, source='matching case'
    ) is None

    wrong_branch = diagnosis(
        noise + token_line(WatchdogTrigger.HEARTBEAT_STARVATION) + noise,
        WatchdogTrigger.EOF,
        'wrong-branch case',
    )
    assert WatchdogTrigger.HEARTBEAT_STARVATION.value in wrong_branch, wrong_branch
    assert watchdog_fire_trigger_report(WatchdogTrigger.EOF) in wrong_branch, wrong_branch
    assert 'wrong-branch case' in wrong_branch, wrong_branch

    unattributable = diagnosis(
        noise + 'TAIL-MARKER\n', WatchdogTrigger.HEARTBEAT_STARVATION, 'no-token case'
    )
    assert 'unattributable' in unattributable, unattributable
    assert (
        watchdog_fire_trigger_report(WatchdogTrigger.HEARTBEAT_STARVATION) in unattributable
    ), unattributable
    assert 'no-token case' in unattributable, unattributable

    both = diagnosis(
        token_line(WatchdogTrigger.EOF) + token_line(WatchdogTrigger.HEARTBEAT_STARVATION),
        WatchdogTrigger.EOF,
        'both-tokens case',
    )
    assert WatchdogTrigger.HEARTBEAT_STARVATION.value in both, both
