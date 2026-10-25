"""Stdout doubles for testing the process-boundary guard, shared by BOTH suites.

:mod:`shared.cli_boundary` is exercised from two packages that cannot import
each other's test trees:

  * ``shared/tests/test_cli_boundary.py`` — the HELPER, against a synthetic CLI.
  * ``fused-memory/tests/test_local_memory_models_eval_corpus.py`` — the
    WIRING, against the real ``build_corpus.py`` end to end.

``shared.tests`` is not an importable package from the fused-memory pytest
rootdir, so the doubles live HERE, in the shipped ``shared`` package — the same
convention as :mod:`shared.testing` and :mod:`shared.testing_stdin` — rather
than being copy-pasted into each suite. That copy-paste is what this module
replaces: the pipe/fd dance, the ``quiet_close`` close-verdict assertion and
the buffering knob must stay in AGREEMENT for the two suites to mean the same
thing, and a change whose whole purpose was INV-5 ``no-lockstep-duplication``
should not have shipped its own second copy of the scaffolding (task 4328).

**These are real OS objects, not mocks, and that is the point.**
:func:`closed_pipe_stdout` opens an actual ``os.pipe()`` and closes the read
end, so the assertion is on a genuine kernel-level write failure; a fix that
only satisfies a mocked ``print`` cannot pass through it. The two ``StringIO``
subclasses are doubles only because ENOSPC cannot be provoked portably, and
they earn a second job by it — see their docstrings on ``fileno()``.

Pure stdlib, and registered as such in
``shared/tests/test_pure_stdlib_leaves.py``: importing a test helper must never
be the thing that drags a third-party package into a purity-checked tree. In
particular it does NOT import ``pytest`` — :func:`closed_pipe_stdout` takes the
``monkeypatch`` fixture as a plain duck-typed argument.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
from collections.abc import Iterator
from typing import Any

__all__ = [
    'StdoutWithAFailingFlush',
    'StdoutWithAFailingWrite',
    'closed_pipe_stdout',
]


@contextlib.contextmanager
def closed_pipe_stdout(
    monkeypatch: Any, *, buffering: int | None = None, quiet_close: bool = True
) -> Iterator[Any]:
    """Point ``sys.stdout`` at a pipe whose reader is already gone (the ``| head`` shape).

    A REAL closed ``os.pipe()``, never a mocked ``print``: a fix that only
    satisfies a mock cannot pass through here, because the assertion is on the
    actual OS-level write failure.

    *buffering* is the load-bearing knob, not a detail — it selects WHICH of
    the two failure regimes the test exercises, so every caller states it:

    * ``None`` (block buffering, the default) — a short write stays in the
      buffer, nothing fails in-band, and only an explicit flush can surface it.
    * ``1`` (line buffering) — every print does a real ``write()``, so the
      failure is raised during the run.

    The exit is itself an assertion. Closing the wrapper stands in for the
    interpreter's finalization-time flush of ``sys.stdout``, which is where a
    closed pipe would otherwise resurface as "Exception ignored ..." noise and
    a forced exit status no ``return`` can override. It runs in a ``finally``
    so a failing assertion in the body cannot skip it or leak the write fd;
    the verdict on it is checked only when the body passed, so a real failure
    is never masked by a second one here.

    *quiet_close* says which LAYER is under test. ``True`` (the PROCESS
    boundary — ``shared/src/shared/cli_boundary.py::run_cli``, or a consumer's
    delegating ``_cli``) means that flush must not raise, because the boundary
    owns fd 1 and redirects it to ``os.devnull``. ``False`` (a ``main`` alone,
    or a report-only call) means it must raise: ``main(argv) -> int``
    deliberately does not mutate a process-global fd on behalf of a caller that
    only asked for an exit code.

    *monkeypatch* is pytest's fixture, taken as a plain argument so this module
    stays a pure-stdlib leaf. Only ``setattr`` is used.
    """
    read_fd, write_fd = os.pipe()
    os.close(read_fd)  # the reader is already gone, like `head` after its window
    if buffering is None:
        pipe_stdout = os.fdopen(write_fd, 'w')
    else:
        pipe_stdout = os.fdopen(write_fd, 'w', buffering=buffering)
    original_stdout = sys.stdout
    monkeypatch.setattr(sys, 'stdout', pipe_stdout)
    close_failure: BaseException | None = None
    try:
        yield pipe_stdout
    finally:
        sys.stdout = original_stdout
        try:
            pipe_stdout.close()
        except BrokenPipeError as exc:
            close_failure = exc
    if quiet_close:
        assert close_failure is None, (
            'the interpreter\'s shutdown flush would have raised a second, '
            f'uncatchable BrokenPipeError: {close_failure!r}'
        )
    else:
        assert close_failure is not None, (
            'this layer took over the caller\'s stdout fd — only the process '
            'boundary may do that'
        )


class StdoutWithAFailingFlush(io.StringIO):
    """A stdout that accepts every write and fails on flush — ENOSPC, not a closed reader.

    ``StringIO.fileno()`` raises ``io.UnsupportedOperation``, so this doubles as
    a pin on the handler's fd guard: a stream with no real descriptor must not
    turn the reported failure into a second, different traceback out of the
    code meant to prevent the first one. (``io.UnsupportedOperation`` subclasses
    both ``OSError`` and ``ValueError``, which is exactly the pair
    ``shared/src/shared/cli_boundary.py::_silence_stream_fd`` names.)
    """

    def __init__(self, exc: OSError):
        super().__init__()
        self._exc = exc

    def flush(self) -> None:
        raise self._exc


class StdoutWithAFailingWrite(io.StringIO):
    """A stdout whose LARGE writes fail outright — the IN-BAND half of the pair above.

    :class:`StdoutWithAFailingFlush` accepts every write and defers the failure
    to the flush, which is what a *block-buffered* full disk does. A real
    ``> /full/disk/report.txt`` also fails the other way: once the write is big
    enough to reach the device — over the 8 KiB buffer, or on any write at all
    when unbuffered — the ``write()`` itself raises, mid-run, with no flush
    involved. Same errno, different frame, different handler needed.

    *min_length* gates which writes fail, so the double stays usable for the
    rest of the run rather than exploding on the first character. The threshold
    is deliberately the CALLER's to choose and to justify: it is a fact about
    the CLI under test (how large its writes actually are), not about this
    double. Against a REAL CLI that means measuring its writes and stating the
    measurement at the call site — too low and the double explodes before the
    run reaches the write under test, too high and nothing fails at all and the
    test goes green asserting nothing. Against a SYNTHETIC CLI the test simply
    picks a line length either side of the threshold it states.

    Keeps the ``io.StringIO`` base for the same second job as above:
    ``fileno()`` raises, re-pinning the handler's fd guard.
    """

    def __init__(self, exc: OSError, *, min_length: int):
        super().__init__()
        self._exc = exc
        self._min_length = min_length

    def write(self, s: str) -> int:
        if len(s) >= self._min_length:
            raise self._exc
        return super().write(s)
