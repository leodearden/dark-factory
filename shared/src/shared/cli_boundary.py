"""The process boundary of a CLI: what happens to a run's exit status and streams on the way out.

WHAT THIS IS FOR. A ``sys.exit(main())`` script whose stdout is a closed pipe
(``cmd | head``), a full disk (``cmd > /full/disk/report.txt``) or a
disconnected terminal has three bad endings available to it, and this module
exists to replace all three with one documented outcome — :data:`EXIT_STDOUT_FAILED`
plus a single ``error: ...`` line:

* a raw traceback, where every other failure in the CLI prints one line;
* exit **0** for a run whose entire output went nowhere, because ``argparse``
  writes its messages inside a suppressed ``except OSError`` and a short
  buffered write never fails in-band at all;
* an exit status OVERRIDDEN by the interpreter, because finalization flushes
  ``sys.stdout``/``sys.stderr`` where no ``except`` can reach, prints
  "Exception ignored ..." and forces a status no ``return`` can undo.

ORIGIN. Built and measured by task 3900 inside
``fused-memory/scripts/local_memory_models_eval/build_corpus.py``, and hoisted
here by task 4328 the first time a second CLI needed it. The facts in these
docstrings were MEASURED, not reasoned — that ``argparse`` suppresses its own
``OSError``s, that a short buffered write only fails at interpreter
finalization, that ``2>&1 | head`` leaves stderr holding bytes too, that
``BrokenPipeError`` subclasses ``OSError`` so arm ORDER is the mechanism —
and re-deriving them costs more than reading them.

WHO SHOULD USE IT. The ~30 sibling ``sys.exit(main())`` CLIs under
``fused-memory/scripts/``, all of which print reports to a stdout an operator
routinely pipes or redirects. Reuse this rather than re-deriving a second copy
(INV-5 ``no-lockstep-duplication``): the adoption shape is two lines, given at
:func:`run_cli`.

LAYERING, which the split between the public and private names here encodes.
:func:`report_broken_pipe` and :func:`report_stdout_failure` report and stop
there — they touch no file descriptor, so a ``main(argv) -> int`` may call
them without mutating a process-global fd behind the back of an in-process
caller that asked only for an exit code. Taking over fd 1 is legitimate ONLY
at the boundary that is about to exit, which is :func:`run_cli` and the
private handlers it calls.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from typing import IO

#: Grown as each name lands (``LoudArgumentParser`` and ``run_cli`` follow),
#: so every commit stays lint-clean rather than carrying an F822 for a symbol
#: that does not exist yet. The finished surface is these six names.
__all__ = [
    'EXIT_STDOUT_FAILED',
    'report_broken_pipe',
    'report_stdout_failure',
    'reset_stdout_failure_state',
]

EXIT_STDOUT_FAILED = 1
"""The conventional "the run could not complete" status.

A consumer with its own exit-code table keeps its own constant and is
responsible for agreeing with this one — the agreement is behaviour, pinned
end-to-end by a real child process's exit status, not by comparing two names.
"""

_STDOUT_FAILURE_REPORTED = False
"""Whether this run already printed its one ``error: ...`` line about stdout.

Per-RUN state, cleared by :func:`reset_stdout_failure_state`. The same failure
is legitimately seen twice on the way out — in-band by ``main()``, then again
by :func:`_flush_stdout`'s flush of the buffer that write left behind — and
reporting it twice would break the single-line convention every other failure
in a CLI follows.
"""

_DETAIL: Callable[[], str | None] | None = None
"""An optional callback naming extra context to append to the failure line.

Per-RUN state, installed through :func:`reset_stdout_failure_state`'s *detail*
keyword and cleared by the next reset — so an in-process caller that runs the
CLI twice never reports the previous run's artifact as this one's output.

A CALLBACK rather than a string because of WHEN each end knows its value: the
consumer installs it at the top of its ``main()``, before it can possibly know
whether this run will write anything, and it is read only at the moment of
failure. Returning ``None`` means "nothing to add", which is how a run that
wrote nothing avoids claiming an artifact.
"""


def reset_stdout_failure_state(*, detail: Callable[[], str | None] | None = None) -> None:
    """Re-arm the once-only reporter and install (or clear) the detail callback.

    Called by :func:`run_cli` at entry, covering a consumer whose ``main``
    knows nothing about this module, and callable again as the first statement
    of that ``main`` to install a detail — which also covers an in-process test
    that drives ``main`` directly rather than through the boundary.
    Double-resetting is idempotent by design.
    """
    global _STDOUT_FAILURE_REPORTED, _DETAIL
    _STDOUT_FAILURE_REPORTED = False
    _DETAIL = detail


def _silence_stream_fd(stream: IO[str]) -> None:
    """Point *stream*'s file descriptor at ``os.devnull``.

    The dup2 dance from the interpreter docs' "Note on SIGPIPE". Python flushes
    ``sys.stdout`` and ``sys.stderr`` during finalization; if the fd still
    points at a closed pipe and the stream still holds the bytes an earlier
    write could not deliver, that flush fails where no ``except`` can reach it,
    printing "Exception ignored ..." and overriding the exit status this module
    documents. Redirecting the fd first means the later flush lands on a device
    that always accepts the write.

    Every step is best-effort: a stream with no real fd (a replaced
    ``sys.stdout``, a captured one under a test runner) raises from
    ``fileno()``, and this is called from the very handlers meant to keep a
    traceback off the screen — so a failure here degrades to "the message may
    be noisier", never to a second crash.
    """
    try:
        fd = stream.fileno()
    except (OSError, ValueError):  # io.UnsupportedOperation subclasses both
        return
    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
    except OSError:
        return
    try:
        os.dup2(devnull_fd, fd)
    except OSError:
        pass
    finally:
        os.close(devnull_fd)


def report_stdout_failure(detail: str) -> int:
    """Print ONE ``error: <detail>`` line about a failed stdout write; return the code.

    Reports only — it does not touch any file descriptor, so it is safe to call
    from a ``main()`` whose contract is "parse and run, return a code" rather
    than "own this process's streams".

    Two things the bare exit code cannot say are folded in here, because the
    message is the only channel left once the code has been spent:

    * :data:`_DETAIL` — whatever the consumer knows and this module cannot,
      typically an artifact that IS on disk even though
      :data:`EXIT_STDOUT_FAILED` says the run could not complete. The
      parentheses are this module's convention; the sentence inside them
      belongs to the caller that knows the artifact.
    * At most one line per run (:data:`_STDOUT_FAILURE_REPORTED`), because the
      same failure is legitimately caught twice on the way out.

    The write to stderr is best-effort. On the ordinary ``| head`` shape only
    stdout is the closed pipe, so it succeeds and looks like every other exit-1
    message; on ``2>&1 | head`` stderr is that same pipe, and the failed write
    leaves bytes in ITS buffer for the interpreter to choke on at shutdown —
    measured as an overridden exit status, so stderr gets the same treatment as
    stdout rather than being left to raise out of the one function whose job is
    to report failure quietly.
    """
    global _STDOUT_FAILURE_REPORTED
    if _STDOUT_FAILURE_REPORTED:
        return EXIT_STDOUT_FAILED
    _STDOUT_FAILURE_REPORTED = True
    extra = _DETAIL() if _DETAIL is not None else None
    if extra is not None:
        detail = f'{detail} ({extra})'
    try:
        print(f'error: {detail}', file=sys.stderr)
    except OSError:
        _silence_stream_fd(sys.stderr)
    return EXIT_STDOUT_FAILED


def report_broken_pipe() -> int:
    """A downstream reader (``| head``) closed stdout: report it, change nothing else.

    The ``print`` calls on a CLI's report path write to ``sys.stdout`` with no
    protection of their own. When the reader on the other end has already
    closed the pipe, that write raises :class:`BrokenPipeError`, which is not a
    bug in the run: it is the same "the run never reached its output" situation
    :data:`EXIT_STDOUT_FAILED` already documents, triggered by nothing being
    there to read. Left uncaught it surfaces as a raw traceback where every
    other failure prints a single ``error: ...`` line.

    Its message is deliberately distinct from :func:`_handle_stdout_error`'s:
    "the reader went away" and "no space left on device" have different
    remedies, so an operator must not be sent down the wrong one.
    """
    return report_stdout_failure(
        'downstream reader closed the output pipe before the run finished'
    )
