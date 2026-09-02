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

import argparse
import os
import sys
from collections.abc import Callable
from typing import IO

__all__ = [
    'EXIT_STDOUT_FAILED',
    'LoudArgumentParser',
    'report_broken_pipe',
    'report_stdout_failure',
    'reset_stdout_failure_state',
    'run_cli',
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

      That callback is the ONE hook this module hands to arbitrary consumers,
      and it runs on the failure path — so it is called inside a deliberately
      WIDE ``except Exception``, the only such arm in the module. A consumer
      that formats a lazily-computed path, touches an object already torn down
      at shutdown, or does any I/O of its own would otherwise raise straight
      out through :func:`_handle_broken_pipe` and :func:`run_cli` as a chained
      traceback under an unhandled-exception status — precisely the two
      outcomes (no single ``error:`` line, no :data:`EXIT_STDOUT_FAILED`) this
      module exists to eliminate. A detail is a nicety; the line is the
      contract, so a broken detail costs its parentheses and nothing else.
      The width is safe HERE and nowhere else in the module: the arm spans one
      call whose only job is to produce a string.
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
    try:
        extra = _DETAIL() if _DETAIL is not None else None
    except Exception:  # deliberately wide: see the docstring's best-effort note
        extra = None
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


def _handle_broken_pipe() -> int:
    """:func:`report_broken_pipe` plus ownership of fd 1 — the PROCESS-boundary half.

    Called only from :func:`run_cli`. Redirecting a process-global file
    descriptor is legitimate at the boundary that is about to exit, and out of
    place in a ``main()``, which owes an in-process caller nothing but an exit
    code. That split is the whole reason this module exports the report-only
    form and keeps this one private.
    """
    _silence_stream_fd(sys.stdout)
    return report_broken_pipe()


def _handle_stdout_error(exc: OSError) -> int:
    """As :func:`_handle_broken_pipe`, for a stdout write that failed some OTHER way.

    A closed reader is not the only way ``> file`` or ``| cmd`` ends badly: a
    full disk or quota (``ENOSPC``/``EDQUOT``) and a disconnected terminal
    (``EIO``) fail the same write, and are at least as likely for a CLI whose
    report is routinely redirected. Reported through the same single
    ``error: ...`` line, with the errno text kept so the remedy is visible —
    "no space left on device" and "closed the output pipe" are different jobs.

    Serves BOTH frames such a failure can surface in, because a handler on one
    is not a handler on the other:

    * DEFERRED — every write is buffered and :func:`_flush_stdout`'s flush is
      what reaches the device.
    * IN-BAND — the write itself reaches the device and raises mid-run, from a
      ``print`` on the report path or from :class:`LoudArgumentParser`'s
      re-raise during ``parse_args``. Neither is inside a flush; both arrive at
      :func:`run_cli`'s ``except OSError``.

    One handler for both, so a full disk produces the same line and the same
    :data:`EXIT_STDOUT_FAILED` whichever frame it was noticed in.
    """
    _silence_stream_fd(sys.stdout)
    return report_stdout_failure(f'cannot write to stdout: {exc}')


def _flush_stdout() -> int | None:
    """Flush stdout; return an exit code if that write failed, else ``None``.

    Deliberately narrow. Only the flush is inside the ``try``, so the widened
    ``except OSError`` cannot reach anything else in the run and mis-attribute
    a store or filesystem failure to stdout — the mis-attribution hazard task
    3757 fixed by moving a store handler down to its own seam.
    """
    try:
        sys.stdout.flush()
    except BrokenPipeError:
        return _handle_broken_pipe()
    except OSError as exc:
        return _handle_stdout_error(exc)
    return None


class LoudArgumentParser(argparse.ArgumentParser):
    """An ``ArgumentParser`` whose help text cannot vanish down a failed stdout.

    ``argparse`` writes its messages inside a suppressed ``except OSError`` — so
    on ``--help | head`` (``BrokenPipeError``, an ``OSError``) or
    ``--help > /full/disk`` (``ENOSPC``) it discards the help text and exits 0
    anyway: a success status for a run whose entire output went nowhere.

    Only reachable when stdout is UNBUFFERED or line-buffered
    (``PYTHONUNBUFFERED=1``, ``python -u``, a tty), where the write hits fd 1
    during ``parse_args``. Block-buffered, the help text is still in the buffer
    at that point and the failure surfaces later at :func:`run_cli`'s explicit
    flush instead — which is why this hid behind the rest of the closed-pipe
    work, and why the tests run both regimes.

    Overrides the PUBLIC :meth:`print_help` rather than argparse's private
    message writer: stdout is the only stream help goes to, so this needs no
    private API and leaves argparse's routing, formatting and exit codes
    untouched — an unrecognized flag still reports to stderr and still exits 2.

    The re-raised exception leaves ``parse_args`` — which runs before a CLI's
    ``main`` body, and outside its own ``try`` — and lands in :func:`run_cli`'s
    stdout handlers: ``except BrokenPipeError`` for a closed reader,
    ``except OSError`` for every other way the write can fail. So a broken
    stdout gets ONE outcome across BOTH buffering regimes AND both failure
    kinds: :data:`EXIT_STDOUT_FAILED` plus a single ``error: ...`` line naming
    which of the two it was.
    """

    def print_help(self, file: IO[str] | None = None) -> None:
        (file or sys.stdout).write(self.format_help())


def run_cli(main: Callable[[], int]) -> int:
    """Process-boundary wrapper: run *main* and force stdout out while it can fail usefully.

    USAGE — the whole adoption shape for a sibling script::

        from shared.cli_boundary import LoudArgumentParser, run_cli

        def _build_parser():
            parser = LoudArgumentParser(...)
            ...

        def main(argv: list[str] | None = None) -> int:
            ...

        if __name__ == '__main__':
            sys.exit(run_cli(main))

    A closed stdout pipe (``cmd | head``) fails in two measurably different
    ways, and only one of them ever reaches ``main()``'s own
    ``except BrokenPipeError``:

    * LARGE write — the report overflows stdout's buffer, so ``print`` flushes
      mid-run, the ``write()`` fails, and the error is raised in-band.
      ``main()`` catches it.
    * SHORT write — a one-line verdict stays in the buffer, so the run raises
      NOTHING. ``main()`` returns its ordinary code and the failure surfaces
      only when the interpreter flushes ``sys.stdout`` during finalization,
      where no ``except`` can intervene and the returned status is overridden.

    The explicit flush is what converts the second case into the first. It is
    NOT redundant with the interpreter's own shutdown flush and must not be
    "simplified" away: its entire purpose is to attempt the buffered write
    *earlier*, while a handler is still on the stack, so a deferred and
    uncatchable failure becomes the same documented :data:`EXIT_STDOUT_FAILED`
    plus single ``error: ...`` line every other failure produces.

    WHERE THAT IS OBSERVABLE, which is worth stating because it is not
    obvious from any in-process test. The status override happens during
    interpreter FINALIZATION, so nothing running inside the process can see
    it: a test suite's interpreter never finalizes mid-suite, and an
    in-process test can only assert on the value ``run_cli`` returned — which
    is precisely the value CPython is free to discard. A REAL CHILD PROCESS is
    the only shape that observes the status the OS actually reports. That
    regime is covered by the ``_spawn`` tests in
    ``shared/tests/test_cli_boundary.py``, across both buffering regimes,
    including the ``2>&1 | head`` shape where BOTH streams are the closed pipe
    and the exit status is the only remaining observable. A change here that
    still passes every in-process test can regress that, so run those too.

    Kept OUT of ``main()`` deliberately. ``main(argv) -> int`` is the seam a
    test suite drives and its contract is "parse and run, return a code";
    interpreter-lifecycle concerns — flushing what is left, and redirecting fd
    1 so finalization cannot fail on it — belong at the process boundary, which
    is this function and the ``__main__`` guard that calls it.

    ``argparse`` accounts for the other two handlers, in complementary
    buffering regimes that are not reachable by each other's:

    * BLOCK-buffered — ``--help`` and an unrecognized flag leave ``parse_args``
      via ``SystemExit``, which happens inside ``main()`` but BEFORE its
      ``try``, with their text still buffered. The flush in that handler is
      what surfaces it.
    * UNBUFFERED (``PYTHONUNBUFFERED=1``, ``python -u``, a tty) — the write
      reaches fd 1 during ``parse_args`` and fails there, inside the
      ``except OSError: pass`` argparse wraps its own message writes in,
      leaving nothing for a later flush to find. :class:`LoudArgumentParser`
      re-raises it, and it arrives at one of the two stdout handlers below.

    A closed reader is not the only way stdout fails, so the arms below are a
    PAIR, and their order is the mechanism rather than a style choice:

    * ``except BrokenPipeError`` — the ``| head`` shape, which gets its own
      message because "the reader went away" has its own remedy.
    * ``except OSError`` — every other stdout failure raised IN-BAND: a full
      disk or quota on a report ``print``, a disconnected tty,
      ``print_help``'s re-raise onto anything that is not a pipe. It must stay
      SECOND, because ``BrokenPipeError`` is an ``OSError`` and a wide arm
      first would swallow the closed-pipe case and its distinct message.

    That second arm is safe HERE and would not be inside ``main()``, which is
    the distinction task 3757 turned on: a consumer converts every non-stdout
    ``OSError`` at the seam that knows what it means — the store read, the
    artifact write — so what is still an ``OSError`` by the time it reaches
    this frame is stdout-shaped by construction. ``main``'s ``try`` wraps the
    WHOLE run, so the same arm there would mis-attribute a store or filesystem
    failure to stdout. It belongs at the process boundary and nowhere else.

    The ``SystemExit`` handler RE-RAISES on a successful flush rather than
    returning a normalised code: ``SystemExit.code`` may be ``None`` or a
    non-int, and re-raising delegates every one of those shapes back to the
    interpreter that defines them instead of re-implementing the mapping here.
    So ``--help`` still exits 0 and a bad flag still exits 2, unchanged.

    ``SystemExit`` specifically, never a bare ``except BaseException``: that
    would swallow ``KeyboardInterrupt`` and genuine bugs behind a tidy
    message — a silent fail-soft, which is the inversion of this module's
    entire purpose (INV-11).

    :func:`reset_stdout_failure_state` at entry deliberately installs NO
    detail. A consumer that wants one installs it from its own ``main()``,
    which runs first and is also the seam an in-process test drives.
    """
    reset_stdout_failure_state()
    try:
        code = main()
    except BrokenPipeError:
        # Only reachable from LoudArgumentParser's re-raise, which happens in
        # parse_args — outside main()'s own try.
        return _handle_broken_pipe()
    except OSError as exc:
        # MUST stay below the BrokenPipeError arm: BrokenPipeError IS an
        # OSError, so the wide arm first would swallow the closed-pipe case and
        # lose its distinct message. Ordering is the whole mechanism here.
        return _handle_stdout_error(exc)
    except SystemExit:
        failed = _flush_stdout()
        if failed is not None:
            return failed
        raise
    failed = _flush_stdout()
    return code if failed is None else failed
