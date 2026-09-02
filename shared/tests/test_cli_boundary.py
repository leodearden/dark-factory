"""Tests for shared/src/shared/cli_boundary.py — the process-boundary stdout guard.

The module under test is the generic half of the closed-stdout guard task 3900
built inside ``fused-memory/scripts/local_memory_models_eval/build_corpus.py``,
hoisted here so the ~30 sibling ``sys.exit(main())`` CLIs under
``fused-memory/scripts/`` reuse it rather than re-derive it (INV-5).

SUBJECT SPLIT, deliberate. This file tests the HELPER against a synthetic
three-line CLI, so a regression here names the helper. The six closed-pipe
classes in ``fused-memory/tests/test_local_memory_models_eval_corpus.py`` stay
where they are and exercise the real ``build_corpus.py`` end to end, so a
regression there names the WIRING. They are not duplicates, and their
unchanged passing is the evidence that this hoist preserved behaviour.

THE THREE REGIMES this behaviour actually turns on, all covered below:

* IN-PROCESS, LINE-BUFFERED — every ``print`` does a real ``write()``, so a
  closed reader raises during the run and an in-band handler sees it.
* IN-PROCESS, BLOCK-BUFFERED — a short write stays in the buffer, nothing
  fails in-band, and only an explicit flush can surface it.
* REAL SUBPROCESS — the only shape that can observe the interpreter's
  finalization-time flush and the exit status the OS actually reports.

The doubles and the ``_closed_pipe_stdout`` context manager are ported from
``fused-memory/tests/test_local_memory_models_eval_corpus.py`` (2721-2777,
2968-3017, 3107-3199) with their MEASURED facts intact; the originals stay in
place serving the consumer-side tests.
"""

from __future__ import annotations

import contextlib
import errno
import io
import os
import sys

import pytest

import shared.cli_boundary as cli_boundary


@pytest.fixture(autouse=True)
def _fresh_reporter_state():
    """Arm the once-only reporter and clear any detail, before AND after each test.

    ``shared.cli_boundary`` keeps per-RUN module state on purpose (the same
    failure is legitimately caught twice on the way out, and the CLI's
    one-``error:``-line convention must survive that). A test that inherited a
    previous test's SPENT flag would assert nothing at all, and one that
    inherited a previous test's detail callback would assert the wrong text —
    so the reset happens on both edges rather than only at entry.
    """
    cli_boundary.reset_stdout_failure_state()
    yield
    cli_boundary.reset_stdout_failure_state()


def _reported_lines(err: str) -> list[str]:
    """The non-blank stderr lines — which this contract says is exactly one."""
    return [line for line in err.splitlines() if line.strip()]


class TestTheReporterPrintsOneErrorLineAndTouchesNoFileDescriptor:
    """The REPORT-ONLY half: a message and a code, with the caller's streams left alone.

    Split out from the fd-owning handlers because the split is the layering.
    ``report_*`` is safe to call from a ``main(argv) -> int``, whose contract
    is "parse and run, return a code" rather than "own this process's streams";
    only the process boundary that is about to exit may redirect fd 1.
    """

    def test_a_closed_reader_is_reported_as_one_error_line_naming_the_pipe(self, capsys):
        code = cli_boundary.report_broken_pipe()

        assert code == cli_boundary.EXIT_STDOUT_FAILED == 1
        reported = _reported_lines(capsys.readouterr().err)
        assert len(reported) == 1
        assert reported[0].startswith('error: ')
        assert 'closed the output pipe' in reported[0]

    def test_a_stdout_failure_that_is_not_a_closed_pipe_names_its_errno_instead(self, capsys):
        """"No space left on device" and "the reader went away" are different remedies.

        So they must be different messages: an operator who reads the wrong one
        looks for the wrong cause.
        """
        exc = OSError(errno.ENOSPC, 'No space left on device')
        code = cli_boundary.report_stdout_failure(f'cannot write to stdout: {exc}')

        assert code == cli_boundary.EXIT_STDOUT_FAILED
        reported = _reported_lines(capsys.readouterr().err)
        assert len(reported) == 1
        assert reported[0].startswith('error: ')
        assert 'No space left on device' in reported[0]
        assert 'closed the output pipe' not in reported[0]

    def test_a_second_report_in_the_same_run_returns_the_code_and_stays_silent(self, capsys):
        """AT MOST ONE line per run — the reason the module keeps a flag at all.

        The same stdout failure is legitimately seen twice on the way out:
        in-band by ``main()``, then again by the boundary's flush of the buffer
        that write left behind. Both must return the code; only the first may
        speak.
        """
        assert cli_boundary.report_broken_pipe() == cli_boundary.EXIT_STDOUT_FAILED
        assert len(_reported_lines(capsys.readouterr().err)) == 1

        assert cli_boundary.report_broken_pipe() == cli_boundary.EXIT_STDOUT_FAILED
        assert capsys.readouterr().err == ''

    def test_resetting_re_arms_the_reporter_for_the_next_run(self, capsys):
        """Otherwise a long-lived process running the CLI twice goes silent on run two."""
        cli_boundary.report_broken_pipe()
        capsys.readouterr()

        cli_boundary.reset_stdout_failure_state()
        assert cli_boundary.report_broken_pipe() == cli_boundary.EXIT_STDOUT_FAILED
        reported = _reported_lines(capsys.readouterr().err)
        assert len(reported) == 1
        assert 'closed the output pipe' in reported[0]

    def test_an_installed_detail_callback_is_appended_in_parentheses(self, capsys):
        """The parentheses are the generic convention; the sentence is the caller's.

        The bare exit code cannot say "the artifact IS on disk" — exit 1 means
        "the run could not complete", and an operator who reads that after the
        artifact was written may rebuild, or treat a good artifact as absent.
        So a consumer installs a callback naming what it landed, and the module
        supplies only the suffix shape.
        """
        cli_boundary.reset_stdout_failure_state(
            detail=lambda: 'the manifest was written to /tmp/x.json'
        )

        assert cli_boundary.report_broken_pipe() == cli_boundary.EXIT_STDOUT_FAILED
        reported = _reported_lines(capsys.readouterr().err)
        assert len(reported) == 1
        assert reported[0].endswith(' (the manifest was written to /tmp/x.json)')
        assert 'closed the output pipe' in reported[0]

    def test_a_detail_callback_that_returns_none_appends_nothing(self, capsys):
        """A run that wrote nothing must not claim an artifact.

        The callback returns ``None`` rather than being uninstalled, because
        the consumer installs it once at the top of ``main()`` — before it can
        possibly know whether this run will write anything.
        """
        cli_boundary.reset_stdout_failure_state(detail=lambda: None)

        assert cli_boundary.report_broken_pipe() == cli_boundary.EXIT_STDOUT_FAILED
        reported = _reported_lines(capsys.readouterr().err)
        assert len(reported) == 1
        assert '(' not in reported[0]

    def test_reporting_does_not_touch_the_stdout_file_descriptor(self, monkeypatch, capsys):
        """The layering assertion, made by DELIVERY rather than by introspection.

        ``fileno()`` would keep returning the same integer even after a
        ``dup2`` pointed it at ``os.devnull``, so the number proves nothing.
        Writing through the descriptor afterwards and reading the bytes off the
        other end of a real pipe is what actually distinguishes "left alone"
        from "silently redirected".

        The read end is set non-blocking so a swallowed write fails fast with
        ``BlockingIOError`` instead of hanging the suite on an empty pipe whose
        write end is still open.
        """
        read_fd, write_fd = os.pipe()
        os.set_blocking(read_fd, False)
        pipe_stdout = os.fdopen(write_fd, 'w', buffering=1)
        try:
            monkeypatch.setattr(sys, 'stdout', pipe_stdout)
            assert cli_boundary.report_broken_pipe() == cli_boundary.EXIT_STDOUT_FAILED
            print('still landing')
            monkeypatch.undo()
            assert os.read(read_fd, 1024) == b'still landing\n'
        finally:
            monkeypatch.undo()
            pipe_stdout.close()
            os.close(read_fd)
        # The report itself still went to stderr, which was never the subject.
        assert 'closed the output pipe' in capsys.readouterr().err


@contextlib.contextmanager
def _closed_pipe_stdout(monkeypatch, *, buffering: int | None = None, quiet_close: bool = True):
    """Point ``sys.stdout`` at a pipe whose reader is already gone (the ``| head`` shape).

    Ported from ``fused-memory/tests/test_local_memory_models_eval_corpus.py``
    (2721-2777), which keeps its own copy serving the consumer-side tests.

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

    *quiet_close* says which LAYER is under test. ``True`` (:func:`run_cli`,
    the process boundary) means that flush must not raise, because the
    boundary owns fd 1 and redirects it to ``os.devnull``. ``False`` (a
    ``main`` alone, or a report-only call) means it must raise:
    ``main(argv) -> int`` deliberately does not mutate a process-global fd on
    behalf of a caller that only asked for an exit code.
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


class _StdoutWithAFailingFlush(io.StringIO):
    """A stdout that accepts every write and fails on flush — ENOSPC, not a closed reader.

    ``StringIO.fileno()`` raises ``io.UnsupportedOperation``, so this doubles as
    a pin on the handler's fd guard: a stream with no real descriptor must not
    turn the reported failure into a second, different traceback out of the
    code meant to prevent the first one. (``io.UnsupportedOperation`` subclasses
    both ``OSError`` and ``ValueError``, which is exactly the pair
    ``_silence_stream_fd`` names.)
    """

    def __init__(self, exc: OSError):
        super().__init__()
        self._exc = exc

    def flush(self) -> None:
        raise self._exc


class _StdoutWithAFailingWrite(io.StringIO):
    """A stdout whose LARGE writes fail outright — the IN-BAND half of the pair above.

    ``_StdoutWithAFailingFlush`` accepts every write and defers the failure to
    the flush, which is what a *block-buffered* full disk does. A real
    ``> /full/disk/report.txt`` also fails the other way: once the write is big
    enough to reach the device — over the 8 KiB buffer, or on any write at all
    when unbuffered — the ``write()`` itself raises, mid-run, with no flush
    involved. Same errno, different frame, different handler needed.

    *min_length* gates which writes fail, so the double stays usable for the
    rest of the run rather than exploding on the first character. (In the
    corpus copy of this double the threshold is MEASURED against that CLI's
    four stdout writes; here the synthetic CLIs choose their own line lengths
    around whatever threshold the test states.)

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


class TestTheFlushIsWhatSurfacesADeferredStdoutFailure:
    """The FD-OWNING half: the handlers that may take over fd 1 because they are exiting.

    The two failure regimes are the axis this behaviour turns on, and only one
    of them can ever reach an in-band handler:

    * LINE-BUFFERED — the write reaches the device during the run and raises
      where a ``try`` around ``main()`` can see it.
    * BLOCK-BUFFERED — a short write stays in the buffer, the run raises
      NOTHING, and the failure surfaces only when something flushes. Without
      an explicit flush that is interpreter finalization, where no ``except``
      can intervene and the returned status is overridden.
    """

    def test_a_healthy_stdout_flushes_quietly_and_reports_nothing(self, capsys):
        """CONTROL: the flush is not a failure detector that fires on success."""
        assert cli_boundary._flush_stdout() is None
        assert capsys.readouterr().err == ''

    def test_a_short_buffered_write_surfaces_at_the_flush_and_nowhere_earlier(
        self, monkeypatch, capsys
    ):
        """The headline shape, and the reason the explicit flush exists at all.

        ``quiet_close=True`` is a second assertion, not decoration: it says the
        handler redirected fd 1 to ``os.devnull``, so the interpreter's later
        flush cannot fail on the buffer this one left behind.
        """
        with _closed_pipe_stdout(monkeypatch, quiet_close=True):
            print('short')  # block-buffered: stays in the buffer, raises nothing
            code = cli_boundary._flush_stdout()
        monkeypatch.undo()

        assert code == cli_boundary.EXIT_STDOUT_FAILED
        reported = _reported_lines(capsys.readouterr().err)
        assert len(reported) == 1
        assert 'closed the output pipe' in reported[0]

    def test_a_flush_failure_that_is_not_a_closed_pipe_routes_to_the_errno_message(
        self, monkeypatch, capsys
    ):
        """Arm ORDER, asserted behaviourally: BrokenPipeError is tried before OSError.

        ``BrokenPipeError`` IS an ``OSError``, so a wide arm placed first would
        swallow the closed-pipe case and print ITS message for everything. This
        test catches the mirror-image mistake — a narrow-only guard — and the
        closed-pipe test above catches the wide-first one.
        """
        monkeypatch.setattr(
            sys, 'stdout', _StdoutWithAFailingFlush(OSError(errno.ENOSPC, 'No space left on device'))
        )
        capsys.readouterr()
        code = cli_boundary._flush_stdout()
        monkeypatch.undo()

        assert code == cli_boundary.EXIT_STDOUT_FAILED
        reported = _reported_lines(capsys.readouterr().err)
        assert len(reported) == 1
        assert 'No space left on device' in reported[0]
        assert 'closed the output pipe' not in reported[0]

    def test_an_in_band_stdout_error_is_handled_by_the_same_pair_of_messages(self, capsys):
        """``_handle_stdout_error`` serves the IN-BAND frame with the same message shape.

        A full disk raised by ``print`` mid-run and one found by a later flush
        are the same operator situation, so they get one handler, one message
        and one code — a reader comparing the two should see the failure site
        differ and nothing else.
        """
        code = cli_boundary._handle_stdout_error(OSError(errno.ENOSPC, 'No space left on device'))

        assert code == cli_boundary.EXIT_STDOUT_FAILED
        reported = _reported_lines(capsys.readouterr().err)
        assert len(reported) == 1
        assert 'No space left on device' in reported[0]
        assert 'closed the output pipe' not in reported[0]

    def test_a_stream_with_no_real_descriptor_does_not_crash_the_handler(self, monkeypatch, capsys):
        """The fd guard, pinned by the doubles' ``fileno()`` raising.

        These handlers exist to keep a traceback off the screen; one that
        itself raised on a captured or replaced stdout would produce a second,
        different traceback out of the code meant to prevent the first.
        """
        monkeypatch.setattr(sys, 'stdout', _StdoutWithAFailingWrite(OSError(errno.EIO, 'io'), min_length=1))
        capsys.readouterr()
        code = cli_boundary._handle_broken_pipe()
        monkeypatch.undo()

        assert code == cli_boundary.EXIT_STDOUT_FAILED
        assert 'closed the output pipe' in capsys.readouterr().err

    def test_only_the_handler_takes_over_fd_one_the_reporter_leaves_it_alone(
        self, monkeypatch, capsys
    ):
        """The layering, asserted from BOTH sides in one place.

        ``quiet_close`` is the discriminator: ``False`` demands the wrapper's
        close still raise (nobody touched fd 1), ``True`` demands it not
        (the handler redirected it). The pair is what stops a future
        "simplification" collapsing the report-only and fd-owning forms into
        one function.
        """
        with _closed_pipe_stdout(monkeypatch, quiet_close=False):
            print('short')
            assert cli_boundary.report_broken_pipe() == cli_boundary.EXIT_STDOUT_FAILED
        monkeypatch.undo()
        capsys.readouterr()
        cli_boundary.reset_stdout_failure_state()

        with _closed_pipe_stdout(monkeypatch, quiet_close=True):
            print('short')
            assert cli_boundary._handle_broken_pipe() == cli_boundary.EXIT_STDOUT_FAILED
        monkeypatch.undo()
        assert 'closed the output pipe' in capsys.readouterr().err


class TestArgparseHelpCannotVanishDownAFailedStdout:
    """``--help | head`` must not exit 0 with its output discarded.

    ``argparse`` writes its messages inside a suppressed ``except OSError``, so
    a stock ``ArgumentParser`` swallows both the ``BrokenPipeError`` from a
    closed reader and the ``ENOSPC`` from a full disk, then leaves via
    ``SystemExit(0)`` — a success status for a run whose entire output went
    nowhere. :class:`~shared.cli_boundary.LoudArgumentParser` re-raises instead.

    Only reachable when stdout is UNBUFFERED or line-buffered
    (``PYTHONUNBUFFERED=1``, ``python -u``, a tty), where the write hits fd 1
    during ``parse_args``. Block-buffered, the help text is still in the buffer
    at that point and the failure surfaces later at :func:`run_cli`'s explicit
    flush instead — which is why both regimes are tested, here and in the
    subprocess suite below.

    The CONTROLS carry as much weight as the headline. The override must keep
    EMITTING the full help text, and must leave argparse's error path,
    formatting and exit codes untouched — a "fix" that merely stopped
    swallowing, or that routed every ``SystemExit`` through the pipe handler,
    would satisfy the headline and break the ordinary contract.
    """

    @staticmethod
    def _parser() -> cli_boundary.LoudArgumentParser:
        parser = cli_boundary.LoudArgumentParser(prog='synthetic')
        parser.add_argument('--alpha', help='the first flag')
        parser.add_argument('--beta', action='store_true', help='the second flag')
        return parser

    def test_help_into_a_closed_pipe_raises_rather_than_exiting_zero(self, monkeypatch):
        """LINE-buffered, so the write reaches fd 1 inside ``parse_args``.

        ``quiet_close=False``: ``parse_args`` is not the process boundary and
        must not take over the caller's fd — it re-raises and lets
        :func:`run_cli` decide. ``SystemExit(0)`` never being reached IS the
        assertion.
        """
        with _closed_pipe_stdout(monkeypatch, buffering=1, quiet_close=False):
            with pytest.raises(BrokenPipeError):
                self._parser().parse_args(['--help'])
        monkeypatch.undo()

    def test_help_onto_a_full_disk_escapes_as_a_plain_oserror(self, monkeypatch):
        """The non-pipe half: a closed pipe can only ever produce EPIPE.

        ``/dev/full``'s ``ENOSPC`` is what separates "the pipe handler catches
        it" from "any stdout failure is handled", and it must escape as a plain
        ``OSError`` so :func:`run_cli`'s second arm can see it.
        """
        monkeypatch.setattr(
            sys,
            'stdout',
            _StdoutWithAFailingWrite(
                OSError(errno.ENOSPC, 'No space left on device'), min_length=1
            ),
        )
        with pytest.raises(OSError) as excinfo:
            self._parser().parse_args(['--help'])
        monkeypatch.undo()

        assert not isinstance(excinfo.value, BrokenPipeError)
        assert excinfo.value.errno == errno.ENOSPC

    def test_help_on_a_healthy_stdout_still_prints_everything_and_exits_zero(self, capsys):
        """CONTROL: pins the override to keep EMITTING, in full, not merely to re-raise."""
        with pytest.raises(SystemExit) as excinfo:
            self._parser().parse_args(['--help'])

        assert excinfo.value.code == 0
        out = capsys.readouterr().out
        assert 'usage:' in out
        # The whole document, not a truncated first line — format_help() output
        # routed through the override unchanged.
        assert '--alpha' in out and '--beta' in out

    def test_an_unrecognized_flag_still_reports_to_stderr_and_exits_two(self, capsys):
        """CONTROL: the override touches stdout routing only.

        argparse's error path writes to STDERR and exits 2. Both must survive
        untouched, or the class has replaced a narrow defect with a wide one.
        """
        with pytest.raises(SystemExit) as excinfo:
            self._parser().parse_args(['--definitely-not-a-flag'])

        assert excinfo.value.code == 2
        captured = capsys.readouterr()
        assert 'unrecognized' in captured.err
        assert 'unrecognized' not in captured.out
