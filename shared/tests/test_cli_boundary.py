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

import errno
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
