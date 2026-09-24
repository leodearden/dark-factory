"""Process-boundary WIRING of the ``fused-memory/scripts`` CLIs migrated onto ``run_cli``.

The helper's behaviour (both buffering regimes, both stdout failure kinds, the
``2>&1 | head`` shape) is owned by ``shared/tests/test_cli_boundary.py``. This
file pins only that each script in :data:`_MIGRATED_SCRIPTS` is WIRED to it: its
parser is a ``LoudArgumentParser`` and its ``__main__`` guard calls
``shared.cli_boundary.run_cli``. The exit status a closed stdout produces is
decided at interpreter finalization, so every test runs the script in a real
child process and observes only argv, exit status and streams. The one
exception reads a script's own ``EXIT_RUN_FAILED``: that is the agreement pin
``shared.cli_boundary.EXIT_STDOUT_FAILED``'s docstring asks each consumer with
its own exit-code table for.

Later migration batches append to :data:`_MIGRATED_SCRIPTS`, and to
:data:`_STORE_FREE_RUNS` for an ordinary run that needs no store, rather than
growing a ``_spawn`` of their own.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from _fm_helpers import load_script_module
from shared.cli_boundary import EXIT_STDOUT_FAILED

_SCRIPTS_DIR = Path(__file__).parent.parent / 'scripts'
_CORPUS_FIXTURE_ARCHIVE = Path(__file__).parent / 'fixtures' / 'transcript_corpus'
_STAMP = '20260101T000000Z'

_MIGRATED_SCRIPTS = [
    ('memory_eval_retrieval_probe.py', '--derive-registry'),
    ('memory_eval_transcript_corpus.py', '--archive-root'),
    ('audit_duplicate_memories.py', '--ann-threshold'),
]
"""``(script, a flag its help must show)`` for every migrated script."""

_MIGRATED = pytest.mark.parametrize(
    ('script', 'help_flag'),
    _MIGRATED_SCRIPTS,
    ids=[Path(script).stem for script, _ in _MIGRATED_SCRIPTS],
)

def _probe_derive_registry_argv(out_root: Path) -> list[str]:
    """Prints ~67 KB of registry candidates and writes no artifact."""
    return ['--derive-registry']


def _corpus_fixture_run_argv(out_root: Path) -> list[str]:
    """Writes three artifacts under *out_root*, then prints a short report."""
    return [
        '--archive-root', str(_CORPUS_FIXTURE_ARCHIVE),
        '--out-root', str(out_root), '--stamp', _STAMP,
    ]


_STORE_FREE_RUNS = pytest.mark.parametrize(
    ('script', 'argv_for'),
    [
        pytest.param(
            'memory_eval_retrieval_probe.py', _probe_derive_registry_argv,
            id='memory_eval_retrieval_probe',
        ),
        pytest.param(
            'memory_eval_transcript_corpus.py', _corpus_fixture_run_argv,
            id='memory_eval_transcript_corpus',
        ),
    ],
)
"""An ordinary run per script that prints a report without opening a store,
as ``(script, argv built from an artifact root)``."""

_BUFFERING = pytest.mark.parametrize(
    'unbuffered', [False, True], ids=['block-buffered', 'unbuffered'],
)
"""The axis a closed stdout's failure turns on: in-band when unbuffered, at a
later flush when block-buffered."""


def _spawn(script: str, *argv: str, closed_stdout: bool, unbuffered: bool) -> tuple[int, str, str]:
    """Run *script* in a child process; return ``(exit status, stdout, stderr)``.

    With *closed_stdout*, fd 1 is a pipe whose read end is closed BEFORE the
    spawn: left open, the child's write would land in the kernel's pipe buffer
    and never fail. *unbuffered* is pinned rather than inherited, because
    ``PYTHONUNBUFFERED`` decides where the write fails and an inherited value
    tests whichever regime the ambient environment supplies.
    """
    env = {**os.environ}
    if unbuffered:
        env['PYTHONUNBUFFERED'] = '1'
    else:
        env.pop('PYTHONUNBUFFERED', None)
    cmd = [sys.executable, str(_SCRIPTS_DIR / script), *argv]

    if closed_stdout:
        read_fd, write_fd = os.pipe()
        os.close(read_fd)
        try:
            proc = subprocess.Popen(cmd, stdout=write_fd, stderr=subprocess.PIPE, env=env)
        finally:
            os.close(write_fd)
    else:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)

    try:
        out, err = proc.communicate(timeout=120)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        raise
    return (
        proc.returncode,
        (out or b'').decode('utf-8', 'replace'),
        (err or b'').decode('utf-8', 'replace'),
    )


def _error_lines(err: str) -> list[str]:
    return [line for line in err.splitlines() if line.startswith('error: ')]


def _assert_one_clean_stdout_failure(code: int, err: str) -> str:
    """The boundary's contract: its status, no shutdown noise, one error line."""
    assert code == EXIT_STDOUT_FAILED, err
    assert 'Traceback' not in err
    assert 'Exception ignored' not in err
    errors = _error_lines(err)
    assert len(errors) == 1, err
    return errors[0]


@_MIGRATED
@_BUFFERING
class TestHelp:
    def test_into_a_closed_pipe_is_one_clean_stdout_failure(self, script, help_flag, unbuffered):
        code, _, err = _spawn(script, '--help', closed_stdout=True, unbuffered=unbuffered)

        error = _assert_one_clean_stdout_failure(code, err)
        assert 'closed the output pipe' in error

    def test_on_a_healthy_stdout_exits_zero_with_the_full_usage(
        self, script, help_flag, unbuffered,
    ):
        code, out, _ = _spawn(script, '--help', closed_stdout=False, unbuffered=unbuffered)

        assert code == 0
        assert 'usage:' in out
        assert help_flag in out


@_MIGRATED
@_BUFFERING
@pytest.mark.parametrize('closed_stdout', [False, True], ids=['healthy-stdout', 'closed-stdout'])
def test_an_unrecognized_flag_still_exits_two(script, help_flag, unbuffered, closed_stdout):
    """Asserts ``usage:`` rather than ``unrecognized``: the audit's required
    ``--project-id`` makes argparse report the missing argument first."""
    code, _, err = _spawn(
        script, '--definitely-not-a-flag', closed_stdout=closed_stdout, unbuffered=unbuffered,
    )

    assert code == 2
    assert 'usage:' in err


@_STORE_FREE_RUNS
@_BUFFERING
def test_an_ordinary_run_into_a_closed_pipe_is_one_clean_stdout_failure(
    script, argv_for, unbuffered, tmp_path,
):
    code, _, err = _spawn(
        script, *argv_for(tmp_path), closed_stdout=True, unbuffered=unbuffered,
    )

    _assert_one_clean_stdout_failure(code, err)


@_BUFFERING
class TestTheStdoutFailureLineNamesOnlyWhatWasWritten:
    """A stdout failure AFTER the artifacts landed names them; a run that
    wrote nothing claims nothing."""

    def test_the_corpus_names_its_report_and_agrees_on_the_exit_status(
        self, unbuffered, tmp_path,
    ):
        corpus = load_script_module(
            _SCRIPTS_DIR / 'memory_eval_transcript_corpus.py',
            mod_name='memory_eval_transcript_corpus',
        )
        report_path = tmp_path / corpus.EVAL_ID / f'report-{_STAMP}.txt'

        code, _, err = _spawn(
            'memory_eval_transcript_corpus.py', *_corpus_fixture_run_argv(tmp_path),
            closed_stdout=True, unbuffered=unbuffered,
        )

        assert code == corpus.EXIT_RUN_FAILED
        error = _assert_one_clean_stdout_failure(code, err)
        assert 'closed the output pipe' in error
        assert error.endswith(f'(the report was written to {report_path})')
        assert report_path.exists()

    def test_the_probes_derive_registry_run_claims_no_artifact(self, unbuffered, tmp_path):
        code, _, err = _spawn(
            'memory_eval_retrieval_probe.py', *_probe_derive_registry_argv(tmp_path),
            closed_stdout=True, unbuffered=unbuffered,
        )

        error = _assert_one_clean_stdout_failure(code, err)
        assert '(' not in error
        assert 'written to' not in error
