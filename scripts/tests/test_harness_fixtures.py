"""Contract tests for the shared fake-command fixtures in scripts/tests/conftest.py (task 5285).

Each test requests a fixture and drives what it installs by subprocess,
asserting only what a consuming suite can observe through the fixture's
public seam.

To run just this suite:

    uv run --project shared pytest scripts/tests/test_harness_fixtures.py -q
"""
from __future__ import annotations

import os
import subprocess


def _run(fake, *args, env=None):
    return subprocess.run(
        [str(fake.path), *args],
        env=os.environ if env is None else env,
        capture_output=True, text=True, timeout=30,
    )


# ---------------------------------------------------------------------------
# fake_recorder
# ---------------------------------------------------------------------------


def test_fake_recorder_records_each_calls_argv_and_environment_in_order(fake_recorder):
    fake = fake_recorder('fake-sweep-recorder')

    _run(fake, '--apply', '--project-id', 'dark_factory')
    _run(fake, '--check', env={**os.environ, 'ONLY_IN_SECOND_CALL': 'yes'})

    assert fake.calls() == [['--apply', '--project-id', 'dark_factory'], ['--check']]
    assert 'ONLY_IN_SECOND_CALL' not in fake.envs()[0]
    assert fake.envs()[1]['ONLY_IN_SECOND_CALL'] == 'yes'


def test_fake_recorder_exits_with_its_install_time_exit_code(fake_recorder):
    default = fake_recorder('succeeds')
    failing = fake_recorder('fails', exit_code=7)

    assert _run(default).returncode == 0
    assert _run(failing).returncode == 7


def test_fake_recorder_answers_a_reply_token_and_records_both_call_kinds(fake_recorder):
    fake = fake_recorder(
        'fake-sweep-recorder',
        exit_code=3,
        replies={'--list-known-projects': {'stdout': 'dark_factory\nreify\n', 'exit_code': 0}},
    )

    listing = _run(fake, '--list-known-projects')
    sweep = _run(fake, '--apply')

    assert (listing.stdout, listing.returncode) == ('dark_factory\nreify\n', 0)
    assert (sweep.stdout, sweep.returncode) == ('', 3)
    assert fake.calls() == [['--list-known-projects'], ['--apply']]


def test_fake_recorder_runs_under_an_empty_path(fake_recorder, tmp_path):
    empty_dir = tmp_path / 'empty-path'
    empty_dir.mkdir()
    fake = fake_recorder('uv')

    result = _run(fake, 'run', '--frozen', env={'PATH': str(empty_dir)})

    assert result.returncode == 0, result.stderr
    assert fake.calls() == [['run', '--frozen']]


def test_fake_recorders_in_different_bin_dirs_keep_separate_call_logs(fake_recorder, tmp_path):
    recorder = fake_recorder('fake-sweep-recorder')
    home_uv = fake_recorder('uv', bin_dir=tmp_path / 'home' / '.local' / 'bin')

    _run(recorder, '--apply')
    _run(home_uv, 'run')
    _run(home_uv, 'sync')

    assert recorder.bin_dir == tmp_path / 'bin'
    assert home_uv.path == tmp_path / 'home' / '.local' / 'bin' / 'uv'
    assert recorder.calls() == [['--apply']]
    assert home_uv.calls() == [['run'], ['sync']]


def test_reinstalling_a_fake_recorder_starts_from_an_empty_call_log(fake_recorder):
    first = fake_recorder('fake-check-recorder')
    _run(first, '--check')

    again = fake_recorder('fake-check-recorder')

    assert again.path == first.path
    assert again.calls() == []
