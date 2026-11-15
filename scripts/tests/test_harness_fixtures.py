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

import pytest
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)


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


@pytest.mark.parametrize('reply', [
    {'stdout': 'dark_factory\n'},
    {'exit_code': 0},
    {'stdout': 'dark_factory\n', 'exit_code': '0'},
    {'stdout': b'dark_factory\n', 'exit_code': 0},
    {'stdout': 'dark_factory\n', 'exit_code': 0, 'stderr': ''},
    'dark_factory\n',
])
def test_fake_recorder_refuses_a_malformed_reply_at_install_naming_the_token(fake_recorder, reply):
    with pytest.raises(ValueError, match='--list-known-projects') as refused:
        fake_recorder('fake-sweep-recorder', replies={'--list-known-projects': reply})

    assert repr(reply) in str(refused.value)


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


# ---------------------------------------------------------------------------
# fake_timer_systemctl
# ---------------------------------------------------------------------------


def _systemctl(fake_timer_systemctl, *args, **extra_env):
    env = {
        **os.environ,
        'PATH': f"{fake_timer_systemctl.bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        **extra_env,
    }
    return _run(fake_timer_systemctl, *args, env=env)


def test_fake_timer_systemctl_strips_user_and_records_every_call_in_order(fake_timer_systemctl):
    _systemctl(fake_timer_systemctl, '--user', 'daemon-reload')
    _systemctl(fake_timer_systemctl, '--user', 'enable', '--now', 'x.timer')

    assert fake_timer_systemctl.calls() == [['daemon-reload'], ['enable', '--now', 'x.timer']]


def test_fake_timer_systemctl_lists_enabled_timers_unless_told_to_omit_them(fake_timer_systemctl):
    _systemctl(fake_timer_systemctl, '--user', 'enable', '--now', 'x.timer')

    listed = _systemctl(fake_timer_systemctl, '--user', 'list-timers', '--all')
    omitted = _systemctl(
        fake_timer_systemctl, '--user', 'list-timers', '--all', FAKE_SYSTEMCTL_OMIT_LIST_TIMERS='1',
    )

    *timer_lines, summary = listed.stdout.splitlines()
    assert any('x.timer' in line and 'x.service' in line for line in timer_lines), listed.stdout
    assert summary == '1 timers listed.'
    assert 'x.timer' not in omitted.stdout and 'x.service' not in omitted.stdout


def test_fake_timer_systemctl_start_succeeds_unless_told_to_fail_naming_the_unit(fake_timer_systemctl):
    plain = _systemctl(fake_timer_systemctl, '--user', 'start', 'x.service')
    no_block = _systemctl(fake_timer_systemctl, '--user', 'start', '--no-block', 'x.service')
    failed = _systemctl(
        fake_timer_systemctl, '--user', 'start', '--no-block', 'x.service',
        FAKE_SYSTEMCTL_FAIL_START='1',
    )

    assert (plain.returncode, no_block.returncode) == (0, 0)
    assert failed.returncode != 0
    assert 'Job for x.service failed.' in failed.stderr, failed.stderr
    assert fake_timer_systemctl.calls()[-1] == ['start', '--no-block', 'x.service']


def test_fake_timer_systemctl_show_answers_the_configured_working_directory(fake_timer_systemctl, tmp_path):
    result = _systemctl(
        fake_timer_systemctl, '--user', 'show', '-p', 'WorkingDirectory', '--value', 'x.service',
        FAKE_SYSTEMCTL_WORKDIR=str(tmp_path / 'checkout'),
    )

    assert result.stdout.strip() == str(tmp_path / 'checkout')


def test_fake_timer_systemctl_refuses_an_unmodelled_verb(fake_timer_systemctl):
    assert _systemctl(fake_timer_systemctl, '--user', 'restart', 'x.service').returncode != 0


def test_fake_timer_systemctl_keeps_state_across_runs_until_reset(fake_timer_systemctl):
    _systemctl(fake_timer_systemctl, '--user', 'enable', '--now', 'first.timer')
    _systemctl(fake_timer_systemctl, '--user', 'enable', '--now', 'second.timer')
    carried = _systemctl(fake_timer_systemctl, '--user', 'list-timers', '--all')

    fake_timer_systemctl.reset()

    assert 'first.timer' in carried.stdout and 'second.timer' in carried.stdout
    assert fake_timer_systemctl.calls() == []
    after_reset = _systemctl(fake_timer_systemctl, '--user', 'list-timers', '--all')
    assert after_reset.stdout.splitlines() == ['0 timers listed.']


# ---------------------------------------------------------------------------
# unit_values
# ---------------------------------------------------------------------------


def _unit(tmp_path, text):
    path = tmp_path / 'x.timer'
    path.write_text(text, encoding='utf-8')
    return path


def test_unit_values_returns_every_value_for_the_key_in_file_order(unit_values, tmp_path):
    unit = _unit(tmp_path, (
        '[Unit]\n'
        'Documentation=man:first(1)\n'
        'Description=x\n'
        'Documentation=man:second(1)\n'
        '[Timer]\n'
        'OnCalendar=daily\n'
    ))

    assert unit_values(unit, 'Unit', 'Documentation') == ['man:first(1)', 'man:second(1)']
    assert unit_values(unit, 'Timer', 'Persistent') == []
    assert unit_values(unit, 'Install', 'WantedBy') == []


def test_unit_values_does_not_read_a_comment_as_a_directive(unit_values, tmp_path):
    unit = _unit(tmp_path, (
        '[Timer]\n'
        '# Persistent=true keeps a missed run\n'
        '; OnCalendar=daily\n'
        'Unit=x.service\n'
    ))

    assert unit_values(unit, 'Timer', 'Persistent') == []
    assert unit_values(unit, 'Timer', 'OnCalendar') == []


def test_unit_values_joins_a_backslash_continued_value(unit_values, tmp_path):
    unit = _unit(tmp_path, (
        '[Service]\n'
        'ExecStart=/bin/a \\\n'
        '    --flag\n'
    ))

    assert unit_values(unit, 'Service', 'ExecStart') == ['/bin/a --flag']
