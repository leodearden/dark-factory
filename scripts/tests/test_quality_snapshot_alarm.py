"""Tests for scripts/quality_snapshot_alarm.py, the OnFailure= handler of quality-snapshot@<id>.service.

Driven through its public seams only: the escalation poster is injected, and
``journalctl`` is a fake first on PATH (the fake-binary pattern of
scripts/tests/test_install_link_heal_timer.py). The one real post goes to an
endpoint nothing listens on, the scripts/tests/test_transcript_secret_sweep.py
precedent. Real systemd and the real escalation server are never touched.
"""
from __future__ import annotations

import datetime
import json
import os
from pathlib import Path

import pytest
import quality_snapshot_alarm as alarm
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

NOW = datetime.datetime(2026, 11, 14, 3, 47, tzinfo=datetime.UTC)
UNIT = 'quality-snapshot@proj_a.service'
FAILED_ENVIRON = {
    'MONITOR_UNIT': UNIT,
    'MONITOR_SERVICE_RESULT': 'exit-code',
    'MONITOR_EXIT_CODE': 'exited',
    'MONITOR_EXIT_STATUS': '2',
    'MONITOR_INVOCATION_ID': 'abc123',
    'QUALITY_SNAPSHOT_ROOT': '/srv/proj-a',
}
JOURNAL = 'quality_snapshot_cadence: `git -C /srv/proj-a rev-parse` exited 128: fatal: not a git repository'
ARGUMENT_KEYS = {'task_id', 'agent_role', 'category', 'severity', 'summary', 'detail', 'evidence'}
ESCALATION_URL = 'http://x/mcp'

_FAKE_JOURNALCTL_SRC = '''#!/usr/bin/env python3
"""Fake `journalctl`: records its argv into $FAKE_JOURNALCTL_ARGV, prints two lines, exits $FAKE_JOURNALCTL_EXIT."""
import json
import os
import sys

with open(os.environ["FAKE_JOURNALCTL_ARGV"], "w") as f:
    json.dump(sys.argv[1:], f)
print("first journal line")
print("second journal line")
sys.exit(int(os.environ.get("FAKE_JOURNALCTL_EXIT", "0")))
'''


@pytest.fixture
def fake_journalctl(tmp_path, monkeypatch) -> Path:
    """A fake ``journalctl`` first on PATH; returns the file its argv is recorded in."""
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    fake = bin_dir / 'journalctl'
    fake.write_text(_FAKE_JOURNALCTL_SRC)
    fake.chmod(0o755)
    record = tmp_path / 'journalctl_argv.json'
    monkeypatch.setenv('PATH', f'{bin_dir}{os.pathsep}{os.environ["PATH"]}')
    monkeypatch.setenv('FAKE_JOURNALCTL_ARGV', str(record))
    return record


def _failed_arguments() -> dict:
    run = alarm.FailedRun.from_environment('proj_a', FAILED_ENVIRON)
    return alarm.escalation_arguments(run, JOURNAL, now=NOW)


class TestEscalationArguments:
    def test_a_failed_run_files_one_infra_issue_info_under_a_job_distinct_task_id(self):
        arguments = _failed_arguments()

        assert set(arguments) == ARGUMENT_KEYS
        assert arguments['task_id'] == 'quality-snapshot-proj_a'
        assert arguments['agent_role'] == 'quality-snapshot-alarm'
        assert arguments['category'] == 'infra_issue'
        assert arguments['severity'] == 'info'

    def test_the_summary_names_the_project_the_unit_the_result_and_the_exit_status(self):
        summary = _failed_arguments()['summary']

        for fact in ('proj_a', UNIT, 'exit-code', '2'):
            assert fact in summary, (fact, summary)

    def test_the_detail_carries_the_root_the_journal_and_where_to_read_rerun_and_inspect(self):
        detail = _failed_arguments()['detail']

        for fact in (
            '/srv/proj-a',
            JOURNAL,
            'abc123',
            f'journalctl --user -u {UNIT}',
            f'systemctl --user start {UNIT}',
            f'systemctl --user cat {UNIT}',
        ):
            assert fact in detail, (fact, detail)

    def test_the_detail_does_not_restate_the_wrapper_argv(self):
        assert '--project-id' not in _failed_arguments()['detail']

    def test_the_evidence_is_one_raw_observation_measured_now(self):
        (entry,) = _failed_arguments()['evidence']

        assert set(entry) == {'observation', 'measured_at', 'ref'}
        assert entry['measured_at'] == NOW.isoformat()

    def test_a_run_without_monitor_facts_or_a_root_still_names_the_unit_and_the_expected_env_file(self):
        run = alarm.FailedRun.from_environment('proj_a', {})

        assert run.unit == UNIT
        assert (run.service_result, run.exit_code, run.exit_status) == ('unknown', 'unknown', 'unknown')
        assert run.invocation_id is None
        assert run.project_root is None
        arguments = alarm.escalation_arguments(run, JOURNAL, now=NOW)
        assert 'unknown' in arguments['summary']
        assert 'root: unknown' in arguments['detail']
        assert 'dark-factory/quality-snapshot/proj_a.env' in arguments['detail']


class TestJournalTail:
    def test_it_reads_the_units_newest_lines_from_the_user_journal(self, fake_journalctl):
        text = alarm.journal_tail(UNIT)

        assert text.splitlines() == ['first journal line', 'second journal line']
        argv = json.loads(fake_journalctl.read_text())
        assert '--user' in argv
        assert argv[argv.index('-u') + 1] == UNIT
        assert argv[argv.index('-o') + 1] == 'cat'
        assert '--no-pager' in argv
        assert '-n' in argv

    def test_a_journalctl_that_fails_is_named_not_raised(self, fake_journalctl, monkeypatch):
        monkeypatch.setenv('FAKE_JOURNALCTL_EXIT', '1')

        text = alarm.journal_tail(UNIT)

        assert text.startswith('(journal unreadable:'), text
        assert 'exited 1' in text, text

    def test_a_path_without_journalctl_is_named_not_raised(self, tmp_path, monkeypatch):
        empty = tmp_path / 'empty-bin'
        empty.mkdir()
        monkeypatch.setenv('PATH', str(empty))

        text = alarm.journal_tail(UNIT)

        assert text.startswith('(journal unreadable:'), text
        assert 'FileNotFoundError' in text, text


class _Recorder:
    def __init__(self, reply: object = None, raises: Exception | None = None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self._reply = {'status': 'queued'} if reply is None else reply
        self._raises = raises

    def __call__(self, base_url: str, arguments: dict) -> object:
        self.calls.append((base_url, arguments))
        if self._raises is not None:
            raise self._raises
        return self._reply


def _main(poster=None, *, url: str = ESCALATION_URL) -> int:
    return alarm.main(
        ['--project-id', 'proj_a', '--escalation-url', url], environ=FAILED_ENVIRON, poster=poster,
    )


class TestMain:
    def test_it_posts_one_escalation_to_the_named_servers_base_and_exits_0(self, fake_journalctl):
        poster = _Recorder()

        assert _main(poster) == 0

        ((base_url, arguments),) = poster.calls
        assert base_url == 'http://x'
        assert set(arguments) == ARGUMENT_KEYS
        assert arguments['task_id'] == 'quality-snapshot-proj_a'
        assert 'first journal line' in arguments['detail']

    def test_a_poster_that_raises_exits_2_naming_the_url_and_the_exception_type(self, fake_journalctl, capsys):
        poster = _Recorder(raises=ConnectionError('refused'))

        assert _main(poster) == 2

        err = capsys.readouterr().err
        ((_, arguments),) = poster.calls
        assert arguments['summary'] in err, err
        naming = [line for line in err.splitlines() if 'http://x' in line and 'ConnectionError' in line]
        assert len(naming) == 1, err
        assert 'Traceback' not in err

    def test_a_refused_escalation_exits_2_naming_the_refusal(self, fake_journalctl, capsys):
        assert _main(_Recorder(reply={'error': 'nope'})) == 2

        assert 'nope' in capsys.readouterr().err

    def test_the_default_poster_fails_loud_against_an_unreachable_server(self, fake_journalctl, capsys):
        assert _main(url='http://127.0.0.1:1/mcp') == 2

        err = capsys.readouterr().err
        assert '127.0.0.1:1' in err, err
        assert 'Traceback' not in err
