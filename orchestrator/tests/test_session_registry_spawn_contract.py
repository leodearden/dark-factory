"""The registry half of the /spawn contract: the result-handback outcome vocabulary and the self-exit input."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from orchestrator import session_registry as sr

_UNSET = object()


class TestResultOutcomeVocabulary:
    @pytest.mark.parametrize(('text', 'expected'), [
        ('outcome: done\nchanged: x\n', 'done'),
        ('---\noutcome: handed-off\nchanged: task/5140\naction_needed: poll\n---\nprose', 'handed-off'),
        ('outcome: blocked', 'blocked'),
        ('outcome: abandoned\n', 'abandoned'),
        ('outcome: in-progress\n', 'in-progress'),
        ('---\noutcome: in-progress\nchanged: half of it\n---\nprose', 'in-progress'),
        ('outcome: maybe\n', None),
        ('', None),
        ('I finished the work.\n', None),
        ('---\nchanged: x\n---\noutcome: done\n', None),
    ])
    def test_header_outcome_or_none(self, text, expected):
        assert sr.parse_result_outcome(text) == expected

    def test_parser_returns_the_typed_outcome(self):
        assert sr.parse_result_outcome('outcome: in-progress\n') is sr.ResultOutcome.IN_PROGRESS

    def test_outcomes_are_spelled_in_trailer_order(self):
        assert sr.RESULT_OUTCOMES == ('done', 'blocked', 'abandoned', 'handed-off', 'in-progress')

    def test_owning_outcomes_are_terminal_and_only_the_checkpoint_is_not(self):
        assert sr.OWNING_OUTCOMES <= sr.TERMINAL_RESULT_OUTCOMES
        assert frozenset(sr.RESULT_OUTCOMES) - {sr.ResultOutcome.IN_PROGRESS} == sr.TERMINAL_RESULT_OUTCOMES
        assert {'done', 'handed-off'} == sr.OWNING_OUTCOMES


def _launch(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], fleet_root: Path, self_exit: object,
) -> tuple[str, sr.SessionRecord]:
    """Run the ``launching`` verb as spawn-claude.sh does; return the printed record dir and the record."""
    monkeypatch.delenv('CLAUDE_SPAWN_SELF_EXIT', raising=False)
    launcher_pid = os.getpid()
    env = {
        'CLAUDE_FLEET_ROOT': str(fleet_root),
        'CLAUDE_SPAWN_ROLE': 'prd',
        'CLAUDE_SPAWN_PROJECT': 'df',
        'CLAUDE_SPAWN_TASK_ID': '7179',
        'CLAUDE_SPAWN_TITLE': 'prd:df#7179 S3',
        'CLAUDE_SPAWN_PROMPT': 'Read /tmp/brief.md and follow it as your brief.',
        'CLAUDE_SPAWN_CWD': str(fleet_root),
        'CLAUDE_SPAWN_LAUNCHER_PID': str(launcher_pid),
    }
    if self_exit is not _UNSET:
        env['CLAUDE_SPAWN_SELF_EXIT'] = str(self_exit)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert sr.main(['launching']) == 0
    record_dir = capsys.readouterr().out.strip()
    slug = sr.build_session_slug('prd', 'df', '7179', launcher_pid)
    return record_dir, sr.read_record(slug, root=fleet_root)


class TestSelfExitInput:
    @pytest.mark.parametrize(('value', 'expected'), [
        ('1', True), ('0', False), ('true', False), ('', False), (_UNSET, False),
    ])
    def test_only_the_exact_value_one_turns_self_exit_on(self, monkeypatch, capsys, tmp_path, value, expected):
        _, record = _launch(monkeypatch, capsys, tmp_path, value)

        assert record.self_exit is expected

    def test_self_exit_survives_the_exit_read_modify_write(self, monkeypatch, capsys, tmp_path):
        record_dir, launched = _launch(monkeypatch, capsys, tmp_path, '1')

        assert sr.main(['exit', '--record', record_dir, '--code', '0']) == 0

        exited = sr.read_record(launched.session_slug, root=tmp_path)
        assert exited.status == sr.Status.EXITED
        assert exited.self_exit is True


class TestSelfExitField:
    def test_defaults_off(self):
        assert sr.SessionRecord(session_slug='s', status=sr.Status.LAUNCHING).self_exit is False

    def test_a_record_written_before_the_field_existed_reads_off(self):
        assert sr.SessionRecord.from_dict({'session_slug': 's', 'status': 'launching'}).self_exit is False

    @pytest.mark.parametrize(('stored', 'expected'), [
        ('true', False), (1, False), ('yes', False), (None, False), (True, True),
    ])
    def test_only_a_json_true_reads_on(self, stored, expected):
        data = sr.SessionRecord(session_slug='s', status=sr.Status.EXITED).to_dict()
        data['self_exit'] = stored

        assert sr.SessionRecord.from_dict(data).self_exit is expected
