"""The registry half of the /spawn contract: the result-handback outcome vocabulary and the end-of-work inputs."""

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


def _set_launch_env(
    monkeypatch: pytest.MonkeyPatch, fleet_root: Path, *, self_exit: object = _UNSET, t3_end: object = _UNSET,
) -> int:
    """Set the env spawn-claude.sh hands the ``launching`` verb, clearing an _UNSET input; return the launcher pid."""
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
    for key, value in (('CLAUDE_SPAWN_SELF_EXIT', self_exit), ('CLAUDE_SPAWN_T3_END', t3_end)):
        if value is _UNSET:
            monkeypatch.delenv(key, raising=False)
        else:
            env[key] = str(value)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return launcher_pid


def _launch(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fleet_root: Path,
    self_exit: object = _UNSET,
    *,
    t3_end: object = _UNSET,
) -> tuple[str, sr.SessionRecord]:
    """Run the ``launching`` verb as spawn-claude.sh does; return the printed record dir and the record."""
    launcher_pid = _set_launch_env(monkeypatch, fleet_root, self_exit=self_exit, t3_end=t3_end)
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


class TestT3EndInput:
    @pytest.mark.parametrize(('value', 'self_exit', 'expected'), [
        ('settle', _UNSET, sr.T3End.SETTLE),
        ('archive', '1', sr.T3End.ARCHIVE),
        ('', _UNSET, None),
        ('none', _UNSET, None),
        (_UNSET, _UNSET, None),
    ])
    def test_the_input_stamps_the_end_of_work_act(self, monkeypatch, capsys, tmp_path, value, self_exit, expected):
        _, record = _launch(monkeypatch, capsys, tmp_path, self_exit, t3_end=value)

        assert record.t3_end is expected

    @pytest.mark.parametrize(('self_exit', 't3_end', 'named'), [
        (_UNSET, 'Archive', ("CLAUDE_SPAWN_T3_END='Archive'",)),
        (_UNSET, 'exit', ("CLAUDE_SPAWN_T3_END='exit'",)),
        ('1', 'settle', ("CLAUDE_SPAWN_SELF_EXIT='1'", "CLAUDE_SPAWN_T3_END='settle'")),
        ('0', 'archive', ("CLAUDE_SPAWN_SELF_EXIT='0'", "CLAUDE_SPAWN_T3_END='archive'")),
        (_UNSET, 'archive', ("CLAUDE_SPAWN_SELF_EXIT=''", "CLAUDE_SPAWN_T3_END='archive'")),
    ])
    def test_an_unknown_or_contradictory_input_writes_no_record_and_names_itself(
        self, monkeypatch, capsys, caplog, tmp_path, self_exit, t3_end, named,
    ):
        _set_launch_env(monkeypatch, tmp_path, self_exit=self_exit, t3_end=t3_end)

        assert sr.main(['launching']) == 0

        assert not list(tmp_path.glob('sessions/*/record.json'))
        assert capsys.readouterr().out == ''
        assert all(name in caplog.text for name in named), caplog.text

    def test_t3_end_survives_the_exit_read_modify_write(self, monkeypatch, capsys, tmp_path):
        record_dir, launched = _launch(monkeypatch, capsys, tmp_path, '1', t3_end='archive')

        assert sr.main(['exit', '--record', record_dir, '--code', '0']) == 0

        exited = sr.read_record(launched.session_slug, root=tmp_path)
        assert exited.status == sr.Status.EXITED
        assert exited.exit_code == 0
        assert exited.t3_end is sr.T3End.ARCHIVE


class TestT3EndField:
    def test_defaults_to_no_act(self):
        assert sr.SessionRecord(session_slug='s', status=sr.Status.LAUNCHING).t3_end is None

    def test_a_record_written_before_the_field_existed_reads_none(self):
        assert sr.SessionRecord.from_dict({'session_slug': 's', 'status': 'launching'}).t3_end is None

    @pytest.mark.parametrize(('act', 'wire'), [(sr.T3End.SETTLE, 'settle'), (sr.T3End.ARCHIVE, 'archive'), (None, None)])
    def test_is_stored_as_its_wire_string(self, act, wire):
        stored = sr.SessionRecord(session_slug='s', status=sr.Status.LAUNCHING, t3_end=act).to_dict()['t3_end']

        assert stored == wire
        assert type(stored) is type(wire)

    @pytest.mark.parametrize(('stored', 'expected'), [
        ('settle', sr.T3End.SETTLE), ('archive', sr.T3End.ARCHIVE),
        ('none', None), ('Settle', None), (1, None), (True, None), ({}, None),
    ])
    def test_only_a_member_wire_string_reads_back_as_an_act(self, stored, expected):
        data = sr.SessionRecord(session_slug='s', status=sr.Status.EXITED).to_dict()
        data['t3_end'] = stored

        assert sr.SessionRecord.from_dict(data).t3_end is expected


_AT = '2026-10-23T00:00:00+00:00'


def _kill_failed_record() -> sr.SessionRecord:
    return sr.SessionRecord(
        session_slug='s',
        status=sr.Status.IDLE,
        exit_request=sr.ExitRequest(
            state=sr.ExitRequestState.KILL_FAILED, reason='ESRCH', pid=7193001, at=_AT, marker_session_id='uuid-a',
        ),
        exit_requested_at=_AT,
    )


def _stored(**body: object) -> sr.SessionRecord:
    data = sr.SessionRecord(session_slug='s', status=sr.Status.IDLE).to_dict()
    data.update(body)
    return sr.SessionRecord.from_dict(data)


class TestExitRequestField:
    def test_defaults_to_no_evaluation(self):
        record = sr.SessionRecord(session_slug='s', status=sr.Status.IDLE)

        assert record.exit_request is None
        assert record.exit_requested_at is None

    def test_a_record_written_before_the_fields_existed_reads_none(self):
        record = sr.SessionRecord.from_dict({'session_slug': 's', 'status': 'idle'})

        assert record.exit_request is None
        assert record.exit_requested_at is None

    def test_round_trips_losslessly(self):
        record = _kill_failed_record()

        assert sr.SessionRecord.from_dict(record.to_dict()) == record
        assert sr.SessionRecord.from_json(record.to_json()) == record

    def test_is_stored_as_wire_scalars(self):
        assert _kill_failed_record().to_dict()['exit_request'] == {
            'state': 'kill_failed', 'reason': 'ESRCH', 'pid': 7193001, 'at': _AT, 'marker_session_id': 'uuid-a',
        }

    def test_state_wire_values(self):
        assert {state.value for state in sr.ExitRequestState} == {'refused', 'attempted', 'kill_failed'}

    @pytest.mark.parametrize('stored', [
        'attempted', [], {'state': 'exploded', 'reason': None, 'pid': 7193001, 'at': _AT}, {},
    ])
    def test_a_junk_body_reads_none(self, stored):
        assert _stored(exit_request=stored).exit_request is None

    @pytest.mark.parametrize('pid', [True, '12', -3])
    def test_a_junk_pid_reads_none(self, pid):
        body = {'state': 'refused', 'reason': 'self-exit-off', 'pid': pid, 'at': _AT, 'marker_session_id': None}

        request = _stored(exit_request=body).exit_request

        assert request is not None
        assert request.pid is None

    @pytest.mark.parametrize('stored', [5, True])
    def test_a_non_str_requested_at_reads_none(self, stored):
        assert _stored(exit_requested_at=stored).exit_requested_at is None
