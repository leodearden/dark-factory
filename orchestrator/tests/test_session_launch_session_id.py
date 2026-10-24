"""The bind-once launch session id: its SessionRecord contract and the hooks that stamp it."""

from __future__ import annotations

import pytest  # pyright: ignore[reportMissingImports]

from orchestrator import session_registry as sr


class TestLaunchSessionIdField:
    def test_defaults_to_none(self) -> None:
        record = sr.SessionRecord(session_slug='s', status=sr.Status.LAUNCHING)

        assert record.claude_launch_session_id is None

    def test_body_written_before_the_field_existed_reads_none(self) -> None:
        body = {'session_slug': 's', 'status': 'running', 'claude_session_id': 'uuid-a'}

        assert sr.SessionRecord.from_dict(body).claude_launch_session_id is None

    def test_round_trips_losslessly_under_its_wire_key(self) -> None:
        record = sr.SessionRecord(
            session_slug='s',
            status=sr.Status.RUNNING,
            claude_session_id='uuid-after-clear',
            claude_launch_session_id='uuid-launch',
        )

        assert record.to_dict()['claude_launch_session_id'] == 'uuid-launch'
        assert sr.SessionRecord.from_dict(record.to_dict()) == record
        assert sr.SessionRecord.from_json(record.to_json()) == record

    @pytest.mark.parametrize('junk', [123, True, '   '])
    def test_junk_body_value_reads_none(self, junk: object) -> None:
        body = {'session_slug': 's', 'status': 'running', 'claude_launch_session_id': junk}

        assert sr.SessionRecord.from_dict(body).claude_launch_session_id is None

    def test_body_value_is_stripped(self) -> None:
        body = {'session_slug': 's', 'status': 'running', 'claude_launch_session_id': '  uuid-launch  '}

        assert sr.SessionRecord.from_dict(body).claude_launch_session_id == 'uuid-launch'
