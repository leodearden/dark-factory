"""The bind-once launch session id: its SessionRecord contract and the hooks that stamp it."""

from __future__ import annotations

from pathlib import Path

import pytest  # pyright: ignore[reportMissingImports]
from _process_tree_fakes import claude_process_tree

from orchestrator import session_hooks as sh
from orchestrator import session_registry as sr

_DF_CWD = '/home/leo/src/dark-factory'


@pytest.fixture(autouse=True)
def _isolate_fleet_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))


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


def _owner_env(slug: str, ppid: int) -> dict[str, str]:
    return {'CLAUDE_SPAWN_SESSION_ID': slug, 'CLAUDE_SPAWN_OWNER_PPID': str(ppid)}


def _hook_input(session_id: str, source: str | None = None) -> dict[str, str]:
    hook_input = {'session_id': session_id, 'cwd': _DF_CWD}
    if source is not None:
        hook_input['source'] = source
    return hook_input


class TestHooksStampLaunchSessionId:
    def test_first_session_start_bind_stamps_the_launch_id(self, tmp_path: Path) -> None:
        slug = 'session-cockpit-6069001'
        sr.write_record(sr.SessionRecord(session_slug=slug, status=sr.Status.LAUNCHING), root=tmp_path)

        sh.run_session_start(
            _hook_input('uuid-launch'),
            _owner_env(slug, 6069501),
            root=tmp_path,
            process_tree=claude_process_tree(6069502, 6069501),
        )

        record = sr.read_record(slug, root=tmp_path)
        assert record.claude_session_id == 'uuid-launch'
        assert record.claude_launch_session_id == 'uuid-launch'

    @pytest.mark.parametrize('source', ['clear', 'compact'])
    def test_owner_remint_rebinds_the_current_id_and_keeps_the_launch_id(
        self, tmp_path: Path, source: str
    ) -> None:
        slug = 'session-cockpit-6069002'
        env = _owner_env(slug, 6069511)
        tree = claude_process_tree(6069512, 6069511)
        sr.write_record(sr.SessionRecord(session_slug=slug, status=sr.Status.LAUNCHING), root=tmp_path)
        sh.run_session_start(_hook_input('uuid-launch'), env, root=tmp_path, process_tree=tree)

        sh.run_session_start(_hook_input('uuid-after', source), env, root=tmp_path, process_tree=tree)

        record = sr.read_record(slug, root=tmp_path)
        assert record.claude_session_id == 'uuid-after'
        assert record.claude_launch_session_id == 'uuid-launch'
        assert len(list(sr.sessions_dir(root=tmp_path).iterdir())) == 1

    def test_first_refresh_path_bind_stamps_the_launch_id(self, tmp_path: Path) -> None:
        slug = 'session-cockpit-6069003'
        sr.write_record(sr.SessionRecord(session_slug=slug, status=sr.Status.RUNNING), root=tmp_path)

        sh.run_stop(
            _hook_input('uuid-launch'),
            _owner_env(slug, 6069521),
            root=tmp_path,
            process_tree=claude_process_tree(6069522, 6069521),
        )

        assert sr.read_record(slug, root=tmp_path).claude_launch_session_id == 'uuid-launch'

    def test_rebind_of_a_record_bound_before_the_field_existed_does_not_backfill(
        self, tmp_path: Path
    ) -> None:
        slug = 'session-cockpit-6069004'
        sr.write_record(
            sr.SessionRecord(
                session_slug=slug,
                status=sr.Status.RUNNING,
                claude_session_id='uuid-before',
                claude_owner_pid=424242,
            ),
            root=tmp_path,
        )

        sh.run_session_start(
            _hook_input('uuid-after', 'clear'),
            {'CLAUDE_SPAWN_SESSION_ID': slug},
            root=tmp_path,
            process_tree=claude_process_tree(424242),
        )

        record = sr.read_record(slug, root=tmp_path)
        assert record.claude_session_id == 'uuid-after'
        assert record.claude_launch_session_id is None

    def test_bind_after_a_junk_reset_binding_keeps_the_launch_id(self, tmp_path: Path) -> None:
        slug = 'session-cockpit-6069005'
        sr.write_record(
            sr.SessionRecord(
                session_slug=slug,
                status=sr.Status.RUNNING,
                claude_launch_session_id='uuid-launch',
            ),
            root=tmp_path,
        )

        sh.run_stop(
            _hook_input('uuid-other'),
            _owner_env(slug, 6069541),
            root=tmp_path,
            process_tree=claude_process_tree(6069542, 6069541),
        )

        record = sr.read_record(slug, root=tmp_path)
        assert record.claude_session_id == 'uuid-other'
        assert record.claude_launch_session_id == 'uuid-launch'
