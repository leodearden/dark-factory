"""Tests for the pid -> session-record-slug pointer and its lease/decision consumers."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest  # pyright: ignore[reportMissingImports]

from orchestrator import session_hooks as sh
from orchestrator import session_registry as sr

# A pid virtually guaranteed dead on any host (the suite-wide idiom; see
# test_session_registry.py's own _DEAD_PID -- deliberately not imported).
_DEAD_PID = 2**31 - 1


@pytest.fixture(autouse=True)
def _isolate_fleet_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))


def _record(
    slug: str,
    *,
    owner_pid: int | None,
    status: sr.Status = sr.Status.RUNNING,
) -> sr.SessionRecord:
    return sr.SessionRecord(
        session_slug=slug,
        status=status,
        claude_session_id='claude-session-uuid',
        claude_owner_pid=owner_pid,
    )


class TestSessionPointerPaths:
    def test_pointer_dir_is_a_sibling_of_the_sessions_dir(self, tmp_path: Path) -> None:
        pointers = sr.session_pointers_dir(tmp_path)

        assert pointers == tmp_path / 'sessions-by-pid'
        assert not pointers.is_relative_to(sr.sessions_dir(tmp_path))

    def test_pointer_path_is_named_by_the_pid(self, tmp_path: Path) -> None:
        assert sr.session_pointer_path_for_pid(1234, root=tmp_path) == (
            sr.session_pointers_dir(tmp_path) / '1234'
        )

    def test_write_creates_the_dir_and_stores_exactly_the_slug(self, tmp_path: Path) -> None:
        assert sr.write_session_pointer(1234, 'role-proj-uuid', root=tmp_path) is True

        path = sr.session_pointer_path_for_pid(1234, root=tmp_path)
        assert path.read_text(encoding='utf-8') == 'role-proj-uuid'

    def test_a_second_write_for_the_same_pid_wins(self, tmp_path: Path) -> None:
        sr.write_session_pointer(1234, 'first-slug', root=tmp_path)
        assert sr.write_session_pointer(1234, 'second-slug', root=tmp_path) is True

        path = sr.session_pointer_path_for_pid(1234, root=tmp_path)
        assert path.read_text(encoding='utf-8') == 'second-slug'

    def test_non_positive_pids_are_refused_without_writing(self, tmp_path: Path) -> None:
        assert sr.write_session_pointer(0, 'role-proj-uuid', root=tmp_path) is False
        assert sr.write_session_pointer(-1, 'role-proj-uuid', root=tmp_path) is False
        assert not sr.session_pointers_dir(tmp_path).exists()

    def test_slugs_that_are_not_record_keys_are_refused_without_writing(
        self, tmp_path: Path
    ) -> None:
        for slug in ('', '..', 'a/b'):
            assert sr.write_session_pointer(1234, slug, root=tmp_path) is False, slug
        assert not sr.session_pointers_dir(tmp_path).exists()

    def test_an_unwritable_pointer_dir_returns_false_instead_of_raising(
        self, tmp_path: Path
    ) -> None:
        sr.session_pointers_dir(tmp_path).write_text('not a directory', encoding='utf-8')

        assert sr.write_session_pointer(1234, 'role-proj-uuid', root=tmp_path) is False
        assert sr.session_pointers_dir(tmp_path).read_text(encoding='utf-8') == 'not a directory'


def _write_pointer_raw(pid: int, content: str | bytes, root: Path) -> Path:
    path = sr.session_pointer_path_for_pid(pid, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding='utf-8')
    return path


class TestResolveSessionSlugForPid:
    _PID = 1234
    _SLUG = 'role-proj-uuid'

    def test_no_pointer_file(self, tmp_path: Path) -> None:
        assert sr.resolve_session_slug_for_pid(self._PID, root=tmp_path) is None

    @pytest.mark.parametrize('content', ['', '   \n'])
    def test_blank_pointer(self, tmp_path: Path, content: str) -> None:
        _write_pointer_raw(self._PID, content, tmp_path)

        assert sr.resolve_session_slug_for_pid(self._PID, root=tmp_path) is None

    @pytest.mark.parametrize(
        ('content', 'traversal_target'),
        [
            ('../../etc', Path('etc')),
            ('a/b', Path('fleet/sessions/a/b')),
            ('..', Path('fleet')),
        ],
    )
    def test_pointer_that_is_not_a_record_key_never_reads_past_sessions_dir(
        self, tmp_path: Path, content: str, traversal_target: Path
    ) -> None:
        root = tmp_path / 'fleet'
        sr.sessions_dir(root).mkdir(parents=True)
        sentinel = tmp_path / traversal_target / 'record.json'
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_text(
            _record('sentinel', owner_pid=self._PID).to_json(), encoding='utf-8'
        )
        _write_pointer_raw(self._PID, content, root)

        assert sr.resolve_session_slug_for_pid(self._PID, root=root) is None

    def test_dangling_pointer(self, tmp_path: Path) -> None:
        sr.write_session_pointer(self._PID, self._SLUG, root=tmp_path)

        assert sr.resolve_session_slug_for_pid(self._PID, root=tmp_path) is None

    def test_record_that_is_not_json(self, tmp_path: Path) -> None:
        path = sr.record_path_for_slug(self._SLUG, root=tmp_path)
        path.parent.mkdir(parents=True)
        path.write_text('{not json', encoding='utf-8')
        sr.write_session_pointer(self._PID, self._SLUG, root=tmp_path)

        assert sr.resolve_session_slug_for_pid(self._PID, root=tmp_path) is None

    def test_record_owned_by_a_different_pid(self, tmp_path: Path) -> None:
        sr.write_record(_record(self._SLUG, owner_pid=self._PID + 1), root=tmp_path)
        sr.write_session_pointer(self._PID, self._SLUG, root=tmp_path)

        assert sr.resolve_session_slug_for_pid(self._PID, root=tmp_path) is None

    def test_record_with_no_owner_pid_no_longer_vouches_for_the_pid(
        self, tmp_path: Path
    ) -> None:
        sr.write_record(_record(self._SLUG, owner_pid=None), root=tmp_path)
        sr.write_session_pointer(self._PID, self._SLUG, root=tmp_path)

        assert sr.resolve_session_slug_for_pid(self._PID, root=tmp_path) is None

    def test_record_owned_by_the_pid_resolves_to_its_slug(self, tmp_path: Path) -> None:
        sr.write_record(_record(self._SLUG, owner_pid=self._PID), root=tmp_path)
        sr.write_session_pointer(self._PID, self._SLUG, root=tmp_path)

        assert sr.resolve_session_slug_for_pid(self._PID, root=tmp_path) == self._SLUG

    def test_directory_in_the_pointer_files_place(self, tmp_path: Path) -> None:
        sr.session_pointer_path_for_pid(self._PID, root=tmp_path).mkdir(parents=True)

        assert sr.resolve_session_slug_for_pid(self._PID, root=tmp_path) is None

    def test_pointer_that_is_not_utf8(self, tmp_path: Path) -> None:
        sr.write_record(_record(self._SLUG, owner_pid=self._PID), root=tmp_path)
        _write_pointer_raw(self._PID, b'\xff\xfe' + self._SLUG.encode(), tmp_path)

        assert sr.resolve_session_slug_for_pid(self._PID, root=tmp_path) is None

    @pytest.mark.parametrize('pid', [0, -1])
    def test_non_positive_pid_never_touches_the_filesystem(
        self, tmp_path: Path, pid: int
    ) -> None:
        assert sr.resolve_session_slug_for_pid(pid, root=tmp_path) is None
        assert not sr.session_pointers_dir(tmp_path).exists()


class TestResolveOwnRecordSlug:
    _PID = 1234
    _SLUG = 'role-proj-uuid'

    def _link(self, root: Path) -> None:
        sr.write_record(_record(self._SLUG, owner_pid=self._PID), root=root)
        sr.write_session_pointer(self._PID, self._SLUG, root=root)

    def test_resolves_the_claude_pid_from_env(self, tmp_path: Path) -> None:
        self._link(tmp_path)

        assert sr.resolve_own_record_slug({'CLAUDE_PID': '1234'}, root=tmp_path) == self._SLUG

    def test_env_none_reads_os_environ(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._link(tmp_path)
        monkeypatch.setenv('CLAUDE_PID', '1234')

        assert sr.resolve_own_record_slug(root=tmp_path) == self._SLUG

    @pytest.mark.parametrize(
        'env',
        [{}, {'CLAUDE_PID': ''}, {'CLAUDE_PID': '   '}, {'CLAUDE_PID': '0'},
         {'CLAUDE_PID': '-1'}, {'CLAUDE_PID': 'not-a-pid'}],
    )
    def test_an_unusable_claude_pid_is_none_and_quiet(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture, env: dict[str, str]
    ) -> None:
        self._link(tmp_path)
        caplog.set_level(logging.DEBUG)

        assert sr.resolve_own_record_slug(env, root=tmp_path) is None
        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []

    def test_a_valid_pid_with_no_pointer_is_none_and_quiet(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.DEBUG)

        assert sr.resolve_own_record_slug({'CLAUDE_PID': '1234'}, root=tmp_path) is None
        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []

    def test_resolve_session_pid_keeps_its_lease_degradation_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.DEBUG)

        assert sr.resolve_session_pid({'CLAUDE_PID': ''}) == 0
        assert any(r.levelno == logging.WARNING for r in caplog.records)


class TestReapStaleSessionPointers:
    _SLUG = 'role-proj-uuid'

    def _pointer_path(self, name: str, root: Path) -> Path:
        return sr.session_pointers_dir(root) / name

    def test_a_live_pids_pointer_to_an_existing_record_is_kept(self, tmp_path: Path) -> None:
        sr.write_record(_record(self._SLUG, owner_pid=os.getpid()), root=tmp_path)
        sr.write_session_pointer(os.getpid(), self._SLUG, root=tmp_path)

        assert sr.reap_stale_session_pointers(root=tmp_path) == []
        assert sr.session_pointer_path_for_pid(os.getpid(), root=tmp_path).is_file()

    def test_a_dead_pids_pointer_is_removed_even_though_its_record_exists(
        self, tmp_path: Path
    ) -> None:
        sr.write_record(_record(self._SLUG, owner_pid=_DEAD_PID), root=tmp_path)
        sr.write_session_pointer(_DEAD_PID, self._SLUG, root=tmp_path)
        pointer = sr.session_pointer_path_for_pid(_DEAD_PID, root=tmp_path)

        assert sr.reap_stale_session_pointers(root=tmp_path) == [pointer]
        assert not pointer.exists()

    def test_a_live_pids_pointer_to_a_reaped_record_is_removed(self, tmp_path: Path) -> None:
        sr.write_session_pointer(os.getpid(), self._SLUG, root=tmp_path)
        pointer = sr.session_pointer_path_for_pid(os.getpid(), root=tmp_path)

        assert sr.reap_stale_session_pointers(root=tmp_path) == [pointer]
        assert not pointer.exists()

    @pytest.mark.parametrize('content', ['', '  \n', '../escape', 'a/b', '..'])
    def test_a_pointer_whose_content_is_not_a_record_key_is_removed(
        self, tmp_path: Path, content: str
    ) -> None:
        pointer = _write_pointer_raw(os.getpid(), content, tmp_path)

        assert sr.reap_stale_session_pointers(root=tmp_path) == [pointer]
        assert not pointer.exists()

    @pytest.mark.parametrize('name', ['not-a-pid', '0', '-5', '12.5'])
    def test_a_file_not_named_by_a_positive_pid_is_removed(
        self, tmp_path: Path, name: str
    ) -> None:
        sr.write_record(_record(self._SLUG, owner_pid=os.getpid()), root=tmp_path)
        pointer = self._pointer_path(name, tmp_path)
        pointer.parent.mkdir(parents=True)
        pointer.write_text(self._SLUG, encoding='utf-8')

        assert sr.reap_stale_session_pointers(root=tmp_path) == [pointer]
        assert not pointer.exists()

    def test_an_absent_pointer_dir_is_an_empty_pass(self, tmp_path: Path) -> None:
        assert sr.reap_stale_session_pointers(root=tmp_path) == []

    def test_a_corrupt_record_is_never_parsed_so_its_pointer_is_kept(
        self, tmp_path: Path
    ) -> None:
        record_path = sr.record_path_for_slug(self._SLUG, root=tmp_path)
        record_path.parent.mkdir(parents=True)
        record_path.write_text('{not json', encoding='utf-8')
        sr.write_session_pointer(os.getpid(), self._SLUG, root=tmp_path)

        assert sr.reap_stale_session_pointers(root=tmp_path) == []
        assert sr.session_pointer_path_for_pid(os.getpid(), root=tmp_path).is_file()

    def test_an_entry_that_cannot_be_removed_is_skipped_and_the_pass_continues(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        stuck = self._pointer_path(str(_DEAD_PID), tmp_path)
        stuck.mkdir(parents=True)
        (stuck / 'occupant').write_text('x', encoding='utf-8')
        stale = self._pointer_path('not-a-pid', tmp_path)
        stale.write_text(self._SLUG, encoding='utf-8')
        caplog.set_level(logging.DEBUG)

        assert sr.reap_stale_session_pointers(root=tmp_path) == [stale]
        assert stuck.is_dir()
        assert not stale.exists()
        assert any(str(_DEAD_PID) in r.getMessage() for r in caplog.records)


def _launching_env(root: Path) -> dict[str, str]:
    return {
        'CLAUDE_FLEET_ROOT': str(root),
        'CLAUDE_SPAWN_ROLE': 'unblock',
        'CLAUDE_SPAWN_PROJECT': 'df',
        'CLAUDE_SPAWN_TASK_ID': '2085',
        'CLAUDE_SPAWN_ESCALATION_ID': 'esc-9',
        'CLAUDE_SPAWN_TITLE': 'unblock:df#2085 routing-mechanism',
        'CLAUDE_SPAWN_PROMPT': '/unblock 2085',
        'CLAUDE_SPAWN_CWD': '/home/leo/src/dark-factory',
        'CLAUDE_SPAWN_LAUNCHER_PID': '4242',
    }


class TestPointerSweepDrivers:
    def test_the_reap_verb_removes_a_dead_pids_pointer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
        sr.write_session_pointer(_DEAD_PID, 'role-proj-uuid', root=tmp_path)

        assert sr.main(['reap']) == 0
        assert not sr.session_pointer_path_for_pid(_DEAD_PID, root=tmp_path).exists()

    def test_every_spawn_removes_a_dead_pids_pointer_and_still_prints_only_its_record_dir(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        for key, value in _launching_env(tmp_path).items():
            monkeypatch.setenv(key, value)
        sr.write_session_pointer(_DEAD_PID, 'role-proj-uuid', root=tmp_path)

        assert sr.main(['launching']) == 0

        slug = sr.build_session_slug('unblock', 'df', '2085', 4242)
        expected_dir = sr.record_path_for_slug(slug, root=tmp_path).parent
        assert capsys.readouterr().out == f'{expected_dir}\n'
        assert not sr.session_pointer_path_for_pid(_DEAD_PID, root=tmp_path).exists()


_CWD = '/home/leo/src/dark-factory'


def _pointer_files(root: Path) -> list[Path]:
    pointers = sr.session_pointers_dir(root)
    return sorted(pointers.iterdir()) if pointers.is_dir() else []


class TestSessionStartStampsPointer:
    _OWNER = 3_215_501

    def test_a_hand_launched_session_is_findable_from_its_claude_pid(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sh, '_owning_claude_pid', lambda: self._OWNER)

        record = sh.run_session_start({'session_id': 'sess-hl', 'cwd': _CWD}, {}, root=tmp_path)

        assert sr.resolve_session_slug_for_pid(self._OWNER, root=tmp_path) == record.session_slug

    def test_the_pointer_is_keyed_on_the_claude_pid_not_the_launcher_pid(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sh, '_owning_claude_pid', lambda: self._OWNER)

        record = sh.run_session_start({'session_id': 'sess-hl', 'cwd': _CWD}, {}, root=tmp_path)

        assert record.launcher_pid != self._OWNER
        assert sr.resolve_session_slug_for_pid(record.launcher_pid, root=tmp_path) is None

    def test_an_unresolvable_claude_pid_writes_the_record_but_no_pointer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sh, '_owning_claude_pid', lambda: None)

        record = sh.run_session_start({'session_id': 'sess-hl', 'cwd': _CWD}, {}, root=tmp_path)

        assert sr.read_record(record.session_slug, root=tmp_path) == record
        assert _pointer_files(tmp_path) == []

    def test_a_session_that_never_binds_gets_no_pointer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sh, '_owning_claude_pid', lambda: self._OWNER)

        record = sh.run_session_start({'session_id': '', 'cwd': _CWD}, {}, root=tmp_path)

        assert record.claude_session_id is None
        assert _pointer_files(tmp_path) == []

    def test_an_unwritable_pointer_dir_never_breaks_the_hook(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sh, '_owning_claude_pid', lambda: self._OWNER)
        sr.session_pointers_dir(tmp_path).write_text('not a directory', encoding='utf-8')

        record = sh.run_session_start({'session_id': 'sess-hl', 'cwd': _CWD}, {}, root=tmp_path)

        assert sr.read_record(record.session_slug, root=tmp_path) == record


class TestRefreshEventsStampPointer:
    _OWNER = 3_215_601
    _HOOK_INPUT = {'session_id': 'sess-hl', 'cwd': _CWD, 'message': 'Proceed?'}

    def _bound_hand_launched_record(self, root: Path) -> str:
        slug = sh.hook_session_slug(self._HOOK_INPUT, {}, root=root)
        sr.write_record(
            sr.SessionRecord(
                session_slug=slug,
                status=sr.Status.RUNNING,
                claude_session_id='sess-hl',
                claude_owner_pid=self._OWNER,
            ),
            root=root,
        )
        return slug

    @pytest.mark.parametrize('handler', [sh.run_notification, sh.run_stop])
    def test_a_session_already_running_at_rollout_is_linked_by_its_next_event(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        handler: Callable[..., str],
    ) -> None:
        monkeypatch.setattr(sh, '_owning_claude_pid', lambda: self._OWNER)
        linked_root = tmp_path / 'linked'
        slug = self._bound_hand_launched_record(linked_root)
        unwritable_root = tmp_path / 'unwritable'
        self._bound_hand_launched_record(unwritable_root)
        sr.session_pointers_dir(unwritable_root).write_text('x', encoding='utf-8')

        retitle = handler(self._HOOK_INPUT, {}, root=linked_root)

        assert sr.resolve_session_slug_for_pid(self._OWNER, root=linked_root) == slug
        assert retitle == handler(self._HOOK_INPUT, {}, root=unwritable_root)

    def test_a_withheld_launch_window_event_stamps_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sh, '_owning_claude_pid', lambda: self._OWNER)
        slug = 'session-cockpit-4237001'
        sr.write_record(
            sr.SessionRecord(
                session_slug=slug,
                status=sr.Status.LAUNCHING,
                launcher_pid=4237001,
                start_ts=datetime.now(UTC).isoformat(),
            ),
            root=tmp_path,
        )

        sh.run_notification(
            self._HOOK_INPUT, {'CLAUDE_SPAWN_SESSION_ID': slug}, root=tmp_path
        )

        assert sr.read_record(slug, root=tmp_path).status == sr.Status.LAUNCHING
        assert not sr.session_pointer_path_for_pid(self._OWNER, root=tmp_path).exists()

    def test_a_nested_claude_points_at_its_forked_record_never_its_spawners(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spawner_pid, nested_pid = 4_237_100, 4_237_200
        spawner_slug = 'session-cockpit-4237100'
        sr.write_record(
            sr.SessionRecord(
                session_slug=spawner_slug,
                status=sr.Status.RUNNING,
                claude_session_id='uuid-spawner',
                claude_owner_pid=spawner_pid,
            ),
            root=tmp_path,
        )
        sr.write_session_pointer(spawner_pid, spawner_slug, root=tmp_path)
        monkeypatch.setattr(sh, '_owning_claude_pid', lambda: nested_pid)

        forked = sh.run_session_start(
            {'session_id': 'uuid-nested', 'cwd': _CWD},
            {'CLAUDE_SPAWN_SESSION_ID': spawner_slug},
            root=tmp_path,
        )

        assert forked.session_slug != spawner_slug
        assert sr.resolve_session_slug_for_pid(nested_pid, root=tmp_path) == forked.session_slug
        assert sr.resolve_session_slug_for_pid(spawner_pid, root=tmp_path) == spawner_slug


_T0 = datetime(2026, 9, 25, 12, 0, 0, tzinfo=UTC)


def _holder(*, pid: int = 0, record_slug: str = '', slug: str = 'watcher-df-1') -> sr.LeaseHolder:
    return sr.LeaseHolder(
        session_slug=slug,
        pid=pid or os.getpid(),
        start_ts=_T0.isoformat(),
        record_slug=record_slug,
    )


class TestLeaseHolderRecordSlug:
    def test_record_slug_round_trips(self) -> None:
        holder = _holder(record_slug='role-proj-uuid')

        assert sr.LeaseHolder.from_dict(holder.to_dict()) == holder
        assert sr.LeaseHolder.from_json(holder.to_json()) == holder
        assert holder.to_dict()['record_slug'] == 'role-proj-uuid'

    def test_a_legacy_body_parses_as_unlinked(self) -> None:
        legacy = {
            'session_slug': 'watcher-df-357458',
            'pid': 357458,
            'start_ts': '2026-08-15T00:00:00+00:00',
        }

        assert sr.LeaseHolder.from_dict(legacy).record_slug == ''

    def test_an_explicit_null_parses_as_unlinked(self) -> None:
        body = {'session_slug': 'watcher-df-1', 'pid': 1, 'start_ts': '', 'record_slug': None}

        assert sr.LeaseHolder.from_dict(body).record_slug == ''

    def test_the_field_defaults_to_unlinked(self) -> None:
        holder = sr.LeaseHolder(session_slug='watcher-df-1', pid=1, start_ts='')

        assert holder.record_slug == ''

    def test_claim_lease_writes_it_and_a_contender_reads_it_back(self, tmp_path: Path) -> None:
        sr.claim_lease('watcher-df', holder=_holder(record_slug='role-proj-uuid'), root=tmp_path)
        body = json.loads(sr.lease_path_for_name('watcher-df', root=tmp_path).read_text())

        contended = sr.claim_lease('watcher-df', holder=_holder(slug='watcher-df-2'), root=tmp_path)

        assert body['record_slug'] == 'role-proj-uuid'
        assert contended.acquired is False
        assert contended.holder is not None
        assert contended.holder.record_slug == 'role-proj-uuid'


class TestHolderRecordState:
    _SLUG = 'role-proj-uuid'

    def test_no_holder_is_unlinked(self, tmp_path: Path) -> None:
        assert sr.holder_record_state(None, root=tmp_path) is sr.HolderRecordState.UNLINKED

    def test_a_holder_without_a_record_slug_is_unlinked(self, tmp_path: Path) -> None:
        state = sr.holder_record_state(_holder(record_slug=''), root=tmp_path)

        assert state is sr.HolderRecordState.UNLINKED

    def test_a_record_slug_that_is_not_a_record_key_is_unreadable(self, tmp_path: Path) -> None:
        state = sr.holder_record_state(_holder(record_slug='../x'), root=tmp_path)

        assert state is sr.HolderRecordState.UNREADABLE

    def test_a_reaped_record_is_absent(self, tmp_path: Path) -> None:
        state = sr.holder_record_state(_holder(record_slug=self._SLUG), root=tmp_path)

        assert state is sr.HolderRecordState.ABSENT

    @pytest.mark.parametrize('status', sorted(set(sr.Status) - sr.TERMINAL_STATUSES))
    def test_a_non_terminal_record_is_active(self, tmp_path: Path, status: sr.Status) -> None:
        sr.write_record(_record(self._SLUG, owner_pid=1, status=status), root=tmp_path)

        state = sr.holder_record_state(_holder(record_slug=self._SLUG), root=tmp_path)

        assert state is sr.HolderRecordState.ACTIVE

    @pytest.mark.parametrize('status', sorted(sr.TERMINAL_STATUSES))
    def test_a_terminal_record_is_exited(self, tmp_path: Path, status: sr.Status) -> None:
        sr.write_record(_record(self._SLUG, owner_pid=1, status=status), root=tmp_path)

        state = sr.holder_record_state(_holder(record_slug=self._SLUG), root=tmp_path)

        assert state is sr.HolderRecordState.EXITED

    def test_a_corrupt_record_is_unreadable(self, tmp_path: Path) -> None:
        path = sr.record_path_for_slug(self._SLUG, root=tmp_path)
        path.parent.mkdir(parents=True)
        path.write_text('{not json', encoding='utf-8')

        state = sr.holder_record_state(_holder(record_slug=self._SLUG), root=tmp_path)

        assert state is sr.HolderRecordState.UNREADABLE

    def test_the_printed_values(self) -> None:
        assert [state.value for state in sr.HolderRecordState] == [
            'unlinked',
            'absent',
            'unreadable',
            'active',
            'exited',
        ]

    def test_no_evidence_and_evidence_of_absence_never_share_a_value(
        self, tmp_path: Path
    ) -> None:
        legacy = sr.LeaseHolder.from_dict({'session_slug': 'watcher-df-1', 'pid': 1, 'start_ts': ''})
        reaped = _holder(record_slug=self._SLUG)

        assert sr.HolderRecordState.UNLINKED != sr.HolderRecordState.ABSENT
        assert sr.holder_record_state(legacy, root=tmp_path) != sr.holder_record_state(
            reaped, root=tmp_path
        )


_LEASE = 'watcher-df'


def _seed_lease(root: Path, body: str | sr.LeaseHolder, *, age_secs: float = 60.0) -> None:
    path = sr.lease_path_for_name(_LEASE, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body if isinstance(body, str) else body.to_json(), encoding='utf-8')
    ts = (_T0 - timedelta(seconds=age_secs)).timestamp()
    os.utime(path, (ts, ts))


def _contend(root: Path, *, record_slug: str = '') -> sr.LeaseClaim:
    contender = _holder(slug='watcher-df-contender', record_slug=record_slug)
    return sr.claim_lease(_LEASE, holder=contender, root=root, now=_T0)


class TestLeaseClaimHolderRecordState:
    _SLUG = 'role-proj-uuid'

    def test_an_acquired_claim_reports_the_claimants_own_linked_record(
        self, tmp_path: Path
    ) -> None:
        sr.write_record(_record(self._SLUG, owner_pid=os.getpid()), root=tmp_path)

        claim = sr.claim_lease(_LEASE, holder=_holder(record_slug=self._SLUG), root=tmp_path)

        assert claim.acquired is True
        assert claim.holder_record_state is sr.HolderRecordState.ACTIVE

    def test_an_acquired_unlinked_claim_is_unlinked(self, tmp_path: Path) -> None:
        claim = sr.claim_lease(_LEASE, holder=_holder(), root=tmp_path)

        assert claim.holder_record_state is sr.HolderRecordState.UNLINKED

    def test_a_contended_claim_reports_the_existing_holders_record(
        self, tmp_path: Path
    ) -> None:
        sr.write_record(
            _record(self._SLUG, owner_pid=1, status=sr.Status.EXITED), root=tmp_path
        )
        _seed_lease(tmp_path, _holder(record_slug=self._SLUG))

        claim = _contend(tmp_path)

        assert claim.acquired is False
        assert claim.holder_record_state is sr.HolderRecordState.EXITED

    def test_a_legacy_body_is_unlinked(self, tmp_path: Path) -> None:
        legacy = {'session_slug': 'watcher-df-1', 'pid': os.getpid(), 'start_ts': ''}
        _seed_lease(tmp_path, json.dumps(legacy))

        assert _contend(tmp_path).holder_record_state is sr.HolderRecordState.UNLINKED

    def test_an_unreadable_body_is_unlinked(self, tmp_path: Path) -> None:
        _seed_lease(tmp_path, '{not json')

        claim = _contend(tmp_path)

        assert claim.holder is None
        assert claim.holder_record_state is sr.HolderRecordState.UNLINKED


class TestHolderRecordIsAnAdditiveAxis:
    _SLUG = 'role-proj-uuid'

    @staticmethod
    def _verdict(claim: sr.LeaseClaim) -> tuple[object, ...]:
        return (claim.decision, claim.acquired, claim.holder_alive, claim.message)

    @pytest.mark.parametrize('existing', ['live', 'dead', 'unreadable'])
    def test_the_claim_verdict_ignores_record_slug(self, tmp_path: Path, existing: str) -> None:
        verdicts = []
        for record_slug in ('', self._SLUG):
            root = tmp_path / (record_slug or 'unlinked')
            sr.write_record(
                _record(self._SLUG, owner_pid=1, status=sr.Status.EXITED), root=root
            )
            if existing == 'unreadable':
                _seed_lease(root, '{not json')
            else:
                pid = os.getpid() if existing == 'live' else _DEAD_PID
                _seed_lease(root, _holder(pid=pid, record_slug=record_slug))
            verdicts.append(self._verdict(_contend(root, record_slug=record_slug)))

        assert verdicts[0] == verdicts[1]

    def test_a_dead_holder_whose_record_exited_still_stands_down_on_a_fresh_heartbeat(
        self, tmp_path: Path
    ) -> None:
        sr.write_record(
            _record(self._SLUG, owner_pid=_DEAD_PID, status=sr.Status.EXITED), root=tmp_path
        )
        _seed_lease(tmp_path, _holder(pid=_DEAD_PID, record_slug=self._SLUG))

        claim = _contend(tmp_path)

        assert claim.decision is sr.LeaseDecision.STAND_DOWN
        assert claim.holder_alive is False
        assert claim.holder_record_state is sr.HolderRecordState.EXITED


def _show(capsys: pytest.CaptureFixture[str]) -> list[str]:
    capsys.readouterr()
    assert sr.main(['lease-show', '--name', _LEASE]) == 0
    return capsys.readouterr().out.splitlines()


class TestLeaseShowHolderRecord:
    _SLUG = 'role-proj-uuid'

    def test_lease_status_carries_the_holders_record_link(self, tmp_path: Path) -> None:
        sr.write_record(_record(self._SLUG, owner_pid=os.getpid()), root=tmp_path)
        _seed_lease(tmp_path, _holder(record_slug=self._SLUG))

        status = sr.lease_status(_LEASE, root=tmp_path, now=_T0)

        assert status.holder_record_slug == self._SLUG
        assert status.holder_record_state is sr.HolderRecordState.ACTIVE

    def test_an_absent_lease_status_is_unlinked(self, tmp_path: Path) -> None:
        status = sr.lease_status(_LEASE, root=tmp_path, now=_T0)

        assert status.holder_record_slug is None
        assert status.holder_record_state is sr.HolderRecordState.UNLINKED

    def test_an_unreadable_lease_status_is_unlinked(self, tmp_path: Path) -> None:
        _seed_lease(tmp_path, '{not json')

        status = sr.lease_status(_LEASE, root=tmp_path, now=_T0)

        assert status.holder_record_slug is None
        assert status.holder_record_state is sr.HolderRecordState.UNLINKED

    def test_an_absent_lease_still_prints_only_name_and_state(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert _show(capsys) == [f'name={_LEASE}', 'state=absent']

    def test_a_linked_lease_appends_both_keys_after_the_existing_ones(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        sr.write_record(
            _record(self._SLUG, owner_pid=1, status=sr.Status.EXITED), root=tmp_path
        )
        _seed_lease(tmp_path, _holder(record_slug=self._SLUG))

        lines = _show(capsys)

        assert [line.split('=', 1)[0] for line in lines] == [
            'name',
            'state',
            'holder_slug',
            'holder_pid',
            'holder_pid_alive',
            'heartbeat_ts',
            'heartbeat_age_secs',
            'reclaimable',
            'holder_record',
            'holder_record_slug',
        ]
        assert lines[-2:] == ['holder_record=exited', f'holder_record_slug={self._SLUG}']

    def test_an_unlinked_lease_prints_holder_record_but_no_slug(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _seed_lease(tmp_path, _holder())

        lines = _show(capsys)

        assert lines[-1] == 'holder_record=unlinked'
        assert not any(line.startswith('holder_record_slug=') for line in lines)

    def test_an_unreadable_lease_prints_holder_record_unlinked_and_no_slug(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _seed_lease(tmp_path, '{not json')

        lines = _show(capsys)

        assert 'state=unreadable' in lines
        assert lines[-1] == 'holder_record=unlinked'
        assert not any(line.startswith('holder_record_slug=') for line in lines)


def _link_pid(pid: int, slug: str, root: Path) -> None:
    sr.write_record(_record(slug, owner_pid=pid), root=root)
    sr.write_session_pointer(pid, slug, root=root)


def _lease_body(root: Path) -> sr.LeaseHolder:
    return sr.LeaseHolder.from_json(sr.lease_path_for_name(_LEASE, root=root).read_text())


class TestLeaseClaimCliRecordSlug:
    _PID = 4_237_300
    _OTHER_PID = 4_237_400

    def test_a_bare_claim_links_the_claimants_own_record(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _link_pid(self._PID, 'watcher-own-record', tmp_path)
        monkeypatch.setenv('CLAUDE_PID', str(self._PID))

        assert sr.main(['lease-claim', '--name', _LEASE]) == 0

        assert _lease_body(tmp_path).record_slug == 'watcher-own-record'

    def test_an_explicit_pid_links_that_pids_record(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _link_pid(self._PID, 'the-env-pids-record', tmp_path)
        _link_pid(self._OTHER_PID, 'the-body-pids-record', tmp_path)
        monkeypatch.setenv('CLAUDE_PID', str(self._PID))

        sr.main(['lease-claim', '--name', _LEASE, '--slug', 'X', '--pid', str(self._OTHER_PID)])

        body = _lease_body(tmp_path)
        assert body.pid == self._OTHER_PID
        assert body.record_slug == 'the-body-pids-record'

    def test_an_unresolvable_record_slug_is_blank_quiet_and_never_blocks(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setenv('CLAUDE_PID', str(self._PID))
        caplog.set_level(logging.DEBUG)

        assert sr.main(['lease-claim', '--name', _LEASE]) == 0

        assert _lease_body(tmp_path).record_slug == ''
        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []

        monkeypatch.delenv('CLAUDE_PID')
        with pytest.raises(SystemExit) as excinfo:
            sr.main(['lease-claim', '--name', 'another-lease'])
        assert excinfo.value.code == 2

    @pytest.mark.parametrize(
        ('argv', 'max_calls'),
        [
            (['lease-claim', '--name', _LEASE], 1),
            (['lease-claim', '--name', _LEASE, '--slug', 'X', '--pid', '4237400'], 0),
        ],
        ids=['bare', 'explicit-slug-and-pid'],
    )
    def test_the_session_pid_is_resolved_at_most_once_and_only_on_demand(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        argv: list[str],
        max_calls: int,
    ) -> None:
        _link_pid(self._PID, 'watcher-own-record', tmp_path)
        monkeypatch.setenv('CLAUDE_PID', str(self._PID))
        calls: list[object] = []
        real = sr.resolve_session_pid

        def counting(env: object = None) -> int:
            calls.append(env)
            return real()

        monkeypatch.setattr(sr, 'resolve_session_pid', counting)

        assert sr.main(argv) == 0

        assert len(calls) <= max_calls
        assert _lease_body(tmp_path).pid in (self._PID, self._OTHER_PID)


def _claim_lines(capsys: pytest.CaptureFixture[str], *extra: str) -> list[str]:
    capsys.readouterr()
    assert sr.main(['lease-claim', '--name', _LEASE, *extra]) == 0
    return capsys.readouterr().out.splitlines()


def _seed_fresh_lease(root: Path, holder: sr.LeaseHolder) -> None:
    path = sr.lease_path_for_name(_LEASE, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(holder.to_json(), encoding='utf-8')


class TestLeaseClaimPrintsHolderRecordLast:
    _PID = 4_237_500
    _OWN_SLUG = 'watcher-own-record'
    _HOLDER_SLUG = 'the-holders-record'

    def test_an_acquired_claim_prints_the_claimants_own_state_as_line_five(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _link_pid(self._PID, self._OWN_SLUG, tmp_path)
        monkeypatch.setenv('CLAUDE_PID', str(self._PID))

        lines = _claim_lines(capsys)

        assert len(lines) == 5
        assert lines[0] == 'decision=acquired'
        assert lines[2:] == [
            'holder_liveness=none',
            f'slug={_LEASE}-{self._PID}',
            'holder_record=active',
        ]

    def test_an_acquired_unlinked_claim_prints_unlinked(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv('CLAUDE_PID', str(self._PID))

        assert _claim_lines(capsys)[-1] == 'holder_record=unlinked'

    @pytest.mark.parametrize(
        ('holder_pid', 'liveness'),
        [(os.getpid(), 'held'), (_DEAD_PID, 'orphaned')],
        ids=['held', 'orphaned'],
    )
    def test_a_contended_claim_prints_the_existing_holders_state_not_the_callers(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        holder_pid: int,
        liveness: str,
    ) -> None:
        sr.write_record(
            _record(self._HOLDER_SLUG, owner_pid=holder_pid, status=sr.Status.EXITED),
            root=tmp_path,
        )
        _seed_fresh_lease(tmp_path, _holder(pid=holder_pid, record_slug=self._HOLDER_SLUG))
        _link_pid(self._PID, self._OWN_SLUG, tmp_path)
        monkeypatch.setenv('CLAUDE_PID', str(self._PID))

        lines = _claim_lines(capsys)

        assert len(lines) == 5
        assert lines[0] == 'decision=stand-down'
        assert lines[2:] == [
            f'holder_liveness={liveness}',
            f'slug={_LEASE}-{self._PID}',
            'holder_record=exited',
        ]

    def test_the_fail_open_path_prints_no_holder_record_line(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _link_pid(self._PID, self._OWN_SLUG, tmp_path)
        monkeypatch.setenv('CLAUDE_PID', str(self._PID))

        def _boom(*_args: object, **_kwargs: object) -> sr.LeaseClaim:
            raise OSError('lease substrate on fire')

        monkeypatch.setattr(sr, 'claim_lease', _boom)

        lines = _claim_lines(capsys)

        assert lines[0] == 'decision=proceed'
        assert lines[2:] == [f'slug={_LEASE}-{self._PID}']
