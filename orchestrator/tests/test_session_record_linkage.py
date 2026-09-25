"""Tests for the pid -> session-record-slug pointer and its lease/decision consumers."""

from __future__ import annotations

from pathlib import Path

import pytest  # pyright: ignore[reportMissingImports]

from orchestrator import session_registry as sr

# A pid virtually guaranteed dead on any host (the suite-wide idiom; see
# test_session_registry.py's own _DEAD_PID -- deliberately not imported).
_DEAD_PID = 2**31 - 1


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
