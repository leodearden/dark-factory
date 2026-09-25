"""Tests for the pid -> session-record-slug pointer and its lease/decision consumers."""

from __future__ import annotations

from pathlib import Path

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
