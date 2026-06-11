"""Tests for orchestrator.verify_cancel — pgid-file path/lifecycle, descendant walk, cancel_request."""

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Step-1: pgid-file path & lifecycle
# ---------------------------------------------------------------------------


class TestPgidFilePath:
    """pgid_file resolves under <worktree_base>/.merge_verify_pgids/ and sanitizes request_id."""

    def test_normal_request_id_resolves_inside_pgid_dir(self, tmp_path: Path):
        from orchestrator.verify_cancel import pgid_file, PGID_DIR_NAME

        wt_base = tmp_path / 'wt'
        path = pgid_file(wt_base, 'abc-123')
        assert path.parent == wt_base / PGID_DIR_NAME
        assert path.name == 'abc-123'

    def test_path_traversal_sanitized(self, tmp_path: Path):
        """A request_id like '../escape' must NOT produce a path outside pgid_dir."""
        from orchestrator.verify_cancel import pgid_file, PGID_DIR_NAME

        wt_base = tmp_path / 'wt'
        path = pgid_file(wt_base, '../escape')
        pgid_dir = wt_base / PGID_DIR_NAME
        # Must stay inside the pgid dir
        assert path.parent == pgid_dir or path.is_relative_to(pgid_dir)
        # The final filename must not be 'escape' at the wrong level
        assert path.parent == pgid_dir

    def test_slash_in_request_id_sanitized(self, tmp_path: Path):
        """A request_id like 'a/b' must not create a sub-directory escape."""
        from orchestrator.verify_cancel import pgid_file, PGID_DIR_NAME

        wt_base = tmp_path / 'wt'
        path = pgid_file(wt_base, 'a/b')
        pgid_dir = wt_base / PGID_DIR_NAME
        assert path.parent == pgid_dir

    def test_uuid_style_request_id(self, tmp_path: Path):
        from orchestrator.verify_cancel import pgid_file, PGID_DIR_NAME

        wt_base = tmp_path / 'wt'
        rid = 'f47ac10b-58cc-4372-a567-0e02b2c3d479'
        path = pgid_file(wt_base, rid)
        assert path.parent == wt_base / PGID_DIR_NAME
        assert path.name == rid


class TestPgidFileLifecycle:
    """write_pgid_file creates parent dirs + writes pid; remove_pgid_file is idempotent."""

    def test_write_creates_dirs_and_readable(self, tmp_path: Path):
        from orchestrator.verify_cancel import pgid_file, write_pgid_file

        path = pgid_file(tmp_path / 'wt', 'my-req')
        assert not path.exists()
        write_pgid_file(path, 4242)
        assert path.read_text().strip() == '4242'

    def test_write_overwrites_existing(self, tmp_path: Path):
        from orchestrator.verify_cancel import pgid_file, write_pgid_file

        path = pgid_file(tmp_path / 'wt', 'my-req')
        write_pgid_file(path, 1111)
        write_pgid_file(path, 2222)
        assert path.read_text().strip() == '2222'

    def test_remove_absent_is_idempotent(self, tmp_path: Path):
        from orchestrator.verify_cancel import pgid_file, remove_pgid_file

        path = pgid_file(tmp_path / 'wt', 'nonexistent')
        # Must not raise even though the file does not exist
        remove_pgid_file(path)

    def test_remove_deletes_existing(self, tmp_path: Path):
        from orchestrator.verify_cancel import pgid_file, write_pgid_file, remove_pgid_file

        path = pgid_file(tmp_path / 'wt', 'my-req')
        write_pgid_file(path, 9999)
        assert path.exists()
        remove_pgid_file(path)
        assert not path.exists()
