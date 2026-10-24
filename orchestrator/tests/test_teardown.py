"""Tests for orchestrator.teardown — primitives that carry a teardown to a
confirmed end (task 6794)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from orchestrator.teardown import path_confirmed_absent


class TestPathConfirmedAbsent:
    def test_never_created_path_is_absent(self, tmp_path: Path) -> None:
        assert path_confirmed_absent(tmp_path / 'never') is True

    def test_existing_directory_is_not_absent(self, tmp_path: Path) -> None:
        d = tmp_path / 'd'
        d.mkdir()
        assert path_confirmed_absent(d) is False

    def test_existing_file_is_not_absent(self, tmp_path: Path) -> None:
        f = tmp_path / 'f'
        f.write_text('x')
        assert path_confirmed_absent(f) is False

    def test_dangling_symlink_is_not_absent(self, tmp_path: Path) -> None:
        link = tmp_path / 'link'
        link.symlink_to(tmp_path / 'gone')
        assert path_confirmed_absent(link) is False

    @pytest.mark.skipif(os.geteuid() == 0, reason='root bypasses permission bits')
    def test_unreadable_parent_is_unknown_not_absent(self, tmp_path: Path) -> None:
        locked = tmp_path / 'locked'
        locked.mkdir()
        inner = locked / 'inner'
        inner.mkdir()
        locked.chmod(0o000)
        try:
            with pytest.raises(PermissionError):
                os.lstat(inner)
            assert path_confirmed_absent(inner) is False
        finally:
            locked.chmod(0o700)
