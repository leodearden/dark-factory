"""Tests for orchestrator.verify_cancel — pgid-file path/lifecycle, descendant walk, cancel_request.

Steps covered:
  1 (test)  pgid-file path & lifecycle
  2 (impl)  verify_cancel module created
  3 (test)  collect_descendants — pure BFS with ppid_map injection
  4 (impl)  read_ppid_map + collect_descendants
  5 (test)  cancel_request — injected spies, all contract cases
  6 (impl)  cancel_request
  7 (test)  start_own_process_group setsid + fallback
  8 (impl)  start_own_process_group
"""

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


# ---------------------------------------------------------------------------
# Step-3: collect_descendants — pure BFS over an injected ppid_map
# ---------------------------------------------------------------------------


class TestCollectDescendants:
    """Pure collect_descendants(root, ppid_map) with injected ppid maps."""

    def _cd(self, root, ppid_map):
        from orchestrator.verify_cancel import collect_descendants
        return collect_descendants(root, ppid_map)

    def test_simple_chain(self):
        # root=100 -> 200 -> 300
        ppid_map = {200: 100, 300: 200}
        assert self._cd(100, ppid_map) == {200, 300}

    def test_tree_with_sibling(self):
        # root=100 -> {200, 201}; 200 -> 300; unrelated 999->1
        ppid_map = {200: 100, 201: 100, 300: 200, 999: 1}
        result = self._cd(100, ppid_map)
        assert result == {200, 201, 300}
        assert 999 not in result
        assert 100 not in result  # root excluded

    def test_missing_root_returns_empty(self):
        ppid_map = {200: 100, 300: 200}
        # root=999 has no children in the map
        assert self._cd(999, ppid_map) == set()

    def test_cyclic_map_does_not_loop(self):
        # Synthetic cycle: 200->100, 100->200 (nonsensical but must terminate)
        ppid_map = {100: 200, 200: 100, 300: 100}
        # Should terminate and return descendants without infinite loop
        result = self._cd(100, ppid_map)
        assert isinstance(result, set)
        assert 100 not in result  # root never included

    def test_empty_map_returns_empty(self):
        assert self._cd(42, {}) == set()

    def test_root_not_in_result(self):
        ppid_map = {200: 100}
        result = self._cd(100, ppid_map)
        assert 100 not in result
        assert 200 in result


# ---------------------------------------------------------------------------
# Step-5: cancel_request — injected spies, all contract cases
# ---------------------------------------------------------------------------


class TestCancelRequest:
    """cancel_request contract with injected ppid_map_provider, kill, killpg spies."""

    def _run(self, path, ppid_map, kill_side_effects=None, killpg_raises=None):
        """Helper: run cancel_request with deterministic injections.

        *kill_side_effects*: dict mapping pid -> exception to raise (or None to succeed).
        *killpg_raises*: exception to raise from killpg (or None to succeed).
        """
        from orchestrator.verify_cancel import cancel_request
        import signal

        kill_calls = []
        killpg_calls = []

        def _kill(pid, sig):
            kill_calls.append((pid, sig))
            if kill_side_effects and pid in kill_side_effects:
                exc = kill_side_effects[pid]
                if exc is not None:
                    raise exc

        def _killpg(pgid, sig):
            killpg_calls.append((pgid, sig))
            if killpg_raises is not None:
                raise killpg_raises

        rc = cancel_request(
            path,
            ppid_map_provider=lambda: ppid_map,
            kill=_kill,
            killpg=_killpg,
        )
        return rc, kill_calls, killpg_calls

    def test_no_file_returns_0_no_kill(self, tmp_path):
        """(a) No pgid file -> return 0, no kill calls."""
        from orchestrator.verify_cancel import pgid_file
        path = pgid_file(tmp_path / 'wt', 'req-a')
        rc, kill_calls, killpg_calls = self._run(path, {})
        assert rc == 0
        assert kill_calls == []
        assert killpg_calls == []

    def test_live_descendants_killed_file_removed(self, tmp_path):
        """(b) File with pgid + live descendants -> SIGKILL to all, killpg backstop, file removed, rc=0."""
        import signal
        from orchestrator.verify_cancel import pgid_file, write_pgid_file

        path = pgid_file(tmp_path / 'wt', 'req-b')
        write_pgid_file(path, 100)

        # ppid_map: 100 -> {200, 201}
        ppid_map = {200: 100, 201: 100}

        rc, kill_calls, killpg_calls = self._run(path, ppid_map)

        assert rc == 0
        killed_pids = {pid for pid, sig in kill_calls}
        assert {100, 200, 201} == killed_pids
        for _, sig in kill_calls:
            assert sig == signal.SIGKILL
        assert len(killpg_calls) == 1
        assert killpg_calls[0] == (100, signal.SIGKILL)
        assert not path.exists()

    def test_already_dead_targets_count_as_success(self, tmp_path):
        """(c) Every target raises ProcessLookupError -> rc=0, file removed."""
        from orchestrator.verify_cancel import pgid_file, write_pgid_file

        path = pgid_file(tmp_path / 'wt', 'req-c')
        write_pgid_file(path, 100)
        ppid_map = {200: 100}

        side_effects = {100: ProcessLookupError(), 200: ProcessLookupError()}
        rc, kill_calls, _ = self._run(path, ppid_map, kill_side_effects=side_effects)

        assert rc == 0
        assert not path.exists()

    def test_permission_error_returns_nonzero_file_retained(self, tmp_path):
        """(d) A live target raises PermissionError -> rc=1, file retained."""
        from orchestrator.verify_cancel import pgid_file, write_pgid_file

        path = pgid_file(tmp_path / 'wt', 'req-d')
        write_pgid_file(path, 100)
        ppid_map = {}

        side_effects = {100: PermissionError()}
        rc, kill_calls, _ = self._run(path, ppid_map, kill_side_effects=side_effects)

        assert rc == 1
        assert path.exists()  # file retained on failure

    def test_corrupt_file_removed_returns_0(self, tmp_path):
        """(e) Corrupt/empty file (non-int) -> remove file, return 0, no kills."""
        from orchestrator.verify_cancel import pgid_file

        path = pgid_file(tmp_path / 'wt', 'req-e')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('not-an-int')

        rc, kill_calls, killpg_calls = self._run(path, {})

        assert rc == 0
        assert kill_calls == []
        assert not path.exists()

    def test_killpg_oserror_suppressed(self, tmp_path):
        """killpg OSError is suppressed; rc still 0 if no PermissionError from kill."""
        from orchestrator.verify_cancel import pgid_file, write_pgid_file

        path = pgid_file(tmp_path / 'wt', 'req-f')
        write_pgid_file(path, 555)
        ppid_map = {}

        rc, _, killpg_calls = self._run(
            path, ppid_map, killpg_raises=OSError('group gone')
        )
        assert rc == 0
        assert not path.exists()
        assert len(killpg_calls) == 1


# ---------------------------------------------------------------------------
# Step-7: start_own_process_group — setsid + fallback
# ---------------------------------------------------------------------------


class TestStartOwnProcessGroup:
    """start_own_process_group: normal setsid path and OSError fallback."""

    def test_setsid_called_and_pgid_returned(self, monkeypatch):
        """Normal path: setsid() succeeds, getpgrp() returns the new group."""
        import os
        import orchestrator.verify_cancel as vc_mod

        setsid_calls = []

        def fake_setsid():
            setsid_calls.append(True)

        monkeypatch.setattr(os, 'setsid', fake_setsid)
        monkeypatch.setattr(os, 'getpgrp', lambda: 12345)

        result = vc_mod.start_own_process_group()

        assert len(setsid_calls) == 1
        assert result == 12345

    def test_oserror_falls_back_to_getpgrp(self, monkeypatch):
        """EPERM path: setsid() raises OSError, fallback returns os.getpgrp()."""
        import errno
        import os
        import orchestrator.verify_cancel as vc_mod

        def fake_setsid():
            raise OSError(errno.EPERM, 'Operation not permitted')

        monkeypatch.setattr(os, 'setsid', fake_setsid)
        monkeypatch.setattr(os, 'getpgrp', lambda: 99999)

        result = vc_mod.start_own_process_group()

        # Must not raise; must return the current group
        assert result == 99999
