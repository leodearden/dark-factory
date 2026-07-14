"""Tests for scripts/legibility/inventory.py — session enumeration (PRD §5.2 point 2).

``inventory.encode_cwd`` mirrors
``orchestrator.session_registry.transcript_path_for_cwd``'s cwd encoding
(both ``/`` and ``.`` map to ``-``). A project's agents span many encoded
dirs (57 for dark-factory today: main checkout + ``.worktrees``/
``.claude-worktrees`` children), so membership is resolved from the
session's REAL ``cwd`` (read from a transcript line) via path-component
semantics (``Path.is_relative_to``) — never a raw string prefix match on
the encoded dir name, which would over-include a sibling project sharing
the same literal prefix (e.g. ``dark-factory-cockpit``).

Imported as ``from legibility import inventory`` (PEP-420 namespace
package; see test_legibility_config.py's module docstring for the import
mechanics).
"""
from __future__ import annotations

import json
from datetime import date as dt_date
from pathlib import Path

from legibility import inventory as mod

MAIN_CWD = '/home/leo/src/dark-factory'
WORKTREE_CWD = '/home/leo/src/dark-factory/.worktrees/2573'
COCKPIT_CWD = '/home/leo/src/dark-factory-cockpit'


class TestEncodeCwd:
    def test_plain_path(self):
        assert mod.encode_cwd(MAIN_CWD) == '-home-leo-src-dark-factory'

    def test_worktrees_child_maps_slash_and_dot(self):
        # Both '/' and '.' -> '-', mirroring transcript_path_for_cwd exactly.
        assert mod.encode_cwd(WORKTREE_CWD) == '-home-leo-src-dark-factory--worktrees-2573'

    def test_cockpit_sibling_shares_literal_prefix(self):
        # This is exactly why a raw string-prefix match over-includes: the
        # encoded cockpit dir name starts with the encoded main dir name.
        encoded_main = mod.encode_cwd(MAIN_CWD)
        encoded_cockpit = mod.encode_cwd(COCKPIT_CWD)
        assert encoded_cockpit.startswith(encoded_main)
        assert encoded_cockpit != encoded_main


def _write_session(dir_path: Path, session_id: str, cwd: str, timestamp: str = '2026-07-13T10:00:00.000Z'):
    dir_path.mkdir(parents=True, exist_ok=True)
    session_path = dir_path / f'{session_id}.jsonl'
    lines = [
        {'type': 'user', 'cwd': cwd, 'timestamp': timestamp, 'message': {'content': 'hello'}},
    ]
    session_path.write_text('\n'.join(json.dumps(line) for line in lines) + '\n')
    return session_path


class TestIsMember:
    """is_member uses Path.is_relative_to path-component semantics."""

    def test_main_dir_is_member(self):
        assert mod.is_member(MAIN_CWD, [MAIN_CWD]) is True

    def test_worktree_child_is_member(self):
        assert mod.is_member(WORKTREE_CWD, [MAIN_CWD]) is True

    def test_cockpit_sibling_is_not_member(self):
        assert mod.is_member(COCKPIT_CWD, [MAIN_CWD]) is False


class TestProjectDirMembershipResolution:
    """End-to-end: a tmp projects_root with main + worktree + cockpit-sibling
    encoded dirs. Membership resolution (iter_project_dirs + is_member on
    each session's real cwd) includes the main dir and worktree child but
    excludes the cockpit sibling — even though the cockpit dir's encoded
    name is a candidate under the cheap prefix pre-filter.
    """

    def _build_tree(self, tmp_path: Path) -> Path:
        projects_root = tmp_path / 'projects'
        _write_session(projects_root / '-home-leo-src-dark-factory', 'main-session', MAIN_CWD)
        _write_session(
            projects_root / '-home-leo-src-dark-factory--worktrees-2573',
            'worktree-session',
            WORKTREE_CWD,
        )
        _write_session(
            projects_root / '-home-leo-src-dark-factory-cockpit',
            'cockpit-session',
            COCKPIT_CWD,
        )
        return projects_root

    def test_iter_project_dirs_over_includes_cockpit_as_a_candidate(self, tmp_path):
        # The cheap encoded-prefix pre-filter is intentionally imprecise —
        # confirms the design premise that a further real-cwd check is needed.
        projects_root = self._build_tree(tmp_path)
        dirs = {d.name for d in mod.iter_project_dirs(projects_root, [MAIN_CWD])}
        assert '-home-leo-src-dark-factory-cockpit' in dirs

    def test_enumerate_membership_excludes_cockpit_includes_worktree(self, tmp_path):
        projects_root = self._build_tree(tmp_path)
        candidate_dirs = mod.iter_project_dirs(projects_root, [MAIN_CWD])
        members = []
        for project_dir in candidate_dirs:
            for session_path in project_dir.glob('*.jsonl'):
                cwd = mod.session_cwd(session_path)
                if cwd is not None and mod.is_member(cwd, [MAIN_CWD]):
                    members.append(session_path.stem)
        assert set(members) == {'main-session', 'worktree-session'}
        assert 'cockpit-session' not in members


class TestSessionCwd:
    def test_reads_cwd_from_first_matching_line(self, tmp_path):
        session_path = _write_session(tmp_path, 'sess', MAIN_CWD)
        assert mod.session_cwd(session_path) == MAIN_CWD

    def test_returns_none_when_no_cwd_anywhere(self, tmp_path):
        # Mirrors real ~/.claude/projects stub files: metadata-only lines
        # (ai-title/agent-name/queue-operation) carry no 'cwd' at all.
        session_path = tmp_path / 'stub.jsonl'
        lines = [
            {'type': 'ai-title', 'aiTitle': 'x', 'sessionId': 'stub'},
            {'type': 'agent-name', 'agentName': 'x', 'sessionId': 'stub'},
        ]
        session_path.write_text('\n'.join(json.dumps(line) for line in lines) + '\n')
        assert mod.session_cwd(session_path) is None

    def test_returns_none_for_unreadable_path(self, tmp_path):
        assert mod.session_cwd(tmp_path / 'does-not-exist.jsonl') is None


class TestEnumerateSessions:
    """enumerate_sessions aggregates across every matching encoded dir
    (never one-dir-per-project), filters by first-timestamp UTC date,
    stamps real size_bytes, and skips non-.jsonl / empty / fully-malformed
    files without raising."""

    TARGET_DATE = dt_date(2026, 7, 13)

    def _build_tree(self, tmp_path: Path) -> Path:
        projects_root = tmp_path / 'projects'
        main_dir = projects_root / '-home-leo-src-dark-factory'
        worktree_dir = projects_root / '-home-leo-src-dark-factory--worktrees-2573'
        main_dir.mkdir(parents=True)
        worktree_dir.mkdir(parents=True)

        # Target-date session in the main dir.
        _write_session(main_dir, 'main-target', MAIN_CWD, timestamp='2026-07-13T09:00:00.000Z')
        # Different-date session in the main dir — must be excluded.
        _write_session(
            main_dir, 'main-other-date', MAIN_CWD, timestamp='2026-07-12T09:00:00.000Z'
        )
        # Target-date session in the worktree dir — proves aggregation
        # across multiple encoded dirs, not just the main one.
        _write_session(
            worktree_dir, 'worktree-target', WORKTREE_CWD, timestamp='2026-07-13T11:00:00.000Z'
        )

        # A non-.jsonl file: excluded by the *.jsonl glob itself.
        (main_dir / 'notes.txt').write_text('not a transcript')
        # An empty .jsonl file: must be skipped, not raise.
        (main_dir / 'empty.jsonl').write_text('')
        # A fully-malformed .jsonl file (no valid JSON line at all, so no
        # cwd/timestamp is derivable): must be skipped, not raise.
        (main_dir / 'garbage.jsonl').write_text('not json\n{{{broken\n')

        return projects_root

    def test_returns_only_target_date_sessions(self, tmp_path):
        projects_root = self._build_tree(tmp_path)
        records = mod.enumerate_sessions(projects_root, [MAIN_CWD], self.TARGET_DATE)
        assert {r.path.stem for r in records} == {'main-target', 'worktree-target'}

    def test_aggregates_across_multiple_encoded_dirs(self, tmp_path):
        projects_root = self._build_tree(tmp_path)
        records = mod.enumerate_sessions(projects_root, [MAIN_CWD], self.TARGET_DATE)
        assert {r.encoded_dir for r in records} == {
            '-home-leo-src-dark-factory',
            '-home-leo-src-dark-factory--worktrees-2573',
        }

    def test_size_bytes_matches_real_file_size(self, tmp_path):
        projects_root = self._build_tree(tmp_path)
        records = mod.enumerate_sessions(projects_root, [MAIN_CWD], self.TARGET_DATE)
        assert records  # sanity: the fixture does produce records
        for record in records:
            assert record.size_bytes == record.path.stat().st_size

    def test_excludes_different_date_session(self, tmp_path):
        projects_root = self._build_tree(tmp_path)
        records = mod.enumerate_sessions(projects_root, [MAIN_CWD], self.TARGET_DATE)
        assert 'main-other-date' not in {r.path.stem for r in records}

    def test_skips_non_jsonl_empty_and_malformed_without_raising(self, tmp_path):
        projects_root = self._build_tree(tmp_path)
        records = mod.enumerate_sessions(projects_root, [MAIN_CWD], self.TARGET_DATE)
        names = {r.path.stem for r in records}
        assert 'notes' not in names
        assert 'empty' not in names
        assert 'garbage' not in names
