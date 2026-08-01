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

import gzip
import importlib.util
import json
from datetime import date as dt_date
from pathlib import Path

from legibility import digest, inventory as mod
from orchestrator import session_registry

MAIN_CWD = '/home/leo/src/dark-factory'
WORKTREE_CWD = '/home/leo/src/dark-factory/.worktrees/2573'
COCKPIT_CWD = '/home/leo/src/dark-factory-cockpit'

# OBSERVED, not guessed (task 3272). Every right-hand side below is a real
# directory name read off a live ``~/.claude/projects`` tree, or a cwd
# confirmed against one. The rule was derived empirically from 738
# (encoded-dir, decoded-cwd) pairs sampled from that tree: the only
# substitutions observed were ``.`` -> ``-``, ``/`` -> ``-`` and
# ``_`` -> ``-``, and the only non-alphanumeric characters appearing in ANY
# sampled cwd were ``- . / _`` — so the three-character rule is complete
# over the observed domain (it reproduces all 738 pairs; the former
# two-character ``/``+``.`` rule mismatched 492 of them).
#
# These are STRING LITERALS on purpose. They must never be produced by
# calling ``encode_cwd`` (or any mirror of it): a fixture built with the
# function under test moves in lockstep with a bug in that function and can
# never detect it, which is exactly why the missing ``_`` rule survived a
# fully green suite. See TestEncoderLockstep below.
REAL_ENCODED_DIR_PAIRS: tuple[tuple[str, str], ...] = (
    (MAIN_CWD, '-home-leo-src-dark-factory'),
    (WORKTREE_CWD, '-home-leo-src-dark-factory--worktrees-2573'),
    (
        '/home/leo/src/dark-factory/.eval-worktrees/df_task_12/run-5383f6a8',
        '-home-leo-src-dark-factory--eval-worktrees-df-task-12-run-5383f6a8',
    ),
    (
        '/home/leo/src/reify/.claude/worktrees/printer-design-v01',
        '-home-leo-src-reify--claude-worktrees-printer-design-v01',
    ),
    # Pins CASE PRESERVATION: the encoder does NOT lowercase. This dir name
    # exists on disk with its capitals intact, ruling out a case-folding step.
    ('/opt/Auto-Claude/resources/backend', '-opt-Auto-Claude-resources-backend'),
    (
        '/home/leo/src/warm-lanes/worktrees/_lane-39',
        '-home-leo-src-warm-lanes-worktrees--lane-39',
    ),
    ('/media/leo/data_lv_1/leo/reify-build', '-media-leo-data-lv-1-leo-reify-build'),
)


class TestEncodeCwd:
    def test_plain_path(self):
        assert mod.encode_cwd(MAIN_CWD) == '-home-leo-src-dark-factory'

    def test_worktrees_child_maps_slash_and_dot(self):
        # Both '/' and '.' -> '-', mirroring transcript_path_for_cwd exactly.
        assert mod.encode_cwd(WORKTREE_CWD) == '-home-leo-src-dark-factory--worktrees-2573'

    def test_underscore_maps_to_dash(self):
        # The character the mirror used to miss (task 3272). Two thirds of the
        # real project dirs sampled contain an underscore.
        assert mod.encode_cwd('/media/leo/data_lv_1/leo/reify-build') == (
            '-media-leo-data-lv-1-leo-reify-build'
        )

    def test_round_trips_real_on_disk_dir_names(self):
        """Every encoding matches a dir name observed on a live ~/.claude/projects tree.

        Table-driven over REAL_ENCODED_DIR_PAIRS, whose expected values are
        hard-coded literals rather than encoder output — the only kind of
        assertion that can catch an encoder which is self-consistently wrong.
        """
        for cwd, expected_dir in REAL_ENCODED_DIR_PAIRS:
            assert mod.encode_cwd(cwd) == expected_dir, cwd

    def test_cockpit_sibling_shares_literal_prefix(self):
        # This is exactly why a raw string-prefix match over-includes: the
        # encoded cockpit dir name starts with the encoded main dir name.
        encoded_main = mod.encode_cwd(MAIN_CWD)
        encoded_cockpit = mod.encode_cwd(COCKPIT_CWD)
        assert encoded_cockpit.startswith(encoded_main)
        assert encoded_cockpit != encoded_main


def _load_sibling_test_module(name: str):
    """Import a sibling scripts/tests module by file path.

    ``scripts/tests`` is not on ``sys.path`` (its conftest inserts
    ``scripts/`` and ``scripts/legibility``, not itself), so a bare
    ``import test_legibility_nightly`` would not resolve under the suite's
    ``--import-mode=importlib`` collection. Loading by path is the sanctioned
    equivalent and avoids restructuring the nightly fixture.
    """
    spec = importlib.util.spec_from_file_location(
        f'_lockstep_{name}', Path(__file__).parent / f'{name}.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestEncoderLockstep:
    """Every in-repo copy of the cwd encoding must agree with the canonical (task 3272).

    The rule is duplicated four times across the repo, and in 3272 ALL FOUR
    copies were found to be missing the same character (``_`` -> ``-``) at
    once. The old ``inventory.encode_cwd`` docstring asserted the mirrors were
    "kept in lockstep with the canonical implementation" — a claim nothing
    checked, and which was false in fact.

    This class replaces that aspiration with an enforced invariant. Each
    mirror is asserted equal to BOTH:

      - ``session_registry.encode_cwd``, the canonical — so a mirror that
        drifts from it fails loudly; and
      - the hard-coded ``REAL_ENCODED_DIR_PAIRS`` dir names — so the
        canonical drifting from REALITY fails too.

    The second assertion is the load-bearing one. A mirror-only check would
    have passed cleanly on the pre-3272 tree, because all four copies were
    consistently wrong together. The same defect explains why 37 green tests
    never caught it: every fixture built its session dirs by calling the
    encoder under test, so the fixtures tracked the bug. Only literals read
    off a real ``~/.claude/projects`` tree can detect an encoder that is
    self-consistently wrong.
    """

    def _mirrors(self):
        """(label, callable) for every in-repo copy of the rule."""
        nightly_tests = _load_sibling_test_module('test_legibility_nightly')
        return (
            ('legibility.inventory.encode_cwd', mod.encode_cwd),
            ('legibility.digest._encode_cwd', digest._encode_cwd),
            ('test_legibility_nightly._encode_cwd', nightly_tests._encode_cwd),
        )

    def test_every_mirror_agrees_with_canonical_and_with_reality(self):
        canonical = session_registry.encode_cwd
        for cwd, expected_dir in REAL_ENCODED_DIR_PAIRS:
            # The canonical itself must match the real on-disk dir name.
            assert canonical(cwd) == expected_dir, f'canonical drifted from reality: {cwd}'
            for label, mirror in self._mirrors():
                got = mirror(cwd)
                assert got == canonical(cwd), f'{label} drifted from canonical: {cwd}'
                assert got == expected_dir, f'{label} drifted from reality: {cwd}'


def _write_session(dir_path: Path, session_id: str, cwd: str, timestamp: str = '2026-07-13T10:00:00.000Z'):
    dir_path.mkdir(parents=True, exist_ok=True)
    session_path = dir_path / f'{session_id}.jsonl'
    lines = [
        {'type': 'user', 'cwd': cwd, 'timestamp': timestamp, 'message': {'content': 'hello'}},
    ]
    session_path.write_text('\n'.join(json.dumps(line) for line in lines) + '\n')
    return session_path


def _write_session_gz(
    dir_path: Path, session_id: str, cwd: str, timestamp: str = '2026-07-13T10:00:00.000Z'
):
    """Gzip sibling of :func:`_write_session`: write a ``<sid>.jsonl.gz``
    fixture in the archived-fleet-transcript format (shared.transcript_archive)."""
    dir_path.mkdir(parents=True, exist_ok=True)
    session_path = dir_path / f'{session_id}.jsonl.gz'
    lines = [
        {'type': 'user', 'cwd': cwd, 'timestamp': timestamp, 'message': {'content': 'hello'}},
    ]
    with gzip.open(session_path, 'wt', encoding='utf-8') as f:
        f.write('\n'.join(json.dumps(line) for line in lines) + '\n')
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


class TestGzAwareReader:
    """The single low-level reader (``_iter_json_lines``, via
    ``_session_cwd_and_date`` / ``session_cwd``) transparently reads
    gzip-compressed ``.jsonl.gz`` transcripts — the archived fleet-transcript
    format (shared.transcript_archive) — keeping byte-parity for plain
    ``.jsonl`` and degrading a corrupt ``.gz`` to ``(None, None)`` rather than
    raising (gzip.BadGzipFile subclasses OSError, so it flows through the
    existing ``except OSError`` degrade path)."""

    def test_session_cwd_reads_gz(self, tmp_path):
        gz_path = _write_session_gz(tmp_path, 'sess', MAIN_CWD)
        assert mod.session_cwd(gz_path) == MAIN_CWD

    def test_session_cwd_and_date_reads_gz(self, tmp_path):
        gz_path = _write_session_gz(
            tmp_path, 'sess', WORKTREE_CWD, timestamp='2026-07-13T10:00:00.000Z'
        )
        cwd, session_date = mod._session_cwd_and_date(gz_path)
        assert cwd == WORKTREE_CWD
        assert session_date == dt_date(2026, 7, 13)

    def test_plain_jsonl_parity(self, tmp_path):
        # A plain .jsonl still reads exactly as before (no gz branch taken).
        plain_path = _write_session(
            tmp_path, 'sess', MAIN_CWD, timestamp='2026-07-13T10:00:00.000Z'
        )
        cwd, session_date = mod._session_cwd_and_date(plain_path)
        assert cwd == MAIN_CWD
        assert session_date == dt_date(2026, 7, 13)

    def test_corrupt_gz_degrades_to_none(self, tmp_path):
        # Raw non-gzip bytes under a .jsonl.gz name: gzip raises BadGzipFile
        # (an OSError subclass) on first read, which _session_cwd_and_date's
        # `except OSError` maps to (None, None) — no raise.
        corrupt = tmp_path / 'corrupt.jsonl.gz'
        corrupt.write_bytes(b'this is not gzip\n{"cwd": "/x", "timestamp": "2026-07-13T10:00:00Z"}\n')
        assert mod.session_cwd(corrupt) is None


class TestResolveAgentTranscriptRoots:
    """resolve_agent_transcript_roots joins each relative root against
    project_root (so mining is independent of the process CWD) and returns
    an already-absolute root unchanged — always as pathlib.Path instances."""

    PROJECT_ROOT = '/home/leo/src/dark-factory'

    def test_relative_root_resolved_against_project_root(self):
        roots = mod.resolve_agent_transcript_roots(
            self.PROJECT_ROOT, ['data/orchestrator/agent-transcripts']
        )
        assert roots == [
            Path('/home/leo/src/dark-factory/data/orchestrator/agent-transcripts')
        ]

    def test_absolute_root_returned_unchanged(self):
        roots = mod.resolve_agent_transcript_roots(
            self.PROJECT_ROOT, ['/var/lib/agent-transcripts']
        )
        assert roots == [Path('/var/lib/agent-transcripts')]

    def test_empty_roots_returns_empty_list(self):
        assert mod.resolve_agent_transcript_roots(self.PROJECT_ROOT, []) == []

    def test_result_elements_are_paths(self):
        roots = mod.resolve_agent_transcript_roots(
            self.PROJECT_ROOT, ['data/orchestrator/agent-transcripts', '/abs/root']
        )
        assert roots and all(isinstance(r, Path) for r in roots)


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


class TestEnumerateArchiveRoots:
    """enumerate_sessions additionally walks agent_transcript_roots — the
    archived fleet-transcript tree written by shared.transcript_archive in
    the production nested layout ``<archive>/<task_id>/<enc>/<sid>.jsonl.gz``
    (+ a plain ``.jsonl`` variant) — recursively, gated solely by
    :func:`is_member` on each session's REAL cwd. The empty-roots path is
    byte-identical to today (the archive loop simply does not execute).
    """

    TARGET_DATE = dt_date(2026, 7, 13)
    WT_ENC = '-home-leo-src-dark-factory--worktrees-2573'

    def _build_archive(self, root: Path) -> Path:
        # Production nested layout: <archive>/<task_id>/<enc>/<sid>.jsonl(.gz)
        enc_dir = root / '2573' / self.WT_ENC
        _write_session_gz(
            enc_dir, 'gz-session', WORKTREE_CWD, timestamp='2026-07-13T09:00:00.000Z'
        )
        _write_session(
            enc_dir, 'plain-session', WORKTREE_CWD, timestamp='2026-07-13T10:00:00.000Z'
        )
        # A non-member cockpit cwd under its own task-id/enc dir: is_member
        # is false, so it is excluded even though it is inside the archive.
        _write_session_gz(
            root / '9999' / '-home-leo-src-dark-factory-cockpit',
            'cockpit-session', COCKPIT_CWD, timestamp='2026-07-13T09:30:00.000Z',
        )
        return root

    def test_enumerates_gz_and_plain_archive_sessions(self, tmp_path):
        archive = self._build_archive(tmp_path / 'archive')
        records = mod.enumerate_sessions(
            tmp_path / 'no-projects', [MAIN_CWD], self.TARGET_DATE,
            agent_transcript_roots=[archive],
        )
        assert {r.path.name for r in records} == {
            'gz-session.jsonl.gz', 'plain-session.jsonl',
        }

    def test_archive_record_fields(self, tmp_path):
        archive = self._build_archive(tmp_path / 'archive')
        records = mod.enumerate_sessions(
            tmp_path / 'no-projects', [MAIN_CWD], self.TARGET_DATE,
            agent_transcript_roots=[archive],
        )
        gz = next(r for r in records if r.path.name == 'gz-session.jsonl.gz')
        assert gz.encoded_dir == self.WT_ENC
        assert gz.cwd == WORKTREE_CWD
        assert gz.date == self.TARGET_DATE
        assert gz.size_bytes == gz.path.stat().st_size

    def test_non_member_cockpit_session_excluded(self, tmp_path):
        archive = self._build_archive(tmp_path / 'archive')
        records = mod.enumerate_sessions(
            tmp_path / 'no-projects', [MAIN_CWD], self.TARGET_DATE,
            agent_transcript_roots=[archive],
        )
        assert 'cockpit-session.jsonl.gz' not in {r.path.name for r in records}

    def test_empty_agent_transcript_roots_is_byte_identical(self, tmp_path):
        # A tree with BOTH a projects-root session and a populated archive.
        projects_root = tmp_path / 'projects'
        _write_session(
            projects_root / '-home-leo-src-dark-factory', 'main-session', MAIN_CWD,
            timestamp='2026-07-13T09:00:00.000Z',
        )
        self._build_archive(tmp_path / 'archive')

        # No agent_transcript_roots kwarg at all == today's behavior.
        default_records = mod.enumerate_sessions(projects_root, [MAIN_CWD], self.TARGET_DATE)
        # Explicit empty tuple == same (the archive loop does not execute).
        empty_records = mod.enumerate_sessions(
            projects_root, [MAIN_CWD], self.TARGET_DATE, agent_transcript_roots=(),
        )
        assert {r.path.name for r in default_records} == {'main-session.jsonl'}
        assert {r.path.name for r in empty_records} == {'main-session.jsonl'}

    def test_absent_archive_root_yields_nothing_and_does_not_raise(self, tmp_path):
        records = mod.enumerate_sessions(
            tmp_path / 'no-projects', [MAIN_CWD], self.TARGET_DATE,
            agent_transcript_roots=[tmp_path / 'does-not-exist'],
        )
        assert records == []


class TestEnumerateSessionsInRange:
    """enumerate_sessions_in_range walks the projects tree ONCE and keeps
    every session whose date falls in the inclusive ``[start_date, end_date]``
    window — the single-walk O(total_files) replacement for calling
    :func:`enumerate_sessions` once per calendar date (which re-opens each
    file window_days times, O(window_days × files))."""

    START_DATE = dt_date(2026, 7, 12)
    END_DATE = dt_date(2026, 7, 14)

    def test_inclusive_boundaries_both_kept(self, tmp_path):
        # A session dated == start_date AND one dated == end_date are BOTH
        # kept: the window is inclusive on both ends.
        projects_root = tmp_path / 'projects'
        main_dir = projects_root / '-home-leo-src-dark-factory'
        _write_session(main_dir, 'at-start', MAIN_CWD, timestamp='2026-07-12T09:00:00.000Z')
        _write_session(main_dir, 'at-end', MAIN_CWD, timestamp='2026-07-14T23:00:00.000Z')
        records = mod.enumerate_sessions_in_range(
            projects_root, [MAIN_CWD], self.START_DATE, self.END_DATE
        )
        assert {r.path.stem for r in records} == {'at-start', 'at-end'}

    def test_out_of_range_excluded(self, tmp_path):
        # One day before start and one day after end are BOTH excluded; a
        # mid-window session is kept.
        projects_root = tmp_path / 'projects'
        main_dir = projects_root / '-home-leo-src-dark-factory'
        _write_session(main_dir, 'before-start', MAIN_CWD, timestamp='2026-07-11T09:00:00.000Z')
        _write_session(main_dir, 'after-end', MAIN_CWD, timestamp='2026-07-15T09:00:00.000Z')
        _write_session(main_dir, 'in-range', MAIN_CWD, timestamp='2026-07-13T09:00:00.000Z')
        records = mod.enumerate_sessions_in_range(
            projects_root, [MAIN_CWD], self.START_DATE, self.END_DATE
        )
        assert {r.path.stem for r in records} == {'in-range'}

    def test_aggregates_across_multiple_encoded_dirs(self, tmp_path):
        # Mirrors TestEnumerateSessions: aggregation spans every matching
        # encoded dir, not just the main one.
        projects_root = tmp_path / 'projects'
        main_dir = projects_root / '-home-leo-src-dark-factory'
        worktree_dir = projects_root / '-home-leo-src-dark-factory--worktrees-2573'
        _write_session(main_dir, 'main-in-range', MAIN_CWD, timestamp='2026-07-12T09:00:00.000Z')
        _write_session(
            worktree_dir, 'worktree-in-range', WORKTREE_CWD, timestamp='2026-07-14T11:00:00.000Z'
        )
        records = mod.enumerate_sessions_in_range(
            projects_root, [MAIN_CWD], self.START_DATE, self.END_DATE
        )
        assert {r.encoded_dir for r in records} == {
            '-home-leo-src-dark-factory',
            '-home-leo-src-dark-factory--worktrees-2573',
        }

    def test_single_walk_opens_each_in_range_file_exactly_once(self, tmp_path, monkeypatch):
        # Three in-range dates across the window. A spy that wraps + delegates
        # to _session_cwd_and_date (the single gz-decompress/open point)
        # records each path it is called with: every in-range file must be
        # passed EXACTLY ONCE — proving the range enumerator is O(total_files),
        # not O(window_days × files) (the per-date loop would open each file 3×).
        projects_root = tmp_path / 'projects'
        main_dir = projects_root / '-home-leo-src-dark-factory'
        _write_session(main_dir, 'day12', MAIN_CWD, timestamp='2026-07-12T09:00:00.000Z')
        _write_session(main_dir, 'day13', MAIN_CWD, timestamp='2026-07-13T09:00:00.000Z')
        _write_session(main_dir, 'day14', MAIN_CWD, timestamp='2026-07-14T09:00:00.000Z')

        real = mod._session_cwd_and_date
        opened = []

        def spy(path):
            opened.append(path)
            return real(path)

        monkeypatch.setattr(mod, '_session_cwd_and_date', spy)

        records = mod.enumerate_sessions_in_range(
            projects_root, [MAIN_CWD], self.START_DATE, self.END_DATE
        )
        assert {r.path.stem for r in records} == {'day12', 'day13', 'day14'}
        # Exactly one open per in-range file — no per-date re-walk.
        for stem in ('day12', 'day13', 'day14'):
            path = main_dir / f'{stem}.jsonl'
            assert opened.count(path) == 1, f'{stem} opened {opened.count(path)}× (want 1)'
        # And no extra opens beyond the three in-range files.
        assert len(opened) == 3


class TestArchiveEncPrefilter:
    """The archive-roots walk cheaply pre-filters by the encoded ``<enc>``
    directory — the archive-root-relative ``parts[1]`` — mirroring
    :func:`iter_project_dirs`' superset pre-filter, so a proven-foreign
    ``<enc>`` is skipped WITHOUT a gz-decompress. :func:`is_member` on the
    real cwd remains the SOLE membership authority for lossy false-positives
    (e.g. a ``-cockpit`` sibling that string-startswith the prefix). ``<enc>``
    is ``parts[1]`` for BOTH the main (``<task>/<enc>/<sid>.jsonl.gz``) and
    subagent (``<task>/<enc>/<sid>/subagents/agent-*.jsonl.gz``) layouts —
    never ``session_path.parent.name`` (== ``'subagents'`` for the subagent
    variant, which would wrongly drop every subagent transcript)."""

    TARGET_DATE = dt_date(2026, 7, 13)
    WT_ENC = '-home-leo-src-dark-factory--worktrees-2573'
    OTHER_CWD = '/home/leo/src/other-project'
    OTHER_ENC = '-home-leo-src-other-project'

    @staticmethod
    def _install_open_spy(monkeypatch) -> list[Path]:
        """Wrap+delegate to _session_cwd_and_date (the single gz-decompress/
        open point), recording every path it is called with."""
        real = mod._session_cwd_and_date
        opened: list[Path] = []

        def spy(path):
            opened.append(path)
            return real(path)

        monkeypatch.setattr(mod, '_session_cwd_and_date', spy)
        return opened

    def test_foreign_enc_excluded_and_never_opened(self, tmp_path, monkeypatch):
        # (a) A foreign <enc> (does NOT startswith the encoded MAIN prefix) is
        # excluded AND its path is never passed to the reader — skipped without
        # a gz-decompress by the cheap <enc> pre-filter.
        archive = tmp_path / 'archive'
        _write_session_gz(
            archive / '2573' / self.OTHER_ENC, 'foreign', self.OTHER_CWD,
            timestamp='2026-07-13T09:00:00.000Z',
        )
        opened = self._install_open_spy(monkeypatch)
        records = mod.enumerate_sessions(
            tmp_path / 'no-projects', [MAIN_CWD], self.TARGET_DATE,
            agent_transcript_roots=[archive],
        )
        foreign_path = archive / '2573' / self.OTHER_ENC / 'foreign.jsonl.gz'
        assert records == []
        assert foreign_path not in opened

    def test_member_enc_kept_and_opened(self, tmp_path, monkeypatch):
        # (b) A member <enc> is kept AND its path WAS passed to the reader.
        archive = tmp_path / 'archive'
        member_path = _write_session_gz(
            archive / '2573' / self.WT_ENC, 'member', WORKTREE_CWD,
            timestamp='2026-07-13T09:00:00.000Z',
        )
        opened = self._install_open_spy(monkeypatch)
        records = mod.enumerate_sessions(
            tmp_path / 'no-projects', [MAIN_CWD], self.TARGET_DATE,
            agent_transcript_roots=[archive],
        )
        assert {r.path.name for r in records} == {'member.jsonl.gz'}
        assert member_path in opened

    def test_lossy_cockpit_false_positive_is_opened_then_is_member_rejected(
        self, tmp_path, monkeypatch
    ):
        # (c) A -cockpit <enc> string-startswith the encoded main prefix (a
        # LOSSY false-positive), so the superset pre-filter admits it as a
        # candidate — it IS opened — but is_member on the real cwd rejects it.
        # The pre-filter is a superset filter; is_member is the sole authority.
        archive = tmp_path / 'archive'
        cockpit_enc = mod.encode_cwd(COCKPIT_CWD)
        cockpit_path = _write_session_gz(
            archive / '9999' / cockpit_enc, 'cockpit', COCKPIT_CWD,
            timestamp='2026-07-13T09:00:00.000Z',
        )
        opened = self._install_open_spy(monkeypatch)
        records = mod.enumerate_sessions(
            tmp_path / 'no-projects', [MAIN_CWD], self.TARGET_DATE,
            agent_transcript_roots=[archive],
        )
        assert records == []
        assert cockpit_path in opened

    def test_subagent_layout_member_kept_and_opened(self, tmp_path, monkeypatch):
        # (d) Subagent layout: <archive>/<task>/<enc>/<sid>/subagents/agent-x.jsonl.gz.
        # <enc> is parts[1] (the member WT_ENC), NOT parent.name (== 'subagents',
        # which never encoded-prefix-matches a cwd and would drop EVERY subagent
        # transcript). The member subagent file is kept + opened, and its
        # encoded_dir is the real <enc>, not 'subagents'.
        archive = tmp_path / 'archive'
        sub_dir = archive / '2573' / self.WT_ENC / 'cafe-sid' / 'subagents'
        sub_path = _write_session_gz(
            sub_dir, 'agent-x', WORKTREE_CWD, timestamp='2026-07-13T09:00:00.000Z',
        )
        opened = self._install_open_spy(monkeypatch)
        records = mod.enumerate_sessions(
            tmp_path / 'no-projects', [MAIN_CWD], self.TARGET_DATE,
            agent_transcript_roots=[archive],
        )
        assert {r.path.name for r in records} == {'agent-x.jsonl.gz'}
        assert sub_path in opened
        record = next(r for r in records if r.path.name == 'agent-x.jsonl.gz')
        assert record.encoded_dir == self.WT_ENC
