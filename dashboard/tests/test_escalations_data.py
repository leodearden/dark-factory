"""Tests for dashboard.data.escalations module.

Tests use tmp_path to materialise fake escalation dirs (root + archive subdirs)
and synthesise minimal escalation-shaped dicts.  No async / MCP traffic.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _esc(esc_id: str, task_id: str = '1', level: int = 0, status: str = 'pending',
          worktree: str | None = None) -> dict:
    """Return a minimal escalation-shaped dict."""
    d = {'id': esc_id, 'task_id': task_id, 'level': level, 'status': status}
    if worktree is not None:
        d['worktree'] = worktree
    return d


def _write_esc(directory: Path, filename: str, data: dict) -> Path:
    """Write escalation dict to a JSON file in directory."""
    path = directory / filename
    path.write_text(json.dumps(data))
    return path


# ---------------------------------------------------------------------------
# Tests for load_queue_escalations
# ---------------------------------------------------------------------------

class TestLoadQueueEscalations:
    """Tests for load_queue_escalations(esc_dir: Path) -> list[dict]."""

    def test_missing_directory_returns_empty(self, tmp_path):
        from dashboard.data.escalations import load_queue_escalations

        result = load_queue_escalations(tmp_path / 'nonexistent')
        assert result == []

    def test_non_directory_path_returns_empty(self, tmp_path):
        """A file (not a directory) at the given path returns []."""
        from dashboard.data.escalations import load_queue_escalations

        fake_dir = tmp_path / 'not_a_dir.json'
        fake_dir.write_text('{}')

        result = load_queue_escalations(fake_dir)
        assert result == []

    def test_empty_directory_returns_empty(self, tmp_path):
        from dashboard.data.escalations import load_queue_escalations

        esc_dir = tmp_path / 'escalations'
        esc_dir.mkdir()

        result = load_queue_escalations(esc_dir)
        assert result == []

    def test_root_level_json_files_are_loaded(self, tmp_path):
        """*.json files at the root of esc_dir are returned as dicts."""
        from dashboard.data.escalations import load_queue_escalations

        esc_dir = tmp_path / 'escalations'
        esc_dir.mkdir()

        esc1 = _esc('esc-1-1', task_id='1', level=0, status='pending')
        esc2 = _esc('esc-2-1', task_id='2', level=1, status='resolved')
        _write_esc(esc_dir, 'esc-1-1.json', esc1)
        _write_esc(esc_dir, 'esc-2-1.json', esc2)

        result = load_queue_escalations(esc_dir)
        assert len(result) == 2
        result_ids = {r['id'] for r in result}
        assert result_ids == {'esc-1-1', 'esc-2-1'}

    def test_archive_subdir_files_are_excluded(self, tmp_path):
        """Files in archive/YYYY-MM-DD/ subdirs are NOT included (root-only glob)."""
        from dashboard.data.escalations import load_queue_escalations

        esc_dir = tmp_path / 'escalations'
        esc_dir.mkdir()

        # Root-level file — should be included
        esc_root = _esc('esc-root-1', task_id='10')
        _write_esc(esc_dir, 'esc-root-1.json', esc_root)

        # Archive subtree — must NOT be included
        archive_day = esc_dir / 'archive' / '2026-05-27'
        archive_day.mkdir(parents=True)
        esc_archived = _esc('esc-archived-1', task_id='20')
        _write_esc(archive_day, 'esc-archived-1.json', esc_archived)

        result = load_queue_escalations(esc_dir)
        ids = {r['id'] for r in result}
        assert 'esc-root-1' in ids
        assert 'esc-archived-1' not in ids

    def test_malformed_json_is_skipped_and_warning_emitted(self, tmp_path, caplog):
        """A file with bad JSON is skipped; valid files are still returned.

        Verifies that logger.warning is called for the bad file.
        """
        from dashboard.data.escalations import load_queue_escalations

        esc_dir = tmp_path / 'escalations'
        esc_dir.mkdir()

        # Valid file
        esc_valid = _esc('esc-good-1', task_id='1')
        _write_esc(esc_dir, 'esc-good-1.json', esc_valid)

        # Malformed JSON
        (esc_dir / 'esc-bad-1.json').write_text('this is not json {{{')

        with caplog.at_level(logging.WARNING, logger='dashboard.data.escalations'):
            result = load_queue_escalations(esc_dir)

        assert len(result) == 1
        assert result[0]['id'] == 'esc-good-1'
        assert caplog.records, 'Expected a WARNING log for the bad JSON file'
        assert any('esc-bad-1.json' in rec.message or 'esc-bad-1' in rec.message
                   for rec in caplog.records), 'WARNING should reference the bad file'

    def test_fields_intact(self, tmp_path):
        """Loaded dict has all original fields intact — nothing stripped or transformed."""
        from dashboard.data.escalations import load_queue_escalations

        esc_dir = tmp_path / 'escalations'
        esc_dir.mkdir()

        esc = _esc('esc-3-1', task_id='3', level=2, status='dismissed',
                   worktree='/home/leo/src/proj/.worktrees/3')
        esc['extra_field'] = 'should-survive'
        _write_esc(esc_dir, 'esc-3-1.json', esc)

        result = load_queue_escalations(esc_dir)
        assert len(result) == 1
        loaded = result[0]
        assert loaded['id'] == 'esc-3-1'
        assert loaded['task_id'] == '3'
        assert loaded['level'] == 2
        assert loaded['status'] == 'dismissed'
        assert loaded['worktree'] == '/home/leo/src/proj/.worktrees/3'
        assert loaded['extra_field'] == 'should-survive'

    # -- the opt-in ``skipped`` out-parameter -------------------------------
    #
    # Skipping an unparseable file is correct — one corrupt escalation must not
    # crash a queue scan.  Doing it with no channel back to the caller is not:
    # the reader holds both the path and the exception at the failure point and
    # threw both away into a log line no payload consumer can read (INV-2,
    # ``structured-facts-at-failure``).  ``skipped`` is that channel.

    def test_skipped_out_parameter_records_unparseable_files(self, tmp_path):
        """An opted-in caller learns WHICH file was dropped and WHY."""
        from dashboard.data.escalations import load_queue_escalations

        esc_dir = tmp_path / 'escalations'
        esc_dir.mkdir()
        _write_esc(esc_dir, 'esc-good-1.json', _esc('esc-good-1', task_id='1'))
        bad = esc_dir / 'esc-bad-1.json'
        bad.write_text('this is not json {{{')

        skipped: list = []
        result = load_queue_escalations(esc_dir, skipped=skipped)

        # The return value is untouched — this is a second channel, not a
        # change to what the reader yields.
        assert len(result) == 1
        assert result[0]['id'] == 'esc-good-1'
        assert len(skipped) == 1
        assert skipped[0]['path'] == bad
        assert isinstance(skipped[0]['error'], str) and skipped[0]['error']

    def test_skipped_records_os_errors_not_just_decode_errors(self, tmp_path):
        """BOTH arms of ``except (JSONDecodeError, OSError)`` report, not just JSON.

        ``Path.glob('*.json')`` yields directories too, so a directory named
        ``weird.json`` makes ``read_text()`` raise ``IsADirectoryError`` — a
        deterministic OSError needing no permission games or monkeypatching.
        A reader that only reported the decode arm would still lose every
        unreadable/permission-denied file silently, which is the more likely
        production failure of the two.
        """
        from dashboard.data.escalations import load_queue_escalations

        esc_dir = tmp_path / 'escalations'
        esc_dir.mkdir()
        _write_esc(esc_dir, 'esc-good-1.json', _esc('esc-good-1', task_id='1'))
        weird = esc_dir / 'weird.json'
        weird.mkdir()

        skipped: list = []
        result = load_queue_escalations(esc_dir, skipped=skipped)

        assert [r['id'] for r in result] == ['esc-good-1']
        assert len(skipped) == 1
        assert skipped[0]['path'] == weird
        assert isinstance(skipped[0]['error'], str) and skipped[0]['error']

    def test_skipped_accumulator_is_appended_not_replaced(self, tmp_path):
        """Entries are APPENDED, so one list can span several queue dirs.

        ``build_escalation_queues`` calls this reader once per orchestrator
        root; a caller that wants the whole fleet's skips in one list must be
        able to reuse the accumulator rather than merge N of them.
        """
        from dashboard.data.escalations import load_queue_escalations

        esc_dir = tmp_path / 'escalations'
        esc_dir.mkdir()
        bad = esc_dir / 'esc-bad-1.json'
        bad.write_text('this is not json {{{')

        sentinel = {'path': tmp_path / 'from-an-earlier-dir.json', 'error': 'earlier'}
        skipped: list = [sentinel]
        load_queue_escalations(esc_dir, skipped=skipped)

        assert len(skipped) == 2
        assert skipped[0] is sentinel
        assert skipped[1]['path'] == bad

    def test_omitting_skipped_leaves_behaviour_unchanged(self, tmp_path, caplog):
        """The default path is byte-identical to today — the back-compat pin.

        Neither of the two ``build_escalation_queues`` call sites opts in, so
        for the escalation views the WARNING log stays the only signal.  (Named
        by function, not by line: a line-number citation in this file is stale
        the moment anything above it moves — the diff that added this test
        pushed those very calls down ~28 lines.)

        This asserts the un-opted-in call still returns only the valid record,
        still does not raise, and still logs — i.e. that the new keyword is
        additive and no existing caller had to change.
        """
        from dashboard.data.escalations import load_queue_escalations

        esc_dir = tmp_path / 'escalations'
        esc_dir.mkdir()
        _write_esc(esc_dir, 'esc-good-1.json', _esc('esc-good-1', task_id='1'))
        (esc_dir / 'esc-bad-1.json').write_text('this is not json {{{')

        with caplog.at_level(logging.WARNING, logger='dashboard.data.escalations'):
            result = load_queue_escalations(esc_dir)

        assert len(result) == 1
        assert result[0]['id'] == 'esc-good-1'
        assert any('esc-bad-1' in rec.message for rec in caplog.records), \
            'the WARNING must survive for callers that do not opt in'


# ---------------------------------------------------------------------------
# Tests for resolve_owning_project — worktree-prefix arm (step 3)
# ---------------------------------------------------------------------------

class TestResolveOwningProjectWorktreeArm:
    """Tests for the worktree-prefix arm of resolve_owning_project."""

    def test_worktree_under_dot_worktrees_resolves_to_project(self, tmp_path):
        """Escalation with worktree under <root>/.worktrees/<id> resolves to root.name."""
        from dashboard.data.escalations import resolve_owning_project

        proj_a = tmp_path / 'projA'
        roots = [(proj_a, [])]

        esc = _esc('esc-1', task_id='42',
                   worktree=str(proj_a / '.worktrees' / '42'))
        result = resolve_owning_project(esc, roots)
        assert result == 'projA'

    def test_worktree_directly_under_root_resolves(self, tmp_path):
        """Worktree that starts with str(root) (not via .worktrees/) also resolves."""
        from dashboard.data.escalations import resolve_owning_project

        proj_b = tmp_path / 'projB'
        roots = [(proj_b, [])]

        esc = _esc('esc-2', task_id='55',
                   worktree=str(proj_b / 'some-subdir'))
        result = resolve_owning_project(esc, roots)
        assert result == 'projB'

    def test_dot_worktrees_prefix_form_also_matches(self, tmp_path):
        """Explicitly test the root/.worktrees/ prefix form resolves."""
        from dashboard.data.escalations import resolve_owning_project

        proj_c = tmp_path / 'projC'
        roots = [(proj_c, [])]

        # Worktree string exactly starts with str(proj_c / '.worktrees')
        wt = str(proj_c / '.worktrees' / '99')
        esc = _esc('esc-3', task_id='99', worktree=wt)
        result = resolve_owning_project(esc, roots)
        assert result == 'projC'

    def test_first_root_wins_when_multiple_could_match(self, tmp_path):
        """When multiple roots could prefix-match, the FIRST one in the list wins."""
        from dashboard.data.escalations import resolve_owning_project

        # proj_first contains proj_second as a subdirectory path-prefix-wise
        # (simulated by using parent/child paths)
        proj_first = tmp_path / 'workspace'
        proj_second = tmp_path / 'workspace' / 'sub'

        # worktree is under workspace/sub/.worktrees — both roots prefix-match
        # because proj_first path is a prefix of proj_second path
        wt = str(proj_second / '.worktrees' / '10')
        esc = _esc('esc-4', task_id='10', worktree=wt)

        roots = [(proj_first, []), (proj_second, [])]
        result = resolve_owning_project(esc, roots)
        # first root wins
        assert result == 'workspace'

    def test_no_matching_root_returns_none(self, tmp_path):
        """Worktree that doesn't match any root prefix returns None."""
        from dashboard.data.escalations import resolve_owning_project

        proj_a = tmp_path / 'projA'
        roots = [(proj_a, [])]

        esc = _esc('esc-5', task_id='7',
                   worktree='/completely/different/path')
        result = resolve_owning_project(esc, roots)
        assert result is None

    def test_missing_worktree_returns_none(self, tmp_path):
        """Escalation without worktree key returns None (no worktree arm match)."""
        from dashboard.data.escalations import resolve_owning_project

        proj_a = tmp_path / 'projA'
        roots = [(proj_a, [])]

        esc = _esc('esc-6', task_id='8')  # no worktree field
        result = resolve_owning_project(esc, roots)
        assert result is None


# ---------------------------------------------------------------------------
# Tests for resolve_owning_project — task-map probe fallback (step 5)
# ---------------------------------------------------------------------------

class TestResolveOwningProjectTaskMapArm:
    """Tests for the task-map fallback arm of resolve_owning_project."""

    def test_task_id_in_second_roots_task_map_resolves(self, tmp_path):
        """Escalation with unmatched worktree but task_id in roots[1].task_map resolves."""
        from dashboard.data.escalations import resolve_owning_project

        proj_a = tmp_path / 'projA'
        proj_b = tmp_path / 'projB'

        task_map_b = [{'id': 42, 'title': 'some task'}]
        roots = [(proj_a, []), (proj_b, task_map_b)]

        esc = _esc('esc-1', task_id='42', worktree='/no-match/path')
        result = resolve_owning_project(esc, roots)
        assert result == 'projB'

    def test_first_root_wins_when_multiple_task_maps_contain_same_task_id(self, tmp_path):
        """When multiple task maps have the same task_id, the first root wins."""
        from dashboard.data.escalations import resolve_owning_project

        proj_a = tmp_path / 'projA'
        proj_b = tmp_path / 'projB'

        task_map_a = [{'id': 42, 'title': 'task in A'}]
        task_map_b = [{'id': 42, 'title': 'task in B'}]
        roots = [(proj_a, task_map_a), (proj_b, task_map_b)]

        esc = _esc('esc-1', task_id='42')
        result = resolve_owning_project(esc, roots)
        assert result == 'projA'

    def test_no_match_in_any_task_map_returns_none(self, tmp_path):
        """task_id not in any task map and no worktree match → None."""
        from dashboard.data.escalations import resolve_owning_project

        proj_a = tmp_path / 'projA'
        roots = [(proj_a, [{'id': 99, 'title': 'other task'}])]

        esc = _esc('esc-1', task_id='55')
        result = resolve_owning_project(esc, roots)
        assert result is None

    def test_task_id_string_vs_int_coercion(self, tmp_path):
        """task_id as string '42' matches task map entry with id=42 (int)."""
        from dashboard.data.escalations import resolve_owning_project

        proj_a = tmp_path / 'projA'
        # task map has int id
        task_map_a = [{'id': 42, 'title': 'some task'}]
        roots = [(proj_a, task_map_a)]

        # esc.task_id is a string
        esc = _esc('esc-1', task_id='42')
        result = resolve_owning_project(esc, roots)
        assert result == 'projA'

    def test_no_worktree_falls_back_to_task_map(self, tmp_path):
        """Escalation without worktree field falls back to task map probe."""
        from dashboard.data.escalations import resolve_owning_project

        proj_a = tmp_path / 'projA'
        task_map_a = [{'id': 7}]
        roots = [(proj_a, task_map_a)]

        esc = _esc('esc-1', task_id='7')  # no worktree key
        result = resolve_owning_project(esc, roots)
        assert result == 'projA'


# ---------------------------------------------------------------------------
# Tests for build_escalation_queues — subsection enumeration (step 7)
# ---------------------------------------------------------------------------

class TestBuildEscalationQueuesSubsections:
    """Tests for build_escalation_queues(config) subsection shape and de-duplication."""

    def _make_config(self, tmp_path, primary: Path, extra: list[Path] | None = None):
        """Build a DashboardConfig with given project roots."""
        from dashboard.config import DashboardConfig

        return DashboardConfig(
            project_root=primary,
            known_project_roots=extra or [],
        )

    def test_orchestrator_subsections_created_per_root(self, tmp_path):
        """One orchestrator subsection per root, plus one reconciliation subsection."""
        from dashboard.data.escalations import build_escalation_queues

        primary = tmp_path / 'primary'
        reify = tmp_path / 'reify'
        primary.mkdir()
        reify.mkdir()

        # Primary escalations
        primary_esc_dir = primary / 'data' / 'escalations'
        primary_esc_dir.mkdir(parents=True)
        _write_esc(primary_esc_dir, 'esc-p1.json', _esc('esc-p1', task_id='1'))

        # Reify escalations
        reify_esc_dir = reify / 'data' / 'escalations'
        reify_esc_dir.mkdir(parents=True)
        _write_esc(reify_esc_dir, 'esc-r1.json', _esc('esc-r1', task_id='2'))

        # Reconciliation escalations
        recon_esc_dir = primary / 'data' / 'reconciliation' / 'escalations'
        recon_esc_dir.mkdir(parents=True)
        _write_esc(recon_esc_dir, 'esc-rc1.json', _esc('esc-rc1', task_id='3'))

        config = self._make_config(tmp_path, primary, [reify])
        result = build_escalation_queues(config)

        assert 'subsections' in result
        subsections = result['subsections']
        assert len(subsections) == 3

    def test_subsection_order_primary_first(self, tmp_path):
        """Primary root is first, then known_project_roots, then reconciliation."""
        from dashboard.data.escalations import build_escalation_queues

        primary = tmp_path / 'primary'
        reify = tmp_path / 'reify'
        primary.mkdir()
        reify.mkdir()
        (primary / 'data' / 'escalations').mkdir(parents=True)
        (reify / 'data' / 'escalations').mkdir(parents=True)
        (primary / 'data' / 'reconciliation' / 'escalations').mkdir(parents=True)

        config = self._make_config(tmp_path, primary, [reify])
        result = build_escalation_queues(config)
        subsections = result['subsections']

        # Use .resolve() to match what DashboardConfig.__post_init__ stores;
        # on macOS /tmp is a symlink to /private/tmp so str(primary) may differ
        # from the stored resolved path.
        assert subsections[0]['id'] == str(primary.resolve())
        assert subsections[0]['label'] == 'primary'
        assert subsections[0]['kind'] == 'orchestrator'

        assert subsections[1]['id'] == str(reify.resolve())
        assert subsections[1]['label'] == 'reify'
        assert subsections[1]['kind'] == 'orchestrator'

        assert subsections[2]['id'] == 'reconciliation'
        assert subsections[2]['label'] == 'fused-memory'
        assert subsections[2]['kind'] == 'reconciliation'

    def test_archive_excluded_from_orchestrator_subsection(self, tmp_path):
        """Files in archive/ subdir are not included in orchestrator subsection escalations."""
        from dashboard.data.escalations import build_escalation_queues

        primary = tmp_path / 'primary'
        primary.mkdir()
        esc_dir = primary / 'data' / 'escalations'
        esc_dir.mkdir(parents=True)

        # Root-level — should appear
        _write_esc(esc_dir, 'esc-root.json', _esc('esc-root', task_id='1'))

        # Archive — should NOT appear
        archive_dir = esc_dir / 'archive' / '2026-05-27'
        archive_dir.mkdir(parents=True)
        _write_esc(archive_dir, 'esc-old.json', _esc('esc-old', task_id='99'))

        (primary / 'data' / 'reconciliation' / 'escalations').mkdir(parents=True)
        config = self._make_config(tmp_path, primary)
        result = build_escalation_queues(config)

        orch = result['subsections'][0]
        ids = {e['id'] for e in orch['escalations']}
        assert 'esc-root' in ids
        assert 'esc-old' not in ids

    def test_dedup_primary_not_repeated_in_known_roots(self, tmp_path):
        """If config.project_root also appears in known_project_roots, it appears once."""
        from dashboard.data.escalations import build_escalation_queues

        primary = tmp_path / 'primary'
        primary.mkdir()
        (primary / 'data' / 'escalations').mkdir(parents=True)
        (primary / 'data' / 'reconciliation' / 'escalations').mkdir(parents=True)

        # known_project_roots includes the primary root — should be de-duped
        config = self._make_config(tmp_path, primary, [primary])
        result = build_escalation_queues(config)

        orchestrator_subsections = [s for s in result['subsections'] if s['kind'] == 'orchestrator']
        ids = [s['id'] for s in orchestrator_subsections]
        # Use .resolve() to match the stored resolved path (DashboardConfig resolves roots).
        assert ids.count(str(primary.resolve())) == 1, 'Primary root should appear exactly once'

    def test_each_subsection_has_escalations_list(self, tmp_path):
        """Each subsection carries an 'escalations' list."""
        from dashboard.data.escalations import build_escalation_queues

        primary = tmp_path / 'primary'
        primary.mkdir()
        esc_dir = primary / 'data' / 'escalations'
        esc_dir.mkdir(parents=True)
        _write_esc(esc_dir, 'esc-1.json', _esc('esc-1'))
        (primary / 'data' / 'reconciliation' / 'escalations').mkdir(parents=True)

        config = self._make_config(tmp_path, primary)
        result = build_escalation_queues(config)

        for sub in result['subsections']:
            assert 'escalations' in sub
            assert isinstance(sub['escalations'], list)

    def test_orchestrator_subsection_reports_skipped_files(self, tmp_path):
        """A corrupt queue file is reported in the subsection's ``skipped`` list.

        INV-2 (``structured-facts-at-failure``): a queue that reports fewer
        escalations than it holds must say so in the payload, not only in a
        WARNING line a human tailing stderr may never see.
        """
        from dashboard.data.escalations import build_escalation_queues

        primary = tmp_path / 'primary'
        primary.mkdir()
        esc_dir = primary / 'data' / 'escalations'
        esc_dir.mkdir(parents=True)
        _write_esc(esc_dir, 'esc-good.json', _esc('esc-good'))
        (esc_dir / 'esc-bad.json').write_text('{not json')
        (primary / 'data' / 'reconciliation' / 'escalations').mkdir(parents=True)

        config = self._make_config(tmp_path, primary)
        result = build_escalation_queues(config)

        primary_sub = next(s for s in result['subsections'] if s['id'] == str(primary.resolve()))

        assert 'skipped' in primary_sub, (
            'each subsection must carry a `skipped` list — pass a fresh accumulator '
            'to load_queue_escalations and stringify its records into the subsection'
        )
        assert isinstance(primary_sub['skipped'], list)
        assert len(primary_sub['skipped']) == 1, (
            f"expected exactly one skip record, got {primary_sub['skipped']!r}"
        )

        entry = primary_sub['skipped'][0]
        assert set(entry.keys()) == {'path', 'error'}, (
            "skip records keep the reader's own {'path', 'error'} shape — no "
            'renamed or extra fields'
        )
        assert isinstance(entry['path'], str), (
            '`path` must be stringified at this payload boundary — a Path reaching '
            'JSONResponse would 500 the endpoint'
        )
        assert entry['path'].endswith('esc-bad.json'), (
            'the record must name the FILE that was dropped, not its directory'
        )
        assert isinstance(entry['error'], str) and entry['error'], (
            '`error` must be a non-empty str naming why the file could not be read'
        )

        # A skip must not drop readable records.
        assert [e['id'] for e in primary_sub['escalations']] == ['esc-good'], (
            'the good escalation must still be returned alongside the skip report'
        )

    def test_skipped_is_per_subsection_not_shared(self, tmp_path):
        """Each subsection gets its OWN skipped list — no shared accumulator.

        ``load_queue_escalations`` **appends** to the accumulator it is handed,
        so one list shared across every call site would attribute every queue's
        skips to every subsection: a single corrupt file in one orchestrator's
        queue would render as N badges across N unrelated projects.  That is a
        worse lie than the current silence.
        """
        from dashboard.data.escalations import build_escalation_queues

        primary = tmp_path / 'primary'
        reify = tmp_path / 'reify'
        primary.mkdir()
        reify.mkdir()

        primary_esc = primary / 'data' / 'escalations'
        primary_esc.mkdir(parents=True)
        _write_esc(primary_esc, 'esc-p1.json', _esc('esc-p1'))
        (primary_esc / 'esc-bad.json').write_text('{not json')

        reify_esc = reify / 'data' / 'escalations'
        reify_esc.mkdir(parents=True)
        _write_esc(reify_esc, 'esc-r1.json', _esc('esc-r1'))

        recon_dir = primary / 'data' / 'reconciliation' / 'escalations'
        recon_dir.mkdir(parents=True)
        _write_esc(recon_dir, 'esc-rc1.json', _esc('esc-rc1'))

        config = self._make_config(tmp_path, primary, [reify])
        result = build_escalation_queues(config)

        primary_id = str(primary.resolve())
        primary_sub = next(s for s in result['subsections'] if s['id'] == primary_id)
        others = [s for s in result['subsections'] if s['id'] != primary_id]

        assert len(others) == 2, 'expected the reify + reconciliation subsections'
        assert len(primary_sub['skipped']) == 1
        assert primary_sub['skipped'][0]['path'].endswith('esc-bad.json')
        for sub in others:
            assert sub['skipped'] == [], (
                f"subsection {sub['id']!r} must not inherit another queue's skips — "
                'pass a FRESH list per load_queue_escalations call site'
            )

    def test_reconciliation_subsection_reports_skipped_files(self, tmp_path):
        """The reconciliation queue reports its own skips, one entry per file."""
        from dashboard.data.escalations import build_escalation_queues

        primary = tmp_path / 'primary'
        primary.mkdir()
        esc_dir = primary / 'data' / 'escalations'
        esc_dir.mkdir(parents=True)
        _write_esc(esc_dir, 'esc-p1.json', _esc('esc-p1'))

        recon_dir = primary / 'data' / 'reconciliation' / 'escalations'
        recon_dir.mkdir(parents=True)
        (recon_dir / 'esc-bad-1.json').write_text('{not json')
        (recon_dir / 'esc-bad-2.json').write_text('also not json')

        config = self._make_config(tmp_path, primary)
        result = build_escalation_queues(config)

        recon_sub = next(s for s in result['subsections'] if s['id'] == 'reconciliation')
        orch_subs = [s for s in result['subsections'] if s['kind'] == 'orchestrator']

        assert len(recon_sub['skipped']) == 2, (
            'one skip record per unreadable file, not one per queue'
        )
        assert {Path(e['path']).name for e in recon_sub['skipped']} == {
            'esc-bad-1.json', 'esc-bad-2.json',
        }
        for sub in orch_subs:
            assert sub['skipped'] == [], (
                "an orchestrator subsection must not inherit the reconciliation queue's skips"
            )

    def test_skipped_reports_os_errors_end_to_end(self, tmp_path):
        """The ``OSError`` arm reaches the payload too, not just ``JSONDecodeError``.

        Every other test here corrupts a file's *content*, exercising only the
        decode arm of the reader's ``except (JSONDecodeError, OSError)``.  The
        OSError arm is the one likelier to hit a whole directory at once
        (permission fault, a file vanishing mid-scan, a truncated mount), so if
        only the decode arm reached the payload the worst real failure would
        still be silent.

        A directory named ``weird.json`` makes ``read_text()`` raise
        ``IsADirectoryError`` — a deterministic OSError needing no chmod games
        (which no-op under root) or monkeypatching.  Same device as
        ``TestLoadQueueEscalations::test_skipped_records_os_errors_not_just_decode_errors``,
        here driven end-to-end through ``build_escalation_queues``.
        """
        from dashboard.data.escalations import build_escalation_queues

        primary = tmp_path / 'primary'
        primary.mkdir()
        esc_dir = primary / 'data' / 'escalations'
        esc_dir.mkdir(parents=True)
        _write_esc(esc_dir, 'esc-p1.json', _esc('esc-p1'))
        (esc_dir / 'weird.json').mkdir()
        (primary / 'data' / 'reconciliation' / 'escalations').mkdir(parents=True)

        config = self._make_config(tmp_path, primary)
        result = build_escalation_queues(config)

        primary_sub = result['subsections'][0]
        assert [e['id'] for e in primary_sub['escalations']] == ['esc-p1'], (
            'an unreadable entry must not drop the readable records beside it'
        )
        assert len(primary_sub['skipped']) == 1
        entry = primary_sub['skipped'][0]
        assert Path(entry['path']).name == 'weird.json'
        assert isinstance(entry['error'], str) and entry['error'], (
            'an OSError skip must carry a non-empty cause, same as a decode skip'
        )
        assert primary_sub['summary']['skipped_count'] == 1

    def test_payload_with_skips_is_json_serializable(self, tmp_path):
        """The built payload survives ``json.dumps`` — the stringify claim, pinned.

        Three docstrings justify stringifying ``path`` at this boundary with "a
        ``Path`` reaching ``JSONResponse`` would 500 the endpoint".  That claim
        was asserted only indirectly (``isinstance(path, str)``); this pins it
        directly, through the shaper the API layer actually calls, so a future
        refactor that lets a ``Path`` back into the payload fails here rather
        than at runtime on the one poll where a queue file is corrupt.
        """
        import json as _json

        from dashboard.data.escalations import build_escalation_queues
        from dashboard.data.redux_api import shape_escalations

        primary = tmp_path / 'primary'
        primary.mkdir()
        esc_dir = primary / 'data' / 'escalations'
        esc_dir.mkdir(parents=True)
        _write_esc(esc_dir, 'esc-p1.json', _esc('esc-p1'))
        (esc_dir / 'esc-bad-1.json').write_text('{not json')
        recon_dir = primary / 'data' / 'reconciliation' / 'escalations'
        recon_dir.mkdir(parents=True)
        (recon_dir / 'esc-bad-2.json').write_text('also not json')

        config = self._make_config(tmp_path, primary)
        queues = build_escalation_queues(config)

        # Raw builder output serializes...
        _json.dumps(queues)
        # ...and so does what the API layer actually hands JSONResponse.
        shaped = shape_escalations(queues, {})
        encoded = _json.dumps(shaped)

        assert 'esc-bad-1.json' in encoded and 'esc-bad-2.json' in encoded, (
            'the serialized payload must still name the unreadable files'
        )
        assert shaped['ESCALATIONS']['summary']['skipped_count'] == 2


# ---------------------------------------------------------------------------
# Tests for build_escalation_queues — summary counts (step 9)
# ---------------------------------------------------------------------------

class TestBuildEscalationQueuesSummary:
    """Tests for per-subsection and top-level summary bucketing."""

    def _make_config(self, tmp_path, primary: Path, extra: list[Path] | None = None):
        from dashboard.config import DashboardConfig

        return DashboardConfig(
            project_root=primary,
            known_project_roots=extra or [],
        )

    def test_per_subsection_summary_shape(self, tmp_path):
        """Each subsection has a summary dict with by_level and by_status keys."""
        from dashboard.data.escalations import build_escalation_queues

        primary = tmp_path / 'primary'
        primary.mkdir()
        esc_dir = primary / 'data' / 'escalations'
        esc_dir.mkdir(parents=True)
        _write_esc(esc_dir, 'esc-1.json', _esc('esc-1', level=0, status='pending'))
        (primary / 'data' / 'reconciliation' / 'escalations').mkdir(parents=True)

        config = self._make_config(tmp_path, primary)
        result = build_escalation_queues(config)

        for sub in result['subsections']:
            assert 'summary' in sub
            s = sub['summary']
            assert 'by_level' in s
            assert 'by_status' in s
            assert set(s['by_level'].keys()) == {0, 1, 2}
            assert set(s['by_status'].keys()) == {'pending', 'resolved', 'dismissed'}

    def test_per_subsection_counts_correct(self, tmp_path):
        """Subsection summary counts only its own escalations."""
        from dashboard.data.escalations import build_escalation_queues

        primary = tmp_path / 'primary'
        reify = tmp_path / 'reify'
        primary.mkdir()
        reify.mkdir()

        primary_esc = primary / 'data' / 'escalations'
        primary_esc.mkdir(parents=True)
        _write_esc(primary_esc, 'esc-p1.json', _esc('esc-p1', level=0, status='pending'))
        _write_esc(primary_esc, 'esc-p2.json', _esc('esc-p2', level=1, status='resolved'))
        _write_esc(primary_esc, 'esc-p3.json', _esc('esc-p3', level=2, status='dismissed'))

        reify_esc = reify / 'data' / 'escalations'
        reify_esc.mkdir(parents=True)
        _write_esc(reify_esc, 'esc-r1.json', _esc('esc-r1', level=1, status='pending'))

        (primary / 'data' / 'reconciliation' / 'escalations').mkdir(parents=True)

        config = self._make_config(tmp_path, primary, [reify])
        result = build_escalation_queues(config)

        # Use .resolve() to match the stored resolved path (DashboardConfig resolves roots).
        primary_sub = next(s for s in result['subsections'] if s['id'] == str(primary.resolve()))
        reify_sub = next(s for s in result['subsections'] if s['id'] == str(reify.resolve()))

        # primary: 1 at level 0, 1 at level 1, 1 at level 2
        assert primary_sub['summary']['by_level'][0] == 1
        assert primary_sub['summary']['by_level'][1] == 1
        assert primary_sub['summary']['by_level'][2] == 1

        # primary: 1 pending, 1 resolved, 1 dismissed
        assert primary_sub['summary']['by_status']['pending'] == 1
        assert primary_sub['summary']['by_status']['resolved'] == 1
        assert primary_sub['summary']['by_status']['dismissed'] == 1

        # reify: 1 at level 1 only
        assert reify_sub['summary']['by_level'][0] == 0
        assert reify_sub['summary']['by_level'][1] == 1

    def test_top_level_summary_aggregates_all_subsections(self, tmp_path):
        """Top-level summary aggregates counts from ALL subsections."""
        from dashboard.data.escalations import build_escalation_queues

        primary = tmp_path / 'primary'
        primary.mkdir()
        esc_dir = primary / 'data' / 'escalations'
        esc_dir.mkdir(parents=True)
        _write_esc(esc_dir, 'esc-1.json', _esc('esc-1', level=0, status='pending'))
        _write_esc(esc_dir, 'esc-2.json', _esc('esc-2', level=1, status='resolved'))

        recon_dir = primary / 'data' / 'reconciliation' / 'escalations'
        recon_dir.mkdir(parents=True)
        _write_esc(recon_dir, 'esc-rc1.json', _esc('esc-rc1', level=2, status='dismissed'))

        config = self._make_config(tmp_path, primary)
        result = build_escalation_queues(config)

        assert 'summary' in result
        top = result['summary']
        assert top['by_level'][0] == 1
        assert top['by_level'][1] == 1
        assert top['by_level'][2] == 1
        assert top['by_status']['pending'] == 1
        assert top['by_status']['resolved'] == 1
        assert top['by_status']['dismissed'] == 1

    def test_unknown_level_and_status_excluded_from_buckets_but_in_list(self, tmp_path):
        """Escalations with unknown level/status are excluded from counts but stay in escalations list."""
        from dashboard.data.escalations import build_escalation_queues

        primary = tmp_path / 'primary'
        primary.mkdir()
        esc_dir = primary / 'data' / 'escalations'
        esc_dir.mkdir(parents=True)

        esc_valid = _esc('esc-valid', level=0, status='pending')
        esc_unknown = {'id': 'esc-unknown', 'task_id': '99', 'level': 3, 'status': 'weird'}
        _write_esc(esc_dir, 'esc-valid.json', esc_valid)
        _write_esc(esc_dir, 'esc-unknown.json', esc_unknown)

        (primary / 'data' / 'reconciliation' / 'escalations').mkdir(parents=True)

        config = self._make_config(tmp_path, primary)
        result = build_escalation_queues(config)

        primary_sub = result['subsections'][0]

        # Both escalations still in the list
        ids = {e['id'] for e in primary_sub['escalations']}
        assert 'esc-valid' in ids
        assert 'esc-unknown' in ids

        # Only the valid one counted
        assert primary_sub['summary']['by_level'][0] == 1
        assert sum(primary_sub['summary']['by_level'].values()) == 1
        assert primary_sub['summary']['by_status']['pending'] == 1
        assert sum(primary_sub['summary']['by_status'].values()) == 1

    def test_per_subsection_summary_carries_skipped_count(self, tmp_path):
        """``skipped_count`` sits beside the counts it explains, and does not inflate them.

        The summary dict is precisely "the per-level/per-status counts that may
        be quietly low"; the skip count is the honest annotation on those counts,
        read by the same consumers at the same nesting.
        """
        from dashboard.data.escalations import build_escalation_queues

        primary = tmp_path / 'primary'
        primary.mkdir()
        esc_dir = primary / 'data' / 'escalations'
        esc_dir.mkdir(parents=True)
        _write_esc(esc_dir, 'esc-good.json', _esc('esc-good', level=1, status='pending'))
        (esc_dir / 'esc-bad.json').write_text('{not json')
        (primary / 'data' / 'reconciliation' / 'escalations').mkdir(parents=True)

        config = self._make_config(tmp_path, primary)
        result = build_escalation_queues(config)

        primary_sub = next(s for s in result['subsections'] if s['id'] == str(primary.resolve()))
        recon_sub = next(s for s in result['subsections'] if s['id'] == 'reconciliation')

        assert primary_sub['summary']['skipped_count'] == 1, (
            'the summary must report how many files this queue could not read'
        )
        assert recon_sub['summary']['skipped_count'] == 0, (
            "a clean queue's skipped_count is 0, not another queue's count"
        )

        # The skip must NOT inflate the level/status counts it qualifies.
        assert sum(primary_sub['summary']['by_level'].values()) == 1
        assert primary_sub['summary']['by_level'][1] == 1
        assert sum(primary_sub['summary']['by_status'].values()) == 1
        assert primary_sub['summary']['by_status']['pending'] == 1

    def test_top_level_summary_aggregates_skipped_count(self, tmp_path):
        """The top-level rollup sums every subsection's skips."""
        from dashboard.data.escalations import build_escalation_queues

        primary = tmp_path / 'primary'
        primary.mkdir()
        esc_dir = primary / 'data' / 'escalations'
        esc_dir.mkdir(parents=True)
        _write_esc(esc_dir, 'esc-good.json', _esc('esc-good'))
        (esc_dir / 'esc-bad.json').write_text('{not json')

        recon_dir = primary / 'data' / 'reconciliation' / 'escalations'
        recon_dir.mkdir(parents=True)
        (recon_dir / 'esc-bad-rc.json').write_text('also not json')

        config = self._make_config(tmp_path, primary)
        result = build_escalation_queues(config)

        assert result['summary']['skipped_count'] == 2, (
            'the top-level summary must aggregate skipped_count across every '
            'subsection, through the same _merge_summaries path the level/status '
            'counts already take'
        )

    def test_skipped_count_zero_when_all_files_readable(self, tmp_path):
        """``skipped_count`` is always present — never conditionally absent.

        A missing key reads as "unknown" and forces every consumer into a
        ``.get(..., 0)`` guess; an explicit 0 states the fact.
        """
        from dashboard.data.escalations import build_escalation_queues

        primary = tmp_path / 'primary'
        reify = tmp_path / 'reify'
        primary.mkdir()
        reify.mkdir()

        esc_dir = primary / 'data' / 'escalations'
        esc_dir.mkdir(parents=True)
        _write_esc(esc_dir, 'esc-good.json', _esc('esc-good'))
        (reify / 'data' / 'escalations').mkdir(parents=True)
        (primary / 'data' / 'reconciliation' / 'escalations').mkdir(parents=True)

        config = self._make_config(tmp_path, primary, [reify])
        result = build_escalation_queues(config)

        for sub in result['subsections']:
            assert sub['summary']['skipped_count'] == 0, (
                f"subsection {sub['id']!r} has no unreadable files — skipped_count "
                'must be present and 0'
            )
        assert result['summary']['skipped_count'] == 0


# ---------------------------------------------------------------------------
# Regression tests for resolve_owning_project — worktree false-positive
# prefix matching (step 11)
# ---------------------------------------------------------------------------

class TestResolveOwningProjectPrefixRegression:
    """Regression tests locking the fix for str.startswith false-positive matches.

    The original implementation used raw ``str.startswith`` for path prefix
    matching.  This causes two classes of false positives:

    1. A root named ``workspace`` incorrectly matches a worktree under a
       *sibling* root ``workspace-2`` because the string ``".../workspace-2/..."``
       starts with ``".../workspace"``.
    2. A root's ``.worktrees`` prefix incorrectly matches paths under a
       ``.worktrees-archive`` sibling directory.

    All three tests below FAIL against the pre-fix ``str.startswith``
    implementation and PASS after the fix (``Path.is_relative_to``).
    """

    def test_sibling_root_prefix_does_not_false_match(self, tmp_path):
        """workspace-2 worktree must resolve to workspace-2, not workspace.

        Pre-fix bug: ``str(tmp_path/'workspace-2'/'.worktrees'/'42').startswith(
        str(tmp_path/'workspace'))`` is True because the string
        ".../workspace-2/..." starts with ".../workspace".
        """
        from dashboard.data.escalations import resolve_owning_project

        ws = tmp_path / 'workspace'
        ws2 = tmp_path / 'workspace-2'
        # workspace FIRST so the first-hit-wins rule amplifies the bug
        roots = [(ws, []), (ws2, [])]

        wt = str(ws2 / '.worktrees' / '42')
        esc = _esc('esc-reg-1', task_id='42', worktree=wt)
        result = resolve_owning_project(esc, roots)
        assert result == 'workspace-2', (
            f"Expected 'workspace-2' but got {result!r} — "
            "sibling-prefix false positive not fixed"
        )

    def test_unrelated_sibling_worktree_returns_none(self, tmp_path):
        """Worktree under workspace-extra must NOT match root workspace.

        Pre-fix bug: ``".../workspace-extra/...".startswith(".../workspace")``
        is True.
        """
        from dashboard.data.escalations import resolve_owning_project

        ws = tmp_path / 'workspace'
        ws_extra = tmp_path / 'workspace-extra'
        roots = [(ws, [])]

        wt = str(ws_extra / '.worktrees' / '42')
        esc = _esc('esc-reg-2', task_id='42', worktree=wt)
        result = resolve_owning_project(esc, roots)
        assert result is None, (
            f"Expected None but got {result!r} — "
            "sibling-prefix false positive not fixed"
        )

    def test_dot_worktrees_archive_suffix_does_not_false_match(self, tmp_path):
        """A sibling dir named .worktrees-archive must NOT match a root named .worktrees.

        The root here is ``projA/.worktrees`` (the worktrees directory itself).
        Its sibling ``projA/.worktrees-archive`` shares the string prefix
        ``.../projA/.worktrees`` but is NOT a child of root — it is a sibling.

        Pre-fix bug: ``str(projA/'.worktrees-archive'/'42').startswith(
        str(projA/'.worktrees'))`` is True because the raw string
        ``.worktrees-archive`` starts with ``.worktrees``.

        Post-fix: ``Path(...).is_relative_to(Path(projA/'.worktrees'))`` is False
        because ``.worktrees-archive`` is a different path component from ``.worktrees``.
        """
        from dashboard.data.escalations import resolve_owning_project

        proj_a = tmp_path / 'projA'
        # The ROOT is the .worktrees dir itself — the path to resolve against.
        worktrees_root = proj_a / '.worktrees'
        roots = [(worktrees_root, [])]

        # Sibling of worktrees_root: .worktrees-archive (same level, different name).
        # Its name string-starts-with ".worktrees" but it is NOT under worktrees_root.
        wt = str(proj_a / '.worktrees-archive' / '42')
        esc = _esc('esc-reg-3', task_id='42', worktree=wt)
        result = resolve_owning_project(esc, roots)
        assert result is None, (
            f"Expected None but got {result!r} — "
            ".worktrees-archive false-matched .worktrees root via string prefix"
        )
