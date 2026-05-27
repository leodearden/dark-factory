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

        assert subsections[0]['id'] == str(primary)
        assert subsections[0]['label'] == 'primary'
        assert subsections[0]['kind'] == 'orchestrator'

        assert subsections[1]['id'] == str(reify)
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
        assert ids.count(str(primary)) == 1, 'Primary root should appear exactly once'

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

        primary_sub = next(s for s in result['subsections'] if s['id'] == str(primary))
        reify_sub = next(s for s in result['subsections'] if s['id'] == str(reify))

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
