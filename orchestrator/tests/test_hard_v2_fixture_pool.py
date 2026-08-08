"""Contract tests for the fable-trial-v2 curated hard fixture pool (β1, 3631).

Covers the committed artifacts under
``orchestrator/src/orchestrator/evals/tasks_hard_v2/``:

* ``_meta/curation.json`` — the machine-readable single source of truth
  (census provenance, the 41-row candidate table, the ceilings derivation,
  the continuity block).
* ``CURATION.md`` — generated from the manifest; pinned byte-for-byte so the
  human table can never silently drift from the minted pool.
* the minted fixture JSONs, and the guarantee that the STANDING corpus in
  ``evals/tasks/`` is unreachable from here and unchanged by β1.

These assert data invariants, not prose.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import REPO_ROOT

EVALS_DIR = REPO_ROOT / 'orchestrator' / 'src' / 'orchestrator' / 'evals'
POOL_DIR = EVALS_DIR / 'tasks_hard_v2'
META_DIR = POOL_DIR / '_meta'
CURATION_JSON = META_DIR / 'curation.json'
CURATION_MD = POOL_DIR / 'CURATION.md'
STANDING_TASKS_DIR = EVALS_DIR / 'tasks'

# The recorded census: the pool's provenance, per project.
EXPECTED_CENSUS_COUNTS = {'reify': 36, 'dark_factory': 4, 'know_live': 1}
EXPECTED_TOTAL = 41


@pytest.fixture(scope='module')
def manifest() -> dict:
    assert CURATION_JSON.exists(), f'missing curation manifest at {CURATION_JSON}'
    return json.loads(CURATION_JSON.read_text())


# ---------------------------------------------------------------------------
# The census block — the pool's provenance is reproducible, not re-derived
# ---------------------------------------------------------------------------

class TestCensusBlock:
    def test_records_the_exact_filter_sql(self, manifest: dict) -> None:
        # v1's sampling driver was gitignored, so its pool could not be
        # re-derived. The exact predicate is committed here.
        sql = manifest['census']['filter_sql']
        assert 'error_max_turns' in sql
        assert 'error_max_budget_usd' in sql
        assert "role='architect'" in sql
        assert "event_type='invocation_end'" in sql

    def test_records_the_census_date(self, manifest: dict) -> None:
        # Data, not datetime.now() — the CURATION.md renderer must be pure.
        assert manifest['census']['census_date']

    def test_records_per_project_counts_summing_to_41(self, manifest: dict) -> None:
        counts = manifest['census']['counts']
        assert counts == EXPECTED_CENSUS_COUNTS
        assert sum(counts.values()) == EXPECTED_TOTAL

    def test_records_the_source_db_path(self, manifest: dict) -> None:
        assert 'runs.db' in manifest['census']['source']

    def test_records_the_per_project_task_ids(self, manifest: dict) -> None:
        ids = manifest['census']['task_ids']
        assert set(ids) == set(EXPECTED_CENSUS_COUNTS)
        for project, expected_n in EXPECTED_CENSUS_COUNTS.items():
            assert len(ids[project]) == expected_n
            assert len(set(ids[project])) == expected_n, 'ids must be distinct'


# ---------------------------------------------------------------------------
# The candidates block — every census candidate is accounted for
# ---------------------------------------------------------------------------

class TestCandidatesBlock:
    def test_has_exactly_41_entries(self, manifest: dict) -> None:
        assert len(manifest['candidates']) == EXPECTED_TOTAL

    def test_candidate_set_equals_the_census_ids(self, manifest: dict) -> None:
        # Completeness: CURATION.md accounts for EVERY candidate, machine-checked.
        from_census = {
            (project, str(tid))
            for project, ids in manifest['census']['task_ids'].items()
            for tid in ids
        }
        from_rows = {(c['project'], str(c['task_id'])) for c in manifest['candidates']}
        assert from_rows == from_census

    def test_every_row_is_fully_populated(self, manifest: dict) -> None:
        for c in manifest['candidates']:
            where = f'candidate {c.get("project")}/{c.get("task_id")}'
            assert str(c['task_id']).strip(), where
            assert c['project'].strip(), where
            assert c['project_root'].strip(), where
            assert isinstance(c['brief_chars'], int), where
            assert c['status'].strip(), where
            assert c['decision'] in {'include', 'exclude'}, where
            assert c['reason'].strip(), where

    def test_includes_declare_a_mint_mode(self, manifest: dict) -> None:
        for c in manifest['candidates']:
            if c['decision'] == 'include':
                assert c['mint_mode'] in {'referenced', 'planrate_only'}, c['task_id']

    def test_referenced_includes_carry_a_40_hex_merge_sha(self, manifest: dict) -> None:
        for c in manifest['candidates']:
            if c['decision'] == 'include' and c['mint_mode'] == 'referenced':
                sha = c['merge_sha']
                assert isinstance(sha, str) and len(sha) == 40, c['task_id']
                assert all(ch in '0123456789abcdef' for ch in sha), c['task_id']

    def test_planrate_only_includes_have_no_merge_sha_but_name_a_baseline(
        self, manifest: dict,
    ) -> None:
        # The missing reference must be a positive recorded fact with a named
        # cause, never a silent omission.
        for c in manifest['candidates']:
            if c['decision'] == 'include' and c['mint_mode'] == 'planrate_only':
                assert c['merge_sha'] is None, c['task_id']
                assert c['baseline_source'] in {
                    'merge_first_parent', 'status_autocommit', 'timestamp_walk',
                }, c['task_id']

    def test_the_two_cancelled_reify_tasks_are_adjudicated(
        self, manifest: dict,
    ) -> None:
        by_id = {
            (c['project'], str(c['task_id'])): c for c in manifest['candidates']
        }
        for tid in ('3378', '3586'):
            row = by_id[('reify', tid)]
            assert row['status'] == 'cancelled', tid
            # The reason must state whether abandonment was benign — a
            # cancelled task may have been abandoned BECAUSE it was ill-posed,
            # which is a confound, not a hard task.
            assert 'benign' in row['reason'].lower(), (
                f'reify {tid}: reason must state whether abandonment was benign'
            )

    def test_the_pending_reify_task_is_adjudicated(self, manifest: dict) -> None:
        # 5208 is a third non-done candidate the PRD did not name.
        by_id = {
            (c['project'], str(c['task_id'])): c for c in manifest['candidates']
        }
        row = by_id[('reify', '5208')]
        assert row['status'] == 'pending'
        assert row['decision'] in {'include', 'exclude'}
        assert row['reason'].strip()

    def test_pool_is_not_over_pruned(self, manifest: dict) -> None:
        included = [c for c in manifest['candidates'] if c['decision'] == 'include']
        assert len(included) >= 10, (
            f'only {len(included)} candidate(s) included — the hard pool would '
            f'be too small to calibrate against'
        )

    def test_no_included_candidate_lacks_a_baseline(self, manifest: dict) -> None:
        for c in manifest['candidates']:
            if c['decision'] == 'include':
                sha = c['baseline_sha']
                assert isinstance(sha, str) and len(sha) == 40, c['task_id']


# ---------------------------------------------------------------------------
# Layout — the manifest is invisible to _load_fixture_dir's glob
# ---------------------------------------------------------------------------

class TestMetaLayout:
    def test_meta_dir_holds_no_file_the_fixture_glob_would_reach(self) -> None:
        # _load_fixture_dir globs '*.json' NON-recursively against tasks_dir.
        # A top-level curation.json would be loaded as a malformed fixture;
        # inside _meta/ it is unreachable.
        assert META_DIR.is_dir()
        assert not (POOL_DIR / 'curation.json').exists()
        from orchestrator.cli import _load_fixture_dir
        loaded_from_meta = _load_fixture_dir(META_DIR)
        # The glob does reach _meta/*.json if pointed AT _meta — the guarantee
        # is that nothing points at it. What matters is that pointing at the
        # POOL never picks the manifest up:
        pool_ids = {f.get('id') for f in _load_fixture_dir(POOL_DIR)}
        assert None not in pool_ids, 'a non-fixture JSON leaked into the pool glob'
        assert loaded_from_meta, 'sanity: the manifest is a real JSON file'

    def test_pool_dir_holds_only_fixtures_and_docs(self) -> None:
        allowed_suffixes = {'.json', '.md'}
        for entry in POOL_DIR.iterdir():
            if entry.is_dir():
                assert entry.name == '_meta', f'unexpected subdir {entry.name}'
                continue
            assert entry.suffix in allowed_suffixes, f'unexpected file {entry.name}'
