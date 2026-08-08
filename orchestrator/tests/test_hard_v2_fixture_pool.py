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

# ---------------------------------------------------------------------------
# The ceilings block — derived and shown not to bind, never guessed
# ---------------------------------------------------------------------------

class TestCeilings:
    def test_max_architect_turns_matches_production(self, manifest: dict) -> None:
        # 120 is know-live's production max_turns.architect — the ceiling whose
        # exhaustion at 121 turns DEFINED the census. Any other value would
        # mint a pool that cannot reproduce the failures it was selected for.
        assert manifest['ceilings']['max_architect_turns'] == 120

    def test_timeout_clears_the_prd_floor(self, manifest: dict) -> None:
        assert manifest['ceilings']['timeout_minutes'] >= 180

    def test_timeout_clears_twice_the_observed_max(self, manifest: dict) -> None:
        # The timeout must provably not bind before the turn/budget ceiling:
        # runner.py deliberately does NOT taint-exclude a timeout, so a binding
        # timeout would score a kept 0.0 and manufacture an artificial failure.
        ceilings = manifest['ceilings']
        derivation = ceilings['derivation']
        assert ceilings['timeout_minutes'] >= 2 * derivation['observed_max_minutes']

    def test_derivation_is_fully_populated(self, manifest: dict) -> None:
        derivation = manifest['ceilings']['derivation']
        for key in ('observed_max_minutes', 'p95_minutes', 'sample_n',
                    'all_architect_max_minutes', 'source'):
            assert key in derivation, key
        assert derivation['sample_n'] > 0
        assert derivation['observed_max_minutes'] > 0
        assert derivation['p95_minutes'] <= derivation['observed_max_minutes']
        assert 'duration_ms' in derivation['source']

    def test_derivation_records_the_substituted_source(self, manifest: dict) -> None:
        # The PRD names "v1 wall-clock dumps" as the source, but
        # data/eval-campaign/fable-architect-only-results.json carries NO
        # duration field. The substitution must be recorded, or the threshold
        # reads as basis-free.
        note = manifest['ceilings']['derivation']['substitution_note']
        assert 'fable-architect-only-results.json' in note
        assert 'runs.db' in note

    def test_timeout_clears_the_all_time_architect_max(self, manifest: dict) -> None:
        derivation = manifest['ceilings']['derivation']
        assert (manifest['ceilings']['timeout_minutes']
                > derivation['all_architect_max_minutes'])


# ---------------------------------------------------------------------------
# CURATION.md is generated — the human table cannot drift from the manifest
# ---------------------------------------------------------------------------

class TestCurationMdIsGenerated:
    def test_exists(self) -> None:
        assert CURATION_MD.exists(), f'missing {CURATION_MD}'

    def test_rerender_is_byte_identical(self, manifest: dict) -> None:
        # The load-bearing invariant: two hand-maintained artifacts drift
        # silently, so the table is a pure function of the manifest.
        import sys
        sys.path.insert(0, str(REPO_ROOT / 'scripts'))
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            '_mint_driver_for_render', REPO_ROOT / 'scripts' / 'mint_hard_v2_fixtures.py',
        )
        assert spec is not None and spec.loader is not None
        driver = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = driver
        spec.loader.exec_module(driver)

        assert driver.render_curation_md(manifest) == CURATION_MD.read_text()

    def test_renderer_is_pure(self, manifest: dict) -> None:
        # No wall-clock / no env, or the byte-equality test above is flaky.
        import importlib.util
        import sys
        spec = importlib.util.spec_from_file_location(
            '_mint_driver_purity', REPO_ROOT / 'scripts' / 'mint_hard_v2_fixtures.py',
        )
        assert spec is not None and spec.loader is not None
        driver = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = driver
        spec.loader.exec_module(driver)
        assert driver.render_curation_md(manifest) == driver.render_curation_md(manifest)

    def test_table_names_every_candidate(self, manifest: dict) -> None:
        text = CURATION_MD.read_text()
        for c in manifest['candidates']:
            assert f'| {c["task_id"]} |' in text, (
                f'candidate {c["project"]}/{c["task_id"]} is missing from the table'
            )

    def test_records_the_split_majority_finding(self, manifest: dict) -> None:
        text = CURATION_MD.read_text()
        assert manifest['merge_sha_availability']['finding'] in text


# ---------------------------------------------------------------------------
# The minted pool — loadable, complete, and carrying what the runner reads
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def pool() -> list[dict]:
    from orchestrator.cli import _load_fixture_dir
    return _load_fixture_dir(POOL_DIR)


class TestMintedPool:
    def test_loads_via_the_real_fixture_loader(self, pool: list[dict]) -> None:
        # The user-observable signal: the pool is loadable by the same code
        # path an eval run uses, with no ClickException.
        assert pool, 'the v2 hard pool holds no fixtures'

    def test_id_set_equals_the_manifest_includes(
        self, pool: list[dict], manifest: dict,
    ) -> None:
        # No orphan fixture, no unminted include.
        expected = set()
        for c in manifest['candidates']:
            if c['decision'] == 'include':
                repo = {'reify': 'reify', 'dark_factory': 'df',
                        'know_live': 'kl'}[c['project']]
                expected.add(f'{repo}_task_{c["task_id"]}')
        assert {f['id'] for f in pool} == expected

    def test_every_fixture_carries_what_the_runner_reads(
        self, pool: list[dict],
    ) -> None:
        for f in pool:
            where = f.get('id')
            assert f['id'], where
            assert f['project_root'], where
            # The sharp edge: run_architect_eval REQUIRES pre_task_commit to
            # create its eval worktree — it is not optional the way reference is.
            pre = f['pre_task_commit']
            assert isinstance(pre, str) and len(pre) == 40, where
            assert all(ch in '0123456789abcdef' for ch in pre), where
            assert f['task_definition']['title'], where
            assert f['task_definition']['description'], where
            assert f['verify_commands'], where

    def test_every_fixture_is_in_the_v2_cohort(self, pool: list[dict]) -> None:
        for f in pool:
            assert f['cohort'] == 'fable-trial-v2-hard', f['id']

    def test_every_fixture_names_its_baseline_rung(self, pool: list[dict]) -> None:
        for f in pool:
            assert f['provenance']['baseline_source'] in {
                'merge_first_parent', 'status_autocommit', 'timestamp_walk',
            }, f['id']

    def test_ceilings_are_pinned_fixture_side(
        self, pool: list[dict], manifest: dict,
    ) -> None:
        # runner.py reads both straight off the task record (task.get with
        # 50 / 60 defaults), so pinning them is pure data — but EVERY fixture
        # must carry them or that fixture silently runs at the wrong ceiling.
        for f in pool:
            assert f['max_architect_turns'] == 120, f['id']
            assert f['timeout_minutes'] == manifest['ceilings']['timeout_minutes'], \
                f['id']

    def test_referenced_fixtures_carry_a_real_reference(
        self, pool: list[dict], manifest: dict,
    ) -> None:
        by_id = _mint_modes(manifest)
        for f in pool:
            if by_id.get(f['id']) != 'referenced':
                continue
            post = f['reference']['post_task_commit']
            assert isinstance(post, str) and len(post) == 40, f['id']
            assert f['reference']['diff_stat']['files'] > 0, f['id']

    def test_planrate_only_fixtures_omit_reference_and_say_why(
        self, pool: list[dict], manifest: dict,
    ) -> None:
        # An empty `reference: {}` is indistinguishable from a capture that
        # silently failed. Omit the key and record the cause instead.
        by_id = _mint_modes(manifest)
        for f in pool:
            if by_id.get(f['id']) != 'planrate_only':
                continue
            assert 'reference' not in f, f['id']
            assert f['provenance']['reference_unavailable'].strip(), f['id']


def _mint_modes(manifest: dict) -> dict[str, str]:
    repo_of_project = {'reify': 'reify', 'dark_factory': 'df', 'know_live': 'kl'}
    return {
        f'{repo_of_project[c["project"]]}_task_{c["task_id"]}': c['mint_mode']
        for c in manifest['candidates'] if c['decision'] == 'include'
    }


# ---------------------------------------------------------------------------
# Isolation — the standing corpus is unreachable from here and unchanged
# ---------------------------------------------------------------------------

class TestStandingCorpusIsolation:
    def test_standing_corpus_still_holds_its_22_fixtures(self) -> None:
        from orchestrator.cli import _load_fixture_dir
        standing = _load_fixture_dir(STANDING_TASKS_DIR)
        assert len(standing) == 22
        assert len({f['id'] for f in standing}) == 22

    def test_pool_and_standing_corpus_are_disjoint(self, pool: list[dict]) -> None:
        from orchestrator.cli import _load_fixture_dir
        standing_ids = {f['id'] for f in _load_fixture_dir(STANDING_TASKS_DIR)}
        assert standing_ids.isdisjoint({f['id'] for f in pool})

    def test_standing_corpus_carries_no_v2_cohort_fixture(self) -> None:
        from orchestrator.cli import _load_fixture_dir
        for f in _load_fixture_dir(STANDING_TASKS_DIR):
            assert f.get('cohort') != 'fable-trial-v2-hard', f['id']


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
