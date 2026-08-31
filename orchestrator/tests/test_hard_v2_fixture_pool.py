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

# Matches the parents[2] convention used by conftest.py and the sibling
# evals tests (test_eval_bootstrap_smoke.py). Defined locally rather than
# imported: the bare `conftest` module name collides across subprojects in
# sys.modules under --import-mode=importlib.
REPO_ROOT = Path(__file__).resolve().parents[2]

EVALS_DIR = REPO_ROOT / 'orchestrator' / 'src' / 'orchestrator' / 'evals'
POOL_DIR = EVALS_DIR / 'tasks_hard_v2'
META_DIR = POOL_DIR / '_meta'
CURATION_JSON = META_DIR / 'curation.json'
CURATION_MD = POOL_DIR / 'CURATION.md'
STANDING_TASKS_DIR = EVALS_DIR / 'tasks'

# The recorded census: the pool's provenance, per project. Deliberately a
# literal HERE and nowhere else — this is the PRD-recorded expectation, the
# independent external pin the manifest is checked against. The driver no
# longer carries its own copy: `expected_census_counts()` reads the committed
# manifest, so a drifted db is measured against the artifact rather than
# against a second literal that could disagree with it.
EXPECTED_CENSUS_COUNTS = {'reify': 36, 'dark_factory': 4, 'know_live': 1}
EXPECTED_TOTAL = 41

# The baseline-ladder rungs a CENSUS candidate's pre_task_commit can come from.
BASELINE_RUNGS = {'merge_first_parent', 'status_autocommit', 'timestamp_walk'}

# A CONTINUITY fixture's baseline is not resolved by the ladder at all — it is
# inherited verbatim from the canonical fixture under evals/tasks/, whose own
# pre/post predate the ladder (df_task_18's pre is not its post's first parent,
# so claiming `merge_first_parent` would be a false provenance).
CONTINUITY_RUNG = 'standing_fixture_inherited'

# The three PRD-named back-filled continuity fixtures. They are deliberately
# re-banded into the v2 cohort under their EXISTING ids, so these three ids are
# the one intended overlap with the standing corpus.
CONTINUITY_IDS = ('reify_task_12', 'reify_task_27', 'df_task_18')


def _fixture_id(row: dict) -> str:
    """The fixture id a manifest candidate row mints to.

    Derived through ``task_sampler.repo_of_project`` — the SAME table
    ``build_fixture_record`` derives the real id from — rather than a
    project→repo literal re-declared in this file, which could drift from it
    without any test noticing.
    """
    from orchestrator.evals.task_sampler import repo_of_project
    return f'{repo_of_project(row["project"])}_task_{row["task_id"]}'


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
                assert c['baseline_source'] in BASELINE_RUNGS, c['task_id']

    def test_the_two_cancelled_reify_tasks_are_adjudicated(
        self, manifest: dict,
    ) -> None:
        by_id = {
            (c['project'], str(c['task_id'])): c for c in manifest['candidates']
        }
        for tid in ('3378', '3586'):
            row = by_id[('reify', tid)]
            assert row['status'] == 'cancelled', tid
            assert row['reason'].strip(), tid
        # The pair is SPLIT, and the split is the invariant worth pinning:
        # `cancelled` alone does not decide membership, so each was adjudicated
        # against its own escalation history. 3378 was abandoned BECAUSE it was
        # ill-posed (its required signature referenced types absent from main),
        # so an exhaustion there measures the spec rather than the model — a
        # confound, excluded. 3586 was abandoned on cost while its brief named
        # concrete deliverables, i.e. a genuine hard task, so it is included.
        assert by_id[('reify', '3378')]['decision'] == 'exclude'
        assert by_id[('reify', '3586')]['decision'] == 'include'

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
# The availability summary — derived from the rows, never hand-authored
# ---------------------------------------------------------------------------

class TestMergeShaAvailability:
    def test_availability_block_is_derived_from_the_committed_rows(
        self, manifest: dict,
    ) -> None:
        # Total equality against the SHARED derivation: no number and no claim
        # in the committed summary can disagree with the rows it summarises.
        # Substring checks could not carry this — the shipped block said
        # "22/41" and "17 of 41" (which do not even sum to their own stated
        # denominator) beside rows reading referenced=20, planrate_only=19
        # over 39 included candidates, and called the SPLIT set a MAJORITY
        # when it is now the smaller of the two.
        driver = _load_driver()
        assert manifest['merge_sha_availability'] == \
            driver.merge_sha_availability_block(manifest['candidates'])


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

    def test_renders_the_availability_finding(self, manifest: dict) -> None:
        # Round-trip only: the derived sentence reaches the human artifact.
        # Its SEMANTICS are carried by
        # TestMergeShaAvailability::test_availability_block_is_derived_from_the_committed_rows
        # — substring presence could never check that the sentence is true.
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
        # No orphan fixture, no unminted include. The pool is exactly the
        # manifest's included census candidates PLUS its continuity block —
        # the two are the only sources a fixture may come from.
        expected = {
            _fixture_id(c) for c in manifest['candidates']
            if c['decision'] == 'include'
        }
        expected |= {e['id'] for e in manifest['continuity']['fixtures']}
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
        # Census fixtures name the ladder rung that produced their baseline;
        # a continuity fixture names the distinct `inherited` provenance
        # instead of falsely claiming a ladder rung it never ran.
        for f in pool:
            assert f['provenance']['baseline_source'] in (
                BASELINE_RUNGS | {CONTINUITY_RUNG}
            ), f['id']

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

    def test_planrate_only_fixtures_claim_no_landed_verify_outcome(
        self, pool: list[dict], manifest: dict,
    ) -> None:
        # The same reasoning as the popped `reference`, applied to the gate
        # result. `{source:'landed', passed:True}` asserts "the task merged to
        # main ⇒ its gates passed at the post commit" — but a planRate-only
        # fixture HAS no post commit (and reify_task_3586 was cancelled and
        # never landed at all), so that claim would be a fabricated ground
        # truth shipped on a majority of the pool.
        by_id = _mint_modes(manifest)
        for f in pool:
            if by_id.get(f['id']) != 'planrate_only':
                continue
            outcome = f['verify_outcome']
            assert outcome['source'] == 'unavailable', f['id']
            assert outcome['passed'] is None, f['id']
            assert outcome['reason'].strip(), f['id']
            assert outcome['commands'] == f['verify_commands'], f['id']

    def test_census_fixtures_carry_their_terminal_task_status(
        self, pool: list[dict], manifest: dict,
    ) -> None:
        # The curation's adjudicated status travels WITH the fixture, so the
        # cancelled case is self-describing from the JSON alone.
        by_id = {
            _fixture_id(c): c['status'] for c in manifest['candidates']
            if c['decision'] == 'include'
        }
        for f in pool:
            if f['id'] not in by_id:
                continue  # continuity fixtures are not census candidates
            assert f['provenance']['task_status'] == by_id[f['id']], f['id']

    def test_no_fixture_ships_an_unretrievable_reference(
        self, pool: list[dict],
    ) -> None:
        # The pool's single most load-bearing durability property: a reference
        # whose post commit is GC-eligible can EVAPORATE, taking the fixture's
        # plan_quality ground truth with it. Every referenced/continuity
        # fixture in this pool shipped with `eval_branch_pinned: false` (the
        # source checkouts are read-only from here), so the only thing keeping
        # them retrievable is that the commit is reachable another way —
        # ancestry of main, or an existing ref. `_pin_or_record_failure`
        # measures that at mint time and writes the verdict into
        # `eval_branch_pin_impact`; an 'ACTION NEEDED' verdict means nothing
        # holds the commit. Refuse to ship one.
        for f in pool:
            if 'reference' not in f:
                continue  # planRate-only: nothing to retrieve, asserted elsewhere
            prov = f['provenance']
            if prov.get('eval_branch_pinned') is True:
                continue  # evals/<id> holds it directly
            impact = prov.get('eval_branch_pin_impact', '')
            assert impact.startswith('None for retrievability'), (
                f'{f["id"]} ships a reference that is neither pinned nor shown '
                f'retrievable: eval_branch_pinned='
                f'{prov.get("eval_branch_pinned")!r}, impact={impact!r}. Its '
                f'reference diff can be lost to GC, and with it the fixture\'s '
                f'plan_quality ground truth.'
            )

    def test_the_cancelled_include_asserts_no_passing_gates(
        self, pool: list[dict],
    ) -> None:
        # Named explicitly: reify_task_3586 is `status: cancelled` with an
        # empty post_task_commit. A `passed: true` here would be flatly false.
        rec = next(f for f in pool if f['id'] == 'reify_task_3586')
        assert rec['provenance']['task_status'] == 'cancelled'
        assert rec['verify_outcome']['passed'] is not True


    def test_every_fixture_declares_base_approximation(
        self, pool: list[dict],
    ) -> None:
        # A readout that cannot tell a true branch point from a guess silently
        # averages the two together. Only `merge_first_parent` (and a
        # continuity base MEASURED equal to M^1) is the real thing; everything
        # else must SAY it is an approximation so a readout can exclude it.
        for f in pool:
            prov = f['provenance']
            flag = prov.get('base_is_approximated')
            assert isinstance(flag, bool), \
                f'{f["id"]}: base_is_approximated is {flag!r}, not a bool'
            if flag:
                assert prov.get('base_approximation_reason', '').strip(), \
                    f'{f["id"]}: approximated with no reason'

    def test_merge_derived_bases_are_not_marked_approximated(
        self, pool: list[dict],
    ) -> None:
        for f in pool:
            if f['provenance']['baseline_source'] == 'merge_first_parent':
                assert f['provenance']['base_is_approximated'] is False, f['id']

    def test_reference_unavailable_only_when_no_landing_merge_exists(
        self, pool: list[dict],
    ) -> None:
        # The DEFECT-2 invariant. A fixture may only claim its reference is
        # unavailable when no single landing merge exists under EITHER
        # accepted subject spelling — the same check that produced the claim.
        driver = _load_driver()
        for f in pool:
            if 'reference_unavailable' not in f['provenance']:
                continue
            root = Path(f['project_root'])
            if not (root / '.git').exists():
                pytest.skip(f'{root} is not checked out on this machine')
            task_id = f['id'].split('_task_', 1)[1]
            assert driver.find_merge_sha(root, task_id) is None, (
                f'{f["id"]} claims reference_unavailable but a single landing '
                f'merge resolves in {root}'
            )

    def test_task_4026_resolves_its_landing_merge(
        self, pool: list[dict],
    ) -> None:
        # Acceptance criterion 3, pinned by name: reify_task_4026's landing
        # merge is 3613bea224, spelled with a colon, and was invisible to the
        # single-spelling matcher.
        by_id = {f['id']: f for f in pool}
        fixture = by_id['reify_task_4026']
        assert 'reference_unavailable' not in fixture['provenance']
        assert fixture.get('reference'), 'reify_task_4026 carries no reference'
        assert fixture['provenance']['merge_sha'].startswith('3613bea224')
        assert fixture['provenance']['baseline_source'] == 'merge_first_parent'
        assert fixture['provenance']['base_is_approximated'] is False


def _load_driver():
    """Import ``scripts/mint_hard_v2_fixtures.py`` by path (not a package).

    Same loader shape as ``test_mint_hard_v2_driver.py``: the pool invariants
    below must be checked with the SAME matcher that stamped the claims, not
    with a second copy that could drift from it.
    """
    import importlib.util
    import sys
    path = REPO_ROOT / 'scripts' / 'mint_hard_v2_fixtures.py'
    spec = importlib.util.spec_from_file_location('mint_hard_v2_fixtures', path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _mint_modes(manifest: dict) -> dict[str, str]:
    return {
        _fixture_id(c): c['mint_mode']
        for c in manifest['candidates'] if c['decision'] == 'include'
    }


# ---------------------------------------------------------------------------
# Isolation — the standing corpus is unreachable from here and unchanged
# ---------------------------------------------------------------------------

class TestStandingCorpusIsolation:
    def test_standing_corpus_still_loads_with_unique_ids(self) -> None:
        # A FLOOR, not an exact count: β1 does not own evals/tasks/ and that
        # corpus is expected to grow (ι2 back-fills references into it), so
        # pinning 22 would fail an unrelated task's legitimate addition from a
        # test named for the v2 hard pool. What β1 actually guarantees — no
        # collision with the standing corpus — is asserted by
        # `test_the_only_id_overlap_is_the_declared_continuity_set` below.
        from orchestrator.cli import _load_fixture_dir
        standing = _load_fixture_dir(STANDING_TASKS_DIR)
        assert len(standing) >= 22
        assert len({f['id'] for f in standing}) == len(standing)

    def test_the_only_id_overlap_is_the_declared_continuity_set(
        self, pool: list[dict], manifest: dict,
    ) -> None:
        # The census half of the pool must not collide with the standing
        # corpus at all. The continuity fixtures DO share their ids by
        # design — they are the same three tasks re-banded into the v2
        # cohort — so the overlap is pinned to exactly that declared set
        # rather than left as an open-ended allowance.
        from orchestrator.cli import _load_fixture_dir
        standing_ids = {f['id'] for f in _load_fixture_dir(STANDING_TASKS_DIR)}
        declared = {e['id'] for e in manifest['continuity']['fixtures']}
        overlap = standing_ids & {f['id'] for f in pool}
        assert overlap == declared

    def test_standing_corpus_carries_no_v2_cohort_fixture(self) -> None:
        from orchestrator.cli import _load_fixture_dir
        for f in _load_fixture_dir(STANDING_TASKS_DIR):
            assert f.get('cohort') != 'fable-trial-v2-hard', f['id']


# ---------------------------------------------------------------------------
# The continuity fixtures — a re-banded copy, machine-checked against drift
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def continuity(manifest: dict) -> list[dict]:
    return manifest['continuity']['fixtures']


def _pre_beta1_anchor() -> str:
    """The revision the standing corpus is compared against: this branch's
    merge-base with main, i.e. the tree before β1's first commit.

    Skips (loudly, never silently passes) when no ``main`` ref exists in the
    checkout — a shallow/detached clone cannot answer "before β1" at all, and
    a vacuous comparison against HEAD would read as a guarantee it is not.
    """
    import subprocess
    for main_ref in ('main', 'origin/main'):
        proc = subprocess.run(
            ['git', 'merge-base', 'HEAD', main_ref],
            cwd=str(REPO_ROOT), capture_output=True, text=True,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    pytest.skip(
        'no main / origin/main ref in this checkout, so the pre-β1 anchor is '
        'unavailable; the standing-corpus isolation invariant is still covered '
        'by the disjointness and no-v2-cohort tests'
    )


def _source_fixture(entry: dict) -> dict:
    return json.loads((REPO_ROOT / entry['source_path']).read_text())


def _minted(entry: dict) -> dict:
    return json.loads((POOL_DIR / f'{entry["id"]}.json').read_text())


class TestContinuityFixtures:
    def test_block_lists_exactly_the_three_prd_named_ids(
        self, continuity: list[dict],
    ) -> None:
        assert {e['id'] for e in continuity} == set(CONTINUITY_IDS)
        assert len(continuity) == len(CONTINUITY_IDS), 'no duplicate entries'

    def test_block_records_why_the_back_fill_exists(self, manifest: dict) -> None:
        # The PRD justification (re-banding under a valid reference closes the
        # v1 n=1 confound) is recorded, not implied.
        assert manifest['continuity']['rationale'].strip()

    def test_each_entry_names_a_source_under_the_standing_corpus(
        self, continuity: list[dict],
    ) -> None:
        for entry in continuity:
            src = entry['source_path']
            assert src.startswith(
                'orchestrator/src/orchestrator/evals/tasks/'
            ), src
            assert (REPO_ROOT / src).exists(), src
            assert Path(src).name == f'{entry["id"]}.json', src

    def test_each_is_minted_into_the_pool(self, continuity: list[dict]) -> None:
        for entry in continuity:
            assert (POOL_DIR / f'{entry["id"]}.json').exists(), entry['id']

    def test_lineage_fields_equal_the_source_fixture(
        self, continuity: list[dict],
    ) -> None:
        # The task's "do not duplicate content divergently" requirement, made
        # a machine check rather than a convention: a re-authored brief or a
        # re-derived baseline would make the v2 record a DIFFERENT task
        # wearing the same id, silently breaking the v1↔v2 comparison the
        # continuity set exists to enable.
        for entry in continuity:
            src, got = _source_fixture(entry), _minted(entry)
            where = entry['id']
            assert got['pre_task_commit'] == src['pre_task_commit'], where
            assert got['post_task_commit'] == src['post_task_commit'], where
            assert got['task_definition'] == src['task_definition'], where
            assert got['project'] == src['project'], where
            assert got['project_root'] == src['project_root'], where

    def test_each_carries_a_reference_captured_from_its_own_shas(
        self, continuity: list[dict],
    ) -> None:
        # Captured directly from the fixture's already-committed pre/post, so
        # β1 is self-contained and does not wait on ι2's back-fill.
        for entry in continuity:
            got = _minted(entry)
            post = got['reference']['post_task_commit']
            assert isinstance(post, str) and len(post) == 40, entry['id']
            assert post == got['post_task_commit'], entry['id']
            assert got['reference']['diff_stat']['files'] > 0, entry['id']

    def test_each_carries_the_v2_cohort_and_ceilings(
        self, continuity: list[dict], manifest: dict,
    ) -> None:
        for entry in continuity:
            got = _minted(entry)
            assert got['cohort'] == 'fable-trial-v2-hard', entry['id']
            assert got['max_architect_turns'] == 120, entry['id']
            assert got['timeout_minutes'] == manifest['ceilings']['timeout_minutes'], \
                entry['id']

    def test_provenance_names_the_source_path(
        self, continuity: list[dict],
    ) -> None:
        for entry in continuity:
            got = _minted(entry)
            assert got['provenance']['derived_from'] == entry['source_path'], \
                entry['id']
            assert got['provenance']['baseline_source'] == CONTINUITY_RUNG, \
                entry['id']

    def test_source_fixtures_are_byte_unchanged_since_before_beta1(
        self, continuity: list[dict],
    ) -> None:
        # β1 must not edit the standing corpus. Byte-level, against git — a
        # weaker "the file still parses" check would not catch a re-authored
        # brief being written back into tasks/.
        #
        # The anchor is this branch's merge-base with main, NOT HEAD. HEAD is
        # vacuous: an edit to evals/tasks/ would arrive as a COMMIT, HEAD would
        # move with it and the comparison would pass unconditionally. The
        # merge-base is the tree as it stood before β1's first commit, so the
        # assertion actually spans the work being reviewed.
        import subprocess
        anchor = _pre_beta1_anchor()
        for entry in continuity:
            src = entry['source_path']
            before = subprocess.run(
                ['git', 'show', f'{anchor}:{src}'],
                cwd=str(REPO_ROOT), capture_output=True, check=True,
            ).stdout
            assert (REPO_ROOT / src).read_bytes() == before, (
                f'{src} differs from {anchor} (this branch\'s merge-base with '
                f'main) — β1 must not touch the standing corpus'
            )

    def test_verify_outcome_provenance_matches_measured_reachability(
        self, continuity: list[dict],
    ) -> None:
        # `build_fixture_record` stamps `{source:'landed', passed:True}`
        # unconditionally, on the premise "the task merged to main ⇒ its gates
        # passed at the post commit". For these three that premise is not
        # established: their post commits are the task-BRANCH TIPS the
        # canonical fixtures recorded, and the canonical fixtures carry no
        # verify_outcome to inherit. So the mint MEASURES reachability and the
        # stamp must follow the measurement — a `passed: true` that no
        # reachable commit backs is the same fabricated ground truth the
        # planRate-only path refuses.
        for entry in continuity:
            got = _minted(entry)
            where = entry['id']
            reachable = got['provenance']['post_commit_reachable_from_main']
            outcome = got['verify_outcome']
            if reachable:
                assert outcome['source'] == 'landed', where
                assert outcome['passed'] is True, where
            else:
                assert outcome['source'] == 'landed_branch_tip', where
                assert outcome['passed'] is None, where
                assert outcome['reason'].strip(), where
                assert outcome['commands'] == got['verify_commands'], where


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
