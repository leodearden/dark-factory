"""Tests for ``scripts/mint_hard_v2_fixtures.py`` — the fable-trial-v2 β1 driver.

Hermetic: the census filter runs against a synthetic sqlite db built inline
(never the live ``data/orchestrator/runs.db``), and the baseline-resolution
ladder runs against temp git repos built per-test. No live data / no network.

``scripts/`` is not an importable package, so the driver is loaded by path via
``importlib.util.spec_from_file_location`` off the module-local ``REPO_ROOT``.
"""

from __future__ import annotations

import importlib.util
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

# Matches the parents[2] convention used by conftest.py and the sibling
# evals tests (test_eval_bootstrap_smoke.py). Defined locally rather than
# imported: the bare `conftest` module name collides across subprojects in
# sys.modules under --import-mode=importlib.
REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_driver() -> ModuleType:
    """Import ``scripts/mint_hard_v2_fixtures.py`` by path (not a package)."""
    path = REPO_ROOT / 'scripts' / 'mint_hard_v2_fixtures.py'
    spec = importlib.util.spec_from_file_location('mint_hard_v2_fixtures', path)
    assert spec is not None and spec.loader is not None, f'no import spec for {path}'
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


driver = _load_driver()


def _git(args: list[str], cwd: Path) -> str:
    """Run a git command in *cwd* and return stripped stdout."""
    return subprocess.run(
        ['git', *args], cwd=str(cwd), check=True, capture_output=True, text=True,
    ).stdout.strip()


def _init_repo(tmp_path: Path, name: str = 'repo') -> Path:
    repo = tmp_path / name
    repo.mkdir()
    _git(['init', '-q', '-b', 'main'], repo)
    _git(['config', 'user.email', 'test@example.com'], repo)
    _git(['config', 'user.name', 'Test User'], repo)
    _git(['config', 'commit.gpgsign', 'false'], repo)
    return repo


def _commit(repo: Path, filename: str, body: str, message: str,
            date: str | None = None) -> str:
    """Write *filename*, commit with *message*, return the new SHA.

    *date* (a git-parseable timestamp) pins both author and committer date so
    the timestamp-walk rung is deterministic.
    """
    (repo / filename).write_text(body)
    _git(['add', '.'], repo)
    env_args = ['commit', '-q', '-m', message]
    if date is None:
        _git(env_args, repo)
    else:
        subprocess.run(
            ['git', *env_args], cwd=str(repo), check=True, capture_output=True,
            text=True, env={
                **_base_env(), 'GIT_AUTHOR_DATE': date, 'GIT_COMMITTER_DATE': date,
            },
        )
    return _git(['rev-parse', 'HEAD'], repo)


def _base_env() -> dict[str, str]:
    import os
    return dict(os.environ)


# ---------------------------------------------------------------------------
# (a) The census filter
# ---------------------------------------------------------------------------

def _make_runs_db(tmp_path: Path, rows: list[dict]) -> Path:
    """Build a synthetic ``runs.db`` with the real ``events`` schema."""
    db = tmp_path / 'runs.db'
    conn = sqlite3.connect(db)
    conn.execute("""
        CREATE TABLE events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            run_id      TEXT    NOT NULL,
            task_id     TEXT,
            event_type  TEXT    NOT NULL,
            phase       TEXT,
            role        TEXT,
            data        TEXT    DEFAULT '{}',
            cost_usd    REAL,
            duration_ms INTEGER
        )
    """)
    for i, row in enumerate(rows):
        conn.execute(
            'INSERT INTO events (timestamp, run_id, task_id, event_type, role, '
            'data, duration_ms) VALUES (?, ?, ?, ?, ?, ?, ?)',
            (
                row.get('timestamp', f'2026-06-0{i % 9 + 1}T00:00:00+00:00'),
                row.get('run_id', f'run{i}'),
                row['task_id'],
                row.get('event_type', 'invocation_end'),
                row.get('role', 'architect'),
                row['data'],
                row.get('duration_ms'),
            ),
        )
    conn.commit()
    conn.close()
    return db


# The discriminating cases. Only the two SELECTED shapes may survive the
# filter: max_turns exhaustion AT the 121-turn ceiling, and budget exhaustion
# at ANY turn count (the 121 clause binds only the max_turns arm).
_CENSUS_ROWS = [
    # SELECTED — max_turns exhaustion at the production ceiling + 1.
    {'task_id': '100', 'data': '{"subtype": "error_max_turns", "turns": 121}',
     'duration_ms': 600_000},
    # REJECTED — max_turns exhaustion at a pre-today, lower ceiling.
    {'task_id': '101', 'data': '{"subtype": "error_max_turns", "turns": 76}'},
    {'task_id': '102', 'data': '{"subtype": "error_max_turns", "turns": 119}'},
    # SELECTED — budget exhaustion terminates at an arbitrary turn count.
    {'task_id': '103', 'data': '{"subtype": "error_max_budget_usd", "turns": 113}',
     'duration_ms': 1_200_000},
    # REJECTED — a clean architect run.
    {'task_id': '104', 'data': '{"subtype": "success", "turns": 40}'},
    # REJECTED — right shape, wrong role.
    {'task_id': '105', 'role': 'implementer',
     'data': '{"subtype": "error_max_turns", "turns": 121}'},
    # REJECTED — right shape, wrong event_type.
    {'task_id': '106', 'event_type': 'invocation_start',
     'data': '{"subtype": "error_max_turns", "turns": 121}'},
    # A SECOND exhaustion for task 100 — must dedupe to one candidate.
    {'task_id': '100', 'data': '{"subtype": "error_max_turns", "turns": 121}',
     'duration_ms': 900_000},
]


class TestCensusFilter:
    def test_selects_only_the_two_exhaustion_shapes(self, tmp_path: Path) -> None:
        db = _make_runs_db(tmp_path, _CENSUS_ROWS)
        got = driver.census_task_ids(db)
        assert got == ['100', '103']

    def test_budget_arm_carries_no_turn_condition(self, tmp_path: Path) -> None:
        # The PRD prose reads as a global "turns == 121" filter; applying it
        # globally drops the budget arm and yields 23, not 41, candidates.
        db = _make_runs_db(tmp_path, [
            {'task_id': '900',
             'data': '{"subtype": "error_max_budget_usd", "turns": 7}'},
        ])
        assert driver.census_task_ids(db) == ['900']

    def test_max_turns_arm_requires_exactly_121(self, tmp_path: Path) -> None:
        db = _make_runs_db(tmp_path, [
            {'task_id': '901', 'data': '{"subtype": "error_max_turns", "turns": 120}'},
            {'task_id': '902', 'data': '{"subtype": "error_max_turns", "turns": 122}'},
        ])
        assert driver.census_task_ids(db) == []

    def test_deduplicates_repeat_exhaustions_of_one_task(
        self, tmp_path: Path,
    ) -> None:
        db = _make_runs_db(tmp_path, _CENSUS_ROWS)
        assert driver.census_task_ids(db).count('100') == 1

    def test_opens_the_db_read_only(self, tmp_path: Path) -> None:
        # A live orchestrator must never be locked by the census. A ro URI
        # connection refuses writes, so this is the observable proof.
        db = _make_runs_db(tmp_path, _CENSUS_ROWS)
        conn = driver.connect_ro(db)
        try:
            with pytest.raises(sqlite3.OperationalError):
                conn.execute("INSERT INTO events (timestamp, run_id, event_type) "
                             "VALUES ('t', 'r', 'e')")
        finally:
            conn.close()

    def test_collects_duration_ms_over_the_census_population(
        self, tmp_path: Path,
    ) -> None:
        # Feeds the timeout_minutes derivation: the ceiling must be shown not
        # to bind, so the observed wall-clock distribution is evidence.
        db = _make_runs_db(tmp_path, _CENSUS_ROWS)
        durations = driver.census_durations_ms(db)
        assert sorted(durations) == [600_000, 900_000, 1_200_000]


# ---------------------------------------------------------------------------
# (b) The three-rung baseline-resolution ladder
# ---------------------------------------------------------------------------

class TestBaselineLadder:
    def test_rung1_returns_merge_first_parent(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        base = _commit(repo, 'a.txt', 'base\n', 'base')
        _git(['checkout', '-q', '-b', 'task/777'], repo)
        _commit(repo, 'b.txt', 'work\n', 'work on 777')
        _git(['checkout', '-q', 'main'], repo)
        _git(['merge', '--no-ff', 'task/777', '-m', 'Merge task/777 into main'], repo)

        sha, rung = driver.resolve_baseline(repo, '777', '2026-06-01T00:00:00+00:00')
        assert sha == base, 'rung 1 must return M^1 (prior main), not M'
        assert rung == 'merge_first_parent'

    def test_rung2_returns_the_in_progress_autocommit(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        _commit(repo, 'a.txt', 'base\n', 'base')
        start = _commit(
            repo, 'tasks.db', 'x\n',
            'chore(tasks): auto-commit after set_task_status(778=in-progress)',
        )
        _commit(repo, 'c.txt', 'later\n', 'unrelated later work')

        sha, rung = driver.resolve_baseline(repo, '778', '2026-06-01T00:00:00+00:00')
        assert sha == start
        assert rung == 'status_autocommit'

    def test_rung2_picks_the_earliest_in_progress_autocommit(
        self, tmp_path: Path,
    ) -> None:
        # A task can be re-started; the baseline is main at FIRST start.
        repo = _init_repo(tmp_path)
        _commit(repo, 'a.txt', 'base\n', 'base')
        first = _commit(
            repo, 'tasks.db', '1\n',
            'chore(tasks): auto-commit after set_task_status(779=in-progress)',
        )
        _commit(
            repo, 'tasks.db', '2\n',
            'chore(tasks): auto-commit after set_task_status(779=in-progress)',
        )
        sha, rung = driver.resolve_baseline(repo, '779', '2026-06-01T00:00:00+00:00')
        assert sha == first
        assert rung == 'status_autocommit'

    def test_rung2_ignores_other_status_transitions(self, tmp_path: Path) -> None:
        # Only the in-progress transition marks task start; done/blocked do not.
        repo = _init_repo(tmp_path)
        old = _commit(repo, 'a.txt', 'base\n', 'base',
                      date='2026-01-01T00:00:00+00:00')
        _commit(repo, 'tasks.db', 'x\n',
                'chore(tasks): auto-commit after set_task_status(780=done)',
                date='2026-02-01T00:00:00+00:00')
        sha, rung = driver.resolve_baseline(repo, '780', '2026-01-15T00:00:00+00:00')
        assert rung == 'timestamp_walk'
        assert sha == old

    def test_rung2_ignores_another_tasks_autocommit(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        old = _commit(repo, 'a.txt', 'base\n', 'base',
                      date='2026-01-01T00:00:00+00:00')
        _commit(repo, 'tasks.db', 'x\n',
                'chore(tasks): auto-commit after set_task_status(9999=in-progress)',
                date='2026-02-01T00:00:00+00:00')
        sha, rung = driver.resolve_baseline(repo, '781', '2026-01-15T00:00:00+00:00')
        assert rung == 'timestamp_walk'
        assert sha == old

    def test_rung3_walks_main_back_to_the_first_invocation(
        self, tmp_path: Path,
    ) -> None:
        repo = _init_repo(tmp_path)
        early = _commit(repo, 'a.txt', 'base\n', 'base',
                        date='2026-01-01T00:00:00+00:00')
        _commit(repo, 'b.txt', 'after\n', 'after the architect ran',
                date='2026-03-01T00:00:00+00:00')

        sha, rung = driver.resolve_baseline(repo, '782', '2026-02-01T00:00:00+00:00')
        assert sha == early, 'must pick the newest main commit BEFORE the ts'
        assert rung == 'timestamp_walk'

    def test_raises_when_every_rung_fails(self, tmp_path: Path) -> None:
        # Loud-over-silent: an empty pre_task_commit would blow up deep inside
        # run_architect_eval's worktree creation instead of here.
        repo = _init_repo(tmp_path)
        _commit(repo, 'a.txt', 'base\n', 'base', date='2026-03-01T00:00:00+00:00')
        with pytest.raises(driver.BaselineUnresolved) as exc:
            driver.resolve_baseline(repo, '783', '2026-01-01T00:00:00+00:00')
        assert '783' in str(exc.value)

    def test_never_returns_an_empty_sha(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        _commit(repo, 'a.txt', 'base\n', 'base', date='2026-01-01T00:00:00+00:00')
        sha, _rung = driver.resolve_baseline(repo, '784', '2026-02-01T00:00:00+00:00')
        assert len(sha) == 40 and sha.strip() == sha


# ---------------------------------------------------------------------------
# find_merge_sha — ambiguity is planRate-only, not a coin flip
# ---------------------------------------------------------------------------

class TestFindMergeSha:
    def test_returns_the_single_merge_sha(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        _commit(repo, 'a.txt', 'base\n', 'base')
        _git(['checkout', '-q', '-b', 'task/801'], repo)
        _commit(repo, 'b.txt', 'work\n', 'work')
        _git(['checkout', '-q', 'main'], repo)
        _git(['merge', '--no-ff', 'task/801', '-m', 'Merge task/801 into main'], repo)
        assert driver.find_merge_sha(repo, '801') == _git(['rev-parse', 'HEAD'], repo)

    def test_returns_none_when_split_landed(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        _commit(repo, 'a.txt', 'base\n', 'base')
        _commit(repo, 'b.txt', 'work\n', 'feat: landed directly, no merge commit')
        assert driver.find_merge_sha(repo, '802') is None

    def test_ambiguous_multi_merge_is_planrate_only(self, tmp_path: Path) -> None:
        # Two merges for one id: picking either would silently invent a
        # reference. planRate-only (None) is the recorded, honest outcome.
        repo = _init_repo(tmp_path)
        _commit(repo, 'a.txt', 'base\n', 'base')
        for n, fn in ((1, 'b.txt'), (2, 'c.txt')):
            _git(['checkout', '-q', '-b', f'task/803-{n}'], repo)
            _commit(repo, fn, f'work{n}\n', f'work {n}')
            _git(['checkout', '-q', 'main'], repo)
            _git(['merge', '--no-ff', f'task/803-{n}', '-m',
                  'Merge task/803 into main'], repo)
        assert driver.find_merge_sha(repo, '803') is None

    def test_does_not_prefix_match_a_longer_id(self, tmp_path: Path) -> None:
        # 'Merge task/8030 into main' must not answer a query for task 803.
        repo = _init_repo(tmp_path)
        _commit(repo, 'a.txt', 'base\n', 'base')
        _git(['checkout', '-q', '-b', 'task/8030'], repo)
        _commit(repo, 'b.txt', 'work\n', 'work')
        _git(['checkout', '-q', 'main'], repo)
        _git(['merge', '--no-ff', 'task/8030', '-m', 'Merge task/8030 into main'], repo)
        assert driver.find_merge_sha(repo, '803') is None


# ---------------------------------------------------------------------------
# _mint_one — a planRate-only fixture must not claim a landed verify outcome
# ---------------------------------------------------------------------------

def _planrate_row(**over: object) -> dict:
    row = {
        'task_id': '3586',
        'project': 'reify',
        'project_root': '/home/leo/src/reify',
        'title': 'PNv2 vertex-side attribute widening',
        'description': 'Widen BRepKind with Vertex and seed per-op attributes.',
        'complexity': 'complex',
        'modules': ['kernel'],
        'status': 'cancelled',
        'merge_sha': None,
        'baseline_sha': 'a' * 40,
        'baseline_source': 'timestamp_walk',
        'mint_mode': 'planrate_only',
    }
    row.update(over)
    return row


def _mint(row: dict) -> dict:
    """Mint one record. The planRate-only branch touches no git and no db."""
    import asyncio
    sampler = driver._import_sampler()
    return asyncio.run(driver._mint_one(
        sampler, row, sampled_at='2026-08-04T00:00:00+00:00', seed=3631,
        ceilings={'max_architect_turns': 120, 'timeout_minutes': 180},
    ))


class TestPlanRateOnlyVerifyOutcome:
    def test_does_not_claim_the_landed_source(self) -> None:
        # build_fixture_record stamps `{source:'landed', passed:True}`
        # unconditionally, on the premise that the task merged to main. A
        # planRate-only fixture has no landed post-commit at all, so that
        # premise — and the gate result it implies — is fabricated.
        rec = _mint(_planrate_row())
        assert rec['verify_outcome']['source'] == 'unavailable'
        assert rec['verify_outcome']['passed'] is None

    def test_records_why_the_outcome_is_unavailable(self) -> None:
        rec = _mint(_planrate_row())
        reason = rec['verify_outcome']['reason']
        assert reason.strip()
        # Self-describing: a reader of the JSON alone learns the terminal
        # status that makes the landed claim impossible.
        assert 'cancelled' in reason

    def test_carries_the_verify_commands_for_provenance(self) -> None:
        rec = _mint(_planrate_row())
        sampler = driver._import_sampler()
        assert rec['verify_outcome']['commands'] == \
            sampler.default_verify_commands('reify')

    def test_carries_the_terminal_task_status(self) -> None:
        for status in ('cancelled', 'done'):
            rec = _mint(_planrate_row(status=status))
            assert rec['provenance']['task_status'] == status

    def test_a_done_planrate_candidate_is_still_unavailable(self) -> None:
        # `done` does not rescue the claim: the gates passed at SOME commit,
        # but this fixture cannot name it, so it cannot assert one.
        rec = _mint(_planrate_row(status='done'))
        assert rec['verify_outcome']['source'] == 'unavailable'
        assert "'done'" in rec['verify_outcome']['reason']


# ---------------------------------------------------------------------------
# (d) --render — the documented regeneration path, with no db access
# ---------------------------------------------------------------------------

class TestRenderMode:
    def test_render_reproduces_the_committed_curation_md(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # `--author` re-derives the manifest from the three LIVE runs.db files
        # and refuses on any census drift, which the driver's own guidance
        # calls expected and harmless — so it cannot be the regeneration path
        # the README and the generated-file header point at. `--render` is,
        # and it reads only the committed manifest.
        import json as _json
        committed = driver.CURATION_MD.read_text()
        manifest = _json.loads(driver.CURATION_JSON.read_text())
        out = tmp_path / 'CURATION.md'
        monkeypatch.setattr(driver, 'CURATION_MD', out)
        assert driver.run_render() == 0
        assert out.read_text() == driver.render_curation_md(manifest)
        assert out.read_text() == committed

    def test_render_refuses_when_the_manifest_is_absent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Loud-over-silent: CURATION.md is generated FROM the manifest, so a
        # missing manifest is an error, never an empty/stale render.
        monkeypatch.setattr(driver, 'CURATION_JSON', tmp_path / 'nope.json')
        monkeypatch.setattr(driver, 'CURATION_MD', tmp_path / 'CURATION.md')
        with pytest.raises(RuntimeError, match='no manifest'):
            driver.run_render()


# ---------------------------------------------------------------------------
# (e) The ceilings derivation refuses to divide by absent evidence
# ---------------------------------------------------------------------------

class TestCeilingsEvidenceGuard:
    def test_missing_duration_evidence_raises_a_named_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # timeout_minutes is DERIVED: its headroom divides by the observed
        # max-at-exhaustion and the all-time architect max. A runs.db with no
        # populated duration_ms (fresh, truncated, schema-drifted) makes both
        # denominators zero, and this script's whole design is to fail loudly
        # at the missing evidence rather than emit a ZeroDivisionError from
        # inside an f-string — or, worse, a basis-free threshold.
        root = tmp_path / 'checkout'
        (root / 'data' / 'orchestrator').mkdir(parents=True)
        db = _make_runs_db(tmp_path, [
            # A census-matching exhaustion, but with NO duration recorded.
            {'task_id': '100', 'data': '{"subtype": "error_max_turns", "turns": 121}'},
        ])
        db.rename(root / 'data' / 'orchestrator' / 'runs.db')
        monkeypatch.setattr(driver, 'SOURCE_CHECKOUTS', {'reify': str(root)})
        with pytest.raises(driver.WallClockEvidenceMissing, match='wall-clock'):
            driver._build_ceilings()


# ---------------------------------------------------------------------------
# (f) Task-db enrichment degrades the same way the sampler's does
# ---------------------------------------------------------------------------

class TestEnrichFromTaskDb:
    def _cand(self) -> object:
        return driver.Candidate(
            task_id='7', project='reify', project_root='/home/leo/src/reify',
            title='pre-existing title', description='pre-existing brief',
            status='done',
        )

    def test_unreadable_db_degrades_to_stubs_instead_of_raising(
        self, tmp_path: Path,
    ) -> None:
        # A locked or schema-drifted db must not abort --mint with a raw
        # sqlite traceback; the stub is refused later, at the mint boundary,
        # where the cause can be named.
        broken = tmp_path / 'tasks.db'
        broken.write_bytes(b'not a sqlite database at all')
        cand = self._cand()
        got = driver.enrich_from_task_db([cand], broken)
        assert got == [cand]
        assert cand.title == 'pre-existing title'

    def test_a_blank_column_does_not_blank_an_existing_value(
        self, tmp_path: Path,
    ) -> None:
        # `or cand.<field>` preservation, matching
        # task_sampler.enrich_candidates_from_task_db: a NULL title in the db
        # must not overwrite a title the caller already had with ''.
        db = tmp_path / 'tasks.db'
        conn = sqlite3.connect(db)
        conn.execute(
            'CREATE TABLE tasks (id TEXT PRIMARY KEY, title TEXT, '
            'description TEXT, status TEXT, metadata TEXT)'
        )
        conn.execute(
            'INSERT INTO tasks (id, title, description, status, metadata) '
            'VALUES (?, ?, ?, ?, ?)', ('7', None, None, None, None),
        )
        conn.commit()
        conn.close()
        cand = self._cand()
        driver.enrich_from_task_db([cand], db)
        assert cand.title == 'pre-existing title'
        assert cand.description == 'pre-existing brief'
        assert cand.status == 'done'
