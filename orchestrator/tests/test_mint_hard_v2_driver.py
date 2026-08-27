"""Tests for ``scripts/mint_hard_v2_fixtures.py`` — the fable-trial-v2 β1 driver.

Hermetic: the census filter runs against a synthetic sqlite db built inline
(never the live ``data/orchestrator/runs.db``), and the baseline-resolution
ladder runs against temp git repos built per-test. No live data / no network.

``scripts/`` is not an importable package, so the driver is loaded by path via
``importlib.util.spec_from_file_location`` off the module-local ``REPO_ROOT``.
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

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


def _merge(repo: Path, branch: str, message: str,
           date: str | None = None) -> str:
    """No-ff merge *branch* into the current branch; return the merge SHA.

    *date* pins author and committer date, mirroring :func:`_commit`, so a
    merge can be placed deterministically relative to the commits it brings in
    — which is what makes the off-mainline timestamp-walk shape expressible.
    """
    args = ['merge', '--no-ff', '-q', branch, '-m', message]
    if date is None:
        _git(args, repo)
    else:
        subprocess.run(
            ['git', *args], cwd=str(repo), check=True, capture_output=True,
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

    def test_rung3_never_returns_an_off_mainline_commit(
        self, tmp_path: Path,
    ) -> None:
        # The measured defect. Without --first-parent, `git rev-list -n1
        # --before=<ts> main` traverses EVERYTHING reachable from main,
        # including commits that only ever lived on a merged-in side branch.
        # It then hands back a tree state that was never a state of main.
        #
        # Reproduces reify_task_4026's committed base e21d047026 (a side-branch
        # commit, 245 commits from the true branch point, and absent from
        # `git rev-list --first-parent main`) and reify_task_4086's.
        repo = _init_repo(tmp_path)
        mainline = _commit(repo, 'a.txt', 'base\n', 'base',
                           date='2026-01-01T00:00:00+00:00')
        _git(['checkout', '-q', '-b', 'task/999'], repo)
        side = _commit(repo, 'b.txt', 'work\n', 'work on someone else task',
                       date='2026-02-01T00:00:00+00:00')
        _git(['checkout', '-q', 'main'], repo)
        _merge(repo, 'task/999', 'Merge task/999 into main',
               date='2026-03-01T00:00:00+00:00')

        # Task 790 has no landing merge and no status auto-commit, so it falls
        # to rung 3. Its first invocation sits between the side commit and the
        # merge that landed it.
        sha, rung = driver.resolve_baseline(repo, '790', '2026-02-15T00:00:00+00:00')
        assert rung == 'timestamp_walk'
        first_parent = _git(['rev-list', '--first-parent', 'main'], repo).split()
        assert sha in first_parent, (
            f'rung 3 returned {sha}, which is not on main\'s first-parent line '
            f'— that tree state never existed on main'
        )
        assert sha != side
        assert sha == mainline

    def test_rung3_still_picks_the_newest_mainline_commit_before_ts(
        self, tmp_path: Path,
    ) -> None:
        # Guard against --first-parent over-pruning: on a linear main it must
        # still return the NEWEST commit before the cutoff, not the oldest.
        repo = _init_repo(tmp_path)
        _commit(repo, 'a.txt', 'base\n', 'oldest',
                date='2026-01-01T00:00:00+00:00')
        middle = _commit(repo, 'b.txt', 'more\n', 'middle',
                         date='2026-02-01T00:00:00+00:00')
        _commit(repo, 'c.txt', 'later\n', 'newest',
                date='2026-03-01T00:00:00+00:00')

        sha, rung = driver.resolve_baseline(repo, '791', '2026-02-15T00:00:00+00:00')
        assert rung == 'timestamp_walk'
        assert sha == middle

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

    def test_finds_colon_spelled_merge_subject(self, tmp_path: Path) -> None:
        # The regressed spelling. Both are in live use on reify's main
        # (censused: 2741 x "Merge task/N into main", 74 x "Merge task/N: ..."),
        # and matching only the first is what stamped a false
        # `reference_unavailable` on fixtures that DO have a landing merge.
        # This is reify_task_4026's real subject, verbatim.
        repo = _init_repo(tmp_path)
        _commit(repo, 'a.txt', 'base\n', 'base')
        _git(['checkout', '-q', '-b', 'task/4026'], repo)
        _commit(repo, 'b.txt', 'work\n', 'work')
        _git(['checkout', '-q', 'main'], repo)
        _git(['merge', '--no-ff', 'task/4026', '-m',
              'Merge task/4026: Add SPEED_OF_LIGHT + BOLTZMANN_CONSTANT '
              'physical constants to std.units'], repo)
        assert driver.find_merge_sha(repo, '4026') == _git(['rev-parse', 'HEAD'], repo)

    def test_colon_spelling_does_not_prefix_match_a_longer_id(
        self, tmp_path: Path,
    ) -> None:
        # Substring-safety must hold for the colon spelling too: 'Merge
        # task/8030: work' cannot answer a query for task 803. Asserted in BOTH
        # directions so a pattern that matched nothing at all could not pass.
        repo = _init_repo(tmp_path)
        _commit(repo, 'a.txt', 'base\n', 'base')
        _git(['checkout', '-q', '-b', 'task/8030'], repo)
        _commit(repo, 'b.txt', 'work\n', 'work')
        _git(['checkout', '-q', 'main'], repo)
        _git(['merge', '--no-ff', 'task/8030', '-m', 'Merge task/8030: work'], repo)
        merge = _git(['rev-parse', 'HEAD'], repo)
        assert driver.find_merge_sha(repo, '803') is None
        assert driver.find_merge_sha(repo, '8030') == merge

    def test_ambiguous_across_spellings_is_planrate_only(
        self, tmp_path: Path,
    ) -> None:
        # The 'ambiguity is planRate-only, never a coin flip' contract holds
        # over the UNION of the two spellings, not per-spelling: two landing
        # merges for one id are ambiguous however each one is worded.
        repo = _init_repo(tmp_path)
        _commit(repo, 'a.txt', 'base\n', 'base')
        for branch, fn, subject in (
            ('task/803-1', 'b.txt', 'Merge task/803 into main'),
            ('task/803-2', 'c.txt', 'Merge task/803: the colon spelling'),
        ):
            _git(['checkout', '-q', '-b', branch], repo)
            _commit(repo, fn, 'work\n', f'work on {branch}')
            _git(['checkout', '-q', 'main'], repo)
            _git(['merge', '--no-ff', branch, '-m', subject], repo)
        assert driver.find_merge_sha(repo, '803') is None

    def test_colon_spelling_requires_the_space_separator(
        self, tmp_path: Path,
    ) -> None:
        # 'Merge task/803:no-space' is not the landing-merge shape and must not
        # count as a second match. If it did, the real merge below would be
        # reported as ambiguous and lost.
        repo = _init_repo(tmp_path)
        _commit(repo, 'a.txt', 'base\n', 'base')
        _git(['checkout', '-q', '-b', 'task/803-decoy'], repo)
        _commit(repo, 'b.txt', 'decoy\n', 'decoy')
        _git(['checkout', '-q', 'main'], repo)
        _git(['merge', '--no-ff', 'task/803-decoy', '-m',
              'Merge task/803:no-space'], repo)
        _git(['checkout', '-q', '-b', 'task/803'], repo)
        _commit(repo, 'c.txt', 'work\n', 'work')
        _git(['checkout', '-q', 'main'], repo)
        _git(['merge', '--no-ff', 'task/803', '-m',
              'Merge task/803: the real landing merge'], repo)
        assert driver.find_merge_sha(repo, '803') == _git(['rev-parse', 'HEAD'], repo)


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
# _mint_one — every fixture declares whether its base is an approximation
# ---------------------------------------------------------------------------

class TestBaseApproximationMarking:
    """Only ``merge_first_parent`` yields the task's TRUE branch point (M^1 of
    its landing merge). Every weaker rung is a guess, and a readout that
    cannot tell the two apart silently averages them together — which is how
    reify_task_3883 shipped a base ~1900 first-parent commits from where its
    work actually started with nothing in the JSON to say so."""

    def test_merge_derived_base_is_not_approximated(self) -> None:
        rec = _mint(_planrate_row(
            mint_mode='planrate_only', baseline_source='merge_first_parent',
        ))
        assert rec['provenance']['base_is_approximated'] is False
        assert 'base_approximation_reason' not in rec['provenance']

    def test_status_autocommit_base_is_approximated_and_says_which_rung(
        self,
    ) -> None:
        rec = _mint(_planrate_row(baseline_source='status_autocommit'))
        assert rec['provenance']['base_is_approximated'] is True
        reason = rec['provenance']['base_approximation_reason']
        assert reason.strip()
        assert 'status_autocommit' in reason

    def test_timestamp_walk_base_is_approximated_and_says_which_rung(
        self,
    ) -> None:
        rec = _mint(_planrate_row(baseline_source='timestamp_walk'))
        assert rec['provenance']['base_is_approximated'] is True
        reason = rec['provenance']['base_approximation_reason']
        assert 'timestamp_walk' in reason
        # Self-describing: a reader of the JSON alone learns this is an
        # approximation a readout should exclude, not a derived branch point.
        assert 'approximation' in reason.lower()
        assert 'exclude' in reason.lower()

    def test_the_flag_is_a_real_bool(self) -> None:
        # A readout filters on this. A truthy string would make every fixture
        # look approximated, including the merge-derived ones.
        for source in ('merge_first_parent', 'status_autocommit', 'timestamp_walk'):
            flag = _mint(_planrate_row(baseline_source=source))[
                'provenance']['base_is_approximated']
            assert isinstance(flag, bool), f'{source}: {flag!r} is not a bool'


# ---------------------------------------------------------------------------
# _mint_continuity_one — the inherited base is MEASURED, never assumed
# ---------------------------------------------------------------------------

def _standing_fixture(repo: Path, task_id: str, pre: str, post: str) -> dict:
    """A minimal canonical ``evals/tasks/<id>.json`` for the continuity path."""
    return {
        'id': f'reify_task_{task_id}',
        'name': f'Standing fixture {task_id}',
        'project': 'reify',
        'project_root': str(repo),
        'pre_task_commit': pre,
        'post_task_commit': post,
        'task_definition': {
            'title': f'Standing fixture {task_id}',
            'description': 'Carried verbatim from the standing corpus.',
        },
        'verify_commands': {'test': 'true'},
        'modules': ['kernel'],
        'complexity': 'complex',
    }


def _mint_continuity(monkeypatch: Any, tmp_path: Path, src_fixture: dict) -> dict:
    """Mint one continuity record from *src_fixture*, written under a fake
    REPO_ROOT so nothing is read from or written to the real corpus."""
    import asyncio
    rel = Path('orchestrator/src/orchestrator/evals/tasks') / f'{src_fixture["id"]}.json'
    dest = tmp_path / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(src_fixture, indent=2))
    monkeypatch.setattr(driver, 'REPO_ROOT', tmp_path)
    sampler = driver._import_sampler()
    return asyncio.run(driver._mint_continuity_one(
        sampler, {'id': src_fixture['id'], 'source_path': str(rel)},
        sampled_at='2026-08-04T00:00:00+00:00', seed=3631,
        ceilings={'max_architect_turns': 120, 'timeout_minutes': 180},
    ))


class TestContinuityBaseApproximation:
    """Continuity fixtures sit OUTSIDE the three-rung ladder: their base is
    carried verbatim from the standing corpus under
    ``CONTINUITY_BASELINE_SOURCE``, so the flag cannot be read off a rung
    label. It is MEASURED against the task's landing merge — the same
    measure-the-premise discipline as ``post_commit_reachable_from_main``."""

    def test_inherited_base_equal_to_merge_first_parent_is_not_approximated(
        self, tmp_path: Path, monkeypatch: Any,
    ) -> None:
        repo = _init_repo(tmp_path, 'reify')
        base = _commit(repo, 'a.txt', 'base\n', 'base')
        _git(['checkout', '-q', '-b', 'task/12'], repo)
        tip = _commit(repo, 'b.txt', 'work\n', 'work on 12')
        _git(['checkout', '-q', 'main'], repo)
        # The colon spelling on purpose: df_task_18 and reify_task_12 both
        # have colon-spelled landing merges in the live checkouts, so the
        # step-2 matcher is what makes them measurable at all.
        merge = _merge(repo, 'task/12', 'Merge task/12: the landing merge')

        rec = _mint_continuity(
            monkeypatch, tmp_path, _standing_fixture(repo, '12', base, tip))
        prov = rec['provenance']
        assert prov['base_is_approximated'] is False
        # The measurement is recorded, not just its verdict: a reader can see
        # WHICH merge the inherited base was checked against.
        assert prov['base_verified_against_merge'] == merge
        assert 'base_approximation_reason' not in prov

    def test_inherited_base_diverging_from_merge_first_parent_is_approximated(
        self, tmp_path: Path, monkeypatch: Any,
    ) -> None:
        repo = _init_repo(tmp_path, 'reify')
        older = _commit(repo, 'a.txt', 'base\n', 'base')
        branch_point = _commit(repo, 'a2.txt', 'more\n', 'later main commit')
        _git(['checkout', '-q', '-b', 'task/13'], repo)
        tip = _commit(repo, 'b.txt', 'work\n', 'work on 13')
        _git(['checkout', '-q', 'main'], repo)
        _merge(repo, 'task/13', 'Merge task/13 into main')

        # The standing fixture inherited `older`, not the real branch point.
        rec = _mint_continuity(
            monkeypatch, tmp_path, _standing_fixture(repo, '13', older, tip))
        prov = rec['provenance']
        assert prov['base_is_approximated'] is True
        assert prov['base_verified_against_merge'] == \
            _git(['rev-parse', 'main'], repo)
        # Both SHAs are reported, so the divergence is legible from the JSON.
        reason = prov['base_approximation_reason']
        assert older in reason and branch_point in reason

    def test_no_landing_merge_leaves_the_base_unverifiable(
        self, tmp_path: Path, monkeypatch: Any,
    ) -> None:
        repo = _init_repo(tmp_path, 'reify')
        base = _commit(repo, 'a.txt', 'base\n', 'base')
        tip = _commit(repo, 'b.txt', 'work\n', 'landed directly, no merge')

        rec = _mint_continuity(
            monkeypatch, tmp_path, _standing_fixture(repo, '14', base, tip))
        prov = rec['provenance']
        assert prov['base_is_approximated'] is True
        assert prov['base_verified_against_merge'] is None
        assert 'no landing merge' in prov['base_approximation_reason'].lower()

    def test_pre_and_post_are_still_carried_verbatim(
        self, tmp_path: Path, monkeypatch: Any,
    ) -> None:
        # The continuity contract forbids divergence; this marking is
        # OBSERVATIONAL only and must never move a commit.
        repo = _init_repo(tmp_path, 'reify')
        older = _commit(repo, 'a.txt', 'base\n', 'base')
        branch_point = _commit(repo, 'a2.txt', 'more\n', 'later main commit')
        _git(['checkout', '-q', '-b', 'task/15'], repo)
        tip = _commit(repo, 'b.txt', 'work\n', 'work on 15')
        _git(['checkout', '-q', 'main'], repo)
        _merge(repo, 'task/15', 'Merge task/15 into main')

        # An approximated base and a verified one must BOTH be carried through
        # untouched — the marking is observational, never corrective.
        for pre in (older, branch_point):
            fixture = _standing_fixture(repo, '15', pre, tip)
            rec = _mint_continuity(monkeypatch, tmp_path, fixture)
            assert rec['pre_task_commit'] == fixture['pre_task_commit']
            assert rec['post_task_commit'] == fixture['post_task_commit']
            assert rec['provenance']['baseline_source'] == \
                driver.CONTINUITY_BASELINE_SOURCE


# ---------------------------------------------------------------------------
# redrive_provenance — re-derive provenance WITHOUT re-censusing
# ---------------------------------------------------------------------------

def _redrive_manifest() -> dict:
    """A manifest with the blocks --redrive must leave alone."""
    return {
        'census': {'date': '2026-08-08', 'n': 41},
        'ceilings': {'max_architect_turns': 120, 'timeout_minutes': 180},
        'continuity': {'fixtures': [{'id': 'reify_task_12'}]},
        'merge_sha_availability': {'referenced': 1, 'planrate_only': 1},
        'candidates': [
            {
                'task_id': '4026', 'project': 'reify',
                'project_root': '/home/leo/src/reify',
                'title': 'Add physical constants', 'status': 'done',
                'brief_chars': 224, 'decision': 'include',
                'reason': 'INCLUDE. The brief states an implementable goal.',
                'merge_sha': None, 'baseline_sha': 'e' * 40,
                'baseline_source': 'timestamp_walk',
                'mint_mode': 'planrate_only',
            },
            {
                'task_id': '3883', 'project': 'reify',
                'project_root': '/home/leo/src/reify',
                'title': 'Add stdlib dynamics', 'status': 'cancelled',
                'brief_chars': 300, 'decision': 'include',
                'reason': 'INCLUDE. The brief states an implementable goal.',
                'merge_sha': None, 'baseline_sha': '2' * 40,
                'baseline_source': 'timestamp_walk',
                'mint_mode': 'planrate_only',
            },
            {
                'task_id': '999', 'project': 'reify',
                'project_root': '/home/leo/src/reify',
                'title': 'Too vague', 'status': 'done',
                'brief_chars': 12, 'decision': 'exclude',
                'reason': 'EXCLUDE. The brief states no implementable goal.',
            },
        ],
    }


def _fake_resolve(table: dict[str, tuple[str | None, str, str]]):
    """Injected resolver: task_id -> (merge_sha, baseline_sha, baseline_source).

    Same dependency-injection discipline as
    ``task_sampler.audit_fixture_corpus``'s ``ref_exists`` — no checkout is
    touched, so the redrive rule itself is testable in isolation.
    """
    def resolve(project_root: str, task_id: str):
        return table[task_id]
    return resolve


class TestRedriveProvenance:
    def test_a_newly_resolvable_row_is_upgraded(self) -> None:
        before = _redrive_manifest()
        after, _changes = driver.redrive_provenance(before, _fake_resolve({
            '4026': ('3613bea224' + 'f' * 30, '794d321596' + 'a' * 30,
                     'merge_first_parent'),
            '3883': (None, '2ceaf9ec17' + 'b' * 30, 'timestamp_walk'),
        }))
        row = after['candidates'][0]
        assert row['merge_sha'] == '3613bea224' + 'f' * 30
        assert row['baseline_sha'] == '794d321596' + 'a' * 30
        assert row['baseline_source'] == 'merge_first_parent'
        assert row['mint_mode'] == 'referenced'

    def test_a_row_that_still_has_no_merge_keeps_planrate_only(self) -> None:
        after, _changes = driver.redrive_provenance(
            _redrive_manifest(), _fake_resolve({
                '4026': (None, 'e' * 40, 'timestamp_walk'),
                '3883': (None, '2ceaf9ec17' + 'b' * 30, 'timestamp_walk'),
            }))
        row = after['candidates'][1]
        assert row['mint_mode'] == 'planrate_only'
        assert row['merge_sha'] is None
        assert row['baseline_source'] == 'timestamp_walk'
        assert row['baseline_sha'] == '2ceaf9ec17' + 'b' * 30

    def test_curation_fields_are_never_re_adjudicated(self) -> None:
        before = _redrive_manifest()
        after, _changes = driver.redrive_provenance(before, _fake_resolve({
            '4026': ('3613bea224' + 'f' * 30, '794d321596' + 'a' * 30,
                     'merge_first_parent'),
            '3883': (None, '2ceaf9ec17' + 'b' * 30, 'timestamp_walk'),
        }))
        curation_keys = ('task_id', 'project', 'project_root', 'decision',
                         'reason', 'title', 'status', 'brief_chars')
        for old_row, new_row in zip(before['candidates'], after['candidates'],
                                    strict=True):
            for key in curation_keys:
                if key in old_row:
                    assert new_row[key] == old_row[key], key

    def test_the_row_set_and_the_exclude_row_are_untouched(self) -> None:
        # --redrive re-derives provenance on rows that already exist; by
        # construction it cannot add or drop a fixture.
        before = _redrive_manifest()
        after, _changes = driver.redrive_provenance(before, _fake_resolve({
            '4026': ('3613bea224' + 'f' * 30, '794d321596' + 'a' * 30,
                     'merge_first_parent'),
            '3883': (None, '2ceaf9ec17' + 'b' * 30, 'timestamp_walk'),
        }))
        assert len(after['candidates']) == len(before['candidates'])
        assert [r['task_id'] for r in after['candidates']] == \
            [r['task_id'] for r in before['candidates']]
        assert after['candidates'][2] == before['candidates'][2]

    def test_census_ceilings_and_continuity_blocks_are_untouched(self) -> None:
        before = _redrive_manifest()
        after, _changes = driver.redrive_provenance(before, _fake_resolve({
            '4026': ('3613bea224' + 'f' * 30, '794d321596' + 'a' * 30,
                     'merge_first_parent'),
            '3883': (None, '2ceaf9ec17' + 'b' * 30, 'timestamp_walk'),
        }))
        for block in ('census', 'ceilings', 'continuity'):
            assert after[block] == before[block]

    def test_does_not_mutate_the_manifest_it_was_given(self) -> None:
        before = _redrive_manifest()
        snapshot = json.dumps(before, sort_keys=True)
        driver.redrive_provenance(before, _fake_resolve({
            '4026': ('3613bea224' + 'f' * 30, '794d321596' + 'a' * 30,
                     'merge_first_parent'),
            '3883': (None, '2ceaf9ec17' + 'b' * 30, 'timestamp_walk'),
        }))
        assert json.dumps(before, sort_keys=True) == snapshot

    def test_the_change_list_names_exactly_what_moved(self) -> None:
        _after, changes = driver.redrive_provenance(
            _redrive_manifest(), _fake_resolve({
                '4026': ('3613bea224' + 'f' * 30, '794d321596' + 'a' * 30,
                         'merge_first_parent'),
                '3883': (None, '2ceaf9ec17' + 'b' * 30, 'timestamp_walk'),
            }))
        # 3883's rung and mode are unchanged; only its baseline_sha moved (the
        # --first-parent fix), so BOTH rows moved and both must be listed.
        assert [c['task_id'] for c in changes] == ['4026', '3883']
        upgraded = changes[0]
        assert upgraded['before']['baseline_source'] == 'timestamp_walk'
        assert upgraded['after']['baseline_source'] == 'merge_first_parent'
        assert upgraded['before']['mint_mode'] == 'planrate_only'
        assert upgraded['after']['mint_mode'] == 'referenced'

    def test_an_unchanged_row_is_not_listed(self) -> None:
        _after, changes = driver.redrive_provenance(
            _redrive_manifest(), _fake_resolve({
                '4026': (None, 'e' * 40, 'timestamp_walk'),
                '3883': (None, '2' * 40, 'timestamp_walk'),
            }))
        assert changes == []


# ---------------------------------------------------------------------------
# base_distance_rows — REPORT the measured distances, never assert them
# ---------------------------------------------------------------------------

def _row(task_id: str, baseline_sha: str, baseline_source: str,
         merge_sha: str | None, mint_mode: str) -> dict:
    return {
        'task_id': task_id, 'project': 'reify',
        'project_root': '/home/leo/src/reify', 'decision': 'include',
        'baseline_sha': baseline_sha, 'baseline_source': baseline_source,
        'merge_sha': merge_sha, 'mint_mode': mint_mode,
    }


_BEFORE_ROWS = [
    _row('4026', 'e21d047026', 'timestamp_walk', None, 'planrate_only'),
    _row('3883', '2ceaf9ec17', 'timestamp_walk', None, 'planrate_only'),
]
_AFTER_ROWS = [
    _row('4026', '794d321596', 'merge_first_parent', '3613bea224', 'referenced'),
    _row('3883', '2ceaf9ec17', 'timestamp_walk', None, 'planrate_only'),
]


def _fake_distance(table: dict[tuple[str, str], int | None]):
    """Injected ``distance(project_root, a, b) -> int | None``."""
    def distance(project_root: str, a: str, b: str) -> int | None:
        return table.get((a, b))
    return distance


_DISTANCES = {
    ('794d321596', 'e21d047026'): 245,
    ('794d321596', '794d321596'): 0,
}


class TestBaseDistanceReport:
    def test_reports_before_and_after_for_each_fixture(self) -> None:
        rows = driver.base_distance_rows(
            _BEFORE_ROWS, _AFTER_ROWS, _fake_distance(_DISTANCES))
        upgraded = rows[0]
        assert upgraded['fixture_id'] == 'reify_task_4026'
        assert upgraded['before']['baseline_source'] == 'timestamp_walk'
        assert upgraded['after']['baseline_source'] == 'merge_first_parent'
        assert upgraded['before']['baseline_sha'] == 'e21d047026'
        assert upgraded['after']['baseline_sha'] == '794d321596'
        assert upgraded['before']['distance_from_branch_point'] == 245
        assert upgraded['after']['distance_from_branch_point'] == 0

    def test_carries_the_shared_approximation_flag(self) -> None:
        rows = driver.base_distance_rows(
            _BEFORE_ROWS, _AFTER_ROWS, _fake_distance(_DISTANCES))
        # From the step-6 helper, so the report and the minted fixtures can
        # never disagree about which bases a readout should exclude.
        assert rows[0]['before']['base_is_approximated'] is True
        assert rows[0]['after']['base_is_approximated'] is False
        assert rows[1]['after']['base_is_approximated'] is True

    def test_an_unknowable_distance_is_reported_not_omitted(self) -> None:
        # reify_3883's real case: no landing merge under either spelling, so
        # the true branch point is not derivable from git at all. Dropping the
        # row would let the report read as full coverage when it is not.
        rows = driver.base_distance_rows(
            _BEFORE_ROWS, _AFTER_ROWS, _fake_distance(_DISTANCES))
        assert [r['fixture_id'] for r in rows] == \
            ['reify_task_4026', 'reify_task_3883']
        unknown = rows[1]
        assert unknown['before']['distance_from_branch_point'] is None
        assert unknown['after']['distance_from_branch_point'] is None
        assert unknown['branch_point'] is None
        assert 'no landing merge' in unknown['note'].lower()
        assert 'approximated' in unknown['note'].lower()

    def test_names_the_branch_point_it_measured_against(self) -> None:
        rows = driver.base_distance_rows(
            _BEFORE_ROWS, _AFTER_ROWS, _fake_distance(_DISTANCES))
        assert rows[0]['branch_point'] == '794d321596'

    def test_is_deterministic_and_wall_clock_free(self) -> None:
        first = driver.base_distance_rows(
            _BEFORE_ROWS, _AFTER_ROWS, _fake_distance(_DISTANCES))
        second = driver.base_distance_rows(
            _BEFORE_ROWS, _AFTER_ROWS, _fake_distance(_DISTANCES))
        assert first == second
        assert json.dumps(first, sort_keys=True) == \
            json.dumps(second, sort_keys=True)

    def test_a_distance_the_measurement_could_not_resolve_stays_none(
        self,
    ) -> None:
        # An empty `git rev-list --count` (an unresolvable SHA in this
        # checkout) must read as "not measured", never as 0.
        rows = driver.base_distance_rows(
            _BEFORE_ROWS, _AFTER_ROWS, _fake_distance({}))
        assert rows[0]['before']['distance_from_branch_point'] is None
        assert rows[0]['branch_point'] == '794d321596'

    def test_the_measurement_is_direction_agnostic(self, tmp_path: Path) -> None:
        # The production measurement must count the SYMMETRIC difference. A
        # one-directional `git rev-list --count A..B` answers 0 whenever B is
        # an ancestor of A — which is exactly reify_task_4026's shape, and
        # would have reported its 245-commit-stale base as a perfect 0.
        repo = _init_repo(tmp_path)
        old = _commit(repo, 'a.txt', 'base\n', 'base')
        _commit(repo, 'b.txt', 'one\n', 'one')
        new = _commit(repo, 'c.txt', 'two\n', 'two')

        assert driver._commit_distance(str(repo), new, old) == 2
        assert driver._commit_distance(str(repo), old, new) == 2
        assert driver._commit_distance(str(repo), new, new) == 0
        assert driver._first_parent_commit_distance(str(repo), new, old) == 2

    def test_an_unresolvable_sha_measures_as_none_not_zero(
        self, tmp_path: Path,
    ) -> None:
        repo = _init_repo(tmp_path)
        head = _commit(repo, 'a.txt', 'base\n', 'base')
        assert driver._commit_distance(str(repo), head, 'f' * 40) is None
        assert driver._first_parent_commit_distance(
            str(repo), head, 'f' * 40) is None

    def test_refuses_rows_that_do_not_line_up(self) -> None:
        # A before/after pair that disagrees on which fixtures it covers would
        # silently mis-attribute every distance in the table.
        with pytest.raises(ValueError):
            driver.base_distance_rows(
                _BEFORE_ROWS, list(reversed(_AFTER_ROWS)),
                _fake_distance(_DISTANCES))


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
    # `Any`, not `object`: the driver is loaded by path (see _load_driver), so
    # `driver.Candidate` is only ever `Any` to a type checker. Narrowing the
    # return to `object` throws that away and makes every field read below a
    # reportAttributeAccessIssue.
    def _cand(self) -> Any:
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
