"""python -m shared.quality_series: the producer's signal, run against a fixture project root."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from quality_series_fixtures import (
    GitRepoBuilder,
    LandingScenario,
    RunsDbBuilder,
    at_hour,
    build_landing_scenario,
)

from shared.quality_series import cli, landings, rows_for, summary
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

_SRC = Path(__file__).resolve().parent.parent / 'src'
POST_MERGE = 'post_merge_verify'


@pytest.fixture(scope='module')
def project(tmp_path_factory: pytest.TempPathFactory) -> LandingScenario:
    """Every landing lands on 2026-10-27 (ISO week 2026-W44); the summary is hand-computed."""
    s = build_landing_scenario(tmp_path_factory.mktemp('cli') / 'proj')
    db = s.db
    db.merge_blocked(at_hour(10.9), '102', POST_MERGE, 'Post-merge verification failed: a')
    db.merge_blocked(at_hour(11.8), '104', POST_MERGE, 'Post-merge verification failed: b')
    db.merge_blocked(at_hour(11.85), '104', POST_MERGE, 'Post-merge verification failed: c')
    db.flake(at_hour(11.6), '104', 'passes_in_isolation')
    db.flake(at_hour(12.5), '101', 'fails_in_isolation')
    db.review_completed(at_hour(10.8), '102', cycle=1, has_blocking=True)
    db.review_completed(at_hour(10.85), '102', cycle=2, has_blocking=False)
    db.review_completed(at_hour(13.5), '107', cycle=1, has_blocking=False)
    meta = s.repo.root / '.worktrees' / '.task-meta' / '101'
    meta.mkdir(parents=True)
    (meta / 'review_state.json').write_text(json.dumps({'review_cycles_total': 1}))
    db.merge_verify(at_hour(10.7), '102', s.f2, passed=True)
    db.merge_verify(at_hour(11.7), '105', s.m3, passed=False)
    db.merge_verify(at_hour(11.9), '105', s.m3, passed=True)
    return s


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, list[str], str]:
    code = cli.main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out.splitlines(), captured.err


def _table(lines: list[str]) -> list[dict[str, str]]:
    header = next(line for line in lines if line.startswith('# columns:'))
    columns = header.removeprefix('# columns:').strip().split('\t')
    return [dict(zip(columns, line.split('\t'), strict=True))
            for line in lines if not line.startswith('#')]


class TestSince:
    def test_prints_one_row_per_landing_after_the_cursor(self, project, capsys) -> None:
        code, lines, _ = _run(capsys, '--root', str(project.repo.root), '--since', project.m1)
        assert code == 0
        rows = _table(lines)
        assert len(rows) == 8
        assert ('101', project.m1) not in {(r['task_id'], r['merge_sha']) for r in rows}
        for column in (
            'ordinal', 'landed_at', 'task_id', 'merge_sha', 'source', 'unit_id', 'unit_size',
            'git_first_parent', 'rejected', 'rejection_events', 'reviewed', 'review_blocking',
            'flake_confirmed', 'flake_real_red', 'reverify_count', 'all_role_usd',
            'implementer_invocations', 'turns', 'turns_capped', 'execute_iterations',
            'review_rounds', 'review_rounds_source', 'charter_class', 'code_lines',
        ):
            assert column in rows[0]

    def test_none_is_rendered_as_none(self, project, capsys) -> None:
        _, lines, _ = _run(capsys, '--root', str(project.repo.root), '--since', project.m1)
        side = next(r for r in _table(lines) if r['task_id'] == '106')
        assert side['code_lines'] == 'none'

    def test_states_the_disagreement_and_the_anchoring(self, project, capsys) -> None:
        _, lines, _ = _run(capsys, '--root', str(project.repo.root), '--since', project.m1)
        assert ('# git first-parent disagreement: 2 '
                '(off_first_parent=1, merges_without_landing=1, late_claims=0)') in lines
        anchoring = next(line for line in lines if line.startswith('# anchoring:'))
        assert 'event timestamp' in anchoring
        assert '(previous landing, landing]' in anchoring
        assert f'# next cursor: {project.m8}' in lines


def test_week_summary_is_the_hand_computed_one(project, capsys) -> None:
    code, lines, _ = _run(capsys, '--root', str(project.repo.root), '--week', '2026-W44')
    assert code == 0
    summary = [line for line in lines if line.startswith('# week') or line.startswith('#   ')]
    assert summary == [
        '# week 2026-W44: 7 units, 9 landings',
        '#   rejections per 100 landings: 37.5 (3 events over 8 landings)',
        '#   landings with >= 1 rejection: 4/8',
        '#   confirmed-flake landings: 3/7',
        '#   real-red landings: 1/7',
        '#   review rounds >= 2: 2/3 (review_completed=2, disk_counter=1)',
        '#   re-verifies per landing: 0.12 (1 over 8 landings)',
    ]


def test_week_summary_denominators_are_known_landings(project) -> None:
    batch = landings(project.db.conn, project.repo.root, since_cursor=None)
    rows = rows_for(batch.landings, project.db.conn, project.repo.root, git_budget_s=60)
    week = summary.week_summary(rows, '2026-W44')
    assert week.rejection_events == summary.EventsPerLanding(3, 8)
    assert week.reverifies == summary.EventsPerLanding(1, 8)


def test_a_relative_root_is_resolved(project, capsys, monkeypatch) -> None:
    _, absolute, _ = _run(capsys, '--root', str(project.repo.root), '--since', project.m1)
    monkeypatch.chdir(project.repo.root)
    code, relative, err = _run(capsys, '--root', '.', '--since', project.m1)
    assert (code, err) == (0, '')
    assert relative == absolute


class TestRefusals:
    def test_a_root_outside_any_repository_exits_2(self, tmp_path: Path, capsys) -> None:
        root = tmp_path / 'plain'
        RunsDbBuilder(root / 'data' / 'orchestrator' / 'runs.db').close()
        code, _, err = _run(capsys, '--root', str(root))
        assert code == 2
        assert str(root) in err

    def test_a_root_without_runs_db_exits_2(self, tmp_path: Path, capsys) -> None:
        repo = GitRepoBuilder(tmp_path / 'bare')
        repo.commit({'README.md': 'x\n'}, 'root')
        code, _, err = _run(capsys, '--root', str(repo.root))
        assert code == 2
        assert 'runs.db' in err

    def test_an_unknown_charter_class_exits_2(self, tmp_path: Path, capsys) -> None:
        s = build_landing_scenario(tmp_path / 'proj', with_charter_class=True)
        s.db.task_result('102', 'run-a', 'blocked', at_hour(10.5), charter_class='bogus')
        code, _, err = _run(capsys, '--root', str(s.repo.root))
        assert code == 2
        assert 'bogus' in err

    def test_a_runs_db_without_the_events_table_exits_2(self, tmp_path: Path, capsys) -> None:
        repo = GitRepoBuilder(tmp_path / 'empty')
        repo.commit({'README.md': 'x\n'}, 'root')
        path = repo.root / 'data' / 'orchestrator' / 'runs.db'
        path.parent.mkdir(parents=True)
        sqlite3.connect(path).execute('CREATE TABLE other (x)').connection.close()
        code, _, err = _run(capsys, '--root', str(repo.root))
        assert code == 2
        assert 'events' in err

    def test_an_unrecognised_train_payload_exits_2(self, tmp_path: Path, capsys) -> None:
        repo = GitRepoBuilder(tmp_path / 'odd')
        repo.commit({'README.md': 'x\n'}, 'root')
        db = RunsDbBuilder(repo.root / 'data' / 'orchestrator' / 'runs.db')
        db.event(at_hour(10.0), 'train_merged', '401', {'train_id': 't', 'verdict': 'other'})
        db.close()
        code, _, err = _run(capsys, '--root', str(repo.root))
        assert code == 2
        assert 'train_merged' in err


def test_the_module_runs_as_a_program(project) -> None:
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join(
        [str(_SRC), *filter(None, [os.environ.get('PYTHONPATH')])]
    )}
    result = subprocess.run(
        [sys.executable, '-m', 'shared.quality_series', '--root', str(project.repo.root)],
        capture_output=True, text=True, env=env, timeout=300, check=False,
    )
    assert result.returncode == 0, result.stderr
