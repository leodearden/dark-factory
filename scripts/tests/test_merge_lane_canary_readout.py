"""End-to-end tests for scripts/merge_lane_canary_readout.py, the canary readout CLI.

Each test drives ``main(argv)`` over real fixtures: a runs.db at
``runs_db(root)`` built by the ``make_runs_db`` conftest factory, a real git
repo whose main reflog the CLI walks, and a journal extract in the
orchestrator journal's short-iso line format.
"""
from __future__ import annotations

import os
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import merge_lane_canary as canary
import merge_lane_canary_readout as readout
import pytest
from shared.project_layout import runs_db
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

LO =datetime(2026, 11, 20, 0, 0, tzinfo=UTC)
HI = LO + timedelta(days=1)
WINDOW = f'{LO.isoformat()}..{HI.isoformat()}'


def at(minutes: float) -> datetime:
    return LO + timedelta(minutes=minutes)


def row(event_type: str, minutes: float, task_id: str | None, phase: str | None = None, **data: Any) -> dict[str, Any]:
    return {'event_type': event_type, 'timestamp': at(minutes).isoformat(), 'task_id': task_id,
            'phase': phase, 'data': data}


# ---------------------------------------------------------------------------
# A git repo whose main reflog carries direct commits and one fast-forward landing.
# ---------------------------------------------------------------------------


def git(repo: Path, *args: str, when: datetime | None = None) -> str:
    env = {**os.environ, 'GIT_AUTHOR_NAME': 't', 'GIT_AUTHOR_EMAIL': 't@e',
           'GIT_COMMITTER_NAME': 't', 'GIT_COMMITTER_EMAIL': 't@e'}
    if when is not None:
        env['GIT_COMMITTER_DATE'] = env['GIT_AUTHOR_DATE'] = when.isoformat()
    done = subprocess.run(['git', '-C', str(repo), *args], check=True, capture_output=True,
                          text=True, env=env, timeout=30)
    return done.stdout.strip()


def commit(repo: Path, message: str, when: datetime) -> str:
    git(repo, '-c', 'commit.gpgsign=false', 'commit', '--allow-empty', '-q', '-m', message, when=when)
    return git(repo, 'rev-parse', 'HEAD')


def build_repo(repo: Path) -> dict[str, str]:
    """c0 (before the window), d1 direct, f1+f2 landed by fast-forward, d2 direct."""
    repo.mkdir()
    git(repo, 'init', '-q', '-b', 'main')
    shas = {'c0': commit(repo, 'c0', LO - timedelta(days=1)), 'd1': commit(repo, 'd1', at(30))}
    git(repo, 'checkout', '-q', '-b', 'feature')
    shas['f1'] = commit(repo, 'f1', at(40))
    shas['f2'] = commit(repo, 'f2', at(50))
    git(repo, 'checkout', '-q', 'main')
    git(repo, 'merge', '-q', '--ff-only', 'feature', when=at(70))
    shas['d2'] = commit(repo, 'd2', at(300))
    return shas


# ---------------------------------------------------------------------------
# A journal extract in the orchestrator journal's short-iso format.
# ---------------------------------------------------------------------------

HARNESS = '[orchestrator.harness] Main-tip integrity sweep:'
LANE = '[orchestrator.merge_lane] merge queue resource-conservation audit'
LEDGER_TEXT = (
    'unregistered on-disk merge worktree /w/.worktrees/_merge-1e483c5f (age 9001s > grace 9000s) '
    'absent from owned ledger — possible leak — scheduled for automatic reclaim'
)


def journal_line(minutes: float, level: str, message: str) -> str:
    moment = at(minutes)
    return (f'{moment.isoformat(timespec="seconds")} leo-MS-7C35 uv[1218128]: '
            f'{moment:%Y-%m-%d %H:%M:%S} {level:<8} {message}')


def journal(shas: dict[str, str]) -> list[str]:
    d1, f2, d2 = shas['d1'][:12], shas['f2'][:12], shas['d2'][:12]
    return [
        journal_line(40, 'INFO', f'{HARNESS} verifying {d1} with a warm build (last cold verdict: 2026-11-18 04:05:17.983865+00:00)'),
        journal_line(50, 'INFO', f'{HARNESS} warm build of {d1} passed (passed)'),
        journal_line(80, 'INFO', f'{HARNESS} verifying {f2} with a warm build (last cold verdict: 2026-11-18 04:05:17.983865+00:00)'),
        journal_line(90, 'INFO', f'{HARNESS} warm build of {f2} failed (test_failure)'),
        journal_line(91, 'WARNING', f'{HARNESS} filed L1 escalation esc-main-sweep-{f2}-1 for SHA {f2} (test_failure)'),
        journal_line(100, 'WARNING', f'{LANE}: 1 violation(s) found (consecutive streak=1): {LEDGER_TEXT}'),
        journal_line(130, 'INFO', f'{LANE} clear after 423 polls / 12769s — the violation set is now empty'),
        journal_line(310, 'INFO', f'{HARNESS} verifying {d2} with a cold build (last cold verdict: 2026-11-18 04:05:17.983865+00:00)'),
        journal_line(320, 'INFO', f'{HARNESS} cold build of {d2} passed (passed)'),
    ]


# ---------------------------------------------------------------------------
# A runs.db for one window, with every alpha field present.
# ---------------------------------------------------------------------------


def alpha_events(shas: dict[str, str]) -> list[dict[str, Any]]:
    reason = {'basis': 'pass_estimate', 'pass_estimate': 0.9, 'row': [0.876, 4], 'queue_depth': 2}
    return [
        row('merge_queued', 10, 'a', 'merge'),
        row('merge_queued', 20, 'b', 'merge'),
        row('merge_verify', 60, 'a', runner='local', merge_sha=shas['f2'], passed=True,
            duration_ms=1_800_000, chain_items=2, chain_task_ids=['a', 'b'], runner_reason='only_host',
            base_relation='main_tip', chain_depth_reason=reason),
        row('merge_reverify_gate', 65, 'a', 'merge', reverify=True, rebased_from=shas['c0'],
            rebased_onto=shas['d1']),
        row('merge_finalized', 70, 'a', 'merge', state='done', merge_sha=shas['f2'], landed_via_chain=1,
            queue_to_land_secs=3600.0),
        row('merge_verify_cancelled', 120, 'c', 'merge', elapsed_ms=600_000, fraction=0.5, cause='void',
            host='laptop'),
        row('verdict_voided', 120, 'c', 'merge', point='dispatch', item_commit='x' * 40,
            killer={'site': 'head_fail', 'task_id': 'a'}, anchor_reason='tip_dead', anchor_base=None,
            dead_link='y' * 40),
        row('merge_heartbeat', 60, None, occupancy={'leases': [{'host': 'laptop', 'activity': 'finished'}]}),
        row('merge_heartbeat', 65, None, occupancy={'leases': []}),
        row('phase_enter', 180, 't', 'verify'),
        row('phase_exit', 190, 't', 'verify'),
        row('workflow_verify', 190.5, 't', passed=True),
        row('phase_enter', 200, 'u', 'verify'),
        row('phase_exit', 210, 'u', 'verify'),
        row('phase_enter', 220, 'u', 'execute'),
        row('merge_finalized', 24 * 60, 'late', 'merge', state='done', merge_sha='z' * 40,
            queue_to_land_secs=60.0),
    ]


@pytest.fixture
def project(tmp_path: Path, make_runs_db) -> tuple[Path, dict[str, str], Path]:
    root = tmp_path / 'project'
    shas = build_repo(root)
    runs_db(root).parent.mkdir(parents=True)
    make_runs_db(alpha_events(shas), directory=runs_db(root).parent)
    extract = tmp_path / 'journal.txt'
    extract.write_text('\n'.join(journal(shas)) + '\n', encoding='utf-8')
    return root, shas, extract


def run_readout(root: Path, extract: Path, out: Path) -> int:
    return readout.main([
        '--window', WINDOW, '--project-root', str(root), '--journal-extract', str(extract), '--out', str(out),
    ])


def test_the_readout_renders_one_row_per_predicate_and_every_section(project, tmp_path):
    root, _shas, extract = project
    out = tmp_path / 'canary.md'

    assert run_readout(root, extract, out) == 0

    text = out.read_text(encoding='utf-8')
    table_rows = [line for line in text.splitlines() if line.startswith('| ')]
    for d12 in canary.D12_ROWS:
        (line,) = [r for r in table_rows if r.startswith(f'| {d12.label} |')]
        assert d12.rule in line
        assert d12.baseline in line
        assert ('| holds |' in line) or ('| FALSE |' in line)
    for heading in (
        '## Landings and lane hours', '## Voids', '## chain_items distribution', '## Depth-reason tally',
        '## Depth-starved verifies', '## Post-red-chain solos', '## Gate re-verify split',
        '## Red-main episodes', '## False predicates',
    ):
        assert heading in text


def test_the_readout_values_follow_the_fixture(project, tmp_path):
    root, shas, extract = project
    out = tmp_path / 'canary.md'
    run_readout(root, extract, out)
    text = out.read_text(encoding='utf-8')

    assert '| gates per landing | 1 | < 0.3 | FALSE |' in text
    assert '| voids per landing | 0 | < 0.1 | holds |' in text
    assert 'discarded before a host (point dispatch, not counted): 1' in text
    assert '| chain-attributed red-main episodes | 1 | = 0 | FALSE |' in text
    assert '| speculation accounting violations | 0 | = 0 | holds |' in text
    assert f'first red {shas["f2"][:12]}' in text
    assert '- task verifies started on red: 2 of 2 (100.0%), 1 failed' in text
    assert '- direct_only: 1' in text


def test_the_window_is_half_open(project, tmp_path):
    root, _shas, extract = project
    out = tmp_path / 'canary.md'
    run_readout(root, extract, out)
    assert '- landings: 1' in out.read_text(encoding='utf-8')


def test_a_pre_alpha_window_is_refused_naming_each_missing_field(tmp_path, make_runs_db):
    root = tmp_path / 'project'
    build_repo(root)
    runs_db(root).parent.mkdir(parents=True)
    make_runs_db([
        row('merge_verify', 60, 'a', runner='local', passed=True, duration_ms=1000, chain_items=1),
        row('merge_finalized', 70, 'a', 'merge', state='done', merge_sha='s'),
    ], directory=runs_db(root).parent)
    extract = tmp_path / 'journal.txt'
    extract.write_text('', encoding='utf-8')
    out = tmp_path / 'canary.md'

    rc = run_readout(root, extract, out)

    text = out.read_text(encoding='utf-8')
    assert rc != 0
    assert text.startswith('REFUSED')
    for field in ('merge_verify.runner_reason', 'merge_verify.base_relation', 'merge_verify_cancelled',
                  'verdict_voided.item_commit', 'verdict_voided.killer', 'merge_finalized.queue_to_land_secs'):
        assert field in text
    assert '| predicate |' not in text


def test_the_journal_parser_types_sweep_verdicts_and_audit_records():
    shas = {'d1': '1' * 40, 'f2': '2' * 40, 'd2': '3' * 40}
    lines = [
        *journal(shas),
        journal_line(400, 'INFO', f'{HARNESS} verifying {"4" * 12} with a warm build (last cold verdict: x)'),
        journal_line(410, 'INFO', f'{HARNESS} warm build of {"4" * 12} failed (infra_timeout)'),
        journal_line(411, 'INFO', f'{HARNESS} failure at {"4" * 12} unconfirmed on current tip '
                     '(subset_confirmed=False tip_unchanged=False) — not filing (stale/transient)'),
        'not a journal line at all',
    ]

    parsed = readout.parse_journal(lines)

    assert [(v.sha12, v.passed, v.filed) for v in parsed.sweeps] == [
        ('1' * 12, True, None), ('2' * 12, False, True), ('3' * 12, True, None), ('4' * 12, False, False),
    ]
    assert parsed.sweeps[1].started_at == at(80)
    assert parsed.sweeps[1].verdict_at == at(90)
    assert [(r.level, r.violations) for r in parsed.audits] == [
        (canary.AuditLevel.WARNING, (LEDGER_TEXT,)), (canary.AuditLevel.INFO, ()),
    ]
    assert parsed.unparsed == 1


def test_a_missing_journal_extract_is_a_fault_not_a_traceback(project, tmp_path, capsys):
    root, _shas, _extract = project
    out = tmp_path / 'canary.md'

    rc = run_readout(root, tmp_path / 'absent-journal.txt', out)

    assert rc == readout.EXIT_FAULT
    assert 'absent-journal.txt' in capsys.readouterr().err
    assert not out.exists()


def test_a_missing_runs_db_is_a_fault_not_a_traceback(tmp_path, capsys):
    root = tmp_path / 'project'
    build_repo(root)
    extract = tmp_path / 'journal.txt'
    extract.write_text('', encoding='utf-8')
    out = tmp_path / 'canary.md'

    rc = run_readout(root, extract, out)

    assert rc == readout.EXIT_FAULT
    assert str(runs_db(root)) in capsys.readouterr().err
    assert not out.exists()
