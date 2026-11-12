"""warm-lane-audit.sh refuses a missing mount, and ages a lane by its record.

Drives the SHIPPED script by subprocess against synthetic mounts. The defects
this pins, and why DF's copy is patched directly: README.md "Delta 16" in
orchestrator/scripts/warm-lane/.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from test_warm_lane_scripts_shipped import (
    _BASH,
    WARM_LANE_SCRIPT_DIR,
    _sanitized_env,
)

from orchestrator.lane_lifecycle import LANE_STATE_DIRNAME, LaneRecord, LaneState

AUDIT = WARM_LANE_SCRIPT_DIR / 'warm-lane-audit.sh'


def _env(root: Path, **extra: str) -> dict[str, str]:
    """A hermetic env: no ``REIFY_*`` knob leaks in, and git stops at *root*."""
    env = {
        key: value
        for key, value in _sanitized_env(ceiling=root).items()
        if not key.startswith('REIFY_')
    }
    env.update(extra)
    return env


def _run_audit(
    args: list[str], env: dict[str, str], cwd: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [_BASH, str(AUDIT), *args],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(cwd),
        env=env,
    )


def _git(path: Path, *args: str) -> None:
    subprocess.run(
        [
            'git', '-c', 'user.email=t@t', '-c', 'user.name=t',
            '-c', 'commit.gpgsign=false', *args,
        ],
        cwd=str(path),
        check=True,
        capture_output=True,
        timeout=60,
        env=_env(path.parent),
    )


def _git_lane(path: Path) -> None:
    """Make *path* a git repo on ``main`` with one commit: a resident lane."""
    path.mkdir(parents=True, exist_ok=True)
    _git(path, 'init', '-q', '-b', 'main')
    _git(path, 'commit', '--allow-empty', '-q', '-m', 'initial')


def _lines_with(stderr: str, *needles: str) -> list[str]:
    return [line for line in stderr.splitlines() if all(n in line for n in needles)]


class TestAuditRefusesAMissingMount:
    """No usable mount is a usage error, never a vacuous ``resident=0`` report."""

    @pytest.mark.parametrize(
        'mount_env', [{}, {'REIFY_WARM_LANE_MOUNT': ''}], ids=['unset', 'empty'],
    )
    def test_no_mount_at_all_exits_2(
        self, tmp_path: Path, mount_env: dict[str, str],
    ) -> None:
        proc = _run_audit([], _env(tmp_path, **mount_env), tmp_path)
        assert proc.returncode == 2, (
            f'an audit with no mount must refuse; rc={proc.returncode} '
            f'stdout={proc.stdout!r} stderr={proc.stderr!r}'
        )
        assert proc.stdout == '', f'a refusal reports nothing; {proc.stdout!r}'
        assert '--mount' in proc.stderr, proc.stderr
        assert 'REIFY_WARM_LANE_MOUNT' in proc.stderr, proc.stderr

    @pytest.mark.parametrize('door', ['flag', 'env'])
    def test_a_nonexistent_mount_exits_2_naming_it(
        self, tmp_path: Path, door: str,
    ) -> None:
        absent = tmp_path / 'absent'
        if door == 'flag':
            proc = _run_audit(['--mount', str(absent)], _env(tmp_path), tmp_path)
        else:
            proc = _run_audit(
                [], _env(tmp_path, REIFY_WARM_LANE_MOUNT=str(absent)), tmp_path,
            )
        assert proc.returncode == 2, (
            f'a nonexistent mount must refuse; rc={proc.returncode} '
            f'stdout={proc.stdout!r} stderr={proc.stderr!r}'
        )
        assert proc.stdout == '', f'a refusal reports nothing; {proc.stdout!r}'
        assert str(absent) in proc.stderr, proc.stderr

    def test_a_regular_file_mount_exits_2(self, tmp_path: Path) -> None:
        not_a_dir = tmp_path / 'a-file'
        not_a_dir.write_text('')
        proc = _run_audit(['--mount', str(not_a_dir)], _env(tmp_path), tmp_path)
        assert proc.returncode == 2, (
            f'a non-directory mount must refuse; rc={proc.returncode} '
            f'stderr={proc.stderr!r}'
        )
        assert proc.stdout == '', f'a refusal reports nothing; {proc.stdout!r}'


class TestAuditWarnsOnAnEmptyPool:
    """A real mount with no residents still reports, and says it looks wrong.

    The fixture is the wrong-level mistake: ``--mount`` names the mount point
    above the worktrees dir that holds the lanes.
    """

    @pytest.fixture
    def pool(self, tmp_path: Path) -> Path:
        pool = tmp_path / 'pool'
        _git_lane(pool / 'worktrees' / '_lane-1')
        return pool

    def test_the_mount_point_above_the_lanes_warns_resident_0(
        self, tmp_path: Path, pool: Path,
    ) -> None:
        proc = _run_audit(['--mount', str(pool)], _env(tmp_path), tmp_path)
        assert proc.returncode == 0, f'stderr={proc.stderr!r}'
        assert 'HEADROOM resident=0' in proc.stdout, proc.stdout
        # One line, because the [info] banner also names the mount.
        assert _lines_with(proc.stderr, '[warn]', str(pool), 'resident=0'), (
            f'a zero-resident mount must warn, naming it and resident=0:\n'
            f'{proc.stderr}'
        )

    def test_the_worktrees_dir_itself_does_not_warn(
        self, tmp_path: Path, pool: Path,
    ) -> None:
        proc = _run_audit(
            ['--mount', str(pool / 'worktrees')], _env(tmp_path), tmp_path,
        )
        assert proc.returncode == 0, f'stderr={proc.stderr!r}'
        assert 'HEADROOM resident=1' in proc.stdout, proc.stdout
        assert not _lines_with(proc.stderr, '[warn]', 'resident=0'), proc.stderr


@dataclass(frozen=True)
class _AgeCase:
    """One lane: its two clocks, and what the audit must make of them.

    ``record`` is minutes ago (an int, negative for a stamp ahead of now), a
    literal ``updated_at`` (a str), or ``None`` for a lane with no record.
    """

    lane: str
    record: int | str | None
    dir_age_min: int
    age_source: str
    age_floor: int
    classification: str


#: Every lane is an idle, unreserved ORPHAN with a non-terminal backing task:
#: the LEAKED predicate's exact shape, so age alone decides LEAKED.
_AGE_CASES = (
    # The reify lane-52 incident: churn keeps the dir mtime at ~0.
    _AgeCase('_lane-churn', 180, 0, 'record', 180, 'LEAKED'),
    # The record wins in BOTH directions, not max(record, dir).
    _AgeCase('_lane-fresh', 0, 90, 'record', 0, 'PRESERVED-OK'),
    # A record written under a skewed clock reads age 0, never negative.
    _AgeCase('_lane-future', -30, 90, 'record', 0, 'PRESERVED-OK'),
    _AgeCase('_lane-norec', None, 90, 'dir-mtime', 90, 'LEAKED'),
    # GNU `date -d ''` answers today's midnight; age_source discriminates.
    _AgeCase('_lane-unstamped', '', 90, 'dir-mtime', 90, 'LEAKED'),
    _AgeCase('_lane-badstamp', 'not-a-timestamp', 90, 'dir-mtime', 90, 'LEAKED'),
)


@dataclass(frozen=True)
class _AgedPoolReport:
    table_lines: list[str]
    table_rows: dict[str, dict[str, str]]
    headroom: dict[str, str]
    json_lanes: dict[str, dict]


def _fields(line: str) -> dict[str, str]:
    """A table row or HEADROOM line's ``key=value`` fields."""
    return dict(field.split('=', 1) for field in line.split() if '=' in field)


@pytest.fixture(scope='module')
def aged_pool(tmp_path_factory: pytest.TempPathFactory) -> _AgedPoolReport:
    root = tmp_path_factory.mktemp('aged-pool')
    mount = root / 'mount'
    state_dir = mount / LANE_STATE_DIRNAME
    state_dir.mkdir(parents=True)
    status_cmd = root / 'status-stub.sh'
    status_cmd.write_text("#!/usr/bin/env bash\nprintf 'pending\\n'\n")
    status_cmd.chmod(0o755)

    for task_id, case in enumerate(_AGE_CASES, start=9201):
        lane = mount / case.lane
        _git_lane(lane)
        _git(lane, 'checkout', '-q', '-b', f'task/{task_id}')
        _git(lane, 'commit', '--allow-empty', '-q', '-m', 'ahead of main')
        if case.record is not None:
            stamp = case.record
            if isinstance(stamp, int):
                stamp = (datetime.now(UTC) - timedelta(minutes=stamp)).isoformat()
            record = LaneRecord(state=LaneState.RELEASED, updated_at=stamp)
            (state_dir / f'{case.lane}.json').write_text(record.to_json())
        dir_mtime = time.time() - case.dir_age_min * 60
        os.utime(lane, (dir_mtime, dir_mtime))

    args = ['--mount', str(mount), '--status-cmd', str(status_cmd),
            '--stale-age-min', '60']
    table = _run_audit(args, _env(root), root)
    assert table.returncode == 0, f'stderr={table.stderr!r}'
    as_json = _run_audit([*args, '--format', 'json'], _env(root), root)
    assert as_json.returncode == 0, f'stderr={as_json.stderr!r}'

    lines = [line for line in table.stdout.splitlines() if line.startswith('lane=')]
    headroom = next(
        line for line in table.stdout.splitlines() if line.startswith('HEADROOM ')
    )
    return _AgedPoolReport(
        table_lines=lines,
        table_rows={row['lane']: row for row in map(_fields, lines)},
        headroom=_fields(headroom),
        json_lanes={row['lane']: row for row in json.loads(as_json.stdout)['lanes']},
    )


class TestAuditAgesALaneByItsRecord:
    """``age_min`` comes from the record's ``updated_at`` when it is judgeable.

    The lane dir's mtime is churned by transient files, so it is only the
    fallback, and ``age_source`` names which clock a row was judged by.
    """

    @pytest.mark.parametrize('case', _AGE_CASES, ids=lambda case: case.lane)
    def test_age_source_names_the_clock(
        self, aged_pool: _AgedPoolReport, case: _AgeCase,
    ) -> None:
        assert aged_pool.json_lanes[case.lane].get('age_source') == case.age_source, (
            aged_pool.json_lanes[case.lane]
        )
        assert aged_pool.table_rows[case.lane].get('age_source') == case.age_source, (
            aged_pool.table_rows[case.lane]
        )

    @pytest.mark.parametrize('case', _AGE_CASES, ids=lambda case: case.lane)
    def test_age_min_is_read_off_that_clock(
        self, aged_pool: _AgedPoolReport, case: _AgeCase,
    ) -> None:
        age_min = aged_pool.json_lanes[case.lane]['age_min']
        assert case.age_floor <= age_min < case.age_floor + 60, (
            f'{case.lane} must age from its {case.age_source} clock '
            f'(>= {case.age_floor} min); {aged_pool.json_lanes[case.lane]}'
        )

    @pytest.mark.parametrize('case', _AGE_CASES, ids=lambda case: case.lane)
    def test_the_age_decides_leaked(
        self, aged_pool: _AgedPoolReport, case: _AgeCase,
    ) -> None:
        row = aged_pool.json_lanes[case.lane]
        assert row['classification'] == case.classification, row

    def test_age_source_is_the_appended_last_column(
        self, aged_pool: _AgedPoolReport,
    ) -> None:
        assert len(aged_pool.table_lines) == len(_AGE_CASES), aged_pool.table_lines
        for line in aged_pool.table_lines:
            fields = line.split()
            assert fields[-1].startswith('age_source='), line
            assert fields[-2].startswith('classification='), line

    def test_headroom_leaked_counts_the_record_aged_verdicts(
        self, aged_pool: _AgedPoolReport,
    ) -> None:
        expected = sum(case.classification == 'LEAKED' for case in _AGE_CASES)
        assert aged_pool.headroom['leaked'] == str(expected), aged_pool.headroom
