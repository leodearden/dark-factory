"""warm-lane-audit.sh refuses a missing mount, and ages a lane by its record.

Drives the SHIPPED script by subprocess against synthetic mounts. The defects
this pins, and why DF's copy is patched directly: README.md "Delta 13" in
orchestrator/scripts/warm-lane/.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from test_warm_lane_scripts_shipped import (  # noqa: E402
    _BASH,
    WARM_LANE_SCRIPT_DIR,
    _sanitized_env,
)

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
