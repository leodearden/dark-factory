"""`client-feed publish`, `snapshot` and `dump-state`: the publisher run from its command line, in process and installed."""

import contextlib
import json
import os
import signal
import stat
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml
from _git_repo import make_repo
from _in_dir import write_engagement
from _stores import EscalationQueueDir, RunsDb, TasksDb

from client_feed.cli import main
from client_feed.contract import (
    FEED_FILENAME,
    STATE_FILENAME,
    MergeLanded,
)
from client_feed.feed_io import read_feed_lines
from client_feed.fold import fold
from client_feed.publisher import PassReport

SCRIPT = Path(sys.executable).with_name('client-feed')
WAIT_SECS = 30
BASE = datetime(2026, 10, 21, 10, 0, tzinfo=UTC)


def at(minutes: float) -> datetime:
    return BASE + timedelta(minutes=minutes)


def write_config(tmp_path: Path, root: Path, **fields: Any) -> Path:
    config = tmp_path / 'policy.yaml'
    config.write_text(
        yaml.safe_dump({
            'project_id': 'acme',
            'project_root': str(root),
            'out_dir': str(tmp_path / 'out'),
            'in_dir': str(tmp_path / 'in'),
            'interval_seconds': 1,
            'visible_globs': ['README.md', 'src/**'],
            **fields,
        })
    )
    return config


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """main holding `Merge task/7 into main`, with task 7 done and its landing in runs.db."""
    repo = make_repo(tmp_path / 'root')
    repo.write('.gitignore', '.taskmaster/\ndata/\n')
    repo.write('README.md', 'hello\n')
    repo.commit('Initial', when=at(-60))
    repo.branch('task/7')
    repo.checkout('task/7')
    repo.write('src/widget.py', 'a = 1\n')
    repo.commit('Add the widget', when=at(-30))
    repo.checkout('main')
    merge = repo.merge_no_ff('task/7', 'Merge task/7 into main', when=at(-20))
    TasksDb.create(repo.root).add(
        7, title='Ship /home/leo/x', status='done', updated_at='2026-10-21T09:00:00.000Z'
    )
    EscalationQueueDir.create(repo.root)
    runs = RunsDb.create(repo.root)
    runs.emit('phase_enter', at=at(-25), task_id=7, phase='merge')
    runs.emit(
        'merge_finalized', at=at(-20), task_id=7, phase='merge', data={'state': 'done', 'merge_sha': merge}
    )
    return repo.root


@pytest.fixture
def out_dir(tmp_path: Path) -> Path:
    path = tmp_path / 'out'
    path.mkdir()
    return path


@pytest.fixture
def config(tmp_path: Path, root: Path, out_dir: Path) -> Path:
    write_engagement(tmp_path / 'in', project_id='acme')
    return write_config(tmp_path, root)


def cli(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def validates(capsys: pytest.CaptureFixture[str], out_dir: Path) -> bool:
    code, out, err = cli(capsys, 'validate-feed', '--out-dir', str(out_dir))
    assert code in (0, 1), out + err
    return code == 0


def merge_landings(out_dir: Path) -> Counter[str]:
    return Counter(line.commit for line in read_feed_lines(out_dir) if isinstance(line, MergeLanded))


def test_snapshot_publishes_one_pass_and_prints_its_report(
    config: Path, out_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, err = cli(capsys, 'snapshot', '--config', str(config))

    assert code == 0, err
    report = PassReport.model_validate_json(out)
    assert report.appended > 0
    assert report.degraded == ()
    assert (out_dir / FEED_FILENAME).is_file() and (out_dir / STATE_FILENAME).is_file()
    assert validates(capsys, out_dir)


def test_snapshot_of_an_engagement_for_another_project_exits_1_naming_both_ids(
    tmp_path: Path, config: Path, out_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_engagement(tmp_path / 'in', project_id='other')

    code, out, err = cli(capsys, 'snapshot', '--config', str(config))

    assert (code, out) == (1, '')
    assert "'other'" in err
    assert "'acme'" in err
    assert not (out_dir / FEED_FILENAME).exists()


def test_every_out_file_is_group_readable_under_any_umask_and_takes_the_setgid_group(
    config: Path, out_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    supplementary = [gid for gid in os.getgroups() if gid != os.getegid()]
    if supplementary:
        os.chown(out_dir, -1, supplementary[0])
    out_dir.chmod(0o2750)
    previous = os.umask(0o077)
    try:
        code, _, err = cli(capsys, 'snapshot', '--config', str(config))
    finally:
        os.umask(previous)

    assert code == 0, err
    directory = out_dir.stat()
    assert directory.st_mode & stat.S_ISGID
    written = sorted(out_dir.iterdir())
    assert {path.name for path in written} >= {FEED_FILENAME, STATE_FILENAME, 'cursor.json'}
    for path in written:
        status = path.stat()
        assert (path.name, status.st_mode & 0o777) == (path.name, 0o640)
        assert (path.name, status.st_gid) == (path.name, directory.st_gid)


def every_path(tmp_path: Path) -> list[Path]:
    return sorted(tmp_path.rglob('*'))


@pytest.mark.parametrize('command', ['snapshot', 'publish'])
def test_a_policy_whose_out_dir_is_not_owned_is_refused_and_nothing_is_written(
    tmp_path: Path, root: Path, command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(tmp_path, root, out_dir='/')
    before = every_path(tmp_path)

    code, out, err = cli(capsys, command, '--config', str(config))

    assert (code, out) == (1, '')
    assert 'out_dir_not_owned' in err
    assert every_path(tmp_path) == before
    assert not Path('/', FEED_FILENAME).exists()


@pytest.mark.parametrize('command', ['snapshot', 'publish'])
def test_a_symlinked_feed_is_refused_and_nothing_is_written(
    tmp_path: Path, config: Path, out_dir: Path, command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / 'elsewhere.jsonl'
    target.write_text('')
    (out_dir / FEED_FILENAME).symlink_to(target)

    code, out, err = cli(capsys, command, '--config', str(config))

    assert (code, out) == (1, '')
    assert 'symlink' in err
    assert [path.name for path in out_dir.iterdir()] == [FEED_FILENAME]
    assert target.read_text() == ''


def test_dump_state_prints_the_fold_of_the_feed(
    config: Path, out_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli(capsys, 'snapshot', '--config', str(config))[0] == 0

    code, out, err = cli(capsys, 'dump-state', '--out-dir', str(out_dir))

    assert code == 0, err
    lines = read_feed_lines(out_dir)
    assert json.loads(out) == fold(lines, project_id=lines[0].project_id).model_dump(mode='json')


def test_dump_state_of_a_missing_feed_exits_1_naming_it(
    out_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, err = cli(capsys, 'dump-state', '--out-dir', str(out_dir))

    assert (code, out) == (1, '')
    assert FEED_FILENAME in err


def heartbeats(out_dir: Path) -> int:
    """The heartbeats on the feed's complete lines, read while a publisher may be appending."""
    try:
        return sum(1 for line in read_feed_lines(out_dir) if line.type == 'heartbeat')
    except FileNotFoundError:
        return 0


def wait_for(condition: Callable[[], bool], process: subprocess.Popen[bytes], log: Path) -> None:
    deadline = time.monotonic() + WAIT_SECS
    while not condition():
        assert process.poll() is None, f'publish exited early: {log.read_text()}'
        assert time.monotonic() < deadline, f'publish made no progress in {WAIT_SECS}s: {log.read_text()}'
        time.sleep(0.1)


@contextlib.contextmanager
def publishing(config: Path, log: Path) -> Iterator[subprocess.Popen[bytes]]:
    """The installed console script publishing under `config`, killed on the way out if still running."""
    assert SCRIPT.is_file(), f'console script not installed at {SCRIPT}; run uv sync --all-packages'
    with log.open('wb') as sink:
        process = subprocess.Popen(
            [str(SCRIPT), 'publish', '--config', str(config)], stdout=sink, stderr=subprocess.STDOUT
        )
        try:
            yield process
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=WAIT_SECS)


def script_validates(out_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(SCRIPT), 'validate-feed', '--out-dir', str(out_dir)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_publish_loops_until_sigterm_then_exits_0(tmp_path: Path, config: Path, out_dir: Path) -> None:
    log = tmp_path / 'publish.log'
    with publishing(config, log) as process:
        wait_for(lambda: heartbeats(out_dir) >= 2, process, log)

        process.send_signal(signal.SIGTERM)

        assert process.wait(timeout=WAIT_SECS) == 0, log.read_text()
    validated = script_validates(out_dir)
    assert validated.returncode == 0, validated.stdout + validated.stderr


def test_a_publisher_killed_mid_run_is_recovered_by_the_next_snapshot(
    tmp_path: Path, config: Path, out_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = tmp_path / 'publish.log'
    with publishing(config, log) as process:
        wait_for(lambda: heartbeats(out_dir) >= 1, process, log)
        process.kill()
        process.wait(timeout=WAIT_SECS)

    code, _, err = cli(capsys, 'snapshot', '--config', str(config))

    assert code == 0, err
    assert validates(capsys, out_dir)
    landed = merge_landings(out_dir)
    assert landed and set(landed.values()) == {1}
