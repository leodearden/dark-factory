"""Tests for scripts/install-load-sampler.sh and the two systemd unit files it
installs (task 3592, leaf δ of plans/load-throttle-harmonisation-prd.md).

Drives the installer via subprocess with `scripts/tests/conftest.py::fake_systemctl`
first on PATH, and reads the committed units through
`scripts/tests/conftest.py::unit_values`. Real systemd is never touched, and
DARK_FACTORY_ROOT is pointed at a tmp tree so the row check never reads the
live corpus.

The units live in dashboard/ rather than scripts/, which is the only way this
installer differs in shape from its siblings.

WHY THE UNIT FILES ARE PINNED HERE. The installer copies them BYTE FOR BYTE,
so for the two properties this task actually turns on -- the `--no-sync` flag
on ExecStart, and the 5 s cadence every sizing number in the
plan rests on -- the committed file IS the behaviour. There is nothing else
to test.
"""
from __future__ import annotations

import os
import sqlite3
import subprocess
import time
from pathlib import Path

import pytest
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

SCRIPT = Path(__file__).parent.parent / 'install-load-sampler.sh'
REPO_ROOT = Path(__file__).parent.parent.parent
TEMPLATES_DIR = REPO_ROOT / 'dashboard'

SERVICE_NAME = 'dark-factory-load-sampler.service'
TIMER_NAME = 'dark-factory-load-sampler.timer'


# ── the committed unit files ────────────────────────────────────────────────


def test_execstart_pins_the_venv_against_mutation(unit_values):
    """--no-sync, and the reason is not cosmetic.

    There is ONE shared root .venv per checkout with every workspace member
    installed editable into it, and a plain `uv run --project <member>` was
    measured UNINSTALLING a sibling from it. At this unit's 5 s cadence an
    un-flagged ExecStart performs 17,280 env re-syncs a day against the MAIN
    checkout's venv -- i.e. it would intermittently break running
    orchestrators and verify runs. `--no-sync` is the flag that prevents it.

    The ABSENCE of a lockfile flag is asserted too, and is the more fragile
    half: `--frozen` and `--locked` both read as strengthening the line, and
    both are no-ops once --no-sync is present (measured on uv 0.11.6 -- see the
    unit's own comment for the three rcs). A no-op flag that looks load-bearing
    is worse than no flag, so the seam that would let one back in is pinned.
    """
    exec_start, = unit_values(TEMPLATES_DIR / SERVICE_NAME, 'Service', 'ExecStart')

    assert '--no-sync' in exec_start.split(), exec_start
    assert not {'--frozen', '--locked'} & set(exec_start.split()), (
        'a lockfile flag is a no-op alongside --no-sync; it buys nothing and '
        f'reads as a guarantee the unit does not have: {exec_start}'
    )
    assert exec_start.endswith('--project sampler python -m sampler'), exec_start


def test_service_is_a_oneshot_in_the_production_checkout(unit_values):
    """Type=oneshot is what makes `systemctl start` block until the tick ends.

    That is what step-22's row check relies on: the installer gets a
    deterministic point at which to query, rather than sleep-polling a 5 s
    timer.
    """
    service = TEMPLATES_DIR / SERVICE_NAME

    assert unit_values(service, 'Service', 'Type') == ['oneshot']
    assert unit_values(service, 'Service', 'WorkingDirectory') == ['%h/src/dark-factory']


def test_timer_fires_every_five_seconds(unit_values):
    """The cadence every sizing number in this task rests on.

    17,280 ticks/day x 25 metrics = 432,000 rows/day = 12.96M rows at the
    30-day retention this task also lands. Pinned rather than assumed, because
    changing it silently invalidates the retention sizing.
    """
    timer = TEMPLATES_DIR / TIMER_NAME

    assert unit_values(timer, 'Timer', 'OnUnitActiveSec') == ['5s']
    assert unit_values(timer, 'Install', 'WantedBy') == ['timers.target']


# ── the installer ───────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
    ts INTEGER NOT NULL,
    metric TEXT NOT NULL,
    value REAL NOT NULL,
    window_mean REAL,
    window_max REAL
);
CREATE INDEX IF NOT EXISTS idx_samples_metric_ts ON samples (metric, ts);
"""


def _seed_db(root: Path, *, rows: bool = True, ts_offset: int = 60) -> Path:
    """Build a DB with the real schema where the installed UNIT would write it.

    *ts_offset* is seconds relative to now. The default is in the FUTURE on
    purpose: the installer records a high-water ts before the kick and then
    looks for a row at or after it, and the fake systemctl's `start` does not
    actually run the sampler, so a row dated "now" would race the script's own
    clock. A NEGATIVE offset plants the row the high-water mark exists to
    reject -- a leftover from an earlier install.
    """
    db_path = root / 'data' / 'load-samples.db'
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(_SCHEMA)
        if rows:
            conn.execute(
                'INSERT INTO samples (ts, metric, value) VALUES (?, ?, ?)',
                (int(time.time()) + ts_offset, 'runqueue_ratio', 4.06),
            )
        conn.commit()
    finally:
        conn.close()
    return db_path


def _run_script(fake_systemctl, tmp_path: Path, *, env=None, reset_state=True):
    if reset_state:
        fake_systemctl.reset()

    full_env = dict(os.environ)
    full_env['PATH'] = f'{fake_systemctl.bin_dir}{os.pathsep}{full_env["PATH"]}'
    if env:
        full_env.update(env)
    return subprocess.run(
        ['bash', str(SCRIPT)],
        env=full_env, capture_output=True, text=True, timeout=60,
    )


def _install_env(tmp_path: Path, root: Path) -> dict[str, str]:
    """Env for one install run: where units go, and where the UNIT will run.

    The DB the installer verifies is derived from the service's effective
    WorkingDirectory, asked of systemd -- so the seam that redirects it at a
    tmp tree is the fake systemctl's answer, not an environment variable the
    real unit never sees. `DARK_FACTORY_ROOT` is deliberately NOT set here:
    the installer must not follow it, and
    test_the_verified_db_follows_the_unit_not_the_invoking_shell holds it to
    that.
    """
    return {
        'XDG_CONFIG_HOME': str(tmp_path / 'xdg-config'),
        'FAKE_SYSTEMCTL_WORKDIR': str(root),
    }


def test_script_is_executable():
    """δ'/ε1/ε2 name this path in metadata.before_done.script, which validates
    path-exists-AND-executable at submit_task time."""
    assert os.access(SCRIPT, os.X_OK), (
        f'Expected {SCRIPT} to be executable (os.X_OK); run: chmod +x {SCRIPT}')


def test_install_copies_both_units_verbatim_and_enables_the_timer(fake_systemctl, tmp_path: Path):
    root = tmp_path / 'root'
    _seed_db(root)

    result = _run_script(fake_systemctl, tmp_path, env=_install_env(tmp_path, root))
    assert result.returncode == 0, (
        f'stdout={result.stdout!r} stderr={result.stderr!r}')

    unit_dir = tmp_path / 'xdg-config' / 'systemd' / 'user'
    for name in (SERVICE_NAME, TIMER_NAME):
        installed = unit_dir / name
        assert installed.is_file(), f'Expected {installed} to exist after install'
        assert installed.read_bytes() == (TEMPLATES_DIR / name).read_bytes()

    calls = fake_systemctl.calls()
    assert ['daemon-reload'] in calls, calls
    assert ['enable', '--now', TIMER_NAME] in calls, calls
    assert ['list-timers', '--all'] in calls, calls
    assert ['start', SERVICE_NAME] in calls, calls


def test_the_timer_is_verified_before_the_service_is_kicked(fake_systemctl, tmp_path: Path):
    """The ordering lesson scripts/install-flag-marker-sweep-timer.sh documents.

    A failing kick ends the run -- by the explicit `exit 1` inside the
    `if ! systemctl ...` wrapper, not by `set -e`, which that wrapper
    suppresses -- so a kick placed first means the install's stated guarantee,
    that the timer IS armed, never gets checked at all.
    """
    root = tmp_path / 'root'
    _seed_db(root)

    result = _run_script(fake_systemctl, tmp_path, env=_install_env(tmp_path, root))
    assert result.returncode == 0, result.stderr

    verbs = [call[0] for call in fake_systemctl.calls()]
    assert 'list-timers' in verbs and 'start' in verbs, verbs
    assert verbs.index('list-timers') < verbs.index('start'), verbs


def test_install_fails_loud_when_the_timer_is_absent_from_list_timers(fake_systemctl, tmp_path: Path):
    """Catches "enable nominally succeeded but the unit is absent" -- the case
    a bare `enable` exit code cannot distinguish. And it must stop there."""
    root = tmp_path / 'root'
    _seed_db(root)

    result = _run_script(fake_systemctl, tmp_path, env={
        **_install_env(tmp_path, root),
        'FAKE_SYSTEMCTL_OMIT_LIST_TIMERS': '1',
    })

    assert result.returncode != 0
    assert TIMER_NAME in result.stderr, result.stderr
    verbs = [call[0] for call in fake_systemctl.calls()]
    assert 'start' not in verbs, f'kicked the service after a failed verify: {verbs}'


def test_a_db_with_a_fresh_row_passes_the_row_check(fake_systemctl, tmp_path: Path):
    root = tmp_path / 'root'
    _seed_db(root)

    result = _run_script(fake_systemctl, tmp_path, env=_install_env(tmp_path, root))

    assert result.returncode == 0, (
        f'stdout={result.stdout!r} stderr={result.stderr!r}')


def test_an_empty_db_fails_the_row_check_naming_the_path(fake_systemctl, tmp_path: Path):
    """The row check must be able to FAIL, not just narrate.

    An installer that always exits 0 would report a healthy install for a
    sampler that writes nothing, which is the entire failure this check is for.
    """
    root = tmp_path / 'root'
    db_path = _seed_db(root, rows=False)

    result = _run_script(fake_systemctl, tmp_path, env=_install_env(tmp_path, root))

    assert result.returncode != 0
    assert str(db_path) in result.stderr, result.stderr


def test_an_absent_db_fails_the_row_check_naming_the_path(fake_systemctl, tmp_path: Path):
    root = tmp_path / 'root'
    root.mkdir()

    result = _run_script(fake_systemctl, tmp_path, env=_install_env(tmp_path, root))

    assert result.returncode != 0
    assert str(root / 'data' / 'load-samples.db') in result.stderr, result.stderr


def test_the_verified_db_follows_the_unit_not_the_invoking_shell(fake_systemctl, tmp_path: Path):
    """The installer must check the file the UNIT writes, not the operator's env.

    sampler/__main__.py resolves <DARK_FACTORY_ROOT or CWD>/data and the unit
    sets no Environment=, so under systemd the root is WorkingDirectory. An
    installer that read DARK_FACTORY_ROOT from its own shell agreed with the
    unit only while the variable was UNSET -- exported (a worktree, the case
    its comment claimed to protect) it verified a file nothing writes, and
    exited 1 saying "the sampler wrote nothing" about a healthy install.

    Here the shell variable names a root holding a perfectly good DB and the
    unit's directory holds none. Following the variable passes; following the
    unit fails, naming the path the sampler would actually have written.
    """
    unit_root = tmp_path / 'unit-root'
    unit_root.mkdir()
    decoy_root = tmp_path / 'decoy-root'
    _seed_db(decoy_root)

    result = _run_script(fake_systemctl, tmp_path, env={
        **_install_env(tmp_path, unit_root),
        'DARK_FACTORY_ROOT': str(decoy_root),
    })

    assert result.returncode != 0, result.stdout
    assert str(unit_root / 'data' / 'load-samples.db') in result.stderr, result.stderr


def test_a_leftover_row_from_an_earlier_install_does_not_pass_the_kick_check(
    fake_systemctl, tmp_path: Path
):
    """What `pre_kick_ts` is FOR, and nothing exercised it.

    Every other row-check test seeds a row dated after the high-water mark, so
    rewriting the query to `SELECT COUNT(*) FROM samples` -- dropping the
    high-water clause entirely -- left all of them green. Then a re-install on
    a host where the sampler had stopped would report success on rows a
    previous install wrote, which is exactly the "the timer is armed but the
    tick wrote nothing" case the check exists to catch.
    """
    root = tmp_path / 'root'
    db_path = _seed_db(root, ts_offset=-3600)

    result = _run_script(fake_systemctl, tmp_path, env=_install_env(tmp_path, root))

    assert result.returncode != 0, result.stdout
    assert str(db_path) in result.stderr, result.stderr


def test_a_failed_kick_stops_the_install_naming_the_service(fake_systemctl, tmp_path: Path):
    """The kick-failure branch, unreachable until the fake could refuse to start.

    It carries the install's most useful sentence -- the timer IS armed even
    though the tick failed -- and the row check below it must not run, because
    a kick that never happened cannot have written a row and the DB message
    would misdirect an operator onto the sampler's writes.
    """
    root = tmp_path / 'root'
    _seed_db(root)

    result = _run_script(fake_systemctl, tmp_path, env={
        **_install_env(tmp_path, root),
        'FAKE_SYSTEMCTL_FAIL_START': '1',
    })

    assert result.returncode != 0, result.stdout
    assert SERVICE_NAME in result.stderr, result.stderr
    assert 'load-samples.db' not in result.stderr, (
        'the row check ran after a kick that never happened: '
        f'{result.stderr}'
    )


def test_running_twice_is_idempotent(fake_systemctl, tmp_path: Path):
    root = tmp_path / 'root'
    _seed_db(root)
    env = _install_env(tmp_path, root)

    first = _run_script(fake_systemctl, tmp_path, env=env)
    assert first.returncode == 0, first.stderr
    unit_dir = tmp_path / 'xdg-config' / 'systemd' / 'user'
    before = {n: (unit_dir / n).read_bytes() for n in (SERVICE_NAME, TIMER_NAME)}

    second = _run_script(fake_systemctl, tmp_path, env=env)
    assert second.returncode == 0, second.stderr
    after = {n: (unit_dir / n).read_bytes() for n in (SERVICE_NAME, TIMER_NAME)}
    assert before == after
