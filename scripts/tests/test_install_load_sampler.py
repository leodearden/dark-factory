"""Tests for scripts/install-load-sampler.sh and the two systemd unit files it
installs (task 3592, leaf δ of plans/load-throttle-harmonisation-prd.md).

Drives the installer via subprocess with a FAKE `systemctl` shimmed onto PATH
(records every invocation, minus `--user`, into a shared JSON state file) --
mirroring test_install_memory_metadata_coverage_census_timer.py. Real systemd
is never touched, and DARK_FACTORY_ROOT is pointed at a tmp tree so the row
check never reads the live corpus.

The units live in dashboard/ rather than scripts/, which is the only way this
installer differs in shape from its siblings.

WHY THE UNIT FILES ARE PINNED HERE. The installer copies them BYTE FOR BYTE,
so for the two properties this task actually turns on -- the `--frozen
--no-sync` flags on ExecStart, and the 5 s cadence every sizing number in the
plan rests on -- the committed file IS the behaviour. There is nothing else
to test.
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import time
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / 'install-load-sampler.sh'
REPO_ROOT = Path(__file__).parent.parent.parent
TEMPLATES_DIR = REPO_ROOT / 'dashboard'

SERVICE_NAME = 'dark-factory-load-sampler.service'
TIMER_NAME = 'dark-factory-load-sampler.timer'


# ── the committed unit files ────────────────────────────────────────────────


def _directives(name: str) -> dict[str, list[tuple[str, str]]]:
    """Parse a systemd unit into `{section: [(key, value), ...]}`.

    A raw-substring check on a unit's text cannot tell a live DIRECTIVE from
    the COMMENT that explains it, and this service unit comments heavily --
    including prose about the very ExecStart flags asserted below. Duplicate
    keys are preserved as separate pairs rather than collapsed, matching the
    sibling suite's parser.
    """
    path = TEMPLATES_DIR / name
    sections: dict[str, list[tuple[str, str]]] = {}
    current = ''
    for raw in path.read_text().splitlines():
        line = raw.strip()
        # FULL-LINE comments only: systemd treats `#`/`;` as a comment lead-in
        # at the start of a line, and a value may legitimately contain either.
        if not line or line[0] in '#;':
            continue
        if line.startswith('[') and line.endswith(']'):
            current = line[1:-1].strip()
            sections.setdefault(current, [])
            continue
        if '=' not in line:
            continue
        key, value = line.split('=', 1)
        sections.setdefault(current, []).append((key.strip(), value.strip()))
    return sections


def _values(directives, section: str, key: str) -> list[str]:
    """Every value declared for `key` under `[section]`, in file order.

    A list, not a scalar: asserting `== ['x']` pins both the value AND that it
    is declared exactly once.
    """
    return [v for k, v in directives.get(section, []) if k == key]


def test_execstart_pins_the_venv_against_mutation():
    """--frozen --no-sync, and the reason is not cosmetic.

    There is ONE shared root .venv per checkout with every workspace member
    installed editable into it, and a plain `uv run --project <member>` was
    measured UNINSTALLING a sibling from it. At this unit's 5 s cadence an
    un-flagged ExecStart performs 17,280 env re-syncs a day against the MAIN
    checkout's venv -- i.e. it would intermittently break running
    orchestrators and verify runs. `--no-sync` is the flag that prevents the
    mutation; `--frozen` additionally pins the lockfile, so a genuinely
    unsynced venv fails loudly in the journal instead of silently repairing
    itself by damaging the venv.
    """
    exec_start, = _values(_directives(SERVICE_NAME), 'Service', 'ExecStart')

    assert '--frozen' in exec_start.split(), exec_start
    assert '--no-sync' in exec_start.split(), exec_start
    assert exec_start.endswith('--project sampler python -m sampler'), exec_start


def test_service_is_a_oneshot_in_the_production_checkout():
    """Type=oneshot is what makes `systemctl start` block until the tick ends.

    That is what step-22's row check relies on: the installer gets a
    deterministic point at which to query, rather than sleep-polling a 5 s
    timer.
    """
    service = _directives(SERVICE_NAME)

    assert _values(service, 'Service', 'Type') == ['oneshot']
    assert _values(service, 'Service', 'WorkingDirectory') == ['%h/src/dark-factory']


def test_timer_fires_every_five_seconds():
    """The cadence every sizing number in this task rests on.

    17,280 ticks/day x 25 metrics = 432,000 rows/day = 12.96M rows at the
    30-day retention this task also lands. Pinned rather than assumed, because
    changing it silently invalidates the retention sizing.
    """
    timer = _directives(TIMER_NAME)

    assert _values(timer, 'Timer', 'OnUnitActiveSec') == ['5s']
    assert _values(timer, 'Install', 'WantedBy') == ['timers.target']


# ── the installer ───────────────────────────────────────────────────────────

_FAKE_SYSTEMCTL_SRC = '''#!/usr/bin/env python3
"""Fake `systemctl` for testing install-load-sampler.sh.

Records every invocation (minus `--user`) into a JSON state file at
$FAKE_SYSTEMCTL_STATE. `enable --now <unit>` marks each non-flag arg as an
enabled unit; `start <unit>` is recorded and always succeeds; `list-timers`
echoes one line per *.timer enabled so far THIS RUN, unless
FAKE_SYSTEMCTL_OMIT_LIST_TIMERS=1 -- simulating the self-verify failure where
`enable` nominally succeeded but the unit is absent from `list-timers`.
"""
import json
import os
import sys

STATE_PATH = os.environ["FAKE_SYSTEMCTL_STATE"]


def _load():
    with open(STATE_PATH) as f:
        return json.load(f)


def _save(state):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f)


def main(argv):
    args = [a for a in argv[1:] if a != "--user"]
    if not args:
        return 1
    verb, rest = args[0], args[1:]

    state = _load()
    state.setdefault("calls", []).append(args)

    if verb == "daemon-reload":
        _save(state)
        return 0

    if verb == "enable":
        units = [a for a in rest if not a.startswith("-")]
        enabled = state.setdefault("enabled_timers", [])
        for u in units:
            if u not in enabled:
                enabled.append(u)
        _save(state)
        return 0

    if verb == "start":
        _save(state)
        return 0

    if verb == "list-timers":
        _save(state)
        if os.environ.get("FAKE_SYSTEMCTL_OMIT_LIST_TIMERS") == "1":
            print("0 timers listed.")
            return 0
        enabled = state.get("enabled_timers", [])
        for unit in enabled:
            service = unit.replace(".timer", ".service")
            print(f"Mon 2026-09-14 05:00:00 UTC 4s left n/a n/a {unit} {service}")
        print(f"{len(enabled)} timers listed.")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
'''

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


def _fake_systemctl(tmp_path: Path):
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir(exist_ok=True)
    fake = bin_dir / 'systemctl'
    fake.write_text(_FAKE_SYSTEMCTL_SRC)
    fake.chmod(0o755)

    state_path = tmp_path / 'systemctl_state.json'
    state_path.write_text(json.dumps({'calls': [], 'enabled_timers': []}))
    return bin_dir, state_path


def _systemctl_calls(tmp_path: Path) -> list[list[str]]:
    state_path = tmp_path / 'systemctl_state.json'
    if not state_path.is_file():
        return []
    return json.loads(state_path.read_text())['calls']


def _seed_db(root: Path, *, rows: bool = True) -> Path:
    """Build a DB with the real schema at the DARK_FACTORY_ROOT-derived path."""
    db_path = root / 'data' / 'load-samples.db'
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(_SCHEMA)
        if rows:
            # Dated slightly in the FUTURE on purpose. The installer records a
            # high-water ts before the kick and then looks for a row at or
            # after it; the fake systemctl's `start` does not actually run the
            # sampler, so a row dated "now" would race the script's own clock.
            conn.execute(
                'INSERT INTO samples (ts, metric, value) VALUES (?, ?, ?)',
                (int(time.time()) + 60, 'runqueue_ratio', 4.06),
            )
        conn.commit()
    finally:
        conn.close()
    return db_path


def _run_script(tmp_path: Path, *, env=None, reset_state=True):
    bin_dir, state_path = _fake_systemctl(tmp_path) if reset_state else (
        tmp_path / 'bin', tmp_path / 'systemctl_state.json')

    full_env = dict(os.environ)
    full_env['PATH'] = f'{bin_dir}{os.pathsep}{full_env["PATH"]}'
    full_env['FAKE_SYSTEMCTL_STATE'] = str(state_path)
    if env:
        full_env.update(env)
    return subprocess.run(
        ['bash', str(SCRIPT)],
        env=full_env, capture_output=True, text=True, timeout=60,
    )


def _install_env(tmp_path: Path, root: Path) -> dict[str, str]:
    return {
        'XDG_CONFIG_HOME': str(tmp_path / 'xdg-config'),
        'DARK_FACTORY_ROOT': str(root),
    }


def test_script_is_executable():
    """δ'/ε1/ε2 name this path in metadata.before_done.script, which validates
    path-exists-AND-executable at submit_task time."""
    assert os.access(SCRIPT, os.X_OK), (
        f'Expected {SCRIPT} to be executable (os.X_OK); run: chmod +x {SCRIPT}')


def test_install_copies_both_units_verbatim_and_enables_the_timer(tmp_path: Path):
    root = tmp_path / 'root'
    _seed_db(root)

    result = _run_script(tmp_path, env=_install_env(tmp_path, root))
    assert result.returncode == 0, (
        f'stdout={result.stdout!r} stderr={result.stderr!r}')

    unit_dir = tmp_path / 'xdg-config' / 'systemd' / 'user'
    for name in (SERVICE_NAME, TIMER_NAME):
        installed = unit_dir / name
        assert installed.is_file(), f'Expected {installed} to exist after install'
        assert installed.read_bytes() == (TEMPLATES_DIR / name).read_bytes()

    calls = _systemctl_calls(tmp_path)
    assert ['daemon-reload'] in calls, calls
    assert ['enable', '--now', TIMER_NAME] in calls, calls
    assert ['list-timers', '--all'] in calls, calls
    assert ['start', SERVICE_NAME] in calls, calls


def test_the_timer_is_verified_before_the_service_is_kicked(tmp_path: Path):
    """The ordering lesson scripts/install-flag-marker-sweep-timer.sh documents.

    Under `set -euo pipefail` a failing kick ABORTS the script, so a kick
    placed first means the install's stated guarantee -- the timer IS armed --
    never gets checked at all.
    """
    root = tmp_path / 'root'
    _seed_db(root)

    result = _run_script(tmp_path, env=_install_env(tmp_path, root))
    assert result.returncode == 0, result.stderr

    verbs = [call[0] for call in _systemctl_calls(tmp_path)]
    assert 'list-timers' in verbs and 'start' in verbs, verbs
    assert verbs.index('list-timers') < verbs.index('start'), verbs


def test_install_fails_loud_when_the_timer_is_absent_from_list_timers(tmp_path: Path):
    """Catches "enable nominally succeeded but the unit is absent" -- the case
    a bare `enable` exit code cannot distinguish. And it must stop there."""
    root = tmp_path / 'root'
    _seed_db(root)

    result = _run_script(tmp_path, env={
        **_install_env(tmp_path, root),
        'FAKE_SYSTEMCTL_OMIT_LIST_TIMERS': '1',
    })

    assert result.returncode != 0
    assert TIMER_NAME in result.stderr, result.stderr
    verbs = [call[0] for call in _systemctl_calls(tmp_path)]
    assert 'start' not in verbs, f'kicked the service after a failed verify: {verbs}'


def test_a_db_with_a_fresh_row_passes_the_row_check(tmp_path: Path):
    root = tmp_path / 'root'
    _seed_db(root)

    result = _run_script(tmp_path, env=_install_env(tmp_path, root))

    assert result.returncode == 0, (
        f'stdout={result.stdout!r} stderr={result.stderr!r}')


def test_an_empty_db_fails_the_row_check_naming_the_path(tmp_path: Path):
    """The row check must be able to FAIL, not just narrate.

    An installer that always exits 0 would report a healthy install for a
    sampler that writes nothing, which is the entire failure this check is for.
    """
    root = tmp_path / 'root'
    db_path = _seed_db(root, rows=False)

    result = _run_script(tmp_path, env=_install_env(tmp_path, root))

    assert result.returncode != 0
    assert str(db_path) in result.stderr, result.stderr


def test_an_absent_db_fails_the_row_check_naming_the_path(tmp_path: Path):
    root = tmp_path / 'root'
    root.mkdir()

    result = _run_script(tmp_path, env=_install_env(tmp_path, root))

    assert result.returncode != 0
    assert str(root / 'data' / 'load-samples.db') in result.stderr, result.stderr


def test_running_twice_is_idempotent(tmp_path: Path):
    root = tmp_path / 'root'
    _seed_db(root)
    env = _install_env(tmp_path, root)

    first = _run_script(tmp_path, env=env)
    assert first.returncode == 0, first.stderr
    unit_dir = tmp_path / 'xdg-config' / 'systemd' / 'user'
    before = {n: (unit_dir / n).read_bytes() for n in (SERVICE_NAME, TIMER_NAME)}

    second = _run_script(tmp_path, env=env)
    assert second.returncode == 0, second.stderr
    after = {n: (unit_dir / n).read_bytes() for n in (SERVICE_NAME, TIMER_NAME)}
    assert before == after
