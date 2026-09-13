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
