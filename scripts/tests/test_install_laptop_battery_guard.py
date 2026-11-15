"""Tests for the laptop battery guard's committed system units and for
install_laptop_battery_guard.sh, which installs them on leo-laptop (task 9051).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

DEPLOY = Path(__file__).parent.parent / "deploy"
GUARD_SRC = Path(__file__).parent.parent / "laptop_battery_guard.py"
UNIT = "laptop-battery-guard"
SERVICE = DEPLOY / f"{UNIT}.service"
TIMER = DEPLOY / f"{UNIT}.timer"
PYTHON_PREFIX = "/usr/bin/python3 "


def _directive(path, key):
    """Values of every `key=` line. Comment lines start with '#' or ';', so
    they never match."""
    prefix = f"{key}="
    return [
        line[len(prefix):].strip()
        for line in path.read_text().splitlines()
        if line.startswith(prefix)
    ]


class TestTimer:
    def test_polls_every_30s(self):
        assert _directive(TIMER, "OnUnitActiveSec") == ["30s"]

    def test_accuracy_keeps_the_poll_at_30s(self):
        assert _directive(TIMER, "AccuracySec") == ["1s"]

    def test_first_poll_a_minute_after_boot(self):
        assert _directive(TIMER, "OnBootSec") == ["1min"]

    def test_enabled_by_timers_target(self):
        assert _directive(TIMER, "WantedBy") == ["timers.target"]

    def test_declares_no_calendar(self):
        assert _directive(TIMER, "OnCalendar") == []


class TestService:
    def test_is_oneshot(self):
        assert _directive(SERVICE, "Type") == ["oneshot"]

    def test_runs_the_installed_guard_with_system_python(self):
        assert _directive(SERVICE, "ExecStart") == [
            f"{PYTHON_PREFIX}/usr/local/lib/{UNIT}/laptop_battery_guard.py",
        ]

    def test_runs_as_root(self):
        assert _directive(SERVICE, "User") == []


def test_guard_source_is_executable():
    assert os.access(GUARD_SRC, os.X_OK)
