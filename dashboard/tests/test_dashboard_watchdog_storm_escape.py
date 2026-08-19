"""B4's literal acceptance signal: exactly ONE born-at-L2 record on disk.

The bulk of the watchdog's tests live in tests/scripts/test_dashboard_watchdog.py,
which runs in the ``scripts/`` verify lane (``uv run --project shared pytest
tests/scripts/``).  That environment is stdlib-only — ``shared`` does not depend
on ``escalation`` — so those tests can only assert the SHAPE of the argv the
watchdog builds.

An argv-shape assertion is not the acceptance criterion.  ``escalation submit``
stamps ``level=2`` directly, bypassing the escalation server's severity
chokepoint, and therefore enforces its own boundary: ``--severity`` restricted
to BORN_AT_L2_SEVERITIES via argparse choices, and an ``--agent-role`` rejected
unless it carries a ``harness-``/``orchestrator-`` sentinel prefix.  A watchdog
whose argv looks plausible but trips either check would go quiet at the restart
ceiling with NOBODY TOLD — the worst of both behaviours, and precisely the
failure this file exists to catch.

This directory can import ``escalation`` (dashboard/pyproject.toml already
declares it as a workspace dependency), so here the fake ``subprocess.run``
dispatches the watchdog's captured argv into the REAL
``escalation.submit.main()`` against a tmp queue dir, and the assertions are
made on the resulting ``esc-*.json`` record.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import types

import pytest
from escalation.models import BORN_AT_L2_SEVERITIES

REPO_ROOT = pathlib.Path(__file__).parents[2]
WATCHDOG_PATH = REPO_ROOT / "scripts" / "dashboard-watchdog.py"


def _load_watchdog() -> types.ModuleType:
    """Load scripts/dashboard-watchdog.py by path (hyphenated → unimportable).

    Re-invoked per simulated tick: the timer fires a fresh ``Type=oneshot``
    process every 30s, so a fresh module load is the honest simulation.
    """
    spec = importlib.util.spec_from_file_location("dashboard_watchdog", WATCHDOG_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _StormHarness:
    """Fake ``subprocess.run`` that dispatches escalations into the real writer.

    * ``systemd-cat``            — swallowed (journal logging).
    * ``systemctl ... show``     — answers with a long-ago activation, so no
                                   tick is ever inside the startup grace window.
    * ``systemctl ... <verb>``   — RECORDED, never executed.  Nothing in a test
                                   may actuate the developer's real dashboard.
    * anything else              — the ``uv run ... escalation submit ...``
                                   invocation: the argv AFTER the console-script
                                   name is handed to escalation.submit.main(),
                                   so the record is produced by production code.
    """

    def __init__(self, activated_secs_ago: int = 10_000) -> None:
        import time

        self.active_enter = int(time.time()) - activated_secs_ago
        self.actuations: list[list[str]] = []
        self.submit_argvs: list[list[str]] = []
        self.submit_rcs: list[int] = []

    @property
    def restarts(self) -> list[list[str]]:
        return [a for a in self.actuations if "restart" in a]

    def run(self, argv, *args, **kwargs):
        from escalation.submit import main as submit_main

        argv_list = list(argv)

        if argv_list and argv_list[0] == "systemd-cat":
            return subprocess.CompletedProcess(argv_list, 0, stdout="", stderr="")

        if argv_list[:2] == ["systemctl", "--user"]:
            if "show" in argv_list:
                return subprocess.CompletedProcess(
                    argv_list,
                    0,
                    stdout=f"ActiveEnterTimestamp=@{self.active_enter}\n",
                    stderr="",
                )
            self.actuations.append(argv_list)
            return subprocess.CompletedProcess(argv_list, 0, stdout="", stderr="")

        # `uv run --project <...>/escalation escalation submit --queue-dir ...`
        # Everything from the 'submit' token onward is what the console script
        # would hand to main() as argv.
        assert "submit" in argv_list, f"unrecognised subprocess argv: {argv_list}"
        cli_args = argv_list[argv_list.index("submit") :]
        self.submit_argvs.append(argv_list)
        rc = submit_main(cli_args)
        self.submit_rcs.append(rc)
        return subprocess.CompletedProcess(argv_list, rc, stdout="", stderr="")


@pytest.fixture()
def storm(monkeypatch, tmp_path):
    """Point state + queue dir at tmp_path and install the dispatching fake."""
    monkeypatch.setenv("DASHBOARD_WATCHDOG_STATE", str(tmp_path / "wd" / "state.json"))
    monkeypatch.setenv("DASHBOARD_WATCHDOG_QUEUE_DIR", str(tmp_path / "escalations"))

    harness = _StormHarness()
    monkeypatch.setattr(subprocess, "run", harness.run)
    return harness


def _drive_failing_ticks(monkeypatch, n: int) -> None:
    """Run *n* ticks whose probe always fails, each in a fresh module load."""
    import urllib.error

    for _ in range(n):
        mod = _load_watchdog()

        def refused(*args, **kwargs):
            raise urllib.error.URLError(ConnectionRefusedError(111))

        monkeypatch.setattr(mod.urllib.request, "urlopen", refused)
        mod.tick()


def _queue_records(queue_dir: pathlib.Path) -> list[dict]:
    return [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(queue_dir.glob("esc-*.json"))
    ]


def test_storm_files_exactly_one_born_at_l2_record(monkeypatch, tmp_path, storm):
    """The acceptance signal, end to end through the real writer.

    A full storm — enough failing ticks to exhaust MAX_RESTARTS and then run
    well past the ceiling — must leave exactly ONE record in the queue dir.
    """
    mod = _load_watchdog()
    # MAX_RESTARTS full streaks exhaust the allowance; the next completed
    # streak trips the ceiling. The extra ticks are the "well past it" part.
    ticks = mod.FAIL_STREAK * (mod.MAX_RESTARTS + 1) + 30
    _drive_failing_ticks(monkeypatch, ticks)

    queue_dir = tmp_path / "escalations"
    records = _queue_records(queue_dir)

    assert len(records) == 1, (
        f"expected exactly one born-at-L2 record, found {len(records)} after "
        f"{ticks} failing ticks: {[r.get('id') for r in records]}"
    )
    assert storm.submit_rcs == [0], (
        "escalation submit rejected the watchdog's argv at its argparse "
        f"boundary (return codes: {storm.submit_rcs}) — the ceiling would trip "
        "with nobody told"
    )


def test_the_record_is_a_real_born_at_l2_escalation(monkeypatch, tmp_path, storm):
    """level == 2, an accepted severity, pending, and correctly attributed.

    ``level`` is the field that actually routes this to a human: submit.py
    stamps it directly because the server's chokepoint is bypassed. A record
    written at any other level would sit in the queue as an ordinary L0 and
    wait for a steward that never comes.
    """
    mod = _load_watchdog()
    _drive_failing_ticks(monkeypatch, mod.FAIL_STREAK * (mod.MAX_RESTARTS + 1))

    (record,) = _queue_records(tmp_path / "escalations")

    assert record["level"] == 2
    assert record["severity"] in BORN_AT_L2_SEVERITIES
    assert record["status"] == "pending"
    assert record["task_id"] == "dashboard-watchdog-restart-ceiling"
    assert record["agent_role"] == "harness-dashboard-watchdog"
    assert record["category"] == "infra_issue"
    assert record["summary"].strip()
    assert mod.DASHBOARD_UNIT in record["summary"]


def test_no_restarts_after_the_ceiling_is_reached(monkeypatch, tmp_path, storm):
    """INV-4. Past the ceiling the watchdog stops actuating — permanently,
    until the episode ends.

    Restarts are capped at MAX_RESTARTS no matter how long the outage runs.
    The 2026-07-30 incident was the absence of exactly this bound: 192
    restarts in 3 hours, each one adding downtime to a service that was never
    going to be repaired by restarting.
    """
    mod = _load_watchdog()
    _drive_failing_ticks(monkeypatch, mod.FAIL_STREAK * (mod.MAX_RESTARTS + 1) + 60)

    assert len(storm.restarts) == mod.MAX_RESTARTS, (
        f"expected at most MAX_RESTARTS={mod.MAX_RESTARTS} restarts, "
        f"got {len(storm.restarts)}"
    )
    assert _load_watchdog().load_state()["ceiling_open"] is True


def test_the_escalation_is_filed_before_the_ceiling_silences_the_watchdog(
    monkeypatch, tmp_path, storm
):
    """Ordering matters: the L2 must exist by the time actuation stops.

    If the flag were persisted first and the submit then failed or was skipped,
    the watchdog would fall silent on a dead dashboard with no record anywhere
    — a silent degradation, and the failure mode the storm escape is supposed
    to eliminate rather than introduce.
    """
    mod = _load_watchdog()
    _drive_failing_ticks(monkeypatch, mod.FAIL_STREAK * (mod.MAX_RESTARTS + 1))

    assert len(_queue_records(tmp_path / "escalations")) == 1
    assert _load_watchdog().load_state()["ceiling_open"] is True

    restarts_at_trip = len(storm.restarts)
    _drive_failing_ticks(monkeypatch, 20)

    assert len(storm.restarts) == restarts_at_trip, "actuated after the trip"
    assert len(_queue_records(tmp_path / "escalations")) == 1, "re-filed while open"
