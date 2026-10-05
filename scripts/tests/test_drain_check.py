"""Tests for scripts/drain_check.py — the STDLIB-ONLY reader of α's (task
2395) per-unit merge-idle heartbeat, consumed by γ's drain gate in
restart-all-orchestrators.sh (task 2397).

step-1: pure classify(heartbeat, now, fresh_window) taxonomy -- idle / busy
/ stale / absent. No filesystem or subprocess I/O in this module.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from cli_subprocess_timeout import cli_timeout_from_env
from drain_check import classify, heartbeat_path, resolve_fleet_dir

FRESH_WINDOW = 120.0
NOW = 1_000_000.0

UNIT = "orchestrator-dark-factory.service"

SCRIPT = Path(__file__).parent.parent / "drain_check.py"


def _heartbeat(**overrides):
    payload = {
        "unit": UNIT,
        "merge_idle": True,
        "depth": 0,
        "queue_empty": True,
        "ts_epoch": NOW,
    }
    payload.update(overrides)
    return payload


def test_fresh_merge_idle_true_is_idle():
    heartbeat = _heartbeat(merge_idle=True, ts_epoch=NOW)
    assert classify(heartbeat, NOW, FRESH_WINDOW) == "idle"


def test_fresh_merge_idle_false_is_busy():
    heartbeat = _heartbeat(merge_idle=False, ts_epoch=NOW)
    assert classify(heartbeat, NOW, FRESH_WINDOW) == "busy"


def test_ts_epoch_older_than_fresh_window_is_stale():
    heartbeat = _heartbeat(merge_idle=True, ts_epoch=NOW - FRESH_WINDOW - 1)
    assert classify(heartbeat, NOW, FRESH_WINDOW) == "stale"


def test_stale_even_when_merge_idle_false():
    """Staleness is decided on ts_epoch alone -- an old busy heartbeat is
    still 'stale', not 'busy' (the unknown-grace path handles it, not the
    busy/force-poll path)."""
    heartbeat = _heartbeat(merge_idle=False, ts_epoch=NOW - FRESH_WINDOW - 1)
    assert classify(heartbeat, NOW, FRESH_WINDOW) == "stale"


def test_none_heartbeat_is_absent():
    assert classify(None, NOW, FRESH_WINDOW) == "absent"


def test_missing_ts_epoch_is_absent():
    heartbeat = _heartbeat()
    del heartbeat["ts_epoch"]
    assert classify(heartbeat, NOW, FRESH_WINDOW) == "absent"


def test_non_numeric_ts_epoch_is_absent():
    heartbeat = _heartbeat(ts_epoch="not-a-number")
    assert classify(heartbeat, NOW, FRESH_WINDOW) == "absent"


def test_fresh_missing_merge_idle_is_busy():
    """Conservative: ambiguous/missing merge_idle on an otherwise-fresh
    heartbeat classifies as busy, protecting an in-flight merge."""
    heartbeat = _heartbeat(ts_epoch=NOW)
    del heartbeat["merge_idle"]
    assert classify(heartbeat, NOW, FRESH_WINDOW) == "busy"


def test_fresh_ambiguous_merge_idle_is_busy():
    heartbeat = _heartbeat(merge_idle="yes", ts_epoch=NOW)
    assert classify(heartbeat, NOW, FRESH_WINDOW) == "busy"


def test_exactly_at_fresh_window_boundary_is_still_fresh():
    """now - ts_epoch == fresh_window is fresh (<=, not <)."""
    heartbeat = _heartbeat(merge_idle=True, ts_epoch=NOW - FRESH_WINDOW)
    assert classify(heartbeat, NOW, FRESH_WINDOW) == "idle"


# ---------------------------------------------------------------------------
# step-3: resolve_fleet_dir / heartbeat_path / drift-vs-α test
# ---------------------------------------------------------------------------

def test_resolve_fleet_dir_honours_env_override():
    env = {"ORCH_FLEET_DIR": "/tmp/some-fleet-dir"}
    assert resolve_fleet_dir(env) == Path("/tmp/some-fleet-dir")


def test_resolve_fleet_dir_falls_back_to_default_when_unset():
    assert resolve_fleet_dir({}) == drain_check_default_fleet_dir()


def test_resolve_fleet_dir_falls_back_to_default_when_empty():
    assert resolve_fleet_dir({"ORCH_FLEET_DIR": ""}) == drain_check_default_fleet_dir()


def test_heartbeat_path_joins_fleet_dir_and_unit_json():
    fleet_dir = Path("/tmp/some-fleet-dir")
    assert heartbeat_path(fleet_dir, UNIT) == fleet_dir / f"{UNIT}.json"


def drain_check_default_fleet_dir():
    from drain_check import DEFAULT_FLEET_DIR

    return DEFAULT_FLEET_DIR


def test_default_fleet_dir_matches_orchestrator_fleet_heartbeat():
    """DRIFT GUARD: the stdlib-only mirror in drain_check.py must never
    silently diverge from α's canonical DEFAULT_FLEET_DIR."""
    import drain_check
    import orchestrator.fleet_heartbeat as fleet_heartbeat

    assert drain_check.DEFAULT_FLEET_DIR == fleet_heartbeat.DEFAULT_FLEET_DIR


# ---------------------------------------------------------------------------
# step-5: CLI (argparse) tests -- drive via subprocess.run
#
# The budget comes from cli_subprocess_timeout.py, shared with
# test_recon_busy_check.py and test_scan_task_toolcall_leaks.py.
# ---------------------------------------------------------------------------

def _write_raw_heartbeat(fleet_dir: Path, unit: str, **overrides):
    fleet_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "unit": unit,
        "merge_idle": True,
        "depth": 0,
        "queue_empty": True,
        "ts_epoch": NOW,
    }
    payload.update(overrides)
    (fleet_dir / f"{unit}.json").write_text(json.dumps(payload))
    return payload


_CLI_TIMEOUT = cli_timeout_from_env("DRAIN_CHECK_TEST_TIMEOUT")


def _run_cli(*args, env=None, timeout=_CLI_TIMEOUT):
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    return subprocess.run(
        ["python3", str(SCRIPT), *args],
        env=full_env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_run_cli_passes_resolved_timeout_to_subprocess_run(monkeypatch):
    captured = {}

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="idle\n", stderr="")

    monkeypatch.setattr(subprocess, "run", spy)
    _run_cli("--unit", UNIT)
    # No bound on the magnitude: the 60.0 default is pinned in
    # test_cli_subprocess_timeout.py, and a bound here would break
    # DRAIN_CHECK_TEST_TIMEOUT whenever it lowers the budget.
    assert captured["timeout"] == _CLI_TIMEOUT

    _run_cli("--unit", UNIT, timeout=3)
    assert captured["timeout"] == 3


def test_cli_prints_idle_for_fresh_merge_idle_heartbeat(tmp_path):
    _write_raw_heartbeat(tmp_path, UNIT, merge_idle=True, ts_epoch=NOW)

    result = _run_cli(
        "--unit", UNIT,
        "--fleet-dir", str(tmp_path),
        "--fresh-window", str(FRESH_WINDOW),
        "--now", str(NOW),
    )

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert result.stdout.strip() == "idle"


def test_cli_prints_busy_for_fresh_merge_idle_false_heartbeat(tmp_path):
    _write_raw_heartbeat(tmp_path, UNIT, merge_idle=False, ts_epoch=NOW)

    result = _run_cli(
        "--unit", UNIT,
        "--fleet-dir", str(tmp_path),
        "--fresh-window", str(FRESH_WINDOW),
        "--now", str(NOW),
    )

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert result.stdout.strip() == "busy"


def test_cli_prints_stale_for_old_heartbeat(tmp_path):
    _write_raw_heartbeat(tmp_path, UNIT, merge_idle=True, ts_epoch=NOW - FRESH_WINDOW - 1)

    result = _run_cli(
        "--unit", UNIT,
        "--fleet-dir", str(tmp_path),
        "--fresh-window", str(FRESH_WINDOW),
        "--now", str(NOW),
    )

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert result.stdout.strip() == "stale"


def test_cli_prints_absent_for_missing_heartbeat_file(tmp_path):
    result = _run_cli(
        "--unit", UNIT,
        "--fleet-dir", str(tmp_path),
        "--fresh-window", str(FRESH_WINDOW),
        "--now", str(NOW),
    )

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert result.stdout.strip() == "absent"


def test_cli_fresh_window_defaults_to_120(tmp_path):
    """Omitting --fresh-window falls back to 120 -- a 100s-old heartbeat
    (< 120) must still read idle."""
    _write_raw_heartbeat(tmp_path, UNIT, merge_idle=True, ts_epoch=NOW - 100)

    result = _run_cli(
        "--unit", UNIT,
        "--fleet-dir", str(tmp_path),
        "--now", str(NOW),
    )

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert result.stdout.strip() == "idle"


def test_cli_now_defaults_to_current_time_when_omitted(tmp_path):
    """Omitting --now falls back to the real current time -- a heartbeat
    written far in the past reads stale against a small fresh-window, and
    one written right now reads idle."""
    _write_raw_heartbeat(tmp_path, UNIT, merge_idle=True, ts_epoch=time.time() - 10_000)

    stale_result = _run_cli(
        "--unit", UNIT,
        "--fleet-dir", str(tmp_path),
        "--fresh-window", "5",
    )
    assert stale_result.returncode == 0, (
        f"stdout={stale_result.stdout!r} stderr={stale_result.stderr!r}"
    )
    assert stale_result.stdout.strip() == "stale"

    _write_raw_heartbeat(tmp_path, UNIT, merge_idle=True, ts_epoch=time.time())

    idle_result = _run_cli(
        "--unit", UNIT,
        "--fleet-dir", str(tmp_path),
        "--fresh-window", "60",
    )
    assert idle_result.returncode == 0, (
        f"stdout={idle_result.stdout!r} stderr={idle_result.stderr!r}"
    )
    assert idle_result.stdout.strip() == "idle"


def test_cli_fleet_dir_defaults_via_resolve_fleet_dir(tmp_path):
    """Omitting --fleet-dir falls back to resolve_fleet_dir() -- honouring
    ORCH_FLEET_DIR from the environment."""
    _write_raw_heartbeat(tmp_path, UNIT, merge_idle=True, ts_epoch=NOW)

    result = _run_cli(
        "--unit", UNIT,
        "--fresh-window", str(FRESH_WINDOW),
        "--now", str(NOW),
        env={"ORCH_FLEET_DIR": str(tmp_path)},
    )

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert result.stdout.strip() == "idle"


# ---------------------------------------------------------------------------
# task 5371: classify under a drain request -- verifying / overdue / refused.
# One test per precedence branch of the contract, in precedence order.
# ---------------------------------------------------------------------------

REQUESTED_TS = 1_000_000 - 60


def _drain_ack(**overrides):
    ack = {"requested_ts": REQUESTED_TS, "admission_halted": True, "refused": None}
    ack.update(overrides)
    return ack


def _verify(deadline_ts, **overrides):
    entry = {
        "task_id": "5371",
        "host": "local",
        "kind": "verify",
        "started_ts": NOW - 100,
        "deadline_ts": deadline_ts,
    }
    entry.update(overrides)
    return entry


def _new_producer_heartbeat(*, drain, verifies, **overrides):
    return _heartbeat(merge_idle=False, drain=drain, verifies_in_flight=verifies, **overrides)


def _classify_under_request(heartbeat):
    return classify(heartbeat, NOW, FRESH_WINDOW, drain_requested_ts=REQUESTED_TS)


def test_absent_and_stale_are_unchanged_under_a_drain_request():
    assert _classify_under_request(None) == "absent"
    assert _classify_under_request(_heartbeat(ts_epoch="x")) == "absent"
    stale = _new_producer_heartbeat(
        drain=_drain_ack(), verifies=[], ts_epoch=NOW - FRESH_WINDOW - 1,
    )
    assert _classify_under_request(stale) == "stale"


def test_a_legacy_producer_is_read_by_merge_idle_under_a_drain_request():
    """A unit still on pre-5371 code has no verifies_in_flight key and cannot
    see the request: it must classify exactly as without one, even if it
    carries a stray `drain` key."""
    assert _classify_under_request(_heartbeat(merge_idle=True)) == "idle"
    assert _classify_under_request(_heartbeat(merge_idle=False)) == "busy"
    assert _classify_under_request(
        _heartbeat(merge_idle=False, drain=_drain_ack()),
    ) == "busy"


def test_an_unacknowledged_request_is_busy():
    no_ack = _new_producer_heartbeat(drain=None, verifies=[])
    other_request = _new_producer_heartbeat(
        drain=_drain_ack(requested_ts=REQUESTED_TS - 1), verifies=[],
    )
    not_a_mapping = _new_producer_heartbeat(drain="halted", verifies=[])
    assert _classify_under_request(no_ack) == "busy"
    assert _classify_under_request(other_request) == "busy"
    assert _classify_under_request(not_a_mapping) == "busy"


def test_a_refusal_is_refused_even_with_nothing_in_flight():
    heartbeat = _new_producer_heartbeat(
        drain=_drain_ack(admission_halted=False, refused="sweep_dead"), verifies=[],
    )
    assert _classify_under_request(heartbeat) == "refused"


def test_a_refusal_wins_over_in_flight_verifies():
    heartbeat = _new_producer_heartbeat(
        drain=_drain_ack(admission_halted=False, refused="invocation_mismatch"),
        verifies=[_verify(NOW + 500)],
    )
    assert _classify_under_request(heartbeat) == "refused"


def test_admission_not_yet_halted_is_busy_not_idle():
    for halted in (False, None, "true", 1):
        heartbeat = _new_producer_heartbeat(
            drain=_drain_ack(admission_halted=halted), verifies=[],
        )
        assert _classify_under_request(heartbeat) == "busy", halted


def test_an_acknowledged_drain_with_nothing_in_flight_is_idle():
    heartbeat = _new_producer_heartbeat(drain=_drain_ack(), verifies=[])
    assert _classify_under_request(heartbeat) == "idle"


def test_a_malformed_verifies_list_is_busy():
    heartbeat = _new_producer_heartbeat(drain=_drain_ack(), verifies=None)
    assert _classify_under_request(heartbeat) == "busy"


def test_every_verify_past_its_deadline_is_overdue():
    heartbeat = _new_producer_heartbeat(
        drain=_drain_ack(), verifies=[_verify(NOW - 1), _verify(NOW)],
    )
    assert _classify_under_request(heartbeat) == "overdue"


def test_one_verify_inside_its_deadline_keeps_the_unit_verifying():
    heartbeat = _new_producer_heartbeat(
        drain=_drain_ack(), verifies=[_verify(NOW - 1), _verify(NOW + 1)],
    )
    assert _classify_under_request(heartbeat) == "verifying"


def test_a_non_numeric_deadline_counts_as_not_overdue():
    for deadline in (None, "soon", True):
        heartbeat = _new_producer_heartbeat(
            drain=_drain_ack(), verifies=[_verify(NOW - 1), _verify(deadline)],
        )
        assert _classify_under_request(heartbeat) == "verifying", deadline


def test_without_a_drain_request_the_new_keys_are_ignored():
    """The watchdog --report calls classify() with no request and must see
    today's four-token vocabulary whatever the heartbeat carries."""
    heartbeat = _new_producer_heartbeat(drain=_drain_ack(), verifies=[_verify(NOW + 9)])
    assert classify(heartbeat, NOW, FRESH_WINDOW) == "busy"
    drained = _heartbeat(merge_idle=True, drain=_drain_ack(), verifies_in_flight=[])
    assert classify(drained, NOW, FRESH_WINDOW) == "idle"


def test_cli_drain_requested_ts_enables_the_drain_verdicts(tmp_path):
    _write_raw_heartbeat(
        tmp_path, UNIT, merge_idle=False, drain=_drain_ack(),
        verifies_in_flight=[_verify(NOW + 500)],
    )
    common = ("--unit", UNIT, "--fleet-dir", str(tmp_path),
              "--fresh-window", str(FRESH_WINDOW), "--now", str(NOW))

    with_request = _run_cli(*common, "--drain-requested-ts", str(REQUESTED_TS))
    without_request = _run_cli(*common)

    assert with_request.returncode == 0, f"stderr={with_request.stderr!r}"
    assert with_request.stdout.strip() == "verifying"
    assert without_request.returncode == 0, f"stderr={without_request.stderr!r}"
    assert without_request.stdout.strip() == "busy"
