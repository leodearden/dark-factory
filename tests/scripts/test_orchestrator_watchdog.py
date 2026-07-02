"""Unit tests for scripts/orchestrator-watchdog.py.

The watchdog module has a hyphenated filename so it cannot be imported via
``import orchestrator_watchdog``.  We use importlib to load it by file path.

No live systemd runtime is needed — all subprocess.run calls are monkeypatched.
"""

import importlib.util
import pathlib
import re
import subprocess
import types

import pytest

REPO_ROOT = pathlib.Path(__file__).parents[2]
WATCHDOG_PATH = REPO_ROOT / "scripts" / "orchestrator-watchdog.py"


def _load_watchdog() -> types.ModuleType:
    """Load scripts/orchestrator-watchdog.py as a module (hyphenated name)."""
    spec = importlib.util.spec_from_file_location("orchestrator_watchdog", WATCHDOG_PATH)
    assert spec is not None, f"Could not build spec from {WATCHDOG_PATH}"
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# probe_port tests
# ---------------------------------------------------------------------------

# Verbatim `ss -ltn "sport = :<port>"` output from systemd 255 / iproute2-6.1.0.
# Note: NO leading Netid column in this version — fields are:
#   State  Recv-Q  Send-Q  Local Address:Port  Peer Address:Port
# (5 fields per data row, index 0-4)
_SS_HEADER = "State  Recv-Q Send-Q Local Address:Port Peer Address:Port\n"

_SS_LISTEN_8102 = (
    _SS_HEADER
    + "LISTEN 0      2048       127.0.0.1:8102      0.0.0.0:*\n"
)

_SS_LISTEN_8100 = (
    _SS_HEADER
    + "LISTEN 0      2048       127.0.0.1:8100      0.0.0.0:*\n"
)

# Decoy rows that must NOT trigger a false positive for port 8102
_SS_DECOYS = (
    _SS_HEADER
    + "LISTEN 0      2048       127.0.0.1:81020     0.0.0.0:*\n"
    + "LISTEN 0      2048       127.0.0.1:48102     0.0.0.0:*\n"
)


def test_probe_port_true_when_listening(monkeypatch: pytest.MonkeyPatch) -> None:
    """probe_port returns True when ss reports a LISTEN row for the exact port."""
    wdog = _load_watchdog()

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        return subprocess.CompletedProcess(cmd, 0, stdout=_SS_LISTEN_8102, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert wdog.probe_port(8102) is True


def test_probe_port_false_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """probe_port returns False when ss output has no LISTEN row for the port."""
    wdog = _load_watchdog()

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        return subprocess.CompletedProcess(cmd, 0, stdout=_SS_HEADER, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert wdog.probe_port(8102) is False


def test_probe_port_not_fooled_by_substring(monkeypatch: pytest.MonkeyPatch) -> None:
    """probe_port must not match :81020 or :48102 when probing for :8102."""
    wdog = _load_watchdog()

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        return subprocess.CompletedProcess(cmd, 0, stdout=_SS_DECOYS, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert wdog.probe_port(8102) is False


def test_probe_port_returns_true_on_ss_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """probe_port must return True (not False) when ss exits non-zero.

    A tooling failure (missing binary, permission error, syntax difference across
    iproute2 versions) must not be misinterpreted as 'port down', which would
    trigger a spurious stop→start cycle on a healthy orchestrator every 60s.
    """
    wdog = _load_watchdog()
    log_messages: list[str] = []

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        if cmd[0] == "ss":
            # Simulate ss exiting non-zero (e.g. binary missing / permission denied)
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="Operation not permitted")
        # systemd-cat call from log() — succeed silently
        log_messages.append(str(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = wdog.probe_port(8102)

    assert result is True, (
        "probe_port must return True (not False) when ss exits non-zero "
        "so a tooling failure does not trigger a spurious restart"
    )
    assert len(log_messages) >= 1, "probe_port must log a diagnostic when ss fails"


def test_probe_port_returns_true_on_missing_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    """probe_port must return True when ss binary is not found (FileNotFoundError).

    If ss is absent (e.g. minimal container image), a FileNotFoundError must not
    propagate — the safe default is True so the unit is not spuriously restarted.
    """
    wdog = _load_watchdog()
    log_messages: list[str] = []

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        if cmd[0] == "ss":
            raise FileNotFoundError(2, "No such file or directory", "ss")
        # systemd-cat call from log()
        log_messages.append(str(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = wdog.probe_port(8102)

    assert result is True, (
        "probe_port must return True when ss binary is missing "
        "so a missing tool does not trigger a spurious restart"
    )
    assert len(log_messages) >= 1, "probe_port must log a diagnostic when ss is not found"


def test_probe_port_returns_true_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """probe_port must return True when the ss call exceeds its timeout.

    A slow probe (subprocess.TimeoutExpired) must be treated as a tooling failure
    and must not be misinterpreted as 'port down'.
    """
    wdog = _load_watchdog()
    log_messages: list[str] = []

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        if cmd[0] == "ss":
            raise subprocess.TimeoutExpired(cmd, 5)
        # systemd-cat call from log()
        log_messages.append(str(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = wdog.probe_port(8102)

    assert result is True, (
        "probe_port must return True when ss times out "
        "so a slow probe does not trigger a spurious restart"
    )
    assert len(log_messages) >= 1, "probe_port must log a diagnostic when ss times out"


# ---------------------------------------------------------------------------
# restart_unit tests
# ---------------------------------------------------------------------------


def test_restart_unit_stop_reset_failed_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """restart_unit must call stop, reset-failed, then start — in that order.

    The three-phase sequence ensures:
    - stop: give the unit a grace period (TimeoutStopSec=30 escalates SIGTERM→SIGKILL)
    - reset-failed: clear StartLimit state so the start is not a silent no-op
    - start: re-launch the unit
    """
    wdog = _load_watchdog()
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    wdog.restart_unit("orchestrator-dark-factory.service")

    assert len(calls) == 3, f"Expected exactly 3 systemctl calls, got {len(calls)}: {calls}"
    assert calls[0] == ["systemctl", "--user", "stop", "orchestrator-dark-factory.service"]
    assert calls[1] == ["systemctl", "--user", "reset-failed", "orchestrator-dark-factory.service"]
    assert calls[2] == ["systemctl", "--user", "start", "orchestrator-dark-factory.service"]


def test_restart_unit_never_uses_kill(monkeypatch: pytest.MonkeyPatch) -> None:
    """restart_unit must not invoke systemctl kill or any -KILL/-9 signal."""
    wdog = _load_watchdog()
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    wdog.restart_unit("orchestrator-dark-factory.service")

    for argv in calls:
        for token in argv:
            assert "kill" not in token.lower(), (
                f"Forbidden 'kill' token found in argv {argv}"
            )
            assert token not in {"-9", "-KILL", "SIGKILL"}, (
                f"Forbidden signal token {token!r} found in argv {argv}"
            )


def test_restart_unit_handles_stop_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """restart_unit must not raise if systemctl stop times out.

    After a stop timeout the reset-failed and start calls must still execute
    so the unit is not left in a permanently-down state.
    """
    wdog = _load_watchdog()
    calls: list[list[str]] = []
    log_messages: list[str] = []

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        calls.append(list(cmd))
        if cmd[:3] == ["systemctl", "--user", "stop"]:
            raise subprocess.TimeoutExpired(cmd, 45)
        if cmd[0] == "systemd-cat":
            log_messages.append(" ".join(cmd))
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    # Must not raise
    wdog.restart_unit("orchestrator-dark-factory.service")

    systemctl_cmds = [c for c in calls if c[0] == "systemctl"]
    verbs = [c[2] for c in systemctl_cmds]
    assert "reset-failed" in verbs, "reset-failed must be called after stop timeout"
    assert "start" in verbs, "start must be called even after stop timeout"
    assert len(log_messages) >= 1, "timeout must be logged via log()"


def test_restart_unit_handles_start_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """restart_unit must not raise if systemctl start times out."""
    wdog = _load_watchdog()
    log_messages: list[str] = []

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        if cmd[:3] == ["systemctl", "--user", "start"]:
            raise subprocess.TimeoutExpired(cmd, 45)
        if cmd[0] == "systemd-cat":
            log_messages.append(" ".join(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    # Must not raise
    wdog.restart_unit("orchestrator-dark-factory.service")

    assert len(log_messages) >= 1, "start timeout must be logged via log()"


def test_restart_unit_handles_reset_failed_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """restart_unit must not raise if systemctl reset-failed times out.

    After a reset-failed timeout the start call must still execute so the
    unit is not left in a permanently-down state (docstring: "remaining
    phases still execute").
    """
    wdog = _load_watchdog()
    calls: list[list[str]] = []
    log_messages: list[str] = []

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        calls.append(list(cmd))
        if cmd[:3] == ["systemctl", "--user", "reset-failed"]:
            raise subprocess.TimeoutExpired(cmd, 10)
        if cmd[0] == "systemd-cat":
            log_messages.append(" ".join(cmd))
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    # Must not raise
    wdog.restart_unit("orchestrator-dark-factory.service")

    systemctl_cmds = [c for c in calls if c[0] == "systemctl"]
    verbs = [c[2] for c in systemctl_cmds]
    assert "start" in verbs, "start must be called even after reset-failed timeout"
    assert len(log_messages) >= 1, "reset-failed timeout must be logged via log()"


# ---------------------------------------------------------------------------
# _unit_start_elapsed_secs direct tests
# ---------------------------------------------------------------------------


def _make_systemctl_show_result(value: str, rc: int = 0) -> subprocess.CompletedProcess:
    """Build a fake systemctl show CompletedProcess with the given property value."""
    stdout = f"ExecMainStartTimestampMonotonic={value}\n"
    return subprocess.CompletedProcess(["systemctl"], rc, stdout=stdout, stderr="")


def test_elapsed_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """_unit_start_elapsed_secs returns correct elapsed seconds on the happy path.

    start=5_000_000 us (= 5.0 s), clock_gettime→305.0 s → elapsed = 300.0 s.
    """
    wdog = _load_watchdog()

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        return _make_systemctl_show_result("5000000", rc=0)

    def fake_clock_gettime(clk_id):  # noqa: ANN001
        assert clk_id == wdog.time.CLOCK_MONOTONIC
        return 305.0

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(wdog.time, "clock_gettime", fake_clock_gettime)

    result = wdog._unit_start_elapsed_secs("some.service")
    assert result == pytest.approx(300.0)


def test_elapsed_nonzero_rc_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """_unit_start_elapsed_secs returns None when systemctl exits non-zero."""
    wdog = _load_watchdog()

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        return _make_systemctl_show_result("5000000", rc=1)

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert wdog._unit_start_elapsed_secs("some.service") is None


def test_elapsed_unparseable_value_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """_unit_start_elapsed_secs returns None when the property value is not an int."""
    wdog = _load_watchdog()

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        return _make_systemctl_show_result("notanint", rc=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert wdog._unit_start_elapsed_secs("some.service") is None


def test_elapsed_zero_sentinel_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """_unit_start_elapsed_secs returns None for the zero sentinel (unit never started)."""
    wdog = _load_watchdog()

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        return _make_systemctl_show_result("0", rc=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert wdog._unit_start_elapsed_secs("some.service") is None


def test_elapsed_clock_failure_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """_unit_start_elapsed_secs returns None when clock_gettime raises OSError."""
    wdog = _load_watchdog()

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        return _make_systemctl_show_result("5000000", rc=0)

    def fake_clock_gettime(clk_id):  # noqa: ANN001
        raise OSError("clock unavailable")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(wdog.time, "clock_gettime", fake_clock_gettime)
    assert wdog._unit_start_elapsed_secs("some.service") is None


def test_elapsed_clamp_future_start_returns_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """_unit_start_elapsed_secs clamps to 0.0 when start_us/1e6 > now (max(0,…))."""
    wdog = _load_watchdog()

    # start is in the future relative to now (clock drift or negative elapsed)
    start_us = int(310.0 * 1_000_000)  # 310 s

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        return _make_systemctl_show_result(str(start_us), rc=0)

    def fake_clock_gettime(clk_id):  # noqa: ANN001
        return 305.0  # now is BEFORE start → elapsed would be negative

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(wdog.time, "clock_gettime", fake_clock_gettime)

    result = wdog._unit_start_elapsed_secs("some.service")
    assert result == pytest.approx(0.0)


def test_elapsed_empty_stdout_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """_unit_start_elapsed_secs returns None when stdout has no '=' line."""
    wdog = _load_watchdog()

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        return subprocess.CompletedProcess(cmd, 0, stdout="\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert wdog._unit_start_elapsed_secs("some.service") is None


def test_elapsed_systemctl_timeout_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """_unit_start_elapsed_secs returns None when the systemctl call times out.

    subprocess.TimeoutExpired from the timeout=5 call is the most likely real-world
    failure mode (systemctl hung).  The outer except Exception guard must convert it
    to None so callers treat the grace window as not applicable and proceed to probe.
    """
    wdog = _load_watchdog()

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        raise subprocess.TimeoutExpired(cmd, 5)

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert wdog._unit_start_elapsed_secs("some.service") is None


def test_elapsed_systemctl_not_found_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """_unit_start_elapsed_secs returns None when systemctl binary is not found.

    FileNotFoundError (systemctl absent) must be absorbed by the outer except
    Exception guard and converted to None so callers proceed to probe normally.
    """
    wdog = _load_watchdog()

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        raise FileNotFoundError(2, "No such file or directory", "systemctl")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert wdog._unit_start_elapsed_secs("some.service") is None


# ---------------------------------------------------------------------------
# Config-drift guard test
# ---------------------------------------------------------------------------


def _extract_escalation_port(cfg: object, config_path: pathlib.Path) -> int:
    """Extract escalation.port from a parsed YAML config with an informative diagnostic.

    Uses .get() chaining (no raw [] indexing) so a schema rename surfaces as an
    AssertionError naming the offending config file path, not a bare KeyError.
    """
    escalation = cfg.get("escalation") if isinstance(cfg, dict) else None
    port = escalation.get("port") if isinstance(escalation, dict) else None
    assert port is not None, (
        f"{config_path}: missing 'escalation.port' (schema may have changed)"
    )
    return port


def test_extract_escalation_port_error_paths() -> None:
    """_extract_escalation_port raises AssertionError naming the config file on bad schema.

    The key behavioral contract: the error message includes the config_path so a schema
    rename (e.g. 'escalation' -> 'escalation_mcp') surfaces as a named-file diagnostic
    instead of a bare KeyError. Verified by matching only on re.escape(str(config_path)).
    """
    config_path = pathlib.Path("/fake/orchestrator/config.yaml")
    path_pattern = re.escape(str(config_path))

    # Missing 'escalation' key at top level
    with pytest.raises(AssertionError, match=path_pattern):
        _extract_escalation_port({"other_key": 5}, config_path)

    # 'escalation' present but missing 'port'
    with pytest.raises(AssertionError, match=path_pattern):
        _extract_escalation_port({"escalation": {"queue_dir": "data/escalations"}}, config_path)

    # Non-dict cfg (e.g. YAML parsed to None or a list)
    with pytest.raises(AssertionError, match=path_pattern):
        _extract_escalation_port(None, config_path)


def test_watched_ports_match_configured_escalation_ports() -> None:
    """WATCHED ports must equal the escalation.port values in each orchestrator's config.

    This is a behavioral drift guard: it passes when the ports are aligned and
    fails if WATCHED or either config file drifts to a different port number.
    The reify config is skipped gracefully if it is not reachable (e.g. CI).
    """
    yaml = pytest.importorskip("yaml")
    wdog = _load_watchdog()

    # Build a unit→port map from WATCHED for convenient lookup
    unit_to_port = {unit: port for port, unit in wdog.WATCHED}

    # --- dark-factory orchestrator ---
    df_config_path = REPO_ROOT / "orchestrator" / "config.yaml"
    df_cfg = yaml.safe_load(df_config_path.read_text())
    df_port = _extract_escalation_port(df_cfg, df_config_path)
    assert unit_to_port["orchestrator-dark-factory.service"] == df_port, (
        f"WATCHED port for orchestrator-dark-factory.service "
        f"({unit_to_port['orchestrator-dark-factory.service']}) != "
        f"orchestrator/config.yaml escalation.port ({df_port})"
    )

    # --- my-solar-challenge orchestrator (check if present) ---
    my_solar_config_path = pathlib.Path("/home/leo/src/my-solar-challenge/orchestrator.yaml")
    if my_solar_config_path.exists():
        my_solar_cfg = yaml.safe_load(my_solar_config_path.read_text())
        my_solar_port = _extract_escalation_port(my_solar_cfg, my_solar_config_path)
        assert unit_to_port["orchestrator-my-solar-challenge.service"] == my_solar_port, (
            f"WATCHED port for orchestrator-my-solar-challenge.service "
            f"({unit_to_port['orchestrator-my-solar-challenge.service']}) != "
            f"my-solar-challenge/orchestrator.yaml escalation.port ({my_solar_port})"
        )

    # --- reify orchestrator (skip if absent in this environment) ---
    reify_config_path = pathlib.Path("/home/leo/src/reify/orchestrator.yaml")
    if not reify_config_path.exists():
        pytest.skip("reify orchestrator.yaml not reachable in this environment")
    reify_cfg = yaml.safe_load(reify_config_path.read_text())
    reify_port = _extract_escalation_port(reify_cfg, reify_config_path)
    assert unit_to_port["orchestrator-reify.service"] == reify_port, (
        f"WATCHED port for orchestrator-reify.service "
        f"({unit_to_port['orchestrator-reify.service']}) != "
        f"reify/orchestrator.yaml escalation.port ({reify_port})"
    )


# ---------------------------------------------------------------------------
# main() tests
# ---------------------------------------------------------------------------


def test_main_targets_expected_pairs() -> None:
    """WATCHED must list the orchestrator (port, unit) pairs in order."""
    wdog = _load_watchdog()
    assert hasattr(wdog, "WATCHED"), "Module must expose a WATCHED constant"
    assert wdog.WATCHED == [
        (8102, "orchestrator-dark-factory.service"),
        (8100, "orchestrator-reify.service"),
        (8106, "orchestrator-my-solar-challenge.service"),
    ]


def test_main_restarts_only_failed_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """main() restarts only the unit whose probe returned False."""
    wdog = _load_watchdog()
    restarted: list[str] = []

    def fake_elapsed(unit: str) -> None:
        return None  # no grace window — proceed to probe

    def fake_probe(port: int) -> bool:
        # 8102 (df) fails, 8100 (reify) is fine
        return port != 8102

    def fake_restart(unit: str) -> None:
        restarted.append(unit)

    def fake_log(msg: str) -> None:
        pass

    monkeypatch.setattr(wdog, "_unit_start_elapsed_secs", fake_elapsed)
    monkeypatch.setattr(wdog, "probe_port", fake_probe)
    monkeypatch.setattr(wdog, "restart_unit", fake_restart)
    monkeypatch.setattr(wdog, "is_unit_enabled", lambda _u: True)
    monkeypatch.setattr(wdog, "log", fake_log)

    wdog.main()

    assert restarted == ["orchestrator-dark-factory.service"], (
        f"Expected only df unit restarted, got: {restarted}"
    )


def test_main_logs_each_action(monkeypatch: pytest.MonkeyPatch) -> None:
    """main() must call log() when restarting a unit."""
    wdog = _load_watchdog()
    log_messages: list[str] = []

    def fake_elapsed(unit: str) -> None:
        return None

    def fake_probe(port: int) -> bool:
        return port != 8102  # df probe fails

    def fake_restart(unit: str) -> None:
        pass

    def fake_log(msg: str) -> None:
        log_messages.append(msg)

    monkeypatch.setattr(wdog, "_unit_start_elapsed_secs", fake_elapsed)
    monkeypatch.setattr(wdog, "probe_port", fake_probe)
    monkeypatch.setattr(wdog, "restart_unit", fake_restart)
    monkeypatch.setattr(wdog, "is_unit_enabled", lambda _u: True)
    monkeypatch.setattr(wdog, "log", fake_log)

    wdog.main()

    assert len(log_messages) >= 1, "Expected at least one log() call for the restarted unit"
    # At least one message must mention the unit being restarted
    assert any("orchestrator-dark-factory.service" in m for m in log_messages), (
        f"No log message mentions orchestrator-dark-factory.service: {log_messages}"
    )


def test_main_isolates_per_unit_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """main() must not propagate an exception from one unit to the next.

    If probe_port raises for the first unit (8102), the second unit (8100)
    must still be processed.
    """
    wdog = _load_watchdog()
    restarted: list[str] = []

    def fake_elapsed(unit: str) -> None:
        return None

    def fake_probe(port: int) -> bool:
        if port == 8102:
            raise RuntimeError("ss exploded")
        return False  # reify probe also fails → should trigger restart

    def fake_restart(unit: str) -> None:
        restarted.append(unit)

    def fake_log(msg: str) -> None:
        pass

    monkeypatch.setattr(wdog, "_unit_start_elapsed_secs", fake_elapsed)
    monkeypatch.setattr(wdog, "probe_port", fake_probe)
    monkeypatch.setattr(wdog, "restart_unit", fake_restart)
    monkeypatch.setattr(wdog, "is_unit_enabled", lambda _u: True)
    monkeypatch.setattr(wdog, "log", fake_log)

    # Must not raise
    wdog.main()

    assert "orchestrator-reify.service" in restarted, (
        "reify unit must still be processed even if df probe raised"
    )


def test_main_skips_probe_in_grace_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """main() must not call probe_port for a unit within STARTUP_GRACE_SECS.

    If an orchestrator started only 30s ago (less than the 120s grace window),
    probe_port must not be called: the port may not yet be bound and a false
    negative would cause a stop→start before the process finishes initializing.
    """
    wdog = _load_watchdog()
    probed: list[int] = []

    def fake_elapsed(unit: str) -> float:
        return 30.0  # started 30s ago, inside grace window

    def fake_probe(port: int) -> bool:
        probed.append(port)
        return True

    def fake_log(msg: str) -> None:
        pass

    monkeypatch.setattr(wdog, "_unit_start_elapsed_secs", fake_elapsed)
    monkeypatch.setattr(wdog, "probe_port", fake_probe)
    monkeypatch.setattr(wdog, "is_unit_enabled", lambda _u: True)
    monkeypatch.setattr(wdog, "log", fake_log)

    wdog.main()

    assert probed == [], (
        f"probe_port must not be called inside the grace window; was called for ports: {probed}"
    )


def test_main_probes_after_grace_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """main() must call probe_port when a unit started beyond STARTUP_GRACE_SECS."""
    wdog = _load_watchdog()
    probed: list[int] = []
    restarted: list[str] = []

    def fake_elapsed(unit: str) -> float:
        return 300.0  # started 300s ago, well past grace window

    def fake_probe(port: int) -> bool:
        probed.append(port)
        return True  # listening — no restart needed

    def fake_restart(unit: str) -> None:
        restarted.append(unit)

    def fake_log(msg: str) -> None:
        pass

    monkeypatch.setattr(wdog, "_unit_start_elapsed_secs", fake_elapsed)
    monkeypatch.setattr(wdog, "probe_port", fake_probe)
    monkeypatch.setattr(wdog, "restart_unit", fake_restart)
    monkeypatch.setattr(wdog, "is_unit_enabled", lambda _u: True)
    monkeypatch.setattr(wdog, "log", fake_log)

    wdog.main()

    assert len(probed) == len(wdog.WATCHED), (
        f"All watched ports must be probed when outside grace window; probed: {probed}"
    )
    assert restarted == [], "No restart expected when ports are listening"


def test_main_grace_window_skipped_when_elapsed_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When _unit_start_elapsed_secs returns None, main() must still probe.

    None indicates the elapsed time could not be determined (e.g. unit never
    started, systemctl unavailable).  The safe default is to probe normally.
    """
    wdog = _load_watchdog()
    probed: list[int] = []

    def fake_elapsed(unit: str) -> None:
        return None  # unknown — treat as no grace window

    def fake_probe(port: int) -> bool:
        probed.append(port)
        return True

    def fake_log(msg: str) -> None:
        pass

    monkeypatch.setattr(wdog, "_unit_start_elapsed_secs", fake_elapsed)
    monkeypatch.setattr(wdog, "probe_port", fake_probe)
    monkeypatch.setattr(wdog, "is_unit_enabled", lambda _u: True)
    monkeypatch.setattr(wdog, "log", fake_log)

    wdog.main()

    assert len(probed) == len(wdog.WATCHED), (
        f"When elapsed is None, all watched ports must be probed; probed: {probed}"
    )


# ---------------------------------------------------------------------------
# is_unit_enabled() tests
# ---------------------------------------------------------------------------


def test_is_unit_enabled_true_on_zero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    """is_unit_enabled returns True when ``systemctl is-enabled`` exits 0."""
    wdog = _load_watchdog()

    class _R:
        returncode = 0

    def fake_run(*_a, **_kw):
        return _R()

    monkeypatch.setattr(wdog.subprocess, "run", fake_run)
    assert wdog.is_unit_enabled("orchestrator-reify.service") is True


def test_is_unit_enabled_false_on_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    """is_unit_enabled returns False when ``systemctl is-enabled`` exits non-zero.

    Non-zero is what ``disabled``/``masked``/unknown all return — the watchdog
    must respect the disabled state and skip the unit.
    """
    wdog = _load_watchdog()

    class _R:
        returncode = 1

    def fake_run(*_a, **_kw):
        return _R()

    monkeypatch.setattr(wdog.subprocess, "run", fake_run)
    assert wdog.is_unit_enabled("orchestrator-reify.service") is False


def test_is_unit_enabled_false_on_missing_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    """is_unit_enabled returns False (skip) if systemctl isn't on PATH."""
    wdog = _load_watchdog()
    logged: list[str] = []

    def fake_run(*_a, **_kw):
        raise FileNotFoundError("systemctl")

    monkeypatch.setattr(wdog.subprocess, "run", fake_run)
    monkeypatch.setattr(wdog, "log", lambda m: logged.append(m))
    assert wdog.is_unit_enabled("orchestrator-reify.service") is False
    assert any("FileNotFoundError" in m for m in logged), (
        "missing-systemctl path must emit a diagnostic"
    )


def test_is_unit_enabled_false_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """is_unit_enabled returns False (skip) if the systemctl call times out."""
    wdog = _load_watchdog()
    logged: list[str] = []

    def fake_run(*_a, **_kw):
        raise subprocess.TimeoutExpired(cmd="systemctl", timeout=5)

    monkeypatch.setattr(wdog.subprocess, "run", fake_run)
    monkeypatch.setattr(wdog, "log", lambda m: logged.append(m))
    assert wdog.is_unit_enabled("orchestrator-reify.service") is False
    assert any("TimeoutExpired" in m for m in logged), (
        "timeout path must emit a diagnostic"
    )


# ---------------------------------------------------------------------------
# main() must skip disabled units entirely (no probe, no restart)
# ---------------------------------------------------------------------------


def test_main_skips_disabled_unit_entirely(monkeypatch: pytest.MonkeyPatch) -> None:
    """main() must NOT probe or restart a unit that is_unit_enabled reports False.

    Disabling is explicit operator intent (e.g. a staged-but-not-yet-active
    deployment). The watchdog must respect it — no probe, no restart, no logs.
    """
    wdog = _load_watchdog()
    probed: list[int] = []
    restarted: list[str] = []

    def fake_enabled(unit: str) -> bool:
        # Only orchestrator-reify is enabled; df is disabled and must be skipped.
        return unit == "orchestrator-reify.service"

    def fake_elapsed(unit: str) -> None:
        return None

    def fake_probe(port: int) -> bool:
        probed.append(port)
        return False  # would normally trigger a restart

    def fake_restart(unit: str) -> None:
        restarted.append(unit)

    monkeypatch.setattr(wdog, "is_unit_enabled", fake_enabled)
    monkeypatch.setattr(wdog, "_unit_start_elapsed_secs", fake_elapsed)
    monkeypatch.setattr(wdog, "probe_port", fake_probe)
    monkeypatch.setattr(wdog, "restart_unit", fake_restart)
    monkeypatch.setattr(wdog, "log", lambda _m: None)

    wdog.main()

    # df is disabled → skipped (port 8102 never probed)
    assert 8102 not in probed, (
        f"Disabled unit's port (8102) must not be probed; probed: {probed}"
    )
    assert "orchestrator-dark-factory.service" not in restarted, (
        "Disabled unit must not be restarted"
    )
    # reify is enabled → probed, and the failed probe triggers a restart
    assert probed == [8100]
    assert restarted == ["orchestrator-reify.service"]


# ---------------------------------------------------------------------------
# _enumerate_running_units tests
# ---------------------------------------------------------------------------

_LIST_UNITS_SIX = (
    "orchestrator-dark-factory.service             loaded active running Dark Factory Orchestrator\n"
    "orchestrator-reify.service                    loaded active running Reify Orchestrator\n"
    "orchestrator-my-solar-challenge.service        loaded active running My Solar Challenge Orchestrator\n"
    "orchestrator-know-live.service                 loaded active running Know Live Orchestrator\n"
    "orchestrator-autopilot-video.service           loaded active running Autopilot Video Orchestrator\n"
    "orchestrator-solar-challenge-platform.service  loaded active running Solar Challenge Platform Orchestrator\n"
)


def test_enumerate_running_units_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """_enumerate_running_units returns the first field of each list-units line, in order."""
    wdog = _load_watchdog()
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout=_LIST_UNITS_SIX, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = wdog._enumerate_running_units()

    assert result == [
        "orchestrator-dark-factory.service",
        "orchestrator-reify.service",
        "orchestrator-my-solar-challenge.service",
        "orchestrator-know-live.service",
        "orchestrator-autopilot-video.service",
        "orchestrator-solar-challenge-platform.service",
    ]
    assert len(calls) == 1, f"Expected exactly one systemctl call, got {calls}"
    assert calls[0] == [
        "systemctl",
        "--user",
        "list-units",
        "orchestrator-*.service",
        "--state=running",
        "--no-legend",
        "--plain",
    ]


def test_enumerate_running_units_empty_on_nonzero_rc(monkeypatch: pytest.MonkeyPatch) -> None:
    """_enumerate_running_units returns [] when systemctl exits non-zero."""
    wdog = _load_watchdog()

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        return subprocess.CompletedProcess(cmd, 1, stdout=_LIST_UNITS_SIX, stderr="error")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert wdog._enumerate_running_units() == []


def test_enumerate_running_units_empty_on_missing_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    """_enumerate_running_units returns [] when systemctl binary is not found."""
    wdog = _load_watchdog()

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        raise FileNotFoundError(2, "No such file or directory", "systemctl")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert wdog._enumerate_running_units() == []


def test_enumerate_running_units_empty_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """_enumerate_running_units returns [] when the systemctl call times out."""
    wdog = _load_watchdog()

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        raise subprocess.TimeoutExpired(cmd, 5)

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert wdog._enumerate_running_units() == []


def test_enumerate_running_units_empty_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    """_enumerate_running_units returns [] when no orchestrator units are running."""
    wdog = _load_watchdog()

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert wdog._enumerate_running_units() == []


def test_enumerate_running_units_excludes_self(monkeypatch: pytest.MonkeyPatch) -> None:
    """_enumerate_running_units must never return the watchdog's own unit.

    orchestrator-watchdog.service matches the `orchestrator-*.service` glob
    just like every other unit. Today it happens to be absent from
    --state=running output because it is a Type=oneshot unit whose SUB state
    while executing is 'start', not 'running' — a fragile invariant this
    test does not rely on: it injects a `list-units` line for the watchdog's
    own unit directly (as if that invariant no longer held, e.g. after a
    Type or RemainAfterExit change) and asserts the explicit exclusion still
    filters it out.
    """
    wdog = _load_watchdog()
    stdout = (
        _LIST_UNITS_SIX
        + "orchestrator-watchdog.service                  loaded active running Watchdog\n"
    )

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = wdog._enumerate_running_units()

    assert wdog.WATCHDOG_UNIT_NAME not in result, (
        f"_enumerate_running_units must exclude its own unit even if enumerated; got {result}"
    )
    assert result == [
        "orchestrator-dark-factory.service",
        "orchestrator-reify.service",
        "orchestrator-my-solar-challenge.service",
        "orchestrator-know-live.service",
        "orchestrator-autopilot-video.service",
        "orchestrator-solar-challenge-platform.service",
    ], f"Non-self units must be preserved in order; got {result}"


# ---------------------------------------------------------------------------
# _unit_start_epoch tests
# ---------------------------------------------------------------------------


def _make_start_epoch_result(value: str, rc: int = 0) -> subprocess.CompletedProcess:
    """Build a fake `systemctl show --timestamp=unix -p ExecMainStartTimestamp` result."""
    stdout = f"ExecMainStartTimestamp={value}\n"
    return subprocess.CompletedProcess(["systemctl"], rc, stdout=stdout, stderr="")


def test_unit_start_epoch_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """_unit_start_epoch parses the '@<epoch>' realtime value to an int."""
    wdog = _load_watchdog()
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        calls.append(list(cmd))
        return _make_start_epoch_result("@1782996274")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = wdog._unit_start_epoch("orchestrator-dark-factory.service")

    assert result == 1782996274
    assert isinstance(result, int)
    assert len(calls) == 1, f"Expected exactly one systemctl call, got {calls}"
    argv = calls[0]
    assert "--timestamp=unix" in argv
    assert "--property=ExecMainStartTimestamp" in argv or (
        "-p" in argv and "ExecMainStartTimestamp" in argv
    ), f"argv must request ExecMainStartTimestamp: {argv}"


def test_unit_start_epoch_nonzero_rc_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """_unit_start_epoch returns None when systemctl exits non-zero."""
    wdog = _load_watchdog()

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        return _make_start_epoch_result("@1782996274", rc=1)

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert wdog._unit_start_epoch("some.service") is None


def test_unit_start_epoch_empty_value_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """_unit_start_epoch returns None when the property value is empty."""
    wdog = _load_watchdog()

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        return _make_start_epoch_result("")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert wdog._unit_start_epoch("some.service") is None


def test_unit_start_epoch_zero_sentinel_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """_unit_start_epoch returns None for the '@0' sentinel (unit never started)."""
    wdog = _load_watchdog()

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        return _make_start_epoch_result("@0")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert wdog._unit_start_epoch("some.service") is None


def test_unit_start_epoch_unparseable_value_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """_unit_start_epoch returns None when the value is not an int."""
    wdog = _load_watchdog()

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        return _make_start_epoch_result("@notanint")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert wdog._unit_start_epoch("some.service") is None


def test_unit_start_epoch_no_equals_line_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """_unit_start_epoch returns None when stdout has no '=' line."""
    wdog = _load_watchdog()

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        return subprocess.CompletedProcess(cmd, 0, stdout="\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert wdog._unit_start_epoch("some.service") is None


def test_unit_start_epoch_missing_binary_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """_unit_start_epoch returns None when systemctl binary is not found."""
    wdog = _load_watchdog()

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        raise FileNotFoundError(2, "No such file or directory", "systemctl")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert wdog._unit_start_epoch("some.service") is None


def test_unit_start_epoch_timeout_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """_unit_start_epoch returns None when the systemctl call times out."""
    wdog = _load_watchdog()

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        raise subprocess.TimeoutExpired(cmd, 5)

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert wdog._unit_start_epoch("some.service") is None


# ---------------------------------------------------------------------------
# _newest_watched_commit_epoch tests
# ---------------------------------------------------------------------------

_EXPECTED_REPO_DIR = "/home/leo/src/dark-factory"
_EXPECTED_WATCHED_PATHS = [
    "orchestrator/src/",
    "escalation/src/",
    "orchestrator/pyproject.toml",
    "orchestrator/uv.lock",
    "escalation/pyproject.toml",
]


def test_newest_watched_commit_epoch_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """_newest_watched_commit_epoch parses git's %ct output to an int."""
    wdog = _load_watchdog()
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="1783013906\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = wdog._newest_watched_commit_epoch()

    assert result == 1783013906
    assert isinstance(result, int)
    assert len(calls) == 1, f"Expected exactly one git call, got {calls}"
    argv = calls[0]
    assert argv[0] == "git"
    assert argv[1] == "-C"
    assert argv[2] == _EXPECTED_REPO_DIR
    assert argv[3:7] == ["log", "-1", "--format=%ct", "HEAD"]
    assert "--" in argv, f"argv must separate revision from pathspec with '--': {argv}"
    watched_args = argv[argv.index("--") + 1 :]
    for path in _EXPECTED_WATCHED_PATHS:
        assert path in watched_args, f"Expected watched path {path!r} in argv {argv}"


def test_newest_watched_commit_epoch_empty_stdout_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_newest_watched_commit_epoch returns None on empty stdout (rc 0).

    Confirmed real behavior: `git log -1 --format=%ct HEAD -- <paths>` exits 0
    with empty stdout when no commit touches the given paths. This must be
    treated as undeterminable, not epoch 0 (which would make everything look
    infinitely stale).
    """
    wdog = _load_watchdog()

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert wdog._newest_watched_commit_epoch() is None


def test_newest_watched_commit_epoch_nonzero_rc_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_newest_watched_commit_epoch returns None when git exits non-zero."""
    wdog = _load_watchdog()

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        return subprocess.CompletedProcess(
            cmd, 128, stdout="", stderr="fatal: not a git repository"
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert wdog._newest_watched_commit_epoch() is None


def test_newest_watched_commit_epoch_unparseable_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_newest_watched_commit_epoch returns None when stdout is not an int."""
    wdog = _load_watchdog()

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        return subprocess.CompletedProcess(cmd, 0, stdout="not-an-epoch\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert wdog._newest_watched_commit_epoch() is None


def test_newest_watched_commit_epoch_missing_binary_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_newest_watched_commit_epoch returns None when git binary is not found."""
    wdog = _load_watchdog()

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        raise FileNotFoundError(2, "No such file or directory", "git")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert wdog._newest_watched_commit_epoch() is None


def test_newest_watched_commit_epoch_timeout_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_newest_watched_commit_epoch returns None when the git call times out."""
    wdog = _load_watchdog()

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        raise subprocess.TimeoutExpired(cmd, 5)

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert wdog._newest_watched_commit_epoch() is None


# ---------------------------------------------------------------------------
# STALENESS_GRACE_SECS tests
# ---------------------------------------------------------------------------


def test_staleness_grace_secs_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """STALENESS_GRACE_SECS defaults to 1800 (30 min) with no env override."""
    monkeypatch.delenv("STALENESS_GRACE_SECS", raising=False)
    wdog = _load_watchdog()
    assert wdog.STALENESS_GRACE_SECS == 1800


def test_staleness_grace_secs_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """STALENESS_GRACE_SECS honors a valid STALENESS_GRACE_SECS env override.

    _load_watchdog() re-execs the module, so an env var set before the call
    is picked up at (re)import time — mirrors restart-all-orchestrators.sh's
    RESTART_VERIFY_TIMEOUT env-with-default pattern.
    """
    monkeypatch.setenv("STALENESS_GRACE_SECS", "60")
    wdog = _load_watchdog()
    assert wdog.STALENESS_GRACE_SECS == 60


def test_staleness_grace_secs_malformed_env_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """A malformed STALENESS_GRACE_SECS env value falls back to 1800, not a crash.

    A typo'd env var must not crash the oneshot watchdog — fall-safe ethos.
    """
    monkeypatch.setenv("STALENESS_GRACE_SECS", "not-an-int")
    wdog = _load_watchdog()
    assert wdog.STALENESS_GRACE_SECS == 1800


# ---------------------------------------------------------------------------
# staleness_pass core tests
#
# These tests set every restraint gate permissive (enabled, past startup
# grace, commit older than STALENESS_GRACE_SECS) so they exercise the core
# per-unit staleness comparison. Later steps add tests for each gate itself.
# ---------------------------------------------------------------------------


def test_staleness_pass_core(monkeypatch: pytest.MonkeyPatch) -> None:
    """staleness_pass restarts only the unit stale w.r.t. the newest watched commit.

    Also exercises I6 convergence: once a restart refreshes a unit's start
    epoch to newer-than-commit, a second pass issues no further restart.
    """
    wdog = _load_watchdog()
    restarted: list[str] = []
    log_messages: list[str] = []

    now = 2_000_000_000.0
    commit_epoch = int(now) - wdog.STALENESS_GRACE_SECS - 100  # older than grace

    stale_unit = "orchestrator-stale.service"
    fresh_unit = "orchestrator-fresh.service"
    unknown_unit = "orchestrator-unknown.service"

    start_epochs = {
        stale_unit: commit_epoch - 100,  # started before the commit -> stale
        fresh_unit: commit_epoch + 100,  # started after the commit -> fresh
        unknown_unit: None,  # undeterminable -> must not restart
    }

    monkeypatch.setattr(
        wdog, "_enumerate_running_units", lambda: [stale_unit, fresh_unit, unknown_unit]
    )
    monkeypatch.setattr(wdog, "is_unit_enabled", lambda _u: True)
    monkeypatch.setattr(wdog, "_unit_start_elapsed_secs", lambda _u: 300.0)
    monkeypatch.setattr(wdog, "_newest_watched_commit_epoch", lambda: commit_epoch)
    monkeypatch.setattr(wdog.time, "time", lambda: now)
    monkeypatch.setattr(wdog, "_unit_start_epoch", lambda u: start_epochs[u])
    monkeypatch.setattr(wdog, "restart_unit", lambda u: restarted.append(u))
    monkeypatch.setattr(wdog, "log", lambda m: log_messages.append(m))

    wdog.staleness_pass()

    assert restarted == [stale_unit], f"Expected only {stale_unit} restarted, got {restarted}"
    assert any(("WARNING" in m and stale_unit in m) for m in log_messages), (
        f"Expected a WARNING log line naming {stale_unit}: {log_messages}"
    )

    # --- Convergence (I6): a real restart refreshes the unit's start epoch,
    # so a second pass must issue no further restart for the same unit.
    restarted.clear()
    start_epochs[stale_unit] = commit_epoch + 50  # as if just restarted
    wdog.staleness_pass()
    assert restarted == [], (
        f"staleness_pass must self-clear once the unit's start epoch is fresh; got {restarted}"
    )


def test_staleness_pass_isolates_per_unit_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """staleness_pass must not let one unit's exception stop later units from processing."""
    wdog = _load_watchdog()
    restarted: list[str] = []

    now = 2_000_000_000.0
    commit_epoch = int(now) - wdog.STALENESS_GRACE_SECS - 100

    boom_unit = "orchestrator-boom.service"
    stale_unit = "orchestrator-stale2.service"

    def fake_start_epoch(unit: str):  # noqa: ANN001
        if unit == boom_unit:
            raise RuntimeError("systemctl exploded")
        return commit_epoch - 100  # stale

    monkeypatch.setattr(wdog, "_enumerate_running_units", lambda: [boom_unit, stale_unit])
    monkeypatch.setattr(wdog, "is_unit_enabled", lambda _u: True)
    monkeypatch.setattr(wdog, "_unit_start_elapsed_secs", lambda _u: 300.0)
    monkeypatch.setattr(wdog, "_newest_watched_commit_epoch", lambda: commit_epoch)
    monkeypatch.setattr(wdog.time, "time", lambda: now)
    monkeypatch.setattr(wdog, "_unit_start_epoch", fake_start_epoch)
    monkeypatch.setattr(wdog, "restart_unit", lambda u: restarted.append(u))
    monkeypatch.setattr(wdog, "log", lambda _m: None)

    # Must not raise
    wdog.staleness_pass()

    assert stale_unit in restarted, (
        f"{stale_unit} must still be processed after {boom_unit} raised; got {restarted}"
    )


def test_staleness_pass_noop_when_commit_epoch_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """staleness_pass takes no action at all when the commit epoch is undeterminable."""
    wdog = _load_watchdog()
    enumerated: list[str] = []

    def fake_enumerate():
        enumerated.append("called")
        return ["orchestrator-x.service"]

    monkeypatch.setattr(wdog, "_newest_watched_commit_epoch", lambda: None)
    monkeypatch.setattr(wdog, "_enumerate_running_units", fake_enumerate)
    monkeypatch.setattr(wdog, "restart_unit", lambda _u: pytest.fail("must not restart"))

    wdog.staleness_pass()

    assert enumerated == [], (
        "staleness_pass must return before enumerating units when commit_epoch is None"
    )


def test_staleness_pass_skips_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """staleness_pass must not restart a stale unit that is_unit_enabled reports False.

    Disabling is explicit operator intent — the backstop must respect it,
    identically to main()'s existing enabled gate (I5).
    """
    wdog = _load_watchdog()
    restarted: list[str] = []

    now = 2_000_000_000.0
    commit_epoch = int(now) - wdog.STALENESS_GRACE_SECS - 100

    disabled_unit = "orchestrator-disabled.service"

    monkeypatch.setattr(wdog, "_enumerate_running_units", lambda: [disabled_unit])
    monkeypatch.setattr(wdog, "is_unit_enabled", lambda _u: False)
    monkeypatch.setattr(wdog, "_unit_start_elapsed_secs", lambda _u: 300.0)
    monkeypatch.setattr(wdog, "_newest_watched_commit_epoch", lambda: commit_epoch)
    monkeypatch.setattr(wdog.time, "time", lambda: now)
    monkeypatch.setattr(wdog, "_unit_start_epoch", lambda _u: commit_epoch - 100)  # stale
    monkeypatch.setattr(wdog, "restart_unit", lambda u: restarted.append(u))
    monkeypatch.setattr(wdog, "log", lambda _m: None)

    wdog.staleness_pass()

    assert restarted == [], f"Disabled unit must not be restarted; got {restarted}"


def test_staleness_pass_skips_startup_grace(monkeypatch: pytest.MonkeyPatch) -> None:
    """staleness_pass must not restart a stale, enabled unit within STARTUP_GRACE_SECS.

    Mirrors main()'s existing grace-window gate: a unit that just (re)started
    may not have converged on the new commit's effects yet, and restarting it
    again would risk an indefinite restart loop (I5).
    """
    wdog = _load_watchdog()
    restarted: list[str] = []

    now = 2_000_000_000.0
    commit_epoch = int(now) - wdog.STALENESS_GRACE_SECS - 100

    grace_unit = "orchestrator-grace.service"

    monkeypatch.setattr(wdog, "_enumerate_running_units", lambda: [grace_unit])
    monkeypatch.setattr(wdog, "is_unit_enabled", lambda _u: True)
    monkeypatch.setattr(wdog, "_unit_start_elapsed_secs", lambda _u: 30.0)  # < 120s grace
    monkeypatch.setattr(wdog, "_newest_watched_commit_epoch", lambda: commit_epoch)
    monkeypatch.setattr(wdog.time, "time", lambda: now)
    monkeypatch.setattr(wdog, "_unit_start_epoch", lambda _u: commit_epoch - 100)  # stale
    monkeypatch.setattr(wdog, "restart_unit", lambda u: restarted.append(u))
    monkeypatch.setattr(wdog, "log", lambda _m: None)

    wdog.staleness_pass()

    assert restarted == [], f"Unit within startup grace must not be restarted; got {restarted}"


def test_staleness_pass_none_elapsed_does_not_block_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """staleness_pass must still restart a stale unit when elapsed is None (undeterminable).

    None means "grace window does not apply" — fall-through, consistent with
    main()'s existing treatment of an undeterminable elapsed time.
    """
    wdog = _load_watchdog()
    restarted: list[str] = []

    now = 2_000_000_000.0
    commit_epoch = int(now) - wdog.STALENESS_GRACE_SECS - 100

    unit = "orchestrator-unknown-elapsed.service"

    monkeypatch.setattr(wdog, "_enumerate_running_units", lambda: [unit])
    monkeypatch.setattr(wdog, "is_unit_enabled", lambda _u: True)
    monkeypatch.setattr(wdog, "_unit_start_elapsed_secs", lambda _u: None)
    monkeypatch.setattr(wdog, "_newest_watched_commit_epoch", lambda: commit_epoch)
    monkeypatch.setattr(wdog.time, "time", lambda: now)
    monkeypatch.setattr(wdog, "_unit_start_epoch", lambda _u: commit_epoch - 100)  # stale
    monkeypatch.setattr(wdog, "restart_unit", lambda u: restarted.append(u))
    monkeypatch.setattr(wdog, "log", lambda _m: None)

    wdog.staleness_pass()

    assert restarted == [unit], (
        f"A None elapsed must not block the staleness restart; got {restarted}"
    )


def test_staleness_pass_commit_grace(monkeypatch: pytest.MonkeyPatch) -> None:
    """staleness_pass restarts nothing when the newest watched commit is too young.

    A commit younger than STALENESS_GRACE_SECS gives the polite event-driven
    restart coordinator its head start — the backstop must not race it (I5).
    This is a fleet-wide gate: it must suppress every unit's restart, not
    just the one under test.

    The commit_epoch-None fall-safe (a complete no-op, no enumeration) is
    covered separately by test_staleness_pass_noop_when_commit_epoch_none.
    """
    wdog = _load_watchdog()
    restarted: list[str] = []

    now = 2_000_000_000.0
    commit_epoch = int(now) - 300  # younger than STALENESS_GRACE_SECS=1800

    stale_unit = "orchestrator-young-commit.service"

    monkeypatch.setattr(wdog, "_enumerate_running_units", lambda: [stale_unit])
    monkeypatch.setattr(wdog, "is_unit_enabled", lambda _u: True)
    monkeypatch.setattr(wdog, "_unit_start_elapsed_secs", lambda _u: 300.0)
    monkeypatch.setattr(wdog, "_newest_watched_commit_epoch", lambda: commit_epoch)
    monkeypatch.setattr(wdog.time, "time", lambda: now)
    monkeypatch.setattr(wdog, "_unit_start_epoch", lambda _u: commit_epoch - 100)  # stale
    monkeypatch.setattr(wdog, "restart_unit", lambda u: restarted.append(u))
    monkeypatch.setattr(wdog, "log", lambda _m: None)

    wdog.staleness_pass()

    assert restarted == [], (
        f"A commit younger than STALENESS_GRACE_SECS must suppress all restarts; got {restarted}"
    )


# ---------------------------------------------------------------------------
# report() tests
# ---------------------------------------------------------------------------


def test_report_mixed_fleet_returns_1_and_lists_all_units(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """report() lists every unit with a verdict, returns 1 if any is stale, mutates nothing (I7).

    Drives report() through its REAL helpers (_enumerate_running_units,
    _newest_watched_commit_epoch, _unit_start_epoch) against a single faked
    subprocess.run, dispatching on argv shape the same way the probe_port
    tests do. This way recorded_calls captures the actual systemctl/git argv
    report() drives, so the "zero mutating calls" assertion below is
    meaningful — a prior version of this test monkeypatched all three
    helpers directly, so subprocess.run was never invoked inside report()
    and that assertion passed vacuously (recorded_calls was always empty).
    restart_unit is also monkeypatched to fail the test outright if called,
    as a direct belt-and-suspenders check.
    """
    wdog = _load_watchdog()

    commit_epoch = 1_800_000_000
    units = [f"orchestrator-unit{i}.service" for i in range(6)]
    # unit0 is stale; the rest are fresh.
    start_epochs = {units[0]: commit_epoch - 100}
    for u in units[1:]:
        start_epochs[u] = commit_epoch + 100

    recorded_calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        recorded_calls.append(list(cmd))
        if cmd[:3] == ["systemctl", "--user", "list-units"]:
            stdout = "".join(f"{u} loaded active running desc\n" for u in units)
            return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")
        if cmd[:3] == ["systemctl", "--user", "show"]:
            unit = cmd[3]
            return subprocess.CompletedProcess(
                cmd, 0, stdout=f"ExecMainStartTimestamp=@{start_epochs[unit]}\n", stderr=""
            )
        if cmd[0] == "git":
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{commit_epoch}\n", stderr="")
        pytest.fail(f"unexpected subprocess.run call inside report(): {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        wdog, "restart_unit", lambda u: pytest.fail(f"report() must never restart {u}")
    )

    exit_code = wdog.report()

    assert exit_code == 1, "report() must return 1 when any unit is stale"

    captured = capsys.readouterr()
    for u in units:
        assert u in captured.out, f"report() output must list {u}: {captured.out}"
    assert "stale" in captured.out
    assert "fresh" in captured.out

    assert recorded_calls, (
        "report() must have driven subprocess.run through its real helpers "
        "for this assertion to mean anything"
    )
    mutating_verbs = {"stop", "start", "restart", "reset-failed"}
    for call in recorded_calls:
        for token in call:
            assert token not in mutating_verbs, (
                f"report() must perform zero mutating systemctl calls; saw {call}"
            )


def test_report_all_fresh_returns_0(monkeypatch: pytest.MonkeyPatch) -> None:
    """report() returns 0 when every unit is fresh."""
    wdog = _load_watchdog()

    commit_epoch = 1_800_000_000
    units = [f"orchestrator-unit{i}.service" for i in range(6)]
    start_epochs = {u: commit_epoch + 100 for u in units}

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""),
    )
    monkeypatch.setattr(wdog, "_enumerate_running_units", lambda: list(units))
    monkeypatch.setattr(wdog, "_newest_watched_commit_epoch", lambda: commit_epoch)
    monkeypatch.setattr(wdog, "_unit_start_epoch", lambda u: start_epochs[u])

    assert wdog.report() == 0


def test_report_unknown_verdict_does_not_force_exit_1(monkeypatch: pytest.MonkeyPatch) -> None:
    """report() returns 0 when verdicts are 'unknown' (undeterminable), not forced to 1.

    An 'unknown' verdict (either epoch undeterminable) must not be treated as
    a failure signal — only a confirmed-stale unit should force exit 1.
    """
    wdog = _load_watchdog()
    units = ["orchestrator-x.service"]

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""),
    )
    monkeypatch.setattr(wdog, "_enumerate_running_units", lambda: list(units))
    monkeypatch.setattr(wdog, "_newest_watched_commit_epoch", lambda: None)  # undeterminable
    monkeypatch.setattr(wdog, "_unit_start_epoch", lambda u: 12345)

    assert wdog.report() == 0


# ---------------------------------------------------------------------------
# _cli tests
# ---------------------------------------------------------------------------


def test_cli_report_flag_routes_to_report_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """_cli(["--report"]) calls report() exactly once and does not run main()/staleness_pass()."""
    wdog = _load_watchdog()
    calls: list[str] = []

    monkeypatch.setattr(wdog, "report", lambda: calls.append("report") or 0)
    monkeypatch.setattr(wdog, "main", lambda: calls.append("main"))
    monkeypatch.setattr(wdog, "staleness_pass", lambda: calls.append("staleness_pass"))

    exit_code = wdog._cli(["--report"])

    assert calls == ["report"], f"Expected only report() called, got {calls}"
    assert exit_code == 0


def test_cli_report_flag_returns_reports_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    """_cli(["--report"]) returns report()'s own return value, and skips the mutating paths."""
    wdog = _load_watchdog()

    monkeypatch.setattr(wdog, "report", lambda: 1)
    monkeypatch.setattr(wdog, "main", lambda: pytest.fail("main() must not run under --report"))
    monkeypatch.setattr(
        wdog,
        "staleness_pass",
        lambda: pytest.fail("staleness_pass() must not run under --report"),
    )

    assert wdog._cli(["--report"]) == 1


def test_cli_default_runs_main_then_staleness_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """_cli([]) runs main() then staleness_pass(), in that order, and never report()."""
    wdog = _load_watchdog()
    calls: list[str] = []

    monkeypatch.setattr(wdog, "report", lambda: calls.append("report") or 0)
    monkeypatch.setattr(wdog, "main", lambda: calls.append("main"))
    monkeypatch.setattr(wdog, "staleness_pass", lambda: calls.append("staleness_pass"))

    wdog._cli([])

    assert calls == ["main", "staleness_pass"], f"Expected main then staleness_pass, got {calls}"


def test_cli_unknown_flag_does_not_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    """_cli(["--bogus"]) must not raise; unknown flags fall through to the timer path."""
    wdog = _load_watchdog()
    calls: list[str] = []

    monkeypatch.setattr(wdog, "report", lambda: calls.append("report") or 0)
    monkeypatch.setattr(wdog, "main", lambda: calls.append("main"))
    monkeypatch.setattr(wdog, "staleness_pass", lambda: calls.append("staleness_pass"))

    # Must not raise
    wdog._cli(["--bogus"])

    assert calls == ["main", "staleness_pass"]


def test_cli_defaults_to_sys_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    """_cli() with no argv argument reads sys.argv[1:]."""
    wdog = _load_watchdog()
    calls: list[str] = []

    monkeypatch.setattr(wdog, "report", lambda: calls.append("report") or 0)
    monkeypatch.setattr(wdog.sys, "argv", ["orchestrator-watchdog.py", "--report"])

    exit_code = wdog._cli()

    assert calls == ["report"]
    assert exit_code == 0

