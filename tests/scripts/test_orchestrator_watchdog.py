"""Unit tests for scripts/orchestrator-watchdog.py.

The watchdog module has a hyphenated filename so it cannot be imported via
``import orchestrator_watchdog``.  We use importlib to load it by file path.

No live systemd runtime is needed — all subprocess.run calls are monkeypatched.
"""

import importlib.util
import pathlib
import subprocess
import types

import pytest
import yaml

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
    wdog.restart_unit("dark-factory-orchestrator.service")

    assert len(calls) == 3, f"Expected exactly 3 systemctl calls, got {len(calls)}: {calls}"
    assert calls[0] == ["systemctl", "--user", "stop", "dark-factory-orchestrator.service"]
    assert calls[1] == ["systemctl", "--user", "reset-failed", "dark-factory-orchestrator.service"]
    assert calls[2] == ["systemctl", "--user", "start", "dark-factory-orchestrator.service"]


def test_restart_unit_never_uses_kill(monkeypatch: pytest.MonkeyPatch) -> None:
    """restart_unit must not invoke systemctl kill or any -KILL/-9 signal."""
    wdog = _load_watchdog()
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    wdog.restart_unit("dark-factory-orchestrator.service")

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
    wdog.restart_unit("dark-factory-orchestrator.service")

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
    wdog.restart_unit("dark-factory-orchestrator.service")

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
    wdog.restart_unit("dark-factory-orchestrator.service")

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


def test_watched_ports_match_configured_escalation_ports() -> None:
    """WATCHED ports must equal the escalation.port values in each orchestrator's config.

    This is a behavioral drift guard: it passes when the ports are aligned and
    fails if WATCHED or either config file drifts to a different port number.
    The reify config is skipped gracefully if it is not reachable (e.g. CI).
    """
    wdog = _load_watchdog()

    # Build a unit→port map from WATCHED for convenient lookup
    unit_to_port = {unit: port for port, unit in wdog.WATCHED}

    # --- dark-factory orchestrator ---
    df_config_path = REPO_ROOT / "orchestrator" / "config.yaml"
    df_cfg = yaml.safe_load(df_config_path.read_text())
    df_port = df_cfg["escalation"]["port"]
    assert unit_to_port["dark-factory-orchestrator.service"] == df_port, (
        f"WATCHED port for dark-factory-orchestrator.service "
        f"({unit_to_port['dark-factory-orchestrator.service']}) != "
        f"orchestrator/config.yaml escalation.port ({df_port})"
    )

    # --- reify orchestrator (skip if absent in this environment) ---
    reify_config_path = pathlib.Path("/home/leo/src/reify/orchestrator.yaml")
    if not reify_config_path.exists():
        pytest.skip("reify orchestrator.yaml not reachable in this environment")
    reify_cfg = yaml.safe_load(reify_config_path.read_text())
    reify_port = reify_cfg["escalation"]["port"]
    assert unit_to_port["reify-orchestrator.service"] == reify_port, (
        f"WATCHED port for reify-orchestrator.service "
        f"({unit_to_port['reify-orchestrator.service']}) != "
        f"reify/orchestrator.yaml escalation.port ({reify_port})"
    )


# ---------------------------------------------------------------------------
# main() tests
# ---------------------------------------------------------------------------


def test_main_targets_expected_pairs() -> None:
    """WATCHED must list the two orchestrator (port, unit) pairs in order."""
    wdog = _load_watchdog()
    assert hasattr(wdog, "WATCHED"), "Module must expose a WATCHED constant"
    assert wdog.WATCHED == [
        (8102, "dark-factory-orchestrator.service"),
        (8100, "reify-orchestrator.service"),
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
    monkeypatch.setattr(wdog, "log", fake_log)

    wdog.main()

    assert restarted == ["dark-factory-orchestrator.service"], (
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
    monkeypatch.setattr(wdog, "log", fake_log)

    wdog.main()

    assert len(log_messages) >= 1, "Expected at least one log() call for the restarted unit"
    # At least one message must mention the unit being restarted
    assert any("dark-factory-orchestrator.service" in m for m in log_messages), (
        f"No log message mentions dark-factory-orchestrator.service: {log_messages}"
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
    monkeypatch.setattr(wdog, "log", fake_log)

    # Must not raise
    wdog.main()

    assert "reify-orchestrator.service" in restarted, (
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
    monkeypatch.setattr(wdog, "log", fake_log)

    wdog.main()

    assert len(probed) == 2, (
        f"Both ports must be probed when outside grace window; probed: {probed}"
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
    monkeypatch.setattr(wdog, "log", fake_log)

    wdog.main()

    assert len(probed) == 2, (
        f"When elapsed is None, both ports must be probed; probed: {probed}"
    )
