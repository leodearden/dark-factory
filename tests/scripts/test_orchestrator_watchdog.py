"""Unit tests for scripts/orchestrator-watchdog.py.

The watchdog module has a hyphenated filename so it cannot be imported via
``import orchestrator_watchdog``.  We use importlib to load it by file path.

No live systemd runtime is needed — all subprocess.run calls are monkeypatched.
"""

import importlib.util
import json
import os
import pathlib
import re
import subprocess
import time
import types

import pytest  # pyright: ignore[reportMissingImports]

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
    df_config_path = REPO_ROOT / "dark-factory-orchestrator.yaml"
    df_cfg = yaml.safe_load(df_config_path.read_text())
    df_port = _extract_escalation_port(df_cfg, df_config_path)
    assert unit_to_port["orchestrator-dark-factory.service"] == df_port, (
        f"WATCHED port for orchestrator-dark-factory.service "
        f"({unit_to_port['orchestrator-dark-factory.service']}) != "
        f"dark-factory-orchestrator.yaml escalation.port ({df_port})"
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

    # --- know-live orchestrator (check if present) ---
    know_live_config_path = pathlib.Path("/home/leo/src/know-live/orchestrator.yaml")
    if know_live_config_path.exists():
        know_live_cfg = yaml.safe_load(know_live_config_path.read_text())
        know_live_port = _extract_escalation_port(know_live_cfg, know_live_config_path)
        assert unit_to_port["orchestrator-know-live.service"] == know_live_port, (
            f"WATCHED port for orchestrator-know-live.service "
            f"({unit_to_port['orchestrator-know-live.service']}) != "
            f"know-live/orchestrator.yaml escalation.port ({know_live_port})"
        )

    # --- autopilot-video orchestrator (check if present) ---
    autopilot_video_config_path = pathlib.Path(
        "/home/leo/src/autopilot-video/orchestrator-config.yaml"
    )
    if autopilot_video_config_path.exists():
        autopilot_video_cfg = yaml.safe_load(autopilot_video_config_path.read_text())
        autopilot_video_port = _extract_escalation_port(
            autopilot_video_cfg, autopilot_video_config_path
        )
        assert unit_to_port["orchestrator-autopilot-video.service"] == autopilot_video_port, (
            f"WATCHED port for orchestrator-autopilot-video.service "
            f"({unit_to_port['orchestrator-autopilot-video.service']}) != "
            f"autopilot-video/orchestrator-config.yaml escalation.port ({autopilot_video_port})"
        )

    # --- solar-challenge-platform orchestrator (check if present) ---
    solar_challenge_platform_config_path = pathlib.Path(
        "/home/leo/src/solar-challenge-platform/orchestrator.yaml"
    )
    if solar_challenge_platform_config_path.exists():
        solar_challenge_platform_cfg = yaml.safe_load(
            solar_challenge_platform_config_path.read_text()
        )
        solar_challenge_platform_port = _extract_escalation_port(
            solar_challenge_platform_cfg, solar_challenge_platform_config_path
        )
        assert (
            unit_to_port["orchestrator-solar-challenge-platform.service"]
            == solar_challenge_platform_port
        ), (
            f"WATCHED port for orchestrator-solar-challenge-platform.service "
            f"({unit_to_port['orchestrator-solar-challenge-platform.service']}) != "
            f"solar-challenge-platform/orchestrator.yaml escalation.port "
            f"({solar_challenge_platform_port})"
        )

    # --- pump-web-ui orchestrator (check if present) ---
    pump_web_ui_config_path = pathlib.Path("/home/leo/src/pump-web-ui/orchestrator.yaml")
    if pump_web_ui_config_path.exists():
        pump_web_ui_cfg = yaml.safe_load(pump_web_ui_config_path.read_text())
        pump_web_ui_port = _extract_escalation_port(pump_web_ui_cfg, pump_web_ui_config_path)
        assert unit_to_port["orchestrator-pump-web-ui.service"] == pump_web_ui_port, (
            f"WATCHED port for orchestrator-pump-web-ui.service "
            f"({unit_to_port['orchestrator-pump-web-ui.service']}) != "
            f"pump-web-ui/orchestrator.yaml escalation.port ({pump_web_ui_port})"
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
        (8105, "orchestrator-know-live.service"),
        (8101, "orchestrator-autopilot-video.service"),
        (8107, "orchestrator-solar-challenge-platform.service"),
        (8108, "orchestrator-pump-web-ui.service"),
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
# _JournalLog.warning() fail-soft swallow tests (follow-up from esc-2032-2)
#
# _JournalLog.warning() routes through the module-level log() helper and is
# itself wrapped in `try/except Exception: pass` so a journald-write failure
# can never convert a probe's return-None contract into a raised exception.
# These tests monkeypatch log() to raise, then force each probe into its
# broad-except branch (where logger.warning(...) is actually invoked) and
# assert the probe still returns None with nothing propagating.
# ---------------------------------------------------------------------------


def test_unit_start_epoch_warning_log_failure_is_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_unit_start_epoch returns None (no raise) when its except-block's
    logger.warning(...) call hits a log() that itself raises.
    """
    wdog = _load_watchdog()

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        raise RuntimeError("boom: systemctl subprocess exploded")

    def fake_log(msg):  # noqa: ANN001
        raise FileNotFoundError(2, "No such file or directory", "systemd-cat")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(wdog, "log", fake_log)

    assert wdog._unit_start_epoch("some.service") is None


def test_newest_watched_commit_epoch_warning_log_failure_is_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_newest_watched_commit_epoch returns None (no raise) when its
    except-block's logger.warning(...) call hits a log() that itself raises.
    """
    wdog = _load_watchdog()

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        raise RuntimeError("boom: git subprocess exploded")

    def fake_log(msg):  # noqa: ANN001
        raise OSError("journald socket unavailable")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(wdog, "log", fake_log)

    assert wdog._newest_watched_commit_epoch() is None


def test_journal_log_warning_swallows_log_failure_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_JournalLog().warning(...) itself returns None without raising when
    the underlying log() helper raises — the fail-soft guarantee in isolation,
    independent of any calling probe.
    """
    wdog = _load_watchdog()

    def fake_log(msg):  # noqa: ANN001
        raise FileNotFoundError(2, "No such file or directory", "systemd-cat")

    monkeypatch.setattr(wdog, "log", fake_log)

    assert wdog._JournalLog().warning("x") is None


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
# ORCH_RESTART_MIN_INTERVAL_SECS + shared fleet-deploy clock tests
# (task 2396, fleet-redeploy β)
#
# These cover the watchdog's clock-awareness primitives in isolation:
# ORCH_RESTART_MIN_INTERVAL_SECS (the env-mirror of
# OrchestratorConfig.orchestrator_restart_min_interval_secs),
# _read_last_fleet_deploy_epoch() (fail-open JSON read of the shared clock
# file), and _within_fleet_deploy_min_interval() (the gate predicate). None of
# these are wired into staleness_pass() yet — that wiring is covered by the
# staleness_pass fleet-deploy clock gate tests further below.
# ---------------------------------------------------------------------------


def test_orch_restart_min_interval_secs_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """ORCH_RESTART_MIN_INTERVAL_SECS defaults to 28800 (8h) with no env override."""
    monkeypatch.delenv("ORCH_RESTART_MIN_INTERVAL_SECS", raising=False)
    wdog = _load_watchdog()
    assert wdog.ORCH_RESTART_MIN_INTERVAL_SECS == 28800


def test_orch_restart_min_interval_secs_matches_config_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The watchdog's env-mirror default must not drift from OrchestratorConfig's.

    The watchdog is a stdlib-only systemd oneshot that cannot import the
    orchestrator package at runtime, so ORCH_RESTART_MIN_INTERVAL_SECS's
    default is a hardcoded mirror of
    OrchestratorConfig.orchestrator_restart_min_interval_secs (28800.0, task
    2371). This drift test — mirroring
    tests/scripts/test_orchestrator_restart_config_drift.py — pins the two
    together so a future config.py change doesn't silently diverge from the
    watchdog's copy (Open-Q1).
    """
    from orchestrator.config import OrchestratorConfig

    monkeypatch.delenv("ORCH_RESTART_MIN_INTERVAL_SECS", raising=False)
    # Anchor ORCH_CONFIG_PATH to this worktree's own committed config so the
    # comparison is deterministic regardless of the ambient shell env (which
    # may point ORCH_CONFIG_PATH at a different checkout).
    monkeypatch.setenv("ORCH_CONFIG_PATH", str(REPO_ROOT / "dark-factory-orchestrator.yaml"))
    wdog = _load_watchdog()
    assert pytest.approx(
        OrchestratorConfig().orchestrator_restart_min_interval_secs
    ) == wdog.ORCH_RESTART_MIN_INTERVAL_SECS


def test_orch_restart_min_interval_secs_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """ORCH_RESTART_MIN_INTERVAL_SECS honors a valid env override.

    _load_watchdog() re-execs the module, so an env var set before the call
    is picked up at (re)import time — mirrors STALENESS_GRACE_SECS's own
    env-override test above.
    """
    monkeypatch.setenv("ORCH_RESTART_MIN_INTERVAL_SECS", "60")
    wdog = _load_watchdog()
    assert wdog.ORCH_RESTART_MIN_INTERVAL_SECS == 60


def test_fleet_deploy_clock_path_matches_across_tiers(monkeypatch: pytest.MonkeyPatch) -> None:
    """The shared clock-path literal must not silently diverge across tiers.

    orchestrator.service_restart.FLEET_DEPLOY_CLOCK_RELPATH is the single
    authoritative relative path (task 2396). Neither the stdlib watchdog
    (FLEET_DEPLOY_CLOCK_PATH) nor restart-all-orchestrators.sh (CLOCK_FILE)
    can import it, so each hardcodes its own mirror. If those mirrors ever
    drifted from the authoritative constant, the watchdog would read a
    different file than the script writes — permanently un-gating the
    staleness backstop and silently reintroducing the I2 hole this task
    closes, with every other test still green. This pins all three copies
    together, mirroring test_orch_restart_min_interval_secs_matches_config_default
    above.
    """
    from orchestrator.service_restart import FLEET_DEPLOY_CLOCK_RELPATH

    # --- watchdog mirror (FLEET_DEPLOY_CLOCK_PATH) ---
    monkeypatch.delenv("ORCH_FLEET_DEPLOY_CLOCK", raising=False)
    wdog = _load_watchdog()
    expected_watchdog_path = str(pathlib.Path(wdog.REPO_DIR) / FLEET_DEPLOY_CLOCK_RELPATH)
    assert expected_watchdog_path == wdog.FLEET_DEPLOY_CLOCK_PATH

    # --- bash script mirror (CLOCK_FILE default) ---
    script_src = (REPO_ROOT / "scripts" / "restart-all-orchestrators.sh").read_text()
    match = re.search(
        r'CLOCK_FILE="\$\{ORCH_FLEET_DEPLOY_CLOCK:-\$REPO_DIR/([^}]+)\}"',
        script_src,
    )
    assert match is not None, (
        "restart-all-orchestrators.sh CLOCK_FILE default pattern not found — "
        "did its literal shape change? Update this regex to match."
    )
    assert match.group(1) == FLEET_DEPLOY_CLOCK_RELPATH


def test_orch_restart_min_interval_secs_malformed_env_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed ORCH_RESTART_MIN_INTERVAL_SECS env value falls back to 28800.

    A typo'd env var must not crash the oneshot watchdog — fall-safe ethos,
    mirroring STALENESS_GRACE_SECS's malformed-env test above.
    """
    monkeypatch.setenv("ORCH_RESTART_MIN_INTERVAL_SECS", "not-an-int")
    wdog = _load_watchdog()
    assert wdog.ORCH_RESTART_MIN_INTERVAL_SECS == 28800


def test_read_last_fleet_deploy_epoch_happy_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """_read_last_fleet_deploy_epoch reads the float `ts` from the clock file.

    Mirrors StaleServiceRestartCoordinator._load_last_fire_wall's {ts, iso}
    schema and fail-open semantics (orchestrator/src/orchestrator/service_restart.py).
    """
    wdog = _load_watchdog()
    clock_file = tmp_path / "last_redeploy_orchestrator.json"
    clock_file.write_text('{"ts": 1783000000.0, "iso": "2026-07-16T00:00:00+00:00"}')
    monkeypatch.setattr(wdog, "FLEET_DEPLOY_CLOCK_PATH", str(clock_file))

    result = wdog._read_last_fleet_deploy_epoch()
    assert result == pytest.approx(1783000000.0)
    assert isinstance(result, float)


def test_read_last_fleet_deploy_epoch_missing_file_returns_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """_read_last_fleet_deploy_epoch returns None when the clock file does not exist.

    A fleet that has never had a verified redeploy (or a fresh checkout) must
    not be treated as perpetually inside the min-interval window.
    """
    wdog = _load_watchdog()
    monkeypatch.setattr(wdog, "FLEET_DEPLOY_CLOCK_PATH", str(tmp_path / "absent.json"))
    assert wdog._read_last_fleet_deploy_epoch() is None


def test_read_last_fleet_deploy_epoch_corrupt_json_returns_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """_read_last_fleet_deploy_epoch returns None (fail-open) on unparseable JSON."""
    wdog = _load_watchdog()
    clock_file = tmp_path / "corrupt.json"
    clock_file.write_text("{not-json")
    monkeypatch.setattr(wdog, "FLEET_DEPLOY_CLOCK_PATH", str(clock_file))
    assert wdog._read_last_fleet_deploy_epoch() is None


def test_read_last_fleet_deploy_epoch_missing_ts_key_returns_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """_read_last_fleet_deploy_epoch returns None when the `ts` key is absent."""
    wdog = _load_watchdog()
    clock_file = tmp_path / "no_ts.json"
    clock_file.write_text('{"iso": "2026-07-16T00:00:00+00:00"}')
    monkeypatch.setattr(wdog, "FLEET_DEPLOY_CLOCK_PATH", str(clock_file))
    assert wdog._read_last_fleet_deploy_epoch() is None


def test_within_fleet_deploy_min_interval_true_when_recent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_within_fleet_deploy_min_interval is True when now - last < min_interval."""
    wdog = _load_watchdog()
    monkeypatch.setattr(wdog, "ORCH_RESTART_MIN_INTERVAL_SECS", 28800)
    monkeypatch.setattr(wdog, "_read_last_fleet_deploy_epoch", lambda: 1_000_000.0)
    monkeypatch.setattr(wdog.time, "time", lambda: 1_000_000.0 + 100.0)  # 100s ago

    assert wdog._within_fleet_deploy_min_interval() is True


def test_within_fleet_deploy_min_interval_false_when_elapsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_within_fleet_deploy_min_interval is False once min_interval has elapsed."""
    wdog = _load_watchdog()
    monkeypatch.setattr(wdog, "ORCH_RESTART_MIN_INTERVAL_SECS", 28800)
    monkeypatch.setattr(wdog, "_read_last_fleet_deploy_epoch", lambda: 1_000_000.0)
    monkeypatch.setattr(wdog.time, "time", lambda: 1_000_000.0 + 28800.0 + 1.0)

    assert wdog._within_fleet_deploy_min_interval() is False


def test_within_fleet_deploy_min_interval_false_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_within_fleet_deploy_min_interval is False when ORCH_RESTART_MIN_INTERVAL_SECS==0.

    0 disables the cap entirely — the read of the clock file must not even
    be attempted.
    """
    wdog = _load_watchdog()
    monkeypatch.setattr(wdog, "ORCH_RESTART_MIN_INTERVAL_SECS", 0)
    monkeypatch.setattr(
        wdog,
        "_read_last_fleet_deploy_epoch",
        lambda: pytest.fail("must not be consulted when the cap is disabled"),
    )

    assert wdog._within_fleet_deploy_min_interval() is False


def test_within_fleet_deploy_min_interval_false_when_clock_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_within_fleet_deploy_min_interval is False when the clock is unreadable/absent.

    A never-deployed fleet (or an unreadable clock file) must not block the
    backstop indefinitely — fail toward restarting, not toward silence.
    """
    wdog = _load_watchdog()
    monkeypatch.setattr(wdog, "ORCH_RESTART_MIN_INTERVAL_SECS", 28800)
    monkeypatch.setattr(wdog, "_read_last_fleet_deploy_epoch", lambda: None)

    assert wdog._within_fleet_deploy_min_interval() is False


# ---------------------------------------------------------------------------
# staleness_pass core tests
#
# These tests set every restraint gate permissive (enabled, past startup
# grace, commit older than STALENESS_GRACE_SECS) so they exercise the core
# per-unit staleness comparison. Later steps add tests for each gate itself.
# ---------------------------------------------------------------------------


def test_staleness_pass_core(monkeypatch: pytest.MonkeyPatch) -> None:
    """staleness_pass delegates a fleet-wide restart exactly once when a unit
    is stale w.r.t. the newest watched commit (not a per-unit restart_unit call).

    Also exercises I6 convergence: once a restart refreshes a unit's start
    epoch to newer-than-commit, a second pass delegates zero further times.
    """
    wdog = _load_watchdog()
    delegated: list[None] = []
    log_messages: list[str] = []

    now = 2_000_000_000.0
    commit_epoch = int(now) - wdog.STALENESS_GRACE_SECS - 100  # older than grace

    stale_unit = "orchestrator-stale.service"
    fresh_unit = "orchestrator-fresh.service"
    unknown_unit = "orchestrator-unknown.service"

    start_epochs = {
        stale_unit: commit_epoch - 100,  # started before the commit -> stale
        fresh_unit: commit_epoch + 100,  # started after the commit -> fresh
        unknown_unit: None,  # undeterminable -> must not count as stale
    }

    monkeypatch.setattr(wdog, "_within_fleet_deploy_min_interval", lambda: False)
    monkeypatch.setattr(
        wdog, "_enumerate_running_units", lambda: [stale_unit, fresh_unit, unknown_unit]
    )
    monkeypatch.setattr(wdog, "is_unit_enabled", lambda _u: True)
    monkeypatch.setattr(wdog, "_unit_start_elapsed_secs", lambda _u: 300.0)
    monkeypatch.setattr(wdog, "_newest_watched_commit_epoch", lambda: commit_epoch)
    monkeypatch.setattr(wdog.time, "time", lambda: now)
    monkeypatch.setattr(wdog, "_unit_start_epoch", lambda u: start_epochs[u])
    monkeypatch.setattr(
        wdog,
        "restart_unit",
        lambda u: pytest.fail(f"staleness_pass must delegate, not call restart_unit for {u}"),
    )
    monkeypatch.setattr(wdog, "_delegate_fleet_restart", lambda: delegated.append(None))
    monkeypatch.setattr(wdog, "log", lambda m: log_messages.append(m))

    wdog.staleness_pass()

    assert len(delegated) == 1, (
        f"Expected exactly one delegated fleet restart, got {len(delegated)}"
    )
    assert any(("WARNING" in m and stale_unit in m) for m in log_messages), (
        f"Expected a WARNING log line naming {stale_unit}: {log_messages}"
    )

    # --- Convergence (I6): a real restart refreshes the unit's start epoch,
    # so a second pass must delegate zero further times.
    delegated.clear()
    start_epochs[stale_unit] = commit_epoch + 50  # as if just restarted
    wdog.staleness_pass()
    assert delegated == [], (
        f"staleness_pass must self-clear once the unit's start epoch is fresh; "
        f"got {len(delegated)} delegation(s)"
    )


def test_staleness_pass_isolates_per_unit_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """staleness_pass must not let one unit's exception stop later units from
    processing, and must still delegate exactly once for a stale unit found
    after the exception.
    """
    wdog = _load_watchdog()
    delegated: list[None] = []

    now = 2_000_000_000.0
    commit_epoch = int(now) - wdog.STALENESS_GRACE_SECS - 100

    boom_unit = "orchestrator-boom.service"
    stale_unit = "orchestrator-stale2.service"

    def fake_start_epoch(unit: str):  # noqa: ANN001
        if unit == boom_unit:
            raise RuntimeError("systemctl exploded")
        return commit_epoch - 100  # stale

    monkeypatch.setattr(wdog, "_within_fleet_deploy_min_interval", lambda: False)
    monkeypatch.setattr(wdog, "_enumerate_running_units", lambda: [boom_unit, stale_unit])
    monkeypatch.setattr(wdog, "is_unit_enabled", lambda _u: True)
    monkeypatch.setattr(wdog, "_unit_start_elapsed_secs", lambda _u: 300.0)
    monkeypatch.setattr(wdog, "_newest_watched_commit_epoch", lambda: commit_epoch)
    monkeypatch.setattr(wdog.time, "time", lambda: now)
    monkeypatch.setattr(wdog, "_unit_start_epoch", fake_start_epoch)
    monkeypatch.setattr(wdog, "_delegate_fleet_restart", lambda: delegated.append(None))
    monkeypatch.setattr(wdog, "log", lambda _m: None)

    # Must not raise
    wdog.staleness_pass()

    assert len(delegated) == 1, (
        f"{stale_unit} must still be processed after {boom_unit} raised, triggering "
        f"exactly one delegated fleet restart; got {len(delegated)}"
    )


def test_staleness_pass_noop_when_commit_epoch_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """staleness_pass takes no action at all when the commit epoch is undeterminable."""
    wdog = _load_watchdog()
    enumerated: list[str] = []

    def fake_enumerate():
        enumerated.append("called")
        return ["orchestrator-x.service"]

    # Neutralize the fleet-deploy clock gate (task 2396 step-11): it is
    # checked BEFORE commit_epoch, and _read_last_fleet_deploy_epoch reads a
    # real on-disk file at the default path — this test must exercise the
    # commit_epoch-None path specifically, not an incidental gate skip.
    monkeypatch.setattr(wdog, "_within_fleet_deploy_min_interval", lambda: False)
    monkeypatch.setattr(wdog, "_newest_watched_commit_epoch", lambda: None)
    monkeypatch.setattr(wdog, "_enumerate_running_units", fake_enumerate)
    monkeypatch.setattr(wdog, "restart_unit", lambda _u: pytest.fail("must not restart"))

    wdog.staleness_pass()

    assert enumerated == [], (
        "staleness_pass must return before enumerating units when commit_epoch is None"
    )


def test_staleness_pass_skips_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """staleness_pass must not delegate a fleet restart for a stale unit that
    is_unit_enabled reports False.

    Disabling is explicit operator intent — the backstop must respect it,
    identically to main()'s existing enabled gate (I5).
    """
    wdog = _load_watchdog()
    delegated: list[None] = []

    now = 2_000_000_000.0
    commit_epoch = int(now) - wdog.STALENESS_GRACE_SECS - 100

    disabled_unit = "orchestrator-disabled.service"

    monkeypatch.setattr(wdog, "_within_fleet_deploy_min_interval", lambda: False)
    monkeypatch.setattr(wdog, "_enumerate_running_units", lambda: [disabled_unit])
    monkeypatch.setattr(wdog, "is_unit_enabled", lambda _u: False)
    monkeypatch.setattr(wdog, "_unit_start_elapsed_secs", lambda _u: 300.0)
    monkeypatch.setattr(wdog, "_newest_watched_commit_epoch", lambda: commit_epoch)
    monkeypatch.setattr(wdog.time, "time", lambda: now)
    monkeypatch.setattr(wdog, "_unit_start_epoch", lambda _u: commit_epoch - 100)  # stale
    monkeypatch.setattr(wdog, "_delegate_fleet_restart", lambda: delegated.append(None))
    monkeypatch.setattr(wdog, "log", lambda _m: None)

    wdog.staleness_pass()

    assert delegated == [], f"Disabled unit must not trigger a delegated restart; got {delegated}"


def test_staleness_pass_skips_startup_grace(monkeypatch: pytest.MonkeyPatch) -> None:
    """staleness_pass must not delegate a fleet restart for a stale, enabled
    unit within STARTUP_GRACE_SECS.

    Mirrors main()'s existing grace-window gate: a unit that just (re)started
    may not have converged on the new commit's effects yet, and restarting it
    again would risk an indefinite restart loop (I5).
    """
    wdog = _load_watchdog()
    delegated: list[None] = []

    now = 2_000_000_000.0
    commit_epoch = int(now) - wdog.STALENESS_GRACE_SECS - 100

    grace_unit = "orchestrator-grace.service"

    monkeypatch.setattr(wdog, "_within_fleet_deploy_min_interval", lambda: False)
    monkeypatch.setattr(wdog, "_enumerate_running_units", lambda: [grace_unit])
    monkeypatch.setattr(wdog, "is_unit_enabled", lambda _u: True)
    monkeypatch.setattr(wdog, "_unit_start_elapsed_secs", lambda _u: 30.0)  # < 120s grace
    monkeypatch.setattr(wdog, "_newest_watched_commit_epoch", lambda: commit_epoch)
    monkeypatch.setattr(wdog.time, "time", lambda: now)
    monkeypatch.setattr(wdog, "_unit_start_epoch", lambda _u: commit_epoch - 100)  # stale
    monkeypatch.setattr(wdog, "_delegate_fleet_restart", lambda: delegated.append(None))
    monkeypatch.setattr(wdog, "log", lambda _m: None)

    wdog.staleness_pass()

    assert delegated == [], (
        f"Unit within startup grace must not trigger a delegated restart; got {delegated}"
    )


def test_staleness_pass_none_elapsed_does_not_block_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """staleness_pass must still delegate a restart for a stale unit when
    elapsed is None (undeterminable).

    None means "grace window does not apply" — fall-through, consistent with
    main()'s existing treatment of an undeterminable elapsed time.
    """
    wdog = _load_watchdog()
    delegated: list[None] = []

    now = 2_000_000_000.0
    commit_epoch = int(now) - wdog.STALENESS_GRACE_SECS - 100

    unit = "orchestrator-unknown-elapsed.service"

    monkeypatch.setattr(wdog, "_within_fleet_deploy_min_interval", lambda: False)
    monkeypatch.setattr(wdog, "_enumerate_running_units", lambda: [unit])
    monkeypatch.setattr(wdog, "is_unit_enabled", lambda _u: True)
    monkeypatch.setattr(wdog, "_unit_start_elapsed_secs", lambda _u: None)
    monkeypatch.setattr(wdog, "_newest_watched_commit_epoch", lambda: commit_epoch)
    monkeypatch.setattr(wdog.time, "time", lambda: now)
    monkeypatch.setattr(wdog, "_unit_start_epoch", lambda _u: commit_epoch - 100)  # stale
    monkeypatch.setattr(wdog, "_delegate_fleet_restart", lambda: delegated.append(None))
    monkeypatch.setattr(wdog, "log", lambda _m: None)

    wdog.staleness_pass()

    assert len(delegated) == 1, (
        f"A None elapsed must not block the delegated staleness restart; got {delegated}"
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

    # Neutralize the fleet-deploy clock gate (task 2396 step-11) — see the
    # comment in test_staleness_pass_noop_when_commit_epoch_none above.
    monkeypatch.setattr(wdog, "_within_fleet_deploy_min_interval", lambda: False)
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


def test_staleness_pass_delegates_exactly_once_for_multiple_stale_units(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """staleness_pass delegates the fleet restart EXACTLY ONCE even when
    multiple units are stale — delegation is fleet-wide, not per-unit.
    """
    wdog = _load_watchdog()
    delegated: list[None] = []

    now = 2_000_000_000.0
    commit_epoch = int(now) - wdog.STALENESS_GRACE_SECS - 100

    stale_units = ["orchestrator-stale-a.service", "orchestrator-stale-b.service"]

    monkeypatch.setattr(wdog, "_within_fleet_deploy_min_interval", lambda: False)
    monkeypatch.setattr(wdog, "_enumerate_running_units", lambda: list(stale_units))
    monkeypatch.setattr(wdog, "is_unit_enabled", lambda _u: True)
    monkeypatch.setattr(wdog, "_unit_start_elapsed_secs", lambda _u: 300.0)
    monkeypatch.setattr(wdog, "_newest_watched_commit_epoch", lambda: commit_epoch)
    monkeypatch.setattr(wdog.time, "time", lambda: now)
    monkeypatch.setattr(wdog, "_unit_start_epoch", lambda _u: commit_epoch - 100)  # both stale
    monkeypatch.setattr(
        wdog,
        "restart_unit",
        lambda u: pytest.fail(f"staleness_pass must never call restart_unit directly for {u}"),
    )
    monkeypatch.setattr(wdog, "_delegate_fleet_restart", lambda: delegated.append(None))
    monkeypatch.setattr(wdog, "log", lambda _m: None)

    wdog.staleness_pass()

    assert len(delegated) == 1, (
        f"Expected exactly one delegation for {len(stale_units)} stale units; "
        f"got {len(delegated)}"
    )


def test_staleness_pass_delegates_zero_times_when_all_fresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """staleness_pass delegates zero times when every enumerated unit is fresh (I6)."""
    wdog = _load_watchdog()
    delegated: list[None] = []

    now = 2_000_000_000.0
    commit_epoch = int(now) - wdog.STALENESS_GRACE_SECS - 100

    fresh_units = ["orchestrator-fresh-a.service", "orchestrator-fresh-b.service"]

    monkeypatch.setattr(wdog, "_within_fleet_deploy_min_interval", lambda: False)
    monkeypatch.setattr(wdog, "_enumerate_running_units", lambda: list(fresh_units))
    monkeypatch.setattr(wdog, "is_unit_enabled", lambda _u: True)
    monkeypatch.setattr(wdog, "_unit_start_elapsed_secs", lambda _u: 300.0)
    monkeypatch.setattr(wdog, "_newest_watched_commit_epoch", lambda: commit_epoch)
    monkeypatch.setattr(wdog.time, "time", lambda: now)
    monkeypatch.setattr(wdog, "_unit_start_epoch", lambda _u: commit_epoch + 100)  # fresh
    monkeypatch.setattr(wdog, "_delegate_fleet_restart", lambda: delegated.append(None))
    monkeypatch.setattr(wdog, "log", lambda _m: None)

    wdog.staleness_pass()

    assert delegated == [], (
        f"Expected zero delegations when all units are fresh; got {delegated}"
    )


# ---------------------------------------------------------------------------
# _delegate_fleet_restart tests (task 2396, fleet-redeploy β, step 11)
# ---------------------------------------------------------------------------


def test_delegate_fleet_restart_argv_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """_delegate_fleet_restart fires a detached, named, drain-enabled systemd-run.

    - starts with ["systemd-run", "--user"]
    - includes --collect (transient unit auto-removed on exit) and --no-block
      (detached — the oneshot never blocks on the restart)
    - fires the fixed --unit=orch-fleet-staleness-redeploy.service name (the
      natural overlap guard: a concurrent tick fails to re-register it)
    - invokes restart-all-orchestrators.sh with --drain (so a watchdog-
      initiated fleet restart drains+stamps identically to an operator- or
      coordinator-driven one)
    """
    wdog = _load_watchdog()
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    wdog._delegate_fleet_restart()

    assert len(calls) == 1, f"Expected exactly one subprocess.run call, got {calls}"
    argv = calls[0]
    assert argv[:2] == ["systemd-run", "--user"], f"argv must start with systemd-run --user: {argv}"
    assert "--collect" in argv, f"argv must include --collect: {argv}"
    assert "--no-block" in argv, f"argv must include --no-block (detached): {argv}"
    assert "--unit=orch-fleet-staleness-redeploy.service" in argv, (
        f"argv must fire the fixed transient unit name (the overlap guard): {argv}"
    )
    assert any(a.endswith("scripts/restart-all-orchestrators.sh") for a in argv), (
        f"argv must invoke restart-all-orchestrators.sh: {argv}"
    )
    assert "--drain" in argv, (
        f"argv must pass --drain so watchdog-initiated restarts drain identically: {argv}"
    )


def test_delegate_fleet_restart_swallows_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """_delegate_fleet_restart must not raise if the systemd-run call times out."""
    wdog = _load_watchdog()
    log_messages: list[str] = []

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        raise subprocess.TimeoutExpired(cmd, 10)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(wdog, "log", lambda m: log_messages.append(m))

    # Must not raise
    wdog._delegate_fleet_restart()

    assert len(log_messages) >= 1, "a systemd-run timeout must be logged"


def test_delegate_fleet_restart_swallows_missing_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    """_delegate_fleet_restart must not raise if systemd-run is not on PATH."""
    wdog = _load_watchdog()
    log_messages: list[str] = []

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        raise FileNotFoundError(2, "No such file or directory", "systemd-run")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(wdog, "log", lambda m: log_messages.append(m))

    # Must not raise
    wdog._delegate_fleet_restart()

    assert len(log_messages) >= 1, "a missing systemd-run binary must be logged"


# ---------------------------------------------------------------------------
# staleness_pass fleet-deploy clock gate tests (task 2396, fleet-redeploy β,
# step 9)
#
# These wire _within_fleet_deploy_min_interval() into staleness_pass() as a
# top-priority, fleet-wide restraint gate — ahead of the existing
# commit-grace/per-unit gates — so the once-per-8h fleet-deploy bound is
# honored by the backstop, not just the event-driven coordinator. main()
# (liveness) is untouched (I5): brokenness is not a scheduled deploy.
# ---------------------------------------------------------------------------


def test_staleness_pass_skips_when_within_fleet_deploy_min_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """staleness_pass returns immediately when the shared fleet-deploy clock
    reports we are still inside the min-interval window: no enumeration, no
    delegation, no restart_unit call — even with a would-be-stale unit
    present — and it logs a line naming the skip (the PRD journal signal) at
    a SKIP_LOG_INTERVAL_SECS bucket boundary.

    _newest_watched_commit_epoch, _enumerate_running_units, and restart_unit
    are all monkeypatched to fail the test outright if consulted, so this
    also pins that the fleet-deploy gate is checked BEFORE the existing
    commit-grace gate (top priority) rather than merely somewhere in the
    pass. time.time() is pinned to an exact SKIP_LOG_INTERVAL_SECS multiple
    (a bucket boundary) so the per-tick log rate-limit exercised by
    test_staleness_pass_suppresses_skip_log_outside_log_bucket below cannot
    make this assertion flaky.
    """
    wdog = _load_watchdog()
    log_messages: list[str] = []

    monkeypatch.setattr(wdog, "_within_fleet_deploy_min_interval", lambda: True)
    monkeypatch.setattr(wdog.time, "time", lambda: wdog.SKIP_LOG_INTERVAL_SECS * 1000.0)
    monkeypatch.setattr(
        wdog,
        "_newest_watched_commit_epoch",
        lambda: pytest.fail("must not be consulted when the fleet-deploy gate is closed"),
    )
    monkeypatch.setattr(
        wdog,
        "_enumerate_running_units",
        lambda: pytest.fail("must not enumerate units when the fleet-deploy gate is closed"),
    )
    monkeypatch.setattr(
        wdog,
        "restart_unit",
        lambda u: pytest.fail("must not restart_unit when the fleet-deploy gate is closed"),
    )
    monkeypatch.setattr(wdog, "log", lambda m: log_messages.append(m))

    wdog.staleness_pass()

    assert any(
        "skip" in m and str(wdog.ORCH_RESTART_MIN_INTERVAL_SECS) in m for m in log_messages
    ), f"Expected a skip log line naming the fleet-deploy min-interval: {log_messages}"


def test_staleness_pass_suppresses_skip_log_outside_log_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The skip line is rate-limited to at most once per SKIP_LOG_INTERVAL_SECS.

    staleness_pass still returns immediately (no enumeration/delegation/
    restart_unit — same top-priority gate as the test above) but does NOT
    log when time.time() falls outside the bucket's logging slot. Without
    this throttle, a single 8h fleet-deploy min-interval window would write
    ~480 near-identical skip lines to the journal (one per ~60s tick),
    burying genuinely actionable watchdog output (reviewer_comprehensive
    amendment, task 2396).
    """
    wdog = _load_watchdog()
    log_messages: list[str] = []

    monkeypatch.setattr(wdog, "_within_fleet_deploy_min_interval", lambda: True)
    # Halfway into the bucket — well outside the logging slot near its start.
    monkeypatch.setattr(
        wdog.time,
        "time",
        lambda: wdog.SKIP_LOG_INTERVAL_SECS * 1000.0 + wdog.SKIP_LOG_INTERVAL_SECS / 2,
    )
    monkeypatch.setattr(
        wdog,
        "_newest_watched_commit_epoch",
        lambda: pytest.fail("must not be consulted when the fleet-deploy gate is closed"),
    )
    monkeypatch.setattr(wdog, "log", lambda m: log_messages.append(m))

    wdog.staleness_pass()

    assert log_messages == [], (
        f"Expected no skip log line outside the log-rate-limit bucket: {log_messages}"
    )


def test_staleness_pass_proceeds_when_fleet_deploy_gate_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """staleness_pass proceeds to its existing detection path when the
    fleet-deploy gate is open (disabled, or the clock is absent/elapsed):
    _enumerate_running_units must still be consulted.
    """
    wdog = _load_watchdog()
    enumerated: list[str] = []

    now = 2_000_000_000.0
    commit_epoch = int(now) - wdog.STALENESS_GRACE_SECS - 100  # older than grace

    def fake_enumerate():
        enumerated.append("called")
        return []

    monkeypatch.setattr(wdog, "_within_fleet_deploy_min_interval", lambda: False)
    monkeypatch.setattr(wdog, "_newest_watched_commit_epoch", lambda: commit_epoch)
    monkeypatch.setattr(wdog.time, "time", lambda: now)
    monkeypatch.setattr(wdog, "_enumerate_running_units", fake_enumerate)
    monkeypatch.setattr(wdog, "log", lambda _m: None)

    wdog.staleness_pass()

    assert enumerated == ["called"], (
        "staleness_pass must still reach enumeration when the fleet-deploy gate is open"
    )


def test_main_liveness_unaffected_by_fleet_deploy_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """I5: the fleet-deploy clock gate must never affect main()'s liveness restarts.

    main() still restarts a port-down unit via restart_unit even when
    _within_fleet_deploy_min_interval() would report True (inside the 8h
    fleet-deploy window) — liveness is uncapped, non-clock-gated, and
    non-stamping; brokenness is not a scheduled deploy.
    """
    wdog = _load_watchdog()
    restarted: list[str] = []

    monkeypatch.setattr(wdog, "_within_fleet_deploy_min_interval", lambda: True)
    monkeypatch.setattr(wdog, "_unit_start_elapsed_secs", lambda _u: None)
    monkeypatch.setattr(wdog, "is_unit_enabled", lambda _u: True)
    monkeypatch.setattr(wdog, "probe_port", lambda port: port != 8102)  # df probe fails
    monkeypatch.setattr(wdog, "restart_unit", lambda u: restarted.append(u))
    monkeypatch.setattr(wdog, "log", lambda _m: None)

    wdog.main()

    assert restarted == ["orchestrator-dark-factory.service"], (
        f"main() must still restart a port-down unit even when the fleet-deploy "
        f"clock gate is engaged; got {restarted}"
    )


# ---------------------------------------------------------------------------
# staleness_pass END-TO-END tests (δ task 2027, scenarios 1-4)
#
# α's staleness_pass tests above stub every helper function directly
# (_enumerate_running_units, is_unit_enabled, _unit_start_elapsed_secs,
# _newest_watched_commit_epoch, _unit_start_epoch, restart_unit). These tests
# drive staleness_pass() through the REAL helpers against a single injected
# fake subprocess.run — extending the report()-tests' argv-shape-dispatch
# pattern (below) with is-enabled, the monotonic elapsed-secs show call,
# restart_unit's stop/reset-failed/start sequence, and systemd-cat (log()) —
# with wdog.time.time and wdog.time.clock_gettime controlled. This is the
# strictly-higher integration level the δ gate calls for: it validates that
# the real helpers' parsing composes correctly with staleness_pass's decision
# logic, not just that the decision logic is correct given pre-decided inputs.
# ---------------------------------------------------------------------------

# Fixed CLOCK_MONOTONIC "now" used by _fleet_fake_run below to realize a
# desired per-unit elapsed-secs via the paired ExecMainStartTimestampMonotonic
# value — callers must monkeypatch wdog.time.clock_gettime to this constant.
_E2E_CLOCK_MONOTONIC_NOW = 1_000_000.0


def _fleet_fake_run(
    *,
    units: list[str],
    commit_epoch: int,
    start_epochs: dict[str, int],
    recorded_calls: list[list[str]],
    log_messages: list[str],
    enabled: dict[str, bool] | None = None,
    elapsed_secs: dict[str, float] | None = None,
):
    """Build a fake subprocess.run for staleness_pass() end-to-end tests.

    Dispatches on argv shape (extends the report()-tests' dispatcher below
    with the mutating/log calls staleness_pass can also issue): list-units /
    is-enabled / the two distinct ``systemctl show`` calls (monotonic
    elapsed-secs, realtime start epoch) / git log / restart_unit's
    stop-reset-failed-start sequence (used only by main(), retained here for
    any test that also exercises liveness) / the systemd-run fleet-restart
    delegation (task 2396) / systemd-cat (log()). Unhandled argv shapes fail
    the test outright rather than returning a default result, so a change to
    the real helpers' argv is caught here instead of silently driving
    staleness_pass() off a wrong assumption.

    ``enabled`` defaults every unit to enabled (True); ``elapsed_secs``
    defaults every unit to 300.0s (past STARTUP_GRACE_SECS=120).
    """
    enabled = {} if enabled is None else enabled
    elapsed_secs = {} if elapsed_secs is None else elapsed_secs

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        recorded_calls.append(list(cmd))
        if cmd[:3] == ["systemctl", "--user", "list-units"]:
            stdout = "".join(f"{u} loaded active running desc\n" for u in units)
            return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")
        if cmd[:3] == ["systemctl", "--user", "is-enabled"]:
            unit = cmd[-1]
            rc = 0 if enabled.get(unit, True) else 1
            return subprocess.CompletedProcess(cmd, rc, stdout="", stderr="")
        if cmd[:3] == ["systemctl", "--user", "show"] and (
            "--property=ExecMainStartTimestampMonotonic" in cmd
        ):
            unit = cmd[3]
            secs = elapsed_secs.get(unit, 300.0)
            start_mono_us = int((_E2E_CLOCK_MONOTONIC_NOW - secs) * 1_000_000)
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=f"ExecMainStartTimestampMonotonic={start_mono_us}\n",
                stderr="",
            )
        if cmd[:3] == ["systemctl", "--user", "show"] and "--timestamp=unix" in cmd:
            unit = cmd[3]
            return subprocess.CompletedProcess(
                cmd, 0, stdout=f"ExecMainStartTimestamp=@{start_epochs[unit]}\n", stderr=""
            )
        if cmd[0] == "git":
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{commit_epoch}\n", stderr="")
        if cmd[:3] in (
            ["systemctl", "--user", "stop"],
            ["systemctl", "--user", "reset-failed"],
            ["systemctl", "--user", "start"],
        ):
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[0] == "systemd-run":
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[0] == "systemd-cat":
            log_messages.append(str(kwargs.get("input", "")))
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        pytest.fail(f"unexpected subprocess.run call inside staleness_pass() e2e: {cmd}")

    return fake_run


def test_staleness_pass_e2e_restarts_stale_unit_then_converges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scenario 1 (I6): end-to-end, staleness_pass() delegates a fleet restart
    for a unit whose real start epoch predates the newest watched commit,
    logs a WARNING naming it, and self-clears on the very next pass once the
    unit reads fresh again.

    Unlike test_staleness_pass_core (which stubs every helper), this drives
    _enumerate_running_units, is_unit_enabled, _unit_start_elapsed_secs,
    _newest_watched_commit_epoch, and _unit_start_epoch all through ONE
    injected fake subprocess.run — the integration level above α's
    helper-stubbed unit test. The delegated systemd-run call itself is also
    driven through the same fake (task 2396 step-11), rather than stubbing
    _delegate_fleet_restart, so this test additionally pins the real argv
    _delegate_fleet_restart builds.
    """
    wdog = _load_watchdog()

    now = 2_000_000_000.0
    commit_epoch = int(now) - wdog.STALENESS_GRACE_SECS - 100  # older than grace

    unit = "orchestrator-know-live.service"
    start_epochs = {unit: commit_epoch - 100}  # started before the commit -> stale

    recorded_calls: list[list[str]] = []
    log_messages: list[str] = []
    fake_run = _fleet_fake_run(
        units=[unit],
        commit_epoch=commit_epoch,
        start_epochs=start_epochs,
        recorded_calls=recorded_calls,
        log_messages=log_messages,
    )

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(wdog, "_within_fleet_deploy_min_interval", lambda: False)
    monkeypatch.setattr(wdog.time, "time", lambda: now)
    monkeypatch.setattr(wdog.time, "clock_gettime", lambda _clk_id: _E2E_CLOCK_MONOTONIC_NOW)

    wdog.staleness_pass()

    delegate_calls = [c for c in recorded_calls if c[0] == "systemd-run"]
    assert len(delegate_calls) == 1, (
        f"Expected exactly one systemd-run delegation for {unit}; got {delegate_calls}"
    )
    argv = delegate_calls[0]
    assert any(a.endswith("scripts/restart-all-orchestrators.sh") for a in argv), (
        f"Delegated systemd-run argv must invoke restart-all-orchestrators.sh: {argv}"
    )
    assert "--drain" in argv, f"Delegated systemd-run argv must pass --drain: {argv}"
    assert any(("WARNING" in m and unit in m) for m in log_messages), (
        f"Expected a WARNING log line naming {unit}: {log_messages}"
    )

    # --- I6 convergence: a real restart would refresh the unit's start
    # epoch, so flip it to newer-than-commit and run staleness_pass() again —
    # no further delegation must be issued (stateless self-clear).
    recorded_calls.clear()
    start_epochs[unit] = commit_epoch + 50

    wdog.staleness_pass()

    delegate_calls_2 = [c for c in recorded_calls if c[0] == "systemd-run"]
    assert delegate_calls_2 == [], (
        f"staleness_pass must self-clear once {unit} reads fresh; got {delegate_calls_2}"
    )


def test_staleness_pass_e2e_commit_grace_suppresses_all_restarts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scenario 2 (I5, end-to-end): a commit younger than STALENESS_GRACE_SECS
    performs zero mutating systemctl calls and never even enumerates units,
    even though the running unit's real start epoch predates the commit.
    """
    wdog = _load_watchdog()

    now = 2_000_000_000.0
    commit_epoch = int(now) - 300  # younger than STALENESS_GRACE_SECS=1800

    unit = "orchestrator-know-live.service"
    start_epochs = {unit: commit_epoch - 100}  # would be stale but for the grace gate

    recorded_calls: list[list[str]] = []
    log_messages: list[str] = []
    fake_run = _fleet_fake_run(
        units=[unit],
        commit_epoch=commit_epoch,
        start_epochs=start_epochs,
        recorded_calls=recorded_calls,
        log_messages=log_messages,
    )

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(wdog, "_within_fleet_deploy_min_interval", lambda: False)
    monkeypatch.setattr(wdog.time, "time", lambda: now)
    monkeypatch.setattr(wdog.time, "clock_gettime", lambda _clk_id: _E2E_CLOCK_MONOTONIC_NOW)

    wdog.staleness_pass()

    assert not any(c[:3] == ["systemctl", "--user", "list-units"] for c in recorded_calls), (
        f"commit-grace gate must return before enumerating units; got {recorded_calls}"
    )
    _assert_zero_mutating_calls(recorded_calls)


def test_staleness_pass_e2e_fresh_unit_not_restarted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scenario 3 (I5, end-to-end): a unit whose real start epoch is newer than
    the newest watched commit performs zero mutating systemctl calls and zero
    fleet-restart delegations.
    """
    wdog = _load_watchdog()

    now = 2_000_000_000.0
    commit_epoch = int(now) - wdog.STALENESS_GRACE_SECS - 100  # older than grace

    unit = "orchestrator-know-live.service"
    start_epochs = {unit: commit_epoch + 100}  # fresh: started after the commit

    recorded_calls: list[list[str]] = []
    log_messages: list[str] = []
    fake_run = _fleet_fake_run(
        units=[unit],
        commit_epoch=commit_epoch,
        start_epochs=start_epochs,
        recorded_calls=recorded_calls,
        log_messages=log_messages,
    )

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(wdog, "_within_fleet_deploy_min_interval", lambda: False)
    monkeypatch.setattr(wdog.time, "time", lambda: now)
    monkeypatch.setattr(wdog.time, "clock_gettime", lambda _clk_id: _E2E_CLOCK_MONOTONIC_NOW)

    wdog.staleness_pass()

    assert recorded_calls, "fresh-unit scenario must still drive real subprocess calls"
    _assert_zero_mutating_calls(recorded_calls)
    assert not any(c[0] == "systemd-run" for c in recorded_calls), (
        f"A fresh unit must not trigger a fleet-restart delegation; got {recorded_calls}"
    )


def test_staleness_pass_e2e_disabled_unit_not_restarted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scenario 4 (I5, end-to-end): a disabled unit that is otherwise
    stale-beyond-grace performs zero mutating systemctl calls and zero
    fleet-restart delegations — operator intent (is-enabled) is respected
    before the staleness comparison ever runs against that unit.
    """
    wdog = _load_watchdog()

    now = 2_000_000_000.0
    commit_epoch = int(now) - wdog.STALENESS_GRACE_SECS - 100  # older than grace

    unit = "orchestrator-know-live.service"
    start_epochs = {unit: commit_epoch - 100}  # stale-beyond-grace, but for is-enabled

    recorded_calls: list[list[str]] = []
    log_messages: list[str] = []
    fake_run = _fleet_fake_run(
        units=[unit],
        commit_epoch=commit_epoch,
        start_epochs=start_epochs,
        recorded_calls=recorded_calls,
        log_messages=log_messages,
        enabled={unit: False},
    )

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(wdog, "_within_fleet_deploy_min_interval", lambda: False)
    monkeypatch.setattr(wdog.time, "time", lambda: now)
    monkeypatch.setattr(wdog.time, "clock_gettime", lambda _clk_id: _E2E_CLOCK_MONOTONIC_NOW)

    wdog.staleness_pass()

    _assert_zero_mutating_calls(recorded_calls)
    assert not any(c[0] == "systemd-run" for c in recorded_calls), (
        f"A disabled unit must not trigger a fleet-restart delegation; got {recorded_calls}"
    )

    # Positive confirmation that the DISABLED gate — not empty enumeration or
    # an early commit-grace return — is what suppressed the restart: an
    # is-enabled call must have been recorded for the unit, ...
    is_enabled_calls = [
        c
        for c in recorded_calls
        if c[:3] == ["systemctl", "--user", "is-enabled"] and c[-1] == unit
    ]
    assert is_enabled_calls, (
        f"Expected an is-enabled call recorded for {unit}, proving the disabled "
        f"gate actually evaluated this unit; recorded calls: {recorded_calls}"
    )
    # ...and no staleness-comparison show call (monotonic elapsed or realtime
    # start epoch) may have been made against it, proving is-enabled=False
    # short-circuited BEFORE the staleness comparison ever ran.
    staleness_show_calls = [
        c
        for c in recorded_calls
        if c[:3] == ["systemctl", "--user", "show"]
        and c[3] == unit
        and (
            "--property=ExecMainStartTimestampMonotonic" in c
            or "--timestamp=unix" in c
        )
    ]
    assert staleness_show_calls == [], (
        f"is-enabled=False must short-circuit before any staleness-comparison "
        f"show call for {unit}; got {staleness_show_calls}"
    )


def _assert_zero_mutating_calls(recorded_calls: list[list[str]]) -> None:
    """Assert no recorded argv contains a mutating systemctl verb.

    Mirrors test_report_mixed_fleet_returns_1_and_lists_all_units's
    mutating-verbs check (report()'s I7 read-only guarantee) — reused here
    for staleness_pass()'s restraint gates (I5).
    """
    mutating_verbs = {"stop", "start", "restart", "reset-failed"}
    for call in recorded_calls:
        for token in call:
            assert token not in mutating_verbs, (
                f"staleness_pass() must perform zero mutating systemctl calls in this "
                f"restraint scenario; saw {call}"
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


def test_report_includes_deploy_age_column(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: pathlib.Path,
) -> None:
    """report() gains a DEPLOY-AGE column: the fleet-wide age since the
    shared fleet-deploy clock (task 2396 β), rendered in hours to one
    decimal and repeated on every row — like the existing NEWEST WATCHED
    COMMIT column. Pre-existing UNIT/START/NEWEST WATCHED COMMIT/VERDICT
    columns must still be present. Fails today: report() has no DEPLOY-AGE
    column yet.
    """
    wdog = _load_watchdog()

    commit_epoch = 1_800_000_000
    unit = "orchestrator-unit0.service"
    start_epoch = commit_epoch + 100  # fresh

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        if cmd[:3] == ["systemctl", "--user", "list-units"]:
            return subprocess.CompletedProcess(
                cmd, 0, stdout=f"{unit} loaded active running desc\n", stderr=""
            )
        if cmd[:3] == ["systemctl", "--user", "show"]:
            return subprocess.CompletedProcess(
                cmd, 0, stdout=f"ExecMainStartTimestamp=@{start_epoch}\n", stderr=""
            )
        if cmd[0] == "git":
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{commit_epoch}\n", stderr="")
        pytest.fail(f"unexpected subprocess.run call inside report(): {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    now = 2_000_000_000.0
    clock_file = tmp_path / "clock.json"
    clock_file.write_text(
        f'{{"ts": {now - 3 * 3600}, "iso": "2026-07-15T21:00:00+00:00"}}'
    )
    monkeypatch.setattr(wdog, "FLEET_DEPLOY_CLOCK_PATH", str(clock_file))
    monkeypatch.setattr(wdog.time, "time", lambda: now)

    wdog.report()

    captured = capsys.readouterr()
    header_line = next(line for line in captured.out.splitlines() if line.startswith("UNIT"))
    for col in ("UNIT", "START", "NEWEST WATCHED COMMIT", "VERDICT", "DEPLOY-AGE"):
        assert col in header_line, f"expected column {col!r} in header: {header_line!r}"

    unit_line = next(line for line in captured.out.splitlines() if line.startswith(unit))
    assert "3.0h" in unit_line, f"expected DEPLOY-AGE ~3.0h in row: {unit_line!r}"


def test_report_deploy_age_unknown_when_clock_absent(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: pathlib.Path,
) -> None:
    """DEPLOY-AGE renders 'unknown' when the shared fleet-deploy clock file
    is absent (no fleet deploy has ever verified fresh, or a fresh checkout
    with no data/ yet) — mirrors _read_last_fleet_deploy_epoch's fail-open
    contract. Fails today: report() has no DEPLOY-AGE column yet.
    """
    wdog = _load_watchdog()

    commit_epoch = 1_800_000_000
    unit = "orchestrator-unit0.service"
    start_epoch = commit_epoch + 100  # fresh

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        if cmd[:3] == ["systemctl", "--user", "list-units"]:
            return subprocess.CompletedProcess(
                cmd, 0, stdout=f"{unit} loaded active running desc\n", stderr=""
            )
        if cmd[:3] == ["systemctl", "--user", "show"]:
            return subprocess.CompletedProcess(
                cmd, 0, stdout=f"ExecMainStartTimestamp=@{start_epoch}\n", stderr=""
            )
        if cmd[0] == "git":
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{commit_epoch}\n", stderr="")
        pytest.fail(f"unexpected subprocess.run call inside report(): {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(wdog, "FLEET_DEPLOY_CLOCK_PATH", str(tmp_path / "absent.json"))
    monkeypatch.setattr(wdog.time, "time", lambda: 2_000_000_000.0)

    wdog.report()

    captured = capsys.readouterr()
    unit_line = next(line for line in captured.out.splitlines() if line.startswith(unit))
    assert "unknown" in unit_line, f"expected DEPLOY-AGE 'unknown' in row: {unit_line!r}"


def test_report_includes_merge_idle_and_would_defer_columns(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: pathlib.Path,
) -> None:
    """report() gains MERGE-IDLE (idle/busy/stale/absent, matching
    drain_check.classify exactly) and WOULD-DEFER (yes iff MERGE-IDLE is
    'busy') columns, populated from real per-unit heartbeat files under a
    tmp ORCH_FLEET_DIR. Fails today: report() has neither column yet.

    All four units share the same start/commit relationship (fresh) so the
    pre-existing VERDICT column can't be confused with the new MERGE-IDLE
    values in this assertion; row values are extracted positionally
    (the last four whitespace-separated tokens are VERDICT, DEPLOY-AGE,
    MERGE-IDLE, WOULD-DEFER, in that order) so the START/NEWEST WATCHED
    COMMIT timestamp columns' embedded spaces can't misalign a naive split.
    """
    wdog = _load_watchdog()

    commit_epoch = 1_800_000_000
    now = 2_000_000_000.0
    # unit->(merge_idle, ts_epoch) heartbeat fixture, or None for "no file".
    units = [
        "orchestrator-alpha.service",  # idle: fresh + merge_idle=True
        "orchestrator-bravo.service",  # busy: fresh + merge_idle=False
        "orchestrator-charlie.service",  # stale: ts_epoch far outside the fresh window
        "orchestrator-delta.service",  # absent: no heartbeat file at all
    ]
    start_epochs = {u: commit_epoch + 100 for u in units}  # all fresh vs. commit

    def fake_run(cmd, **kwargs):  # noqa: ANN001
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
    monkeypatch.setattr(wdog.time, "time", lambda: now)
    # Keep DEPLOY-AGE deterministically 'unknown' — irrelevant to this test.
    monkeypatch.setattr(wdog, "FLEET_DEPLOY_CLOCK_PATH", str(tmp_path / "absent_clock.json"))

    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir()
    monkeypatch.setenv("ORCH_FLEET_DIR", str(fleet_dir))

    def _write_heartbeat(unit: str, *, merge_idle: bool, ts_epoch: float) -> None:
        (fleet_dir / f"{unit}.json").write_text(
            json.dumps(
                {
                    "unit": unit,
                    "merge_idle": merge_idle,
                    "depth": 0,
                    "queue_empty": merge_idle,
                    "ts_epoch": ts_epoch,
                }
            )
        )

    _write_heartbeat(units[0], merge_idle=True, ts_epoch=now - 10)
    _write_heartbeat(units[1], merge_idle=False, ts_epoch=now - 10)
    _write_heartbeat(units[2], merge_idle=True, ts_epoch=now - 100_000)
    # units[3] ("delta"/absent): deliberately no heartbeat file written.

    wdog.report()

    captured = capsys.readouterr()
    header_line = next(line for line in captured.out.splitlines() if line.startswith("UNIT"))
    assert "MERGE-IDLE" in header_line, f"expected MERGE-IDLE in header: {header_line!r}"
    assert "WOULD-DEFER" in header_line, f"expected WOULD-DEFER in header: {header_line!r}"

    expected_merge = {
        units[0]: "idle",
        units[1]: "busy",
        units[2]: "stale",
        units[3]: "absent",
    }
    expected_defer = {
        units[0]: "no",
        units[1]: "yes",
        units[2]: "no",
        units[3]: "no",
    }
    for unit in units:
        line = next(entry for entry in captured.out.splitlines() if entry.startswith(unit))
        tokens = line.split()
        _verdict_tok, _deploy_age_tok, merge_tok, defer_tok = tokens[-4:]
        assert merge_tok == expected_merge[unit], (
            f"{unit}: expected MERGE-IDLE={expected_merge[unit]!r}, got {merge_tok!r} "
            f"in line {line!r}"
        )
        assert defer_tok == expected_defer[unit], (
            f"{unit}: expected WOULD-DEFER={expected_defer[unit]!r}, got {defer_tok!r} "
            f"in line {line!r}"
        )


def test_report_merge_idle_degrades_to_unknown_when_drain_check_raises(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: pathlib.Path,
) -> None:
    """_classify_unit_heartbeat's fail-soft branch (scripts/orchestrator-
    watchdog.py): when the lazy drain_check reuse path raises for any reason
    (here, drain_check.resolve_fleet_dir() itself blowing up), report() must
    still complete rather than crash, render MERGE-IDLE='unknown' with
    WOULD-DEFER='no' for the affected row (never silently drop the row,
    never guess a verdict), and log a WARNING naming the swallowed
    exception -- honoring the loud-over-silent-degradation norm.

    Regression lock for the except-block in _classify_unit_heartbeat, which
    the four real-heartbeat-file cases above (idle/busy/stale/absent) never
    exercise -- a future refactor that turned the swallow into a crash, or
    that dropped the WARNING, would otherwise pass CI.
    """
    wdog = _load_watchdog()
    import drain_check

    commit_epoch = 1_800_000_000
    now = 2_000_000_000.0
    unit = "orchestrator-foxtrot.service"
    start_epoch = commit_epoch + 100  # fresh vs. commit -- irrelevant to this test

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        if cmd[:3] == ["systemctl", "--user", "list-units"]:
            return subprocess.CompletedProcess(
                cmd, 0, stdout=f"{unit} loaded active running desc\n", stderr=""
            )
        if cmd[:3] == ["systemctl", "--user", "show"]:
            return subprocess.CompletedProcess(
                cmd, 0, stdout=f"ExecMainStartTimestamp=@{start_epoch}\n", stderr=""
            )
        if cmd[0] == "git":
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{commit_epoch}\n", stderr="")
        pytest.fail(f"unexpected subprocess.run call inside report(): {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(wdog.time, "time", lambda: now)
    monkeypatch.setattr(wdog, "FLEET_DEPLOY_CLOCK_PATH", str(tmp_path / "absent_clock.json"))

    def _boom():
        raise OSError("simulated: fleet dir unreadable")

    monkeypatch.setattr(drain_check, "resolve_fleet_dir", _boom)

    logged: list[str] = []
    monkeypatch.setattr(wdog, "log", lambda msg: logged.append(msg))

    exit_code = wdog.report()

    assert exit_code == 0
    captured = capsys.readouterr()
    line = next(entry for entry in captured.out.splitlines() if entry.startswith(unit))
    tokens = line.split()
    _verdict_tok, _deploy_age_tok, merge_tok, defer_tok = tokens[-4:]
    assert merge_tok == "unknown", (
        f"expected MERGE-IDLE='unknown' when drain_check raises, got {merge_tok!r} "
        f"in line {line!r}"
    )
    assert defer_tok == "no", (
        f"expected WOULD-DEFER='no' when MERGE-IDLE is unknown, got {defer_tok!r} "
        f"in line {line!r}"
    )
    assert any(
        "WARNING" in msg and "_classify_unit_heartbeat" in msg for msg in logged
    ), f"expected a WARNING naming _classify_unit_heartbeat's swallowed exception, got {logged!r}"


def test_report_extended_columns_stay_read_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """I8 guard: report(), now populating DEPLOY-AGE/MERGE-IDLE/WOULD-DEFER,
    still performs zero mutating systemctl calls and never writes the shared
    fleet-deploy clock file.

    Pre-seeds BOTH the clock file (with a sentinel payload, captured
    byte-for-byte) and a per-unit heartbeat under a tmp ORCH_FLEET_DIR — so
    the new columns are actually populated from real reads, not the
    trivially-true empty/absent path — before driving report() through its
    real helpers with a recording fake subprocess.run. Regression lock for
    the read-only contract, independent of the mixed-fleet acceptance test
    (scenario 9) added later.
    """
    wdog = _load_watchdog()

    commit_epoch = 1_800_000_000
    now = 2_000_000_000.0
    unit = "orchestrator-echo.service"
    start_epoch = commit_epoch + 100  # fresh

    recorded_calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        recorded_calls.append(list(cmd))
        if cmd[:3] == ["systemctl", "--user", "list-units"]:
            return subprocess.CompletedProcess(
                cmd, 0, stdout=f"{unit} loaded active running desc\n", stderr=""
            )
        if cmd[:3] == ["systemctl", "--user", "show"]:
            return subprocess.CompletedProcess(
                cmd, 0, stdout=f"ExecMainStartTimestamp=@{start_epoch}\n", stderr=""
            )
        if cmd[0] == "git":
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{commit_epoch}\n", stderr="")
        pytest.fail(f"unexpected subprocess.run call inside report(): {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(wdog.time, "time", lambda: now)
    monkeypatch.setattr(
        wdog, "restart_unit", lambda u: pytest.fail(f"report() must never restart {u}")
    )

    clock_file = tmp_path / "clock.json"
    clock_payload = '{"ts": 1783000000.0, "iso": "2026-07-16T00:00:00+00:00"}'
    clock_file.write_text(clock_payload)
    monkeypatch.setattr(wdog, "FLEET_DEPLOY_CLOCK_PATH", str(clock_file))

    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir()
    monkeypatch.setenv("ORCH_FLEET_DIR", str(fleet_dir))
    (fleet_dir / f"{unit}.json").write_text(
        json.dumps(
            {
                "unit": unit,
                "merge_idle": True,
                "depth": 0,
                "queue_empty": True,
                "ts_epoch": now - 10,
            }
        )
    )

    exit_code = wdog.report()

    assert exit_code == 0
    assert recorded_calls, "report() must have driven subprocess.run for this test to mean anything"
    _assert_zero_mutating_calls(recorded_calls)
    assert clock_file.read_text() == clock_payload, (
        "report() must never write the shared fleet-deploy clock file"
    )


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
    # No-op the fused-memory liveness row (B4) so this test drives no real
    # ss/urllib I/O. raising=False: tolerates the attribute not existing yet
    # (pre-B4-impl) so this test can be written before the impl lands.
    monkeypatch.setattr(wdog, "_print_fused_memory_liveness", lambda: None, raising=False)

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
    # No-op the fused-memory liveness row (B4) — see comment in
    # test_cli_report_flag_routes_to_report_only above.
    monkeypatch.setattr(wdog, "_print_fused_memory_liveness", lambda: None, raising=False)

    assert wdog._cli(["--report"]) == 1


def test_cli_default_runs_main_then_staleness_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """_cli([]) runs main(), fused_memory_liveness_pass(), staleness_pass(), then
    fused_memory_staleness_pass() (the fm staleness backstop, task 2714), never report()."""
    wdog = _load_watchdog()
    calls: list[str] = []

    monkeypatch.setattr(wdog, "report", lambda: calls.append("report") or 0)
    monkeypatch.setattr(wdog, "main", lambda: calls.append("main"))
    monkeypatch.setattr(
        wdog, "fused_memory_liveness_pass", lambda: calls.append("fused_memory_liveness_pass")
    )
    monkeypatch.setattr(wdog, "staleness_pass", lambda: calls.append("staleness_pass"))
    monkeypatch.setattr(
        wdog, "fused_memory_staleness_pass", lambda: calls.append("fused_memory_staleness_pass")
    )

    wdog._cli([])

    assert calls == [
        "main",
        "fused_memory_liveness_pass",
        "staleness_pass",
        "fused_memory_staleness_pass",
    ], f"Expected liveness passes then both staleness passes (fm last), got {calls}"


def test_cli_unknown_flag_does_not_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    """_cli(["--bogus"]) must not raise; unknown flags fall through to the timer path."""
    wdog = _load_watchdog()
    calls: list[str] = []

    monkeypatch.setattr(wdog, "report", lambda: calls.append("report") or 0)
    monkeypatch.setattr(wdog, "main", lambda: calls.append("main"))
    monkeypatch.setattr(
        wdog, "fused_memory_liveness_pass", lambda: calls.append("fused_memory_liveness_pass")
    )
    monkeypatch.setattr(wdog, "staleness_pass", lambda: calls.append("staleness_pass"))
    monkeypatch.setattr(
        wdog, "fused_memory_staleness_pass", lambda: calls.append("fused_memory_staleness_pass")
    )

    # Must not raise
    wdog._cli(["--bogus"])

    assert calls == [
        "main",
        "fused_memory_liveness_pass",
        "staleness_pass",
        "fused_memory_staleness_pass",
    ]


def test_cli_report_does_not_run_fm_staleness_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """--report must NOT invoke fused_memory_staleness_pass() (I7 at the CLI
    boundary): the fm staleness backstop runs on the timer path only, never in
    the read-only doctor mode."""
    wdog = _load_watchdog()

    monkeypatch.setattr(wdog, "report", lambda: 0)
    monkeypatch.setattr(wdog, "_print_fused_memory_liveness", lambda: None, raising=False)
    monkeypatch.setattr(
        wdog,
        "fused_memory_staleness_pass",
        lambda: pytest.fail("fused_memory_staleness_pass() must not run under --report"),
    )

    assert wdog._cli(["--report"]) == 0


def test_cli_defaults_to_sys_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    """_cli() with no argv argument reads sys.argv[1:]."""
    wdog = _load_watchdog()
    calls: list[str] = []

    monkeypatch.setattr(wdog, "report", lambda: calls.append("report") or 0)
    monkeypatch.setattr(wdog.sys, "argv", ["orchestrator-watchdog.py", "--report"])
    # No-op the fused-memory liveness row (B4) — see comment in
    # test_cli_report_flag_routes_to_report_only above.
    monkeypatch.setattr(wdog, "_print_fused_memory_liveness", lambda: None, raising=False)

    exit_code = wdog._cli()

    assert calls == ["report"]
    assert exit_code == 0


# ---------------------------------------------------------------------------
# Boundary-scenario acceptance gate (task 2399, ε of
# plans/orchestrator-fleet-redeploy-throughput-prd.md's §Boundary-test-sketch,
# scenarios 1-10). Ties invariants I1-I9 to end-to-end behavior across the
# now-landed α (fleet_heartbeat.py) / β (shared fleet-deploy clock) / γ
# (drain gate) / δ (coordinator fire-while-busy) work.
#
# NOT to be confused with the "staleness_pass END-TO-END tests (δ task 2027,
# scenarios 1-4)" block earlier in this file -- that numbering belongs to the
# OLDER, now-superseded plans/orchestrator-fleet-staleness-prd.md. Every test
# below is prefixed test_boundaryN_ (N = THIS PRD's own 1-10 numbering) to
# keep the two schemes unambiguous.
#
# Per-contract UNIT coverage already exists in the α/β/γ/δ suites (this
# module's own β-adjacent tests above, scripts/tests/test_restart_all_
# orchestrators.py, orchestrator/tests/test_fleet_staleness_composition.py,
# orchestrator/tests/test_merge_queue_restart_hook.py) -- this section adds
# the acceptance-level restatement the PRD calls for, not duplicate unit
# tests.
# ---------------------------------------------------------------------------


def test_boundary1_staleness_inside_window_real_clock_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """Scenario 1 (I1) -- staleness inside the 8h window, via a REAL on-disk
    fleet-deploy clock file (not a monkeypatched _within_fleet_deploy_min_
    interval -- see test_staleness_pass_skips_when_within_fleet_deploy_min_
    interval above for that unit-level variant). Exercises the real
    _read_last_fleet_deploy_epoch() file-read + _within_fleet_deploy_min_
    interval() comparison end-to-end, with a would-be-stale unit present in
    the fleet, via the _fleet_fake_run harness.

    Asserts neither tier acts (zero mutating systemctl calls AND zero
    systemd-run fleet-restart delegation) and that a "skip: within
    fleet-deploy min-interval" journal line naming
    ORCH_RESTART_MIN_INTERVAL_SECS is logged -- "now" is pinned to a
    SKIP_LOG_INTERVAL_SECS bucket boundary so the skip line is guaranteed
    (not suppressed by the log-rate-limit throttle exercised by
    test_staleness_pass_suppresses_skip_log_outside_log_bucket above).
    """
    wdog = _load_watchdog()

    now = wdog.SKIP_LOG_INTERVAL_SECS * 1000.0
    clock_ts = now - 2 * 3600  # ~2h ago -- well inside the 28800s default window
    clock_file = tmp_path / "clock.json"
    clock_file.write_text(json.dumps({"ts": clock_ts, "iso": "2026-07-16T00:00:00+00:00"}))
    monkeypatch.setattr(wdog, "FLEET_DEPLOY_CLOCK_PATH", str(clock_file))

    commit_epoch = int(now) - wdog.STALENESS_GRACE_SECS - 100  # older than grace, if ever reached
    unit = "orchestrator-know-live.service"
    start_epochs = {unit: commit_epoch - 100}  # would-be-stale if the gate did not close first

    recorded_calls: list[list[str]] = []
    log_messages: list[str] = []
    fake_run = _fleet_fake_run(
        units=[unit],
        commit_epoch=commit_epoch,
        start_epochs=start_epochs,
        recorded_calls=recorded_calls,
        log_messages=log_messages,
    )

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(wdog.time, "time", lambda: now)
    monkeypatch.setattr(wdog.time, "clock_gettime", lambda _clk_id: _E2E_CLOCK_MONOTONIC_NOW)

    wdog.staleness_pass()

    assert not any(c[:3] == ["systemctl", "--user", "list-units"] for c in recorded_calls), (
        f"the real fleet-deploy clock gate must close before enumeration; got {recorded_calls}"
    )
    _assert_zero_mutating_calls(recorded_calls)
    assert not any(c[0] == "systemd-run" for c in recorded_calls), (
        f"must not delegate a fleet restart while inside the real clock's "
        f"min-interval window; got {recorded_calls}"
    )
    assert any(
        "skip" in m and str(wdog.ORCH_RESTART_MIN_INTERVAL_SECS) in m for m in log_messages
    ), f"Expected a skip log line naming the fleet-deploy min-interval: {log_messages}"


# ---------------------------------------------------------------------------
# Boundary scenarios 2-6 (I1/I2/I3/I4/I6): drive the REAL
# scripts/restart-all-orchestrators.sh --drain via subprocess, against a
# local fake multi-unit `systemctl` on PATH + crafted heartbeats.
#
# Reimplements -- rather than imports across test directories, which isn't
# clean -- the proven harnesses in tests/scripts/test_restart_all_
# orchestrators.py (fake-binary-on-PATH + clock-stamp pattern) and
# scripts/tests/test_restart_all_orchestrators.py (multi-unit fake systemctl
# with per-unit state + the drain env-knob techniques: force-fire=0 for
# force, unknown-grace=0 for absent, a bounded subprocess timeout for defer).
# ---------------------------------------------------------------------------

RESTART_ALL_SCRIPT = REPO_ROOT / "scripts" / "restart-all-orchestrators.sh"

# Stateful fake `systemctl`: `list-units` reports every entry in
# running_units; `show -p FIELDS UNIT` and `restart UNIT` operate on
# state["units"][UNIT], keyed by a "scenario" ("fresh" advances MainPID/
# ActiveState/ActiveEnterTimestampMonotonic on restart, simulating a verified
# restart; "stale" -- the default -- never advances, simulating a restart
# that never came back up fresh; "delayed-fresh" (task 2967) reports stale
# for the first `fresh_after` post-restart `show` calls, then flips fresh --
# simulating a slow-draining unit that only verifies during the
# VERIFY_TIMEOUT grace re-probe, task 2961). Every call is recorded into
# state["calls"] for assertions. Verbatim reimplementation of
# scripts/tests/test_restart_all_orchestrators.py's FAKE_SYSTEMCTL_SRC.
_BOUNDARY_FAKE_SYSTEMCTL_SRC = '''#!/usr/bin/env python3
"""Fake multi-unit `systemctl` for ε's --drain boundary scenarios."""
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
    state.setdefault("calls", []).append(argv[1:])

    if verb == "list-units":
        for unit in state.get("running_units", []):
            print(f"{unit} loaded active running Orchestrator")
        _save(state)
        return 0

    if verb == "restart":
        unit = rest[0] if rest else ""
        units = state.setdefault("units", {})
        ustate = units.setdefault(unit, {})
        scenario = ustate.get("scenario", "stale")
        if scenario == "fresh":
            ustate["MainPID"] = ustate.get("MainPID", 1000) + 1
            ustate["ActiveState"] = "active"
            ustate["ActiveEnterTimestampMonotonic"] = (
                ustate.get("ActiveEnterTimestampMonotonic", 0) + 5_000_000
            )
            ustate["ActiveEnterTimestamp"] = "restarted"
        elif scenario == "delayed-fresh":
            ustate["restarted"] = True
            ustate["post_restart_shows"] = 0
        _save(state)
        return 0

    if verb == "show":
        fields = None
        unit = None
        i = 0
        while i < len(rest):
            tok = rest[i]
            if tok == "-p":
                fields = rest[i + 1]
                i += 2
            elif tok.startswith("--property="):
                fields = tok.split("=", 1)[1]
                i += 1
            elif tok.startswith("-"):
                i += 1
            else:
                unit = tok
                i += 1
        ustate = state.get("units", {}).get(unit, {})
        if ustate.get("scenario") == "delayed-fresh" and ustate.get("restarted"):
            ustate["post_restart_shows"] = ustate.get("post_restart_shows", 0) + 1
            if ustate["post_restart_shows"] > ustate.get("fresh_after", 0):
                ustate["MainPID"] = ustate.get("MainPID", 1000) + 1
                ustate["ActiveState"] = "active"
                ustate["ActiveEnterTimestampMonotonic"] = (
                    ustate.get("ActiveEnterTimestampMonotonic", 0) + 5_000_000
                )
                ustate["ActiveEnterTimestamp"] = "restarted"
        current = {
            "MainPID": str(ustate.get("MainPID", 0)),
            "ActiveState": ustate.get("ActiveState", "active"),
            "ActiveEnterTimestamp": ustate.get("ActiveEnterTimestamp", "baseline"),
            "ActiveEnterTimestampMonotonic": str(ustate.get("ActiveEnterTimestampMonotonic", 0)),
        }
        keys = fields.split(",") if fields else list(current.keys())
        for k in keys:
            print(f"{k}={current.get(k, '')}")
        _save(state)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
'''


def _boundary_make_fake_systemctl(base_dir, *, running_units, units=None):
    """Write a fake multi-unit `systemctl` into <base_dir>/bin/.

    Returns (bin_dir, state_path). parents=True/exist_ok=True so callers may
    pass a not-yet-created base_dir (e.g. a fresh sub-scenario directory).
    """
    bin_dir = base_dir / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    fake = bin_dir / "systemctl"
    fake.write_text(_BOUNDARY_FAKE_SYSTEMCTL_SRC)
    fake.chmod(0o755)

    state_path = base_dir / "systemctl_state.json"
    state_path.write_text(json.dumps({
        "running_units": list(running_units),
        "units": units or {},
        "calls": [],
    }))
    return bin_dir, state_path


def _boundary_write_heartbeat(fleet_dir, unit, **overrides):
    """Write a heartbeat JSON matching fleet_heartbeat.py's on-disk contract."""
    fleet_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "unit": unit,
        "merge_idle": True,
        "depth": 0,
        "queue_empty": True,
        "ts_epoch": time.time(),
    }
    payload.update(overrides)
    (fleet_dir / f"{unit}.json").write_text(json.dumps(payload))


def _boundary_run_drain_script(
    bin_dir, state_path, fleet_dir, clock_file, *, env=None, timeout=20
):
    """Run the REAL restart-all-orchestrators.sh --drain with the fake
    systemctl prepended onto PATH."""
    full_env = dict(os.environ)
    full_env["PATH"] = f"{bin_dir}{os.pathsep}{full_env['PATH']}"
    full_env["FAKE_SYSTEMCTL_STATE"] = str(state_path)
    full_env["ORCH_FLEET_DIR"] = str(fleet_dir)
    full_env["ORCH_FLEET_DEPLOY_CLOCK"] = str(clock_file)
    if env:
        full_env.update(env)
    return subprocess.run(
        ["bash", str(RESTART_ALL_SCRIPT), "--drain"],
        env=full_env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _boundary_load_state(state_path):
    return json.loads(state_path.read_text())


def _boundary_decode(maybe_bytes):
    """subprocess.run's TimeoutExpired attaches partial output as bytes even
    when text=True was passed to the original call -- normalize."""
    if maybe_bytes is None:
        return ""
    if isinstance(maybe_bytes, bytes):
        return maybe_bytes.decode(errors="replace")
    return maybe_bytes


_UNIT_SUITE_FAKE_SYSTEMCTL_PATH = REPO_ROOT / "scripts" / "tests" / "test_restart_all_orchestrators.py"


def _fake_systemctl_functional_body(fake_systemctl_src: str) -> str:
    """Strip the module-level shebang/docstring preamble, returning the fake
    systemctl's actual behavior (from the first `import json` line onward).

    The two copies' module docstrings are intentionally different -- each is
    contextualized to its own test file -- so comparing from here down
    isolates what actually matters for drift (the verbs/fields the fake
    models) from that cosmetic difference.
    """
    marker = "\nimport json\n"
    idx = fake_systemctl_src.index(marker)
    return fake_systemctl_src[idx:]


def test_boundary_fake_systemctl_matches_unit_suite_verbatim() -> None:
    """DRIFT GUARD: _BOUNDARY_FAKE_SYSTEMCTL_SRC (this file's local
    reimplementation, used by scenarios 2-6 above) must stay in lockstep
    with scripts/tests/test_restart_all_orchestrators.py's
    FAKE_SYSTEMCTL_SRC, which it deliberately reimplements rather than
    imports across test directories (see the module comment above the
    boundary-harness block). If the real restart-all-orchestrators.sh
    starts querying a systemctl field neither fake models, or the unit
    suite's fake gains support for it and this module's copy doesn't, this
    test fails loudly instead of the two suites silently drifting apart.
    """
    unit_suite_src = _UNIT_SUITE_FAKE_SYSTEMCTL_PATH.read_text()
    match = re.search(r"^FAKE_SYSTEMCTL_SRC = '''(.*?)'''$", unit_suite_src, re.S | re.M)
    assert match, (
        f"could not locate FAKE_SYSTEMCTL_SRC in {_UNIT_SUITE_FAKE_SYSTEMCTL_PATH} "
        "-- has it been renamed or restructured?"
    )
    unit_suite_fake = match.group(1)

    assert _fake_systemctl_functional_body(unit_suite_fake) == _fake_systemctl_functional_body(
        _BOUNDARY_FAKE_SYSTEMCTL_SRC
    ), (
        "_BOUNDARY_FAKE_SYSTEMCTL_SRC has drifted from "
        f"{_UNIT_SUITE_FAKE_SYSTEMCTL_PATH}'s FAKE_SYSTEMCTL_SRC -- update this "
        "file's copy (or promote both to a shared fixture) to match."
    )


def test_boundary2_all_idle_restarts_and_stamps_clock(tmp_path: pathlib.Path) -> None:
    """Scenario 2 (I1/I2/I6) -- staleness past 8h, all idle: the REAL
    restart-all-orchestrators.sh --drain restarts every unit, verifies each
    fresh, and stamps the shared fleet-deploy clock afterward.

    Drives the actual bash script + drain_check.py end-to-end via subprocess
    against a local fake multi-unit `systemctl` on PATH, with >=2 running
    units, all heartbeats fresh + merge_idle=True (idle -> transparent, no
    defer/force line). The clock file does not exist beforehand -- the
    script's own stamp_fleet_deploy_clock is what creates it: a new
    integration combo (drain-idle + verify + stamp together) beyond the
    pre-existing per-contract suites, which test drain and stamping
    separately.
    """
    fleet_dir = tmp_path / "fleet"
    unit_a = "orchestrator-alpha.service"
    unit_b = "orchestrator-bravo.service"
    bin_dir, state_path = _boundary_make_fake_systemctl(
        tmp_path,
        running_units=[unit_a, unit_b],
        units={unit_a: {"scenario": "fresh"}, unit_b: {"scenario": "fresh"}},
    )
    _boundary_write_heartbeat(fleet_dir, unit_a, merge_idle=True, ts_epoch=time.time())
    _boundary_write_heartbeat(fleet_dir, unit_b, merge_idle=True, ts_epoch=time.time())

    clock_file = tmp_path / "clock.json"
    assert not clock_file.exists()

    result = _boundary_run_drain_script(
        bin_dir, state_path, fleet_dir, clock_file,
        env={"RESTART_VERIFY_TIMEOUT": "5"},
    )

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    state = _boundary_load_state(state_path)
    assert ["--user", "restart", unit_a] in state["calls"], state["calls"]
    assert ["--user", "restart", unit_b] in state["calls"], state["calls"]
    assert clock_file.exists(), "a fully-verified drain-aware restart must stamp the clock"
    stamped = json.loads(clock_file.read_text())
    assert isinstance(stamped["ts"], (int, float)), f"ts must be numeric; got {stamped!r}"


def test_boundary3_failed_verify_leaves_clock_unchanged(tmp_path: pathlib.Path) -> None:
    """Scenario 3 (I2 negative) -- stamp-on-verify only, WITH --drain: when
    one unit's fake systemctl never advances ActiveEnterTimestampMonotonic
    (never verifies fresh) after --drain restarts it, the script exits 1 and
    the shared fleet-deploy clock is left byte-identical -- a failed/partial
    verify must NOT stamp even under --drain, so a failed detached deploy
    can never silence the watchdog backstop for a full min-interval window.
    """
    fleet_dir = tmp_path / "fleet"
    unit_ok = "orchestrator-alpha.service"
    unit_bad = "orchestrator-bravo.service"
    bin_dir, state_path = _boundary_make_fake_systemctl(
        tmp_path,
        running_units=[unit_ok, unit_bad],
        units={unit_ok: {"scenario": "fresh"}, unit_bad: {"scenario": "stale"}},
    )
    _boundary_write_heartbeat(fleet_dir, unit_ok, merge_idle=True, ts_epoch=time.time())
    _boundary_write_heartbeat(fleet_dir, unit_bad, merge_idle=True, ts_epoch=time.time())

    clock_file = tmp_path / "clock.json"
    sentinel = '{"ts": 1.0}'
    clock_file.write_text(sentinel)

    result = _boundary_run_drain_script(
        bin_dir, state_path, fleet_dir, clock_file,
        # RESTART_VERIFY_GRACE_SECS (task 2961): the real script re-probes
        # for this many additional seconds past RESTART_VERIFY_TIMEOUT
        # before declaring a unit failed -- kept small here so a
        # genuinely-never-fresh unit still fails within this test's own
        # bounded subprocess timeout.
        env={"RESTART_VERIFY_TIMEOUT": "2", "RESTART_VERIFY_GRACE_SECS": "2"},
    )

    assert result.returncode == 1, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert clock_file.read_text() == sentinel, (
        f"clock file must be byte-identical after a failed verify even under "
        f"--drain; got {clock_file.read_text()!r}"
    )


def test_boundary4_defers_busy_unit_while_others_proceed(tmp_path: pathlib.Path) -> None:
    """Scenario 4 (I3) -- drain-defer: a unit R with a fresh
    merge_idle:false heartbeat is withheld from restart while a large
    ORCH_RESTART_FORCE_FIRE_AFTER_SECS is in effect -- proven via a bounded
    subprocess timeout (the script is still polling, not merely fast),
    mirroring scripts/tests/test_restart_all_orchestrators.py::
    test_defer_withholds_restart_while_busy. R is ordered AFTER a plain idle
    unit, so the idle unit's restart being recorded before the timeout shows
    other units proceed while R defers.
    """
    fleet_dir = tmp_path / "fleet"
    unit_idle = "orchestrator-alpha.service"
    unit_r = "orchestrator-reify.service"
    bin_dir, state_path = _boundary_make_fake_systemctl(
        tmp_path,
        running_units=[unit_idle, unit_r],
        units={unit_idle: {"scenario": "fresh"}, unit_r: {"scenario": "fresh"}},
    )
    _boundary_write_heartbeat(fleet_dir, unit_idle, merge_idle=True, ts_epoch=time.time())
    _boundary_write_heartbeat(fleet_dir, unit_r, merge_idle=False, ts_epoch=time.time())

    clock_file = tmp_path / "clock.json"

    with pytest.raises(subprocess.TimeoutExpired) as exc_info:
        _boundary_run_drain_script(
            bin_dir, state_path, fleet_dir, clock_file,
            env={
                "RESTART_VERIFY_TIMEOUT": "5",
                "ORCH_RESTART_FORCE_FIRE_AFTER_SECS": "99999",
                "ORCH_DRAIN_POLL_INTERVAL_SECS": "1",
            },
            # Wide margin over the couple of bash+python3 subprocess spawns
            # needed to reach the assertion point below (SELF_UNIT/list-units,
            # the idle unit's drain-check+baseline+restart, R's drain-check)
            # so a loaded CI host can't push the idle unit's restart past the
            # cutoff and flake the ordering assertion; FORCE_FIRE_AFTER_SECS
            # is 99999s, so widening this can't accidentally let R's own
            # restart land before the timeout fires.
            #
            # 20s (not 8s): under full tests/scripts/ suite load (32-way
            # xdist), the handful of subprocess spawns above can collectively
            # take long enough under CPU contention that an 8s wall-clock cap
            # kills the child before R's "deferring restart of ...: mid-merge"
            # line (a plain, unbuffered bash `echo`) is even reached -- not a
            # buffering issue, just insufficient scheduling margin. 20s
            # matches _boundary_run_drain_script's own default timeout, which
            # every other caller in this file already relies on safely.
            timeout=20,
        )

    stdout = _boundary_decode(exc_info.value.stdout)
    assert f"deferring restart of {unit_r}: mid-merge" in stdout, (
        f"expected a stable defer-prefix line naming {unit_r}; got stdout={stdout!r}"
    )
    state = _boundary_load_state(state_path)
    assert ["--user", "restart", unit_r] not in state["calls"], (
        f"{unit_r}'s restart must NOT have been recorded yet; got calls={state['calls']!r}"
    )
    assert ["--user", "restart", unit_idle] in state["calls"], (
        f"the idle unit ordered before {unit_r} must already be restarted "
        f"while {unit_r} defers; got calls={state['calls']!r}"
    )


def test_boundary5_force_restarts_busy_unit_after_grace(tmp_path: pathlib.Path) -> None:
    """Scenario 5 (I3) -- drain force after grace: a unit R continuously
    busy (fresh merge_idle:false heartbeat) is force-restarted once
    ORCH_RESTART_FORCE_FIRE_AFTER_SECS elapses -- here 0, so immediately --
    printing a "force-restarting" line, actually issuing the restart call,
    and exiting 0 (one re-verified merge accepted; recover_pending_merges
    makes it crash-safe -- see test_boundary10 below).
    """
    fleet_dir = tmp_path / "fleet"
    unit_r = "orchestrator-reify.service"
    bin_dir, state_path = _boundary_make_fake_systemctl(
        tmp_path, running_units=[unit_r], units={unit_r: {"scenario": "fresh"}},
    )
    _boundary_write_heartbeat(fleet_dir, unit_r, merge_idle=False, ts_epoch=time.time())

    clock_file = tmp_path / "clock.json"

    result = _boundary_run_drain_script(
        bin_dir, state_path, fleet_dir, clock_file,
        env={
            "RESTART_VERIFY_TIMEOUT": "5",
            "ORCH_RESTART_FORCE_FIRE_AFTER_SECS": "0",
        },
    )

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "force-restarting" in result.stdout.lower(), (
        f"expected a force-restart line; got stdout={result.stdout!r}"
    )
    state = _boundary_load_state(state_path)
    assert ["--user", "restart", unit_r] in state["calls"], (
        f"expected a restart call for {unit_r}; got calls={state['calls']!r}"
    )


def test_boundary6_absent_and_stale_heartbeat_proceed_after_grace(tmp_path: pathlib.Path) -> None:
    """Scenario 6 (I4) -- absent/stale heartbeat proceeds after the short
    unknown-grace: a unit with NO heartbeat file at all (absent) still
    restarts once ORCH_DRAIN_UNKNOWN_GRACE_SECS elapses -- here 0, so
    immediately -- a not-reporting unit must not block the fleet restart
    forever (fail-toward-convergence, the opposite fail direction from the
    confirmed-busy defer in test_boundary4/5 above). A second, stale-
    heartbeat sub-case (heartbeat file present but its ts_epoch is far
    outside the freshness window) exercises the same outcome via the other
    "unknown" branch drain_check.classify() recognizes.
    """
    fleet_dir = tmp_path / "fleet"
    unit_absent = "orchestrator-alpha.service"
    bin_dir, state_path = _boundary_make_fake_systemctl(
        tmp_path, running_units=[unit_absent], units={unit_absent: {"scenario": "fresh"}},
    )
    # No heartbeat file written for unit_absent at all.
    clock_file = tmp_path / "clock.json"

    result = _boundary_run_drain_script(
        bin_dir, state_path, fleet_dir, clock_file,
        env={
            "RESTART_VERIFY_TIMEOUT": "5",
            "ORCH_DRAIN_UNKNOWN_GRACE_SECS": "0",
        },
    )

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    state = _boundary_load_state(state_path)
    assert ["--user", "restart", unit_absent] in state["calls"], (
        f"expected a restart call for {unit_absent}; got calls={state['calls']!r}"
    )

    # --- stale-heartbeat sub-case: same outcome via the other unknown branch ---
    fleet_dir_2 = tmp_path / "fleet2"
    unit_stale_hb = "orchestrator-bravo.service"
    bin_dir_2, state_path_2 = _boundary_make_fake_systemctl(
        tmp_path / "run2",
        running_units=[unit_stale_hb], units={unit_stale_hb: {"scenario": "fresh"}},
    )
    _boundary_write_heartbeat(
        fleet_dir_2, unit_stale_hb, merge_idle=True, ts_epoch=time.time() - 99999
    )
    clock_file_2 = tmp_path / "clock2.json"

    result_2 = _boundary_run_drain_script(
        bin_dir_2, state_path_2, fleet_dir_2, clock_file_2,
        env={
            "RESTART_VERIFY_TIMEOUT": "5",
            "ORCH_DRAIN_UNKNOWN_GRACE_SECS": "0",
            "ORCH_DRAIN_FRESH_WINDOW_SECS": "120",
        },
    )

    assert result_2.returncode == 0, f"stdout={result_2.stdout!r} stderr={result_2.stderr!r}"
    state_2 = _boundary_load_state(state_path_2)
    assert ["--user", "restart", unit_stale_hb] in state_2["calls"], (
        f"expected a restart call for {unit_stale_hb}; got calls={state_2['calls']!r}"
    )


def test_boundary7_liveness_during_window_does_not_stamp_clock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """Scenario 7 (I5) -- liveness during the 8h window, non-stamping: with
    a REAL on-disk fleet-deploy clock dated ~1h ago (inside the window,
    sentinel {ts, iso}), main() still immediately revives a port-down unit
    (liveness is uncapped and not clock-gated) AND the clock file is left
    byte-identical afterward -- a single wedged-unit revive must never
    advance the fleet-wide deploy clock. Extends
    test_main_liveness_unaffected_by_fleet_deploy_gate above (which
    monkeypatches _within_fleet_deploy_min_interval directly) with a real
    on-disk clock file and the file-level non-stamping assertion.
    """
    wdog = _load_watchdog()

    now = 2_000_000_000.0
    clock_ts = now - 3600  # ~1h ago -- inside the 8h default window
    clock_file = tmp_path / "clock.json"
    sentinel_payload = json.dumps({"ts": clock_ts, "iso": "2026-07-15T23:00:00+00:00"})
    clock_file.write_text(sentinel_payload)
    monkeypatch.setattr(wdog, "FLEET_DEPLOY_CLOCK_PATH", str(clock_file))
    monkeypatch.setattr(wdog.time, "time", lambda: now)

    # Sanity: the fleet-deploy min-interval gate IS engaged for this pair.
    assert wdog._within_fleet_deploy_min_interval() is True

    restarted: list[str] = []
    monkeypatch.setattr(wdog, "_unit_start_elapsed_secs", lambda _u: None)
    monkeypatch.setattr(wdog, "is_unit_enabled", lambda _u: True)
    monkeypatch.setattr(wdog, "probe_port", lambda port: port != 8102)  # df probe fails (down)
    monkeypatch.setattr(wdog, "restart_unit", lambda u: restarted.append(u))
    monkeypatch.setattr(wdog, "log", lambda _m: None)

    wdog.main()

    assert restarted == ["orchestrator-dark-factory.service"], (
        f"main() must immediately revive a port-down unit even inside the "
        f"fleet-deploy min-interval window; got {restarted}"
    )
    assert clock_file.read_text() == sentinel_payload, (
        "main()'s per-unit liveness revive must never stamp the shared "
        "fleet-deploy clock -- that clock records a verified FLEET-WIDE "
        "deploy, not a single wedged-unit revive"
    )


def test_boundary8_coordinator_fire_while_busy_link_seam(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """Scenario 8 (I6) -- coordinator fire-while-busy, cross-tier LINK/
    composition assertion. The coordinator's internal arm/debounce/force-
    fire decision logic is owned by
    orchestrator/tests/test_fleet_staleness_composition.py and is NOT
    re-derived here -- ε only asserts the integration seam that makes
    fire-while-busy safe under the shared 8h cap:

    (a) the watchdog's FLEET_DEPLOY_CLOCK_PATH resolves to the SAME
        data/orchestrator/last_redeploy_orchestrator.json relative path the
        coordinator persists to (orchestrator.service_restart.
        FLEET_DEPLOY_CLOCK_RELPATH) -- the single shared clock both tiers
        honor (restates test_fleet_deploy_clock_path_matches_across_tiers
        above as part of this scenario's acceptance signal).
    (b) a bare OrchestratorConfig() exposes
        orchestrator_restart_force_fire_after_secs (δ's fire-while-busy
        knob), default 4500 (75 min).

    Uses pytest.importorskip("orchestrator.config") so the otherwise
    stdlib-only watchdog suite still collects and passes in a minimal env
    where the orchestrator package/venv is not importable.
    """
    pytest.importorskip("orchestrator.config")
    from orchestrator.config import OrchestratorConfig
    from orchestrator.service_restart import FLEET_DEPLOY_CLOCK_RELPATH

    monkeypatch.delenv("ORCH_FLEET_DEPLOY_CLOCK", raising=False)
    wdog = _load_watchdog()
    expected_path = str(pathlib.Path(wdog.REPO_DIR) / FLEET_DEPLOY_CLOCK_RELPATH)
    assert expected_path == wdog.FLEET_DEPLOY_CLOCK_PATH, (
        "the watchdog and the coordinator must honor the exact same shared "
        "fleet-deploy clock path for fire-while-busy to be safe under the "
        "8h cap"
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ORCH_CONFIG_PATH", raising=False)
    cfg = OrchestratorConfig()
    assert cfg.orchestrator_restart_force_fire_after_secs == 4500.0


def test_boundary9_report_mixed_fleet_seven_columns(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: pathlib.Path,
) -> None:
    """Scenario 9 (I8) -- `--report` on a mixed fleet: the flagship
    user-observable signal, combining what
    test_report_mixed_fleet_returns_1_and_lists_all_units /
    test_report_includes_deploy_age_column /
    test_report_includes_merge_idle_and_would_defer_columns /
    test_report_extended_columns_stay_read_only exercise separately into one
    integrated acceptance scenario: >=2 units (one stale start_epoch with a
    busy heartbeat, one fresh with an idle heartbeat) and a tmp
    ORCH_FLEET_DEPLOY_CLOCK ~3h old, pre-seeded and byte-captured.

    Asserts report() lists every unit; the header carries all seven columns
    (UNIT/START/NEWEST WATCHED COMMIT/VERDICT/DEPLOY-AGE/MERGE-IDLE/
    WOULD-DEFER); each row's VERDICT/DEPLOY-AGE/MERGE-IDLE/WOULD-DEFER
    values (extracted positionally -- the last four whitespace-separated
    tokens -- since the START/NEWEST WATCHED COMMIT date columns embed
    spaces, mirroring test_report_includes_merge_idle_and_would_defer_
    columns's technique) are correct; report() returns 1 (a stale unit is
    present); recorded subprocess argv contains ZERO mutating systemctl
    verbs; and the clock file is byte-identical afterward (no clock write).
    """
    wdog = _load_watchdog()

    commit_epoch = 1_800_000_000
    now = 2_000_000_000.0
    unit_stale = "orchestrator-stale.service"  # started before the commit, busy heartbeat
    unit_fresh = "orchestrator-fresh.service"  # started after the commit, idle heartbeat
    units = [unit_stale, unit_fresh]
    start_epochs = {unit_stale: commit_epoch - 100, unit_fresh: commit_epoch + 100}

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
    monkeypatch.setattr(wdog.time, "time", lambda: now)
    monkeypatch.setattr(
        wdog, "restart_unit", lambda u: pytest.fail(f"report() must never restart {u}")
    )

    clock_file = tmp_path / "clock.json"
    clock_payload = json.dumps({"ts": now - 3 * 3600, "iso": "2026-07-15T21:00:00+00:00"})
    clock_file.write_text(clock_payload)
    monkeypatch.setattr(wdog, "FLEET_DEPLOY_CLOCK_PATH", str(clock_file))

    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir()
    monkeypatch.setenv("ORCH_FLEET_DIR", str(fleet_dir))
    (fleet_dir / f"{unit_stale}.json").write_text(
        json.dumps(
            {
                "unit": unit_stale,
                "merge_idle": False,
                "depth": 1,
                "queue_empty": False,
                "ts_epoch": now - 10,
            }
        )
    )
    (fleet_dir / f"{unit_fresh}.json").write_text(
        json.dumps(
            {
                "unit": unit_fresh,
                "merge_idle": True,
                "depth": 0,
                "queue_empty": True,
                "ts_epoch": now - 10,
            }
        )
    )

    exit_code = wdog.report()

    assert exit_code == 1, "a stale unit is present -> report() must return 1"

    captured = capsys.readouterr()
    header_line = next(line for line in captured.out.splitlines() if line.startswith("UNIT"))
    for col in (
        "UNIT", "START", "NEWEST WATCHED COMMIT", "VERDICT",
        "DEPLOY-AGE", "MERGE-IDLE", "WOULD-DEFER",
    ):
        assert col in header_line, f"expected column {col!r} in header: {header_line!r}"

    for unit in units:
        assert unit in captured.out, f"report() output must list {unit}: {captured.out}"

    stale_line = next(line for line in captured.out.splitlines() if line.startswith(unit_stale))
    fresh_line = next(line for line in captured.out.splitlines() if line.startswith(unit_fresh))

    s_verdict, s_deploy_age, s_merge, s_defer = stale_line.split()[-4:]
    f_verdict, f_deploy_age, f_merge, f_defer = fresh_line.split()[-4:]

    assert s_verdict == "stale"
    assert s_deploy_age == "3.0h"
    assert s_merge == "busy"
    assert s_defer == "yes"

    assert f_verdict == "fresh"
    assert f_deploy_age == "3.0h"
    assert f_merge == "idle"
    assert f_defer == "no"

    assert recorded_calls, "report() must have driven subprocess.run through its real helpers"
    _assert_zero_mutating_calls(recorded_calls)
    assert clock_file.read_text() == clock_payload, (
        "report() must never write the shared fleet-deploy clock file"
    )


def test_boundary10_recover_pending_merges_link_seam() -> None:
    """Scenario 10 (I9) -- crash-safe force-restart, existing-behavior LINK
    test. The drain gate's force-restart-after-grace path (scenario 5 above)
    can safely kill a unit mid-merge only because a crash-safe recovery path
    exists: on boot, `recover_pending_merges` replays the durable merge-queue
    journal and re-enqueues surviving records while dropping any branch
    that's gone / already an ancestor of main (idempotency -- no double-land)
    -- so a force-restarted merge is neither double-landed nor lost.

    That behavior -- idempotent recovery, no double-land, task-not-lost -- is
    owned and already exercised by
    orchestrator/tests/test_merge_queue_restart_hook.py and is NOT
    re-derived here, per the task's explicit "existing-behavior link test"
    framing (PRD scenario 10 / I9). ε only asserts the integration seam: the
    recovery function the drain gate's crash-safety story depends on is
    present and callable.

    Uses pytest.importorskip("orchestrator.merge_queue_store") so the
    otherwise stdlib-only watchdog suite still collects and passes in a
    minimal env where the orchestrator package/venv is not importable.
    """
    pytest.importorskip("orchestrator.merge_queue_store")
    from orchestrator.merge_queue_store import recover_pending_merges

    assert callable(recover_pending_merges), (
        "the drain gate's force-restart-mid-merge path (scenario 5) relies "
        "on recover_pending_merges for crash-safe recovery on boot -- it "
        "must be present and callable"
    )


# NOTE: this test is intentionally NOT named test_boundaryN_ -- that prefix
# is reserved for THIS PRD's own fixed 1-10 scenario numbering (see the
# module comment above test_boundary1), which scenarios 1-10 above already
# fully occupy. This is follow-up coverage (task 2967, from task 2961's
# grace re-probe + a reviewer test-coverage gap), not an 11th PRD scenario.
def test_drain_grace_reprobe_delayed_fresh_unit_verifies_and_stamps_clock(
    tmp_path: pathlib.Path,
) -> None:
    """Follow-up (task 2967, from task 2961's VERIFY_TIMEOUT grace re-probe
    + a reviewer test-coverage gap): the SUCCESSFUL grace re-probe under
    `--drain`.

    restart_and_verify()'s grace re-probe (task 2961,
    scripts/restart-all-orchestrators.sh: "...re-probing for up to
    ${VERIFY_GRACE}s more..." then "OK (... verified fresh during grace
    re-probe)") is covered on the non-drain path by tests/scripts/
    test_restart_all_orchestrators.py::
    test_unit_fresh_only_during_grace_still_verifies_and_stamps. Scenarios
    2-6 above cover fresh-on-first-check (test_boundary2) and never-fresh
    (test_boundary3) under --drain, but none exercise a unit that is still
    stale when VERIFY_TIMEOUT expires and only turns fresh during the grace
    re-probe window while --drain is active. This closes that gap, mirroring
    the non-drain reference test's timing (verify_timeout=1/grace=5/
    fresh_after=2) against the boundary suite's REAL script + fake
    multi-unit systemctl harness.

    The unit's heartbeat is fresh+idle so drain_gate is transparent (returns
    immediately with zero extra `show` calls), keeping the post-restart
    show-counter cadence identical to the non-drain reference test.
    """
    fleet_dir = tmp_path / "fleet"
    unit_r = "orchestrator-reify.service"
    bin_dir, state_path = _boundary_make_fake_systemctl(
        tmp_path,
        running_units=[unit_r],
        units={unit_r: {"scenario": "delayed-fresh", "fresh_after": 2}},
    )
    _boundary_write_heartbeat(fleet_dir, unit_r, merge_idle=True, ts_epoch=time.time())

    clock_file = tmp_path / "clock.json"
    assert not clock_file.exists()

    result = _boundary_run_drain_script(
        bin_dir, state_path, fleet_dir, clock_file,
        env={"RESTART_VERIFY_TIMEOUT": "1", "RESTART_VERIFY_GRACE_SECS": "5"},
    )

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "FAILED" not in result.stdout, (
        f"must not declare FAILED; got stdout={result.stdout!r}"
    )
    assert "re-probing" in result.stdout, (
        f"expected the grace re-probe line; got stdout={result.stdout!r}"
    )
    state = _boundary_load_state(state_path)
    assert ["--user", "restart", unit_r] in state["calls"], (
        f"expected a restart call for {unit_r}; got calls={state['calls']!r}"
    )
    assert clock_file.exists(), "a verified-fresh drain-aware restart must stamp the clock"
    stamped = json.loads(clock_file.read_text())
    assert isinstance(stamped["ts"], (int, float)), f"ts must be numeric; got {stamped!r}"


# ---------------------------------------------------------------------------
# Part B: fused-memory liveness — constants + probe_health() (B1)
# ---------------------------------------------------------------------------


def test_fused_memory_constants_exposed() -> None:
    """The module exposes the fused-memory unit name, port, and health URL."""
    wdog = _load_watchdog()
    assert wdog.FUSED_MEMORY_UNIT == "fused-memory.service"
    assert wdog.FUSED_MEMORY_PORT == 8002
    assert hasattr(wdog, "FUSED_MEMORY_HEALTH_URL"), (
        "Module must expose a FUSED_MEMORY_HEALTH_URL constant"
    )
    assert "8002" in wdog.FUSED_MEMORY_HEALTH_URL
    assert "/health" in wdog.FUSED_MEMORY_HEALTH_URL


def test_fused_memory_port_matches_configured_server_port() -> None:
    """FUSED_MEMORY_PORT must equal fused-memory/config/config.yaml's server.port.

    Config-drift guard mirroring test_watched_ports_match_configured_escalation_ports
    above: skipped gracefully if the config file is unreachable in this environment.
    """
    yaml = pytest.importorskip("yaml")
    wdog = _load_watchdog()

    config_path = REPO_ROOT / "fused-memory" / "config" / "config.yaml"
    if not config_path.exists():
        pytest.skip(f"{config_path} not reachable in this environment")
    cfg = yaml.safe_load(config_path.read_text())
    server = cfg.get("server") if isinstance(cfg, dict) else None
    port = server.get("port") if isinstance(server, dict) else None
    assert port is not None, f"{config_path}: missing 'server.port' (schema may have changed)"
    assert port == wdog.FUSED_MEMORY_PORT, (
        f"FUSED_MEMORY_PORT ({wdog.FUSED_MEMORY_PORT}) != "
        f"fused-memory/config/config.yaml server.port ({port})"
    )


class _FakeHealthResponse:
    """Minimal context-manager stand-in for urllib.request.urlopen's return value."""

    def __init__(self, status: int = 200) -> None:
        self.status = status

    def __enter__(self) -> "_FakeHealthResponse":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False

    def getcode(self) -> int:
        return self.status


def test_probe_health_true_on_200(monkeypatch: pytest.MonkeyPatch) -> None:
    """probe_health returns True when urlopen succeeds with a 2xx response."""
    wdog = _load_watchdog()

    def fake_urlopen(*args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        return _FakeHealthResponse(200)

    monkeypatch.setattr(wdog.urllib.request, "urlopen", fake_urlopen)
    assert wdog.probe_health() is True


def test_probe_health_true_on_503_degraded(monkeypatch: pytest.MonkeyPatch) -> None:
    """probe_health returns True on HTTPError (e.g. 503) — the loop IS serving.

    fused-memory's /health returns 503 when a backing store (FalkorDB/Qdrant)
    is degraded, but the asyncio event loop DID respond within the timeout —
    restarting the process would not fix a down store, so this response must
    NOT be treated as dead/wedged.
    """
    wdog = _load_watchdog()

    def fake_urlopen(*args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        raise wdog.urllib.error.HTTPError(
            "http://127.0.0.1:8002/health", 503, "Service Unavailable", None, None
        )

    monkeypatch.setattr(wdog.urllib.request, "urlopen", fake_urlopen)
    assert wdog.probe_health() is True


def test_probe_health_false_on_url_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """probe_health returns False when urlopen raises URLError (no response received)."""
    wdog = _load_watchdog()

    def fake_urlopen(*args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        raise wdog.urllib.error.URLError("connection refused")

    monkeypatch.setattr(wdog.urllib.request, "urlopen", fake_urlopen)
    assert wdog.probe_health() is False


def test_probe_health_false_on_connection_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """probe_health returns False when urlopen raises ConnectionRefusedError directly."""
    wdog = _load_watchdog()

    def fake_urlopen(*args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        raise ConnectionRefusedError("connection refused")

    monkeypatch.setattr(wdog.urllib.request, "urlopen", fake_urlopen)
    assert wdog.probe_health() is False


def test_probe_health_false_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """probe_health returns False when urlopen times out — the true wedged/dead signal.

    socket.timeout is an alias of TimeoutError (Python 3.10+), so a single
    TimeoutError-raising test covers both names.
    """
    wdog = _load_watchdog()

    def fake_urlopen(*args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        raise TimeoutError("timed out")

    monkeypatch.setattr(wdog.urllib.request, "urlopen", fake_urlopen)
    assert wdog.probe_health() is False


# ---------------------------------------------------------------------------
# Part B: _fused_memory_liveness_verdict() (B2)
# ---------------------------------------------------------------------------


def test_liveness_verdict_port_down(monkeypatch: pytest.MonkeyPatch) -> None:
    """_fused_memory_liveness_verdict returns 'port-down' when probe_port is False."""
    wdog = _load_watchdog()

    monkeypatch.setattr(wdog, "probe_port", lambda _port: False)
    monkeypatch.setattr(
        wdog, "probe_health", lambda: pytest.fail("probe_health must not run when port is down")
    )

    assert wdog._fused_memory_liveness_verdict() == "port-down"


def test_liveness_verdict_healthy(monkeypatch: pytest.MonkeyPatch) -> None:
    """_fused_memory_liveness_verdict returns 'healthy' when port is up and health succeeds."""
    wdog = _load_watchdog()

    monkeypatch.setattr(wdog, "probe_port", lambda _port: True)
    monkeypatch.setattr(wdog, "probe_health", lambda: True)

    assert wdog._fused_memory_liveness_verdict() == "healthy"


def test_liveness_verdict_wedged(monkeypatch: pytest.MonkeyPatch) -> None:
    """_fused_memory_liveness_verdict returns 'wedged' when port is up but health fails."""
    wdog = _load_watchdog()

    monkeypatch.setattr(wdog, "probe_port", lambda _port: True)
    monkeypatch.setattr(wdog, "probe_health", lambda: False)

    assert wdog._fused_memory_liveness_verdict() == "wedged"


# ---------------------------------------------------------------------------
# Part B: fused_memory_liveness_pass() (B3)
# ---------------------------------------------------------------------------


def test_liveness_pass_revives_on_port_down(monkeypatch: pytest.MonkeyPatch) -> None:
    """USER-SIGNAL: fused_memory_liveness_pass() restarts the unit when the port is down."""
    wdog = _load_watchdog()
    restarted: list[str] = []

    monkeypatch.setattr(wdog, "is_unit_enabled", lambda _u: True)
    monkeypatch.setattr(wdog, "_unit_start_elapsed_secs", lambda _u: None)
    monkeypatch.setattr(wdog, "probe_port", lambda _port: False)
    monkeypatch.setattr(
        wdog, "probe_health", lambda: pytest.fail("probe_health must not run when port is down")
    )
    monkeypatch.setattr(wdog, "restart_unit", lambda unit: restarted.append(unit))
    monkeypatch.setattr(wdog, "log", lambda _m: None)

    wdog.fused_memory_liveness_pass()

    assert restarted == ["fused-memory.service"], (
        f"Expected fused-memory.service restarted exactly once, got: {restarted}"
    )


def test_liveness_pass_no_restart_when_healthy(monkeypatch: pytest.MonkeyPatch) -> None:
    """fused_memory_liveness_pass() must not restart when port is up and health succeeds."""
    wdog = _load_watchdog()
    restarted: list[str] = []

    monkeypatch.setattr(wdog, "is_unit_enabled", lambda _u: True)
    monkeypatch.setattr(wdog, "_unit_start_elapsed_secs", lambda _u: None)
    monkeypatch.setattr(wdog, "probe_port", lambda _port: True)
    monkeypatch.setattr(wdog, "probe_health", lambda: True)
    monkeypatch.setattr(wdog, "restart_unit", lambda unit: restarted.append(unit))
    monkeypatch.setattr(wdog, "log", lambda _m: None)

    wdog.fused_memory_liveness_pass()

    assert restarted == [], "No restart expected when port is up and health succeeds"


def test_liveness_pass_restarts_when_wedged(monkeypatch: pytest.MonkeyPatch) -> None:
    """fused_memory_liveness_pass() restarts when the port is up but the health fetch fails."""
    wdog = _load_watchdog()
    restarted: list[str] = []

    monkeypatch.setattr(wdog, "is_unit_enabled", lambda _u: True)
    monkeypatch.setattr(wdog, "_unit_start_elapsed_secs", lambda _u: None)
    monkeypatch.setattr(wdog, "probe_port", lambda _port: True)
    monkeypatch.setattr(wdog, "probe_health", lambda: False)
    monkeypatch.setattr(wdog, "restart_unit", lambda unit: restarted.append(unit))
    monkeypatch.setattr(wdog, "log", lambda _m: None)

    wdog.fused_memory_liveness_pass()

    assert restarted == ["fused-memory.service"], (
        f"Expected fused-memory.service restarted when wedged, got: {restarted}"
    )


def test_liveness_pass_skips_disabled_unit(monkeypatch: pytest.MonkeyPatch) -> None:
    """fused_memory_liveness_pass() must not probe or restart a disabled unit.

    Disabling is explicit operator intent, mirroring main()'s
    test_main_skips_disabled_unit_entirely.
    """
    wdog = _load_watchdog()
    probed: list[int] = []
    restarted: list[str] = []

    monkeypatch.setattr(wdog, "is_unit_enabled", lambda _u: False)
    monkeypatch.setattr(wdog, "probe_port", lambda port: probed.append(port) or True)
    monkeypatch.setattr(wdog, "restart_unit", lambda unit: restarted.append(unit))
    monkeypatch.setattr(wdog, "log", lambda _m: None)

    wdog.fused_memory_liveness_pass()

    assert probed == [], f"probe_port must not be called for a disabled unit; probed: {probed}"
    assert restarted == []


def test_liveness_pass_skips_probe_in_grace_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """fused_memory_liveness_pass() must not probe within STARTUP_GRACE_SECS of unit start.

    Mirrors main()'s test_main_skips_probe_in_grace_window: a freshly (re)started
    unit may not have bound its port yet, so probing would risk a false restart.
    """
    wdog = _load_watchdog()
    probed: list[int] = []
    restarted: list[str] = []

    monkeypatch.setattr(wdog, "is_unit_enabled", lambda _u: True)
    monkeypatch.setattr(wdog, "_unit_start_elapsed_secs", lambda _u: 30.0)
    monkeypatch.setattr(wdog, "probe_port", lambda port: probed.append(port) or True)
    monkeypatch.setattr(wdog, "restart_unit", lambda unit: restarted.append(unit))
    monkeypatch.setattr(wdog, "log", lambda _m: None)

    wdog.fused_memory_liveness_pass()

    assert probed == [], (
        f"probe_port must not be called inside the grace window; probed: {probed}"
    )
    assert restarted == []


def test_liveness_pass_isolates_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """An exception raised inside the probe/verdict must not propagate out of the pass.

    Mirrors main()'s test_main_isolates_per_unit_failure — a single-unit
    analogue since fused_memory_liveness_pass() only ever handles one unit.
    """
    wdog = _load_watchdog()

    monkeypatch.setattr(wdog, "is_unit_enabled", lambda _u: True)
    monkeypatch.setattr(wdog, "_unit_start_elapsed_secs", lambda _u: None)

    def fake_probe(_port: int) -> bool:
        raise RuntimeError("ss exploded")

    monkeypatch.setattr(wdog, "probe_port", fake_probe)
    monkeypatch.setattr(wdog, "log", lambda _m: None)

    # Must not raise
    wdog.fused_memory_liveness_pass()


# ---------------------------------------------------------------------------
# Part B: _print_fused_memory_liveness() + --report wiring (B4)
# ---------------------------------------------------------------------------


_SS_LISTEN_8002 = _SS_HEADER + "LISTEN 0      2048       127.0.0.1:8002      0.0.0.0:*\n"


def test_print_fused_memory_liveness_row(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """USER-SIGNAL: --report's fused-memory row names the unit + verdict, mutates nothing.

    Drives _print_fused_memory_liveness() through the REAL
    _fused_memory_liveness_verdict() -> probe_port()/probe_health() chain for
    each of the three verdicts, faking subprocess.run (the `ss` port probe)
    and urllib.request.urlopen (the /health fetch) rather than monkeypatching
    the verdict function directly, so the "zero mutating calls" assertion
    below is meaningful — mirrors test_report_mixed_fleet_returns_1_and_lists_all_units's
    rationale for driving through real helpers instead of stubbing them.
    restart_unit is also monkeypatched to fail the test outright if called,
    as a direct belt-and-suspenders check that this read-only print helper
    never mutates (I7/I8 extended to the fused-memory row).
    """
    wdog = _load_watchdog()

    # (verdict token, `ss` stdout, /health outcome: None (unreached), an int
    # status code, or an exception instance for urlopen to raise)
    scenarios = [
        ("port-down", _SS_HEADER, None),
        ("healthy", _SS_LISTEN_8002, 200),
        ("wedged", _SS_LISTEN_8002, TimeoutError("timed out")),
    ]

    for verdict_token, ss_stdout, health_outcome in scenarios:
        recorded_calls: list[list[str]] = []

        # The loop variables are bound explicitly as KEYWORD-ONLY defaults (so no
        # positional caller can ever reach them). A closure defined in a loop
        # otherwise reads whatever the name holds when it is CALLED, not when it
        # was defined (B023) — benign only while every call stays inside the same
        # iteration, which nothing here enforces. `_recorded` is the same list
        # object `recorded_calls` names, so the assertion below still sees the
        # appends.
        def fake_run(cmd, *, _recorded=recorded_calls, _ss=ss_stdout, **kwargs):  # noqa: ANN001
            _recorded.append(list(cmd))
            assert cmd[0] == "ss", f"unexpected subprocess.run call: {cmd}"
            return subprocess.CompletedProcess(cmd, 0, stdout=_ss, stderr="")

        def fake_urlopen(*args, _outcome=health_outcome, **kwargs):  # noqa: ANN001, ANN002, ANN003
            if isinstance(_outcome, Exception):
                raise _outcome
            return _FakeHealthResponse(_outcome)

        monkeypatch.setattr(subprocess, "run", fake_run)
        monkeypatch.setattr(wdog.urllib.request, "urlopen", fake_urlopen)
        monkeypatch.setattr(
            wdog, "restart_unit", lambda u: pytest.fail(f"must never restart {u}")
        )
        # probe_health()'s no-response branch calls log(), which itself
        # shells out to `systemd-cat` via subprocess.run — no-op it so
        # fake_run only ever sees the `ss` probe call it's built to handle.
        monkeypatch.setattr(wdog, "log", lambda _m: None)
        # The enriched row (step 16) also reads the fm deploy clock and the
        # recon-busy verdict; stub both so this liveness-focused test neither
        # hits a real socket for recon-busy nor depends on a real clock file.
        monkeypatch.setattr(wdog, "_read_last_fm_deploy_epoch", lambda: None)
        monkeypatch.setattr(wdog, "_fused_memory_recon_busy_verdict", lambda: "idle")

        wdog._print_fused_memory_liveness()

        captured = capsys.readouterr()
        assert "fused-memory.service" in captured.out, (
            f"[{verdict_token}] expected unit name in output: {captured.out!r}"
        )
        assert verdict_token in captured.out, (
            f"[{verdict_token}] expected verdict token in output: {captured.out!r}"
        )
        _assert_zero_mutating_calls(recorded_calls)


def test_print_fused_memory_liveness_row_survives_verdict_exception(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """_print_fused_memory_liveness() must not crash --report on a probe failure.

    Mirrors fused_memory_liveness_pass()'s per-unit try/except isolation (B3):
    an unexpected exception surfacing from _fused_memory_liveness_verdict()
    (e.g. a non-defensive OSError/PermissionError propagating out of
    probe_port, which only catches FileNotFoundError/TimeoutExpired) must be
    caught, logged, and degrade to an 'unknown' row rather than propagating
    and aborting the read-only --report doctor path after report() has
    already computed its exit code.
    """
    wdog = _load_watchdog()

    def _boom() -> str:
        raise OSError("permission denied")

    monkeypatch.setattr(wdog, "_fused_memory_liveness_verdict", _boom)
    monkeypatch.setattr(
        wdog, "restart_unit", lambda u: pytest.fail(f"must never restart {u}")
    )
    # The enriched row also reads the fm clock and recon-busy verdict; stub
    # both so this fail-soft test stays hermetic (no real socket / clock file).
    monkeypatch.setattr(wdog, "_read_last_fm_deploy_epoch", lambda: None)
    monkeypatch.setattr(wdog, "_fused_memory_recon_busy_verdict", lambda: "idle")
    logged: list[str] = []
    monkeypatch.setattr(wdog, "log", lambda m: logged.append(m))

    # Must not raise.
    wdog._print_fused_memory_liveness()

    captured = capsys.readouterr()
    assert "fused-memory.service" in captured.out
    assert "unknown" in captured.out
    assert logged, "the swallowed exception must be logged"


def test_cli_report_includes_fused_memory_row(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """_cli(["--report"]) output includes the fused-memory liveness row.

    report() is stubbed to a silent no-op so only _print_fused_memory_liveness()
    can be the source of this output; report()'s own exit code (0) must still
    be what _cli returns, confirming the fm row is informational-only and
    does not alter report()'s staleness-only exit-code contract.
    """
    wdog = _load_watchdog()

    monkeypatch.setattr(wdog, "report", lambda: 0)
    monkeypatch.setattr(wdog, "_fused_memory_liveness_verdict", lambda: "healthy")
    # Stub the enriched-row helpers (step 16) so this test neither hits a real
    # socket for the recon-busy verdict nor depends on a real clock file.
    monkeypatch.setattr(wdog, "_read_last_fm_deploy_epoch", lambda: None)
    monkeypatch.setattr(wdog, "_fused_memory_recon_busy_verdict", lambda: "idle")

    exit_code = wdog._cli(["--report"])

    captured = capsys.readouterr()
    assert "fused-memory.service" in captured.out
    assert "healthy" in captured.out
    assert exit_code == 0, "report()'s staleness-only exit code must be unaffected by the fm row"


# ---------------------------------------------------------------------------
# Enriched --report fused-memory row: DEPLOY-AGE + recon-busy (steps 15/16)
#
# _fused_memory_recon_busy_verdict() lazily reuses scripts/recon_busy_check.py
# — the SAME busy/idle/unreachable gate restart-fused-memory.sh's defer-if-busy
# path consumes — so this column predicts the restart gate exactly; it degrades
# to 'unknown' on any fetch/import failure. _print_fused_memory_liveness() is
# enriched with a DEPLOY-AGE field (fm-clock age in hours) and a recon-busy
# field, staying strictly read-only (no mutation, no clock write).
# ---------------------------------------------------------------------------


class _FakeHealthBodyResponse:
    """urlopen stand-in whose .read() returns a fixed /health body as bytes.

    Distinct from _FakeHealthResponse (which only models a status code for the
    liveness probe): _fused_memory_recon_busy_verdict() reads the response
    BODY and runs it through recon_busy_check.parse_health()/classify().
    """

    def __init__(self, body: str) -> None:
        self._body = body.encode("utf-8")

    def __enter__(self) -> "_FakeHealthBodyResponse":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False

    def read(self) -> bytes:
        return self._body


def _run_recon_busy_verdict_with_body(
    wdog: types.ModuleType, monkeypatch: pytest.MonkeyPatch, body: str
) -> str:
    """Drive _fused_memory_recon_busy_verdict() with a faked /health *body*
    flowing through the REAL lazy-imported recon_busy_check.classify()."""

    def fake_urlopen(*args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        return _FakeHealthBodyResponse(body)

    monkeypatch.setattr(wdog.urllib.request, "urlopen", fake_urlopen)
    return wdog._fused_memory_recon_busy_verdict()


def test_fused_memory_recon_busy_verdict_busy(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-empty recon_busy list classifies as 'busy' (a full cycle is in flight)."""
    wdog = _load_watchdog()
    body = json.dumps(
        {"status": "ok", "recon_busy": [{"project_id": "dark_factory", "run_id": "r1"}]}
    )
    assert _run_recon_busy_verdict_with_body(wdog, monkeypatch, body) == "busy"


def test_fused_memory_recon_busy_verdict_idle_empty_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty recon_busy list classifies as 'idle'."""
    wdog = _load_watchdog()
    body = json.dumps({"status": "ok", "recon_busy": []})
    assert _run_recon_busy_verdict_with_body(wdog, monkeypatch, body) == "idle"


def test_fused_memory_recon_busy_verdict_idle_absent_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An absent recon_busy field classifies as 'idle'."""
    wdog = _load_watchdog()
    body = json.dumps({"status": "ok"})
    assert _run_recon_busy_verdict_with_body(wdog, monkeypatch, body) == "idle"


def test_fused_memory_recon_busy_verdict_unreachable_on_blank_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A blank/whitespace body (unreachable/degraded endpoint) → 'unreachable'.

    The fetch SUCCEEDS but returns an unparseable body — recon_busy_check.
    parse_health() returns None and classify() maps that to 'unreachable'.
    This is the case distinct from an outright fetch exception (→ 'unknown').
    """
    wdog = _load_watchdog()
    assert _run_recon_busy_verdict_with_body(wdog, monkeypatch, "   ") == "unreachable"


def test_fused_memory_recon_busy_verdict_unknown_on_fetch_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Any fetch/import exception degrades to 'unknown' (fail-soft) and is logged."""
    wdog = _load_watchdog()

    def fake_urlopen(*args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        raise ConnectionRefusedError("connection refused")

    monkeypatch.setattr(wdog.urllib.request, "urlopen", fake_urlopen)
    warnings: list[str] = []
    monkeypatch.setattr(
        wdog.logger, "warning", lambda m, *a, **k: warnings.append(str(m))
    )

    assert wdog._fused_memory_recon_busy_verdict() == "unknown"
    assert warnings, "the swallowed fetch exception must be logged"


def test_print_fused_memory_liveness_row_includes_deploy_age(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The enriched fm row renders DEPLOY-AGE as fm-clock age in hours (one decimal).

    Mirrors report()'s DEPLOY-AGE column: (now - fm-clock epoch) / 3600 to one
    decimal place, sourced from _read_last_fm_deploy_epoch (fm's OWN clock).
    """
    wdog = _load_watchdog()
    now = 2_000_000_000.0
    monkeypatch.setattr(wdog, "_fused_memory_liveness_verdict", lambda: "healthy")
    monkeypatch.setattr(wdog, "_fused_memory_recon_busy_verdict", lambda: "idle")
    monkeypatch.setattr(wdog, "_read_last_fm_deploy_epoch", lambda: now - 3 * 3600)
    monkeypatch.setattr(wdog.time, "time", lambda: now)

    wdog._print_fused_memory_liveness()

    out = capsys.readouterr().out
    assert "DEPLOY-AGE" in out, f"expected DEPLOY-AGE label in fm row: {out!r}"
    assert "3.0h" in out, f"expected DEPLOY-AGE ~3.0h in fm row: {out!r}"


def test_print_fused_memory_liveness_row_deploy_age_unknown_when_clock_absent(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """DEPLOY-AGE renders 'unknown' when the fm deploy clock is absent (fail-open).

    Mirrors _read_last_fm_deploy_epoch's fail-open contract (None when the fm
    clock file has never been stamped / is unreadable).
    """
    wdog = _load_watchdog()
    monkeypatch.setattr(wdog, "_fused_memory_liveness_verdict", lambda: "healthy")
    monkeypatch.setattr(wdog, "_fused_memory_recon_busy_verdict", lambda: "idle")
    monkeypatch.setattr(wdog, "_read_last_fm_deploy_epoch", lambda: None)

    wdog._print_fused_memory_liveness()

    out = capsys.readouterr().out
    assert "DEPLOY-AGE" in out
    assert "unknown" in out, f"expected DEPLOY-AGE 'unknown' in fm row: {out!r}"


def test_print_fused_memory_liveness_row_includes_recon_busy(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The enriched fm row carries a labelled recon-busy field from the verdict."""
    wdog = _load_watchdog()
    monkeypatch.setattr(wdog, "_fused_memory_liveness_verdict", lambda: "healthy")
    monkeypatch.setattr(wdog, "_read_last_fm_deploy_epoch", lambda: None)
    monkeypatch.setattr(wdog, "_fused_memory_recon_busy_verdict", lambda: "busy")

    wdog._print_fused_memory_liveness()

    out = capsys.readouterr().out
    assert "recon-busy: busy" in out, (
        f"expected labelled recon-busy field carrying the verdict in fm row: {out!r}"
    )


def test_print_fused_memory_liveness_row_enriched_stays_read_only(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: pathlib.Path,
) -> None:
    """I8 guard extended to the enriched fm row: no mutating systemctl call and
    no write to FM_DEPLOY_CLOCK_PATH.

    Drives the REAL _fused_memory_liveness_verdict() -> probe_port()/
    probe_health() chain (faking `ss` + urlopen) so the zero-mutating-calls
    assertion is meaningful, stubs the recon-busy verdict to avoid a second
    socket, points the fm deploy clock at an absent tmp file, and asserts the
    read-only row never creates it.
    """
    wdog = _load_watchdog()

    recorded_calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        recorded_calls.append(list(cmd))
        assert cmd[0] == "ss", f"unexpected subprocess.run call: {cmd}"
        return subprocess.CompletedProcess(cmd, 0, stdout=_SS_LISTEN_8002, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        wdog.urllib.request, "urlopen", lambda *a, **k: _FakeHealthResponse(200)
    )
    monkeypatch.setattr(
        wdog, "restart_unit", lambda u: pytest.fail(f"must never restart {u}")
    )
    monkeypatch.setattr(wdog, "log", lambda _m: None)
    monkeypatch.setattr(wdog, "_fused_memory_recon_busy_verdict", lambda: "idle")

    clock = tmp_path / "last_redeploy_fused_memory.json"
    monkeypatch.setattr(wdog, "FM_DEPLOY_CLOCK_PATH", str(clock))

    wdog._print_fused_memory_liveness()

    captured = capsys.readouterr()
    assert "fused-memory.service" in captured.out
    _assert_zero_mutating_calls(recorded_calls)
    assert not clock.exists(), (
        "the read-only fm row must never write the fm deploy clock file"
    )


# ---------------------------------------------------------------------------
# Part C: fused-memory staleness — constants (step 1)
#
# fm-staleness siblings of the orchestrator staleness constants. These pin the
# new FM_* constants that fused_memory_staleness_pass() and its clock/delegate
# helpers consume: the watched-paths list, fm's OWN deploy-clock file + env
# override, the min-interval knob (env-with-fallback like
# ORCH_RESTART_MIN_INTERVAL_SECS), and the fixed transient redeploy unit name.
# ---------------------------------------------------------------------------


def test_fm_watched_paths_constant() -> None:
    """FM_WATCHED_PATHS is exactly [fused-memory/src/, shared/src/].

    fused-memory imports shared.* (e.g. shared.task_metadata), so a change to
    shared/src/ can alter fm's behavior and must count toward fm staleness —
    hence both prefixes are watched.
    """
    wdog = _load_watchdog()
    assert wdog.FM_WATCHED_PATHS == ["fused-memory/src/", "shared/src/"]


def test_fm_deploy_clock_path_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """FM_DEPLOY_CLOCK_PATH defaults to fm's OWN clock file under REPO_DIR."""
    monkeypatch.delenv("FM_DEPLOY_CLOCK", raising=False)
    wdog = _load_watchdog()
    assert os.path.join(
        wdog.REPO_DIR, "data", "fused-memory", "last_redeploy_fused_memory.json"
    ) == wdog.FM_DEPLOY_CLOCK_PATH


def test_fm_deploy_clock_path_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """FM_DEPLOY_CLOCK_PATH honors the FM_DEPLOY_CLOCK env override.

    _load_watchdog() re-execs the module, so an env var set before the call is
    picked up at (re)import time — mirrors FLEET_DEPLOY_CLOCK_PATH's
    ORCH_FLEET_DEPLOY_CLOCK override.
    """
    monkeypatch.setenv("FM_DEPLOY_CLOCK", "/tmp/custom_fm_clock.json")
    wdog = _load_watchdog()
    assert wdog.FM_DEPLOY_CLOCK_PATH == "/tmp/custom_fm_clock.json"


def test_fm_deploy_clock_path_separate_from_fleet_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """fm's deploy clock must be a DIFFERENT file than the orchestrator fleet clock.

    fused-memory and the orchestrator fleet are independent deploy targets
    whose redeploy cadences must not couple — an orchestrator fleet redeploy
    must not reset fm's min-interval window and vice-versa.
    """
    monkeypatch.delenv("FM_DEPLOY_CLOCK", raising=False)
    monkeypatch.delenv("ORCH_FLEET_DEPLOY_CLOCK", raising=False)
    wdog = _load_watchdog()
    assert wdog.FM_DEPLOY_CLOCK_PATH != wdog.FLEET_DEPLOY_CLOCK_PATH


def test_fm_restart_min_interval_secs_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """FM_RESTART_MIN_INTERVAL_SECS defaults to 28800 (8h) with no env override.

    Mirrors ORCH_RESTART_MIN_INTERVAL_SECS's 8h backstop cadence, but as fm's
    OWN independent knob.
    """
    monkeypatch.delenv("FM_RESTART_MIN_INTERVAL_SECS", raising=False)
    wdog = _load_watchdog()
    assert wdog.FM_RESTART_MIN_INTERVAL_SECS == 28800


def test_fm_restart_min_interval_secs_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """FM_RESTART_MIN_INTERVAL_SECS honors a valid env override."""
    monkeypatch.setenv("FM_RESTART_MIN_INTERVAL_SECS", "60")
    wdog = _load_watchdog()
    assert wdog.FM_RESTART_MIN_INTERVAL_SECS == 60


def test_fm_restart_min_interval_secs_malformed_env_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed FM_RESTART_MIN_INTERVAL_SECS env value falls back to 28800.

    A typo'd env var must not crash the oneshot watchdog — fall-safe ethos,
    mirroring ORCH_RESTART_MIN_INTERVAL_SECS's malformed-env test.
    """
    monkeypatch.setenv("FM_RESTART_MIN_INTERVAL_SECS", "not-an-int")
    wdog = _load_watchdog()
    assert wdog.FM_RESTART_MIN_INTERVAL_SECS == 28800


def test_fm_staleness_redeploy_unit_constant() -> None:
    """FM_STALENESS_REDEPLOY_UNIT is the fixed transient redeploy unit name."""
    wdog = _load_watchdog()
    assert wdog.FM_STALENESS_REDEPLOY_UNIT == "fm-staleness-redeploy.service"


# ---------------------------------------------------------------------------
# Part C: _unit_active_enter_epoch() (step 3)
#
# fm sibling of _unit_start_epoch: structurally identical, but queries
# ActiveEnterTimestamp (when the unit signalled readiness — the field
# restart-all-orchestrators.sh's own freshness verify reads) instead of
# ExecMainStartTimestamp. fused_memory_staleness_pass() compares this against
# the newest fm-watched commit.
# ---------------------------------------------------------------------------


def _make_active_enter_epoch_result(value: str, rc: int = 0) -> subprocess.CompletedProcess:
    """Build a fake `systemctl show --timestamp=unix -p ActiveEnterTimestamp` result."""
    stdout = f"ActiveEnterTimestamp={value}\n"
    return subprocess.CompletedProcess(["systemctl"], rc, stdout=stdout, stderr="")


def test_unit_active_enter_epoch_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """_unit_active_enter_epoch parses the '@<epoch>' realtime value to an int.

    Also pins the field choice: the argv must request ActiveEnterTimestamp and
    must NOT request ExecMainStartTimestamp (the sibling helper's field).
    """
    wdog = _load_watchdog()
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        calls.append(list(cmd))
        return _make_active_enter_epoch_result("@1782996274")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = wdog._unit_active_enter_epoch("fused-memory.service")

    assert result == 1782996274
    assert isinstance(result, int)
    assert len(calls) == 1, f"Expected exactly one systemctl call, got {calls}"
    argv = calls[0]
    assert "--timestamp=unix" in argv
    assert any("ActiveEnterTimestamp" in tok for tok in argv), (
        f"argv must request ActiveEnterTimestamp: {argv}"
    )
    assert not any("ExecMainStartTimestamp" in tok for tok in argv), (
        f"argv must NOT request ExecMainStartTimestamp (that is _unit_start_epoch's field): {argv}"
    )


def test_unit_active_enter_epoch_nonzero_rc_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """_unit_active_enter_epoch returns None when systemctl exits non-zero."""
    wdog = _load_watchdog()

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        return _make_active_enter_epoch_result("@1782996274", rc=1)

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert wdog._unit_active_enter_epoch("some.service") is None


def test_unit_active_enter_epoch_empty_value_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """_unit_active_enter_epoch returns None when the property value is empty."""
    wdog = _load_watchdog()

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        return _make_active_enter_epoch_result("")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert wdog._unit_active_enter_epoch("some.service") is None


def test_unit_active_enter_epoch_zero_sentinel_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """_unit_active_enter_epoch returns None for the '@0' sentinel (never activated)."""
    wdog = _load_watchdog()

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        return _make_active_enter_epoch_result("@0")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert wdog._unit_active_enter_epoch("some.service") is None


def test_unit_active_enter_epoch_unparseable_value_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_unit_active_enter_epoch returns None when the value is not an int."""
    wdog = _load_watchdog()

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        return _make_active_enter_epoch_result("@notanint")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert wdog._unit_active_enter_epoch("some.service") is None


def test_unit_active_enter_epoch_missing_binary_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_unit_active_enter_epoch returns None when systemctl binary is not found."""
    wdog = _load_watchdog()

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        raise FileNotFoundError(2, "No such file or directory", "systemctl")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert wdog._unit_active_enter_epoch("some.service") is None


def test_unit_active_enter_epoch_timeout_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """_unit_active_enter_epoch returns None when the systemctl call times out."""
    wdog = _load_watchdog()

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        raise subprocess.TimeoutExpired(cmd, 5)

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert wdog._unit_active_enter_epoch("some.service") is None


# ---------------------------------------------------------------------------
# Part C: _newest_fm_watched_commit_epoch() (step 5)
#
# fm sibling of _newest_watched_commit_epoch: identical body but diffs
# FM_WATCHED_PATHS (fused-memory/src/ + shared/src/) rather than WATCHED_PATHS.
# ---------------------------------------------------------------------------

_EXPECTED_FM_WATCHED_PATHS = ["fused-memory/src/", "shared/src/"]


def test_newest_fm_watched_commit_epoch_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """_newest_fm_watched_commit_epoch parses git's %ct output to an int.

    Also pins that the argv diffs the fm-watched paths (both fused-memory/src/
    and shared/src/) rather than the orchestrator WATCHED_PATHS.
    """
    wdog = _load_watchdog()
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="1783013906\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = wdog._newest_fm_watched_commit_epoch()

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
    for path in _EXPECTED_FM_WATCHED_PATHS:
        assert path in watched_args, f"Expected fm-watched path {path!r} in argv {argv}"


def test_newest_fm_watched_commit_epoch_empty_stdout_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_newest_fm_watched_commit_epoch returns None on empty stdout (rc 0).

    `git log -1 --format=%ct HEAD -- <paths>` exits 0 with empty stdout when no
    commit touches the paths — this must be treated as undeterminable, NOT
    epoch 0 (which would make fused-memory look infinitely stale).
    """
    wdog = _load_watchdog()

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert wdog._newest_fm_watched_commit_epoch() is None


def test_newest_fm_watched_commit_epoch_nonzero_rc_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_newest_fm_watched_commit_epoch returns None when git exits non-zero."""
    wdog = _load_watchdog()

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        return subprocess.CompletedProcess(
            cmd, 128, stdout="", stderr="fatal: not a git repository"
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert wdog._newest_fm_watched_commit_epoch() is None


def test_newest_fm_watched_commit_epoch_unparseable_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_newest_fm_watched_commit_epoch returns None when stdout is not an int."""
    wdog = _load_watchdog()

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        return subprocess.CompletedProcess(cmd, 0, stdout="not-an-epoch\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert wdog._newest_fm_watched_commit_epoch() is None


def test_newest_fm_watched_commit_epoch_broad_error_returns_none_and_warns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_newest_fm_watched_commit_epoch returns None (and warns) on a broad subprocess error."""
    wdog = _load_watchdog()
    logged: list[str] = []

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        raise RuntimeError("boom: git subprocess exploded")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(wdog, "log", lambda m: logged.append(m))

    assert wdog._newest_fm_watched_commit_epoch() is None
    assert any("WARNING" in m for m in logged), (
        f"a swallowed subprocess error must emit a WARNING via logger.warning: {logged}"
    )


# ---------------------------------------------------------------------------
# Part C: fm deploy-clock trio + --stamp-fm-deploy-clock CLI (step 7)
#
# fm's OWN deploy clock (data/fused-memory/last_redeploy_fused_memory.json):
# _read_last_fm_deploy_epoch() (fail-open {ts,iso} reader), _stamp_fm_deploy_clock()
# (atomic write, unlike restart-all-orchestrators.sh, restart-fused-memory.sh
# does NOT self-stamp), _within_fm_deploy_min_interval() (the gate predicate),
# and the `--stamp-fm-deploy-clock` CLI subcommand the detached restart chains
# on its verified exit-0.
# ---------------------------------------------------------------------------


def test_stamp_and_read_fm_deploy_clock_roundtrip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """_stamp_fm_deploy_clock writes a {ts,iso} body (creating the parent dir)
    that _read_last_fm_deploy_epoch reads back as float(ts)."""
    wdog = _load_watchdog()
    # Parent dir does NOT exist yet — the stamp must create it (makedirs).
    clock_file = tmp_path / "data" / "fused-memory" / "last_redeploy_fused_memory.json"
    assert not clock_file.parent.exists()
    monkeypatch.setattr(wdog, "FM_DEPLOY_CLOCK_PATH", str(clock_file))
    monkeypatch.setattr(wdog.time, "time", lambda: 1783000000.5)

    wdog._stamp_fm_deploy_clock()

    assert clock_file.exists(), "stamp must create the clock file and its parent dir"
    body = json.loads(clock_file.read_text())
    assert "ts" in body and "iso" in body, f"clock body must carry ts+iso: {body}"
    assert wdog._read_last_fm_deploy_epoch() == pytest.approx(1783000000.0)
    assert isinstance(wdog._read_last_fm_deploy_epoch(), float)


def test_read_last_fm_deploy_epoch_missing_file_returns_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """_read_last_fm_deploy_epoch returns None when the clock file is absent."""
    wdog = _load_watchdog()
    monkeypatch.setattr(wdog, "FM_DEPLOY_CLOCK_PATH", str(tmp_path / "absent.json"))
    assert wdog._read_last_fm_deploy_epoch() is None


def test_read_last_fm_deploy_epoch_corrupt_json_returns_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """_read_last_fm_deploy_epoch returns None (fail-open) on a partial/corrupt file."""
    wdog = _load_watchdog()
    clock_file = tmp_path / "corrupt.json"
    clock_file.write_text('{"ts": 178300')  # truncated / partially written
    monkeypatch.setattr(wdog, "FM_DEPLOY_CLOCK_PATH", str(clock_file))
    assert wdog._read_last_fm_deploy_epoch() is None


def test_within_fm_deploy_min_interval_true_when_recent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_within_fm_deploy_min_interval is True when now - last < FM_RESTART_MIN_INTERVAL_SECS."""
    wdog = _load_watchdog()
    monkeypatch.setattr(wdog, "FM_RESTART_MIN_INTERVAL_SECS", 28800)
    monkeypatch.setattr(wdog, "_read_last_fm_deploy_epoch", lambda: 1_000_000.0)
    monkeypatch.setattr(wdog.time, "time", lambda: 1_000_000.0 + 100.0)  # 100s ago

    assert wdog._within_fm_deploy_min_interval() is True


def test_within_fm_deploy_min_interval_false_when_elapsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_within_fm_deploy_min_interval is False once FM_RESTART_MIN_INTERVAL_SECS has elapsed."""
    wdog = _load_watchdog()
    monkeypatch.setattr(wdog, "FM_RESTART_MIN_INTERVAL_SECS", 28800)
    monkeypatch.setattr(wdog, "_read_last_fm_deploy_epoch", lambda: 1_000_000.0)
    monkeypatch.setattr(wdog.time, "time", lambda: 1_000_000.0 + 28800.0 + 1.0)

    assert wdog._within_fm_deploy_min_interval() is False


def test_within_fm_deploy_min_interval_false_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_within_fm_deploy_min_interval is False when FM_RESTART_MIN_INTERVAL_SECS<=0.

    0 disables the cap entirely — the clock file must not even be read.
    """
    wdog = _load_watchdog()
    monkeypatch.setattr(wdog, "FM_RESTART_MIN_INTERVAL_SECS", 0)
    monkeypatch.setattr(
        wdog,
        "_read_last_fm_deploy_epoch",
        lambda: pytest.fail("must not be consulted when the cap is disabled"),
    )

    assert wdog._within_fm_deploy_min_interval() is False


def test_within_fm_deploy_min_interval_false_when_clock_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_within_fm_deploy_min_interval is False when the clock is unreadable/absent.

    A never-deployed fm (or an unreadable clock) must not block the backstop
    indefinitely — fail toward running the backstop, not toward silence.
    """
    wdog = _load_watchdog()
    monkeypatch.setattr(wdog, "FM_RESTART_MIN_INTERVAL_SECS", 28800)
    monkeypatch.setattr(wdog, "_read_last_fm_deploy_epoch", lambda: None)

    assert wdog._within_fm_deploy_min_interval() is False


def test_cli_stamp_fm_deploy_clock_subcommand(monkeypatch: pytest.MonkeyPatch) -> None:
    """_cli(["--stamp-fm-deploy-clock"]) stamps exactly once, returns 0, and runs
    NONE of the liveness/staleness/report paths.

    This is the subcommand the detached _delegate_fm_restart chains after a
    verified restart (`restart-fused-memory.sh && <self> --stamp-fm-deploy-clock`),
    so it must do exactly one thing: stamp the fm clock.
    """
    wdog = _load_watchdog()
    stamped: list[None] = []

    monkeypatch.setattr(
        wdog, "_stamp_fm_deploy_clock", lambda: stamped.append(None), raising=False
    )
    monkeypatch.setattr(wdog, "main", lambda: pytest.fail("main() must not run under --stamp"))
    monkeypatch.setattr(
        wdog,
        "fused_memory_liveness_pass",
        lambda: pytest.fail("fused_memory_liveness_pass() must not run under --stamp"),
    )
    monkeypatch.setattr(
        wdog, "staleness_pass", lambda: pytest.fail("staleness_pass() must not run under --stamp")
    )
    monkeypatch.setattr(
        wdog,
        "fused_memory_staleness_pass",
        lambda: pytest.fail("fused_memory_staleness_pass() must not run under --stamp"),
        raising=False,
    )
    monkeypatch.setattr(wdog, "report", lambda: pytest.fail("report() must not run under --stamp"))

    exit_code = wdog._cli(["--stamp-fm-deploy-clock"])

    assert exit_code == 0
    assert stamped == [None], f"Expected exactly one stamp call, got {stamped}"


# ---------------------------------------------------------------------------
# Part C: _delegate_fm_restart() (step 9)
#
# fm sibling of _delegate_fleet_restart: detached systemd-run with a fixed
# transient unit name (overlap guard) + fail-soft registration. Diverges from
# the orchestrator path in TWO ways: (1) it invokes restart-fused-memory.sh in
# its DEFAULT defer-if-busy mode (no --now/--drain), and (2) it chains
# `&& <self> --stamp-fm-deploy-clock` so the fm clock is stamped only on the
# restart script's verified exit-0 (restart-fused-memory.sh, unlike
# restart-all-orchestrators.sh, does not self-stamp).
# ---------------------------------------------------------------------------


def test_delegate_fm_restart_argv_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """_delegate_fm_restart fires a detached, named systemd-run that runs
    restart-fused-memory.sh (default defer-if-busy mode) and chains the stamp
    on its verified exit-0."""
    wdog = _load_watchdog()
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    wdog._delegate_fm_restart()

    assert len(calls) == 1, f"Expected exactly one subprocess.run call, got {calls}"
    argv = calls[0]
    assert argv[:2] == ["systemd-run", "--user"], f"argv must start with systemd-run --user: {argv}"
    assert "--collect" in argv, f"argv must include --collect: {argv}"
    assert "--no-block" in argv, f"argv must include --no-block (detached): {argv}"
    assert "--unit=fm-staleness-redeploy.service" in argv, (
        f"argv must fire the fixed transient unit name (the overlap guard): {argv}"
    )

    # systemd-run --user does NOT propagate this process's env into the detached
    # unit, so the chained --stamp-fm-deploy-clock must be pinned to the reader's
    # resolved clock path via --setenv or it would default a divergent path.
    assert f"--setenv=FM_DEPLOY_CLOCK={wdog.FM_DEPLOY_CLOCK_PATH}" in argv, (
        "argv must forward the reader's resolved FM_DEPLOY_CLOCK_PATH via "
        f"--setenv so the detached stamp writes the same file the reader reads: {argv}"
    )

    # The bash -c payload chains restart-fused-memory.sh && <self> --stamp.
    payload = next(
        (a for a in argv if "restart-fused-memory.sh" in a), None
    )
    assert payload is not None, f"argv must carry a restart-fused-memory.sh payload: {argv}"
    assert "&&" in payload, (
        f"payload must chain the stamp with `&&` so it runs only on exit-0: {payload!r}"
    )
    assert "--stamp-fm-deploy-clock" in payload, (
        f"payload must chain the fm-clock stamp subcommand: {payload!r}"
    )
    assert "--now" not in payload, (
        f"payload must NOT pass --now (uses the default defer-if-busy path): {payload!r}"
    )
    assert "--drain" not in payload, (
        f"payload must NOT pass --drain (that is the orchestrator path): {payload!r}"
    )

    # The referenced restart script must actually exist on disk.
    assert (REPO_ROOT / "scripts" / "restart-fused-memory.sh").exists(), (
        "scripts/restart-fused-memory.sh must exist on disk (task 2703 δ dependency)"
    )


def test_delegate_fm_restart_forwards_clock_override_via_setenv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A FM_DEPLOY_CLOCK override is forwarded into the detached unit via --setenv.

    systemd-run --user runs the transient unit under the systemd user manager's
    environment, not the watchdog process's — so without an explicit --setenv the
    chained --stamp-fm-deploy-clock would recompute FM_DEPLOY_CLOCK_PATH from the
    session env and default it, diverging from the overridden path the in-process
    reader (_within_fm_deploy_min_interval) consults. Forwarding keeps
    writer==reader so the min-interval cap actually engages under an override.
    """
    monkeypatch.setenv("FM_DEPLOY_CLOCK", "/tmp/custom_fm_clock.json")
    wdog = _load_watchdog()
    # Sanity: the reader resolved the override at import.
    assert wdog.FM_DEPLOY_CLOCK_PATH == "/tmp/custom_fm_clock.json"

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    wdog._delegate_fm_restart()

    assert len(calls) == 1, f"Expected exactly one subprocess.run call, got {calls}"
    argv = calls[0]
    # The forwarded --setenv value must equal the reader's resolved override —
    # writer (detached stamp) and reader consult the identical file.
    assert "--setenv=FM_DEPLOY_CLOCK=/tmp/custom_fm_clock.json" in argv, (
        f"argv must forward the FM_DEPLOY_CLOCK override into the detached unit: {argv}"
    )
    assert f"--setenv=FM_DEPLOY_CLOCK={wdog.FM_DEPLOY_CLOCK_PATH}" in argv, (
        "the forwarded --setenv value must track the reader's FM_DEPLOY_CLOCK_PATH"
    )


def test_delegate_fm_restart_swallows_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """_delegate_fm_restart must not raise if the systemd-run call times out."""
    wdog = _load_watchdog()
    log_messages: list[str] = []

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        raise subprocess.TimeoutExpired(cmd, 10)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(wdog, "log", lambda m: log_messages.append(m))

    # Must not raise
    wdog._delegate_fm_restart()

    assert len(log_messages) >= 1, "a systemd-run timeout must be logged"


def test_delegate_fm_restart_swallows_missing_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    """_delegate_fm_restart must not raise if systemd-run is not on PATH."""
    wdog = _load_watchdog()
    log_messages: list[str] = []

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        raise FileNotFoundError(2, "No such file or directory", "systemd-run")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(wdog, "log", lambda m: log_messages.append(m))

    # Must not raise
    wdog._delegate_fm_restart()

    assert len(log_messages) >= 1, "a missing systemd-run binary must be logged"


# ---------------------------------------------------------------------------
# Part C: fused_memory_staleness_pass() (step 11)
#
# Single-unit mirror of staleness_pass() over FUSED_MEMORY_UNIT: min-interval
# gate (fm clock) -> commit epoch -> commit-grace head-start -> enabled /
# startup-grace / ActiveEnterTimestamp-vs-commit -> delegate once. Every helper
# is stubbed directly (the α-style unit level); the fm-deploy min-interval gate
# is neutralized to False except where it is the subject under test.
# ---------------------------------------------------------------------------


def test_fused_memory_staleness_pass_core_stale_delegates_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Case 1: an enabled, past-startup-grace fused-memory.service whose
    ActiveEnterTimestamp predates the newest fm-watched commit delegates a
    fused-memory restart exactly once and logs a WARNING naming the unit."""
    wdog = _load_watchdog()
    delegated: list[None] = []
    log_messages: list[str] = []

    now = 2_000_000_000.0
    commit_epoch = int(now) - wdog.STALENESS_GRACE_SECS - 100  # older than grace

    monkeypatch.setattr(wdog, "_within_fm_deploy_min_interval", lambda: False)
    monkeypatch.setattr(wdog, "_newest_fm_watched_commit_epoch", lambda: commit_epoch)
    monkeypatch.setattr(wdog.time, "time", lambda: now)
    monkeypatch.setattr(wdog, "is_unit_enabled", lambda _u: True)
    monkeypatch.setattr(wdog, "_unit_start_elapsed_secs", lambda _u: 300.0)
    monkeypatch.setattr(wdog, "_unit_active_enter_epoch", lambda _u: commit_epoch - 100)  # stale
    monkeypatch.setattr(wdog, "_delegate_fm_restart", lambda: delegated.append(None))
    monkeypatch.setattr(wdog, "log", lambda m: log_messages.append(m))

    wdog.fused_memory_staleness_pass()

    assert len(delegated) == 1, f"Expected exactly one fm restart delegation, got {len(delegated)}"
    assert any(("WARNING" in m and wdog.FUSED_MEMORY_UNIT in m) for m in log_messages), (
        f"Expected a WARNING log naming {wdog.FUSED_MEMORY_UNIT}: {log_messages}"
    )


def test_fused_memory_staleness_pass_fresh_does_not_delegate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Case 2: ActiveEnterTimestamp >= commit (fm already running the newest
    code) → no delegation."""
    wdog = _load_watchdog()
    delegated: list[None] = []

    now = 2_000_000_000.0
    commit_epoch = int(now) - wdog.STALENESS_GRACE_SECS - 100

    monkeypatch.setattr(wdog, "_within_fm_deploy_min_interval", lambda: False)
    monkeypatch.setattr(wdog, "_newest_fm_watched_commit_epoch", lambda: commit_epoch)
    monkeypatch.setattr(wdog.time, "time", lambda: now)
    monkeypatch.setattr(wdog, "is_unit_enabled", lambda _u: True)
    monkeypatch.setattr(wdog, "_unit_start_elapsed_secs", lambda _u: 300.0)
    monkeypatch.setattr(wdog, "_unit_active_enter_epoch", lambda _u: commit_epoch + 100)  # fresh
    monkeypatch.setattr(wdog, "_delegate_fm_restart", lambda: delegated.append(None))
    monkeypatch.setattr(wdog, "log", lambda _m: None)

    wdog.fused_memory_staleness_pass()

    assert delegated == [], f"A fresh unit must not delegate; got {delegated}"


def test_fused_memory_staleness_pass_within_min_interval_logs_inside_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Case 3a: within fm's min-interval → early return (no commit read, no
    delegate) and a throttled skip-log line emitted at a bucket boundary."""
    wdog = _load_watchdog()
    log_messages: list[str] = []

    monkeypatch.setattr(wdog, "_within_fm_deploy_min_interval", lambda: True)
    monkeypatch.setattr(wdog.time, "time", lambda: wdog.SKIP_LOG_INTERVAL_SECS * 1000.0)
    monkeypatch.setattr(
        wdog,
        "_newest_fm_watched_commit_epoch",
        lambda: pytest.fail("must not read commit when the fm-deploy gate is closed"),
    )
    monkeypatch.setattr(
        wdog,
        "_delegate_fm_restart",
        lambda: pytest.fail("must not delegate when the fm-deploy gate is closed"),
    )
    monkeypatch.setattr(wdog, "log", lambda m: log_messages.append(m))

    wdog.fused_memory_staleness_pass()

    assert any(
        "skip" in m and str(wdog.FM_RESTART_MIN_INTERVAL_SECS) in m for m in log_messages
    ), f"Expected a skip log naming the fm-deploy min-interval: {log_messages}"


def test_fused_memory_staleness_pass_within_min_interval_suppresses_log_outside_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Case 3b: within fm's min-interval but outside the log bucket → still
    returns early, but emits NO skip line (rate-limit throttle)."""
    wdog = _load_watchdog()
    log_messages: list[str] = []

    monkeypatch.setattr(wdog, "_within_fm_deploy_min_interval", lambda: True)
    monkeypatch.setattr(
        wdog.time,
        "time",
        lambda: wdog.SKIP_LOG_INTERVAL_SECS * 1000.0 + wdog.SKIP_LOG_INTERVAL_SECS / 2,
    )
    monkeypatch.setattr(
        wdog,
        "_newest_fm_watched_commit_epoch",
        lambda: pytest.fail("must not read commit when the fm-deploy gate is closed"),
    )
    monkeypatch.setattr(wdog, "log", lambda m: log_messages.append(m))

    wdog.fused_memory_staleness_pass()

    assert log_messages == [], f"Expected no skip log outside the bucket: {log_messages}"


def test_fused_memory_staleness_pass_commit_grace_suppresses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Case 4: a commit younger than STALENESS_GRACE_SECS gives the fm
    event-driven coordinator its head start — no delegation."""
    wdog = _load_watchdog()
    delegated: list[None] = []

    now = 2_000_000_000.0
    commit_epoch = int(now) - 300  # younger than STALENESS_GRACE_SECS=1800

    monkeypatch.setattr(wdog, "_within_fm_deploy_min_interval", lambda: False)
    monkeypatch.setattr(wdog, "_newest_fm_watched_commit_epoch", lambda: commit_epoch)
    monkeypatch.setattr(wdog.time, "time", lambda: now)
    monkeypatch.setattr(
        wdog,
        "_unit_active_enter_epoch",
        lambda _u: pytest.fail("must not probe activation inside the commit-grace window"),
    )
    monkeypatch.setattr(wdog, "_delegate_fm_restart", lambda: delegated.append(None))
    monkeypatch.setattr(wdog, "log", lambda _m: None)

    wdog.fused_memory_staleness_pass()

    assert delegated == [], f"A young commit must suppress delegation; got {delegated}"


def test_fused_memory_staleness_pass_noop_when_commit_epoch_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Case 5: an undeterminable commit epoch is a complete no-op (no
    enabled/active probe, no delegate)."""
    wdog = _load_watchdog()

    monkeypatch.setattr(wdog, "_within_fm_deploy_min_interval", lambda: False)
    monkeypatch.setattr(wdog, "_newest_fm_watched_commit_epoch", lambda: None)
    monkeypatch.setattr(
        wdog,
        "is_unit_enabled",
        lambda _u: pytest.fail("must not probe enabled when commit epoch is None"),
    )
    monkeypatch.setattr(
        wdog,
        "_delegate_fm_restart",
        lambda: pytest.fail("must not delegate when commit epoch is None"),
    )
    monkeypatch.setattr(wdog, "log", lambda _m: None)

    wdog.fused_memory_staleness_pass()


def test_fused_memory_staleness_pass_skips_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Case 6: a disabled fused-memory.service (operator intent) → no
    activation probe, no delegate."""
    wdog = _load_watchdog()
    delegated: list[None] = []

    now = 2_000_000_000.0
    commit_epoch = int(now) - wdog.STALENESS_GRACE_SECS - 100

    monkeypatch.setattr(wdog, "_within_fm_deploy_min_interval", lambda: False)
    monkeypatch.setattr(wdog, "_newest_fm_watched_commit_epoch", lambda: commit_epoch)
    monkeypatch.setattr(wdog.time, "time", lambda: now)
    monkeypatch.setattr(wdog, "is_unit_enabled", lambda _u: False)
    monkeypatch.setattr(
        wdog,
        "_unit_active_enter_epoch",
        lambda _u: pytest.fail("must not probe activation for a disabled unit"),
    )
    monkeypatch.setattr(wdog, "_delegate_fm_restart", lambda: delegated.append(None))
    monkeypatch.setattr(wdog, "log", lambda _m: None)

    wdog.fused_memory_staleness_pass()

    assert delegated == [], f"A disabled unit must not delegate; got {delegated}"


def test_fused_memory_staleness_pass_skips_startup_grace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Case 7: a just-restarted fm within STARTUP_GRACE_SECS → no delegate
    (avoid an indefinite restart loop before the new version converges)."""
    wdog = _load_watchdog()
    delegated: list[None] = []

    now = 2_000_000_000.0
    commit_epoch = int(now) - wdog.STALENESS_GRACE_SECS - 100

    monkeypatch.setattr(wdog, "_within_fm_deploy_min_interval", lambda: False)
    monkeypatch.setattr(wdog, "_newest_fm_watched_commit_epoch", lambda: commit_epoch)
    monkeypatch.setattr(wdog.time, "time", lambda: now)
    monkeypatch.setattr(wdog, "is_unit_enabled", lambda _u: True)
    monkeypatch.setattr(wdog, "_unit_start_elapsed_secs", lambda _u: 30.0)  # < 120s grace
    monkeypatch.setattr(
        wdog,
        "_unit_active_enter_epoch",
        lambda _u: pytest.fail("must not probe activation inside the startup-grace window"),
    )
    monkeypatch.setattr(wdog, "_delegate_fm_restart", lambda: delegated.append(None))
    monkeypatch.setattr(wdog, "log", lambda _m: None)

    wdog.fused_memory_staleness_pass()

    assert delegated == [], f"A unit within startup grace must not delegate; got {delegated}"


def test_fused_memory_staleness_pass_active_none_does_not_delegate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Case 8: an undeterminable ActiveEnterTimestamp (None) → no delegate
    (don't guess staleness)."""
    wdog = _load_watchdog()
    delegated: list[None] = []

    now = 2_000_000_000.0
    commit_epoch = int(now) - wdog.STALENESS_GRACE_SECS - 100

    monkeypatch.setattr(wdog, "_within_fm_deploy_min_interval", lambda: False)
    monkeypatch.setattr(wdog, "_newest_fm_watched_commit_epoch", lambda: commit_epoch)
    monkeypatch.setattr(wdog.time, "time", lambda: now)
    monkeypatch.setattr(wdog, "is_unit_enabled", lambda _u: True)
    monkeypatch.setattr(wdog, "_unit_start_elapsed_secs", lambda _u: 300.0)
    monkeypatch.setattr(wdog, "_unit_active_enter_epoch", lambda _u: None)  # undeterminable
    monkeypatch.setattr(wdog, "_delegate_fm_restart", lambda: delegated.append(None))
    monkeypatch.setattr(wdog, "log", lambda _m: None)

    wdog.fused_memory_staleness_pass()

    assert delegated == [], f"A None ActiveEnterTimestamp must not delegate; got {delegated}"


def test_fused_memory_staleness_pass_isolates_probe_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Case 9: an exception raised inside the probe chain is caught (no raise),
    mirroring fused_memory_liveness_pass()'s single-unit try/except isolation."""
    wdog = _load_watchdog()

    now = 2_000_000_000.0
    commit_epoch = int(now) - wdog.STALENESS_GRACE_SECS - 100

    def _boom(_u: str) -> bool:
        raise RuntimeError("systemctl exploded")

    monkeypatch.setattr(wdog, "_within_fm_deploy_min_interval", lambda: False)
    monkeypatch.setattr(wdog, "_newest_fm_watched_commit_epoch", lambda: commit_epoch)
    monkeypatch.setattr(wdog.time, "time", lambda: now)
    monkeypatch.setattr(wdog, "is_unit_enabled", _boom)
    monkeypatch.setattr(wdog, "log", lambda _m: None)

    # Must not raise
    wdog.fused_memory_staleness_pass()


def test_fused_memory_staleness_pass_e2e_converges(monkeypatch: pytest.MonkeyPatch) -> None:
    """Case 10 (I6): a first pass with ActiveEnterTimestamp<commit delegates
    once; after the restart advances ActiveEnterTimestamp past the commit, the
    next pass no-ops — stateless self-heal, no stored flap state."""
    wdog = _load_watchdog()
    delegated: list[None] = []

    now = 2_000_000_000.0
    commit_epoch = int(now) - wdog.STALENESS_GRACE_SECS - 100
    active = {"epoch": commit_epoch - 100}  # starts stale

    monkeypatch.setattr(wdog, "_within_fm_deploy_min_interval", lambda: False)
    monkeypatch.setattr(wdog, "_newest_fm_watched_commit_epoch", lambda: commit_epoch)
    monkeypatch.setattr(wdog.time, "time", lambda: now)
    monkeypatch.setattr(wdog, "is_unit_enabled", lambda _u: True)
    monkeypatch.setattr(wdog, "_unit_start_elapsed_secs", lambda _u: 300.0)
    monkeypatch.setattr(wdog, "_unit_active_enter_epoch", lambda _u: active["epoch"])
    monkeypatch.setattr(wdog, "_delegate_fm_restart", lambda: delegated.append(None))
    monkeypatch.setattr(wdog, "log", lambda _m: None)

    wdog.fused_memory_staleness_pass()
    assert len(delegated) == 1, f"first pass must delegate once; got {len(delegated)}"

    # Restart refreshed the unit — ActiveEnterTimestamp now past the commit.
    delegated.clear()
    active["epoch"] = commit_epoch + 50
    wdog.fused_memory_staleness_pass()
    assert delegated == [], f"a refreshed unit must self-clear; got {len(delegated)} delegation(s)"


# ---------------------------------------------------------------------------
# log() bounding tests (task 3392 — follow-up from task 3308 / commit
# 87ff5d1870, which fixed the identical unbounded-systemd-cat shape in
# scripts/dashboard-watchdog.py). log() was the one subprocess call in this
# file with no timeout and no exception handling at all: a systemd-cat
# blocked on a stuck journald or a full /run would hang the tick forever,
# and since orchestrator-watchdog.service is Type=oneshot (TimeoutStartSec
# disabled by default for that type) driven by a timer whose OnUnitActiveSec
# is measured from this unit's LAST ACTIVATION, that hang would never let the
# timer fire again — supervision stops with no signal, arriving through the
# LOGGING path.
# ---------------------------------------------------------------------------


def test_log_bounds_systemd_cat_with_a_five_second_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """log() must pass an explicit timeout=5 to its systemd-cat subprocess call.

    Without it, systemd-cat inherits no bound at all and a wedged journald or
    a full /run hangs the oneshot tick forever.
    """
    wdog = _load_watchdog()
    seen_kwargs: list[dict] = []

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        seen_kwargs.append(kwargs)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    wdog.log("hello")

    assert len(seen_kwargs) == 1, f"expected exactly one subprocess.run call: {seen_kwargs}"
    assert seen_kwargs[0].get("timeout") == 5, (
        f"log() must bound systemd-cat with timeout=5, got {seen_kwargs[0]!r}"
    )


def test_log_falls_through_to_stderr_on_missing_binary(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing systemd-cat binary (OSError) must not raise — log() must fall
    back to printing on stderr, which StandardError=journal routes to the same
    journal.
    """
    wdog = _load_watchdog()

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        raise FileNotFoundError(2, "No such file or directory", "systemd-cat")

    monkeypatch.setattr(subprocess, "run", fake_run)

    wdog.log("hello")  # must not raise

    captured = capsys.readouterr()
    assert "hello" in captured.err, f"expected the message on stderr, got: {captured!r}"


def test_log_falls_through_to_stderr_on_timeout(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A systemd-cat call that exceeds its bound (TimeoutExpired, a
    subprocess.SubprocessError) must not raise — this is the exact wedge
    the whole-tick TimeoutStartSec backstop exists to catch, and this handler
    is the first line of defense: the tick continues instead of hanging here.
    """
    wdog = _load_watchdog()

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        raise subprocess.TimeoutExpired(cmd, 5)

    monkeypatch.setattr(subprocess, "run", fake_run)

    wdog.log("hello")  # must not raise

    captured = capsys.readouterr()
    assert "hello" in captured.err, f"expected the message on stderr, got: {captured!r}"


def test_log_swallows_only_os_and_subprocess_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """log() must not widen its except clause into a bare `except Exception`:
    an unrelated bug (e.g. a TypeError from a bad call site) must still
    surface rather than being silently swallowed alongside the two
    tooling-failure cases it is meant to catch.
    """
    wdog = _load_watchdog()

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        raise ValueError("not a systemd-cat failure at all")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(ValueError):
        wdog.log("hello")


def test_log_never_raises_when_the_stderr_fallback_itself_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fallback print is best-effort too: stderr can be a broken pipe or a
    full/failing journal socket, and that OSError must not escape log().

    main()'s per-unit handler calls log() from inside its ``except Exception``
    block, so an exception escaping log() there aborts the for-loop and leaves
    the remaining WATCHED units unprobed for that tick.
    """
    wdog = _load_watchdog()

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        raise FileNotFoundError("systemd-cat not found")

    class _BrokenStderr:
        def write(self, _s: str) -> int:
            raise BrokenPipeError("stderr is gone too")

        def flush(self) -> None:
            raise BrokenPipeError("stderr is gone too")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(wdog.sys, "stderr", _BrokenStderr())

    wdog.log("hello")  # must not raise — both journal routes are gone


# ---------------------------------------------------------------------------
# orchestrator-watchdog.service TimeoutStartSec pin (task 3392)
# ---------------------------------------------------------------------------


def _unit_sections(content: str) -> dict[str, list[str]]:
    """Split unit-file text into {section_name: [lines]} (header line excluded).

    Local copy of tests/scripts/test_orchestrator_service_files.py's
    ``_parse_sections``: that module holds the other orchestrator-watchdog.service
    pins and is where this assertion ultimately belongs, but pytest runs here with
    ``--import-mode=importlib`` and tests/scripts/ is not a package, so a sibling
    test module cannot be imported. Fold this back into ``_parse_sections`` if the
    pin ever moves next to ``test_watchdog_service_structure``.
    """
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in content.splitlines():
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    return sections


def test_service_bounds_the_whole_tick() -> None:
    """TimeoutStartSec must be present under [Service], finite, and above the
    script's own worst-case sequential subprocess bound.

    systemd disables TimeoutStartSec for Type=oneshot by default, and the
    timer's OnUnitActiveSec measures from the unit's last activation — so a
    tick that never returns does not merely run late, it ENDS supervision:
    the timer never re-triggers and nothing reports it. The bound must clear
    the script's own worst realistic sequential path (main() walking all 7
    WATCHED units plus fused_memory_liveness_pass plus the staleness passes,
    each unit's own children already individually bounded) — roughly 1100s —
    or the whole-tick kill could land mid-way through a legitimate multi-unit
    restart rather than only on a genuine wedge.

    systemd honours TimeoutStartSec= only in [Service]; under [Unit] or
    [Install] it is silently ignored, which would leave the tick unbounded
    again while a presence-only check still passed — so the section is
    asserted, not just the line.
    """
    service_path = REPO_ROOT / "scripts" / "orchestrator-watchdog.service"
    unit = service_path.read_text(encoding="utf-8")
    sections = _unit_sections(unit)

    def _values(lines: list[str]) -> list[str]:
        return [
            ln.split("=", 1)[1].strip()
            for ln in lines
            if ln.startswith("TimeoutStartSec=")
        ]

    misplaced = {
        name: _values(lines)
        for name, lines in sections.items()
        if name != "Service" and _values(lines)
    }
    assert not misplaced, (
        f"TimeoutStartSec= outside [Service] is silently ignored by systemd, "
        f"leaving the tick unbounded: {misplaced}"
    )

    values = _values(sections.get("Service", []))
    assert len(values) == 1, (
        f"expected exactly one TimeoutStartSec= under [Service]: {values}"
    )
    assert values[0].isdigit(), (
        f"TimeoutStartSec={values[0]!r} is not a plain seconds count; "
        "'infinity' would leave the tick unbounded and a unit suffix is not "
        "parsed here — keep it a bare integer"
    )
    assert int(values[0]) > 1100, (
        f"TimeoutStartSec={values[0]} does not clear the script's own "
        "~1100s worst-case sequential path (7 WATCHED units + fm liveness + "
        "the staleness passes), so the whole-tick kill could land on a "
        "legitimate multi-unit restart tick instead of only a genuine wedge"
    )

