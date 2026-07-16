"""Unit tests for scripts/orchestrator-watchdog.py.

The watchdog module has a hyphenated filename so it cannot be imported via
``import orchestrator_watchdog``.  We use importlib to load it by file path.

No live systemd runtime is needed — all subprocess.run calls are monkeypatched.
"""

import importlib.util
import json
import pathlib
import re
import subprocess
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
    monkeypatch.setenv("ORCH_CONFIG_PATH", str(REPO_ROOT / "orchestrator" / "config.yaml"))
    wdog = _load_watchdog()
    assert wdog.ORCH_RESTART_MIN_INTERVAL_SECS == pytest.approx(
        OrchestratorConfig().orchestrator_restart_min_interval_secs
    )


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
    assert wdog.FLEET_DEPLOY_CLOCK_PATH == expected_watchdog_path

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

