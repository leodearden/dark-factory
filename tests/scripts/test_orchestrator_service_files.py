"""File-content tests for the orchestrator systemd service files.

These tests read the source-controlled service definition files directly —
no systemd runtime is required.  They guard the shape of:
  - scripts/dark-factory-orchestrator.service
  - scripts/reify-orchestrator.service
  - scripts/orchestrator-watchdog.service
  - scripts/orchestrator-watchdog.timer

See also:
  - tests/scripts/test_dashboard_service_template.py — pattern reference
"""

import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).parents[2]
DF_SERVICE = REPO_ROOT / "scripts" / "dark-factory-orchestrator.service"
REIFY_SERVICE = REPO_ROOT / "scripts" / "reify-orchestrator.service"
WATCHDOG_SERVICE = REPO_ROOT / "scripts" / "orchestrator-watchdog.service"
WATCHDOG_TIMER = REPO_ROOT / "scripts" / "orchestrator-watchdog.timer"


# ---------------------------------------------------------------------------
# dark-factory-orchestrator.service
# ---------------------------------------------------------------------------


def test_dark_factory_orchestrator_service_structure() -> None:
    """scripts/dark-factory-orchestrator.service must have the required systemd shape."""
    content = DF_SERVICE.read_text(encoding="utf-8")

    # --- sections ---
    assert "[Unit]" in content
    assert "[Service]" in content
    assert "[Install]" in content

    # --- [Unit] ---
    assert "Dark Factory" in content, "Description must mention 'Dark Factory'"
    # After= must chain all four dependencies
    after_line = next(
        (ln for ln in content.splitlines() if ln.startswith("After=")), None
    )
    assert after_line is not None, "After= line not found"
    assert "network.target" in after_line
    assert "fused-memory.service" in after_line
    assert "reify-jobserver.service" in after_line
    assert "pytest-jobserver.service" in after_line
    assert "Requires=fused-memory.service" in content

    # --- [Service] ---
    assert "Type=simple" in content
    assert "WorkingDirectory=/home/leo/src/dark-factory" in content
    assert (
        "uv run --project orchestrator orchestrator run --config /home/leo/src/dark-factory/orchestrator/config.yaml"
        in content
    ), "ExecStart must invoke the orchestrator with the df config"
    assert "Restart=on-failure" in content
    assert "RestartSec=10" in content
    assert "RestartMaxDelaySec=60" in content
    assert "StartLimitIntervalSec=600" in content
    assert "StartLimitBurst=10" in content
    assert "TimeoutStopSec=30" in content
    assert "StandardOutput=journal" in content
    assert "StandardError=journal" in content

    # --- [Install] ---
    assert "WantedBy=default.target" in content


# ---------------------------------------------------------------------------
# reify-orchestrator.service
# ---------------------------------------------------------------------------


def test_reify_orchestrator_service_structure() -> None:
    """scripts/reify-orchestrator.service must have the required systemd shape."""
    content = REIFY_SERVICE.read_text(encoding="utf-8")

    # --- sections ---
    assert "[Unit]" in content
    assert "[Service]" in content
    assert "[Install]" in content

    # --- [Unit] ---
    assert "Reify" in content, "Description must mention 'Reify'"
    after_line = next(
        (ln for ln in content.splitlines() if ln.startswith("After=")), None
    )
    assert after_line is not None, "After= line not found"
    assert "network.target" in after_line
    assert "fused-memory.service" in after_line
    assert "reify-jobserver.service" in after_line
    assert "pytest-jobserver.service" in after_line
    assert "Requires=fused-memory.service" in content

    # --- [Service] ---
    assert "Type=simple" in content
    # WorkingDirectory MUST be dark-factory (uv --project resolves relative to cwd;
    # /home/leo/src/reify/orchestrator/ does not exist).
    assert "WorkingDirectory=/home/leo/src/dark-factory" in content, (
        "WorkingDirectory must be /home/leo/src/dark-factory for the reify unit too "
        "(uv --project orchestrator requires the orchestrator/ subdir to live under cwd)"
    )
    assert (
        "uv run --project orchestrator orchestrator run --config /home/leo/src/reify/orchestrator.yaml"
        in content
    ), "ExecStart must invoke the orchestrator with the reify config"
    assert "Restart=on-failure" in content
    assert "RestartSec=10" in content
    assert "RestartMaxDelaySec=60" in content
    assert "StartLimitIntervalSec=600" in content
    assert "StartLimitBurst=10" in content
    assert "TimeoutStopSec=30" in content
    assert "StandardOutput=journal" in content
    assert "StandardError=journal" in content

    # --- [Install] ---
    assert "WantedBy=default.target" in content


def test_reify_and_df_differ_only_in_config_and_description() -> None:
    """The two orchestrator service files must be identical except Description and --config path.

    This guards the 'same shape' invariant: any structural drift (missing key,
    different Restart policy, etc.) that appears in one but not the other will
    break this test.
    """
    df_lines = DF_SERVICE.read_text(encoding="utf-8").splitlines()
    reify_lines = REIFY_SERVICE.read_text(encoding="utf-8").splitlines()

    assert len(df_lines) == len(reify_lines), (
        f"Service files have different line counts: df={len(df_lines)} reify={len(reify_lines)}"
    )

    diff_lines: list[tuple[int, str, str]] = []
    for i, (dl, rl) in enumerate(zip(df_lines, reify_lines)):
        if dl != rl:
            diff_lines.append((i + 1, dl, rl))

    # Classify each diff — must be either a Description or an ExecStart (--config) line
    allowed_df_fragments = {
        "Dark Factory Orchestrator",
        "/home/leo/src/dark-factory/orchestrator/config.yaml",
    }
    allowed_reify_fragments = {
        "Reify Orchestrator",
        "/home/leo/src/reify/orchestrator.yaml",
    }

    unexpected: list[tuple[int, str, str]] = []
    for lineno, dl, rl in diff_lines:
        df_ok = any(frag in dl for frag in allowed_df_fragments)
        reify_ok = any(frag in rl for frag in allowed_reify_fragments)
        if not (df_ok and reify_ok):
            unexpected.append((lineno, dl, rl))

    assert not unexpected, (
        "Unexpected differences between df and reify service files "
        "(only Description and --config path should differ):\n"
        + "\n".join(f"  line {n}:\n    df:    {d!r}\n    reify: {r!r}" for n, d, r in unexpected)
    )


# ---------------------------------------------------------------------------
# orchestrator-watchdog.timer
# ---------------------------------------------------------------------------


def test_watchdog_timer_structure() -> None:
    """scripts/orchestrator-watchdog.timer must fire every 60s."""
    content = WATCHDOG_TIMER.read_text(encoding="utf-8")

    assert "[Unit]" in content
    assert "[Timer]" in content
    assert "[Install]" in content

    # Description must hint at the 60-second interval
    desc_line = next(
        (ln for ln in content.splitlines() if ln.startswith("Description=")), None
    )
    assert desc_line is not None, "Description= line not found"
    assert "60" in desc_line or "every 60" in desc_line.lower(), (
        f"Description must mention '60' to document the probe interval: {desc_line!r}"
    )

    assert "OnBootSec=30" in content
    assert "OnUnitActiveSec=60" in content
    assert "WantedBy=timers.target" in content


# ---------------------------------------------------------------------------
# orchestrator-watchdog.service
# ---------------------------------------------------------------------------


def test_watchdog_service_structure() -> None:
    """scripts/orchestrator-watchdog.service must be a oneshot launcher for the Python watchdog."""
    content = WATCHDOG_SERVICE.read_text(encoding="utf-8")

    assert "[Unit]" in content
    assert "[Service]" in content

    desc_line = next(
        (ln for ln in content.splitlines() if ln.startswith("Description=")), None
    )
    assert desc_line is not None, "Description= line not found"

    assert "Type=oneshot" in content
    # ExecStart must be the absolute path to the Python script (single token, no sh -c)
    exec_line = next(
        (ln for ln in content.splitlines() if ln.startswith("ExecStart=")), None
    )
    assert exec_line is not None, "ExecStart= line not found"
    assert exec_line == "ExecStart=/home/leo/src/dark-factory/scripts/orchestrator-watchdog.py", (
        f"ExecStart must be the bare absolute path to orchestrator-watchdog.py, got: {exec_line!r}"
    )
