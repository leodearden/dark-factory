"""File-content tests for the orchestrator systemd service files.

These tests read the source-controlled service definition files directly —
no systemd runtime is required.  They guard the shape of:
  - scripts/orchestrator-dark-factory.service
  - scripts/orchestrator-reify.service
  - scripts/orchestrator-watchdog.service
  - scripts/orchestrator-watchdog.timer

See also:
  - tests/scripts/test_dashboard_service_template.py — pattern reference
"""

import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).parents[2]
DF_SERVICE = REPO_ROOT / "scripts" / "orchestrator-dark-factory.service"
REIFY_SERVICE = REPO_ROOT / "scripts" / "orchestrator-reify.service"
AUTOPILOT_SERVICE = REPO_ROOT / "scripts" / "orchestrator-autopilot-video.service"
WATCHDOG_SERVICE = REPO_ROOT / "scripts" / "orchestrator-watchdog.service"
WATCHDOG_TIMER = REPO_ROOT / "scripts" / "orchestrator-watchdog.timer"


# ---------------------------------------------------------------------------
# orchestrator-dark-factory.service
# ---------------------------------------------------------------------------


def test_dark_factory_orchestrator_service_structure() -> None:
    """scripts/orchestrator-dark-factory.service must have the required systemd shape."""
    content = DF_SERVICE.read_text(encoding="utf-8")

    # --- sections ---
    assert "[Unit]" in content
    assert "[Service]" in content
    assert "[Install]" in content

    # --- [Unit] ---
    # After= must chain all four dependencies
    after_line = next(
        (ln for ln in content.splitlines() if ln.startswith("After=")), None
    )
    assert after_line is not None, "After= line not found"
    assert "network.target" in after_line
    assert "fused-memory.service" in after_line
    assert "reify-jobserver.service" in after_line
    assert "pytest-jobserver.service" in after_line
    # Wants=, NOT Requires=: a hard Requires= turns a single fused-memory
    # first-start failure (boot race) into a permanent cancel of our start
    # job — systemd never retries a dependency-failed job. See 2026-05-27
    # post-powercut hardening.
    assert "Wants=fused-memory.service" in content
    assert "Requires=fused-memory.service" not in content, (
        "Requires=fused-memory.service is a regression — use Wants= so a "
        "transient fused-memory blip doesn't permanently cancel our start job"
    )

    # --- [Service] ---
    assert "Type=simple" in content
    assert "WorkingDirectory=/home/leo/src/dark-factory" in content
    # Port-wait ExecStartPre: gate startup on fused-memory's MCP port (8002)
    # being live. Pairs with Wants= above — waits cleanly instead of
    # crash-looping while fused-memory comes up.
    assert (
        "ExecStartPre=/home/leo/bin/wait-for-port.py --timeout 280 127.0.0.1:8002"
        in content
    ), "Missing ExecStartPre wait-for-port gate on fused-memory's port"
    assert (
        "uv run --frozen --project orchestrator orchestrator run --config /home/leo/src/dark-factory/orchestrator/config.yaml"
        in content
    ), "ExecStart must invoke the orchestrator with the df config, frozen"
    # --frozen: process start must NEVER implicitly re-sync the shared
    # dark-factory/.venv (the 2026-05-29 ghost-venv fix — a frozen start fails
    # fast instead of bootstrapping/mutating the runtime interpreter).
    assert "uv run --frozen" in content, (
        "ExecStart must pass --frozen so unit start never re-syncs the shared venv"
    )
    assert "Restart=on-failure" in content
    assert "RestartSec=10" in content
    assert "RestartMaxDelaySec=60" in content
    assert "StartLimitIntervalSec=600" in content
    assert "StartLimitBurst=10" in content
    assert "TimeoutStopSec=90" in content
    # TimeoutStartSec must exceed the ExecStartPre poll budget (280s) so a slow
    # fused-memory cold-start is covered by ONE start attempt rather than
    # burning StartLimit.
    assert "TimeoutStartSec=300" in content
    assert "StandardOutput=journal" in content
    assert "StandardError=journal" in content

    # --- [Install] ---
    assert "WantedBy=default.target" in content


# ---------------------------------------------------------------------------
# orchestrator-reify.service
# ---------------------------------------------------------------------------


def test_reify_orchestrator_service_structure() -> None:
    """scripts/orchestrator-reify.service must have the required systemd shape."""
    content = REIFY_SERVICE.read_text(encoding="utf-8")

    # --- sections ---
    assert "[Unit]" in content
    assert "[Service]" in content
    assert "[Install]" in content

    # --- [Unit] ---
    after_line = next(
        (ln for ln in content.splitlines() if ln.startswith("After=")), None
    )
    assert after_line is not None, "After= line not found"
    assert "network.target" in after_line
    assert "fused-memory.service" in after_line
    assert "reify-jobserver.service" in after_line
    assert "pytest-jobserver.service" in after_line
    assert "Wants=fused-memory.service" in content
    assert "Requires=fused-memory.service" not in content, (
        "Requires=fused-memory.service is a regression — use Wants= so a "
        "transient fused-memory blip doesn't permanently cancel our start job"
    )

    # --- [Service] ---
    assert "Type=simple" in content
    # WorkingDirectory MUST be dark-factory (uv --project resolves relative to cwd;
    # /home/leo/src/reify/orchestrator/ does not exist).
    assert "WorkingDirectory=/home/leo/src/dark-factory" in content, (
        "WorkingDirectory must be /home/leo/src/dark-factory for the reify unit too "
        "(uv --project orchestrator requires the orchestrator/ subdir to live under cwd)"
    )
    assert (
        "ExecStartPre=/home/leo/bin/wait-for-port.py --timeout 280 127.0.0.1:8002"
        in content
    ), "Missing ExecStartPre wait-for-port gate on fused-memory's port"
    assert (
        "uv run --frozen --project orchestrator orchestrator run --config /home/leo/src/reify/orchestrator.yaml"
        in content
    ), "ExecStart must invoke the orchestrator with the reify config, frozen"
    # --frozen: see the df structure test — unit start must never re-sync the
    # shared dark-factory/.venv that the reify orchestrator also runs under.
    assert "uv run --frozen" in content, (
        "ExecStart must pass --frozen so unit start never re-syncs the shared venv"
    )
    assert "Restart=on-failure" in content
    assert "RestartSec=10" in content
    assert "RestartMaxDelaySec=60" in content
    assert "StartLimitIntervalSec=600" in content
    assert "StartLimitBurst=10" in content
    assert "TimeoutStopSec=90" in content
    assert "TimeoutStartSec=300" in content
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
# orchestrator-autopilot-video.service
# ---------------------------------------------------------------------------


def test_autopilot_video_service_exists_and_structure() -> None:
    """scripts/orchestrator-autopilot-video.service must exist with the right shape.

    Until the 2026-05-29 venv-isolation fix this unit was live in
    ~/.config/systemd/user/ but had NO source template in scripts/ — so
    setup-host.sh would never reinstall it and it could not pick up --frozen.
    This test guards the now-tracked template going forward.
    """
    assert AUTOPILOT_SERVICE.exists(), (
        "scripts/orchestrator-autopilot-video.service must exist as a tracked "
        "template (it was previously installed but untracked)"
    )
    content = AUTOPILOT_SERVICE.read_text(encoding="utf-8")

    # --- sections ---
    assert "[Unit]" in content
    assert "[Service]" in content
    assert "[Install]" in content

    # --- [Unit] ---
    assert "Description=Autopilot Video Orchestrator" in content
    after_line = next(
        (ln for ln in content.splitlines() if ln.startswith("After=")), None
    )
    assert after_line is not None, "After= line not found"
    assert "network.target" in after_line
    assert "fused-memory.service" in after_line
    assert "Wants=fused-memory.service" in content
    assert "Requires=fused-memory.service" not in content

    # --- [Service] ---
    assert "Type=simple" in content
    # CWD must be dark-factory like the other two (uv --project resolves the
    # orchestrator/ subdir relative to cwd); the target is selected via --config.
    assert "WorkingDirectory=/home/leo/src/dark-factory" in content
    assert (
        "ExecStartPre=/home/leo/bin/wait-for-port.py --timeout 280 127.0.0.1:8002"
        in content
    )
    assert (
        "uv run --frozen --project orchestrator orchestrator run --config /home/leo/src/autopilot-video/orchestrator-config.yaml"
        in content
    ), "ExecStart must invoke the orchestrator with the autopilot-video config, frozen"
    assert "uv run --frozen" in content
    assert "Restart=on-failure" in content
    assert "StartLimitIntervalSec=600" in content
    assert "StartLimitBurst=10" in content
    assert "TimeoutStopSec=90" in content
    assert "TimeoutStartSec=300" in content
    assert "StandardOutput=journal" in content
    assert "StandardError=journal" in content

    # --- [Install] ---
    assert "WantedBy=default.target" in content


def test_autopilot_video_start_limit_directives_under_unit_section() -> None:
    """StartLimit directives must be under [Unit], not [Service] (systemd >=230)."""
    sections = _parse_sections(AUTOPILOT_SERVICE.read_text(encoding="utf-8"))
    unit_text = "\n".join(sections.get("Unit", []))
    service_text = "\n".join(sections.get("Service", []))
    assert "StartLimitIntervalSec=600" in unit_text
    assert "StartLimitBurst=10" in unit_text
    assert "StartLimitIntervalSec" not in service_text
    assert "StartLimitBurst" not in service_text


# ---------------------------------------------------------------------------
# orchestrator-watchdog.timer
# ---------------------------------------------------------------------------


def test_watchdog_timer_structure() -> None:
    """scripts/orchestrator-watchdog.timer must fire every 60s."""
    content = WATCHDOG_TIMER.read_text(encoding="utf-8")

    assert "[Unit]" in content
    assert "[Timer]" in content
    assert "[Install]" in content

    # Description field must exist (prose content is not pinned — the functional
    # interval invariant is asserted via OnUnitActiveSec=60 below)
    desc_line = next(
        (ln for ln in content.splitlines() if ln.startswith("Description=")), None
    )
    assert desc_line is not None, "Description= line not found"

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


# ---------------------------------------------------------------------------
# StartLimit directives must live in [Unit], not [Service]
# ---------------------------------------------------------------------------


def _parse_sections(content: str) -> dict[str, list[str]]:
    """Split unit-file text into {section_name: [lines]} (header line excluded)."""
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in content.splitlines():
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    return sections


@pytest.mark.parametrize(
    "service_path",
    [
        pytest.param(
            REPO_ROOT / "scripts" / "orchestrator-dark-factory.service",
            id="orchestrator-dark-factory",
        ),
        pytest.param(
            REPO_ROOT / "scripts" / "orchestrator-reify.service",
            id="orchestrator-reify",
        ),
    ],
)
def test_start_limit_directives_under_unit_section(
    service_path: pathlib.Path,
) -> None:
    """StartLimitIntervalSec and StartLimitBurst must be in [Unit], NOT [Service].

    Since systemd v230 these directives are only honoured under [Unit].
    Under [Service] systemd 255 (the target host) treats them as unknown keys
    and silently ignores them, which would disable the restart-rate cap and
    allow an unbounded 10s restart loop — especially dangerous when combined
    with the watchdog.
    """
    content = service_path.read_text(encoding="utf-8")
    sections = _parse_sections(content)

    unit_text = "\n".join(sections.get("Unit", []))
    service_text = "\n".join(sections.get("Service", []))

    # Must be present in [Unit]
    assert "StartLimitIntervalSec=600" in unit_text, (
        f"StartLimitIntervalSec=600 must be under [Unit] in {service_path.name}"
    )
    assert "StartLimitBurst=10" in unit_text, (
        f"StartLimitBurst=10 must be under [Unit] in {service_path.name}"
    )

    # Must NOT appear in [Service] (where systemd 255 ignores them)
    assert "StartLimitIntervalSec" not in service_text, (
        f"StartLimitIntervalSec must NOT be under [Service] in {service_path.name} "
        "(systemd >=230 only honours it under [Unit])"
    )
    assert "StartLimitBurst" not in service_text, (
        f"StartLimitBurst must NOT be under [Service] in {service_path.name} "
        "(systemd >=230 only honours it under [Unit])"
    )
