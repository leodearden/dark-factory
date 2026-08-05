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
import re

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
        "uv run --frozen --project orchestrator orchestrator run --config /home/leo/src/dark-factory/dark-factory-orchestrator.yaml"
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
    # The cap above is INERT without this pairing: systemd warns and discards
    # RestartMaxDelaySec= when no RestartSteps= accompanies it. This pins the
    # VALUE; the relational cap-implies-steps invariant is enforced fleet-wide
    # in tests/scripts/test_systemd_restart_backoff.py.
    assert "RestartSteps=4" in content
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
    # Primary prevention for the Jul-3 registration-wipe incident: the reify
    # unit's warm-lane pool storage (XFS loop image at /home/leo/src/warm-lanes)
    # must gate startup via RequiresMountsFor, or a post-crash reboot can start
    # this orchestrator before the mount is up and its first sweep's
    # `git worktree prune` runs against an empty mountpoint.
    assert "RequiresMountsFor=/home/leo/src/warm-lanes" in content, (
        "Missing RequiresMountsFor=/home/leo/src/warm-lanes — without it systemd "
        "has no ordering on the warm-lane loop-image mount and can start this "
        "orchestrator before the mount is up (Jul-3 registration-wipe incident)"
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
        "uv run --frozen --project orchestrator orchestrator run --config /home/leo/src/reify/dark-factory-orchestrator.yaml"
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
    # The cap above is INERT without this pairing: systemd warns and discards
    # RestartMaxDelaySec= when no RestartSteps= accompanies it. This pins the
    # VALUE; the relational cap-implies-steps invariant is enforced fleet-wide
    # in tests/scripts/test_systemd_restart_backoff.py.
    assert "RestartSteps=4" in content
    assert "StartLimitIntervalSec=600" in content
    assert "StartLimitBurst=10" in content
    assert "TimeoutStopSec=90" in content
    assert "TimeoutStartSec=300" in content
    assert "StandardOutput=journal" in content
    assert "StandardError=journal" in content

    # --- [Install] ---
    assert "WantedBy=default.target" in content


def test_reify_and_df_differ_only_in_config_and_description() -> None:
    """The two orchestrator service files must be identical except Description,
    --config path, and the reify-only warm-lane mount gate.

    This guards the 'same shape' invariant: any structural drift (missing key,
    different Restart policy, etc.) that appears in one but not the other will
    break this test.
    """
    df_lines = DF_SERVICE.read_text(encoding="utf-8").splitlines()
    reify_lines = REIFY_SERVICE.read_text(encoding="utf-8").splitlines()

    # orchestrator-reify.service alone gates on the warm-lane pool storage
    # mount (XFS loop image at /home/leo/src/warm-lanes) — dark-factory runs
    # entirely on the root filesystem and has no such mount to wait on. This
    # is a deliberate, permanent structural difference (Jul-3 registration-
    # wipe incident prevention), so it's stripped before the line-for-line
    # comparison below rather than being treated as drift.
    warm_lane_block_start = (
        "# Warm-lane pool storage (XFS loop image) mount gate: the fstab entry for"
    )
    warm_lane_directive = "RequiresMountsFor=/home/leo/src/warm-lanes"
    if warm_lane_block_start in reify_lines:
        start_idx = reify_lines.index(warm_lane_block_start)
        end_idx = reify_lines.index(warm_lane_directive, start_idx)
        # Drop the comment block, the directive itself, and the single
        # trailing blank line that separates it from the next block.
        del reify_lines[start_idx : end_idx + 2]

    assert len(df_lines) == len(reify_lines), (
        f"Service files have different line counts: df={len(df_lines)} reify={len(reify_lines)}"
    )

    diff_lines: list[tuple[int, str, str]] = []
    for i, (dl, rl) in enumerate(zip(df_lines, reify_lines)):
        if dl != rl:
            diff_lines.append((i + 1, dl, rl))

    # Classify each diff — must be either a Description, an ExecStart (--config)
    # line, or the self-identifying ORCH_UNIT line (each unit names itself).
    allowed_df_fragments = {
        "Dark Factory Orchestrator",
        "/home/leo/src/dark-factory/dark-factory-orchestrator.yaml",
    }
    allowed_reify_fragments = {
        "Reify Orchestrator",
        "/home/leo/src/reify/dark-factory-orchestrator.yaml",
    }
    # The ORCH_UNIT line must match EXACTLY, not merely contain the unit's
    # basename — a fragment-only "in" check would also wave through an
    # unrelated line that happens to embed the basename (e.g. a new
    # After=/Requires= referencing the unit), masking real structural drift.
    expected_df_orch_unit_line = "Environment=ORCH_UNIT=orchestrator-dark-factory.service"
    expected_reify_orch_unit_line = "Environment=ORCH_UNIT=orchestrator-reify.service"

    unexpected: list[tuple[int, str, str]] = []
    for lineno, dl, rl in diff_lines:
        if (
            dl.strip() == expected_df_orch_unit_line
            and rl.strip() == expected_reify_orch_unit_line
        ):
            continue
        df_ok = any(frag in dl for frag in allowed_df_fragments)
        reify_ok = any(frag in rl for frag in allowed_reify_fragments)
        if not (df_ok and reify_ok):
            unexpected.append((lineno, dl, rl))

    assert not unexpected, (
        "Unexpected differences between df and reify service files "
        "(only Description, --config path, and ORCH_UNIT should differ):\n"
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
        "uv run --frozen --project orchestrator orchestrator run --config /home/leo/src/autopilot-video/dark-factory-orchestrator.yaml"
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


# ---------------------------------------------------------------------------
# ORCH_UNIT self-identification (task 2064 — 2004 self-kill root-cause fix)
# ---------------------------------------------------------------------------
#
# DeterministicRunner._default_resolve_own_unit() (orchestrator/src/orchestrator/
# deterministic_runner.py) resolves the orchestrator's own systemd unit purely
# from the ORCH_UNIT env var, failing open to '' when unset. An empty own-unit
# means self_target is False for EVERY target_unit, so a fleet restart-all
# self-restart deploy takes the blocking cross-unit path and SIGTERMs its own
# deploy script mid-run instead of scheduling a detached self-restart. Every
# orchestrator unit template must set ORCH_UNIT=<its own basename> so the
# runner can detect a self-target deploy.

ALL_ORCHESTRATOR_SERVICE_FILES = sorted(
    (REPO_ROOT / "scripts").glob("orchestrator-*.service")
)

_EXPECTED_ORCHESTRATOR_SERVICE_BASENAMES = {
    "orchestrator-dark-factory.service",
    "orchestrator-reify.service",
    "orchestrator-solar-challenge-platform.service",
    "orchestrator-my-solar-challenge.service",
    "orchestrator-autopilot-video.service",
    "orchestrator-know-live.service",
    "orchestrator-pump-web-ui.service",
    "orchestrator-watchdog.service",
}


def test_orchestrator_service_glob_covers_all_known_units() -> None:
    """Coverage guard: the glob must be non-empty and include all eight known units.

    A wrong CWD or other glob mishap would silently shrink the parametrized
    ORCH_UNIT lint below to zero cases — a zero-case parametrize collects no
    tests and reports no failure, which would mask a missing requirement
    instead of catching it.
    """
    discovered = {p.name for p in ALL_ORCHESTRATOR_SERVICE_FILES}
    assert discovered, "glob discovered no orchestrator-*.service templates"
    missing = _EXPECTED_ORCHESTRATOR_SERVICE_BASENAMES - discovered
    assert not missing, f"glob is missing known orchestrator unit templates: {missing}"


@pytest.mark.parametrize(
    "service_path",
    ALL_ORCHESTRATOR_SERVICE_FILES,
    ids=lambda p: p.name,
)
def test_orchestrator_service_sets_own_orch_unit(
    service_path: pathlib.Path,
) -> None:
    """Every orchestrator unit template must self-identify via ORCH_UNIT=<own basename>.

    The value must equal the unit's OWN basename (self-target detection
    compares this string against `before_done.target_unit`), and the line
    must live in [Service] — systemd only honours Environment= there; under
    [Unit]/[Install] it is silently ignored (same failure class as the
    StartLimit directives guarded above).
    """
    content = service_path.read_text(encoding="utf-8")
    expected_line = f"Environment=ORCH_UNIT={service_path.name}"

    sections = _parse_sections(content)
    service_text = "\n".join(sections.get("Service", []))
    assert expected_line in service_text.splitlines(), (
        f"{service_path.name} must set `{expected_line}` in its [Service] section "
        "(systemd only honours Environment= under [Service])"
    )


# ---------------------------------------------------------------------------
# Canonical orchestrator-config filename (task 3641; completes task 3512's sweep)
# ---------------------------------------------------------------------------

CANONICAL_CONFIG_BASENAME = "dark-factory-orchestrator.yaml"


def _exec_start_config_arg(content: str) -> str | None:
    """Return the `--config` argument of the unit's ExecStart=, or None if absent.

    None is a real answer, not a parse failure: orchestrator-watchdog.service
    runs a probe script that takes no --config at all.  Callers must skip on
    None rather than fail, or the watchdog gets dragged into an invariant that
    does not apply to it.
    """
    exec_line = next(
        (ln for ln in content.splitlines() if ln.startswith("ExecStart=")), None
    )
    if exec_line is None:
        return None
    tokens = exec_line.split()
    for i, token in enumerate(tokens):
        if token == "--config":
            return tokens[i + 1] if i + 1 < len(tokens) else None
        if token.startswith("--config="):
            return token.split("=", 1)[1]
    return None


@pytest.mark.parametrize(
    "service_path",
    ALL_ORCHESTRATOR_SERVICE_FILES,
    ids=lambda p: p.name,
)
def test_orchestrator_service_points_at_canonical_config_filename(
    service_path: pathlib.Path,
) -> None:
    """Every orchestrator unit's --config must name the CANONICAL config filename.

    CLAUDE.md makes ``<project_root>/dark-factory-orchestrator.yaml`` the
    canonical, required filename — the dashboard's escalation-URL discovery
    (``_discover_escalation_urls``) keys on it, and legacy spellings
    (``orchestrator.yaml``, ``orchestrator-config.yaml``,
    ``orchestrator/config.yaml``) are honoured only as a discovery fallback for
    not-yet-migrated projects, never as a supported choice.  Task 2698
    canonicalized the filename; task 2719 then RETIRED dark-factory's own
    transitional symlinks (guarded by test_legacy_config_symlinks_retired.py),
    which is the precedent that makes the legacy spelling a defect here rather
    than merely an inconsistency.

    The failure this catches is latent, not active, and that is exactly why it
    needs a guard: a target project that still keeps ``orchestrator.yaml`` as a
    SYMLINK to the canonical file resolves the legacy path fine today, so a
    stale unit template starts cleanly and nothing looks wrong.  It breaks on
    the day that project retires its symlink the way this repo already did —
    at which point the unit fails to start and the cause is a line nobody has
    looked at in months.

    The ``--config`` predicate is load-bearing: orchestrator-watchdog.service
    runs a probe script with no ``--config``, so it is skipped rather than
    failed.  Keying on the flag's presence (rather than naming the watchdog as
    an exception) means a future unit is covered the day it lands.
    """
    content = service_path.read_text(encoding="utf-8")
    config_arg = _exec_start_config_arg(content)
    if config_arg is None:
        pytest.skip(f"{service_path.name} has no ExecStart --config argument")

    actual = pathlib.PurePosixPath(config_arg).name
    assert actual == CANONICAL_CONFIG_BASENAME, (
        f"{service_path.name} points --config at {config_arg!r}, whose basename "
        f"is {actual!r}. It must be the canonical {CANONICAL_CONFIG_BASENAME!r} "
        "(CLAUDE.md: the dashboard's _discover_escalation_urls keys on that "
        "exact filename). If the target project still has a legacy "
        "orchestrator.yaml symlink, the legacy path resolves today and breaks "
        "the moment that project retires the symlink — as dark-factory already "
        "did under task 2719."
    )


# ---------------------------------------------------------------------------
# setup-host.sh installer coverage (task 3641)
# ---------------------------------------------------------------------------
#
# A committed unit template that setup-host.sh never installs is invisible: it
# looks like a supported unit, but a fresh host simply does not get it, and an
# existing host keeps whatever hand-installed copy it happens to have.  That is
# the exact defect this task exists to fix — orchestrator-know-live.service had
# a committed template since e5273d8623 and was never wired in, and
# orchestrator-pump-web-ui.service ran on the host for weeks with no committed
# source at all.  Fixing the two instances without a guard leaves the CLASS
# open, so this asserts the installer's coverage mechanically.

SETUP_HOST_SH = REPO_ROOT / "scripts" / "setup-host.sh"


def _unit_has_install_section(content: str) -> bool:
    """True if the unit declares an [Install] section (i.e. it is enable-able).

    ``systemctl enable`` on a unit with no [Install] section is an error, not a
    no-op — such a unit is 'static'.  orchestrator-watchdog.service is exactly
    that: it is pulled in by orchestrator-watchdog.timer, which carries the
    install instead.
    """
    return "Install" in _parse_sections(content)


def _shell_statements(script_text: str) -> list[str]:
    """Non-comment, non-blank statements of a shell script, whitespace-stripped.

    Comments are dropped so that a unit merely NAMED in the section header
    prose above the cp block cannot satisfy an assertion that the installer
    actually copies it.
    """
    return [
        stripped
        for line in script_text.splitlines()
        if (stripped := line.strip()) and not stripped.startswith("#")
    ]


def test_setup_host_install_predicate_discriminates() -> None:
    """Coverage guard: the glob is non-empty AND the [Install] predicate splits it.

    Mirrors test_orchestrator_service_glob_covers_all_known_units, plus the
    silent-green risk specific to the parametrized guard below.  That test's
    enable half is CONDITIONAL on _unit_has_install_section, so a predicate
    that answered False for everything (a broken _parse_sections, a change in
    unit shape) would vacuously satisfy every case while checking nothing.
    Asserting the predicate discriminates in BOTH directions keeps it honest:
    at least one enable-able unit, and at least one static unit — today the
    seven project orchestrators and orchestrator-watchdog.service respectively.
    """
    assert ALL_ORCHESTRATOR_SERVICE_FILES, (
        "glob discovered no orchestrator-*.service templates"
    )
    by_install = {
        p.name: _unit_has_install_section(p.read_text(encoding="utf-8"))
        for p in ALL_ORCHESTRATOR_SERVICE_FILES
    }
    assert any(by_install.values()), (
        "no orchestrator template has an [Install] section — the enable half of "
        "test_setup_host_installs_every_orchestrator_unit would pass vacuously "
        f"for every unit. Parsed: {by_install}"
    )
    assert not all(by_install.values()), (
        "every orchestrator template has an [Install] section, so the static-unit "
        "branch is never exercised. orchestrator-watchdog.service is expected to "
        f"have none (it is pulled in by its .timer). Parsed: {by_install}"
    )


@pytest.mark.parametrize(
    "service_path",
    ALL_ORCHESTRATOR_SERVICE_FILES,
    ids=lambda p: p.name,
)
def test_setup_host_installs_every_orchestrator_unit(
    service_path: pathlib.Path,
) -> None:
    """setup-host.sh must copy every orchestrator template, and enable each one
    that has an [Install] section.

    setup-host.sh is how a unit reaches a host.  A template it does not copy is
    a unit that does not exist on a fresh machine, however correct its contents
    — so 'committed' and 'installed' silently diverge with nothing to detect it.

    The enable obligation is DERIVED from the [Install] predicate rather than
    hard-coded around a named watchdog exception, for the same reason
    test_systemd_restart_backoff.py discovers by content instead of by list: a
    hand-maintained exception list has to be edited for every future unit, and
    a unit nobody remembers to add is precisely the bug being fixed here.  With
    the predicate, a new orchestrator template turns this red on the day it
    lands unless the installer learns about it.

    This does NOT assert the converse (that the installer names no unknown
    unit) — a `cp` of a template this glob cannot see would fail the installer
    loudly at run time, so it needs no test.
    """
    basename = service_path.name
    statements = _shell_statements(SETUP_HOST_SH.read_text(encoding="utf-8"))

    expected_cp = f'cp "$REPO_ROOT/scripts/{basename}" "$UNIT_DIR/"'
    copied = any(
        stmt.startswith("cp ") and f"scripts/{basename}" in stmt for stmt in statements
    )
    assert copied, (
        f"{SETUP_HOST_SH.name} never copies {basename} into $UNIT_DIR, so a fresh "
        f"host never gets that unit. Add a line to the cp block:\n    {expected_cp}"
    )

    content = service_path.read_text(encoding="utf-8")
    if not _unit_has_install_section(content):
        return

    expected_enable = f"systemctl --user enable {basename}"
    enabled = any(stmt.startswith(expected_enable) for stmt in statements)
    assert enabled, (
        f"{basename} declares an [Install] section but {SETUP_HOST_SH.name} never "
        "enables it, so it is copied to the host and then never starts at boot. "
        f"Add a line to the enable block:\n    {expected_enable}"
    )


# ---------------------------------------------------------------------------
# SETUP.md operator remediation coverage (task 3641, review feedback)
# ---------------------------------------------------------------------------
#
# setup-host.sh installs and ENABLES orchestrator units for several projects
# that are not part of this repo.  SETUP.md's "Known gap" callout is the only
# operator-facing remediation for that: it hands a non-maintainer a copy-paste
# `systemctl --user disable --now ...` command naming the units to turn off.
#
# That block is EXECUTABLE, not prose — which makes it the same class of
# artifact as setup-host.sh's own cp/enable lines guarded above, and it drifts
# the same way.  Every unit the installer enables but the doc omits is left
# running on a non-maintainer's host with a --config path that does not exist:
# at the next login/default.target it crash-loops until StartLimitBurst is
# exhausted, then sits failed.  And every unit the doc names but the installer
# does NOT install breaks the operator's copy-paste outright, because
# `systemctl --user disable --now` on a nonexistent unit file exits non-zero.
# So both drift directions are real defects and both are asserted, separately.

SETUP_MD = REPO_ROOT / "SETUP.md"

# The one enabled orchestrator unit that IS genuinely the reader's own, which
# SETUP.md explicitly tells them to keep ("orchestrator-dark-factory.service
# itself is genuinely yours and is fine to run").  Keying the exclusion on this
# repo's OWN unit — rather than hand-listing the foreign ones — means a seventh
# project is covered by this guard the day its unit lands, matching the
# [Install]-predicate reasoning used by the installer-coverage guard above.
_OWN_PROJECT_UNIT = "orchestrator-dark-factory.service"

_DISABLE_COMMAND = "systemctl --user disable --now"

_ORCHESTRATOR_UNIT_RE = re.compile(r"orchestrator-[\w.-]+?\.service")


def _enabled_orchestrator_service_units() -> set[str]:
    """Orchestrator .service units that setup-host.sh actually enables.

    Derived from the same comment-stripped view of setup-host.sh that the
    installer-coverage guard uses, so the doc is checked against what the
    installer DOES, never against what its section header prose says.

    The ``.service`` filter naturally drops ``orchestrator-watchdog.timer``,
    which setup-host.sh also enables and which SETUP.md correctly tells the
    reader to KEEP enabled (it skips disabled units, so it is harmless).
    """
    statements = _shell_statements(SETUP_HOST_SH.read_text(encoding="utf-8"))
    prefix = "systemctl --user enable "
    units = set()
    for stmt in statements:
        if not stmt.startswith(prefix):
            continue
        unit = stmt.split()[-1]
        if unit.startswith("orchestrator-") and unit.endswith(".service"):
            units.add(unit)
    return units


def _fenced_code_blocks(md_text: str) -> list[str]:
    """Bodies of every ``` -fenced code block, in document order.

    Parsed by walking fences rather than by line number so the guard survives
    any edit above the block it cares about.
    """
    blocks: list[str] = []
    current: list[str] | None = None
    for line in md_text.splitlines():
        if line.lstrip().startswith("```"):
            if current is None:
                current = []
            else:
                blocks.append("\n".join(current))
                current = None
            continue
        if current is not None:
            current.append(line)
    return blocks


def _setup_md_disable_blocks() -> list[str]:
    """Every fenced block in SETUP.md carrying the disable-the-foreign-units command."""
    return [
        block
        for block in _fenced_code_blocks(SETUP_MD.read_text(encoding="utf-8"))
        if _DISABLE_COMMAND in block
    ]


def test_setup_md_disable_block_is_discoverable() -> None:
    """Coverage guard: exactly one disable block, and a non-empty derived set.

    Mirrors the intent of test_setup_host_install_predicate_discriminates.  The
    parity assertion below compares two derived sets, and BOTH can silently
    collapse to empty: a fence-parse mishap returning no block, or an enable-
    statement scan that matches nothing.  Empty-vs-empty is a passing set
    comparison that checks nothing — the same silent-green failure mode the
    other coverage guards in this file exist to prevent — so both inputs are
    asserted non-degenerate here, before the parity test relies on them.
    """
    blocks = _setup_md_disable_blocks()
    assert len(blocks) == 1, (
        f"expected exactly one fenced block in {SETUP_MD.name} containing "
        f"{_DISABLE_COMMAND!r}, found {len(blocks)}. The parity guard below "
        "assumes a single copy-paste remediation block; if the doc genuinely "
        "grew a second one, teach the helper which is which rather than "
        "letting it pick arbitrarily."
    )

    enabled = _enabled_orchestrator_service_units()
    assert enabled, (
        f"derived no `{_DISABLE_COMMAND.replace('disable --now', 'enable')} "
        f"orchestrator-*.service` statements from {SETUP_HOST_SH.name} — the "
        "parity guard below would pass vacuously against an empty set."
    )
    assert enabled - {_OWN_PROJECT_UNIT}, (
        f"{SETUP_HOST_SH.name} enables no orchestrator unit other than "
        f"{_OWN_PROJECT_UNIT}, so the foreign-unit set is empty and the parity "
        "guard below checks nothing."
    )


def test_setup_md_disable_block_covers_every_foreign_orchestrator_unit() -> None:
    """SETUP.md's disable block must name exactly the foreign units setup-host.sh enables.

    Scope, deliberate: this guards ONLY the executable
    ``systemctl --user disable --now`` block — the artifact an operator actually
    copy-pastes.  It does NOT pin the surrounding prose (the "four projects"
    count, or the sentence-form list of unit names), because asserting on
    documentation wording is a brittle meta-test that rots on the next rewrite
    and buys nothing this block-level assertion does not already give.  That
    prose is corrected by hand.  Do not "helpfully" extend this test into a
    wording pin.
    """
    foreign = _enabled_orchestrator_service_units() - {_OWN_PROJECT_UNIT}
    block = _setup_md_disable_blocks()[0]
    named = set(_ORCHESTRATOR_UNIT_RE.findall(block))

    missing = sorted(foreign - named)
    assert not missing, (
        f"{SETUP_HOST_SH.name} enables {', '.join(missing)} but {SETUP_MD.name}'s "
        f"`{_DISABLE_COMMAND}` block does not name "
        f"{'them' if len(missing) > 1 else 'it'}. A non-maintainer who follows "
        "the documented remediation verbatim is left with "
        f"{'those units' if len(missing) > 1 else 'that unit'} enabled, pointing "
        "at a --config path that does not exist on their machine; at the next "
        "login/default.target they crash-loop until StartLimitBurst is exhausted "
        f"and then sit failed. Fix: add {', '.join(missing)} to the "
        f"`{_DISABLE_COMMAND}` block in {SETUP_MD.name}."
    )

    extra = sorted(named - foreign - {_OWN_PROJECT_UNIT})
    assert not extra, (
        f"{SETUP_MD.name}'s `{_DISABLE_COMMAND}` block names "
        f"{', '.join(extra)}, which {SETUP_HOST_SH.name} does not enable. "
        "`systemctl --user disable --now` on a unit file that does not exist "
        "fails with 'Unit file does not exist' and returns NON-ZERO, so a stale "
        "entry breaks the whole copy-paste command for the operator — this is a "
        f"real defect, not cosmetic. Fix: remove {', '.join(extra)} from the "
        f"block in {SETUP_MD.name} (or, if the unit should be installed after "
        f"all, add it to {SETUP_HOST_SH.name}'s enable block)."
    )
