"""File-content tests for the shape of the orchestrator systemd unit templates.

These tests read source-controlled files directly — no systemd runtime is
required.  Every suite here takes a .service / .timer template as its SUBJECT:

1. Per-unit shape (named units, then fleet-wide parametrized invariants):
     - scripts/orchestrator-dark-factory.service
     - scripts/orchestrator-reify.service
     - scripts/orchestrator-watchdog.service
     - scripts/orchestrator-watchdog.timer
     - every scripts/orchestrator-*.service, via ALL_ORCHESTRATOR_SERVICE_FILES
       (ORCH_UNIT self-identification; canonical --config filename)

WHAT LEFT, AND WHY (task 3746).  Two further suites lived here until this
module reached ~1300 lines: scripts/setup-host.sh installer coverage and
SETUP.md operator-remediation coverage (both task 3641).  Neither reads a
.service file as its subject — they assert what the INSTALLER and the OPERATOR
DOCS do with the templates — and both now live in
tests/scripts/test_setup_host_unit_installation.py, together with the two
coverage guards that keep them from going silently green.

That split was gated on a helper extraction this docstring previously called
"a lift waiting to happen", and task 3746 performed it: ``parse_sections`` and
``ALL_ORCHESTRATOR_SERVICE_FILES`` moved into
tests/scripts/systemd_unit_invariants.py, because the split leaves a consumer
on both sides of each.  They are imported back below under their public names.

Extraction state of the rest is unchanged.  CANONICAL_CONFIG_BASENAME,
MalformedExecStart and the ``--config`` token scan live in
systemd_unit_invariants.py (task 3773), a second consumer having hand-copied
them (tests/scripts/test_know_live_installed_unit_parity.py) and drifted.
``_exec_start_line`` did NOT move — it still has exactly one consumer, and
this directory's lift trigger is a second consumer, not proximity.
``shell_statements`` had already moved to the sibling setup_host_parsing.py
for the same reason; it is no longer imported here at all, having left with
the installer suite that was its only consumer in this module.

tests/scripts/test_orchestrator_watchdog.py's ``_unit_sections`` remains a
third hand-copy of the section parse, and is now a straightforward de-dup
against systemd_unit_invariants.parse_sections; filed as a follow-up.

See also:
  - tests/scripts/test_setup_host_unit_installation.py — the installer and
    operator-doc suites that left this module
  - tests/scripts/test_dashboard_service_template.py — pattern reference
  - tests/scripts/test_systemd_restart_backoff.py — content-discovered
    restart-backoff sweep over every unit in the tree
"""

import pathlib

import pytest

# Shared with tests/scripts/test_know_live_installed_unit_parity.py, which had
# hand-copied the ExecStart parsers from this module and drifted; task 3773
# lifted those into the directory's shared helper module (the parse contract is
# stated once there, on config_arg_from_exec_start).  parse_sections and
# ALL_ORCHESTRATOR_SERVICE_FILES joined them under task 3746, when the
# installer and operator-doc suites moved to
# tests/scripts/test_setup_host_unit_installation.py and left a consumer on
# BOTH sides of them.  Importable by name only because
# tests/scripts/conftest.py puts this directory on sys.path, which pytest's
# --import-mode=importlib deliberately does not.
# Imported UNALIASED: the sibling modules import these under exactly these
# public names, and a private alias here would obscure that they are one
# definition rather than several.
from systemd_unit_invariants import (
    ALL_ORCHESTRATOR_SERVICE_FILES,
    CANONICAL_CONFIG_BASENAME,
    MalformedExecStart,
    config_arg_from_exec_start,
    parse_sections,
)

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
    for i, (dl, rl) in enumerate(zip(df_lines, reify_lines, strict=True)):
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
    sections = parse_sections(AUTOPILOT_SERVICE.read_text(encoding="utf-8"))
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
    """scripts/orchestrator-watchdog.timer must fire every 60s, tightly.

    ``AccuracySec=5s`` is asserted here because this is the one place the
    number is justified, and it is not cosmetic. systemd's DEFAULT
    AccuracySec is 1min, so ``OnUnitActiveSec=60`` WITHOUT this directive
    gives an elapse window of [60s, 120s] rather than [60s, 65s] — roughly
    halving how fast the fleet watchdog notices a dead orchestrator, for a
    probe whose entire job is noticing that quickly.

    The value is not new policy. The live host has been running it since
    2026-05-26 (``systemctl --user show orchestrator-watchdog.timer -p
    AccuracyUSec`` => ``AccuracyUSec=5s``, measured 2026-08-02), and the
    committed timer never carried it (``git log -S AccuracySec --
    scripts/orchestrator-watchdog.timer`` is empty) — the installed copy is
    the older hand-written original and the committed transcription (task
    1368) dropped the directive.

    That asymmetry is why this assertion exists: task 3424 installs the
    committed timer onto the host, and without this line that install would
    have been a supervision REGRESSION wearing the costume of a parity fix.
    This test makes the committed copy the safe one to install.
    """
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
    assert "AccuracySec=5s" in content, (
        "orchestrator-watchdog.timer must declare AccuracySec=5s. Without it "
        "systemd's 1min default widens this 60s probe's elapse window from "
        "[60s, 65s] to [60s, 120s], roughly halving fleet-watchdog "
        "responsiveness. See this test's docstring."
    )
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
    sections = parse_sections(content)

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

    sections = parse_sections(content)
    service_text = "\n".join(sections.get("Service", []))
    assert expected_line in service_text.splitlines(), (
        f"{service_path.name} must set `{expected_line}` in its [Service] section "
        "(systemd only honours Environment= under [Service])"
    )


# ---------------------------------------------------------------------------
# Canonical orchestrator-config filename (task 3641; completes task 3512's sweep)
#
# CANONICAL_CONFIG_BASENAME, MalformedExecStart and the token scan itself are
# imported from tests/scripts/systemd_unit_invariants.py, where the parse
# contract is stated once (config_arg_from_exec_start). What stays here is the
# FILE-CONTENT half: finding the effective ExecStart= line, which has exactly
# one consumer and so did not meet this directory's lift trigger.
# ---------------------------------------------------------------------------


def _exec_start_line(content: str, unit_name: str = "<unit>") -> str:
    """The unit's EFFECTIVE ExecStart= line, stripped.  Raises if it has none.

    LAST occurrence wins, matching systemd itself and
    systemd_unit_invariants.restart_directive — which is what the sibling
    parity module feeds the shared scan from, so a first-match read here would
    have the two layers disagreeing about the same unit.  A drop-in override
    lands as an empty ``ExecStart=`` list RESET followed by the real command;
    reading the reset line would find no ``--config`` token and answer None,
    i.e. silently skip a unit whose real command may well be wrong — the exact
    direction the shared parser's contract refuses (see
    systemd_unit_invariants.config_arg_from_exec_start).  An effective
    ExecStart= with no command after the ``=`` raises for that same reason
    instead of degrading to that None.

    Lines are stripped before matching because systemd permits leading
    whitespace on a directive; the trailing ``=`` in the prefix is what keeps
    ExecStartPre= out of the match.
    """
    exec_lines = [
        stripped
        for ln in content.splitlines()
        if (stripped := ln.strip()).startswith("ExecStart=")
    ]
    if not exec_lines:
        raise MalformedExecStart(
            f"{unit_name} declares no ExecStart= line. Every orchestrator unit "
            "must have one — systemd refuses to start a Type=simple service "
            "without it. Treating this as 'takes no --config' would silently "
            "skip the unit out of the canonical-config-filename guard below."
        )
    exec_line = exec_lines[-1]
    if not exec_line.partition("=")[2].strip():
        raise MalformedExecStart(
            f"{unit_name}'s effective ExecStart= carries no command "
            f"({exec_line!r}): the last assignment is a list RESET with nothing "
            "appended after it, so systemd has no command to run at all. "
            "Treating this as 'takes no --config' would silently skip a unit "
            "that cannot start."
        )
    return exec_line


def _exec_start_config_arg(content: str, unit_name: str = "<unit>") -> str | None:
    """Return the `--config` argument of the unit's ExecStart=, or None if absent.

    This wrapper owns only the locating half — the effective ExecStart= line
    inside unit FILE CONTENT (``_exec_start_line`` above).  When None comes
    back and when MalformedExecStart is raised is the contract of
    systemd_unit_invariants.config_arg_from_exec_start, stated there once.

    The sibling parity module hands that same scan an already-extracted value
    instead (a ``restart_directive`` result, or a ``systemctl show`` ``argv[]``
    segment); the scan is prefix-agnostic, so both shapes reach it without
    either side normalising first.
    """
    return config_arg_from_exec_start(_exec_start_line(content, unit_name), unit_name)


# ---------------------------------------------------------------------------
# Fixture-string coverage for the FILE-CONTENT half of the --config parse
#
# The token scan itself is shared (systemd_unit_invariants.config_arg_from_
# exec_start) and its negative cases are owned ONCE, by the PARSER-layer
# section of tests/scripts/test_know_live_installed_unit_parity.py — this
# directory's one-owner convention — so they are deliberately NOT re-pinned
# here.  What these fixtures own is the OTHER half: locating the effective
# ExecStart= line inside unit FILE CONTENT (_exec_start_line), which had no
# fixture-string coverage at all.  Its only exercise was the parametrized
# sweep over real committed templates, every one of which is well-formed and
# carries exactly one ExecStart=, so its raise branches, its
# last-occurrence rule and the None branch reached through file content were
# asserted nowhere.  One positive case is kept, to pin that file content
# reaches the shared scan at all.
#
# Inline fixtures rather than tmp_path files, matching how sibling guards in
# this directory build unit text (cf. test_check_dashboard_unit_parity.py's
# _SAMPLE_UNIT): both helpers take a string, so a file would add I/O without
# adding coverage.
# ---------------------------------------------------------------------------


def _unit_fixture(*service_lines: str) -> str:
    """A minimal [Service]-shaped unit carrying *service_lines*, for the parsers."""
    body = "\n".join(service_lines)
    return f"[Unit]\nDescription=Fixture\n\n[Service]\nType=simple\n{body}\n"


@pytest.mark.parametrize(
    ("exec_start", "expected"),
    [
        pytest.param(
            "ExecStart=/usr/bin/uv run orchestrator run "
            "--config /home/leo/src/x/dark-factory-orchestrator.yaml",
            "/home/leo/src/x/dark-factory-orchestrator.yaml",
            id="space-separated",
        ),
        pytest.param(
            "ExecStart=/usr/bin/python3 /home/leo/src/x/scripts/orchestrator-watchdog.py",
            None,
            id="no-config-flag-at-all",
        ),
    ],
)
def test_exec_start_config_arg_answers_from_unit_content(exec_start: str, expected) -> None:
    """_exec_start_config_arg answers from whole FILE CONTENT, not a bare value.

    The scan's own spelling matrix (`--config x` vs `--config=x`) belongs to
    its single owner in the parity module; what is pinned here is that file
    content reaches that scan at all, and the None branch it reaches through.

    That None is load-bearing rather than incidental:
    test_orchestrator_service_points_at_canonical_config_filename SKIPs on it
    for orchestrator-watchdog.service (a probe script that legitimately takes
    no --config), and test_exec_start_config_parser_answers_for_every_
    orchestrator_run_unit asserts that skip branch stays genuinely exercised.
    """
    assert _exec_start_config_arg(_unit_fixture(exec_start), "fixture.service") == expected


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(
            _unit_fixture("ExecStartPre=/bin/mkdir -p /run/fixture"),
            id="no-execstart-line-at-all",
        ),
        pytest.param(
            _unit_fixture(
                "ExecStart=/usr/bin/uv run orchestrator run --config /tmp/staging.yaml",
                "ExecStart=",
            ),
            id="effective-execstart-is-an-empty-reset",
        ),
    ],
)
def test_exec_start_line_raises_on_unit_with_no_usable_command(content: str) -> None:
    """A unit with no usable ExecStart= FAILS, naming itself — it does not skip.

    _exec_start_line's own negative cases, owned here because that helper
    stayed local when the token scan was lifted (the scan's negative cases
    stayed with THEIR owner, the parity module, and are not duplicated here).
    Neither may be answered with None: the canonical-filename guard SKIPs on
    None, so a unit systemd could not even start would be waved straight
    through it.  Both must raise the SHARED MalformedExecStart that the
    parser's callers already catch, not a locally redefined look-alike.
    """
    with pytest.raises(MalformedExecStart) as excinfo:
        _exec_start_config_arg(content, "fixture.service")
    assert "fixture.service" in str(excinfo.value), (
        "the raise must name the unit it was asked about — these parsers are "
        "applied by a parametrized sweep over every committed template, so a "
        "message that does not name the offender leaves the reader to guess "
        f"which case failed. Got: {excinfo.value}"
    )


def test_exec_start_config_arg_reads_the_last_execstart_assignment() -> None:
    """An empty ExecStart= RESET followed by the real command resolves to the real one.

    The drop-in override shape: systemd merges <unit>.d/*.conf by APPENDING,
    so overriding a command means first resetting the list with a bare
    `ExecStart=`.  Last-occurrence-wins is what systemd does, and what
    systemd_unit_invariants.restart_directive — the sibling parity module's
    source for the very same scan — documents.  A first-match read here would
    have the two layers answering differently about one unit: this one reading
    the empty reset, finding no --config and SKIPPING the canonical-filename
    guard, while the parity layer asserted against the real command.
    """
    content = _unit_fixture(
        "ExecStart=",
        "ExecStart=/usr/bin/uv run orchestrator run "
        "--config /home/leo/src/x/dark-factory-orchestrator.yaml",
    )
    assert _exec_start_config_arg(content, "fixture.service") == (
        "/home/leo/src/x/dark-factory-orchestrator.yaml"
    )


def test_exec_start_config_arg_ignores_exec_start_pre() -> None:
    """ExecStartPre= must not be mistaken for ExecStart=, --config and all.

    Pins the trailing-``=`` discrimination _exec_start_line's docstring calls
    out.  A prefix match on "ExecStart" alone reads the FIRST ExecStartPre=
    line as the unit's command, so a pre-command that happens to carry its own
    --config (a config-rendering or validation step, exactly the shape a
    preparatory ExecStartPre= takes) silently answers for the real one — and
    the guard reports on a path the service never runs with.
    """
    content = _unit_fixture(
        "ExecStartPre=/usr/bin/uv run orchestrator validate --config /tmp/staging.yaml",
        "ExecStart=/usr/bin/uv run orchestrator run "
        "--config /home/leo/src/x/dark-factory-orchestrator.yaml",
    )
    assert _exec_start_config_arg(content, "fixture.service") == (
        "/home/leo/src/x/dark-factory-orchestrator.yaml"
    )


# Marker identifying a unit that launches the orchestrator CLI proper (as
# opposed to orchestrator-watchdog.service, which runs a bare probe script).
# `orchestrator run` REQUIRES a --config, so this is the mechanical predicate
# for "must be asserted, must not skip" in the coverage guard below.
_ORCHESTRATOR_RUN_MARKER = "orchestrator run"


def test_exec_start_config_parser_answers_for_every_orchestrator_run_unit() -> None:
    """Coverage guard: the --config parser must ANSWER, not skip, for real units.

    Mirrors test_orchestrator_service_glob_covers_all_known_units and
    test_setup_host_unit_installation.py's
    test_setup_host_install_predicate_discriminates.  The parametrized guard
    below is CONDITIONAL — it skips whenever _exec_start_config_arg returns
    None — so a parser regression (a change in ExecStart= line shape, a
    continuation-line variant, a renamed flag) would return None for all eight
    templates, skip every case, and report green while checking nothing.

    The expectation is DERIVED, not a hard-coded count that rots the day an
    eighth project lands: a unit whose ExecStart= invokes ``orchestrator run``
    needs a --config by construction, and one that does not (the watchdog probe
    script) legitimately has none.  Both directions are asserted so the skip
    branch stays genuinely exercised rather than becoming the only branch.
    """
    assert ALL_ORCHESTRATOR_SERVICE_FILES, (
        "glob discovered no orchestrator-*.service templates"
    )

    parsed: dict[str, str | None] = {}
    runs_orchestrator: set[str] = set()
    for path in ALL_ORCHESTRATOR_SERVICE_FILES:
        content = path.read_text(encoding="utf-8")
        # A malformed ExecStart= raises out of here — deliberately, so it is a
        # hard failure of this guard rather than a skipped parametrized case.
        parsed[path.name] = _exec_start_config_arg(content, path.name)
        if _ORCHESTRATOR_RUN_MARKER in _exec_start_line(content, path.name):
            runs_orchestrator.add(path.name)

    assert runs_orchestrator, (
        "no orchestrator template has an ExecStart= invoking "
        f"{_ORCHESTRATOR_RUN_MARKER!r}, so this guard has nothing to check and "
        f"the parametrized test below would skip everything. Parsed: {parsed}"
    )

    silent = sorted(name for name in runs_orchestrator if parsed[name] is None)
    assert not silent, (
        f"{', '.join(silent)} run `{_ORCHESTRATOR_RUN_MARKER}` but no --config "
        "argument could be parsed from their ExecStart=. Either the unit really "
        "lost its --config (the orchestrator cannot start without one), or "
        "_exec_start_config_arg has regressed — in which case "
        "test_orchestrator_service_points_at_canonical_config_filename is "
        f"silently skipping instead of asserting. Parsed: {parsed}"
    )

    assert any(value is None for value in parsed.values()), (
        "every orchestrator template yielded a --config argument, so the skip "
        "branch of test_orchestrator_service_points_at_canonical_config_filename "
        "is never exercised. orchestrator-watchdog.service is expected to have "
        f"none (it runs a probe script, not the orchestrator). Parsed: {parsed}"
    )


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
    an exception) means a future unit is covered the day it lands.  That skip
    is itself guarded — see
    test_exec_start_config_parser_answers_for_every_orchestrator_run_unit,
    which fails if a unit that runs the orchestrator ever lands in it — and a
    MALFORMED ExecStart= raises MalformedExecStart out of here rather than
    skipping, because "broken unit" is not "unit takes no --config".
    """
    content = service_path.read_text(encoding="utf-8")
    config_arg = _exec_start_config_arg(content, service_path.name)
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
