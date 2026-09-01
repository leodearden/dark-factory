"""Tests for scripts/check_fused_memory_unit_parity.py.

All drift-logic tests run against tmp_path fixtures — never the host's installed
unit — so the suite is portable across CI and dev machines that may lack
~/.config/systemd/user/fused-memory.service.

Module loading: scripts/ is not a package / not on sys.path, so we load
check_fused_memory_unit_parity.py via importlib.util.spec_from_file_location,
mirroring the pattern in tests/scripts/test_orchestrator_watchdog.py::_load_watchdog.
"""

import importlib.util
import pathlib
import subprocess
import sys
import types

import pytest
from setup_host_sections import (
    enabled_units,
    run_section,
    slice_section,
    systemctl_calls,
    usage_error_checker,
    write_checker,
)

REPO_ROOT = pathlib.Path(__file__).parents[2]
CHECKER_PATH = REPO_ROOT / "scripts" / "check_fused_memory_unit_parity.py"
TEMPLATE_PATH = REPO_ROOT / "scripts" / "fused-memory.service.template"


def _load_checker() -> types.ModuleType:
    """Load scripts/check_fused_memory_unit_parity.py by file path."""
    spec = importlib.util.spec_from_file_location(
        "check_fused_memory_unit_parity", CHECKER_PATH
    )
    assert spec is not None, f"Could not build spec from {CHECKER_PATH}"
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# parse_unit_sections tests  (step-1 / step-2)
# ---------------------------------------------------------------------------

_SAMPLE_UNIT = """\
[Unit]
Description=Test Service
# this is a comment
; this too

[Service]
Type=simple
# commented-out line
Environment=FOO=bar
WatchdogSec=120

[Install]
WantedBy=default.target
"""


def test_parse_unit_sections_keys():
    """parse_unit_sections returns a dict keyed by section name."""
    mod = _load_checker()
    sections = mod.parse_unit_sections(_SAMPLE_UNIT)
    assert set(sections.keys()) == {"Unit", "Service", "Install"}


def test_parse_unit_sections_skips_comments_and_blanks():
    """Comment and blank lines are not included in section values."""
    mod = _load_checker()
    sections = mod.parse_unit_sections(_SAMPLE_UNIT)
    service = sections["Service"]
    assert not any(line.startswith("#") for line in service)
    assert not any(line.startswith(";") for line in service)
    assert "" not in service


def test_parse_unit_sections_service_directives():
    """Directives under [Service] are returned in sections['Service']."""
    mod = _load_checker()
    sections = mod.parse_unit_sections(_SAMPLE_UNIT)
    assert "Environment=FOO=bar" in sections["Service"]
    assert "WatchdogSec=120" in sections["Service"]
    # The commented-out line must NOT appear
    assert "# commented-out line" not in sections["Service"]


def test_parse_unit_sections_unit_directives():
    """Directives under [Unit] are returned in sections['Unit']."""
    mod = _load_checker()
    sections = mod.parse_unit_sections(_SAMPLE_UNIT)
    assert "Description=Test Service" in sections["Unit"]


# ---------------------------------------------------------------------------
# find_drift tests  (step-3 / step-4)
# ---------------------------------------------------------------------------

_CLEAN_UNIT = """\
[Unit]
Description=Clean Service

[Service]
Type=simple
Environment=MEM0_TELEMETRY=false
WatchdogSec=120
ExecStartPre=/usr/bin/docker compose -f /repo/fused-memory/docker/docker-compose.yml up -d falkordb qdrant
Restart=on-failure
RestartSec=5
RestartSteps=4
TimeoutStartSec=300
TimeoutStopSec=90

[Install]
WantedBy=default.target
"""

_MISSING_MEM0_UNIT = """\
[Unit]
Description=Drifted Service

[Service]
Type=simple
WatchdogSec=120

[Install]
WantedBy=default.target
"""

_COMMENTED_MEM0_UNIT = """\
[Unit]
Description=Commented Service

[Service]
Type=simple
# Environment=MEM0_TELEMETRY=false
WatchdogSec=120

[Install]
WantedBy=default.target
"""


def test_find_drift_returns_empty_for_clean_unit():
    """find_drift returns [] when all required directives are present."""
    mod = _load_checker()
    assert mod.find_drift(_CLEAN_UNIT) == []


def test_find_drift_detects_missing_mem0_telemetry():
    """find_drift returns ['Environment=MEM0_TELEMETRY=false'] when line is absent."""
    mod = _load_checker()
    drift = mod.find_drift(_MISSING_MEM0_UNIT)
    assert "Environment=MEM0_TELEMETRY=false" in drift


def test_find_drift_treats_commented_line_as_missing():
    """A commented-out directive still counts as missing (non-comment match required)."""
    mod = _load_checker()
    drift = mod.find_drift(_COMMENTED_MEM0_UNIT)
    assert "Environment=MEM0_TELEMETRY=false" in drift


def test_find_drift_does_not_report_present_watchdog():
    """WatchdogSec=120 present and uncommented is not flagged."""
    mod = _load_checker()
    drift = mod.find_drift(_CLEAN_UNIT)
    assert "WatchdogSec=120" not in drift


# ---------------------------------------------------------------------------
# find_drift tests: restart-relevant EXACT directives (A1)
# ---------------------------------------------------------------------------

_MISSING_RESTART_DIRECTIVES_UNIT = """\
[Unit]
Description=Drifted Service Missing Restart Directives

[Service]
Type=notify
Environment=MEM0_TELEMETRY=false
WatchdogSec=120
ExecStartPre=/usr/bin/docker compose -f /repo/fused-memory/docker/docker-compose.yml up -d falkordb qdrant

[Install]
WantedBy=default.target
"""

# The pre-fix restart shape: a backoff cap declared with a floor but no steps.
# systemd parses RestartMaxDelaySec=, logs "Service has RestartMaxDelaySec= but
# no RestartSteps= setting. Ignoring." at unit load, and discards the cap — so
# this unit's backoff never grows past RestartSec. Nothing in its text says so.
_MISSING_RESTART_STEPS_UNIT = """\
[Unit]
Description=Installed Fused Memory (cap declared, no steps)

[Service]
Type=notify
Environment=MEM0_TELEMETRY=false
WatchdogSec=120
ExecStartPre=/usr/bin/docker compose -f /repo/fused-memory/docker/docker-compose.yml up -d falkordb qdrant
Restart=on-failure
RestartSec=5
RestartMaxDelaySec=60
TimeoutStartSec=300
TimeoutStopSec=90

[Install]
WantedBy=default.target
"""

# A fully-populated [Service] except an injected divergence: TimeoutStopSec=45
# instead of the required TimeoutStopSec=90. Exact-match semantics mean a
# *different* value for the same key still counts as the required directive
# being absent.
_DIVERGENT_TIMEOUT_STOP_SEC_UNIT = """\
[Unit]
Description=Installed Fused Memory (divergent TimeoutStopSec)

[Service]
Type=notify
Environment=MEM0_TELEMETRY=false
WatchdogSec=120
ExecStartPre=/usr/bin/docker compose -f /repo/fused-memory/docker/docker-compose.yml up -d falkordb qdrant
Restart=on-failure
RestartSec=5
TimeoutStartSec=300
TimeoutStopSec=45

[Install]
WantedBy=default.target
"""


def test_find_drift_detects_missing_restart_directives():
    """find_drift flags all four restart-relevant directives when absent."""
    mod = _load_checker()
    drift = mod.find_drift(_MISSING_RESTART_DIRECTIVES_UNIT)
    assert "Restart=on-failure" in drift
    assert "RestartSec=5" in drift
    assert "TimeoutStartSec=300" in drift
    assert "TimeoutStopSec=90" in drift


def test_find_drift_detects_missing_restart_steps():
    """find_drift must flag a missing RestartSteps= on the INSTALLED host unit.

    Why the CHECKER has to know this directive, not just the template:
    scripts/setup-host.sh runs this checker against
    ~/.config/systemd/user/fused-memory.service, and --fix appends whatever it
    reports missing.  Fixing only scripts/fused-memory.service.template would
    leave the LIVE unit's RestartMaxDelaySec= inert — systemd parses the cap,
    warns, and discards it — while the drift check reported green.  That is the
    same silently-ignored-directive failure class this task exists to kill,
    merely relocated from the unit file to the installer.

    The fixture below is exactly the pre-fix shape: a cap declared with a floor
    and no steps to interpolate over.
    """
    mod = _load_checker()
    drift = mod.find_drift(_MISSING_RESTART_STEPS_UNIT)
    assert "RestartSteps=4" in drift, (
        "find_drift did not flag the absent RestartSteps=4 on a unit that "
        "declares RestartMaxDelaySec=60. setup-host.sh relies on this checker "
        "to carry the directive to the installed unit; without it the host's "
        "backoff cap stays inert while the drift check reports clean."
    )


def test_find_drift_detects_injected_timeout_stop_sec_divergence():
    """A divergent TimeoutStopSec value (45 vs required 90) is flagged as missing.

    Exact-match semantics: the required directive is the full string
    'TimeoutStopSec=90', so a unit carrying a *different* value for the same
    key still fails the check — the correct value is effectively absent.
    """
    mod = _load_checker()
    drift = mod.find_drift(_DIVERGENT_TIMEOUT_STOP_SEC_UNIT)
    assert "TimeoutStopSec=90" in drift


def test_parity_checker_subprocess_exit_1_on_divergent_timeout_stop_sec(
    tmp_path: pathlib.Path,
):
    """The divergent-TimeoutStopSec unit yields exit code 1 through the real CLI.

    Mirrors test_parity_checker_callable_as_subprocess's subprocess-boundary
    pattern, but for the injected TimeoutStopSec divergence case (A1c).
    """
    divergent = _write_unit(tmp_path, _DIVERGENT_TIMEOUT_STOP_SEC_UNIT, name="divergent.service")
    result = subprocess.run(
        [sys.executable, str(CHECKER_PATH), "--installed", str(divergent)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, (
        f"Expected exit 1 for divergent TimeoutStopSec unit; got {result.returncode}. "
        f"stderr: {result.stderr}"
    )


# ---------------------------------------------------------------------------
# Template parity test  (step-5 / step-6)
# ---------------------------------------------------------------------------


def test_template_satisfies_required_directives():
    """The committed template itself must pass find_drift (no missing safety switches).

    This is RED until Environment=MEM0_TELEMETRY=false is added to the template
    (step-6), because the template currently lacks that line.
    """
    mod = _load_checker()
    template_text = TEMPLATE_PATH.read_text(encoding="utf-8")
    drift = mod.find_drift(template_text)
    assert drift == [], (
        f"scripts/fused-memory.service.template is missing host-invariant safety "
        f"directives: {drift}. Add them to the template [Service] section."
    )


# ---------------------------------------------------------------------------
# fix_unit_text tests  (step-7 / step-8)
# ---------------------------------------------------------------------------

_DRIFTED_WITH_HOST_SPECIFIC = """\
[Unit]
Description=Installed Fused Memory

[Service]
Type=notify
WatchdogSec=120
ExecStartPre=/usr/bin/docker compose -f /repo/fused-memory/docker/docker-compose.yml up -d falkordb qdrant
Environment=DASHBOARD_KNOWN_PROJECT_ROOTS=/home/leo/src/dark-factory,/home/leo/src/other

[Install]
WantedBy=default.target
"""


def test_fix_unit_text_adds_missing_directive():
    """fix_unit_text inserts Environment=MEM0_TELEMETRY=false into [Service]."""
    mod = _load_checker()
    fixed = mod.fix_unit_text(_DRIFTED_WITH_HOST_SPECIFIC)
    sections = mod.parse_unit_sections(fixed)
    assert "Environment=MEM0_TELEMETRY=false" in sections["Service"]


def test_fix_unit_text_preserves_host_specific_line():
    """fix_unit_text must not remove the host-specific DASHBOARD_KNOWN_PROJECT_ROOTS."""
    mod = _load_checker()
    fixed = mod.fix_unit_text(_DRIFTED_WITH_HOST_SPECIFIC)
    assert (
        "Environment=DASHBOARD_KNOWN_PROJECT_ROOTS=/home/leo/src/dark-factory,/home/leo/src/other"
        in fixed
    )


def test_fix_unit_text_is_idempotent():
    """Applying fix_unit_text twice must not duplicate the appended directive."""
    mod = _load_checker()
    once = mod.fix_unit_text(_DRIFTED_WITH_HOST_SPECIFIC)
    twice = mod.fix_unit_text(once)
    assert once == twice


# ---------------------------------------------------------------------------
# find_drift / fix_unit_text tests: ExecStartPre presence (A2)
# ---------------------------------------------------------------------------

_ALL_EXACT_NO_EXECSTARTPRE_UNIT = """\
[Unit]
Description=All Exact Directives Present, No ExecStartPre

[Service]
Type=notify
Environment=MEM0_TELEMETRY=false
WatchdogSec=120
Restart=on-failure
RestartSec=5
RestartSteps=4
TimeoutStartSec=300
TimeoutStopSec=90

[Install]
WantedBy=default.target
"""


def test_find_drift_flags_missing_execstartpre():
    """find_drift flags 'ExecStartPre=' as missing when no [Service] line has that prefix.

    RED until find_drift gains a required_prefixes parameter — today it only
    checks REQUIRED_SERVICE_DIRECTIVES via exact membership, so a unit that
    satisfies every exact directive but lacks any ExecStartPre= line is
    (incorrectly) reported as fully clean.
    """
    mod = _load_checker()
    drift = mod.find_drift(_ALL_EXACT_NO_EXECSTARTPRE_UNIT)
    assert "ExecStartPre=" in drift


def test_find_drift_does_not_flag_present_execstartpre():
    """find_drift does not flag 'ExecStartPre=' when a matching-prefix line is present."""
    mod = _load_checker()
    drift = mod.find_drift(_CLEAN_UNIT)
    assert "ExecStartPre=" not in drift


def test_fix_unit_text_never_synthesizes_bare_execstartpre():
    """fix_unit_text appends the missing EXACT directives but never a bare 'ExecStartPre=' line.

    A host-specific prefix value (e.g. carrying __REPO_ROOT__ or
    /home/leo/bin paths) cannot be synthesized by --fix — only its presence
    can be checked, never its correct value guessed. Exercised against a unit
    missing BOTH ExecStartPre and the exact directives.
    """
    mod = _load_checker()
    fixed = mod.fix_unit_text(_MISSING_MEM0_UNIT)
    sections = mod.parse_unit_sections(fixed)
    assert "Environment=MEM0_TELEMETRY=false" in sections["Service"]
    assert "Restart=on-failure" in sections["Service"]
    assert "RestartSec=5" in sections["Service"]
    assert "TimeoutStartSec=300" in sections["Service"]
    assert "TimeoutStopSec=90" in sections["Service"]
    assert not any(line.startswith("ExecStartPre=") for line in sections["Service"]), (
        "fix_unit_text must never synthesize an ExecStartPre= line (host-specific value)"
    )


# ---------------------------------------------------------------------------
# main(argv) CLI tests  (step-9 / step-10)
# ---------------------------------------------------------------------------


def _write_unit(tmp_path: pathlib.Path, content: str, name: str = "fused-memory.service") -> pathlib.Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def test_main_returns_0_on_parity(tmp_path: pathlib.Path):
    """main() returns 0 when the installed unit has all required directives."""
    mod = _load_checker()
    installed = _write_unit(tmp_path, _CLEAN_UNIT)
    rc = mod.main(["--installed", str(installed)])
    assert rc == 0


def test_main_returns_1_on_drift(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture):
    """main() returns 1 and prints missing directives when the unit has drift."""
    mod = _load_checker()
    installed = _write_unit(tmp_path, _MISSING_MEM0_UNIT)
    rc = mod.main(["--installed", str(installed)])
    assert rc == 1
    captured = capsys.readouterr()
    assert "MEM0_TELEMETRY" in captured.out or "MEM0_TELEMETRY" in captured.err


def test_main_returns_2_when_installed_absent(tmp_path: pathlib.Path):
    """main() returns 2 when the installed unit file does not exist."""
    mod = _load_checker()
    missing = tmp_path / "nonexistent.service"
    rc = mod.main(["--installed", str(missing)])
    assert rc == 2


def test_main_fix_rewrites_file(tmp_path: pathlib.Path):
    """main(--fix) rewrites the installed unit to add missing directives."""
    mod = _load_checker()
    installed = _write_unit(tmp_path, _DRIFTED_WITH_HOST_SPECIFIC)
    rc = mod.main(["--installed", str(installed), "--fix"])
    # After --fix the unit must have parity
    fixed_text = installed.read_text(encoding="utf-8")
    sections = mod.parse_unit_sections(fixed_text)
    assert "Environment=MEM0_TELEMETRY=false" in sections["Service"]
    assert rc == 0


def test_main_fix_calls_daemon_reload(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    """main(--fix) on a drifted unit invokes daemon_reload()."""
    mod = _load_checker()
    installed = _write_unit(tmp_path, _DRIFTED_WITH_HOST_SPECIFIC)
    reload_calls: list[None] = []

    def _fake_reload() -> None:
        reload_calls.append(None)

    monkeypatch.setattr(mod, "daemon_reload", _fake_reload)
    mod.main(["--installed", str(installed), "--fix"])
    assert len(reload_calls) == 1


# ---------------------------------------------------------------------------
# LOG_TAG contract  (task 3909)
# ---------------------------------------------------------------------------
#
# Modelled on the sibling pin at
# tests/scripts/test_check_dashboard_unit_parity.py::test_main_every_emitted_line_carries_the_log_tag.
#
# setup-host.sh's fused-memory gate believes an exit status only when the
# checker's own bracketed tag is present in the captured output. That test is
# CONCLUSIVE rather than heuristic only if EVERY emitted line carries the tag —
# an untagged continuation line is a line the gate cannot attribute, and a
# report whose tag it cannot find reads to the gate as "it did not run".


def test_main_every_emitted_line_carries_the_log_tag(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture
):
    """Every printed line is prefixed with the log tag, INCLUDING continuations.

    Driven down the MULTI-LINE drift path on purpose. This checker's drift
    report is ONE print carrying a joined `  - {directive}` list:

        [drift] <path>: missing required directives:
          - Restart=on-failure
          - RestartSec=5

    so a tag prefixed only to the first physical line would leave every
    continuation untagged — and the "absence of the tag is conclusive" claim
    setup-host.sh's gate rests on would be false on the one path an operator
    most needs to trust.
    """
    mod = _load_checker()
    assert mod.LOG_TAG == "fused_memory_unit_parity"

    # Missing FIVE required directives — comfortably past one, so the report is
    # unambiguously multi-line and the continuation lines are exercised.
    installed = _write_unit(tmp_path, _MISSING_RESTART_DIRECTIVES_UNIT)
    rc = mod.main(["--installed", str(installed)])
    captured = capsys.readouterr()

    assert rc == 1, f"{captured.out}\n{captured.err}"
    assert captured.out.strip(), "The checker must report something."
    for line in (captured.out + captured.err).splitlines():
        if not line.strip():
            continue
        assert line.startswith(f"[{mod.LOG_TAG}]"), f"Untagged output line: {line!r}"


def test_log_tags_every_physical_line_of_a_multi_line_message(
    capsys: pytest.CaptureFixture,
):
    """`_log` itself splits and tags, rather than prefixing the message once.

    Pinned on the helper directly, not only through main(), because the
    invariant is a property of the helper: this checker interpolates FOREIGN
    text into its output (a captured `exc.stderr.decode()` in daemon_reload),
    whose shape no test of main() can enumerate. Tagging per physical line is
    what makes the invariant hold for text the checker did not author.
    """
    mod = _load_checker()

    mod._log("first\nsecond")
    out = capsys.readouterr().out

    assert out.splitlines() == [
        f"[{mod.LOG_TAG}] first",
        f"[{mod.LOG_TAG}] second",
    ], out


# ---------------------------------------------------------------------------
# CLI --fix residual (un-synthesizable prefix) tests  (amendment)
# ---------------------------------------------------------------------------


def test_main_fix_returns_1_when_execstartpre_unsynthesizable(
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
):
    """main(--fix) returns 1 (not a false 0) when a host-specific prefix directive remains.

    _MISSING_MEM0_UNIT lacks every exact directive AND any ExecStartPre= line.
    --fix appends the exact directives but cannot synthesize the host-specific
    ExecStartPre value, so parity is NOT reached. The CLI must report the
    residual drift and exit 1 — matching what a follow-up plain verify would
    return — instead of falsely signalling parity with exit 0.
    """
    mod = _load_checker()
    monkeypatch.setattr(mod, "daemon_reload", lambda: None)
    installed = _write_unit(tmp_path, _MISSING_MEM0_UNIT)
    rc = mod.main(["--installed", str(installed), "--fix"])
    assert rc == 1
    combined = "".join(capsys.readouterr())
    assert "ExecStartPre=" in combined, (
        "The residual, un-synthesizable directive must be named in the output."
    )


def test_main_fix_appended_count_excludes_unsynthesizable_prefix(
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
):
    """The '[fixed] Appended N' count reflects only the EXACT directives appended.

    find_drift(_MISSING_MEM0_UNIT) returns the 5 missing exact directives PLUS
    the ExecStartPre= prefix miss (6 total), but fix_unit_text only appends the
    5 exact directives. The reported count must therefore be 5, never the
    inflated len(drift)==6.
    """
    mod = _load_checker()
    monkeypatch.setattr(mod, "daemon_reload", lambda: None)
    full_drift = mod.find_drift(_MISSING_MEM0_UNIT)
    exact_only = mod.find_drift(_MISSING_MEM0_UNIT, required_prefixes=())
    assert len(full_drift) == len(exact_only) + 1  # exactly the ExecStartPre miss
    installed = _write_unit(tmp_path, _MISSING_MEM0_UNIT)
    mod.main(["--installed", str(installed), "--fix"])
    out = capsys.readouterr().out
    assert f"Appended {len(exact_only)} directive" in out
    assert f"Appended {len(full_drift)} directive" not in out


# ---------------------------------------------------------------------------
# CLI subprocess test  (step-11 / step-12)
# ---------------------------------------------------------------------------


def test_parity_checker_callable_as_subprocess(tmp_path: pathlib.Path):
    """The checker is invokable as a standalone subprocess.

    Validates the actual CLI entry point (the interface that setup-host.sh and
    operators use) by running it against tmp_path fixtures and asserting on
    exit codes rather than on source-text substrings.

    - Clean unit  → exit 0
    - Drifted unit → exit 1 and prints the missing directive to stdout
    """
    clean = _write_unit(tmp_path, _CLEAN_UNIT, name="clean.service")
    result_clean = subprocess.run(
        [sys.executable, str(CHECKER_PATH), "--installed", str(clean)],
        capture_output=True,
        text=True,
    )
    assert result_clean.returncode == 0, (
        f"Expected exit 0 for clean unit; got {result_clean.returncode}. "
        f"stderr: {result_clean.stderr}"
    )

    drifted = _write_unit(tmp_path, _MISSING_MEM0_UNIT, name="drifted.service")
    result_drifted = subprocess.run(
        [sys.executable, str(CHECKER_PATH), "--installed", str(drifted)],
        capture_output=True,
        text=True,
    )
    assert result_drifted.returncode == 1, (
        f"Expected exit 1 for drifted unit; got {result_drifted.returncode}. "
        f"stderr: {result_drifted.stderr}"
    )
    assert "MEM0_TELEMETRY" in result_drifted.stdout, (
        "Expected drift output to name the missing directive on stdout."
    )


# ---------------------------------------------------------------------------
# The parity GATE in setup-host.sh is wired so it can actually stop something
# ---------------------------------------------------------------------------
#
# Everything above tests the CHECKER. This group tests its WIRING — the block
# in scripts/setup-host.sh that runs it and decides what its exit status meant.
#
# The defect these pin: that block believed a bare exit status, and 2 is
# overloaded three ways — the checker's benign "not installed on this host",
# `python3` refusing to open a missing script, and argparse rejecting an
# unknown flag. So renaming the checker or one of its flags made the installer
# print a reassuring "not installed ... (skipping parity check)" and move on: a
# gate reporting green because it never ran, which is the exact silent-drift
# failure the checker exists to catch, reproduced one level up in its own
# wiring.
#
# Nothing here touches ~/.config/systemd/user or real systemd: REPO_ROOT and
# UNIT_DIR are tmp_path trees and `systemctl` is a PATH stub that exits 0.

# Anchored on the block's hoisted `_fm_parity_script=` assignment — CODE, and
# unique to this site — not on the section comment above it. A comment anchor
# turns a reworded comment or a fixed typo into a red CI run for no behavioural
# change, and it is the same line the structural sweep in
# test_check_dashboard_unit_parity.py keys on, so both share one anchor.
_GATE_START = "_fm_parity_script="
_GATE_END = "\nfi\n"

# The status is believed only when the checker's own tag is present — the same
# test the dashboard, orchestrator and lms gates apply, which the checker's
# LOG_TAG (task 3909) made available on this side too. Not line-anchored and
# not a regex in the gate itself: setup-host.sh uses bash's own
# `[[ "$out" != *'[fused_memory_unit_parity]'* ]]` containment test.
# Containment is safe here where a bare `[ok]` alternation was not, because no
# imposter emits this string — measured on both, below.
#
# test_gate_tag_appears_on_every_real_exit_path_and_neither_collision is the
# contract pin keeping the tag from drifting out from under the gate.
_REPORT_TAG = "[fused_memory_unit_parity]"


def _gate_repo(
    tmp_path: pathlib.Path,
    *,
    checker_body: str | None = None,
    with_checker: bool = True,
) -> pathlib.Path:
    """A tmp repo root holding the template and (optionally) the checker.

    The checker is copied from the real repo so the gate drives the real one;
    only the TREE is fake. It imports nothing from scripts/, so one file is the
    whole dependency.
    """
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True, exist_ok=True)
    (repo / "scripts" / "fused-memory.service.template").write_text(
        TEMPLATE_PATH.read_text(encoding="utf-8"), encoding="utf-8"
    )
    if with_checker:
        write_checker(repo, CHECKER_PATH.name, body=checker_body)
    return repo


def _gate_unit_dir(
    tmp_path: pathlib.Path, *, content: str | None = None
) -> pathlib.Path:
    """A tmp UNIT_DIR, optionally holding an installed fused-memory.service."""
    unit_dir = tmp_path / "installed"
    unit_dir.mkdir(parents=True, exist_ok=True)
    if content is not None:
        (unit_dir / "fused-memory.service").write_text(content, encoding="utf-8")
    return unit_dir


def _run_gate(
    tmp_path: pathlib.Path, repo: pathlib.Path, unit_dir: pathlib.Path
) -> subprocess.CompletedProcess:
    return run_section(
        tmp_path,
        slice_section(_GATE_START, _GATE_END),
        repo_root=repo,
        unit_dir=unit_dir,
    )


# The argparse-shaped stub: exit 2, usage-shaped stderr, and no report marker —
# what renaming a flag in a future refactor would actually produce. The real
# flag spellings are kept because their bracketed tokens ([-h], [--fix]) are
# precisely what a LOOSE bracket match would misread as a report; this gate now
# tests containment of its own [fused_memory_unit_parity] tag, which the stub
# never emits.
_USAGE_ERROR_CHECKER = usage_error_checker(
    CHECKER_PATH.name,
    "[-h] [--installed INSTALLED] [--template TEMPLATE] [--fix]",
    "--installed",
)


def test_gate_tag_appears_on_every_real_exit_path_and_neither_collision(
    tmp_path: pathlib.Path,
):
    """CONTRACT PIN: the TAG the gate depends on, on every path that produced it.

    The gate believes a status only when the checker's own
    [fused_memory_unit_parity] tag is in the captured output.
    test_main_every_emitted_line_carries_the_log_tag pins that the checker puts
    it on every line it emits; this pins the other half — that the three REAL
    exit paths each emit something, and that NEITHER exit-2 imposter does.
    Together they make the gate's inference sound in both directions.

    A future change to the checker's output then fails as a legible
    contract-test failure, instead of silently wedging the installer's gate
    into permanent "it did not run".

    stdout and stderr are MERGED throughout, because the [skip] line and the
    drift trailer go to stderr — which is also why the gate captures 2>&1.
    """
    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    def _merged(argv: list[str]) -> tuple[int, str]:
        result = subprocess.run(
            [sys.executable, *argv],
            capture_output=True,
            text=True,
        )
        return result.returncode, result.stdout + result.stderr

    # exit 0 — a unit that carries every required directive.
    parity = _write_unit(tmp_path, template, name="parity.service")
    rc, out = _merged([str(CHECKER_PATH), "--installed", str(parity)])
    assert rc == 0, out
    assert _REPORT_TAG in out, f"exit 0 emitted no tagged report:\n{out}"

    # exit 1 — a required directive removed.
    drifted = _write_unit(tmp_path, _MISSING_MEM0_UNIT, name="drifted.service")
    rc, out = _merged([str(CHECKER_PATH), "--installed", str(drifted)])
    assert rc == 1, out
    assert _REPORT_TAG in out, f"exit 1 emitted no tagged report:\n{out}"

    # exit 2 — the installed unit is absent.
    rc, out = _merged(
        [str(CHECKER_PATH), "--installed", str(tmp_path / "nonexistent.service")]
    )
    assert rc == 2, out
    assert _REPORT_TAG in out, f"exit 2 emitted no tagged report:\n{out}"

    # COLLISION 1 — argparse usage error also exits 2. Its output carries
    # bracketed tokens ([-h], [--fix]) and names the script, but neither is the
    # tag: `usage: check_fused_memory_unit_parity.py` carries no brackets around
    # the name, and the flag spellings are not it.
    rc, out = _merged([str(CHECKER_PATH), "--bogus"])
    assert rc == 2, out
    assert _REPORT_TAG not in out, (
        f"An argparse usage error must NOT look like a report:\n{out}"
    )

    # COLLISION 2 — python3 refusing to open a renamed/moved script also exits
    # 2. Its message carries `[Errno 2]` and the bare (bracket-free) path.
    rc, out = _merged([str(tmp_path / "check_fused_memory_unit_parity_RENAMED.py")])
    assert rc == 2, out
    assert _REPORT_TAG not in out, (
        f"A missing script must NOT look like a report:\n{out}"
    )


def test_gate_missing_checker_does_not_read_as_not_installed(
    tmp_path: pathlib.Path,
):
    """EXIT-CODE COLLISION: `python3 <missing script>` also exits 2.

    2 is the checker's benign "not installed on this host". If the checker were
    renamed or moved, python3's own 2 would land in that same branch and the
    gate would report the reassuring "skipping parity check" — telling the
    operator a check was skipped for a benign reason when in fact no check ran
    at all.
    """
    repo = _gate_repo(tmp_path, with_checker=False)
    unit_dir = _gate_unit_dir(
        tmp_path, content=TEMPLATE_PATH.read_text(encoding="utf-8")
    )

    result = _run_gate(tmp_path, repo, unit_dir)

    assert result.returncode == 0, (
        "The gate is non-fatal — fail() only printfs, so it must not abort the "
        f"health-check section.\n{result.stdout}\n{result.stderr}"
    )
    assert "not installed at" not in result.stdout, (
        "A missing checker was reported as the benign 'not installed on this "
        f"host'.\n{result.stdout}"
    )
    assert "FAIL " in result.stdout, (
        f"A gate that did not run must say so loudly.\n{result.stdout}"
    )


def test_gate_usage_error_does_not_read_as_not_installed(tmp_path: pathlib.Path):
    """SAME COLLISION, second source: argparse exits 2 on any usage error.

    Simulated with a stub checker exiting 2 with argparse-shaped stderr and no
    report marker — what renaming a flag in a future refactor would produce.
    The marker, not the exit code, is what makes a status believable.
    """
    repo = _gate_repo(tmp_path, checker_body=_USAGE_ERROR_CHECKER)
    unit_dir = _gate_unit_dir(
        tmp_path, content=TEMPLATE_PATH.read_text(encoding="utf-8")
    )

    result = _run_gate(tmp_path, repo, unit_dir)

    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "not installed at" not in result.stdout, result.stdout
    assert "FAIL " in result.stdout, (
        f"A gate that did not run must say so loudly.\n{result.stdout}"
    )


# A checker that RAN and reported in the legacy marker vocabulary, exiting 0,
# but emits no tag — what reshaping or replacing the checker would produce.
# The bare `[ok]` is deliberate: it is precisely what the gate used to believe.
_UNTAGGED_OK_CHECKER = (
    "import sys\n"
    "print('[ok] parity — all required directives present.')\n"
    "sys.exit(0)\n"
)


def test_gate_untagged_report_is_not_believed(tmp_path: pathlib.Path):
    """A checker reporting WITHOUT the tag gets no verdict, even exiting 0.

    The failure this closes is not a renamed script or a rejected flag — both
    already covered — but a checker that was RESHAPED or REPLACED. Exit 0 plus
    a plausible-looking report is the most believable possible input, and it is
    still not evidence about this host unless the tag says which script
    produced it. The old marker alternation believed exactly this stub.
    """
    repo = _gate_repo(tmp_path, checker_body=_UNTAGGED_OK_CHECKER)
    unit_dir = _gate_unit_dir(
        tmp_path, content=TEMPLATE_PATH.read_text(encoding="utf-8")
    )

    result = _run_gate(tmp_path, repo, unit_dir)

    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "FAIL " in result.stdout, (
        f"An untagged report must not be believed.\n{result.stdout}"
    )
    assert "OK " not in result.stdout, (
        f"An untagged exit 0 was reported as a verdict.\n{result.stdout}"
    )
    assert "parity with template" not in result.stdout, result.stdout


def test_gate_reports_parity_when_the_installed_unit_matches(
    tmp_path: pathlib.Path,
):
    """Happy path — the fix must not degenerate into 'always report failure'."""
    repo = _gate_repo(tmp_path)
    unit_dir = _gate_unit_dir(
        tmp_path, content=TEMPLATE_PATH.read_text(encoding="utf-8")
    )

    result = _run_gate(tmp_path, repo, unit_dir)

    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "OK " in result.stdout, result.stdout
    assert "parity with template" in result.stdout, result.stdout
    assert "FAIL " not in result.stdout, result.stdout


def test_gate_reports_drift_when_a_required_directive_is_missing(
    tmp_path: pathlib.Path,
):
    """Real drift stays a DRIFT warning naming --fix — not a "did not run" fail."""
    repo = _gate_repo(tmp_path)
    unit_dir = _gate_unit_dir(tmp_path, content=_MISSING_MEM0_UNIT)

    result = _run_gate(tmp_path, repo, unit_dir)

    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "DRIFT detected" in result.stdout, result.stdout
    assert "--fix" in result.stdout, result.stdout
    assert "FAIL " not in result.stdout, (
        f"Drift is a real verdict from a gate that RAN.\n{result.stdout}"
    )


def test_gate_reports_skip_when_the_unit_is_genuinely_not_installed(
    tmp_path: pathlib.Path,
):
    """The benign branch must survive: a real exit 2 still reads as 'not installed'."""
    repo = _gate_repo(tmp_path)
    unit_dir = _gate_unit_dir(tmp_path)  # no fused-memory.service written

    result = _run_gate(tmp_path, repo, unit_dir)

    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "not installed at" in result.stdout, result.stdout
    assert "FAIL " not in result.stdout, (
        f"A genuine 'not installed' is a real verdict.\n{result.stdout}"
    )


# ---------------------------------------------------------------------------
# ACCEPTANCE 3 — the checker, INCLUDING --fix, cannot undo the preservation
# (step-7)
# ---------------------------------------------------------------------------
# Task 4796 made setup-host.sh render this unit through
# scripts/render_dashboard_unit.py so a host's extra
# DASHBOARD_KNOWN_PROJECT_ROOTS entries survive a re-provision. The obvious
# question is whether THIS checker then undoes it on the next run.
#
# It does not, and the reason is structural rather than incidental:
# DASHBOARD_KNOWN_PROJECT_ROOTS is absent from REQUIRED_SERVICE_DIRECTIVES, and
# fix_unit_text only APPENDS missing members of `required` — "Never removes or
# reorders any existing line". So (i) and (ii) below are expected to pass the
# moment they are written. They are REGRESSION GUARDS over a latent property,
# not a red-to-green cycle, and (iii) is what makes them load-bearing rather
# than decorative: it demonstrates that the hazard they forbid is real.
#
# THE HAZARD. REQUIRED_SERVICE_DIRECTIVES' own comment says "Extend this list to
# guard additional safety flags". Add
# `Environment=DASHBOARD_KNOWN_PROJECT_ROOTS=<repo_root>` there and, on a
# correctly-preserved MULTI-root host, find_drift's exact WHOLE-LINE membership
# reports it missing — so --fix appends the single-root line after the last
# [Service] line, where systemd's LAST-WINS silently beats the preserved value.
# On the unit that governs reconciliation, and invisibly: the checker then
# exits 0. No code inside either module can prevent an edit to a constant in
# the other, which is why the instrument is a cross-module invariant test.

_PRESERVED_NAME = "DASHBOARD_KNOWN_PROJECT_ROOTS"

# The nine project roots measured on this host, the value the old truncating
# `sed ... > "$UNIT_DIR/fused-memory.service"` collapsed to one on every
# re-run. Spelled here rather than imported from
# tests/scripts/test_render_dashboard_unit.py: a test module importing another
# test module makes collection order load-bearing, and this is fixture data
# whose only contract is "more than one root, realistically shaped".
_MULTI_ROOTS = ",".join(
    f"/home/leo/src/{name}"
    for name in (
        "dark-factory",
        "reify",
        "autopilot-video",
        "autotrade",
        "know-live",
        "solar-challenge",
        "mission-control",
        "solar-challenge-platform",
        "pump-web-ui",
    )
)

_FM_REPO_ROOT = "/home/leo/src/dark-factory"
_FM_UV_PATH = "/home/leo/.local/bin/uv"


def _render_fused_unit(repo_root: str = _FM_REPO_ROOT) -> str:
    """The committed template rendered — through the REAL renderer's substitution."""
    import render_dashboard_unit  # pyright: ignore[reportMissingImports]

    return render_dashboard_unit.render_template(
        TEMPLATE_PATH.read_text(encoding="utf-8"),
        repo_root=repo_root,
        uv_path=_FM_UV_PATH,
    )


def _multi_root_unit(repo_root: str = _FM_REPO_ROOT) -> str:
    """A correctly-configured host's installed unit: rendered, then nine roots in.

    Built BY RENDERING rather than hand-written, so the fixture cannot drift
    into a shape the real installer would never produce — the multi-root line's
    real shape is the whole subject here.
    """
    text = _render_fused_unit(repo_root)
    single = f"Environment={_PRESERVED_NAME}={repo_root}"
    assert single in text, f"fixture anchor {single!r} not found in the rendered template"
    return text.replace(single, f"Environment={_PRESERVED_NAME}={_MULTI_ROOTS}")


def _known_roots_lines(text: str) -> list[str]:
    prefix = f"Environment={_PRESERVED_NAME}="
    return [line for line in text.splitlines() if line.strip().startswith(prefix)]


def test_checker_and_fix_leave_a_multi_root_unit_untouched(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture
):
    """(i) BEHAVIOURAL: a correctly-preserved host sees neither drift nor a correction.

    Run BOTH ways against the same fixture, because they fail differently:
    a plain run that reported drift would send the operator to `--fix`, and a
    `--fix` run that rewrote or appended would re-clobber the value the
    installer had just preserved.
    """
    mod = _load_checker()
    installed = _write_unit(tmp_path, _multi_root_unit())
    before = installed.read_bytes()

    assert mod.main(["--installed", str(installed), "--template", str(TEMPLATE_PATH)]) == 0
    out = capsys.readouterr().out
    assert "[ok]" in out and "parity" in out, out

    assert (
        mod.main(
            ["--installed", str(installed), "--template", str(TEMPLATE_PATH), "--fix"]
        )
        == 0
    )
    capsys.readouterr()

    after = installed.read_text(encoding="utf-8")
    assert _known_roots_lines(after) == _known_roots_lines(
        _multi_root_unit()
    ), "--fix changed the preserved line"
    assert _MULTI_ROOTS in after, "--fix dropped this host's extra project roots"
    assert installed.read_bytes() == before, "--fix rewrote a unit that was at parity"
    assert mod.find_drift(after) == [], mod.find_drift(after)


def test_preserved_names_are_disjoint_from_required_service_directives():
    """(ii) THE INVARIANT, across BOTH modules and EVERY registered unit.

    No name any unit's spec preserves may appear as the VARIABLE of an exact
    `Environment=<NAME>=...` entry on this checker's required list. The two
    mechanisms are incompatible by construction: preservation says "this value
    is the host's", exact whole-line membership says "this value is the
    committed one", and when they disagree --fix wins by appending last.

    Iterates render_dashboard_unit.UNITS rather than naming one variable, so a
    third unit — or a second preserved name on an existing one — is covered the
    day it is registered.

    Held HERE rather than by an import in either module: check_* and render_*
    deliberately do not import each other (see render_dashboard_unit's module
    docstring — a cross-module import would ImportError under the section-8
    tmp-repo tests that replace one of them with a stub).
    """
    import render_dashboard_unit  # pyright: ignore[reportMissingImports]

    mod = _load_checker()

    preserved = {
        name
        for spec in render_dashboard_unit.UNITS.values()
        for name in spec.host_local_environment
    }
    required_env_vars = {
        directive[len("Environment=") :].split("=", 1)[0]
        for directive in mod.REQUIRED_SERVICE_DIRECTIVES
        if directive.startswith("Environment=") and "=" in directive[len("Environment=") :]
    }

    overlap = preserved & required_env_vars
    assert not overlap, (
        f"{sorted(overlap)} is BOTH preserved by scripts/render_dashboard_unit.py "
        "and exact-matched by REQUIRED_SERVICE_DIRECTIVES. On a host whose value "
        "differs from the committed one, find_drift reports the required line as "
        "missing and --fix APPENDS it after the last [Service] line, where "
        "systemd's last-wins silently beats the preserved value — and the "
        "checker then exits 0. See "
        "test_a_required_known_project_roots_line_would_reclobber for the "
        "demonstration."
    )


def test_a_required_known_project_roots_line_would_reclobber():
    """(iii) THE DEMONSTRATION that makes (ii) load-bearing rather than decorative.

    An invariant nobody can see the point of gets deleted as pedantry. This runs
    the forbidden configuration and shows the damage.

    The hypothetical extension is passed through fix_unit_text's EXISTING
    `required` parameter rather than by mutating the module constant: monkeypatching
    a module-level tuple would leak into whatever ran next in the same process,
    and the parameter is the supported way to ask "what would this list do?".
    """
    mod = _load_checker()
    unit = _multi_root_unit()
    assert len(_known_roots_lines(unit)) == 1, "fixture should start with one line"

    single_root_line = f"Environment={_PRESERVED_NAME}={_FM_REPO_ROOT}"
    fixed = mod.fix_unit_text(
        unit,
        required=(*mod.REQUIRED_SERVICE_DIRECTIVES, single_root_line),
    )

    lines = _known_roots_lines(fixed)
    assert len(lines) == 2, (
        f"expected the hypothetical required line to be APPENDED alongside the "
        f"host's, found {len(lines)}: {lines}"
    )
    assert lines[0].strip() == f"Environment={_PRESERVED_NAME}={_MULTI_ROOTS}", lines
    assert lines[-1].strip() == single_root_line, (
        "the appended single-root line is not LAST, which is the only reason "
        f"this is a silent clobber under systemd's last-wins: {lines}"
    )

    # And the checker would then report the clobbered unit as being at parity,
    # which is what makes the loss invisible rather than merely bad.
    assert mod.find_drift(fixed, (*mod.REQUIRED_SERVICE_DIRECTIVES, single_root_line)) == []


# ---------------------------------------------------------------------------
# setup-host.sh SECTION 4 renders through the renderer, not a redirect (step-9)
# ---------------------------------------------------------------------------
#
# ACCEPTANCE 1 and 2 at the INSTALLER layer, which is where the defect actually
# lives. Section 4 used to install this unit with
#
#     sed -e "s|__REPO_ROOT__|$REPO_ROOT|g" ... > "$UNIT_DIR/fused-memory.service"
#
# and the template renders DASHBOARD_KNOWN_PROJECT_ROOTS to a SINGLE root, so
# every re-run of the sanctioned install path collapsed this host's registered
# reconciliation scope — silently, because the post-install parity gate checks
# only host-invariant safety directives and cannot see this variable's value.
#
# These tests live in THIS module rather than in test_render_dashboard_unit.py
# because running a setup-host.sh slice needs subprocess and a stubbed PATH,
# while that module's docstring pins "ALL FIXTURES ARE tmp_path OR IN-MEMORY
# STRINGS" and it contains zero subprocess calls. Same split task 4793 used for
# the dashboard's section-8 tests.
#
# Nothing here touches ~/.config/systemd/user or real systemd: REPO_ROOT and
# UNIT_DIR are tmp_path trees and `systemctl` is a PATH stub that exits 0 while
# RECORDING its argv — which is how "was fused-memory enabled?" is OBSERVED
# rather than assumed.

# Anchored on the block's hoisted `_fm_render_script=` assignment — CODE, and
# unique to this site — for the same reason _GATE_START is: a comment anchor
# turns a reworded comment into a red CI run for no behavioural change. Mirrors
# section 8's `_dash_render_script=`.
_RENDER_START = "_fm_render_script="
_RENDER_END = "\nfi\n"
# The slice must run THROUGH the systemctl block, whose closing `fi` is not the
# first one after the start (the render if/elif/else chain closes first). This
# third anchor moves the end search past it — the same construct section 8's
# slice uses.
_RENDER_END_AFTER = 'if [ "$_fm_rendered"'

_FM_SERVICE = "fused-memory.service"

# A renderer that RAN and refused. Tagged the way the real one tags, so the
# assertion that the section reported the failure is not satisfied by the stub
# merely existing.
_FAILING_FM_RENDERER = (
    "import sys\n"
    "sys.stderr.write('[fused_memory_unit_render] FAILED: cannot read template\\n')\n"
    "sys.exit(1)\n"
)


def _section_4_repo(tmp_path: pathlib.Path, *, with_renderer: bool = True) -> pathlib.Path:
    """A tmp repo holding the template and (optionally) the REAL renderer.

    The renderer is copied from the repo rather than stubbed, so the slice
    drives the real preservation logic and only the TREE is fake. It imports
    systemd_unit_parity, so that sibling is copied too — the renderer's own
    docstring explains why that dependency went DOWN into a shared module
    instead of sideways into a checker.
    """
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True, exist_ok=True)
    (repo / "scripts" / "fused-memory.service.template").write_text(
        TEMPLATE_PATH.read_text(encoding="utf-8"), encoding="utf-8"
    )
    if with_renderer:
        for name in ("render_dashboard_unit.py", "systemd_unit_parity.py"):
            (repo / "scripts" / name).write_text(
                (REPO_ROOT / "scripts" / name).read_text(encoding="utf-8"),
                encoding="utf-8",
            )
    return repo


def _run_section_4(
    tmp_path: pathlib.Path, repo: pathlib.Path, unit_dir: pathlib.Path
) -> subprocess.CompletedProcess:
    """Run the section-4 slice. UV_PATH is assigned upstream in the real script."""
    return run_section(
        tmp_path,
        slice_section(_RENDER_START, _RENDER_END, end_after=_RENDER_END_AFTER),
        repo_root=repo,
        unit_dir=unit_dir,
        env_extra={"UV_PATH": "/usr/bin/uv"},
    )


def _fm_calls(tmp_path: pathlib.Path) -> list[list[str]]:
    """Every systemctl call naming the fused-memory unit."""
    return [
        argv
        for argv in systemctl_calls(tmp_path)
        if any(tok.startswith("fused-memory") for tok in argv)
    ]


def test_section_4_never_redirects_into_the_unit_file():
    """A redirect ANYWHERE in the block truncates the destination before python reads it.

    `python3 render.py ... > "$UNIT_DIR/fused-memory.service"` is the same defect
    one level up from the `sed >` it replaced: bash truncates the destination
    before python ever opens it, so the installed value is gone before it can be
    read and the tool preserves nothing while reporting success. The renderer
    OWNS the destination via --output; the slice must contain no redirect into
    it at all.
    """
    section = slice_section(_RENDER_START, _RENDER_END, end_after=_RENDER_END_AFTER)

    assert f'> "$UNIT_DIR/{_FM_SERVICE}"' not in section, section
    assert "--output" in section, (
        "section 4 does not hand the renderer --output, so it is not the "
        f"read-then-write path this task installs:\n{section}"
    )


def test_section_4_preserves_a_multi_root_installed_unit(tmp_path: pathlib.Path):
    """(a) ACCEPTANCE 1 at the installer layer: the reconciliation scope survives."""
    repo = _section_4_repo(tmp_path)
    unit_dir = _gate_unit_dir(tmp_path, content=_multi_root_unit("/old/root"))

    result = _run_section_4(tmp_path, repo, unit_dir)

    assert result.returncode == 0, result.stderr
    text = (unit_dir / _FM_SERVICE).read_text(encoding="utf-8")
    assert _MULTI_ROOTS in text, (
        f"the re-render collapsed this host's project roots:\n{text}"
    )
    for name in ("PROJECT_ROOT", "CONFIG_PATH", "TASKMASTER_DIR"):
        prefix = f"Environment={name}="
        line = next(
            line for line in text.splitlines() if line.strip().startswith(prefix)
        )
        assert str(repo) in line, (
            f"{line!r} was preserved from the OLD checkout instead of being "
            f"re-derived at {repo}"
        )
    assert "__REPO_ROOT__" not in text and "__UV_PATH__" not in text, text


def test_section_4_greenfield_installs_and_enables(tmp_path: pathlib.Path):
    """(b) ACCEPTANCE 2: a host with no installed unit gets the committed default."""
    repo = _section_4_repo(tmp_path)
    unit_dir = _gate_unit_dir(tmp_path)

    result = _run_section_4(tmp_path, repo, unit_dir)

    assert result.returncode == 0, result.stderr
    installed = unit_dir / _FM_SERVICE
    assert installed.is_file(), f"{_FM_SERVICE} was not rendered into {unit_dir}"
    text = installed.read_text(encoding="utf-8")
    assert f"Environment={_PRESERVED_NAME}={repo}" in text, text
    assert "__REPO_ROOT__" not in text and "__UV_PATH__" not in text, text
    assert "fused-memory" in enabled_units(tmp_path), enabled_units(tmp_path)


def test_section_4_missing_renderer_leaves_the_unit_alone(tmp_path: pathlib.Path):
    """(c) NO sed FALLBACK, and no systemctl on a unit that was not written.

    Rendering "the old way" when the renderer is missing looks like graceful
    degradation and is the opposite: it reinstates the exact truncating clobber
    this task removes, on the one path where nothing is left watching for it.
    Leaving the unit ALONE is the recoverable direction — stale-but-intact is
    fixable on the next run; de-registered-from-reconciliation is not noticed
    until projects start failing with UnknownProjectError.

    The systemctl assertion is the other half. `fail` in setup-host.sh is a
    printf, not an exit, so without the `_fm_rendered` flag the section would
    still reach `enable` (targeting a unit that does not exist on a greenfield
    host) and `restart` — bouncing the server that backs the orchestrators, the
    dashboard and this session's own MCP tooling on the strength of an install
    that did not happen.
    """
    repo = _section_4_repo(tmp_path, with_renderer=False)
    unit_dir = _gate_unit_dir(tmp_path, content=_multi_root_unit("/old/root"))
    before = (unit_dir / _FM_SERVICE).read_bytes()

    result = _run_section_4(tmp_path, repo, unit_dir)

    assert (unit_dir / _FM_SERVICE).read_bytes() == before, (
        "the missing-renderer path modified the installed unit"
    )
    assert "FAIL" in result.stdout + result.stderr, result.stdout + result.stderr
    assert _fm_calls(tmp_path) == [], _fm_calls(tmp_path)


def test_section_4_render_failure_leaves_the_unit_alone(tmp_path: pathlib.Path):
    """(d) A renderer that RAN and refused must not fall through silently green.

    The `elif ...; then ok` construct's blind spot: with no else branch a failing
    render falls out of the chain with status 0 and the operator is told nothing
    about the unit that did not get written.
    """
    repo = _section_4_repo(tmp_path)
    (repo / "scripts" / "render_dashboard_unit.py").write_text(
        _FAILING_FM_RENDERER, encoding="utf-8"
    )
    unit_dir = _gate_unit_dir(tmp_path, content=_multi_root_unit("/old/root"))
    before = (unit_dir / _FM_SERVICE).read_bytes()

    result = _run_section_4(tmp_path, repo, unit_dir)

    assert (unit_dir / _FM_SERVICE).read_bytes() == before, (
        "a refused render modified the installed unit"
    )
    assert "FAIL" in result.stdout + result.stderr, result.stdout + result.stderr
    assert _fm_calls(tmp_path) == [], _fm_calls(tmp_path)
