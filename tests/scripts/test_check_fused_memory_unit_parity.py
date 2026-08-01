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
    """find_drift flags all four restart-relevant directives when absent.

    RED until REQUIRED_SERVICE_DIRECTIVES grows to include Restart=on-failure,
    RestartSec=5, TimeoutStartSec=300, and TimeoutStopSec=90 — today it only
    lists MEM0_TELEMETRY + WatchdogSec, so find_drift returns [] here.
    """
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

    RED until REQUIRED_SERVICE_DIRECTIVES includes RestartSteps=4 — today the
    checker does not know the directive exists, so find_drift reports it
    nowhere.
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
