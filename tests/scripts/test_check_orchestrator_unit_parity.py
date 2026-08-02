"""Tests for scripts/check_orchestrator_unit_parity.py.

All drift-logic tests run against inline fixture strings and tmp_path
directories — NEVER the host's real ~/.config/systemd/user/ — mirroring the
rule tests/scripts/test_check_fused_memory_unit_parity.py and
tests/scripts/test_check_dashboard_unit_parity.py each state in their own
docstrings.

That rule is load-bearing here for a specific, measured reason rather than
mere portability. As measured on 2026-08-02, five of the seven registered
units (the orchestrator-*.service ones) diverge from their committed copies on
this host: every installed copy lacks ``RestartSteps=4``, so its
``RestartMaxDelaySec=60`` is silently discarded by systemd; four still name
legacy config filenames in ExecStart; reify's lacks RequiresMountsFor. Fixing
them is owned by a named follow-up task (see the checker's module docstring) —
so a test asserting parity against the live host would be red on landing, and
one asserting drift would flip red the moment that task lands. Either encodes
host state rather than checker behaviour.

The only real-tree reads are REPO-side: the committed scripts/*.service and
*.timer files, and scripts/setup-host.sh, used by the registry staleness guard.

Module loading: scripts/ is not a package, so the checker is loaded via
importlib.util.spec_from_file_location, mirroring
tests/scripts/test_check_fused_memory_unit_parity.py::_load_checker. The
shared parser module is imported by NAME (``import systemd_unit_parity``)
rather than by path, because that is the import the checker itself performs
and it is what the identity guard below needs to be exercising.
"""

import importlib.util
import pathlib
import re
import types

REPO_ROOT = pathlib.Path(__file__).parents[2]
CHECKER_PATH = REPO_ROOT / "scripts" / "check_orchestrator_unit_parity.py"
DASHBOARD_CHECKER_PATH = REPO_ROOT / "scripts" / "check_dashboard_unit_parity.py"
SETUP_HOST_PATH = REPO_ROOT / "scripts" / "setup-host.sh"


def _load_module(name: str, path: pathlib.Path) -> types.ModuleType:
    """Load a scripts/ module by file path (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None, f"Could not build spec from {path}"
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_checker() -> types.ModuleType:
    """Load scripts/check_orchestrator_unit_parity.py by file path."""
    return _load_module("check_orchestrator_unit_parity", CHECKER_PATH)


def _load_dashboard_checker() -> types.ModuleType:
    """Load scripts/check_dashboard_unit_parity.py by file path."""
    return _load_module("check_dashboard_unit_parity", DASHBOARD_CHECKER_PATH)


# ---------------------------------------------------------------------------
# Shared parser extraction  (step-1 / step-2)
# ---------------------------------------------------------------------------

# Deliberately exercises every parsing rule at once: a pre-header directive,
# both comment spellings, a blank line, a repeated key, the same key in two
# different sections, a value containing '=', and a backslash continuation
# spanning three physical lines.
_SAMPLE_UNIT = """\
Type=simple
[Unit]
Description=Test Service
# hash comment
; semicolon comment

After=network.target

[Service]
Environment=A=1
Environment=B=2
ExecStart=/usr/bin/thing \\
    --flag-one \\
    --flag-two
Restart=always
"""


def test_shared_parser_module_importable_and_exposes_the_parser():
    """``import systemd_unit_parity`` resolves and exposes both parser functions.

    Import by NAME, not by path: this is the exact import
    check_orchestrator_unit_parity.py performs, and it resolves in both
    contexts the checker runs in — at CLI runtime python puts the script's own
    directory (scripts/) at sys.path[0], and under pytest
    tests/scripts/conftest.py explicitly inserts scripts/ onto sys.path
    (pyproject's ``--import-mode=importlib`` deliberately does NOT).
    """
    import systemd_unit_parity

    assert callable(systemd_unit_parity.parse_unit_directives)
    assert callable(systemd_unit_parity._join_continuations)


def test_shared_parser_parses_sections_keys_and_values():
    """The lifted parser yields ``{section: {key: [values]}}`` with systemd's rules.

    Pins every classification rule the checker depends on: comments (both
    spellings) and blank lines dropped, backslash continuations joined into
    one logical line, repeated keys accumulated into the values LIST (systemd
    applies every occurrence), lines before the first ``[Section]`` header
    dropped rather than attributed, and the split taken on the FIRST ``=``
    only so ``Environment=A=1`` yields value ``A=1``.
    """
    import systemd_unit_parity

    parsed = systemd_unit_parity.parse_unit_directives(_SAMPLE_UNIT)

    assert set(parsed) == {"Unit", "Service"}

    # Pre-header `Type=simple` is dropped, not attributed to [Unit].
    assert "Type" not in parsed["Unit"]
    assert "Type" not in parsed["Service"]

    # Comments (both spellings) and blanks contribute nothing.
    assert parsed["Unit"] == {
        "Description": ["Test Service"],
        "After": ["network.target"],
    }

    # Repeated key accumulates in order; split on the FIRST '=' only.
    assert parsed["Service"]["Environment"] == ["A=1", "B=2"]

    # Continuations joined into ONE logical line, so the flags are reachable
    # as part of ExecStart rather than stranded on lines of their own.
    assert parsed["Service"]["ExecStart"] == [
        "/usr/bin/thing --flag-one --flag-two"
    ]
    assert parsed["Service"]["Restart"] == ["always"]


def test_dashboard_checker_consumes_the_lifted_parser():
    """IDENTITY guard: the dashboard checker uses the lifted parser, not a copy.

    This is the assertion that makes the extraction worth doing. Re-exporting
    the parser from check_dashboard_unit_parity.py keeps that module's surface
    (and its 1677-line test suite) intact, but re-export is only meaningful if
    it is the SAME function object — if someone later pastes a second copy of
    the parser back into the dashboard checker, every other test in both
    suites would still pass while the two implementations quietly drifted
    apart, which is precisely the failure mode these parity checkers exist to
    catch. Asserting object identity is the only check that fires on that.
    """
    import systemd_unit_parity

    dashboard = _load_dashboard_checker()

    assert dashboard.parse_unit_directives is systemd_unit_parity.parse_unit_directives
    assert dashboard._join_continuations is systemd_unit_parity._join_continuations


# ---------------------------------------------------------------------------
# The unit registry, and its staleness guard  (step-3 / step-4)
# ---------------------------------------------------------------------------

# The seven units scripts/setup-host.sh installs by copying VERBATIM. Stated
# as a literal here so a bug in the shell-parsing helper below cannot make the
# equality assertion vacuously true against itself.
_EXPECTED_UNITS = {
    "orchestrator-watchdog.service",
    "orchestrator-watchdog.timer",
    "orchestrator-dark-factory.service",
    "orchestrator-reify.service",
    "orchestrator-autopilot-video.service",
    "orchestrator-my-solar-challenge.service",
    "orchestrator-solar-challenge-platform.service",
}

# Matches setup-host.sh's install lines, e.g.
#   cp "$REPO_ROOT/scripts/orchestrator-reify.service"   "$UNIT_DIR/"
# Anchored on BOTH endpoints (the scripts/ source and the $UNIT_DIR
# destination) so an unrelated `cp` elsewhere in the installer cannot be
# mistaken for a unit install.
_CP_UNIT_RE = re.compile(
    r'cp\s+"\$REPO_ROOT/scripts/(?P<unit>[^"]+\.(?:service|timer))"\s+"\$UNIT_DIR/?"'
)


def _units_installed_by_setup_host() -> set[str]:
    """Extract the unit names setup-host.sh `cp`s verbatim into $UNIT_DIR."""
    return set(
        m.group("unit")
        for m in _CP_UNIT_RE.finditer(SETUP_HOST_PATH.read_text(encoding="utf-8"))
    )


def test_registry_covers_the_seven_verbatim_copied_units():
    """UNITS registers exactly the seven units setup-host.sh copies verbatim."""
    checker = _load_checker()

    assert set(checker.UNITS) == _EXPECTED_UNITS, (
        "check_orchestrator_unit_parity.UNITS does not match the expected set.\n"
        f"  missing from UNITS: {sorted(_EXPECTED_UNITS - set(checker.UNITS))}\n"
        f"  unexpected in UNITS: {sorted(set(checker.UNITS) - _EXPECTED_UNITS)}"
    )


def test_every_registered_unit_has_a_committed_copy():
    """Every registry entry's committed repo path exists under scripts/.

    A registry entry naming a path that does not exist is not a harmless
    typo: the checker reports it as ``[vanished]`` on every run, which is
    indistinguishable from a real "the file I compare against is gone" and
    spends the gate's credibility on nothing.
    """
    checker = _load_checker()

    for name, relpath in checker.UNITS.items():
        assert (REPO_ROOT / relpath).is_file(), (
            f"UNITS registers {name} -> {relpath}, but "
            f"{REPO_ROOT / relpath} does not exist."
        )


def test_registry_matches_setup_host_cp_lines():
    """STALENESS GUARD: the registry equals setup-host.sh's own `cp` unit set.

    This is the load-bearing test of the three. A hand-curated registry has
    exactly the failure shape this whole checker exists to kill: add a unit to
    the installer's `cp` block and it becomes installed-but-UNCHECKED —
    silently, forever, while the gate still reports green. That is the same
    silent-drift defect one level up, reproduced inside the tool built to
    catch it.

    Asserting set equality in BOTH directions makes the registry
    self-maintaining: a unit added to the installer fails here until it is
    registered, and a unit dropped from the installer fails here until it is
    deregistered. The derivation deliberately lives in the TEST and not in the
    checker — the runtime registry stays an explicit literal, so a bug in the
    shell parsing below can never disarm the checker itself.
    """
    checker = _load_checker()
    installed_by_setup_host = _units_installed_by_setup_host()

    # Guard the guard: if the regex silently matched nothing, the equality
    # assertion below would still fail, but with a message pointing at the
    # registry rather than at the parsing.
    assert installed_by_setup_host, (
        f"Parsed ZERO `cp` unit lines out of {SETUP_HOST_PATH}. The installer's "
        "cp idiom has changed and _CP_UNIT_RE no longer matches it — fix the "
        "regex, do not weaken this test."
    )

    assert installed_by_setup_host == set(checker.UNITS), (
        "The unit registry and the installer disagree about which units are "
        "installed by verbatim copy.\n"
        f"  copied by {SETUP_HOST_PATH} but NOT registered in "
        f"scripts/check_orchestrator_unit_parity.py UNITS: "
        f"{sorted(installed_by_setup_host - set(checker.UNITS))}\n"
        f"  registered in UNITS but NOT copied by {SETUP_HOST_PATH}: "
        f"{sorted(set(checker.UNITS) - installed_by_setup_host)}\n"
        "A unit the installer copies but the registry omits is "
        "installed-but-unchecked: it can drift forever while this gate "
        "reports green."
    )


# ---------------------------------------------------------------------------
# compare_unit — full symmetric directive equality  (step-5 / step-6)
# ---------------------------------------------------------------------------

_BASE_TIMER = """\
[Unit]
Description=Orchestrator escalation-MCP health probe (every 60s)

[Timer]
OnBootSec=30
OnUnitActiveSec=60

[Install]
WantedBy=timers.target
"""


def _drift_keys(drifts) -> list[tuple[str, str]]:
    """Reduce a Drift list to comparable (section, key) pairs."""
    return [(d.section, d.key) for d in drifts]


def test_identical_texts_report_no_drift():
    """Byte-identical copies are parity, full stop."""
    checker = _load_checker()

    assert checker.compare_unit("u.timer", _BASE_TIMER, _BASE_TIMER) == []


def test_differing_value_is_reported_with_both_sides():
    """A directive present on both sides with different VALUES is drift.

    Value comparison is what catches present-but-WRONG. A presence-only check
    would wave through an installed TimeoutStartSec=300 against a committed
    1800 — the bound would look enforced and be four minutes short of a
    half-hour.
    """
    checker = _load_checker()

    repo = "[Service]\nTimeoutStartSec=1800\n"
    installed = "[Service]\nTimeoutStartSec=300\n"

    drifts = checker.compare_unit("u.service", repo, installed)

    assert len(drifts) == 1, drifts
    (drift,) = drifts
    assert drift.unit == "u.service"
    assert drift.section == "Service"
    assert drift.key == "TimeoutStartSec"
    assert drift.repo_value == "1800"
    assert drift.installed_value == "300"
    assert "differ" in drift.reason


def test_directive_only_in_repo_copy_is_drift():
    """A committed directive missing from the installed copy is drift.

    This is the measured task-3392 case: the committed watchdog service gained
    TimeoutStartSec=1800 and the installed copy never did, so a whole-tick
    bound on the fleet probe was inert on this host for the entire time it
    appeared to be in force.
    """
    checker = _load_checker()

    repo = "[Service]\nType=oneshot\nTimeoutStartSec=1800\n"
    installed = "[Service]\nType=oneshot\n"

    drifts = checker.compare_unit("u.service", repo, installed)

    assert _drift_keys(drifts) == [("Service", "TimeoutStartSec")]
    assert drifts[0].repo_value == "1800"
    assert drifts[0].installed_value == checker._ABSENT
    assert "absent from the installed copy" in drifts[0].reason


def test_directive_only_in_installed_copy_is_drift():
    """An INSTALLED-only directive is drift — the case curation cannot see.

    This is the whole reason the rule here is full symmetric equality rather
    than the curated (section, key) registry check_dashboard_unit_parity.py
    uses. A curated key list can only ever contain keys someone thought to
    register, so it is structurally blind to a directive that exists ONLY on
    the installed side.

    The fixture is the real measured case: the installed
    orchestrator-watchdog.timer carries AccuracySec=5s and no committed copy
    ever has. systemd's default AccuracySec is 1min, so propagating the
    committed timer without noticing this would have widened the 60s probe's
    elapse window from [60s, 65s] to [60s, 120s] — a supervision regression
    shipped as a "parity fix". A registry-based checker would have reported
    green on it.
    """
    checker = _load_checker()

    repo = "[Timer]\nOnUnitActiveSec=60\n"
    installed = "[Timer]\nOnUnitActiveSec=60\nAccuracySec=5s\n"

    drifts = checker.compare_unit("u.timer", repo, installed)

    assert _drift_keys(drifts) == [("Timer", "AccuracySec")]
    assert drifts[0].repo_value == checker._ABSENT
    assert drifts[0].installed_value == "5s"
    assert "absent from the repo copy" in drifts[0].reason


def test_comment_only_divergence_is_not_drift():
    """Comments and blank lines must NOT fire the gate.

    Task 3392 added ~35 comment lines to the committed watchdog service. If a
    comment reflow reported as drift, the gate would fire on edits that change
    nothing systemd reads — and a gate nobody believes is worse than no gate.
    This is the credibility cost that curation exists to avoid, paid by the
    parser instead, which is what lets the comparison itself stay unbounded.
    """
    checker = _load_checker()

    repo = (
        "# Rewritten by task 3392 to bound the whole tick.\n"
        "# The probe may block on a slow socket; without this the unit\n"
        "# could hang indefinitely.\n"
        "[Service]\n"
        "Type=oneshot\n"
        "\n"
        "TimeoutStartSec=1800\n"
    )
    installed = (
        "[Service]\n"
        "; a semicolon-style comment, the other spelling systemd accepts\n"
        "Type=oneshot\n"
        "TimeoutStartSec=1800\n"
    )

    assert checker.compare_unit("u.service", repo, installed) == []


def test_changed_occurrence_count_of_a_repeated_directive_is_drift():
    """A repeated directive losing an occurrence is drift.

    systemd APPLIES every occurrence of Environment=, so comparing only the
    first value would wave through a dropped variable. The comparison is over
    the whole values LIST for that reason.
    """
    checker = _load_checker()

    repo = "[Service]\nEnvironment=A=1\nEnvironment=B=2\n"
    installed = "[Service]\nEnvironment=A=1\n"

    drifts = checker.compare_unit("u.service", repo, installed)

    assert _drift_keys(drifts) == [("Service", "Environment")]
    # Both sides declare the key, so neither renders as <absent> — the
    # difference is in the list, and the report must show both lists.
    assert "B=2" in drifts[0].repo_value
    assert drifts[0].installed_value != checker._ABSENT


def test_same_key_in_different_sections_is_not_conflated():
    """[Unit] Description and [Timer] Description are different directives.

    A parser keyed on bare directive names would fold these together and
    could report parity on a pair of units whose sections disagree.
    """
    checker = _load_checker()

    repo = "[Unit]\nDescription=probe\n\n[Timer]\nDescription=timer-desc\n"
    installed = "[Unit]\nDescription=probe\n\n[Timer]\nDescription=CHANGED\n"

    drifts = checker.compare_unit("u.timer", repo, installed)

    assert _drift_keys(drifts) == [("Timer", "Description")]
    assert drifts[0].repo_value == "timer-desc"


def test_section_present_on_one_side_only_surfaces_as_drifts():
    """A whole missing section reports its keys as drift, and does not crash.

    Walking the UNION of sections (not the intersection) is what makes this
    work; an intersection walk would silently ignore an entire dropped
    [Install] section, i.e. a unit that is no longer enable-able.
    """
    checker = _load_checker()

    repo = "[Timer]\nOnBootSec=30\n\n[Install]\nWantedBy=timers.target\n"
    installed = "[Timer]\nOnBootSec=30\n"

    drifts = checker.compare_unit("u.timer", repo, installed)

    assert _drift_keys(drifts) == [("Install", "WantedBy")]
    assert drifts[0].repo_value == "timers.target"
    assert drifts[0].installed_value == checker._ABSENT


def test_drift_ordering_is_deterministic():
    """Drifts come back sorted by (section, key) so reports are diffable.

    An operator comparing two runs should see a stable report; dict-insertion
    ordering would make the same drift set render differently depending on
    the order directives happened to appear in the file.
    """
    checker = _load_checker()

    repo = "[Timer]\nZeta=1\nAlpha=1\n\n[Install]\nWantedBy=timers.target\n"
    installed = "[Timer]\nZeta=2\nAlpha=2\n\n[Install]\nWantedBy=other.target\n"

    drifts = checker.compare_unit("u.timer", repo, installed)

    assert _drift_keys(drifts) == [
        ("Install", "WantedBy"),
        ("Timer", "Alpha"),
        ("Timer", "Zeta"),
    ]
