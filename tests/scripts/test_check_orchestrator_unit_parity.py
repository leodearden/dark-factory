"""Tests for scripts/check_orchestrator_unit_parity.py.

All drift-logic tests run against inline fixture strings and tmp_path
directories — NEVER the host's real ~/.config/systemd/user/ — mirroring the
rule tests/scripts/test_check_fused_memory_unit_parity.py and
tests/scripts/test_check_dashboard_unit_parity.py each state in their own
docstrings.

That rule is load-bearing here for a specific, measured reason rather than
mere portability. As measured on 2026-08-02, five of the (then seven, now
nine — see the REBASE NOTE below) registered units (the orchestrator-*.service
ones) diverge from their committed copies on this host, and — importantly —
NOT all in the same direction: every installed copy lacks ``RestartSteps=4``
(so its ``RestartMaxDelaySec=60`` is silently discarded by systemd, and the
REPO copy is the correct one), while four have an ExecStart ``--config`` path
where the INSTALLED copy is the correct one (it names the canonical
``dark-factory-orchestrator.yaml``; the committed copy still names a legacy
filename that, for reify and autopilot-video, does not exist at all). Fixing
them is owned by a named follow-up task (see the checker's module docstring)
— so a test asserting parity against the live host would be red on landing,
and one asserting drift would flip red the moment that task lands. Either
encodes host state rather than checker behaviour.

REBASE NOTE (2026-08-06): this branch was rebased onto a main that had, in
the interim, wired orchestrator-know-live.service and
orchestrator-pump-web-ui.service into setup-host.sh (task 3641) and repointed
four of the five --config paths above at the canonical filename (task 3512,
commit 4fcd43eec0, landed 2026-08-04). The two new units are registered below
to keep the staleness guard satisfied; see
scripts/check_orchestrator_unit_parity.py's own REBASE NOTE for both — neither
the two new units nor the 2026-08-02 numbers above were re-measured against
the live host as part of this rebase.

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
import os
import pathlib
import re
import subprocess
import sys
import types

import pytest

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

    The ``# pyright: ignore[reportMissingImports]`` on the import is a
    STATIC-ANALYSIS artifact, not a papering-over: pyright never executes
    conftest.py, so it cannot see that sys.path insertion, and the root
    pyproject's ``[tool.pyright] extraPaths`` deliberately omits ``scripts/``.
    Do NOT "fix" this by adding scripts/ to extraPaths — scripts/ is knowingly
    not yet pyright-clean, which is exactly why scripts/orchestrator.yaml
    declines to declare a ``type_check_command``; widening extraPaths would
    pull that whole tree into resolution for every consumer. The suppression
    is the convention already in force at three sibling sites here
    (test_migrate_metadata_modules_to_files.py, test_repair_wiped_metadata_files.py).
    The runtime import is the assertion; these tests passing IS its proof.
    """
    import systemd_unit_parity  # pyright: ignore[reportMissingImports]

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
    import systemd_unit_parity  # pyright: ignore[reportMissingImports]

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
    import systemd_unit_parity  # pyright: ignore[reportMissingImports]

    dashboard = _load_dashboard_checker()

    assert dashboard.parse_unit_directives is systemd_unit_parity.parse_unit_directives
    assert dashboard._join_continuations is systemd_unit_parity._join_continuations


# ---------------------------------------------------------------------------
# The unit registry, and its staleness guard  (step-3 / step-4)
# ---------------------------------------------------------------------------

# The nine units scripts/setup-host.sh installs by copying VERBATIM. Stated
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
    "orchestrator-know-live.service",
    "orchestrator-pump-web-ui.service",
}

# setup-host.sh no longer carries one literal `cp` line per unit; it declares
# the set once and loops. This matches that declaration:
#
#   _orch_units=(
#     orchestrator-reify.service        # reify orchestrator, escalation 8100
#     ...
#   )
#
# Anchored on a column-0 `_orch_units=(` ... `)` pair so no other parenthesised
# construct in the installer can be mistaken for it.
_ORCH_UNITS_ARRAY_RE = re.compile(
    r"^_orch_units=\(\n(?P<body>.*?)^\)$", re.MULTILINE | re.DOTALL
)

# The four statements that make the array MEAN something. Replacing the old
# both-endpoints-anchored `cp` regex with this chain preserves everything that
# regex bought and adds the per-unit link: a unit named in the array but never
# judged, never added to the cleared set, or copied somewhere other than
# $UNIT_DIR, is as uninstalled as a unit nobody listed at all.
#
#   declaration  ->  per-unit decision  ->  cleared set  ->  copy
_DECISION_LOOP_HEADER = 'for _unit in "${_orch_units[@]}"; do'
_INSTALL_LIST_APPEND = '_orch_install_units+=("$_unit")'
_INSTALL_LOOP_HEADER = 'for _unit in "${_orch_install_units[@]}"; do'
_INSTALL_LOOP_CP = 'if cp "$REPO_ROOT/scripts/$_unit" "$UNIT_DIR/"; then'


def _shell_statements(script_text: str) -> list[str]:
    """Non-comment, non-blank statements, whitespace-stripped.

    Mirrors tests/scripts/test_orchestrator_service_files.py's helper of the
    same name, and for the same reason: a unit or a command merely NAMED in
    the section's header prose must not satisfy an assertion that the
    installer actually runs it.
    """
    return [
        stripped
        for line in script_text.splitlines()
        if (stripped := line.strip()) and not stripped.startswith("#")
    ]


def _units_installed_by_setup_host() -> set[str]:
    """Extract the unit names setup-host.sh installs, from its `_orch_units` array."""
    match = _ORCH_UNITS_ARRAY_RE.search(SETUP_HOST_PATH.read_text(encoding="utf-8"))
    if match is None:
        return set()
    units = set()
    for line in match.group("body").splitlines():
        # Each entry may carry a trailing `# rationale` comment — that prose is
        # the per-unit justification the old enable block held, and it must not
        # be parsed as part of the unit name.
        if entry := line.split("#", 1)[0].strip():
            units.add(entry)
    return units


def test_registry_covers_the_nine_verbatim_copied_units():
    """UNITS registers exactly the nine units setup-host.sh copies verbatim."""
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
        f"Parsed ZERO units out of {SETUP_HOST_PATH}'s `_orch_units` array. The "
        "installer's declaration idiom has changed and _ORCH_UNITS_ARRAY_RE no "
        "longer matches it — fix the regex, do not weaken this test."
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


def test_the_install_loop_actually_consumes_the_declared_unit_array():
    """The `_orch_units` array is what DRIVES the copy, not decoration.

    The staleness guard above derives the registry from that array, so the
    array has to be the thing the installer acts on — otherwise a unit could be
    declared, registered, checked, and never installed, and every test here
    would still be green.

    Four statements are asserted, and together they reconstruct everything the
    old both-endpoints-anchored `cp` regex bought before section 5 was made
    declarative, plus the per-unit link the gate now depends on:

        declaration -> per-unit decision -> cleared set -> copy into $UNIT_DIR

    Break any one link and the array stops meaning "these units get installed":
    a decision loop over some other list judges the wrong units, an install
    loop over `_orch_units` rather than the cleared set installs the skipped
    ones, and a copy to a staging path leaves the unit as uninstalled as no
    copy at all.

    Read through the comment-stripped view so a `cp` shown as an EXAMPLE in the
    section's header prose cannot satisfy it.
    """
    statements = _shell_statements(_installer_section())

    for expected, why in (
        (
            _DECISION_LOOP_HEADER,
            "nothing walks the declared `_orch_units` array, so the array is "
            "documentation — and the registry staleness guard above is derived "
            "from documentation",
        ),
        (
            _INSTALL_LIST_APPEND,
            "the per-unit decision never adds a cleared unit to "
            "`_orch_install_units`, so the decision has no effect",
        ),
        (
            _INSTALL_LOOP_HEADER,
            "the install loop does not iterate the CLEARED set, so a unit the "
            "gate declined would be copied anyway",
        ),
        (
            _INSTALL_LOOP_CP,
            "the install loop does not copy into $UNIT_DIR; the destination is "
            "asserted, not just the source",
        ),
    ):
        assert expected in statements, (
            f"{SETUP_HOST_PATH}: {why}.\nExpected the statement:\n"
            f"    {expected}\nSection statements: {statements}"
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


def test_repeated_values_render_with_unambiguous_occurrence_boundaries():
    """A repeated directive's occurrences must be separable in the report.

    A bare ", "-join is ambiguous the moment a value contains a comma:
    ``Environment=OPTS=a,b`` next to ``Environment=X=1`` renders as
    ``[OPTS=a,b, X=1]``, where an operator cannot tell where one occurrence
    ends. Comparison runs on the raw lists so this can never be a false
    negative — but the report is the deliverable of a checker whose value is
    being believed, and a diff you have to guess at is not actionable.
    """
    checker = _load_checker()

    repo = "[Service]\nEnvironment=OPTS=a,b\nEnvironment=X=1\n"
    installed = "[Service]\nEnvironment=OPTS=a\n"

    rendered = checker.compare_unit("u.service", repo, installed)[0].repo_value

    assert rendered == "['OPTS=a,b', 'X=1']", (
        "Repeated values render ambiguously — an operator cannot tell where "
        f"one occurrence ends: {rendered}"
    )


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


# ---------------------------------------------------------------------------
# Drop-in override detection  (step-7 / step-8)
# ---------------------------------------------------------------------------

# A real registered unit name, so --unit (which uses argparse `choices`)
# accepts it in the CLI-level tests below.
_REAL_UNIT = "orchestrator-watchdog.timer"
_REAL_UNIT_RELPATH = "scripts/orchestrator-watchdog.timer"


def _make_trees(
    tmp_path: pathlib.Path,
    *,
    repo_text: str | None = _BASE_TIMER,
    installed_text: str | None = _BASE_TIMER,
    unit: str = _REAL_UNIT,
    relpath: str = _REAL_UNIT_RELPATH,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Build a (repo_root, installed_dir) pair under tmp_path.

    Passing None for either text omits that side's file, which is how the
    ``[vanished]`` (no committed copy) and ``[skip]`` (no installed copy)
    branches are exercised. NEVER touches ~/.config/systemd/user.
    """
    repo_root = tmp_path / "repo"
    installed_dir = tmp_path / "installed"
    (repo_root / "scripts").mkdir(parents=True, exist_ok=True)
    installed_dir.mkdir(parents=True, exist_ok=True)

    if repo_text is not None:
        (repo_root / relpath).write_text(repo_text, encoding="utf-8")
    if installed_text is not None:
        (installed_dir / unit).write_text(installed_text, encoding="utf-8")

    return repo_root, installed_dir


def test_find_dropins_returns_empty_when_no_dropin_dir(tmp_path: pathlib.Path):
    """No `<unit>.d/` directory at all => no overrides."""
    checker = _load_checker()
    _, installed_dir = _make_trees(tmp_path)

    assert checker.find_dropins(installed_dir, _REAL_UNIT) == []


def test_find_dropins_finds_an_override_conf(tmp_path: pathlib.Path):
    """A `<unit>.d/override.conf` — what `systemctl --user edit` writes."""
    checker = _load_checker()
    _, installed_dir = _make_trees(tmp_path)

    dropin_dir = installed_dir / f"{_REAL_UNIT}.d"
    dropin_dir.mkdir()
    override = dropin_dir / "override.conf"
    override.write_text("[Timer]\nAccuracySec=1min\n", encoding="utf-8")

    assert checker.find_dropins(installed_dir, _REAL_UNIT) == [override]


def test_find_dropins_returns_all_conf_files_sorted(tmp_path: pathlib.Path):
    """Several drop-ins all count, in sorted order (systemd merges them all)."""
    checker = _load_checker()
    _, installed_dir = _make_trees(tmp_path)

    dropin_dir = installed_dir / f"{_REAL_UNIT}.d"
    dropin_dir.mkdir()
    for fname in ("20-second.conf", "10-first.conf"):
        (dropin_dir / fname).write_text("[Timer]\n", encoding="utf-8")

    found = checker.find_dropins(installed_dir, _REAL_UNIT)

    assert [p.name for p in found] == ["10-first.conf", "20-second.conf"]


def test_find_dropins_ignores_non_conf_files(tmp_path: pathlib.Path):
    """Non-`.conf` files are ignored, matching what systemd actually merges.

    Counting a stray `override.conf.bak` would report an override that has no
    effect — a false positive in a report whose whole value is being believed.
    """
    checker = _load_checker()
    _, installed_dir = _make_trees(tmp_path)

    dropin_dir = installed_dir / f"{_REAL_UNIT}.d"
    dropin_dir.mkdir()
    (dropin_dir / "override.conf.bak").write_text("[Timer]\n", encoding="utf-8")
    (dropin_dir / "notes.txt").write_text("hello\n", encoding="utf-8")

    assert checker.find_dropins(installed_dir, _REAL_UNIT) == []


def test_dropin_over_identical_units_does_not_report_parity(
    tmp_path: pathlib.Path,
):
    """BYTE-IDENTICAL unit files + a drop-in must NOT report parity.

    This is the behavioural claim that makes find_dropins worth having.
    `systemctl --user edit` never modifies the unit file; it writes
    `<unit>.d/override.conf`, which systemd merges OVER the unit at load
    time. So every compared directive can match character for character while
    the EFFECTIVE configuration differs — the precise claim this gate exists
    to make checkable.

    Not hypothetical on this host: `~/.config/systemd/user/orchestrator-reify.service.d/`
    exists, so the mechanism is already in live use here.

    The override shares exit 1 with drift because "I could not verify" belongs
    with "I found a difference", not with the benign "not installed here" that
    2 denotes — but it is WORDED apart ([override], not [drift]) so the
    operator is not sent hunting for a directive diff that does not exist.
    """
    checker = _load_checker()
    repo_root, installed_dir = _make_trees(tmp_path)

    dropin_dir = installed_dir / f"{_REAL_UNIT}.d"
    dropin_dir.mkdir()
    (dropin_dir / "override.conf").write_text(
        "[Timer]\nAccuracySec=1min\n", encoding="utf-8"
    )

    rc = checker.main(
        [
            "--installed-dir", str(installed_dir),
            "--repo-root", str(repo_root),
            "--unit", _REAL_UNIT,
        ]
    )

    assert rc == 1, "A drop-in override must not be reported as parity."


# ---------------------------------------------------------------------------
# CLI exit-code contract  (step-9 / step-10)
# ---------------------------------------------------------------------------

_SECOND_UNIT = "orchestrator-watchdog.service"
_SECOND_UNIT_RELPATH = "scripts/orchestrator-watchdog.service"

_BASE_SERVICE = """\
[Unit]
Description=Orchestrator escalation-MCP health probe

[Service]
Type=oneshot
TimeoutStartSec=1800
ExecStart=/usr/bin/python3 /home/leo/src/dark-factory/scripts/orchestrator-watchdog.py
"""

# The drifted timer: OnBootSec differs, so exactly one directive disagrees.
_DRIFTED_TIMER = _BASE_TIMER.replace("OnBootSec=30", "OnBootSec=60")


def _run_cli(
    repo_root: pathlib.Path,
    installed_dir: pathlib.Path,
    *units: str,
    extra_args: tuple[str, ...] = (),
) -> subprocess.CompletedProcess:
    """Invoke the checker as a real subprocess against tmp_path trees.

    Uses sys.executable rather than a bare `python3`: commit 5178360711 fixed
    exactly that defect in the sibling scanner CLI tests, where a bare python3
    resolved to a different interpreter than the one running the suite.

    ``extra_args`` appends flags after the two tree flags, which is how the
    machine-readable-verdict tests run the SAME invocation with and without
    ``--print-verdicts`` to prove the flag is a pure addition.
    """
    argv = [
        sys.executable,
        str(CHECKER_PATH),
        "--installed-dir", str(installed_dir),
        "--repo-root", str(repo_root),
    ]
    for unit in units:
        argv += ["--unit", unit]
    argv += list(extra_args)
    return subprocess.run(argv, capture_output=True, text=True)


def test_cli_matching_copies_exit_zero(tmp_path: pathlib.Path):
    """Matching copies => 0, reporting the number actually COMPARED."""
    checker = _load_checker()
    repo_root, installed_dir = _make_trees(tmp_path)

    result = _run_cli(repo_root, installed_dir, _REAL_UNIT)

    assert result.returncode == 0, (
        f"Expected 0 for matching copies; got {result.returncode}. "
        f"stdout: {result.stdout} stderr: {result.stderr}"
    )
    assert "[ok] parity" in result.stdout
    # The count is what was COMPARED, never what was selected: with 7 units
    # registered and 1 selected, claiming 7 would overstate the check.
    assert "1 unit(s)" in result.stdout

    assert checker.main(
        ["--installed-dir", str(installed_dir), "--repo-root", str(repo_root),
         "--unit", _REAL_UNIT]
    ) == 0


def test_cli_drifted_copy_exits_one_and_names_the_directive(
    tmp_path: pathlib.Path,
):
    """A drifted copy => 1, with the offending directive named on stdout.

    Asserting on the directive NAME, not just the exit code: an exit code
    tells an operator something is wrong, the report is what tells them what.
    """
    repo_root, installed_dir = _make_trees(
        tmp_path, installed_text=_DRIFTED_TIMER
    )

    result = _run_cli(repo_root, installed_dir, _REAL_UNIT)

    assert result.returncode == 1, (
        f"Expected 1 for a drifted copy; got {result.returncode}. "
        f"stdout: {result.stdout}"
    )
    assert "[drift]" in result.stdout
    assert "OnBootSec" in result.stdout
    assert "30" in result.stdout and "60" in result.stdout


def test_cli_installed_unit_absent_exits_two(tmp_path: pathlib.Path):
    """An absent INSTALLED unit => 2 and a [skip] line.

    2 is the benign "not installed on this host" that setup-host.sh treats as
    a skip rather than something to act on.
    """
    repo_root, installed_dir = _make_trees(tmp_path, installed_text=None)

    result = _run_cli(repo_root, installed_dir, _REAL_UNIT)

    assert result.returncode == 2, (
        f"Expected 2 for an absent installed unit; got {result.returncode}. "
        f"stdout: {result.stdout}"
    )
    assert "[skip]" in result.stdout


def test_cli_vanished_committed_unit_exits_one(tmp_path: pathlib.Path):
    """An absent COMMITTED unit => 1 and [vanished], worded apart from [drift].

    A missing source of truth is not a diff to propagate. Before the sibling
    checker distinguished these, a typo'd --repo-root compared nothing and
    still printed "parity — N unit(s) match", silently disarming the check.
    Telling an operator to hunt for a directive diff here would waste the trip,
    hence the separate word.
    """
    repo_root, installed_dir = _make_trees(tmp_path, repo_text=None)

    result = _run_cli(repo_root, installed_dir, _REAL_UNIT)

    assert result.returncode == 1, (
        f"Expected 1 for a vanished committed unit; got {result.returncode}. "
        f"stdout: {result.stdout}"
    )
    assert "[vanished]" in result.stdout
    assert "[drift]" not in result.stdout


def test_cli_undecodable_installed_unit_reports_unreadable_not_a_traceback(
    tmp_path: pathlib.Path,
):
    """A unit that EXISTS but cannot be decoded => 1 and [unreadable].

    is_file() establishes existence, not readability: a unit can be mode 000
    or not valid UTF-8. Unguarded, read_text raises and the run dies with a
    Python traceback and exit 1 — which setup-host.sh reports as "drift or
    unverifiable state, see the [orchestrator_unit_parity] report above",
    sending the operator to look for a directive diff in a report that is a
    stack trace. Worded apart for the same reason [vanished] and [override]
    are: it shares the exit code, not the remedy.
    """
    repo_root, installed_dir = _make_trees(tmp_path)
    (installed_dir / _REAL_UNIT).write_bytes(b"[Timer]\nOnBootSec=\xff\xfe30\n")

    result = _run_cli(repo_root, installed_dir, _REAL_UNIT)

    assert result.returncode == 1, (
        f"Expected 1 for an undecodable unit; got {result.returncode}. "
        f"stdout: {result.stdout} stderr: {result.stderr}"
    )
    assert "[unreadable]" in result.stdout, result.stdout
    assert "Traceback" not in result.stderr, result.stderr
    # Names the file that ACTUALLY failed, not the other side — an operator
    # sent to check permissions on the wrong copy learns nothing.
    assert str(installed_dir / _REAL_UNIT) in result.stdout, result.stdout
    assert "[drift]" not in result.stdout, result.stdout
    # A run that could not read a unit never counts it as compared.
    assert "[ok] parity" not in result.stdout, result.stdout


def test_cli_undecodable_repo_unit_names_the_repo_side(tmp_path: pathlib.Path):
    """The same guard on the committed side, reporting the committed path."""
    repo_root, installed_dir = _make_trees(tmp_path)
    (repo_root / _REAL_UNIT_RELPATH).write_bytes(b"[Timer]\nOnBootSec=\xff\xfe30\n")

    result = _run_cli(repo_root, installed_dir, _REAL_UNIT)

    assert result.returncode == 1, result.stdout
    assert "[unreadable]" in result.stdout, result.stdout
    assert str(repo_root / _REAL_UNIT_RELPATH) in result.stdout, result.stdout


def test_cli_drift_dominates_absence(tmp_path: pathlib.Path):
    """PRECEDENCE: one unit drifted + another absent => 1, not 2.

    With nine units a single run can hit both at once. Returning 2 would let
    an unrelated uninstalled unit MASK an actionable finding, because
    setup-host.sh treats 2 as a benign skip. The absent unit is still
    reported — dominated, not hidden.
    """
    repo_root, installed_dir = _make_trees(
        tmp_path, installed_text=_DRIFTED_TIMER
    )
    # Second unit: committed but not installed => would be exit 2 alone.
    (repo_root / _SECOND_UNIT_RELPATH).write_text(_BASE_SERVICE, encoding="utf-8")

    result = _run_cli(repo_root, installed_dir, _REAL_UNIT, _SECOND_UNIT)

    assert result.returncode == 1, (
        f"Drift must dominate absence; got {result.returncode}. "
        f"stdout: {result.stdout}"
    )
    assert "[drift]" in result.stdout
    # Dominated, NOT hidden.
    assert "[skip]" in result.stdout
    assert _SECOND_UNIT in result.stdout


def test_cli_unit_flag_restricts_the_run(tmp_path: pathlib.Path):
    """--unit scopes the run to the named unit(s).

    The drifted second unit is invisible when only the clean one is selected,
    which is what lets an operator scope a run around the five
    orchestrator-*.service units that are knowingly red on this host.
    """
    checker = _load_checker()
    repo_root, installed_dir = _make_trees(tmp_path)
    (repo_root / _SECOND_UNIT_RELPATH).write_text(_BASE_SERVICE, encoding="utf-8")
    (installed_dir / _SECOND_UNIT).write_text(
        _BASE_SERVICE.replace("TimeoutStartSec=1800", "TimeoutStartSec=300"),
        encoding="utf-8",
    )

    base = ["--installed-dir", str(installed_dir), "--repo-root", str(repo_root)]

    # Scoped to the clean unit: parity.
    assert checker.main(base + ["--unit", _REAL_UNIT]) == 0
    # Scoped to the drifted one: drift.
    assert checker.main(base + ["--unit", _SECOND_UNIT]) == 1
    # Unscoped: the drift is found among the rest.
    assert checker.main(base) == 1


def test_cli_run_that_compared_nothing_never_reports_parity(
    tmp_path: pathlib.Path,
):
    """A run that compared ZERO units must never return 0.

    A --repo-root naming a tree with no units compares nothing. Reporting
    parity there would be a checker that is green precisely because it looked
    at nothing — the failure mode that silently disarmed the sibling checker
    before it reported the COMPARED count instead of the SELECTED one.
    """
    checker = _load_checker()
    empty_repo = tmp_path / "empty-repo"
    (empty_repo / "scripts").mkdir(parents=True)
    _, installed_dir = _make_trees(tmp_path)

    rc = checker.main(
        ["--installed-dir", str(installed_dir), "--repo-root", str(empty_repo)]
    )

    assert rc != 0, "A run that compared zero units must not report parity."


# ---------------------------------------------------------------------------
# Machine-readable per-unit verdicts  (--print-verdicts)
# ---------------------------------------------------------------------------
#
# The human report above is written to be READ. setup-host.sh needs the same
# findings PER UNIT so it can install the clean units and skip only the drifted
# ones, and re-parsing the free-form report for that would be a second, brittle
# parser of prose that exists to be prose. `--print-verdicts` is the machine
# channel: one line per SELECTED unit, carrying the same [orchestrator_unit_parity]
# tag the installer's "did the gate actually run" guard already keys on.
#
# It is OPT-IN and additive by construction: without the flag nothing changes,
# and with it neither the exit codes nor the human report move. Both halves are
# asserted below, because a "machine-readable mode" that perturbs the exit code
# would silently re-point the installer's 0/2/other branch.

# The tag prefix every line of this checker's output carries. Spelled literally
# rather than derived from checker.LOG_TAG so a rename of the constant cannot
# quietly re-point BOTH sides of the assertion at each other — setup-host.sh
# greps for this exact string.
_LOG_TAG_PREFIX = "[orchestrator_unit_parity] "

_VERDICT_LINE_RE = re.compile(
    r"^" + re.escape(_LOG_TAG_PREFIX) + r"verdict (?P<unit>\S+) (?P<kinds>\S+)$"
)


def _verdict_lookalikes(stdout: str) -> list[str]:
    """Every line that is trying to be a verdict line, tagged or not.

    Deliberately looser than _VERDICT_LINE_RE: an UNTAGGED verdict line must
    fail loudly here rather than be invisible to the strict parser, where it
    would surface as a confusing "no verdict emitted for <unit>".
    """
    return [line for line in stdout.splitlines() if " verdict " in line]


def _parse_verdicts(stdout: str) -> dict[str, list[str]]:
    """Parse the verdict lines into ``{unit: [kind, ...]}``.

    Duplicates raise instead of merging. Two lines for one unit would let a
    last-write-wins consumer and a first-write-wins one disagree about the same
    run — and the consumer here is a bash associative array, which silently
    takes the last. A checker that can emit a unit twice must fail a test, not
    a host.
    """
    known_kinds = set(_load_checker().VERDICT_KINDS)

    verdicts: dict[str, list[str]] = {}
    for line in _verdict_lookalikes(stdout):
        match = _VERDICT_LINE_RE.match(line)
        assert match is not None, (
            f"Verdict-shaped line does not match the documented format "
            f"'{_LOG_TAG_PREFIX}verdict <unit> <kind>[,<kind>...]':\n  {line!r}\n"
            "setup-host.sh refuses to believe any output that does not carry "
            "the [orchestrator_unit_parity] tag, so an untagged or reshaped "
            "verdict line reaches the installer as NO verdict at all."
        )
        unit = match.group("unit")
        assert unit not in verdicts, (
            f"{unit} got more than one verdict line in a single run:\n{stdout}"
        )
        kinds = match.group("kinds").split(",")

        # Checked HERE, in the shared parser, so it holds for every verdict
        # case in this file rather than one dedicated test. VERDICT_KINDS is
        # the published vocabulary setup-host.sh writes a case arm for; a kind
        # emitted outside it reaches the installer's `*)` fallback and produces
        # a warning naming no actionable cause. The vocabulary must not be able
        # to grow silently.
        unknown = [k for k in kinds if k not in known_kinds]
        assert not unknown, (
            f"{unit} was given verdict kind(s) {unknown} that are not in "
            f"check_orchestrator_unit_parity.VERDICT_KINDS "
            f"({sorted(known_kinds)}). Add the kind to that tuple AND a case "
            "arm to setup-host.sh's _orch_skip_reason, or the installer will "
            "skip the unit without telling the operator why."
        )
        verdicts[unit] = kinds
    return verdicts


def _add_dropin(
    installed_dir: pathlib.Path,
    unit: str,
    *,
    body: str = "[Timer]\nAccuracySec=1min\n",
    name: str = "override.conf",
) -> pathlib.Path:
    """Write a ``<unit>.d/<name>`` drop-in — what `systemctl --user edit` does."""
    dropin_dir = installed_dir / f"{unit}.d"
    dropin_dir.mkdir(exist_ok=True)
    dropin = dropin_dir / name
    dropin.write_text(body, encoding="utf-8")
    return dropin


def _verdict_for(
    tmp_path: pathlib.Path,
    *,
    repo_text: str | None = _BASE_TIMER,
    installed_text: str | None = _BASE_TIMER,
    dropin: bool = False,
    installed_bytes: bytes | None = None,
) -> list[str]:
    """Build a one-unit tree in the given shape and return that unit's kinds.

    Returns ``[]`` when the checker emitted no verdict line for it — which is
    itself a finding, since the installer reads a missing verdict as
    "unverified" and declines to install.
    """
    repo_root, installed_dir = _make_trees(
        tmp_path, repo_text=repo_text, installed_text=installed_text
    )
    if installed_bytes is not None:
        (installed_dir / _REAL_UNIT).write_bytes(installed_bytes)
    if dropin:
        _add_dropin(installed_dir, _REAL_UNIT)

    result = _run_cli(
        repo_root, installed_dir, _REAL_UNIT, extra_args=("--print-verdicts",)
    )
    return _parse_verdicts(result.stdout).get(_REAL_UNIT, [])


def test_verdict_kinds_is_the_published_vocabulary():
    """VERDICT_KINDS is public, non-empty, deduplicated and ordered.

    Guard the guard for _parse_verdicts' membership check above and for the
    cross-artifact derivation against setup-host.sh: both iterate this tuple,
    and an empty or duplicate-bearing one would let them pass while checking
    less than they claim.
    """
    checker = _load_checker()

    assert checker.VERDICT_KINDS, "VERDICT_KINDS is empty"
    assert len(set(checker.VERDICT_KINDS)) == len(checker.VERDICT_KINDS), (
        f"VERDICT_KINDS has duplicates: {checker.VERDICT_KINDS}"
    )
    # The two install-eligible kinds must exist by these names: setup-host.sh's
    # install condition names them literally.
    assert {"clean", "absent"} <= set(checker.VERDICT_KINDS), checker.VERDICT_KINDS


def test_verdict_drifted_unit_reports_drift(tmp_path: pathlib.Path):
    """A real directive difference => `drift`."""
    assert _verdict_for(tmp_path, installed_text=_DRIFTED_TIMER) == ["drift"]


def test_verdict_comment_only_difference_plus_dropin_is_override_not_drift(
    tmp_path: pathlib.Path,
):
    """The LIVE reify shape: a drop-in over a comment-only difference.

    ``~/.config/systemd/user/orchestrator-reify.service.d/warm-lane.conf`` is a
    deliberate, in-use drop-in on this host. Its unit file differs from the
    committed copy only in comments, which the shared parser strips before
    comparison — so this must read as `override` ALONE.

    Reporting `drift` here would send the operator hunting for a directive diff
    that does not exist, and would make the installer's skip warning name the
    wrong remedy: a drop-in is removed with `systemctl --user cat/edit`, not by
    reconciling a directive.
    """
    commented = _BASE_TIMER + "# a comment-only edit, invisible to the parser\n"

    assert _verdict_for(tmp_path, installed_text=commented, dropin=True) == [
        "override"
    ]


def test_verdict_drift_and_dropin_render_in_precedence_order(
    tmp_path: pathlib.Path,
):
    """Both conditions => exactly `drift,override`, in VERDICT_KINDS order.

    Order is a CONTRACT, not an artifact of set iteration: a consumer diffing
    one run's verdicts against another's (or one host against another) can only
    do that if the same findings always render the same string.
    """
    assert _verdict_for(tmp_path, installed_text=_DRIFTED_TIMER, dropin=True) == [
        "drift",
        "override",
    ]


def test_verdict_absent_installed_copy_reports_absent(tmp_path: pathlib.Path):
    """No installed copy => `absent`, which is install-eligible but NOT `clean`.

    Kept apart from `clean` deliberately: this checker may only claim what it
    verified, and an absent copy was never compared. Calling it `clean` would
    be a parity assertion about a measurement that never happened.
    """
    assert _verdict_for(tmp_path, installed_text=None) == ["absent"]


def test_verdict_vanished_committed_copy_reports_vanished(tmp_path: pathlib.Path):
    """No COMMITTED copy => `vanished`.

    Load-bearing for the installer: with no source file, a bare `cp` would fail
    and abort the whole `set -e` script. This kind is what lets the installer
    skip that unit instead of dying.
    """
    assert _verdict_for(tmp_path, repo_text=None) == ["vanished"]


def test_verdict_undecodable_unit_reports_unreadable(tmp_path: pathlib.Path):
    """A unit that exists but cannot be decoded => `unreadable`, never `clean`."""
    assert _verdict_for(
        tmp_path, installed_bytes=b"[Timer]\nOnBootSec=\xff\xfe30\n"
    ) == ["unreadable"]


def test_verdict_lines_are_restricted_to_the_selected_units(
    tmp_path: pathlib.Path,
):
    """--unit scopes the machine channel exactly as it scopes the report.

    A verdict for an unselected unit would tell the installer something the run
    did not measure.
    """
    repo_root, installed_dir = _make_trees(tmp_path)
    (repo_root / _SECOND_UNIT_RELPATH).write_text(_BASE_SERVICE, encoding="utf-8")
    (installed_dir / _SECOND_UNIT).write_text(_BASE_SERVICE, encoding="utf-8")

    scoped = _run_cli(
        repo_root, installed_dir, _REAL_UNIT, extra_args=("--print-verdicts",)
    )
    assert _parse_verdicts(scoped.stdout) == {_REAL_UNIT: ["clean"]}, scoped.stdout

    both = _run_cli(
        repo_root,
        installed_dir,
        _REAL_UNIT,
        _SECOND_UNIT,
        extra_args=("--print-verdicts",),
    )
    assert _parse_verdicts(both.stdout) == {
        _REAL_UNIT: ["clean"],
        _SECOND_UNIT: ["clean"],
    }, both.stdout


def test_every_verdict_kind_is_reachable(tmp_path: pathlib.Path):
    """COVERAGE: every VERDICT_KINDS member is actually emitted by some tree.

    A kind nobody can produce is dead vocabulary that still forces a case arm
    in setup-host.sh, and — worse — hides the reverse defect: a bucket the
    checker populates but never renders leaves the installer with NO verdict
    for that unit, which it fail-safes into "skip". A unit silently skipped
    forever looks identical to a unit correctly declined.
    """
    checker = _load_checker()
    commented = _BASE_TIMER + "# comment-only\n"

    emitted = set()
    emitted.update(_verdict_for(tmp_path / "clean"))
    emitted.update(_verdict_for(tmp_path / "drift", installed_text=_DRIFTED_TIMER))
    emitted.update(
        _verdict_for(tmp_path / "override", installed_text=commented, dropin=True)
    )
    emitted.update(_verdict_for(tmp_path / "absent", installed_text=None))
    emitted.update(_verdict_for(tmp_path / "vanished", repo_text=None))
    emitted.update(
        _verdict_for(
            tmp_path / "unreadable",
            installed_bytes=b"[Timer]\nOnBootSec=\xff\xfe30\n",
        )
    )

    assert emitted == set(checker.VERDICT_KINDS), (
        "VERDICT_KINDS and the kinds this checker can actually emit disagree.\n"
        f"  declared but never emitted: {sorted(set(checker.VERDICT_KINDS) - emitted)}\n"
        f"  emitted but not declared:   {sorted(emitted - set(checker.VERDICT_KINDS))}"
    )


def test_print_verdicts_emits_exactly_one_clean_line_per_selected_unit(
    tmp_path: pathlib.Path,
):
    """Byte-identical trees => one `verdict <unit> clean` line per selected unit.

    Compared as a DICT against the selected set, not with `in` checks: a
    missing line and a duplicated line are both defects the installer would
    experience as "this unit has no verdict" / "this unit's verdict is whichever
    line came last", and only an exact comparison catches either.
    """
    repo_root, installed_dir = _make_trees(tmp_path)
    (repo_root / _SECOND_UNIT_RELPATH).write_text(_BASE_SERVICE, encoding="utf-8")
    (installed_dir / _SECOND_UNIT).write_text(_BASE_SERVICE, encoding="utf-8")

    result = _run_cli(
        repo_root,
        installed_dir,
        _REAL_UNIT,
        _SECOND_UNIT,
        extra_args=("--print-verdicts",),
    )

    assert result.returncode == 0, (
        f"Expected 0 for matching copies; got {result.returncode}. "
        f"stdout: {result.stdout} stderr: {result.stderr}"
    )
    assert _parse_verdicts(result.stdout) == {
        _REAL_UNIT: ["clean"],
        _SECOND_UNIT: ["clean"],
    }, result.stdout


def test_verdict_lines_carry_the_parity_log_tag(tmp_path: pathlib.Path):
    """Every verdict line is prefixed with [orchestrator_unit_parity].

    Not cosmetic. setup-host.sh believes NO status from this checker unless
    that tag appears in the output (the guard that stops a renamed script's
    argparse exit 2 from reading as the benign "not installed on this host").
    Emitting the machine channel through the same tagged writer extends that
    guard to it for free: rename the flag, argparse exits 2, no tag and no
    verdict lines appear, and every unit falls back to "unverified" — so the
    installer copies nothing. An untagged prefix would have re-opened exactly
    the silent-green hole the tag guard exists to close.
    """
    repo_root, installed_dir = _make_trees(tmp_path)

    result = _run_cli(
        repo_root, installed_dir, _REAL_UNIT, extra_args=("--print-verdicts",)
    )

    lines = _verdict_lookalikes(result.stdout)
    assert lines, f"--print-verdicts emitted no verdict line at all:\n{result.stdout}"
    for line in lines:
        assert line.startswith(_LOG_TAG_PREFIX), (
            f"Verdict line is not tagged: {line!r}. setup-host.sh greps for "
            f"{_LOG_TAG_PREFIX!r} before believing anything this checker says."
        )


def test_no_verdict_lines_without_the_flag(tmp_path: pathlib.Path):
    """The channel is OPT-IN: a default run's output is unchanged.

    An operator reading the human report should not have to skim past nine
    machine lines to find the finding, and — more importantly — the report is
    the artifact whose wording other tests and the docstring pin. Additive
    means additive.
    """
    repo_root, installed_dir = _make_trees(tmp_path)

    result = _run_cli(repo_root, installed_dir, _REAL_UNIT)

    assert _verdict_lookalikes(result.stdout) == [], (
        "A run WITHOUT --print-verdicts printed verdict lines:\n" + result.stdout
    )
    assert "[ok] parity" in result.stdout, result.stdout
    assert "1 unit(s)" in result.stdout, result.stdout


def test_print_verdicts_never_changes_the_exit_code(tmp_path: pathlib.Path):
    """The flag is observationally pure on every exit-code path.

    setup-host.sh's gate branches on 0 / 2 / other, and this flag is added to
    that very invocation. If turning it on moved any exit code, the gate would
    silently re-classify a host — a fresh machine reading as drift, or drift
    reading as benign. Each path is run twice, with and without, and compared.
    """
    # parity => 0, drift => 1, installed copy absent => 2.
    cases = {
        "parity": _make_trees(tmp_path / "parity"),
        "drift": _make_trees(tmp_path / "drift", installed_text=_DRIFTED_TIMER),
        "absent": _make_trees(tmp_path / "absent", installed_text=None),
    }

    observed: dict[str, int] = {}
    for label, (repo_root, installed_dir) in cases.items():
        plain = _run_cli(repo_root, installed_dir, _REAL_UNIT)
        with_flag = _run_cli(
            repo_root, installed_dir, _REAL_UNIT, extra_args=("--print-verdicts",)
        )
        assert plain.returncode == with_flag.returncode, (
            f"--print-verdicts changed the {label} exit code: "
            f"{plain.returncode} -> {with_flag.returncode}.\n"
            f"plain stdout: {plain.stdout}\nflagged stdout: {with_flag.stdout}"
        )
        observed[label] = plain.returncode

    # Guard the guard: three fixtures that all happened to return the same code
    # would satisfy the equality above while exercising ONE path. Pin which
    # code each case is supposed to produce.
    assert observed == {"parity": 0, "drift": 1, "absent": 2}, observed


# ---------------------------------------------------------------------------
# setup-host.sh wiring contract  (step-13 / step-14)
# ---------------------------------------------------------------------------


def _setup_host_text() -> str:
    return SETUP_HOST_PATH.read_text(encoding="utf-8")


def _orchestrator_gate_block(text: str) -> str:
    """Slice the gate's actual `if ... fi` construct out of setup-host.sh.

    Both endpoints are DERIVED, never byte counts. An earlier version of these
    tests sliced a fixed 1200-character window from the first mention of the
    checker; measured, that overshot the closing `fi` by ~300 characters into
    the `cp` lines below it, so a `exit 1` landing anywhere downstream would
    have failed the gate's own warn-only assertion with a misleading message,
    and the assertion passed only because an in-block comment happened to
    capitalise the word. A test that claims a semantic property must not be
    pinned to incidental file layout.

    Start: the beginning of the line carrying the first mention of the checker
    script. End: the first column-0 `fi` after it — the gate's own `if` closes
    there, and every `fi` nested inside it is indented.
    """
    mention = text.index("check_orchestrator_unit_parity.py")
    start = text.rfind("\n", 0, mention) + 1
    end = text.index("\nfi\n", mention) + len("\nfi\n")
    return text[start:end]


def _orchestrator_install_block(text: str) -> str:
    """Slice the construct that performs the orchestrator unit `cp`s.

    Derived the same way: from the column-0 `if` preceding the install loop's
    `cp` to the column-0 `fi` that closes it. Empty-ish slices are impossible —
    the caller asserts the loop is inside.

    The end anchor is a COLUMN-0 `fi`, so the loops' own indented `fi`/`done`
    are invisible to it and the slice still terminates at the install
    construct's close.
    """
    first_cp = text.index(_INSTALL_LOOP_CP)
    start = text.rindex("\nif ", 0, first_cp) + 1
    end = text.index("\nfi\n", first_cp) + len("\nfi\n")
    return text[start:end]


def test_setup_host_invokes_the_orchestrator_parity_gate():
    """setup-host.sh actually RUNS the checker, passing both trees.

    Without this the checker could ship fully tested and never execute — a
    gate that exists only in the test suite reports nothing about the host.
    """
    text = _setup_host_text()

    assert "check_orchestrator_unit_parity.py" in text, (
        "scripts/setup-host.sh never invokes check_orchestrator_unit_parity.py. "
        "A parity gate that the installer does not run reports nothing."
    )

    # The flags must be inside the gate construct, not merely somewhere in the
    # file (both appear in the dashboard and fused-memory blocks too).
    gate = _orchestrator_gate_block(text)
    assert "python3 " in gate, gate
    assert '--installed-dir "$UNIT_DIR"' in gate, gate
    assert '--repo-root     "$REPO_ROOT"' in gate or \
           '--repo-root "$REPO_ROOT"' in gate, gate


def test_parity_gate_runs_BEFORE_the_units_are_copied():
    """ORDERING: the gate must run before the `cp` block overwrites the host.

    This is a semantic claim, not a style pin. Once the installer has copied
    the committed units over the installed ones, there is no drift left to
    observe — a post-install-only check would report green on exactly the
    divergence this task exists to surface. setup-host.sh already learned
    this for the dashboard units, which is why it carries both a pre-install
    gate and a post-install sanity check there.
    """
    text = _setup_host_text()

    gate_at = text.index("check_orchestrator_unit_parity.py")
    first_cp_at = text.index(_INSTALL_LOOP_CP)

    assert gate_at < first_cp_at, (
        "check_orchestrator_unit_parity.py is invoked at offset "
        f"{gate_at}, AFTER the first orchestrator unit `cp` at "
        f"{first_cp_at}. A parity check that runs after the install can no "
        "longer observe host drift — it would report green on precisely the "
        "divergence it exists to surface."
    )


def test_parity_gate_distinguishes_exit_2_from_exit_1():
    """The branch treats 2 (not installed here) apart from 1 (actionable).

    Matching the existing fused-memory and dashboard blocks: 2 is a benign
    "not installed on this host, skipping", and collapsing it into the drift
    branch would make a fresh host look like it had a supervision problem.
    """
    text = _setup_host_text()
    block = _orchestrator_gate_block(text)

    assert "-eq 2" in block, (
        "The orchestrator parity branch does not distinguish exit 2 "
        "(not installed on this host) from exit 1 (actionable drift)."
    )
    # Non-fatal: the gate reports and declines to install, but must never
    # abort setup-host.sh itself — five units are knowingly red on this host
    # today, and killing the run there would take every later section with it.
    #
    # Matched as a shell COMMAND (line start, or after a `;`/`&&`/`||`) rather
    # than as the substring "exit 1", which would also hit the word in a
    # comment and make this semantic claim hostage to capitalisation.
    aborts = re.search(r"(?:^|[;&|])[ \t]*exit\b", block, re.MULTILINE)
    assert aborts is None, (
        "The orchestrator parity gate runs `exit` at "
        f"{aborts.start() if aborts else -1} in its own block. It must be "
        "non-fatal: drift declines the unit install, it does not abort the "
        "installer.\n" + block
    )


# ---------------------------------------------------------------------------
# The gate is WIRED such that it can actually stop something  (amendment)
# ---------------------------------------------------------------------------
#
# The tests above read setup-host.sh as text. These ones RUN the gate + install
# section of it, against tmp trees and a stub `systemctl`, because the property
# at stake is behavioural: on a drifted host the installer must not overwrite
# the installed units, and it must not do so QUIETLY on the strength of an exit
# code the checker never produced.
#
# Nothing here touches ~/.config/systemd/user or the real systemd: REPO_ROOT
# and UNIT_DIR are tmp_path trees and `systemctl` is a PATH stub that exits 0.


def _installer_section() -> str:
    """The gate + unit-install section of setup-host.sh, verbatim.

    From the line carrying the first mention of the checker through the `fi`
    that closes the install construct — endpoints derived, so this follows a
    reflow of the block instead of pinning one.
    """
    text = _setup_host_text()
    start = text.rfind("\n", 0, text.index("check_orchestrator_unit_parity.py")) + 1
    first_cp = text.index(_INSTALL_LOOP_CP)
    end = text.index("\nfi\n", first_cp) + len("\nfi\n")
    return text[start:end]


def _fake_repo(
    tmp_path: pathlib.Path, *, checker_body: str | None = None, with_checker: bool = True
) -> pathlib.Path:
    """A tmp repo root holding the nine committed units (+ optionally the checker).

    The unit files are copied from the real repo so the comparison under test is
    the real one; only the TREE is fake.
    """
    checker = _load_checker()
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True, exist_ok=True)
    for relpath in checker.UNITS.values():
        (repo / relpath).write_text(
            (REPO_ROOT / relpath).read_text(encoding="utf-8"), encoding="utf-8"
        )
    if with_checker:
        target = repo / "scripts" / "check_orchestrator_unit_parity.py"
        if checker_body is None:
            for name in ("check_orchestrator_unit_parity.py", "systemd_unit_parity.py"):
                (repo / "scripts" / name).write_text(
                    (REPO_ROOT / "scripts" / name).read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
        else:
            target.write_text(checker_body, encoding="utf-8")
    return repo


_SYSTEMCTL_LOG = "systemctl-calls.log"


def _run_installer_section(
    tmp_path: pathlib.Path,
    repo: pathlib.Path,
    unit_dir: pathlib.Path,
    *,
    env_extra: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Execute the sliced section under bash with setup-host.sh's own preamble."""
    stub_bin = tmp_path / "stub-bin"
    stub_bin.mkdir(exist_ok=True)
    systemctl = stub_bin / "systemctl"
    # The stub RECORDS its argv (one call per line) before exiting 0, so the
    # enable half of the install is observable and not merely assumed. Half of
    # what this section does is `systemctl --user enable`, and a per-unit gate
    # that copied the right files while enabling the wrong set would pass every
    # file-content assertion in this module. Purely additive: tests that do not
    # read the log are unaffected.
    systemctl.write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%s\\n' \"$*\" >> {tmp_path / _SYSTEMCTL_LOG}\n"
        "exit 0\n",
        encoding="utf-8",
    )
    systemctl.chmod(0o755)

    script = tmp_path / "section.sh"
    script.write_text(
        "set -euo pipefail\n"
        f'REPO_ROOT="{repo}"\n'
        f'UNIT_DIR="{unit_dir}"\n'
        'mkdir -p "$UNIT_DIR"\n'
        "info()  { printf '==> %s\\n' \"$*\"; }\n"
        "ok()    { printf 'OK %s\\n' \"$*\"; }\n"
        "warn()  { printf 'WARN %s\\n' \"$*\"; }\n"
        "fail()  { printf 'FAIL %s\\n' \"$*\"; }\n"
        + _installer_section(),
        encoding="utf-8",
    )

    env = dict(os.environ)
    env["PATH"] = f"{stub_bin}:{env.get('PATH', '')}"
    env.update(env_extra or {})
    return subprocess.run(
        ["bash", str(script)], capture_output=True, text=True, env=env
    )


def _systemctl_calls(tmp_path: pathlib.Path) -> list[list[str]]:
    """Every `systemctl` invocation the run made, as argv token lists."""
    log = tmp_path / _SYSTEMCTL_LOG
    if not log.is_file():
        return []
    return [
        line.split() for line in log.read_text(encoding="utf-8").splitlines() if line
    ]


def _enabled_units(tmp_path: pathlib.Path) -> list[str]:
    """The units passed to `systemctl ... enable <unit>` during the run.

    Token-matched rather than substring-matched: `enable` naming one unit must
    never be satisfied by a line naming a different one.
    """
    enabled: list[str] = []
    for argv in _systemctl_calls(tmp_path):
        if "enable" in argv:
            enabled.extend(argv[argv.index("enable") + 1 :])
    return enabled


def _install_all_units(repo: pathlib.Path, unit_dir: pathlib.Path) -> None:
    """Seed *unit_dir* with byte-identical copies of every committed unit."""
    checker = _load_checker()
    unit_dir.mkdir(parents=True, exist_ok=True)
    for name, relpath in checker.UNITS.items():
        (unit_dir / name).write_text(
            (repo / relpath).read_text(encoding="utf-8"), encoding="utf-8"
        )


def test_installer_copies_the_units_when_the_gate_reports_parity(
    tmp_path: pathlib.Path,
):
    """The happy path still installs — the gate must not block a clean host."""
    repo = _fake_repo(tmp_path)
    unit_dir = tmp_path / "installed"
    _install_all_units(repo, unit_dir)

    result = _run_installer_section(tmp_path, repo, unit_dir)

    assert result.returncode == 0, result.stderr
    assert "SKIPPING" not in result.stdout, result.stdout
    assert "installed and enabled" in result.stdout, result.stdout


def test_installer_does_not_overwrite_units_the_gate_reported_drift_on(
    tmp_path: pathlib.Path,
):
    """DRIFT => the destructive `cp` block is SKIPPED, not merely warned about.

    A warning is not an intervention point in a non-interactive `set -e`
    script: it scrolls past and the next line overwrites the installed units.
    And drift does not mean the installed copy is the stale one — measured
    2026-08-02, two COMMITTED units name --config paths that do not exist on
    this host, so an unconditional copy would break those orchestrators on
    their next restart. The installer must decline to act on an unverified
    diff.
    """
    repo = _fake_repo(tmp_path)
    unit_dir = tmp_path / "installed"
    _install_all_units(repo, unit_dir)
    drifted = unit_dir / "orchestrator-watchdog.timer"
    drifted.write_text(
        drifted.read_text(encoding="utf-8") + "\n[Timer]\nAccuracySec=5s\n",
        encoding="utf-8",
    )
    before = drifted.read_text(encoding="utf-8")

    result = _run_installer_section(tmp_path, repo, unit_dir)

    assert result.returncode == 0, (
        "The gate must be NON-FATAL — it declines the unit install, it does "
        f"not abort the installer.\n{result.stdout}\n{result.stderr}"
    )
    assert "SKIPPING" in result.stdout, result.stdout
    assert drifted.read_text(encoding="utf-8") == before, (
        "The installer overwrote a unit the parity gate reported drift on."
    )


def test_installer_copies_over_drift_when_explicitly_opted_in(
    tmp_path: pathlib.Path,
):
    """DF_INSTALL_ORCH_UNITS=1 is the operator's override.

    The skip is a default, not a lock: an operator who has read the report and
    decided the committed side is correct must be able to proceed without
    editing the installer.
    """
    repo = _fake_repo(tmp_path)
    unit_dir = tmp_path / "installed"
    _install_all_units(repo, unit_dir)
    drifted = unit_dir / "orchestrator-watchdog.timer"
    drifted.write_text(
        drifted.read_text(encoding="utf-8") + "\n[Timer]\nAccuracySec=5s\n",
        encoding="utf-8",
    )

    result = _run_installer_section(
        tmp_path, repo, unit_dir, env_extra={"DF_INSTALL_ORCH_UNITS": "1"}
    )

    assert result.returncode == 0, result.stderr
    assert "SKIPPING" not in result.stdout, result.stdout
    assert drifted.read_text(encoding="utf-8") == (
        repo / "scripts" / "orchestrator-watchdog.timer"
    ).read_text(encoding="utf-8"), (
        "DF_INSTALL_ORCH_UNITS=1 must still install the committed units."
    )


def _warnings(stdout: str) -> str:
    """Just the harness's `WARN ` lines, joined.

    The installer echoes the checker's whole report to stdout, and that report
    NAMES every drifted unit. Asserting "no warning names a clean unit" against
    raw stdout would therefore be asserting against the checker's own output.
    Only the installer's own warnings are the installer's claim.
    """
    return "\n".join(
        line for line in stdout.splitlines() if line.startswith("WARN ")
    )


def _warnings_naming(stdout: str, unit: str) -> str:
    """The installer's own WARN lines that NAME *unit*, joined.

    Scoped to one unit because the aggregate gate warnings describe the run as
    a whole — asserting a phrase is merely "in the warnings" would be satisfied
    by a line about some other unit entirely.
    """
    return "\n".join(line for line in _warnings(stdout).splitlines() if unit in line)


def test_one_drifted_unit_does_not_block_its_clean_siblings(
    tmp_path: pathlib.Path,
):
    """PER-UNIT: a single drifted unit must not decline the other eight.

    This is the whole point of the task. The old gate was all-or-nothing: any
    unit reporting drift, an override or an unverifiable state skipped the
    install of ALL nine — so the deliberate, permanent
    orchestrator-reify.service.d/warm-lane.conf drop-in on this host also
    blocked the watchdog pair from ever being reinstalled and re-enabled by a
    plain setup-host.sh run. A supervision safety net that cannot be repaired
    by the installer is a worse outcome than the drift the gate was protecting.

    The policy is UNCHANGED and still ratified: a drifted unit is never
    overwritten without DF_INSTALL_ORCH_UNITS=1, because drift does not tell
    you which side is stale. Only the blast radius shrinks, from nine to one.

    The clean sibling checked here is one whose installed copy is DELETED
    first, so its reinstall is positively observable — asserting on a unit that
    was already byte-identical would be satisfied by doing nothing at all.
    """
    repo = _fake_repo(tmp_path)
    unit_dir = tmp_path / "installed"
    _install_all_units(repo, unit_dir)

    drifted = unit_dir / "orchestrator-reify.service"
    drifted.write_text(
        drifted.read_text(encoding="utf-8") + "\n[Service]\nTimeoutStopSec=99\n",
        encoding="utf-8",
    )
    before = drifted.read_text(encoding="utf-8")

    reinstalled = unit_dir / "orchestrator-watchdog.timer"
    reinstalled.unlink()

    result = _run_installer_section(tmp_path, repo, unit_dir)

    assert result.returncode == 0, (
        "The gate must stay NON-FATAL — it declines individual units, it does "
        f"not abort the installer.\n{result.stdout}\n{result.stderr}"
    )
    assert reinstalled.is_file(), (
        "orchestrator-watchdog.timer was clean (absent) but was not installed — "
        "one unrelated unit's drift is still blocking the whole set.\n"
        + result.stdout
    )
    assert reinstalled.read_text(encoding="utf-8") == (
        repo / "scripts" / "orchestrator-watchdog.timer"
    ).read_text(encoding="utf-8")

    assert drifted.read_text(encoding="utf-8") == before, (
        "The installer overwrote a unit the parity gate reported drift on. The "
        "per-unit gate narrows the skip; it does not relax it."
    )

    warnings = _warnings(result.stdout)
    assert "orchestrator-reify.service" in warnings, (
        "The skip warning does not NAME the unit that was skipped. An operator "
        f"cannot act on 'something drifted'.\n{result.stdout}"
    )
    assert "byte-drift" in warnings, (
        "The skip warning does not say WHICH KIND of divergence was found. "
        "byte-drift is reconciled by editing a directive; a drop-in is removed "
        f"with systemctl --user edit — different remedies.\n{result.stdout}"
    )

    checker = _load_checker()
    wrongly_named = sorted(
        unit
        for unit in set(checker.UNITS) - {"orchestrator-reify.service"}
        if unit in warnings
    )
    assert not wrongly_named, (
        f"The installer warned about clean units {wrongly_named}, which it "
        f"installed. A warning naming a unit nothing was wrong with trains an "
        f"operator to ignore the whole block.\n{result.stdout}"
    )


def test_the_live_reify_dropin_blocks_only_reify_and_the_watchdog_is_re_enabled(
    tmp_path: pathlib.Path,
):
    """THE LIVE SCENARIO, end to end, including the ENABLE half.

    Reproduces this host exactly: ``orchestrator-reify.service`` carries a
    deliberate, permanent ``orchestrator-reify.service.d/warm-lane.conf``
    drop-in over a unit file that differs from the committed copy only in
    comments — so its verdict is `override`, not `drift`. Under the old
    all-or-nothing gate that ONE unit declined the install of all nine, which
    is why the watchdog pair could not be reinstalled and re-enabled by a plain
    setup-host.sh run. On 2026-08-10 that left the fleet 31.8h stale: the
    supervision safety net was the thing the gate was protecting from repair.

    The enable half is asserted from the systemctl stub's call log, not
    inferred from file contents, because it is a genuinely separate obligation:
    copying orchestrator-watchdog.timer into place does nothing at all until it
    is enabled. Three distinct properties are checked:
      - the timer IS enabled (the repair path the stale fleet needed);
      - orchestrator-watchdog.service is NOT (it is static — no [Install] — and
        `systemctl enable` on a static unit is an ERROR, not a no-op, so under
        `set -e` that would abort the installer outright);
      - orchestrator-reify.service is NOT (a unit that did not clear the gate is
        neither copied NOR enabled — enabling a unit whose install was declined
        would act on exactly the state the skip refused to act on).
    """
    repo = _fake_repo(tmp_path)
    unit_dir = tmp_path / "installed"
    _install_all_units(repo, unit_dir)

    # Reify: comment-only divergence + a drop-in. The parser strips comments,
    # so this must read as `override` ALONE — the phrasing matters because the
    # remedies differ (edit a directive vs. `systemctl --user cat/edit`).
    reify = unit_dir / "orchestrator-reify.service"
    reify.write_text(
        reify.read_text(encoding="utf-8")
        + "\n# host-local annotation, invisible to the parser\n",
        encoding="utf-8",
    )
    reify_before = reify.read_text(encoding="utf-8")
    dropin = _add_dropin(
        unit_dir,
        "orchestrator-reify.service",
        body="[Unit]\nWants=reify-warm-lane.service\nAfter=reify-warm-lane.service\n",
        name="warm-lane.conf",
    )
    dropin_before = dropin.read_text(encoding="utf-8")

    # Deleted so their reinstall is positively observable: asserting on units
    # that were already byte-identical would be satisfied by doing nothing.
    (unit_dir / "orchestrator-watchdog.timer").unlink()
    (unit_dir / "orchestrator-watchdog.service").unlink()

    result = _run_installer_section(tmp_path, repo, unit_dir)

    assert result.returncode == 0, (
        "The gate must stay NON-FATAL — it declines individual units, it does "
        f"not abort the installer.\n{result.stdout}\n{result.stderr}"
    )

    for name in ("orchestrator-watchdog.service", "orchestrator-watchdog.timer"):
        installed = unit_dir / name
        assert installed.is_file(), (
            f"{name} was absent (install-eligible) but was not installed — "
            "reify's drop-in is still blocking the watchdog pair, which is the "
            f"exact failure this task exists to fix.\n{result.stdout}"
        )
        assert installed.read_text(encoding="utf-8") == (
            repo / "scripts" / name
        ).read_text(encoding="utf-8")

    enabled = _enabled_units(tmp_path)
    assert "orchestrator-watchdog.timer" in enabled, (
        "orchestrator-watchdog.timer was copied but never enabled. A timer "
        "unit file on disk supervises nothing; enabling it is the repair the "
        f"31.8h-stale fleet needed.\ncalls: {_systemctl_calls(tmp_path)}"
    )
    assert "orchestrator-watchdog.service" not in enabled, (
        "orchestrator-watchdog.service is STATIC (no [Install]) — `systemctl "
        "enable` on it is an error, not a no-op, so under `set -e` this would "
        f"abort the installer.\ncalls: {_systemctl_calls(tmp_path)}"
    )
    assert "orchestrator-reify.service" not in enabled, (
        "A unit the gate declined to install was still enabled. The skip must "
        "cover BOTH halves: enabling a unit whose install was declined acts on "
        f"exactly the unverified state the skip refused.\nenabled: {enabled}"
    )

    reload_calls = [argv for argv in _systemctl_calls(tmp_path) if "daemon-reload" in argv]
    assert len(reload_calls) == 1, (
        "daemon-reload must run exactly once, AFTER the copies and BEFORE the "
        "enables — systemd must not be asked to enable a unit it has not "
        f"re-read.\ncalls: {_systemctl_calls(tmp_path)}"
    )

    assert reify.read_text(encoding="utf-8") == reify_before, (
        "The installer overwrote a unit the gate reported an override on."
    )
    assert dropin.read_text(encoding="utf-8") == dropin_before, (
        "The installer disturbed the deliberate warm-lane.conf drop-in."
    )

    reify_warning = _warnings_naming(result.stdout, "orchestrator-reify.service")
    assert "drop-in" in reify_warning, (
        "The skip warning for a drop-in'd unit does not say `drop-in`. The "
        "remedy is `systemctl --user cat/edit`, not reconciling a directive."
        f"\n{result.stdout}"
    )
    assert "byte-drift" not in reify_warning, (
        "The skip warning blames byte-drift on a unit whose only divergence is "
        "a drop-in over comment-only edits — sending the operator hunting for "
        f"a directive diff that does not exist.\n{result.stdout}"
    )


def _verdict_stub(exit_code: int, verdicts: dict[str, str]) -> str:
    """A stub checker printing a TAGGED report and the given verdict lines.

    Used to reach states the real checker cannot be driven into from a tmp tree
    — chiefly "the report is well formed but a unit has no verdict line", which
    is what an older checker, a refactor that dropped the emit, or a registry
    that does not know a unit would each produce.
    """
    lines = ["[orchestrator_unit_parity] stub report over 9 units"]
    lines += [
        f"[orchestrator_unit_parity] verdict {unit} {kinds}"
        for unit, kinds in sorted(verdicts.items())
    ]
    return (
        "import sys\n"
        + "".join(f"print({line!r})\n" for line in lines)
        + f"sys.exit({exit_code})\n"
    )


def test_a_report_with_no_verdict_lines_installs_nothing(tmp_path: pathlib.Path):
    """FAIL-SAFE: a tagged, exit-0 report carrying NO verdicts installs NOTHING.

    The third face of the same collision the two tests below cover, and the one
    the machine channel newly opens: here the gate DID run, its tag IS present
    and it exited 0 — everything the installer checks before reading verdicts
    says "green" — yet it said nothing per-unit. A checker refactored to drop
    the emit, an older copy on a rebuilt host, or a registry that does not know
    these units all land exactly here.

    Reading that as "no findings, install everything" would make the per-unit
    gate strictly WEAKER than the all-or-nothing one it replaces: the states
    that produce no verdict are precisely "nothing was checked", and installing
    on the strength of nothing is the silent-drift failure this gate exists to
    catch. A unit with no verdict is therefore BLOCKED, and the warning must
    say so — "skipped" with no cause is indistinguishable from a real finding.
    """
    repo = _fake_repo(tmp_path, checker_body=_verdict_stub(0, {}))
    unit_dir = tmp_path / "installed"
    _install_all_units(repo, unit_dir)
    deleted = unit_dir / "orchestrator-watchdog.timer"
    deleted.unlink()

    result = _run_installer_section(tmp_path, repo, unit_dir)

    assert result.returncode == 0, result.stderr
    assert not deleted.exists(), (
        "A unit with NO verdict was installed anyway. The gate reported no "
        "finding because it measured nothing, not because there was nothing "
        f"to find.\n{result.stdout}"
    )
    assert _enabled_units(tmp_path) == [], (
        "Units were enabled on a run where nothing cleared the gate.\n"
        f"{_systemctl_calls(tmp_path)}"
    )
    assert "SKIPPING" in result.stdout, result.stdout

    warning = _warnings_naming(result.stdout, "orchestrator-watchdog.timer")
    assert "no verdict" in warning, (
        "The skip warning does not tell the operator the gate returned NO "
        "VERDICT for this unit. Without that, a checker that silently stopped "
        "reporting is indistinguishable from a host with real drift — and the "
        f"remedies are opposite.\n{result.stdout}"
    )


def test_a_vanished_committed_unit_is_skipped_not_fatal(tmp_path: pathlib.Path):
    """A missing SOURCE file must skip that unit, never abort the installer.

    `cp` of a nonexistent source fails, and under `set -euo pipefail` that
    aborts the whole script — taking every section after this one with it. The
    `vanished` verdict is what lets the loop decline that unit instead of dying
    on it, so this is the regression that must never land: the per-unit gate
    made `cp` run inside a loop over a computed set, which is exactly where an
    unguarded missing source would first bite.
    """
    repo = _fake_repo(tmp_path)
    unit_dir = tmp_path / "installed"
    _install_all_units(repo, unit_dir)
    (repo / "scripts" / "orchestrator-know-live.service").unlink()

    survivor = unit_dir / "orchestrator-watchdog.timer"
    survivor.unlink()

    result = _run_installer_section(tmp_path, repo, unit_dir)

    assert result.returncode == 0, (
        "A vanished committed unit ABORTED the installer instead of being "
        f"skipped.\n{result.stdout}\n{result.stderr}"
    )
    assert survivor.is_file(), (
        "The vanished unit stopped its clean siblings from being installed.\n"
        + result.stdout
    )
    assert "orchestrator-know-live.service" in _warnings_naming(
        result.stdout, "orchestrator-know-live.service"
    ), f"The vanished unit was skipped silently.\n{result.stdout}"
    assert "orchestrator-know-live.service" not in _enabled_units(tmp_path), (
        "A unit with no committed copy was enabled.\n"
        f"{_systemctl_calls(tmp_path)}"
    )


def test_force_all_installs_even_drifted_and_unverified_units(
    tmp_path: pathlib.Path,
):
    """DF_INSTALL_ORCH_UNITS=1 still means ALL, per-unit gate notwithstanding.

    The override is the operator's escape hatch after they have read the report
    and decided the committed side is correct. A per-unit gate that quietly
    kept skipping the no-verdict units under the override would leave them with
    no way to install those units at all without editing the installer.
    """
    checker = _load_checker()
    unverified = "orchestrator-watchdog.timer"
    drifted = "orchestrator-reify.service"
    stub = _verdict_stub(
        1,
        {
            unit: ("drift" if unit == drifted else "clean")
            for unit in checker.UNITS
            if unit != unverified
        },
    )
    repo = _fake_repo(tmp_path, checker_body=stub)
    unit_dir = tmp_path / "installed"
    unit_dir.mkdir(parents=True, exist_ok=True)

    result = _run_installer_section(
        tmp_path, repo, unit_dir, env_extra={"DF_INSTALL_ORCH_UNITS": "1"}
    )

    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "SKIPPING" not in result.stdout, (
        f"The override skipped a unit anyway.\n{result.stdout}"
    )
    assert "installing over the reported drift" in result.stdout, (
        "The override installed over a reported finding SILENTLY. The escape "
        "hatch must still say out loud that it acted against the gate — that "
        f"line is the only record in the run's output.\n{result.stdout}"
    )
    missing = sorted(
        name for name in checker.UNITS if not (unit_dir / name).is_file()
    )
    assert not missing, (
        f"DF_INSTALL_ORCH_UNITS=1 did not install {missing}. The override is "
        f"force-ALL; the per-unit gate narrows the default, not the escape "
        f"hatch.\n{result.stdout}"
    )
    assert unverified in _enabled_units(tmp_path), (
        f"{unverified} was force-installed but not enabled.\n"
        f"{_systemctl_calls(tmp_path)}"
    )


def test_force_all_still_cannot_copy_a_source_that_does_not_exist(
    tmp_path: pathlib.Path,
):
    """The override installs over a FINDING; it cannot conjure a missing file.

    Measured before this guard existed: DF_INSTALL_ORCH_UNITS=1 with one
    committed template deleted exits 1 at `cp: cannot stat ...`, and under
    `set -euo pipefail` that aborts setup-host.sh outright — no daemon-reload,
    no enables, and every LATER section of the host installer (jCodeMunch,
    Claude config, ...) silently never runs. The operator asked to install over
    a reported drift; they did not ask to halt the host setup.

    The guard is deliberately PHYSICAL (does the source file exist) rather than
    a fourth verdict arm, because it must also hold for a unit the gate never
    reported on at all — `_orch_units` and the checker's registry are kept in
    step by a test, but under the override a unit with no verdict is installed
    on trust, and trust does not create a file.
    """
    repo = _fake_repo(tmp_path)
    unit_dir = tmp_path / "installed"
    _install_all_units(repo, unit_dir)
    (repo / "scripts" / "orchestrator-know-live.service").unlink()

    survivor = unit_dir / "orchestrator-watchdog.timer"
    survivor.unlink()

    result = _run_installer_section(
        tmp_path, repo, unit_dir, env_extra={"DF_INSTALL_ORCH_UNITS": "1"}
    )

    assert result.returncode == 0, (
        "DF_INSTALL_ORCH_UNITS=1 with a vanished committed unit ABORTED the "
        "installer. Every section after this one was skipped.\n"
        f"{result.stdout}\n{result.stderr}"
    )
    assert survivor.is_file(), (
        "The vanished unit stopped its siblings from being installed even "
        f"under the override.\n{result.stdout}"
    )
    assert "orchestrator-know-live.service" in _warnings_naming(
        result.stdout, "orchestrator-know-live.service"
    ), f"The unit with no source was skipped silently.\n{result.stdout}"


@pytest.mark.skipif(
    os.geteuid() == 0,
    reason="root ignores the permission bits this test uses to make `cp` fail",
)
def test_a_unit_that_cannot_be_copied_does_not_abort_the_installer(
    tmp_path: pathlib.Path,
):
    """An UNWRITABLE destination must skip that unit, never kill the run.

    The sibling guard above covers a source that does not EXIST. This is the
    other half, and the existence test cannot reach it: the checker raises
    `unreadable` on ``(OSError, UnicodeDecodeError)``, and the OSError half is
    precisely a file `cp` cannot touch. Reproduced here the way an operator
    meets it — DF_INSTALL_ORCH_UNITS=1 over an installed unit at mode 000, i.e.
    the override told the installer to write a file this user cannot open for
    writing. `cp` exits 1 at "cannot create regular file: Permission denied",
    and under `set -euo pipefail` a bare `cp` would abort setup-host.sh right
    there: no daemon-reload, no enables, and every LATER section of the host
    installer (jCodeMunch, Claude config, ...) silently never runs.

    The unwritable unit is deliberately ordered BEFORE the survivor in
    `_orch_units`, so a run that aborts on it cannot accidentally satisfy the
    survivor assertion.
    """
    repo = _fake_repo(tmp_path)
    unit_dir = tmp_path / "installed"
    _install_all_units(repo, unit_dir)

    unwritable = unit_dir / "orchestrator-know-live.service"
    unwritable.chmod(0o000)

    # Copied last by the array's order, and absent, so only a run that got past
    # the failure can produce it.
    survivor = unit_dir / "orchestrator-watchdog.timer"
    survivor.unlink()

    result = _run_installer_section(
        tmp_path, repo, unit_dir, env_extra={"DF_INSTALL_ORCH_UNITS": "1"}
    )
    unwritable.chmod(0o644)

    assert result.returncode == 0, (
        "A unit that could not be copied ABORTED the installer. Every section "
        f"after this one was skipped.\n{result.stdout}\n{result.stderr}"
    )
    assert survivor.is_file(), (
        "The uncopyable unit stopped its siblings from being installed.\n"
        + result.stdout
    )
    assert "orchestrator-watchdog.timer" in _enabled_units(tmp_path), (
        "The run survived the failure but never reached the enable loop.\n"
        f"{_systemctl_calls(tmp_path)}"
    )
    assert "FAILED" in _warnings_naming(
        result.stdout, "orchestrator-know-live.service"
    ), (
        "The failed copy was not reported. A unit silently not installed is "
        f"the state this whole section exists to make observable.\n{result.stdout}"
    )
    assert "orchestrator-know-live.service" not in _enabled_units(tmp_path), (
        "A unit whose copy FAILED was enabled anyway — that acts on bytes "
        f"nobody managed to write.\n{_systemctl_calls(tmp_path)}"
    )


def test_a_missing_checker_does_not_read_as_not_installed_here(
    tmp_path: pathlib.Path,
):
    """EXIT-CODE COLLISION: `python3 <missing script>` also exits 2.

    2 is the checker's benign "not installed on this host, installing below".
    If the checker were renamed or moved, python3's own 2 would land in that
    same branch and the installer would print a reassuring line and copy the
    units anyway — a gate reporting green because it never ran, which is the
    exact silent-drift failure the checker exists to catch, reproduced one
    level up in its own wiring.
    """
    repo = _fake_repo(tmp_path, with_checker=False)
    unit_dir = tmp_path / "installed"
    _install_all_units(repo, unit_dir)
    (unit_dir / "orchestrator-watchdog.timer").write_text(
        "[Timer]\nOnBootSec=999\n", encoding="utf-8"
    )
    before = (unit_dir / "orchestrator-watchdog.timer").read_text(encoding="utf-8")

    result = _run_installer_section(tmp_path, repo, unit_dir)

    assert result.returncode == 0, result.stderr
    assert "not yet installed" not in result.stdout, (
        "A missing checker was reported as the benign 'not installed on this "
        f"host'.\n{result.stdout}"
    )
    assert "SKIPPING" in result.stdout, result.stdout
    assert (unit_dir / "orchestrator-watchdog.timer").read_text(
        encoding="utf-8"
    ) == before


def test_a_usage_error_does_not_read_as_not_installed_here(
    tmp_path: pathlib.Path,
):
    """SAME COLLISION, second source: argparse exits 2 on any usage error.

    Simulated with a stub checker that exits 2 with argparse-shaped stderr and
    no [orchestrator_unit_parity] report — what renaming a flag in a future
    refactor would produce. The tag, not the exit code, is what makes a status
    believable.
    """
    repo = _fake_repo(
        tmp_path,
        checker_body=(
            "import sys\n"
            "sys.stderr.write('usage: check_orchestrator_unit_parity.py "
            "[-h]\\nerror: unrecognized arguments: --installed-dir\\n')\n"
            "sys.exit(2)\n"
        ),
    )
    unit_dir = tmp_path / "installed"
    _install_all_units(repo, unit_dir)
    (unit_dir / "orchestrator-watchdog.timer").write_text(
        "[Timer]\nOnBootSec=999\n", encoding="utf-8"
    )
    before = (unit_dir / "orchestrator-watchdog.timer").read_text(encoding="utf-8")

    result = _run_installer_section(tmp_path, repo, unit_dir)

    assert result.returncode == 0, result.stderr
    assert "not yet installed" not in result.stdout, result.stdout
    assert "SKIPPING" in result.stdout, result.stdout
    assert (unit_dir / "orchestrator-watchdog.timer").read_text(
        encoding="utf-8"
    ) == before


# A `_orch_skip_reason` case arm: `*,drift,*)`, or a `|`-alternated one like
# `*,drift,override,*|*,drift,*,override,*)`. Alternation is included because
# an arm the parse cannot see reads as a MISSING arm, and this guard's failure
# would then name a kind that is in fact handled.
_SKIP_ARM_RE = re.compile(r"^\s*(\*,[a-z,*|]+,\*)\)\s*$", re.M)
# A literal kind named in the install-eligible condition: `[ "$_kinds" = clean ]`.
_INSTALL_ELIGIBLE_RE = re.compile(r'\[\s*"\$_kinds"\s*=\s*([a-z]+)\s*\]')


def _skip_arm_kinds(section: str) -> set[str]:
    """Kinds named by `_orch_skip_reason`'s case arms."""
    return {
        kind
        for arm in _SKIP_ARM_RE.findall(section)
        for kind in re.findall(r"[a-z]+", arm)
    }


def _install_eligible_kinds(section: str) -> set[str]:
    """Kinds the install condition accepts literally."""
    return set(_INSTALL_ELIGIBLE_RE.findall(section))


def _skip_reason(kinds: str, unit: str = "orchestrator-x.service") -> str:
    """Run setup-host.sh's own `_orch_skip_reason` for *kinds*, under bash.

    Extracted from the live script and EXECUTED rather than read, because the
    defect class here is a case pattern that parses fine, reads plausibly and
    never matches — invisible to any test that only greps for the arm.
    """
    section = _installer_section()
    start = section.index("_orch_skip_reason() {")
    end = section.index("\n}\n", start) + len("\n}\n")
    result = subprocess.run(
        ["bash", "-c", f'{section[start:end]}\n_orch_skip_reason "$1" "$2"', "_", kinds, unit],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"_orch_skip_reason({kinds!r}) failed: {result.stderr}"
    return result.stdout


def test_every_skip_reason_arm_fires_for_the_kinds_it_claims():
    """Each arm is EXECUTED, because an arm that never matches is invisible.

    Measured: the combined arm was written `*,drift,*,override,*)`, which can
    never match `,drift,override,` — the comma the first `,drift,` consumes is
    the same one `,override,` needs. It parsed, it read correctly, and it
    silently lost every match to the drift-only arm below it.

    The consequence is not cosmetic. A unit with BOTH byte-drift and a drop-in
    would be reported as byte-drift alone; the operator reconciles the
    directive, re-runs, and the unit is skipped AGAIN — for a drop-in nobody
    ever told them about. A warning that names an incomplete remedy is worse
    than one that names none, because it looks like progress.
    """
    checker = _load_checker()
    blocking = [k for k in checker.VERDICT_KINDS if k not in {"clean", "absent"}]

    for kind in [*blocking, "unverified"]:
        message = _skip_reason(kind).strip()
        assert message, f"{kind} produced an EMPTY reason"

    combined = _skip_reason("drift,override")
    assert "drift" in combined and "drop-in" in combined, (
        "A unit with BOTH byte-drift and a drop-in override is told about only "
        f"one of them: {combined!r}\nThe operator fixes what they were told "
        "about, re-runs, and is skipped again for the other."
    )

    # Each single kind must still get ITS OWN message, not the combined one —
    # a combined arm widened until it swallows the singles is the same defect
    # with the sign flipped.
    assert "drop-in" not in _skip_reason("drift"), _skip_reason("drift")
    assert "byte-drift" not in _skip_reason("override"), _skip_reason("override")

    unknown = _skip_reason("masked")
    assert "masked" in unknown, (
        "The `*)` fallback does not name the unhandled kind VERBATIM, so an "
        f"operator cannot tell which kind went unhandled: {unknown!r}"
    )
    assert "setup-host.sh" in unknown, (
        "The fallback does not name the file that has to change. It is the one "
        f"message whose reader has no other clue where to look: {unknown!r}"
    )


def test_setup_host_handles_every_verdict_kind_the_checker_can_emit():
    """CROSS-ARTIFACT: the checker's vocabulary and the shell's arms agree.

    The two artifacts are coupled by a string protocol with no compiler between
    them. A kind added to VERDICT_KINDS later — by someone reading only the
    checker — reaches setup-host.sh's `*)` fallback, and the operator gets a
    unit skipped for a cause the installer cannot name. That is the precise
    shape of silent degradation these gates exist to prevent: the install still
    "works", it just quietly stops doing something, and the one line that could
    have explained it says nothing actionable.

    Kinds are matched by PARSING the case arms and the install condition, not
    by searching the region for the word. The region's comments name every kind
    repeatedly, so a substring check would pass on documentation alone —
    exactly the vacuous guard this is meant not to be.
    """
    checker = _load_checker()
    section = _installer_section()

    # Guard the guard: every assertion below is over a derived set, and an
    # empty derivation would pass while checking nothing.
    assert section.strip(), "The installer section sliced EMPTY."
    assert "_orch_skip_reason" in section, (
        "The sliced region does not contain _orch_skip_reason — the slice "
        "anchors have drifted off the block this test is about."
    )
    assert checker.VERDICT_KINDS, "VERDICT_KINDS is empty"

    arms = _skip_arm_kinds(section)
    eligible = _install_eligible_kinds(section)
    assert arms, (
        "Parsed ZERO case arms out of _orch_skip_reason. Fix the arm regex — "
        "do NOT weaken this test; a zero-arm parse makes every assertion below "
        "vacuously true."
    )
    assert eligible, (
        "Parsed ZERO kinds out of the install-eligible condition. Fix the "
        "regex rather than dropping the assertion."
    )

    unhandled = sorted(set(checker.VERDICT_KINDS) - arms - eligible)
    assert not unhandled, (
        f"check_orchestrator_unit_parity.VERDICT_KINDS can emit {unhandled}, "
        "which setup-host.sh neither treats as install-eligible nor phrases in "
        "_orch_skip_reason. Add a case arm (or name it in the install "
        "condition) — otherwise a unit is skipped with a warning that names no "
        "actionable cause."
    )

    # The shell-only kind: no verdict line at all. It never appears in
    # VERDICT_KINDS (the checker cannot emit "I said nothing"), so it would be
    # invisible to the loop above — and it is the single most likely kind an
    # operator actually meets, since it is what a missing or older checker
    # produces for every unit at once.
    assert "unverified" in arms, (
        "setup-host.sh's `unverified` default — a unit with NO verdict line — "
        "has no _orch_skip_reason arm, so the most common real-world skip "
        "would fall through to the unhandled-kind fallback."
    )

    assert "clean" in eligible and "absent" in eligible, (
        "The two install-eligible kinds are no longer named literally in the "
        f"install condition (found {sorted(eligible)}), so this guard can no "
        "longer tell an install-eligible kind from an unhandled one."
    )

    assert re.search(r"^\s*\*\)\s*$", section, re.M), (
        "_orch_skip_reason has no `*)` fallback arm. Under a future kind the "
        "case would fall through and print nothing at all, leaving the warning "
        "reading 'SKIPPING <unit> — ; its installed copy is UNCHANGED'."
    )


def test_setup_host_parses_cleanly():
    """`bash -n scripts/setup-host.sh` — the added block must not break the script."""
    result = subprocess.run(
        ["bash", "-n", str(SETUP_HOST_PATH)], capture_output=True, text=True
    )

    assert result.returncode == 0, (
        f"bash -n rejected {SETUP_HOST_PATH}: {result.stderr}"
    )
