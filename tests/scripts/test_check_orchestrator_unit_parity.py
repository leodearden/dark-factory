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
) -> subprocess.CompletedProcess:
    """Invoke the checker as a real subprocess against tmp_path trees.

    Uses sys.executable rather than a bare `python3`: commit 5178360711 fixed
    exactly that defect in the sibling scanner CLI tests, where a bare python3
    resolved to a different interpreter than the one running the suite.
    """
    argv = [
        sys.executable,
        str(CHECKER_PATH),
        "--installed-dir", str(installed_dir),
        "--repo-root", str(repo_root),
    ]
    for unit in units:
        argv += ["--unit", unit]
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

    Derived the same way: from the column-0 `if` preceding the first
    orchestrator `cp` to the column-0 `fi` that closes it. Empty-ish slices
    are impossible — the caller asserts the `cp` lines are inside.
    """
    first_cp = text.index('cp "$REPO_ROOT/scripts/orchestrator-dark-factory.service"')
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
    first_cp_at = text.index(
        'cp "$REPO_ROOT/scripts/orchestrator-watchdog.service"'
    )

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
    first_cp = text.index('cp "$REPO_ROOT/scripts/orchestrator-dark-factory.service"')
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
    systemctl.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
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


def test_setup_host_parses_cleanly():
    """`bash -n scripts/setup-host.sh` — the added block must not break the script."""
    result = subprocess.run(
        ["bash", "-n", str(SETUP_HOST_PATH)], capture_output=True, text=True
    )

    assert result.returncode == 0, (
        f"bash -n rejected {SETUP_HOST_PATH}: {result.stderr}"
    )
