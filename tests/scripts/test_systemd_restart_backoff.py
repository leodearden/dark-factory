"""Fleet-wide guard: a declared restart-backoff cap must actually engage.

This module owns the RELATIONAL invariant for the whole repo — every file
that declares ``RestartMaxDelaySec=`` must pair it with ``RestartSteps=``,
because systemd parses the cap, logs "Service has RestartMaxDelaySec= but no
RestartSteps= setting. Ignoring." at unit load, and then discards it.  The
backoff the unit's own comment advertises never happens and nothing in the
file's text reveals that.

The assertion itself is NOT defined here.  It was written for task 3333
(dashboard unit + template) as a private helper inside
tests/scripts/test_dashboard_service_template.py, and task 3408 lifted it
into the importable tests/scripts/systemd_unit_invariants.py so that this
fleet-wide sweep and the dashboard suite share ONE implementation rather
than two copies that can drift apart.  test_restart_backoff_guard_rejects_
ineffective_units in the dashboard module remains the helper's negative-case
guard; this module supplies its fleet-wide application.
"""
import pathlib
import subprocess

import pytest
from systemd_unit_invariants import (
    assert_restart_backoff_effective,
    restart_directive,
)

REPO_ROOT = pathlib.Path(__file__).parents[2]

# What discovery refuses to treat as a unit, excluded by CATEGORY rather than
# by naming individual paths.  Both categories are files that CONTAIN a unit as
# quoted text rather than files systemd can load, and both break the invariant
# the same way: restart_directive is last-occurrence-wins FILE-WIDE, so on a
# file holding more than one embedded unit it splices a directive out of one and
# a directive out of another and reports a verdict about neither.
#
#   **/tests/**  — parity suites embed whole units as column-0 triple-quoted
#     fixtures.  Measured: tests/scripts/test_check_fused_memory_unit_parity.py
#     was swept as a 13th "unit" and PASSED by accident, splicing a cap out of
#     the NEGATIVE fixture (which deliberately models the defect) together with
#     RestartSteps= out of an unrelated POSITIVE one.  The glob form is
#     load-bearing: a plain `:!tests/` excludes only the top-level directory and
#     leaves fused-memory/tests/, orchestrator/tests/, scripts/tests/ and
#     dashboard/tests/ swept — and fused-memory/tests/test_systemd_unit_config.py
#     already parses systemd units, so one fixture there gaining a column-0 cap
#     would drag a .py file back in.
#
#   **/*.md — prose.  A doc may legitimately show the DEFECT: a PRD or
#     postmortem for this very task would carry a "before" fence (cap, no steps)
#     next to an "after" fence, and no mechanical rule distinguishes a
#     cautionary example from a prescription.  plans/afk-C1-systemd.md is the
#     live instance — an as-built record of what was deployed, already diverged
#     from the fleet in three visible ways (`Requires=fused-memory.service`,
#     which the real units reject and test_orchestrator_service_files.py asserts
#     is ABSENT; an obsolete `--config orchestrator/config.yaml`; no `--frozen`).
#     Editing a directive inside it would falsify the record without making any
#     unit correct.  Excluding the category rather than the path means the next
#     doc quoting a unit does not turn CI red and does not have to be
#     hand-added to a constant in a test file.
#
# The cost is that a doc which IS a copy-source for real units must opt back in
# explicitly.  Exactly one does — skills/factory-init/references/supervised-unit
# .md — and its guard below is UNCONDITIONAL, i.e. strictly stronger than the
# sweep, not a weaker substitute for it.
_NON_UNIT_PATHSPECS = (
    ":(exclude,glob)**/tests/**",
    ":(exclude,glob)**/*.md",
)

# Every path the sweep is known to cover today.  Guards against a discovery
# that silently returns fewer files than it should — see
# test_discovery_covers_every_known_unit.
_EXPECTED_SWEPT_PATHS = frozenset(
    {
        "dashboard/dark-factory-dashboard.service",
        "fused-memory/fused-memory.service.example-systemd-config",
        "scripts/dashboard.service.template",
        "scripts/fused-memory.service.template",
        "scripts/jcodemunch-watcher.service.template",
        "scripts/orchestrator-autopilot-video.service",
        "scripts/orchestrator-dark-factory.service",
        "scripts/orchestrator-know-live.service",
        "scripts/orchestrator-my-solar-challenge.service",
        "scripts/orchestrator-pump-web-ui.service",
        "scripts/orchestrator-reify.service",
        "scripts/orchestrator-solar-challenge-platform.service",
    }
)

_FACTORY_INIT_REFERENCE = "skills/factory-init/references/supervised-unit.md"


def discover_units_declaring_a_restart_cap() -> list[str]:
    """Return every git-tracked file declaring ``RestartMaxDelaySec=``, sans exclusions.

    Discovery is by CONTENT, not by filename glob and not by a hand-maintained
    list, because the stated goal is that a FUTURE unit cannot reintroduce this
    bug.  A hand-maintained list fails that outright — a unit added next month
    is simply not in it.  A glob cannot work either: the affected files span
    four naming conventions (``*.service``, ``*.service.template``,
    ``*.example-systemd-config``, and a fenced ini block inside a ``.md``)
    across four directories, so any glob broad enough to catch them all is
    broader than the invariant.  Keying on the declared cap keys on exactly the
    thing that creates the obligation: a new unit anywhere in the tree is
    covered the moment it declares one, and a unit that deliberately declares
    no cap is never dragged in.

    The returncode assertion is load-bearing.  ``git grep`` exits 0 on matches,
    1 on no matches, and >1 on a real error; a helper that swallows non-zero
    returns [], the parametrize below collects ZERO cases, and the sweep
    reports green while checking nothing.  That is the same silent-empty
    failure mode test_orchestrator_service_glob_covers_all_known_units exists
    to prevent for the sibling glob, except a subprocess can fail in more ways
    than a glob can.

    Test directories at any depth and markdown files are excluded via
    _NON_UNIT_PATHSPECS — see the rationale on that constant.  Both hold units
    as quoted text rather than as files systemd loads, and the file-wide
    last-wins read splices their embedded units together.

    The grep pattern tolerates leading whitespace and whitespace around the
    separator, matching both systemd.syntax and restart_directive's own anchor.
    A column-0-only pattern would leave a valid ``  RestartMaxDelaySec = 60``
    undiscovered, so the file would never be swept, the pairing never checked,
    and the sweep would report green — a silent skip in a guard whose whole
    purpose is catching a silently-ignored directive.
    """
    proc = subprocess.run(
        [
            "git",
            "grep",
            "-lE",
            r"^[ \t]*RestartMaxDelaySec[ \t]*=",
            "--",
            ".",
            *_NON_UNIT_PATHSPECS,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (
        f"`git grep` for RestartMaxDelaySec= exited {proc.returncode} in "
        f"{REPO_ROOT} (0=matches, 1=no matches, >1=error), so unit discovery "
        "produced nothing to check. A non-zero exit here must fail loudly: "
        "silently returning no paths would collect zero parametrized cases and "
        f"report this whole sweep green while checking nothing. stderr: "
        f"{proc.stderr.strip()!r}"
    )
    return sorted(line.strip() for line in proc.stdout.splitlines() if line.strip())


# Filename shapes a systemd unit definition is allowed to take in this repo.
# Used to police what discovery drags IN, not what it must contain — see
# test_discovery_sweeps_only_systemd_units.  `.md` is deliberately NOT here:
# markdown is prose that quotes units, never a unit, and _NON_UNIT_PATHSPECS
# keeps it out of discovery in the first place.
_UNIT_FILE_SUFFIXES = (
    ".service",
    ".service.template",
    ".example-systemd-config",
)


def test_discovery_covers_every_known_unit() -> None:
    """Coverage guard: discovery must be non-empty and cover every known unit.

    Mirrors test_orchestrator_service_glob_covers_all_known_units. Without it a
    broken discovery shrinks the sweep to zero cases, and a zero-case
    parametrize collects no tests and reports no failure — masking the very
    defect the sweep exists to catch.

    This half is deliberately ONE-SIDED — it checks only for MISSING paths.
    Asserting `discovered == _EXPECTED_SWEPT_PATHS` would defeat the whole
    point of content-based discovery: a genuinely new unit declaring a cap is
    supposed to be picked up automatically, and an equality assertion would
    turn that success into a test failure demanding the constant be edited.
    The opposite risk — an EXTRA path that is not a unit at all — is covered by
    test_discovery_sweeps_only_systemd_units instead, which constrains extras
    by SHAPE rather than by enumeration.
    """
    discovered = set(discover_units_declaring_a_restart_cap())
    assert discovered, "discovery found no files declaring RestartMaxDelaySec="
    missing = _EXPECTED_SWEPT_PATHS - discovered
    assert not missing, f"discovery is missing known units declaring a cap: {missing}"


def test_discovery_sweeps_only_systemd_units() -> None:
    """The other half of the coverage guard: extras must still be UNITS.

    test_discovery_covers_every_known_unit can only see paths that went
    missing; an extra path is invisible to a `_EXPECTED - discovered` check.
    That asymmetry is not theoretical — this sweep shipped briefly with
    tests/scripts/test_check_fused_memory_unit_parity.py in it, because that
    module embeds systemd units as column-0 fixture strings, and the guard
    written to detect discovery drift could not see the drift.

    Constraining extras by SHAPE keeps the auto-coverage property intact (a new
    .service file is swept the day it lands, with no constant to update) while
    refusing anything that is not a unit definition.  A Python module, a JSON
    fixture, a markdown doc or a shell script matching this directive means
    discovery has escaped its subject, and the failure names the path rather
    than misreporting it as a misconfigured unit several assertions later.

    The test-directory check is by path SEGMENT, not by prefix, so a unit-shaped
    fixture parked at fused-memory/tests/fixtures/whatever.service is caught too
    — the same depth blindness that a prefix-matching `:!tests/` pathspec has,
    which is why _NON_UNIT_PATHSPECS uses the glob form.
    """
    discovered = discover_units_declaring_a_restart_cap()
    strays = [
        p
        for p in discovered
        if not p.endswith(_UNIT_FILE_SUFFIXES) or "/tests/" in f"/{p}"
    ]
    assert not strays, (
        f"discovery swept files that are not systemd unit definitions: {strays}. "
        "Something other than a unit declares RestartMaxDelaySec= — most likely "
        "a test fixture embedding a unit as a triple-quoted string, or a doc "
        "quoting one inside a fence. Such a file must be excluded from the "
        "pathspec, not checked: the invariant reads the LAST occurrence of each "
        "directive file-wide, so on a file holding more than one embedded unit "
        "it splices values from unrelated ones together and reports a nonsense "
        "verdict about a file that is not a unit."
    )


@pytest.mark.parametrize(
    "rel_path",
    discover_units_declaring_a_restart_cap(),
    ids=lambda p: p,
)
def test_every_unit_declaring_a_restart_cap_pairs_it_with_steps(rel_path: str) -> None:
    """A declared RestartMaxDelaySec= must be paired with RestartSteps= to engage.

    This is the fleet-wide application of the shared relational invariant.  It
    replaces per-unit string matching, which is what let this bug live: those
    asserts checked that the cap was WRITTEN, never that it was OBEYED, and
    stayed green the whole time systemd was discarding it at load.
    """
    assert_restart_backoff_effective(REPO_ROOT / rel_path)


def test_factory_init_reference_unit_ships_effective_backoff() -> None:
    """The factory-init reference unit must declare the full backoff triple.

    This file is markdown, so _NON_UNIT_PATHSPECS keeps it OUT of the sweep and
    this guard is its only coverage.  That is not a downgrade: the guard is
    UNCONDITIONAL where the sweep is conditional, i.e. strictly stronger for
    this one file, so nothing was traded away by excluding the category.

    The sweep's contract is "IF you declare a cap, pair it with steps", which
    correctly lets a unit opt out of backoff entirely by declaring no cap.  But
    this file is not a unit; it is the template new supervised units are
    modelled on (its own prose says the scripts/orchestrator-*.service files
    are cp'd verbatim by setup-host.sh and points at orchestrator-reify.service
    as the model).  A reference that simply DROPPED RestartMaxDelaySec= would
    satisfy the conditional invariant while still failing to teach the pairing,
    and every unit minted from it thereafter would need this same fix applied
    by hand.  So require all three directives to be present here, not merely
    consistent with each other.

    restart_directive reads raw text with ^-anchored MULTILINE regexes, so it
    works on the fenced ini block as-is: the directives sit at column 0 inside
    the fence.  That read is only sound while the doc carries ONE unit block —
    the directive lookup is last-occurrence-wins file-wide, so a second fence
    showing a "before"/broken variant would splice the two.  If this reference
    ever grows a second unit block, this guard must gain block scoping rather
    than keep trusting a file-wide read; the same hazard is why markdown is
    excluded from discovery wholesale.
    """
    path = REPO_ROOT / _FACTORY_INIT_REFERENCE
    assert path.exists(), (
        f"{_FACTORY_INIT_REFERENCE} does not exist. New supervised units are "
        "modelled on it, so if it moved this guard must follow it rather than "
        "silently stop checking anything."
    )

    for directive in ("RestartSec", "RestartSteps", "RestartMaxDelaySec"):
        assert restart_directive(path, directive) is not None, (
            f"{_FACTORY_INIT_REFERENCE} does not declare {directive}=. This is "
            "the block new supervised units are copied from, so every unit "
            "minted from it inherits the omission — the whole backoff triple "
            "(floor, steps, cap) has to be present to be taught."
        )

    assert_restart_backoff_effective(path)
