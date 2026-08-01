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
import importlib
import pathlib
import subprocess

import pytest

import systemd_unit_invariants
from systemd_unit_invariants import (
    assert_restart_backoff_effective,
    restart_directive,
)

REPO_ROOT = pathlib.Path(__file__).parents[2]

# Files that declare the cap but are deliberately NOT fixed.
#
# plans/ holds design docs and PRDs for past work (CLAUDE.md "Repo Map"), and
# afk-C1-systemd.md is an as-built record of what was deployed at the time.  It
# has already diverged from the live fleet in three visible ways — it still
# shows `Requires=fused-memory.service`, which the real units deliberately
# reject and test_orchestrator_service_files.py explicitly asserts is ABSENT,
# plus an obsolete `--config orchestrator/config.yaml` path and no `--frozen`.
# It is manifestly not a copy-source (unlike
# skills/factory-init/references/supervised-unit.md, which IS fixed and is
# guarded unconditionally below).  Editing one directive inside it would
# falsify the record without making any unit correct.
#
# The exclusion is named and tested rather than silent: see
# test_historical_record_exclusions_are_live.
_HISTORICAL_RECORD_EXCLUSIONS = ("plans/afk-C1-systemd.md",)

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
        "scripts/orchestrator-reify.service",
        "scripts/orchestrator-solar-challenge-platform.service",
        "skills/factory-init/references/supervised-unit.md",
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
    """
    proc = subprocess.run(
        ["git", "grep", "-lE", "^RestartMaxDelaySec=", "--", "."],
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
    paths = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    return sorted(p for p in paths if p not in _HISTORICAL_RECORD_EXCLUSIONS)


def test_discovery_covers_every_known_unit() -> None:
    """Coverage guard: discovery must be non-empty and cover every known unit.

    Mirrors test_orchestrator_service_glob_covers_all_known_units. Without it a
    broken discovery shrinks the sweep to zero cases, and a zero-case
    parametrize collects no tests and reports no failure — masking the very
    defect the sweep exists to catch.
    """
    discovered = set(discover_units_declaring_a_restart_cap())
    assert discovered, "discovery found no files declaring RestartMaxDelaySec="
    missing = _EXPECTED_SWEPT_PATHS - discovered
    assert not missing, f"discovery is missing known units declaring a cap: {missing}"


def test_historical_record_exclusions_are_live() -> None:
    """Every excluded path must still exist AND still declare the cap.

    A stale exclusion is a silent change of scope in either direction: if the
    file were renamed or deleted, the constant would quietly protect nothing;
    if the directive were removed from it, the exclusion would no longer be
    describing a real conflict and should be dropped.  Either way the sweep's
    coverage would have drifted from what the constant claims, so assert the
    exclusion is still load-bearing rather than trusting it forever.
    """
    for rel in _HISTORICAL_RECORD_EXCLUSIONS:
        path = REPO_ROOT / rel
        assert path.exists(), (
            f"excluded path {rel} no longer exists; the exclusion in "
            "_HISTORICAL_RECORD_EXCLUSIONS is stale and should be removed."
        )
        assert restart_directive(path, "RestartMaxDelaySec") is not None, (
            f"excluded path {rel} no longer declares RestartMaxDelaySec=, so "
            "the sweep would not have picked it up anyway; the exclusion in "
            "_HISTORICAL_RECORD_EXCLUSIONS is stale and should be removed."
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

    This guard is UNCONDITIONAL where the sweep above is conditional, and
    deliberately so — it is a strictly stronger invariant for this one file.

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
    the fence.
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


def test_restart_backoff_helper_is_shared_not_duplicated() -> None:
    """The dashboard suite must USE the shared helper, not its own fork of it.

    A "both modules define a function of this name" test would pass happily
    against two independently-drifting copies, which is precisely the failure
    this task exists to repair: per-unit string asserts forked from the real
    invariant and stayed green for months while the directive they guarded was
    inert.  So the check is object IDENTITY — a future author who re-inlines a
    private copy into either module fails here immediately, at the moment of
    forking, rather than years later when one copy has quietly stopped
    catching the defect.
    """
    for name in ("restart_directive", "assert_restart_backoff_effective"):
        assert callable(getattr(systemd_unit_invariants, name, None)), (
            f"systemd_unit_invariants must export a callable {name!r}; it is "
            "the single shared implementation of the restart-backoff invariant."
        )

    dashboard = importlib.import_module("test_dashboard_service_template")

    assert (
        dashboard._assert_restart_backoff_effective
        is systemd_unit_invariants.assert_restart_backoff_effective
    ), (
        "test_dashboard_service_template._assert_restart_backoff_effective is "
        "not the shared systemd_unit_invariants.assert_restart_backoff_effective. "
        "The invariant has been forked into a private copy; import the shared "
        "one instead so both modules cannot drift apart."
    )
    assert (
        dashboard._restart_directive is systemd_unit_invariants.restart_directive
    ), (
        "test_dashboard_service_template._restart_directive is not the shared "
        "systemd_unit_invariants.restart_directive. The directive reader "
        "encodes measured systemd 255.4 last-wins scalar semantics; a private "
        "copy of it will drift out of agreement with the assertion that uses it."
    )
