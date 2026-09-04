"""Host-coupled parity guard: the INSTALLED dark-factory-dashboard.service unit.

Deliberately HOST-COUPLED — the opposite contract of every fixture-only
parity suite in this directory. test_check_dashboard_unit_parity.py states
that rule plainly: "All drift-logic tests run against inline fixture strings
and tmp_path directories — NEVER the host's real ~/.config/systemd/user/".
That rule is correct for a general parity CHECKER, but it cannot answer the
one question task 3868 needs answered: has THIS host's installed unit, and
its systemd --user manager, actually been reconciled with the committed
template. A run on a host with the unit installed and reconciled gets a live
green answer about ITS install; a fresh checkout or CI runner with no
installed unit, no user D-Bus session, or a pre-254 systemd that has never
heard of RestartSteps= degrades to a skip (see the guards below), never a
false failure.

THREE-WAY DISTINCTION — this module must not be confused with either of the
two existing dashboard test modules, which answer different questions:
  - test_check_dashboard_unit_parity.py tests scripts/check_dashboard_unit_
    parity.py's DRIFT LOGIC against inline fixtures and tmp_path directories,
    never the real host — it is deliberately NOT host-coupled (see its own
    docstring) and would stay green or red independent of whether THIS
    host's installed unit is reconciled.
  - test_dashboard_service_template.py compares the COMMITTED template
    (scripts/dashboard.service.template) against the COMMITTED hardcoded
    copy (dashboard/dark-factory-dashboard.service) — both in-repo files,
    no systemd runtime involved at all.
  - THIS module is the only one of the three that reads the INSTALLED unit
    under ~/.config/systemd/user/ and the live systemd --user manager.

Sibling modules: test_pump_web_ui_installed_unit_parity.py (task 3763),
test_know_live_installed_unit_parity.py (task 3642), and
test_fused_memory_installed_unit_parity.py (task 3868, this task's step-1)
do the identical job for their respective units. This module is structured
on the pump-web-ui one (the leaner sibling — no local helpers) and is
near-identical to test_fused_memory_installed_unit_parity.py, differing only
in UNIT_BASENAME, COMMITTED_UNIT_PATH and _REMEDIATION (see below).

Scope is deliberately narrow: ONE invariant of exactly one unit — that its
RestartMaxDelaySec=/RestartSteps= backoff pair actually engages — checked at
two layers, not a byte-parity sweep across the fleet:

  1. FILE layer — the installed unit FILE on disk.
  2. MANAGER layer — systemd --user's LOADED view of the unit, via
     `systemctl --user show`. Not redundant with the file layer: hand-
     editing a corrected unit into place without `daemon-reload` leaves the
     manager holding the stale unit, so a file-only check would bless a
     host whose backoff is still inert.

BOTH directives are pinned at BOTH layers, deliberately. The pair is the
invariant; either directive alone is inert, so pinning only one leaves a
hole. The file layer's shared helper is CONDITIONAL — it returns early when
no cap is declared — which means a stale re-install or hand drop-in that
dropped BOTH lines would pass it vacuously; a manager layer reading only
RestartSteps would in turn pass a drop-in that pinned steps with no cap.
Each layer covers the other's blind spot only if both read both.

The expected RestartSteps value is DERIVED from the committed template at
assertion time, never hard-coded, for the same reason as the sibling
modules: a hard-coded '4' turns any future legitimate template edit into a
RED test on a fully reconciled host, with a misdirecting failure message.

DEVIATION from the orchestrator-*.service siblings (shared with this task's
fused-memory module): COMMITTED_UNIT_PATH points at a differently named
`.template` file, not at `scripts/<UNIT_BASENAME>`. dark-factory-
dashboard.service is rendered by `sed` placeholder substitution from
scripts/dashboard.service.template (scripts/setup-host.sh:790-794), not
`cp`-ed from an identically-named committed file. This needs no
__REPO_ROOT__/__UV_PATH__ placeholder normalisation before use: the only
scalar this module reads from that file is the bare `RestartSteps=4` line,
which contains neither placeholder.

Byte-parity against the committed template is deliberately NOT the invariant
pinned here. The installed unit is KNOWN to diverge from the template on
purpose: both declare Environment=DASHBOARD_KNOWN_PROJECT_ROOTS=, but the
template pins it to __REPO_ROOT__ alone while the installed copy carries
nine project roots (dark-factory, reify, autopilot-video, autotrade,
know-live, solar-challenge, mission-control, solar-challenge-platform,
pump-web-ui) — scripts/check_dashboard_unit_parity.py's DIVERGENCE_ALLOWLIST
records this exact variable as a deliberate value-divergence. A byte-parity
assertion here would not just be fragile to a future comment-only template
edit; it would actively demand DESTROYING that host-specific data, since the
only way to satisfy byte-parity would be to strip the eight extra roots back
out of the installed unit.

UPDATED 2026-08-27 (task 4793). This paragraph used to continue: "That
destructive path — a `sed` re-render or a setup-host.sh re-run — is exactly
what this module's _REMEDIATION warns against." The re-render is no longer
destructive. setup-host.sh section 8 now installs this unit through
scripts/render_dashboard_unit.py, which reads the ALREADY-INSTALLED unit's
host-local Environment= values and puts them back
(render_dashboard_unit.HOST_LOCAL_ENVIRONMENT), so a re-run preserves the nine
roots instead of collapsing them to one. _REMEDIATION below is rewritten
accordingly. What is unchanged is the argument in the paragraph above: BYTE
parity is still the wrong invariant for this module, because the divergence is
deliberate and permanent, not because re-installing is unsafe.
Also deliberately NOT asserted: ActiveState — liveness is the watchdog's
job, and pinning it here would make this suite fail during any legitimate
restart window.

DELIBERATELY OUT OF SCOPE, recorded here as a measured observation so a
green guard is not misread as "this unit is fully reconciled": the installed
dark-factory-dashboard.service is stale on THREE FURTHER axes beyond
RestartSteps, none of which this module asserts. Measured 2026-08-19 by
diffing the sed-rendered committed template against the installed unit:
  - `SuccessExitStatus=143` is ABSENT from the installed unit entirely
    (committed by d50235fa66).
  - `Environment=DASHBOARD_PROJECT_ROOT=` is ABSENT from the installed unit
    entirely (committed by task 3572's 226674af75); only the differently
    named DASHBOARD_KNOWN_PROJECT_ROOTS is present.
  - `--timeout-keep-alive 5` is installed where the committed template says
    `--timeout-keep-alive 6` (task 4087's 26cf6c6751).
Filed as follow-up task 4445 ("Installed dark-factory-dashboard.service is
stale on three axes beyond RestartSteps"), which this task's own plan
analysis also records as out of scope. Fixing those is a different
invariant with its own review surface — this module pins ONE invariant of
this unit at two layers, exactly as its siblings do for theirs.

PRE-FIX BASELINE, recorded so a future reader knows this module arrived RED
and was not written to match an already-green host. Measured by the
implementer 2026-08-19 against systemd 255.4-1ubuntu8.17:

  - The installed unit declared Restart=on-failure, RestartSec=5 and
    RestartMaxDelaySec=60 with NO RestartSteps= line at all (`grep -n
    Restart ~/.config/systemd/user/dark-factory-dashboard.service`).
  - `systemd-analyze --user verify` printed on stderr, at exit 0:
    "dark-factory-dashboard.service: Service has RestartMaxDelaySec= but no
    RestartSteps= setting. Ignoring."
  - `systemctl --user show dark-factory-dashboard.service -p RestartSteps -p
    RestartMaxDelayUSec` reported RestartSteps=0 (systemd's zero default —
    the manager had never loaded a RestartSteps= value for this unit) and
    RestartMaxDelayUSec=1min (the cap IS parsed and shown even though it is
    discarded at runtime, which is why RestartSteps rather than
    RestartMaxDelayUSec is the only discriminating manager property).
  - The committed scripts/dashboard.service.template already declared
    RestartSteps=4 (commit 7e1070f360) — restart_directive() against it
    returns '4' — so the defect was purely host-side install drift, not a
    repo defect.

POST-FIX CONFIRMATION, measured 2026-08-19 after a manual surgical edit —
inserting the single line `RestartSteps=4` immediately after `RestartSec=5`
in the installed unit (matching the committed template's directive
ordering: Restart=on-failure / RestartSec=5 / RestartSteps=4 /
RestartMaxDelaySec=60), changing nothing else — followed by `systemctl
--user daemon-reload`:

  - `diff` of the installed unit against its pre-fix backup is exactly
    `36a37 > RestartSteps=4` — one line appended, nothing removed or
    reordered. `grep -n Restart` now shows RestartSteps=4 at line 37,
    between RestartSec=5 (line 36) and RestartMaxDelaySec=60 (line 38).
  - `Environment=DASHBOARD_KNOWN_PROJECT_ROOTS=` still lists all nine
    project roots (dark-factory, reify, autopilot-video, autotrade,
    know-live, solar-challenge, mission-control, solar-challenge-platform,
    pump-web-ui), and the three known out-of-scope staleness deltas
    (missing SuccessExitStatus=143, missing DASHBOARD_PROJECT_ROOT=, and
    --timeout-keep-alive 5 vs the committed 6) were left exactly as they
    were — not opportunistically fixed; see follow-up task 4445.
  - `systemctl --user show dark-factory-dashboard.service -p RestartSteps
    -p RestartMaxDelayUSec` now reports RestartSteps=4 and
    RestartMaxDelayUSec=1min (previously RestartSteps=0).
  - `systemd-analyze --user verify` no longer prints the "Service has
    RestartMaxDelaySec= but no RestartSteps= setting. Ignoring." line for
    either dark-factory unit (the remaining jcodemunch-serve.service "No
    such file or directory" line is Reify-owned, out of scope — see the
    plan analysis).
  - dark-factory-dashboard.service remained `active` throughout; no restart
    was performed.
  - Both tests in this module pass under `-m integration`; run together
    with test_fused_memory_installed_unit_parity.py, all four tests across
    both new modules pass (`-m integration`: 4 passed).

Marking: both host-touching tests carry `@pytest.mark.integration`, which is
registered and DESELECTED by the ROOT pyproject.toml's default addopts
(`-m 'not smoke and not integration and not warm_lane_bash'`). A bare
`pytest` run therefore reports success while asserting nothing about the
host — run these explicitly:

    uv run --project shared pytest \\
        tests/scripts/test_dashboard_installed_unit_parity.py -m integration

(`--project shared` is required: tests/scripts/ is not a workspace member,
and a bare `uv run pytest` resolves against the root project, which declares
no pytest. This mirrors tests/scripts/orchestrator.yaml's test_command.)
That deselection is the same mechanism keeping a legitimate operator action
against ~/.config/systemd/user (a manual `systemctl --user stop`, a hand
drop-in, a fleet redeploy in progress) from turning every concurrent task
worktree's default test run red. The marker is applied per-FUNCTION rather
than as a module-level `pytestmark`, matching the sibling's convention, so
that any future portable test added here keeps running off-host.

Importable `from systemd_unit_invariants import ...` because
tests/scripts/conftest.py puts this directory on sys.path — pytest's
--import-mode=importlib (pyproject.toml addopts) deliberately does not do
that on its own.
"""

import pathlib
import shutil

import pytest
from systemd_unit_invariants import (
    INSTALLED_UNIT_DIR,
    SYSTEMCTL_SKIP_REASON,
    assert_restart_backoff_effective,
    require_installed_unit,
    restart_directive,
    systemctl_user_show,
)

UNIT_BASENAME = "dark-factory-dashboard.service"
INSTALLED_UNIT_PATH = INSTALLED_UNIT_DIR / UNIT_BASENAME

# The committed template the installed copy is rendered FROM by
# scripts/render_dashboard_unit.py (`__REPO_ROOT__` / `__UV_PATH__` placeholder
# substitution, invoked by setup-host.sh section 8) — NOT a plain
# `cp`, which is why this is a differently-named `.template` file rather
# than `parents[2]/"scripts"/UNIT_BASENAME` as in the orchestrator-*.service
# sibling modules. See the module docstring's DEVIATION paragraph for why no
# __REPO_ROOT__/__UV_PATH__ placeholder normalisation is needed before
# reading RestartSteps= from it. Read at assertion time rather than
# hard-coding the expected value: a hard-coded '4' turns any future edit of
# the template into a RED test on a FULLY RECONCILED host, whose failure
# message would then misdirect the operator by claiming the template says 4
# when it does not — and whose printed remediation would already have been
# done. parents[2] because this file is <repo>/tests/scripts/<name>.py.
COMMITTED_UNIT_PATH = (
    pathlib.Path(__file__).parents[2] / "scripts" / "dashboard.service.template"
)

# What `systemctl show -p RestartMaxDelayUSec` reports for a unit that
# declares NO cap — measured on this host (systemd 255.4) across every
# installed --user unit with no RestartMaxDelaySec= line: uniformly
# `infinity`, never `0` and never absent. So "the property is present and
# non-zero" is NOT evidence a cap was loaded; only "present and not this
# sentinel" is.
_NO_CAP_SENTINEL = "infinity"

# Remediation for either layer below. TWO routes now, where there used to be
# one — task 4793 removed the hazard that made the surgical edit the only safe
# option. setup-host.sh section 8 installs this unit through
# scripts/render_dashboard_unit.py, which reads the installed unit's host-local
# Environment= values and puts them back, so a re-run no longer strips this
# host's extra DASHBOARD_KNOWN_PROJECT_ROOTS entries. It is the SANCTIONED path
# and it is what check_dashboard_unit_parity.py's own drift report tells the
# operator to run; it also propagates every other committed fix, which for this
# unit means the three further stale axes recorded in the module docstring
# (task 4445) — a larger change than the single line this module asserts, which
# is the only reason the surgical route is still offered rather than retired.
#
# scripts/check_dashboard_unit_parity.py still has NO --fix flag by design;
# re-running the installer IS the propagation path.
#
# The daemon-reload is not optional on the surgical route — without it the
# manager keeps serving the stale unit and only the FILE-layer test goes green.
# (setup-host.sh runs its own.) Both layers really do interpolate this: the
# FILE layer delegates its assertion to the shared helper (whose generic
# message points at an unrelated template), so it re-raises with this appended
# rather than leaving the operator the one message that never mentions
# re-installing THIS unit.
_REMEDIATION = (
    "To reconcile, either: (a) re-run `scripts/setup-host.sh` — the sanctioned "
    "path, which since task 4793 PRESERVES this host's local Environment= "
    "values (DASHBOARD_KNOWN_PROJECT_ROOTS) instead of stripping them, and "
    "which re-renders the whole unit, so it also picks up the other committed "
    "fixes this module does not assert; or (b) surgically, changing nothing "
    "else: insert `RestartSteps=4` immediately after `RestartSec=5` in "
    f"{INSTALLED_UNIT_PATH}, then `systemctl --user daemon-reload`."
)


# ---------------------------------------------------------------------------
# FILE layer — the installed unit on disk
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_installed_unit_file_restart_backoff_effective() -> None:
    """The INSTALLED unit file's RestartMaxDelaySec= must be paired with RestartSteps=.

    Arrived RED (see the module docstring's PRE-FIX BASELINE): the installed
    copy declared RestartMaxDelaySec=60 with no RestartSteps= line, which
    systemd silently drops — so the advertised 5s->60s escalating backoff
    never ran. Reuses systemd_unit_invariants.assert_restart_backoff_
    effective, the same RELATIONAL invariant test_systemd_restart_backoff.py
    already applies to the COMMITTED template — this is that check applied to
    the INSTALLED file instead. Its last-occurrence-wins directive reader is
    correct here because a <unit>.d/ drop-in merges by appending.

    The shared helper's own failure message is written for the COMMITTED
    fleet and points at scripts/jcodemunch-watcher.service.template as the
    worked example — accurate, but it never mentions re-installing THIS unit
    or running daemon-reload, which is the only thing that fixes a drifted
    HOST copy. Since this is the layer an operator hits first, the assertion
    is re-raised with _REMEDIATION appended rather than delegating the
    operator-facing guidance to a message that cannot know about the host.
    """
    path = require_installed_unit(UNIT_BASENAME)
    try:
        assert_restart_backoff_effective(path)
    except AssertionError as exc:
        raise AssertionError(f"{exc}\n{_REMEDIATION}") from exc


# ---------------------------------------------------------------------------
# MANAGER layer — systemd --user's LOADED view of the unit
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("systemctl") is None, reason=SYSTEMCTL_SKIP_REASON)
def test_installed_unit_manager_restart_steps_effective() -> None:
    """systemd --user's LOADED view must carry BOTH halves of the backoff pair.

    Not redundant with the file-layer check above: hand-editing a corrected
    unit into place without `systemctl --user daemon-reload` leaves the
    MANAGER holding the old unit, so a file-only check would bless a host
    whose backoff is still inert. This layer is the only thing that proves a
    reload actually happened. Arrived RED reporting RestartSteps=0 —
    systemd's zero-value default, confirming the manager had never loaded a
    RestartSteps= value for this unit at all.

    Pins the PAIR, not just RestartSteps, because the invariant this module
    exists to protect is the 5s->60s curve, which takes both directives. The
    file layer's helper is CONDITIONAL — it returns early when no cap is
    declared — so a stale re-install or hand drop-in that dropped BOTH
    directives would leave the file layer vacuously green, and a manager
    layer reading only RestartSteps would pass a drop-in that pinned steps
    with no cap. Neither half alone closes that.

    Two measured systemd 255.4 facts drive the exact shape below, both
    non-obvious enough that getting them wrong yields a guard that silently
    checks nothing:

      - The queryable property is `RestartMaxDelayUSec`, NOT the directive
        spelling `RestartMaxDelaySec`. Verified: `systemctl --user show <unit>
        -p RestartMaxDelaySec` exits 0 with EMPTY stdout — the
        unsupported-property shape systemctl_user_show deliberately reports as
        ABSENT — so asserting on the directive spelling would skip on every
        host forever.
      - A unit declaring NO cap reports `RestartMaxDelayUSec=infinity`, not
        `0` and not absent (verified across every installed --user unit on
        this host lacking a RestartMaxDelaySec= line). So the assertion is
        "not the infinity sentinel", not "non-zero" — the latter is vacuously
        true exactly when the cap is missing.

    The manager renders the loaded cap as a duration (`RestartMaxDelaySec=60`
    in the file surfaces here as `1min`), so this asserts a cap IS loaded
    rather than pinning a rendering — reproducing systemd's duration
    formatter to predict the exact string would be a second, more brittle
    guard for no extra signal. The file layer already pins the declared value.
    """
    require_installed_unit(UNIT_BASENAME)
    properties = ("RestartSteps", "RestartMaxDelayUSec")
    shown = systemctl_user_show(UNIT_BASENAME, *properties)
    if shown is None:
        pytest.skip(
            "systemctl --user show could not be queried (no user D-Bus "
            "session in this runner)"
        )
    missing = [prop for prop in properties if prop not in shown]
    if missing:
        pytest.skip(
            f"systemctl --user show {UNIT_BASENAME} returned no "
            f"{', '.join(missing)} property at all (empty stdout, not merely "
            "an empty value) — most likely this host's systemd predates 254, "
            "which introduced RestartSteps=/RestartMaxDelaySec=. Verified: "
            "`systemctl --user show <unit> -p <unsupported-property>` exits "
            "0 with empty stdout, so there is nothing for this guard to "
            "assert against an unsupported property."
        )

    # Derived, never hard-coded: see COMMITTED_UNIT_PATH's comment. A literal
    # here would go RED on a correctly-reconciled host the day the template
    # changes, and would print a remediation that had already been performed.
    assert COMMITTED_UNIT_PATH.is_file(), (
        f"{COMMITTED_UNIT_PATH} is missing, so this guard cannot derive the "
        "RestartSteps= value the host is supposed to have. That path is "
        "in-repo, not host state — a missing committed template is a repo "
        "defect, not an environment fact, so this fails rather than skips."
    )
    expected_steps = restart_directive(COMMITTED_UNIT_PATH, "RestartSteps")
    assert expected_steps is not None, (
        f"the committed {COMMITTED_UNIT_PATH} declares no RestartSteps= line, "
        "so its own RestartMaxDelaySec= is inert at the source and there is "
        "nothing coherent for this host guard to require. Fix the template "
        "first — tests/scripts/test_systemd_restart_backoff.py is the gate "
        "that owns that invariant for committed units."
    )

    steps = shown["RestartSteps"]
    assert steps == expected_steps, (
        f"systemctl --user show {UNIT_BASENAME} -p RestartSteps reports "
        f"RestartSteps={steps!r}, but the committed {COMMITTED_UNIT_PATH} "
        f"declares RestartSteps={expected_steps}. A value of '0' is systemd's "
        "default and means the manager never loaded that line, so the cap is "
        "being ignored and every restart waits a flat RestartSec. "
        f"{_REMEDIATION}"
    )

    cap = shown["RestartMaxDelayUSec"]
    assert cap != _NO_CAP_SENTINEL, (
        f"systemctl --user show {UNIT_BASENAME} -p RestartMaxDelayUSec "
        f"reports {cap!r}, systemd's no-cap default. RestartSteps="
        f"{steps} is loaded but has no ceiling to interpolate TOWARD, so the "
        "escalating backoff this unit advertises does not run as described. "
        "The manager is serving a unit with no RestartMaxDelaySec= — either a "
        f"stale copy or a <unit>.d/ drop-in overriding it. {_REMEDIATION}"
    )
