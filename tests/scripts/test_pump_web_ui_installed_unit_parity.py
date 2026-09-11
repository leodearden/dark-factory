"""Host-coupled parity guard: the INSTALLED orchestrator-pump-web-ui.service unit.

Deliberately HOST-COUPLED — the opposite contract of every fixture-only
parity suite in this directory. test_check_dashboard_unit_parity.py states
that rule plainly: "All drift-logic tests run against inline fixture strings
and tmp_path directories — NEVER the host's real ~/.config/systemd/user/".
That rule is correct for a general parity CHECKER, but it cannot answer the
one question task 3763 needs answered: has THIS host's installed unit, and
its systemd --user manager, actually been reconciled with the committed
template. scripts/check_orchestrator_unit_parity.py (a registry-driven
installed-vs-committed gate) has since LANDED on main — task 3424, picked up
here when this branch was rebased onto it — and setup-host.sh now runs it as
a pre-install gate. That does not make this module redundant, and the two
answer different questions: the checker compares two FILES for symmetric
equality, so it is silent about the MANAGER layer below (an install without a
daemon-reload passes it), and it is a script an operator must run rather than
an assertion the suite enforces. What it does retire is the older claim here
that no portable checker existed to exercise against a fixture — one now
does, and drift-logic coverage of it belongs in its own fixture-based suite
(tests/scripts/test_check_orchestrator_unit_parity.py), not here. A run on a host
with the unit installed and reconciled gets a live green answer about ITS
install; a fresh checkout or CI runner with no installed unit, no user D-Bus
session, or a pre-254 systemd that has never heard of RestartSteps= degrades
to a skip (see the guards below), never a false failure.

Sibling module: test_know_live_installed_unit_parity.py (task 3642) does the
identical job for orchestrator-know-live.service, and this module is
structured on it.

Scope is deliberately narrow: ONE invariant of exactly one unit — that its
RestartMaxDelaySec=/RestartSteps= backoff pair actually engages — checked at
two layers, not a byte-parity sweep across the fleet:

  1. FILE layer — the installed unit FILE on disk.
  2. MANAGER layer — systemd --user's LOADED view of the unit, via
     `systemctl --user show`. Not redundant with the file layer: `cp`-ing a
     corrected unit into place without `daemon-reload` leaves the manager
     holding the stale unit, so a file-only check would bless a host whose
     backoff is still inert.

BOTH directives are pinned at BOTH layers, deliberately. The pair is the
invariant; either directive alone is inert, so pinning only one leaves a
hole. The file layer's shared helper is CONDITIONAL — it returns early when
no cap is declared — which means a stale re-install or hand drop-in that
dropped BOTH lines would pass it vacuously; a manager layer reading only
RestartSteps would in turn pass a drop-in that pinned steps with no cap.
Each layer covers the other's blind spot only if both read both.

The expected RestartSteps value is DERIVED from the committed template at
assertion time, never hard-coded. Hard-coding it means the day the template
legitimately changes, a fully reconciled host goes RED and the failure
message actively misdirects — asserting the template says something it no
longer says, and prescribing an install+reload that has already been done.

Byte-parity against the committed template is deliberately NOT the invariant
pinned here, even though task 3763's step-4 does produce a byte-identical
file: any future comment-only edit to the committed template would then turn
this host red until an operator re-ran setup-host.sh, converting a
documentation change into a spurious host failure. Fleet-wide byte-parity is
task 3424's job. Also deliberately NOT asserted: ActiveState — liveness is
the watchdog's job (scripts/orchestrator-watchdog.py), and pinning it here
would make this suite fail during any legitimate restart window.

Narrower than the sibling by one property, for a MEASURED reason: know-live
also guards its ExecStart= `--config` basename, because know-live had a
SECOND, opposite-direction drift there (installed ahead of committed). A
full `diff -u` of installed vs committed pump-web-ui (architect, 2026-08-08)
found RestartSteps=4 to be the ONLY non-comment delta in the entire file —
ExecStart= is already byte-identical between the two — so there is no such
drift here and no parser for this module to own. Consequently this module
defines NO helpers of its own at all, and needs no portable PARSER-layer
section: every helper it uses lives in systemd_unit_invariants.py and is
negative-case-owned elsewhere. Task 3763 lifted three of them there rather
than copying them out of the sibling — `systemctl_user_show` (whose
four-case parametrized guard stayed put in the sibling, mirroring how
test_dashboard_service_template.py owns assert_restart_backoff_effective's),
plus `INSTALLED_UNIT_DIR`, `require_installed_unit` and
`SYSTEMCTL_SKIP_REASON` when this module became their second consumer.
INSTALLED_UNIT_DIR especially: it mirrors a path written in another
language's file (scripts/setup-host.sh), and a mis-mirrored copy does not
fail loudly — it degrades to require_installed_unit() skipping, a guard that
silently checks nothing, which is precisely what the mirroring exists to
prevent. One copy can be kept right; N copies drift.

PRE-FIX BASELINE, recorded so a future reader knows this module arrived RED
and was not written to match an already-green host. Measured by the
architect 2026-08-08 against systemd 255.4-1ubuntu8.16 (well past 254, which
introduced RestartSteps=/RestartMaxDelaySec=, so this was a real defect and
not an unsupported-directive artifact), and independently re-measured by the
implementer immediately before the fix:

  - The installed unit declared Restart=on-failure, RestartSec=10 and
    RestartMaxDelaySec=60 with NO RestartSteps= line at all.
  - `journalctl --user -u orchestrator-pump-web-ui.service` carried repeated
    "orchestrator-pump-web-ui.service: Service has RestartMaxDelaySec= but
    no RestartSteps= setting. Ignoring." warnings — last seen Aug 08
    01:03:00 (architect) and still recurring at Aug 08 06:09:35
    (implementer, re-measured).
  - `systemd-analyze --user verify` printed that same warning on stderr at
    exit 0.
  - `systemctl --user show orchestrator-pump-web-ui.service -p RestartSteps`
    reported `RestartSteps=0` — systemd's zero default, i.e. the manager had
    never loaded a RestartSteps= value for this unit.

So the unit's advertised 10s->60s escalating backoff was inert. Task 3763's
step-4 installed the committed template and ran `daemon-reload`; both tests
below pin the reconciled state so a stale re-install regresses loudly rather
than silently.

Marking: both host-touching tests carry `@pytest.mark.integration`, which is
registered and DESELECTED by the ROOT pyproject.toml's default addopts
(`-m 'not smoke and not integration and not warm_lane_bash'`). A bare
`pytest` run therefore reports success while asserting nothing about the
host — run these explicitly:

    uv run --project shared pytest \\
        tests/scripts/test_pump_web_ui_installed_unit_parity.py -m integration

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

UNIT_BASENAME = "orchestrator-pump-web-ui.service"
INSTALLED_UNIT_PATH = INSTALLED_UNIT_DIR / UNIT_BASENAME

# The committed template the installed copy is propagated FROM. Read at
# assertion time rather than hard-coding the expected RestartSteps value:
# a hard-coded '4' turns any future edit of the template (say to
# RestartSteps=5, correctly re-installed) into a RED test on a FULLY
# RECONCILED host, whose failure message would then misdirect the operator
# by claiming the template says 4 when it does not — and whose printed
# remediation (`install` + `daemon-reload`) would already have been done.
# parents[2] because this file is <repo>/tests/scripts/<name>.py.
COMMITTED_UNIT_PATH = pathlib.Path(__file__).parents[2] / "scripts" / UNIT_BASENAME

# What `systemctl show -p RestartMaxDelayUSec` reports for a unit that
# declares NO cap — measured on this host (systemd 255.4) across every
# installed --user unit with no RestartMaxDelaySec= line: uniformly
# `infinity`, never `0` and never absent. So "the property is present and
# non-zero" is NOT evidence a cap was loaded; only "present and not this
# sentinel" is.
_NO_CAP_SENTINEL = "infinity"

# Remediation for either layer below: both failures have the same fix, and
# the second command is not optional — without the reload the manager keeps
# serving the stale unit and only the FILE-layer test goes green. Both
# layers really do interpolate it: the FILE layer delegates its assertion to
# the shared helper (whose generic message points at an unrelated template),
# so it re-raises with this appended rather than leaving the operator the
# one message that never mentions re-installing THIS unit.
_REMEDIATION = (
    f"To reconcile: `install -m 0644 scripts/{UNIT_BASENAME} "
    f"{INSTALLED_UNIT_PATH}` then `systemctl --user daemon-reload`."
)


# ---------------------------------------------------------------------------
# FILE layer — the installed unit on disk
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_installed_unit_file_restart_backoff_effective() -> None:
    """The INSTALLED unit file's RestartMaxDelaySec= must be paired with RestartSteps=.

    Arrived RED (see the module docstring's PRE-FIX BASELINE): the installed
    copy declared RestartMaxDelaySec=60 with no RestartSteps= line, which
    systemd silently drops — so the advertised 10s->60s escalating backoff
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

    Not redundant with the file-layer check above: `cp`-ing a corrected unit
    into place without `systemctl --user daemon-reload` leaves the MANAGER
    holding the old unit, so a file-only check would bless a host whose
    backoff is still inert. This layer is the only thing that proves a reload
    actually happened. Arrived RED reporting RestartSteps=0 — systemd's
    zero-value default, confirming the manager had never loaded a
    RestartSteps= value for this unit at all.

    Pins the PAIR, not just RestartSteps, because the invariant this module
    exists to protect is the 10s->60s curve, which takes both directives. The
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
