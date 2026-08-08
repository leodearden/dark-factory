"""Host-coupled parity guard: the INSTALLED orchestrator-pump-web-ui.service unit.

Deliberately HOST-COUPLED — the opposite contract of every fixture-only
parity suite in this directory. test_check_dashboard_unit_parity.py states
that rule plainly: "All drift-logic tests run against inline fixture strings
and tmp_path directories — NEVER the host's real ~/.config/systemd/user/".
That rule is correct for a general parity CHECKER, but it cannot answer the
one question task 3763 needs answered: has THIS host's installed unit, and
its systemd --user manager, actually been reconciled with the committed
template. scripts/check_orchestrator_unit_parity.py (a registry-driven
installed-vs-committed gate) is confirmed absent from main — its commits
live only on the unmerged task/3424 branch — so there is no portable checker
this module could exercise against a fixture instead. A future run on a host
with the unit installed and reconciled gets a live green answer about ITS
install; a fresh checkout or CI runner with no installed unit, no user D-Bus
session, or a pre-254 systemd that has never heard of RestartSteps= degrades
to a skip (see the guards below), never a false failure.

Sibling module: test_know_live_installed_unit_parity.py (task 3642) does the
identical job for orchestrator-know-live.service, and this module is
structured on it.

Scope is deliberately narrow: ONE property of exactly one unit, at two
layers — not a byte-parity sweep across the fleet:

  1. FILE layer — the installed unit FILE on disk.
  2. MANAGER layer — systemd --user's LOADED view of the unit, via
     `systemctl --user show`. Not redundant with the file layer: `cp`-ing a
     corrected unit into place without `daemon-reload` leaves the manager
     holding the stale unit, so a file-only check would bless a host whose
     backoff is still inert.

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
defines no helpers of its own beyond the skip guard, and needs no portable
PARSER-layer section: the one non-trivial helper it reuses,
`systemctl_user_show`, is negative-case-owned by the sibling module (task
3763 lifted it into systemd_unit_invariants.py when this module became its
second consumer; its four-case parametrized guard stayed put).

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
    assert_restart_backoff_effective,
    systemctl_user_show,
)

# Mirrors scripts/setup-host.sh:114 (`UNIT_DIR="$HOME/.config/systemd/user"`)
# exactly. Deliberately NOT XDG_CONFIG_HOME-aware: the installer itself does
# not honour that variable, so making this guard honour it would only ever
# point the guard at a directory setup-host.sh never writes to — the unit
# would not be found there, _require_installed_unit would skip, and the
# guard would silently check nothing while the real installed unit (at
# $HOME/.config/systemd/user, unconditionally) drifted. Mirroring the
# installer's actual, non-configurable path is the whole point of a parity
# guard.
UNIT_DIR = pathlib.Path.home() / ".config" / "systemd" / "user"

UNIT_BASENAME = "orchestrator-pump-web-ui.service"
INSTALLED_UNIT_PATH = UNIT_DIR / UNIT_BASENAME

# Remediation for either layer below: both failures have the same fix, and
# the second command is not optional — without the reload the manager keeps
# serving the stale unit and only the FILE-layer test goes green.
_REMEDIATION = (
    f"To reconcile: `install -m 0644 scripts/{UNIT_BASENAME} "
    f"{INSTALLED_UNIT_PATH}` then `systemctl --user daemon-reload`."
)

_SYSTEMCTL_SKIP_REASON = (
    "systemctl is not installed; this test requires a live systemd --user "
    "manager and has no fixture-based fallback (see module docstring)"
)


def _require_installed_unit() -> pathlib.Path:
    """Return INSTALLED_UNIT_PATH, or skip if it does not exist on this host.

    A fresh checkout or a CI runner has no ~/.config/systemd/user at all —
    that is an environment fact, not a defect in the unit, so this skips
    rather than fails.
    """
    if not INSTALLED_UNIT_PATH.exists():
        pytest.skip(
            f"{INSTALLED_UNIT_PATH} does not exist on this host (fresh "
            "checkout or CI runner with no installed orchestrator units)"
        )
    return INSTALLED_UNIT_PATH


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
    """
    path = _require_installed_unit()
    assert_restart_backoff_effective(path)


# ---------------------------------------------------------------------------
# MANAGER layer — systemd --user's LOADED view of the unit
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("systemctl") is None, reason=_SYSTEMCTL_SKIP_REASON)
def test_installed_unit_manager_restart_steps_effective() -> None:
    """systemd --user's LOADED view must report RestartSteps=4, not just the file.

    Not redundant with the file-layer check above: `cp`-ing a corrected unit
    into place without `systemctl --user daemon-reload` leaves the MANAGER
    holding the old unit, so a file-only check would bless a host whose
    backoff is still inert. This layer is the only thing that proves a reload
    actually happened. Arrived RED reporting RestartSteps=0 — systemd's
    zero-value default, confirming the manager had never loaded a
    RestartSteps= value for this unit at all.
    """
    _require_installed_unit()
    shown = systemctl_user_show(UNIT_BASENAME, "RestartSteps")
    if shown is None:
        pytest.skip(
            "systemctl --user show could not be queried (no user D-Bus "
            "session in this runner)"
        )
    if "RestartSteps" not in shown:
        pytest.skip(
            f"systemctl --user show {UNIT_BASENAME} -p RestartSteps returned "
            "no RestartSteps= property at all (empty stdout, not merely an "
            "empty value) — most likely this host's systemd predates 254, "
            "which introduced RestartSteps=/RestartMaxDelaySec=. Verified: "
            "`systemctl --user show <unit> -p <unsupported-property>` exits "
            "0 with empty stdout, so there is nothing for this guard to "
            "assert against an unsupported property."
        )
    steps = shown.get("RestartSteps")
    assert steps == "4", (
        f"systemctl --user show {UNIT_BASENAME} -p RestartSteps reports "
        f"RestartSteps={steps!r}, not '4'. The committed "
        f"scripts/{UNIT_BASENAME} declares RestartSteps=4; a value of '0' is "
        "systemd's default and means the manager never loaded that line, so "
        "RestartMaxDelaySec=60 is being ignored and every restart waits a "
        f"flat RestartSec=10. {_REMEDIATION}"
    )
