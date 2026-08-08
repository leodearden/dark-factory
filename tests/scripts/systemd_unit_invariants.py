"""Shared systemd unit invariants, imported by more than one test module.

This module holds NO test functions of its own.  It exists because the
restart-backoff invariant below is applied from two places — the dashboard
unit/template suite (tests/scripts/test_dashboard_service_template.py, which
also owns the helper's negative-case guard) and the fleet-wide sweep
(tests/scripts/test_systemd_restart_backoff.py) — and duplicating it into
both is how the two copies drift until one silently stops catching the
defect.  Written for task 3333, lifted here by task 3408.

Almost every line of the docstrings below is measured systemd 255.4
behaviour, not restatement of the code.  Preserve it: it is the reason the
helper is correct.

Importable from tests/scripts/test_*.py only because tests/scripts/conftest.py
puts this directory on sys.path — pytest's --import-mode=importlib (set in
pyproject.toml addopts) deliberately does not.
"""
import pathlib
import re
import subprocess

# ---------------------------------------------------------------------------
# Restart backoff
#
# RestartMaxDelaySec= is silently INERT unless RestartSteps= accompanies it.
# systemd parses the cap, logs "Service has RestartMaxDelaySec= but no
# RestartSteps= setting. Ignoring." at load time, and then discards it — so the
# interpolated 5s -> 60s backoff the unit's own comment advertises never
# happens and every restart waits exactly RestartSec, forever.  Nothing in the
# unit's text reveals this; the only signal is a load-time warning nobody reads.
#
# The invariant below is RELATIONAL and CONDITIONAL, mirroring
# _assert_drain_bounded in the dashboard suite: the defect is the missing
# PAIRING, not the absence of either directive on its own.  RestartSteps= alone
# is meaningless and RestartMaxDelaySec= alone is thrown away, while a unit that
# deliberately declares no cap is not in violation of anything.
# ---------------------------------------------------------------------------


def restart_directive(path: pathlib.Path, name: str) -> str | None:
    r"""Return the effective value of ``<name>=`` in *path*, or None if absent.

    Mirrors the opaque-token-then-parse style of _timeout_stop_sec: the value is
    captured as ``(.*)`` and interpreted by the caller, so a valid-but-unexpected
    spelling (``RestartMaxDelaySec=60s``, ``RestartSec=1min``) is reported as the
    present directive it is rather than misdiagnosed as a missing line.  Matching
    ``(\d+)`` here would read ``RestartMaxDelaySec=60s`` as no cap at all and
    skip the pairing invariant silently — the worst possible failure for a guard
    whose entire job is to notice a directive that is being ignored.

    LAST occurrence wins, not the first, which is why this uses ``findall`` and
    not ``search`` (the same reasoning _success_exit_statuses applies to
    repeated directives, reaching the opposite conclusion only because
    SuccessExitStatus= is one of the few systemd directives that ACCUMULATES).
    These restart directives are scalars, and systemd overwrites on each repeat.
    Measured on this host (systemd 255.4): a unit carrying ``RestartSteps=4``
    followed by ``RestartSteps=0`` draws the pairing warning "Service has
    RestartMaxDelaySec= but no RestartSteps= setting. Ignoring." — i.e. systemd
    applied the trailing 0 — while the reverse order (0 then 4) is silent.  A
    first-match read would report 4, and this guard would bless a unit whose
    backoff systemd has in fact discarded.  Repeats are not hypothetical: a
    drop-in under <unit>.d/ is merged by appending, so an override that pins one
    of these values lands as exactly this shape.

    The anchor tolerates leading whitespace and whitespace around the separator
    (``^[ \t]*Name[ \t]*=``) because systemd.syntax does: ``    RestartMaxDelaySec
    = 60`` is a perfectly valid assignment that a column-0 anchor reads as no
    directive at all.  That mis-read degrades in the FAILURE-MASKING direction —
    the pairing invariant below returns early, the unit is blessed, and a guard
    written to catch a silently-ignored directive silently ignores it in turn.
    Contrast the opaque ``(.*)`` capture above, which is deliberately loose for
    the opposite reason: it degrades LOUDLY, reporting an unexpected spelling as
    the present directive it is rather than skipping the check.
    """
    matches = re.findall(
        rf"^[ \t]*{re.escape(name)}[ \t]*=(.*)$",
        path.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    return matches[-1].strip() if matches else None


def assert_restart_backoff_effective(path: pathlib.Path) -> None:
    """Assert *path*'s restart backoff actually engages, given that it declares a cap.

    Conditional on RestartMaxDelaySec= being present: a unit that asks for no
    backoff cap violates nothing.  But once the cap is written down, systemd
    needs RestartSteps= to interpolate between RestartSec and that cap; without
    it the cap is parsed, warned about, and dropped.
    """
    cap = restart_directive(path, "RestartMaxDelaySec")
    if cap is None:
        return

    steps = restart_directive(path, "RestartSteps")
    assert steps is not None, (
        f"{path} declares RestartMaxDelaySec={cap} but no RestartSteps=. "
        "systemd logs 'Service has RestartMaxDelaySec= but no RestartSteps= "
        "setting. Ignoring.' at unit load and then drops the cap entirely, so "
        "the backoff this unit advertises never engages — every restart waits "
        "exactly RestartSec and the delay never grows. Add RestartSteps= to "
        "make the cap effective; scripts/jcodemunch-watcher.service.template "
        "already carries this fix for the identical restart shape."
    )
    assert steps.isdigit(), (
        f"could not parse RestartSteps={steps!r} in {path} as an integer; "
        "systemd accepts only a plain unsigned integer here."
    )
    assert int(steps) >= 1, (
        f"RestartSteps={steps} in {path} produces no backoff curve at all: with "
        "zero steps there is nothing to interpolate between RestartSec and "
        f"RestartMaxDelaySec={cap}, leaving the cap as inert as if it had been "
        "omitted. Use at least 1 step."
    )

    floor = restart_directive(path, "RestartSec")
    assert floor is not None, (
        f"{path} declares RestartMaxDelaySec={cap} and RestartSteps={steps} but "
        "no RestartSec=. The curve is interpolated FROM RestartSec TO "
        "RestartMaxDelaySec, so without an explicit floor the unit silently "
        "starts from systemd's 100ms default and the backoff that runs is not "
        "the one the file describes."
    )


# ---------------------------------------------------------------------------
# systemd MANAGER view
#
# The file-layer helpers above read a unit FILE.  This one reads what the
# systemd --user MANAGER has actually LOADED, which is a different question
# with a different answer: `cp`-ing a corrected unit into place without a
# `daemon-reload` leaves the manager on the stale unit, so a file-only check
# blesses a host whose defect is still live.
#
# Written for task 3642 (as a module-local helper in
# tests/scripts/test_know_live_installed_unit_parity.py, whose own docstring
# named lifting it here "the better long-term fix" — declined then only
# because this module sat outside that task's locked scope).  Lifted here
# VERBATIM by task 3763, when tests/scripts/test_pump_web_ui_installed_unit_
# parity.py became its second consumer: exactly the trigger condition this
# module's own docstring describes.  Its negative-case guard stays where it
# was written, in test_know_live_installed_unit_parity.py, mirroring how
# test_dashboard_service_template.py owns assert_restart_backoff_effective's.
# ---------------------------------------------------------------------------


def systemctl_user_show(unit: str, *properties: str) -> dict[str, str] | None:
    """Run `systemctl --user show <unit> -p <prop> ...`, parsed to a dict.

    Returns None — never raises — when the query cannot be answered at all:
    a non-zero exit, output naming a bus connection failure ("Failed to
    connect to bus" — the shape seen from a container/CI sandbox with no
    user D-Bus session), or the query timing out — a wedged systemd --user
    manager or a stuck D-Bus leaving this call hung past its 30s timeout,
    which surfaces as subprocess.TimeoutExpired. That last case is caught
    deliberately and degrades to the same skip: subprocess.TimeoutExpired
    is a subprocess.SubprocessError, NOT an OSError (verified MRO:
    TimeoutExpired -> SubprocessError -> Exception), so the handler below
    must name both classes — narrowing it back to `except OSError` alone
    is the exact regression a prior review caught here. Callers treat None
    as "skip", not "fail": this invariant requires a live systemd --user
    manager to answer, and its absence is an environment fact rather than
    a defect in the unit.

    A property systemd does not implement at all (verified: `-p
    SomeUnknownProperty` against a live unit exits 0 with EMPTY stdout, on
    this host's systemd 255.4) is NOT surfaced as None here — it is simply
    absent from the returned dict, distinct from a recognised-but-blank
    value. Callers that care about that distinction (e.g. RestartSteps=,
    unsupported before systemd 254) must check `"Prop" not in shown`
    themselves; collapsing "unsupported" into the same falsy shape as
    "supported and empty" is how a guard meant to skip cleanly on an old
    systemd instead fails loudly and misleadingly on one.
    """
    argv = ["systemctl", "--user", "show", unit]
    for prop in properties:
        argv += ["-p", prop]
    try:
        result = subprocess.run(
            argv, capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    combined = f"{result.stdout}{result.stderr}"
    if result.returncode != 0 or "failed to connect to bus" in combined.lower():
        return None
    parsed: dict[str, str] = {}
    for line in result.stdout.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            parsed[key] = value
    return parsed
