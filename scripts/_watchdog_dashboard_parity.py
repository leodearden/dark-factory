"""The orchestrator watchdog's hourly read-only dashboard unit-parity check and its --report row (task 4883).

scripts/check_dashboard_unit_parity.py used to run only when an operator ran
setup-host.sh, so an installed dashboard unit could drift for weeks with
nothing reporting it. unit_parity_pass() runs it read-only on the watchdog's
tick instead: the watchdog executes the script FROM THE REPO CHECKOUT on an
already-armed timer, so the gate is live on merge with no install action.

Inputs are injected: *repo_dir* is the checkout whose units are checked and
under which this tier's clock lives, and *log* is the caller's journal sink,
so every line lands on the CALLER's log. It imports only _watchdog_state,
downward, and is stdlib-only like its importer.

Import-resolution contract, the same as scripts/_watched_module_removals.py's:
this stays a flat sibling in scripts/, which a directly executed script finds
because its own directory is sys.path[0].
"""

from __future__ import annotations

import enum
import os
import subprocess
import sys

from _watchdog_state import LogSink, read_clock_epoch, stamp_clock, within_min_interval

DASHBOARD_PARITY_SCRIPT_RELPATH = os.path.join("scripts", "check_dashboard_unit_parity.py")
# Every line the checker prints carries this tag (its LOG_TAG), so without it the
# checker did not report, whatever its exit code says — the rule
# scripts/setup-host.sh::_parity_verdict applies.
DASHBOARD_PARITY_TAG = "[dashboard_unit_parity]"

# The parity check's OWN clock, independent of every deploy and restart clock
# the watchdog keeps, so a redeploy never resets the parity cadence and a
# parity check never moves a deploy gate. Not a deploy clock:
# df_pytest_isolation.py does not protect it. ORCH_UNIT_PARITY_CLOCK overrides
# it (see unit_parity_clock_path).
UNIT_PARITY_CLOCK_RELPATH = os.path.join("data", "orchestrator", "last_unit_parity_check.json")

# Cost is not the constraint (stdlib-only, reads six files); journal legibility
# is. Ungated, a drifted host would log ~1440 near-identical WARNINGs a day.
# <=0 disables the throttle. Env-with-fallback: a typo must not crash the oneshot.
try:
    UNIT_PARITY_MIN_INTERVAL_SECS = int(os.environ["ORCH_UNIT_PARITY_MIN_INTERVAL_SECS"])
except (KeyError, ValueError):
    UNIT_PARITY_MIN_INTERVAL_SECS = 3600

UNIT_PARITY_TIMEOUT_SECS = 30


class UnitParityVerdict(enum.StrEnum):
    """What unit_parity_verdict() concluded; its value is the --report row's word.

    DRIFT is the checker's exit 1, which covers three findings its report tags
    apart: [drift], [override] (a drop-in) and [vanished] (a committed unit).
    """

    PARITY = "parity"
    DRIFT = "drift"
    ABSENT = "absent"
    UNKNOWN = "unknown"


# The checker's documented exit codes, believed only once DASHBOARD_PARITY_TAG
# shows it reported: exit 2 is also python3's can't-open-file and argparse's
# usage-error status, and an uncaught traceback exits 1. Any other code is
# not a parity claim and reads as UNKNOWN.
UNIT_PARITY_VERDICTS: dict[int, UnitParityVerdict] = {
    0: UnitParityVerdict.PARITY,
    1: UnitParityVerdict.DRIFT,
    2: UnitParityVerdict.ABSENT,
}


def unit_parity_clock_path(repo_dir: str) -> str:
    """Return the parity clock's path: ORCH_UNIT_PARITY_CLOCK, else under *repo_dir*."""
    return os.environ.get(
        "ORCH_UNIT_PARITY_CLOCK", os.path.join(repo_dir, UNIT_PARITY_CLOCK_RELPATH)
    )


def unit_parity_verdict(repo_dir: str) -> tuple[UnitParityVerdict, str]:
    """Run *repo_dir*'s dashboard unit-parity checker; return ``(verdict, report)``.

    *verdict* is PARITY / DRIFT / ABSENT (the checker's own exit codes 0/1/2,
    see UNIT_PARITY_VERDICTS) or UNKNOWN when the checker could not run, ran
    but produced no DASHBOARD_PARITY_TAG report of its own, or exited with
    anything else. The tag is checked FIRST, as
    scripts/setup-host.sh::_parity_verdict does. *report* is the checker's
    combined stdout and stderr — how an operator learns WHICH finding it
    made, or why the checker did not report.

    ``--installed-dir`` is deliberately not passed: the checker's own default
    is the one setup-host.sh gates against, and re-deriving it here would be a
    second copy of it.

    Read-only, and never raises — same contract as the watchdog's probe_port
    and log(): a tooling failure in an observational check must not abort the
    tick.
    """
    try:
        result = subprocess.run(
            [
                sys.executable,
                os.path.join(repo_dir, DASHBOARD_PARITY_SCRIPT_RELPATH),
                "--repo-root",
                repo_dir,
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=UNIT_PARITY_TIMEOUT_SECS,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return UnitParityVerdict.UNKNOWN, f"{type(exc).__name__}: {exc}"
    report = "\n".join(
        part.strip() for part in (result.stdout, result.stderr) if part and part.strip()
    )
    if DASHBOARD_PARITY_TAG not in report:
        return UnitParityVerdict.UNKNOWN, (
            f"checker produced no {DASHBOARD_PARITY_TAG} report "
            f"(exit {result.returncode}): {report or 'no output'}"
        )
    verdict = UNIT_PARITY_VERDICTS.get(result.returncode)
    if verdict is None:
        return UnitParityVerdict.UNKNOWN, f"checker exited {result.returncode}: {report}"
    return verdict, report


def unit_parity_pass(repo_dir: str, log: LogSink) -> None:
    """Report dashboard unit drift to *log*, at most once per min-interval.

    Purely observational and read-only: it runs the checker and logs, and
    never installs, reloads or restarts anything — there is no --fix by
    design (see the checker's module docstring).

    The clock is stamped on ATTEMPT, before the checker runs, so a
    persistently broken checker is retried hourly rather than on every tick.
    Only DRIFT is a WARNING. ABSENT is silent: the dashboard units are not
    installed on this host, which setup-host.sh already treats as benign. A
    checker that could not run gets one plain line, so a broken gate stays
    distinguishable from a green one.

    Never raises: an unexpected failure is logged so it cannot abort the tick.
    """
    try:
        clock = unit_parity_clock_path(repo_dir)
        if within_min_interval(
            UNIT_PARITY_MIN_INTERVAL_SECS,
            lambda: read_clock_epoch(clock, "unit-parity clock", log),
        ):
            return
        stamp_clock(clock, log)
        verdict, report = unit_parity_verdict(repo_dir)
        if verdict == UnitParityVerdict.DRIFT:
            log(
                "WARNING: dashboard unit parity: directive drift, a drop-in "
                "override, or a vanished committed unit — the checker's "
                f"[drift] / [override] / [vanished] report says which:\n{report}"
            )
        elif verdict == UnitParityVerdict.UNKNOWN:
            log(f"dashboard unit parity: checker could not run, parity unknown: {report}")
    except Exception as exc:  # noqa: BLE001
        log(f"dashboard unit parity pass error: {exc!r}")


def print_unit_parity(repo_dir: str, log: LogSink) -> None:
    """Print the dashboard unit-parity row for ``--report``.

    The operator's right-now view of the check unit_parity_pass() runs hourly.
    NOT clock-gated — an operator asking now wants an answer now — and it
    writes no clock, so running --report never shifts the timer path's
    cadence.

    On DRIFT the checker's own tagged report follows the row, naming the
    finding without a second command; on UNKNOWN the reason the checker could
    not run follows it instead.

    Informational only, like the watchdog's fm row: never alters report()'s
    exit code. An unexpected failure degrades the verdict to a logged UNKNOWN
    rather than crashing --report after report() has already computed its
    exit code.
    """
    try:
        verdict, detail = unit_parity_verdict(repo_dir)
    except Exception as exc:  # noqa: BLE001
        log(f"watchdog error printing the dashboard unit parity row: {exc}")
        verdict, detail = UnitParityVerdict.UNKNOWN, ""
    print(f"dashboard unit parity: {verdict} | CHECKER: {DASHBOARD_PARITY_SCRIPT_RELPATH}")
    if verdict in (UnitParityVerdict.DRIFT, UnitParityVerdict.UNKNOWN) and detail:
        print(detail)
