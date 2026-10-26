"""Parity checker for the installed orchestrator systemd units.

Verifies that the eighteen units committed under ``scripts/`` still agree with
the copies actually installed in ``~/.config/systemd/user/``:

    orchestrator-watchdog.service          orchestrator-watchdog.timer
    orchestrator-dark-factory.service      orchestrator-reify.service
    orchestrator-autopilot-video.service   orchestrator-my-solar-challenge.service
    orchestrator-solar-challenge-platform.service
    orchestrator-know-live.service         orchestrator-pump-web-ui.service
    orchestrator-stream-touch.service

and the eight ``orchestrator-<project>.socket`` units (one per project
orchestrator above; the watchdog pair has none) that hold each project's
escalation MCP port across a restart. Each socket's committed copy lives at
``scripts/<unit>.template`` — see the ``UNITS`` registry below for why.

The two know-live / pump-web-ui entries were added 2026-08-06, when this
branch was rebased onto a main that had, in the interim, wired both units
into ``setup-host.sh`` (task 3641). They are registered here to keep the
staleness guard below satisfied — a unit the installer copies but this
registry omits is installed-but-unchecked, the exact silent-drift failure
this checker exists to catch. The REBASE NOTE that stood here, saying
neither had had a parity MEASUREMENT taken against this host, is DISCHARGED:
task 4882 ran this checker against ``~/.config/systemd/user`` on 2026-09-05
and both are ``clean``. See the KNOWN RED section below for the full
measurement and for why the figures here are dated rather than asserted as
current.

The purpose is narrow and concrete: make it OBSERVABLE when a repo-side unit
change never reaches the running system.  A unit edit that stays repo-side is
indistinguishable from no edit at all, and nothing reported that until this
check existed.  Task 3392 added ``TimeoutStartSec=1800`` to the committed
watchdog service — a whole-tick bound on the fleet probe — and as measured on
2026-08-02 the installed copy still did not carry it, so the bound had been
inert on this host for the entire time it appeared to be in force.

Why FULL SYMMETRIC EQUALITY, and not a curated key registry
-----------------------------------------------------------
``scripts/check_dashboard_unit_parity.py`` bounds its comparison to a curated
per-unit ``(section, key)`` registry, because one of its units is RENDERED
from a template (``__REPO_ROOT__`` / ``__UV_PATH__`` substitution) and an
unbounded diff there would be a false-positive machine.

That reasoning does not transfer.  ``setup-host.sh`` installs all eighteen
units here by ``cp``-ing them VERBATIM — no substitution at all, not even for the
host paths, which are hardcoded in the committed files.  So the two sides are
supposed to be character-identical by construction, every parsed-directive
difference is real drift, and the false-positive risk that forced curation
does not exist.

The equality rule is also strictly STRONGER in the way that turned out to
matter.  A curated key list can only ever see keys someone thought to
register, so it is structurally blind to an installed-ONLY directive.  The
single most consequential divergence measured on this host is exactly that:
the installed timer carries ``AccuracySec=5s`` (live-verified via
``systemctl --user show orchestrator-watchdog.timer -p AccuracyUSec``) and no
committed copy ever has.  systemd's default AccuracySec is 1min, so
propagating the committed timer without it would have widened the 60s probe's
elapse window from [60s, 65s] to [60s, 120s] — roughly halving fleet-watchdog
responsiveness, as a "parity fix".  No plausible curated key list would have
contained ``AccuracySec``; a registry-based checker would have reported green
on the one divergence that could have caused a supervision regression.

The credibility cost curation exists to avoid is paid by the PARSER instead:
comments (``#`` and ``;``) and blank lines are dropped before comparison, so a
comment-only edit — task 3392 added ~35 comment lines — does not fire the
gate.  A gate nobody believes is worse than no gate.

Exit codes
----------
The family's codes are defined once, in
``systemd_unit_parity.py::ParityExit``. Of the report blocks that share
exit 1, only ``[drift]`` is a directive difference to propagate. A consumer
that needs to act per UNIT rather than per run wants the
``--print-verdicts`` channel described next, not these codes.

Machine-readable verdicts
-------------------------
The exit code above is a WHOLE-RUN verdict: with eighteen units, one drifted
unit collapses the entire run to 1 and tells a consumer nothing about the
other fifteen.  That is fine for an operator reading the report, and was not
fine for ``setup-host.sh``, which used it to decline the install of ALL eighteen
units when
any one of them was unverifiable — so a single deliberate drop-in on
``orchestrator-reify.service`` also blocked the watchdog pair from being
reinstalled and re-enabled, which is a supervision regression dressed as
caution.

``--print-verdicts`` is the per-unit channel that fixes that.  It is OPT-IN
and purely additive: with the flag absent, the report and every exit code are
byte-identical.  With it, one extra line per SELECTED unit, in sorted order,
emitted before the human summary blocks::

    [orchestrator_unit_parity] verdict <unit> <kind>[,<kind>...]

The kinds, what each means, and their rendering order are defined in
``systemd_unit_parity.py::VERDICT_KINDS``. ``clean`` (safe to overwrite) and
``absent`` (nothing to overwrite) are the two kinds a unit can be installed
under.

The sole consumer is ``scripts/setup-host.sh`` section 5, which installs the
units whose verdict is ``clean`` or ``absent`` and skips the rest, naming each
skipped unit and its kind.  A unit with NO verdict line — the checker missing,
renamed, run without the flag, or simply not knowing that unit — is treated by
the installer as unverified and is NOT installed.  That is why the lines carry
the same ``[orchestrator_unit_parity]`` tag as everything else here: the
installer already refuses to believe any output lacking that tag, so a renamed
flag exits 2 with no tag, produces no verdicts, and installs nothing.

KNOWN RED on this host
----------------------
CURRENT STATE, measured 2026-09-05 with ``--print-verdicts`` (task 4882).
Seven of the nine units are ``clean``.  The family is NO LONGER red on
arrival, and the two findings that remain are both DELIBERATE and OWNED:

- ``orchestrator-watchdog.service`` — ``drift``.  The sole differing
  directive is an INSTALLED-ONLY ``Environment=ORCH_RESTART_MIN_INTERVAL_SECS=259200``
  (72h) that the committed copy does not carry.  It is a documented,
  self-expiring deploy pause added host-side 2026-09-03 (the installed
  unit's own header comment carries the rationale and the measurements),
  throttling the watchdog's STALENESS redeploy tier from its 8h default.
  OWNED by task 5020, whose escalation is the trigger to remove it.
- ``orchestrator-reify.service`` — ``override``.  The
  ``orchestrator-reify.service.d/warm-lane.conf`` drop-in is deliberate
  (task 2100's ``RequiresMountsFor=/home/leo/src/warm-lanes``, carried as a
  drop-in rather than in the unit).  Task 4198 made the installer's skip
  PER UNIT precisely so this permanent, intentional divergence stops
  withholding the install from the other eight.

Note the DIRECTION of the live drift, because it is the mirror image of the
2026-08-02 first bullet below: there the INSTALLED copies were the ones
missing a directive, so the repo side was right; here the installed copy is
the SUPERSET and the repo side is the one that must not be propagated.
Installing the committed watchdog unit today would silently delete a
deliberate throttle and resume 8h fleet redeploys — which is the
``[drift] BUT CHECK DIRECTION FIRST`` line's live, present-tense
justification, no longer merely a historical one.

HISTORICAL (2026-08-02), RESOLVED by commit 4fcd43eec0 (task 3512, landed
2026-08-04).  Both bullets below were live when this checker was written and
are retained because they are the recorded EVIDENCE for policies that are
still in force — not because they still describe this host:

- HISTORICAL: all five installed ``orchestrator-*.service`` copies lacked
  ``RestartSteps=4`` while declaring ``RestartMaxDelaySec=60``.  systemd
  requires the former for the latter to take effect — without it it logs
  "has RestartMaxDelaySec= but no RestartSteps= setting. Ignoring." and
  DISCARDS the cap — so the restart backoff ceiling those units appeared to
  have was silently inert.  Here the REPO copy was the correct one.
- HISTORICAL: four disagreed on the ExecStart ``--config`` path, and in the
  direction OPPOSITE to the one above: the INSTALLED copies named the
  canonical ``dark-factory-orchestrator.yaml`` that CLAUDE.md requires,
  while the COMMITTED copies still named legacy ``orchestrator.yaml`` /
  ``orchestrator-config.yaml``.  Here the INSTALLED copy was the correct
  one.  Re-measured 2026-09-05: every committed ``--config`` path now names
  the canonical spelling and every one EXISTS; the two legacy paths named
  below do not exist and are named by no committed unit.

DO NOT "fix" a ``--config`` disagreement by forcing an install.  This is a
POLICY, and the instance that motivated it is the second bullet above:
measured 2026-08-02, ``/home/leo/src/reify/orchestrator.yaml`` and
``/home/leo/src/autopilot-video/orchestrator-config.yaml`` — the paths the
COMMITTED units then named — DID NOT EXIST.  Only the canonical
``dark-factory-orchestrator.yaml`` did.  So the ``cp`` block in
``setup-host.sh`` would have overwritten two working installed units with
committed ones pointing at absent config files, breaking both orchestrators
on their next restart.  THAT PARTICULAR HAZARD IS DISCHARGED (4fcd43eec0);
the policy it justifies is not, and the 2026-09-05 watchdog drift above is a
current instance of the same shape in a different directive.  The policy is
why ``setup-host.sh`` runs this check PRE-install and, on a finding, SKIPS
that unit's install rather than merely warning — a warning in a
non-interactive ``set -e`` script scrolls past and the next line does the
overwrite anyway.  An operator who has read the report and decided the
committed side is right proceeds with ``DF_INSTALL_ORCH_UNITS=1``.  The gate
never aborts the installer itself.

Since task 4198 that skip is PER UNIT — the installer reads the verdict
channel above and declines only the units that did not clear, installing the
rest on the same run.  The policy is unchanged: a unit with a finding is
still never overwritten without ``DF_INSTALL_ORCH_UNITS=1``.  So the two
units named at the top of this section are protected from exactly the
overwrite this section warns about, while a permanent, deliberate divergence
on one unit no longer withholds the install from every other.

Naming the owner here is not bookkeeping.  ``check_dashboard_unit_parity.py``
records that a permanently-red gate gets switched off within a week — taking
the accidental drift it exists to catch with it — and accepted its own
red-on-arrival unit only because a named task owned the fix.  That condition
is still met: task 5020 owns the watchdog throttle (and it self-expires
anyway), and the reify drop-in is a standing intentional divergence that
task 4198 already accommodated per-unit.  The red is OWNED, not ambient.
The repeatable ``--unit`` filter lets an operator scope a run meanwhile.
The five-unit convergence that ticket ``tkt_0RRZWH0V5F9PPG86A1WDJ3NV2R``
(filed from task 3424) was raised for is DONE — task 3512 landed it, per the
HISTORICAL bullets above.

REBASE NOTES (2026-08-06) — DISCHARGED 2026-09-05, both by measurement.
Two notes stood here and at the head of this docstring, and the whole content
of each was "these numbers were not re-measured against the live host".  Task
4882 ran this checker against ``~/.config/systemd/user`` and recorded the
result above, which is what they asked for, so they are retired rather than
left standing — an un-actioned "this may be stale" note is what makes a
section stale BY CONSTRUCTION, since it never expires on its own.  For the
record: ``orchestrator-know-live.service`` and ``orchestrator-pump-web-ui.service``
(the two units the head-of-docstring note added without a measurement) are
both ``clean``, and the ExecStart ``--config`` hazard the second note
suspected 4fcd43eec0 had already fixed had indeed been fixed.

READING THIS SECTION LATER: it is dated on purpose.  The host has moved under
it three times (2026-08-02, 2026-08-16, 2026-09-05), disagreeing each time.
Re-run ``python3 scripts/check_orchestrator_unit_parity.py --print-verdicts``
— it is read-only and cannot mutate the host — before relying on any figure
here.

Design notes
------------
- Stdlib-only (argparse, pathlib, sys) plus the shared
  ``scripts/systemd_unit_parity.py``, which is itself stdlib-only —
  runs under a plain python3, exactly like the two sibling checkers whose
  idioms this script follows deliberately rather than inventing a pattern.
- The parser is imported from ``scripts/systemd_unit_parity.py``, which was
  lifted out of ``check_dashboard_unit_parity.py`` when this second consumer
  appeared rather than copied into it.  See that module's docstring.
- No ``--fix``.  Propagating repo units into ``~/.config/systemd/user/`` and
  daemon-reloading is already what ``scripts/setup-host.sh`` does; duplicating
  it here would be a fourth copy of the install logic, and an actively unsafe
  one — it could silently repoint a live orchestrator's ExecStart or re-arm a
  timer someone deliberately left disarmed.  Drift is reported with a
  remediation pointer instead.

Usage
-----
  # verify with defaults (~/.config/systemd/user vs this repo)
  python3 scripts/check_orchestrator_unit_parity.py

  # verify explicit trees
  python3 scripts/check_orchestrator_unit_parity.py \\
      --installed-dir ~/.config/systemd/user \\
      --repo-root /home/leo/src/dark-factory

  # verify only the watchdog units
  python3 scripts/check_orchestrator_unit_parity.py \\
      --unit orchestrator-watchdog.service \\
      --unit orchestrator-watchdog.timer

Testing note
------------
All drift-logic tests run against ``tmp_path`` fixtures — never the host's
real ``~/.config/systemd/user/``.  With five units knowingly red here, a test
asserting parity against the live host would be red on landing, and one
asserting drift would flip red the moment the follow-up task fixes it.  Either
encodes host state rather than checker behaviour.
"""

import argparse
import pathlib
import sys
from collections.abc import Sequence

from systemd_unit_parity import (
    ABSENT,
    VERDICT_KINDS,
    BlockText,
    Drift,
    ParityRun,
    ReportText,
    UnitRef,
    parse_unit_directives,
    run_parity,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Prefixed onto every line this script prints, so a parity report is greppable
# the same way the dashboard checker's [dashboard_unit_parity] report is.
LOG_TAG = "orchestrator_unit_parity"

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_DEFAULT_INSTALLED_DIR = pathlib.Path.home() / ".config" / "systemd" / "user"


# ---------------------------------------------------------------------------
# Drift records and comparison
# ---------------------------------------------------------------------------

def _render(values: list[str] | None) -> str:
    """Render one side's values for the operator's report.

    A single-valued directive renders as its bare value (the common case, and
    what an operator expects to see next to a key).  A repeated directive
    renders as a bracketed list of QUOTED occurrences, because systemd APPLIES
    every occurrence and showing only the first would misreport what is
    actually in force.

    The quoting is not decoration.  A bare ``, ``-join is ambiguous the moment
    a value contains a comma — two ``Environment=`` lines could render as
    ``[OPTS=a,b, X=1]``, where an operator cannot tell where one occurrence
    ends and the next begins.  Comparison happens on the raw lists, so this
    could never cause a false negative; but the report IS the deliverable of a
    checker whose whole value is being believed, and an operator who has to
    guess at occurrence boundaries has been handed a diff they cannot act on.
    """
    if values is None:
        return ABSENT
    if len(values) == 1:
        return values[0]
    return "[" + ", ".join(repr(v) for v in values) + "]"


def compare_unit(
    name: str, repo_text: str, installed_text: str
) -> list[Drift]:
    """Compare two unit texts by FULL SYMMETRIC directive equality.

    Both sides are parsed into ``{section: {key: [values]}}``, then the UNION
    of sections and — per section — the UNION of keys is walked in sorted
    order, emitting a Drift wherever the two value LISTS differ.

    Three properties are load-bearing, each pinned by a test:

    1. **UNION, not intersection**, on both sections and keys.  An
       intersection walk sees only what both sides already agree exists, so
       it silently ignores a whole dropped ``[Install]`` section (a unit that
       is no longer enable-able) and, more importantly, every INSTALLED-only
       directive.  That last case is the entire reason this checker compares
       by equality instead of against a curated key registry — see the module
       docstring, and the measured ``AccuracySec=5s``.
    2. **The whole values LIST is compared**, not the first value.  systemd
       applies every occurrence of a repeated directive, so an installed copy
       that dropped one of two ``Environment=`` lines is drift even though
       the key is present on both sides and its first value matches.
    3. **Sorted order**, so two runs of the same drift set render identically
       and an operator can diff one report against another.

    Comments and blank lines never reach here — the shared parser drops them —
    which is what lets the comparison stay unbounded without firing on a
    comment reflow.
    """
    repo = parse_unit_directives(repo_text)
    installed = parse_unit_directives(installed_text)

    drifts: list[Drift] = []
    for section in sorted(set(repo) | set(installed)):
        repo_section = repo.get(section, {})
        installed_section = installed.get(section, {})
        for key in sorted(set(repo_section) | set(installed_section)):
            repo_values = repo_section.get(key)
            installed_values = installed_section.get(key)
            if repo_values == installed_values:
                continue

            # Worded apart so the report sends the operator the right way: a
            # value difference is something to reconcile, whereas a one-sided
            # directive is something to add on the side that lacks it.
            if installed_values is None:
                reason = (
                    f"{key} declared in the repo copy, absent from the "
                    "installed copy"
                )
            elif repo_values is None:
                reason = (
                    f"{key} declared in the installed copy, absent from the "
                    "repo copy"
                )
            else:
                reason = f"{key} values differ"

            drifts.append(
                Drift(
                    unit=name,
                    section=section,
                    key=key,
                    repo_value=_render(repo_values),
                    installed_value=_render(installed_values),
                    reason=reason,
                )
            )
    return drifts


# ---------------------------------------------------------------------------
# The unit registry
# ---------------------------------------------------------------------------

# unit name -> repo-relative path of its committed copy.
#
# No per-unit key curation, unlike check_dashboard_unit_parity.UNITS: these are
# all installed by verbatim `cp`, so the comparison is full symmetric equality
# over the whole parsed directive set (see the module docstring).  What IS
# curated is WHICH units are covered, and that is guarded against rot by
# tests/scripts/test_check_orchestrator_unit_parity.py, which derives the
# expected set from setup-host.sh's own `cp` lines and asserts equality in both
# directions.  A unit added to the installer but not here would be
# installed-but-unchecked — the same silent drift one level up.
#
# The registry stays an explicit literal (the derivation lives in the test)
# precisely so a bug in that shell parsing can never disarm this checker.
UNITS: dict[str, str] = {
    "orchestrator-watchdog.service": "scripts/orchestrator-watchdog.service",
    "orchestrator-watchdog.timer": "scripts/orchestrator-watchdog.timer",
    "orchestrator-dark-factory.service": "scripts/orchestrator-dark-factory.service",
    "orchestrator-reify.service": "scripts/orchestrator-reify.service",
    "orchestrator-autopilot-video.service": "scripts/orchestrator-autopilot-video.service",
    "orchestrator-my-solar-challenge.service": (
        "scripts/orchestrator-my-solar-challenge.service"
    ),
    "orchestrator-solar-challenge-platform.service": (
        "scripts/orchestrator-solar-challenge-platform.service"
    ),
    # Added by this branch's 2026-08-06 rebase onto main: task 3641 wired both
    # of these into setup-host.sh's cp block after this branch forked. Both
    # measured `clean` 2026-09-05 (task 4882) — see the module docstring's
    # KNOWN RED section, which retired the REBASE NOTE that stood here.
    "orchestrator-know-live.service": "scripts/orchestrator-know-live.service",
    "orchestrator-pump-web-ui.service": "scripts/orchestrator-pump-web-ui.service",
    # stream-touch joined 2026-10-22 (escalation 8110), installed and committed together.
    "orchestrator-stream-touch.service": "scripts/orchestrator-stream-touch.service",
    # The eight escalation-port-holding sockets (task: escalation socket
    # activation). Each pairs with the like-named .service above and is
    # installed by the SAME verbatim `cp` as every other unit here — `.socket`
    # is only not itself the committed EXTENSION, because it is not in the
    # shared lock-charter extension allowlist. The committed source is
    # `<unit>.template`; comparison still runs on the parsed directives, so
    # that source's extra leading comment block (required by the charter) is
    # invisible to it exactly like every other unit's header comment (see
    # "Why FULL SYMMETRIC EQUALITY" above — comments are dropped before either
    # side is compared).
    "orchestrator-dark-factory.socket": "scripts/orchestrator-dark-factory.socket.template",
    "orchestrator-reify.socket": "scripts/orchestrator-reify.socket.template",
    "orchestrator-autopilot-video.socket": (
        "scripts/orchestrator-autopilot-video.socket.template"
    ),
    "orchestrator-my-solar-challenge.socket": (
        "scripts/orchestrator-my-solar-challenge.socket.template"
    ),
    "orchestrator-solar-challenge-platform.socket": (
        "scripts/orchestrator-solar-challenge-platform.socket.template"
    ),
    "orchestrator-know-live.socket": "scripts/orchestrator-know-live.socket.template",
    "orchestrator-pump-web-ui.socket": (
        "scripts/orchestrator-pump-web-ui.socket.template"
    ),
    "orchestrator-stream-touch.socket": (
        "scripts/orchestrator-stream-touch.socket.template"
    ),
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser (shared by main and the tests)."""
    parser = argparse.ArgumentParser(
        description=(
            "Verify parity between the in-repo and installed orchestrator units."
        )
    )
    parser.add_argument(
        "--installed-dir",
        type=pathlib.Path,
        default=_DEFAULT_INSTALLED_DIR,
        help="Directory holding the installed units (default: %(default)s)",
    )
    parser.add_argument(
        "--repo-root",
        type=pathlib.Path,
        default=_REPO_ROOT,
        help="Repo root holding the committed units (default: %(default)s)",
    )
    parser.add_argument(
        "--unit",
        action="append",
        choices=sorted(UNITS),
        metavar="UNIT",
        help=(
            "Restrict the run to this unit (repeatable). "
            "Default: all of " + ", ".join(sorted(UNITS))
        ),
    )
    parser.add_argument(
        "--print-verdicts",
        action="store_true",
        help=(
            "Additionally emit one machine-readable "
            f"'[{LOG_TAG}] verdict <unit> <kind>[,<kind>...]' line per selected "
            "unit. Purely additive: the human report and every exit code are "
            "unchanged. Kinds: " + ", ".join(VERDICT_KINDS)
        ),
    )
    return parser


def _compare(unit: UnitRef, repo_text: str, installed_text: str) -> list[Drift]:
    return compare_unit(unit.name, repo_text, installed_text)


# The second [drift] tail line is deliberately NOT "the installed copy is
# stale": which side is right has varied per unit on every measurement (see the
# KNOWN RED section above), so installing blindly can delete a deliberate
# installed-only setting.
_REPORT = ReportText(
    skip="{name}: installed unit not found: {path} (not installed on this host)",
    drift=BlockText(
        "drift",
        "{count} directive(s) differ between repo and installed units:",
        (
            "To propagate the committed units to this host, run: "
            "DF_INSTALL_ORCH_UNITS=1 bash scripts/setup-host.sh  (a plain run "
            "runs this same check and DECLINES the unit install on drift, "
            "because the committed side is not always the correct one — see "
            "below. This checker is read-only by design; the module docstring "
            "says why there is no --fix.)",
            "BUT CHECK DIRECTION FIRST: drift does not mean the installed copy is "
            "the stale one. Confirm the committed value is the one you want on "
            "this host before installing — see the KNOWN RED section of this "
            "script's module docstring.",
        ),
    ),
)


def main(argv: Sequence[str]) -> int:
    """Parse args and run the parity check through ``systemd_unit_parity.run_parity``.

    Exit codes: see ``systemd_unit_parity.ParityExit``.
    """
    args = _build_parser().parse_args(argv)
    units = [
        UnitRef(name, args.repo_root / UNITS[name], args.installed_dir / name)
        for name in args.unit or sorted(UNITS)
    ]
    run = ParityRun(LOG_TAG, units, _compare, _REPORT, print_verdicts=args.print_verdicts)
    return run_parity(run, out=sys.stdout)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
