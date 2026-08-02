"""Parity checker for the installed orchestrator systemd units.

Verifies that the seven units committed under ``scripts/`` still agree with
the copies actually installed in ``~/.config/systemd/user/``:

    orchestrator-watchdog.service          orchestrator-watchdog.timer
    orchestrator-dark-factory.service      orchestrator-reify.service
    orchestrator-autopilot-video.service   orchestrator-my-solar-challenge.service
    orchestrator-solar-challenge-platform.service

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

That reasoning does not transfer.  ``setup-host.sh`` installs all seven units
here by ``cp``-ing them VERBATIM — no substitution at all, not even for the
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
0 — parity (every compared directive agrees, at least one unit was actually
    compared, and nothing overrides the units that were)
1 — drift, OR a committed unit VANISHED (no committed copy, so nothing could
    be verified for it), OR an installed unit carries a drop-in override (so
    the unit file was compared but the EFFECTIVE configuration was not)
2 — installed unit absent (no installed copy found for one or more units)

PRECEDENCE: drift (1) DOMINATES absence (2).  With seven units a single run
can hit both at once, and returning 2 there would let an unrelated
uninstalled unit mask an actionable finding — ``setup-host.sh`` treats 2 as a
benign "not installed on this host, skipping" and only 1 as something to act
on.  An absent unit is still REPORTED in that case: dominated, not hidden.

A run that compared ZERO units can NEVER report parity.  This vocabulary is
adopted unchanged from ``check_dashboard_unit_parity.py`` so setup-host.sh's
existing 0/2/other branch shape carries over verbatim.

KNOWN RED on this host
----------------------
As measured 2026-08-02, the five ``orchestrator-*.service`` entries all
diverge from their committed copies and this checker reports drift for them:

- ALL FIVE installed copies lack ``RestartSteps=4`` while declaring
  ``RestartMaxDelaySec=60``.  systemd requires the former for the latter to
  take effect — without it it logs "has RestartMaxDelaySec= but no
  RestartSteps= setting. Ignoring." and DISCARDS the cap — so the restart
  backoff ceiling those units appear to have is silently inert today.  Here
  the REPO copy is the correct one.
- FOUR disagree on the ExecStart ``--config`` path, and in the direction
  opposite to the one above: the INSTALLED copies name the canonical
  ``dark-factory-orchestrator.yaml`` that CLAUDE.md requires, while the
  COMMITTED copies still name legacy ``orchestrator.yaml`` /
  ``orchestrator-config.yaml``.  Here the INSTALLED copy is the correct one.
- ``orchestrator-reify.service``'s installed copy lacks
  ``RequiresMountsFor=/home/leo/src/warm-lanes`` — but carries a drop-in
  (``orchestrator-reify.service.d/warm-lane.conf``) instead, which is why
  this unit also reports ``[override]``.

DO NOT "fix" that second bullet by running ``setup-host.sh``.  Measured
2026-08-02: ``/home/leo/src/reify/orchestrator.yaml`` and
``/home/leo/src/autopilot-video/orchestrator-config.yaml`` — the paths the
COMMITTED units name — DO NOT EXIST.  Only the canonical
``dark-factory-orchestrator.yaml`` does.  So the ``cp`` block in
``setup-host.sh`` would overwrite two working installed units with committed
ones pointing at absent config files, breaking both orchestrators on their
next restart.  This is the exact hazard the "no ``--fix``" note below
describes, and it is live TODAY: for these four units the committed file is
the thing that is wrong, and it must be corrected in the repo BEFORE any
install propagates it.  This is also why the setup-host.sh gate is warn-only
and pre-install: it makes the divergence visible without acting on it.

Fixing all five is OWNED by the follow-up filed from task 3424 as ticket
``tkt_0RRZWH0V5F9PPG86A1WDJ3NV2R``.  Task 3424 deliberately converged only
the two watchdog units on the live host, and the measurement above is why
that scoping was right: the five need a per-unit judgement about WHICH side
is correct (repo for RestartSteps, installed for ExecStart) plus a restart of
the running fleet — not a blanket copy in either direction.

Naming the owner here is not bookkeeping.  ``check_dashboard_unit_parity.py``
records that a permanently-red gate gets switched off within a week — taking
the accidental drift it exists to catch with it — and accepted its own
red-on-arrival unit only because a named task owned the fix.  That is the
condition being met here, so the red is OWNED rather than ambient.  The
repeatable ``--unit`` filter lets an operator scope a run meanwhile.

Design notes
------------
- Stdlib-only (argparse, dataclasses, pathlib, sys) plus the shared parser —
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
import dataclasses
import pathlib
import sys
from typing import Sequence

from systemd_unit_parity import parse_unit_directives

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Prefixed onto every line this script prints, so a parity report is greppable
# the same way the dashboard checker's [dashboard_unit_parity] report is.
LOG_TAG = "orchestrator_unit_parity"

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_DEFAULT_INSTALLED_DIR = pathlib.Path.home() / ".config" / "systemd" / "user"


def _log(message: str, *, stream=None) -> None:
    """Print *message* prefixed with the log tag."""
    print(f"[{LOG_TAG}] {message}", file=stream if stream is not None else sys.stdout)


# ---------------------------------------------------------------------------
# Drift records and comparison
# ---------------------------------------------------------------------------

# Rendered in place of a value on whichever side does not declare the
# directive at all.  Deliberately not '' or None: it appears verbatim in the
# operator's report, where "<absent>" reads unambiguously and an empty string
# would look like a directive set to nothing.
_ABSENT = "<absent>"


@dataclasses.dataclass(frozen=True)
class Drift:
    """One disagreement between the repo copy and the installed copy."""

    unit: str
    section: str
    key: str
    repo_value: str
    installed_value: str
    reason: str


def _render(values: list[str] | None) -> str:
    """Render one side's values for the operator's report.

    A single-valued directive renders as its bare value (the common case, and
    what an operator expects to see next to a key).  A repeated directive
    renders as a bracketed list, because systemd APPLIES every occurrence and
    showing only the first would misreport what is actually in force.
    """
    if values is None:
        return _ABSENT
    if len(values) == 1:
        return values[0]
    return "[" + ", ".join(values) + "]"


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


def find_dropins(installed_dir: pathlib.Path, unit_name: str) -> list[pathlib.Path]:
    """Return the ``.conf`` drop-ins systemd would layer over *unit_name*.

    ``systemctl --user edit`` does not modify the unit file; it writes
    ``<unit>.d/override.conf`` beside it, and systemd merges that over the
    unit at load time.  Reading only ``<installed-dir>/<unit>`` is therefore
    blind to it: a drop-in setting ``AccuracySec=1min`` or ``Restart=no``
    leaves every compared directive matching while the RUNNING configuration
    is not the committed one — the precise claim this gate exists to make
    checkable.

    Not hypothetical on this host: no orchestrator unit has a drop-in today,
    but ``~/.config/systemd/user/orchestrator-reify.service.d/`` exists, so
    the mechanism is already in live use here.

    Returns the sorted ``.conf`` files, or ``[]`` when the directory is absent
    or holds none (systemd ignores non-``.conf`` files there, so this counts
    exactly what would take effect — counting a stray ``override.conf.bak``
    would report an override that has no effect at all).

    Consulted EVEN WHEN the unit file itself is at parity; see main().
    """
    dropin_dir = installed_dir / f"{unit_name}.d"
    if not dropin_dir.is_dir():
        return []
    return sorted(p for p in dropin_dir.glob("*.conf") if p.is_file())


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
    return parser


def main(argv: Sequence[str]) -> int:
    """Parse args and run the parity check.

    Returns:
        0 — parity (and at least one unit was actually compared)
        1 — drift, OR a committed unit vanished, OR a drop-in override
        2 — one or more installed units absent

    Drift DOMINATES absence: a run that hits both returns 1.  An absent unit
    is still reported in that case — dominated, not hidden.  A run that
    compared ZERO units never reports parity: see the exit-code table in the
    module docstring.
    """
    args = _build_parser().parse_args(argv)

    installed_dir: pathlib.Path = args.installed_dir
    repo_root: pathlib.Path = args.repo_root
    selected = args.unit or sorted(UNITS)

    drifts: list[tuple[Drift, pathlib.Path, pathlib.Path]] = []
    missing: list[tuple[str, pathlib.Path]] = []
    vanished: list[tuple[str, pathlib.Path]] = []
    overridden: list[tuple[str, list[pathlib.Path]]] = []
    # Units that actually reached compare_unit. The success line reports THIS
    # count, not len(selected): a report may only claim what it verified.
    compared: list[str] = []

    for name in selected:
        repo_path = repo_root / UNITS[name]
        installed_path = installed_dir / name

        if not repo_path.is_file():
            # The committed unit is the source of truth; without it there is
            # nothing to compare against, so this unit was NOT checked.
            vanished.append((name, repo_path))
            continue

        if not installed_path.is_file():
            missing.append((name, installed_path))
            continue

        # Checked even when the unit itself is at parity: a drop-in is layered
        # OVER a matching unit file, so it is invisible to the text
        # comparison below.
        dropins = find_dropins(installed_dir, name)
        if dropins:
            overridden.append((name, dropins))

        compared.append(name)
        for drift in compare_unit(
            name,
            repo_path.read_text(encoding="utf-8"),
            installed_path.read_text(encoding="utf-8"),
        ):
            drifts.append((drift, repo_path, installed_path))

    for name, path in missing:
        _log(f"[skip] {name}: installed unit not found: {path} (not installed on this host)")

    if vanished:
        # Deliberately worded apart from the drift block below. Both exit 1,
        # but they send the operator to different places: a drift is a
        # directive diff to propagate, whereas this is "the file I compare
        # against is gone" — telling them to hunt for a diff would waste the
        # trip.
        _log(
            f"[vanished] {len(vanished)} committed unit(s) not found — "
            "nothing was verified for them:"
        )
        for name, repo_path in vanished:
            _log(f"  {name}: expected committed copy at {repo_path}")
        _log(
            "[vanished] The committed unit is this checker's source of truth. "
            "Check --repo-root, and whether the unit was renamed or moved "
            "(the paths live in UNITS in this script)."
        )

    if overridden:
        # Worded apart from [drift] for the same reason [vanished] is: this is
        # not a directive diff to propagate. The unit files may match exactly;
        # what the run could not establish is that the EFFECTIVE configuration
        # matches, because systemd merges these over the unit at load time.
        # It shares exit 1 because "I could not verify" belongs with "I found
        # a difference", not with the benign "not installed here" that 2
        # denotes — reporting parity here would overstate what was checked.
        _log(
            f"[override] {len(overridden)} unit(s) have drop-in overrides — "
            "the unit files were compared, but the EFFECTIVE configuration "
            "was NOT verified:"
        )
        for name, dropins in overridden:
            for dropin in dropins:
                _log(f"  {name}: {dropin}")
        _log(
            "[override] systemd merges these over the unit at load time, so a "
            "directive set here silently wins over the committed value. "
            "Inspect with: systemctl --user cat <unit>  (and remove the "
            "drop-in, or move the setting into the committed unit)."
        )

    if drifts:
        _log(
            f"[drift] {len(drifts)} directive(s) differ between repo and "
            "installed units:"
        )
        for drift, repo_path, installed_path in drifts:
            _log(f"  {drift.unit} [{drift.section}] {drift.key}")
            _log(f"      {drift.reason}")
            _log(f"      repo      {repo_path}: {drift.repo_value}")
            _log(f"      installed {installed_path}: {drift.installed_value}")
        _log(
            "[drift] To propagate the committed units to this host, run: "
            "scripts/setup-host.sh  (this checker is read-only by design — "
            "see the module docstring for why there is no --fix)"
        )
        # Deliberately NOT phrased as "the installed copy is stale". The
        # report above shows drift, not direction: measured 2026-08-02, some
        # of these units have the correct value on the REPO side
        # (RestartSteps) and others on the INSTALLED side (the ExecStart
        # --config path, where two committed units name config files that do
        # not exist). Propagating blindly would break a running orchestrator.
        _log(
            "[drift] BUT CHECK DIRECTION FIRST: drift does not mean the "
            "installed copy is the stale one. Confirm the committed value is "
            "the one you want on this host before installing — see the KNOWN "
            "RED section of this script's module docstring."
        )

    if drifts or vanished or overridden:
        return 1

    if missing:
        return 2

    if not compared:
        # UNREACHABLE BY CONSTRUCTION, kept as defence — not a tested path.
        # Every unit that was not compared took an earlier `continue` into
        # `vanished` or `missing`, and both return above, so reaching here
        # requires `selected` itself to be empty: impossible while UNITS is
        # non-empty (`args.unit or sorted(UNITS)`) and argparse `choices`
        # rejects an unknown --unit. (The "compared nothing" TEST reaches
        # exit 1 through the `vanished` branch, which is the same invariant
        # enforced one step earlier.) It stays because that invariant — a run
        # that verified nothing must never report parity — is one a future
        # early-`continue` could quietly break, and holding it costs three
        # lines.
        _log("[error] no units were compared — nothing was verified.")
        return 1

    _log(f"[ok] parity — {len(compared)} unit(s) match their committed copies.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
