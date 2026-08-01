"""Parity checker for the installed dashboard systemd units.

Verifies that the three dashboard units committed under ``dashboard/`` still
agree with the copies actually installed in ``~/.config/systemd/user/``:

    dark-factory-dashboard.service
    dark-factory-dashboard-watchdog.service
    dark-factory-dashboard-watchdog.timer

The purpose is narrow and concrete: make it OBSERVABLE when a repo-side unit
change never reaches the running system.  Two such changes exist already —
task 3306 added ``--timeout-graceful-shutdown 8`` / ``--timeout-keep-alive 5``
to the dashboard ExecStart, and task 3308 replaced the watchdog's inline-shell
ExecStart with ``scripts/dashboard-watchdog.py`` plus ``TimeoutStartSec=300``.
A unit edit that stays repo-side is indistinguishable from no edit at all, and
nothing reported that until this check existed.

Exit codes
----------
0 — parity (every compared directive agrees, and at least one unit was
    actually compared)
1 — drift (one or more compared directives disagree) OR a committed unit
    VANISHED (no committed copy found, so nothing could be verified for it)
2 — installed unit absent (no installed copy found for one or more units)

PRECEDENCE: drift (1) DOMINATES absence (2).  With three units a single run can
hit both at once, and returning 2 there would let an unrelated uninstalled unit
mask an actionable finding — ``setup-host.sh`` treats 2 as a benign "not
installed on this host, skipping" and only 1 as something to act on.

A run that compared ZERO units can NEVER report parity.  A vanished committed
unit shares exit 1 with drift rather than minting a third code, for the same
reason: the committed copy is this checker's source of truth, so its absence
means the gate verified nothing — the opposite of the benign "not installed
here" that 2 denotes, and ``setup-host.sh`` already branches on this 0/1/2
vocabulary.  Before that was true, ``--repo-root`` naming a tree with no units
compared nothing and still printed "parity — 3 unit(s) match", so a typo'd
path, a renamed unit or a ``git mv`` of ``dashboard/*.service`` silently
disarmed the whole check.  The success line therefore reports the number of
units actually COMPARED, never the number selected.

Usage
-----
  # verify with defaults (~/.config/systemd/user vs this repo)
  python3 scripts/check_dashboard_unit_parity.py

  # verify explicit trees
  python3 scripts/check_dashboard_unit_parity.py \\
      --installed-dir ~/.config/systemd/user \\
      --repo-root /home/leo/src/dark-factory

  # verify a single unit
  python3 scripts/check_dashboard_unit_parity.py \\
      --unit dark-factory-dashboard-watchdog.timer

What is compared (PRD open question 4, resolved here)
-----------------------------------------------------
Comparison is BOUNDED to a curated per-unit key registry (``UNITS``), with the
committed repo copy as the source of truth.  An unbounded diff would fire on
Description, After and every comment reflow — a false-positive machine, and a
gate nobody believes is worse than no gate.  Within that registry, directives
are split by CLASS, because one rule cannot fit all of them:

- **Value-compared** — host-INVARIANT literals carrying no paths (Type,
  Restart, RestartSec, RestartMaxDelaySec, TimeoutStopSec, TimeoutStartSec,
  StandardOutput/Error, OnBootSec, OnUnitActiveSec, WantedBy).  These can be
  value-compared with no false-positive risk, and value comparison is what
  catches present-but-WRONG — an installed ``TimeoutStopSec=90`` against a
  committed 15 is exactly the failure mode a presence check would wave through.
- **Presence-only** — directives whose value embeds a host path (ExecStart,
  WorkingDirectory, Documentation).  Value-comparing them would report drift
  on every machine that is not this one.  Their meaningful content is reached
  instead through per-flag ExecStart comparison.
- **ExecStart flags** — the uvicorn flags are extracted from the
  continuation-JOINED logical ExecStart of both copies and compared
  individually, so the ``uv`` binary path and repo root are ignored while a
  stale ``--timeout-graceful-shutdown`` is not.
- **Environment=** — compared by variable-NAME set (a dropped variable is
  always drift), with values compared only for names off
  ``DIVERGENCE_ALLOWLIST``.  That allowlist exists because the one measured
  divergence on this host — DASHBOARD_KNOWN_PROJECT_ROOTS, 9 installed roots
  vs 1 committed — is DELIBERATE and documented in the committed unit itself.
  Value-comparing it would fire on every run of a correctly-configured host,
  and a permanently-red gate gets disabled within a week, taking the
  accidental drift it exists to catch with it.  Allowlisting is scoped to a
  variable NAME, not to Environment= as a whole, so blessing the nine-root
  value does not also bless the variable disappearing.

One unit has THREE sites, not two: ``setup-host.sh`` installs
``dark-factory-dashboard.service`` by RENDERING
``scripts/dashboard.service.template`` (``__REPO_ROOT__`` / ``__UV_PATH__``
substitution), and only ``cp``s the two watchdog units verbatim — so the
committed ``dashboard/dark-factory-dashboard.service`` this checker treats as
truth is not the source of the copy it compares against.  The two repo-side
files are held in lockstep by a staleness test in
``tests/scripts/test_check_dashboard_unit_parity.py`` (see the comment on that
registry entry for why ``repo_relpath`` was not simply retargeted at the
template).  Editing one without the other would otherwise leave this checker
reporting drift whose stated remediation — run ``setup-host.sh`` — reinstalls
the same template and changes nothing.

Design notes
------------
- Stdlib-only (argparse, dataclasses, pathlib, re, sys) — runs under plain
  python3, exactly like scripts/check_fused_memory_unit_parity.py, whose
  idioms this script follows deliberately rather than inventing a new pattern.
- No ``--fix``, unlike that precedent.  Propagating repo units into
  ``~/.config/systemd/user/`` and daemon-reloading is already what
  ``scripts/setup-host.sh`` does; duplicating it here would be a fourth copy of
  the install logic.  It would also be actively unsafe today: the repo
  watchdog timer's own comment records that RE-ARMING the installed timer is
  task 3289's job, so a ``--fix`` that installed and reloaded it would
  silently re-arm a watchdog someone deliberately left disarmed — a
  supervision change disguised as a parity fix.  Drift is reported with a
  remediation pointer instead.

Testing note
------------
All drift-logic tests run against ``tmp_path`` fixtures — never the host's
real ``~/.config/systemd/user/`` — mirroring the rule the fused-memory test
module states in its own docstring.  This is not merely for portability: as
measured on 2026-08-01 the installed watchdog service is still the
pre-incident inline-shell copy, so this checker exits 1 against the live host
today.  That is the CORRECT signal (installing the post-3308 units belongs to
task 3289), but a test asserting parity against the live host would be red on
landing, and one asserting drift would flip red the moment 3289 fixes it.
Either encodes host state rather than checker behaviour.
"""

import argparse
import dataclasses
import pathlib
import re
import sys
from typing import Sequence

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Prefixed onto every line this script prints, so a parity report is greppable
# the same way `journalctl --user -t dashboard-watchdog` makes the watchdog's
# decisions greppable.  This is also the token `metadata.delivered_checks`
# greps for over scripts/ (dashboard_unit_parity|DASHBOARD_UNIT_PARITY).
LOG_TAG = "dashboard_unit_parity"

# Environment variable NAMES whose VALUE is permitted to differ between the
# committed unit and the installed one, each with the documented reason it is
# blessed.  Presence is still required — see the Environment= branch of
# compare_unit: allowlisting a value must never bless the variable vanishing.
#
# THIS IS A HOLE IN THE GATE.  Keep it small, and keep every entry's reason
# specific enough that a reviewer can check it.  The gate's whole value comes
# from being believable when it fires.
DIVERGENCE_ALLOWLIST: dict[str, str] = {
    "DASHBOARD_KNOWN_PROJECT_ROOTS": (
        "Cost/burndown aggregation roots. The committed unit's own comment "
        "declares the divergence deliberate: 'additional project roots are "
        "LOCAL settings, added to the installed unit, not committed here'. "
        "Measured 2026-08-01: the installed unit carries 9 roots, the "
        "committed one carries this repo only. Value-comparing it would "
        "therefore report drift on every run of a correctly-configured host, "
        "and a gate that is always red gets switched off — taking the "
        "accidental drift it exists to catch with it."
    ),
}

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_DEFAULT_INSTALLED_DIR = pathlib.Path.home() / ".config" / "systemd" / "user"


def _log(message: str, *, stream=None) -> None:
    """Print *message* prefixed with the log tag."""
    print(f"[{LOG_TAG}] {message}", file=stream if stream is not None else sys.stdout)


# ---------------------------------------------------------------------------
# Unit parser
# ---------------------------------------------------------------------------


def _join_continuations(text: str) -> list[str]:
    """Return *text*'s lines with backslash continuations joined into one line.

    While a line ends in ``\\``, the backslash is dropped and the NEXT line's
    stripped form is appended after a single space.  Mirrors ``_logical_exec_start``
    in tests/scripts/test_dashboard_service_template.py, generalised from "the
    ExecStart line" to "every line".

    Joining happens BEFORE comment classification, which matches systemd's own
    behaviour: a comment line ending in ``\\`` continues, and its continuation
    is part of the comment.  Both real dashboard units rely on this — the
    watchdog service quotes the old inline-shell ExecStart across two ``#``
    lines joined by a backslash, and classifying first would leave the second
    half of that quote looking like a directive.
    """
    joined: list[str] = []
    pending: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        continued = line.endswith("\\")
        if continued:
            line = line[:-1].rstrip()
        piece = line if pending is None else f"{pending} {line.strip()}"
        if continued:
            pending = piece
            continue
        joined.append(piece)
        pending = None
    if pending is not None:
        # Trailing backslash on the final line — keep what we have rather than
        # silently dropping the directive.
        joined.append(pending)
    return joined


def parse_unit_directives(text: str) -> dict[str, dict[str, list[str]]]:
    """Parse a systemd unit into ``{section: {key: [value, ...]}}``.

    Classification rules are taken verbatim from the precedent's
    ``parse_unit_sections`` (scripts/check_fused_memory_unit_parity.py):

    - ``[X]`` opens section X.
    - Lines starting with ``#`` or ``;`` are comments — skipped.
    - Blank lines are skipped.
    - Lines before the first section header are DROPPED, not attributed.

    Two deliberate divergences from that precedent, each required here:

    1. **key → values LIST, not a flat line list.**  This checker compares
       directives BY KEY, which a flat list of lines cannot express, and it
       needs the several ``Environment=`` lines of a unit addressable as a
       group rather than as unrelated strings.
    2. **Backslash continuations are JOINED.**  ``parse_unit_sections``
       documents that it does not join them, which is harmless for its exact
       whole-line membership checks.  It is fatal here: the dashboard
       ExecStart spans four physical lines, so without joining every uvicorn
       flag task 3306 added lives on a line the parser never associates with
       ``ExecStart`` — the checker would report parity on a command it never
       actually read.

    Each surviving line is split on the FIRST ``=`` only, so
    ``Environment=A=1`` yields key ``Environment`` and value ``A=1``.  A line
    with no ``=`` is skipped (systemd has no valueless directives).
    """
    sections: dict[str, dict[str, list[str]]] = {}
    current: str | None = None
    for joined_line in _join_continuations(text):
        line = joined_line.strip()
        if not line:
            continue
        if line.startswith("#") or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            sections.setdefault(current, {})
            continue
        if current is None:
            continue
        key, sep, value = line.partition("=")
        if not sep:
            continue
        sections[current].setdefault(key.strip(), []).append(value.strip())
    return sections


# ---------------------------------------------------------------------------
# Drift records and unit specs
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


@dataclasses.dataclass(frozen=True)
class UnitSpec:
    """What to compare for one unit, and where its repo copy lives.

    Expected VALUES are never stated here — they are read from the committed
    repo unit at run time.  Only the KEY registry is curated.  Restating the
    values (the way the fused-memory precedent's REQUIRED_SERVICE_DIRECTIVES
    does) would create a THIRD site that must agree with the repo unit and the
    installed unit — reintroducing, in the tool built to close it, exactly the
    lockstep duplication this check exists to catch.  Worse, it would defeat
    the purpose: when a future change edits TimeoutStopSec in the repo unit, a
    stale literal would keep passing against the OLD value on both sides, and
    the change would again fail to reach the running system with nothing
    reporting it.
    """

    name: str
    repo_relpath: str
    # (section, key) pairs whose VALUES must agree. Host-INVARIANT literals
    # only — anything carrying a host path belongs on present_only.
    compared: tuple[tuple[str, str], ...] = ()
    # (section, key) pairs whose PRESENCE must agree but whose values cannot
    # be compared. ExecStart, WorkingDirectory and Documentation all embed
    # absolute host paths (/home/leo/.local/bin/uv, the repo root), so
    # value-comparing them would report drift on every machine that is not
    # this one — a gate that fires unconditionally is a gate that gets
    # switched off. Their MEANINGFUL content is still reached, just not by
    # string equality: exec_start_flags compares the uvicorn flags inside
    # ExecStart individually, ignoring the host-specific prefix.
    present_only: tuple[tuple[str, str], ...] = ()
    # Section whose Environment= directives are compared by variable-NAME set
    # (values compared only for names off DIVERGENCE_ALLOWLIST). None disables
    # the branch entirely, which is right for the two watchdog units — they
    # declare no Environment= at all.
    environment_section: str | None = None
    # Bare uvicorn flag names (no leading '--') compared INSIDE [Service]
    # ExecStart. This is how a presence-only ExecStart still gets its
    # meaningful content checked: the host-specific `uv` path and repo-root
    # prefix are ignored, while a stale --timeout-graceful-shutdown is not.
    exec_start_flags: tuple[str, ...] = ()


def _render(values: list[str] | None) -> str:
    """Render a directive's values for a Drift record / the report."""
    if not values:
        return _ABSENT
    if len(values) == 1:
        return values[0]
    return " | ".join(values)


def _exec_start_flag(service_directives: dict[str, list[str]], flag: str) -> str | None:
    """Return the argument of ``--<flag>`` in ExecStart, or None if absent.

    Reads the ALREADY-JOINED ExecStart value out of a parsed section — never
    the raw file text.  That scoping is load-bearing, and the hazard is real
    rather than hypothetical: both dashboard units discuss ``--timeout-keep-alive``
    and ``--timeout-graceful-shutdown`` in the explanatory comment block
    directly above ExecStart.  A whole-file regex would therefore keep
    reporting a value after the flag had actually been deleted from the
    command — reporting parity on a unit that lost the very flag being
    checked.  Comments are already gone from the parse, so the correct
    behaviour is inherited for free.

    Both click spellings are accepted (``--flag value`` and ``--flag=value``):
    uvicorn's parser treats them identically, so rejecting one would report a
    reformatted-but-correct command as a missing flag.

    Returns the RAW token, not an int (unlike ``_uvicorn_int_flag`` in
    tests/scripts/test_dashboard_service_template.py, whose idiom this
    borrows), so non-numeric flags like ``--host 127.0.0.1`` work too.
    """
    values = service_directives.get("ExecStart")
    if not values:
        return None
    command = " ".join(values)
    match = re.search(rf"--{re.escape(flag)}[=\s]+(\S+)", command)
    return match.group(1) if match else None


def _compare_exec_start_flags(
    spec: UnitSpec,
    repo: dict[str, dict[str, list[str]]],
    installed: dict[str, dict[str, list[str]]],
) -> list[Drift]:
    """Compare each of ``spec.exec_start_flags`` inside [Service] ExecStart."""
    drifts: list[Drift] = []
    repo_service = repo.get("Service", {})
    installed_service = installed.get("Service", {})

    for flag in spec.exec_start_flags:
        repo_value = _exec_start_flag(repo_service, flag)
        installed_value = _exec_start_flag(installed_service, flag)
        if repo_value == installed_value:
            continue
        if repo_value is None:
            reason = f"--{flag} present on the installed command, absent from the repo command"
        elif installed_value is None:
            reason = f"--{flag} present on the repo command, absent from the installed command"
        else:
            reason = f"--{flag} argument differs between the repo and installed commands"
        drifts.append(
            Drift(
                unit=spec.name,
                section="Service",
                # Keyed by FLAG, not by 'ExecStart': the command is long and
                # mostly host-specific, so naming the whole directive would
                # send the operator back to a manual diff to find the token
                # that actually moved.
                key=f"ExecStart --{flag}",
                repo_value=repo_value if repo_value is not None else _ABSENT,
                installed_value=installed_value if installed_value is not None else _ABSENT,
                reason=reason,
            )
        )

    return drifts


def _environment_map(
    directives: dict[str, dict[str, list[str]]],
    section: str,
) -> dict[str, str]:
    """Return ``{VAR: value}`` for every ``Environment=`` line in *section*.

    Each value is split on its FIRST ``=`` — ``Environment=A=b=c`` sets A to
    ``b=c``.  A later occurrence of the same variable wins, matching systemd,
    which applies the directives in file order.
    """
    env: dict[str, str] = {}
    for assignment in directives.get(section, {}).get("Environment", []):
        name, sep, value = assignment.partition("=")
        if not sep:
            continue
        env[name.strip()] = value
    return env


def _compare_environment(
    spec: UnitSpec,
    repo: dict[str, dict[str, list[str]]],
    installed: dict[str, dict[str, list[str]]],
) -> list[Drift]:
    """Compare ``Environment=`` by variable-NAME set, then by value.

    Two rules, and the split between them is the whole point:

    - The SET of variable names must agree.  A variable declared on one side
      and not the other is drift regardless of the allowlist — the allowlist
      says "this variable's VALUE is a local setting", never "this variable is
      optional".  An installed unit that dropped
      DASHBOARD_KNOWN_PROJECT_ROOTS entirely would silently lose every
      aggregation root, which is a real regression, not a local preference.
    - For names present in BOTH, values must agree unless the name is on
      DIVERGENCE_ALLOWLIST.

    Drifts are keyed ``Environment=<VAR>`` so the report names the offending
    variable rather than the directive class — with several Environment= lines
    in a unit, "Environment differs" would send the operator back to a diff.
    """
    section = spec.environment_section
    if section is None:
        return []

    repo_env = _environment_map(repo, section)
    installed_env = _environment_map(installed, section)
    drifts: list[Drift] = []

    for name in sorted(set(repo_env) | set(installed_env)):
        in_repo = name in repo_env
        in_installed = name in installed_env
        if in_repo and in_installed:
            if name in DIVERGENCE_ALLOWLIST or repo_env[name] == installed_env[name]:
                continue
            reason = "environment variable value differs (not on DIVERGENCE_ALLOWLIST)"
        elif in_repo:
            reason = "environment variable declared in the repo copy, absent from the installed copy"
        else:
            reason = "environment variable declared in the installed copy, absent from the repo copy"
        drifts.append(
            Drift(
                unit=spec.name,
                section=section,
                key=f"Environment={name}",
                repo_value=repo_env.get(name, _ABSENT),
                installed_value=installed_env.get(name, _ABSENT),
                reason=reason,
            )
        )

    return drifts


def compare_unit(
    spec: UnitSpec,
    repo_text: str,
    installed_text: str,
) -> list[Drift]:
    """Return every drift between the repo and installed copies of *spec*.

    Comparison is SYMMETRIC over the curated key set: a compared directive the
    installed copy declares and the repo copy does not is drift just as much
    as the reverse.  A missing key is treated as ``_ABSENT`` on its side, so
    both asymmetric cases fall out of the same equality test.

    The full values LIST is compared, not just the first value, so a repeated
    directive that gained or lost an occurrence is caught — systemd applies
    every occurrence, so the checker must see every occurrence.

    ``spec.present_only`` keys are checked for PRESENCE only: a drift is
    emitted when one copy declares the directive and the other does not, never
    on a value difference.  See UnitSpec.present_only for why.
    """
    repo = parse_unit_directives(repo_text)
    installed = parse_unit_directives(installed_text)
    drifts: list[Drift] = []

    for section, key in spec.compared:
        repo_values = repo.get(section, {}).get(key)
        installed_values = installed.get(section, {}).get(key)
        if repo_values == installed_values:
            continue
        if repo_values is None:
            reason = "declared in the installed copy, absent from the repo copy"
        elif installed_values is None:
            reason = "declared in the repo copy, absent from the installed copy"
        else:
            reason = "value differs between the repo copy and the installed copy"
        drifts.append(
            Drift(
                unit=spec.name,
                section=section,
                key=key,
                repo_value=_render(repo_values),
                installed_value=_render(installed_values),
                reason=reason,
            )
        )

    for section, key in spec.present_only:
        repo_present = key in repo.get(section, {})
        installed_present = key in installed.get(section, {})
        if repo_present == installed_present:
            continue
        drifts.append(
            Drift(
                unit=spec.name,
                section=section,
                key=key,
                repo_value=_render(repo.get(section, {}).get(key)),
                installed_value=_render(installed.get(section, {}).get(key)),
                reason=(
                    "required directive absent from the installed copy"
                    if repo_present
                    else "required directive absent from the repo copy"
                ),
            )
        )

    drifts.extend(_compare_exec_start_flags(spec, repo, installed))
    drifts.extend(_compare_environment(spec, repo, installed))

    return drifts


# ---------------------------------------------------------------------------
# The unit registry
# ---------------------------------------------------------------------------

# Every entry names WHY a key is on its list. Description= and After= are
# deliberately absent from all three: they are cosmetic, they legitimately
# differ (the installed timer's Description predates the incident rewrite),
# and comparing them would spend the gate's credibility on nothing.
#
# Registry KEYS are curated; expected VALUES are not — they are read from the
# committed unit at run time. See UnitSpec's docstring for why. The keys are
# guarded against rot by the staleness tests in
# tests/scripts/test_check_dashboard_unit_parity.py, which assert every key
# listed here is genuinely declared in the committed unit; a key that is not
# would compare absent-to-absent and check nothing, forever, while still
# reporting green.
UNITS: dict[str, UnitSpec] = {
    "dark-factory-dashboard.service": UnitSpec(
        name="dark-factory-dashboard.service",
        # THREE SITES, NOT TWO — read this before editing either file.
        # setup-host.sh does not `cp` this unit the way it does the two
        # watchdog units. It RENDERS the installed copy from
        # scripts/dashboard.service.template (setup-host.sh:362-367,
        # substituting __REPO_ROOT__ / __UV_PATH__), so the path below is NOT
        # the source of the copy this checker compares against. Edit one of
        # the two repo-side files and you must edit the other; they are held
        # in lockstep by
        # tests/scripts/test_check_dashboard_unit_parity.py::
        # test_committed_dashboard_unit_agrees_with_the_installed_template,
        # which fails the moment they diverge.
        #
        # WHY repo_relpath still points at the committed unit rather than at
        # the template: every value-compared key here is a host-invariant
        # literal, so pointing at the template would additionally require
        # normalising __REPO_ROOT__/__UV_PATH__ before every comparison to
        # keep present_only and the ExecStart-prefix territory honest — and it
        # would leave dashboard/dark-factory-dashboard.service an unchecked
        # orphan. Guarding the pair with the same staleness mechanism already
        # used against key rot is the smaller change and closes the same hole.
        repo_relpath="dashboard/dark-factory-dashboard.service",
        compared=(
            # Restart policy — the task's named minimum coverage. All
            # host-invariant literals: an installed copy that drifted here
            # would change availability behaviour with nothing reporting it.
            ("Service", "Type"),
            ("Service", "Restart"),
            ("Service", "RestartSec"),
            ("Service", "RestartMaxDelaySec"),
            # 15 is sized against uvicorn's own 8s drain bound (see the unit's
            # comment); a drifted value silently re-opens the SIGKILL window
            # that produced the ~16s dead restarts.
            ("Service", "TimeoutStopSec"),
            # Without these, `journalctl --user -u dark-factory-dashboard`
            # goes quiet — the failure is invisible rather than absent.
            ("Service", "StandardOutput"),
            ("Service", "StandardError"),
            ("Install", "WantedBy"),
        ),
        present_only=(
            # Both carry absolute host paths (/home/leo/.local/bin/uv, the
            # repo root). Presence only; content reached via exec_start_flags.
            ("Service", "ExecStart"),
            ("Service", "WorkingDirectory"),
        ),
        environment_section="Service",
        exec_start_flags=(
            # The two flags task 3306 added. Comparing them is what makes
            # "the unit change reached the running system" a checkable claim.
            "timeout-graceful-shutdown",
            "timeout-keep-alive",
            # The bind address is a contract with the watchdog probe and the
            # reverse proxy; a drifted port means a healthy dashboard nothing
            # can reach.
            "host",
            "port",
        ),
    ),
    "dark-factory-dashboard-watchdog.service": UnitSpec(
        name="dark-factory-dashboard-watchdog.service",
        repo_relpath="dashboard/dark-factory-dashboard-watchdog.service",
        compared=(
            # oneshot is load-bearing: every timer tick must be a FRESH
            # process, which is why the failure streak is persisted to disk
            # rather than held in memory.
            ("Service", "Type"),
            # 3308's bound on the WHOLE tick. systemd disables TimeoutStartSec
            # for Type=oneshot by default, and the timer's OnUnitActiveSec
            # measures from this unit's last activation — so a tick that never
            # returns is not a slow tick, it is the END of supervision, with
            # nothing saying so. Its absence from the installed copy is
            # precisely the drift measured on this host.
            ("Service", "TimeoutStartSec"),
            ("Service", "StandardOutput"),
            ("Service", "StandardError"),
        ),
        # Absolute path to scripts/dashboard-watchdog.py — host-specific.
        # No [Install] entry: this unit deliberately has no [Install] section
        # (it is activated by the timer, never enabled directly).
        present_only=(("Service", "ExecStart"),),
    ),
    "dark-factory-dashboard-watchdog.timer": UnitSpec(
        name="dark-factory-dashboard-watchdog.timer",
        repo_relpath="dashboard/dark-factory-dashboard-watchdog.timer",
        compared=(
            ("Timer", "OnBootSec"),
            # The cadence is load-bearing, not a free knob: the watchdog needs
            # FAIL_STREAK (=3) consecutive failed probes, so 3 x 30s sets the
            # ~90s sustained-outage detection latency. A drifted interval
            # changes that latency in the same proportion, silently.
            ("Timer", "OnUnitActiveSec"),
            # Dropping [Install] would disarm the watchdog while looking like
            # a fix — the exact hazard the committed unit's own comment warns
            # about.
            ("Install", "WantedBy"),
        ),
    ),
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str]) -> int:
    """Parse args and run the parity check.

    Returns:
        0 — parity (and at least one unit was actually compared)
        1 — drift, OR a committed unit vanished
        2 — one or more installed units absent

    Drift DOMINATES absence: a run that hits both returns 1.  An absent unit
    is still reported in that case — dominated, not hidden.  A run that
    compared ZERO units never reports parity: see the exit-code table in the
    module docstring.
    """
    parser = argparse.ArgumentParser(
        description="Verify parity between the in-repo and installed dashboard units."
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
    args = parser.parse_args(argv)

    installed_dir: pathlib.Path = args.installed_dir
    repo_root: pathlib.Path = args.repo_root
    selected = args.unit or sorted(UNITS)

    drifts: list[tuple[Drift, pathlib.Path, pathlib.Path]] = []
    missing: list[pathlib.Path] = []
    vanished: list[tuple[str, pathlib.Path]] = []
    # Units that actually reached compare_unit. The success line reports THIS
    # count, not len(selected): a report may only claim what it verified.
    compared: list[str] = []

    for name in selected:
        spec = UNITS[name]
        repo_path = repo_root / spec.repo_relpath
        installed_path = installed_dir / name

        if not repo_path.is_file():
            # The committed unit is the source of truth; without it there is
            # nothing to compare against, so this unit was NOT checked.
            vanished.append((name, repo_path))
            continue

        if not installed_path.is_file():
            missing.append(installed_path)
            continue

        compared.append(name)
        for drift in compare_unit(
            spec,
            repo_path.read_text(encoding="utf-8"),
            installed_path.read_text(encoding="utf-8"),
        ):
            drifts.append((drift, repo_path, installed_path))

    for path in missing:
        _log(f"[skip] installed unit not found: {path} (not installed on this host)")

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

    if drifts:
        _log(f"[drift] {len(drifts)} directive(s) differ between repo and installed units:")
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

    if drifts or vanished:
        return 1

    if missing:
        return 2

    if not compared:
        # Belt and braces on the return-0 path ONLY: every path that compared
        # nothing for a KNOWN reason has already returned above (1 for a
        # vanished committed unit, 2 for "not installed on this host"), so
        # reaching here means we compared nothing for no stated reason. A run
        # that verified nothing must not report parity.
        _log("[error] no units were compared — nothing was verified.")
        return 1

    _log(f"[ok] parity — {len(compared)} unit(s) match their committed copies.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
