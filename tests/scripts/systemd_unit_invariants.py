"""Shared systemd unit invariants, imported by more than one test module.

This module holds NO test functions of its own.  It exists because the
restart-backoff invariant below is applied from two places — the dashboard
unit/template suite (tests/scripts/test_dashboard_service_template.py, which
also owns the helper's negative-case guard) and the fleet-wide sweep
(tests/scripts/test_systemd_restart_backoff.py) — and duplicating it into
both is how the two copies drift until one silently stops catching the
defect.  Written for task 3333, lifted here by task 3408.  The same
reasoning brings the effective ExecStart= command parse here, together with
the discovery scope the content-discovered sweeps share.

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

import pytest

# ---------------------------------------------------------------------------
# Unit-file section parsing, and the orchestrator template glob
#
# Lifted here by task 3746, when the setup-host.sh installer suite and the
# SETUP.md operator-remediation suite moved out of
# tests/scripts/test_orchestrator_service_files.py into
# tests/scripts/test_setup_host_unit_installation.py.  Both helpers were
# private to that module; the split puts a consumer on BOTH sides of it — the
# per-unit shape suite that stayed, and the installer / operator-doc suites
# that left — which is exactly this module's admission criterion.
#
# Duplicating them into the new module instead is what the origin module's own
# docstring argued against, and for parse_sections it would have produced the
# THIRD hand-copy in this directory: tests/scripts/test_orchestrator_watchdog.py
# carried a private _unit_sections copy of the same six lines at the time of
# this decision. Task 3913 later retired that copy in favour of parse_sections.
#
# Neither helper carries a test function here (this module holds none); their
# guards stay in test_orchestrator_service_files.py, which still owns
# test_orchestrator_service_glob_covers_all_known_units for the glob and
# exercises parse_sections through the [Unit]/[Service] placement suite.
# ---------------------------------------------------------------------------

REPO_ROOT = pathlib.Path(__file__).parents[2]


def parse_sections(content: str) -> dict[str, list[str]]:
    """Split unit-file text into {section_name: [lines]} (header line excluded)."""
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in content.splitlines():
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    return sections


# Discovered by glob rather than hand-listed, so a new orchestrator template is
# covered by every consuming guard the day it lands.  The counterpart guard
# test_orchestrator_service_glob_covers_all_known_units pins the glob against a
# known-basename set, so a glob that silently matched nothing cannot make the
# suites that iterate this list pass vacuously.
ALL_ORCHESTRATOR_SERVICE_FILES = sorted(
    (REPO_ROOT / "scripts").glob("orchestrator-*.service")
)


# ---------------------------------------------------------------------------
# Installed-unit location
#
# Where setup-host.sh actually writes.  Lifted here by task 3763 from
# tests/scripts/test_know_live_installed_unit_parity.py (task 3642) when
# tests/scripts/test_pump_web_ui_installed_unit_parity.py became the second
# host-coupled parity module and would otherwise have copied all three
# symbols verbatim — the same trigger condition, and the same reasoning, that
# brought systemctl_user_show here.  UNIT_DIR is the sharpest case for
# single-sourcing: it mirrors a path in another language's file, so every
# additional copy is another chance to mis-mirror, and a mis-mirror does not
# fail — it degrades to require_installed_unit() skipping, i.e. a guard that
# silently checks nothing, which is the exact failure mode the mirroring
# exists to prevent.
# ---------------------------------------------------------------------------

# Mirrors scripts/setup-host.sh:114 (`UNIT_DIR="$HOME/.config/systemd/user"`)
# exactly.  Deliberately NOT XDG_CONFIG_HOME-aware: the installer itself does
# not honour that variable, so making these guards honour it would only ever
# point them at a directory setup-host.sh never writes to — the unit would not
# be found there, require_installed_unit would skip, and the guard would
# silently check nothing while the real installed unit (at
# $HOME/.config/systemd/user, unconditionally) drifted.  Mirroring the
# installer's actual, non-configurable path is the whole point of a parity
# guard.
INSTALLED_UNIT_DIR = pathlib.Path.home() / ".config" / "systemd" / "user"

SYSTEMCTL_SKIP_REASON = (
    "systemctl is not installed; this test requires a live systemd --user "
    "manager and has no fixture-based fallback (see module docstring)"
)


def require_installed_unit(basename: str) -> pathlib.Path:
    """Return INSTALLED_UNIT_DIR/*basename*, or skip if it is absent on this host.

    A fresh checkout or a CI runner has no ~/.config/systemd/user at all —
    that is an environment fact, not a defect in the unit, so this skips
    rather than fails.
    """
    path = INSTALLED_UNIT_DIR / basename
    if not path.exists():
        pytest.skip(
            f"{path} does not exist on this host (fresh checkout or CI "
            "runner with no installed orchestrator units)"
        )
    return path


# ---------------------------------------------------------------------------
# The `--config` argument of an ExecStart=
#
# LIFT TRIGGER: a SECOND consumer appeared.  The canonical parser lived in
# tests/scripts/test_orchestrator_service_files.py, task 3642 hand-copied it
# (and CANONICAL_CONFIG_BASENAME) into test_know_live_installed_unit_parity.py,
# and task 3773 lifted both here — same trigger and same reasoning that brought
# systemctl_user_show here under task 3763, except that these two copies had
# ALREADY drifted, answering a dangling `--config` two different ways.
#
# The single reconciled contract that resolved that drift is stated ONCE, on
# config_arg_from_exec_start below; every other site in this directory points
# at that docstring rather than restating it, because prose copies drift the
# same way code copies do and nothing keeps them in step.
#
# WHAT DID NOT MOVE, and why: `_argv_from_exec_start_show` (systemctl struct
# -> argv) stayed in test_know_live_installed_unit_parity.py.  It still has
# exactly ONE consumer, and this module's lift trigger is a second consumer,
# not proximity or tidiness: lifting a single-consumer helper buys no
# de-duplication while widening this module's surface.  Its negative-case
# guard stays with it, per the same convention that kept systemctl_user_show's
# where it was written.
# ---------------------------------------------------------------------------

# CLAUDE.md makes `<project_root>/dark-factory-orchestrator.yaml` the
# canonical, REQUIRED filename: the dashboard's escalation-URL discovery
# (`_discover_escalation_urls`) keys on that exact string, and legacy spellings
# (`orchestrator.yaml`, `orchestrator-config.yaml`, `orchestrator/config.yaml`)
# are honoured only as a discovery fallback for not-yet-migrated projects,
# never as a supported choice for new ones.
CANONICAL_CONFIG_BASENAME = "dark-factory-orchestrator.yaml"


class MalformedExecStart(ValueError):
    """A unit's ExecStart= is broken — never a legitimate "no --config" answer.

    Kept distinct from the ``None`` return below so the two can be handled
    oppositely: ``None`` means "this unit takes no ``--config``" and callers
    SKIP on it, while this class means the unit (or the parser) is defective
    and must FAIL.  Which inputs land where is the contract on
    config_arg_from_exec_start below, stated there once.

    Also raised by the OTHER half of a parse — locating the ExecStart= text
    before the scan sees it (logical_exec_start below) — so a broken unit
    surfaces as one class whichever layer notices it first.
    """


def config_arg_from_exec_start(
    exec_start_value: str, unit_name: str = "<unit>"
) -> str | None:
    """Return the `--config` argument in *exec_start_value*, or None if absent.

    THE CONTRACT, stated here ONCE — every other site points at this docstring.

    None means exactly ONE thing: the string carries no ``--config`` flag at
    all.  That is a real answer, not a parse failure — orchestrator-watchdog.
    service runs a probe script that takes no ``--config`` — and callers SKIP
    on it, so the watchdog is not dragged into an invariant that does not apply
    to it.  Every OTHER way of producing no value raises MalformedExecStart: a
    dangling ``--config`` as the final token, and the ``--config=`` spelling
    with an empty value.  Both describe a unit that would start the
    orchestrator with no config path at all, and collapsing either into None is
    how a guard waves through the drift it exists to catch.  Verified before
    the two copies were reconciled onto this contract: every committed unit
    uses the space-separated form with a real path, so tightening moved no live
    verdict — only the failure text.  (logical_exec_start below, which LOCATES
    the ExecStart= text first, owns the third no-value case, a unit with no
    usable ExecStart= line, and raises the same class for the same reason.)

    *exec_start_value* may be a whole ``ExecStart=`` line, just its value, or
    the ``argv[]=`` segment of a ``systemctl show`` struct: the scan looks only
    for ``--config`` tokens and is prefix-agnostic.  That looseness is not
    laxity — the call sites genuinely hold different shapes (a command value,
    a ``systemctl show`` ``argv[]=`` segment), and normalising at the boundary
    would have meant a wrapper or a copy per shape.  *unit_name* is pure
    diagnostics, interpolated into both raises so the caller's context (a
    unit path, or a ``systemctl --user show ...`` provenance string) survives
    into the failure; the messages say "command line" rather than "ExecStart=
    line" precisely because two of those three accepted shapes are not one.
    """
    tokens = exec_start_value.split()
    for i, token in enumerate(tokens):
        if token == "--config":
            if i + 1 >= len(tokens):
                raise MalformedExecStart(
                    f"{unit_name}: `--config` is the last token of the command "
                    "line inspected, with no value after it. The orchestrator "
                    "would start with no config path at all. Command line "
                    f"inspected: {' '.join(tokens)!r}"
                )
            return tokens[i + 1]
        if token.startswith("--config="):
            value = token.split("=", 1)[1]
            if not value:
                raise MalformedExecStart(
                    f"{unit_name}: `--config=` carries an empty value. The "
                    "orchestrator would start with no config path at all — the "
                    "same defect as a dangling `--config`, in the other "
                    f"spelling. Command line inspected: {' '.join(tokens)!r}"
                )
            return value
    return None


# ---------------------------------------------------------------------------
# The effective ExecStart= command
#
# Read by tests/scripts/test_orchestrator_service_files.py::
# _exec_start_config_arg, tests/scripts/test_dashboard_service_template.py::
# _uvicorn_int_flag and tests/scripts/test_uv_run_venv_isolation.py::
# discover_uv_run_units.  Its home is here because those three carried three
# disagreeing copies.  Its negative-case guard lives in
# test_orchestrator_service_files.py's fixture-string section.
# ---------------------------------------------------------------------------

# The trailing `=` keeps ExecStartPre= out; whitespace around the `=` is legal
# systemd.syntax.  A discovery grep must use this SAME anchor as the parser
# (tests/scripts/test_uv_run_venv_isolation.py::discover_exec_start_files), or
# a unit it discovers is then reported as declaring no ExecStart= at all.
EXEC_START_PREFIX = r"ExecStart[ \t]*="
_EXEC_START_RE = re.compile(rf"^{EXEC_START_PREFIX}")


def logical_exec_start(text: str, unit_name: str = "<unit>") -> str:
    """Return the effective ExecStart= COMMAND in unit *text* as one logical line.

    LAST OCCURRENCE WINS, as in systemd and restart_directive below: a drop-in
    under <unit>.d/ merges by appending, so an override lands as an empty
    ``ExecStart=`` list RESET followed by the real command.  A first-match read
    answers about the reset, or about the overridden command — either way a
    command systemd never runs.

    CONTINUATIONS ARE JOINED: the ExecStart= of scripts/dashboard.service.
    template and scripts/fused-memory.service.template spans several physical
    lines, and an unjoined read sees one fragment — passing a flag check
    vacuously until the day that flag moves to a continuation line.

    Returns the command WITHOUT the directive prefix.  Raises MalformedExecStart
    when *text* has no ExecStart= at all, or when the effective one carries no
    command: neither is a legitimate "this command lacks X" answer, the
    None-vs-raise split config_arg_from_exec_start's contract states.
    *unit_name* is diagnostics only, named in both raises.
    """
    lines = text.splitlines()
    start_indices = [
        i for i, ln in enumerate(lines) if _EXEC_START_RE.match(ln.strip())
    ]
    if not start_indices:
        raise MalformedExecStart(
            f"{unit_name} declares no ExecStart= line, so there is no command "
            "to inspect. Treating this as an answer would silently drop the "
            "unit out of whichever guard asked."
        )

    parts: list[str] = []
    idx = start_indices[-1]
    while True:
        line = lines[idx].strip()
        continued = line.endswith("\\")
        if continued:
            line = line[:-1]
        parts.append(line.strip())
        if not continued or idx + 1 >= len(lines):
            break
        idx += 1

    command = _EXEC_START_RE.sub("", " ".join(p for p in parts if p), count=1).strip()
    if not command:
        raise MalformedExecStart(
            f"{unit_name}'s effective ExecStart= carries no command: the last "
            "assignment is a list RESET with nothing appended after it, so "
            "systemd has no command to run at all. Treating this as an answer "
            "would silently drop a unit that cannot start out of whichever "
            "guard asked."
        )
    return command


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
# Unit discovery scope
#
# Shared by the two content-discovered sweeps:
# tests/scripts/test_systemd_restart_backoff.py::
# discover_units_declaring_a_restart_cap and
# tests/scripts/test_uv_run_venv_isolation.py::discover_exec_start_files.
# ---------------------------------------------------------------------------

# What discovery refuses to treat as a unit, excluded by CATEGORY rather than
# by naming individual paths.  Both categories are files that CONTAIN a unit as
# quoted text rather than files systemd can load, and both break the sweeps
# the same way: restart_directive and logical_exec_start above are
# last-occurrence-wins FILE-WIDE, so on a file holding more than one embedded
# unit they splice a directive out of one and a directive out of another and
# report a verdict about neither.
#
#   **/tests/**  — parity suites embed whole units as column-0 triple-quoted
#     fixtures.  Measured in test_systemd_restart_backoff.py's sweep:
#     tests/scripts/test_check_fused_memory_unit_parity.py was swept as a 13th
#     "unit" and PASSED by accident, splicing a RestartMaxDelaySec= cap out of
#     the NEGATIVE fixture (which deliberately models the defect) together with
#     RestartSteps= out of an unrelated POSITIVE one.  The glob form is
#     load-bearing: a plain `:!tests/` excludes only the top-level directory and
#     leaves fused-memory/tests/, orchestrator/tests/, scripts/tests/ and
#     dashboard/tests/ swept — and fused-memory/tests/test_systemd_unit_config.py
#     already parses systemd units, so one fixture there gaining a column-0
#     directive either sweep anchors on would drag a .py file back in.
#
#   **/*.md — prose.  A doc may legitimately show the DEFECT a sweep guards
#     against: a PRD or postmortem quoting the defective unit next to the fixed
#     one, and no mechanical rule distinguishes a cautionary example from a
#     prescription.  For test_systemd_restart_backoff.py that "before" fence is
#     a cap with no RestartSteps=; for test_uv_run_venv_isolation.py it is a
#     `uv run` missing run-level `--no-sync`, or carrying a stale `--frozen`
#     beside it.  plans/afk-C1-systemd.md is the live instance — an as-built
#     record of what was deployed, already diverged from the fleet in three
#     visible ways (`Requires=fused-memory.service`, which the real units reject
#     and test_orchestrator_service_files.py asserts is ABSENT; an obsolete
#     `--config orchestrator/config.yaml`; no `--no-sync`).
#     Editing a directive inside it would falsify the record without making any
#     unit correct.  Excluding the category rather than the path means the next
#     doc quoting a unit does not turn CI red and does not have to be
#     hand-added to a constant in a test file.
#
# The cost is that a doc which IS a copy-source for real units must opt back in
# explicitly: FACTORY_INIT_REFERENCE below.
NON_UNIT_PATHSPECS = (
    ":(exclude,glob)**/tests/**",
    ":(exclude,glob)**/*.md",
)

# The copy-source new projects' supervised units are minted from, and the only
# markdown file each sweep opts back in.  Each consumer guards it
# UNCONDITIONALLY, i.e. strictly more strongly than its sweep would.
FACTORY_INIT_REFERENCE = "skills/factory-init/references/supervised-unit.md"


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
