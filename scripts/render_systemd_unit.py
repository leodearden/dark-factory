#!/usr/bin/env python3
"""Render a systemd unit, PRESERVING this host's local Environment= values.

ONE RENDERER FOR EVERY UNIT IN THE ``UNITS`` REGISTRY below, selected by
``--unit``: dark-factory-dashboard.service (task 4793), fused-memory.service
(task 4796), and client-feed@.service (task 6974), which declares no
Environment= policy and so preserves nothing. The preservation core here was
already unit-neutral — only the operator-facing log tag and the preserve set
were dashboard-specific — so each further unit is a registry entry rather than
a forked near-copy.

THE FILE WAS RENAMED OFF ITS DASHBOARD-ERA NAME (task 4990), once it also
rendered fused-memory.service (task 4796) — a module named
``render_dashboard_unit`` invoked from setup-host.sh to render the unit that
governs RECONCILIATION was a misnomer at the most safety-critical install site
in the repo. The rename was deliberately deferred out of task 4796: at that
point the old name was referenced at ~35 sites across 8 tracked files, two of
which (in tests/scripts/test_check_dashboard_unit_parity.py) WRITE and UNLINK
the script BY LITERAL PATH to build setup-host.sh's section-8 tmp repo, making
the rename its own mechanical pass rather than something to smuggle into the
task that fixed the defect. The half of that naming problem that actually cost
anything at 3am was already OPERATOR-facing before this rename: a shared tag
would have printed ``[dashboard_unit_render]`` lines describing the
fused-memory unit in a long bring-up log, and task 4796 fixed that half by
making the tag per-unit (``LOG_TAG`` below is unchanged by this rename — it is
a distinct string from the module name, and stays the dashboard call site's
value).

WHAT THIS EXISTS TO PREVENT, measured rather than hypothetical. setup-host.sh
used to install the dashboard unit with a plain truncating redirect::

    sed -e "s|__REPO_ROOT__|$REPO_ROOT|g" -e "s|__UV_PATH__|$UV_PATH|g" \\
        "$REPO_ROOT/scripts/dashboard.service.template" \\
        > "$UNIT_DIR/dark-factory-dashboard.service"

scripts/dashboard.service.template declares
``Environment=DASHBOARD_KNOWN_PROJECT_ROOTS=__REPO_ROOT__``, so that render
collapsed this host's NINE measured cost/burndown aggregation roots to one —
every re-run of the sanctioned install path silently un-configured the
dashboard's whole aggregation scope.

And the damage was INVISIBLE afterwards. DASHBOARD_KNOWN_PROJECT_ROOTS is on
check_dashboard_unit_parity.DIVERGENCE_ALLOWLIST, which waives the cross-copy
VALUE comparison (the variable's whole point is that extra roots are a local
setting), so the post-install parity check reported parity on the unit that had
just been clobbered. The checker's own comment calls that entry "A HOLE IN THE
GATE". Worse, the PRE-install gate's remediation line tells the operator to run
scripts/setup-host.sh — so following the gate's advice was what caused the loss.

This renderer closes that: it reads the values off the unit that is already
installed and puts the host-local ones back into the freshly rendered text.

THE SECOND UNIT IS THE HIGHER-STAKES ONE (task 4796). Section 4 of the same
script installed fused-memory.service the same truncating way, and
scripts/fused-memory.service.template carries the same
``Environment=DASHBOARD_KNOWN_PROJECT_ROOTS=__REPO_ROOT__`` line — but on THAT
unit the variable is not a dashboard view setting. fused_memory/models/scope.py
reads it as ``KNOWN_PROJECT_ROOTS_ENV``, and reconciliation/harness.py raises
``UnknownProjectError`` for a project absent from the resulting set. Collapsing
it therefore de-registers every other project from RECONCILIATION, and the loss
is again invisible: the post-install gate (check_fused_memory_unit_parity.py)
checks only host-invariant safety directives and is structurally incapable of
saying anything about this variable's value.

WHY A SEPARATE SCRIPT RATHER THAN A ``--fix`` ON THE PARITY CHECKER. That
module's docstring argues at length for staying READ-ONLY, and names the
specific hazard a writing mode would inherit: re-arming a watchdog timer
someone deliberately left disarmed. Nothing here installs, enables or reloads
anything — it renders text into a file setup-host.sh names, and setup-host.sh
still owns the install. A checker that stays a checker keeps being believable
when it fires, which is the only thing it has.

IT DOES NOT IMPORT check_dashboard_unit_parity, or any checker. The section-8
harness in tests/scripts/test_check_dashboard_unit_parity.py replaces that
checker with a stub (``write_checker(body=...)``) or omits it
(``with_checker=False``) to test that the install still happens when the gate
did not run, and an import would ImportError there. What the renderer shares
with the checkers (``environment_map``, ``ENV_POLICY``) lives in
scripts/systemd_unit_parity.py instead.

Stdlib-only, so it stays runnable under a plain ``python3`` with no environment
set up — the same rule its ``check_*_unit_parity.py`` neighbours follow, and a
hard requirement for something setup-host.sh calls before any venv exists.
"""

import argparse
import contextlib
import dataclasses
import os
import pathlib
import shlex
import sys
import tempfile
from collections.abc import Sequence

# Prefixed onto every line this script prints, so its report is greppable and
# so a shell caller can route an operator to it BY TAG in a long bring-up run.
# Same convention, and the same load-bearing reason, as the
# check_*_unit_parity.py family: setup-host.sh's parity gates believe no status
# unless the checker's own tag appears in the output it produced, because an
# exit code alone cannot distinguish "ran and reported" from "never ran".
#
# This is the DASHBOARD's tag and `_log`'s default. Since task 4796 the tag is
# per-unit (UNITS below) and `main` threads the selected spec's tag through
# `_log(..., log_tag=...)` explicitly. Kept as a module-level name because
# task 4793's suite asserts on it and the dashboard call site passes no --unit.
LOG_TAG = "dashboard_unit_render"

# The bare sibling import resolves in both contexts this script runs in: as a
# CLI, python puts scripts/ at sys.path[0]; under pytest, tests/scripts/
# conftest.py inserts scripts/ explicitly (pyproject's --import-mode=importlib
# deliberately does not). Same mechanics the check_*_unit_parity.py family
# documents; see systemd_unit_parity.py's "Import mechanics" section.
import systemd_unit_parity  # noqa: E402


def render_template(template_text: str, *, repo_root: str, uv_path: str) -> str:
    """Substitute the two sentinels in *template_text*.

    Exactly setup-host.sh's two ``sed -e`` expressions, and deliberately the
    ONLY spelling of them left in the tree:

        sed 's|__REPO_ROOT__|$REPO_ROOT|g'   ->  replace __REPO_ROOT__
        sed 's|__UV_PATH__|$UV_PATH|g'       ->  replace __UV_PATH__

    GLOBAL (every occurrence — the committed template carries __REPO_ROOT__ at
    several sites), UNANCHORED, and LITERAL. Literal is the property the shell
    form could not actually promise: its sed delimiter is ``|``, so a repo root
    containing one would have ended the expression. ``str.replace`` has no such
    hazard, which is a strict improvement rather than a behaviour change on any
    path that worked before.

    Pure text in, pure text out. No filesystem access, so the substitution
    contract is testable without touching a unit dir.
    """
    return template_text.replace("__REPO_ROOT__", repo_root).replace(
        "__UV_PATH__", uv_path
    )


@dataclasses.dataclass(frozen=True)
class UnitRenderSpec:
    """Everything about rendering ONE unit that is not shared by every unit.

    The preservation mechanism (``render_template`` / ``preserved_values`` /
    ``_environment_token`` / ``apply_preserved``) is unit-neutral: it is
    parameterised by ``names``, which ``render_unit`` reads off a spec, and
    --template/--output are CLI flags. What is unit-specific is the log tag and
    the policy family whose host-local ENV_POLICY rows are the preserve set (why
    each row is so: plans/review-all-s25-scripts-instruments-prd.md D5). A
    family of None is a unit with no Environment= policy, which preserves
    nothing. Adding a unit is therefore a registry entry, not a fork.

    FROZEN: a spec mutated per invocation would make "which names does this host
    preserve?" depend on argument-parsing order.
    """

    log_tag: str
    family: systemd_unit_parity.UnitFamily | None

    # The committed template this unit is rendered FROM, REPO-RELATIVE. It is
    # what makes the registry self-describing rather than a bare tag/preserve-set
    # pair, and it is what the staleness guard reads to pair a preserve set with
    # the template that must actually declare it
    # (tests/scripts/test_render_systemd_unit.py::
    # test_every_preserved_name_is_declared_exactly_once_in_its_units_template).
    # A preserve set checked against the WRONG unit's template would pass while
    # preserving nothing on the host, since the two templates declare overlapping
    # but not identical Environment= sets. MEASURED: pointing this field at the
    # other unit's template leaves that staleness guard GREEN — which is why the
    # field is also pinned to what setup-host.sh actually passes as --template
    # for this --unit, by ::test_setup_host_renders_each_unit_from_the_template_
    # its_spec_names in the same module. No production code reads this field, so
    # that pairing is the whole of its meaning.
    #
    # --template STAYS A REQUIRED CLI FLAG and is deliberately NOT defaulted from
    # this field. Do not "tidy" that into a default: setup-host.sh passes an
    # absolute path built from $REPO_ROOT, so defaulting would introduce a SECOND
    # path-resolution rule (script-relative vs caller-supplied) for no caller that
    # wants one — and it would silently resolve against the checkout this file
    # happens to live in rather than the one being installed, which is the exact
    # class of wrong-checkout bug the preserve set excludes PROJECT_ROOT to avoid.
    template: str

    @property
    def host_local_environment(self) -> tuple[str, ...]:
        """The names this unit preserves: ENV_POLICY's host-local rows for its family."""
        if self.family is None:
            return ()
        return systemd_unit_parity.ENV_POLICY.host_local_names(self.family)


# The registry --unit selects from. Keys are the operator-facing unit NAMES as
# setup-host.sh spells them on the command line.
#
# THE TAGS MUST STAY DISTINCT. Every unit is rendered by this same script in
# the same bring-up run, and setup-host.sh routes an operator to a report BY TAG
# — a shared tag would attribute the fused-memory unit's preserved/skipped lines
# to the dashboard, on the unit where mis-attribution costs the most.
UNITS: "dict[str, UnitRenderSpec]" = {
    "dashboard": UnitRenderSpec(
        log_tag=LOG_TAG,
        family=systemd_unit_parity.UnitFamily.DASHBOARD,
        template="scripts/dashboard.service.template",
    ),
    "fused-memory": UnitRenderSpec(
        log_tag="fused_memory_unit_render",
        family=systemd_unit_parity.UnitFamily.FUSED_MEMORY,
        template="scripts/fused-memory.service.template",
    ),
    "client-feed": UnitRenderSpec(
        log_tag="client_feed_unit_render",
        family=None,
        template="scripts/client-feed.service.template",
    ),
}

# What a call that names no unit renders, on the CLI and in render_unit alike:
# setup-host.sh section 8 predates --unit and passes none.
_DEFAULT_UNIT = "dashboard"

# Reasons a name in the preserve set did NOT survive into the render. Both are
# legitimate, and both are REPORTED rather than taken silently — see the module
# docstring: DASHBOARD_KNOWN_PROJECT_ROOTS is allowlisted, so setup-host.sh's
# post-install parity check is structurally incapable of saying anything about
# its value, and this record is the only trace that the variable was handled.
#
# TWO LAYERS, DELIBERATELY. `preserved_values` returns the stable CODE; the
# sentence is looked up only when the report is printed. The distinction
# absent-vs-empty is a behavioural contract (one is a greenfield host, the other
# is a host whose setting was blanked), and pinning it through substrings of the
# operator prose made rewording the prose a test failure while letting a
# reworded sentence that happened to contain the other word satisfy the wrong
# assertion. The code is what tests assert on; the sentence is free to change.
# Both are emitted, the code as a bracketed token, so the operator-facing line
# is greppable AND readable.
SKIP_ABSENT = "absent"
SKIP_EMPTY = "empty"

_SKIP_SENTENCE = {
    SKIP_ABSENT: "absent from the installed unit — rendered default used",
    SKIP_EMPTY: (
        "declared empty in the installed unit — rendered default used (an "
        "empty value is not a usable setting and would be worse than the "
        "default)"
    ),
}


def preserved_values(
    installed_text: str, names: "tuple[str, ...] | list[str]"
) -> "tuple[dict[str, str], dict[str, str]]":
    """Read the host-local values of *names* off an ALREADY-INSTALLED unit.

    Returns ``(preserved, skipped)``:

    - ``preserved`` — ``{name: value}`` for every name the installed unit
      declares with a non-empty stripped value.
    - ``skipped`` — ``{name: code}`` for every other name in *names*, where the
      code is ``SKIP_ABSENT`` or ``SKIP_EMPTY``. A stable code rather than the
      operator sentence, so callers (and tests) can branch on WHICH fallback
      happened without depending on how it is worded; ``_SKIP_SENTENCE`` holds
      the prose ``main`` prints.

    The skipped map is not diagnostic garnish. Taking the rendered default is
    the RIGHT answer on a greenfield host and the WRONG one to take silently on
    a configured one, and nothing downstream can tell the operator which
    happened: the variable this exists for is on the parity checker's
    DIVERGENCE_ALLOWLIST, so the post-install gate reports parity either way.

    Parsing goes through ``systemd_unit_parity`` — the SAME reader
    check_dashboard_unit_parity.py compares these variables with. That is the
    point of the lift rather than a convenience: a value the gate can see has
    to be exactly a value this can preserve. In particular the multi-assignment
    spelling (``Environment=FOO=1 BAR=2``) and quoted values are read the way
    systemd reads them, where a naive split would preserve nothing and report
    success.

    An empty installed unit (``""``) yields every name skipped as absent, which
    is the greenfield case — not an error.
    """
    env = systemd_unit_parity.environment_map(
        systemd_unit_parity.parse_unit_directives(installed_text),
        systemd_unit_parity.SERVICE_SECTION,
    )

    preserved: dict[str, str] = {}
    skipped: dict[str, str] = {}
    for name in names:
        if name not in env:
            skipped[name] = SKIP_ABSENT
        elif not env[name].strip():
            skipped[name] = SKIP_EMPTY
        else:
            preserved[name] = env[name]
    return preserved, skipped


def _environment_token(name: str, value: str) -> str:
    """Spell ``NAME=VALUE`` so the same reader gets *value* back, verbatim.

    THE ASYMMETRY THIS CLOSES. ``environment_map`` reads an ``Environment=``
    line the way systemd does — several assignments per line, values quotable —
    so a value carrying whitespace is perfectly readable off an installed unit.
    Writing it back BARE was not the inverse of that: installed
    ``Environment="DASHBOARD_KNOWN_PROJECT_ROOTS=/a b,/c"`` read as ``/a b,/c``
    and re-emitted unquoted becomes two tokens, of which systemd keeps the
    first — the value silently truncates to ``/a`` on a re-render. A padded
    value (``"KNOWN=   /a"``) was worse: non-empty by the stripped test, then
    written unquoted and re-read as EMPTY. Both are the silent-clobber class
    this module exists to remove, one asymmetry in, and both are invisible
    afterwards because this variable is value-blessed by the parity checker's
    DIVERGENCE_ALLOWLIST.

    ``shlex.quote`` is a no-op for the values that actually occur (a
    comma-separated path list quotes to itself), so the committed unit and every
    ordinary host render stay BYTE-IDENTICAL; only a value that needs quoting
    gets any.

    The round trip is then VERIFIED, through the very reader the parity gate
    compares with, and a value that does not survive it raises rather than
    installs. That is the backstop for spellings quoting cannot rescue — a value
    containing a newline cannot live on one ``Environment=`` line at all — and
    it keeps the promise structural instead of resting on shlex and systemd
    agreeing about every escape.
    """
    token = shlex.quote(f"{name}={value}")
    roundtrip = systemd_unit_parity.environment_map(
        systemd_unit_parity.parse_unit_directives(
            f"[{systemd_unit_parity.SERVICE_SECTION}]\nEnvironment={token}\n"
        ),
        systemd_unit_parity.SERVICE_SECTION,
    )
    if roundtrip.get(name) != value:
        raise ValueError(
            f"{name}: the installed value {value!r} cannot be written to a "
            f"single Environment= line — quoted as {token!r} it reads back as "
            f"{roundtrip.get(name)!r}. Refusing to install a unit whose "
            "host-local value would differ from the one it was preserved from; "
            "the truncation would be invisible afterwards, because this "
            "variable is value-blessed by the parity checker's "
            "DIVERGENCE_ALLOWLIST."
        )
    return token


def apply_preserved(rendered_text: str, preserved: "dict[str, str]") -> str:
    """Put each *preserved* value back into *rendered_text*, one line each.

    LINE-SCOPED rewrite, never a whole-text regex, for the reason
    ``_exec_start_flag`` documents in check_dashboard_unit_parity.py: the
    committed template discusses these variables in COMMENT PROSE directly
    above the directives they describe. A whole-text substitution would edit
    the explanation as readily as the setting.

    RAISES ValueError, naming the variable, unless EXACTLY ONE line's stripped
    form starts with ``Environment=<NAME>=``:

    - ZERO means the template changed shape underneath this code. The value
      would be read off the installed unit and dropped on the floor while the
      install reported success — precisely the silent clobber this module
      exists to remove, one layer in.
    - MORE THAN ONE is ambiguous: systemd applies every occurrence and the last
      wins, so rewriting one and leaving the other installs a value nobody
      chose, invisibly.

    ALSO raises ValueError, from ``_environment_token``, for a value that cannot
    be written to one ``Environment=`` line such that the same reader gets it
    back. The written line is quoted when quoting is what makes it round-trip.

    Every other byte of *rendered_text* is left alone.
    """
    lines = rendered_text.splitlines(keepends=True)
    for name, value in preserved.items():
        prefix = f"Environment={name}="
        matches = [i for i, line in enumerate(lines) if line.strip().startswith(prefix)]
        if len(matches) != 1:
            raise ValueError(
                f"{name}: expected exactly one line beginning "
                f"{prefix!r} in the rendered unit, found {len(matches)}. "
                "Zero means the template no longer declares the variable, so "
                "preserving the installed value would silently do nothing; "
                "more than one is ambiguous, and systemd's last-wins would "
                "hide which was chosen. Refusing to guess."
            )
        index = matches[0]
        newline = "\n" if lines[index].endswith("\n") else ""
        lines[index] = f"Environment={_environment_token(name, value)}{newline}"
    return "".join(lines)


def render_unit(
    template_text: str,
    *,
    repo_root: str,
    uv_path: str,
    installed_text: str = "",
    spec: UnitRenderSpec = UNITS[_DEFAULT_UNIT],
    preserve: bool = True,
) -> "tuple[str, dict[str, str], dict[str, str]]":
    """Render *template_text*, then put this host's local values back.

    The composition: ``render_template`` -> ``preserved_values`` ->
    ``apply_preserved``. Returns ``(text, preserved, skipped)`` — the final unit
    plus the whole preservation RECORD, so a caller can report what it decided
    rather than merely doing it.

    *installed_text* is the unit ALREADY on the host (``""`` on a greenfield
    one). Only *spec*'s ``host_local_environment`` is read from it; everything
    else in the output comes from the freshly rendered template. That asymmetry
    is the design: DASHBOARD_PROJECT_ROOT is on the parity checker's
    DIVERGENCE_ALLOWLIST too, but it is a RENDERED value that must equal the
    same copy's WorkingDirectory=, so preserving it would pin the data root at
    the previous checkout. Each name's policy: systemd_unit_parity.py::ENV_POLICY.

    ``preserve=False`` empties the preserve SET rather than merely reading
    nothing, so the names are not then reported as absent from an installed
    unit nobody read (``main``'s --no-preserve).

    PURE TEXT — no filesystem access at all. ``main`` owns the reading and the
    atomic write, so the entire preservation contract stays testable without
    touching a unit dir, which is the rule every sibling suite states.
    """
    names = spec.host_local_environment if preserve else ()
    rendered = render_template(template_text, repo_root=repo_root, uv_path=uv_path)
    preserved, skipped = preserved_values(installed_text, names)
    return apply_preserved(rendered, preserved), preserved, skipped


def _log(message: str, *, stream=None, log_tag: str = LOG_TAG) -> None:
    """Print *message* prefixed with the log tag.

    Every line this script emits goes through here, on stdout and stderr alike
    — pinned by test_main_every_emitted_line_carries_the_log_tag. Same shape
    and same reason as check_dashboard_unit_parity._log.

    *log_tag* IS THREADED, NOT GLOBAL (task 4796). Since one script now renders
    several units, the tag varies per invocation — but rebinding the module-global
    ``LOG_TAG`` from ``main`` would make every call site's output depend on
    argument-parsing order, invisibly at the call site, and would leak across an
    in-process second call (which is exactly how this module's suite invokes the
    CLI). An explicit parameter defaulting to ``LOG_TAG`` keeps the dashboard's
    behaviour and every existing call site unchanged.
    """
    print(f"[{log_tag}] {message}", file=stream if stream is not None else sys.stdout)


def main(argv: "Sequence[str] | None" = None) -> int:
    """Render the --unit's template into --output, preserving host-local values.

    WHICH UNIT is chosen by ``--unit`` off the ``UNITS`` registry, defaulting to
    ``dashboard`` so the call site that predates the flag (setup-host.sh section
    8) is unchanged. The unit decides exactly two things — the preserve set and
    the log tag — and both are read off the spec here rather than from module
    state: the tag is passed to every ``_log`` call explicitly, never by
    rebinding the global, so a second in-process call cannot inherit the first
    one's tag. Everything else below is unit-neutral.

    READING AND WRITING THE SAME PATH IS THE INTENDED PRODUCTION CALL:
    setup-host.sh passes the installed unit as --output, and this reads that
    file FIRST as the "installed" side before overwriting it. That is precisely
    why there is an --output flag instead of printing to stdout for a shell
    redirect — ``python3 render.py ... > "$UNIT_DIR/<unit>"`` has bash TRUNCATE
    the destination before python ever opens it, so the installed value would be
    gone before it could be read: the tool would preserve nothing, take the
    rendered default, and report success. Owning the destination makes the
    read-then-write ordering structural rather than the caller's responsibility.

    THE WRITE IS ATOMIC — a temp file in --output's OWN parent directory (so the
    rename cannot cross a filesystem) followed by ``os.replace``. A render that
    fails part-way therefore leaves the host's existing unit BYTE-UNCHANGED
    rather than truncated: "stale but working" is recoverable, and setup-host.sh's
    pre-install parity gate reports it on the next run; "no unit at all" is not.
    The temp file is REMOVED on every failing path (a bare ``delete=False``
    would otherwise leave one ``.<unit>.<rand>.tmp`` per failed run beside the
    unit), and --output's MODE is carried across the replace — a fresh
    NamedTemporaryFile is 0600, so replacing without restoring would silently
    re-permission a unit this tool exists to leave alone.

    ``--no-preserve`` renders the template and nothing else — it empties the
    preserve SET rather than merely skipping the read, so the names are not then
    reported as absent from a file this run never opened. It is for
    REGENERATING THE REPO-SIDE committed copy, where preserving is exactly
    wrong: there the --output being read is the artifact under regeneration, so
    a stale committed value would be carried straight back into the "fresh"
    render. Never use it against an installed unit — that is the clobber the
    rest of this module exists to prevent, and setup-host.sh does not pass it.

    Returns 0 on success, non-zero with a tagged error line on any failure,
    having written nothing.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Render a dark-factory systemd unit from its template, preserving "
            "host-local Environment= values already present in --output. "
            "--unit selects WHICH unit (and therefore which values are "
            "host-local): dark-factory-dashboard.service, fused-memory.service "
            "or client-feed@.service."
        )
    )
    parser.add_argument(
        "--unit",
        choices=sorted(UNITS),
        default=_DEFAULT_UNIT,
        help=(
            "Which unit is being rendered. Selects the preserve set and the log "
            "tag from the UNITS registry. Defaults to the dashboard, so the "
            "call site that predates this flag is unchanged. `choices` is what "
            "makes an unknown value a USAGE ERROR rather than a silent fallback "
            "to a preserve set that does not match the unit being written."
        ),
    )
    parser.add_argument("--template", required=True, help="Path to the .template file")
    parser.add_argument("--repo-root", required=True, help="Value for __REPO_ROOT__")
    parser.add_argument("--uv-path", required=True, help="Value for __UV_PATH__")
    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Unit file to write. READ FIRST as the installed copy — see this "
            "function's docstring for why this is a flag and not a redirect."
        ),
    )
    parser.add_argument(
        "--no-preserve",
        action="store_true",
        help=(
            "Do not read host-local values out of --output. For regenerating "
            "the REPO-SIDE committed unit, where the file being overwritten is "
            "the artifact under regeneration; never for an installed unit."
        ),
    )
    args = parser.parse_args(argv)

    # The whole of what is unit-specific, resolved once. Everything below is
    # unit-neutral and reads only `spec`.
    spec = UNITS[args.unit]
    log_tag = spec.log_tag

    template_path = pathlib.Path(args.template)
    output_path = pathlib.Path(args.output)

    try:
        template_text = template_path.read_text(encoding="utf-8")
    except OSError as exc:
        _log(
            f"FAILED: cannot read template {template_path}: {exc}",
            stream=sys.stderr,
            log_tag=log_tag,
        )
        _log(
            f"  {output_path} was NOT modified — it keeps whatever this host "
            "already had.",
            stream=sys.stderr,
            log_tag=log_tag,
        )
        return 1

    # The installed side. An absent --output is the greenfield case, not an
    # error: there is simply nothing to preserve.
    installed_text = ""
    if args.no_preserve:
        _log(
            f"--no-preserve: rendering the template as-is; no host-local "
            f"Environment= values were read from {output_path}",
            log_tag=log_tag,
        )
    elif output_path.is_file():
        try:
            installed_text = output_path.read_text(encoding="utf-8")
        except OSError as exc:
            _log(
                f"FAILED: cannot read the installed unit {output_path}: {exc}",
                stream=sys.stderr,
                log_tag=log_tag,
            )
            _log(
                "  Refusing to render over a unit whose host-local values "
                "could not be read.",
                stream=sys.stderr,
                log_tag=log_tag,
            )
            return 1

    try:
        text, preserved, skipped = render_unit(
            template_text,
            repo_root=args.repo_root,
            uv_path=args.uv_path,
            installed_text=installed_text,
            spec=spec,
            preserve=not args.no_preserve,
        )
    except ValueError as exc:
        _log(f"FAILED: {exc}", stream=sys.stderr, log_tag=log_tag)
        _log(
            f"  {output_path} was NOT modified — its host-local values are "
            "intact and the unit may simply be stale.",
            stream=sys.stderr,
            log_tag=log_tag,
        )
        return 1

    # The REPORT. DASHBOARD_KNOWN_PROJECT_ROOTS is on the parity checker's
    # DIVERGENCE_ALLOWLIST, so setup-host.sh's post-install check is
    # structurally incapable of saying anything about its value: these lines are
    # the only record that the variable was handled at all.
    for name, value in preserved.items():
        _log(f"preserved host-local Environment={name}={value}", log_tag=log_tag)
    for name, code in skipped.items():
        # The CODE is emitted as a bracketed token beside the sentence: stable
        # for anything matching on it, while the prose stays free to change.
        _log(
            f"default used for Environment={name} [{code}]: {_SKIP_SENTENCE[code]}",
            log_tag=log_tag,
        )

    # `temp_name` is bound INSIDE the `with` but read in the `except`, so it is
    # declared here and assigned before the write: a failing `handle.write`
    # (ENOSPC, I/O error) must still leave the name available to clean up.
    temp_name: str | None = None
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_name = handle.name
            handle.write(text)
        # Carry the destination's mode across the replace. tempfile creates
        # 0600, so without this a re-render silently re-permissions the unit —
        # an unreported side effect in a tool whose whole contract is to leave
        # this host's state alone. 0644 when there is nothing to carry, which is
        # what the `sed >` redirect this replaced produced under a normal umask.
        try:
            mode = output_path.stat().st_mode & 0o7777
        except OSError:
            mode = 0o644
        os.chmod(temp_name, mode)
        os.replace(temp_name, output_path)
    except OSError as exc:
        # Remove the temp file rather than orphaning one `.<unit>.<rand>.tmp`
        # per failed run in the unit directory. suppress(OSError) because the
        # cleanup must not mask the failure being reported.
        if temp_name is not None:
            with contextlib.suppress(OSError):
                os.unlink(temp_name)
        _log(
            f"FAILED: cannot write {output_path}: {exc}",
            stream=sys.stderr,
            log_tag=log_tag,
        )
        _log(
            f"  {output_path} was NOT modified — the write goes through a temp "
            "file and a rename, so it is byte-unchanged.",
            stream=sys.stderr,
            log_tag=log_tag,
        )
        return 1

    _log(f"rendered {output_path}", log_tag=log_tag)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
