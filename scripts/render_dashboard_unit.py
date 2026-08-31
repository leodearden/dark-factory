#!/usr/bin/env python3
"""Render dark-factory-dashboard.service, PRESERVING this host's local Environment= values.

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

WHY A SEPARATE SCRIPT RATHER THAN A ``--fix`` ON THE PARITY CHECKER. That
module's docstring argues at length for staying READ-ONLY, and names the
specific hazard a writing mode would inherit: re-arming a watchdog timer
someone deliberately left disarmed. Nothing here installs, enables or reloads
anything — it renders text into a file setup-host.sh names, and setup-host.sh
still owns the install. A checker that stays a checker keeps being believable
when it fires, which is the only thing it has.

WHY THIS DOES NOT IMPORT check_dashboard_unit_parity, even though that is where
the Environment= reader used to live. tests/scripts/test_check_dashboard_unit_parity.py
builds the section-8 tmp repo with ``write_checker(body=...)``, which REPLACES
check_dashboard_unit_parity.py with an argparse-usage-error STUB — and with
``with_checker=False`` omits it and its siblings altogether — precisely to test
that the install still happens when the parity gate did not run. A renderer
importing that module would ImportError under exactly those two tests, turning
them red for a reason unrelated to what they assert. So the shared dependency
was pushed DOWN into scripts/systemd_unit_parity.py (``environment_map``)
rather than sideways. Do not "tidy" this into a cross-module import.

Stdlib-only, so it stays runnable under a plain ``python3`` with no environment
set up — the same rule its ``check_*_unit_parity.py`` neighbours follow, and a
hard requirement for something setup-host.sh calls before any venv exists.
"""

import argparse
import contextlib
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


# The Environment= variables that are HOST-LOCAL and must therefore SURVIVE a
# re-render. This is the host-local SUBSET of
# check_dashboard_unit_parity.DIVERGENCE_ALLOWLIST — NOT that allowlist itself,
# and the difference is the whole design.
#
# That allowlist has two entries, on it for OPPOSITE reasons:
#
#   * DASHBOARD_KNOWN_PROJECT_ROOTS — the declared HOLE. The committed unit's
#     own comment says "additional project roots are LOCAL settings, added to
#     the installed unit, not committed here", and the checker's entry calls it
#     "A HOLE IN THE GATE". Genuinely host-local; preserving it is the point.
#   * DASHBOARD_PROJECT_ROOT — NOT host-local, despite sitting on the same list.
#     Its value is RENDERED from __REPO_ROOT__ per host, and the checker still
#     CHECKS it — intra-copy, against the SAME file's WorkingDirectory=
#     (UnitSpec.env_matches_directive). Preserving a PREVIOUS host's value would
#     pin the dashboard's data root at the OLD repo root while WorkingDirectory=
#     moved to the new one, manufacturing precisely the intra-copy drift
#     _compare_env_matches_directive exists to report — on a host that had just
#     been correctly reinstalled.
#
# So: adding a name here that is NOT on DIVERGENCE_ALLOWLIST turns the parity
# gate permanently red (the checker would value-compare a variable this
# installer deliberately makes differ), and adding DASHBOARD_PROJECT_ROOT
# specifically breaks the unit in the way just described. Neither failure is
# hypothetical and neither is visible in this file alone.
#
# THE SUBSET RELATION AND THE EXCLUSION ARE HELD BY TESTS, NOT BY AN IMPORT:
# tests/scripts/test_render_dashboard_unit.py::
# test_host_local_environment_is_a_subset_of_the_divergence_allowlist and
# ::test_host_local_environment_excludes_project_root, plus a staleness guard
# asserting every name here is really declared in the committed template.
#
# The concrete obstacle to importing the checker instead — do not "tidy" this
# into a cross-module import: tests/scripts/test_check_dashboard_unit_parity.py's
# section-8 harness builds a tmp repo where write_checker(body=...) REPLACES
# check_dashboard_unit_parity.py with an argparse stub, and with_checker=False
# omits it and its siblings entirely. An import here would ImportError under
# test_section_8_missing_checker_does_not_read_as_not_yet_installed and
# test_section_8_usage_error_does_not_read_as_not_yet_installed, turning both
# red for a reason unrelated to what they assert. This is the same
# held-by-a-test-rather-than-shared-code arrangement the checker's own UNITS
# registry has with setup-host.sh's _orch_units array.
HOST_LOCAL_ENVIRONMENT: tuple[str, ...] = ("DASHBOARD_KNOWN_PROJECT_ROOTS",)

# The section every dashboard Environment= directive lives in. Named rather
# than searched: check_dashboard_unit_parity's UnitSpec pins the same
# `environment_section="Service"`, and the two must agree or the installer
# preserves out of a section the gate does not read.
_ENVIRONMENT_SECTION = "Service"

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
        _ENVIRONMENT_SECTION,
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
            f"[{_ENVIRONMENT_SECTION}]\nEnvironment={token}\n"
        ),
        _ENVIRONMENT_SECTION,
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
    names: "tuple[str, ...] | list[str]" = HOST_LOCAL_ENVIRONMENT,
) -> "tuple[str, dict[str, str], dict[str, str]]":
    """Render *template_text*, then put this host's local values back.

    The composition: ``render_template`` -> ``preserved_values`` ->
    ``apply_preserved``. Returns ``(text, preserved, skipped)`` — the final unit
    plus the whole preservation RECORD, so a caller can report what it decided
    rather than merely doing it.

    *installed_text* is the unit ALREADY on the host (``""`` on a greenfield
    one). Only names in *names* are read from it; everything else in the output
    comes from the freshly rendered template. That asymmetry is the design:
    DASHBOARD_PROJECT_ROOT is on the parity checker's DIVERGENCE_ALLOWLIST too,
    but it is a RENDERED value that must equal the same copy's
    WorkingDirectory=, so preserving it would pin the data root at the previous
    checkout. See HOST_LOCAL_ENVIRONMENT above for the full argument.

    PURE TEXT — no filesystem access at all. ``main`` owns the reading and the
    atomic write, so the entire preservation contract stays testable without
    touching a unit dir, which is the rule every sibling suite states.
    """
    rendered = render_template(template_text, repo_root=repo_root, uv_path=uv_path)
    preserved, skipped = preserved_values(installed_text, names)
    return apply_preserved(rendered, preserved), preserved, skipped


def _log(message: str, *, stream=None) -> None:
    """Print *message* prefixed with the log tag.

    Every line this script emits goes through here, on stdout and stderr alike
    — pinned by test_main_every_emitted_line_carries_the_log_tag. Same shape
    and same reason as check_dashboard_unit_parity._log.
    """
    print(f"[{LOG_TAG}] {message}", file=stream if stream is not None else sys.stdout)


def main(argv: "Sequence[str] | None" = None) -> int:
    """Render the dashboard unit into --output, preserving host-local values.

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

    ``--no-preserve`` renders the template and nothing else. It is for
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
            "Render dark-factory-dashboard.service from its template, "
            "preserving host-local Environment= values already present in "
            "--output."
        )
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

    template_path = pathlib.Path(args.template)
    output_path = pathlib.Path(args.output)

    try:
        template_text = template_path.read_text(encoding="utf-8")
    except OSError as exc:
        _log(f"FAILED: cannot read template {template_path}: {exc}", stream=sys.stderr)
        _log(
            f"  {output_path} was NOT modified — it keeps whatever this host "
            "already had.",
            stream=sys.stderr,
        )
        return 1

    # The installed side. An absent --output is the greenfield case, not an
    # error: there is simply nothing to preserve.
    installed_text = ""
    names: tuple[str, ...] = HOST_LOCAL_ENVIRONMENT
    if args.no_preserve:
        # Not merely "read nothing": the preserve SET is emptied too, so the
        # names do not then get reported as absent-from-the-installed-unit,
        # which would be a true sentence about a file this run never read.
        names = ()
        _log(
            f"--no-preserve: rendering the template as-is; no host-local "
            f"Environment= values were read from {output_path}"
        )
    elif output_path.is_file():
        try:
            installed_text = output_path.read_text(encoding="utf-8")
        except OSError as exc:
            _log(
                f"FAILED: cannot read the installed unit {output_path}: {exc}",
                stream=sys.stderr,
            )
            _log(
                "  Refusing to render over a unit whose host-local values "
                "could not be read.",
                stream=sys.stderr,
            )
            return 1

    try:
        text, preserved, skipped = render_unit(
            template_text,
            repo_root=args.repo_root,
            uv_path=args.uv_path,
            installed_text=installed_text,
            names=names,
        )
    except ValueError as exc:
        _log(f"FAILED: {exc}", stream=sys.stderr)
        _log(
            f"  {output_path} was NOT modified — its host-local values are "
            "intact and the unit may simply be stale.",
            stream=sys.stderr,
        )
        return 1

    # The REPORT. DASHBOARD_KNOWN_PROJECT_ROOTS is on the parity checker's
    # DIVERGENCE_ALLOWLIST, so setup-host.sh's post-install check is
    # structurally incapable of saying anything about its value: these lines are
    # the only record that the variable was handled at all.
    for name, value in preserved.items():
        _log(f"preserved host-local Environment={name}={value}")
    for name, code in skipped.items():
        # The CODE is emitted as a bracketed token beside the sentence: stable
        # for anything matching on it, while the prose stays free to change.
        _log(f"default used for Environment={name} [{code}]: {_SKIP_SENTENCE[code]}")

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
        _log(f"FAILED: cannot write {output_path}: {exc}", stream=sys.stderr)
        _log(
            f"  {output_path} was NOT modified — the write goes through a temp "
            "file and a rename, so it is byte-unchanged.",
            stream=sys.stderr,
        )
        return 1

    _log(f"rendered {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
