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
_SKIP_ABSENT = "absent from the installed unit — rendered default used"
_SKIP_EMPTY = (
    "declared empty in the installed unit — rendered default used (an empty "
    "value is not a usable setting and would be worse than the default)"
)


def preserved_values(
    installed_text: str, names: "tuple[str, ...] | list[str]"
) -> "tuple[dict[str, str], dict[str, str]]":
    """Read the host-local values of *names* off an ALREADY-INSTALLED unit.

    Returns ``(preserved, skipped)``:

    - ``preserved`` — ``{name: value}`` for every name the installed unit
      declares with a non-empty stripped value.
    - ``skipped`` — ``{name: reason}`` for every other name in *names*.

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
            skipped[name] = _SKIP_ABSENT
        elif not env[name].strip():
            skipped[name] = _SKIP_EMPTY
        else:
            preserved[name] = env[name]
    return preserved, skipped


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
        lines[index] = f"{prefix}{value}{newline}"
    return "".join(lines)
