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
