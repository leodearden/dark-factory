"""Shared parsing of scripts/setup-host.sh section 5, for the tests that read it.

Two suites derive claims from the same region of the installer:

* ``test_orchestrator_service_files.py`` — every committed
  ``orchestrator-*`` template must be installed, and each one with an
  ``[Install]`` section enabled.
* ``test_check_orchestrator_unit_parity.py`` — the checker's ``UNITS``
  registry must equal the set the installer actually copies (a unit the
  installer installs but the registry omits is installed-but-UNCHECKED),
  and the behavioural harness slices the live section and executes it.

Both parse the same ``_orch_units`` array and pin the same loop statements,
and both were carrying private copies of this code.  Duplicated regexes over
one brittle source text is the failure mode where a reflow of section 5 turns
two suites red for one cause and each names the other's helper — so the
parsing lives here once, alongside ``systemd_unit_invariants.py``, which is
the same pattern for the unit files themselves.

Importable by NAME (``from setup_host_parsing import ...``) because
tests/scripts/conftest.py puts this directory on ``sys.path``; pytest's
``--import-mode=importlib`` deliberately does not.
"""

import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).parents[2]
SETUP_HOST_PATH = REPO_ROOT / "scripts" / "setup-host.sh"

# Destination the installer must copy units into: ~/.config/systemd/user, via
# the $UNIT_DIR variable setup-host.sh defines.  Named because a `cp` to
# anywhere else leaves the unit uninstalled just as surely as no `cp` at all.
UNIT_DIR_VAR = "$UNIT_DIR"

# setup-host.sh declares the units it installs ONCE and loops over the array
# for both the copy and the enable, so "does the installer install X" is a
# membership question:
#
#   _orch_units=(
#     orchestrator-reify.service        # reify orchestrator, escalation 8100
#     ...
#   )
#
# Anchored on a column-0 `_orch_units=(` ... `)` pair so no other
# parenthesised construct in the installer can be mistaken for it.
ORCH_UNITS_ARRAY_RE = re.compile(
    r"^_orch_units=\(\n(?P<body>.*?)^\)$", re.MULTILINE | re.DOTALL
)

# The statements that make that membership MEAN something:
#
#   declaration -> per-unit gate decision -> cleared set -> copy into $UNIT_DIR
#
# Break any one link and the array stops meaning "these units get installed":
# a decision loop over some other list judges the wrong units, an install loop
# over `_orch_units` rather than the cleared set installs the skipped ones, and
# a copy to a staging path leaves the unit as uninstalled as no copy at all.
DECISION_LOOP_HEADER = 'for _unit in "${_orch_units[@]}"; do'
INSTALL_LIST_APPEND = '_orch_install_units+=("$_unit")'
INSTALL_LOOP_HEADER = 'for _unit in "${_orch_install_units[@]}"; do'
# Pinned as the whole `if ...; then` rather than the bare `cp`, so the copy's
# FAILURE handling is asserted too.  A bare `cp` under `set -euo pipefail`
# aborts setup-host.sh on the first unit it cannot write — one uncopyable unit
# would take daemon-reload, every enable and every later installer section
# with it.
INSTALL_LOOP_CP = f'if cp "$REPO_ROOT/scripts/$_unit" "{UNIT_DIR_VAR}/"; then'


def shell_statements(script_text: str) -> list[str]:
    """Non-comment, non-blank statements of a shell script, whitespace-stripped.

    Comments are dropped so that a unit or a command merely NAMED in a
    section's header prose cannot satisfy an assertion that the installer
    actually runs it.
    """
    return [
        stripped
        for line in script_text.splitlines()
        if (stripped := line.strip()) and not stripped.startswith("#")
    ]


def declared_orchestrator_units(
    setup_host_path: pathlib.Path = SETUP_HOST_PATH,
) -> set[str]:
    """The unit basenames setup-host.sh's ``_orch_units`` array declares.

    Returns the empty set when the array cannot be found — callers assert it
    is non-empty ("guard the guard"), so a silently unmatched regex fails with
    a message about the PARSING rather than one blaming the installer for
    omitting a unit it in fact lists.

    Trailing ``# rationale`` comments on an entry are stripped: that prose is
    the per-unit justification the old one-line-per-unit enable block carried,
    and it is not part of the unit name.
    """
    match = ORCH_UNITS_ARRAY_RE.search(
        setup_host_path.read_text(encoding="utf-8")
    )
    if match is None:
        return set()
    units = set()
    for line in match.group("body").splitlines():
        if entry := line.split("#", 1)[0].strip():
            units.add(entry)
    return units
