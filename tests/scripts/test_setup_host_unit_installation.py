"""setup-host.sh installer coverage, and SETUP.md operator-remediation coverage.

Two suites whose SUBJECT is not a .service file at all, split out of
tests/scripts/test_orchestrator_service_files.py by task 3746 (the reviewer's
preferred fix from task 3641, which could not apply it inside its own lock
set).  That module asserts the SHAPE of individual unit templates; these two
assert what the installer and the operator documentation DO with them, which
is a different subject with a different failure mode, and holding all three in
one ~1300-line module is what task 3641's reviewer objected to.

  1. setup-host.sh installer coverage — every scripts/orchestrator-*.service
     template must be declared in the installer's `_orch_units` array, copied
     into $UNIT_DIR, and (if it declares an [Install] section) enabled.
  2. SETUP.md operator-remediation coverage — the copy-paste
     `systemctl --user disable --now` block must name exactly the FOREIGN
     orchestrator units setup-host.sh enables.

The two travel together rather than as two more modules: suite 2 derives its
input set through suite 1's `_unit_has_install_section` predicate, so a split
between them would either duplicate that predicate or re-export it for a
single consumer.

COVERAGE GUARDS.  Both suites compare derived sets, and both can collapse to
empty-vs-empty — a passing comparison that checks nothing.
test_setup_host_install_predicate_discriminates and
test_setup_md_disable_block_is_discoverable exist solely to stop that, and
moved here WITH the suites they protect.  Do not relocate one without the
other.  (The third such guard in the origin module,
test_exec_start_config_parser_answers_for_every_orchestrator_run_unit,
deliberately did NOT move: it protects the ExecStart-parser suite, which
stayed.)

Shared helpers are imported, not copied: the setup-host.sh section-5 parse
comes from tests/scripts/setup_host_parsing.py (task 4198), and the unit-file
section parse plus the template glob from
tests/scripts/systemd_unit_invariants.py (lifted there by this task, since the
split put a consumer on both sides of it).  All three are importable by name
only because tests/scripts/conftest.py puts this directory on sys.path —
pytest's --import-mode=importlib (set in pyproject.toml addopts) deliberately
does not.
"""
import pathlib
import re

import pytest

# setup-host.sh section 5 parsing, shared with the origin module and with
# tests/scripts/test_check_orchestrator_unit_parity.py.  Aliased to the private
# spellings these suites already used, matching the origin module's convention
# for this import block.
from setup_host_parsing import (
    DECISION_LOOP_HEADER as _DECISION_LOOP_HEADER,
)
from setup_host_parsing import (
    INSTALL_LIST_APPEND as _INSTALL_LIST_APPEND,
)
from setup_host_parsing import (
    INSTALL_LOOP_CP as _INSTALL_LOOP_CP,
)
from setup_host_parsing import (
    INSTALL_LOOP_HEADER as _INSTALL_LOOP_HEADER,
)
from setup_host_parsing import (
    UNIT_DIR_VAR as _UNIT_DIR_VAR,
)
from setup_host_parsing import (
    declared_orchestrator_units as _declared_orchestrator_units,
)
from setup_host_parsing import (
    shell_statements as _shell_statements,
)

# Imported UNALIASED, for the same reason the origin module states: the sibling
# modules import these under exactly these public names, and a private alias
# here would obscure that they are one definition rather than several.
from systemd_unit_invariants import (
    ALL_ORCHESTRATOR_SERVICE_FILES,
    parse_sections,
)

REPO_ROOT = pathlib.Path(__file__).parents[2]


# ---------------------------------------------------------------------------
# setup-host.sh installer coverage (task 3641)
# ---------------------------------------------------------------------------
#
# A committed unit template that setup-host.sh never installs is invisible: it
# looks like a supported unit, but a fresh host simply does not get it, and an
# existing host keeps whatever hand-installed copy it happens to have.  That is
# the exact defect this task exists to fix — orchestrator-know-live.service had
# a committed template since e5273d8623 and was never wired in, and
# orchestrator-pump-web-ui.service ran on the host for weeks with no committed
# source at all.  Fixing the two instances without a guard leaves the CLASS
# open, so this asserts the installer's coverage mechanically.

SETUP_HOST_SH = REPO_ROOT / "scripts" / "setup-host.sh"


def _unit_has_install_section(content: str) -> bool:
    """True if the unit declares an [Install] section (i.e. it is enable-able).

    ``systemctl enable`` on a unit with no [Install] section is an error, not a
    no-op — such a unit is 'static'.  orchestrator-watchdog.service is exactly
    that: it is pulled in by orchestrator-watchdog.timer, which carries the
    install instead.
    """
    return "Install" in parse_sections(content)


# Parsed once at import and shared by every consumer below (the parametrized
# installer guard, one case per template, plus the SETUP.md parity guard) —
# setup-host.sh does not change under a test run, so re-reading and re-parsing
# it per case bought nothing.
SETUP_HOST_STATEMENTS = _shell_statements(SETUP_HOST_SH.read_text(encoding="utf-8"))

# The enable obligation is decided at RUN TIME from the unit's own [Install]
# section — the same rule _unit_has_install_section expresses here in Python —
# rather than from a hand-maintained exception list naming the static watchdog
# service.  See test_setup_host_installs_every_orchestrator_unit.
#
# Pinned as the whole `if ...; then` statement rather than the bare grep, so
# this asserts the predicate is used as a GUARD.  A grep whose result is
# discarded would enable the static watchdog service — an error, not a no-op,
# which under `set -e` aborts the installer.
_ENABLE_INSTALL_GUARD = (
    "if grep -q '^\\[Install\\]' \"$REPO_ROOT/scripts/$_unit\"; then"
)
_ENABLE_STATEMENT = 'systemctl --user enable "$_unit"'


def test_setup_host_install_predicate_discriminates() -> None:
    """Coverage guard: the glob is non-empty AND the [Install] predicate splits it.

    Mirrors test_orchestrator_service_glob_covers_all_known_units, plus the
    silent-green risk specific to the parametrized guard below.  That test's
    enable half is CONDITIONAL on _unit_has_install_section, so a predicate
    that answered False for everything (a broken parse_sections, a change in
    unit shape) would vacuously satisfy every case while checking nothing.
    Asserting the predicate discriminates in BOTH directions keeps it honest:
    at least one enable-able unit, and at least one static unit — today the
    seven project orchestrators and orchestrator-watchdog.service respectively.
    """
    assert ALL_ORCHESTRATOR_SERVICE_FILES, (
        "glob discovered no orchestrator-*.service templates"
    )
    by_install = {
        p.name: _unit_has_install_section(p.read_text(encoding="utf-8"))
        for p in ALL_ORCHESTRATOR_SERVICE_FILES
    }
    assert any(by_install.values()), (
        "no orchestrator template has an [Install] section — the enable half of "
        "test_setup_host_installs_every_orchestrator_unit would pass vacuously "
        f"for every unit. Parsed: {by_install}"
    )
    assert not all(by_install.values()), (
        "every orchestrator template has an [Install] section, so the static-unit "
        "branch is never exercised. orchestrator-watchdog.service is expected to "
        f"have none (it is pulled in by its .timer). Parsed: {by_install}"
    )


@pytest.mark.parametrize(
    "service_path",
    ALL_ORCHESTRATOR_SERVICE_FILES,
    ids=lambda p: p.name,
)
def test_setup_host_installs_every_orchestrator_unit(
    service_path: pathlib.Path,
) -> None:
    """setup-host.sh must copy every orchestrator template, and enable each one
    that has an [Install] section.

    setup-host.sh is how a unit reaches a host.  A template it does not copy is
    a unit that does not exist on a fresh machine, however correct its contents
    — so 'committed' and 'installed' silently diverge with nothing to detect it.

    The enable obligation is DERIVED from the [Install] predicate rather than
    hard-coded around a named watchdog exception, for the same reason
    test_systemd_restart_backoff.py discovers by content instead of by list: a
    hand-maintained exception list has to be edited for every future unit, and
    a unit nobody remembers to add is precisely the bug being fixed here.  With
    the predicate, a new orchestrator template turns this red on the day it
    lands unless the installer learns about it.

    This does NOT assert the converse (that the installer names no unknown
    unit) — a `cp` of a template this glob cannot see would fail the installer
    loudly at run time, so it needs no test.

    Shape note: setup-host.sh declares its units once as `_orch_units` and
    loops, so coverage is a MEMBERSHIP question here plus two assertions that
    the loops act on the array.  Those two are what keep membership meaningful:
    a unit listed in an array nothing iterates is exactly as uninstalled as a
    unit nobody listed.
    """
    basename = service_path.name
    declared = _declared_orchestrator_units()

    # Guard the guard: an array parse that silently returned nothing would make
    # the membership assertion below fail with a message blaming the installer
    # for omitting a unit it in fact lists.
    assert declared, (
        f"Parsed ZERO units out of {SETUP_HOST_SH.name}'s `_orch_units` array. "
        "The installer's declaration idiom has changed and "
        "_ORCH_UNITS_ARRAY_RE no longer matches it — fix the regex, do not "
        "weaken this test."
    )

    assert basename in declared, (
        f"{SETUP_HOST_SH.name}'s `_orch_units` array does not list {basename}, so "
        "a fresh host never gets that unit — however correct its contents. Add "
        f"it to the array:\n    {basename}\nDeclared today: {sorted(declared)}"
    )

    for expected, why in (
        (
            _DECISION_LOOP_HEADER,
            "declares `_orch_units` but nothing iterates it, so listing "
            f"{basename} there installs nothing",
        ),
        (
            _INSTALL_LIST_APPEND,
            "never adds a unit that cleared the parity gate to "
            "`_orch_install_units`, so nothing is ever queued for install",
        ),
        (
            _INSTALL_LOOP_HEADER,
            "does not install from the CLEARED set, so the per-unit gate "
            "decision has no effect on what reaches the host",
        ),
        (
            _INSTALL_LOOP_CP,
            f"does not copy into {_UNIT_DIR_VAR} with the copy's failure "
            "handled; the destination is asserted, not just the source (a cp "
            "of the template to a staging path leaves the unit just as "
            "uninstalled as no cp at all), and so is the `if` around it (a "
            "bare cp aborts the whole installer on the first unwritable unit)",
        ),
    ):
        assert expected in SETUP_HOST_STATEMENTS, (
            f"{SETUP_HOST_SH.name} {why}. Expected the statement:\n    {expected}"
        )

    content = service_path.read_text(encoding="utf-8")
    if not _unit_has_install_section(content):
        return

    assert _ENABLE_STATEMENT in SETUP_HOST_STATEMENTS, (
        f"{basename} declares an [Install] section but {SETUP_HOST_SH.name} has no "
        "enable loop, so it is copied to the host and then never starts at boot. "
        f"Expected the statement:\n    {_ENABLE_STATEMENT}"
    )
    assert _ENABLE_INSTALL_GUARD in SETUP_HOST_STATEMENTS, (
        f"{SETUP_HOST_SH.name}'s enable loop does not gate on the unit's own "
        "[Install] section, so it would either skip enable-able units or run "
        "`systemctl enable` on a static one — which is an ERROR, not a no-op, "
        "and under `set -e` aborts the installer. Expected:\n"
        f"    {_ENABLE_INSTALL_GUARD}"
    )


# ---------------------------------------------------------------------------
# SETUP.md operator remediation coverage (task 3641, review feedback)
# ---------------------------------------------------------------------------
#
# setup-host.sh installs and ENABLES orchestrator units for several projects
# that are not part of this repo.  SETUP.md's "Known gap" callout is the only
# operator-facing remediation for that: it hands a non-maintainer a copy-paste
# `systemctl --user disable --now ...` command naming the units to turn off.
#
# That block is EXECUTABLE, not prose — which makes it the same class of
# artifact as setup-host.sh's own cp/enable lines guarded above, and it drifts
# the same way.  Every unit the installer enables but the doc omits is left
# running on a non-maintainer's host with a --config path that does not exist:
# at the next login/default.target it crash-loops until StartLimitBurst is
# exhausted, then sits failed.  And every unit the doc names but the installer
# does NOT install as a foreign unit breaks the operator's copy-paste outright:
# `systemctl --user disable --now` on a nonexistent unit file exits non-zero,
# and on THIS repo's own orchestrator it turns off the one the operator wants.
# So both drift directions are real defects and both are asserted, separately.

SETUP_MD = REPO_ROOT / "SETUP.md"

# The one enabled orchestrator unit that IS genuinely the reader's own, which
# SETUP.md explicitly tells them to keep ("orchestrator-dark-factory.service
# itself is genuinely yours and is fine to run").  Keying the exclusion on this
# repo's OWN unit — rather than hand-listing the foreign ones — means a seventh
# project is covered by this guard the day its unit lands, matching the
# [Install]-predicate reasoning used by the installer-coverage guard above.
_OWN_PROJECT_UNIT = "orchestrator-dark-factory.service"

_DISABLE_COMMAND = "systemctl --user disable --now"

_ORCHESTRATOR_UNIT_RE = re.compile(r"orchestrator-[\w.-]+?\.service")


def _enabled_orchestrator_service_units() -> set[str]:
    """Orchestrator .service units that setup-host.sh actually enables.

    Derived from the declared ``_orch_units`` array — what the installer DOES,
    never what its section header prose says — filtered by the SAME two rules
    the installer's enable loop applies at run time:

    1. The ``.service`` suffix, which naturally drops
       ``orchestrator-watchdog.timer``.  setup-host.sh enables that too, and
       SETUP.md correctly tells the reader to KEEP it enabled (it skips
       disabled units, so it is harmless).
    2. The unit's own ``[Install]`` section, which is the shell's
       ``grep -q '^\\[Install\\]'`` predicate expressed here in Python.  This
       drops ``orchestrator-watchdog.service``: it is static, and
       ``systemctl enable`` on it is an error rather than a no-op.

    Deriving instead of scanning for one literal ``systemctl --user enable
    <unit>`` statement per unit is what keeps this in step now that the
    installer loops: the loop carries ONE enable statement for all nine units,
    so a per-statement scan would collapse this set to empty and make the
    SETUP.md parity guard below pass vacuously.
    """
    units = set()
    for unit in _declared_orchestrator_units():
        if not (unit.startswith("orchestrator-") and unit.endswith(".service")):
            continue
        template = REPO_ROOT / "scripts" / unit
        if not template.is_file():
            continue
        if _unit_has_install_section(template.read_text(encoding="utf-8")):
            units.add(unit)
    return units


def _fenced_code_blocks(md_text: str) -> list[str]:
    """Bodies of every ``` -fenced code block, in document order.

    Parsed by walking fences rather than by line number so the guard survives
    any edit above the block it cares about.
    """
    blocks: list[str] = []
    current: list[str] | None = None
    for line in md_text.splitlines():
        if line.lstrip().startswith("```"):
            if current is None:
                current = []
            else:
                blocks.append("\n".join(current))
                current = None
            continue
        if current is not None:
            current.append(line)
    return blocks


def _setup_md_disable_blocks() -> list[str]:
    """Every fenced block in SETUP.md carrying the disable-the-foreign-units command."""
    return [
        block
        for block in _fenced_code_blocks(SETUP_MD.read_text(encoding="utf-8"))
        if _DISABLE_COMMAND in block
    ]


def test_setup_md_disable_block_is_discoverable() -> None:
    """Coverage guard: exactly one disable block, and a non-empty derived set.

    Mirrors the intent of test_setup_host_install_predicate_discriminates.  The
    parity assertion below compares two derived sets, and BOTH can silently
    collapse to empty: a fence-parse mishap returning no block, or an enable-
    statement scan that matches nothing.  Empty-vs-empty is a passing set
    comparison that checks nothing — the same silent-green failure mode the
    other coverage guards in this file exist to prevent — so both inputs are
    asserted non-degenerate here, before the parity test relies on them.
    """
    blocks = _setup_md_disable_blocks()
    assert len(blocks) == 1, (
        f"expected exactly one fenced block in {SETUP_MD.name} containing "
        f"{_DISABLE_COMMAND!r}, found {len(blocks)}. The parity guard below "
        "assumes a single copy-paste remediation block; if the doc genuinely "
        "grew a second one, teach the helper which is which rather than "
        "letting it pick arbitrarily."
    )

    enabled = _enabled_orchestrator_service_units()
    assert enabled, (
        f"derived no `{_DISABLE_COMMAND.replace('disable --now', 'enable')} "
        f"orchestrator-*.service` statements from {SETUP_HOST_SH.name} — the "
        "parity guard below would pass vacuously against an empty set."
    )
    assert enabled - {_OWN_PROJECT_UNIT}, (
        f"{SETUP_HOST_SH.name} enables no orchestrator unit other than "
        f"{_OWN_PROJECT_UNIT}, so the foreign-unit set is empty and the parity "
        "guard below checks nothing."
    )


def test_setup_md_disable_block_covers_every_foreign_orchestrator_unit() -> None:
    """SETUP.md's disable block must name exactly the foreign units setup-host.sh enables.

    Scope, deliberate: this guards ONLY the executable
    ``systemctl --user disable --now`` block — the artifact an operator actually
    copy-pastes.  It does NOT pin the surrounding prose (the "four projects"
    count, or the sentence-form list of unit names), because asserting on
    documentation wording is a brittle meta-test that rots on the next rewrite
    and buys nothing this block-level assertion does not already give.  That
    prose is corrected by hand.  Do not "helpfully" extend this test into a
    wording pin.

    Note the asymmetry in how _OWN_PROJECT_UNIT is used: it is excluded when
    DERIVING the foreign set (setup-host.sh enables it, but it is not foreign),
    and deliberately NOT excluded from the ``extra`` check.  The block naming
    this repo's own orchestrator would directly contradict the sentence right
    above it — "orchestrator-dark-factory.service itself is genuinely yours and
    is fine to run" — and hand the operator a copy-paste that shuts off the one
    orchestrator they actually want.  That is drift this bidirectional guard
    exists to catch, so it is reported like any other extra.
    """
    foreign = _enabled_orchestrator_service_units() - {_OWN_PROJECT_UNIT}
    block = _setup_md_disable_blocks()[0]
    named = set(_ORCHESTRATOR_UNIT_RE.findall(block))

    missing = sorted(foreign - named)
    assert not missing, (
        f"{SETUP_HOST_SH.name} enables {', '.join(missing)} but {SETUP_MD.name}'s "
        f"`{_DISABLE_COMMAND}` block does not name "
        f"{'them' if len(missing) > 1 else 'it'}. A non-maintainer who follows "
        "the documented remediation verbatim is left with "
        f"{'those units' if len(missing) > 1 else 'that unit'} enabled, pointing "
        "at a --config path that does not exist on their machine; at the next "
        "login/default.target they crash-loop until StartLimitBurst is exhausted "
        f"and then sit failed. Fix: add {', '.join(missing)} to the "
        f"`{_DISABLE_COMMAND}` block in {SETUP_MD.name}."
    )

    extra = sorted(named - foreign)
    own_unit_note = (
        (
            f" {_OWN_PROJECT_UNIT} in particular is THIS repo's own orchestrator: "
            f"{SETUP_MD.name} tells the reader two sentences earlier that it "
            "'is genuinely yours and is fine to run', so naming it here makes "
            "the doc contradict itself and shuts off the one orchestrator the "
            "operator actually wants."
        )
        if _OWN_PROJECT_UNIT in extra
        else ""
    )
    assert not extra, (
        f"{SETUP_MD.name}'s `{_DISABLE_COMMAND}` block names "
        f"{', '.join(extra)}, which is not a foreign unit {SETUP_HOST_SH.name} "
        "enables. `systemctl --user disable --now` on a unit file that does not "
        "exist fails with 'Unit file does not exist' and returns NON-ZERO, so a "
        "stale entry breaks the whole copy-paste command for the operator — this "
        f"is a real defect, not cosmetic.{own_unit_note} Fix: remove "
        f"{', '.join(extra)} from the block in {SETUP_MD.name} (or, if the unit "
        f"should be installed after all, add it to {SETUP_HOST_SH.name}'s enable "
        "block)."
    )
