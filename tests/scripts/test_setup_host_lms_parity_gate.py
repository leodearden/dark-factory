"""The lms-arm@ parity gate in scripts/setup-host.sh.

Asserts against the TEXT of setup-host.sh, the idiom this repo already uses for
shell-script contracts that cannot be exercised hermetically (cf.
scripts/tests/test_lms_ctl.py::test_installer_never_pipes_systemctl_into_grep
and the slice-and-assert family in
tests/scripts/test_check_orchestrator_unit_parity.py).

Why a host-wide gate exists at all: an installer-only check leaves an override
invisible BETWEEN installs, which is when it actually bites. A landed worktree
stays on disk for months (213 measured, task 3267), so a drop-in pinning
WorkingDirectory at one can outlive its worktree indefinitely with nothing on
this host reporting it.

Parsing is shared with the other suites that read this file, via
setup_host_parsing: duplicated regexes over one brittle source text is the
failure mode where a reflow of one section turns several suites red for one
cause, each naming the other's helper.
"""

import re

import pytest
from setup_host_parsing import (  # pyright: ignore[reportMissingImports]
    SETUP_HOST_PATH,
    shell_statements,
)

CHECKER = "check_lms_unit_parity.py"
LOG_TAG = "[lms_unit_parity]"


@pytest.fixture(scope="module")
def setup_host_text() -> str:
    return SETUP_HOST_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def lms_block(setup_host_text: str) -> str:
    """The shell statement that invokes the lms checker, with its branches.

    Sliced from the parsed statement list rather than by line number, so the
    assertions below survive a reflow of the surrounding sections.
    """
    matches = [
        statement
        for statement in shell_statements(setup_host_text)
        if CHECKER in statement
    ]
    assert matches, f"scripts/setup-host.sh does not invoke {CHECKER}"
    assert len(matches) == 1, (
        f"expected exactly one {CHECKER} gate, found {len(matches)}"
    )
    return matches[0]


def test_setup_host_runs_the_lms_parity_checker(lms_block: str):
    """The gate reads the same two trees the installer does.

    Passing only one would compare the committed template against a directory
    the installer never wrote, or vice versa — a gate reporting on state
    nobody produces.
    """
    assert "--installed-dir" in lms_block
    assert "--repo-root" in lms_block


def test_exit_2_is_the_benign_not_installed_branch(lms_block: str):
    """Exit 2 must not warn: on most hosts it is the CORRECT state.

    setup-host.sh installs no lms units — it contains no reference to
    install-lms or local-model-serving — so on any host that never installed
    the arms, 2 is the expected outcome. Warning there every single run is how
    operators learn to ignore the gate, taking the real finding with it.
    """
    assert re.search(r'-eq\s+2', lms_block), lms_block

    # The exit-2 arm reports, it does not warn.
    benign = re.search(
        r'-eq\s+2\s*\]\s*;\s*then(?P<arm>.*?)\belse\b', lms_block, re.DOTALL
    )
    assert benign, lms_block
    assert "info " in benign.group("arm")
    assert "warn " not in benign.group("arm")


def test_setup_host_does_not_install_the_lms_units(setup_host_text: str):
    """This gate REPORTS; it must not start installing the arms.

    The units hold the whole GPU, and which arm runs when is an operator
    decision per the template's own stated rationale — never a side effect of
    running setup. This is also what makes exit 2 benign rather than a defect.
    """
    assert "install-lms" not in setup_host_text
    assert "local-model-serving" not in setup_host_text


def test_the_override_case_is_worded_apart_from_a_directive_diff(lms_block: str):
    """Exit 1 warns, and names the OVERRIDE case distinctly.

    Same wording discipline as the dashboard gate: the checker deliberately
    words [override] apart from [drift] so an operator is not sent hunting a
    directive diff that does not exist. Collapsing that back into a single
    "drift" warning here would undo it at the last step.
    """
    assert "warn " in lms_block
    lowered = lms_block.lower()
    assert "drop-in" in lowered or "override" in lowered
    # The remediation an operator can actually act on.
    assert "systemctl --user cat" in lms_block
    assert "remove-lms-arm-worktree-dropin.sh" in lms_block


def test_the_warning_routes_to_the_report_by_tag(lms_block: str):
    """The warn text names [lms_unit_parity], the family convention.

    A one-line warning in a long bring-up run has no reliable way to point at
    its own output by position. The dashboard block names
    [dashboard_unit_parity] and the orchestrator block names
    [orchestrator_unit_parity] for exactly this reason.
    """
    assert LOG_TAG in lms_block


def test_the_lms_gate_is_warn_only(lms_block: str):
    """Never exit, die or fail from this block.

    setup-host.sh is the host bring-up path, and this task deliberately
    REFUSES to remove a drop-in. A hard failure would therefore brick bring-up
    on precisely the state we chose not to auto-fix. (Contrast the orchestrator
    gate, which may fail — that one guards an install setup-host.sh actually
    performs; this one reports on units it does not install.)
    """
    for statement in lms_block.splitlines():
        stripped = statement.strip()
        if stripped.startswith("#"):
            continue
        assert not re.match(r"(exit|die|fail)\b", stripped), stripped


def test_the_gate_cannot_false_red_when_its_inputs_are_absent(lms_block: str):
    """Guarded, so an absent checker or committed template is not a warning.

    A checkout without the checker is not a host with a drop-in, and reporting
    it as one is the same credibility leak as warning on exit 2.
    """
    assert re.search(r"\[\s*-f\s", lms_block), lms_block
