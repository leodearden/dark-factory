"""Regression tests that guard the CI enforcement of the TYPE_CHECKING Protocol
conformance block in test_workflow_e2e.py.

The ``if TYPE_CHECKING:`` Protocol conformance block near the bottom of
test_workflow_e2e.py is invisible to pytest — pyright must be wired into the
commit gate for it to catch Protocol drift.
These tests pin the two preconditions that make the gate effective:

1. orchestrator/pyproject.toml declares ``[tool.pyright] include = ["src", "tests"]``
   so pyright type-checks test_workflow_e2e.py.
2. hooks/project-checks runs the orchestrator module's registered
   type_check_command (via scripts/hook_module_checks.py) on main-branch commits
   staging orchestrator .py, so any reportAssignmentType error blocks the commit.

If either precondition disappears, these tests fail loudly at normal ``pytest`` time
— before the real failure mode (undetected Protocol drift) can bite.

See task 699 and commit 357fa4d6a5 for full context.
"""

from __future__ import annotations

import re
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import yaml

# Locate package root (orchestrator/) from the tests/ directory.
_PACKAGE_ROOT = Path(__file__).parent.parent

# Locate repo root (dark-factory/) from orchestrator/tests/.
_REPO_ROOT = Path(__file__).parents[2]

_HELPER = _REPO_ROOT / "scripts" / "hook_module_checks.py"


def _load_pyright_config() -> dict[str, Any]:
    """Load and return the ``[tool.pyright]`` section of orchestrator/pyproject.toml.

    Returns an empty dict when the section is absent so callers can assert
    truthiness or membership with clean messages.  The pyproject.toml path is
    resolved via ``_PACKAGE_ROOT``, mirroring the ``_load_package_defaults``
    pattern in orchestrator/tests/test_config.py.
    """
    toml_path = _PACKAGE_ROOT / "pyproject.toml"
    assert toml_path.is_file(), f"pyproject.toml not found at {toml_path}"
    with open(toml_path, "rb") as fh:
        config = tomllib.load(fh)
    return config.get("tool", {}).get("pyright", {})


# ---------------------------------------------------------------------------
# (a) Sanity: [tool.pyright] section exists in orchestrator/pyproject.toml
# ---------------------------------------------------------------------------


def test_orchestrator_pyproject_has_pyright_section() -> None:
    """orchestrator/pyproject.toml must have a [tool.pyright] section."""
    pyright_config = _load_pyright_config()
    toml_path = _PACKAGE_ROOT / "pyproject.toml"
    assert pyright_config, (
        f"[tool.pyright] section missing from {toml_path}. "
        "The TYPE_CHECKING Protocol conformance block in test_workflow_e2e.py "
        "relies on pyright being configured for this package. See task 699."
    )


# ---------------------------------------------------------------------------
# (b) pyright include list must contain 'tests'
# ---------------------------------------------------------------------------


def test_orchestrator_pyright_include_contains_tests() -> None:
    """[tool.pyright] include must list 'tests' so pyright covers test_workflow_e2e.py.

    If this test fails, the TYPE_CHECKING Protocol conformance block near the
    bottom of test_workflow_e2e.py silently becomes a no-op because pyright no
    longer type-checks this file. See task 699.
    """
    pyright_config = _load_pyright_config()
    toml_path = _PACKAGE_ROOT / "pyproject.toml"
    include = pyright_config.get("include", [])
    assert "tests" in include, (
        f"'tests' is not in [tool.pyright] include = {include!r} "
        f"in {toml_path}. "
        "Without this entry pyright does not type-check test_workflow_e2e.py, "
        "so the TYPE_CHECKING Protocol conformance block near the bottom of "
        "test_workflow_e2e.py silently becomes a no-op. Re-add 'tests' to "
        "[tool.pyright] include, or update the conformance strategy. See task 699."
    )


# ---------------------------------------------------------------------------
# (c) hooks/project-checks must run orchestrator's type check
# ---------------------------------------------------------------------------


def test_project_checks_hook_invokes_pyright_on_orchestrator() -> None:
    """A staged test_workflow_e2e.py must list orchestrator's registered type check.

    hooks/project-checks runs whatever scripts/hook_module_checks.py lists, so
    this drives the real helper with that file staged. The listed
    ``module:orchestrator:type`` command must be orchestrator/orchestrator.yaml's
    type_check_command, and pyright must be pointed at ``tests/``. If this test
    fails, the commit gate that enforces the TYPE_CHECKING Protocol conformance
    block in test_workflow_e2e.py no longer exists for the orchestrator package,
    and Protocol drift between FakeScheduler/Scheduler and SchedulerFacade goes
    undetected until runtime. See task 699.
    """
    registered = yaml.safe_load((_PACKAGE_ROOT / "orchestrator.yaml").read_text(encoding="utf-8"))
    proc = subprocess.run(
        [sys.executable, str(_HELPER)],
        cwd=_REPO_ROOT,
        input="orchestrator/tests/test_workflow_e2e.py\n",
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    units = dict(line.split("\t", 1) for line in proc.stdout.splitlines())
    command = units.get("module:orchestrator:type")
    assert command == registered["type_check_command"], (
        f"{_HELPER} listed {command!r} for module:orchestrator:type, not "
        f"orchestrator/orchestrator.yaml's type_check_command "
        f"{registered['type_check_command']!r}. See task 699."
    )
    argv = shlex.split(command)
    assert "tests/" in argv[argv.index("pyright"):], (
        f"orchestrator's type check {command!r} does not name tests/, so pyright "
        "never sees test_workflow_e2e.py. See task 699."
    )


# ---------------------------------------------------------------------------
# Meta-test: no stale line-number citations in this file's own docstrings
# ---------------------------------------------------------------------------


def test_module_docstrings_do_not_cite_stale_line_numbers() -> None:
    """This file must not contain line-number citations into test_workflow_e2e.py.

    Line numbers go stale every time test_workflow_e2e.py is edited. Protocol
    conformance is enforced by pyright-on-assignment via the TYPE_CHECKING block
    in test_workflow_e2e.py, so line numbers add no value and only mislead.
    This meta-test catches any future regression to line-number citations
    without requiring a code reviewer to notice.

    The regex patterns used here deliberately do not self-match:
    - ``test_workflow_e2e\\.py:\\d+`` contains ``\\d+`` literal characters
    - ``\\bline 2\\d{3}\\b`` requires a literal ``line 2XXX`` with real digits
    """
    content = Path(__file__).read_text()
    pattern = re.compile(
        r"test_workflow_e2e\.py:\d+"
        r"|test_workflow_e2e\.py.*?\bline \d+"
        r"|\bline 2\d{3}\b"
    )
    matches = pattern.findall(content)
    assert not matches, (
        f"Stale line-number citations found in {__file__}: {matches}. "
        "Refer to the if TYPE_CHECKING Protocol conformance block by content, "
        "not line number — line numbers go stale every time "
        "test_workflow_e2e.py is edited."
    )
