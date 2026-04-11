"""Regression tests that guard the CI enforcement of the TYPE_CHECKING Protocol
conformance block in test_workflow_e2e.py.

The ``if TYPE_CHECKING:`` block at test_workflow_e2e.py:2094-2099 is invisible to
pytest — pyright must be wired into the commit gate for it to catch Protocol drift.
These tests pin the two preconditions that make the gate effective:

1. orchestrator/pyproject.toml declares ``[tool.pyright] include = ["src", "tests"]``
   so pyright type-checks test_workflow_e2e.py.
2. hooks/project-checks runs ``uv run pyright`` from orchestrator/ on main-branch
   commits, so any reportAssignmentType error blocks the commit.

If either precondition disappears, these tests fail loudly at normal ``pytest`` time
— before the real failure mode (undetected Protocol drift) can bite.

See task 699 and commit 357fa4d6a5 for full context.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

# Locate package root (orchestrator/) from the tests/ directory.
_PACKAGE_ROOT = Path(__file__).parent.parent

# Locate repo root (dark-factory/) from orchestrator/tests/.
_REPO_ROOT = Path(__file__).parents[2]


# ---------------------------------------------------------------------------
# (a) Sanity: [tool.pyright] section exists in orchestrator/pyproject.toml
# ---------------------------------------------------------------------------


def test_orchestrator_pyproject_has_pyright_section() -> None:
    """orchestrator/pyproject.toml must have a [tool.pyright] section."""
    toml_path = _PACKAGE_ROOT / "pyproject.toml"
    assert toml_path.is_file(), f"pyproject.toml not found at {toml_path}"
    with open(toml_path, "rb") as fh:
        config = tomllib.load(fh)
    assert "pyright" in config.get("tool", {}), (
        f"[tool.pyright] section missing from {toml_path}. "
        "The TYPE_CHECKING Protocol conformance block in test_workflow_e2e.py "
        "relies on pyright being configured for this package. See task 699."
    )


# ---------------------------------------------------------------------------
# (b) pyright include list must contain 'tests'
# ---------------------------------------------------------------------------


def test_orchestrator_pyright_include_contains_tests() -> None:
    """[tool.pyright] include must list 'tests' so pyright covers test_workflow_e2e.py.

    If this test fails, the TYPE_CHECKING Protocol conformance block at
    test_workflow_e2e.py:2094 silently becomes a no-op because pyright no longer
    type-checks this file. See task 699.
    """
    toml_path = _PACKAGE_ROOT / "pyproject.toml"
    assert toml_path.is_file(), f"pyproject.toml not found at {toml_path}"
    with open(toml_path, "rb") as fh:
        config = tomllib.load(fh)
    include = config.get("tool", {}).get("pyright", {}).get("include", [])
    assert "tests" in include, (
        f"'tests' is not in [tool.pyright] include = {include!r} "
        f"in {toml_path}. "
        "Without this entry pyright does not type-check test_workflow_e2e.py, "
        "so the TYPE_CHECKING Protocol conformance block at line 2094 silently "
        "becomes a no-op. Re-add 'tests' to [tool.pyright] include, or update "
        "the conformance strategy. See task 699."
    )


# ---------------------------------------------------------------------------
# (c) hooks/project-checks must invoke pyright on orchestrator
# ---------------------------------------------------------------------------


def test_project_checks_hook_invokes_pyright_on_orchestrator() -> None:
    """hooks/project-checks must list 'orchestrator' in PYRIGHT_PACKAGES and invoke pyright.

    If this test fails, the commit-gate that enforces the TYPE_CHECKING Protocol
    conformance block in test_workflow_e2e.py no longer exists for the orchestrator
    package. Protocol drift between FakeScheduler/_EvalScheduler and _SchedulerLike
    will go undetected until runtime. See task 699.
    """
    hook_path = _REPO_ROOT / "hooks" / "project-checks"
    assert hook_path.is_file(), (
        f"hooks/project-checks not found at {hook_path}. "
        "This hook is the commit-time gate for pyright on orchestrator. See task 699."
    )
    content = hook_path.read_text()
    assert "orchestrator" in content, (
        f"'orchestrator' does not appear in {hook_path}. "
        "The hook must list orchestrator in PYRIGHT_PACKAGES so that "
        "`uv run pyright` is invoked from orchestrator/ on main-branch commits. "
        "See task 699."
    )
    assert "uv run pyright" in content, (
        f"'uv run pyright' does not appear in {hook_path}. "
        "The hook must invoke pyright to enforce Protocol conformance. See task 699."
    )


# ---------------------------------------------------------------------------
# (d) The TYPE_CHECKING conformance block must still exist in test_workflow_e2e.py
# ---------------------------------------------------------------------------


def test_workflow_e2e_conformance_block_is_present() -> None:
    """The TYPE_CHECKING Protocol conformance block must be present in test_workflow_e2e.py.

    This pins the block's existence so it cannot be deleted without also updating
    this regression test. Without the block, Protocol drift between FakeScheduler/
    _EvalScheduler and _SchedulerLike would be undetectable by pyright regardless
    of how the gate is configured. See task 699.
    """
    e2e_path = _PACKAGE_ROOT / "tests" / "test_workflow_e2e.py"
    assert e2e_path.is_file(), f"test_workflow_e2e.py not found at {e2e_path}"
    content = e2e_path.read_text()
    required_fragments = [
        "if TYPE_CHECKING:",
        "_fake_scheduler_conforms: _SchedulerLike",
        "_eval_scheduler_conforms: _SchedulerLike",
    ]
    for fragment in required_fragments:
        assert fragment in content, (
            f"Expected fragment {fragment!r} not found in {e2e_path}. "
            "The TYPE_CHECKING Protocol conformance block must be present for "
            "pyright to enforce _SchedulerLike conformance on FakeScheduler and "
            "_EvalScheduler. See task 699."
        )
