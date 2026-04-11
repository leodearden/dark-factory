"""Regression tests that guard the CI enforcement of the TYPE_CHECKING Protocol
conformance block in test_workflow_e2e.py.

The ``if TYPE_CHECKING:`` Protocol conformance block near the bottom of
test_workflow_e2e.py is invisible to pytest — pyright must be wired into the
commit gate for it to catch Protocol drift.
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

import re
import tomllib
from pathlib import Path
from typing import Any

# Locate package root (orchestrator/) from the tests/ directory.
_PACKAGE_ROOT = Path(__file__).parent.parent

# Locate repo root (dark-factory/) from orchestrator/tests/.
_REPO_ROOT = Path(__file__).parents[2]

# Pre-compiled pattern for detecting a PYRIGHT_PACKAGES for-loop in hook content.
_LOOP_PATTERN = re.compile(
    r'for\s+\w+\s+in\s+"\$\{PYRIGHT_PACKAGES\[@\]\}"\s*;\s*do(.*?)done',
    re.DOTALL,
)


def _parse_pyright_packages(content: str) -> list[str] | None:
    """Parse PYRIGHT_PACKAGES=(...) from hook file content, return the list or None.

    Returns the whitespace-split list of package names inside the Bash array
    declaration.  Returns None (not an empty list) when no PYRIGHT_PACKAGES
    declaration is found, so callers can distinguish ``missing declaration``
    from ``declaration present but empty``.

    Uses ``re.MULTILINE`` with a ``^`` anchor so only a real array declaration
    at the start of a line is matched, not a PYRIGHT_PACKAGES substring that
    appears inside a comment or an unrelated string.

    Bash line continuations (a ``\\`` backslash followed by a newline) are
    normalized to a single space before matching, so a multi-line declaration
    such as::

        PYRIGHT_PACKAGES=( \\
            fused-memory \\
            orchestrator \\
        )

    is parsed correctly without stray ``\\`` tokens in the result.
    """
    normalized = content.replace("\\\n", " ")
    match = re.search(r"^PYRIGHT_PACKAGES=\(([^)]*)\)", normalized, re.MULTILINE)
    return match.group(1).split() if match else None


def _hook_invokes_pyright_in_loop(content: str) -> bool:
    """Return True iff ``uv run pyright`` appears inside a PYRIGHT_PACKAGES for-loop.

    Captures the body of the ``for pkg in "${PYRIGHT_PACKAGES[@]}"; do ... done``
    block using ``re.DOTALL`` and checks that ``uv run pyright`` is present
    *inside* the captured body.  A stray ``uv run pyright`` elsewhere in the
    file (e.g. in a comment) does not falsely pass, and a loop that iterates
    PYRIGHT_PACKAGES but calls some other command inside fails correctly.
    """
    match = _LOOP_PATTERN.search(content)
    return bool(match and "uv run pyright" in match.group(1))


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
# (c) hooks/project-checks must invoke pyright on orchestrator
# ---------------------------------------------------------------------------


def test_project_checks_hook_invokes_pyright_on_orchestrator() -> None:
    """hooks/project-checks must list 'orchestrator' in PYRIGHT_PACKAGES and loop over it.

    Uses _parse_pyright_packages to read the PYRIGHT_PACKAGES=(...) array
    specifically (not ALL_PACKAGES, which is used for ruff).  Uses
    _hook_invokes_pyright_in_loop to verify that ``uv run pyright`` appears
    inside the loop body — not merely somewhere in the file.

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
    packages = _parse_pyright_packages(content)
    assert packages is not None, (
        f"PYRIGHT_PACKAGES=(...) declaration not found in {hook_path}. "
        "The hook must declare PYRIGHT_PACKAGES so pyright runs on each listed "
        "package on main-branch commits. See task 699."
    )
    assert "orchestrator" in packages, (
        f"'orchestrator' not in PYRIGHT_PACKAGES={packages} in {hook_path}. "
        "Without this, pyright never runs on orchestrator and the TYPE_CHECKING "
        "Protocol conformance block in test_workflow_e2e.py silently becomes "
        "un-enforced. Note that ALL_PACKAGES (used for ruff) is NOT sufficient — "
        "pyright must be wired to PYRIGHT_PACKAGES specifically. See task 699."
    )
    assert _hook_invokes_pyright_in_loop(content), (
        f"{hook_path} declares PYRIGHT_PACKAGES but does not iterate it with "
        "'uv run pyright' in the loop body. The hook must loop over "
        "PYRIGHT_PACKAGES and invoke pyright inside the loop. See task 699."
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


# ---------------------------------------------------------------------------
# Unit tests for _parse_pyright_packages helper
# ---------------------------------------------------------------------------


def test_parse_pyright_packages_extracts_listed_packages() -> None:
    """_parse_pyright_packages extracts the list of packages from PYRIGHT_PACKAGES=(...)."""
    content = "PYRIGHT_PACKAGES=(fused-memory orchestrator dashboard)\n"
    result = _parse_pyright_packages(content)
    assert result == ["fused-memory", "orchestrator", "dashboard"], (
        f"Expected ['fused-memory', 'orchestrator', 'dashboard'], got {result!r}"
    )


def test_parse_pyright_packages_distinguishes_from_all_packages() -> None:
    """_parse_pyright_packages must read PYRIGHT_PACKAGES, not ALL_PACKAGES.

    This is the critical regression canary: if orchestrator is in ALL_PACKAGES
    but removed from PYRIGHT_PACKAGES, the naive ``'orchestrator' in content``
    check still passes (ALL_PACKAGES names it), silently disabling pyright
    for orchestrator. The helper must parse the PYRIGHT_PACKAGES array specifically.
    """
    content = (
        "ALL_PACKAGES=(shared escalation fused-memory orchestrator dashboard)\n"
        "PYRIGHT_PACKAGES=(fused-memory dashboard)\n"
    )
    result = _parse_pyright_packages(content)
    assert result is not None, "PYRIGHT_PACKAGES declaration should be found in content"
    assert "orchestrator" not in result, (
        f"'orchestrator' found in {result!r} but should not be — "
        "it was deliberately removed from PYRIGHT_PACKAGES. "
        "This confirms the helper reads from PYRIGHT_PACKAGES, not ALL_PACKAGES."
    )


def test_parse_pyright_packages_returns_none_when_missing() -> None:
    """_parse_pyright_packages returns None when PYRIGHT_PACKAGES is absent."""
    content = "ALL_PACKAGES=(shared escalation fused-memory orchestrator dashboard)\n"
    result = _parse_pyright_packages(content)
    assert result is None, (
        f"Expected None when PYRIGHT_PACKAGES is absent, got {result!r}"
    )


def test_parse_pyright_packages_returns_empty_list_when_declared_empty() -> None:
    """_parse_pyright_packages returns [] (not None) for PYRIGHT_PACKAGES=().

    This pins the intentional empty-vs-None distinction: a present-but-empty
    declaration PYRIGHT_PACKAGES=() returns an empty list, while an absent
    declaration returns None.  Callers use this distinction to report
    'declaration missing' vs 'declaration present but empty' in their messages.
    """
    content = "PYRIGHT_PACKAGES=()\n"
    result = _parse_pyright_packages(content)
    assert result == [], (
        f"Expected [] when PYRIGHT_PACKAGES=() is declared empty, got {result!r}. "
        "The empty-vs-None distinction is intentional: [] means declared-but-empty, "
        "None means absent."
    )


def test_parse_pyright_packages_handles_line_continuations() -> None:
    """_parse_pyright_packages correctly parses a Bash array written with line continuations.

    A multi-line PYRIGHT_PACKAGES declaration using ``\\`` continuation tokens:

        PYRIGHT_PACKAGES=( \\
            fused-memory \\
            orchestrator \\
            dashboard \\
        )

    must return ['fused-memory', 'orchestrator', 'dashboard'] without stray
    '\\\\' tokens in the result.  Without normalization, .split() on the captured
    body leaves stray '\\\\' tokens and the result would be
    ['\\\\', 'fused-memory', '\\\\', 'orchestrator', '\\\\', 'dashboard', '\\\\'],
    which would hide a real regression because 'orchestrator' happens to survive
    the split even with the stray tokens.
    """
    content = (
        "PYRIGHT_PACKAGES=( \\\n"
        "    fused-memory \\\n"
        "    orchestrator \\\n"
        "    dashboard \\\n"
        ")\n"
    )
    result = _parse_pyright_packages(content)
    assert result == ["fused-memory", "orchestrator", "dashboard"], (
        f"Expected ['fused-memory', 'orchestrator', 'dashboard'], got {result!r}. "
        "Line-continuation tokens ('\\\\') must be stripped from the result."
    )


# ---------------------------------------------------------------------------
# Unit tests for _hook_invokes_pyright_in_loop helper
# ---------------------------------------------------------------------------


def test_hook_invokes_pyright_in_loop_detects_loop_body() -> None:
    """_hook_invokes_pyright_in_loop returns True when uv run pyright is inside the loop body."""
    content = (
        'for pkg in "${PYRIGHT_PACKAGES[@]}"; do\n'
        '    (cd "$dir" && uv run pyright)\n'
        "done\n"
    )
    assert _hook_invokes_pyright_in_loop(content) is True


def test_hook_invokes_pyright_in_loop_rejects_bare_pyright() -> None:
    """_hook_invokes_pyright_in_loop returns False when uv run pyright appears outside any loop."""
    content = "uv run pyright\n"
    assert _hook_invokes_pyright_in_loop(content) is False


def test_hook_invokes_pyright_in_loop_rejects_loop_without_pyright() -> None:
    """_hook_invokes_pyright_in_loop returns False when the PYRIGHT_PACKAGES loop lacks pyright."""
    content = (
        'for pkg in "${PYRIGHT_PACKAGES[@]}"; do\n'
        '    echo "$pkg"\n'
        "done\n"
    )
    assert _hook_invokes_pyright_in_loop(content) is False


# ---------------------------------------------------------------------------
# Meta-test: no stale line-number citations in this file's own docstrings
# ---------------------------------------------------------------------------


def test_module_docstrings_do_not_cite_stale_line_numbers() -> None:
    """This file must not contain line-number citations into test_workflow_e2e.py.

    Line numbers go stale every time test_workflow_e2e.py is edited. The block
    existence is pinned by content in test_workflow_e2e_conformance_block_is_present,
    so line numbers add no value and only mislead. This meta-test catches any
    future regression to line-number citations without requiring a code reviewer
    to notice.

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
