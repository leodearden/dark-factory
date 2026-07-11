"""Tests for orchestrator.verify_plan — derive_verify_plan() + FileKind.

Task γ of the verify-plan PRD (plans/verify-plan-prd.md §Contract·derive_verify_plan).
Unifies the twice-fixed scope decision (scope_module_config + _build_fallback_config)
behind a single pure ``derive_verify_plan()`` and a ``FileKind`` enum, so file
classification happens exactly once instead of being reimplemented per call site.

No source stub exists yet — every test in this module is RED until
orchestrator/src/orchestrator/verify_plan.py is created (step-2 onward).

GOLDEN fixtures below reconstruct the historical incident diffs from the cited
fix commits (all git-verified present on this branch) rather than inventing
arbitrary file lists — see PRD resolved-decision 6.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# GOLDEN incident fixtures
# ---------------------------------------------------------------------------

# task-1077: conftest.py must trigger the full unscoped suite, never be passed
# directly to pytest as a target (pytest >= 9 exits 1 "no tests ran" on a bare
# conftest target). The same fix landed twice — scope_module_config
# (d7504d432d) and _build_fallback_config (cb7277926d) — the exact "same bug
# fixed in both functions" class derive_verify_plan closes by construction.
ROOT_CONFTEST_DIFF: list[str] = ['orchestrator/tests/conftest.py']

# task-1852: a non-test data module under tests/ — a test-tree member but NOT
# pytest-collectable (passing it to pytest produces rc=5 "no tests ran").
# Fixed twice: scope_module_config (4fbed6c4fb, has_test_data -> full suite)
# and _build_fallback_config (7c9b316260, bare-fallback -> SKIPPED/None).
DATA_MODULE_DIFF: list[str] = ['shared/tests/silent_fallthrough_allowlist.py']

# A Protocol-defining source file (D2): file-scoped pyright cannot verify
# cross-file Protocol conformance, so a STRUCTURAL file must widen pyright to
# the unscoped package-wide command in BOTH the module and fallback paths —
# the latent gap _build_fallback_config never closed (only scope_module_config
# widened for this case).
STRUCTURAL_DIFF: list[str] = ['orchestrator/src/orchestrator/interfaces.py']

# Canned file contents for the dict-backed fake worktree_reader below. Only
# STRUCTURAL_DIFF's file has real (Protocol-bearing) content; every other
# path — including ROOT_CONFTEST_DIFF/DATA_MODULE_DIFF's files, and any path
# absent from this dict entirely — reads back as None, which classify_file
# must treat as "STRUCTURAL simply not detected", never an error.
_FAKE_FILE_CONTENTS: dict[str, str] = {
    STRUCTURAL_DIFF[0]: 'class Foo(Protocol):\n    def method(self) -> None: ...\n',
}


def fake_worktree_reader(path: str) -> str | None:
    """Dict-backed stand-in for real file I/O (``Callable[[str], str | None]``).

    Keeps derive_verify_plan pure and unit-testable without touching a real
    filesystem: returns the canned Protocol content for STRUCTURAL_DIFF's
    file, else None.
    """
    return _FAKE_FILE_CONTENTS.get(path)
