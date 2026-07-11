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

from orchestrator.verify_plan import FileKind, classify_file

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


# ---------------------------------------------------------------------------
# FileKind / classify_file (step-1: RED)
# ---------------------------------------------------------------------------


class TestFileKindMembers:
    """FileKind is a plain Enum with exactly the six classification kinds."""

    def test_members_present(self):
        names = {member.name for member in FileKind}
        assert names == {
            'CONFTEST', 'COLLECTABLE_TEST', 'TEST_DATA', 'STRUCTURAL', 'SOURCE', 'INERT',
        }


class TestClassifyFile:
    """classify_file(path, content) -> FileKind runs the classification ladder exactly once.

    Precedence: CONFTEST > COLLECTABLE_TEST > TEST_DATA > STRUCTURAL > SOURCE > INERT.
    """

    # -- one representative path per FileKind ---------------------------------

    def test_conftest_under_subdirectory(self):
        assert classify_file('orchestrator/tests/conftest.py', None) is FileKind.CONFTEST

    def test_conftest_at_root(self):
        assert classify_file('conftest.py', None) is FileKind.CONFTEST

    def test_collectable_test_prefix(self):
        assert classify_file('a/test_x.py', None) is FileKind.COLLECTABLE_TEST

    def test_collectable_test_suffix(self):
        assert classify_file('a/x_test.py', None) is FileKind.COLLECTABLE_TEST

    def test_data_module_under_tests_dir(self):
        """Task-1852 golden: not conftest, not collectable, but a test-tree member."""
        assert classify_file(DATA_MODULE_DIFF[0], None) is FileKind.TEST_DATA

    def test_structural_protocol_source_file(self):
        content = _FAKE_FILE_CONTENTS[STRUCTURAL_DIFF[0]]
        assert classify_file(STRUCTURAL_DIFF[0], content) is FileKind.STRUCTURAL

    def test_structural_typeddict_source_file(self):
        content = 'class Bar(TypedDict):\n    name: str\n'
        assert classify_file('orchestrator/src/orchestrator/types.py', content) is FileKind.STRUCTURAL

    def test_plain_source_file(self):
        content = 'def do_thing(x: int) -> str:\n    return str(x)\n'
        assert classify_file('orchestrator/src/orchestrator/utils.py', content) is FileKind.SOURCE

    def test_non_python_path_is_inert(self):
        assert classify_file('docs/README.md', None) is FileKind.INERT
        assert classify_file('scripts/deploy.yaml', None) is FileKind.INERT
        assert classify_file('crates/foo/src/lib.rs', None) is FileKind.INERT

    # -- precedence assertions -------------------------------------------------

    def test_test_data_beats_structural(self):
        """A data module under tests/ that ALSO defines a Protocol stays TEST_DATA.

        TEST_DATA must outrank STRUCTURAL so a Protocol-defining data module
        under tests/ still full-suites (D1) rather than merely widening pyright.
        """
        content = 'class Foo(Protocol):\n    def method(self) -> None: ...\n'
        assert classify_file(DATA_MODULE_DIFF[0], content) is FileKind.TEST_DATA

    def test_conftest_beats_test_data(self):
        """conftest.py under tests/ classifies CONFTEST, never TEST_DATA."""
        assert classify_file('shared/tests/conftest.py', None) is FileKind.CONFTEST

    def test_none_content_never_raises_and_skips_structural(self):
        """content=None must never raise — STRUCTURAL is simply not detected."""
        result = classify_file('orchestrator/src/orchestrator/foo.py', None)
        assert result is FileKind.SOURCE
