"""Comment hygiene tests for maintenance entrypoints.

These tests use source introspection to assert that stale comments have been
replaced with accurate ones.  They fail if the stale text is still present and
pass only after the implementation steps have been applied.
"""

import pathlib

import pytest

# Locate source files relative to this test file
_MAINTENANCE = (
    pathlib.Path(__file__).parent.parent
    / "src"
    / "fused_memory"
    / "maintenance"
)

MAINTENANCE_FILES = [
    _MAINTENANCE / "cleanup_stale_edges.py",
    _MAINTENANCE / "verify_zombie_edges.py",
    _MAINTENANCE / "reindex.py",
]

STALE_TEXT = "CONFIG_PATH restoration below always runs"
CORRECT_TEXT = "Catch close() errors so they do not propagate out of the finally block"


@pytest.mark.parametrize("source_file", MAINTENANCE_FILES, ids=lambda p: p.name)
def test_stale_comment_absent(source_file: pathlib.Path) -> None:
    """The old, misleading CONFIG_PATH comment must not appear in the source."""
    text = source_file.read_text()
    assert STALE_TEXT not in text, (
        f"{source_file.name} still contains stale comment: {STALE_TEXT!r}"
    )


@pytest.mark.parametrize("source_file", MAINTENANCE_FILES, ids=lambda p: p.name)
def test_correct_comment_present(source_file: pathlib.Path) -> None:
    """The corrected comment explaining the real purpose must be present."""
    text = source_file.read_text()
    assert CORRECT_TEXT in text, (
        f"{source_file.name} is missing expected comment: {CORRECT_TEXT!r}"
    )
