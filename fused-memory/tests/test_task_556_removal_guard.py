"""Task-556 transient scaffolding — slated for deletion in step-3.

This file exists solely to provide a TDD red→green signal for the removal of
TestDocstringAccuracyForceDryRunClass from test_graphiti_quality_433.py.

DO NOT keep this file after task 556 step-3 is committed.
"""
from pathlib import Path


def test_docstring_accuracy_class_has_been_removed() -> None:
    """Assert that 'class TestDocstringAccuracyForceDryRunClass' is absent from
    the sibling test file.

    Fails (red) while the class is still present; passes (green) once it has
    been deleted in step-2.
    """
    sibling = Path(__file__).parent / "test_graphiti_quality_433.py"
    source = sibling.read_text()
    assert "class TestDocstringAccuracyForceDryRunClass" not in source, (
        "TestDocstringAccuracyForceDryRunClass must be deleted from "
        "test_graphiti_quality_433.py (task 556)"
    )
