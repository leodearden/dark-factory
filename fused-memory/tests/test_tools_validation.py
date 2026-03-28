"""Regression tests verifying that tools.py delegates validation to the shared validators.

These tests replace the original vacuous tests that either called shared validators
directly (not through tools.py at all) or checked hasattr() on closures that were
never module-level attributes.
"""

from __future__ import annotations

import fused_memory.server.tools as tools_module
from fused_memory.utils.validation import validate_project_id, validate_project_root


class TestToolsDelegateToSharedValidators:
    """Object identity checks proving tools.py imports and re-exports the shared validators.

    If tools.py ever re-introduces private closures for validation, these tests fail
    immediately — the module attribute would no longer be the same object as the
    shared validator imported from utils.validation.
    """

    def test_validate_project_id_is_shared_validator(self):
        """tools.validate_project_id must be the exact same object as utils.validation.validate_project_id.

        tools.py line 14: `from fused_memory.utils.validation import validate_project_id`
        This creates a module-level attribute that is identical (by identity) to the
        shared validator. Any re-implementation would break this check.
        """
        assert tools_module.validate_project_id is validate_project_id, (
            "tools.validate_project_id is not the shared validator from utils.validation. "
            "tools.py must delegate to the shared validator, not re-implement it."
        )

    def test_validate_project_root_is_shared_validator(self):
        """tools.validate_project_root must be the exact same object as utils.validation.validate_project_root."""
        assert tools_module.validate_project_root is validate_project_root, (
            "tools.validate_project_root is not the shared validator from utils.validation. "
            "tools.py must delegate to the shared validator, not re-implement it."
        )
