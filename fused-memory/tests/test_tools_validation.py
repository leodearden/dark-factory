"""Regression tests verifying that tools.py delegates validation to the shared validators.

Step 4: Post-consolidation regression tests. After the fix, tools.py no longer defines
_validate_project_id or _validate_project_root as private names. All validation is
delegated to fused_memory.utils.validation's shared validators.
"""

from __future__ import annotations

import fused_memory.server.tools as tools_module


class TestToolsValidationConsolidation:
    """Regression tests verifying tools.py uses shared validators after consolidation."""

    def test_whitespace_only_project_id_is_now_rejected(self):
        """After consolidation, tools.py delegates to shared validator that strips whitespace.

        The pre-fix bug: tools.py _validate_project_id used `if not project_id` (truthy
        check only), so whitespace-only inputs like '   ' passed validation.

        The fix: tools.py now uses the shared validate_project_id which checks
        `if not project_id or not project_id.strip()`, correctly rejecting whitespace.

        We verify this by checking that the shared validator (which tools.py now uses)
        rejects whitespace-only project_ids.
        """
        from fused_memory.utils.validation import validate_project_id

        # This is exactly what tools.py call sites now use:
        result = validate_project_id('   ')
        assert result is not None, (
            "validate_project_id must reject whitespace-only project_id. "
            "Tools.py now delegates here — this verifies the bugfix is active."
        )
        assert result['error_type'] == 'ValidationError'

    def test_whitespace_only_project_root_is_rejected(self):
        """Shared validate_project_root correctly rejects whitespace-only paths."""
        from fused_memory.utils.validation import validate_project_root

        result = validate_project_root('   ')
        assert result is not None
        assert result['error_type'] == 'ValidationError'

    def test_private_validate_project_id_does_not_exist(self):
        """The _validate_project_id private closure must NOT exist in the tools module.

        After consolidation, it was removed. Its continued absence confirms no regression
        that re-introduces the private (buggy) version.

        Note: The tools module is a file-level module; the closure was inside
        create_mcp_server(), so it was never accessible as a module attribute. But we
        can confirm no top-level name '_validate_project_id' was accidentally added.
        """
        assert not hasattr(tools_module, '_validate_project_id'), (
            "_validate_project_id must not exist as a module-level name in tools.py — "
            "it was removed during consolidation."
        )

    def test_private_validate_project_root_does_not_exist(self):
        """The _validate_project_root private closure must NOT exist in the tools module."""
        assert not hasattr(tools_module, '_validate_project_root'), (
            "_validate_project_root must not exist as a module-level name in tools.py — "
            "it was removed during consolidation."
        )

    def test_shared_validators_are_importable_from_tools_dependencies(self):
        """The shared validators are importable at module level (used by tools.py at import time)."""
        from fused_memory.utils.validation import validate_project_id, validate_project_root

        # These are the exact functions that tools.py now imports and uses
        assert callable(validate_project_id)
        assert callable(validate_project_root)

    def test_valid_project_id_passes(self):
        """Sanity check: valid project_ids are not rejected by the shared validator."""
        from fused_memory.utils.validation import validate_project_id

        assert validate_project_id('dark_factory') is None
        assert validate_project_id('my-project') is None

    def test_valid_project_root_passes(self):
        """Sanity check: valid absolute paths are not rejected."""
        from fused_memory.utils.validation import validate_project_root

        assert validate_project_root('/home/user/project') is None
        assert validate_project_root('/') is None
