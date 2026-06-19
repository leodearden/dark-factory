"""Tests for per-project reconciliation policy helpers.

Covers:
- AUTOPILOT_VIDEO_SNAPSHOT_WRITES_BLOCKED flag (policies/autopilot_video.py)
- SNAPSHOT_WRITE_BLOCKED_PROJECTS registry (policies/__init__.py)
- is_snapshot_write_blocked(project_id) helper (policies/__init__.py)

RED until step-2 adds the constants and helper.
"""

from __future__ import annotations


class TestIsSnapshotWriteBlocked:
    """is_snapshot_write_blocked(project_id) routes membership to the registry.

    All assertions are runtime-behavior only — no docstring/introspection checks.
    RED: names do not exist yet.
    """

    def test_autopilot_video_is_blocked(self):
        """is_snapshot_write_blocked('autopilot_video') must return True."""
        from fused_memory.reconciliation.policies import is_snapshot_write_blocked

        assert is_snapshot_write_blocked('autopilot_video') is True, (
            "is_snapshot_write_blocked('autopilot_video') must return True; "
            "snapshot write paths are blocked-by-design for this project"
        )

    def test_dark_factory_is_not_blocked(self):
        """is_snapshot_write_blocked('dark_factory') must return False."""
        from fused_memory.reconciliation.policies import is_snapshot_write_blocked

        assert is_snapshot_write_blocked('dark_factory') is False, (
            "is_snapshot_write_blocked('dark_factory') must return False; "
            "dark_factory is not in the blocked set"
        )

    def test_reify_is_not_blocked(self):
        """is_snapshot_write_blocked('reify') must return False."""
        from fused_memory.reconciliation.policies import is_snapshot_write_blocked

        assert is_snapshot_write_blocked('reify') is False

    def test_empty_string_is_not_blocked(self):
        """is_snapshot_write_blocked('') must return False (not raise)."""
        from fused_memory.reconciliation.policies import is_snapshot_write_blocked

        assert is_snapshot_write_blocked('') is False

    def test_none_is_not_blocked(self):
        """is_snapshot_write_blocked(None) must return False (not raise)."""
        from fused_memory.reconciliation.policies import is_snapshot_write_blocked

        assert is_snapshot_write_blocked(None) is False  # type: ignore[arg-type]

    def test_autopilot_video_snapshot_writes_blocked_flag_is_truthy(self):
        """AUTOPILOT_VIDEO_SNAPSHOT_WRITES_BLOCKED must be truthy."""
        from fused_memory.reconciliation.policies.autopilot_video import (
            AUTOPILOT_VIDEO_SNAPSHOT_WRITES_BLOCKED,
        )

        assert AUTOPILOT_VIDEO_SNAPSHOT_WRITES_BLOCKED, (
            "AUTOPILOT_VIDEO_SNAPSHOT_WRITES_BLOCKED must be True; "
            "both snapshot write paths are blocked-by-design"
        )

    def test_autopilot_video_in_snapshot_write_blocked_projects(self):
        """'autopilot_video' must be a member of SNAPSHOT_WRITE_BLOCKED_PROJECTS."""
        from fused_memory.reconciliation.policies import SNAPSHOT_WRITE_BLOCKED_PROJECTS

        assert 'autopilot_video' in SNAPSHOT_WRITE_BLOCKED_PROJECTS, (
            f"'autopilot_video' must be in SNAPSHOT_WRITE_BLOCKED_PROJECTS; "
            f"got {SNAPSHOT_WRITE_BLOCKED_PROJECTS!r}"
        )

    def test_snapshot_write_blocked_projects_is_frozenset(self):
        """SNAPSHOT_WRITE_BLOCKED_PROJECTS must be a frozenset."""
        from fused_memory.reconciliation.policies import SNAPSHOT_WRITE_BLOCKED_PROJECTS

        assert isinstance(SNAPSHOT_WRITE_BLOCKED_PROJECTS, frozenset), (
            f"SNAPSHOT_WRITE_BLOCKED_PROJECTS must be a frozenset; "
            f"got {type(SNAPSHOT_WRITE_BLOCKED_PROJECTS)!r}"
        )

    def test_autopilot_video_project_id_constant_value(self):
        """AUTOPILOT_VIDEO_PROJECT_ID must equal 'autopilot_video' (no regression)."""
        from fused_memory.reconciliation.policies.autopilot_video import (
            AUTOPILOT_VIDEO_PROJECT_ID,
        )

        assert AUTOPILOT_VIDEO_PROJECT_ID == 'autopilot_video'
