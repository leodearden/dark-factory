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

    def test_dark_factory_is_blocked(self):
        """is_snapshot_write_blocked('dark_factory') must return True (task 2325).

        dark_factory has never written a task_count_snapshot — the per-project
        census is not in use for it, so the ABSENCE of a snapshot is
        correct-by-design.
        """
        from fused_memory.reconciliation.policies import is_snapshot_write_blocked

        assert is_snapshot_write_blocked('dark_factory') is True, (
            "is_snapshot_write_blocked('dark_factory') must return True; "
            "snapshot write paths are blocked-by-design for this project"
        )

    def test_solar_challenge_platform_is_blocked(self):
        """is_snapshot_write_blocked('solar_challenge_platform') must return True (task 2325).

        solar_challenge_platform has never written a task_count_snapshot — the
        per-project census is not in use for it, so the ABSENCE of a snapshot
        is correct-by-design.
        """
        from fused_memory.reconciliation.policies import is_snapshot_write_blocked

        assert is_snapshot_write_blocked('solar_challenge_platform') is True, (
            "is_snapshot_write_blocked('solar_challenge_platform') must return True; "
            "snapshot write paths are blocked-by-design for this project"
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

    def test_autopilot_video_project_id_constant_value(self):
        """AUTOPILOT_VIDEO_PROJECT_ID must equal 'autopilot_video' (no regression)."""
        from fused_memory.reconciliation.policies.autopilot_video import (
            AUTOPILOT_VIDEO_PROJECT_ID,
        )

        assert AUTOPILOT_VIDEO_PROJECT_ID == 'autopilot_video'

    def test_know_live_is_blocked(self):
        """is_snapshot_write_blocked('know_live') must return True (task 1943)."""
        from fused_memory.reconciliation.policies import is_snapshot_write_blocked

        assert is_snapshot_write_blocked('know_live') is True, (
            "is_snapshot_write_blocked('know_live') must return True; "
            "snapshot write paths are blocked-by-design for this project"
        )

    def test_know_live_project_id_constant_value(self):
        """KNOW_LIVE_PROJECT_ID must equal 'know_live' (task 1943)."""
        from fused_memory.reconciliation.policies.know_live import (
            KNOW_LIVE_PROJECT_ID,
        )

        assert KNOW_LIVE_PROJECT_ID == 'know_live'

    def test_dark_factory_project_id_constant_value(self):
        """DARK_FACTORY_PROJECT_ID must equal 'dark_factory' (task 2325)."""
        from fused_memory.reconciliation.policies.dark_factory import (
            DARK_FACTORY_PROJECT_ID,
        )

        assert DARK_FACTORY_PROJECT_ID == 'dark_factory'

    def test_solar_challenge_platform_project_id_constant_value(self):
        """SOLAR_CHALLENGE_PLATFORM_PROJECT_ID must equal 'solar_challenge_platform' (task 2325)."""
        from fused_memory.reconciliation.policies.solar_challenge_platform import (
            SOLAR_CHALLENGE_PLATFORM_PROJECT_ID,
        )

        assert SOLAR_CHALLENGE_PLATFORM_PROJECT_ID == 'solar_challenge_platform'

    def test_snapshot_write_blocked_projects_registry_includes_new_exemptions(self):
        """SNAPSHOT_WRITE_BLOCKED_PROJECTS must include both new exemptions (task 2325)."""
        from fused_memory.reconciliation.policies import SNAPSHOT_WRITE_BLOCKED_PROJECTS

        assert {'dark_factory', 'solar_challenge_platform'} <= SNAPSHOT_WRITE_BLOCKED_PROJECTS
