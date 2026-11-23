"""Tests for per-project reconciliation policy helpers.

Covers:
- AUTOPILOT_VIDEO_CONTAMINATION_CEILING_RETIRED flag (policies/autopilot_video.py)
- CONTAMINATION_CEILING_RETIRED_PROJECTS registry (policies/__init__.py)
- is_contamination_ceiling_retired(project_id) helper (policies/__init__.py)
"""

from __future__ import annotations


class TestIsContaminationCeilingRetired:
    """is_contamination_ceiling_retired(project_id) routes membership to the registry.

    The autopilot_video Stage-1 task-ID "contamination ceiling" was retired
    (task 2818 decision B; live memories retired 2026-07-19).

    All assertions are runtime-behavior only — no docstring/introspection checks.
    RED: names do not exist yet.
    """

    def test_autopilot_video_ceiling_retired_constant_is_true(self):
        """AUTOPILOT_VIDEO_CONTAMINATION_CEILING_RETIRED must be True."""
        from fused_memory.reconciliation.policies.autopilot_video import (
            AUTOPILOT_VIDEO_CONTAMINATION_CEILING_RETIRED,
        )

        assert AUTOPILOT_VIDEO_CONTAMINATION_CEILING_RETIRED is True

    def test_autopilot_video_is_ceiling_retired(self):
        """is_contamination_ceiling_retired('autopilot_video') must return True."""
        from fused_memory.reconciliation.policies import is_contamination_ceiling_retired

        assert is_contamination_ceiling_retired('autopilot_video') is True, (
            "is_contamination_ceiling_retired('autopilot_video') must return True; "
            "the Stage-1 task-ID contamination ceiling is retired for this project"
        )

    def test_reify_is_not_ceiling_retired(self):
        """is_contamination_ceiling_retired('reify') must return False."""
        from fused_memory.reconciliation.policies import is_contamination_ceiling_retired

        assert is_contamination_ceiling_retired('reify') is False

    def test_empty_string_is_not_ceiling_retired(self):
        """is_contamination_ceiling_retired('') must return False (not raise)."""
        from fused_memory.reconciliation.policies import is_contamination_ceiling_retired

        assert is_contamination_ceiling_retired('') is False

    def test_none_is_not_ceiling_retired(self):
        """is_contamination_ceiling_retired(None) must return False (not raise)."""
        from fused_memory.reconciliation.policies import is_contamination_ceiling_retired

        assert is_contamination_ceiling_retired(None) is False  # type: ignore[arg-type]

    def test_ceiling_retired_projects_registry_includes_autopilot_video(self):
        """CONTAMINATION_CEILING_RETIRED_PROJECTS must include 'autopilot_video'."""
        from fused_memory.reconciliation.policies import (
            CONTAMINATION_CEILING_RETIRED_PROJECTS,
        )

        assert 'autopilot_video' in CONTAMINATION_CEILING_RETIRED_PROJECTS
