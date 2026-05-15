"""Tests for orchestrator.agents.skill_prompt.load_skill_system_prompt.

Step 1 — failing tests (module does not exist yet).
"""

from __future__ import annotations

import pytest


class TestLoadSkillSystemPrompt:
    """Tests for the shared SKILL.md loader."""

    def test_unblock_auto_strips_frontmatter(self) -> None:
        """For 'unblock-auto', the returned body must not start with '---'."""
        from orchestrator.agents.skill_prompt import load_skill_system_prompt

        body = load_skill_system_prompt('unblock-auto')
        assert not body.startswith('---'), (
            "load_skill_system_prompt('unblock-auto') must strip the YAML frontmatter"
        )

    def test_unblock_auto_contains_known_phrase(self) -> None:
        """The unblock-auto body contains a recognisable phrase from the skill."""
        from orchestrator.agents.skill_prompt import load_skill_system_prompt

        body = load_skill_system_prompt('unblock-auto')
        # A stable phrase in skills/unblock-auto/SKILL.md
        assert 'dry-run' in body.lower() or 'unblock' in body.lower(), (
            "Expected 'unblock-auto' SKILL.md body to contain a recognisable phrase"
        )

    def test_escalation_watcher_auto_strips_frontmatter(self) -> None:
        """For 'escalation-watcher-auto', body must not start with '---'."""
        from orchestrator.agents.skill_prompt import load_skill_system_prompt

        body = load_skill_system_prompt('escalation-watcher-auto')
        assert not body.startswith('---'), (
            "load_skill_system_prompt('escalation-watcher-auto') must strip the YAML frontmatter"
        )

    def test_escalation_watcher_auto_non_empty(self) -> None:
        """The escalation-watcher-auto body must be non-empty."""
        from orchestrator.agents.skill_prompt import load_skill_system_prompt

        body = load_skill_system_prompt('escalation-watcher-auto')
        assert len(body.strip()) > 0, (
            "load_skill_system_prompt('escalation-watcher-auto') must return non-empty body"
        )

    def test_unknown_skill_raises_file_not_found(self) -> None:
        """An unknown skill name raises FileNotFoundError."""
        from orchestrator.agents.skill_prompt import load_skill_system_prompt

        with pytest.raises(FileNotFoundError):
            load_skill_system_prompt('nonexistent-skill-xyzzy-12345')
