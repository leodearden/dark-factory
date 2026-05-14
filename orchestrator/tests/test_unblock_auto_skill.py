"""Tests for the unblock-auto skill's SKILL.md frontmatter and content.

Pins the SKILL.md to ensure it has the required name, description wording,
and body instructions about emitting proposals to metadata.  A failing test
here means the file was accidentally deleted, renamed, or its contract-critical
wording was changed.
"""

from __future__ import annotations

from pathlib import Path

import yaml

# Project root is three levels up from orchestrator/tests/
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_SKILL_PATH = _PROJECT_ROOT / 'skills' / 'unblock-auto' / 'SKILL.md'


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split '---\\n<YAML>\\n---\\n<body>' into (frontmatter_dict, body_str).

    Returns (empty-dict, text) if the file has no frontmatter delimiters.
    """
    if not text.startswith('---'):
        return {}, text
    parts = text.split('---', maxsplit=2)
    # parts[0] is the empty string before the first ---
    # parts[1] is the YAML block
    # parts[2] is the body
    if len(parts) < 3:
        return {}, text
    frontmatter = yaml.safe_load(parts[1]) or {}
    body = parts[2].strip()
    return frontmatter, body


class TestUnblockAutoSkillMd:
    """Pins the contract of skills/unblock-auto/SKILL.md."""

    def test_skill_md_has_required_frontmatter(self):
        """SKILL.md exists, has name==unblock-auto, and the description mentions
        dry-run or read-only or autonomous intent."""
        assert _SKILL_PATH.exists(), (
            f'skills/unblock-auto/SKILL.md not found at {_SKILL_PATH}. '
            'Run step-2 (impl) to create the file.'
        )

        raw = _SKILL_PATH.read_text()
        frontmatter, body = _parse_frontmatter(raw)

        # name must be exactly 'unblock-auto'
        assert frontmatter.get('name') == 'unblock-auto', (
            f"Expected name=='unblock-auto', got {frontmatter.get('name')!r}"
        )

        # description must be a non-empty string
        description = frontmatter.get('description', '')
        assert isinstance(description, str) and description.strip(), (
            'SKILL.md frontmatter must have a non-empty description field'
        )

        # description must convey the dry-run / read-only / autonomous nature
        description_lower = description.lower()
        assert any(kw in description_lower for kw in ('dry-run', 'read-only', 'autonomous')), (
            f'description must mention "dry-run", "read-only", or "autonomous". Got: {description!r}'
        )

        # body must instruct the agent to emit a proposal (to be written to metadata)
        body_lower = body.lower()
        assert 'proposal' in body_lower, (
            'SKILL.md body must mention "proposal" (the agent emits a proposal to be written to metadata)'
        )
        assert 'metadata' in body_lower, (
            'SKILL.md body must mention "metadata" (proposals are persisted via dry_run_proposals[])'
        )
