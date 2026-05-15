"""Shared SKILL.md loader for orchestrator agent invocations.

Provides a single, canonical function for loading the body of a skill's
SKILL.md file from the monorepo's ``skills/<skill_name>/SKILL.md`` path.

Previously each caller (e.g. dry_run_unblock._load_skill_system_prompt)
hardcoded its own parent-walk.  This module centralises the logic so
additional skills (e.g. escalation-watcher-auto) can reuse it without
duplication.  Task 1326.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_skill_system_prompt(skill_name: str) -> str:
    """Load the body of ``skills/<skill_name>/SKILL.md``, stripping frontmatter.

    Walks up from this file's location through parent directories until it
    finds a directory containing ``skills/<skill_name>/SKILL.md`` or reaches
    the repo root (identified by a ``.git`` marker).

    Frontmatter is the leading ``---`` … ``---`` block (YAML front-matter as
    used by most SKILL.md files).  It is stripped before returning; only the
    body text is returned.

    Args:
        skill_name: Directory name under ``skills/``, e.g. ``'unblock-auto'``
            or ``'escalation-watcher-auto'``.

    Returns:
        The stripped body text of the SKILL.md file.

    Raises:
        FileNotFoundError: If the SKILL.md cannot be found by walking up to
            the repo root.  The error message identifies the skill name and
            the search start path to help diagnose installation issues.
    """
    here = Path(__file__).resolve()
    skill_path: Path | None = None

    for parent in here.parents:
        candidate = parent / 'skills' / skill_name / 'SKILL.md'
        if candidate.exists():
            skill_path = candidate
            break
        # Stop at the repo root to avoid wandering above the monorepo
        if (parent / '.git').exists():
            break

    if skill_path is None:
        msg = (
            f'skills/{skill_name}/SKILL.md not found: searched upward from {here} — '
            f'is the orchestrator installed outside the monorepo?'
        )
        logger.error('skill_prompt: %s', msg)
        raise FileNotFoundError(msg)

    raw = skill_path.read_text()
    if not raw.startswith('---'):
        return raw
    parts = raw.split('---', maxsplit=2)
    return parts[2].strip() if len(parts) >= 3 else raw
