"""Read a markup journal back the way an operator would: one JSON object per line.

The one reader for suites that assert on
``orchestrator/src/orchestrator/mcp/markup_journal.py``'s files, so a change to
where a journal lives or how a line is encoded is one edit here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from orchestrator.mcp import markup_journal


def journal_lines(root: Path, server_label: str) -> list[dict[str, Any]]:
    """Every line of *server_label*'s journal under *root*; none if it was never written."""
    path = markup_journal.journal_path(root, server_label)
    if not path.exists():
        return []
    text = path.read_text(encoding='utf-8')
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def lines_of(root: Path, server_label: str, fact: str) -> list[dict[str, Any]]:
    """The lines of *server_label*'s journal whose ``fact`` is *fact*."""
    return [line for line in journal_lines(root, server_label) if line.get('fact') == fact]
