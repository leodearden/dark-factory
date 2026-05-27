"""Worktree identity helpers — guard against task-id recycling collisions.

Standalone module (no orchestrator imports) so both :mod:`git_ops` and the
harness can use it without re-introducing an artifacts↔git_ops import cycle.

A worktree is keyed purely on its numeric task id.  When an id is recycled —
an abandoned task is deleted and its id re-minted onto unrelated work (reify
task 3770) — an orphaned worktree on disk can be adopted for the WRONG task,
because the only structural guard compares ``plan.task_id`` to the directory
name (both equal the recycled id).  These helpers compare the worktree's
STORED title against the live DB task's title so a content mismatch is caught.

Match on **title only** — descriptions get edited/truncated, which would
produce false-positive quarantines.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# Leading bracketed tag groups the orchestrator prepends to titles, e.g.
# "[auto-eval redo] Fix X" or "[redo] [hotfix] Fix X".  Stripped before
# comparison so a benign tag prefix doesn't read as an identity mismatch.
_LEADING_TAGS_RE = re.compile(r'^(?:\s*\[[^\]]*\]\s*)+')
_WS_RE = re.compile(r'\s+')


def normalize_identity(title: str | None) -> str:
    """Normalise a task title for identity comparison.

    Strips leading ``[...]`` tag groups, collapses internal whitespace, and
    casefolds.  Returns ``''`` for a blank/``None`` input.
    """
    if not title:
        return ''
    stripped = _LEADING_TAGS_RE.sub('', title)
    collapsed = _WS_RE.sub(' ', stripped).strip()
    return collapsed.casefold()


def identities_match(stored: str | None, live: str | None) -> bool:
    """Whether a worktree's stored title matches the live DB task's title.

    **Fail-open**: if either side normalises to ``''`` (a legacy/title-less
    worktree, or a task with no readable title) the function returns ``True`` —
    a missing identity cannot disprove a match, and a false-positive quarantine
    would relocate live work.  Returns ``False`` only when BOTH sides are
    present and differ.
    """
    ns = normalize_identity(stored)
    nl = normalize_identity(live)
    if not ns or not nl:
        return True
    return ns == nl


def read_worktree_title(worktree: Path) -> str | None:
    """Read the worktree's stored task title from its ``.task`` sidecar.

    Prefers ``.task/metadata.json`` ``title``, falls back to
    ``.task/plan.json`` ``title``.  Returns ``None`` if neither is readable or
    carries a non-blank title — callers treat ``None`` as "unprovable", which
    :func:`identities_match` fails open on.
    """
    for rel in ('metadata.json', 'plan.json'):
        path = worktree / '.task' / rel
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(data, dict):
            title = data.get('title')
            if isinstance(title, str) and title.strip():
                return title
    return None
