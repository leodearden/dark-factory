"""Cancelled-premise blocklist for the task curator gate.

Provides a pure, deterministic guard that the curator consults *before* any
LLM call. Operators add an entry per revert to record premises that were
tried, shipped, and then reverted — ensuring the same hallucinated task cannot
recur without an explicit human override.

Usage::

    from fused_memory.middleware.cancelled_premise_blocklist import (
        BlocklistEntry,
        load_blocklist,
        match_candidate,
    )

    entries = load_blocklist(Path("config/cancelled_premise_blocklist.yaml"))
    hit = match_candidate(candidate, entries)
    if hit is not None:
        # return CuratorDecision(action='drop', justification=...)
        ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from fused_memory.middleware.task_curator import CandidateTask

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BlocklistEntry:
    """One entry in the cancelled-premise blocklist.

    Matching logic (evaluated case-insensitively):
    - ALL strings in ``title_substrings`` must appear in the candidate title.
    - AT LEAST ONE string in ``description_substrings`` must appear in the
      combined candidate description + details text.
    """

    name: str
    reason: str
    title_substrings: list[str]
    description_substrings: list[str]


def load_blocklist(path: Path | None) -> list[BlocklistEntry]:
    """Load the cancelled-premise blocklist from a YAML file.

    Returns an empty list (without warning) when *path* is ``None``.
    Returns an empty list and emits one WARNING when the file is missing,
    unreadable, or not valid YAML.
    Skips malformed individual entries with one WARNING each while returning
    the well-formed entries from the same file.

    The function never raises — all failures degrade gracefully to [].
    """
    if path is None:
        return []

    # Missing-file / unreadable
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning(
            "cancelled_premise_blocklist: file not found: %s — blocklist disabled", path
        )
        return []
    except OSError as exc:
        logger.warning(
            "cancelled_premise_blocklist: cannot read %s: %s — blocklist disabled",
            path, exc,
        )
        return []

    # Parse
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        logger.warning(
            "cancelled_premise_blocklist: YAML parse error in %s: %s — blocklist disabled",
            path, exc,
        )
        return []

    if not isinstance(data, list):
        logger.warning(
            "cancelled_premise_blocklist: expected a YAML list in %s, got %s — blocklist disabled",
            path, type(data).__name__,
        )
        return []

    entries: list[BlocklistEntry] = []
    for item in data:
        if not isinstance(item, dict):
            logger.warning(
                "cancelled_premise_blocklist: skipping non-dict entry in %s: %r", path, item
            )
            continue

        # Validate required fields
        missing = [
            f for f in ("name", "reason", "title_substrings", "description_substrings")
            if f not in item
        ]
        if missing:
            logger.warning(
                "cancelled_premise_blocklist: skipping entry missing fields %s in %s: %r",
                missing, path, item.get("name", "<unnamed>"),
            )
            continue

        title_subs = item["title_substrings"]
        desc_subs = item["description_substrings"]
        if not isinstance(title_subs, list) or not isinstance(desc_subs, list):
            logger.warning(
                "cancelled_premise_blocklist: skipping entry %r — title_substrings and "
                "description_substrings must be lists",
                item.get("name", "<unnamed>"),
            )
            continue

        entries.append(
            BlocklistEntry(
                name=str(item["name"]),
                reason=str(item["reason"]),
                title_substrings=[str(s) for s in title_subs],
                description_substrings=[str(s) for s in desc_subs],
            )
        )

    return entries


def match_candidate(
    candidate: CandidateTask,
    entries: list[BlocklistEntry],
) -> BlocklistEntry | None:
    """Return the first matching blocklist entry for *candidate*, or ``None``.

    Matching is case-insensitive string search:
    - ALL strings in ``entry.title_substrings`` must appear in
      ``candidate.title``.
    - AT LEAST ONE string in ``entry.description_substrings`` must appear in
      the combined ``candidate.description + " " + candidate.details``.

    Returns the first match in list order (deterministic). Returns ``None``
    when *entries* is empty or no entry matches.

    Pure string operations — no regex, no async, no I/O.
    """
    title_lower = (candidate.title or "").lower()
    desc_lower = (
        (candidate.description or "") + " " + (candidate.details or "")
    ).lower()

    for entry in entries:
        # All title substrings must match
        if not all(sub.lower() in title_lower for sub in entry.title_substrings):
            continue
        # At least one description substring must match
        if not any(sub.lower() in desc_lower for sub in entry.description_substrings):
            continue
        return entry

    return None
