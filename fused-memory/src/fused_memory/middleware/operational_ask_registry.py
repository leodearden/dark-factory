"""Operational-ask registry for the task curator filing-policy gate.

Provides a pure, deterministic guard that the curator consults *before* any
LLM call. Operators add an entry per recurring operational live-data/
live-mutation ask — one whose deliverable is an already-built+unit-tested
script and whose remaining work is dry-run -> human-review -> apply against
live external stores, modifying ZERO repo files. The TDD architect cannot
author a RED->GREEN plan for that shape of work, so a match here routes the
candidate straight to a deterministic PURE-GATE (born-at-L2 milestone gate)
instead of letting it reach the architect and bounce as "unactionable".

Usage::

    from fused_memory.middleware.operational_ask_registry import (
        OperationalAskEntry,
        load_operational_registry,
        match_candidate,
    )

    entries = load_operational_registry(Path("config/operational_ask_registry.yaml"))
    hit = match_candidate(candidate, entries)
    if hit is not None:
        # return CuratorDecision(action='route_deterministic', justification=...)
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
class OperationalAskEntry:
    """One entry in the operational-ask registry.

    Matching logic (evaluated case-insensitively):
    - ALL strings in ``title_substrings`` must appear in the candidate title.
    - AT LEAST ONE string in ``description_substrings`` must appear in the
      combined candidate description + details text.
    """

    name: str
    reason: str
    title_substrings: list[str]
    description_substrings: list[str]


def load_operational_registry(path: Path | None) -> list[OperationalAskEntry]:
    """Load the operational-ask registry from a YAML file.

    Returns an empty list (without warning) when *path* is ``None``.
    Returns an empty list and emits one WARNING when the file is missing,
    unreadable, not valid YAML, or its top-level document is not a list.
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
            "operational_ask_registry: file not found: %s — registry disabled", path
        )
        return []
    except OSError as exc:
        logger.warning(
            "operational_ask_registry: cannot read %s: %s — registry disabled",
            path, exc,
        )
        return []

    # Parse
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        logger.warning(
            "operational_ask_registry: YAML parse error in %s: %s — registry disabled",
            path, exc,
        )
        return []

    if not isinstance(data, list):
        logger.warning(
            "operational_ask_registry: expected a YAML list in %s, got %s — registry disabled",
            path, type(data).__name__,
        )
        return []

    entries: list[OperationalAskEntry] = []
    for item in data:
        if not isinstance(item, dict):
            logger.warning(
                "operational_ask_registry: skipping non-dict entry in %s: %r", path, item
            )
            continue

        # Validate required fields
        missing = [
            f for f in ("name", "reason", "title_substrings", "description_substrings")
            if f not in item
        ]
        if missing:
            logger.warning(
                "operational_ask_registry: skipping entry missing fields %s in %s: %r",
                missing, path, item.get("name", "<unnamed>"),
            )
            continue

        title_subs = item["title_substrings"]
        desc_subs = item["description_substrings"]
        if not isinstance(title_subs, list) or not isinstance(desc_subs, list):
            logger.warning(
                "operational_ask_registry: skipping entry %r — title_substrings and "
                "description_substrings must be lists",
                item.get("name", "<unnamed>"),
            )
            continue

        entries.append(
            OperationalAskEntry(
                name=str(item["name"]),
                reason=str(item["reason"]),
                title_substrings=[str(s) for s in title_subs],
                description_substrings=[str(s) for s in desc_subs],
            )
        )

    return entries


def match_candidate(
    candidate: CandidateTask,
    entries: list[OperationalAskEntry],
) -> OperationalAskEntry | None:
    """Return the first matching registry entry for *candidate*, or ``None``.

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
