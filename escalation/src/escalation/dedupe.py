"""Deduplication helpers for infra_issue escalations.

Provides:
- DedupeConfig  — configuration knobs (defaults: enabled, 600s window,
                  infra_issue category only).
- summary_dedupe_key() — pure function; normalises a summary string and
                         returns the first ≤3 tokens as a tuple.
- find_dedupe_parent() — scans the live queue and returns the oldest
                         pending parent id whose key matches the candidate,
                         or None.

Design contracts (see plan.json design_decisions for rationale):
- find_dedupe_parent() does NOT check DedupeConfig.infra_dedupe_enabled.
  That gate is the caller's responsibility (server._submit_or_dedupe).
  This keeps the function pure/testable and avoids action-at-a-distance.
- Cross-task: get_pending() scans all tasks, so infra fan-out (same
  summary from 30 task_ids simultaneously) collapses into a single parent.
"""

from __future__ import annotations

__all__ = ['DedupeConfig', 'find_dedupe_parent', 'summary_dedupe_key']

import string
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from escalation.models import Escalation
    from escalation.queue import EscalationQueue

_PUNCT_TABLE = str.maketrans('', '', string.punctuation)


@dataclass
class DedupeConfig:
    """Configuration knobs for infra_issue deduplication.

    Defaults represent the recommended AFK-hardening settings:
    - enabled         : True  — dedupe is on by default.
    - window_secs     : 600.0 — 10-minute look-back window.
    - categories      : ('infra_issue',) — only fold infra noise.
    """

    infra_dedupe_enabled: bool = True
    infra_dedupe_window_secs: float = 600.0
    infra_dedupe_categories: tuple[str, ...] = ('infra_issue',)


def summary_dedupe_key(summary: str) -> tuple[str, ...]:
    """Return a normalised prefix key for *summary*.

    Normalisation steps:
    1. Casefold (Unicode-aware lower-case).
    2. Strip all ASCII punctuation (``string.punctuation``).
    3. Split on whitespace (collapses multiple spaces / tabs).
    4. Return the first three tokens as a tuple (fewer if the summary
       has fewer than three words).

    Examples::

        >>> summary_dedupe_key("Fused-memory  CONNECTION timeout!")
        ('fusedmemory', 'connection', 'timeout')
        >>> summary_dedupe_key("fused-memory connection timeout on port 8002")
        ('fusedmemory', 'connection', 'timeout')
        >>> summary_dedupe_key("lost link")
        ('lost', 'link')
        >>> summary_dedupe_key("")
        ()
    """
    normalised = summary.casefold().translate(_PUNCT_TABLE)
    tokens = normalised.split()
    return tuple(tokens[:3])


def find_dedupe_parent(
    queue: EscalationQueue,
    candidate: Escalation,
    config: DedupeConfig,
    now: datetime | None = None,
) -> str | None:
    """Return the id of the oldest matching pending parent, or None.

    A parent matches when ALL of the following hold:
    - ``parent.status == 'pending'`` (get_pending() already ensures this).
    - ``parent.category == candidate.category``.
    - ``parent.category in config.infra_dedupe_categories``.
    - ``summary_dedupe_key(parent.summary) == summary_dedupe_key(candidate.summary)``.
    - ``(now - parsed(parent.timestamp)) <= window_secs``.

    The ``config.infra_dedupe_enabled`` flag is intentionally NOT checked
    here — the server callers gate on that flag before ever calling this
    function.  This keeps the function's contract simple and testable.

    Returns the id of the *oldest* survivor (minimum timestamp) so that
    repeated duplicates always fold into the same canonical first record.
    """
    effective_now = now if now is not None else datetime.now(UTC)
    window = timedelta(seconds=config.infra_dedupe_window_secs)
    candidate_key = summary_dedupe_key(candidate.summary)
    candidate_category = candidate.category

    matches: list[tuple[datetime, str]] = []  # (timestamp, id) for sorting

    for parent in queue.get_pending():
        # Category filter
        if parent.category != candidate_category:
            continue
        if parent.category not in config.infra_dedupe_categories:
            continue
        # Key filter
        if summary_dedupe_key(parent.summary) != candidate_key:
            continue
        # Time-window filter
        try:
            parent_ts = datetime.fromisoformat(parent.timestamp)
        except (ValueError, AttributeError):
            continue
        # Ensure timezone-aware for comparison
        if parent_ts.tzinfo is None:
            parent_ts = parent_ts.replace(tzinfo=UTC)
        if effective_now - parent_ts > window:
            continue
        matches.append((parent_ts, parent.id))

    if not matches:
        return None

    # Return the oldest by timestamp (first match we should fold into)
    matches.sort(key=lambda pair: pair[0])
    return matches[0][1]
