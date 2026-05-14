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

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from escalation.models import Escalation
    from escalation.queue import EscalationQueue

_NON_WORD_PATTERN = re.compile(r'[^\w\s]', flags=re.UNICODE)  # strips punctuation, symbols, control; keeps word chars and whitespace


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
    2. Strip all non-word, non-whitespace characters (Unicode punctuation,
       symbols, controls), including en/em dashes, curly quotes, and ASCII
       punctuation.
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
    normalised = _NON_WORD_PATTERN.sub('', summary.casefold())
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
    - ``summary_dedupe_key(parent.summary) == summary_dedupe_key(candidate.summary)``.
    - ``(now - parsed(parent.timestamp)) <= window_secs``.
    - ``candidate_key != ()`` — empty keys are never matched to prevent
      unrelated empty-summary escalations from collapsing into one parent.

    The ``config.infra_dedupe_enabled`` flag and category membership are
    intentionally NOT checked here — the server callers gate on those before
    calling this function.  This keeps the function's contract simple and
    testable.

    Returns the id of the *oldest* survivor (minimum timestamp) so that
    repeated duplicates always fold into the same canonical first record.

    Performance note: calls ``queue.get_pending()`` which glob-reads the queue
    root on every invocation (O(N) disk reads, N = pending escalations).
    This is acceptable given the bounded window and low submit rate; if this
    path becomes hot, maintain an in-memory (category, key) → [(ts, id)] index
    populated by submit/resolve callbacks instead.
    """
    effective_now = now if now is not None else datetime.now(UTC)
    window = timedelta(seconds=config.infra_dedupe_window_secs)
    candidate_key = summary_dedupe_key(candidate.summary)
    # Empty key means the summary was blank/whitespace — never match to avoid
    # collapsing unrelated escalations that happen to have no useful summary.
    if not candidate_key:
        return None
    candidate_category = candidate.category

    matches: list[tuple[datetime, str]] = []  # (timestamp, id) for sorting

    for parent in queue.get_pending():
        # Category filter — caller already verified candidate_category is in
        # infra_dedupe_categories, so checking equality is sufficient.
        if parent.category != candidate_category:
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
