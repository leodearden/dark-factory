"""Deduplication helpers for infra_issue escalations.

Provides:
- DedupeConfig  — configuration knobs (defaults: enabled, 600s window,
                  infra_issue category only).
- summary_dedupe_key() — pure function; normalises a summary string and
                         returns the first ≤3 tokens as a tuple.
- find_dedupe_parent() — scans the live queue and returns the oldest
                         pending parent id whose key matches the candidate,
                         or None.
- compute_content_fingerprint() — deterministic sha256-based fingerprint
                                   keyed on finding identity for recon dedup.

Design contracts (see plan.json design_decisions for rationale):
- find_dedupe_parent() does NOT check DedupeConfig.infra_dedupe_enabled.
  That gate is the caller's responsibility (server._submit_or_dedupe).
  This keeps the function pure/testable and avoids action-at-a-distance.
- Cross-task: get_pending() scans all tasks, so infra fan-out (same
  summary from 30 task_ids simultaneously) collapses into a single parent.
"""

from __future__ import annotations

__all__ = ['DedupeConfig', 'compute_content_fingerprint', 'find_dedupe_parent', 'summary_dedupe_key']

import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from escalation.models import Escalation
    from escalation.queue import EscalationQueue

_NON_WORD_PATTERN = re.compile(r'[^\w\s]', flags=re.UNICODE)  # strips punctuation, symbols, control; keeps word chars and whitespace
_WHITESPACE_PATTERN = re.compile(r'\s+')  # collapse runs of whitespace

# Unit separator: never appears in category names or entity IDs, so the join is
# collision-free without hashing the readable prefix components.
_FIELD_SEP = '\x1f'


def _normalize_description(text: str) -> str:
    """Normalise a description string for the empty-affected_ids tiebreak.

    Steps:
    1. Casefold (Unicode-aware lower-case).
    2. Strip all non-word, non-whitespace characters (reuses _NON_WORD_PATTERN).
    3. Collapse multiple whitespace runs to a single space and strip edges.

    Reuses _NON_WORD_PATTERN so the normalisation is consistent with
    summary_dedupe_key's stripping stage.
    """
    casefolded = text.casefold()
    stripped = _NON_WORD_PATTERN.sub('', casefolded)
    return _WHITESPACE_PATTERN.sub(' ', stripped).strip()


def compute_content_fingerprint(
    escalation_category: str,
    finding_category: str,
    affected_ids: list[str],
    description: str = '',
) -> str:
    """Return a deterministic sha256 fingerprint keyed on finding identity.

    Identity composition:
    - ``escalation_category``, ``finding_category``, and a *body* joined by
      the unit separator ``\\x1f`` (collision-free since the separator never
      appears in category names or entity IDs).
    - When ``affected_ids`` is NON-EMPTY: body = sorted(affected_ids) joined
      by ``\\x1f``.  The ``description`` is intentionally ignored so that
      recurring findings on the same targets fold even as their prose drifts
      cycle to cycle.
    - When ``affected_ids`` is EMPTY: body = ``'desc:'`` + first 16 hex chars
      of ``sha256(normalised_description)`` so that description-only findings
      with identical normalised text still fold, while genuinely distinct
      descriptions do not.

    Uses ``hashlib.sha256`` (NOT Python's builtin ``hash()``) so the result is
    deterministic across processes regardless of ``PYTHONHASHSEED``.

    Returns the full 64-character hex digest of the sha256 of the composed
    identity string encoded as UTF-8.
    """
    if affected_ids:
        body = _FIELD_SEP.join(sorted(affected_ids))
    else:
        norm = _normalize_description(description)
        desc_hash = hashlib.sha256(norm.encode()).hexdigest()[:16]
        body = 'desc:' + desc_hash

    identity = _FIELD_SEP.join([escalation_category, finding_category, body])
    return hashlib.sha256(identity.encode()).hexdigest()


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
       punctuation.  Note: underscore (U+005F, category Pc) is part of
       ``\\w`` and is therefore *preserved* — ``fused_memory`` stays
       ``fused_memory``, not ``fusedmemory``.  This is a deliberate
       divergence from the previous translate-table implementation, which
       stripped all Unicode Pc (connector punctuation) characters.  In
       practice escalation summaries do not use underscores, so the
       divergence is harmless.

       Symbols (Unicode categories Sm/Sc/Sk/So such as ``+``, ``=``,
       ``$``) are also stripped, which can merge adjacent tokens (e.g.
       ``cpu+memory`` → ``cpumemory``).
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
        >>> summary_dedupe_key("cpu+memory leak")
        ('cpumemory', 'leak')
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
    return min(matches, key=lambda pair: pair[0])[1]
