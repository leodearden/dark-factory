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

__all__ = [
    'DedupeConfig',
    'compute_content_fingerprint',
    'content_fingerprint_key',
    'find_dedupe_parent',
    'submit_or_dedupe',
    'summary_dedupe_key',
]

import hashlib
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from shared.timestamps import parse_timestamp_or_warn

if TYPE_CHECKING:
    from escalation.models import Escalation
    from escalation.queue import EscalationQueue

# Type alias for injectable key functions.  A key function maps an Escalation
# to a hashable value used for matching.  None return (e.g. unset fingerprint)
# is treated as "never fold" by the empty-key guard in find_dedupe_parent.
KeyFn = Callable[['Escalation'], Any]

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


def content_fingerprint_key(esc: Escalation) -> str | None:
    """Key adapter for content-fingerprint dedup.

    Returns ``esc.dedupe_fingerprint`` directly.  When the fingerprint is
    ``None`` (unstamped escalation), the falsy-key guard in
    ``find_dedupe_parent`` treats it as "never fold", so this function is safe
    to use even for escalations that were not pre-stamped by A7b.
    """
    return esc.dedupe_fingerprint


def _default_summary_key(esc: Escalation) -> tuple[str, ...]:
    """Default key fn: wraps summary_dedupe_key for the key_fn=None path.

    This wrapper is resolved at the find_dedupe_parent use-site (not stored as
    a dataclass default) to avoid the descriptor-binding gotcha with bare
    function defaults on dataclass fields.  The result is identical to calling
    summary_dedupe_key(esc.summary) directly.
    """
    return summary_dedupe_key(esc.summary)


@dataclass
class DedupeConfig:
    """Configuration knobs for escalation deduplication.

    Defaults represent the recommended AFK-hardening settings:
    - enabled         : True  — dedupe is on by default.
    - window_secs     : 600.0 — 10-minute look-back window.
    - categories      : ('infra_issue',) — only fold infra noise.
    - key_fn          : None  — use summary_dedupe_key (default, infra path).

    The ``infra_dedupe_*`` field names are historical; the config is
    general-purpose.  Use ``DedupeConfig.for_recon()`` for the recon path.

    ``key_fn`` is resolved at the ``find_dedupe_parent`` use-site: None maps
    to ``_default_summary_key`` (wrapping ``summary_dedupe_key``).  Storing
    None rather than the function directly avoids the dataclass
    descriptor-binding gotcha and keeps the default path byte-identical to
    the pre-A7a implementation.
    """

    infra_dedupe_enabled: bool = True
    infra_dedupe_window_secs: float = 600.0
    infra_dedupe_categories: tuple[str, ...] = ('infra_issue',)
    key_fn: KeyFn | None = None  # None => _default_summary_key (summary prefix key)

    @classmethod
    def for_recon(cls) -> DedupeConfig:
        """Return a DedupeConfig configured for recon integrity dedup.

        Properties:
        - ``infra_dedupe_enabled``     : True
        - ``infra_dedupe_window_secs`` : float('inf') — unbounded window so
          recurring findings over hours/days always fold into the same parent.
        - ``infra_dedupe_categories``  : ('recon_integrity_issue',) — only fold
          recon integrity findings; recon_failure / recon_backlog_overflow /
          recon_stale_run are intentionally excluded to preserve distinct
          blocking signals.
        - ``key_fn``                   : content_fingerprint_key — folds on
          esc.dedupe_fingerprint rather than the summary prefix.

        The ``infra_dedupe_*`` prefix is historical / general-purpose.
        """
        return cls(
            infra_dedupe_enabled=True,
            infra_dedupe_window_secs=float('inf'),
            infra_dedupe_categories=('recon_integrity_issue',),
            key_fn=content_fingerprint_key,
        )


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

    Short-circuit: if ``candidate_key`` (the resolved key of the candidate) is
    falsy (None, empty tuple, empty string), the function returns ``None``
    immediately — falsy keys are never matched to prevent unrelated or
    unstamped escalations from collapsing.

    A parent matches when ALL of the following hold:
    - ``parent.status == 'pending'`` (get_pending() already ensures this).
    - ``parent.category == candidate.category``.
    - ``key_fn(parent) == key_fn(candidate)`` where key_fn is resolved from
      ``config.key_fn`` (None → ``_default_summary_key`` wrapping
      ``summary_dedupe_key``).
    - Age filter: ``(now - parsed(parent.timestamp)) <= window_secs``, UNLESS
      ``config.infra_dedupe_window_secs`` is ``float('inf')`` (unbounded) in
      which case the age filter is skipped entirely.  Parent timestamps are
      still parsed for the oldest-selection sort in both modes.

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
    unbounded = math.isinf(config.infra_dedupe_window_secs)
    # Only build the timedelta when the window is finite; timedelta(seconds=inf)
    # raises OverflowError.
    window = None if unbounded else timedelta(seconds=config.infra_dedupe_window_secs)

    # Resolve the key function: None sentinel -> default summary-prefix key.
    _key_fn: KeyFn = config.key_fn if config.key_fn is not None else _default_summary_key

    candidate_key = _key_fn(candidate)
    # Falsy key (None fingerprint, empty tuple, empty string) — never match to
    # avoid collapsing unrelated or unstamped escalations.
    if not candidate_key:
        return None
    candidate_category = candidate.category

    matches: list[tuple[datetime, str]] = []  # (timestamp, id) for sorting

    for parent in queue.get_pending():
        # Category filter — caller already verified candidate_category is in
        # infra_dedupe_categories, so checking equality is sufficient.
        if parent.category != candidate_category:
            continue
        # Key filter using the resolved key function.
        if _key_fn(parent) != candidate_key:
            continue
        # Parse timestamp for age filter and oldest-match selection.
        # fallback=datetime.max: corrupt-ts parent is retained (not dropped) and
        # sorts LAST so it never displaces a valid older match. With datetime.min,
        # effective_now - datetime.min >> 600s window → parent re-dropped (same bug).
        parent_ts, _ = parse_timestamp_or_warn(
            parent.timestamp,
            fallback=datetime.max.replace(tzinfo=UTC),
            context='dedupe.find_dedupe_parent',
        )
        # Time-window filter — skipped entirely when window is unbounded (inf).
        if window is not None and effective_now - parent_ts > window:
            continue
        matches.append((parent_ts, parent.id))

    if not matches:
        return None

    # Return the oldest by timestamp (first match we should fold into)
    return min(matches, key=lambda pair: pair[0])[1]


def submit_or_dedupe(
    queue: EscalationQueue,
    esc: Escalation,
    config: DedupeConfig,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Submit *esc* to *queue* or fold it into an existing pending parent.

    This is the central gated orchestration wrapper that centralises:
    - Gate 1: ``config.infra_dedupe_enabled``
    - Gate 2: ``esc.category in config.infra_dedupe_categories``
    - Parent lookup via ``find_dedupe_parent``
    - TOCTOU guard: ``attach_dedupe_child`` returns ``None`` when the parent
      was resolved between the find scan and the attach call; in that case
      fall through to ``queue.submit()`` so the escalation is never dropped.

    Response shapes (identical to server._submit_or_dedupe):
    - Queued:        ``{'id': esc_id, 'status': 'queued'}``
    - Dedup-skipped: ``{'id': parent_id, 'status': 'dedup_skipped',
                        'parent_id': parent_id, 'child_id': esc.id}``

    Recon (A7b) calls this directly with ``DedupeConfig.for_recon()`` instead
    of ``queue.submit()``, routing through the same gate + TOCTOU logic used
    by the infra path.

    *now* is forwarded to ``find_dedupe_parent`` for deterministic testing.
    """
    # Gate 1 (enabled) and Gate 2 (category membership) both short-circuit
    # in pure memory before any disk I/O via find_dedupe_parent.
    if config.infra_dedupe_enabled and esc.category in config.infra_dedupe_categories:
        parent_id = find_dedupe_parent(queue, esc, config, now=now)
        # TOCTOU guard: attach_dedupe_child returns None when the parent was
        # resolved/archived between the find scan and this call.  Fall through
        # to submit() so the escalation is not silently dropped.
        if parent_id is not None and queue.attach_dedupe_child(parent_id, esc.id, child_severity=esc.severity) is not None:
            return {
                'id': parent_id,
                'status': 'dedup_skipped',
                'parent_id': parent_id,
                'child_id': esc.id,
            }
    esc_id = queue.submit(esc)
    return {'id': esc_id, 'status': 'queued'}
