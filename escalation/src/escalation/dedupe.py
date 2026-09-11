"""Deduplication helpers for infra_issue escalations.

Provides:
- DedupeConfig  — configuration knobs (defaults: enabled, 600s window,
                  infra_issue category only).  Named constructors for the
                  non-infra paths: DedupeConfig.for_recon() (recon integrity
                  findings, key_fn=content_fingerprint_key) and
                  DedupeConfig.for_gate_backlog() (stale gate backlog,
                  key_fn=gate_backlog_fingerprint_key — a superset that also
                  recovers pre-stamp parents); both use an unbounded window.
- summary_dedupe_key() — pure function; normalises a summary string and
                         returns the first ≤3 tokens as a tuple.
- find_dedupe_parent() — scans the live queue and returns the oldest
                         pending parent id whose key matches the candidate,
                         or None.
- compute_content_fingerprint() — deterministic sha256-based fingerprint
                                   keyed on finding identity for recon dedup.
- content_fingerprint_key() / gate_backlog_fingerprint_key() — key adapters.
                  The latter additionally recomputes the identity of a
                  gate-backlog record filed before the fingerprint stamp
                  existed, so the live backlog migrates itself in place.

Design contracts (see plan.json design_decisions for rationale):
- find_dedupe_parent() does NOT check DedupeConfig.infra_dedupe_enabled.
  That gate is the caller's responsibility (server._submit_or_dedupe).
  This keeps the function pure/testable and avoids action-at-a-distance.
- Cross-task: get_pending() scans all tasks, so infra fan-out (same
  summary from 30 task_ids simultaneously) collapses into a single parent.
- Cross-LEVEL folding never happens (task 3236): find_dedupe_parent
  requires parent.level == candidate.level.  The ladder levels have
  different consumers by contract (L0 → steward, L1 →
  escalation-watcher-auto, L2 → human), so folding across them would
  hand the record to the wrong consumer.  Note this is orthogonal to the
  born-at-L2 dedupe bypass in server._submit_or_dedupe, which is keyed on
  SEVERITY: a level-1 filing carries severity='blocking' and therefore
  does route through this matcher.
"""

from __future__ import annotations

__all__ = [
    'DedupeConfig',
    'compute_content_fingerprint',
    'content_fingerprint_key',
    'find_dedupe_parent',
    'gate_backlog_fingerprint_key',
    'submit_or_dedupe',
    'summary_dedupe_key',
]

import hashlib
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from shared.timestamps import parse_timestamp_or_warn

# The casefold/non-word/whitespace transform is NOT owned here — it lives once
# in the leaf module escalation.canonical, which queue.py also imports for
# root-cause matching.  Both call sites in this module pin punctuation='strip'
# (the legacy, deletion-flavoured policy); see _normalize_description.
from escalation.canonical import canonical_text

# observed_submit_response lives next to the record it re-reads (queue), not
# here — this module owns fold logic only.  Re-exported by neither module's
# ``__all__``; server imports it from ``escalation.queue`` directly.
from escalation.queue import observed_submit_response

if TYPE_CHECKING:
    from escalation.models import Escalation
    from escalation.queue import EscalationQueue

# Type alias for injectable key functions.  A key function maps an Escalation
# to a hashable value used for matching.  None return (e.g. unset fingerprint)
# is treated as "never fold" by the empty-key guard in find_dedupe_parent.
KeyFn = Callable[['Escalation'], Any]

logger = logging.getLogger(__name__)

# Unit separator: never appears in category names or entity IDs, so the join is
# collision-free without hashing the readable prefix components.
_FIELD_SEP = '\x1f'


def _normalize_description(text: str) -> str:
    """Normalise a description string for the empty-affected_ids tiebreak.

    Delegates to the single implementation in :mod:`escalation.canonical`
    (casefold -> map non-word chars -> collapse whitespace -> strip).

    THE ``strip`` POLICY IS PINNED HERE DELIBERATELY and must NOT be changed to
    the ``separator`` policy the root-cause match site uses.  This function's
    only consumer is :func:`compute_content_fingerprint`, whose sha256 digests
    are already persisted across the live recon corpus: under separator
    semantics every digest changes, so every already-fingerprinted recon finding
    stops matching its own past self and the whole corpus silently un-dedupes.
    ``tests/test_dedupe.py::TestNormalisationLiftedToCanonical`` pins four
    reference digests against exactly that drift.
    """
    return canonical_text(text, punctuation='strip')


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


# The one recovery site for a legacy gate-backlog record's project_id: the
# emitter (stage1_stall_detector.maybe_escalate_stalled_gate_backlog) writes
# ``detail_parts[0] = f'project_id: {project_id}'``.  Kept as a module constant
# so the coupling to that emitter is named rather than inlined.
_GATE_BACKLOG_DETAIL_PROJECT_PREFIX = 'project_id: '


def gate_backlog_fingerprint_key(esc: Escalation) -> str | None:
    """Key adapter for gate-backlog dedup that tolerates LEGACY unstamped parents.

    A superset of ``content_fingerprint_key``: stamped records take the same
    fast path, and only unstamped ones pay for prose recovery.

    Why this exists — it is a self-executing, in-place migration.  Every
    ``reconciliation_stale_gate_backlog`` record filed before the fingerprint
    stamp landed carries ``dedupe_fingerprint: None`` (78 such records measured
    on the live queue, i.e. 100% of the pending backlog).  Under the plain
    stamped-only adapter each of those keys falsy, ``find_dedupe_parent``
    short-circuits, and the first post-change cycle mints a DUPLICATE record at
    ``dedupe_count 0`` for every stalled gate.  Worse, the legacy record's key
    stays falsy forever, so it never becomes a fold target again.  Recovering
    the record's true identity from its own ``detail`` fixes the whole backlog
    with no operator step, and also covers records filed in the window between
    the stamp landing and any backfill being run.

    Keyed on ``(category, project_id, task_id)`` and deliberately NOT on
    ``(category, task_id)``: the escalation queue is SHARED ACROSS PROJECTS
    (7 observed live — dark_factory 37, reify 27, autopilot_video 5,
    know_live 3, solar_challenge_platform 2, pump_web_ui 2, solar_challenge 2)
    and task ids are small per-project integers.  A task_id-only fallback would
    cross-fold two different projects' gates into a single record and silently
    discard an escalation a human is waiting on.  Today's snapshot happens to
    hold 78 distinct task_ids, but that is a property of the current backlog,
    not an invariant.

    ``project_id`` is NOT a persisted field on ``Escalation`` — it appears
    nowhere in the on-disk record's key set — so ``detail``'s first line is the
    only recovery site.  That couples this helper to the emitter's format: a
    change to ``detail_parts[0]`` in ``stage1_stall_detector`` must update this
    parser.  Because the parse fails CLOSED, the blast radius of such a drift is
    duplicate records (visible, self-correcting once the new stamped record
    becomes the parent), never wrong folds (which silently destroy an
    escalation).  The same asymmetry drives taking the line remainder VERBATIM
    rather than via ``\\S+``: truncating ``my project`` to ``my`` would turn a
    parse ambiguity into a different, possibly colliding key.

    The literal token ``None`` is deliberately not special-cased: the emitter
    writes ``f'project_id: {project_id}'`` and stamps children as
    ``f'{project_id}:{task_id}'``, so a filing made with ``project_id=None``
    yields ``'None:645'`` on BOTH sides.  Reproducing ``str(project_id)``
    byte-for-byte is exactly what makes parent and child keys agree.

    ACCEPTED COST — a legacy parent kept alive here keeps its STALE SUMMARY.
    ``attach_dedupe_child`` increments ``dedupe_count`` but never rewrites
    ``summary``/``detail``, and the records this adapter rescues were filed
    before task 3520 replaced the relative-age phrasing (``'Gate task 166 has
    awaited a human decision for 48.7h'``) with an absolute ``since <ISO>``
    anchor.  A compact drain projects ``summary`` and drops ``detail``
    (``_COMPACT_ESCALATION_FIELDS``, escalation/server.py), so a steward
    triaging a gate that has now been open 400h still reads ``48.7h``, with
    only ``dedupe_count`` hinting that it recurred.  This is NOT a regression
    introduced by folding — under the ``has_open_l1`` skip this adapter
    replaced, the same record was equally permanent and equally stale, since
    a fresher filing was suppressed outright rather than merged.  The
    alternative (let each legacy record be superseded once by a correctly
    anchored one) trades a permanent duplicate-shaped blip for a truthful
    summary; it was rejected only because it re-pins ``dedupe_count`` at 0 for
    every gate in the backlog, which is the defect this task exists to remove.
    Re-anchoring a legacy parent in place means mutating live production
    escalation records from inside a key-resolution helper, so it is
    deliberately left to a separately reviewable operator action (see
    plan.json design_decisions for task 3522).

    Returns the stamped fingerprint, the recomputed one, or ``None`` (never
    fold) when the record's identity cannot be recovered.
    """
    # Fast path — post-stamp records never touch the prose.
    if esc.dedupe_fingerprint:
        return esc.dedupe_fingerprint

    task_id = esc.task_id
    detail = esc.detail or ''
    first_line = detail.split('\n', 1)[0]
    if not first_line.startswith(_GATE_BACKLOG_DETAIL_PROJECT_PREFIX):
        return None
    # Verbatim remainder (only \r from a CRLF detail is dropped, which is not
    # part of any project id) — see the docstring on why not \S+.
    project_id = first_line[len(_GATE_BACKLOG_DETAIL_PROJECT_PREFIX):].rstrip('\r')

    # Fail CLOSED: a falsy key hits find_dedupe_parent's short-circuit, so an
    # unrecoverable parent simply does not fold and we mint a fresh, correctly
    # stamped record instead of guessing an identity.  An EMPTY project_id
    # (a bare ``project_id: `` line) is rejected here too, not mirrored: it far
    # more likely means a truncated detail than a genuine ``project_id=''``
    # filing, and a ``':166'`` key would collide with every other malformed
    # record for task 166 across all projects — a wrong fold, the one
    # unrecoverable outcome.
    if not task_id or not project_id:
        return None

    # Identical construction to the child's stamp in stage1_stall_detector, so
    # parent and child keys agree by construction rather than by a second,
    # drift-prone formula.
    #
    # ``esc.category`` rather than the literal category string: every record
    # reaching this line is already known to carry it — submit_or_dedupe gates
    # the candidate on ``config.infra_dedupe_categories`` and find_dedupe_parent
    # skips any parent whose category differs — so reading it off the record is
    # correct by construction, whereas a repeated literal would silently stop
    # matching stage1's _GATE_BACKLOG_ESCALATION_CATEGORY if that constant were
    # ever renamed (folding would fail open into duplicates, with no test that
    # catches it).
    return compute_content_fingerprint(
        esc.category, '', [f'{project_id}:{task_id}'], ''
    )


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
    general-purpose.  Use ``DedupeConfig.for_recon()`` for the recon
    integrity path and ``DedupeConfig.for_gate_backlog()`` for the
    stale-gate-backlog path — deliberate siblings, not one widened config
    (see ``for_gate_backlog``'s docstring for why).

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

    @classmethod
    def for_gate_backlog(cls) -> DedupeConfig:
        """Return a DedupeConfig configured for stale-gate-backlog dedup.

        Used by ``fused_memory.reconciliation.stage1_stall_detector.
        maybe_escalate_stalled_gate_backlog``, which re-files an L1 every
        Stage-1 cycle a gate task stays past its human-decision threshold.
        Folding those into one parent is what makes ``dedupe_count`` a
        recurrence/triage-order signal instead of a constant 0.

        Properties:
        - ``infra_dedupe_enabled``     : True
        - ``infra_dedupe_window_secs`` : float('inf') — UNBOUNDED, and this is
          load-bearing: a gate that has sat open 300h must still fold into its
          original parent.  Any bounded window silently mints a second pending
          record for the same gate and re-pins ``dedupe_count`` at 0, which is
          exactly the defect this config exists to prevent.
        - ``infra_dedupe_categories``  : ('reconciliation_stale_gate_backlog',)
        - ``key_fn``                   : gate_backlog_fingerprint_key — folds on
          ``esc.dedupe_fingerprint`` when present, and otherwise recovers a
          LEGACY parent's identity from its ``detail``.  The requirement to
          stamp therefore binds the CANDIDATE, not the parent: the caller
          treats an empty fingerprint as a hard error rather than filing
          (``find_dedupe_parent`` short-circuits to None on a falsy key, so an
          unstamped candidate would silently never fold), while an unstamped
          PARENT filed before the stamp landed is still a valid fold target.

        Why a SIBLING of ``for_recon()`` rather than widening it: the tuple
        returned by ``for_recon().infra_dedupe_categories`` is consumed as the
        eligible-collapse set by ``fused-memory/scripts/
        backfill_recon_escalations.py`` (:168) and as the complement defining
        that script's ``blocking_pending`` report field (:290).  Widening
        ``for_recon`` would silently admit live gate-backlog records into that
        one-shot operator script's collapse plan and change its report's
        meaning, and ``for_recon``'s own docstring documents its exclusions as
        deliberate.

        The ``infra_dedupe_*`` prefix is historical / general-purpose.
        """
        return cls(
            infra_dedupe_enabled=True,
            infra_dedupe_window_secs=float('inf'),
            infra_dedupe_categories=('reconciliation_stale_gate_backlog',),
            key_fn=gate_backlog_fingerprint_key,
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
    # Same single implementation as _normalize_description, and the ``strip``
    # policy is pinned here for the same reason: these tuples are already
    # persisted fleet-wide as find_dedupe_parent's key, so a change to the
    # transform silently re-partitions every existing dedupe cluster.  The
    # helper's extra whitespace-collapse-and-strip (which the old inline
    # expression did not do) is absorbed by the .split() below — verified
    # byte-identical on the five examples above and on all 2796 real summaries
    # in the live queue.
    normalised = canonical_text(summary, punctuation='strip')
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
    - ``parent.level == candidate.level`` — dedupe never folds across ladder
      levels, because the levels have different consumers by contract (L0 →
      steward, L1 → escalation-watcher-auto, L2 → human), so folding an L1
      into an L0 parent would hand the record to the wrong consumer.
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
        # Level filter (task 3236) — dedupe NEVER folds across ladder levels.
        # Placed alongside the category filter so it short-circuits before the
        # key computation and the timestamp parse.  The levels have different
        # consumers by contract (models.py module header: L0 → steward,
        # L1 → escalation-watcher-auto, L2 → human), so folding an L1 into an
        # L0 parent hands the record to the wrong consumer.  Concretely: a
        # steward re-escalating an infra_issue at level=1 would otherwise fold
        # straight back into the pending level-0 record it was handling.
        if parent.level != candidate.level:
            continue
        # Key filter using the resolved key function.
        if _key_fn(parent) != candidate_key:
            continue
        # Parse timestamp for age filter and oldest-match selection.
        # fallback=datetime.max: corrupt-ts parent is retained (not dropped) and
        # sorts LAST so it never displaces a valid older match. With datetime.min,
        # effective_now - datetime.min >> 600s window → parent re-dropped (same bug).
        # NOTE — asymmetry: because effective_now - datetime.max is always negative
        # (< 0 < window), a corrupt-ts parent is NEVER aged out by the window filter.
        # Valid parents expire after infra_dedupe_window_secs; corrupt ones don't.
        # This is intentional (a malformed timestamp is a bug worth investigating,
        # not silently discarding), but means a corrupt parent persists as a fold
        # target until it is explicitly resolved or the queue is cleared.
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

    Response shapes (identical to server._submit_or_dedupe).  ``level`` is on
    EVERY branch, so the documented "echo confirms the level landed" contract
    holds no matter which one a caller lands on:
    - Queued:        ``{'id': esc_id, 'status': 'queued', 'level': <persisted>}``
    - Auto-resolved/dismissed (the record was NOT pending after the write —
      e.g. a concurrent sweep won the race): ``{'id', 'status', 'resolution',
      'resolved_by', 'level'}``.  See ``queue.observed_submit_response``: the
      response reports observed post-write state, never write intent, and
      fails open to ``'queued'`` (still carrying ``level``) if the re-read is
      unavailable.
    - Dedup-skipped: ``{'id': parent_id, 'status': 'dedup_skipped',
                        'parent_id': parent_id, 'child_id': esc.id,
                        'level': esc.level}``
      (the id/parent/child keys are deliberately unchanged — they are already
      explicit that the filing folded into another record; ``level`` is
      additive.  It is the CHILD's level, which since task 3236 equals the
      parent's: ``find_dedupe_parent`` requires ``parent.level ==
      candidate.level``, so no extra read is needed to report it.)

    Recon (A7b) calls this directly with ``DedupeConfig.for_recon()`` instead
    of ``queue.submit()``, routing through the same gate + TOCTOU logic used
    by the infra path.

    *now* is forwarded to ``find_dedupe_parent`` for deterministic testing.
    """
    # Gate 1 (enabled) and Gate 2 (category membership) both short-circuit
    # in pure memory before any disk I/O via find_dedupe_parent.
    # NOTE: the post-write response is shaped by observed_submit_response, not
    # by a hardcoded 'queued' — see that function's docstring.
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
                # Level-scoped folding (task 3236) means the parent's level is
                # the child's, so echoing esc.level costs no extra read and
                # keeps the 'level' key present on every response branch.
                'level': esc.level,
            }
    esc_id = queue.submit(esc)
    return observed_submit_response(queue, esc_id, fallback_level=esc.level)

