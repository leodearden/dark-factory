"""Write-time containment tripwire for leaked MCP envelope markup (task 3141).

A recurring harness serialization bug leaks raw MCP envelope fragments — the
closing/opening tags of the tool-call wire format — into the *payload* of
fused-memory writes. Two observed vectors: memory ``content`` arriving with a
``</content>``/``</invoke>`` tail (permanent specimens now sitting in the mem0
and Graphiti corpora), and task text arriving with a ``<parameter name=``
fragment that the interceptor's description parser then mis-parses *silently*
(reify task 3210 was filed ``priority=high`` and stored as ``medium``).

This module is the CONTAINMENT half of that story (PRD
``docs/prds/memory-write-path-convergence.md`` §9 leaf ο, contract C3): it
REJECTS such a write at the boundary, loudly and with the matched pattern
named, so no further specimen enters the corpus and no parser gets the chance
to derive a wrong value from the fragment. Root cause, the Qdrant payload
text-match read tool, and the retroactive corpus sweep belong to **DF task
3083** and are deliberately out of scope here.

:data:`MCP_MARKUP_PATTERNS` is the SINGLE write-time pattern list (INV-5) —
every one of the four MCP write boundaries (``add_memory``, ``add_episode``,
``submit_task``, ``update_task``) rejects against exactly this tuple, and
nothing else in the package enumerates these literals.

Calibration vs. the retrospective scanner
-----------------------------------------
``scripts/scan_task_toolcall_leaks.py`` (task 2939) detects the *same* leak
retrospectively, but with a deliberately DIFFERENT and much narrower matcher
(its ``LEAK_TAIL`` regex — see that module's docstring for the discriminator
and the evidence behind it; it is not restated here). The two are calibrated in
opposite directions on purpose, because the cost of each error mode is
inverted:

* There, a false positive sends a human off to inspect a task that never
  leaked, so precision is bought with a discriminated regex.
* Here, a false positive is a *recoverable* rejection that carries its own
  remediation and an explicit override (``metadata={'allow_mcp_markup': True}``)
  — one retry clears it. A false NEGATIVE, by contrast, is another permanent
  corpus specimen that DF 3083 has to hunt down by eye.

So this write-time check matches bare, case-sensitive literal substrings and
accepts the resulting over-reporting. If you are tempted to "fix" one of the
two pattern sets to match the other: don't — they are answering different
questions.

Structure mirrors the sibling :mod:`fused_memory.server.near_duplicate_guard`:
module-level constants plus pure synchronous functions that do no I/O and never
raise on empty/``None`` input, so callers can hand raw handler arguments
straight through.
"""

from __future__ import annotations

import json
import time
from collections import deque
from collections.abc import Callable
from typing import Any

# Raw MCP envelope fragments that must never appear inside a write payload.
#
# Matched as bare, CASE-SENSITIVE substrings (see the module docstring for why
# this deliberately over-reports relative to scripts/scan_task_toolcall_leaks.py).
# This is the single write-time source of truth (INV-5); the same-file drift
# guard in tests/server/test_markup_tripwire.py must be updated alongside it.
MCP_MARKUP_PATTERNS: tuple[str, ...] = ('</content>', '<parameter name=', '</invoke>')

# Write-time-only control flag that bypasses the tripwire for markup a caller is
# quoting DELIBERATELY (DF 3083's own task description quotes all three literals
# in prose, so without this the very sibling this leaf exists to feed could not
# be updated). Only a literal boolean ``True`` enables it, and it is stripped
# from metadata before persistence at every boundary — mirroring the established
# ``allow_near_duplicate`` lifecycle in ``server/tools.py``. An accidental
# harness serialization leak never sets an explicit flag; an author can.
MARKUP_OVERRIDE_KEY = 'allow_mcp_markup'

# Surfaced in the rejection dict, the four tool docstrings and
# FUSED_MEMORY_INSTRUCTIONS so the remediation and the escalation pointer are
# discoverable at the point of rejection, not just in documentation the writer
# may never have read (mirrors near_duplicate_guard._NEAR_DUPLICATE_HINT).
_MARKUP_HINT = (
    'This write carries raw MCP envelope markup (see matched_pattern/field), '
    'which indicates the caller serialized part of its own tool-call envelope '
    'into the payload. Strip the leaked envelope fragment and resubmit. Do NOT '
    'work around this by rewording the payload around the fragment: DF task '
    '3083 owns the root cause and the retroactive corpus sweep, so report a '
    'recurrence there. If the markup is quoted deliberately (e.g. documenting '
    "the leak itself), override with metadata={'" + MARKUP_OVERRIDE_KEY + "': True}."
)

# Storm thresholds are plain module constants rather than FusedMemoryConfig
# fields, following the _PLACEHOLDER_DROP_STORM_* precedent (harness.py:200-215):
# this leaf owns both the predicate and its only consumer, so a config field
# would add hot-reload tier surface and a schema migration for no operator gain.
_MARKUP_STORM_THRESHOLD = 3
_MARKUP_STORM_WINDOW_SECONDS = 3600.0

# Fired only on a BURST. The wording matters: the alarm-worthy conclusion is
# that the upstream serialization leak is ACTIVE, not that this tripwire has
# started misfiring — an operator who reads it the second way would disable the
# containment and let specimens back into the corpus.
_MARKUP_STORM_HINT = (
    'Multiple MCP-envelope-markup writes were rejected in a short window: the '
    'upstream serialization leak is ACTIVE right now. This is not a sign the '
    'tripwire is misfiring — do NOT disable it. DF task 3083 owns the root '
    'cause and the retroactive corpus sweep; attach the offending agent_id, '
    'field and matched_pattern there.'
)


def find_markup_pattern(text: object) -> str | None:
    """Return the first :data:`MCP_MARKUP_PATTERNS` literal occurring in *text*.

    "First" is by POSITION IN THE TEXT, not by position in the pattern tuple:
    when several patterns are present the earliest one is reported, so the
    caller is told where the leaked envelope actually starts rather than
    whichever literal happens to be listed first.

    Matching is case-sensitive — the harness emits lowercase tags, and
    case-folding would only widen the guard onto prose that shouts a tag name.

    Pure and synchronous. *text* is expected to be a handler argument's value
    (``str``) but anything else — ``None``, an absent optional field, a dict —
    returns ``None`` without raising, so call sites need no pre-validation.
    """
    if not text or not isinstance(text, str):
        return None
    best_index = -1
    best_pattern: str | None = None
    for pattern in MCP_MARKUP_PATTERNS:
        index = text.find(pattern)
        if index == -1:
            continue
        if best_index == -1 or index < best_index:
            best_index = index
            best_pattern = pattern
    return best_pattern


def find_markup_violation(fields: dict[str, object]) -> tuple[str, str] | None:
    """Return ``(field_name, matched_pattern)`` for the first violating field.

    *fields* maps a caller-facing field name (``'content'``, ``'description'``,
    …) to its raw value. Fields are checked in dict INSERTION ORDER, so a call
    site controls which field is named when several are dirty. Returns ``None``
    when every field is clean, empty, ``None`` or non-``str``.

    Pure and synchronous; never raises (an empty map is simply not a violation).
    """
    for field_name, value in fields.items():
        pattern = find_markup_pattern(value)
        if pattern is not None:
            return field_name, pattern
    return None


def _as_metadata_dict(metadata: object) -> dict[str, Any] | None:
    """Best-effort read of *metadata* as a dict, else ``None``.

    ``submit_task``/``update_task`` accept metadata as an object OR a JSON
    string, so both shapes are understood. Anything unparseable — malformed
    JSON, a non-dict JSON payload, a wrong type entirely — yields ``None``
    without raising: validating metadata is not this module's job, and a write
    must never fail because the override helper choked on a field it does not
    own.
    """
    if isinstance(metadata, dict):
        return metadata
    if isinstance(metadata, str):
        try:
            parsed = json.loads(metadata)
        except (ValueError, TypeError):
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


def markup_override_requested(metadata: object) -> bool:
    """Return True iff *metadata* carries an explicit :data:`MARKUP_OVERRIDE_KEY` opt-in.

    Fail-closed: ONLY a literal boolean ``True`` counts, mirroring add_memory's
    ``metadata.get('allow_near_duplicate') is True`` check (``tools.py``:1199).
    A truthy-but-not-``True`` value (``'yes'``, ``1``) is far more likely to be
    unrelated data than a considered decision to write raw MCP envelope markup
    into the corpus — and the failure mode being contained, an accidental
    serialization leak, never sets an explicit flag at all.

    Never raises, for any input.
    """
    parsed = _as_metadata_dict(metadata)
    if parsed is None:
        return False
    return parsed.get(MARKUP_OVERRIDE_KEY) is True


def strip_markup_override(metadata: Any) -> Any:
    """Return *metadata* without :data:`MARKUP_OVERRIDE_KEY`, in the same shape.

    The override is a write-time-only control flag: it must never be persisted
    into stored memory metadata or the task metadata vocabulary. Returning the
    shape it was given (dict in / dict out, JSON string in / JSON string out)
    lets a call site substitute the result inline before forwarding downstream.

    NON-mutating — the caller's own dict is left intact, since the handler may
    still need the original and quietly mutating caller-owned metadata is
    action-at-a-distance this guard should not introduce. (This is the one
    deliberate divergence from ``allow_near_duplicate``'s in-place
    ``cleaned_meta.pop`` at ``tools.py``:1266, which operates on a dict it has
    already copied.)

    Unparseable input passes straight through unchanged, never raising.
    """
    if isinstance(metadata, dict):
        if MARKUP_OVERRIDE_KEY not in metadata:
            return metadata
        return {k: v for k, v in metadata.items() if k != MARKUP_OVERRIDE_KEY}
    if isinstance(metadata, str):
        parsed = _as_metadata_dict(metadata)
        if parsed is None or MARKUP_OVERRIDE_KEY not in parsed:
            return metadata
        return json.dumps({k: v for k, v in parsed.items() if k != MARKUP_OVERRIDE_KEY})
    return metadata


def build_markup_block(
    agent_id: str | None,
    field: str,
    pattern: str,
    text: str,
    *,
    storm: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the structured rejection dict returned by the four write tools.

    Mirrors :func:`near_duplicate_guard.build_near_duplicate_block`'s flat
    ``error``/``error_type``/``agent_id``/``content_excerpt``/``hint`` shape so
    both guards' agent-facing diagnostics stay uniform, and adds *field* and
    *matched_pattern* — the write has already been refused, so this dict is the
    only machine-readable account of WHICH pattern tripped and WHERE (INV-1).

    *storm* is folded in only when a rejection burst actually fired: the
    ``'storm'`` key is OMITTED entirely otherwise, rather than set to ``None``, so
    its presence is an unambiguous signal (INV-4).
    """
    block: dict[str, Any] = {
        'error': 'mcp_markup_write_blocked',
        'error_type': 'McpEnvelopeMarkupWriteRejected',
        'agent_id': agent_id,
        'field': field,
        'matched_pattern': pattern,
        'content_excerpt': text[:200],
        'hint': _MARKUP_HINT,
    }
    if storm is not None:
        block['storm'] = storm
    return block


class MarkupStormCounter:
    """Rolling-window burst detector over markup rejections (INV-4).

    One bounced write is routine; a BURST means the upstream serialization leak
    is actively running, which is the condition worth escalating rather than
    merely logging. Reproduces the established storm-counter body from
    ``reconciliation/harness.py::_record_resume_failure`` and
    ``reconciliation/bulk_reset_guard.py`` — append, prune to the window, count,
    compare to the threshold, then rate-limit to one fire per window — using
    bulk_reset_guard's guard-side injectable-clock convention
    (``time_provider`` stored as ``self._now``) so the 3600s window can be
    tested by advancing a fake clock instead of sleeping.

    State is PROCESS-LOCAL and resets on restart, like every other in-process
    storm counter in this codebase: the counter exists to catch a live burst, not
    to keep durable statistics. It is also per-instance, and ``server/tools.py``
    instantiates one per ``create_mcp_server`` call rather than as a module
    global, so no state bleeds between servers (or between tests).

    Not thread-safe by construction; the MCP tool handlers that call it run on a
    single event loop and ``record`` never awaits.
    """

    def __init__(
        self,
        threshold: int = _MARKUP_STORM_THRESHOLD,
        window_seconds: float = _MARKUP_STORM_WINDOW_SECONDS,
        time_provider: Callable[[], float] = time.time,
    ) -> None:
        self._threshold = threshold
        self._window_seconds = window_seconds
        self._now = time_provider
        self._events: deque[float] = deque()
        self._last_fire_ts: float | None = None

    def record(self) -> dict[str, Any] | None:
        """Record one rejection; return a storm summary iff a burst just fired.

        Returns ``None`` when the count within the window is below the threshold,
        AND when the threshold is met but a previous fire is still inside the
        window (the rate limit — without it, a leak emitting hundreds of writes
        would escalate hundreds of times for one incident).

        Otherwise stamps the rate-limit timestamp and returns a JSON-serializable
        summary with ``count``, ``threshold``, ``window_seconds`` and ``hint``.
        """
        now = self._now()

        # Append, then prune. The window is half-open: an event aged exactly
        # window_seconds is already out.
        self._events.append(now)
        cutoff = now - self._window_seconds
        while self._events and self._events[0] <= cutoff:
            self._events.popleft()

        count = len(self._events)
        if count < self._threshold:
            return None

        # Threshold crossed — apply the per-window rate limit.
        if self._last_fire_ts is not None and (now - self._last_fire_ts) < self._window_seconds:
            return None

        self._last_fire_ts = now
        return {
            'count': count,
            'threshold': self._threshold,
            'window_seconds': self._window_seconds,
            'hint': _MARKUP_STORM_HINT,
        }
