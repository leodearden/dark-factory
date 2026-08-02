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
import logging
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fused_memory.server.storm_counter import StormCounter

if TYPE_CHECKING:
    from escalation.queue import EscalationQueue  # type: ignore[import-untyped]

# Defensive import of the optional ``escalation`` workspace package, mirroring
# middleware/candidate_key_escalation.py: when it is missing (minimal CI envs,
# deployments that have not installed it) the storm escalation becomes a logged
# no-op. This module sits on the MCP write path, so it must never make a write
# fail — the rejection is already decided by the time escalation is attempted.
try:
    from escalation.models import Escalation  # type: ignore[import-untyped]
    from escalation.queue import EscalationQueue  # type: ignore[import-untyped,no-redef]
    HAS_ESCALATION = True
except ImportError:  # pragma: no cover — exercised only in minimal envs
    HAS_ESCALATION = False

logger = logging.getLogger(__name__)

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

# Escalation wiring, copied shape-for-shape from
# middleware/candidate_key_escalation.py. _ANCHOR_TASK_ID is a stable per-project
# anchor (not a real task id) so the resulting ids form one greppable
# ``esc-markup-tripwire-N`` series and the dedup check has something to key on.
_QUEUE_DIRNAME: str = 'data/escalations'
_ANCHOR_TASK_ID: str = 'markup-tripwire'
_AGENT_ROLE: str = 'fused-memory/markup-tripwire'
_CATEGORY: str = 'mcp_markup_write_storm'


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


def find_markup_violation(fields: Mapping[str, object]) -> tuple[str, str] | None:
    """Return ``(field_name, matched_pattern)`` for the first violating field.

    *fields* maps a caller-facing field name (``'content'``, ``'description'``,
    …) to its raw value. Fields are checked in dict INSERTION ORDER, so a call
    site controls which field is named when several are dirty. Returns ``None``
    when every field is clean, empty, ``None`` or non-``str``.

    Typed ``Mapping[str, object]`` rather than ``dict[str, object]``: this
    function only READS the map, and ``dict`` is invariant in its value type, so
    a ``dict`` annotation would reject an ordinary homogeneous call site such as
    ``{'title': a, 'description': b}`` (inferred ``dict[str, str]``). ``Mapping``
    is covariant in the value type and accepts every dict a caller can build.

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
    merely logging.

    A THIN ADAPTER over the shared ``server/storm_counter.StormCounter``, which
    is the single home for the append/prune/count/rate-limit body (INV-5, task
    3088 — this class was its third copy). Everything markup-specific stays
    here: the module-constant defaults, the ``project=`` parameter name, the
    ``projects`` result key and the ``hint``.

    The per-project dimension is load-bearing, not decoration: one server
    instance serves every known project, so a window that mixed two projects
    into a bare count would let the caller attribute the whole burst to
    whichever write happened to cross the threshold — and file the escalation
    into that project's queue while the actually-leaking project got nothing.

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
        self._counter = StormCounter(time_provider=time_provider)

    def record(self, project: str | None = None) -> dict[str, Any] | None:
        """Record one rejection; return a storm summary iff a burst just fired.

        *project* is the label the rejection belongs to — the resolved
        ``project_root``, or ``None`` when the boundary could not resolve one
        (``add_memory``/``add_episode`` take a ``project_id``, which for a project
        absent from the known-projects registry maps to no root at all).
        Unlabelled rejections still count toward the burst but are not named.

        Returns ``None`` when the count within the window is below the threshold,
        AND when the threshold is met but a previous fire is still inside the
        window (the rate limit — without it, a leak emitting hundreds of writes
        would escalate hundreds of times for one incident).

        Otherwise returns a JSON-serializable summary with ``count``,
        ``threshold``, ``window_seconds``, ``hint`` and ``projects`` — the sorted
        DISTINCT non-``None`` labels seen in the window, so the caller can
        attribute the burst (and escalate to every project it touched) instead of
        blaming whichever write crossed the threshold.

        This counter's threshold and window are module constants, so passing them
        through on each call is a no-op for reload semantics here; the shared
        class takes them per call for the benefit of consumers whose knobs come
        from a green-tier config leaf (see ``storm_counter``'s module docstring).
        """
        summary = self._counter.record(
            threshold=self._threshold,
            window_seconds=self._window_seconds,
            label=project,
        )
        if summary is None:
            return None
        return {
            'count': summary['count'],
            'threshold': summary['threshold'],
            'window_seconds': summary['window_seconds'],
            'projects': summary['labels'],
            'hint': _MARKUP_STORM_HINT,
        }


def emit_markup_storm_escalation(
    project_root: str | None,
    storm: dict[str, Any],
) -> str | None:
    """File an ``mcp_markup_write_storm`` escalation for a rejection burst (INV-4).

    Returns the escalation id — freshly filed, or the id of an already-open
    escalation for this project (dedup) — or ``None`` when filing is impossible
    or fails.

    NEVER raises. This is called from an MCP write path whose rejection has
    already been decided, so escalation is purely additive: every failure mode
    degrades to ``None`` plus a log line rather than changing the write's
    outcome. Copied shape-for-shape from
    :func:`middleware.candidate_key_escalation.emit_residual_candidate_key_escalation`,
    adding an early return for a ``None`` *project_root* — ``add_memory`` /
    ``add_episode`` take a ``project_id``, which for an unknown project resolves
    to no root at all, and that must be a quiet no-op rather than a crash.

    The anchor dedup matters beyond the counter's own per-window rate limit: a
    leak running for hours would otherwise file one escalation per window, so
    those collapse into the single open record until an operator resolves it.
    """
    if project_root is None:
        logger.debug(
            'markup_tripwire: no project_root resolved; storm %r will not be escalated',
            storm,
        )
        return None
    if not HAS_ESCALATION:
        logger.debug(
            'markup_tripwire: escalation package unavailable; storm %r in '
            'project_root=%r will not be escalated',
            storm, project_root,
        )
        return None

    try:
        queue = EscalationQueue(Path(project_root) / _QUEUE_DIRNAME)
    except Exception:
        logger.exception(
            'markup_tripwire: failed to open the escalation queue for '
            'project_root=%r; storm %r not escalated',
            project_root, storm,
        )
        return None

    # Best-effort dedup: a read failure falls THROUGH to filing rather than
    # bailing out — losing duplicate-suppression is strictly better than losing
    # the alarm for an actively running leak.
    try:
        existing = queue.get_by_task(_ANCHOR_TASK_ID, status='pending')
    except Exception:
        logger.exception(
            'markup_tripwire: failed to check for an existing open escalation '
            'for project_root=%r; proceeding to file a new one',
            project_root,
        )
        existing = []
    if existing:
        logger.info(
            'markup_tripwire: %s already open for project_root=%r (storm %r now); '
            'not filing a duplicate',
            existing[0].id, project_root, storm,
        )
        return existing[0].id

    count = storm.get('count')
    window_seconds = storm.get('window_seconds')
    detail = '\n'.join([
        f'project_root={project_root!r}',
        f'rejected_writes_in_window={count!r}',
        f'threshold={storm.get("threshold")!r}',
        f'window_seconds={window_seconds!r}',
        # The window is shared across every project this server serves, so the
        # count above may span more than this project_root. Naming them all
        # keeps the record honest and points at the co-affected queues.
        f'projects_in_window={storm.get("projects")!r}',
        '',
        'The fused-memory MCP write tripwire (task 3141) rejected multiple '
        'writes carrying raw MCP envelope markup within one rolling window. A '
        'burst means the upstream harness serialization leak is ACTIVE right '
        'now, not that the tripwire is misfiring — do NOT disable it, or '
        'further specimens will land permanently in the corpus.',
        '',
        'DF task 3083 owns the root cause, the Qdrant payload text-match read '
        'tool and the retroactive corpus sweep. Attach the agent_id, field and '
        'matched_pattern from the rejection responses (grep the server logs for '
        "'markup_tripwire_storm') to 3083.",
    ])

    try:
        esc = Escalation(  # type: ignore[possibly-unbound]
            id=queue.make_id(_ANCHOR_TASK_ID),
            task_id=_ANCHOR_TASK_ID,
            agent_role=_AGENT_ROLE,
            severity='blocking',
            category=_CATEGORY,
            summary=(
                f'{count} MCP write(s) rejected for leaked envelope markup in '
                f'{window_seconds}s — serialization leak active (see DF 3083)'
            ),
            detail=detail,
            suggested_action=(
                'identify the leaking caller from the rejection logs and report '
                'it on DF task 3083'
            ),
            level=1,
        )
        esc_id = queue.submit(esc)
    except Exception:
        # A queue I/O failure must not propagate: the write has already been
        # rejected, and the ERROR log at the call site has already recorded the
        # burst. The operator simply loses the queued heads-up.
        logger.exception(
            'markup_tripwire: failed to submit storm escalation for '
            'project_root=%r (storm %r)',
            project_root, storm,
        )
        return None

    logger.warning(
        'markup_tripwire: queued %s for project_root=%r (storm %r)',
        esc_id, project_root, storm,
    )
    return esc_id
