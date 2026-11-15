"""Pure renderers for Stage 1's LLM payload sections.

Also holds the instant parsing that Stage 1's "new since last reconciliation"
freshness filter shares, so a record timestamp is compared as an instant and
never as a string.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from datetime import UTC, datetime

from fused_memory.models.reconciliation import (
    ContextFetchFailure,
    ContextItem,
    ReconciliationEvent,
    Watermark,
)
from fused_memory.reconciliation.child_memory_link import render_parent_link
from fused_memory.reconciliation.dangling_citation import render_dangling_citation_line
from fused_memory.reconciliation.task_filter import strip_snapshot_lines
from fused_memory.utils.clip_disclosure import elide_text

logger = logging.getLogger(__name__)

_RECORD_BODY_CHARS = 2_000  # basis: plans/recon-evidence-caps-measurement-2026-10.md

CONTEXT_FETCH_FAILED_HEADING = 'Context fetch failed'


def to_utc(dt: datetime) -> datetime:
    """Coerce *dt* to a tz-aware UTC datetime, assuming UTC when naive.

    Shared normalization step for both sides of ``is_newer_than_watermark``'s
    comparison, mirroring ``backends/graphiti_client.py::_as_sortable_utc``'s
    "naive means UTC" convention for its sort key.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def parse_instant(raw_ts: str | None) -> datetime | None:
    """Parse *raw_ts* into a tz-aware UTC datetime, or None if it is
    missing, empty, or unparseable ("undatable").

    Catches both ``ValueError`` (malformed ISO-8601 text) and ``TypeError``
    (a non-str, non-None value such as a ``datetime``, an int epoch, or a
    dict — ``datetime.fromisoformat`` raises ``TypeError`` rather than
    ``ValueError`` for these) so one malformed record is classified as
    undatable rather than raising out of the caller's filter loop.
    """
    if not raw_ts:
        return None
    try:
        ts = datetime.fromisoformat(raw_ts)
    except (TypeError, ValueError):
        return None
    return to_utc(ts)


def is_newer_than_watermark(raw_ts: str | None, watermark: datetime) -> bool:
    """True iff *raw_ts* is a parseable instant STRICTLY after *watermark*.

    Both sides are compared as instants, never as strings: *raw_ts* is
    parsed with ``datetime.fromisoformat`` and *watermark* is taken as
    given, then each is coerced to a tz-aware UTC datetime — naive
    (tzinfo-less) values on EITHER side are assumed to already be UTC
    rather than raising on naive-vs-aware comparison, mirroring
    ``backends/graphiti_client.py::_as_sortable_utc`` — before comparing
    with ``>`` (strictly after, not ``>=``, so an instant exactly equal to
    *watermark* is not re-surfaced). A non-UTC offset is therefore compared
    correctly by its true instant, not by its printed digits.

    Comparing these values as STRINGS is wrong: ``str(datetime)`` renders
    with a space separator (``'2026-08-20 12:00:00+00:00'``) while
    ``created_at`` values produced by
    ``services/memory_service.py::_created_at_to_utc_iso`` use ``T``
    (``'2026-08-20T01:52:27+00:00'``) — a lexical ``>`` then short-circuits
    at the separator and degenerates to date-granularity (task 4574).

    *raw_ts* that is missing, empty, or unparseable returns False — an
    undatable record cannot be shown to be newer than the watermark, so it
    is treated as not-newer rather than raising.
    """
    ts = parse_instant(raw_ts)
    if ts is None:
        return False
    return ts > to_utc(watermark)


def format_events(events: list[ReconciliationEvent]) -> str:
    if not events:
        return 'No events.'
    lines = []
    for e in events:
        lines.append(f'- [{e.type.value}] {e.timestamp.isoformat()}: {json.dumps(e.payload)}')
    return '\n'.join(lines)


def append_note(body: str, note: str) -> str:
    return f'{body}\n\n{note}' if note else body


def format_episodes(episodes: list[dict]) -> tuple[str, int]:
    if not episodes:
        return 'No new episodes.', 0
    lines = []
    total_stripped = 0
    for ep in episodes:
        raw = ep.get('content') or ''
        cleaned, n = strip_snapshot_lines(raw)
        total_stripped += n
        content = elide_text(cleaned, _RECORD_BODY_CHARS, field='content').text
        lines.append(f'- [{ep.get("uuid", "?")}] {ep.get("created_at", "?")}: {content}')
    return '\n'.join(lines), total_stripped


def format_memories(memories: list[dict]) -> tuple[str, int]:
    if not memories:
        return 'No memories.', 0
    lines = []
    total_stripped = 0
    for m in memories:
        raw = m.get('memory') or ''
        cleaned, n = strip_snapshot_lines(raw)
        total_stripped += n
        content = elide_text(cleaned, _RECORD_BODY_CHARS, field='content').text
        meta = m.get('metadata', {}) or {}
        cat = meta.get('category', '?')
        lines.append(f'- [{m.get("id", "?")}]{render_parent_link(meta)} ({cat}): {content}')
    return '\n'.join(lines), total_stripped


def format_context_items(items: dict[str, ContextItem]) -> tuple[str, int]:
    """Format context items grouped by source, stripping count-snapshot lines."""
    if not items:
        return 'No related context.', 0
    by_source: dict[str, list[ContextItem]] = {}
    for item in items.values():
        by_source.setdefault(item.source, []).append(item)
    sections = []
    total_stripped = 0
    for source, source_items in sorted(by_source.items()):
        sections.append(f'**{source}** ({len(source_items)})')
        for item in source_items:
            cleaned, n = strip_snapshot_lines(item.formatted)
            total_stripped += n
            sections.append(cleaned)
    return '\n'.join(sections), total_stripped


def format_context_fetch_failures(failures: Sequence[ContextFetchFailure]) -> str:
    """Name each event whose related context is unknown; '' when every fetch succeeded."""
    if not failures:
        return ''
    lines = [f'**{CONTEXT_FETCH_FAILED_HEADING}** ({len(failures)})']
    lines.extend(f'- [{f.event_type}] {f.event_id}: {f.error}' for f in failures)
    lines.append(
        'The related context for these events is UNKNOWN, not absent: do not infer '
        'from its absence that no duplicate, contradiction or related record exists.'
    )
    return '\n'.join(lines)


def format_findings(findings: list[dict]) -> str:
    """Render a list of finding dicts as a human-readable string.

    Each finding is formatted with its header fields (description, severity,
    category, suggested_action) followed by the four typed citation lists
    (cited_entities, cited_edges, cited_tasks, cited_memories) from PRD §9.3,
    then the dangling-citation line naming the repair ids when the finding
    carries that handoff record (task 5749).
    Empty citation lists are omitted from output.

    Rendering limits (reviewer finding incomplete_rendering):
    - Each citation list is capped at 5 entries (``... +N more`` appended).
    - ``fact_text_snapshot`` is truncated at 120 characters to avoid inflating
      payload size when feeding findings into the next reconciliation context.
    - cited_tasks includes the task title so downstream consumers can read
      findings without re-fetching tasks.
    """
    _CITE_CAP = 5
    _SNAPSHOT_CAP = 120
    # Caps for LLM-controlled free-text fields (reviewer finding incomplete_rendering):
    # without a cap, verbose descriptions from a handful of findings can dominate
    # the downstream stage payload and swamp per-citation limits.
    _DESC_CAP = 500
    _ACTION_CAP = 500

    if not findings:
        return 'No findings.'
    lines = []
    for i, f in enumerate(findings, 1):
        desc = f.get('description', '?')
        severity = f.get('severity', '?')
        category = f.get('category', '?')
        action = f.get('suggested_action', '?')

        if len(desc) > _DESC_CAP:
            desc = desc[:_DESC_CAP] + '...'
        if len(action) > _ACTION_CAP:
            action = action[:_ACTION_CAP] + '...'

        parts = [
            f'{i}. [{severity}/{category}] {desc}',
            f'   Suggested action: {action}',
        ]

        cited_entities = f.get('cited_entities') or []
        if cited_entities:
            shown = cited_entities[:_CITE_CAP]
            entity_strs = ', '.join(
                f"{e.get('canonical_name', '?')} ({e.get('entity_uuid', '?')})"
                for e in shown
            )
            extra = len(cited_entities) - len(shown)
            if extra:
                entity_strs += f', ... +{extra} more'
                logger.debug('format_findings: finding %d cited_entities truncated (%d dropped)', i, extra)
            parts.append(f'   Cited entities: {entity_strs}')

        cited_edges = f.get('cited_edges') or []
        if cited_edges:
            shown = cited_edges[:_CITE_CAP]
            edge_parts = []
            for e in shown:
                snapshot = e.get('fact_text_snapshot', '?')
                if len(snapshot) > _SNAPSHOT_CAP:
                    snapshot = snapshot[:_SNAPSHOT_CAP] + '...'
                edge_parts.append(f"{e.get('edge_uuid', '?')}: {snapshot}")
            edge_strs = ', '.join(edge_parts)
            extra = len(cited_edges) - len(shown)
            if extra:
                edge_strs += f', ... +{extra} more'
                logger.debug('format_findings: finding %d cited_edges truncated (%d dropped)', i, extra)
            parts.append(f'   Cited edges: {edge_strs}')

        cited_tasks = f.get('cited_tasks') or []
        if cited_tasks:
            shown = cited_tasks[:_CITE_CAP]
            task_strs = ', '.join(
                f"{t.get('project_id', '?')}/{t.get('task_id', '?')} ({t.get('title', '?')})"
                for t in shown
            )
            extra = len(cited_tasks) - len(shown)
            if extra:
                task_strs += f', ... +{extra} more'
                logger.debug('format_findings: finding %d cited_tasks truncated (%d dropped)', i, extra)
            parts.append(f'   Cited tasks: {task_strs}')

        cited_memories = f.get('cited_memories') or []
        if cited_memories:
            shown = cited_memories[:_CITE_CAP]
            memory_strs = ', '.join(
                f"{m.get('memory_id', '?')} ({m.get('store', '?')})"
                for m in shown
            )
            extra = len(cited_memories) - len(shown)
            if extra:
                memory_strs += f', ... +{extra} more'
                logger.debug('format_findings: finding %d cited_memories truncated (%d dropped)', i, extra)
            parts.append(f'   Cited memories: {memory_strs}')

        line = render_dangling_citation_line(f)
        if line is not None:
            parts.append(f'   {line}')

        lines.append('\n'.join(parts))
    return '\n'.join(lines)


def format_watermark(watermark: Watermark, *, include_freshness_cutoffs: bool = False) -> str:
    """Render the "Previous Reconciliation" section.

    *include_freshness_cutoffs* must be True ONLY from the caller that
    actually ran the episode/mem0 "new since last reconciliation" filters
    against these cutoffs — today that is exclusively the time-windowed
    ``assemble_payload`` path. ``_format_assembled_payload`` (the
    ContextAssembler/token-budget path) does no freshness filtering at all
    and formats ``### Related Context`` instead of a filtered episode/mem0
    count, so it leaves this False (the default): showing the cutoffs there
    would claim a relationship to the displayed items that does not exist
    (task 4574 review amendment).
    """
    if watermark.last_full_run_completed is None:
        return 'First run — no previous reconciliation.'
    lines = [
        f'Last full run: {watermark.last_full_run_id} '
        f'at {watermark.last_full_run_completed.isoformat()}'
    ]
    if include_freshness_cutoffs:
        # Disclose the freshness cutoffs the episode/mem0 "new since last
        # reconciliation" filters actually compared against (task 4574).
        # Previously invisible here — the payload showed a filtered COUNT
        # ("New Episodes Since Last Reconciliation (N)") with no way to see
        # what cursor N was computed against, which is why the 894fbe90
        # incident survived three full cycles of re-investigation undiagnosed.
        # Rendered via .isoformat() (never str(datetime), which uses a space
        # separator — see is_newer_than_watermark) and omitted entirely
        # (rather than printed as the literal 'None') when unset, e.g. on a
        # watermark that has a completed full run but has not yet recorded an
        # episode or memory cutoff.
        if watermark.last_episode_timestamp is not None:
            lines.append(
                f'Episode freshness cutoff: {watermark.last_episode_timestamp.isoformat()}'
            )
        if watermark.last_memory_timestamp is not None:
            lines.append(
                f'Mem0 memory freshness cutoff: {watermark.last_memory_timestamp.isoformat()}'
            )
    return '\n'.join(lines)
