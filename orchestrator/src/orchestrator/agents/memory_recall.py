"""Recall memory for one dispatch and render it as the prompt's ``# Context`` block.

Each fused-memory reply is parsed once into a typed result, facts tagged to
another project are dropped, and what survives is rendered as markdown.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


FOREIGN_PROJECT_TAG_KEYS = ('src_project', 'project_id', 'group_id', 'project')
"""Metadata keys, in precedence order, that name a memory result's owning project.

``src_project`` comes first: a CGL-eta rehomed entry (task 2273) physically
lives in ``dst_project``'s collection but describes ``src_project``'s work, so
its origin must win over any co-present ``project_id``/``group_id``.
``dst_project`` is deliberately absent: it names where a fact was relocated
TO, so consulting it would certify a rehomed foreign fact as local.

``fused-memory/src/fused_memory/server/grouped_read.py::ORIGIN_PROJECT_TAG_KEYS``
stamps the tags this reads; the two are copies, pinned equal by a test.
"""

GROUPED_CHILD_KEYS = ('amendments', 'matched_children')
"""Where fused-memory nests child bodies inside a kept result's ``grouped`` block.

``group_search_results`` hangs amendment digests under ``amendments`` and
full swallowed bodies under ``matched_children``. Both render into the
``# Context`` block, so both are filtered and rendered. ``grouped['parent']``
is absent because only ``get_memory_by_id`` produces it, and recall never
calls that tool.
"""


def _canonical_project(value: str) -> str:
    """Canonicalise a project identifier the way fused-memory does.

    Mirrors ``canonicalize_project_id`` (strip, lowercase, ``'-'`` -> ``'_'``)
    so ``dark-factory`` is not mistaken for a project distinct from
    ``dark_factory``. Re-implemented because the orchestrator has no runtime
    dependency on fused-memory.
    """
    return value.strip().lower().replace('-', '_')


def _result_project(entry: dict) -> tuple[str, str] | None:
    """The ``(key, value)`` of the first non-empty string project tag, or None if untagged."""
    metadata = entry.get('metadata')
    if not isinstance(metadata, dict):
        return None
    for key in FOREIGN_PROJECT_TAG_KEYS:
        tag = metadata.get(key)
        if isinstance(tag, str) and tag.strip():
            return key, tag
    return None


def _foreign_tag(entry: dict, target: str) -> tuple[str, str] | None:
    """The ``(key, value)`` of *entry*'s project tag when it names a FOREIGN project.

    The single drop decision, shared by the top-level and the nested loop so a
    nested child is judged by exactly the rule a result is. *target* must
    already be canonical. None means keep: the entry is untagged or its tag
    names *target*.
    """
    match = _result_project(entry)
    if match is None:
        return None
    key, tag = match
    if _canonical_project(tag) == target:
        return None
    return key, tag


def _filter_child_list(children: list, parent_id: Any, target: str) -> list:
    """The children of one grouped list that are not tagged to another project.

    A non-dict child is unclassifiable, not foreign, so it is kept.
    """
    kept = []
    for child in children:
        if isinstance(child, dict) and (foreign := _foreign_tag(child, target)) is not None:
            tag_key, tag = foreign
            logger.debug(
                f'filter_foreign_project_results: dropped nested {child.get("id")!r} '
                f'under {parent_id!r} ({tag_key}={tag!r})'
            )
            continue
        kept.append(child)
    return kept


def _filter_grouped_children(entry: dict, target: str) -> tuple[dict, int]:
    """Drop cross-project children nested in a kept result, never editing *entry*.

    Returns the entry (a copy with only the shortened child lists replaced,
    when something was dropped) and how many children were dropped.
    Surgical: counts, ``truncated`` and every other ``grouped`` key stay
    exactly as the server sent them, because a count is the store's exact
    value and recomputing it from a shortened list would fabricate a number.
    Fails open: a ``grouped`` value or child collection of an unexpected type
    is left as received and logged at WARNING.
    """
    grouped = entry.get('grouped')
    if grouped is None:
        return entry, 0
    if not isinstance(grouped, dict):
        logger.warning(
            f'filter_foreign_project_results: entry {entry.get("id")!r} has a '
            f"'grouped' value of type {type(grouped).__name__}, not a dict; "
            'nested children left unfiltered'
        )
        return entry, 0
    shortened: dict[str, list] = {}
    for key in GROUPED_CHILD_KEYS:
        children = grouped.get(key)
        if children is None:
            continue
        if not isinstance(children, list):
            logger.warning(
                f'filter_foreign_project_results: entry {entry.get("id")!r} has '
                f'grouped[{key!r}] of type {type(children).__name__}, not a list; '
                'left unfiltered'
            )
            continue
        kept = _filter_child_list(children, entry.get('id'), target)
        if len(kept) != len(children):
            shortened[key] = kept
    if not shortened:
        return entry, 0
    dropped = sum(len(grouped[key]) - len(kept) for key, kept in shortened.items())
    return {**entry, 'grouped': {**grouped, **shortened}}, dropped


@dataclass(frozen=True)
class FilteredResults:
    """What survived the cross-project filter, and how much did not.

    ``dropped`` counts results removed; ``nested_dropped`` counts children
    removed from inside results that survived. They stay apart because a
    dropped child only shortens a kept fact, and the drop note an operator
    reads must say which kind of leak was blocked.
    """

    kept: tuple[Any, ...]
    dropped: int
    nested_dropped: int


def filter_foreign_project_results(results: Sequence[Any], project_id: str) -> FilteredResults:
    """Drop results, and children nested in kept results, tagged to another project.

    Untagged results are KEPT: Graphiti-sourced results carry no project tag,
    so dropping untagged ones would empty the block for most queries;
    :data:`MEMORY_CONTEXT_CAVEAT` covers that channel instead. A non-dict
    result, or one with non-dict metadata, is kept as untagged. *results* is
    never mutated.
    """
    target = _canonical_project(project_id)
    kept = []
    dropped = 0
    nested_dropped = 0
    for entry in results:
        if not isinstance(entry, dict):
            kept.append(entry)
            continue
        if (foreign := _foreign_tag(entry, target)) is not None:
            key, tag = foreign
            dropped += 1
            logger.debug(
                f'filter_foreign_project_results: dropped {entry.get("id")!r} ({key}={tag!r})'
            )
            continue
        # Only a KEPT entry is descended into: a dropped parent takes its
        # subtree with it, so counting its children would double-count.
        entry, nested = _filter_grouped_children(entry, target)
        nested_dropped += nested
        kept.append(entry)
    return FilteredResults(tuple(kept), dropped, nested_dropped)


UNCATEGORIZED = 'uncategorized'
UNDATED = 'undated'
UNKNOWN_STORE = 'unknown'
"""Placeholders for the three tag fields a result may not carry.

Graphiti-sourced results carry no category or ``created_at``, and graph edges
usually no ``temporal``. The tag is best-effort and the content is not: a
missing field renders as one of these words, never as ``None``, and never
suppresses the entry it describes.
"""


def _entry_category(entry: dict) -> str:
    """Label an entry: its own category, else its metadata copy, else its kind."""
    metadata = entry.get('metadata')
    metadata_category = metadata.get('category') if isinstance(metadata, dict) else None
    for value in (entry.get('category'), metadata_category, entry.get('kind')):
        if isinstance(value, str) and value:
            return value
    return UNCATEGORIZED


def _entry_date(entry: dict) -> str:
    """Date an entry, date-only: its ``created_at``, else the date its fact became valid."""
    temporal = entry.get('temporal')
    valid_at = temporal.get('valid_at') if isinstance(temporal, dict) else None
    for value in (entry.get('created_at'), valid_at):
        if isinstance(value, str) and len(value) >= 10:
            return value[:10]
    return UNDATED


def _memory_bullet(entry: Any, store: str, indent: str = '') -> str | None:
    """Render one recalled entry as ``- [category · date · store] content``.

    None for an entry with no text: an empty bullet spends tokens announcing
    something unreadable. Content renders WHOLE, with continuation lines
    indented so a multi-paragraph memory stays inside its own bullet.
    """
    if not isinstance(entry, dict):
        return None
    content = entry.get('content') or entry.get('digest')
    if not isinstance(content, str) or not content.strip():
        return None
    body = content.strip().replace('\n', '\n' + indent + '  ')
    return f'{indent}- [{_entry_category(entry)} · {_entry_date(entry)} · {store}] {body}'


def render_memory_results(results: Sequence[Any]) -> str:
    """Render search results as markdown bullets, '' when nothing is renderable.

    Each grouped child (:data:`GROUPED_CHILD_KEYS`) renders as a nested bullet
    tagged with its parent's store, since a collapsed child has none of its own.
    """
    bullets: list[str] = []
    for entry in results:
        if not isinstance(entry, dict):
            continue
        store = entry.get('source_store')
        store = store if isinstance(store, str) and store else UNKNOWN_STORE
        bullet = _memory_bullet(entry, store)
        if bullet is None:
            continue
        bullets.append(bullet)
        grouped = entry.get('grouped')
        if not isinstance(grouped, dict):
            continue
        for key in GROUPED_CHILD_KEYS:
            children = grouped.get(key)
            if not isinstance(children, list):
                continue
            bullets.extend(
                child_bullet
                for child in children
                if (child_bullet := _memory_bullet(child, store, indent='  ')) is not None
            )
    return '\n'.join(bullets)


def render_entity_block(payload: Mapping[str, Any], expected_name: str) -> str:
    """Render a ``get_entity`` reply, but ONLY for a node named exactly *expected_name*.

    ``get_entity`` falls back to fuzzy matching, so a missing task is answered
    with a neighbouring task's node, whose facts would read as this task's
    own. Exact-name admission is what makes the channel safe to render. An
    edge's date is best-effort and never required. Returns '' for a missing,
    empty or wrong-named node.
    """
    nodes = payload.get('nodes')
    nodes = nodes if isinstance(nodes, list) else []
    node = next(
        (n for n in nodes if isinstance(n, dict) and n.get('name') == expected_name),
        None,
    )
    if node is None:
        # A degraded reply cannot answer "no such node", because the store
        # that would hold it is the one that failed; the level says which.
        degraded = bool(payload.get('degraded'))
        logger.log(
            logging.WARNING if degraded else logging.DEBUG,
            f'render_entity_block: no node named exactly {expected_name!r} in '
            f'the reply; nothing rendered (degraded={degraded})',
        )
        return ''

    summary = node.get('summary')
    heading = f'**{expected_name}**'
    if isinstance(summary, str) and summary.strip():
        heading += f' — {summary.strip()}'

    lines = [heading]
    edges = payload.get('edges')
    for edge in edges if isinstance(edges, list) else []:
        if not isinstance(edge, dict):
            continue
        fact = edge.get('fact')
        if not isinstance(fact, str) or not fact.strip():
            continue
        temporal = edge.get('temporal')
        valid_at = temporal.get('valid_at') if isinstance(temporal, dict) else None
        dated = f' ({valid_at[:10]})' if isinstance(valid_at, str) and len(valid_at) >= 10 else ''
        lines.append(f'- {fact.strip()}{dated}')
    return '\n'.join(lines)


class MemoryFailure(Enum):
    """Why a memory query produced nothing, when the answer is "it broke".

    A timeout says the service is alive and slow; a transport failure says it
    is unreachable; a malformed reply says it answered with no usable result
    (a JSON-RPC error, a tool-level ``isError``, or a document with no results
    list). "The corpus holds nothing" is not a failure at all. The values are
    what the prompt's notices render.
    """

    TIMEOUT = 'timeout'
    TRANSPORT = 'transport'
    MALFORMED = 'malformed'


@dataclass(frozen=True)
class SearchReply:
    """A ``search`` reply that parsed into a results list."""

    results: tuple[Any, ...]
    failed_stores: tuple[str, ...] = ()


@dataclass(frozen=True)
class UnparsedReply:
    """Reply text that is not JSON at all, rendered verbatim and unfiltered.

    Failing open is deliberate: blanking a section on a serialisation surprise
    would silently lose context from every prompt. The known instance is a
    reply split across several text blocks, which join into invalid JSON.
    """

    text: str


def reported_failed_stores(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """The stores a reply reports failing.

    ``degraded``/``failed_stores`` are emitted only on a fault, so their
    presence is itself the signal.
    """
    if not payload.get('degraded'):
        return ()
    stores = payload.get('failed_stores')
    if not isinstance(stores, list):
        return ()
    return tuple(store for store in stores if isinstance(store, str) and store)


def parse_search_reply(text: str) -> SearchReply | UnparsedReply | MemoryFailure:
    """Parse a ``search`` reply once.

    JSON without a results list (the server's ``{"error", "error_type"}``
    rejection, say) recalled nothing, so it is MALFORMED rather than rendered.
    """
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return UnparsedReply(text)
    if not isinstance(payload, dict) or not isinstance(payload.get('results'), list):
        return MemoryFailure.MALFORMED
    return SearchReply(tuple(payload['results']), reported_failed_stores(payload))


MEMORY_EMPTY_NOTICE = '_No memory context available._'
MEMORY_OUTAGE_NOTICE = '_Memory unavailable ({reasons}) — proceed with codebase exploration._'
"""The two "nothing recalled" outcomes, worded apart so a reader can tell
"the corpus had nothing" from "the service failed"."""

MEMORY_OUTAGE_STREAK_THRESHOLD = 5
"""Consecutive dispatches recalling NOTHING before the outage is logged at ERROR (INV-4).

The streak counts consecutive dispatches, is reset by any dispatch that
recalled something, and re-alarms at every multiple of the threshold, so a
permanent outage keeps reporting without logging once per dispatch. The
ratified behaviour is recorded in ``docs/prds/memory-briefing-and-fusion.md``.
"""

MEMORY_SECTION_FAILURE_NOTICE = (
    '_The **{section}** section is missing: the memory query failed ({reason})._'
)
MEMORY_DEGRADED_STORES_NOTICE = (
    '_Partial recall for **{section}**: the {stores} store(s) failed._'
)


@dataclass(frozen=True)
class MemoryQueryOutcome:
    """What one memory channel produced: rendered facts, or a named reason there are none.

    ``rendered`` is markdown ready for the prompt. Neither it nor ``failure``
    being set is the honest empty answer: the query worked and the corpus had
    nothing.
    """

    rendered: str = ''
    failure: MemoryFailure | None = None
    dropped: int = 0
    nested_dropped: int = 0
    failed_stores: tuple[str, ...] = ()


def _section_notices(section: str, outcome: MemoryQueryOutcome) -> list[str]:
    """The lines a dispatch owes its reader about one channel's health (D6).

    Each names its *section*, so a reader can tell WHICH question went
    unanswered.
    """
    notices = []
    if outcome.failure:
        notices.append(MEMORY_SECTION_FAILURE_NOTICE.format(
            section=section, reason=outcome.failure.value,
        ))
    if outcome.failed_stores:
        notices.append(MEMORY_DEGRADED_STORES_NOTICE.format(
            section=section, stores=', '.join(outcome.failed_stores),
        ))
    return notices


ENTITY_CHANNEL_SUFFIX = ' (knowledge graph)'
"""Names D3's graph channel apart from the semantic one in a notice.

Both answer the same section, so an unqualified notice could not say which
corpus went missing.
"""

MEMORY_CONTEXT_CAVEAT = (
    "_This context was recalled from the `{project_id}` project's memory — "
    'it is NOT a description of this worktree. It may name tasks, repos, '
    'crates, or file paths that do not exist here. Do not assume a recalled '
    'path is real: verify it exists before reading it or `cd`-ing into it._'
)
"""Standing provenance caveat rendered right after the ``# Context`` heading.

Covers the leak channel :func:`filter_foreign_project_results` cannot reach:
untagged (Graphiti-sourced) results survive the filter unclassified, so the
caveat is what makes an agent verify a recalled path before using it.
Formatted with the project id.
"""
