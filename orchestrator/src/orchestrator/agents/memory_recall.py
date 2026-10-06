"""Recall memory for one dispatch and render it as the prompt's ``# Context`` block.

:class:`MemoryRecall` is the one entry point. Each fused-memory reply is
parsed once into a typed result, facts tagged to another project are dropped,
and what survives is rendered as markdown and composed into the block.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from shared.briefing_queries import BriefingQuerySpec, BriefingScope, queries_for

from orchestrator.mcp_lifecycle import (
    is_timeout_failure,
    mcp_call,
    tool_error_text,
    tool_text_blocks,
)

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

CONTESTED_CHILD_KEY = 'contested'
"""The flag fused-memory stamps on a grouped child that contests its parent.

``fused-memory/src/fused_memory/server/grouped_read.py::_digest_entry`` stamps
it from ``is_contested_child``, the single definition of contested. It is read
here exactly as stamped (``is True``), never re-derived from the child's own
``x_contested`` metadata. A copy rather than an import, because the
orchestrator has no runtime dependency on fused-memory; the recorded replies
under ``orchestrator/tests/fixtures/grouped_search_contesting_child/`` pin it
(see their ``PROVENANCE.md``).
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

Graphiti-sourced results carry no category, and a Graphiti row is undated only
when it carries neither the edge's ``created_at`` nor ``temporal.valid_at``.
The tag is best-effort and the content is not: a missing field renders as one
of these words, never as ``None``, and never suppresses the entry it describes.
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
    """Date an entry, date-only, by when it entered memory: its ``created_at``.

    One meaning for every store, so a Graphiti edge shows its birth even when it
    also carries ``temporal.valid_at``; the date its fact became valid is only
    the fallback for a hit with no ``created_at``.
    """
    temporal = entry.get('temporal')
    valid_at = temporal.get('valid_at') if isinstance(temporal, dict) else None
    for value in (entry.get('created_at'), valid_at):
        if isinstance(value, str) and len(value) >= 10:
            return value[:10]
    return UNDATED


def _entry_id(entry: dict) -> str | None:
    """An entry's id when it is a non-empty string, the only kind usable as a key."""
    entry_id = entry.get('id')
    return entry_id if isinstance(entry_id, str) and entry_id else None


def _entry_text(entry: Any) -> str | None:
    """What an entry has to read, its ``content`` else its ``digest``; None when blank."""
    if not isinstance(entry, dict):
        return None
    text = entry.get('content') or entry.get('digest')
    return text if isinstance(text, str) and text.strip() else None


def _memory_bullet(
    entry: Any, store: str, indent: str = '', label: str | None = None,
) -> str | None:
    """Render one recalled entry as ``- [category · date · store] content``.

    A *label* leads the tag when given. None for an entry with no text: an
    empty bullet spends tokens announcing something unreadable. Content
    renders WHOLE, with continuation lines indented so a multi-paragraph
    memory stays inside its own bullet.
    """
    text = _entry_text(entry)
    if text is None:
        return None
    body = text.strip().replace('\n', '\n' + indent + '  ')
    tag = f'{_entry_category(entry)} · {_entry_date(entry)} · {store}'
    if label is not None:
        tag = f'{label} · {tag}'
    return f'{indent}- [{tag}] {body}'


def _grouped_children(entry: dict) -> list[dict]:
    """The dict children nested in *entry*'s ``grouped`` block, in key order.

    A ``grouped`` value, child collection or child of an unexpected type
    contributes nothing.
    """
    grouped = entry.get('grouped')
    if not isinstance(grouped, dict):
        return []
    return [
        child
        for key in GROUPED_CHILD_KEYS
        if isinstance(children := grouped.get(key), list)
        for child in children
        if isinstance(child, dict)
    ]


def _is_contesting(child: dict) -> bool:
    return child.get(CONTESTED_CHILD_KEY) is True


def _entry_store(entry: dict, default: str) -> str:
    """The store an entry names in ``source_store``, else *default*."""
    store = entry.get('source_store')
    return store if isinstance(store, str) and store else default


def _contesting_bullet(
    child: dict, hits_by_id: Mapping[str, dict], parent_store: str,
) -> str | None:
    """Render a child that contests its parent, tagged ``contests its parent``.

    The child's full body is on the wire only as its own top-level hit, so
    that hit renders, under its own store, when the reply carries one. The
    child entry itself is the fallback, under its parent's store.
    """
    label = 'contests its parent'
    child_id = _entry_id(child)
    hit = hits_by_id.get(child_id) if child_id else None
    if hit is not None:
        bullet = _memory_bullet(hit, _entry_store(hit, parent_store), indent='  ', label=label)
        if bullet is not None:
            return bullet
    return _memory_bullet(child, parent_store, indent='  ', label=label)


def _result_bullets(entry: dict, hits_by_id: Mapping[str, dict]) -> list[str]:
    """One result's bullet, then its contesting children, then its other children."""
    store = _entry_store(entry, UNKNOWN_STORE)
    bullet = _memory_bullet(entry, store)
    if bullet is None:
        return []
    children = _grouped_children(entry)
    contesting = [
        child_bullet
        for child in children
        if _is_contesting(child)
        and (child_bullet := _contesting_bullet(child, hits_by_id, store)) is not None
    ]
    others = [
        child_bullet
        for child in children
        if not _is_contesting(child)
        and (child_bullet := _memory_bullet(child, store, indent='  ')) is not None
    ]
    return [bullet, *contesting, *others]


def render_memory_results(results: Sequence[Any]) -> str:
    """Render search results as markdown bullets, '' when nothing is renderable.

    Each grouped child (:data:`GROUPED_CHILD_KEYS`) renders as a nested bullet
    tagged with its parent's store, since a collapsed child has none of its own.
    A child marked :data:`CONTESTED_CHILD_KEY` renders first under its parent,
    tagged ``contests its parent``, from its own top-level hit and that hit's
    store when the reply carries one; that hit is then not rendered again at
    its own rank.
    """
    entries = [entry for entry in results if isinstance(entry, dict)]
    hits_by_id: dict[str, dict] = {}
    for entry in entries:
        if (entry_id := _entry_id(entry)) is not None:
            hits_by_id.setdefault(entry_id, entry)
    rendered_under_parent = {
        child_id
        for entry in entries
        if _entry_text(entry) is not None
        for child in _grouped_children(entry)
        if _is_contesting(child) and (child_id := _entry_id(child)) is not None
    }
    bullets: list[str] = []
    for entry in entries:
        if _entry_id(entry) in rendered_under_parent:
            continue
        bullets.extend(_result_bullets(entry, hits_by_id))
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
    (a JSON-RPC error, a tool-level ``isError``, a search document with no
    results list, or a graph reply that is not a JSON object). "The corpus
    holds nothing" is not a failure at all. The values are
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

_LOOP_FAILURE_NOTICE = (
    '_Memory unavailable for the remaining queries — proceed with '
    'codebase exploration for anything not covered above._'
)


@dataclass(frozen=True)
class RecallTally:
    """One dispatch's recall: what rendered, what it owes the reader, how each search went.

    ``searches`` holds the search channel's outcomes only. The graph channel
    reports through ``notices`` but is not counted, because the outage
    verdict and the drop note's query count are about the search table.
    ``loop_failure`` is set when the recall loop itself broke.
    """

    sections: tuple[str, ...] = ()
    notices: tuple[str, ...] = ()
    searches: tuple[MemoryQueryOutcome, ...] = ()
    loop_failure: MemoryFailure | None = None

    @property
    def reasons(self) -> tuple[MemoryFailure, ...]:
        """Each distinct search failure in order, then the loop failure."""
        failures = [search.failure for search in self.searches if search.failure is not None]
        if self.loop_failure is not None:
            failures.append(self.loop_failure)
        return tuple(dict.fromkeys(failures))

    @property
    def is_outage(self) -> bool:
        """Nothing worked: no section recalled, and every search failed or the loop broke.

        One failed query among two is a partial recall, which the section
        notices report; a loop that broke after a section was recalled still
        put memory in the prompt.
        """
        failed = sum(1 for search in self.searches if search.failure is not None)
        return (
            not self.sections
            and bool(self.reasons)
            and (self.loop_failure is not None or failed == len(self.searches))
        )

    @property
    def drop_note(self) -> str:
        """How much the cross-project filter blocked, or '' when it blocked nothing.

        Result slots and nested records are named apart, and the query count
        is named too, because one foreign fact can match every query.
        """
        dropped = sum(search.dropped for search in self.searches)
        nested = sum(search.nested_dropped for search in self.searches)
        if dropped <= 0 and nested <= 0:
            return ''
        counted = []
        if dropped > 0:
            counted.append(f'{dropped} memory result slot(s)')
        if nested > 0:
            counted.append(f'{nested} nested memory record(s)')
        queries = len(self.searches)
        query_word = 'query' if queries == 1 else 'queries'
        return (
            f'{" and ".join(counted)} across {queries} {query_word} '
            'were tagged to another project and filtered out'
        )


def render_context_block(tally: RecallTally, project_id: str) -> str:
    """Compose the ``# Context`` block from one dispatch's tally.

    With nothing recalled, the outage or empty notice leads (the legibility
    digest recognises the block by that leading line), followed by every
    section notice and the drop note. With something recalled, the
    provenance caveat leads, and no later failure may suppress it.
    """
    drop_note = tally.drop_note
    if not tally.sections:
        family = (
            MEMORY_OUTAGE_NOTICE.format(reasons=', '.join(r.value for r in tally.reasons))
            if tally.is_outage else MEMORY_EMPTY_NOTICE
        )
        body = '\n\n'.join([family, *tally.notices])
        if drop_note:
            body += f'\n\n_Note: {drop_note}._'
        return f'# Context\n\n{body}'

    caveat = MEMORY_CONTEXT_CAVEAT.format(project_id=project_id)
    if drop_note:
        caveat += f'\n\n_In total, {drop_note}._'
    notices = list(tally.notices)
    if tally.loop_failure is not None:
        notices.append(_LOOP_FAILURE_NOTICE)
    parts = list(tally.sections)
    if notices:
        parts.append('\n'.join(notices))
    return '# Context\n\n' + caveat + '\n\n' + '\n\n---\n\n'.join(parts)


def _search_outcome(text: str, subject: str, project_id: str) -> MemoryQueryOutcome:
    """Parse, filter and render one search reply's text.

    No text is an honest empty answer: the tool replied and recalled nothing.
    """
    if not text:
        return MemoryQueryOutcome()
    reply = parse_search_reply(text)
    if isinstance(reply, MemoryFailure):
        logger.warning(f'{subject} answered without a results list: {text!r}')
        return MemoryQueryOutcome(failure=reply)
    if isinstance(reply, UnparsedReply):
        logger.warning(f'{subject} answered with text that is not JSON; rendering it unfiltered')
        return MemoryQueryOutcome(rendered=reply.text)
    filtered = filter_foreign_project_results(reply.results, project_id)
    return MemoryQueryOutcome(
        rendered=render_memory_results(filtered.kept),
        dropped=filtered.dropped,
        nested_dropped=filtered.nested_dropped,
        failed_stores=reply.failed_stores,
    )


def _entity_outcome(text: str, expected_name: str) -> MemoryQueryOutcome:
    """Decode one ``get_entity`` reply once, render it and read its store health.

    Unlike a search reply, a reply that is not a JSON object cannot fail open:
    the exact-name admission in :func:`render_entity_block` needs the parsed
    node, so rendering the text verbatim would let a fuzzy neighbour through.
    It is MALFORMED instead.
    """
    if not text:
        return MemoryQueryOutcome()
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        logger.warning(f'MCP get_entity reply for {expected_name!r} is not JSON ({e})')
        return MemoryQueryOutcome(failure=MemoryFailure.MALFORMED)
    if not isinstance(payload, dict):
        logger.warning(f'MCP get_entity reply for {expected_name!r} is not a JSON object: {text!r}')
        return MemoryQueryOutcome(failure=MemoryFailure.MALFORMED)
    return MemoryQueryOutcome(
        rendered=render_entity_block(payload, expected_name),
        failed_stores=reported_failed_stores(payload),
    )


class MemoryRecall:
    """Recalls fused-memory for each dispatch and renders its ``# Context`` block."""

    def __init__(self, memory_url: str, project_id: str):
        self._memory_url = memory_url
        self._project_id = project_id
        self._outage_streak = 0
        """Consecutive dispatches whose recall produced nothing at all.

        Counted across every workflow this process serves, because the
        question is about the shared SERVICE; any dispatch that recalled
        something resets it.
        """

    async def context_block(self, scope: BriefingScope, caller_agent_id: str) -> str:
        """The ``# Context`` block for a dispatch about *scope*, asked by *caller_agent_id*."""
        tally = await self._recall(scope, caller_agent_id)
        self._note_outage(tally.is_outage)
        if tally.drop_note:
            logger.info(
                f'MemoryRecall.context_block: {tally.drop_note} of the context assembled '
                f'for {self._project_id!r}'
            )
        return render_context_block(tally, self._project_id)

    async def _recall(self, scope: BriefingScope, caller_agent_id: str) -> RecallTally:
        """Fire the query table for *scope* and tally what each query produced."""
        sections: list[str] = []
        notices: list[str] = []
        searches: list[MemoryQueryOutcome] = []
        try:
            for spec, query in queries_for(scope):
                outcome = await self._search(spec, query, caller_agent_id, scope.task_id)
                searches.append(outcome)
                notices.extend(_section_notices(spec.section_title, outcome))
                blocks = [outcome.rendered]
                if spec.wants_entity_block and scope.task_id:
                    entity = await self._task_entity(scope.task_id)
                    notices.extend(_section_notices(
                        spec.section_title + ENTITY_CHANNEL_SUFFIX, entity,
                    ))
                    blocks.append(entity.rendered)
                body = '\n\n'.join(block for block in blocks if block)
                if body:
                    sections.append(f'## {spec.section_title}\n\n{body}')
        except Exception as e:
            # The loop itself broke rather than one query, so no section can
            # name it; it still counts toward the outage verdict.
            logger.warning(f'Failed to fetch memory context: {e}')
            return RecallTally(
                tuple(sections), tuple(notices), tuple(searches), MemoryFailure.TRANSPORT,
            )
        return RecallTally(tuple(sections), tuple(notices), tuple(searches))

    async def _search(
        self,
        spec: BriefingQuerySpec,
        query: str,
        caller_agent_id: str,
        caller_task_id: str | None,
    ) -> MemoryQueryOutcome:
        """Ask one query, scoped by its spec and declaring who asks (D8).

        An empty ``stores``/``categories`` tuple is omitted rather than sent
        empty, so the server applies its own routing instead of a filter that
        matches nothing.
        """
        arguments: dict[str, Any] = {
            'query': query,
            'project_id': self._project_id,
            'limit': spec.limit,
            'caller_agent_id': caller_agent_id,
        }
        if spec.stores:
            arguments['stores'] = list(spec.stores)
        if spec.categories:
            arguments['categories'] = list(spec.categories)
        if caller_task_id:
            arguments['caller_task_id'] = caller_task_id
        subject = f'Memory search for {query!r}'
        text = await self._call_tool('search', arguments, subject)
        if isinstance(text, MemoryFailure):
            return MemoryQueryOutcome(failure=text)
        return _search_outcome(text, subject, self._project_id)

    async def _task_entity(self, task_id: str) -> MemoryQueryOutcome:
        """What the knowledge graph records ABOUT this task: D3's second channel."""
        expected_name = f'Task {task_id}'
        text = await self._call_tool(
            'get_entity',
            {'name': expected_name, 'project_id': self._project_id},
            f'MCP get_entity for {expected_name!r}',
        )
        if isinstance(text, MemoryFailure):
            return MemoryQueryOutcome(failure=text)
        return _entity_outcome(text, expected_name)

    async def _call_tool(
        self, tool_name: str, arguments: dict[str, Any], subject: str,
    ) -> str | MemoryFailure:
        """Call one fused-memory tool: its joined text blocks, or why there are none.

        Every failure is NAMED and logged at WARNING, so an unreachable
        service never reads as an empty corpus.
        """
        try:
            result = await mcp_call(
                f'{self._memory_url}/mcp',
                'tools/call',
                {'name': tool_name, 'arguments': arguments},
                timeout=10,
            )
        except Exception as e:
            # The TYPE raised says nothing: on retry exhaustion mcp_call
            # re-raises a plain RuntimeError that keeps a timeout only as
            # __cause__, which is_timeout_failure unwraps.
            failure = MemoryFailure.TIMEOUT if is_timeout_failure(e) else MemoryFailure.TRANSPORT
            logger.warning(f'{subject} failed ({failure.value}): {type(e).__name__}: {e}')
            return failure

        reply = result.get('result') if isinstance(result, dict) else None
        if not isinstance(reply, dict):
            logger.warning(f'{subject} answered with no tool result: {result!r}')
            return MemoryFailure.MALFORMED
        error_text = tool_error_text(reply)
        if error_text is not None:
            logger.warning(f'{subject} returned a tool error: {error_text!r}')
            return MemoryFailure.MALFORMED
        return '\n'.join(tool_text_blocks(reply))

    def _note_outage(self, outage: bool) -> None:
        """Track the outage streak; log ERROR whenever it reaches a multiple of the threshold."""
        if not outage:
            self._outage_streak = 0
            return
        self._outage_streak += 1
        if self._outage_streak % MEMORY_OUTAGE_STREAK_THRESHOLD == 0:
            logger.error(
                f'MemoryRecall: {self._outage_streak} consecutive dispatches recalled '
                f'no memory at all for {self._project_id!r} — the memory service looks '
                'unavailable, and every briefing since the streak began was assembled '
                'without it'
            )
