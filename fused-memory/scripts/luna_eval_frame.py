#!/usr/bin/env python3
"""Draw the frozen, read-only sample frame and fixtures that task 9945's gpt-4o-mini vs gpt-6-luna eval runs over.

The extraction frame stratifies every non-decision add_memory write by its
category, with the 4716 audit's own hash rule and window semantics, so
replay_overreach_extraction.py replays and scores it unchanged. Graphs are read
over GRAPH.RO_QUERY only, through fused_memory/backends/falkor_read_only.py's
ReadOnlyGraphReader; episode bodies are read only for sampled episodes, and only
the worksheet carries them, never the frame.
"""
from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from fused_memory.backends.falkor_read_only import PagedQuery, ReadOnlyGraphReader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_ruling_overreach import (
    Episode,
    SampleDefinition,
    in_window,
    parse_source_description,
    sample_key,
)

# --------------------------------------------------------------------------- #
# The extraction frame
# --------------------------------------------------------------------------- #

TEMPORAL, RELATIONS, OTHER_CATEGORIES = 'temporal_facts', 'entities_and_relations', 'other_categories'
EXTRACTION_STRATA: tuple[str, ...] = (TEMPORAL, RELATIONS, OTHER_CATEGORIES)

EXCLUDED_CATEGORY = 'decisions_and_rationale'
"""The ruling class the 4716 audit and levers O1/O6 measured; this frame is every other write."""

EPISODE_METADATA = PagedQuery(
    match='MATCH (e:Episodic)', row='e', returns=('e.uuid', 'e.created_at', 'e.source_description'),
)
"""Every episode, paged and census-proven, without its body."""

EPISODE_CONTENT_CYPHER = 'MATCH (e:Episodic) WHERE e.uuid IN $uuids RETURN e.uuid, e.content'
"""The bodies of the sampled episodes only: at most cap x strata rows per graph."""


class FrameReadIncomplete(RuntimeError):
    """A graph read that cannot prove it saw every episode, so no frame is drawn from it."""


@dataclass(frozen=True)
class ExtractionFrame:
    frame: dict[str, Any]
    """frame.json: the keys replay_overreach_extraction.py::Baseline.from_report reads, and no body."""
    worksheet: tuple[dict[str, Any], ...]
    """One row per sampled episode, body included, in replay_overreach_extraction's worksheet shape."""


def extraction_stratum(category: str | None) -> str | None:
    """The frame stratum of a write's category; None for the excluded decision class."""
    if category == EXCLUDED_CATEGORY:
        return None
    return category if category in (TEMPORAL, RELATIONS) else OTHER_CATEGORIES


def _metadata_episode(graph: str, row: list[Any]) -> Episode:
    uuid, created_at, source_description = (list(row) + [None] * 3)[:3]
    return Episode(
        graph=graph, uuid=str(uuid), created_at='' if created_at is None else str(created_at),
        source=parse_source_description(None if source_description is None else str(source_description)),
        content='',
    )


async def _window_cells(
    graph: str, reader: ReadOnlyGraphReader, definition: SampleDefinition,
) -> dict[str, list[Episode]]:
    """stratum -> the graph's in-window episodes of that stratum, in sample_key order."""
    read = await reader.read_paged(EPISODE_METADATA)
    if not read.complete:
        raise FrameReadIncomplete(f'{graph}: {read.reason}')
    cells: dict[str, list[Episode]] = {stratum: [] for stratum in definition.strata}
    for row in read.rows:
        episode = _metadata_episode(graph, row)
        stratum = extraction_stratum(episode.source.category)
        if stratum is not None and in_window(episode, definition.window_start, definition.window_end):
            cells[stratum].append(episode)
    for episodes in cells.values():
        episodes.sort(key=lambda episode: sample_key(graph, episode.uuid))
    return cells


async def _contents(graph: str, reader: ReadOnlyGraphReader, uuids: list[str]) -> dict[str, str]:
    """Each sampled episode's body; one the graph no longer holds refuses the frame."""
    if not uuids:
        return {}
    rows = await reader.read_unpaged(EPISODE_CONTENT_CYPHER, {'uuids': uuids})
    contents = {str(uuid): '' if content is None else str(content) for uuid, content in rows}
    if missing := sorted(set(uuids) - contents.keys()):
        raise FrameReadIncomplete(f'{graph}: sampled episodes vanished before their bodies were read: {missing}')
    return contents


def _worksheet_row(episode: Episode, stratum: str, content: str) -> dict[str, Any]:
    return {
        'graph': episode.graph, 'uuid': episode.uuid, 'stratum': stratum,
        'category': episode.source.category, 'created_at': episode.created_at, 'content': content,
    }


async def extraction_frame(
    readers: Mapping[str, ReadOnlyGraphReader], definition: SampleDefinition,
) -> ExtractionFrame:
    """The first ``definition.cap`` in-window episodes of each (graph, stratum) cell by sample_key.

    *definition* must carry EXTRACTION_STRATA: the frame's verdict file carries
    it, and replay_overreach_extraction's score validates against it.
    """
    if tuple(definition.strata) != EXTRACTION_STRATA:
        raise ValueError(f'an extraction frame has strata {EXTRACTION_STRATA}, not {definition.strata}')
    population: dict[str, dict[str, dict[str, int]]] = {}
    sizes: dict[str, dict[str, dict[str, int]]] = {}
    worksheet: list[dict[str, Any]] = []
    for graph, reader in readers.items():
        cells = await _window_cells(graph, reader, definition)
        sampled = {stratum: episodes[: definition.cap] for stratum, episodes in cells.items()}
        contents = await _contents(graph, reader, [e.uuid for episodes in sampled.values() for e in episodes])
        population[graph] = {stratum: {'window_population': len(cells[stratum])} for stratum in definition.strata}
        sizes[graph] = {stratum: {'episodes': len(sampled[stratum])} for stratum in definition.strata}
        worksheet.extend(
            _worksheet_row(episode, stratum, contents[episode.uuid])
            for stratum in definition.strata for episode in sampled[stratum]
        )
    frame = {
        'graphs': list(readers),
        'sample': {'definition': definition.to_dict(), 'sizes': sizes, 'window_days': definition.window_days},
        'strata': population,
    }
    return ExtractionFrame(frame=frame, worksheet=tuple(worksheet))
