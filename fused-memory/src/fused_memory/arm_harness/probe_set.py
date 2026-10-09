"""A known item is a reference Episodic node cited by a RELATES_TO edge, queried by its first K words (K = the median real transcript query's word count); rationale in plans/local-memory-models-eval-embedding-report.md."""

import math
import re
import statistics
from collections.abc import Sequence, Set
from pathlib import Path
from typing import Final, Literal

from graphiti_core.search import search_utils
from pydantic import Field
from shared.memory_eval_metrics import canonical_json_text

from fused_memory.arm_harness import corpus
from fused_memory.arm_harness.arm_spec import ContentSha
from fused_memory.arm_harness.frozen_model import FrozenModel
from fused_memory.backends.falkor_fulltext import build_query

SCHEMA_VERSION: Final = 1
TRANSCRIPT_QUERY_CAP = 500
FULLTEXT_TERM_CUTOFF = search_utils.MAX_QUERY_LENGTH

_ONE_SEARCH_GROUP = ('evalmem_probe_set',)
"""A known-item search passes exactly one group id, the arm's scratch graph."""

_ALNUM_RUN = re.compile(r'[^\W_]+')


class ProbeSetError(ValueError):
    """The probe set's inputs cannot yield a probe that measures what it claims to."""


class FrozenReference(FrozenModel):
    graph: str
    node_count: int = Field(ge=0)
    edge_count: int = Field(ge=0)
    topology_hash: ContentSha


class KnownItem(FrozenModel):
    episode_uuid: str
    query: str


class Mem0KnownItem(FrozenModel):
    topic: str
    phrasing: str
    held_out: bool
    canonical_content_hash: str
    canonical_last_known_id: str | None


class TranscriptPin(FrozenModel):
    path: str
    sha256: ContentSha
    queries: tuple[str, ...] = Field(max_length=TRANSCRIPT_QUERY_CAP)


class Mem0SnapshotPin(FrozenModel):
    source: str
    sha256: ContentSha
    point_count: int = Field(ge=0)
    excluded_empty: int = Field(ge=0)


class ProbeSet(FrozenModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    corpus_sha: ContentSha
    """δ's corpus, which built the reference graph: provenance, not this probe's identity."""
    reference: FrozenReference
    query_words: int = Field(ge=1)
    known_items: tuple[KnownItem, ...]
    uncited_episodes: int = Field(ge=0)
    transcript: TranscriptPin
    mem0_snapshot: Mem0SnapshotPin
    mem0_known_items: tuple[Mem0KnownItem, ...]


def known_item_query(content: str, words: int) -> str:
    return ' '.join(content.split()[:words])


def derive_query_words(transcript_queries: Sequence[str]) -> int:
    if not transcript_queries:
        raise ProbeSetError('no transcript query to take a median word count from')
    median = statistics.median([len(query.split()) for query in transcript_queries])
    words = math.floor(median + 0.5)
    if words < 1:
        raise ProbeSetError(f'the median transcript query has {words} words; a query needs one')
    if not _bm25_leg_survives(words):
        raise ProbeSetError(
            f'a {words}-word query can empty the fulltext query (cutoff {FULLTEXT_TERM_CUTOFF}), '
            'which collapses with-indices into embedding-only'
        )
    return words


def _bm25_leg_survives(term_count: int) -> bool:
    terms = ' '.join(f'term{index}' for index in range(term_count))
    return build_query(terms, list(_ONE_SEARCH_GROUP), FULLTEXT_TERM_CUTOFF) != ''


def build_probe_set(
    *,
    corpus_sha: str,
    reference: FrozenReference,
    episodes: Sequence[tuple[str, str]],
    cited_episode_uuids: Set[str],
    control_ok_episode_uuids: Set[str],
    transcript: TranscriptPin,
    mem0_snapshot: Mem0SnapshotPin,
    mem0_known_items: Sequence[Mem0KnownItem],
) -> ProbeSet:
    content_by_uuid = dict(episodes)
    _require_control_episodes(set(content_by_uuid), control_ok_episode_uuids)
    _require_cited_episodes(cited_episode_uuids, set(content_by_uuid))
    words = derive_query_words(transcript.queries)
    known_items = tuple(
        KnownItem(episode_uuid=uuid, query=known_item_query(content_by_uuid[uuid], words))
        for uuid in sorted(cited_episode_uuids)
    )
    for item in known_items:
        _require_bm25_leg(item)
    return ProbeSet(
        corpus_sha=corpus_sha,
        reference=reference,
        query_words=words,
        known_items=known_items,
        uncited_episodes=len(content_by_uuid) - len(known_items),
        transcript=transcript,
        mem0_snapshot=mem0_snapshot,
        mem0_known_items=tuple(
            sorted(mem0_known_items, key=lambda item: (item.topic, item.phrasing))
        ),
    )


def _require_control_episodes(reference: Set[str], control_ok: Set[str]) -> None:
    if reference != control_ok:
        raise ProbeSetError(
            "the reference's Episodic uuids differ from control A's ok replays: "
            f'only in the reference {sorted(reference - control_ok)}, '
            f"only in control A's replays {sorted(control_ok - reference)}"
        )


def _require_cited_episodes(cited: Set[str], reference: Set[str]) -> None:
    stray = sorted(cited - reference)
    if stray:
        raise ProbeSetError(f'cited episode uuids that are no reference episode: {stray}')
    if not cited:
        raise ProbeSetError('no reference episode is cited by a RELATES_TO edge: no known item')


def _require_bm25_leg(item: KnownItem) -> None:
    """Every searchable term holds an alphanumeric run, so the run count bounds the terms."""
    runs = len(_ALNUM_RUN.findall(item.query))
    if not _bm25_leg_survives(runs):
        raise ProbeSetError(
            f'known item {item.episode_uuid}: its query splits into up to {runs} terms, '
            'enough to empty the fulltext query'
        )


def serialize_probe_set(probe_set: ProbeSet) -> str:
    return canonical_json_text(probe_set.model_dump(mode='json'))


def load_probe_set(path: Path) -> ProbeSet:
    return ProbeSet.model_validate_json(Path(path).read_text(encoding='utf-8'))


def probe_set_sha(data: bytes) -> str:
    """Every embedding arm's ``corpus_sha``: the probe set is that axis's corpus."""
    return corpus.corpus_sha(data)
