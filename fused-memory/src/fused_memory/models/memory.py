"""Response and result models for the fused memory system."""

from enum import StrEnum

from pydantic import BaseModel, Field

from fused_memory.models.enums import MemoryCategory, QueryType, SourceStore


class EpisodeStatus(StrEnum):
    """Processing status of an add_episode request."""

    queued = 'queued'
    processed = 'processed'
    error = 'error'


class ClassificationResult(BaseModel):
    """Result of write-path classification."""

    primary: MemoryCategory
    secondary: MemoryCategory | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ''


class MemoryResult(BaseModel):
    """Unified result returned from search across both stores."""

    id: str
    content: str
    category: MemoryCategory | None = None
    source_store: SourceStore
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    provenance: list[str] = Field(default_factory=list)
    temporal: dict | None = None  # {valid_at, invalid_at}
    entities: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    # Mem0 server-stamped write time, passed through verbatim from the Mem0 search result.
    # Typically an offset-aware ISO-8601 string (e.g. "2026-05-15T10:00:00+00:00"), but the
    # UTC offset is not guaranteed — Mem0 may stamp in any offset or return a non-UTC value.
    # None when the key is absent in the Mem0 response or for Graphiti-sourced results.
    created_at: str | None = None
    # COMPUTED RETRIEVAL STATE, not stored metadata: never round-trips to a store, and is
    # never read back off a payload. True means this result was PROMOTED into the returned
    # window by the topic pin (services/topic_anchor.py, task 3111) rather than by its own
    # rank, so `relevance_score` is not meaningful for it. Always present in model_dump()
    # so consumers can read it unconditionally.
    topic_anchored: bool = False


class AddEpisodeResponse(BaseModel):
    """Response from add_episode."""

    # A CORRELATION id for the queued write — explicitly NOT a Graphiti episode
    # uuid (task 3561). add_episode returns synchronously at ENQUEUE time, so no
    # node exists yet; the real uuid is minted by graphiti_core when the queued
    # write executes, and _execute_graphiti_write logs the two together.
    #
    # The 'corr_' prefix is the runtime enforcement of that: a caller who copies
    # this value into delete_episode fails self-describingly rather than getting
    # a silent no-op against a nonexistent node, which is what happened while
    # this field returned a bare uuid4.
    #
    # The field NAME stays `episode_id`: it is part of the published MCP
    # response schema and of the write-journal `result_summary` shape
    # (services/journal.py), so renaming it is a breaking change. The
    # description below is a Field(...) rather than a comment so the demotion
    # travels with the MCP schema instead of living only in source.
    episode_id: str | None = Field(
        default=None,
        description=(
            'Correlation id for the queued write (prefixed "corr_"), NOT a '
            'Graphiti episode uuid. The episode node does not exist until the '
            'queued write executes, so this value is not resolvable via '
            'delete_episode or any other uuid-keyed operation.'
        ),
    )
    status: EpisodeStatus
    message: str = ''


class AddMemoryResponse(BaseModel):
    """Response from add_memory."""

    memory_ids: list[str] = Field(default_factory=list)
    stores_written: list[SourceStore] = Field(default_factory=list)
    category: MemoryCategory | None = None
    message: str = ''


class ReadRouteResult(BaseModel):
    """Result of read-path query routing."""

    query_type: QueryType
    stores: list[SourceStore]
    primary_store: SourceStore
