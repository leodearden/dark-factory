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
    # The 'corr_' prefix is a LEGIBILITY aid, not an enforced guard — nothing
    # validates or rejects it. It makes the value self-describing in logs and
    # in error messages: a caller who copies it into delete_episode reads
    # `corr_...` back in the resulting NodeNotFoundError and can see at a
    # glance that it passed a correlation id where an episode uuid was needed.
    # The failure is loud either way — remove_episode's first statement is
    # EpisodicNode.get_by_uuid (backends/graphiti_client.py::GraphitiBackend.remove_episode),
    # which raises for ANY unresolvable uuid, prefixed or bare — so the bare
    # uuid4 this field used to return failed identically, just without saying
    # why. The prefix changes an opaque failure into a self-explaining one; it
    # does not change a silent one into a loud one.
    #
    # The field NAME stays `episode_id`: the key is on the wire in the
    # add_episode MCP tool's returned dict (server/tools.py returns
    # `result.model_dump()`), and in the write-journal `result_summary` shape
    # written by services/write_journal.py::WriteJournal.log_write_op and read
    # by reconciliation/journal.py, so renaming it is a breaking change. The
    # description below is a Field(...) rather than a comment for the source
    # reader's benefit only: that tool is annotated `-> dict[str, Any]`, so
    # this model's JSON schema is NOT published to MCP clients. The
    # caller-visible channel for the demotion is the `message` string
    # add_episode returns (services/memory_service.py::MemoryService.add_episode).
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
