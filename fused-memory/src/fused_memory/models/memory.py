"""Response and result models for the fused memory system."""

from pydantic import BaseModel, Field

from fused_memory.models.enums import MemoryCategory, QueryType, SourceStore


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


class AddEpisodeResponse(BaseModel):
    """Response from add_episode."""

    episode_id: str | None = None
    status: str  # 'queued' | 'processed' | 'error'
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
