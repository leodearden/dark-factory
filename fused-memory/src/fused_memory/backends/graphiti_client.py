"""Thin async wrapper around the Graphiti client."""

import asyncio
import contextlib
import functools
import importlib.util
import inspect
import logging
import re
import time
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, NamedTuple, TypedDict, cast
from urllib.parse import urlparse

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.driver.driver import GraphDriver
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.edges import EntityEdge
from graphiti_core.embedder import OpenAIEmbedder
from graphiti_core.embedder.openai import OpenAIEmbedderConfig
from graphiti_core.errors import EdgeNotFoundError
from graphiti_core.errors import NodeNotFoundError as GraphitiCoreNodeNotFoundError
from graphiti_core.llm_client import OpenAIClient
from graphiti_core.llm_client.config import LLMConfig as GraphitiLLMConfig
from graphiti_core.nodes import EpisodeType, EpisodicNode

from fused_memory.config.schema import FusedMemoryConfig, OpenAIProviderConfig
from fused_memory.utils.async_utils import gather_or_raise
from fused_memory.utils.validation import canonicalize_project_id

logger = logging.getLogger(__name__)


_MIN_OPENAI_VERSION: tuple[int, ...] = (1, 91, 0)
_MIN_OPENAI_VERSION_STR = '1.91.0'


def _leading_version_tuple(version_str: str) -> tuple[int, ...]:
    """Parse the leading run of dotted integer components of a version string.

    E.g. ``'1.91.0'`` -> ``(1, 91, 0)``; ``'1.91.0rc1'`` -> ``(1, 91, 0)``.
    Returns an empty tuple when no leading numeric component can be found
    (e.g. ``'?'``) — callers should treat that as inconclusive rather than
    a failure, since it means the version string couldn't be parsed, not
    that the version is actually too low.
    """
    parts: list[int] = []
    for chunk in version_str.split('.'):
        digits = ''
        for char in chunk:
            if not char.isdigit():
                break
            digits += char
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def check_openai_responses_api() -> None:
    """Raise if the installed openai SDK lacks the Responses API surface.

    graphiti-core's OpenAIClient calls ``client.responses.create(...)``,
    which lazily resolves ``openai.resources.responses`` — a submodule
    introduced in openai 1.66.0 (graphiti-core 0.28.2 hard-requires
    openai>=1.91.0). An incompatible openai raises ``ModuleNotFoundError``
    deep inside Graphiti write-path LLM extraction, where the durable queue
    treats it as non-retriable and dead-letters silently after exhausting
    retries (task 2053). Fail fast at startup instead, with an actionable
    message.

    Two independent guards feed the same actionable error (review
    hardening, task 2053):

    - Module presence (``importlib.util.find_spec``): catches openai <1.66
      (module never existed) and any future openai release that
      renames/removes the surface. If resolving the submodule spec itself
      raises — e.g. a broken parent ``openai``/``openai.resources`` import —
      that is also treated as "missing" rather than letting a raw
      Import/ModuleNotFoundError escape and bury the actionable message.
    - Version floor (``openai.__version__ >= 1.91.0``): catches the narrow
      window [1.66.0, 1.91.0) where the submodule exists but graphiti-core's
      declared floor is still violated. Best-effort: an unparseable version
      string is treated as inconclusive, not a failure — module presence
      remains the authoritative guard.
    """
    try:
        module_present = importlib.util.find_spec('openai.resources.responses') is not None
    except (ImportError, ModuleNotFoundError):
        module_present = False

    import openai

    installed_version = getattr(openai, '__version__', '?')
    parsed_version = _leading_version_tuple(installed_version)
    version_too_old = bool(parsed_version) and parsed_version < _MIN_OPENAI_VERSION

    if module_present and not version_too_old:
        return

    if not module_present:
        reason = "is missing the module 'openai.resources.responses'"
    else:
        reason = (
            f"provides 'openai.resources.responses' but at version {installed_version}, "
            "below graphiti-core's required floor"
        )

    raise RuntimeError(
        f"Installed openai {installed_version} {reason}, which graphiti-core's "
        "OpenAIClient requires (client.responses.create). This module was "
        f"added in openai 1.66.0; graphiti-core 0.28.2 requires "
        f"openai>={_MIN_OPENAI_VERSION_STR}. Run `uv sync` in fused-memory "
        "and restart the service to install a compatible openai version. "
        "(task 2053)"
    )


def _canonicalize_group_args(func):
    """Canonicalize bound ``group_id``/``group_ids`` arguments at method entry.

    PRD seam S4 (task γ, plans/cross-graph-entity-leak-prd.md): every public
    GraphitiBackend method taking a project group_id/group_ids must
    canonicalize those ARGUMENTS via α's ``canonicalize_project_id`` before
    the method body runs, so the FalkorDB graph KEY (``_driver_for`` /
    ``_graph_for``), the node/edge ``group_id`` PROPERTY (graphiti_core's
    ``client.add_episode``), and any direct-Cypher ``$group_id`` FILTER
    always agree (RCA §4).

    Computes ``inspect.signature(func)`` once at decoration time. On each
    call, binds the actual args/kwargs and:

    - if a ``group_id`` argument is bound and is a ``str``, replaces it with
      ``canonicalize_project_id(group_id)``;
    - if a ``group_ids`` argument is bound and is not ``None``, replaces it
      with a list where each ``str`` element is independently canonicalized
      via ``canonicalize_project_id`` and any non-``str`` element passes
      through untouched — mirroring the ``group_id`` scalar guard so a
      malformed element (e.g. an accidental ``None``) doesn't crash inside
      ``canonicalize_project_id`` itself. ``group_ids=None`` — meaning
      global/no-scope — passes through untouched.

    Method BODIES are never touched by this decorator — it only normalizes
    the two argument names above, before delegating to *func* unchanged.
    Handles both async callables (every DB-facing method) and the one sync
    accessor, ``_identity_lock_for``, via ``inspect.iscoroutinefunction``.
    """
    sig = inspect.signature(func)

    def _canonicalize_bound(bound: inspect.BoundArguments) -> None:
        if 'group_id' in bound.arguments:
            value = bound.arguments['group_id']
            if isinstance(value, str):
                bound.arguments['group_id'] = canonicalize_project_id(value)
        if 'group_ids' in bound.arguments:
            value = bound.arguments['group_ids']
            if value is not None:
                bound.arguments['group_ids'] = [
                    canonicalize_project_id(g) if isinstance(g, str) else g for g in value
                ]

    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            _canonicalize_bound(bound)
            return await func(*bound.args, **bound.kwargs)

        return async_wrapper

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        _canonicalize_bound(bound)
        return func(*bound.args, **bound.kwargs)

    return sync_wrapper


class EdgeDict(TypedDict):
    """Normalised edge dict returned by GraphitiBackend._edge_dict.

    Consumed by get_valid_edges_for_node, get_all_valid_edges,
    _canonical_facts, and rebuild_entity_from_edges.
    """

    uuid: str
    fact: str
    name: str


class StaleSummaryResult(NamedTuple):
    """Structured return type for detect_stale_with_edges.

    Use named attribute access — the canonical idiom after Task 438/465:

    - ``result.stale`` — list of stale entity dicts (each has uuid, name, summary, etc.)
    - ``result.all_edges`` — dict[uuid, list[EdgeDict]] of valid edges for every scanned entity
    - ``result.total_count`` — total number of entity nodes scanned

    Because StaleSummaryResult is a NamedTuple (a tuple subclass), positional
    unpacking still works at runtime, but named access is the preferred idiom
    across the codebase.
    """

    stale: list[dict]
    all_edges: dict[str, list[EdgeDict]]
    total_count: int


class NodeNotFoundError(Exception):
    """Raised when a node UUID is not found in FalkorDB."""


class ActiveEdgesError(Exception):
    """Raised when an entity node still has valid active edges and force=False.

    Pass ``force=True`` to ``delete_entity`` to override the guard and delete
    the node despite its active edges.
    """


class AmbiguousEntityError(Exception):
    """Raised when multiple entity nodes share the same name.

    The error message includes all matching UUIDs so the caller can
    disambiguate and call refresh_entity_summary with a specific UUID.
    """


def _as_sortable_utc(created_at: datetime | None) -> datetime:
    """Coerce an episode's created_at to a tz-aware UTC datetime for sorting.

    Naive (tzinfo-less) datetimes are assumed to already be UTC rather than
    raising on naive-vs-aware comparison; None sorts as the minimum possible
    value (oldest/unknown), landing last in a descending sort.
    """
    if created_at is None:
        return datetime.min.replace(tzinfo=UTC)
    if created_at.tzinfo is None:
        return created_at.replace(tzinfo=UTC)
    return created_at.astimezone(UTC)


def _normalize_fact_for_grouping(fact: str | None) -> str:
    """Normalize an edge fact for duplicate-grouping comparison.

    Mirrors ``MemoryService._normalize_fact`` (lowercase + collapse
    whitespace) and graphiti-core's ``_normalize_string_exact`` — kept as a
    small local copy rather than an import because the backend layer must
    not depend on the services layer. A None/missing fact coerces to ''.
    """
    return re.sub(r'\s+', ' ', (fact or '').lower()).strip()


class _MultiTenantFalkorDriver(FalkorDriver):
    """FalkorDriver that suppresses auto-indexing.

    Upstream ``__init__`` schedules ``build_indices_and_constraints()``
    against ``_database`` as a fire-and-forget task.  In multi-tenant
    mode indices are built explicitly via ``_ensure_indices()`` — the
    fire-and-forget path is suppressed here to prevent redundant
    CREATE INDEX commands from saturating FalkorDB's single-threaded
    execution.

    ``clone()`` is overridden to return another ``_MultiTenantFalkorDriver``
    so cloned per-graph drivers also suppress auto-indexing.
    """

    async def build_indices_and_constraints(self, delete_existing=False):
        pass

    def clone(self, database: str) -> 'GraphDriver':
        if database == self._database:
            return self
        cloned = _MultiTenantFalkorDriver(falkor_db=self.client, database=database)
        return cloned


class GraphitiBackend:
    """Owns the Graphiti client lifecycle.

    FalkorDB is multi-tenant: each project's data lives in its own graph
    (named after the project_id / group_id).  The driver is cloned per-request
    so every operation targets the correct graph.
    """

    def __init__(self, config: FusedMemoryConfig):
        self.config = config
        self.client: Graphiti | None = None
        self._driver: FalkorDriver | None = None
        self._read_timeout: float = config.queue.backend_read_timeout_seconds
        self._write_timeout: float = config.queue.backend_write_timeout_seconds
        self._indexed_graphs: set[str] = set()
        self._cloned_drivers: dict[str, GraphDriver] = {}
        self._identity_locks: dict[str, asyncio.Lock] = {}
        self._llm_client = None
        self._embedder = None
        self._cross_encoder = None
        self._group_clients: dict[str, Graphiti] = {}

    # --- Per-request driver routing ---

    def _driver_for(self, group_id: str) -> GraphDriver:
        """Return a cached driver clone targeting the FalkorDB graph for *group_id*.

        Caches cloned drivers to avoid creating new connections per request.
        """
        cached = self._cloned_drivers.get(group_id)
        if cached is not None:
            return cached
        driver = self._require_driver()
        cloned = driver.clone(database=group_id)
        self._cloned_drivers[group_id] = cloned
        return cloned

    def _client_for(self, group_id: str) -> Graphiti:
        """Return a cached Graphiti client dedicated to *group_id*.

        Each client is built with ``graph_driver=self._driver_for(group_id)``,
        whose ``_database`` already equals *group_id*. Upstream
        ``Graphiti.add_episode`` only mutates ``self.driver``/``self.clients.driver``
        when ``group_id != self.driver._database`` (graphiti_core 0.28.2,
        graphiti.py:889-890) — since that condition is never true for a
        per-group client, the shared-driver mutation race that misroutes
        concurrent cross-group writes is structurally unreachable here.
        The llm_client/embedder/cross_encoder sub-clients are shared (hoisted
        in ``initialize()``) across every per-group client and the base
        ``self.client``.

        ``_group_clients`` is unbounded and never evicted, mirroring
        ``_cloned_drivers``/``_identity_locks`` above — safe because
        *group_id* is always a project_id, so cardinality is bounded by the
        (small, fixed) number of projects, not by request volume. If
        *group_id* ever becomes an ephemeral/per-session value, this cache
        would need a size bound or LRU eviction.

        Each per-group client is constructed with its own
        ``max_coroutines=self.config.queue.graphiti_max_coroutines``. This
        does NOT raise the aggregate OpenAI concurrency ceiling versus the
        prior single-shared-client design: graphiti_core's
        ``semaphore_gather`` (graphiti_core/helpers.py) builds a brand-new
        ``asyncio.Semaphore(max_coroutines)`` on every invocation rather than
        reusing one stored on the client, so ``max_coroutines`` was always a
        per-call fan-out bound (scoped to a single ``add_episode``/etc.
        invocation's internal LLM/embedding sub-tasks), never a budget
        shared across concurrent top-level calls — even under the old
        shared-client design, two concurrent ``add_episode`` calls already
        raced with two independent semaphores, not one. The real
        cross-group aggregate ceiling is, and remains, governed one level up
        by ``DurableWriteQueue``'s ``semaphore_limit`` (bounding how many
        top-level ``add_episode`` calls run concurrently) multiplied by
        ``graphiti_max_coroutines`` (each call's own internal fan-out) — by
        default 3 × 5 = 15 concurrent OpenAI calls at most, unchanged by
        this per-group cache.
        """
        cached = self._group_clients.get(group_id)
        if cached is not None:
            return cached
        client = Graphiti(
            graph_driver=self._driver_for(group_id),
            llm_client=self._llm_client,
            embedder=self._embedder,
            cross_encoder=self._cross_encoder,
            max_coroutines=self.config.queue.graphiti_max_coroutines,
        )
        self._group_clients[group_id] = client
        return client

    def _graph_for(self, group_id: str) -> Any:
        """Return the FalkorGraph object for *group_id* (for direct Cypher)."""
        driver = self._require_driver()
        return driver._get_graph(group_id)

    @_canonicalize_group_args
    def _identity_lock_for(self, group_id: str) -> asyncio.Lock:
        """Return the per-group_id write-time-identity lock, creating it lazily.

        Mirrors the DurableWriteQueue._group_locks idiom (durable_queue.py:136,
        259-260): one asyncio.Lock per group_id, created on first access and
        cached thereafter so repeated calls for the same group_id return the
        exact same Lock instance.

        This registry is SEPARATE from DurableWriteQueue._group_locks, which
        only guards _claim_next — not _process_item/add_episode — so with
        workers_per_group > 1 two add_episode calls for the same group can run
        concurrently and would race on entity-name resolution without their
        own lock.

        Callers (MemoryService) are expected to hold this lock across an
        add_episode + reconcile critical section (which includes any call to
        _resolve_or_create_entity), and must NEVER hold it across a Mem0
        write. Synchronous accessor — returns the Lock object itself, not a
        coroutine, so callers write ``async with backend._identity_lock_for(gid):``.
        """
        lock = self._identity_locks.get(group_id)
        if lock is None:
            lock = asyncio.Lock()
            self._identity_locks[group_id] = lock
        return lock

    def _require_driver(self) -> FalkorDriver:
        if self._driver is None:
            raise RuntimeError('GraphitiBackend not initialized — call initialize() first')
        return self._driver

    def _require_falkor_client(self) -> Any:
        """Return the FalkorDB client from the underlying driver."""
        driver = self._require_driver()
        return cast(Any, driver).client

    async def _ensure_indices(self, group_id: str) -> None:
        """Build indices on *group_id*'s graph if not already done this session."""
        if group_id in self._indexed_graphs:
            return
        driver = self._driver_for(group_id)
        await driver.build_indices_and_constraints()
        self._indexed_graphs.add(group_id)
        logger.debug('Ensured indices on graph %r', group_id)

    async def initialize(self) -> None:
        """Create FalkorDriver + Graphiti client from unified config."""
        cfg = self.config

        # --- LLM client ---
        llm_client = None
        if cfg.llm.provider == 'openai' and cfg.llm.providers.openai:
            api_key = cfg.llm.providers.openai.api_key
            if api_key:
                check_openai_responses_api()
                llm_config = GraphitiLLMConfig(
                    api_key=api_key,
                    model=cfg.llm.model,
                    small_model=cfg.llm.model,
                    temperature=cfg.llm.temperature or 0.0,
                    max_tokens=cfg.llm.max_tokens,
                )
                llm_client = OpenAIClient(config=llm_config)
                logger.info(f'Graphiti LLM: {cfg.llm.provider}/{cfg.llm.model}')
        elif cfg.llm.provider == 'anthropic' and cfg.llm.providers.anthropic:
            api_key = cfg.llm.providers.anthropic.api_key
            if api_key:
                try:
                    from graphiti_core.llm_client.anthropic_client import AnthropicClient

                    llm_config = GraphitiLLMConfig(
                        api_key=api_key,
                        model=cfg.llm.model,
                        temperature=cfg.llm.temperature or 0.0,
                        max_tokens=cfg.llm.max_tokens,
                    )
                    llm_client = AnthropicClient(config=llm_config)
                    logger.info(f'Graphiti LLM: {cfg.llm.provider}/{cfg.llm.model}')
                except ImportError:
                    logger.warning('Anthropic client not available for Graphiti')

        # --- Embedder ---
        embedder_client = None
        if cfg.embedder.provider == 'openai' and cfg.embedder.providers.openai:
            api_key = cfg.embedder.providers.openai.api_key
            if api_key:
                embedder_config = OpenAIEmbedderConfig(
                    api_key=api_key,
                    embedding_model=cfg.embedder.model,
                    base_url=cfg.embedder.providers.openai.api_url,
                    embedding_dim=cfg.embedder.dimensions,
                )
                embedder_client = OpenAIEmbedder(config=embedder_config)
                logger.info(f'Graphiti embedder: {cfg.embedder.provider}/{cfg.embedder.model}')

        # --- FalkorDB driver ---
        # The driver is created with a placeholder database.  Actual graph
        # selection happens per-request via _driver_for() / _graph_for().
        falkor_cfg = cfg.graphiti.falkordb
        if falkor_cfg.database is not None:
            logger.warning(
                'graphiti.falkordb.database=%r is ignored — graph name is '
                'derived from group_id at request time',
                falkor_cfg.database,
            )
        parsed = urlparse(falkor_cfg.uri)
        host = parsed.hostname or 'localhost'
        port = parsed.port or 6379

        self._driver = _MultiTenantFalkorDriver(
            host=host,
            port=port,
            password=falkor_cfg.password,
        )

        self._llm_client = llm_client
        self._embedder = embedder_client

        # --- Cross-encoder (reranker) ---
        # Shared across the base client and every per-group client (see
        # _client_for). Mirror the llm_client/embedder_client guard so a
        # configured api_key/base_url (e.g. a proxy endpoint) is honored
        # instead of silently falling back to env-based defaults. The
        # reranker always talks to the OpenAI API regardless of
        # cfg.llm.provider, so it sources credentials from whichever
        # provider block actually configures OpenAI: cfg.llm.providers.openai
        # is preferred (the common case — cfg.llm.provider defaults to
        # 'openai'), falling back to cfg.embedder.providers.openai (always
        # OpenAI — EmbedderConfig.provider has no other option) when the llm
        # block has no OpenAI api_key, e.g. cfg.llm.provider='anthropic' with
        # OpenAI only configured for embeddings/reranking.
        reranker_provider: OpenAIProviderConfig | None = None
        if cfg.llm.providers.openai and cfg.llm.providers.openai.api_key:
            reranker_provider = cfg.llm.providers.openai
        elif cfg.embedder.providers.openai and cfg.embedder.providers.openai.api_key:
            reranker_provider = cfg.embedder.providers.openai

        reranker_config: GraphitiLLMConfig | None = None
        if reranker_provider:
            reranker_config = GraphitiLLMConfig(
                api_key=reranker_provider.api_key,
                base_url=reranker_provider.api_url,
            )
        self._cross_encoder = OpenAIRerankerClient(config=reranker_config)

        self.client = Graphiti(
            graph_driver=self._driver,
            llm_client=self._llm_client,
            embedder=self._embedder,
            cross_encoder=self._cross_encoder,
            max_coroutines=cfg.queue.graphiti_max_coroutines,
        )

        # Build indices on all existing project graphs (lazy set avoids repeats).
        try:
            existing = await self._require_falkor_client().list_graphs()
            for graph_name in existing:
                if graph_name != 'default_db' and not graph_name.endswith('_db'):
                    await self._ensure_indices(graph_name)
        except Exception:
            logger.warning('Could not enumerate existing graphs for index setup', exc_info=True)

        # Startup identity-integrity sweep (task 2210, W6-ε): dup-NODE alarm +
        # one-shot dup-uuid-EDGE repair, per graph. A scan failure must never
        # break backend startup — this is a safety net, not a startup gate.
        try:
            await self._run_startup_identity_scan()
        except Exception:
            logger.warning('Startup identity-integrity scan failed', exc_info=True)

        logger.info(f'GraphitiBackend initialized (FalkorDB {host}:{port})')

    def _require_client(self) -> Graphiti:
        if self.client is None:
            raise RuntimeError('GraphitiBackend not initialized — call initialize() first')
        return self.client

    @_canonicalize_group_args
    async def add_episode(
        self,
        name: str,
        content: str,
        source: EpisodeType = EpisodeType.text,
        group_id: str = 'main',
        source_description: str = '',
        reference_time: datetime | None = None,
        entity_types: dict | None = None,
        uuid: str | None = None,
        temporal_context: str | None = None,
    ) -> Any:
        """Add an episode to Graphiti and return the result."""
        client = self._client_for(group_id)
        ref_time = reference_time or datetime.now(UTC)
        if temporal_context is not None:
            source_description = f'[temporal:{temporal_context}] {source_description}'
        return await asyncio.wait_for(
            client.add_episode(
                name=name,
                episode_body=content,
                source=source,
                group_id=group_id,
                source_description=source_description,
                reference_time=ref_time,
                entity_types=entity_types,
                uuid=uuid,
            ),
            timeout=self._write_timeout,
        )

    @_canonicalize_group_args
    async def search(
        self,
        query: str,
        group_ids: list[str] | None = None,
        num_results: int = 10,
        center_node_uuid: str | None = None,
    ) -> list[Any]:
        """Search for entity edges (facts)."""
        client = self._require_client()
        gids = group_ids or []
        driver = self._driver_for(gids[0]) if gids else None
        try:
            return await asyncio.wait_for(
                client.search(
                    query=query,
                    group_ids=gids,
                    num_results=num_results,
                    center_node_uuid=center_node_uuid,
                    driver=driver,
                ),
                timeout=self._read_timeout,
            )
        except TimeoutError:
            logger.warning(f'Graphiti search timed out after {self._read_timeout}s')
            return []

    @_canonicalize_group_args
    async def search_nodes(
        self,
        query: str,
        group_ids: list[str] | None = None,
        max_nodes: int = 10,
    ) -> list[Any]:
        """Search for entity nodes."""
        client = self._require_client()
        from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF

        gids = group_ids or []
        driver = self._driver_for(gids[0]) if gids else None
        try:
            results = await asyncio.wait_for(
                client.search_(
                    query=query,
                    config=NODE_HYBRID_SEARCH_RRF,
                    group_ids=gids,
                    driver=driver,
                ),
                timeout=self._read_timeout,
            )
            return (results.nodes or [])[:max_nodes]
        except TimeoutError:
            logger.warning(f'Graphiti search_nodes timed out after {self._read_timeout}s')
            return []

    @_canonicalize_group_args
    async def retrieve_episodes(
        self,
        group_ids: list[str],
        last_n: int = 10,
        reference_time: datetime | None = None,
    ) -> list[Any]:
        """Retrieve recent episodes by group, ordered by created_at (most recent first) and truncated to last_n.

        EpisodicNode.get_by_group_ids truncates via ``ORDER BY uuid DESC LIMIT``,
        which is unrelated to recency, so we fetch the group's full episode set
        (limit=None) and sort/truncate by created_at ourselves.
        """
        driver = self._driver_for(group_ids[0]) if group_ids else self._require_driver()
        try:
            # Tradeoff: limit=None fetches the group's ENTIRE episode set on every
            # call (no Cypher LIMIT), then we sort/truncate in Python. This is what
            # makes the created_at ordering correct given that get_by_group_ids'
            # own ORDER BY uuid DESC LIMIT truncates on the wrong key before we'd
            # ever see the data. Acceptable today because episode reads are a cold
            # path and per-project episode counts are bounded (reconciliation GC;
            # last_n is separately capped at 1000 in tools.py). If per-group episode
            # volume grows large, revisit with a created_at-indexed Cypher query
            # (``ORDER BY e.created_at DESC LIMIT $limit``) to push the bound into
            # the DB instead of transferring+sorting the full set here.
            episodes = await asyncio.wait_for(
                EpisodicNode.get_by_group_ids(
                    driver, group_ids, limit=None
                ),
                timeout=self._read_timeout,
            )
            episodes = sorted(
                episodes or [],
                # Secondary key (uuid) makes this a total order: get_by_group_ids
                # truncates via ORDER BY uuid DESC, so without a tie-breaker,
                # episodes sharing created_at (batch/rapid co-ingestion, or
                # reduced-precision storage) fall back to that upstream order,
                # which is not guaranteed reproducible across executions and can
                # read as non-monotonic to a strict-descending observer.
                key=lambda ep: (
                    _as_sortable_utc(getattr(ep, 'created_at', None)),
                    getattr(ep, 'uuid', None) or '',
                ),
                reverse=True,
            )
            return episodes[:last_n]
        except TimeoutError:
            logger.warning(f'Graphiti retrieve_episodes timed out after {self._read_timeout}s')
            return []

    @_canonicalize_group_args
    async def get_episode_by_uuid(self, episode_uuid: str, *, group_id: str) -> EpisodicNode | None:
        """Fetch an episode node by UUID.

        Mirrors remove_episode's driver-selection + EpisodicNode.get_by_uuid
        call, but is fail-safe rather than propagating: a missing episode
        (graphiti_core's NodeNotFoundError) or a timeout (matching
        retrieve_episodes' timeout handling) both return None instead of
        raising, since callers (e.g. the reconciliation promotion-time
        batch-plan gate, task 2033) treat "can't determine content" as a
        graceful fallback rather than a hard failure.
        """
        driver = self._driver_for(group_id)
        try:
            return await asyncio.wait_for(
                EpisodicNode.get_by_uuid(driver, episode_uuid),
                timeout=self._read_timeout,
            )
        except GraphitiCoreNodeNotFoundError:
            return None
        except TimeoutError:
            logger.warning(f'Graphiti get_episode_by_uuid timed out after {self._read_timeout}s')
            return None

    @_canonicalize_group_args
    async def remove_episode(self, episode_uuid: str, *, group_id: str) -> None:
        """Delete an episode by UUID."""
        driver = self._driver_for(group_id)
        node = await EpisodicNode.get_by_uuid(driver, episode_uuid)
        await asyncio.wait_for(
            node.delete(driver),
            timeout=self._write_timeout,
        )

    @_canonicalize_group_args
    async def remove_edge(self, edge_uuid: str, *, group_id: str) -> None:
        """Delete an entity edge (fact) by UUID. Idempotent — missing edges are ignored."""
        driver = self._driver_for(group_id)
        try:
            edge = await EntityEdge.get_by_uuid(driver, edge_uuid)
        except EdgeNotFoundError:
            logger.info(f'Edge {edge_uuid} not found (already deleted or episode-cascaded)')
            return
        await asyncio.wait_for(
            edge.delete(driver),
            timeout=self._write_timeout,
        )

    @_canonicalize_group_args
    async def update_edge(
        self, edge_uuid: str, fact: str | None = None, *, group_id: str,
        invalid_at: datetime | None = None,
        clear_invalid_at: bool = False,
    ) -> dict[str, Any]:
        """Update an existing edge's fact text and/or invalidate it.

        At least one of ``fact``, ``invalid_at``, or ``clear_invalid_at`` must
        be provided. When ``fact`` is set, the edge's fact text is replaced and
        its embedding is regenerated. When ``invalid_at`` is set, the edge is
        marked superseded as of that timestamp (no re-embedding needed). Both
        may be combined.

        When ``clear_invalid_at=True``, the edge's ``invalid_at`` field is
        reset to ``None``, restoring it to an active (non-superseded) state.
        This takes precedence over ``invalid_at`` if both are supplied.
        Compatible with ``fact`` (update text and un-supersede in one call).
        Because graphiti's FalkorDB map-based ``edge.save()`` (``MERGE ... SET
        e = $edge_data``) does not reliably clear a null-valued property, the
        clear is force-persisted by an explicit direct-Cypher
        ``SET e.invalid_at = NULL`` write, issued after ``edge.save()`` and
        before the endpoint summary refresh below (so the restored edge is
        counted valid by that refresh's ``WHERE e.invalid_at IS NULL`` filter).

        After saving, both source and target entity node summaries are rebuilt
        from their current valid edges so they stay consistent.
        """
        if fact is None and invalid_at is None and not clear_invalid_at:
            raise ValueError('update_edge requires fact, invalid_at, or clear_invalid_at to be set')
        driver = self._driver_for(group_id)
        edge = await EntityEdge.get_by_uuid(driver, edge_uuid)
        if fact is not None:
            edge.fact = fact
            embedder = self._require_client().embedder
            await edge.generate_embedding(embedder)
        if invalid_at is not None:
            edge.invalid_at = invalid_at
        if clear_invalid_at:
            edge.invalid_at = None
        await asyncio.wait_for(edge.save(driver), timeout=self._write_timeout)

        if clear_invalid_at:
            # edge.save()'s map-based SET does not reliably clear a
            # null-valued property on FalkorDB — force it deterministically.
            graph = self._graph_for(group_id)
            await asyncio.wait_for(
                graph.query(
                    'MATCH ()-[e:RELATES_TO {uuid: $uuid}]->() '
                    'SET e.invalid_at = NULL',
                    {'uuid': edge_uuid},
                ),
                timeout=self._write_timeout,
            )

        # Deterministically refresh both endpoint entity summaries so they
        # reflect the updated fact text (no LLM — just fact concatenation).
        refreshed: list[str] = []
        for node_uuid in (edge.source_node_uuid, edge.target_node_uuid):
            try:
                await self.refresh_entity_summary(node_uuid, group_id=group_id)
                refreshed.append(node_uuid)
            except Exception as exc:
                logger.warning(
                    'update_edge: failed to refresh summary for node %s: %s',
                    node_uuid, exc,
                )

        return {
            'uuid': edge.uuid,
            'fact': edge.fact,
            'refreshed_nodes': refreshed,
        }

    @_canonicalize_group_args
    async def build_communities(self, group_ids: list[str] | None = None) -> None:
        """Build community summaries.

        FalkorDB's multi-tenant model gives each group_id its own physical
        graph, reachable only through that group's own driver
        (``_driver_for``) — a single upstream call can target just one
        driver/graph. Forwarding the full *group_ids* list through against
        only the first group's driver would silently scope (or entirely
        skip) communities for the remaining groups, so when more than one
        group_id is requested each is built via its own call against its
        own driver. ``group_ids=None``/empty preserves the pre-existing
        whole-graph fallback (no driver override — upstream falls back to
        the shared client's current driver).
        """
        client = self._require_client()
        if not group_ids:
            await asyncio.wait_for(
                client.build_communities(group_ids=group_ids, driver=None),
                timeout=self._write_timeout,
            )
            return
        for group_id in group_ids:
            await asyncio.wait_for(
                client.build_communities(
                    group_ids=[group_id], driver=self._driver_for(group_id)
                ),
                timeout=self._write_timeout,
            )

    @_canonicalize_group_args
    async def query_stale_node_embeddings(
        self, expected_dim: int, *, group_id: str
    ) -> list[tuple[str, str, int]]:
        """Return (uuid, name, dim) for Entity nodes whose embedding dim != expected_dim.

        FalkorDB's ``size()`` does not work on Vectorf32 properties, so we
        return all nodes with embeddings and filter client-side by parsing the
        raw vector text representation (``<v1, v2, ...>``).
        """
        graph = self._graph_for(group_id)
        cypher = (
            'MATCH (n:Entity) '
            'WHERE n.name_embedding IS NOT NULL '
            'RETURN n.uuid, n.name, n.name_embedding'
        )
        result = await graph.ro_query(cypher)
        stale: list[tuple[str, str, int]] = []
        for row in result.result_set or []:
            raw = row[2]
            if isinstance(raw, bytes):
                raw = raw.decode('utf-8', errors='replace')
            dim = len(str(raw).strip('<>').split(', '))
            if dim != expected_dim:
                stale.append((row[0], row[1], dim))
        return stale

    @_canonicalize_group_args
    async def query_stale_edge_embeddings(
        self, expected_dim: int, *, group_id: str
    ) -> list[tuple[str, str, int]]:
        """Return (uuid, name, dim) for RELATES_TO edges whose embedding dim != expected_dim.

        See ``query_stale_node_embeddings`` for why client-side filtering is needed.
        """
        graph = self._graph_for(group_id)
        cypher = (
            'MATCH (n)-[e:RELATES_TO]->(m) '
            'WHERE e.fact_embedding IS NOT NULL '
            'RETURN e.uuid, e.name, e.fact_embedding'
        )
        result = await graph.ro_query(cypher)
        stale: list[tuple[str, str, int]] = []
        for row in result.result_set or []:
            raw = row[2]
            if isinstance(raw, bytes):
                raw = raw.decode('utf-8', errors='replace')
            dim = len(str(raw).strip('<>').split(', '))
            if dim != expected_dim:
                stale.append((row[0], row[1], dim))
        return stale

    @_canonicalize_group_args
    async def query_edges_by_time_range(
        self, start: str, end: str, *, group_id: str
    ) -> list[dict]:
        """Return edges whose valid_at falls within [start, end] (ISO 8601 strings).

        Uses ro_query since no writes are performed.

        Args:
            start: ISO 8601 string for the lower bound (inclusive).
            end: ISO 8601 string for the upper bound (inclusive).
            group_id: Project graph to query.

        Returns:
            List of dicts with keys: uuid, fact, name, valid_at, invalid_at.
        """
        graph = self._graph_for(group_id)
        cypher = (
            'MATCH ()-[e:RELATES_TO]->() '
            'WHERE e.valid_at >= $start AND e.valid_at <= $end '
            'RETURN e.uuid, e.fact, e.name, e.valid_at, e.invalid_at'
        )
        result = await graph.ro_query(cypher, {'start': start, 'end': end})
        return [
            {
                'uuid': row[0],
                'fact': row[1],
                'name': row[2],
                'valid_at': row[3],
                'invalid_at': row[4],
            }
            for row in (result.result_set or [])
        ]

    @_canonicalize_group_args
    async def get_valid_edges_for_node(self, node_uuid: str, *, group_id: str) -> list[EdgeDict]:
        """Return all currently-valid RELATES_TO edges for an Entity node.

        Matches the node as either source or target (undirected) and filters
        edges where invalid_at IS NULL (i.e. not yet invalidated).

        Deduplicates in Python, keyed on e.uuid. RELATES_TO edge uuids are
        unique graph-wide (task 2207 W6-delta stopped minting copied uuids in
        redirect_node_edges; task 2210 W6-epsilon repaired legacy dup-uuid
        edges), so keying on e.uuid is equivalent to the prior element-identity
        dedup while collapsing the undirected self-loop double-match (an A->A
        edge matches the same uuid twice).

        Args:
            node_uuid: UUID of the Entity node.
            group_id: Project graph to query.

        Returns:
            List of dicts with keys: uuid, fact, name.
        """
        graph = self._graph_for(group_id)
        cypher = (
            'MATCH (n:Entity {uuid: $uuid})-[e:RELATES_TO]-() '
            'WHERE e.invalid_at IS NULL '
            'RETURN e.uuid, e.fact, e.name'
        )
        result = await graph.ro_query(cypher, {'uuid': node_uuid})
        seen: dict[str, EdgeDict] = {}
        edges: list[EdgeDict] = []
        for row in (result.result_set or []):
            edge_uuid = row[0]
            if edge_uuid in seen:
                # Diagnostic only: the uuid-keyed dedup premise (task 2213 W6-zeta)
                # is that RELATES_TO edge uuids are unique graph-wide post
                # tasks 2207/2210. If that invariant is ever violated, two
                # genuinely-distinct edges sharing a uuid would silently
                # collapse to the first-seen row. Surface it at debug level
                # rather than failing silently.
                dup = self._edge_dict(row[0], row[1], row[2])
                prior = seen[edge_uuid]
                if dup['fact'] != prior['fact'] or dup['name'] != prior['name']:
                    logger.debug(
                        'get_valid_edges_for_node: edge uuid %s seen again with '
                        'differing fact/name (kept fact=%r name=%r, saw fact=%r '
                        'name=%r) — uuid-uniqueness invariant may be violated; '
                        'keeping first-seen row',
                        edge_uuid, prior['fact'], prior['name'], dup['fact'], dup['name'],
                    )
                continue
            edge = self._edge_dict(row[0], row[1], row[2])
            seen[edge_uuid] = edge
            edges.append(edge)
        return edges

    @_canonicalize_group_args
    async def get_connected_entity_uuids(self, uuid: str, *, group_id: str) -> list[str]:
        """Return distinct UUID strings of entities connected to the given node via valid edges.

        Queries all RELATES_TO edges where invalid_at IS NULL and excludes the
        node itself (self-loops excluded via ``m.uuid <> $uuid``).

        Collects neighbours BEFORE deletion so the caller can refresh their
        summaries after the target node is removed (edges vanish with DETACH DELETE).

        Args:
            uuid: UUID of the Entity node whose neighbours to find.
            group_id: Project graph to query.

        Returns:
            List of distinct neighbour UUID strings. Empty list when isolated.

        Raises:
            RuntimeError: if the backend is not initialized.
        """
        graph = self._graph_for(group_id)
        cypher = (
            'MATCH (n:Entity {uuid: $uuid})-[e:RELATES_TO]-(m:Entity) '
            'WHERE e.invalid_at IS NULL AND m.uuid <> $uuid '
            'RETURN DISTINCT m.uuid'
        )
        result = await graph.ro_query(cypher, {'uuid': uuid})
        return [row[0] for row in (result.result_set or [])]

    @_canonicalize_group_args
    async def get_all_valid_edges(self, *, group_id: str) -> dict[str, list[EdgeDict]]:
        """Return all currently-valid RELATES_TO edges grouped by entity UUID.

        Bulk variant of get_valid_edges_for_node that issues a single Cypher query
        instead of O(N) per-entity round-trips.  The undirected MATCH pattern causes
        each directed edge to appear under both its source and target entity: for a
        directed A→B edge, traversal matches it from A's side (row: A.uuid, e.uuid)
        and from B's side (row: B.uuid, e.uuid) — two genuinely distinct rows because
        n.uuid differs.

        Deduplicates in Python, keyed on (n.uuid, e.uuid). RELATES_TO edge uuids
        are unique graph-wide (task 2207 W6-delta stopped minting copied uuids in
        redirect_node_edges; task 2210 W6-epsilon repaired legacy dup-uuid edges),
        so keying on the (entity, edge-uuid) pair is equivalent to the prior
        element-identity dedup: it preserves the intended double-attribution (each
        directed edge appears once under each endpoint entity, as distinct
        (n.uuid, e.uuid) pairs) and still collapses the undirected self-loop
        double-match (A→A edges, where both traversal directions yield the
        identical (n.uuid, e.uuid) pair).

        Uses ro_query since no writes are performed.

        Args:
            group_id: Project graph to query.

        Returns:
            Dict mapping entity UUID → list of edge dicts with keys: uuid, fact, name.
            fact and name default to empty string when the property is NULL.
            Each directed edge appears under both its source and target entity UUID
            (double-attribution from the undirected MATCH pattern).

        Note:
            Using a directed pattern (n:Entity)-[e:RELATES_TO]->() would give
            single-appearance semantics per edge if ever needed.
        """
        graph = self._graph_for(group_id)
        cypher = (
            'MATCH (n:Entity)-[e:RELATES_TO]-() '
            'WHERE e.invalid_at IS NULL '
            'RETURN n.uuid, e.uuid, e.fact, e.name'
        )
        result = await graph.ro_query(cypher)
        seen: dict[tuple[str, str], EdgeDict] = {}
        grouped: dict[str, list[EdgeDict]] = {}
        for row in (result.result_set or []):
            entity_uuid, edge_uuid = row[0], row[1]
            key = (entity_uuid, edge_uuid)
            if key in seen:
                # Diagnostic only: see get_valid_edges_for_node — the same
                # uuid-uniqueness invariant underpins this method's dedup key.
                dup = self._edge_dict(row[1], row[2], row[3])
                prior = seen[key]
                if dup['fact'] != prior['fact'] or dup['name'] != prior['name']:
                    logger.debug(
                        'get_all_valid_edges: (entity, edge) pair %s seen again '
                        'with differing fact/name (kept fact=%r name=%r, saw '
                        'fact=%r name=%r) — uuid-uniqueness invariant may be '
                        'violated; keeping first-seen row',
                        key, prior['fact'], prior['name'], dup['fact'], dup['name'],
                    )
                continue
            edge = self._edge_dict(row[1], row[2], row[3])
            seen[key] = edge
            grouped.setdefault(entity_uuid, []).append(edge)
        return grouped

    @_canonicalize_group_args
    async def bulk_remove_edges(self, uuids: list[str], *, group_id: str) -> int:
        """Delete RELATES_TO edges by UUID list. Returns count of actually matched edges.

        Uses a pre-count MATCH query before deletion to return the true number of
        edges that exist (and will be deleted), rather than the input list length.
        This is critical for irreversible operations where accuracy matters.

        Args:
            uuids: List of edge UUIDs to delete.
            group_id: Project graph to query.

        Returns:
            Number of edges that matched (and were deleted). 0 for empty list.
        """
        if not uuids:
            return 0
        logger.info('Deleting %d edge(s)', len(uuids))
        logger.debug('Edge UUIDs to delete: %s', uuids)
        graph = self._graph_for(group_id)
        # Pre-count: how many of the requested UUIDs actually exist as edges
        count_cypher = (
            'MATCH ()-[e:RELATES_TO]->() '
            'WHERE e.uuid IN $uuids '
            'RETURN count(e) AS found'
        )
        count_result = await graph.query(count_cypher, {'uuids': uuids})
        found = int(count_result.result_set[0][0]) if count_result.result_set else 0
        # Delete the edges
        delete_cypher = (
            'MATCH ()-[e:RELATES_TO]->() '
            'WHERE e.uuid IN $uuids '
            'DELETE e'
        )
        await graph.query(delete_cypher, {'uuids': uuids})
        return found

    @_canonicalize_group_args
    async def dedup_valid_edges_for_node(self, node_uuid: str, *, group_id: str) -> int:
        """Collapse post-merge parallel duplicate valid edges incident to a node.

        Task 2118: ``redirect_node_edges`` (the ``merge_entities`` helper)
        redirects a deprecated node's edges onto the surviving node by
        blindly copying ``old.uuid`` onto the recreated edge, without
        checking whether the survivor already has an equivalent edge to the
        same neighbor. When the deprecated/surviving pair started as
        exact-name duplicate nodes each holding their own copy of the same
        fact, this leaves the survivor with two distinct-uuid ``RELATES_TO``
        edges to the same neighbor sharing an identical (normalized) fact
        and ``valid_at`` — a duplicate that neither the pre-merge
        ``MemoryService._dedup_episode_edges`` sweep nor graphiti-core's
        ``resolve_extracted_edges`` fast-path can catch, since those only
        see the edges before they converge onto the same node pair.

        Queries the node's currently-valid (``invalid_at IS NULL``)
        incident edges (undirected, so directed and reverse-directed exact
        duplicates both collapse), delegates grouping + survivor selection
        to ``_duplicate_edge_uuids``, and deletes any non-survivor uuids via
        ``bulk_remove_edges``.

        Args:
            node_uuid: UUID of the Entity node whose valid edges to dedup
                (typically the surviving node of a merge).
            group_id: Project graph to query.

        Returns:
            Number of duplicate edges removed. 0 when there is nothing to
            dedup — ``bulk_remove_edges`` is not called in that case.

        Raises:
            RuntimeError: if the backend is not initialized.
        """
        graph = self._graph_for(group_id)
        cypher = (
            'MATCH (n:Entity {uuid: $uuid})-[e:RELATES_TO]-(m:Entity) '
            'WHERE e.invalid_at IS NULL '
            'RETURN m.uuid, e.uuid, e.fact, e.valid_at'
        )
        result = await graph.ro_query(cypher, {'uuid': node_uuid})
        duplicate_uuids = self._duplicate_edge_uuids(result.result_set or [])
        if not duplicate_uuids:
            return 0
        return await self.bulk_remove_edges(duplicate_uuids, group_id=group_id)

    @_canonicalize_group_args
    async def redirect_node_edges(
        self, deprecated_uuid: str, surviving_uuid: str, *, group_id: str
    ) -> dict:
        """Redirect all RELATES_TO edges from deprecated node to surviving node.

        Three Cypher phases:
        (1) Count and delete inter-node edges between the two nodes (they become
            meaningless self-loops after merge).
        (2) Enumerate outgoing edges (deprecated→target) by stable internal
            element ID(old) — not old.uuid, which may already be duplicated by
            a prior buggy merge — then redirect one edge per query:
            deprecated→target becomes surviving→target, minting a FRESH uuid4
            for the new edge and recording the original as
            new.superseded_edge_uuid for audit (all other properties copied
            individually to preserve vecf32 embedding type).
        (3) Symmetrically, enumerate incoming edges (source→deprecated) by
            ID(old) and redirect one edge per query: source→deprecated becomes
            source→surviving, likewise minting a fresh uuid4 and recording
            new.superseded_edge_uuid.

        Redirecting per-edge and keying on the internal ID(old) — rather than
        issuing a single bulk query keyed on old.uuid — preserves the
        graph-wide "uuid is unique per RELATES_TO edge" invariant: keying on
        old.uuid would silently coalesce any pre-existing dup-uuid edges
        instead of redirecting each one individually (task 2207 W6-δ).

        This trades a single bulk statement per direction for one query per
        edge (N+1 round-trips) — the deliberate cost of the ID(old) keying
        above. Entity merges are rare and touch modest-degree nodes in
        practice, so this is acceptable; if a hub node with hundreds of
        edges ever makes this hot, batch via UNWIND over a
        [{eid, new_uuid}, ...] parameter list while still keying each
        redirect on ID(old) (task 2207 W6-δ Open-Q2, deferred).

        Neither this method nor merge_entities() as a whole runs inside a
        transaction, so a crash or query error partway through Phase 2/3
        leaves some edges already redirected onto the survivor and others
        still on the deprecated node. Retrying redirect_node_edges from the
        top after such a failure is safe: each phase re-enumerates edges
        live off the deprecated node, so already-redirected edges (now
        anchored on the survivor) are simply not seen again.

        Args:
            deprecated_uuid: UUID of the entity node to be deleted.
            surviving_uuid: UUID of the entity node that will absorb the edges.

        Returns:
            Dict with keys: outgoing_redirected, incoming_redirected,
            inter_node_deleted. The redirected counts are incremented only
            after each per-edge query completes successfully, so they
            reflect edges actually redirected rather than merely enumerated.
        """
        graph = self._graph_for(group_id)

        # Phase 1: Delete inter-node edges (edges between the two merging nodes)
        count_inter = await graph.query(
            'MATCH (dep:Entity {uuid: $dep_uuid})-[e:RELATES_TO]-(sur:Entity {uuid: $sur_uuid}) '
            'RETURN count(e) AS cnt',
            {'dep_uuid': deprecated_uuid, 'sur_uuid': surviving_uuid},
        )
        inter_node_deleted = (
            int(count_inter.result_set[0][0]) if count_inter.result_set else 0
        )
        await graph.query(
            'MATCH (dep:Entity {uuid: $dep_uuid})-[e:RELATES_TO]-(sur:Entity {uuid: $sur_uuid}) '
            'DELETE e',
            {'dep_uuid': deprecated_uuid, 'sur_uuid': surviving_uuid},
        )

        # Phase 2: Redirect outgoing edges (deprecated → target). Enumerate
        # the redirect set by stable internal element ID(old) — NOT old.uuid,
        # which may already be duplicated by a prior buggy merge — then
        # redirect one edge per query, minting a fresh uuid4 per edge and
        # recording the original uuid as new.superseded_edge_uuid for audit.
        out_enum = await graph.query(
            'MATCH (dep:Entity {uuid: $dep_uuid})-[old:RELATES_TO]->() '
            'RETURN ID(old) AS eid',
            {'dep_uuid': deprecated_uuid},
        )
        out_eids = [row[0] for row in (out_enum.result_set or [])]
        outgoing_redirected = 0
        for eid in out_eids:
            new_uuid = str(uuid.uuid4())
            await graph.query(
                'MATCH (dep:Entity {uuid: $dep_uuid})-[old:RELATES_TO]->(target) '
                'WHERE ID(old) = $eid '
                'MATCH (sur:Entity {uuid: $sur_uuid}) '
                'CREATE (sur)-[new:RELATES_TO]->(target) '
                'SET new.uuid = $new_uuid, '
                '    new.superseded_edge_uuid = old.uuid, '
                '    new.name = old.name, '
                '    new.fact = old.fact, '
                '    new.fact_embedding = old.fact_embedding, '
                '    new.valid_at = old.valid_at, '
                '    new.invalid_at = old.invalid_at, '
                '    new.created_at = old.created_at, '
                '    new.group_id = old.group_id, '
                '    new.episodes = old.episodes, '
                '    new.source_node_uuid = $sur_uuid '
                'DELETE old',
                {
                    'dep_uuid': deprecated_uuid,
                    'sur_uuid': surviving_uuid,
                    'eid': eid,
                    'new_uuid': new_uuid,
                },
            )
            outgoing_redirected += 1

        # Phase 3: Redirect incoming edges (source → deprecated). Enumerate
        # the redirect set by stable internal element ID(old) — NOT old.uuid,
        # which may already be duplicated by a prior buggy merge — then
        # redirect one edge per query, minting a fresh uuid4 per edge and
        # recording the original uuid as new.superseded_edge_uuid for audit.
        in_enum = await graph.query(
            'MATCH (source)-[old:RELATES_TO]->(dep:Entity {uuid: $dep_uuid}) '
            'RETURN ID(old) AS eid',
            {'dep_uuid': deprecated_uuid},
        )
        in_eids = [row[0] for row in (in_enum.result_set or [])]
        incoming_redirected = 0
        for eid in in_eids:
            new_uuid = str(uuid.uuid4())
            await graph.query(
                'MATCH (source)-[old:RELATES_TO]->(dep:Entity {uuid: $dep_uuid}) '
                'WHERE ID(old) = $eid '
                'MATCH (sur:Entity {uuid: $sur_uuid}) '
                'CREATE (source)-[new:RELATES_TO]->(sur) '
                'SET new.uuid = $new_uuid, '
                '    new.superseded_edge_uuid = old.uuid, '
                '    new.name = old.name, '
                '    new.fact = old.fact, '
                '    new.fact_embedding = old.fact_embedding, '
                '    new.valid_at = old.valid_at, '
                '    new.invalid_at = old.invalid_at, '
                '    new.created_at = old.created_at, '
                '    new.group_id = old.group_id, '
                '    new.episodes = old.episodes, '
                '    new.target_node_uuid = $sur_uuid '
                'DELETE old',
                {
                    'dep_uuid': deprecated_uuid,
                    'sur_uuid': surviving_uuid,
                    'eid': eid,
                    'new_uuid': new_uuid,
                },
            )
            incoming_redirected += 1

        logger.info(
            'redirect_node_edges: dep=%s sur=%s inter_deleted=%d out=%d in=%d',
            deprecated_uuid, surviving_uuid, inter_node_deleted,
            outgoing_redirected, incoming_redirected,
        )
        return {
            'outgoing_redirected': outgoing_redirected,
            'incoming_redirected': incoming_redirected,
            'inter_node_deleted': inter_node_deleted,
        }

    async def _repair_duplicate_edge_uuids(self, group_id: str) -> int:
        """One-shot idempotent repair: re-mint fresh uuids on legacy dup-uuid
        RELATES_TO edges so per-uuid count(*) <= 1 — B6 dup-uuid-edge repair.

        Reuses redirect_node_edges's (task 2207 W6-δ) fresh-uuid4 +
        superseded_edge_uuid + ID(e)-keyed per-edge convention, but re-mints
        IN PLACE via a property SET rather than δ's CREATE-new/DELETE-old
        redirect: this repair never moves an edge to a different endpoint, so
        an in-place SET preserves both endpoints and the vecf32
        fact_embedding automatically (no CREATE/DELETE, no per-property
        copy). Keying on the stable internal ID(e) — not e.uuid — is
        mandatory precisely because the uuid is duplicated and cannot target
        a single edge.

        Within each dup-uuid group the first enumerated edge (eids[0]) keeps
        its original uuid (the survivor); only eids[1:] are re-minted. This
        is the minimal write set that restores per-uuid count(*) <= 1 while
        preserving the full edge set (no edge created or deleted), and it
        makes the repair idempotent: a re-run's detection query finds no
        group with count(*) > 1 and performs zero writes.

        GRAPH-KEY-scoped (NO group_id filter) — deliberately broader than the
        dup-node alarm's `n.group_id = $group_id` scoping. ζ's forthcoming
        uuid-keyed read dedup operates on the graph key regardless of an
        edge's group_id property, so EVERY RELATES_TO edge in this graph key
        must reach per-uuid uniqueness; a group_id filter here could leave a
        dup-uuid pair unrepaired and silently break ζ's global
        uuid-uniqueness premise.

        Performance note: the detection query aggregates over every
        RELATES_TO edge in the graph, every time this runs (i.e. on every
        backend startup, not just the first "one-shot" pass) — there is
        deliberately no sentinel/version marker gating it, since a legacy
        graph could always be mutated by an older code path between
        restarts. On a large, long-clean graph this is startup-path DB CPU
        spent to confirm a no-op. Per-graph duration and the detected
        dup-uuid-group count are logged at DEBUG so this cost is observable;
        see also _run_startup_identity_scan's aggregate elapsed_ms.

        Concurrency note: repair writes are NOT serialized against another
        process's concurrent initialize() (no cross-process lock is taken).
        `_identity_lock_for` is an in-process asyncio.Lock keyed on this
        instance's own `_identity_locks` dict — a second orchestrator process
        has an entirely separate registry, so acquiring it here would not
        actually prevent two processes from racing on the same dup-uuid
        group. This is accepted as safe: each racing writer targets the same
        ID(e) with its own fresh uuid4, so the last SET simply wins — the
        per-uuid uniqueness postcondition still holds either way. The only
        cost is harmless churn (an extra re-mint) and the possibility that
        `superseded_edge_uuid` ends up recording whichever writer's original
        `old_uuid` lost the race, which is acceptable for an audit-only
        property.

        Args:
            group_id: Project graph to repair.

        Returns:
            Count of edges re-minted (0 on a clean graph — no-op).
        """
        graph = self._graph_for(group_id)
        detect_cypher = (
            'MATCH ()-[e:RELATES_TO]->() '
            'WITH e.uuid AS u, count(*) AS c, collect(ID(e)) AS eids '
            'WHERE c > 1 '
            'RETURN u, eids'
        )
        start = time.monotonic()
        detect_result = await graph.query(detect_cypher)
        dup_groups = detect_result.result_set or []
        repaired = 0
        for old_uuid, eids in dup_groups:
            for eid in eids[1:]:
                new_uuid = str(uuid.uuid4())
                await graph.query(
                    'MATCH ()-[e:RELATES_TO]->() '
                    'WHERE ID(e) = $eid '
                    'SET e.uuid = $new_uuid, e.superseded_edge_uuid = $old_uuid',
                    {'eid': eid, 'new_uuid': new_uuid, 'old_uuid': old_uuid},
                )
                repaired += 1
        elapsed_ms = (time.monotonic() - start) * 1000
        logger.debug(
            'startup identity scan: dup-uuid-edge repair for graph %r took %.1fms '
            '(%d dup-uuid group(s) found, %d edge(s) re-minted)',
            group_id, elapsed_ms, len(dup_groups), repaired,
        )
        if repaired:
            logger.info(
                'startup identity scan: re-minted %d duplicate-uuid RELATES_TO edge(s) '
                'in graph %r',
                repaired, group_id,
            )
        return repaired

    @_canonicalize_group_args
    async def merge_entities(
        self, deprecated_uuid: str, surviving_uuid: str, *, group_id: str
    ) -> dict:
        """Merge two entity nodes by redirecting edges and deleting the deprecated node.

        Orchestrates the full merge workflow:
        1. Validate both nodes exist via get_node_text (raises NodeNotFoundError if
           either is missing).
        2. Redirect all RELATES_TO edges from deprecated to surviving via
           redirect_node_edges.
        3. Delete the deprecated node via delete_entity_node.
        4. Collapse any parallel duplicate edges left on the surviving node via
           dedup_valid_edges_for_node (task 2118 — redirect_node_edges mints a
           fresh uuid per redirected edge, but the survivor may already hold
           an equivalent (neighbor, fact, valid_at) edge, so this
           uuid-agnostic pass is still required to collapse those parallel
           duplicates).
        5. Rebuild the surviving node's summary via refresh_entity_summary.

        Args:
            deprecated_uuid: UUID of the entity node to be deleted.
            surviving_uuid: UUID of the entity node that absorbs the edges.

        Returns:
            Audit dict with keys: surviving_uuid, surviving_name, deprecated_uuid,
            deprecated_name, edges_redirected (sub-dict with redirect counts),
            duplicate_edges_removed (count collapsed post-redirect),
            surviving_summary (dict with old/new summary and edge_count).

        Raises:
            NodeNotFoundError: if either UUID does not exist.
            RuntimeError: if the backend is not initialized.
        """
        # Validate both nodes exist and capture their names
        dep_name, _ = await self.get_node_text(deprecated_uuid, group_id=group_id)
        sur_name, _ = await self.get_node_text(surviving_uuid, group_id=group_id)

        # Redirect edges
        edges_redirected = await self.redirect_node_edges(
            deprecated_uuid, surviving_uuid, group_id=group_id,
        )

        # Delete the deprecated node
        await self.delete_entity_node(deprecated_uuid, group_id=group_id)

        # Collapse parallel duplicates left on the survivor by the redirect above
        duplicate_edges_removed = await self.dedup_valid_edges_for_node(
            surviving_uuid, group_id=group_id,
        )

        # Rebuild the surviving node's summary
        refresh_result = await self.refresh_entity_summary(surviving_uuid, group_id=group_id)

        logger.info(
            'merge_entities: dep=%s (%r) sur=%s (%r) redirected=%s duplicate_edges_removed=%d',
            deprecated_uuid, dep_name, surviving_uuid, sur_name, edges_redirected,
            duplicate_edges_removed,
        )
        return {
            'surviving_uuid': surviving_uuid,
            'surviving_name': sur_name,
            'deprecated_uuid': deprecated_uuid,
            'deprecated_name': dep_name,
            'edges_redirected': edges_redirected,
            'duplicate_edges_removed': duplicate_edges_removed,
            'surviving_summary': {
                'before': refresh_result.get('old_summary', ''),
                'after': refresh_result.get('new_summary', ''),
                'edge_count': refresh_result.get('edge_count', 0),
            },
        }

    @_canonicalize_group_args
    async def delete_entity(
        self,
        uuid: str,
        *,
        group_id: str,
        force: bool = False,
    ) -> dict:
        """Delete a Graphiti Entity node, optionally refreshing connected neighbours.

        Orchestrates the full delete workflow:
        1. Validate the node exists via get_node_text (raises NodeNotFoundError if missing).
        2. Guard: if the node has any valid active edges and force=False, raises
           ActiveEdgesError with the edge count.
        3. Collect valid-edge neighbours BEFORE deletion (edges vanish after DETACH DELETE).
        4. Delete the node via delete_entity_node.
        5. Refresh each neighbour's summary via refresh_entity_summary.

        Args:
            uuid: UUID of the Entity node to delete.
            group_id: Project graph to target.
            force: When True, bypass the active-edges guard and delete anyway.

        Returns:
            Audit dict with keys: deleted_uuid, deleted_name, active_edge_count,
            forced, connected_refreshed (list of successfully refreshed neighbour UUIDs),
            refresh_errors (list of neighbour UUIDs whose summary refresh failed).
            The node is already gone when refresh runs, so refresh failures are
            non-fatal — callers should inspect refresh_errors for partial outcomes.

        Raises:
            NodeNotFoundError: if the node does not exist.
            ActiveEdgesError: if the node has valid active edges and force=False.
            RuntimeError: if the backend is not initialized.
        """
        # 1. Validate existence and capture name
        name, _ = await self.get_node_text(uuid, group_id=group_id)

        # 2. Guard: check active edges
        active_edges = await self.get_valid_edges_for_node(uuid, group_id=group_id)
        if active_edges and not force:
            raise ActiveEdgesError(
                f'Entity {uuid} has {len(active_edges)} valid active edge(s); '
                f'pass force=True to delete'
            )

        # 3. Collect neighbours BEFORE delete (edges vanish with DETACH DELETE)
        neighbours = await self.get_connected_entity_uuids(uuid, group_id=group_id)

        # 4. Delete the node
        await self.delete_entity_node(uuid, group_id=group_id)

        # 5. Refresh each neighbour's summary — best-effort.
        # The node is already deleted (irreversible), so a refresh failure must not
        # propagate as a top-level error and must not abort remaining refreshes.
        # Collect both the successful and failed sets for the audit dict.
        refreshed: list[str] = []
        refresh_errors: list[str] = []
        for nbr in neighbours:
            try:
                await self.refresh_entity_summary(nbr, group_id=group_id)
                refreshed.append(nbr)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    'delete_entity: refresh_entity_summary failed for nbr=%s '
                    '(deleted node uuid=%s): %s',
                    nbr, uuid, exc,
                )
                refresh_errors.append(nbr)

        logger.info(
            'delete_entity: deleted uuid=%s name=%r active_edges=%d force=%s '
            'refreshed=%d errors=%d',
            uuid, name, len(active_edges), force, len(refreshed), len(refresh_errors),
        )
        return {
            'deleted_uuid': uuid,
            'deleted_name': name,
            'active_edge_count': len(active_edges),
            'forced': force,
            'connected_refreshed': refreshed,
            'refresh_errors': refresh_errors,
        }

    @_canonicalize_group_args
    async def delete_entity_node(self, uuid: str, *, group_id: str) -> None:
        """Delete an Entity node and all remaining relationships.

        Validates that the node exists first, then issues DETACH DELETE.
        Pre-check uses ro_query since it performs no writes; the DETACH DELETE
        itself uses graph.query.

        Args:
            uuid: UUID of the Entity node to delete.
            group_id: Project graph to query.

        Raises:
            NodeNotFoundError: if no node with that UUID exists.
            RuntimeError: if the backend is not initialized.
        """
        graph = self._graph_for(group_id)
        # Pre-check: verify node exists before deleting (read-only)
        check_result = await graph.ro_query(
            'MATCH (n:Entity {uuid: $uuid}) RETURN n.name, n.summary',
            {'uuid': uuid},
        )
        if not check_result.result_set:
            raise NodeNotFoundError(f'Entity node not found: {uuid}')
        await graph.query(
            'MATCH (n:Entity {uuid: $uuid}) DETACH DELETE n',
            {'uuid': uuid},
        )
        logger.info('delete_entity_node: deleted node=%s', uuid)

    @_canonicalize_group_args
    async def get_node_text(self, uuid: str, *, group_id: str) -> tuple[str, str]:
        """Return (name, summary) for the Entity node with the given UUID.

        Uses ro_query since no writes are performed.

        Raises:
            NodeNotFoundError: if no node with that UUID exists.
        """
        graph = self._graph_for(group_id)
        cypher = (
            'MATCH (n:Entity {uuid: $uuid}) '
            'RETURN n.name, n.summary'
        )
        result = await graph.ro_query(cypher, {'uuid': uuid})
        if not result.result_set:
            raise NodeNotFoundError(f'Entity node not found: {uuid}')
        row = result.result_set[0]
        return (row[0], row[1] or '')

    @_canonicalize_group_args
    async def resolve_entity_by_name(self, name: str, *, group_id: str) -> str:
        """Resolve an entity name to its UUID via an exact Cypher lookup.

        Uses ro_query since no writes are performed.

        Args:
            name: Exact name of the Entity node to resolve.
            group_id: Project graph to query.

        Returns:
            The UUID string of the matching entity.

        Raises:
            NodeNotFoundError: if no entity with that name exists.
            AmbiguousEntityError: if multiple entities share the same name,
                with all matching UUIDs listed in the error message.
            RuntimeError: if the backend is not initialized.
        """
        graph = self._graph_for(group_id)
        cypher = 'MATCH (n:Entity {name: $name}) RETURN n.uuid, n.name'
        result = await graph.ro_query(cypher, {'name': name})
        rows = result.result_set
        if not rows:
            raise NodeNotFoundError(f'No entity found with name: {name!r}')
        if len(rows) > 1:
            uuids = [row[0] for row in rows]
            raise AmbiguousEntityError(
                f'Multiple entities found with name {name!r}: {uuids}'
            )
        return rows[0][0]

    @_canonicalize_group_args
    async def get_nodes_by_exact_name(self, name: str, *, group_id: str) -> list[dict]:
        """Resolve an entity name to full node data via an exact, case-sensitive Cypher match.

        Sibling to resolve_entity_by_name: same `MATCH (n:Entity {name: $name})` shape,
        but returns full node data (uuid, name, summary, labels) as a list[dict] and
        never raises on zero or multiple matches — callers (e.g. MemoryService.get_entity)
        treat an empty result as "fall back to fuzzy search" and pick nodes[0] on a hit.

        Scoped by an explicit `n.group_id = $group_id` property predicate (2026-07-06
        amendment), not just the graph key selected via _graph_for — task-2115's active
        cross-graph leak can plant a misrouted foreign node (group_id property of
        ANOTHER project) physically inside this graph key, and this predicate keeps
        such a clone from ever surfacing here.

        Uses ro_query since no writes are performed.

        Args:
            name: Exact name of the Entity node(s) to resolve.
            group_id: Project graph to query.

        Returns:
            List of dicts with keys: uuid, name, summary (None when NULL), labels
            (defaults to [] when NULL/absent). Empty list when no entity matches.

        Raises:
            RuntimeError: if the backend is not initialized.
        """
        graph = self._graph_for(group_id)
        cypher = (
            'MATCH (n:Entity {name: $name}) WHERE n.group_id = $group_id '
            'RETURN n.uuid, n.name, n.summary, labels(n)'
        )
        result = await graph.ro_query(cypher, {'name': name, 'group_id': group_id})
        return [
            {
                'uuid': row[0],
                'name': row[1],
                'summary': row[2],
                'labels': row[3] or [],
            }
            for row in (result.result_set or [])
        ]

    @_canonicalize_group_args
    async def find_duplicate_entity_nodes(self, name: str, *, group_id: str) -> list[dict]:
        """Return every Entity node sharing an exact name, canonical-ordered.

        Sibling to resolve_entity_by_name / get_nodes_by_exact_name: same
        `MATCH (n:Entity {name: $name})` exact, case-sensitive match shape,
        but scoped to surfacing exact-name DUPLICATES for the post-write
        node-dedup sweep (MemoryService._dedup_episode_nodes) rather than
        resolving a single canonical node. Results are ordered
        canonical-first — most valid edges, then oldest created_at, then
        uuid — so callers can treat matches[0] as the merge survivor and
        matches[1:] as the deprecated duplicates to fold into it.

        Scoped by an explicit `n.group_id = $group_id` property predicate (2026-07-06
        amendment), not just the graph key selected via _graph_for — task-2115's active
        cross-graph leak can plant a misrouted foreign node (group_id property of
        ANOTHER project) physically inside this graph key, and this predicate keeps
        such a clone from ever being treated as a duplicate to collapse.

        Uses ro_query since no writes are performed.

        Args:
            name: Exact name of the Entity node(s) to look up.
            group_id: Project graph to query.

        Returns:
            List of dicts with keys: uuid, created_at, edge_count — ordered
            canonical (survivor) first. Empty list when no entity matches;
            a single-element list when the name is unique (no duplicate).

        Raises:
            RuntimeError: if the backend is not initialized.
        """
        graph = self._graph_for(group_id)
        cypher = (
            'MATCH (n:Entity {name: $name}) '
            'WHERE n.group_id = $group_id '
            'OPTIONAL MATCH (n)-[e:RELATES_TO]-() WHERE e.invalid_at IS NULL '
            'WITH n, count(DISTINCT e) AS edge_count '
            'RETURN n.uuid, n.created_at, edge_count '
            'ORDER BY edge_count DESC, n.created_at ASC, n.uuid ASC'
        )
        result = await graph.ro_query(cypher, {'name': name, 'group_id': group_id})
        return [
            {
                'uuid': row[0],
                'created_at': row[1],
                'edge_count': row[2],
            }
            for row in (result.result_set or [])
        ]

    async def _scan_duplicate_entity_names(self, group_id: str) -> list[tuple[str, int]]:
        """Detect exact-name duplicate Entity nodes in *group_id*'s graph — B5 dup-node alarm.

        A safety net for the write-time identity gate (_resolve_or_create_entity,
        task 2198/α + task 2202/β): that gate should prevent same-name duplicate
        Entity nodes from ever being created, but this scan is a POSITIVE
        DETECTION signal, not input rejection — if duplicates slip through
        anyway (e.g. a gate bug, or data written before the gate existed), a
        LOUD WARN surfaces them at startup instead of letting them silently
        accumulate. It never mutates or rejects anything.

        Scoped by an explicit `n.group_id = $group_id` property predicate
        (2026-07-06 advisory), the same scoping task 2198 added to
        get_nodes_by_exact_name / find_duplicate_entity_nodes above —
        task-2115's active cross-graph leak can plant a misrouted foreign
        node (group_id property of ANOTHER project) physically inside this
        graph key, and that foreign-group clone is a 2115 artifact routed to
        2115, not alarmed here — only SAME-group duplicates count.

        Uses ro_query since no writes are performed.

        Args:
            group_id: Project graph to scan.

        Returns:
            List of (name, count) tuples for every duplicated name (count > 1).
            Empty list when the graph has no exact-name duplicates.
        """
        graph = self._graph_for(group_id)
        cypher = (
            'MATCH (n:Entity) WHERE n.group_id = $group_id '
            'WITH n.name AS name, count(*) AS cnt '
            'WHERE cnt > 1 '
            'RETURN name, cnt'
        )
        start = time.monotonic()
        result = await graph.ro_query(cypher, {'group_id': group_id})
        elapsed_ms = (time.monotonic() - start) * 1000
        duplicates = [(row[0], row[1]) for row in (result.result_set or [])]
        logger.debug(
            'startup identity scan: dup-node name scan for graph %r took %.1fms '
            '(%d duplicated name(s) found)',
            group_id, elapsed_ms, len(duplicates),
        )
        for name, cnt in duplicates:
            logger.warning(
                'startup identity scan: %d duplicate Entity nodes named %r in group %r '
                '(exact-name identity gate should prevent this — investigate)',
                cnt, name, group_id,
            )
        return duplicates

    async def _run_startup_identity_scan(self) -> dict:
        """Startup identity-integrity sweep — per-graph orchestrator (W6-ε).

        Enumerates every project graph via list_graphs() and, for each, runs
        both the dup-NODE alarm (_scan_duplicate_entity_names) and the
        dup-uuid-EDGE repair (_repair_duplicate_edge_uuids). Each graph is
        processed best-effort: a failure on one graph is caught and logged,
        never aborting the sweep over the rest — mirrors the initialize()
        index-setup loop idiom (graphiti_client.py:378-385, try/except
        Exception: logger.warning).

        Called from initialize() (itself wrapped in a try/except there) so a
        total scan failure never breaks backend startup — this sweep is a
        safety net, not a startup gate.

        This runs synchronously on the initialize() critical path on every
        boot (not just once), scanning every RELATES_TO edge and every
        Entity node grouping in every project graph — see the performance
        note on _repair_duplicate_edge_uuids. The aggregate elapsed_ms below,
        plus each sub-method's per-graph DEBUG timing log, make that cost
        observable so a startup slowdown can be diagnosed without guessing.

        Returns:
            Aggregate stats dict:
            - graphs_scanned: graphs attempted (counted whether or not that
              graph's scan/repair completed cleanly).
            - dup_name_groups: total duplicated names found across all graphs
              (sum of len(...) of each graph's _scan_duplicate_entity_names
              result).
            - edges_repaired: total dup-uuid edges re-minted across all
              graphs (sum of each graph's _repair_duplicate_edge_uuids
              result).
            - elapsed_ms: wall-clock duration of the full sweep (all graphs),
              in milliseconds.
        """
        graphs = await self.list_graphs()
        stats: dict[str, int | float] = {
            'graphs_scanned': 0, 'dup_name_groups': 0, 'edges_repaired': 0,
        }
        start = time.monotonic()
        for graph_name in graphs:
            stats['graphs_scanned'] += 1
            try:
                duplicates = await self._scan_duplicate_entity_names(graph_name)
                stats['dup_name_groups'] += len(duplicates)
                repaired = await self._repair_duplicate_edge_uuids(graph_name)
                stats['edges_repaired'] += repaired
            except Exception:
                logger.warning(
                    'Startup identity-integrity scan failed for graph %r',
                    graph_name, exc_info=True,
                )
        stats['elapsed_ms'] = (time.monotonic() - start) * 1000
        logger.info(
            'startup identity scan complete: graphs_scanned=%d dup_name_groups=%d '
            'edges_repaired=%d elapsed_ms=%.1f',
            stats['graphs_scanned'], stats['dup_name_groups'], stats['edges_repaired'],
            stats['elapsed_ms'],
        )
        return stats

    async def _resolve_or_create_entity(self, name: str, *, group_id: str) -> str | None:
        """Exact-name write-time-identity chokepoint: resolve, or collapse duplicates.

        MUST be called only while the caller holds ``_identity_lock_for(group_id)``
        — this method performs no locking of its own. Idempotent: calling it
        repeatedly for the same (name, group_id) converges and stays converged.

        Behaviour by match count (via get_nodes_by_exact_name, group_id-scoped):
        - 0 matches: returns None. Documented no-op — node minting stays
          graphiti_core's job; this primitive only resolves/collapses existing
          nodes, it never creates one.
        - 1 match: returns that node's uuid directly (pure resolve, no writes).
        - >=2 matches: collapses duplicates via find_duplicate_entity_nodes
          (already survivor-first: edge_count DESC, created_at ASC, uuid ASC)
          and merge_entities, folding every non-canonical duplicate into the
          survivor. Returns the survivor's uuid.

        Args:
            name: Exact name of the Entity to resolve.
            group_id: Project graph to target.

        Returns:
            The UUID of the single canonical Entity node with this name in
            group_id's graph, or None if none existed.

        Post-condition on return (non-None): exactly one Entity node with
        this name remains in group_id's graph.
        """
        nodes = await self.get_nodes_by_exact_name(name, group_id=group_id)
        if not nodes:
            return None
        if len(nodes) == 1:
            return nodes[0]['uuid']
        dups = await self.find_duplicate_entity_nodes(name, group_id=group_id)
        if not dups:
            # Defensive: get_nodes_by_exact_name and find_duplicate_entity_nodes
            # are separate queries and, under the lock contract, are expected to
            # filter identically. If they ever diverge (e.g. a future change
            # narrows find_duplicate_entity_nodes to exclude edgeless nodes),
            # degrade to a no-op rather than raise IndexError on dups[0].
            return None
        survivor = dups[0]
        for dup in dups[1:]:
            await self.merge_entities(dup['uuid'], survivor['uuid'], group_id=group_id)
        return survivor['uuid']

    @staticmethod
    def _edge_dict(uuid: str, fact: str | None, name: str | None) -> EdgeDict:
        """Build a normalised edge dict, coercing NULL fact/name to empty string.

        Args:
            uuid: Edge UUID. Must not be None — a NULL uuid from the graph would
                propagate silently through downstream callers and is treated as
                a hard error.
            fact: Edge fact text, or None when the property is NULL in the graph.
            name: Edge name, or None when the property is NULL in the graph.

        Returns:
            EdgeDict with keys: uuid, fact, name. fact and name default to '' when None.

        Raises:
            ValueError: If uuid is None.
        """
        if uuid is None:
            raise ValueError('edge uuid must not be None')
        return {
            'uuid': uuid,
            'fact': fact if fact is not None else '',
            'name': name if name is not None else '',
        }

    @staticmethod
    def _duplicate_edge_uuids(rows: Sequence[Sequence[Any]]) -> list[str]:
        """Return non-survivor edge uuids from valid-edge rows sharing a duplicate key.

        Task 2118: after ``redirect_node_edges`` (the ``merge_entities`` helper)
        blindly copies ``old.uuid`` onto redirected edges, a surviving node can
        end up with two distinct-uuid ``RELATES_TO`` edges to the same neighbor
        carrying an identical fact and ``valid_at`` — a parallel duplicate that
        neither the pre-merge ``MemoryService._dedup_episode_edges`` sweep nor
        graphiti-core's ``resolve_extracted_edges`` fast-path can catch, since
        they only converge onto the same node pair *after* the merge.

        Args:
            rows: Iterable of positional rows ``(neighbor_uuid, edge_uuid, fact,
                valid_at)`` — the shape returned by
                ``dedup_valid_edges_for_node``'s Cypher query.

        Returns:
            Sorted list of edge uuids to delete (empty when no group has a
            duplicate). For each group of rows sharing the same
            ``(neighbor_uuid, fact, valid_at)`` key, the lexicographically
            lowest edge uuid is kept as the survivor and every other uuid in
            the group is returned for deletion.

        Note:
            Rows are first deduplicated by edge uuid (keeping the first
            occurrence) so that an undirected Cypher MATCH double-matching a
            self-loop edge doesn't fabricate a spurious duplicate pair.
        """
        seen_edge_uuids: set[str] = set()
        deduped_rows: list[Sequence[Any]] = []
        for row in rows:
            edge_uuid = row[1]
            if edge_uuid in seen_edge_uuids:
                continue
            seen_edge_uuids.add(edge_uuid)
            deduped_rows.append(row)

        groups: dict[tuple[str, str, str], list[str]] = {}
        for row in deduped_rows:
            neighbor_uuid, edge_uuid, fact, valid_at = row[0], row[1], row[2], row[3]
            key = (neighbor_uuid, _normalize_fact_for_grouping(fact), str(valid_at))
            groups.setdefault(key, []).append(edge_uuid)

        duplicates: list[str] = []
        for uuids in groups.values():
            if len(uuids) > 1:
                survivor = min(uuids)
                duplicates.extend(u for u in uuids if u != survivor)
        return sorted(duplicates)

    @staticmethod
    def _canonical_facts(edges: Sequence[Mapping[str, Any]]) -> list[str]:
        """Deduplicate edge facts preserving insertion order, skipping non-string, empty, and whitespace-only values.

        Args:
            edges: List of edge dicts, each optionally containing a 'fact' key.

        Returns:
            List of unique non-empty fact strings in their first-seen order.
        """
        # f is assigned by the walrus := before isinstance checks str; rejects non-str, empty, whitespace-only
        return list(dict.fromkeys(f for e in edges if isinstance(f := e.get('fact'), str) and f and not f.isspace()))

    @staticmethod
    def _build_stale_entry(
        entity: dict,
        edges: Sequence[Mapping[str, Any]],
    ) -> dict | None:
        """Compute a stale-entry diagnostic dict for *entity*, or None if up-to-date.

        Encapsulates the repeated logic shared between
        ``detect_stale_with_edges`` and
        ``detect_stale_dry_run``:

        1. Return None if entity summary is empty (not stale by definition).
        2. Compute canonical facts via ``_canonical_facts`` and join with newlines.
        3. Return None if summary already equals the canonical string (up-to-date).
        4. Compute diagnostic counts (duplicate_count, stale_line_count,
           valid_fact_count, summary_line_count) and return the stale-entry dict.

        Args:
            entity: Entity dict with keys 'uuid', 'name', 'summary'.
            edges:  Valid edges for this entity (same schema as get_valid_edges_for_node).

        Returns:
            None if the entity is not stale; otherwise a dict with keys:
            uuid, name, summary, duplicate_count, stale_line_count,
            valid_fact_count, summary_line_count.
        """
        summary = entity['summary']
        if not summary:
            return None
        valid_facts = GraphitiBackend._canonical_facts(edges)
        canonical = '\n'.join(valid_facts)
        if summary == canonical:
            return None
        # Compute diagnostic counts
        summary_lines = summary.split('\n')
        valid_fact_set = set(valid_facts)
        # duplicate_count: sum of extra occurrences for each unique line that
        # appears more than once in the current summary.
        line_counts = Counter(summary_lines)
        duplicate_count = sum(c - 1 for c in line_counts.values() if c > 1)
        # stale_line_count: lines in summary not in the valid fact set
        stale_line_count = sum(1 for line in summary_lines if line not in valid_fact_set)
        return {
            'uuid': entity['uuid'],
            'name': entity['name'],
            'summary': summary,
            'duplicate_count': duplicate_count,
            'stale_line_count': stale_line_count,
            'valid_fact_count': len(valid_facts),
            'summary_line_count': len(summary_lines),
        }

    @_canonicalize_group_args
    async def refresh_entity_summary(
        self,
        node_uuid: str,
        *,
        group_id: str,
        name: str | None = None,
        old_summary: str | None = None,
    ) -> dict[str, Any]:
        """Regenerate an Entity node's summary from its currently-valid edges.

        Fetches the node's current name and summary, queries all valid
        (non-invalidated) RELATES_TO edges, deduplicates their facts
        (preserving order), joins them with newlines, and writes the result
        back to the node's summary property.

        Summary regeneration uses simple fact concatenation (deduped), consistent
        with Graphiti's own _extract_entity_summaries_batch pattern — no LLM call.

        For bulk use see ``rebuild_entity_from_edges``, which accepts
        caller-supplied edges, name, and old_summary to avoid per-entity
        ``get_node_text`` and ``get_valid_edges_for_node`` round-trips when
        rebuilding many entities at once.  The two methods are an intentional
        fork: ``refresh_entity_summary`` is self-contained for single-entity
        callers; ``rebuild_entity_from_edges`` is batch-internal and consumes
        pre-fetched data from the ``MemoryService.rebuild_entity_summaries`` pipeline.

        Args:
            node_uuid: UUID of the Entity node to refresh.
            group_id: Project graph to target.
            name: Optional entity name (must be paired with old_summary). When
                both are supplied, get_node_text is skipped — useful when the
                caller already has this data (e.g. rebuild_entity_from_edges).
            old_summary: Optional current summary text (must be paired with name).

        Returns:
            Dict with keys: uuid, name, old_summary, new_summary, edge_count.

        Raises:
            ValueError: if exactly one of name/old_summary is provided.
        """
        if (name is None) != (old_summary is None):
            raise ValueError('name and old_summary must both be provided or both omitted')
        if name is None:
            name, old_summary = await self.get_node_text(node_uuid, group_id=group_id)
        edges = await self.get_valid_edges_for_node(node_uuid, group_id=group_id)
        facts = self._canonical_facts(edges)
        new_summary = '\n'.join(facts)
        await self.update_node_summary(node_uuid, new_summary, group_id=group_id)
        logger.info(
            'refresh_entity_summary: node=%s name=%r edges=%d old_len=%d new_len=%d',
            node_uuid, name, len(edges), len(old_summary or ''), len(new_summary),
        )
        return {
            'uuid': node_uuid,
            'name': name,
            'old_summary': old_summary,
            'new_summary': new_summary,
            'edge_count': len(edges),
        }

    @_canonicalize_group_args
    async def set_entity_summary(
        self,
        node_uuid: str,
        summary: str,
        *,
        group_id: str,
    ) -> dict[str, Any]:
        """Overwrite an Entity node's summary with explicit text, verbatim.

        Unlike ``refresh_entity_summary`` (which regenerates the summary by
        deduplicating and joining the entity's currently-valid edge facts),
        this method writes ``summary`` exactly as given — it never reads or
        derives from edges. An empty string clears the summary entirely.

        This is the operator/reconciliation escape hatch for stale narrative
        text that edge-derived regeneration cannot remove (e.g. when the
        stale sentence is itself still carried by a valid edge fact).

        Validates that the node exists via ``get_node_text`` (which raises
        NodeNotFoundError for a missing UUID) before writing, so a bad UUID
        fails loudly instead of silently matching zero rows.

        Args:
            node_uuid: UUID of the Entity node to overwrite.
            summary: Exact text to write as the new summary. May be '' to clear.
            group_id: Project graph to target.

        Returns:
            Dict with keys: uuid, name, old_summary, new_summary.

        Raises:
            NodeNotFoundError: if no node with that UUID exists.
            RuntimeError: if the backend is not initialized.
        """
        name, old_summary = await self.get_node_text(node_uuid, group_id=group_id)
        await self.update_node_summary(node_uuid, summary, group_id=group_id)
        logger.info(
            'set_entity_summary: node=%s name=%r old_len=%d new_len=%d',
            node_uuid, name, len(old_summary or ''), len(summary),
        )
        return {
            'uuid': node_uuid,
            'name': name,
            'old_summary': old_summary,
            'new_summary': summary,
        }

    @_canonicalize_group_args
    async def rename_entity_node(
        self,
        node_uuid: str,
        new_name: str,
        *,
        group_id: str,
    ) -> dict[str, Any]:
        """Overwrite an Entity node's name property, then best-effort refresh its embedding.

        Used to correct non-canonical entity node names — e.g. task-entity nodes
        minted by graphiti-core's LLM extraction as 'task 132' or 'tasks 153'
        instead of the canonical 'Task 132' form (see
        ``fused_memory.utils.task_naming.canonicalize_task_node_name``). Both the
        write-path post-add_episode normalization hook
        (``MemoryService._normalize_task_node_names``) and the operator-facing
        ``rename_entity`` MCP tool delegate here.

        Validates that the node exists via ``get_node_text`` (which raises
        NodeNotFoundError for a missing UUID) before writing, so a bad UUID
        fails loudly instead of silently matching zero rows.

        After the name property is updated, the node's name_embedding is
        regenerated so fuzzy hybrid node search reflects the new name. This
        step is best-effort: an embedder failure is logged and swallowed
        rather than raised, since the name rewrite itself (the primary fix,
        which keeps exact-name Cypher lookups correct) has already succeeded.

        Args:
            node_uuid: UUID of the Entity node to rename.
            new_name: Exact new name to write.
            group_id: Project graph to target.

        Returns:
            Dict with keys: uuid, old_name, new_name.

        Raises:
            NodeNotFoundError: if no node with that UUID exists.
            RuntimeError: if the backend is not initialized.
        """
        old_name, _ = await self.get_node_text(node_uuid, group_id=group_id)
        await self.update_node_name(node_uuid, new_name, group_id=group_id)
        try:
            embedding = await self._require_client().embedder.create(new_name)
            await self.update_node_embedding(node_uuid, embedding, group_id=group_id)
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            # Best-effort: the name property is already updated (the primary
            # fix); a stale embedding only affects fuzzy hybrid node search.
            logger.warning(
                'rename_entity_node: failed to regenerate name_embedding for '
                'node=%s new_name=%r; name property was still updated',
                node_uuid, new_name, exc_info=True,
            )
        logger.info(
            'rename_entity_node: node=%s old_name=%r new_name=%r',
            node_uuid, old_name, new_name,
        )
        return {
            'uuid': node_uuid,
            'old_name': old_name,
            'new_name': new_name,
        }

    @_canonicalize_group_args
    async def list_entity_nodes(self, *, group_id: str) -> list[dict]:
        """Return all Entity nodes (uuid, name, summary) for a given group_id.

        FalkorDB is multi-tenant — each project lives in its own graph, so no
        group_id filter is needed in the Cypher itself.  Uses ro_query since
        no writes are performed.

        Args:
            group_id: Project graph to query.

        Returns:
            List of dicts with keys: uuid, name, summary (summary defaults to
            empty string when the node property is NULL).
        """
        graph = self._graph_for(group_id)
        cypher = 'MATCH (n:Entity) RETURN n.uuid, n.name, n.summary'
        result = await graph.ro_query(cypher)
        return [
            {
                'uuid': row[0],
                'name': row[1] or '',
                'summary': row[2] or '',
            }
            for row in (result.result_set or [])
        ]

    @_canonicalize_group_args
    async def detect_stale_with_edges(
        self, *, group_id: str
    ) -> StaleSummaryResult:
        """Detect stale summaries and return a StaleSummaryResult.

        Shared by detect_stale_summaries (public API) and MemoryService.rebuild_entity_summaries
        to avoid a duplicate bulk edge fetch when both are needed.

        Args:
            group_id: Project graph to query.

        Returns:
            StaleSummaryResult with fields:
              .stale       - list of stale entity dicts
              .all_edges   - dict[uuid, list[EdgeDict]] of valid edges for all entities
              .total_count - total number of entity nodes scanned
        """
        entities = await self.list_entity_nodes(group_id=group_id)
        all_edges = await self.get_all_valid_edges(group_id=group_id)
        stale: list[dict] = []
        for entity in entities:
            edges = all_edges.get(entity['uuid'], [])
            entry = self._build_stale_entry(entity, edges)
            if entry is not None:
                stale.append(entry)
        return StaleSummaryResult(stale=stale, all_edges=all_edges, total_count=len(entities))

    @_canonicalize_group_args
    async def detect_stale_dry_run(
        self, *, group_id: str, max_concurrency: int = 10
    ) -> tuple[list[dict], int]:
        """Detect stale summaries using per-entity edge fetching (dry_run variant).

        Memory-cheaper alternative to ``detect_stale_with_edges`` for use
        in the ``force=False, dry_run=True`` code path.  Unlike the bulk variant, this
        method never materialises the O(E) all-edges dict because:

        - The dry_run path short-circuits before ``rebuild_entity_from_edges``, so
          the edges dict is only needed for staleness comparison, not for writing.
        - Fetching edges per-entity (only for non-empty-summary entities) avoids
          holding the full graph's edge data in Python memory when none of it will
          be used to write.

        Trade-off vs ``detect_stale_with_edges``:
        - Issues up-to-N targeted ``get_valid_edges_for_node`` queries rather than a
          single bulk ``get_all_valid_edges`` query.
        - Entities with empty summaries are skipped without any edge query (matching
          the existing empty-summary semantics, adding a Pareto improvement for graphs
          with many empty-summary entities).

        Args:
            group_id: Project graph to query.
            max_concurrency: Maximum number of concurrent ``get_valid_edges_for_node``
                requests in flight at once.  Defaults to 10.  Raise or lower to tune
                the trade-off between latency and DB connection pressure.

        Returns:
            Tuple of (stale_list, total_count) where stale_list contains the same
            per-entity dict schema as ``detect_stale_with_edges``
            (uuid, name, summary, duplicate_count, stale_line_count, valid_fact_count,
            summary_line_count) and total_count is len(all entities).  Order of entries
            in stale_list matches the order returned by ``list_entity_nodes``.
        """
        if max_concurrency < 1:
            raise ValueError(f'max_concurrency must be >= 1, got {max_concurrency}')

        entities = await self.list_entity_nodes(group_id=group_id)

        # Separate entities: empty-summary ones are cheap-skipped without any I/O.
        fetch_entities = [e for e in entities if e['summary']]

        sem = asyncio.Semaphore(max_concurrency)

        async def _fetch_one(entity: dict) -> list:
            async with sem:
                return await self.get_valid_edges_for_node(entity['uuid'], group_id=group_id)

        # Two-tier check via gather_or_raise (fused_memory.utils.async_utils).
        # Pass 1 (inside gather_or_raise): propagates CancelledError / KeyboardInterrupt
        # before accumulation. Pass 2 (inside gather_or_raise): logs every captured
        # Exception at WARNING, then re-raises the positionally-first — raise-first
        # semantics are unchanged from the hand-rolled version, with per-failure
        # WARNING logging added.
        gather_results = await gather_or_raise(
            (_fetch_one(e) for e in fetch_entities),
            label='detect_stale_dry_run: get_valid_edges_for_node failed',
            logger=logger,
        )

        stale: list[dict] = []
        for entity, result in zip(fetch_entities, gather_results, strict=True):
            entry = self._build_stale_entry(entity, result)
            if entry is not None:
                stale.append(entry)

        return (stale, len(entities))

    @_canonicalize_group_args
    async def detect_stale_summaries(self, *, group_id: str) -> list[dict]:
        """Identify Entity nodes whose summary is out of sync with valid edge facts.

        For each entity node, fetches its valid RELATES_TO edges and computes
        the canonical summary (deduped facts joined with newlines).  An entity
        is considered *stale* when:

        - Its current summary is non-empty (empty summaries are skipped), AND
        - Its current summary differs from the canonical summary.

        Diagnostic fields help callers understand *why* an entity is stale:
        - ``duplicate_count``: extra occurrences of duplicated lines (lines that
          appear more than once in the current summary).
        - ``stale_line_count``: lines in the current summary that are not backed
          by any valid edge fact.
        - ``valid_fact_count``: number of unique valid edge facts.
        - ``summary_line_count``: number of lines in the current summary.

        Args:
            group_id: Project graph to query.

        Returns:
            List of dicts (one per stale entity) with keys: uuid, name, summary,
            duplicate_count, stale_line_count, valid_fact_count,
            summary_line_count. The ``summary`` key holds the current
            (pre-rebuild) entity summary text so callers can diff it against
            the canonical fact set without a second DB query.
        """
        result = await self.detect_stale_with_edges(group_id=group_id)
        return result.stale

    @_canonicalize_group_args
    async def rebuild_entity_from_edges(
        self, uuid: str, name: str, edges: list[EdgeDict], *, group_id: str,
        old_summary: str,
    ) -> dict[str, Any]:
        """Rebuild one Entity node's summary from pre-fetched edges.

        Accepts the edges already fetched by the bulk call, avoiding a
        per-entity get_valid_edges_for_node round-trip.

        For single-entity use (not bulk) see ``refresh_entity_summary``, which
        fetches its own name/old_summary via ``get_node_text`` and its own valid
        edges via ``get_valid_edges_for_node``.  This method exists as the
        bulk-optimised counterpart: it accepts caller-supplied edges and
        old_summary to eliminate per-entity DB round-trips when rebuilding many
        entities at once.

        .. note:: TOCTOU / eventual-consistency risk:
            The ``edges`` argument is pre-fetched by the caller in a single
            bulk query.  By the time this method runs (potentially after a
            concurrency gap), the graph may have changed — new edges added,
            existing edges invalidated.  Summaries written here therefore
            reflect a snapshot, not necessarily the current DB state.  This
            is an accepted trade-off: callers that require stronger consistency
            should re-fetch edges per entity via ``refresh_entity_summary``.

        Args:
            uuid: Entity UUID.
            name: Entity name (for logging / result dict).
            edges: Pre-fetched valid edge dicts (uuid, fact, name).
            group_id: Graph to update.
            old_summary: Current summary text (caller must supply — avoids
                per-entity ``get_node_text`` DB round-trip).

        Returns:
            Dict with keys: uuid, name, old_summary, new_summary, edge_count.
        """
        facts = self._canonical_facts(edges)
        new_summary = '\n'.join(facts)
        await self.update_node_summary(uuid, new_summary, group_id=group_id)
        logger.info(
            'rebuild_entity_from_edges: node=%s name=%r edges=%d new_len=%d',
            uuid, name, len(edges), len(new_summary),
        )
        return {
            'uuid': uuid,
            'name': name,
            'old_summary': old_summary,
            'new_summary': new_summary,
            'edge_count': len(edges),
        }

    @_canonicalize_group_args
    async def update_node_summary(self, uuid: str, summary: str, *, group_id: str) -> None:
        """Update the summary text property on an Entity node.

        Args:
            uuid: UUID of the Entity node to update.
            summary: New summary text (may be empty string to clear).
            group_id: Project graph to query.
        """
        graph = self._graph_for(group_id)
        cypher = (
            'MATCH (n:Entity {uuid: $uuid}) '
            'SET n.summary = $summary'
        )
        await graph.query(cypher, {'uuid': uuid, 'summary': summary})

    @_canonicalize_group_args
    async def update_node_name(self, uuid: str, name: str, *, group_id: str) -> None:
        """Update the name property on an Entity node.

        Args:
            uuid: UUID of the Entity node to update.
            name: New name text.
            group_id: Project graph to query.
        """
        graph = self._graph_for(group_id)
        cypher = (
            'MATCH (n:Entity {uuid: $uuid}) '
            'SET n.name = $name'
        )
        await graph.query(cypher, {'uuid': uuid, 'name': name})

    @_canonicalize_group_args
    async def get_edge_text(self, uuid: str, *, group_id: str) -> tuple[str, str]:
        """Return (name, fact) for the RELATES_TO edge with the given UUID.

        Uses ro_query since no writes are performed.

        Raises:
            EdgeNotFoundError: if no edge with that UUID exists.
        """
        graph = self._graph_for(group_id)
        cypher = (
            'MATCH ()-[e:RELATES_TO {uuid: $uuid}]->() '
            'RETURN e.name, e.fact'
        )
        result = await graph.ro_query(cypher, {'uuid': uuid})
        if not result.result_set:
            raise EdgeNotFoundError(f'RELATES_TO edge not found: {uuid}')
        row = result.result_set[0]
        return (row[0] or '', row[1] or '')

    @_canonicalize_group_args
    async def get_edge_invalid_at(self, uuid: str, *, group_id: str) -> Any:
        """Return the raw stored ``invalid_at`` for the RELATES_TO edge with the given UUID.

        Returns ``None`` when the property is null or absent (i.e. the edge is
        active/non-superseded). The value is returned verbatim (no parsing) —
        used by MemoryService.update_edge to verify that a clear_invalid_at
        write actually persisted, independent of the fact-text readback.

        Uses ro_query since no writes are performed.

        Raises:
            EdgeNotFoundError: if no edge with that UUID exists.
        """
        graph = self._graph_for(group_id)
        cypher = (
            'MATCH ()-[e:RELATES_TO {uuid: $uuid}]->() '
            'RETURN e.invalid_at'
        )
        result = await graph.ro_query(cypher, {'uuid': uuid})
        if not result.result_set:
            raise EdgeNotFoundError(f'RELATES_TO edge not found: {uuid}')
        return result.result_set[0][0]

    @_canonicalize_group_args
    async def update_node_embedding(self, uuid: str, embedding: list[float], *, group_id: str) -> None:
        """Update the name_embedding vector for an Entity node using vecf32()."""
        graph = self._graph_for(group_id)
        cypher = (
            'MATCH (n:Entity {uuid: $uuid}) '
            'SET n.name_embedding = vecf32($embedding)'
        )
        await graph.query(cypher, {'uuid': uuid, 'embedding': embedding})

    @_canonicalize_group_args
    async def update_edge_embedding(self, uuid: str, embedding: list[float], *, group_id: str) -> None:
        """Update the fact_embedding vector for a RELATES_TO edge using vecf32()."""
        graph = self._graph_for(group_id)
        cypher = (
            'MATCH ()-[e:RELATES_TO {uuid: $uuid}]->() '
            'SET e.fact_embedding = vecf32($embedding)'
        )
        await graph.query(cypher, {'uuid': uuid, 'embedding': embedding})

    @_canonicalize_group_args
    async def list_indices(self, *, group_id: str) -> list[dict]:
        """Return parsed index records from the graph.

        Uses ro_query since no writes are performed.

        Each record is a dict with keys: label, field, type, entity_type.

        Note on the CALL db.indexes() procedure and the read-only path:
        ``CALL db.indexes()`` is the *only* stored-procedure call sent on the
        read-only path in this file — all other ``ro_query`` callers use plain
        MATCH queries.  Stored procedures are sometimes classified as
        write-capable by graph databases, so this usage was validated
        empirically against FalkorDB module v41800 (4.18.0): the call is
        accepted via ``GRAPH.RO_QUERY`` without error.

        The live verification is pinned in
        ``fused-memory/tests/test_list_indices_integration.py``
        (Task 530 / esc-486-49).  If a future FalkorDB upgrade rejects
        ``CALL`` on the RO path, revert this call to ``graph.query(...)``
        (the write-capable command) and update the integration test to pin
        the new behavior.
        """
        graph = self._graph_for(group_id)
        # CALL db.indexes() is a read-only procedure; FalkorDB accepts it via
        # GRAPH.RO_QUERY (verified via test_list_indices_integration.py).
        result = await graph.ro_query('CALL db.indexes()')
        indices = []
        for row in (result.result_set or []):
            indices.append({
                'label': row[0],
                'field': row[1],
                'type': row[2],
                'entity_type': row[3],
            })
        return indices

    @_canonicalize_group_args
    async def drop_index(self, label: str, field: str, *, group_id: str) -> None:
        """Drop an index on the given label and field (FalkorDB syntax)."""
        graph = self._graph_for(group_id)
        cypher = f'DROP INDEX ON :{label}({field})'
        await graph.query(cypher)

    @_canonicalize_group_args
    async def drop_vector_indices(self, *, group_id: str) -> list[dict]:
        """Drop all VECTOR-type indices in the graph.

        Calls list_indices() to find indices with type == 'VECTOR', then calls
        drop_index() for each.  Returns a list of {'label': ..., 'field': ...}
        dicts for each dropped index.
        """
        indices = await self.list_indices(group_id=group_id)
        dropped: list[dict] = []
        for entry in indices:
            if entry.get('type') == 'VECTOR':
                await self.drop_index(entry['label'], entry['field'], group_id=group_id)
                dropped.append({'label': entry['label'], 'field': entry['field']})
        logger.info(f'Dropped {len(dropped)} VECTOR index(es)')
        return dropped

    async def list_graphs(self) -> list[str]:
        """Enumerate non-empty FalkorDB graphs (excluding default_db)."""
        client = self._require_falkor_client()
        all_graphs = await client.list_graphs()
        return [g for g in all_graphs if g != 'default_db' and not g.endswith('_db')]

    async def node_count(self, graph_name: str) -> int:
        """Count nodes in a specific FalkorDB graph.

        Uses ro_query since no writes are performed.
        """
        graph: Any = self._graph_for(graph_name)
        result = await graph.ro_query('MATCH (n) RETURN count(n) as count')
        return result.result_set[0][0] if result.result_set else 0

    async def close(self) -> None:
        """Shut down the driver."""
        for cloned in self._cloned_drivers.values():
            with contextlib.suppress(Exception):
                await cloned.close()
        self._cloned_drivers.clear()
        if self._driver is not None:
            with contextlib.suppress(Exception):
                await self._driver.close()
        self.client = None
        self._driver = None
        # Per-group clients each hold a driver that aliases an entry already
        # closed above via the _cloned_drivers loop (or self._driver itself) —
        # drop the references only; do NOT call .close() per-client, which
        # would double-close the single shared FalkorDB connection.
        self._group_clients.clear()
        self._llm_client = None
        self._embedder = None
        self._cross_encoder = None
