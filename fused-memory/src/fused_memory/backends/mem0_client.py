"""Per-project Mem0 AsyncMemory instance manager."""

import asyncio
import logging
from typing import Any

from mem0 import AsyncMemory

from fused_memory.config.env_precedence import warn_if_ambient_base_url_is_overridden
from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.models.scope import Scope

logger = logging.getLogger(__name__)

# The canonical Qdrant payload key for a memory's text, with the historical
# fallbacks. Mirrors memory_service._MEM0_CONTENT_KEYS and
# scripts/clear_malformed_empty_memory._CONTENT_KEYS so "what counts as a
# memory's content" is judged identically everywhere.
_MEM0_TEXT_KEY = 'data'
_MEM0_TEXT_KEYS = (_MEM0_TEXT_KEY, 'memory', 'content')

# How much of a matching record's text to echo back in a scan hit.
_EXCERPT_LEN = 240


def _extract_payload_text(payload: dict[str, Any]) -> str | None:
    """Return a Qdrant payload's memory text, in canonical key order."""
    for key in _MEM0_TEXT_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return None


# ---------------------------------------------------------------------------
# mem0-owned metadata keys
# ---------------------------------------------------------------------------

#: Keys mem0's ``AsyncMemory._update_memory``
#: (site-packages/mem0/memory/main.py, ~line 2449) never trusts from a
#: forwarded metadata dict: ``data``/``hash``/``created_at``/``updated_at``
#: are unconditionally recomputed from the update call's own arguments and
#: the existing stored point, and
#: ``user_id``/``agent_id``/``run_id``/``actor_id``/``role`` are restored
#: from the *currently stored* payload (unconditionally for ``actor_id``;
#: whenever absent from what was forwarded for the rest).  Forwarding stale
#: copies of these works only because mem0 keeps overwriting/re-deriving
#: them -- an implicit coupling to mem0 internals.  Stripping them makes
#: the intent explicit: preserve only a record's CUSTOM provenance keys.
#:
#: DEFENSIVE NOTE (PRD D12 / task 3055 §6, extracted here by task 3195):
#: this module is the DECIDED HOME for this set.  ``memory_metadata.py``,
#: ``scripts/tag_cgl_eta_rehome_scope.py`` (which defined it first) and the
#: in-place ``update_memory`` tool (task 3088, via
#: :func:`split_managed_metadata` below and ``server/tools.py``) bind this
#: same object rather than rebuilding it; task 3195 landed first and task
#: 3088 imports rather than re-extracts.  Never two copies (INV-5) -- a
#: second definition that drifts from this one is exactly the failure D12
#: exists to prevent, and is asserted against by object identity in
#: ``tests/test_memory_metadata.py::TestKeyLayers``.
MEM0_MANAGED_METADATA_KEYS = frozenset({
    'data', 'hash', 'created_at', 'updated_at',
    'user_id', 'agent_id', 'run_id', 'actor_id', 'role',
})


# Payload keys FUSED-MEMORY owns, which a caller-supplied metadata delta must
# not be able to destroy by omission. Distinct from the mem0-owned set above:
# mem0 recomputes-or-restores its own keys, so losing one of those is
# self-healing, whereas nothing restores these.
#
# 'category': Mem0Backend.search pushes it down to Qdrant as a payload filter
# (`filters = {'category': categories[0]}`), and every record carries it —
# MemoryService.add_memory and add_system_record both stamp
# meta['category'] = resolved_category.value before the write. A record that
# loses the key is therefore permanently invisible to every category-scoped
# search, with no error and no other symptom: the point still exists, still
# has its content, and still answers a direct get. That silence is what makes
# the key worth protecting rather than merely documenting.
#
# Protected is not frozen: an explicit metadata_patch={'category': ...} still
# overrides the carried-through value, so deliberate re-categorization works.
# Registering a new key here makes update_memory's replace mode carry it
# through; it lives beside the mem0-owned set so a reader asking "which
# payload keys are protected from a caller-supplied delta?" finds both answers
# in one place (INV-5).
_FUSED_MEMORY_OWNED_METADATA_KEYS = frozenset({'category'})


def split_managed_metadata(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Partition a stored Qdrant payload into (mem0-owned, custom) subsets.

    Unknown keys land in the CUSTOM subset: anything mem0 does not
    recompute-or-restore is provenance the caller is responsible for
    preserving, so the fail-safe direction for an unrecognised key is
    "preserve it", not "drop it".

    Both returned dicts are fresh shallow copies -- callers mutate the custom
    subset in place (the metadata merge/delete arms) and must not disturb the
    payload they read.
    """
    managed: dict[str, Any] = {}
    custom: dict[str, Any] = {}
    for key, value in payload.items():
        if key in MEM0_MANAGED_METADATA_KEYS:
            managed[key] = value
        else:
            custom[key] = value
    return managed, custom


# mem0's own default embedding dimensionality, shared by its Qdrant vector
# store (mem0/configs/vector_stores/qdrant.py) and its OpenAI embedder
# (mem0/embeddings/openai.py). Emitting the embedder key at this value would
# be a behaviour CHANGE, not a no-op — see _build_config_dict.
_MEM0_DEFAULT_EMBEDDING_DIMS = 1536


class Mem0Backend:
    """Lazily creates AsyncMemory instances keyed by project_id."""

    def __init__(self, config: FusedMemoryConfig):
        self.config = config
        self._instances: dict[str, AsyncMemory] = {}
        self._read_timeout: float = config.queue.backend_read_timeout_seconds
        self._write_timeout: float = config.queue.backend_write_timeout_seconds
        self._async_qdrant_client = None  # Lazy async client for count/list ops

    def _build_config_dict(self, collection_name: str) -> dict[str, Any]:
        """Build a Mem0 config dict from the unified config."""
        cfg = self.config
        config_dict: dict[str, Any] = {
            'version': 'v1.1',
            'vector_store': {
                'provider': 'qdrant',
                'config': {
                    'url': cfg.mem0.qdrant_url,
                    'collection_name': collection_name,
                    # Note the deliberate asymmetry with the embedder's
                    # 'embedding_dims' below: DIFFERENT key name, different
                    # upstream semantics, different emission rule. This one
                    # controls the dimensionality Qdrant CREATES the collection
                    # with, and mem0's own default is already 1536
                    # (mem0/configs/vector_stores/qdrant.py), so emitting it
                    # unconditionally is a no-op at the shipped config.
                    #
                    # OPERATOR HAZARD at a NON-1536 embedder.dimensions on an
                    # EXISTING deployment: neither key was plumbed before, so
                    # such a config was inert — mem0 created the collection at
                    # its own 1536 default and requested 1536-wide vectors.
                    # Both now follow the config, but mem0's
                    # QdrantDB.create_col short-circuits when the collection
                    # already exists ('Collection ... already exists. Skipping
                    # creation.'), leaving a 1536-wide collection that every
                    # N-wide upsert then fails against at runtime. Changing
                    # embedder.dimensions requires recreating and re-embedding
                    # the Qdrant collection; it is not a live-migratable knob.
                    'embedding_model_dims': cfg.embedder.dimensions,
                },
            },
        }

        # LLM
        if cfg.llm.provider == 'openai' and cfg.llm.providers.openai:
            # See the openai_base_url comment below: config now wins over the
            # ambient env fallback, which is an egress change for a gateway
            # deployment that set only OPENAI_BASE_URL. Report it, don't
            # redirect silently.
            warn_if_ambient_base_url_is_overridden(
                cfg.llm.providers.openai.api_url, context='mem0 LLM',
            )
            config_dict['llm'] = {
                'provider': 'openai',
                'config': {
                    'model': cfg.llm.model,
                    'temperature': cfg.llm.temperature or 0.1,
                    'max_tokens': cfg.llm.max_tokens,
                    'api_key': cfg.llm.providers.openai.api_key,
                    # Make config authoritative over the ambient
                    # OPENAI_BASE_URL / OPENAI_API_BASE env fallback that
                    # mem0/llms/openai.py would otherwise apply. The default
                    # resolves to https://api.openai.com/v1, so this is
                    # byte-identical for anyone not setting those vars.
                    # OpenAIConfig-only — do NOT add it to the anthropic
                    # branch below, which would TypeError out of mem0's
                    # factory.
                    'openai_base_url': cfg.llm.providers.openai.api_url,
                },
            }
        elif cfg.llm.provider == 'anthropic' and cfg.llm.providers.anthropic:
            config_dict['llm'] = {
                'provider': 'anthropic',
                'config': {
                    'model': cfg.llm.model,
                    'temperature': cfg.llm.temperature or 0.1,
                    'max_tokens': cfg.llm.max_tokens,
                    'api_key': cfg.llm.providers.anthropic.api_key,
                },
            }

        # Embedder
        if cfg.embedder.provider == 'openai' and cfg.embedder.providers.openai:
            warn_if_ambient_base_url_is_overridden(
                cfg.embedder.providers.openai.api_url, context='mem0 embedder',
            )
            embedder_config: dict[str, Any] = {
                'model': cfg.embedder.model,
                'api_key': cfg.embedder.providers.openai.api_key,
                # Read from the EMBEDDER provider block, not the llm one, so
                # the two endpoints stay independent.
                'openai_base_url': cfg.embedder.providers.openai.api_url,
            }
            # Emitted ONLY at a non-default dimensionality. This guard is
            # mandatory, not stylistic: mem0/embeddings/openai.py does
            #     self._pass_dimensions_to_api = self.config.embedding_dims is not None
            # so emitting the key at all — even as 1536 — would start sending a
            # `dimensions` field on EVERY embeddings request under the shipped
            # config, which is not byte-identical. Do not "simplify" this away.
            # (Note the key name: the embedder takes `embedding_dims`;
            # `embedding_model_dims` is the vector store's, set above.)
            if cfg.embedder.dimensions != _MEM0_DEFAULT_EMBEDDING_DIMS:
                embedder_config['embedding_dims'] = cfg.embedder.dimensions
            config_dict['embedder'] = {
                'provider': 'openai',
                'config': embedder_config,
            }

        return config_dict

    async def _get_instance(self, scope: Scope) -> AsyncMemory:
        """Lazily create and cache an AsyncMemory instance for a project."""
        project_id = scope.project_id
        if project_id not in self._instances:
            collection_name = scope.mem0_collection_name(self.config.mem0.collection_prefix)
            config_dict = self._build_config_dict(collection_name)
            instance = AsyncMemory.from_config(config_dict)
            self._instances[project_id] = instance
            logger.info(f'Mem0 instance created for project {project_id} (collection: {collection_name})')
        return self._instances[project_id]

    async def add(
        self,
        content: str,
        scope: Scope,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add a memory to Mem0.

        Callers into this backend (add_memory, classify_and_add, queued
        mem0_add dispatch) all pass already-distilled content per the
        CLAUDE.md contract. We pin ``infer=False`` so Mem0 stores the
        content verbatim and returns the assigned ID, rather than running
        its LLM fact-extractor over the input — which silently drops any
        content the extractor does not classify as a declarative fact
        (normative/procedural/behavioral text) and returns
        ``{'results': []}`` with no error.
        """
        instance = await self._get_instance(scope)
        return await asyncio.wait_for(
            instance.add(
                messages=content,
                user_id=scope.mem0_user_id,
                agent_id=scope.agent_id,
                run_id=scope.session_id,
                metadata=metadata,
                infer=False,
            ),
            timeout=self._write_timeout,
        )

    async def add_system_record(
        self,
        content: str,
        scope: Scope,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Dedup-exempt system-write path (task 2222 / W5-δ).

        This is the dedup-exempt system-write path used by deterministic
        recon-stage mirrors (e.g. the cycle-summary Mem0 mirror). ``infer=False``
        is pinned LOCALLY in this dedicated method — NOT inherited from
        :meth:`add` — so the dedup-exempt guarantee this method exists to
        provide survives any future change to the general ``add()`` (task
        decision #2: "so a future re-enabling of Mem0 dedup can't silently
        re-break recon"). Sharing :meth:`add`'s ``infer=False`` pin instead of
        having its own would mean the day someone flips that pin (or makes it
        config-driven), this path silently re-breaks along with it.

        ``infer=False`` reaches mem0's fresh-uuid ``_create_memory``
        direct-insert primitive and never the update-vs-add branch — dedup
        (similarity search + update-vs-add choice) only exists in mem0's
        ``infer=True`` path, so ``infer=False`` structurally cannot dedup:
        every call unconditionally mints a new id and inserts a fresh point.
        Task 2221 (W5-γ) empirically confirmed this against a real Qdrant.

        MUST NEVER pass ``infer=True`` — doing so would reintroduce the
        similarity-based dedup this method exists to be exempt from.
        """
        instance = await self._get_instance(scope)
        return await asyncio.wait_for(
            instance.add(
                messages=content,
                user_id=scope.mem0_user_id,
                agent_id=scope.agent_id,
                run_id=scope.session_id,
                metadata=metadata,
                infer=False,
            ),
            timeout=self._write_timeout,
        )

    async def search(
        self,
        query: str,
        scope: Scope,
        limit: int = 10,
        categories: list[str] | None = None,
    ) -> dict[str, Any]:
        """Search memories in Mem0.

        Args:
            query: The search query string.
            scope: The project/agent/session scope for this search.
            limit: Maximum number of results to return (default 10).
            categories: Optional list of category names to restrict the search.
                When provided, the filter is pushed down to the Qdrant vector
                store via Mem0's ``filters`` kwarg so that the top-N ranking
                occurs *within* the category-scoped subset.  This prevents the
                false-negative pattern (task 1083) where a matching memory of
                the requested category can be rank-pushed out of the top-N by
                higher-similarity memories belonging to other categories.
                Single-category → ``{'category': 'name'}`` (equality).
                Multi-category → ``{'category': {'in': [...]}}`` (OR match).
        """
        instance = await self._get_instance(scope)
        # Build Qdrant payload filter for category scoping.
        # Pushing the filter down here ensures top-N is computed within the
        # requested category subset, eliminating the false-negative described
        # in task 1083 where post-filtering on an already-truncated top-N
        # silently drops matching memories that ranked below the limit.
        filters: dict[str, Any] | None = None
        if categories and len(categories) == 1:
            filters = {'category': categories[0]}
        elif categories and len(categories) > 1:
            filters = {'category': {'in': list(categories)}}
        try:
            return await asyncio.wait_for(
                instance.search(
                    query=query,
                    user_id=scope.mem0_user_id,
                    agent_id=None,
                    run_id=None,
                    limit=limit,
                    filters=filters,
                ),
                timeout=self._read_timeout,
            )
        except TimeoutError:
            logger.warning(f'Mem0 search timed out after {self._read_timeout}s')
            return {}

    async def get_all(
        self,
        scope: Scope,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Get all memories for a scope."""
        instance = await self._get_instance(scope)
        try:
            return await asyncio.wait_for(
                instance.get_all(
                    user_id=scope.mem0_user_id,
                    agent_id=scope.agent_id,
                    run_id=scope.session_id,
                    limit=limit,
                ),
                timeout=self._read_timeout,
            )
        except TimeoutError:
            logger.warning(f'Mem0 get_all timed out after {self._read_timeout}s')
            return {}

    async def get(self, memory_id: str, scope: Scope) -> dict[str, Any] | None:
        """Get a single memory by ID."""
        instance = await self._get_instance(scope)
        try:
            return await asyncio.wait_for(
                instance.get(memory_id),
                timeout=self._read_timeout,
            )
        except TimeoutError:
            logger.warning(f'Mem0 get timed out after {self._read_timeout}s')
            return None

    async def update(
        self,
        memory_id: str,
        data: str,
        scope: Scope,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Update a memory.

        ``metadata``, when passed, is forwarded to mem0's
        ``AsyncMemory.update``. mem0's ``_update_memory`` OVERWRITES the
        whole Qdrant payload on update -- it starts a fresh payload from
        ``deepcopy(metadata) if metadata else {}`` and only re-preserves
        ``created_at``/session ids from the existing point, so calling this
        with ``metadata=None`` (the default) wipes any custom payload keys
        (e.g. ``kind``/``src_project``/``dst_project``) the point already
        carried. To edit content in place WITHOUT losing that provenance,
        callers must pass the point's full existing metadata payload back
        as ``metadata=``.
        """
        instance = await self._get_instance(scope)
        return await asyncio.wait_for(
            instance.update(memory_id, data, metadata=metadata),
            timeout=self._write_timeout,
        )

    async def set_payload(
        self,
        memory_id: str,
        payload: dict[str, Any],
        scope: Scope,
    ) -> None:
        """Partial-merge *payload* into a point's stored payload. No re-embed.

        Straight to Qdrant's ``set_payload`` — a genuine storage-layer partial
        merge, so pre-existing keys not named in *payload* survive and no
        read-modify-write is needed to compute the result. Deliberately bypasses
        mem0's ``AsyncMemory.update`` (see :meth:`update`), which would re-embed
        the content, rewrite ``updated_at`` and append a mem0 history row for
        what may be a purely cosmetic tag.

        A write timeout PROPAGATES (raises ``TimeoutError``) rather than being
        swallowed into a falsy return — the posture of
        :meth:`get_point_by_id` / :meth:`count_by_metadata`, in deliberate
        contrast to :meth:`get`. A caller must never mistake an unreachable
        Qdrant for a completed write (no-silent-fail invariant).

        NOTE: Qdrant answers ``acknowledged``/``completed`` for an UNKNOWN point
        id — a no-op, not an error. Callers must confirm the point exists (see
        :meth:`get_point_by_id`) before treating a return here as proof that
        anything was written.
        """
        collection_name = scope.mem0_collection_name(self.config.mem0.collection_prefix)
        client = await self._get_async_qdrant()
        await asyncio.wait_for(
            client.set_payload(
                collection_name=collection_name,
                payload=payload,
                points=[memory_id],
            ),
            timeout=self._write_timeout,
        )

    async def delete_payload(
        self,
        memory_id: str,
        keys: list[str],
        scope: Scope,
    ) -> None:
        """Remove exactly *keys* from a point's stored payload. No re-embed.

        Straight to Qdrant's ``delete_payload``; keys not named are untouched.
        Same bypass rationale, same propagating-timeout posture and the same
        unknown-point-id caveat as :meth:`set_payload`.
        """
        collection_name = scope.mem0_collection_name(self.config.mem0.collection_prefix)
        client = await self._get_async_qdrant()
        await asyncio.wait_for(
            client.delete_payload(
                collection_name=collection_name,
                keys=keys,
                points=[memory_id],
            ),
            timeout=self._write_timeout,
        )

    async def overwrite_payload(
        self,
        memory_id: str,
        payload: dict[str, Any],
        scope: Scope,
    ) -> None:
        """Replace a point's ENTIRE stored payload with *payload*. No re-embed.

        Straight to Qdrant's ``overwrite_payload``. Unlike :meth:`set_payload`
        this is NOT a merge: every key absent from *payload* is destroyed. In
        particular the mem0-owned keys (``_MEM0_MANAGED_METADATA_KEYS``) must be
        read back and re-attached by the caller, or the point becomes unreadable
        by mem0's own ``get``/``search``.

        Same bypass rationale, same propagating-timeout posture and the same
        unknown-point-id caveat as :meth:`set_payload`.
        """
        collection_name = scope.mem0_collection_name(self.config.mem0.collection_prefix)
        client = await self._get_async_qdrant()
        await asyncio.wait_for(
            client.overwrite_payload(
                collection_name=collection_name,
                payload=payload,
                points=[memory_id],
            ),
            timeout=self._write_timeout,
        )

    async def delete(self, memory_id: str, scope: Scope) -> dict[str, Any]:
        """Delete a memory."""
        instance = await self._get_instance(scope)
        return await asyncio.wait_for(
            instance.delete(memory_id),
            timeout=self._write_timeout,
        )

    async def _get_async_qdrant(self):
        """Get or create a shared async Qdrant client for lightweight ops."""
        if self._async_qdrant_client is None:
            from qdrant_client import AsyncQdrantClient

            self._async_qdrant_client = AsyncQdrantClient(
                url=self.config.mem0.qdrant_url,
                timeout=int(self._read_timeout),
            )
        return self._async_qdrant_client

    async def count(self, scope: Scope) -> int:
        """Count memories using native async Qdrant count API."""
        collection_name = scope.mem0_collection_name(self.config.mem0.collection_prefix)
        client = await self._get_async_qdrant()
        result = await asyncio.wait_for(
            client.count(
                collection_name=collection_name,
                exact=True,
            ),
            timeout=self._read_timeout,
        )
        return result.count

    async def count_by_metadata(
        self,
        scope: Scope,
        filters: dict[str, Any],
    ) -> int:
        """Deterministic count of memories whose payload matches all *filters*.

        Goes straight to Qdrant's exact-count API with a payload ``Filter`` —
        this is a key-equality lookup, NOT semantic search.  Use this when you
        need a reliable count of memories tagged with specific metadata
        (e.g. persistence markers, escalation markers) rather than a top-N
        similarity ranking that can silently drop matches off the bottom.

        Mem0 stores ``add_memory(metadata=...)`` fields as top-level keys on the
        Qdrant payload, so ``filters={'source': 'X', 'flag_id': 'Y'}`` matches
        against ``payload.source == 'X' AND payload.flag_id == 'Y'``.

        Returns the exact match count.  Empty ``filters`` is rejected (would
        otherwise count every memory in the collection — almost certainly a
        bug at the caller).
        """
        if not filters:
            raise ValueError(
                'count_by_metadata requires at least one filter; '
                'use count() to count all memories in the collection',
            )
        from qdrant_client.http import models as qmodels  # noqa: PLC0415

        collection_name = scope.mem0_collection_name(self.config.mem0.collection_prefix)
        client = await self._get_async_qdrant()
        must: list[qmodels.Condition] = [
            qmodels.FieldCondition(key=k, match=qmodels.MatchValue(value=v))
            for k, v in filters.items()
        ]
        qdrant_filter = qmodels.Filter(must=must)
        result = await asyncio.wait_for(
            client.count(
                collection_name=collection_name,
                count_filter=qdrant_filter,
                exact=True,
            ),
            timeout=self._read_timeout,
        )
        return result.count

    async def scroll_by_metadata(
        self,
        scope: Scope,
        filters: dict[str, Any],
        limit: int = 1000,
        *,
        with_vectors: bool = False,
    ) -> list[dict[str, Any]]:
        """Deterministic enumeration of memories whose payload matches all *filters*.

        Goes straight to Qdrant's scroll API with a payload ``Filter`` —
        a key-equality lookup, NOT semantic search.  Use this when you need
        the full list of memories tagged with specific metadata (e.g. to find
        the oldest pool members for GC) rather than a top-N similarity ranking
        that can silently drop matches off the bottom.

        Closes the standing TODO in task_knowledge_sync.py that requested this
        primitive for GC-correct pool enumeration.

        Mem0 stores ``add_memory(metadata=...)`` fields as top-level keys on the
        Qdrant payload, so ``filters={'recon_pool': 'stage2_cycle_summary'}``
        matches against ``payload.recon_pool == 'stage2_cycle_summary'``.

        Args:
            scope: Project/agent/session scope.
            filters: Non-empty dict of key→value equality filters (same as
                count_by_metadata).  Empty dict is rejected to avoid silently
                enumerating every memory in the collection.
            limit: Maximum number of points to return (default 1000).
            with_vectors: When False (default) only the payload is fetched,
                preserving the payload-only contract every existing caller
                relies on and paying no bandwidth for vectors nobody reads.
                When True, each point's stored vector is fetched and lifted
                onto its result dict under ``'vector'``.  This exists for ANN
                candidate generation (``scripts/audit_duplicate_memories.py``):
                re-using the vector Mem0 already wrote means the caller can
                query Qdrant for neighbours without making a single embedding
                API call, and without risking a second metric space.

        Returns:
            List of dicts ``{'id': ..., 'created_at': ..., 'metadata': {...}}``
            where ``created_at`` is the raw string from the Qdrant payload (or
            ``None`` if absent) and ``metadata`` is the full payload dict.
            When *with_vectors* is True each dict additionally carries
            ``'vector'`` — the stored embedding, or ``None`` if Qdrant
            returned that point without one.  A vector-less point is still
            returned (degraded, not dropped) so the caller can count and
            report it rather than losing it silently; the ``'vector'`` key is
            absent entirely when *with_vectors* is False.

        Raises:
            ValueError: If *filters* is empty.
            TimeoutError: If the Qdrant scroll exceeds the read timeout —
                propagated (NOT swallowed into an empty list), matching
                count_by_metadata, so a timed-out read is never mistaken for
                an empty result (no-silent-fail invariant).
        """
        if not filters:
            raise ValueError(
                'scroll_by_metadata requires at least one filter; '
                'use get_all() to retrieve all memories in the collection',
            )
        from qdrant_client.http import models as qmodels  # noqa: PLC0415

        collection_name = scope.mem0_collection_name(self.config.mem0.collection_prefix)
        client = await self._get_async_qdrant()
        must: list[qmodels.Condition] = [
            qmodels.FieldCondition(key=k, match=qmodels.MatchValue(value=v))
            for k, v in filters.items()
        ]
        qdrant_filter = qmodels.Filter(must=must)
        points, _next_offset = await asyncio.wait_for(
            client.scroll(
                collection_name=collection_name,
                scroll_filter=qdrant_filter,
                with_payload=True,
                with_vectors=with_vectors,
                limit=limit,
            ),
            timeout=self._read_timeout,
        )

        result = []
        for point in points:
            payload: dict[str, Any] = dict(point.payload) if point.payload else {}
            record: dict[str, Any] = {
                'id': point.id,
                'created_at': payload.get('created_at'),
                'metadata': payload,
            }
            if with_vectors:
                # getattr-with-default, not point.vector: Qdrant can return a
                # point carrying no vector at all.  Degrade to None so the
                # caller can count it as a disclosure — raising here would
                # discard an entire otherwise-good scan over one bad point.
                record['vector'] = getattr(point, 'vector', None)
            result.append(record)

        if len(points) == limit:
            logger.warning(
                'Mem0 scroll_by_metadata returned exactly limit=%d points '
                '(collection=%s, filters=%s); results may be truncated — '
                'if the pool exceeds this limit, some members will not be enumerated.',
                limit,
                collection_name,
                filters,
            )

        return result

    async def scan_payload_text(
        self,
        scope: Scope,
        needles: list[str] | None = None,
        *,
        filters: dict[str, Any] | None = None,
        exhaustive: bool = False,
        page_size: int = 256,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Literal substring scan over Qdrant payload TEXT for leaked tool-call XML.

        (a) THIS IS NOT SEMANTIC SEARCH AND NOT METADATA EQUALITY. It is the
        gap that made the corpus unsweepable: :meth:`search` ranks by
        embedding similarity, and a leaked serialized tool-call fragment
        carries almost no semantic signal, so it is unfindable that way (a
        live 2026-07-26 probe returned zero). :meth:`count_by_metadata` /
        :meth:`scroll_by_metadata` match payload KEYS by equality, which
        cannot see inside the memory text at all. This method walks the
        records and applies the shared detector to their content.

        (b) PREFILTER. Unless *exhaustive*, one
        ``FieldCondition(key='data', match=MatchText(text=needle))`` per
        needle is OR-combined via ``Filter(should=[...])`` and pushed to
        Qdrant. On an UN-INDEXED payload field, ``MatchText`` performs a
        literal, case-sensitive, order-preserving SUBSTRING match — measured
        on qdrant 1.17.1 over 19,321 points at ~70-90 ms for an exact count.

        (c) THE PREFILTER IS AN OPTIMISATION ONLY. Every returned record is
        RE-VERIFIED with :func:`find_toolcall_xml_leak`, which is the
        authoritative verdict. This matters because those are un-indexed-field
        FALLBACK semantics: creating a text payload index on ``data`` would
        SILENTLY flip ``MatchText`` to tokenized word-matching. That failure is
        safe here rather than merely unlikely — tokenized matching is strictly
        MORE permissive for these needles (a record containing a literal
        serialized opening tag necessarily contains its constituent word
        tokens), so the prefilter remains a superset and the Python detector
        still yields the exact answer. Only speed degrades, never correctness.
        ``tests/test_mem0_client.py::TestMem0BackendScanPayloadTextIntegration``
        is the tripwire that fails loudly if those semantics ever change.

        (d) ``exhaustive=True`` skips the prefilter entirely and walks the
        whole collection. Use it when the answer must not depend on prefilter
        semantics at all — notably when establishing the TRUE incidence rate,
        so that claim rests on nothing but the shared detector.

        Both modes PAGINATE, looping on the ``next_page_offset`` that
        :meth:`scroll_by_metadata` discards. That method's single-shot
        ``limit=1000`` cap is deliberately not reused: a silently-capped scan
        would report a plausible-looking undercount, which is the same
        silent-wrong-value class this scan exists to measure.

        Args:
            scope: Project/agent/session scope (selects the collection).
            needles: Literal substrings for the prefilter. ``None`` or empty
                defaults to
                :data:`~fused_memory.utils.toolcall_xml_leak.PREFILTER_NEEDLES`
                — never to "no needles", which would scan for nothing and
                report a clean corpus. Ignored when *exhaustive*.
            filters: Optional key→value payload equality filters, AND-ed in
                via ``must`` exactly as :meth:`count_by_metadata` builds them,
                to narrow the scan (e.g. ``{'category': ...}``). Applies in
                both modes.
            exhaustive: Skip the prefilter and walk every point.
            page_size: Points per scroll request.
            limit: Maximum number of points to WALK. Must be strictly positive
                when given. When the walk stops early, ``truncated`` is True
                and a WARNING is logged — the truncation is never silent.

        Raises:
            ValueError: If *limit* is non-positive. A ``limit`` of 0 would make
                every scroll page request 0 points, so the walk would return
                ``{'matches': [], 'scanned': 0, 'truncated': False}`` — a
                result INDISTINGUISHABLE from a genuinely clean corpus, and one
                that a caller's exit-code predicate reads as a complete,
                successful sweep. Rejecting it is the same no-silent-wrong-value
                rule that makes a scan timeout propagate rather than collapse
                into an empty list.

        Returns:
            ``{'matches': [...], 'scanned': int, 'truncated': bool}`` where
            each match is ``{'id', 'created_at', 'matched_fragments',
            'excerpt', 'metadata'}``. ``scanned`` counts every point walked,
            including non-matching ones, so it is a correct denominator for an
            incidence rate.

        Raises:
            TimeoutError: If any scroll page exceeds the read timeout —
                PROPAGATED (never swallowed into an empty result), matching
                count_by_metadata/scroll_by_metadata/get_point_by_id. A
                timed-out scan must never be mistaken for a clean corpus.
        """
        if limit is not None and limit <= 0:
            raise ValueError(
                f'scan_payload_text limit must be strictly positive, got {limit}; a '
                'non-positive limit walks zero points and would report an empty '
                'result as a clean corpus'
            )

        from qdrant_client.http import models as qmodels  # noqa: PLC0415

        from fused_memory.utils.toolcall_xml_leak import (  # noqa: PLC0415
            PREFILTER_NEEDLES,
            find_toolcall_xml_leak,
        )

        collection_name = scope.mem0_collection_name(self.config.mem0.collection_prefix)
        client = await self._get_async_qdrant()

        must: list[qmodels.Condition] = [
            qmodels.FieldCondition(key=k, match=qmodels.MatchValue(value=v))
            for k, v in (filters or {}).items()
        ]
        should: list[qmodels.Condition] = []
        if not exhaustive:
            should = [
                qmodels.FieldCondition(key=_MEM0_TEXT_KEY, match=qmodels.MatchText(text=needle))
                for needle in (needles or PREFILTER_NEEDLES)
            ]
        scroll_filter = qmodels.Filter(must=must, should=should) if (must or should) else None

        matches: list[dict[str, Any]] = []
        scanned = 0
        truncated = False
        offset: Any = None
        while True:
            page_limit = page_size
            if limit is not None:
                page_limit = min(page_size, limit - scanned)
            points, next_offset = await asyncio.wait_for(
                client.scroll(
                    collection_name=collection_name,
                    scroll_filter=scroll_filter,
                    with_payload=True,
                    with_vectors=False,
                    limit=page_limit,
                    offset=offset,
                ),
                timeout=self._read_timeout,
            )

            for point in points:
                scanned += 1
                payload: dict[str, Any] = dict(point.payload) if point.payload else {}
                hits = find_toolcall_xml_leak(_extract_payload_text(payload))
                if not hits:
                    continue
                text = _extract_payload_text(payload) or ''
                matches.append({
                    'id': point.id,
                    'created_at': payload.get('created_at'),
                    'matched_fragments': [hit.fragment for hit in hits],
                    'excerpt': text[:_EXCERPT_LEN] + ('…' if len(text) > _EXCERPT_LEN else ''),
                    'metadata': payload,
                })

            if next_offset is None:
                break
            if limit is not None and scanned >= limit:
                truncated = True
                logger.warning(
                    'Mem0 scan_payload_text stopped at limit=%d (collection=%s, '
                    'exhaustive=%s) with more pages available — results are '
                    'TRUNCATED and any incidence rate derived from them is an '
                    'undercount. Re-run without --limit for the true rate.',
                    limit,
                    collection_name,
                    exhaustive,
                )
                break
            offset = next_offset

        return {'matches': matches, 'scanned': scanned, 'truncated': truncated}

    async def get_point_by_id(self, memory_id: str, scope: Scope) -> dict[str, Any] | None:
        """Direct Qdrant point-fetch by id (non-semantic) → raw payload dict, or None.

        Fetches a single Mem0 record straight from Qdrant's ``retrieve`` API by
        its raw point-id (the Mem0 memory UUID) and returns the FULL raw payload
        dict — bypassing both semantic ranking (``search``) and metadata-equality
        filtering (``count_by_metadata`` / ``scroll_by_metadata``).

        Unlike :meth:`get` (mem0 ``AsyncMemory.get``, which swallows a read
        timeout into ``None``), a Qdrant read-timeout is PROPAGATED (raises
        ``TimeoutError``), never swallowed — mirroring ``count_by_metadata`` /
        ``scroll_by_metadata`` so a timed-out read is never mistaken for a
        genuine not-found (no-silent-fail invariant). That timeout-distinguishing
        behaviour is the whole reason this bypasses ``get``.

        Returns the point's full raw payload dict, or ``None`` when the point is
        absent (empty ``retrieve`` result). A single-id ``retrieve`` returning
        more than one record is not expected; the first is used and a WARNING is
        logged for observability (defense-in-depth, mirroring
        ``scroll_by_metadata``'s truncation warning).
        """
        collection_name = scope.mem0_collection_name(self.config.mem0.collection_prefix)
        client = await self._get_async_qdrant()
        records = await asyncio.wait_for(
            client.retrieve(
                collection_name=collection_name,
                ids=[memory_id],
                with_payload=True,
                with_vectors=False,
            ),
            timeout=self._read_timeout,
        )
        if not records:
            return None
        if len(records) > 1:
            logger.warning(
                'Mem0 get_point_by_id retrieved %d points for a single id '
                '(collection=%s, memory_id=%s); using the first — a single-id '
                'retrieve returning >1 point is unexpected.',
                len(records),
                collection_name,
                memory_id,
            )
        return dict(records[0].payload or {})

    async def close(self) -> None:
        """Close all cached AsyncMemory instances and release their connections."""
        import contextlib
        for instance in self._instances.values():
            with contextlib.suppress(Exception):
                client = getattr(getattr(instance, 'vector_store', None), 'client', None)
                if client is not None and hasattr(client, 'close'):
                    await client.close()
        self._instances.clear()
        if self._async_qdrant_client is not None:
            with contextlib.suppress(Exception):
                await self._async_qdrant_client.close()
            self._async_qdrant_client = None

    async def list_projects(self) -> list[tuple[str, str]]:
        """Enumerate projects by scanning Qdrant collections matching the prefix.

        Returns list of (project_id, collection_name) tuples.
        """
        client = await self._get_async_qdrant()
        prefix = f'{self.config.mem0.collection_prefix}_'
        result = []
        collections = await client.get_collections()
        for c in collections.collections:
            if c.name.startswith(prefix):
                project_id = c.name[len(prefix):]
                if project_id:
                    result.append((project_id, c.name))
        return result
