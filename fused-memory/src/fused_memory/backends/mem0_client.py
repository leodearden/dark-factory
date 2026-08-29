"""Per-project Mem0 AsyncMemory instance manager."""

import asyncio
import logging
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from contextlib import aclosing
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


#: Default cap on how many pages a single full-enumeration scroll may walk.
#: A pager with no budget loops forever if Qdrant keeps handing back a live
#: ``next_offset``, so the budget travels with the loop.  THE single home for
#: this number: ``scripts/census_memory_metadata.DEFAULT_MAX_PAGES`` aliases
#: it rather than restating 200 (INV-5).
DEFAULT_SCROLL_MAX_PAGES = 200


class ScrollPageBudgetExhausted(RuntimeError):
    """A paged scroll consumed *max_pages* with ``next_offset`` still live.

    The stream is TRUNCATED, so the pager raises rather than ending short —
    a caller that folded a short stream into counters would under-report with
    no error surface (INV-10 no-silent-fail-soft).

    ``scripts/census_memory_metadata.CensusScanIncomplete`` is a module-level
    ALIAS of this class (not a subclass): the census's ``except
    CensusScanIncomplete`` must catch exactly what the backend raises.
    """


class ScrollPointBudgetExhausted(RuntimeError):
    """A paged scroll consumed a caller-supplied *max_points* cap.

    Raised, not returned short, for the same reason as its sibling: the
    primitive never truncates silently (INV-10 no-silent-fail-soft).  A caller
    that wants a quiet capped read converts this into its own flag; a caller
    that does not pass ``max_points`` can never see it.

    Deliberately a STANDALONE sibling of :class:`ScrollPageBudgetExhausted` —
    neither is a subclass of the other.  They are different events and the
    split is what lets the CALLER choose the posture per event:

    * *max_points* is a cap the caller explicitly asked for, so being stopped
      by it is an expected outcome that a caller may reasonably fold into a
      ``truncated`` flag (:meth:`Mem0Backend.scan_payload_text` does exactly
      this, keeping the flag-and-WARNING posture it had before its walk moved
      onto the shared pager).
    * *max_pages* is a safety backstop nobody asked for, so exhausting it is
      an error that should keep propagating even in a caller that tolerates
      the first — reporting a backstop truncation as if the caller had asked
      for it would hand a sweep a plausible-looking undercount.

    An inheritance link in either direction would collapse that choice back
    into one, and would additionally change what
    ``scripts/census_memory_metadata``'s ``except CensusScanIncomplete``
    (an alias of the page-budget class) catches — for a cap the census does
    not pass.
    """


def is_missing_collection_error(exc: BaseException) -> bool:
    """True iff *exc* is Qdrant's "that collection doesn't exist" 404.

    A collection that was never provisioned holds no memories, so the honest
    answer for a read against it is an EMPTY RESULT — semantically the same 0
    / ``[]`` that ``task_curator.corpus_count``/``search_corpus`` return for an
    absent collection, and distinct from a failure to read.

    Everything else returns False and MUST keep propagating to the caller's
    error path: a non-404 ``UnexpectedResponse`` (a real backend failure), a
    generic exception, and above all a ``TimeoutError`` — rendering a
    transient failure as "no data" is precisely the silent fail-soft the
    no-silent-fail invariant bans.  Matching is therefore narrow by
    construction (404 AND the message), so a 404 raised about some other
    Qdrant resource can never be read as an empty collection.

    Callers that degrade on this predicate should still say so out loud (log
    the missing collection), so an operator can tell "collection absent" from
    "collection genuinely empty".
    """
    from qdrant_client.http.exceptions import UnexpectedResponse  # noqa: PLC0415

    if not isinstance(exc, UnexpectedResponse) or exc.status_code != 404:
        return False
    content = exc.content
    if isinstance(content, bytes | bytearray):
        content = content.decode('utf-8', errors='replace')
    text = str(content).lower()
    # BOTH tokens, never the phrase alone: Qdrant words several other not-found
    # errors identically ("Snapshot `x` doesn't exist!", alias/shard variants),
    # and only the COLLECTION one means "zero memories".  Requiring 'collection'
    # is what makes the narrowness the docstring promises actually hold.
    # Verified against a live Qdrant scroll on an absent collection:
    # {"status":{"error":"Not found: Collection `x` doesn't exist!"}}.
    return "doesn't exist" in text and 'collection' in text


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

    def _build_payload_filter(
        self,
        filters: dict[str, Any] | None,
        *,
        text_needles: Sequence[str] | None = None,
    ):
        """Build the Qdrant payload ``Filter`` for a key→value equality dict.

        THE single construction site for this filter (INV-5).  Every
        metadata-addressed Qdrant read in this class routes through here:
        :meth:`count_by_metadata`, :meth:`scroll_by_metadata`,
        :meth:`scroll_all_by_metadata` and :meth:`scan_payload_text`.

        The last of those is why the builder carries a second, OPTIONAL arm.
        :meth:`scan_payload_text` narrows by metadata exactly as the other
        three do (``must``) but ALSO pushes a literal substring prefilter down
        to Qdrant (``should``, one ``MatchText`` per needle).  It built that
        combined filter inline until task 3682, which left the "one
        construction site" claim true of three reads and quietly false of the
        fourth — the one whose whole job is measuring a true incidence rate.

        This is a correctness requirement, not tidiness.  The metadata census
        (``scripts/census_memory_metadata.py``) reconciles its SCROLL against
        :meth:`count_by_metadata`'s COUNT to decide ``coverage.complete``.  If
        the two filter constructions ever drifted, that reconciliation would
        silently compare two *different* point sets and still report complete
        coverage — a no-silent-fail violation with no error surface.  The
        sharing is pinned by an equality assertion at each entry point in
        ``tests/test_mem0_client.py::TestMem0BackendPayloadFilterSingleHome``.

        Args:
            filters: Dict of key→value equality filters, AND-ed into ``must``.
                Mem0 stores ``add_memory(metadata=...)`` fields as top-level
                keys on the Qdrant payload, so ``{'source': 'X'}`` matches
                against ``payload.source == 'X'``.  May be empty/``None`` only
                when *text_needles* is non-empty.
            text_needles: Optional literal substrings, OR-ed into ``should`` as
                one ``FieldCondition(key='data', match=MatchText(...))`` each,
                in the given order.  On an UN-INDEXED payload field
                ``MatchText`` is a literal case-sensitive substring match; see
                :meth:`scan_payload_text` for why that is an optimisation only
                and never the authoritative verdict.

        Returns:
            A ``qdrant_client.http.models.Filter``.  ``must`` holds one
            ``FieldCondition``/``MatchValue`` per *filters* item in
            dict-insertion order; ``should`` holds one
            ``FieldCondition``/``MatchText`` per needle in the given order.

            An arm with no members is OMITTED (``None``), never emitted as
            ``[]``.  Not cosmetic: an empty ``should`` is a no-op server-side
            (measured on qdrant 1.17.1 — ``Filter(must=[c])`` and
            ``Filter(must=[c], should=[])`` return the same count), but the two
            are UNEQUAL under pydantic structural equality.  The anti-drift
            assertions that pin this single home compare the filter one entry
            point hands Qdrant against another's with ``==``, so an emitted
            empty arm would make the sharing unprovable at that entry point.

        Raises:
            ValueError: If BOTH arms would be empty — an unfiltered ``Filter``
                selects the WHOLE collection, which is a bug at every caller.
                A ``should``-only filter is fine: it still selects a proper
                subset, so needles-with-no-filters is a legitimate call.
                Callers validate first with their own message naming the right
                unfiltered alternative (``count()`` / ``get_all()``); this
                guard is the backstop so the shared builder can never become
                the hole that lets one through.
        """
        if not filters and not text_needles:
            raise ValueError(
                '_build_payload_filter requires at least one filter or text needle; '
                'an empty filter would select every point in the collection',
            )
        from qdrant_client.http import models as qmodels  # noqa: PLC0415

        must: list[qmodels.Condition] = [
            qmodels.FieldCondition(key=k, match=qmodels.MatchValue(value=v))
            for k, v in (filters or {}).items()
        ]
        should: list[qmodels.Condition] = [
            qmodels.FieldCondition(key=_MEM0_TEXT_KEY, match=qmodels.MatchText(text=needle))
            for needle in (text_needles or ())
        ]
        # `or None` on BOTH arms — the omit-an-unused-arm rule, stated once.
        return qmodels.Filter(must=must or None, should=should or None)

    def _normalise_point(self, point: Any, *, with_vectors: bool) -> dict[str, Any]:
        """Normalise one raw Qdrant point into the standard record dict.

        THE single home for this shape (INV-5), shared by the list-returning
        :meth:`scroll_by_metadata` and the streaming
        :meth:`scroll_all_by_metadata` so the two APIs cannot drift in what a
        record looks like — a caller migrating between them must see
        byte-identical dicts.

        Returns ``{'id', 'created_at', 'metadata'}`` where ``created_at`` is
        the raw string from the payload (or ``None`` if absent) and
        ``metadata`` is the full payload dict.  When *with_vectors* is True
        the record additionally carries ``'vector'``; that key is absent
        entirely when False.
        """
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
        return record

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
        collection_name = scope.mem0_collection_name(self.config.mem0.collection_prefix)
        client = await self._get_async_qdrant()
        qdrant_filter = self._build_payload_filter(filters)
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
        collection_name = scope.mem0_collection_name(self.config.mem0.collection_prefix)
        client = await self._get_async_qdrant()
        qdrant_filter = self._build_payload_filter(filters)
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

        result = [self._normalise_point(point, with_vectors=with_vectors) for point in points]

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
        max_pages: int = DEFAULT_SCROLL_MAX_PAGES,
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

        (e) BOTH MODES PAGINATE, and the walk is DELEGATED to
        :meth:`scroll_collection_pages` rather than re-implemented here — one
        home for the offset/next_offset loop, the per-page read bound and the
        page budget. The caller-supplied *limit* rides on that pager's
        ``max_points``, so the cap is pushed down into each page request
        instead of being layered on top with a ``break``: a capped scan
        therefore costs no look-ahead round-trip to discover there is more.
        :meth:`scroll_by_metadata`'s single-shot ``limit=1000`` cap is
        deliberately not reused: a silently-capped scan would report a
        plausible-looking undercount, which is the same silent-wrong-value
        class this scan exists to measure.

        The two exhaustion postures below are DELIBERATE and caller-chosen,
        not an inconsistency to be tidied away. The pager RAISES on both
        budgets so the primitive can never truncate silently (INV-2); this
        method then catches :class:`ScrollPointBudgetExhausted` and reports it
        as ``truncated=True`` + a WARNING, while
        :class:`ScrollPageBudgetExhausted` propagates. Being stopped by a
        *limit* the caller passed is an expected outcome; being stopped by the
        safety backstop is not, and reporting the latter as if the caller had
        asked for it would hand a sweep a plausible-looking undercount. That
        asymmetry is exactly what invites a well-meaning "unify these two
        paths" fix, which is why it is written down at both sites.

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
            max_pages: Page budget for the underlying walk, forwarded to
                :meth:`scroll_collection_pages`. Unlike *limit*, exhausting it
                RAISES (see below).

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
            ScrollPageBudgetExhausted: If *max_pages* is consumed with more
                pages still available. PROPAGATED rather than folded into
                ``truncated``, which is deliberate: *limit* is a cap the
                caller asked for, so being stopped by it is a normal capped
                result, but the page budget is a safety backstop nobody asked
                for. Reporting a backstop truncation as if the caller had
                requested it would hand a sweep a plausible-looking undercount
                carrying a flag it was not told to expect.

                The numbers plainly, re-measured against live Qdrant on
                2026-08-27 (exact per-collection counts, not the older
                single-collection reading): at the default ``page_size=256``
                and ``max_pages=200`` the ceiling is 51,200 points walked.
                The BINDING collection is ``fused_reify`` at 33,163 points —
                1.54x headroom, i.e. one exhaustive sweep of it already
                consumes 65% of the budget. ``fused_dark_factory``, the sweep
                script's default scope, is 25,635 (2.0x). So the backstop is
                still not reachable today, but the margin is roughly HALF
                what the stale ~19,321-point/2.6x figure implied, and a
                further ~1.5x growth of the largest collection reaches it.
                (The prefiltered mode bounds only MATCHING points, so it is
                the ``exhaustive=True`` walk that meets this ceiling first.)

                *max_pages* is the escape hatch for when the corpus outgrows
                it, but it is a PYTHON-API-level override ONLY: it is not
                exposed by
                ``services/memory_service.py::MemoryService.scan_memory_content``,
                by the ``scan_memory_content`` MCP tool
                (``server/tools.py::scan_memory_content``), or by
                ``scripts/sweep_toolcall_xml_leak.py``, which has no
                ``--max-pages`` flag. The operational sweep therefore always
                runs at the default ceiling; raising it there is a plumbing
                change to those three surfaces, not a flag an operator can
                pass today.
        """
        if limit is not None and limit <= 0:
            raise ValueError(
                f'scan_payload_text limit must be strictly positive, got {limit}; a '
                'non-positive limit walks zero points and would report an empty '
                'result as a clean corpus'
            )

        from fused_memory.utils.toolcall_xml_leak import (  # noqa: PLC0415
            PREFILTER_NEEDLES,
            find_toolcall_xml_leak,
        )

        collection_name = scope.mem0_collection_name(self.config.mem0.collection_prefix)

        # Built at the shared home so an exhaustive scan and the count it is
        # divided by cannot select different point sets (INV-5).  The call is
        # GUARDED rather than unconditional: with neither arm populated
        # (exhaustive, no filters) the builder correctly refuses to make a
        # whole-collection Filter, but a whole-collection walk is exactly what
        # this mode wants — expressed as scroll_filter=None, not as an
        # unfiltered Filter.
        needle_arm = None if exhaustive else list(needles or PREFILTER_NEEDLES)
        scroll_filter = (
            self._build_payload_filter(filters, text_needles=needle_arm)
            if (filters or needle_arm)
            else None
        )

        matches: list[dict[str, Any]] = []
        scanned = 0
        truncated = False
        try:
            # The caller's `limit` rides on the pager's points cap, so the walk
            # itself has ONE home.  There is deliberately no `break` — the cap
            # is expressed as max_points so the pager can shrink each request
            # rather than being stopped from out here.
            #
            # `aclosing` rather than a bare `async for` because `async for`
            # closes the generator only when the ITERATION ends or raises: a
            # failure in the LOOP BODY below (a malformed payload reaching the
            # detector) or a cancellation would otherwise leave the pager
            # suspended mid-walk for the event loop's async-generator hooks to
            # finalise at some later, unpredictable point.  This makes the
            # close deterministic on EVERY exit path.
            async with aclosing(
                self.scroll_collection_pages(
                    collection_name,
                    scroll_filter=scroll_filter,
                    page_size=page_size,
                    max_pages=max_pages,
                    max_points=limit,
                    with_vectors=False,
                )
            ) as pages:
                async for point in pages:
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
        except ScrollPointBudgetExhausted:
            # THIS caller's posture, chosen here rather than at the primitive:
            # being stopped by a limit the caller itself passed is an expected
            # outcome, so it is disclosed as a flag + WARNING instead of an
            # error.  The clause names exactly ONE exception: a broad except
            # would fold a timed-out or page-budget-exhausted walk into a
            # clean-looking capped result.  `scanned` is exact — the pager
            # raises only after the cap-th point has been yielded and consumed.
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

        return {'matches': matches, 'scanned': scanned, 'truncated': truncated}

    async def scroll_collection_pages(
        self,
        collection_name: str,
        *,
        scroll_filter: Any = None,
        page_size: int = 1000,
        max_pages: int = DEFAULT_SCROLL_MAX_PAGES,
        max_points: int | None = None,
        with_vectors: bool = False,
    ) -> AsyncGenerator[Any, None]:
        """Yield every Qdrant point in *collection_name*, paging on ``next_offset``.

        THE single home for the offset/next_offset walk (INV-5).  Every
        paging caller sits on top of it: :meth:`scroll_all_by_metadata`
        (Scope+filter-addressed, normalised records — what
        ``scripts/census_memory_metadata.py`` drives),
        ``scripts/consolidate_namespace_families.merge_collection`` (raw
        points, no filter — which enters at THIS layer) and
        :meth:`scan_payload_text` (a bounded substring scan).

        *max_points* exists precisely so that last one does not need a second
        copy of the walk.  It stops after N POINTS rather than N pages, and
        expressing that as a caller-side ``break`` would both duplicate the
        loop and cost a look-ahead round-trip; owning the cap here makes it
        free.

        The two budgets are DELIBERATELY distinct events with distinct
        exceptions, so the CALLER chooses the posture rather than this
        primitive imposing one.  Both raise here — the pager never truncates
        silently (INV-2) — but :meth:`scan_payload_text` catches only
        :class:`ScrollPointBudgetExhausted`, converting the cap it asked for
        into a ``truncated`` flag while letting the backstop propagate.  Do
        not "unify" the two exceptions or make one inherit from the other:
        that asymmetry is the whole mechanism, and collapsing it would also
        change what ``census_memory_metadata``'s ``except CensusScanIncomplete``
        (an alias of the page-budget class) catches.

        Deliberately collection-name-addressed rather than
        :class:`~fused_memory.models.scope.Scope`-addressed, and
        filter-OPTIONAL, because consolidate_namespace_families scrolls
        LEGACY mis-named collections (``fused_dark-factory``, ``reify_reify``,
        ``autopilot_video_autopilot_video``) that a Scope structurally cannot
        produce, with no filter at all.  Raw points are yielded, not
        normalised records, because its ``merge_collection`` reads
        ``point.id``/``point.vector``/``point.payload`` to rebuild
        ``PointStruct``s.

        Points are yielded one at a time (never accumulated) so a caller
        folding them into counters holds one page in memory regardless of
        collection size.

        Args:
            collection_name: The Qdrant collection, passed through VERBATIM.
            scroll_filter: Optional pre-built payload ``Filter`` (see
                :meth:`_build_payload_filter`).  ``None`` scrolls the whole
                collection — safe here, unlike at the metadata-addressed APIs,
                because the caller named the collection explicitly.
            page_size: Points requested per page (Qdrant ``limit``).
            max_pages: Page budget; exhausting it raises rather than
                truncating.
            max_points: Optional cap on how many points to WALK.  Must be
                strictly positive when given (see Raises).  ``None`` (the
                default) is inert — the walk is bounded only by *max_pages* —
                so every caller that does not opt in is structurally unable to
                see :class:`ScrollPointBudgetExhausted`.

                The cap is pushed down into the per-page request (each page
                asks for ``min(page_size, remaining)``), so a capped walk
                never over-fetches and never pays a look-ahead round-trip to
                discover there is more.  A caller layering its own cap on top
                with ``break`` could not have that: it learns "there is more"
                only by pulling a point past the cap.
            with_vectors: Fetch each point's stored vector.  Costs bandwidth,
                so it is opt-in.

        Yields:
            Raw Qdrant point objects, in page order.  Annotated
            ``AsyncGenerator`` rather than ``AsyncIterator`` on purpose: the
            narrower type carries ``aclose()``, so a caller that may abandon
            the walk part-way (a raise from ITS loop body, a cancellation) can
            wrap it in :func:`contextlib.aclosing` and close it deterministically
            instead of leaving it suspended for the event loop's
            async-generator hooks.  :meth:`scan_payload_text` does exactly that.

        Raises:
            ValueError: If *max_points* is given and non-positive — raised on
                the first iteration, before any scroll is awaited.  A cap of 0
                shrinks every page request to zero points, so a walk ending on
                a ``None`` offset would yield nothing and raise nothing, which
                is indistinguishable from a genuinely empty collection; a
                negative cap would additionally send a negative ``limit`` down
                to the client.  :meth:`scan_payload_text` validates its own
                *limit* first with a message naming that parameter; this guard
                is the backstop, so the shared pager can never become the hole
                that lets one through.
            ScrollPointBudgetExhausted: If *max_points* is consumed while
                ``next_offset`` is still live.  Deliberately a DISTINCT event
                from the page budget below, and checked FIRST when a single
                page exhausts both: being stopped by the cap the caller itself
                set is an expected outcome, and attributing it to an internal
                safety limit would misreport it.  Reaching the cap exactly as
                the stream ends is NOT this event — nothing was left behind,
                so nothing is raised.
            ScrollPageBudgetExhausted: If *max_pages* is consumed while
                ``next_offset`` is still live — the stream is truncated, so it
                raises instead of ending short.
            TimeoutError: If a single page request exceeds ``_read_timeout``.
                PROPAGATED, never swallowed into an empty stream — same
                posture as :meth:`count_by_metadata` /
                :meth:`scroll_by_metadata` / :meth:`get_point_by_id`, so a
                timed-out read is never mistaken for an empty collection.
        """
        if max_points is not None and max_points <= 0:
            raise ValueError(
                f'scroll_collection_pages max_points must be strictly positive, got '
                f'{max_points}; a non-positive cap shrinks every page request to zero '
                'points, so a walk that ends on a None offset yields nothing and raises '
                'nothing — a result indistinguishable from a genuinely empty collection. '
                'Same no-silent-wrong-value rule as scan_payload_text\'s limit guard, '
                'kept here so the shared pager can never become the hole that lets one '
                'through.'
            )

        def _point_budget_exhausted(next_offset: Any) -> ScrollPointBudgetExhausted:
            """Build the points-cap error — ONE home for a message raised twice.

            The cap is checked at two sites (per-yield and post-page) and a
            copied f-string could drift between them.  A ``None`` offset here
            is NOT a contradiction and must not read as one: it means the
            server returned MORE points than the shrunk request asked for, so
            points were still dropped.
            """
            cause = (
                f'with next_offset={next_offset!r} still live'
                if next_offset is not None
                else 'because the server returned more points than the request asked for'
            )
            return ScrollPointBudgetExhausted(
                f'scroll of collection={collection_name!r} exhausted its point budget '
                f'of {max_points} {cause} — the scan is truncated. Raise max_points to '
                'walk further.',
            )

        client = await self._get_async_qdrant()
        offset: Any = None
        pages = 0
        yielded = 0
        while True:
            # Ask for only what is still wanted, so a capped walk never
            # over-fetches and never needs a look-ahead page to discover it
            # has more.
            page_limit = page_size if max_points is None else min(page_size, max_points - yielded)
            # Bound each PAGE, not the whole scan: a per-scan bound would
            # abort a long-but-healthy multi-page enumeration.
            points, next_offset = await asyncio.wait_for(
                client.scroll(
                    collection_name=collection_name,
                    scroll_filter=scroll_filter,
                    with_payload=True,
                    with_vectors=with_vectors,
                    limit=page_limit,
                    offset=offset,
                ),
                timeout=self._read_timeout,
            )
            pages += 1
            for point in points:
                # Enforced per-YIELD, not per-page: a server that ignores the
                # shrunk page_limit still cannot walk the caller past its cap.
                if max_points is not None and yielded >= max_points:
                    raise _point_budget_exhausted(next_offset)
                yield point
                yielded += 1

            # A clean end is checked FIRST: reaching the cap exactly as the
            # stream runs out left nothing behind, so it is not a truncation.
            if next_offset is None:
                return
            # The caller's own cap outranks the safety backstop when one page
            # exhausts both — it is the expected outcome, not an internal limit.
            if max_points is not None and yielded >= max_points:
                raise _point_budget_exhausted(next_offset)
            if pages >= max_pages:
                raise ScrollPageBudgetExhausted(
                    f'scroll of collection={collection_name!r} exhausted its page budget '
                    f'after {pages} page(s) of {page_size} with next_offset={next_offset!r} '
                    f'still live — the scan is truncated. Raise max_pages or page_size.',
                )
            offset = next_offset

    def scroll_all_by_metadata(
        self,
        scope: Scope,
        filters: dict[str, Any],
        *,
        page_size: int = 1000,
        max_pages: int = DEFAULT_SCROLL_MAX_PAGES,
        with_vectors: bool = False,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream EVERY memory matching *filters*, paging until the match set is exhausted.

        The intended primitive for full-enumeration callers.  Same addressing
        as :meth:`scroll_by_metadata` (Scope + non-empty equality filters) and
        the same per-record shape — both normalise through
        :meth:`_normalise_point` — but this one walks Qdrant's
        ``next_offset`` to completion instead of returning a single capped
        page.

        There is deliberately NO ``limit`` and NO truncation warning here:
        this API does not truncate.  Enumeration is exhaustive, and if the
        collection is so large that the page budget runs out it RAISES
        (:class:`ScrollPageBudgetExhausted`) rather than quietly returning a
        short stream.  ``scroll_by_metadata`` is untouched and keeps its
        one-shot capped-list semantics for callers that genuinely want a
        bounded read (``scripts/audit_duplicate_memories.py``).

        Records are yielded one at a time, so a caller folding them into
        counters (``scripts/census_memory_metadata.py``) holds one page in
        memory regardless of corpus size.

        Args:
            scope: Project/agent/session scope — resolves the collection
                exactly as :meth:`scroll_by_metadata` does.
            filters: Non-empty dict of key→value equality filters, built into
                a payload filter by the shared :meth:`_build_payload_filter`
                so this scroll and :meth:`count_by_metadata` provably select
                the same points.
            page_size: Points fetched per Qdrant round-trip.
            max_pages: Page budget for the whole enumeration.
            with_vectors: Lift each point's stored vector onto its record.

        Yields:
            ``{'id', 'created_at', 'metadata'}`` dicts (plus ``'vector'`` when
            *with_vectors*) — identical in shape to
            :meth:`scroll_by_metadata`'s list elements.

        Argument validation is EAGER: the ``ValueError`` (and any error from
        resolving the collection or building the payload filter) raises when
        this method is CALLED, matching the coroutine sibling
        :meth:`scroll_by_metadata`, so a caller that builds the stream and
        then conditionally never iterates it still gets the error. Only the
        paging is deferred — this is a plain ``def`` returning
        :meth:`_scroll_all_records`, because an ``async def`` containing
        ``yield`` would be an async-generator function whose body, guard
        included, does not run until the first ``__anext__``.

        Raises:
            ValueError: If *filters* is empty — at CALL time.
            ScrollPageBudgetExhausted: If the enumeration is truncated by the
                page budget.
            TimeoutError: If a single page exceeds the read timeout —
                propagated, never swallowed into an empty stream.
        """
        if not filters:
            raise ValueError(
                'scroll_all_by_metadata requires at least one filter; '
                'use get_all() to retrieve all memories in the collection',
            )
        collection_name = scope.mem0_collection_name(self.config.mem0.collection_prefix)
        scroll_filter = self._build_payload_filter(filters)
        return self._scroll_all_records(
            collection_name,
            scroll_filter,
            page_size=page_size,
            max_pages=max_pages,
            with_vectors=with_vectors,
        )

    async def _scroll_all_records(
        self,
        collection_name: str,
        scroll_filter: Any,
        *,
        page_size: int,
        max_pages: int,
        with_vectors: bool,
    ) -> AsyncIterator[dict[str, Any]]:
        """Lazy paging half of :meth:`scroll_all_by_metadata`.

        Assumes PRE-VALIDATED arguments — the public method is the sole entry
        point and owns the empty-filters guard, the collection resolution and
        the payload-filter build. See it for the streaming, budget and
        timeout contracts.
        """
        async for point in self.scroll_collection_pages(
            collection_name,
            scroll_filter=scroll_filter,
            page_size=page_size,
            max_pages=max_pages,
            with_vectors=with_vectors,
        ):
            yield self._normalise_point(point, with_vectors=with_vectors)

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
