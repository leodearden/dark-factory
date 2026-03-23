"""Read router — routes queries to the appropriate store(s)."""

import json
import logging
import re

from openai import AsyncOpenAI

from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.models.enums import QueryType, SourceStore
from fused_memory.models.memory import ReadRouteResult
from fused_memory.routing.json_extract import extract_json

logger = logging.getLogger(__name__)

# Heuristic patterns for query routing (from DESIGN.md)
_QUERY_PATTERNS: dict[QueryType, list[re.Pattern]] = {
    QueryType.entity_lookup: [
        re.compile(r'\b(what is|what does|who is|describe|tell me about)\b', re.I),
        re.compile(r'\b(how are .+ and .+ related)\b', re.I),
    ],
    QueryType.temporal: [
        re.compile(r'\b(what changed|when did|before|after|history of|timeline|since when)\b', re.I),
    ],
    QueryType.relational: [
        re.compile(r'\b(what depends on|who owns|what uses|connected to|related to|links between)\b', re.I),
    ],
    QueryType.preference: [
        re.compile(r'\b(how should|what\'s the convention|style for|preference|standard for|rule for)\b', re.I),
    ],
    QueryType.procedural: [
        re.compile(r'\b(how do I|steps to|process for|how to|procedure for|workflow)\b', re.I),
    ],
    QueryType.broad: [
        re.compile(r'\b(overview|summary|recap|everything about|general|all about)\b', re.I),
    ],
}

# Mapping: query type → store routing
_QUERY_TYPE_ROUTING: dict[QueryType, ReadRouteResult] = {
    QueryType.entity_lookup: ReadRouteResult(
        query_type=QueryType.entity_lookup,
        stores=[SourceStore.graphiti, SourceStore.mem0],
        primary_store=SourceStore.graphiti,
    ),
    QueryType.temporal: ReadRouteResult(
        query_type=QueryType.temporal,
        stores=[SourceStore.graphiti],
        primary_store=SourceStore.graphiti,
    ),
    QueryType.relational: ReadRouteResult(
        query_type=QueryType.relational,
        stores=[SourceStore.graphiti, SourceStore.mem0],
        primary_store=SourceStore.graphiti,
    ),
    QueryType.preference: ReadRouteResult(
        query_type=QueryType.preference,
        stores=[SourceStore.mem0, SourceStore.graphiti],
        primary_store=SourceStore.mem0,
    ),
    QueryType.procedural: ReadRouteResult(
        query_type=QueryType.procedural,
        stores=[SourceStore.mem0, SourceStore.graphiti],
        primary_store=SourceStore.mem0,
    ),
    QueryType.broad: ReadRouteResult(
        query_type=QueryType.broad,
        stores=[SourceStore.graphiti, SourceStore.mem0],
        primary_store=SourceStore.graphiti,
    ),
}

ROUTING_SYSTEM_PROMPT = """\
Given a search query from an agent, determine which memory store(s) to search.

Query types:
- entity_lookup: "what is X", "what does X do", "how are X and Y related"
- temporal: "what changed", "when did", "before/after", "history of"
- relational: "what depends on X", "who owns", "what uses"
- preference: "how should I", "what's the convention", "style for"
- procedural: "how do I", "steps to", "process for"
- broad: general topic query, recap, summary

Respond as JSON:
{
  "query_type": "<type>",
  "stores": ["graphiti", "mem0"],
  "primary_store": "graphiti"
}"""


class ReadRouter:
    """Routes read queries to the appropriate store(s)."""

    def __init__(self, config: FusedMemoryConfig):
        self.config = config
        self._openai_client: AsyncOpenAI | None = None

    def _get_openai_client(self) -> AsyncOpenAI:
        if self._openai_client is None:
            cfg = self.config.llm
            api_key = None
            if cfg.providers.openai:
                api_key = cfg.providers.openai.api_key
            self._openai_client = AsyncOpenAI(api_key=api_key)
        return self._openai_client

    async def route(
        self,
        query: str,
        stores_override: list[SourceStore] | None = None,
    ) -> ReadRouteResult:
        """Route a query to stores. If stores_override is set, skip classification."""
        if stores_override:
            return ReadRouteResult(
                query_type=QueryType.broad,
                stores=stores_override,
                primary_store=stores_override[0],
            )

        if self.config.routing.use_heuristics:
            result = self._heuristic_route(query)
            if result is not None:
                return result

        if self.config.routing.llm_fallback:
            return await self._llm_route(query)

        # Default: query both
        return _QUERY_TYPE_ROUTING[QueryType.broad]

    def _heuristic_route(self, query: str) -> ReadRouteResult | None:
        """Pattern-match query to determine routing."""
        matches: list[QueryType] = []
        for query_type, patterns in _QUERY_PATTERNS.items():
            if any(p.search(query) for p in patterns):
                matches.append(query_type)

        if len(matches) == 1:
            return _QUERY_TYPE_ROUTING[matches[0]]

        if not matches:
            return None

        # Ambiguous — fall through to LLM or default to broad
        return None

    async def _llm_route(self, query: str) -> ReadRouteResult:
        """Use LLM to classify the query type."""
        client = self._get_openai_client()
        try:
            response = await client.chat.completions.create(
                model=self.config.llm.model,
                messages=[
                    {'role': 'system', 'content': ROUTING_SYSTEM_PROMPT},
                    {'role': 'user', 'content': query[:1000]},
                ],
                temperature=0.0,
                max_tokens=256,
            )
            raw = response.choices[0].message.content or ''

            json_str = extract_json(raw)
            if not json_str:
                logger.warning(f'LLM routing returned no JSON: {raw[:200]}')
                return _QUERY_TYPE_ROUTING[QueryType.broad]

            data = json.loads(json_str)
            query_type = QueryType(data['query_type'])
            stores = [SourceStore(s) for s in data.get('stores', ['graphiti', 'mem0'])]
            primary_store = SourceStore(data.get('primary_store', 'graphiti'))

            return ReadRouteResult(
                query_type=query_type,
                stores=stores,
                primary_store=primary_store,
            )
        except Exception as e:
            logger.error(f'LLM routing failed: {e}')
            return _QUERY_TYPE_ROUTING[QueryType.broad]
