"""Write classifier — routes content to the correct memory category."""

import json
import logging
import re

from openai import AsyncOpenAI

from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.models.enums import MemoryCategory
from fused_memory.models.memory import ClassificationResult
from fused_memory.routing.json_extract import extract_json

logger = logging.getLogger(__name__)

# Keyword patterns per category (from DESIGN.md heuristic pre-filter)
_CATEGORY_PATTERNS: dict[MemoryCategory, list[re.Pattern]] = {
    MemoryCategory.entities_and_relations: [
        re.compile(r'\b(depends on|owns|uses|contains|is a|has a|belongs to|relates to|connected to|part of)\b', re.I),
    ],
    MemoryCategory.temporal_facts: [
        re.compile(r'\b(changed|was|since|before|after|deprecated|updated|migrated|as of|until|from \d{4})\b', re.I),
    ],
    MemoryCategory.decisions_and_rationale: [
        re.compile(r'\b(chose|decided|because|trade-?off|rationale|reasoning|opted for|went with|selected)\b', re.I),
    ],
    MemoryCategory.preferences_and_norms: [
        re.compile(r'\b(prefer|always|never|should|convention|style|standard|rule|must|don\'t|avoid)\b', re.I),
    ],
    MemoryCategory.procedural_knowledge: [
        re.compile(r'\b(to do|steps?|first[\s,].*then|run|execute|process|workflow|procedure|how to)\b', re.I),
    ],
    MemoryCategory.observations_and_summaries: [
        re.compile(r'\b(overall|summary|in general|observation|takeaway|recap|noticed|pattern|trend)\b', re.I),
    ],
}

CLASSIFICATION_SYSTEM_PROMPT = """\
Given a memory extracted from an agent interaction, classify it into exactly one
primary category. If the memory has a strong secondary nature, also identify that.

Categories:
1. entities_and_relations — facts about things and how they connect
2. temporal_facts — state that changes over time, with temporal markers
3. decisions_and_rationale — choices made and why
4. preferences_and_norms — how things should be done, conventions, style
5. procedural_knowledge — how to do things, workflows, steps
6. observations_and_summaries — high-level takeaways, session recaps

Respond as JSON:
{
  "primary": "<category>",
  "secondary": "<category or null>",
  "confidence": <0.0-1.0>,
  "reasoning": "<brief>"
}"""


class WriteClassifier:
    """Classifies content into a MemoryCategory for write routing."""

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

    async def classify(self, content: str) -> ClassificationResult:
        """Classify content — heuristic first, LLM fallback."""
        if self.config.routing.use_heuristics:
            result = self._heuristic_classify(content)
            if result is not None and result.confidence >= self.config.routing.confidence_threshold:
                return result

        if self.config.routing.llm_fallback:
            return await self._llm_classify(content)

        # Pure heuristic mode, below threshold — return best-effort
        result = self._heuristic_classify(content)
        if result is not None:
            return result

        return ClassificationResult(
            primary=MemoryCategory.observations_and_summaries,
            confidence=0.3,
            reasoning='No confident classification; defaulting to observations',
        )

    def _heuristic_classify(self, content: str) -> ClassificationResult | None:
        """Pattern-match on keyword sets per category."""
        scores: dict[MemoryCategory, int] = {}
        for category, patterns in _CATEGORY_PATTERNS.items():
            count = sum(1 for p in patterns if p.search(content))
            if count > 0:
                scores[category] = count

        if not scores:
            return None

        # If only one category matched, high confidence
        if len(scores) == 1:
            cat = next(iter(scores))
            return ClassificationResult(
                primary=cat,
                confidence=0.85,
                reasoning=f'Heuristic: single category match on {cat.value}',
            )

        # Multiple matches — pick top, but low confidence (triggers LLM fallback)
        sorted_cats = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        primary = sorted_cats[0][0]
        secondary = sorted_cats[1][0] if len(sorted_cats) > 1 else None

        return ClassificationResult(
            primary=primary,
            secondary=secondary,
            confidence=0.5,
            reasoning=f'Heuristic: ambiguous — matched {[c.value for c, _ in sorted_cats]}',
        )

    async def _llm_classify(self, content: str) -> ClassificationResult:
        """Use LLM for classification."""
        client = self._get_openai_client()
        try:
            response = await client.chat.completions.create(
                model=self.config.llm.model,
                messages=[
                    {'role': 'system', 'content': CLASSIFICATION_SYSTEM_PROMPT},
                    {'role': 'user', 'content': content[:2000]},
                ],
                temperature=0.0,
                max_tokens=256,
            )
            raw = response.choices[0].message.content or ''

            # Extract JSON from the response (handles nested braces and code fences)
            json_str = extract_json(raw)
            if not json_str:
                logger.warning(f'LLM classification returned no JSON: {raw[:200]}')
                return ClassificationResult(
                    primary=MemoryCategory.observations_and_summaries,
                    confidence=0.4,
                    reasoning='LLM returned non-JSON response',
                )

            data = json.loads(json_str)
            primary = MemoryCategory(data['primary'])
            secondary = MemoryCategory(data['secondary']) if data.get('secondary') else None
            confidence = float(data.get('confidence', 0.7))
            reasoning = data.get('reasoning', '')

            return ClassificationResult(
                primary=primary,
                secondary=secondary,
                confidence=confidence,
                reasoning=f'LLM: {reasoning}',
            )
        except Exception as e:
            logger.error(f'LLM classification failed: {e}')
            return ClassificationResult(
                primary=MemoryCategory.observations_and_summaries,
                confidence=0.3,
                reasoning=f'LLM classification error: {e}',
            )
