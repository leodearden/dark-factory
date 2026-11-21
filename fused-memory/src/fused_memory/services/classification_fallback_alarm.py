"""INV-4 storm escape for WriteClassifier fallbacks on the unlabelled write path.

An add_memory write with no category, and every fact extracted from an episode,
is classified by ``routing/classifier.py::WriteClassifier``. When its LLM tier
fails, the write defaults to observations_and_summaries, which lives in Mem0
only. This alarm counts those failures per project and escalates a burst. It
is an alarm, never a rate limiter: no write is blocked.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any

from shared.storm_counter import KeyedStormCounters

from fused_memory.middleware._folded_escalation import file_folded_escalation
from fused_memory.models.enums import LLM_CLASSIFIER_FAILURES, ClassificationFallback

logger = logging.getLogger(__name__)

ANCHOR_TASK_ID = 'write-classifier-fallback-storm'
_AGENT_ROLE = 'fused-memory/write-classifier'
_CATEGORY = 'write_classifier_fallback_storm'
LOG_EVENT = 'write_classifier_fallback_storm'
JOURNAL_PARAM_KEY = 'classification_fallback'
DEFAULT_THRESHOLD = 5
DEFAULT_WINDOW_SECONDS = 3600.0


def journal_params(fallback: ClassificationFallback | None) -> dict[str, str]:
    """The write-journal params entry naming *fallback*; empty for a real classification."""
    return {} if fallback is None else {JOURNAL_PARAM_KEY: fallback.value}


def emit_classification_fallback_storm_escalation(
    project_root: str, *, project_id: str, storm: dict[str, Any],
) -> str | None:
    """File *storm* under the alarm's anchor, folding into an open record. Never raises."""
    count = storm['count']
    window_seconds = storm['window_seconds']
    detail = '\n'.join((
        f'project_id={project_id}',
        f'project_root={project_root}',
        f'count={count}',
        f'threshold={storm["threshold"]}',
        f'window_seconds={window_seconds}',
        f'reasons={",".join(storm["labels"])}',
        '',
        'WriteClassifier could not classify these writes with its LLM, so unlabelled '
        'add_memory writes and episode-derived facts were routed to '
        'observations_and_summaries (Mem0 only). Graphiti-bound memories (entities, '
        'temporal facts, decisions) missed the graph.',
        '',
        "Evidence: write_ops rows with operation='add_memory' whose params carry "
        f"{JOURNAL_PARAM_KEY!r}; each row's result_summary names its memory_ids.",
        '',
        'The writes were not blocked; this alarm only reports.',
        '',
        'This is the ONE open record for this project until it is resolved: later '
        'bursts fold into it and file nothing of their own, so the count above '
        'describes the FIRST burst observed and is not a running total.',
    ))
    return file_folded_escalation(
        project_root,
        anchor_task_id=ANCHOR_TASK_ID,
        agent_role=_AGENT_ROLE,
        category=_CATEGORY,
        severity='blocking',
        summary=(
            f'{count} LLM-classifier fallback(s) in {window_seconds}s for '
            f'project_id={project_id!r} (the FIRST burst observed; later bursts fold '
            'into this record until it is resolved): unlabelled writes were routed '
            'to observations_and_summaries, Mem0 only'
        ),
        detail=detail,
        suggested_action=(
            "Check that the WriteClassifier's OpenAI calls succeed "
            '(llm.providers.openai.api_key, model llm.model in the fused-memory '
            "config; grep the fused-memory log for 'LLM classification failed'). "
            'Then find the affected memories through the write_ops evidence in the '
            'detail and re-write each with an explicit category.'
        ),
        logger=logger,
        log_label=LOG_EVENT,
        context=f'project_id={project_id!r}',
    )


class ClassificationFallbackAlarm:
    """Count LLM-classifier fallbacks per project_id; escalate a burst."""

    def __init__(
        self,
        *,
        threshold: int = DEFAULT_THRESHOLD,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        time_provider: Callable[[], float] = time.time,
    ) -> None:
        self._threshold = threshold
        self._window_seconds = window_seconds
        self._time_provider = time_provider
        self._counters: KeyedStormCounters[str] = KeyedStormCounters()

    @property
    def tracked_projects(self) -> frozenset[str]:
        return self._counters.tracked_keys

    async def record(
        self,
        fallback: ClassificationFallback | None,
        *,
        project_id: str,
        project_root: str | None,
    ) -> str | None:
        """Count *fallback*; return the escalation id iff this call filed or folded one.

        Never raises.
        """
        if fallback is None or fallback not in LLM_CLASSIFIER_FAILURES:
            return None
        storm = self._counters.record(
            project_id,
            threshold=self._threshold,
            window_seconds=self._window_seconds,
            now=self._time_provider(),
            label=fallback.value,
        )
        if storm is None:
            return None
        logger.error(
            '%s: %d LLM-classifier fallback(s) in %ss for project_id=%r (reasons=%s)',
            LOG_EVENT, storm['count'], storm['window_seconds'], project_id,
            ','.join(storm['labels']),
        )
        if not project_root:
            logger.warning(
                '%s: cannot resolve project_root for project_id=%r; NOT escalating '
                '%d LLM-classifier fallback(s). Wire '
                'MemoryService.set_known_projects(build_known_projects_map(...)) at '
                'server startup to restore this alarm.',
                LOG_EVENT, project_id, storm['count'],
            )
            return None
        try:
            # The queue write fsyncs, so it runs off the event loop.
            return await asyncio.to_thread(
                emit_classification_fallback_storm_escalation,
                project_root,
                project_id=project_id,
                storm=storm,
            )
        except Exception:
            logger.exception(
                '%s: filing the escalation for project_id=%r failed', LOG_EVENT, project_id,
            )
            return None
