"""Test helpers shared by the TaskCurator test modules and the tests that inject a curator.

A plain module rather than conftest.py, for the reason ``_fm_helpers.py`` gives.
"""

from __future__ import annotations

from collections.abc import Iterable
from unittest.mock import AsyncMock, MagicMock

from shared.cli_invoke import AgentResult

from fused_memory.config.schema import CuratorConfig, FusedMemoryConfig
from fused_memory.middleware.curator_types import (
    CandidateTask,
    CuratorDecision,
    PoolEntry,
    PoolWithheld,
    PreparedCandidate,
    empty_pool_sizes,
    is_combine_eligible_status,
)


def make_config() -> FusedMemoryConfig:
    cfg = FusedMemoryConfig()
    cfg.curator = CuratorConfig()
    return cfg


def agent_result(structured: dict | None = None, output: str = '') -> AgentResult:
    """A successful LLM call whose structured verdict is *structured*."""
    return AgentResult(
        success=True,
        output=output,
        structured_output=structured,
        cost_usd=0.01,
    )


def pool_with_ids(*pairs: tuple[str, str]) -> list[PoolEntry]:
    """Build a pool with the given (task_id, status) pairs."""
    return [
        PoolEntry(
            task_id=tid,
            title='t',
            description='',
            details='',
            files_to_modify=[],
            module_keys=[],
            status=status,
            priority='medium',
            source='module',
            combine_eligible=is_combine_eligible_status(status),
        )
        for tid, status in pairs
    ]


def prepared(
    title: str, pool: Iterable[PoolEntry] = (), corpus_error: str | None = None,
) -> PreparedCandidate:
    """A prepared bundle for *title* over *pool*; *corpus_error* marks a failed assembly."""
    entries = list(pool)
    return PreparedCandidate(
        candidate=CandidateTask(title=title),
        pool=entries,
        pool_sizes={**empty_pool_sizes(), 'module': len(entries)},
        prompt_tokens=0,
        corpus_error=corpus_error,
    )


def zot_agent_result() -> AgentResult:
    """An LLM call that hung with zero output until its timeout killed it."""
    return AgentResult(
        success=False, output='', subtype='error_empty_output',
        timed_out=True, turns=0, cost_usd=0.0, duration_ms=181_000,
    )


async def empty_corpus(*_args, **_kwargs) -> tuple[list[PoolEntry], dict[str, int], PoolWithheld]:
    """A ``TaskCurator._build_corpus`` stand-in that assembles an empty pool."""
    return [], empty_pool_sizes(), PoolWithheld()


def escalator_double() -> AsyncMock:
    """A curator escalator whose failure and degraded-streak reports file nothing."""
    escalator = AsyncMock()
    escalator.report_failure = AsyncMock(return_value=None)
    escalator.report_consecutive_degraded = AsyncMock(return_value=None)
    return escalator


def curator_returning(*decisions: CuratorDecision) -> MagicMock:
    """A curator double for ``TaskInterceptor(curator=...)``; ``curate`` returns *decisions*."""
    curator = MagicMock()
    curator.prepare_candidate = AsyncMock(return_value=None)
    curator.curate = AsyncMock(side_effect=list(decisions))
    curator.note_created = MagicMock()
    curator.record_task = AsyncMock()
    curator.reembed_task = AsyncMock(return_value=None)
    curator.close = AsyncMock()
    return curator
