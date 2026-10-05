"""Test helpers shared by the TaskCurator test modules.

A plain module rather than conftest.py, for the reason ``_fm_helpers.py`` gives.
"""

from __future__ import annotations

from shared.cli_invoke import AgentResult

from fused_memory.config.schema import CuratorConfig, FusedMemoryConfig
from fused_memory.middleware.task_curator import _PoolEntry, is_combine_eligible_status


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


def pool_with_ids(*pairs: tuple[str, str]) -> list[_PoolEntry]:
    """Build a pool with the given (task_id, status) pairs."""
    return [
        _PoolEntry(
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
