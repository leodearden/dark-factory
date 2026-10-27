"""Sidecar records and a non-judging shadow for tests over a real overreach sidecar.

Three test modules write judged episodes into a real ``OverreachShadowStore``:
the store and runner tests, the Stage 1 flag-handoff tests, and the periodic
checkpoint tests. They build the same ``JudgedEpisode`` and
``(JudgeEdge, EdgeJudgement)`` records from here, so a change to either
record's constructor is one edit. Imported as
``from _overreach_shadow_fixtures import ...``, which follows the
``tests/_*.py`` helper convention described in ``tests/_graphiti_fake.py``.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from fused_memory.config.schema import OverreachGuardConfig
from fused_memory.services.overreach_guard import PROMPT_VERSION, EdgeJudgement, JudgeEdge
from fused_memory.services.overreach_shadow import (
    EdgeReader,
    JudgedEpisode,
    OverreachShadow,
    OverreachShadowStore,
    ShadowOutcome,
)

JUDGE_MODEL = 'judge-m'


def judged_episode(uuid: str = 'ep1', *, graph: str = 'g', outcome: ShadowOutcome = ShadowOutcome.judged,
                   minted: int = 2, error: str | None = None) -> JudgedEpisode:
    return JudgedEpisode(
        graph=graph, episode_uuid=uuid, judge_model=JUDGE_MODEL, prompt_version=PROMPT_VERSION,
        classifiers=('category_decisions', 'ruling_lexeme_head'), minted_count=minted,
        outcome=outcome, error=error,
    )


def judged_edge(edge_uuid: str, label: str, *, fact: str = '') -> tuple[JudgeEdge, EdgeJudgement]:
    return (
        JudgeEdge(uuid=edge_uuid, fact=fact or f'fact of {edge_uuid}', source_name='S', target_name='T'),
        EdgeJudgement(edge_uuid=edge_uuid, label=label, rationale=f'span for {edge_uuid}'),
    )


class NeverJudges:
    """An LLM double for a path that must never judge: any call fails the test."""

    async def generate_response(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError('this path must never judge an edge')


async def no_surviving_edges(uuids: Sequence[str], *, group_id: str) -> list[dict[str, Any]]:
    return []


def non_judging_shadow(store: OverreachShadowStore,
                       read_edges: EdgeReader = no_surviving_edges) -> OverreachShadow:
    return OverreachShadow(store, NeverJudges(), JUDGE_MODEL, read_edges, lambda: OverreachGuardConfig())
