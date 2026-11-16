"""The ``add_memory`` write-path harness shared by the triage, near-duplicate,
entities and completion-claim tool suites.

* ``mock_service.config`` is a plain ``types.SimpleNamespace``. An unspecced
  ``AsyncMock``'s attribute chain auto-generates a truthy ``Mock`` for every
  hop, so ``config.write_triage.enabled`` would read as a Mock rather than a
  bool — which is exactly the shape the resolvers are built to REFUSE, and the
  test would silently exercise the flag-off path while claiming otherwise.

A plain module rather than a cross-suite import or a conftest fixture, for the
reasons ``tests/reconciliation/consolidator_fixtures.py`` records.
"""

from __future__ import annotations

import types
from unittest.mock import AsyncMock

from fused_memory.config.schema import ProceduralTopicCluster
from fused_memory.models.enums import MemoryCategory, SourceStore
from fused_memory.models.memory import MemoryResult
from fused_memory.services.memory_service import RRF_K

_CANONICAL_CONTENT = (
    'Do not use git stash in any dark-factory worktree — refs/stash is shared '
    'across every checkout, not per-worktree.'
)

# Calibrated bands for the harness. Chosen so the importing suites' fixtures sit
# unambiguously inside one band each, rather than near a boundary where a
# rounding change would silently reclassify them.
T_HIGH = 0.90
T_LOW = 0.70

#: Content matching the phrases of the cluster `topic_cluster` builds.
TOPIC_MATCH_CONTENT = 'run create_plan against the missing plan-tools MCP server'


def search_candidate(
    id_: str = 'm1',
    score: float = 0.97,
    content: str = _CANONICAL_CONTENT,
    category: MemoryCategory = MemoryCategory.procedural_knowledge,
    store_rank: int = 1,
    extra_metadata: dict | None = None,
) -> MemoryResult:
    """Build the POST-RRF result shape the tool really receives from search().

    *score* is the Mem0 COSINE and lands in ``metadata['store_score']``;
    ``relevance_score`` carries the ordinal RRF value, deliberately unrelated
    to it, so a regression that bands on the RRF ordinal fails here rather
    than silently disabling triage for every input (task 3658).

    ``RRF_K`` comes from production rather than the literal 60, so a retune
    carries this fixture with it.

    ``extra_metadata`` is merged in last and is how a CHILD candidate's shape
    (``{'kind': ..., 'parent_id': ...}``) is built — children arrive from
    ``search`` as ordinary hits, because grouping is applied at the MCP
    boundary and never inside the service.
    """
    return MemoryResult(
        id=id_,
        content=content,
        category=category,
        source_store=SourceStore.mem0,
        relevance_score=1.0 / (RRF_K + store_rank),
        metadata={
            'store_rank': store_rank,
            'store_score': score,
            **(extra_metadata or {}),
        },
    )


def configure_triage_config(
    mock_service: AsyncMock,
    *,
    enabled: bool = True,
    candidate_k: int = 20,
    t_high: float | None = T_HIGH,
    t_low: float | None = T_LOW,
    near_dup_guard_enabled: bool = True,
    near_dup_threshold: float = 0.90,
    topic_clusters: list | None = None,
    judge_enabled: bool = False,
    min_parent_cosine: float | None = None,
    hoist_max_cosine_gap: float | None = None,
) -> None:
    """Stand in for ``memory_service.config`` with plain namespaces.

    Carries BOTH sections: ``write_triage`` for the new path and
    ``reconciliation`` for the two reject guards it supersedes, so a single
    harness can assert the flag-on and flag-off behaviours against identical
    config in every other respect.

    ``judge_enabled`` DEFAULTS FALSE, diverging from production's default-True
    on purpose. Once `tools.py` passes the real `judge_write` into
    `triage_write`, every middle-band write in an importing suite reaches an LLM
    provider — and `OPENAI_API_KEY` is set in this environment, so those calls
    would SUCCEED: a unit suite silently billing an account and depending on a
    network round-trip for a routing assertion. Leaving the attribute off the
    namespace entirely is what would do that, because the resolver defaults it
    True when absent. Tests that mean to exercise the judge patch
    `tools.judge_write` outright (see
    `tests/server/test_add_memory_write_triage_gate.py::TestTheRealJudgeIsWiredAtTheToolSeam`),
    which is both cheaper and a stronger assertion than a live call.

    The attach-guard knobs (task 7158) DEFAULT None, i.e. both rules off, so an
    importing suite stays on pre-guard routing unless it opts in.
    """
    mock_service.config = types.SimpleNamespace(
        write_triage=types.SimpleNamespace(
            enabled=enabled,
            candidate_k=candidate_k,
            t_high=t_high,
            t_low=t_low,
            judge_enabled=judge_enabled,
            min_parent_cosine=min_parent_cosine,
            hoist_max_cosine_gap=hoist_max_cosine_gap,
        ),
        reconciliation=_reject_guard_section(
            near_dup_guard_enabled, near_dup_threshold, topic_clusters
        ),
    )


def configure_reject_guard_config(
    mock_service: AsyncMock,
    *,
    near_dup_guard_enabled: bool,
    near_dup_threshold: float,
    topic_clusters: list | None = None,
) -> None:
    """Stand in for ``memory_service.config`` with the ``reconciliation`` section ONLY.

    ``write_triage`` is left off the namespace, so its resolver reads the
    production default (off) and the two reject guards are the live path. This
    is the shape for a suite about the guards themselves; use
    ``configure_triage_config`` to compare them against triage.
    """
    mock_service.config = types.SimpleNamespace(
        reconciliation=_reject_guard_section(
            near_dup_guard_enabled, near_dup_threshold, topic_clusters
        ),
    )


def _reject_guard_section(
    near_dup_guard_enabled: bool,
    near_dup_threshold: float,
    topic_clusters: list | None,
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        procedural_knowledge_near_dup_guard_enabled=near_dup_guard_enabled,
        procedural_knowledge_near_dup_threshold=near_dup_threshold,
        procedural_knowledge_topic_guard_clusters=topic_clusters or [],
    )


def topic_cluster(
    topic_id: str = 'test-topic',
    phrases: tuple[str, ...] = ('plan-tools', 'create_plan'),
    min_phrase_hits: int = 2,
    hint: str = '',
) -> ProceduralTopicCluster:
    return ProceduralTopicCluster(
        topic_id=topic_id,
        phrases=list(phrases),
        min_phrase_hits=min_phrase_hits,
        hint=hint,
    )
