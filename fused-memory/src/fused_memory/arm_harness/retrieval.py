"""Known-item retrieval math and the arm-graph retrieval-utility probe.

Recall@k and MRR are defined here, with no copy elsewhere in the repo. A known
item on the LLM axis is the replayed episode itself: a search result matches when
its ``episodes`` cite that episode's uuid. Ranks are 1-based, and ``None`` is a miss.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from shared.memory_eval_metrics import Metric

from fused_memory.arm_harness.arm_spec import LlmArmSpec
from fused_memory.arm_harness.metrics_record import EmbeddingMetricId
from fused_memory.arm_harness.replay import ArmGraph
from fused_memory.arm_harness.replay_types import EpisodeOutcome, ReplayItem
from fused_memory.arm_harness.scratch_guard import GuardCheckpoint, require_scratch_name

RETRIEVAL_UTILITY_K = 10

Rank = int | None


def known_item_rank(results: Sequence[Any], is_target: Callable[[Any], bool]) -> Rank:
    return next((rank for rank, result in enumerate(results, start=1) if is_target(result)), None)


def recall_at_k(ranks: Sequence[Rank], k: int) -> tuple[int, int]:
    if k < 1:
        raise ValueError(f'recall@k needs k >= 1, got k={k}')
    _require_one_based(ranks)
    hits = sum(1 for rank in ranks if rank is not None and rank <= k)
    return hits, len(ranks)


def recall_metric(metric_id: str, ranks: Sequence[Rank], k: int) -> Metric | None:
    hits, n = recall_at_k(ranks, k)
    if n == 0:
        return None
    return Metric(
        metric_id=metric_id,
        kind='proportion',
        value=hits / n,
        n=n,
        denominator=n,
        direction='lower_is_worse',
    )


def mrr_metric(ranks: Sequence[Rank]) -> Metric | None:
    _require_one_based(ranks)
    if not ranks:
        return None
    reciprocal = [1 / rank if rank is not None else 0.0 for rank in ranks]
    return Metric(
        metric_id=EmbeddingMetricId.MRR,
        kind='scalar',
        value=sum(reciprocal) / len(ranks),
        n=len(ranks),
    )


def _require_one_based(ranks: Sequence[Rank]) -> None:
    invalid = [rank for rank in ranks if rank is not None and rank < 1]
    if invalid:
        raise ValueError(f'a rank is 1-based or None (a miss); got {invalid}')


def provenance_matcher(replay_episode_uuid: str) -> Callable[[Any], bool]:
    def cites_the_episode(result: Any) -> bool:
        return replay_episode_uuid in getattr(result, 'episodes', ())

    return cites_the_episode


async def probe_retrieval_utility(
    graph: ArmGraph,
    spec: LlmArmSpec,
    outcomes: Sequence[EpisodeOutcome],
    *,
    items: Sequence[ReplayItem],
    k: int = RETRIEVAL_UTILITY_K,
) -> tuple[Rank, ...]:
    """One rank per ok outcome, in outcome order, each searched by its own episode content.

    The query is the corpus episode's own content, so it depends on neither the arm
    nor a reference run.
    """
    require_scratch_name(spec.scratch_group_id, checkpoint=GuardCheckpoint.REPLAY)
    ok = [outcome for outcome in outcomes if outcome.ok]
    content_by_id = _content_for(ok, {item.episode_id: item for item in items})
    target_by_id = _replay_uuids_for(ok)
    ranks: list[Rank] = []
    for outcome in ok:
        results = await graph.search(
            content_by_id[outcome.episode_id],
            group_ids=[spec.scratch_group_id],
            num_results=k,
        )
        ranks.append(known_item_rank(results, provenance_matcher(target_by_id[outcome.episode_id])))
    return tuple(ranks)


def _content_for(
    outcomes: Sequence[EpisodeOutcome], items_by_id: Mapping[str, ReplayItem]
) -> dict[str, str]:
    missing = sorted(o.episode_id for o in outcomes if o.episode_id not in items_by_id)
    if missing:
        raise ValueError(f'ok outcomes without a replay item to query by: {missing}')
    return {o.episode_id: items_by_id[o.episode_id].content for o in outcomes}


def _replay_uuids_for(outcomes: Sequence[EpisodeOutcome]) -> dict[str, str]:
    """Each ok outcome's replayed episode uuid, the known item its search must find."""
    uuids = {o.episode_id: o.replay_episode_uuid for o in outcomes}
    missing = sorted(episode_id for episode_id, uuid in uuids.items() if uuid is None)
    if missing:
        raise ValueError(f'ok outcomes without a replay_episode_uuid to match: {missing}')
    return {episode_id: uuid for episode_id, uuid in uuids.items() if uuid is not None}
