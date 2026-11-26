"""The LLM axis's metrics for one arm run, each built from the run's episode outcomes.

One function per metric; ``llm_axis_records`` composes them into MetricsRecords. A
metric with nothing to measure is absent, never null or zero. Missing token usage
on an ok episode raises ``TokenAccountingError`` (INV-4: no silent absorb).
"""

import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from types import MappingProxyType
from typing import TypeVar

from shared.memory_eval_metrics import Metric

from fused_memory.arm_harness.arm_spec import LlmArmSpec, TokenPricing
from fused_memory.arm_harness.conformance import ConformanceCounts, conformance_rate_metric
from fused_memory.arm_harness.frozen_model import FrozenModel
from fused_memory.arm_harness.metrics_record import LlmMetricId, MetricsRecord, record_for
from fused_memory.arm_harness.replay_types import (
    ArmRunResult,
    EpisodeOutcome,
    normalize_entity_name,
)
from fused_memory.arm_harness.retrieval import RETRIEVAL_UTILITY_K, Rank, recall_metric
from fused_memory.backends.llm_token_usage import LlmTokenUsage

GRAPH_SAMENESS_DETAILS_FILENAME = 'graph_sameness_details.json'
LATENCY_PERCENTILES: Mapping[LlmMetricId, float] = MappingProxyType({
    LlmMetricId.EPISODE_LATENCY_P50: 0.50,
    LlmMetricId.EPISODE_LATENCY_P95: 0.95,
})
_Ranked = TypeVar('_Ranked')


class TokenAccountingError(RuntimeError):
    """Ok episodes carry no token usage, so the arm's token and cost metrics are unmeasurable."""

    def __init__(self, episode_ids: tuple[str, ...]) -> None:
        self.episode_ids = episode_ids
        super().__init__(
            f'{len(episode_ids)} ok episode(s) have no recorded LLM token usage: '
            f'{list(episode_ids)}; their tokens are unknown, not zero'
        )


def _ok(outcomes: Sequence[EpisodeOutcome]) -> list[EpisodeOutcome]:
    return [outcome for outcome in outcomes if outcome.ok]


def episode_failure_rate_metric(result: ArmRunResult) -> Metric | None:
    attempted = len(result.outcomes)
    if attempted == 0:
        return None
    failed = sum(1 for outcome in result.outcomes if not outcome.ok)
    return Metric(
        metric_id=LlmMetricId.EPISODE_FAILURE_RATE,
        kind='proportion',
        value=failed / attempted,
        n=attempted,
        denominator=attempted,
        direction='higher_is_worse',
    )


def nearest_rank(sorted_values: Sequence[_Ranked], p: float) -> _Ranked:
    """The value at 1-based rank ceil(p·n): an observed value, deterministic for small n."""
    return sorted_values[max(1, math.ceil(p * len(sorted_values))) - 1]


def latency_metric(metric_id: LlmMetricId, outcomes: Sequence[EpisodeOutcome]) -> Metric | None:
    if metric_id not in LATENCY_PERCENTILES:
        raise ValueError(
            f'{metric_id.value!r} is not a latency metric '
            f'(latency metrics: {", ".join(sorted(LATENCY_PERCENTILES))})'
        )
    durations = sorted(outcome.duration_ms for outcome in _ok(outcomes))
    if not durations:
        return None
    return Metric(
        metric_id=metric_id,
        kind='scalar',
        value=nearest_rank(durations, LATENCY_PERCENTILES[metric_id]),
        n=len(durations),
    )


def ok_token_usages(outcomes: Sequence[EpisodeOutcome]) -> list[LlmTokenUsage]:
    ok = _ok(outcomes)
    unaccounted = tuple(outcome.episode_id for outcome in ok if outcome.tokens is None)
    if unaccounted:
        raise TokenAccountingError(unaccounted)
    return [outcome.tokens for outcome in ok if outcome.tokens is not None]


def tokens_per_episode_metric(outcomes: Sequence[EpisodeOutcome]) -> Metric | None:
    usages = ok_token_usages(outcomes)
    if not usages:
        return None
    return Metric(
        metric_id=LlmMetricId.TOKENS_PER_EPISODE,
        kind='scalar',
        value=sum(usage.total_tokens for usage in usages) / len(usages),
        n=len(usages),
    )


def _usd(usage: LlmTokenUsage, pricing: TokenPricing | None) -> float:
    if pricing is None:
        return 0.0
    return pricing.usd_for(usage.input_tokens, usage.output_tokens)


def usd_per_episode_metric(
    spec: LlmArmSpec, outcomes: Sequence[EpisodeOutcome]
) -> Metric | None:
    usages = ok_token_usages(outcomes)
    if not usages:
        return None
    return Metric(
        metric_id=LlmMetricId.USD_PER_EPISODE,
        kind='scalar',
        value=sum(_usd(usage, spec.pricing) for usage in usages) / len(usages),
        n=len(usages),
    )


class EpisodeSameness(FrozenModel):
    episode_id: str
    arm_entity_count: int
    ref_entity_count: int
    arm_edge_count: int
    ref_edge_count: int
    entity_jaccard: float
    edge_triple_jaccard: float


class GraphSamenessDetails(FrozenModel):
    episodes: tuple[EpisodeSameness, ...]
    excluded_arm_ids: tuple[str, ...]
    excluded_reference_ids: tuple[str, ...]


def _jaccard(left: frozenset[object], right: frozenset[object]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _entity_set(outcome: EpisodeOutcome) -> frozenset[object]:
    return frozenset(normalize_entity_name(name) for name in outcome.entity_names)


def _episode_sameness(arm: EpisodeOutcome, ref: EpisodeOutcome) -> EpisodeSameness:
    arm_entities, ref_entities = _entity_set(arm), _entity_set(ref)
    arm_edges, ref_edges = frozenset(arm.edge_triples), frozenset(ref.edge_triples)
    return EpisodeSameness(
        episode_id=arm.episode_id,
        arm_entity_count=len(arm_entities),
        ref_entity_count=len(ref_entities),
        arm_edge_count=len(arm_edges),
        ref_edge_count=len(ref_edges),
        entity_jaccard=_jaccard(arm_entities, ref_entities),
        edge_triple_jaccard=_jaccard(arm_edges, ref_edges),
    )


def graph_sameness_details(
    outcomes: Sequence[EpisodeOutcome], reference: Sequence[EpisodeOutcome]
) -> GraphSamenessDetails:
    """Per-episode comparison over the episodes ok in both the arm and the reference."""
    arm_ok = {outcome.episode_id: outcome for outcome in _ok(outcomes)}
    ref_ok = {outcome.episode_id: outcome for outcome in _ok(reference)}
    compared = sorted(arm_ok.keys() & ref_ok.keys())
    return GraphSamenessDetails(
        episodes=tuple(_episode_sameness(arm_ok[i], ref_ok[i]) for i in compared),
        excluded_arm_ids=tuple(sorted({o.episode_id for o in outcomes} - set(compared))),
        excluded_reference_ids=tuple(sorted({o.episode_id for o in reference} - set(compared))),
    )


def graph_sameness_metric(
    outcomes: Sequence[EpisodeOutcome], reference: Sequence[EpisodeOutcome]
) -> Metric | None:
    episodes = graph_sameness_details(outcomes, reference).episodes
    if not episodes:
        return None
    return Metric(
        metric_id=LlmMetricId.GRAPH_SAMENESS,
        kind='scalar',
        value=sum(episode.entity_jaccard for episode in episodes) / len(episodes),
        n=len(episodes),
        details_path=GRAPH_SAMENESS_DETAILS_FILENAME,
    )


def llm_axis_records(
    spec: LlmArmSpec,
    result: ArmRunResult,
    conformance: ConformanceCounts,
    *,
    reference: Sequence[EpisodeOutcome] | None,
    retrieval_ranks: Sequence[Rank] | None,
    measured_at: datetime,
) -> tuple[MetricsRecord, ...]:
    metrics = (
        episode_failure_rate_metric(result),
        conformance_rate_metric(conformance),
        *(latency_metric(metric_id, result.outcomes) for metric_id in LATENCY_PERCENTILES),
        tokens_per_episode_metric(result.outcomes),
        usd_per_episode_metric(spec, result.outcomes),
        graph_sameness_metric(result.outcomes, reference) if reference is not None else None,
        recall_metric(LlmMetricId.RETRIEVAL_UTILITY, retrieval_ranks, RETRIEVAL_UTILITY_K)
        if retrieval_ranks is not None
        else None,
    )
    return tuple(
        record_for(spec, metric, measured_at=measured_at, incomplete=result.incomplete)
        for metric in metrics
        if metric is not None
    )
