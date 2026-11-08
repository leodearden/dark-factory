"""The values a replay consumes and produces.

In: ``ReplayItem`` (one corpus episode) and ``ReplaySettings``. Out: one
``EpisodeOutcome`` per attempted episode, an ``ArmAbort`` when INV-4 stops the run,
and the ``ArmRunResult`` that holds them. These are plain data, so any module that
reads replay results imports this one without the engine (``replay.py``) or a backend.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from fused_memory.arm_harness.frozen_model import FrozenModel
from fused_memory.arm_harness.metrics_record import IndexConfiguration
from fused_memory.backends.llm_token_usage import LlmTokenUsage
from fused_memory.config.schema import FusedMemoryConfig


@dataclass(frozen=True)
class ReplayItem:
    episode_id: str
    name: str
    content: str
    source_description: str
    reference_time: datetime


@dataclass(frozen=True)
class ReplaySettings:
    concurrency: int
    episode_timeout_s: float
    index_configuration: IndexConfiguration
    clock: Callable[[], float] = time.perf_counter

    def __post_init__(self) -> None:
        if self.concurrency < 1:
            raise ValueError(f'concurrency must be >= 1, got {self.concurrency}')
        if self.episode_timeout_s <= 0:
            raise ValueError(f'episode_timeout_s must be > 0, got {self.episode_timeout_s}')


def default_replay_settings(
    base_config: FusedMemoryConfig,
    *,
    concurrency: int,
    index_configuration: IndexConfiguration,
) -> ReplaySettings:
    """Settings whose episode budget is the production write timeout, read rather than re-declared."""
    return ReplaySettings(
        concurrency=concurrency,
        episode_timeout_s=base_config.queue.backend_write_timeout_seconds,
        index_configuration=index_configuration,
    )


def normalize_entity_name(name: str) -> str:
    """The form ``EpisodeOutcome.entity_names`` and edge-triple endpoints are compared in."""
    return ' '.join(name.lower().split())


class EpisodeOutcome(FrozenModel):
    episode_id: str
    ok: bool
    error_class: str | None
    duration_ms: float
    tokens: LlmTokenUsage | None
    replay_episode_uuid: str | None
    entity_names: tuple[str, ...]
    edge_triples: tuple[tuple[str, str, str], ...]


class ArmAbort(FrozenModel):
    arm_id: str
    item_ids: tuple[str, ...]
    error_classes: tuple[str, ...]


@dataclass(frozen=True)
class ArmRunResult:
    arm_id: str
    outcomes: tuple[EpisodeOutcome, ...]
    cancelled_ids: tuple[str, ...]
    abort: ArmAbort | None

    @property
    def incomplete(self) -> bool:
        return self.abort is not None
