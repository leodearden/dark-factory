#!/usr/bin/env python3
"""The reranker ARMS the ρ1 eval measures, behind one narrow interface.

PRD ``plans/write-triage-flip-readiness-prd.md`` §9 leaf ρ1, decision D1.

An arm is an :class:`ArmSpec`: a stable name, its class, the model it runs, and
an ``open`` factory yielding a :class:`Scorer` for the length of a run. A
scorer answers one question — score these candidates against this entry — and
reports the device it ran on. ``eval_write_triage_reranker.py`` owns everything
done with the scores; this module owns only how they are obtained, and imports
nothing from it.

Every third-party import (torch, sentence-transformers, openai, httpx) happens
inside ``open``, so this module imports on a bare interpreter. An arm whose
dependency or credential is missing raises :class:`ArmUnavailable` there, and
the eval records it as a skipped row rather than crashing.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class ArmClass(StrEnum):
    local_cross_encoder = 'local_cross_encoder'
    llm_pairwise = 'llm_pairwise'
    hosted_api = 'hosted_api'
    jev_choice = 'jev_choice'


class ArmStatus(StrEnum):
    measured = 'measured'
    skipped = 'skipped'


class SkipReason(StrEnum):
    no_credential = 'no_credential'
    dependency_unavailable = 'dependency_unavailable'
    over_budget = 'over_budget'
    error = 'error'


class ArmUnavailable(Exception):
    """An arm that cannot run on this host; *reason* is what the report records."""

    def __init__(self, reason: SkipReason, detail: str) -> None:
        super().__init__(f'{reason}: {detail}')
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class SlateScores:
    """One slate's scores in candidate order, and what scoring it cost.

    ``pairs_over_max_length`` is None when the arm cannot see its own truncation.
    """

    scores: tuple[float, ...]
    cost_usd: float | None
    pairs_over_max_length: int | None


@dataclass(frozen=True)
class ScorerFacts:
    device: str
    vram_peak_mib: float | None
    max_length: int | None


class Scorer(Protocol):
    def score(self, entry: str, candidate_texts: Sequence[str]) -> SlateScores: ...

    def facts(self) -> ScorerFacts: ...


@dataclass(frozen=True)
class ArmContext:
    """The run's knobs every arm reads; each is recorded in the report's provenance."""

    device: str
    local_batch_size: int
    vram_cap_gib: float
    pairwise_concurrency: int


@dataclass(frozen=True)
class ArmSpec:
    name: str
    arm_class: ArmClass
    model: str
    open: Callable[[ArmContext], AbstractContextManager[Scorer]]
