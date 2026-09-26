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

import concurrent.futures
import contextlib
import importlib
import math
import os
import types
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol


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


#: How to get every arm's dependencies into the worktree venv.
INSTALL_HINT = 'install the eval group: `uv sync --all-packages --group reranker`'


def _require_env(*names: str) -> str:
    """The first of *names* that is set: the one credential probe every remote arm uses."""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    raise ArmUnavailable(SkipReason.no_credential, f'{" / ".join(names)} unset')


def _require_module(name: str) -> types.ModuleType:
    """Import *name* lazily; a missing module is an unavailable arm, not a crash."""
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise ArmUnavailable(
            SkipReason.dependency_unavailable, f'{name} is not importable ({exc}); {INSTALL_HINT}',
        ) from exc


SAME_CLAIM_INSTRUCTION = (
    'Does the CANDIDATE state the same claim as the ENTRY: a restatement or a '
    'rediscovery of it, not merely something on the same topic?'
)
_PAIRWISE_SYSTEM_PROMPT = (
    'You compare two notes from a shared memory store. '
    f'{SAME_CLAIM_INSTRUCTION} Answer with exactly one word: yes or no.'
)

PAIRWISE_MODEL = 'gpt-4o-mini'

#: USD per 1M (input, output) tokens, standard tier
#: (developers.openai.com/api/docs/pricing, read 2026-09-26).
GPT_4O_MINI_PRICE_USD_PER_MTOK = (0.15, 0.60)


def yes_probability(top_logprobs: Iterable[tuple[str, float]]) -> float:
    """p(yes) / (p(yes) + p(no)) over one token's top alternatives, spellings folded."""
    mass = {'yes': 0.0, 'no': 0.0}
    seen: list[str] = []
    for token, logprob in top_logprobs:
        seen.append(token)
        answer = token.strip().lower()
        if answer in mass:
            mass[answer] += math.exp(logprob)
    total = mass['yes'] + mass['no']
    if total == 0.0:
        raise ValueError(f'neither yes nor no among the top tokens {seen!r}')
    return mass['yes'] / total


class PairwiseScorer:
    """One single-token yes/no completion per candidate; a slate's calls are in flight together.

    The score is read from the model's own token distribution, not a
    confidence it writes out.
    """

    def __init__(
        self,
        client: Any,
        *,
        model: str = PAIRWISE_MODEL,
        concurrency: int,
        price_usd_per_mtok: tuple[float, float] = GPT_4O_MINI_PRICE_USD_PER_MTOK,
    ) -> None:
        self._client = client
        self._model = model
        self._price = price_usd_per_mtok
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=concurrency)

    def score(self, entry: str, candidate_texts: Sequence[str]) -> SlateScores:
        answers = list(self._executor.map(lambda text: self._ask(entry, text), candidate_texts))
        return SlateScores(
            scores=tuple(p_yes for p_yes, _ in answers),
            cost_usd=math.fsum(cost for _, cost in answers),
            pairs_over_max_length=None,
        )

    def _ask(self, entry: str, candidate: str) -> tuple[float, float]:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {'role': 'system', 'content': _PAIRWISE_SYSTEM_PROMPT},
                {'role': 'user', 'content': f'ENTRY:\n{entry}\n\nCANDIDATE:\n{candidate}'},
            ],
            temperature=0,
            max_tokens=1,
            logprobs=True,
            top_logprobs=5,
        )
        top = response.choices[0].logprobs.content[0].top_logprobs
        input_price, output_price = self._price
        usage = response.usage
        cost = (usage.prompt_tokens * input_price + usage.completion_tokens * output_price) / 1e6
        return yes_probability((alt.token, alt.logprob) for alt in top), cost

    def facts(self) -> ScorerFacts:
        return ScorerFacts(device='remote:api.openai.com', vram_peak_mib=None, max_length=None)

    def close(self) -> None:
        self._executor.shutdown(wait=True)


@contextlib.contextmanager
def open_pairwise(context: ArmContext) -> Iterator[PairwiseScorer]:
    """gpt-4o-mini over a sync client; the SDK's own retries absorb transient failures."""
    key = _require_env('OPENAI_API_KEY')
    openai = _require_module('openai')
    client = openai.OpenAI(api_key=key, timeout=60.0)
    scorer = PairwiseScorer(client, concurrency=context.pairwise_concurrency)
    try:
        yield scorer
    finally:
        scorer.close()
        client.close()
