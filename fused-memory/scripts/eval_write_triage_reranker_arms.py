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
import gc
import importlib
import math
import os
import types
import urllib.parse
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Protocol, TypeGuard


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


def _require_modules(*names: str) -> tuple[types.ModuleType, ...]:
    """Import *names* lazily; any missing module makes the arm unavailable, not a crash.

    Every name is tried, so the skip reason lists all that are missing.
    """
    modules: list[types.ModuleType] = []
    missing: list[str] = []
    for name in names:
        try:
            modules.append(importlib.import_module(name))
        except ImportError as exc:
            missing.append(f'{name} ({exc})')
    if missing:
        raise ArmUnavailable(
            SkipReason.dependency_unavailable,
            f'not importable: {"; ".join(missing)}; {INSTALL_HINT}',
        )
    return tuple(modules)


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
    (openai,) = _require_modules('openai')
    client = openai.OpenAI(api_key=key, timeout=60.0)
    scorer = PairwiseScorer(client, concurrency=context.pairwise_concurrency)
    try:
        yield scorer
    finally:
        scorer.close()
        client.close()


def _is_number(value: Any) -> TypeGuard[int | float]:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _remote_facts(url: str) -> ScorerFacts:
    host = urllib.parse.urlsplit(url).hostname
    return ScorerFacts(device=f'remote:{host}', vram_peak_mib=None, max_length=None)


_HTTP_TIMEOUT_SECONDS = 60.0


@contextlib.contextmanager
def _http_client() -> Iterator[Any]:
    (httpx,) = _require_modules('httpx')
    client = httpx.Client(timeout=_HTTP_TIMEOUT_SECONDS)
    try:
        yield client
    finally:
        client.close()


@dataclass(frozen=True)
class HostedRerankAPI:
    """One vendor's rerank endpoint as data; :class:`HostedRerankScorer` serves every vendor."""

    name: str
    model: str
    url: str
    env_vars: tuple[str, ...]
    top_n_field: str
    results_key: str
    cost: Callable[[Mapping[str, Any]], float | None]


def scores_by_index(results: Sequence[Mapping[str, Any]], count: int) -> tuple[float, ...]:
    """Map a vendor's ranked ``{index, relevance_score}`` list back to candidate order.

    Strict: every position in range exactly once with a numeric score, or
    ValueError. A misaligned ranking must never be published as a measurement.
    """
    scores: dict[int, float] = {}
    for item in results:
        index, score = item.get('index'), item.get('relevance_score')
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < count:
            raise ValueError(f'result index {index!r} is not a candidate position below {count}')
        if index in scores:
            raise ValueError(f'result index {index} returned twice')
        if not _is_number(score):
            raise ValueError(f'result {index} has a non-numeric relevance_score {score!r}')
        scores[index] = float(score)
    if len(scores) != count:
        raise ValueError(f'{count - len(scores)} of {count} candidates were never scored')
    return tuple(scores[index] for index in range(count))


class HostedRerankScorer:
    def __init__(self, api: HostedRerankAPI, client: Any, key: str) -> None:
        self._api = api
        self._client = client
        self._key = key

    def score(self, entry: str, candidate_texts: Sequence[str]) -> SlateScores:
        api = self._api
        response = self._client.post(
            api.url,
            headers={'Authorization': f'Bearer {self._key}'},
            json={
                'model': api.model,
                'query': entry,
                'documents': list(candidate_texts),
                api.top_n_field: len(candidate_texts),
            },
        )
        response.raise_for_status()
        body = response.json()
        return SlateScores(
            scores=scores_by_index(body[api.results_key], len(candidate_texts)),
            cost_usd=api.cost(body),
            pairs_over_max_length=None,
        )

    def facts(self) -> ScorerFacts:
        return _remote_facts(self._api.url)


def open_hosted(
    api: HostedRerankAPI,
) -> Callable[[ArmContext], AbstractContextManager[HostedRerankScorer]]:
    @contextlib.contextmanager
    def open_(context: ArmContext) -> Iterator[HostedRerankScorer]:
        key = _require_env(*api.env_vars)
        with _http_client() as client:
            yield HostedRerankScorer(api, client, key)

    return open_


def _unpriced(body: Mapping[str, Any]) -> None:
    """No list price could be confirmed from the vendor's own page; a guess would be fabricated."""
    return None


#: USD per 1M billed tokens for rerank-2.5 (docs.voyageai.com/docs/pricing, read 2026-09-26).
VOYAGE_RERANK_USD_PER_MTOK = 0.05


def _voyage_cost(body: Mapping[str, Any]) -> float | None:
    tokens = (body.get('usage') or {}).get('total_tokens')
    return tokens * VOYAGE_RERANK_USD_PER_MTOK / 1e6 if _is_number(tokens) else None


#: Each vendor's flagship text reranker per its own docs on 2026-09-26. Jina publishes
#: token packages, not a per-token price; Cohere's page lists two rerank prices
#: without naming their models. Both are therefore unpriced.
HOSTED_APIS = (
    HostedRerankAPI(
        name='jina-reranker', model='jina-reranker-v3.5', url='https://api.jina.ai/v1/rerank',
        env_vars=('JINA_API_KEY',), top_n_field='top_n', results_key='results', cost=_unpriced,
    ),
    HostedRerankAPI(
        name='voyage-rerank', model='rerank-2.5', url='https://api.voyageai.com/v1/rerank',
        env_vars=('VOYAGE_API_KEY',), top_n_field='top_k', results_key='data', cost=_voyage_cost,
    ),
    HostedRerankAPI(
        name='cohere-rerank', model='rerank-v4.0-pro', url='https://api.cohere.com/v2/rerank',
        env_vars=('COHERE_API_KEY', 'CO_API_KEY'), top_n_field='top_n', results_key='results',
        cost=_unpriced,
    ),
)


JEV_URL = 'https://api.typesafe.ai/v1/systemone'
JEV_MODEL = 'jev-1.13.0'
JEV_MAX_OPTIONS = 255
#: USD per 1M input tokens; output tokens are free (docs.typesafe.ai/models, read 2026-09-26).
JEV_USD_PER_MTOK_INPUT = 0.042

_JEV_QUESTION = 'same_claim'
_JEV_INSTRUCTIONS = (
    'The state is a new ENTRY and each option is a stored CANDIDATE. Pick the candidate '
    'that states the same claim as the entry: a restatement or a rediscovery of it, not '
    'merely something on the same topic.'
)


def scores_by_option(probabilities: Mapping[str, Any], names: Sequence[str]) -> tuple[float, ...]:
    """A choice answer's name -> probability mapping in option order, as strict as the index map."""
    if set(probabilities) != set(names):
        raise ValueError(
            f'the distribution names {sorted(probabilities)!r}, the question asked {list(names)!r}',
        )
    for name in names:
        if not _is_number(probabilities[name]):
            raise ValueError(f'option {name} has a non-numeric probability {probabilities[name]!r}')
    return tuple(float(probabilities[name]) for name in names)


def _jev_cost(body: Mapping[str, Any]) -> float | None:
    tokens = (body.get('usage') or {}).get('input_tokens')
    return tokens * JEV_USD_PER_MTOK_INPUT / 1e6 if _is_number(tokens) else None


class JevChoiceScorer:
    """TypeSafe Jev: one ``choice`` question whose options are the slate, per docs.typesafe.ai/api.

    Its per-option probabilities are the scores; a choice cannot score a pair
    outside the slate.
    """

    def __init__(self, client: Any, key: str, *, model: str = JEV_MODEL) -> None:
        self._client = client
        self._key = key
        self._model = model

    def score(self, entry: str, candidate_texts: Sequence[str]) -> SlateScores:
        if len(candidate_texts) > JEV_MAX_OPTIONS:
            raise ValueError(
                f'{len(candidate_texts)} candidates exceed a choice\'s {JEV_MAX_OPTIONS} options',
            )
        names = [f'candidate_{position}' for position in range(len(candidate_texts))]
        response = self._client.post(
            JEV_URL,
            headers={'Authorization': f'Bearer {self._key}'},
            json={
                'state': entry,
                'model': self._model,
                'questions': {_JEV_QUESTION: {
                    'type': 'choice',
                    'instructions': _JEV_INSTRUCTIONS,
                    'criteria': dict(zip(names, candidate_texts, strict=True)),
                }},
            },
        )
        response.raise_for_status()
        body = response.json()
        return SlateScores(
            scores=scores_by_option(body['answers'][_JEV_QUESTION]['probabilities'], names),
            cost_usd=_jev_cost(body),
            pairs_over_max_length=None,
        )

    def facts(self) -> ScorerFacts:
        return _remote_facts(JEV_URL)


@contextlib.contextmanager
def open_jev(context: ArmContext) -> Iterator[JevChoiceScorer]:
    key = _require_env('TYPESAFE_API_KEY')
    with _http_client() as client:
        yield JevChoiceScorer(client, key)


def cuda_facts(torch: Any) -> ScorerFacts:
    """The device a local arm ran on and its peak allocation since the last reset."""
    if not torch.cuda.is_available():
        return ScorerFacts(device='cpu', vram_peak_mib=None, max_length=None)
    index = torch.cuda.current_device()
    return ScorerFacts(
        device=f'cuda:{index} ({torch.cuda.get_device_name(index)})',
        vram_peak_mib=torch.cuda.max_memory_allocated(index) / 2**20,
        max_length=None,
    )


def apply_vram_cap(torch: Any, cap_gib: float) -> None:
    """Cap this process's CUDA allocations, so an OOM lands here rather than in a resident service."""
    if not torch.cuda.is_available():
        return
    index = torch.cuda.current_device()
    total = torch.cuda.get_device_properties(index).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, cap_gib * 2**30 / total), index)


class CrossEncoderScorer:
    """A sentence-transformers CrossEncoder scoring (entry, candidate) pairs in micro-batches."""

    def __init__(
        self,
        model: Any,
        token_count: Callable[[str], int],
        *,
        batch_size: int,
        max_length: int,
        device_facts: Callable[[], ScorerFacts],
    ) -> None:
        self._model = model
        self._token_count = token_count
        self._batch_size = batch_size
        self._max_length = max_length
        self._device_facts = device_facts

    def score(self, entry: str, candidate_texts: Sequence[str]) -> SlateScores:
        if self._model is None:
            raise RuntimeError('the cross-encoder scorer is closed; its model was released')
        pairs = [(entry, candidate) for candidate in candidate_texts]
        scores = self._model.predict(pairs, batch_size=self._batch_size, show_progress_bar=False)
        entry_tokens = self._token_count(entry)
        return SlateScores(
            scores=tuple(float(score) for score in scores),
            cost_usd=0.0,
            pairs_over_max_length=sum(
                entry_tokens + self._token_count(candidate) > self._max_length
                for candidate in candidate_texts
            ),
        )

    def facts(self) -> ScorerFacts:
        return replace(self._device_facts(), max_length=self._max_length)

    def close(self) -> None:
        """Drop the model, so a caller still holding the scorer does not keep its weights alive."""
        self._model = None


def _token_counter(tokenizer: Any) -> Callable[[str], int]:
    return lambda text: len(tokenizer(text, add_special_tokens=False)['input_ids'])


def _cpu_facts() -> ScorerFacts:
    return ScorerFacts(device='cpu', vram_peak_mib=None, max_length=None)


def open_cross_encoder(
    model_id: str, max_length: int,
) -> Callable[[ArmContext], AbstractContextManager[CrossEncoderScorer]]:
    """An ``ArmSpec.open`` loading *model_id* with its repo's own published configuration.

    No template is transcribed and no instruction is tuned here: a model that
    sentence-transformers cannot load is a recorded error, not a hand fix.
    """

    @contextlib.contextmanager
    def open_(context: ArmContext) -> Iterator[CrossEncoderScorer]:
        torch, sentence_transformers = _require_modules('torch', 'sentence_transformers')
        device = context.device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        on_cuda = device.startswith('cuda')
        if on_cuda:
            gc.collect()
            torch.cuda.empty_cache()
            apply_vram_cap(torch, context.vram_cap_gib)
            torch.cuda.reset_peak_memory_stats()
        model = sentence_transformers.CrossEncoder(
            model_id, device=device, max_length=max_length,
            model_kwargs={'dtype': torch.bfloat16} if on_cuda else None,
        )
        scorer = CrossEncoderScorer(
            model,
            _token_counter(model.tokenizer),
            batch_size=context.local_batch_size,
            max_length=max_length,
            device_facts=(lambda: cuda_facts(torch)) if on_cuda else _cpu_facts,
        )
        del model
        try:
            yield scorer
        finally:
            scorer.close()
            gc.collect()
            if on_cuda:
                torch.cuda.empty_cache()

    return open_


def _local_arm(name: str, model_id: str, max_length: int) -> ArmSpec:
    return ArmSpec(
        name=name, arm_class=ArmClass.local_cross_encoder, model=model_id,
        open=open_cross_encoder(model_id, max_length),
    )


#: PRD D1's arms in report row order. The names are the stable row keys the
#: report and Γ2's reader see. bge-reranker-v2-m3 runs at 1024 tokens, the
#: length its model card tunes it for, so its pairs_over_max_length is large.
D1_ARMS: tuple[ArmSpec, ...] = (
    _local_arm('qwen3-reranker-0.6b', 'Qwen/Qwen3-Reranker-0.6B', 8192),
    _local_arm('mxbai-rerank-base-v2', 'mixedbread-ai/mxbai-rerank-base-v2', 8192),
    _local_arm('bge-reranker-v2-m3', 'BAAI/bge-reranker-v2-m3', 1024),
    ArmSpec(
        name='gpt-4o-mini-pairwise', arm_class=ArmClass.llm_pairwise, model=PAIRWISE_MODEL,
        open=open_pairwise,
    ),
    *(
        ArmSpec(name=api.name, arm_class=ArmClass.hosted_api, model=api.model, open=open_hosted(api))
        for api in HOSTED_APIS
    ),
    ArmSpec(name='jev-choice', arm_class=ArmClass.jev_choice, model=JEV_MODEL, open=open_jev),
)
