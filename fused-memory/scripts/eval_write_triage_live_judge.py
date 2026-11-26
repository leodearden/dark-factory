#!/usr/bin/env python3
"""How each eval case's verdict is OBTAINED from the SHIPPED write-triage judge.

PRD ``docs/prds/memory-write-path-convergence.md`` §9 leaf γ, decision D10.

:func:`build_judge_fn` (seeded slates) and :func:`build_retrieved_judge_fn`
(retrieved slates) each return a ``judge_fn`` that drives
``server/write_triage_judge.py::judge_write`` with a duck-typed service
exposing only ``config``, so the eval measures the production prompt, the
production provider fan-out and the production parser. Both refuse, through :func:`require_plan_judge_route`, a
config that would judge anywhere but the ChatGPT-plan proxy. The answer is an
``eval_write_triage_scoring.py::JudgeAnswer``, the one thing imported from
there. ``eval_write_triage_judge.py`` owns everything done with the answers;
this module owns only how they are obtained, imports nothing from it, and is
the only eval module that reaches an LLM.
"""
from __future__ import annotations

import asyncio
import types
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fused_memory.models.enums import SourceStore
from fused_memory.models.memory import MemoryResult
from fused_memory.server.write_triage import OUTCOME_JUDGE, BandDecision, JudgeUsage
from fused_memory.server.write_triage_judge import (
    judge_prompt_elisions,
    judge_write,
    resolve_judge_api_key,
    resolve_judge_api_url,
    resolve_judge_field_chars,
    resolve_judge_provider,
)
from fused_memory.utils.script_loader import load_script

_SCRIPTS = Path(__file__).resolve().parent


# Pyright cannot follow the by-path loader, so it alone resolves `_scoring` through
# the scripts/ extraPath; at runtime the module is always the by-path load.
if TYPE_CHECKING:
    import eval_write_triage_scoring as _scoring
else:
    _scoring = load_script(
        _SCRIPTS / 'eval_write_triage_scoring.py', 'eval_write_triage_scoring',
    )


#: Synthetic per-store cosines stamped on the eval's candidates. The eval
#: SUPPLIES its slate rather than retrieving it, but ``select_judge_candidates``
#: orders by ``metadata['store_score']`` and DROPS anything with no numeric
#: one, so a slate carrying no scores would arrive at the model empty.
#:
#: Descending by slate position, which preserves the order
#: ``eval_write_triage_cases.py::build_judge_cases`` chose — the band winner
#: first. These numbers never reach the model: ``build_judge_prompt`` renders
#: id and text only, no metadata at all (PRD C1). They exist solely to survive
#: the selector.
_SYNTHETIC_TOP_SCORE = 0.90
_SYNTHETIC_SCORE_STEP = 0.01

#: Passed to ``judge_write`` to satisfy its signature. It reaches no prompt —
#: C1 forbids repo or task context from reaching the judge, and
#: ``build_judge_prompt`` interpolates no metadata whatsoever.
_EVAL_PROJECT_ID = 'dark_factory'


def _usage_row(usage: JudgeUsage | None) -> dict[str, int] | None:
    """The verdict's usage under the artifact's ESTABLISHED keys, or ``None``.

    The key names are kept because
    ``fused-memory/scripts/eval_write_triage_scoring.py::_spend`` and
    ``fused-memory/scripts/score_write_triage_pairs.py::JudgedCase.from_row``
    read them, and the row carries nothing they do not read. The mapping is
    exact: output (``completion_tokens``) already includes reasoning on both
    OpenAI APIs.
    """
    if usage is None:
        return None
    return {
        'prompt_tokens': usage.input_tokens,
        'completion_tokens': usage.output_tokens,
        'total_tokens': usage.input_tokens + usage.output_tokens,
    }


def _ask_judge(
    service: Any,
    content: str,
    slate: Sequence[Any],
    *,
    outcome: str,
    band_winner_id: str | None,
    similarity: float | None,
) -> _scoring.JudgeAnswer:
    """Drive the SHIPPED judge for one case, or restate a route that decided itself.

    Only a case routed to the judge reaches an LLM, exactly as ``triage_write``
    routes it through ``route_band``: the middle band, plus a restate-band
    write that does not repeat its band winner, or the winner's canonical,
    near-verbatim. A near-verbatim ``restated`` and a below-floor ``stored``
    are returned with no call, no rendered prompt and no spend, so none of the
    three is reported.

    ``asyncio.run`` per case, which is why the runner is synchronous:
    ``judge_write`` is the only awaitable in the pipeline and it needs no
    initialized ``MemoryService``.
    """
    if outcome != OUTCOME_JUDGE:
        return _scoring.JudgeAnswer(outcome=outcome)

    elisions = judge_prompt_elisions(
        content, slate, field_chars=resolve_judge_field_chars(service),
    )
    verdict = asyncio.run(judge_write(
        memory_service=service,
        content=content,
        project_id=_EVAL_PROJECT_ID,
        decision=BandDecision(
            outcome=OUTCOME_JUDGE,
            canonical_id=band_winner_id,
            similarity=similarity,
            t_high=None,
            t_low=None,
        ),
        candidates=list(slate),
    ))
    return _scoring.JudgeAnswer(
        outcome=verdict.outcome,
        verdict=verdict.outcome,
        entry_elided=elisions.entry,
        candidates_elided=elisions.candidates,
        usage=_usage_row(verdict.usage),
        candidate_id=verdict.candidate_id,
    )


def _as_memory_result(record: Mapping[str, Any], metadata: Mapping[str, Any]) -> Any:
    """One eval record as the ``MemoryResult`` the shipped judge consumes."""
    return MemoryResult(
        id=str(record['memory_id']),
        content=str(record['content']),
        source_store=SourceStore.mem0,
        metadata=dict(metadata),
    )


def require_plan_judge_route(config: Any) -> None:
    """Refuse *config* unless its judge has its own endpoint and key.

    An unset ``write_triage.judge_api_url`` or ``judge_api_key`` makes the
    shipped judge inherit ``llm.providers.<provider>``, which bills the API
    credit, while this eval judges only through the ChatGPT-plan proxy (task
    9946; Leo's billing ruling on task 9045). Unset is read through the
    shipped resolvers, so a blank leaf counts as unset exactly as it does in
    production.
    """
    service = types.SimpleNamespace(config=config)
    unset = [
        leaf for leaf, resolved in (
            ('judge_api_url', resolve_judge_api_url(service)),
            ('judge_api_key', resolve_judge_api_key(service)),
        )
        if resolved is None
    ]
    if unset:
        raise ValueError(
            f'write_triage.{" and write_triage.".join(unset)} unset: the judge would '
            f'inherit llm.providers.{resolve_judge_provider(service)} and bill the API '
            'credit (OPENAI_API_KEY), but this eval judges only through the '
            'ChatGPT-plan proxy (task 9946). Set the plan-proxy environment '
            'config.yaml reads both leaves from.',
        )


def build_judge_fn(config: Any) -> Any:
    """The SEEDED live edge: drive the SHIPPED judge, not a re-implementation.

    ``server/write_triage_judge.judge_write`` is called with a duck-typed
    service exposing only ``config`` — which is all it reads — so this eval
    measures the production prompt, the production provider fan-out and the
    production parser. Re-implementing any of the three here would measure
    this script instead of the thing the operator is deciding about.

    The band decision is SYNTHESIZED rather than retrieved: the eval supplies
    its slate directly, so there is no retrieval to derive a winner from. It
    names the cluster canonical, which is exactly what a correct retrieval
    would have found, and marks the middle band — the only band that reaches a
    judge at all.
    """
    require_plan_judge_route(config)
    service = types.SimpleNamespace(config=config)

    def judge_fn(
        case: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]],
    ) -> _scoring.JudgeAnswer:
        slate = [
            _as_memory_result(record, {
                'store_score': _SYNTHETIC_TOP_SCORE - index * _SYNTHETIC_SCORE_STEP,
            })
            for index, record in enumerate(candidates)
        ]
        return _ask_judge(
            service,
            str(case['content']),
            slate,
            outcome=OUTCOME_JUDGE,
            band_winner_id=slate[0].id if slate else None,
            similarity=_SYNTHETIC_TOP_SCORE,
        )

    return judge_fn


def build_retrieved_judge_fn(config: Any) -> Any:
    """The RETRIEVED live edge: the case already carries production's routing.

    The slate, the band winner, the route and the similarity all came from
    ``route_band``/``select_judge_candidates`` over a real retrieval, so this
    only rebuilds the records and defers to the same judge edge. A case
    reaches the LLM when ``route_band`` routed it to the judge: the middle
    band, plus a restate-band write that does not repeat its band winner, or
    the winner's canonical, near-verbatim. The candidate metadata is passed
    through whole because it carries the cosine the shipped selector re-reads.
    """
    require_plan_judge_route(config)
    service = types.SimpleNamespace(config=config)

    def judge_fn(
        case: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]],
    ) -> _scoring.JudgeAnswer:
        slate = [
            _as_memory_result(record, record.get('metadata') or {})
            for record in candidates
        ]
        return _ask_judge(
            service,
            str(case['content']),
            slate,
            outcome=str(case['routed']),
            band_winner_id=case['band_winner_id'],
            similarity=case['similarity'],
        )

    return judge_fn
