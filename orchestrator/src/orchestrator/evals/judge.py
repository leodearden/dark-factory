"""LLM pairwise blinded comparison of eval results."""

from __future__ import annotations

import json
import logging
import random
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

from orchestrator.agents.invoke import invoke_agent

from .elo import (
    MAX_PER_PAIR,
    TaskPool,
    _pair_key,
    ensure_config_in_pool,
    next_matchup,
    record_match,
)
from .snapshots import get_diff

logger = logging.getLogger(__name__)


@dataclass
class JudgeVerdict:
    """Result of one pairwise comparison."""

    task_id: str
    config_a: str
    config_b: str
    winner: str       # 'A', 'B', or 'tie'
    confidence: float
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


JUDGE_SCHEMA = {
    'type': 'object',
    'properties': {
        'winner': {'type': 'string', 'enum': ['A', 'B', 'tie']},
        'confidence': {'type': 'number', 'minimum': 0.0, 'maximum': 1.0},
        'reasoning': {'type': 'string'},
    },
    'required': ['winner', 'confidence', 'reasoning'],
}


def _strip_metadata(diff: str) -> str:
    """Strip identifying metadata from a diff to prevent judge bias."""
    # Remove session IDs, timestamps, model names
    cleaned = re.sub(r'session[_-]?id[:\s]+\S+', '', diff, flags=re.IGNORECASE)
    cleaned = re.sub(r'claude|codex|gemini|opus|sonnet|gpt-\S+|o4-\S+', '[model]',
                     cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}', '[timestamp]', cleaned)
    return cleaned


async def run_judge(
    result_a: dict,
    result_b: dict,
    task: dict,
) -> JudgeVerdict:
    """Blind pairwise comparison using Claude opus with max thinking.

    result_a/b are EvalResult dicts with 'worktree_path' and 'config_name'.
    """
    # Get diffs (use pre-computed diff if available, e.g. for reference impl).
    # Thread the authoritative base commit so get_diff yields the full
    # committed diff on the eval branch. run_judge's callers always load the
    # task JSON, so pre_task_commit is present (hard key); the reference
    # contender still short-circuits get_diff via result.get('diff').
    base = task['pre_task_commit']
    diff_a = result_a.get('diff') or await get_diff(
        Path(result_a['worktree_path']), base,
    )
    diff_b = result_b.get('diff') or await get_diff(
        Path(result_b['worktree_path']), base,
    )

    # Randomize assignment to prevent position bias
    swapped = random.random() > 0.5
    if swapped:
        diff_a, diff_b = diff_b, diff_a

    # Strip identifying metadata
    diff_a = _strip_metadata(diff_a)
    diff_b = _strip_metadata(diff_b)

    task_name = task.get('name', task.get('id', 'unknown'))
    task_desc = task.get('task_definition', {}).get('description', '')

    prompt = f"""You are evaluating two implementations of the same task.

Task: {task_name}
Task description: {task_desc}

## Agent A's implementation
```diff
{diff_a}
```

## Agent B's implementation
```diff
{diff_b}
```

Compare these implementations on:
1. **Correctness**: Does it solve the task completely and correctly?
2. **Code quality**: Readability, idiom, minimal unnecessary changes
3. **Test quality**: Meaningful assertions, edge cases, good coverage
4. **Design coherence**: Fits existing codebase patterns and conventions

Output JSON: {{"winner": "A" or "B" or "tie", "confidence": 0.0-1.0, "reasoning": "..."}}"""

    result = await invoke_agent(
        prompt=prompt,
        system_prompt='You are an expert code reviewer performing blinded pairwise comparison. Be precise and fair.',
        cwd=Path('/tmp'),
        model='opus',
        effort='max',
        backend='claude',
        max_budget_usd=5.0,
        output_schema=JUDGE_SCHEMA,
    )

    # Parse verdict
    try:
        verdict = result.structured_output or json.loads(result.output)
    except (json.JSONDecodeError, TypeError):
        logger.warning(f'Judge produced unparseable output: {result.output[:200]}')
        verdict = {'winner': 'tie', 'confidence': 0.0, 'reasoning': 'Judge output parse failure'}

    # Account for swap
    winner = verdict.get('winner', 'tie')
    if swapped:
        winner = {'A': 'B', 'B': 'A', 'tie': 'tie'}.get(winner, 'tie')

    return JudgeVerdict(
        task_id=task.get('id', 'unknown'),
        config_a=result_a['config_name'],
        config_b=result_b['config_name'],
        winner=winner,
        confidence=verdict.get('confidence', 0.0),
        reasoning=verdict.get('reasoning', ''),
    )


async def run_tournament(
    results: list[dict],
    task: dict,
    rounds_per_pair: int = 3,
) -> list[JudgeVerdict]:
    """Run all pairwise comparisons, N rounds each for consistency.

    .. deprecated:: Use :func:`run_elo_tournament` for budget-efficient judging.
    """
    verdicts = []
    pairs = list(combinations(results, 2))

    for a, b in pairs:
        for round_num in range(rounds_per_pair):
            logger.info(
                f'Judge: {a["config_name"]} vs {b["config_name"]} '
                f'(round {round_num + 1}/{rounds_per_pair})'
            )
            v = await run_judge(a, b, task)
            verdicts.append(v)

    return verdicts


async def run_elo_tournament(
    result_dicts: list[dict],
    task: dict,
    pool: TaskPool,
    max_rounds: int = 50,
) -> int:
    """Run Elo-based matchup selection until budget exhausted or all pairs maxed.

    Mutates *pool* in place.  Returns the number of judge calls made.
    """
    # Seed any new configs into the pool
    for r in result_dicts:
        ensure_config_in_pool(pool, r['config_name'])

    # Build lookup: config_name → result dict
    by_config: dict[str, dict] = {r['config_name']: r for r in result_dicts}

    rounds_used = 0
    while rounds_used < max_rounds:
        matchup = next_matchup(pool)
        if matchup is None:
            logger.info('All pairs maxed out, stopping')
            break

        config_a, config_b = matchup

        # Skip if either config has no result available (e.g. worktree deleted)
        if config_a not in by_config or config_b not in by_config:
            key = _pair_key(config_a, config_b)
            pool.pair_counts[key] = MAX_PER_PAIR  # avoid infinite loop
            continue

        logger.info(
            f'Elo match: {config_a} ({pool.ratings[config_a]:.0f}) vs '
            f'{config_b} ({pool.ratings[config_b]:.0f})'
        )

        verdict = await run_judge(by_config[config_a], by_config[config_b], task)
        record_match(
            pool, config_a, config_b,
            verdict.winner, verdict.confidence, verdict.reasoning,
        )
        rounds_used += 1

    return rounds_used


# ---------------------------------------------------------------------------
# Plan-quality scoring (eval-revival θ)
#
# The architect is the #1 cost line (42.7%/39.2% DF/reify spend) yet had ZERO
# eval coverage — the framework FROZE the plan. θ invokes the architect LIVE
# and scores the produced plan two ways:
#   (1) score_plan_structure — a DETERMINISTIC structural rubric that is always
#       a non-sentinel floor (no LLM); and
#   (2) judge_plan_quality — an LLM plan judge scoring the produced plan against
#       the REAL landed diff, guided by the SAME code-owned PLAN_QUALITY_RUBRIC.
# Both read ONLY the produced plan artifact (+ the committed reference diff),
# never a transcript (Invariant P4). run_architect_eval degrades to (1) whenever
# (2) fails, so an architect-eval run ALWAYS emits a non-sentinel plan_quality.
# ---------------------------------------------------------------------------


def _plan_has_steps(plan: dict) -> bool:
    return bool(plan.get('steps'))


def _plan_tdd_alternation(plan: dict) -> bool:
    """True when the plan interleaves test and impl steps (TDD discipline).

    Satisfied when at least one ``test`` step is IMMEDIATELY followed by an
    ``impl`` step — the RED→GREEN pairing a TDD plan is built around.
    """
    steps = plan.get('steps') or []
    types = [s.get('type') for s in steps if isinstance(s, dict)]
    return any(a == 'test' and b == 'impl' for a, b in zip(types, types[1:]))


def _plan_files_declared(plan: dict) -> bool:
    return bool(plan.get('files'))


def _plan_design_decisions_present(plan: dict) -> bool:
    return bool(plan.get('design_decisions'))


def _plan_reuse_items_present(plan: dict) -> bool:
    return bool(plan.get('reuse'))


def _plan_confirmed(plan: dict) -> bool:
    """True when the plan carries the ``_finalized_at`` completeness marker.

    ``confirm_plan`` (and ``artifacts.stamp_plan_provenance``) stamp
    ``_finalized_at`` once the architect finalizes a complete plan, so its
    presence is the code-observable "confirmed" signal.
    """
    return bool(plan.get('_finalized_at'))


# Code-owned structural rubric. Each criterion is
# ``{name, weight, description}`` and ``name`` dispatches through this closed
# detector registry (a pure predicate over the produced plan dict). The
# names/descriptions are embedded verbatim into the LLM plan-judge prompt
# (:func:`judge_plan_quality`) so the deterministic floor and the semantic
# judge score the SAME named signals. An unregistered criterion name raises
# loudly (code-owned rubric typo).
_PLAN_QUALITY_DETECTORS: dict[str, Callable[[dict], bool]] = {
    'has_steps': _plan_has_steps,
    'tdd_alternation': _plan_tdd_alternation,
    'files_declared': _plan_files_declared,
    'design_decisions_present': _plan_design_decisions_present,
    'reuse_items_present': _plan_reuse_items_present,
    'plan_confirmed': _plan_confirmed,
}

PLAN_QUALITY_RUBRIC: dict[str, Any] = {
    'description': (
        'Structural quality of a produced implementation plan: a well-formed '
        'TDD plan declares alternating test/impl steps, names the files it '
        'touches, records its design decisions and code reuse, and is confirmed '
        'complete.'
    ),
    'criteria': [
        {'name': 'has_steps', 'weight': 2.0,
         'description': 'The plan contains at least one concrete step.'},
        {'name': 'tdd_alternation', 'weight': 2.0,
         'description': 'Steps interleave test (RED) and impl (GREEN) — TDD discipline.'},
        {'name': 'files_declared', 'weight': 1.0,
         'description': 'The plan names the files it will create or modify.'},
        {'name': 'design_decisions_present', 'weight': 1.0,
         'description': 'The plan records the design decisions it made and their rationale.'},
        {'name': 'reuse_items_present', 'weight': 1.0,
         'description': 'The plan identifies existing code to reuse rather than reinvent.'},
        {'name': 'plan_confirmed', 'weight': 1.0,
         'description': 'The plan is marked finalized / confirmed complete.'},
    ],
}


def score_plan_structure(
    plan: dict | None, rubric: dict[str, Any] = PLAN_QUALITY_RUBRIC,
) -> float:
    """Deterministic structural plan-quality score in ``[0, 1]``.

    Reads ONLY the produced plan dict — no LLM, no worktree, no transcript. A
    plan with no steps (empty / ``None`` / empty ``steps`` list) is not a plan
    and scores ``0.0`` outright. Otherwise returns the weight-weighted fraction
    of satisfied structural criteria,
    ``round(satisfied_weight / total_weight, 4)``, clamped to ``[0, 1]``.

    This is the ALWAYS-non-sentinel floor :func:`run_architect_eval` degrades to
    when the LLM plan judge fails, so an architect-eval run never emits a null
    ``plan_quality`` (unlike recovery scoring, which degrades to ``None``).

    Raises ``ValueError`` on an empty ``criteria`` list, an unknown criterion
    name, or a non-positive total weight — the rubric is code-owned, so a typo
    must fail loudly (structured-facts-at-failure / loud-over-silent).
    """
    if not plan or not isinstance(plan, dict) or not plan.get('steps'):
        return 0.0
    criteria = rubric.get('criteria') or []
    if not criteria:
        raise ValueError(
            'plan-quality rubric has no criteria — a code-owned rubric must '
            'define >=1 criterion'
        )
    total_weight = 0.0
    satisfied_weight = 0.0
    for criterion in criteria:
        name = criterion.get('name')
        if name not in _PLAN_QUALITY_DETECTORS:
            raise ValueError(
                f'unknown plan-quality criterion: {name!r} '
                f'(registered: {sorted(_PLAN_QUALITY_DETECTORS)})'
            )
        weight = float(criterion.get('weight', 1.0))
        total_weight += weight
        if _PLAN_QUALITY_DETECTORS[name](plan):
            satisfied_weight += weight
    if total_weight <= 0.0:
        raise ValueError('plan-quality rubric total weight must be > 0')
    score = satisfied_weight / total_weight
    return round(min(max(score, 0.0), 1.0), 4)
