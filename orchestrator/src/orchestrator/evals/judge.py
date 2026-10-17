"""LLM pairwise blinded comparison of eval results."""

from __future__ import annotations

import functools
import json
import logging
import math
import random
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Any

from shared.cli_invoke import no_mcp_servers_config

from orchestrator.agents.invoke import invoke_agent

from .elo import (
    MAX_PER_PAIR,
    TaskPool,
    _pair_key,
    ensure_config_in_pool,
    next_matchup,
    record_match,
)
from .measurement import (
    MeasurementStage,
    Scored,
    Unmeasured,
    json_answer,
    measured_call,
)
from .plan_predicates import is_scorable_plan, plan_confirmed
from .snapshots import get_diff

if TYPE_CHECKING:
    from shared.cli_invoke import AgentResult

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
        # The judge needs no MCP: it answers through output_schema. The truthy
        # zero-server config is what makes the strict flag bite (task 4757);
        # mcp_config=None would leave the cwd's ambient .mcp.json merged.
        mcp_config=no_mcp_servers_config(),
        strict_mcp_config=True,
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
    return any(a == 'test' and b == 'impl' for a, b in zip(types, types[1:], strict=False))


def _plan_files_declared(plan: dict) -> bool:
    return bool(plan.get('files'))


def _plan_design_decisions_present(plan: dict) -> bool:
    return bool(plan.get('design_decisions'))


def _plan_reuse_items_present(plan: dict) -> bool:
    return bool(plan.get('reuse'))


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
    'plan_confirmed': plan_confirmed,
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


def clamp_unit_score(score: float) -> float:
    """THE output-range contract shared by the two plan-quality instruments.

    :func:`score_plan_structure` and :func:`judge_plan_quality` are a
    two-instrument pair reduced onto ONE axis (``EvalMetrics.plan_quality``) —
    exactly the shape ``plan_predicates.is_scorable_plan`` already solved for
    scorability by making every call site literally that one function.
    The range invariant gets the same treatment within THIS pair — the ONE
    named contract both plan-quality instruments call, rather than a copied
    expression the two can silently drift apart on the next time either is
    edited. This is also THE canonical explanation of the schema-gap/NaN
    reasoning below — other call sites in this module and its tests point
    back here rather than restating it, so the rationale cannot drift out of
    step with itself across copies the way the clamp expression used to.

    Scoped to this instrument pair, not the whole ``evals`` package: the
    identical ``round(min(max(x, 0.0), 1.0), 4)`` idiom still exists verbatim
    at ``scoring.py``'s recovery rubric and ``metrics.py``'s composite blend —
    different instruments, on different axes, deliberately left unmigrated
    here to avoid widening this fix's blast radius (see this task's plan). A
    future unification of those is a separate decision, not an oversight of
    this one.

    Covers a real schema gap: ``PLAN_QUALITY_SCHEMA`` declares
    ``{'minimum': 0.0, 'maximum': 1.0}``, but that only constrains the
    ``structured_output`` delivery path — the ``json.loads(result.output)``
    fallback in ``measurement.json_answer`` bypasses schema enforcement entirely, so an
    out-of-range judge answer reaches this runtime clamp regardless of which
    path delivered it.

    Clamps to ``[0, 1]`` and rounds to 4 decimal places — the precision
    :func:`score_plan_structure` has always used. ``min``/``max`` are ordering
    operations, so if this function accepted non-finite input, a NaN would
    pass straight through unclamped (``round(min(max(nan, 0.0), 1.0), 4)`` is
    ``nan``) and ``+/-Infinity`` would clamp mechanically (to ``1.0``/``0.0``)
    while actually being a fabricated best/worst score rather than a
    measurement — not something this function could tell apart from a real
    one. Rather than leave that a convention callers must remember, this
    function REJECTS all non-finite input outright with ``ValueError`` — a
    loud floor, not just a documented one (structured-facts-at-failure). The
    one caller with untrusted input (``_parse_plan_verdict``) already
    pre-checks with its own ``math.isfinite`` guard, which ``measured_call``
    records as an unparseable answer (with a WARNING) before ever reaching
    here, so in practice this raise is a defense-in-depth backstop for a future second
    caller that forgets to — the same class of gap a second caller re-opens
    by skipping ``plan_predicates.is_scorable_plan``. This
    function's contract is finite-in, ``[0, 1]``/4dp-out, ``ValueError`` on
    anything else.
    """
    if not math.isfinite(score):
        raise ValueError(
            f'clamp_unit_score requires finite input, got {score!r}'
        )
    return round(min(max(score, 0.0), 1.0), 4)


def score_plan_structure(
    plan: dict | None, rubric: dict[str, Any] = PLAN_QUALITY_RUBRIC,
) -> float:
    """Deterministic structural plan-quality score in ``[0, 1]``.

    Reads ONLY the produced plan dict — no LLM, no worktree, no transcript. A
    plan that is not :func:`is_scorable_plan` (empty / ``None`` / no steps /
    the header-only ``create_plan`` stub) is not a plan and scores ``0.0``
    outright. Otherwise returns the weight-weighted fraction
    of satisfied structural criteria, ``satisfied_weight / total_weight``, put
    through :func:`clamp_unit_score` — the ``[0, 1]``/4dp contract this
    function and :func:`judge_plan_quality` share.

    This is the ALWAYS-non-sentinel floor :func:`run_architect_eval` degrades to
    when the LLM plan judge fails, so an architect-eval run never emits a null
    ``plan_quality`` (unlike recovery scoring, which degrades to ``None``).

    Raises ``ValueError`` on an empty ``criteria`` list, an unknown criterion
    name, or a non-positive total weight — the rubric is code-owned, so a typo
    must fail loudly (structured-facts-at-failure / loud-over-silent).
    """
    # Exactly equivalent to the former inline
    # ``not plan or not isinstance(plan, dict) or not plan.get('steps')``
    # (an empty dict and a None both fail the steps test), now expressed via the
    # ONE predicate the runner's taint decision also consults.
    if not is_scorable_plan(plan):
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
    return clamp_unit_score(score)


@dataclass(frozen=True)
class PlanQualityVerdict:
    """The LLM plan judge's answer: a ``[0, 1]`` score and its explanation."""

    plan_quality: float
    per_criterion: dict[str, Any]
    reasoning: str


PLAN_QUALITY_SCHEMA = {
    'type': 'object',
    'properties': {
        'plan_quality': {'type': 'number', 'minimum': 0.0, 'maximum': 1.0},
        'per_criterion': {'type': 'object'},
        'reasoning': {'type': 'string'},
    },
    'required': ['plan_quality', 'reasoning'],
}


async def judge_plan_quality(
    plan: dict | None,
    reference_diff: str,
    task: dict,
    rubric: dict[str, Any] = PLAN_QUALITY_RUBRIC,
) -> Scored[PlanQualityVerdict] | Unmeasured:
    """LLM plan judge: score a produced *plan* against the real landed diff.

    ``Scored`` when the judge answered parseably, or when the artifact is not
    :func:`is_scorable_plan` — then the judge is never invoked and the score is
    :func:`score_plan_structure`'s floor at ``cost_usd=0.0``. ``Unmeasured``
    (stage ``judge``) when the judge's call was refused or its answer was
    unparseable, null or non-finite; a finite out-of-range answer is clamped
    with a WARNING. ``cost_usd`` is the judge invocation's own spend on every
    path that reached it. Rationale: docs/plan-scoring-and-judge.md.
    """
    # Name WHICH cell to go look at, whichever key the caller populated.
    cell = task.get('id') or task.get('name') or 'unknown'

    if not is_scorable_plan(plan):
        floor = score_plan_structure(plan, rubric=rubric)
        logger.warning(
            f'Plan judge asked to score an UNJUDGEABLE artifact for {cell}: '
            f'the plan carries no steps '
            f'(is_scorable_plan=False) — LLM judge SKIPPED, scored on the '
            f'deterministic structural floor ({floor}). The caller did not '
            f'gate; run_architect_eval does (task 3302/3303).'
        )
        return Scored(
            PlanQualityVerdict(
                plan_quality=floor,
                per_criterion={},
                reasoning=(
                    'plan carries no steps — not a judgeable artifact; scored on '
                    f'the deterministic structural floor ({floor})'
                ),
            ),
            cost_usd=0.0,
        )

    return await measured_call(
        invoke_agent,
        functools.partial(_parse_plan_verdict, cell=cell),
        stage=MeasurementStage.JUDGE,
        backend='claude',
        label=f'Plan judge for {cell}',
        prompt=_plan_judge_prompt(plan, reference_diff, task, rubric),
        system_prompt=(
            'You are an expert software architect grading an implementation '
            'plan against the change that actually landed. Be precise and fair.'
        ),
        cwd=Path('/tmp'),
        model='opus',
        effort='max',
        max_budget_usd=5.0,
        output_schema=PLAN_QUALITY_SCHEMA,
        mcp_config=no_mcp_servers_config(),
        strict_mcp_config=True,
    )


def _plan_judge_prompt(
    plan: dict, reference_diff: str, task: dict, rubric: dict[str, Any],
) -> str:
    task_name = task.get('name', task.get('id', 'unknown'))
    task_desc = task.get('task_definition', {}).get('description', '')

    criteria_lines = '\n'.join(
        f'- {c["name"]} (weight {c["weight"]}): {c["description"]}'
        for c in rubric['criteria']
    )
    plan_json = json.dumps(plan, indent=2, default=str)
    # Strip identifying metadata from the landed diff (bias prevention), exactly
    # as run_judge does for the contender diffs.
    clean_diff = _strip_metadata(reference_diff)

    return f"""You are evaluating the QUALITY of an implementation PLAN a \
software architect produced for a task, BEFORE any code was written.

Task: {task_name}
Task description: {task_desc}

## The architect's produced plan
```json
{plan_json}
```

## The change that ACTUALLY landed for this task (reference diff)
```diff
{clean_diff}
```

Score how well the produced PLAN anticipates the change that actually landed, \
guided by this plan-quality rubric:
{criteria_lines}

A strong plan decomposes the landed change into coherent TDD steps, names the \
right files, records its design decisions and code reuse, and would plausibly \
PRODUCE the landed diff if executed. Judge the PLAN, not the diff.

Output JSON: {{"plan_quality": 0.0-1.0, "per_criterion": {{"<criterion>": 0.0-1.0, ...}}, "reasoning": "..."}}"""


def _parse_plan_verdict(result: AgentResult, *, cell: str) -> PlanQualityVerdict:
    """The judge's verdict, clamped; ``ValueError``/``KeyError`` if it has none.

    A null or non-finite ``plan_quality`` is not a judgement (see
    :func:`clamp_unit_score`). A finite out-of-range one is clamped, loudly.
    """
    payload = json_answer(result)
    raw = payload['plan_quality']
    if raw is None:
        raise ValueError('plan_quality is null')
    raw_quality = float(raw)
    if not math.isfinite(raw_quality):
        raise ValueError(f'non-finite plan_quality {raw_quality!r}')
    plan_quality = clamp_unit_score(raw_quality)
    if raw_quality < 0.0 or raw_quality > 1.0:
        logger.warning(
            f'Plan judge for {cell} answered OUTSIDE its declared '
            f'[0, 1] contract (raw plan_quality={raw_quality!r}) — '
            f'clamped to {plan_quality!r}.'
        )
    return PlanQualityVerdict(
        plan_quality=plan_quality,
        per_criterion=payload.get('per_criterion') or {},
        reasoning=payload.get('reasoning', ''),
    )
