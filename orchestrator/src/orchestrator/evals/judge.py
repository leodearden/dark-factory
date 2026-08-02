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
from typing import Any, TypeGuard

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
    return any(a == 'test' and b == 'impl' for a, b in zip(types, types[1:], strict=False))


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


def is_scorable_plan(plan: object) -> TypeGuard[dict]:
    """THE single test for "does this artifact carry content worth scoring?".

    A plan with no steps is not a plan — whether it is absent (``None``), empty
    (``{}``), not a dict at all, or the header-only stub ``create_plan`` writes
    on the architect's FIRST plan-tools call (``plan_tools._create_plan``: a
    TRUTHY dict with ``task_id`` / ``title`` / ``analysis`` / ``files`` and
    ``steps: []``).

    Deliberately shared by :func:`score_plan_structure`'s ``0.0``
    short-circuit and by ``run_architect_eval``'s cap-taint predicate. Those
    were once two independent tests — raw dict truthiness in the runner versus
    ``not plan.get('steps')`` here — and they disagreed exactly on the
    header-only stub, so a session cap landing right after ``create_plan`` left
    the cell UNtainted and then floored it to a fabricated ``0.0``. Making both
    call sites literally this function makes that drift structurally impossible.

    Note this is a different question from the ``has_steps`` RUBRIC criterion
    (:func:`_plan_has_steps`), which is scored per-plan against a real plan; do
    not conflate the two roles.

    Typed as a ``TypeGuard`` (a plain ``bool`` at runtime) so it also carries
    the ``isinstance`` narrowing the inline guard it replaced used to give
    :func:`score_plan_structure`.
    """
    return isinstance(plan, dict) and bool(plan.get('steps'))


def clamp_unit_score(score: float) -> float:
    """THE output-range contract shared by both plan-quality instruments.

    :func:`score_plan_structure` and :func:`judge_plan_quality` are a
    two-instrument pair reduced onto ONE axis (``EvalMetrics.plan_quality``) —
    exactly the shape :func:`is_scorable_plan` (above) already solved for
    scorability. That function's docstring records the rule: "making both call
    sites literally this function makes that drift structurally impossible."
    The range invariant gets the same treatment — the ONE named contract both
    instruments call, rather than a copied expression the two can silently
    drift apart on the next time either is edited.

    Covers a real schema gap: ``PLAN_QUALITY_SCHEMA`` declares
    ``{'minimum': 0.0, 'maximum': 1.0}``, but that only constrains the
    ``structured_output`` delivery path — the documented
    ``result.structured_output or json.loads(result.output)`` fallback in
    :func:`judge_plan_quality` bypasses schema enforcement entirely, so an
    out-of-range judge answer reaches this runtime clamp regardless of which
    path delivered it.

    Clamps to ``[0, 1]`` and rounds to 4 decimal places — the precision
    :func:`score_plan_structure` has always used. Does NOT handle NaN: ``min``/
    ``max`` are ordering operations and NaN is unordered, so a NaN input passes
    straight through unclamped (``round(min(max(nan, 0.0), 1.0), 4)`` is
    ``nan``). A caller with untrusted input (:func:`judge_plan_quality`) must
    check for that itself before calling this.
    """
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


@dataclass
class PlanQualityVerdict:
    """Result of the LLM plan judge (eval-revival θ).

    ``plan_quality`` is a ``float`` in ``[0, 1]`` when the judge scored the
    plan, or ``None`` — the parse-failure sentinel — when the judge output could
    not be parsed. ``run_architect_eval`` treats ``None`` as its signal to
    degrade to the deterministic :func:`score_plan_structure` floor, so the
    persisted ``plan_quality`` is a non-sentinel float even when the judge
    fails (a judge failure never nulls a cell that has a real plan behind it).
    """

    plan_quality: float | None
    per_criterion: dict[str, Any]
    reasoning: str
    # Set when the JUDGE'S OWN invocation was refused at the transport layer (a
    # 429 cap hit / auth failure), so this ``None`` plan_quality is an INFRA
    # failure — we never got a judgement — rather than an unparseable answer.
    # Trailing and defaulted, so every existing 3-arg construction (including
    # the parse-failure fallback below) is unaffected and keeps reading None.
    invocation_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
) -> PlanQualityVerdict:
    """LLM plan judge: score a produced *plan* against the real landed diff.

    Mirrors :func:`run_judge`'s structure — an opus judge with an
    ``output_schema``, ``structured_output``-or-``json.loads`` parse, and a
    defined fallback verdict on parse failure. The prompt embeds the produced
    plan, the landed *reference_diff* (``pre_task_commit..post_task_commit``,
    the always-available ground truth since ζ fixtures frequently carry
    ``plan: null``), and the code-owned *rubric*'s criterion names — so the
    semantic judge scores the SAME named structural signals
    :func:`score_plan_structure` weights deterministically.

    THE ANTI-FABRICATION FLOOR APPLIES UNIFORMLY, WHICHEVER SCORING PATH RUNS
    (task 3303). An artifact that is not :func:`is_scorable_plan` carries
    nothing to judge, so this function refuses it up front — before a prompt is
    built, before ``invoke_agent`` — and returns the deterministic floor
    instead. The guard is the SAME predicate :func:`score_plan_structure`
    short-circuits on and ``run_architect_eval`` gates on, so the three cannot
    drift; and the refusal's score is DERIVED by calling
    :func:`score_plan_structure` — with THIS call's own *rubric*, not the module
    default — rather than hardcoding ``0.0``, so a future change to the floor's
    semantics (including a rubric-sensitive one) carries this path with it
    structurally rather than by convention. Without it, this instrument's
    coherence rested entirely on its ONE caller remembering to gate, and any
    second caller (a backfill/re-scoring script, a new eval path, ``prompt_opt``,
    a resume wave) silently re-opened the reported defect: the 2026-07-29 cell
    ``reify_task_12__architect-opus-high__52c66767.json`` records
    ``plan_steps=0`` alongside ``plan_quality=0.31`` — a value the floor cannot
    even express (its outputs are multiples of ``0.125``), so it can only have
    come from an ungated judge scoring an empty artifact.

    That refusal is ``plan_quality=0.0``-not-``None`` DELIBERATELY: ``None``
    means "no judgement available, degrade to the floor" (parse failure /
    transport refusal), whereas here the judgement is definite and no infra
    failure occurred. ``invocation_error`` stays ``None`` for the same reason —
    nothing was refused at the transport layer. Keeping the two apart preserves
    the content-failure / infra-failure distinction tasks 3118 and 3302 turn on:
    a stepless plan from a healthy architect is a real reliability signal worth
    ``0.0``, never a cap-tainted exclusion.

    A successfully parsed ``plan_quality`` is put through
    :func:`clamp_unit_score` — the same ``[0, 1]``/4dp contract
    :func:`score_plan_structure` uses — rather than returned as a raw float.
    This matters because ``PLAN_QUALITY_SCHEMA``'s declared bounds only
    constrain the ``structured_output`` delivery path: the
    ``json.loads(result.output)`` fallback above bypasses schema enforcement
    entirely, so an out-of-range judge answer must still be caught at runtime
    regardless of which path delivered it.

    On any parse failure the verdict's ``plan_quality`` is ``None`` (the
    sentinel :func:`run_architect_eval` degrades on), never a crash. When the
    judge's OWN invocation was refused at the transport layer (a 429 cap hit /
    auth failure) the verdict additionally carries ``invocation_error``, so that
    infra cause stays distinguishable from an unparseable answer; the caller
    still degrades to the deterministic structural floor, which remains a
    legitimate content-derived score whenever a real plan exists.
    """
    if not is_scorable_plan(plan):
        # THIS call's rubric, not the module default. Today the floor
        # short-circuits on ``is_scorable_plan`` before it ever reads a rubric,
        # so every rubric yields 0.0 here and the argument is inert — but
        # dropping it would make "the score is DERIVED from the floor" true of
        # only ONE of the floor's two arguments, and would silently re-open the
        # two-instrument disagreement 3303 closes the day that short-circuit
        # becomes rubric-sensitive (a rubric-defined minimum, or criteria that
        # change what "unscorable" means).
        floor = score_plan_structure(plan, rubric=rubric)
        # Name WHICH cell to go look at, whichever key the caller populated:
        # ``run_architect_eval`` passes ``id``, but a second caller — precisely
        # the scenario this guard defends against — may carry only ``name``
        # (``run_judge`` and the prompt builder below both key off ``name``
        # first). Falling through absent AND empty values means a malformed
        # task dict degrades the log line rather than turning an
        # anti-fabrication guard into a crash.
        cell = task.get('id') or task.get('name') or 'unknown'
        # LOUD, not a silent 0.0: reaching here means a caller did NOT gate,
        # which is exactly how the reported cell was written. WARNING matches
        # ``run_architect_eval``'s own no-scorable-plan log and the
        # transport-refusal log below — an expected-but-notable degradation,
        # not an error. It cannot double-log through the normal path because
        # the runner's own ``is_scorable_plan`` gate short-circuits before the
        # judge is ever awaited.
        logger.warning(
            f'Plan judge asked to score an UNJUDGEABLE artifact for {cell}: '
            f'the plan carries no steps '
            f'(is_scorable_plan=False) — LLM judge SKIPPED, scored on the '
            f'deterministic structural floor ({floor}). The caller did not '
            f'gate; run_architect_eval does (task 3302/3303).'
        )
        return PlanQualityVerdict(
            plan_quality=floor,
            per_criterion={},
            reasoning=(
                'plan carries no steps — not a judgeable artifact; scored on '
                f'the deterministic structural floor ({floor})'
            ),
        )

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

    prompt = f"""You are evaluating the QUALITY of an implementation PLAN a \
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

    result = await invoke_agent(
        prompt=prompt,
        system_prompt=(
            'You are an expert software architect grading an implementation '
            'plan against the change that actually landed. Be precise and fair.'
        ),
        cwd=Path('/tmp'),
        model='opus',
        effort='max',
        backend='claude',
        max_budget_usd=5.0,
        output_schema=PLAN_QUALITY_SCHEMA,
    )

    # Was the JUDGE ITSELF refused at the transport layer? Checked BEFORE the
    # parse block, because a 429 body is not JSON and would otherwise land in
    # the parse-failure fallback — making an infra refusal indistinguishable
    # from a judge that answered badly. Local import: keeps judge.py's
    # module-level import surface unchanged (and there is no cycle either way —
    # metrics.py does not import judge).
    from .metrics import detect_invocation_error

    invocation_error = detect_invocation_error(result)
    if invocation_error:
        logger.warning(
            f'Plan judge invocation REFUSED at the transport layer: '
            f'{invocation_error}'
        )
        return PlanQualityVerdict(
            plan_quality=None,
            per_criterion={},
            reasoning=f'plan judge invocation refused: {invocation_error}',
            invocation_error=invocation_error,
        )

    # Parse verdict — structured_output first, else json.loads(output); a
    # missing/None/non-numeric plan_quality degrades to the None sentinel.
    try:
        verdict = result.structured_output or json.loads(result.output)
        raw_quality = verdict['plan_quality']
        plan_quality = (
            clamp_unit_score(float(raw_quality)) if raw_quality is not None else None
        )
    except (json.JSONDecodeError, TypeError, KeyError, ValueError):
        logger.warning(
            f'Plan judge produced unparseable output: {str(result.output)[:200]}'
        )
        return PlanQualityVerdict(
            plan_quality=None,
            per_criterion={},
            reasoning='Plan judge output parse failure',
        )

    return PlanQualityVerdict(
        plan_quality=plan_quality,
        per_criterion=verdict.get('per_criterion', {}) or {},
        reasoning=verdict.get('reasoning', ''),
    )
