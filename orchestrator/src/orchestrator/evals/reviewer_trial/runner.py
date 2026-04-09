"""Runner for reviewer panel trial.

Executes reviewer panels against corpus diffs, calling invoke_agent()
directly (bypassing the OrchestratorConfig reviewer dispatch) so each
ReviewerSpec controls its own model, budget, and effort.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from orchestrator.agents.invoke import invoke_agent
from orchestrator.evals.reviewer_trial.corpus import CorpusDiff, CorpusManifest
from orchestrator.evals.reviewer_trial.variants import (
    ReviewerSpec,
    VariantConfig,
    build_trial_reviewer_role,
)

logger = logging.getLogger(__name__)

# Review output schema (identical to production)
REVIEW_SCHEMA = {
    'type': 'object',
    'properties': {
        'reviewer': {'type': 'string'},
        'verdict': {'type': 'string', 'enum': ['PASS', 'ISSUES_FOUND']},
        'issues': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'severity': {'type': 'string', 'enum': ['blocking', 'suggestion']},
                    'location': {'type': 'string'},
                    'category': {'type': 'string'},
                    'description': {'type': 'string'},
                    'suggested_fix': {'type': 'string'},
                },
                'required': ['severity', 'location', 'category', 'description'],
            },
        },
        'summary': {'type': 'string'},
    },
    'required': ['reviewer', 'verdict', 'issues', 'summary'],
}

# Default working directory for diffs without a project-specific cwd
_DEFAULT_CWD = Path('/home/leo/src/dark-factory')

# Results directory
_RESULTS_DIR = Path(__file__).parent / 'results'


@dataclass
class PanelRunResult:
    """Result of running one panel variant against one corpus diff."""

    variant_name: str
    diff_id: str
    reviews: dict[str, dict] = field(default_factory=dict)   # reviewer_name -> review JSON
    total_cost_usd: float = 0.0
    wall_clock_ms: int = 0
    errors: list[str] = field(default_factory=list)


def _build_reviewer_prompt(diff_text: str) -> str:
    """Build a simple reviewer prompt with just the diff.

    Simplified from BriefingAssembler.build_reviewer_prompt — no memory
    context needed for the trial. Reviewers get the diff + codebase access
    via read-only tools + their system prompt.
    """
    if len(diff_text) > 50_000:
        diff_text = diff_text[:50_000] + '\n\n... [diff truncated] ...'

    return f"""\
# Code Diff to Review

```diff
{diff_text}
```

# Action

Review the diff according to your specialization. Explore the codebase as needed for context. Output your review as pure JSON.
"""


async def _run_single_reviewer(
    spec: ReviewerSpec,
    diff: CorpusDiff,
    max_retries: int = 2,
) -> tuple[str, dict | None, float]:
    """Run a single reviewer against a diff.

    Returns (reviewer_name, review_dict_or_None, cost_usd).
    """
    role = build_trial_reviewer_role(spec)
    prompt = _build_reviewer_prompt(diff.diff_text)
    cwd = diff.cwd or _DEFAULT_CWD

    for attempt in range(1, max_retries + 1):
        try:
            result = await invoke_agent(
                prompt=prompt,
                system_prompt=role.system_prompt,
                cwd=cwd,
                model=role.default_model,
                max_turns=role.default_max_turns,
                max_budget_usd=role.default_budget,
                allowed_tools=role.allowed_tools,
                disallowed_tools=role.disallowed_tools,
                output_schema=REVIEW_SCHEMA,
                effort=spec.effort,
            )

            cost = result.cost_usd

            # Try structured output first, then JSON parse
            if result.structured_output:
                return spec.name, result.structured_output, cost

            try:
                review = json.loads(result.output)
                return spec.name, review, cost
            except (json.JSONDecodeError, TypeError):
                if attempt < max_retries:
                    logger.warning(
                        'Reviewer %s attempt %d: unparseable output, retrying',
                        spec.name, attempt,
                    )
                    continue
                logger.warning(
                    'Reviewer %s: unparseable output after %d attempts: %s',
                    spec.name, max_retries, result.output[:200],
                )
                return spec.name, {
                    'reviewer': spec.name,
                    'verdict': 'ERROR',
                    'issues': [],
                    'summary': f'Unparseable output: {result.output[:200]}',
                }, cost

        except Exception as exc:
            if attempt < max_retries:
                logger.warning(
                    'Reviewer %s attempt %d failed: %s, retrying',
                    spec.name, attempt, exc,
                )
                continue
            logger.error('Reviewer %s failed after %d attempts: %s', spec.name, max_retries, exc)
            return spec.name, None, 0.0

    # Unreachable, but satisfy type checker
    return spec.name, None, 0.0


async def run_panel(
    variant: VariantConfig,
    corpus_diff: CorpusDiff,
    stagger_secs: float = 2.0,
    max_retries: int = 2,
) -> PanelRunResult:
    """Run one panel variant against one corpus diff.

    Launches reviewers with a stagger delay to avoid thundering herd.
    """
    start = time.monotonic()
    reviews: dict[str, dict] = {}
    errors: list[str] = []
    total_cost = 0.0

    # Launch reviewers with stagger
    tasks: list[asyncio.Task] = []
    for i, spec in enumerate(variant.reviewers):
        if i > 0 and stagger_secs > 0:
            await asyncio.sleep(stagger_secs)
        task = asyncio.create_task(
            _run_single_reviewer(spec, corpus_diff, max_retries=max_retries),
            name=f'{variant.name}__{spec.name}__{corpus_diff.diff_id}',
        )
        tasks.append(task)

    # Gather results
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for r in results:
        if isinstance(r, Exception):
            errors.append(str(r))
            continue
        name, review, cost = r
        total_cost += cost
        if review is not None:
            reviews[name] = review
        else:
            errors.append(f'{name}: no result')

    elapsed_ms = int((time.monotonic() - start) * 1000)

    return PanelRunResult(
        variant_name=variant.name,
        diff_id=corpus_diff.diff_id,
        reviews=reviews,
        total_cost_usd=total_cost,
        wall_clock_ms=elapsed_ms,
        errors=errors,
    )


async def run_trial(
    variants: list[VariantConfig],
    corpus: CorpusManifest,
    max_parallel_panels: int = 3,
) -> list[PanelRunResult]:
    """Run all (variant x diff) pairs with bounded concurrency.

    Persists results incrementally to the results/ directory.
    """
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(max_parallel_panels)
    all_results: list[PanelRunResult] = []

    async def _run_and_save(variant: VariantConfig, diff: CorpusDiff) -> PanelRunResult:
        async with semaphore:
            result_path = _RESULTS_DIR / f'{variant.name}__{diff.diff_id}.json'

            # Skip if already completed
            if result_path.exists():
                try:
                    existing = json.loads(result_path.read_text())
                    logger.info('Skipping %s__%s (already exists)', variant.name, diff.diff_id)
                    return PanelRunResult(
                        variant_name=existing['variant_name'],
                        diff_id=existing['diff_id'],
                        reviews=existing.get('reviews', {}),
                        total_cost_usd=existing.get('total_cost_usd', 0.0),
                        wall_clock_ms=existing.get('wall_clock_ms', 0),
                        errors=existing.get('errors', []),
                    )
                except (json.JSONDecodeError, KeyError):
                    pass  # Re-run if result file is corrupt

            logger.info('Running %s against %s', variant.name, diff.diff_id)
            result = await run_panel(variant, diff)

            # Persist incrementally
            result_path.write_text(json.dumps(asdict(result), indent=2))
            logger.info(
                'Completed %s__%s: cost=$%.2f, wall=%dms, errors=%d',
                variant.name, diff.diff_id,
                result.total_cost_usd, result.wall_clock_ms, len(result.errors),
            )
            return result

    # Build all (variant, diff) pairs
    tasks = [
        _run_and_save(variant, diff)
        for variant in variants
        for diff in corpus.diffs
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)
    for r in results:
        if isinstance(r, Exception):
            logger.error('Panel run failed: %s', r)
        else:
            all_results.append(r)

    return all_results
