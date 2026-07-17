"""Runner for reviewer panel trial.

Executes reviewer panels against corpus diffs, calling invoke_agent()
directly (bypassing the OrchestratorConfig reviewer dispatch) so each
ReviewerSpec controls its own model, budget, and effort.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from orchestrator.agents.invoke import invoke_agent
from orchestrator.artifacts import TaskArtifacts
from orchestrator.evals.reviewer_trial.corpus import CorpusDiff, CorpusManifest
from orchestrator.evals.reviewer_trial.variants import (
    ReviewerSpec,
    VariantConfig,
    build_trial_reviewer_role,
)
from orchestrator.mcp_lifecycle import verdict_tools_mcp_server

logger = logging.getLogger(__name__)

# Orchestrator package root (contains src/, tests/, pyproject.toml) — used
# as verdict_tools_mcp_server's --project fallback arg. runner.py lives at
# src/orchestrator/evals/reviewer_trial/runner.py, four levels below the
# package root; matches workflow.py's _ORCH_PROJECT_DIR (two levels below
# its own shallower path).
_ORCH_PROJECT_DIR = Path(__file__).resolve().parents[4]

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
    via read-only tools + their system prompt, which (task 2493) instructs
    them to submit their verdict via the submit_review_verdict tool rather
    than emit JSON/prose — so this action prompt no longer tells them to
    "output pure JSON" (that would contradict the frozen CONTRACT).
    """
    if len(diff_text) > 50_000:
        diff_text = diff_text[:50_000] + '\n\n... [diff truncated] ...'

    return f"""\
# Code Diff to Review

```diff
{diff_text}
```

# Action

Review the diff according to your specialization. Explore the codebase as needed for context.
"""


async def _run_single_reviewer(
    spec: ReviewerSpec,
    diff: CorpusDiff,
    max_retries: int = 2,
    prices: dict | None = None,
    meta_root: Path | None = None,
) -> tuple[str, dict | None, float]:
    """Run a single reviewer against a diff via the live verdict-tools transport.

    Mirrors workflow.TaskWorkflow._run_reviewer's read+fail-safe contract
    (task 2493 trial-parity decision): the trial reviewer role is built from
    the same PromptSpec as production (build_trial_reviewer_role), so its
    CONTRACT tells the agent to submit its verdict by calling the
    submit_review_verdict tool rather than emit prose/JSON — the verdict is
    therefore read back from the verdicts/ artifact instead of
    result.structured_output/json.loads(result.output). An invocation that
    didn't successfully produce a valid verdict degrades to an ERROR
    disposition rather than raising or returning nothing.

    *meta_root* is injectable so tests can observe/pre-seed the verdict
    artifact path; production leaves it unset and gets a fresh temp dir per
    invocation (the trial has no real task worktree/.task root to anchor
    TaskArtifacts to). That auto-created temp dir is owned by this call —
    unlike production's TaskWorkflow, the trial harness has no
    worktree/.task-meta lifecycle to reclaim it later — so it is removed in
    a ``finally`` block; an injected *meta_root* is left untouched for the
    caller to inspect.

    Returns (reviewer_name, review_dict_or_None, cost_usd).

    Threads the spec's cross-family dispatch fields (``backend``,
    ``env_overrides``, and an ``os.environ``-resolved ``oauth_token`` from
    ``spec.oauth_token_env``) plus an optional *prices* table into
    ``invoke_agent``. All default-off: a native Claude spec (backend='claude',
    no env/token overrides) with ``prices=None`` reproduces prior behaviour
    exactly (invoke_agent treats those defaults as today).

    Raises ``RuntimeError`` if ``spec.oauth_token_env`` is set but that
    environment variable is unset — fails loudly before dispatch rather than
    silently resolving ``oauth_token=None`` and deferring to a confusing
    downstream auth error inside the backend invocation.
    """
    role = build_trial_reviewer_role(spec)
    prompt = _build_reviewer_prompt(diff.diff_text)
    cwd = diff.cwd or _DEFAULT_CWD
    owns_root = meta_root is None
    root = meta_root if meta_root is not None else Path(tempfile.mkdtemp(prefix='reviewer_trial_'))
    try:
        artifacts = TaskArtifacts(cwd, meta_root=root)
        mcp_config = {
            'mcpServers': {
                'verdict-tools': verdict_tools_mcp_server(
                    _ORCH_PROJECT_DIR, cwd, role.name,
                    python_executable=sys.executable, meta_root=root,
                ),
            },
        }

        if spec.oauth_token_env:
            oauth_token = os.environ.get(spec.oauth_token_env)
            if oauth_token is None:
                raise RuntimeError(
                    f'Reviewer {spec.name!r} sets oauth_token_env={spec.oauth_token_env!r} '
                    f'but that environment variable is not set. Refusing to dispatch with a '
                    f'silently-missing oauth token.'
                )
        else:
            oauth_token = None

        for attempt in range(1, max_retries + 1):
            # I-FRESH: never consume a stale verdict from a prior attempt/run on
            # this same meta_root (mirrors workflow._run_reviewer's pre-spawn
            # clear_verdict).
            artifacts.clear_verdict(role.name)
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
                    mcp_config=mcp_config,
                    effort=spec.effort,
                    backend=spec.backend,
                    env_overrides=spec.env_overrides,
                    oauth_token=oauth_token,
                    prices=prices,
                )
            except Exception as exc:
                if attempt < max_retries:
                    logger.warning(
                        'Reviewer %s attempt %d failed: %s, retrying',
                        spec.name, attempt, exc,
                    )
                    continue
                logger.error('Reviewer %s failed after %d attempts: %s', spec.name, max_retries, exc)
                return spec.name, None, 0.0

            cost = result.cost_usd

            # Read the reviewer's structured verdict instead of the
            # structured_output/json.loads cascade (task 2493, mirrors task
            # 2484's production change). A dict envelope with a dict 'verdict'
            # payload carrying verdict∈{PASS,ISSUES_FOUND} is trusted only when
            # the invocation itself also reported success — an invocation
            # failure is untrusted even if it happened to write a verdict
            # before failing.
            envelope = artifacts.read_verdict(role.name)
            payload = envelope.get('verdict') if isinstance(envelope, dict) else None
            if result.success and isinstance(payload, dict) and payload.get('verdict') in {'PASS', 'ISSUES_FOUND'}:
                return spec.name, payload, cost

            if attempt < max_retries:
                logger.warning(
                    'Reviewer %s attempt %d: no/invalid verdict (result.success=%s), retrying',
                    spec.name, attempt, result.success,
                )
                continue
            logger.warning(
                'Reviewer %s: no/invalid verdict after %d attempts (result.success=%s)',
                spec.name, max_retries, result.success,
            )
            return spec.name, {
                'reviewer': role.name,
                'verdict': 'ERROR',
                'issues': [],
                'summary': (
                    f'Reviewer emitted no/invalid verdict after {max_retries} '
                    f'attempt(s) (result.success={result.success}).'
                ),
            }, cost

        # Unreachable, but satisfy type checker
        return spec.name, None, 0.0
    finally:
        if owns_root:
            shutil.rmtree(root, ignore_errors=True)


async def run_panel(
    variant: VariantConfig,
    corpus_diff: CorpusDiff,
    stagger_secs: float = 2.0,
    max_retries: int = 2,
    prices: dict | None = None,
) -> PanelRunResult:
    """Run one panel variant against one corpus diff.

    Launches reviewers with a stagger delay to avoid thundering herd.

    *prices* (default-off) is threaded to each reviewer's ``invoke_agent``
    call so non-native-cost backends (codex/gemini) price their cost from it.
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
            _run_single_reviewer(spec, corpus_diff, max_retries=max_retries, prices=prices),
            name=f'{variant.name}__{spec.name}__{corpus_diff.diff_id}',
        )
        tasks.append(task)

    # Gather results
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for r in results:
        if isinstance(r, BaseException):
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
    prices: dict | None = None,
) -> list[PanelRunResult]:
    """Run all (variant x diff) pairs with bounded concurrency.

    Persists results incrementally to the results/ directory.

    *prices* (default-off) is threaded through to each ``run_panel`` call so
    cross-family (codex/gemini) reviewers get a non-sentinel cost column.
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
            result = await run_panel(variant, diff, prices=prices)

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
        if isinstance(r, BaseException):
            logger.error('Panel run failed: %s', r)
        else:
            all_results.append(r)

    return all_results
