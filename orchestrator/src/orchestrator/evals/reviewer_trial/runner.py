"""Runner for reviewer panel trial.

Executes reviewer panels against corpus diffs, calling invoke_agent()
directly (bypassing the OrchestratorConfig reviewer dispatch) so each
ReviewerSpec controls its own model, budget, and effort. Every panel runs in
a scratch clone of its fixture's reviewed tree and is briefed by the
production BriefingAssembler builders with an empty memory block, so the
trial scores the prompt production dispatches.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from shared.cli_invoke import invoke_with_cap_retry
from shared.usage_gate import UsageGate

from orchestrator.agents.briefing import DIFF_ONLY_DIFF_CAP_CHARS, BriefingAssembler
from orchestrator.agents.invoke import invoke_agent
from orchestrator.agents.review_briefing import ReviewBriefing, ReviewSubject
from orchestrator.agents.roles import AgentRole
from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import OrchestratorConfig, project_review_config
from orchestrator.evals.reviewer_trial.checkout import (
    FixtureCheckout,
    FixtureEnrichmentMissing,
    FixtureNotMaterialisable,
    fixture_checkout,
)
from orchestrator.evals.reviewer_trial.corpus import (
    DEFAULT_FIXTURE_CWD,
    CorpusDiff,
    CorpusManifest,
)
from orchestrator.evals.reviewer_trial.variants import (
    ReviewerSpec,
    VariantConfig,
    build_trial_reviewer_role,
)
from orchestrator.mcp_lifecycle import verdict_tools_mcp_server
from orchestrator.review_inputs import gather_review_subject
from orchestrator.review_payload import ReviewPayload, ReviewPayloadError, Verdict, normalize_review
from orchestrator.tree_removal import remove_tree

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

# Results directory
_RESULTS_DIR = Path(__file__).parent / 'results'

# Production's build_reviewer_prompt truncates the diff at this many chars;
# the trial mirrors it by default and lets a campaign lift it.
DIFF_CAP_CHARS = DIFF_ONLY_DIFF_CAP_CHARS

_NO_PLAN_REASON = 'the fixture records neither a plan nor why it has none'


@dataclass
class PanelRunResult:
    """Result of running one panel variant against one corpus diff."""

    variant_name: str
    diff_id: str
    reviews: dict[str, dict] = field(default_factory=dict)   # reviewer_name -> review JSON
    total_cost_usd: float = 0.0
    wall_clock_ms: int = 0
    errors: list[str] = field(default_factory=list)


@functools.cache
def _assembler() -> BriefingAssembler:
    """The production briefing builder. Every trial prompt passes ``context=''``, so it never recalls memory."""
    return BriefingAssembler(OrchestratorConfig())


async def _panel_prompts(
    variant: VariantConfig,
    roles: Sequence[AgentRole],
    diff: CorpusDiff,
    checkout: FixtureCheckout,
    diff_cap_chars: int,
) -> list[str]:
    """Each role's user prompt, built by the production builder of *variant*'s briefing."""
    assembler = _assembler()
    if variant.briefing is ReviewBriefing.DIFF_ONLY:
        return [
            await assembler.build_reviewer_prompt(
                role.name, diff.diff_text, context='', diff_cap_chars=diff_cap_chars,
            )
            for role in roles
        ]
    subject = await _review_subject(diff, checkout)
    return [
        await assembler.build_context_complete_reviewer_prompt(role.name, subject, context='')
        for role in roles
    ]


async def _review_subject(diff: CorpusDiff, checkout: FixtureCheckout) -> ReviewSubject:
    """What production would gather for *diff*'s change, read from its scratch clone.

    The conventions file is the one *diff*'s own project configures.
    """
    if diff.context is None:
        raise FixtureEnrichmentMissing([diff.diff_id])
    if checkout.change_base is None:
        raise FixtureNotMaterialisable(
            diff.diff_id, checkout.unapplied_reason or 'its change is not committed in the clone',
        )
    conventions_relpath = project_review_config(diff.cwd or DEFAULT_FIXTURE_CWD).briefing_path
    if not (checkout.root / conventions_relpath).is_file():
        logger.info(
            'Fixture %s: no conventions at %s in its tree; its d2 panel is briefed without them',
            diff.diff_id, conventions_relpath,
        )
    return await gather_review_subject(
        checkout.root,
        checkout.change_base,
        task=diff.context.task.as_task_dict(),
        plan=diff.context.plan,
        plan_absent_reason=diff.context.plan_reason or _NO_PLAN_REASON,
        conventions_relpath=conventions_relpath,
    )


async def _dispatch_reviewer(usage_gate: UsageGate | None, label: str, **kwargs):
    """Invoke one reviewer, through the shared OAuth pool when a gate is given.

    Without a gate the CLI's own default credentials are used (the
    historical trial behaviour). With a gate, ``invoke_with_cap_retry``
    leases a pool account per attempt and fails over on cap hits, so a
    campaign never burns the interactive account and survives a capped
    pool. A spec-supplied ``oauth_token`` still wins over the pool.
    """
    if usage_gate is None or kwargs.get('oauth_token'):
        return await invoke_agent(**kwargs)
    backend = kwargs.pop('backend', 'claude')
    kwargs.pop('oauth_token', None)
    return await invoke_with_cap_retry(
        usage_gate, label, invoke_fn=invoke_agent, backend=backend, **kwargs,
    )


async def _run_single_reviewer(
    spec: ReviewerSpec,
    diff: CorpusDiff,
    *,
    max_retries: int = 2,
    meta_root: Path | None = None,
) -> tuple[str, ReviewPayload | None, float]:
    """Review *diff* in place at its own cwd, briefed by the production diff-only builder.

    For a caller whose cwd already holds the reviewed tree
    (``evals/rereview.py`` replays a live eval worktree); a trial panel
    reviews a scratch clone of its fixture instead (:func:`run_panel`).
    """
    role = build_trial_reviewer_role(spec)
    prompt = await _assembler().build_reviewer_prompt(role.name, diff.diff_text, context='')
    return await _review(
        spec, role, prompt, cwd=diff.cwd or DEFAULT_FIXTURE_CWD, diff_id=diff.diff_id,
        max_retries=max_retries, meta_root=meta_root,
    )


def _submitted_review(envelope: dict | None) -> ReviewPayload | None:
    if envelope is None:
        return None
    try:
        review = normalize_review(envelope)
    except ReviewPayloadError as exc:
        logger.warning('Reviewer verdict envelope is off-contract: %s', exc)
        return None
    return None if review.verdict is Verdict.ERROR else review


async def _review(
    spec: ReviewerSpec,
    role: AgentRole,
    prompt: str,
    *,
    cwd: Path,
    diff_id: str,
    max_retries: int = 2,
    prices: dict | None = None,
    meta_root: Path | None = None,
    usage_gate: UsageGate | None = None,
) -> tuple[str, ReviewPayload | None, float]:
    """Run one reviewer at *cwd* on *prompt* via the live verdict-tools transport.

    Mirrors workflow.TaskWorkflow._run_reviewer's read+fail-safe contract
    (task 2493 trial-parity decision): *role* is production's reviewer role
    (build_trial_reviewer_role), so its CONTRACT tells the agent to submit its verdict by calling the
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

    Returns (reviewer_name, ReviewPayload_or_None, cost_usd), where cost_usd is
    the summed spend across every invocation attempt (not just the last):
    a verdict-less attempt that triggers a retry still costs money, so
    run_panel's total_cost_usd rollup would under-report multi-attempt trial
    spend if only the final attempt's cost were returned (task 2493 amendment).

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
    owns_root = meta_root is None
    root = meta_root if meta_root is not None else Path(tempfile.mkdtemp(prefix='reviewer_trial_'))
    # Sum spend across ALL invocation attempts, not just the last one: an
    # attempt that invokes successfully but produces no valid verdict still
    # costs money and triggers a retry, so overwriting (rather than
    # accumulating) would silently drop that attempt's real spend from the
    # returned cost (task 2493 amendment). The live-transport path makes such
    # verdict-less retries more likely (a reviewer that never calls
    # submit_review_verdict), so accumulate a running total.
    total_cost = 0.0
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
                result = await _dispatch_reviewer(
                    usage_gate,
                    f'reviewer_trial {spec.name} x {diff_id}',
                    prompt=prompt,
                    system_prompt=role.system_prompt,
                    cwd=cwd,
                    model=role.default_model,
                    max_turns=role.default_max_turns,
                    max_budget_usd=role.default_budget,
                    allowed_tools=role.allowed_tools,
                    disallowed_tools=role.disallowed_tools,
                    mcp_config=mcp_config,
                    # The CLI ambient-merges the .mcp.json at cwd (the live
                    # escalation + fused-memory servers) under
                    # bypassPermissions; scope the reviewer to the
                    # verdict-tools server above and nothing else.
                    strict_mcp_config=True,
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
                return spec.name, None, total_cost

            total_cost += result.cost_usd

            # A submitted verdict is trusted only when the invocation itself
            # also reported success — an invocation failure is untrusted even
            # if it happened to write a verdict before failing.
            review = _submitted_review(artifacts.read_verdict(role.name))
            if result.success and review is not None:
                return spec.name, review, total_cost

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
            return spec.name, ReviewPayload(
                verdict=Verdict.ERROR,
                issues=(),
                reviewer=role.name,
                summary=(
                    f'Reviewer emitted no/invalid verdict after {max_retries} '
                    f'attempt(s) (result.success={result.success}).'
                ),
            ), total_cost

        # Unreachable, but satisfy type checker
        return spec.name, None, total_cost
    finally:
        if owns_root:
            await remove_tree(root, ignore_errors=True)


async def run_panel(
    variant: VariantConfig,
    corpus_diff: CorpusDiff,
    stagger_secs: float = 2.0,
    max_retries: int = 2,
    prices: dict | None = None,
    usage_gate: UsageGate | None = None,
    diff_cap_chars: int = DIFF_CAP_CHARS,
) -> PanelRunResult:
    """Run one panel variant against one corpus diff, in a scratch clone of its reviewed tree.

    Launches reviewers with a stagger delay to avoid thundering herd.

    *prices* (default-off) is threaded to each reviewer's ``invoke_agent``
    call so non-native-cost backends (codex/gemini) price their cost from it.

    Raises FixtureNotMaterialisable when the clone cannot be built, or when a
    context-complete panel's change is not committed in it.
    """
    start = time.monotonic()
    reviews: dict[str, dict] = {}
    errors: list[str] = []
    total_cost = 0.0

    async with fixture_checkout(corpus_diff) as checkout:
        roles = [build_trial_reviewer_role(spec, variant.briefing) for spec in variant.reviewers]
        prompts = await _panel_prompts(variant, roles, corpus_diff, checkout, diff_cap_chars)
        tasks: list[asyncio.Task] = []
        for i, (spec, role, prompt) in enumerate(zip(variant.reviewers, roles, prompts, strict=True)):
            if i > 0 and stagger_secs > 0:
                await asyncio.sleep(stagger_secs)
            task = asyncio.create_task(
                _review(
                    spec, role, prompt, cwd=checkout.root, diff_id=corpus_diff.diff_id,
                    max_retries=max_retries, prices=prices, usage_gate=usage_gate,
                ),
                name=f'{variant.name}__{spec.name}__{corpus_diff.diff_id}',
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

    for r in results:
        if isinstance(r, BaseException):
            errors.append(str(r))
            continue
        name, review, cost = r
        total_cost += cost
        if review is not None:
            reviews[name] = review.to_record()
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
    results_dir: Path | None = None,
    usage_gate: UsageGate | None = None,
    diff_cap_chars: int = DIFF_CAP_CHARS,
) -> list[PanelRunResult]:
    """Run all (variant x diff) pairs with bounded concurrency.

    Persists results incrementally to the results/ directory.

    *prices* (default-off) is threaded through to each ``run_panel`` call so
    cross-family (codex/gemini) reviewers get a non-sentinel cost column.

    Raises FixtureEnrichmentMissing, before any reviewer runs, when a
    context-complete variant meets a fixture with no ``context``. A pair whose
    fixture cannot be materialised is logged and yields no result.
    """
    if any(variant.briefing is ReviewBriefing.CONTEXT_COMPLETE for variant in variants):
        unenriched = [diff.diff_id for diff in corpus.diffs if diff.context is None]
        if unenriched:
            raise FixtureEnrichmentMissing(unenriched)

    out_dir = results_dir or _RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(max_parallel_panels)
    all_results: list[PanelRunResult] = []

    async def _run_and_save(variant: VariantConfig, diff: CorpusDiff) -> PanelRunResult | None:
        async with semaphore:
            result_path = out_dir / f'{variant.name}__{diff.diff_id}.json'

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
            try:
                result = await run_panel(
                    variant, diff, prices=prices, usage_gate=usage_gate,
                    diff_cap_chars=diff_cap_chars,
                )
            except FixtureNotMaterialisable as exc:
                logger.warning(
                    'Skipping %s__%s, no result recorded: %s',
                    variant.name, diff.diff_id, exc.reason,
                )
                return None

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
        elif r is not None:
            all_results.append(r)

    return all_results
