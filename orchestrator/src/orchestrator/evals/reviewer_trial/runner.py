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
import hashlib
import json
import logging
import os
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from shared.cli_invoke import invoke_with_cap_retry
from shared.safe_io import atomic_write_text
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
from orchestrator.evals.reviewer_trial.scorer import ScoringResult, score_panel_run
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


@dataclass(frozen=True)
class PanelRunKey:
    """What, besides the variant and the fixture, decides a panel run's result."""

    prices_sha256: str | None
    gated: bool
    diff_cap_chars: int

    @classmethod
    def of(
        cls,
        *,
        prices: dict | None,
        usage_gate: UsageGate | None,
        diff_cap_chars: int,
    ) -> PanelRunKey:
        prices_sha256 = None if prices is None else hashlib.sha256(
            json.dumps(prices, sort_keys=True, default=str).encode(),
        ).hexdigest()
        return cls(
            prices_sha256=prices_sha256,
            gated=usage_gate is not None,
            diff_cap_chars=diff_cap_chars,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PanelRunKey:
        return cls(
            prices_sha256=data['prices_sha256'],
            gated=data['gated'],
            diff_cap_chars=data['diff_cap_chars'],
        )


@dataclass
class PanelRunResult:
    """Result of running one panel variant against one corpus diff."""

    variant_name: str
    diff_id: str
    reviews: dict[str, dict] = field(default_factory=dict)   # reviewer_name -> review JSON
    total_cost_usd: float = 0.0
    wall_clock_ms: int = 0
    errors: list[str] = field(default_factory=list)
    run_key: PanelRunKey | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PanelRunResult:
        run_key = data.get('run_key')
        return cls(
            variant_name=data['variant_name'],
            diff_id=data['diff_id'],
            reviews=dict(data.get('reviews', {})),
            total_cost_usd=data.get('total_cost_usd', 0.0),
            wall_clock_ms=data.get('wall_clock_ms', 0),
            errors=list(data.get('errors', [])),
            run_key=None if run_key is None else PanelRunKey.from_dict(run_key),
        )


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
        run_key=PanelRunKey.of(prices=prices, usage_gate=usage_gate, diff_cap_chars=diff_cap_chars),
    )


def _slot_filename(variant_name: str, diff_id: str, run_key: PanelRunKey | None) -> str:
    """The file holding the panel run under *run_key*, and its score; a keyless legacy panel has the bare name."""
    if run_key is None:
        return f'{variant_name}__{diff_id}.json'
    key_digest = hashlib.sha256(json.dumps(asdict(run_key), sort_keys=True).encode()).hexdigest()[:12]
    return f'{variant_name}__{diff_id}__{key_digest}.json'


_Entry = TypeVar('_Entry', PanelRunResult, ScoringResult)


def _load_entry(
    path: Path, from_dict: Callable[[Mapping[str, Any]], _Entry], variant_name: str, diff_id: str,
) -> _Entry | None:
    """The entry stored at *path* for this (variant, diff), or None when absent, corrupt or foreign."""
    try:
        entry = from_dict(json.loads(path.read_text()))
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        logger.info('Not reusing %s: it is unreadable (%s)', path, exc)
        return None
    if (entry.variant_name, entry.diff_id) != (variant_name, diff_id):
        logger.info('Not reusing %s: it records %s__%s', path, entry.variant_name, entry.diff_id)
        return None
    return entry


async def run_panel_cached(
    variant: VariantConfig,
    diff: CorpusDiff,
    *,
    results_dir: Path | None = None,
    stagger_secs: float = 2.0,
    prices: dict | None = None,
    usage_gate: UsageGate | None = None,
    diff_cap_chars: int = DIFF_CAP_CHARS,
) -> PanelRunResult:
    """*variant*'s panel on *diff*, reused from *results_dir* only when stored under an equal PanelRunKey.

    Each key has its own file. A keyless entry written before the key existed
    is reused when the key's own file is absent. Why, and how to re-measure
    one: docs/plan-scoring-and-judge.md, "Reviewer-trial panel cache and
    score loop (task 6741)".

    Raises FixtureNotMaterialisable when the panel has to run and its clone cannot be built.
    """
    out_dir = results_dir or _RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = PanelRunKey.of(prices=prices, usage_gate=usage_gate, diff_cap_chars=diff_cap_chars)
    path = out_dir / _slot_filename(variant.name, diff.diff_id, wanted)

    cached = _load_entry(path, PanelRunResult.from_dict, variant.name, diff.diff_id)
    if cached is not None and cached.run_key == wanted:
        logger.info('Reusing %s', path)
        return cached
    legacy_path = out_dir / _slot_filename(variant.name, diff.diff_id, None)
    legacy = _load_entry(legacy_path, PanelRunResult.from_dict, variant.name, diff.diff_id)
    if legacy is not None and legacy.run_key is None:
        logger.info('Reusing %s, a legacy panel that records no run key', legacy_path)
        return legacy
    if cached is not None:
        logger.info('Re-running %s: stored under %s, wanted %s', path, cached.run_key, wanted)

    logger.info('Running %s against %s', variant.name, diff.diff_id)
    result = await run_panel(
        variant, diff, stagger_secs=stagger_secs, prices=prices, usage_gate=usage_gate,
        diff_cap_chars=diff_cap_chars,
    )
    atomic_write_text(path, json.dumps(asdict(result), indent=2))
    logger.info(
        'Completed %s__%s: cost=$%.2f, wall=%dms, errors=%d',
        variant.name, diff.diff_id,
        result.total_cost_usd, result.wall_clock_ms, len(result.errors),
    )
    return result


async def run_trial(
    variants: list[VariantConfig],
    corpus: CorpusManifest,
    max_parallel_panels: int = 3,
    prices: dict | None = None,
    results_dir: Path | None = None,
    usage_gate: UsageGate | None = None,
    diff_cap_chars: int = DIFF_CAP_CHARS,
    stagger_secs: float = 2.0,
) -> list[PanelRunResult]:
    """Run all (variant x diff) pairs with bounded concurrency, each through run_panel_cached.

    *prices* (default-off) is threaded through to each ``run_panel`` call so
    cross-family (codex/gemini) reviewers get a non-sentinel cost column.
    Reused legacy panels are counted in one warning.

    Raises FixtureEnrichmentMissing, before any reviewer runs, when a
    context-complete variant meets a fixture with no ``context``. A pair whose
    fixture cannot be materialised is logged and yields no result.
    """
    if any(variant.briefing is ReviewBriefing.CONTEXT_COMPLETE for variant in variants):
        unenriched = [diff.diff_id for diff in corpus.diffs if diff.context is None]
        if unenriched:
            raise FixtureEnrichmentMissing(unenriched)

    semaphore = asyncio.Semaphore(max_parallel_panels)
    all_results: list[PanelRunResult] = []

    async def _run_and_save(variant: VariantConfig, diff: CorpusDiff) -> PanelRunResult | None:
        async with semaphore:
            try:
                return await run_panel_cached(
                    variant, diff, results_dir=results_dir, stagger_secs=stagger_secs,
                    prices=prices, usage_gate=usage_gate, diff_cap_chars=diff_cap_chars,
                )
            except FixtureNotMaterialisable as exc:
                logger.warning(
                    'Skipping %s__%s, no result recorded: %s',
                    variant.name, diff.diff_id, exc.reason,
                )
                return None

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

    legacy = sum(1 for r in all_results if r.run_key is None)
    if legacy:
        logger.warning(
            'Reused %d legacy panel(s) from %s that record no run key; see docs/plan-scoring-and-judge.md, '
            '"Reviewer-trial panel cache and score loop (task 6741)"',
            legacy, results_dir or _RESULTS_DIR,
        )
    return all_results


async def run_and_score(
    variants: Sequence[VariantConfig],
    corpus: CorpusManifest,
    *,
    results_dir: Path | None = None,
    score_dir: Path | None = None,
    max_parallel_panels: int = 3,
    stagger_secs: float = 2.0,
    prices: dict | None = None,
    usage_gate: UsageGate | None = None,
    diff_cap_chars: int = DIFF_CAP_CHARS,
    on_scored: Callable[[ScoringResult], None] | None = None,
) -> list[ScoringResult]:
    """Every reviewer-trial subcommand's score loop: run_trial's panels, each scored through *usage_gate*.

    With *score_dir*, a score is stored beside the key of the panel it scored:
    a stored score is reused, and only a measured one is stored. *on_scored*
    sees each score as it lands. See
    docs/plan-scoring-and-judge.md, "Reviewer-trial panel cache and score loop (task 6741)".
    """
    panels = await run_trial(
        list(variants), corpus, max_parallel_panels=max_parallel_panels, prices=prices,
        results_dir=results_dir, usage_gate=usage_gate, diff_cap_chars=diff_cap_chars,
        stagger_secs=stagger_secs,
    )
    scores: list[ScoringResult] = []
    for panel in panels:
        diff = corpus.get_diff(panel.diff_id)
        if diff is None:
            logger.warning('Not scoring %s__%s: its diff is not in the corpus', panel.variant_name, panel.diff_id)
            continue
        score = await _score_cached(panel, diff, score_dir=score_dir, usage_gate=usage_gate)
        if on_scored is not None:
            on_scored(score)
        scores.append(score)
    return scores


async def _score_cached(
    panel: PanelRunResult,
    diff: CorpusDiff,
    *,
    score_dir: Path | None,
    usage_gate: UsageGate | None,
) -> ScoringResult:
    if score_dir is None:
        return await score_panel_run(panel, diff, usage_gate=usage_gate)
    score_dir.mkdir(parents=True, exist_ok=True)
    path = score_dir / _slot_filename(panel.variant_name, panel.diff_id, panel.run_key)
    cached = _load_entry(path, ScoringResult.from_dict, panel.variant_name, panel.diff_id)
    if cached is not None:
        return cached
    score = await score_panel_run(panel, diff, usage_gate=usage_gate)
    if score.unmeasured is None:
        atomic_write_text(path, json.dumps(asdict(score), indent=2))
    return score
