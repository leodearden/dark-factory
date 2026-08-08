"""Metric collection and composite scoring for eval runs."""

from __future__ import annotations

import asyncio
import logging
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

# The W4 cap/error classification seam (task 3118). Module-level import is safe
# for the same reason agents.invoke above is: shared.invocation_outcome reaches
# only shared.cli_invoke and has no import back into orchestrator, so there is
# no cycle. detect_invocation_error below delegates to it rather than forking
# its cap/auth string tables into the eval layer.
from shared.invocation_outcome import (
    OK,
    AuthFailed,
    CapHit,
    ModelNotFound,
    ServerError,
    ZeroOutputWedge,
    classify_invocation,
)

# Cost primitives (the USD/1M fallback rate + the PriceEntry-or-dict accessor)
# have a SINGLE home in agents.invoke (task 2459); this module re-exports them
# rather than re-declaring so the three-way copy can never drift (reviewer:
# code-reuse). resolve_cost_usd below mirrors invoke's Invariant-P5 policy, and
# report.build_price_table imports _rate FROM here. agents.invoke has no import
# cycle back to evals (it reaches only config/fm_retry/shared.*), so a
# module-level import is safe.
from orchestrator.agents.invoke import _FALLBACK_PRICE, _rate

if TYPE_CHECKING:
    from shared.cli_invoke import AgentResult

    from orchestrator.workflow import TaskWorkflow

logger = logging.getLogger(__name__)


@dataclass
class EvalMetrics:
    """Metrics collected from a single eval run."""

    # Correctness (pass/fail gate). ``None`` means "unknown / suspicious" —
    # see the false-green and null-work guards at the bottom of
    # ``collect_metrics``.
    tests_pass: bool | None = False
    lint_clean: bool | None = False
    typecheck_clean: bool | None = False
    plan_completion_pct: float = 0.0
    plan_steps: int = 0

    # Efficiency
    cost_usd: float = 0.0
    # Cost provenance (Invariant P5): one of 'price_table' | 'cli' |
    # 'unpriced_proxy' | 'mixed'. The first three are what
    # :func:`resolve_cost_usd` resolves for a SINGLE invocation's spend.
    # 'mixed' is producible only by the ARCHITECT path, whose cost_usd is a
    # TWO-COMPONENT sum (architect spend + plan-judge spend): when those two
    # components' sources disagree, :func:`compose_cost_source` says so rather
    # than letting one label silently describe both. Defaults to 'cli' (the
    # trustworthy native-cloud source) so a result JSON persisted before this
    # field existed reads back sanely.
    cost_source: str = 'cli'
    workflow_duration_ms: int = 0
    turns_used: int = 0
    iterations: int = 0          # implementer re-invocations
    debug_cycles: int = 0        # debugger invocations

    # Judge spend, populated by TWO producers: ``collect_metrics`` copies it
    # from ``workflow.metrics`` on the implementer path (the completion judge,
    # ζ), and ``run_architect_eval`` sets it from the plan judge's verdict on
    # the architect path (θ, eval-revival υ). judge_cost_usd is a SUBSET of
    # cost_usd on BOTH paths, not disjoint — report generators should not
    # double-count. A $0.00 judge_cost_usd on an architect cell does not mean
    # the judge ran for free; it means the judge was either SKIPPED (tainted,
    # refused-with-a-plan, or an unscorable plan — judge_invocations reads 0)
    # or RAISED after being asked (judge_invocations ALSO reads 0 there,
    # because whether ``invoke_agent`` was ever reached before the raise
    # isn't determinable from the runner — but ``invocation_error`` then
    # names ``judge:raised: ...``, so the cell is legible as "judge raised,
    # spend unknown", not silently byte-indistinguishable from a skip).
    judge_invocations: int = 0
    judge_cost_usd: float = 0.0
    judge_early_exits: int = 0

    # Token usage
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_create_tokens: int = 0

    # Quality signals
    review_blocking_issues: int = 0
    review_suggestions: int = 0
    lines_changed: int = 0
    files_changed: int = 0

    # Inference speed
    tokens_per_second: float = 0.0       # output_tokens / generation_seconds
    # True when env_overrides has ANTHROPIC_BASE_URL — i.e. a PROXIED endpoint.
    # Repurposed by Invariant P5 as the cost-source signal: a proxied endpoint's
    # CLI cost is untrustworthy, so an unlisted model there resolves to
    # 'unpriced_proxy' (see resolve_cost_usd) rather than the raw CLI number.
    is_local_model: bool = False

    # Derived
    composite_score: float = 0.0

    # Recovery-behavior rubric (eval-revival η) — a rubric DISTINCT from the
    # base composite, populated ONLY for adversarial fixtures. ``None`` is the
    # C4 ``recovery_score | null`` sentinel for non-adversarial runs, kept
    # distinct from a genuinely scored ``0.0``.
    recovery_score: float | None = None

    # Whether this run scored an ADVERSARIAL fixture (eval-revival η), threaded
    # explicitly from the task record. Kept distinct from ``recovery_score`` so
    # that a recovery-scoring FAILURE (which the ``collect_metrics`` guard
    # degrades to ``recovery_score=None``) stays distinguishable from a
    # genuinely non-adversarial run: the report keys its ``adversarial`` column
    # on THIS flag, not on ``recovery_score is not None``. Otherwise an
    # adversarial run whose recovery scoring raised would be silently mislabeled
    # ``adversarial: false``, hiding the failure.
    adversarial: bool = False

    # Plan-quality rubric (eval-revival θ) — the architect-eval analogues of
    # recovery_score/adversarial above. ``plan_quality`` is a rubric DISTINCT
    # from the base composite, populated ONLY for architect runs
    # (role_under_test=='architect'); ``None`` is the C4 ``plan_quality | null``
    # sentinel for non-architect (implementer) runs, kept distinct from a
    # genuinely scored ``0.0``. Unlike recovery scoring (which degrades to None
    # on failure), an architect run that was actually ASKED always emits a
    # non-sentinel float — ``run_architect_eval`` degrades to the deterministic
    # ``score_plan_structure`` floor if the LLM plan judge fails.
    #
    # ``None`` therefore has exactly TWO causes, disambiguated by the markers
    # below: (1) not an architect run at all (``role_under_test != 'architect'``,
    # ``cap_tainted`` False), or (2) an architect run we could NOT measure
    # because the invocation was refused at the transport layer — a 429 cap hit
    # or auth failure — which carries ``cap_tainted=True`` and a stage-prefixed
    # ``invocation_error`` (task 3118). A scoring FAILURE is still never a cause:
    # it degrades to the structural floor, not to None.
    plan_quality: float | None = None

    # Which role this run put UNDER TEST (eval-revival θ / C4 role_under_test),
    # threaded from the candidate's ``EvalConfig.role``. ``None`` for legacy
    # runs; ``'architect'`` for an architect-eval run, ``'implementer'`` for the
    # ordinary path. The plan-quality report keys its column on this being
    # ``'architect'``.
    role_under_test: str | None = None

    # Infra-failure markers (task 3118) — the pair that keeps a TRANSPORT-layer
    # refusal from being read as a CONTENT-domain score.
    #
    # ``invocation_error`` records WHAT happened: a stage-prefixed marker for an
    # infra failure observed while producing this cell, e.g.
    # ``"architect:cap_hit: You've hit your session limit · resets 8pm"``,
    # ``"architect:model_not_found: ..."``, ``"architect:wedge: ..."``,
    # ``"architect:server_error: HTTP 529"`` or ``"judge:api_error: HTTP
    # 429"``. ``None`` means no infra failure was observed — including for an
    # ordinary content failure (an architect that ran fine and simply
    # produced a bad/absent plan), which must keep scoring on content.
    #
    # ``cap_tainted`` is the SCORING DECISION derived from it: this cell is not
    # a content measurement at all, so the plan_quality aggregates EXCLUDE it
    # rather than average in a fabricated zero (which would penalise whichever
    # candidate happened to be scheduled inside a cap window). Scope note
    # (REVISED by task 3099): the plan_quality surfaces exclude, and so do
    # ``build_composite_report``'s composite/quality pools for a PLAN-ONLY
    # trial. Those pools used to COUNT a tainted trial, on the grounds that they
    # measured the implementer-path gates — which a tainted architect cell
    # reports as an honest failure, not a fabricated score. That no longer holds
    # now that a plan-only composite is DERIVED from plan_quality: a fabricated
    # 0.0 would land in exactly the number ``select_survivors`` ranks on. The
    # ``trials`` denominator still counts every trial (the sample is never
    # silently shrunk) and ``plan_quality_cap_excluded`` reports the skips.
    #
    # NAME vs SCOPE (reviewer: design-coherence): the field name is ``cap_``
    # because the campaign that motivated it was a session-cap window, but the
    # set of causes is broader — any refusal that left NO model content to score
    # (cap hit, auth failure, model-not-found, zero-output wedge, harness
    # error). A cap hit is SCHEDULE-attributable (rerun later); a model-not-found
    # is a PERMANENT config error. Those must not read alike, so the CAUSE is
    # always carried in ``invocation_error`` and the report surfaces break the
    # exclusion count out by cause rather than reporting a bare "cap" total.
    #
    # ``cap_tainted`` is kept as an explicit boolean — not inferred from
    # ``plan_quality is None`` — for the same reason ``adversarial`` is kept
    # distinct from ``recovery_score``: a null score would otherwise be ambiguous
    # between "not an architect run" and "architect run we could not measure".
    # This is the same null-gating discipline ``_is_false_green`` /
    # ``_is_null_work`` already apply to ``tests_pass``, applied to the cap path.
    #
    # Both defaults are safe, so results persisted before these fields existed
    # read back unchanged (no marker, not tainted).
    invocation_error: str | None = None
    cap_tainted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def produced_a_plan(metrics: dict[str, Any]) -> bool:
    """THE plan-production predicate: did this architect actually emit a plan?

    Reads the PERSISTED ``plan_steps`` field above and nothing else. This is the
    metrics-dict twin of :func:`orchestrator.evals.judge.is_scorable_plan`, which
    asks the same question of the plan ARTIFACT — the form the report layer can
    never use, because by report time it holds only a persisted metrics dict.
    ``run_architect_eval`` derives ``plan_steps`` from the identical
    ``len(plan.get('steps') or [])``, so the two are equivalent BY CONSTRUCTION
    (pinned by ``TestProducedAPlan.test_equivalent_to_the_artifact_level_twin``
    rather than left to coincidence — the drift hazard
    ``report._has_plan_quality_score`` was written to close).

    **Why not ``plan_quality > 0``** (the rule this replaces everywhere, task
    3302): the two plan scorers disagreed exactly on a stepless artifact.
    :func:`judge.score_plan_structure` returns ``0.0`` for one as a deliberate
    ANTI-FABRICATION guard; the LLM plan judge used to have no such guard and
    could score the same artifact nonzero (Graphiti episode e2066ec6) — closed
    at the instrument by task 3303, which gave ``judge_plan_quality`` the same
    floor. Regardless of which scorer produced it, a nonzero ``plan_quality``
    is not evidence a plan exists, and a zero one is not evidence it does not
    — the number is the thing being decided, not the evidence for it.

    **Why not ``outcome == 'done'``** (task 2863 AMENDMENT §1): done-without-a-plan
    and blocked-with-a-good-plan cells both occur, so the workflow outcome
    answers a different question.

    Two in-repo surfaces already got this right and this predicate makes the
    pipeline agree with them rather than contradict them:
    ``scripts/eval_bootstrap_smoke.sh`` gates its smoke on ``metrics.plan_steps
    > 0 & outcome == 'done'``, and ``plans/eval-architect-effort-verdict-2026-07-27.md``
    hand-computed the campaign's ``planRate`` / ``meanPQ_all`` from
    ``plan_steps > 0`` because the pipeline's own ranking could not be used.

    Pure: no I/O, no mutation. Tolerates a missing key and an explicit ``None``
    (the empty shape ``len(... or [])`` guards) — both mean "no plan".
    """
    return int(metrics.get('plan_steps') or 0) > 0


def _is_false_green(m: EvalMetrics, max_iterations: int) -> bool:
    """The 404-bug signature: iteration cap hit with zero work but T/T/T.

    When every agent subprocess errors at the network layer (e.g. the vLLM
    bridge 404 bug, 2026-04-08), the workflow burns through its iteration
    budget making no code changes, verify then runs against the untouched
    baseline, and reports clean gates for whatever the pre-task tree already
    passed. Cost lands at $0 because the CLI never completed a usage-tracked
    turn. See ``docs/vllm-eval-status.md`` (2026-04-08 afternoon).
    """
    return (
        bool(m.tests_pass)
        and m.lines_changed == 0
        and m.files_changed == 0
        and m.iterations >= max_iterations
        and m.cost_usd == 0.0
    )


def _is_null_work(m: EvalMetrics) -> bool:
    """The NULL-byte implementer signature: real inference but zero code changes.

    When a vLLM model emits pad tokens (\\u0000) instead of tool calls — e.g.
    the reap-139b minimax_m2 parser bug (2026-04-08) — the Claude CLI reports
    "success" on each turn but no tool calls are executed, so the worktree is
    unchanged.  Cost is non-zero because inference actually ran, distinguishing
    this from the 404-bug (``_is_false_green``).  Verify gates then pass
    against the unchanged baseline → false T/T/T.
    """
    return (
        m.iterations > 0
        and m.lines_changed == 0
        and m.files_changed == 0
        and m.cost_usd > 0
    )


# Cap/auth reasons are free-form CLI text; clip them so the marker stays a
# single short line in result JSON and in the report tables.
_MARKER_REASON_CHARS = 80


def detect_invocation_error(
    result: AgentResult | None, *, backend: str = 'claude',
) -> str | None:
    """Name the TRANSPORT-layer refusal behind a failed invocation, if any.

    The third infra-failure guard alongside :func:`_is_false_green` /
    :func:`_is_null_work`, and the input to the ``cap_tainted`` scoring
    decision: it answers "did we ever actually get to ask the model?" so a CLI
    429 stops being scored as if the model had answered badly.

    Classification is DELEGATED to
    :func:`shared.invocation_outcome.classify_invocation` rather than
    re-matching 429/cap strings here — that seam already owns the cap/auth/
    model-not-found tables (under a drift guard) and short-circuits ``OK`` on
    ``result.success``, which is what stops a healthy run whose OUTPUT merely
    quotes a cap string from being falsely tainted. Forking those tables into
    the eval layer would reproduce exactly the false-positive class they were
    built to prevent.

    The explicit ``api_error_status == 429`` fallback is still required: 429 is
    deliberately EXCLUDED from the ``AuthFailed`` variant (it normally carries a
    cap-message body that the cap-TEXT tier recognises), so a BODY-LESS 429
    would otherwise fall through to ``Failure(unclassified)`` and go unmarked.
    It is checked BEFORE the wedge tier so a 429 that also timed out with zero
    turns is reported as the 429 it is, not as a generic wedge.

    ``ServerError`` (HTTP 5xx, including 529 Overloaded) is marked too: the
    provider refused, so the model never answered and the cell is unmeasurable
    for the same reason a 429 is. It is attributed ABOVE the wedge tier because
    a SIGTERM-flushed 5xx on a watchdog-killed CLI is a PROVIDER outage, not a
    local wedge — the 2026-07-29 incident this tier exists for. This arm also
    REPAIRS a silent regression: before ``classify_invocation`` grew the
    ``ServerError`` tier, a timed-out flushed 5xx reached this ladder as a
    ``ZeroOutputWedge`` and was marked; once that tier was ranked above the
    wedge upstream, the same case silently fell through to ``None`` until this
    arm was added.

    ``ZeroOutputWedge`` is marked too (reviewer: robustness): a full-timeout
    invocation with zero transcript turns produced no model answer at all, so it
    is unmeasurable for exactly the reason a 429 is, and scoring it 0.0 would
    reintroduce the fabricated zero this helper exists to remove.

    Returns ``None`` — deliberately, not a marker — for ``result is None``, for
    a successful invocation, and for every remaining variant (``NearCap``, which
    is a WARNING that did not block this invocation; ``CliLocalError``, which is
    explicitly not a cap; ``Failure(unclassified)``). An ORDINARY content failure
    (an architect that really ran and simply produced a bad or absent plan) is a
    genuine reliability signal that must keep scoring on content; laundering it
    into an exclusion would hide it.

    Pure: no I/O, no LLM, no mutation. Raising is left to the caller to guard
    (``run_architect_eval`` wraps the call) so a classifier bug degrades that
    one cell rather than being silently swallowed everywhere.
    """
    if result is None:
        return None

    outcome = classify_invocation(result, strict_confirm=True, backend=backend)
    if isinstance(outcome, CapHit):
        return f'cap_hit: {outcome.reason[:_MARKER_REASON_CHARS]}'
    if isinstance(outcome, AuthFailed):
        return f'auth_failed: HTTP {outcome.status}'
    if isinstance(outcome, ModelNotFound):
        return f'model_not_found: {outcome.reason[:_MARKER_REASON_CHARS]}'
    if isinstance(outcome, OK):
        # Preserve classify_invocation's OK short-circuit through the fallback
        # below: a successful invocation is never an infra refusal.
        return None
    if getattr(result, 'api_error_status', None) == 429:
        return 'api_error: HTTP 429'
    # ServerError sits BELOW the 429 fallback, because a 5xx is never a 429
    # and 429-body semantics must not move; it sits ABOVE the wedge branch so
    # this ladder's order tracks classify_invocation's own CapHit/NearCap >
    # ServerError > ZeroOutputWedge precedence (shared/src/shared/
    # invocation_outcome.py:398, :523) — a SIGTERM-flushed 5xx on a
    # watchdog-killed CLI is a provider outage, not a local wedge.
    if isinstance(outcome, ServerError):
        return f'server_error: HTTP {outcome.status}'
    if isinstance(outcome, ZeroOutputWedge):
        return 'wedge: zero-output timeout (no transcript turns)'
    return None


def compute_composite(m: EvalMetrics) -> float:
    """Pure quality score bounded to 0..1.

    - Fails tests (or ``tests_pass=None`` from the false-green/null-work
      guards) → score 0
    - blocking_rate = blocking_issues / plan_steps (larger tasks tolerate more issues)
    - debug_cycles get a light penalty (the system self-correcting is good)
    - Final score = quality, clamped to [0, 1]

    ``plan_completion_pct`` is still collected as a diagnostic signal (visible
    in result JSON) but no longer gates the composite. It was previously a
    multiplier, but local models that write correct code without updating
    plan.json status fields would get score 0 despite T/T/T gates.
    """
    if not m.tests_pass:
        return 0.0
    steps = max(m.plan_steps, 1)
    blocking_rate = m.review_blocking_issues / steps
    quality = 1.0 - (blocking_rate * 2.0) - (m.debug_cycles * 0.05)
    quality = max(quality, 0.0)
    quality = min(quality, 1.0)
    return round(quality, 4)


# The C4 efficiency-adjusted composite weights (decision 11 / this task λ):
# quality dominant, with cost + latency as secondary tie-breakers, summing to
# 1.0 so a best-on-every-axis run scores exactly 1.0. Code-owned (not config) so
# price can never silently override correctness.
DEFAULT_COMPOSITE_WEIGHTS: dict[str, float] = {
    'quality': 0.6,
    'cost': 0.2,
    'latency': 0.2,
}


def blend_composite(
    quality: float,
    cost_score: float,
    latency_score: float,
    *,
    tests_pass: bool | None,
    plan_only: bool = False,
    no_plan: bool = False,
    weights: dict[str, float] = DEFAULT_COMPOSITE_WEIGHTS,
) -> float:
    """The C4 efficiency-adjusted ``composite``: *quality* blended with
    normalized cost + latency scores, bounded to ``[0, 1]``.

    TWO HARD GATES, one per path, resting on ONE argument: a cheap+fast WRONG
    answer must never outrank a correct one.

    - Keeps ``compute_composite``'s gate (decision 11): a failing (or ``None``)
      *tests_pass* returns ``0.0`` regardless of the efficiency axes.
    - **no_plan** (task 3302): the architect produced NO PLAN, so the plan-only
      path returns ``0.0`` the same way. The asymmetry the earlier docstring got
      wrong is that the plan-only path bypasses *tests_pass* because "no test
      signal was collected" is not "the answer was wrong" — but "the architect
      produced no plan" IS the answer being wrong, and it is the one quality
      signal a plan-only cell ALWAYS has. It therefore gates exactly the way a
      failing workflow trial does.

    The bound *no_plan* closes: flooring such a cell's *quality* axis to ``0.0``
    (the report layer's :func:`_plan_quality_score`) bounds only the 0.6 quality
    weight. The remaining 0.2 cost + 0.2 latency is still collected, so an
    ungated no-plan cell caps at ``0.40`` — and a cell that is the sole member of
    its ``(fixture, 'plan_only')`` group takes the report layer's all-trials
    fallback baseline, earning ratios of ``1.0`` on both axes and banking the
    full ``0.40`` for having FAILED. Measured (task 3302 review): a no-plan cell
    at ``0.40`` outranked a config that produced a real 6-step plan at ``0.26``
    and survived ``select_survivors(top_k=2)``, while the plan-producing one was
    cut. Barring the cell from SEEDING its group's floor closes only the
    intra-group route; this gate closes the cross-group one.

    The gate is UNCONDITIONAL rather than scoped to *plan_only*, so a caller
    cannot silently bypass it by forgetting the flag. The report layer computes
    it as ``plan_only and not produced_a_plan(m)``, so it is never ``True`` on
    the workflow path — where the plan-production question does not exist.

    *quality* is the pure :func:`compute_composite` score; *cost_score* and
    *latency_score* are per-fixture NORMALIZED efficiency scores in ``[0, 1]``
    (``1.0`` == the cheapest / fastest run of the fixture — see
    ``report._ratio_score``), supplied by the report layer where the
    cross-config context to normalize exists. PURE and additive: this does not
    touch ``compute_composite``.

    **plan_only** (task 3099) — the cell under test is a PLAN-ONLY run (an
    architect eval freezes implementer/debugger/reviewer/verify), so no test was
    ever run and there is no test signal to gate on. The caller then supplies
    *quality* as the θ-rubric ``plan_quality`` and the ``tests_pass`` hard gate
    is deliberately bypassed: keeping it would read "no signal collected" as
    "the answer was wrong" and zero the number that drives survivor selection.

    Under *plan_only* the CALLER — not this function — owns the exclusion of an
    UNMEASURABLE cell: a cap-tainted trial (no ``plan_quality`` at all) must be
    dropped from the pool by the report layer, never passed here as a fabricated
    ``0.0``, which would penalise whichever candidate happened to be scheduled
    inside a cap window (the task-3118 invariant, one layer up). *no_plan* is
    owned by the caller for the same reason it owns *tests_pass* and *plan_only*:
    this function stays PURE over floats + bools and never sees a metrics dict.
    """
    # The two hard gates, as one parallel pair — not one gate plus a special
    # case. Left: the workflow path's answer was wrong. Right: the plan-only
    # path's answer was nothing.
    if not plan_only and not tests_pass:
        return 0.0
    if no_plan:
        return 0.0
    blended = (
        weights['quality'] * quality
        + weights['cost'] * cost_score
        + weights['latency'] * latency_score
    )
    return round(min(max(blended, 0.0), 1.0), 4)


def is_proxied_endpoint(env_overrides: dict[str, str] | None) -> bool:
    """Whether *env_overrides* points the CLI at a PROXIED endpoint.

    ``ANTHROPIC_BASE_URL`` is the single leaf that tells a vLLM/proxy candidate
    apart from a native-cloud one, and THREE sites key on it: ``runner
    .build_eval_orch_config`` (which seeds :func:`claude_endpoint_price_table`
    into ``config.prices`` for a proxied candidate, task 2820),
    :func:`collect_metrics` and ``runner.run_architect_eval`` (both of which
    pass it to :func:`resolve_cost_usd` as the do-not-trust-the-CLI-cost flag).
    Those three MUST agree — a seeded price table beside a "native, trust the
    CLI figure" flag would resolve one candidate's cost two different ways — so
    the predicate gets ONE home rather than three copies a future change to the
    proxy signal (an additional env leaf, say) could silently desynchronise
    (the single-home discipline of task 2459; reviewer: code-reuse).

    Total: ``None``, an empty mapping and an empty URL are all native-cloud.
    """
    return bool((env_overrides or {}).get('ANTHROPIC_BASE_URL'))


# ``_FALLBACK_PRICE`` (the DEFINED, logged degradation rate used only for a
# PROXIED endpoint whose model is unlisted — Invariant P5 / the
# loud-over-silent-degradation norm) and ``_rate`` (the PriceEntry-or-dict
# accessor) are imported from agents.invoke at the top of this module — a single
# home so the copy cannot drift (reviewer: code-reuse).
def resolve_cost_usd(
    input_tokens: int,
    output_tokens: int,
    *,
    model: str,
    prices: dict[str, Any] | None,
    cli_cost_usd: float,
    is_local_model: bool,
) -> tuple[float, str]:
    """Resolve ``(cost_usd, cost_source)`` for a run, per Invariant P5.

    ``cost_source`` ∈ ``{'price_table', 'cli', 'unpriced_proxy'}``:

    - **price_table** — the run's *model* is listed in *prices*: cost is the
      token-weighted price-table figure ``(in*input_per_1m +
      out*output_per_1m)/1e6`` (the identical formula
      ``invoke._estimate_cost`` uses). This wins even for a proxied endpoint —
      a listed price is always preferred to the (proxy-untrustworthy) CLI
      number.
    - **cli** — a NATIVE cloud endpoint (``is_local_model=False``) whose model
      is unlisted: the CLI's own ``cli_cost_usd`` is trustworthy for the real
      Anthropic API, so it is used verbatim.
    - **unpriced_proxy** — a PROXIED endpoint (``is_local_model=True`` —
      ``ANTHROPIC_BASE_URL`` set) whose model is unlisted: the CLI cost is
      WRONG for a proxied endpoint, so emit a loud WARNING (mirroring task
      2459) and fall back to a DEFINED ``_FALLBACK_PRICE`` figure — never a
      silent or raw-CLI number.

    Pure: no I/O, no config access; *prices* is passed in by the caller.
    Accepts ``PriceEntry`` objects and plain dicts interchangeably (via
    :func:`_rate`), exactly as invoke.py does.
    """
    entry = prices.get(model) if prices else None
    if entry is not None:
        cost = (
            input_tokens * _rate(entry, 'input_per_1m')
            + output_tokens * _rate(entry, 'output_per_1m')
        ) / 1_000_000
        return cost, 'price_table'

    if not is_local_model:
        # Native cloud endpoint — the CLI's own cost figure is trustworthy.
        return cli_cost_usd, 'cli'

    # Proxied endpoint with an unlisted model: the CLI cost is wrong here, so
    # warn loudly and fall back to a DEFINED rate rather than a silent number.
    logger.warning(
        'No configured price for proxied-endpoint model %r; the CLI cost is '
        'unreliable for a proxied endpoint, falling back to $%.2f/1M input, '
        '$%.2f/1M output (add it to config.prices to silence)',
        model, _FALLBACK_PRICE['input_per_1m'], _FALLBACK_PRICE['output_per_1m'],
    )
    cost = (
        input_tokens * _FALLBACK_PRICE['input_per_1m']
        + output_tokens * _FALLBACK_PRICE['output_per_1m']
    ) / 1_000_000
    return cost, 'unpriced_proxy'


def compose_cost_source(primary: str, *, secondary_cost_usd: float) -> str:
    """Collapse a TWO-COMPONENT cost sum's provenance to ONE honest label.

    :func:`resolve_cost_usd` answers "where did THIS figure come from?" for a
    single invocation. ``run_architect_eval``'s ``cost_usd`` is not a single
    invocation: it is the architect's spend PLUS the plan judge's spend. Only
    the ARCHITECT component is resolved — the plan judge is always a
    native-cloud opus call (``judge_plan_quality`` does not take the
    candidate's model or ``env_overrides``), so its CLI figure is trustworthy
    by construction and re-resolving it against the candidate's price table
    would price opus tokens at a vLLM rate.

    That is the mixed-provenance problem this helper exists to solve: the
    moment the architect side resolves to ``'price_table'`` or
    ``'unpriced_proxy'``, a single ``cost_source`` would silently describe two
    differently-sourced components. Rather than let one label stand in for two
    sources (the defect one level up, just quieter) or persist a second
    ``judge_cost_source`` field (which every result JSON written before it
    would read back as a default, mislabelling the existing corpus), the label
    COMPOSES:

    - *secondary_cost_usd* ``<= 0`` — no second component entered the sum, so
      the label describes the primary alone. This is every judge-SKIPPED
      architect branch (tainted, refused-with-a-plan, unscorable plan), plus
      the defensive negative case: :func:`coerce_cost_usd` already floors a
      nonsense figure to ``0.0`` upstream, and spend that did not happen must
      never MANUFACTURE a ``'mixed'`` label.
    - the two sources AGREE — that source. Today's architect cell is a native
      candidate resolving ``'cli'`` beside a ``'cli'`` judge, so a spending
      judge leaves it ``'cli'``, byte-identical to what already landed.
    - they DISAGREE — ``'mixed'``.

    ``'mixed'`` is deliberately the SAME label
    :func:`~orchestrator.evals.report._summarize_cost_source` already emits for
    the structurally identical situation one aggregation level up ("this figure
    blends more than one distinct cost source, so no single source honestly
    labels it"). Reusing the word keeps ONE vocabulary across the cell and the
    report row, needs zero report-side change (a single distinct cell label
    passes straight through ``_summarize_cost_source``), and means an operator
    reading ``mixed`` need not learn which level produced it.

    The second component's source is ``'cli'`` BY CONSTRUCTION (the judge's
    native-cloud opus call), so it is NOT a parameter: a ``secondary=`` keyword
    would be speculative generality with no caller to justify it (reviewer:
    over-engineering). A future caller whose second component IS differently
    sourced adds the keyword then — a non-breaking change, since every call
    site today means ``'cli'``.

    Pure: no I/O, no config, mirroring :func:`resolve_cost_usd`.
    """
    if secondary_cost_usd <= 0.0:
        return primary
    return primary if primary == 'cli' else 'mixed'


def coerce_cost_usd(value: object) -> float:
    """Safely coerce an untrusted ``cost_usd``-shaped *value* to a valid,
    non-negative float, defaulting to ``0.0`` for anything that is not one.

    Amendment (eval-revival υ, reviewer robustness + design-coherence): the
    ONE shared home for the "is this actually a usable dollar figure?"
    check, used at BOTH the plan judge's producer boundary
    (:class:`~orchestrator.evals.judge.PlanQualityVerdict` construction, all
    four POST-invoke return paths) and the architect runner's defensive read
    (``runner._verdict_cost_usd``) — so a Mock, ``None``, NaN, Infinity, or
    negative value degrades to ``0.0`` in exactly ONE place rather than two
    independently maintained checks that could drift apart. This closes the
    gap where the dataclass declared ``cost_usd: float`` but only the
    RUNNER's read enforced it — a second future consumer (a backfill
    script, ``prompt_opt``, ``verdict.to_dict()`` persisted anywhere) reading
    the verdict directly would otherwise inherit an untrusted value with no
    guard, the same "the instrument's coherence rests on its ONE caller
    remembering" shape task 3303 already closed for the scorability gate.

    isinstance-gated rather than a bare ``float(value or 0.0)`` coercion:
    ``unittest.mock.MagicMock`` configures ``__float__`` to return ``1.0`` by
    default, so a naive ``float()`` call would silently FABRICATE a dollar
    figure for any test double that never set the attribute (verified: a
    bare ``MagicMock()``'s ``float()`` returns ``1.0`` without raising) —
    the opposite of loud-over-silent-degradation, and worse than the status
    quo it would replace. ``isinstance`` correctly recognizes an
    unconfigured Mock (or ``None``, or any other non-numeric placeholder) as
    "not a real number" and returns the honest ``0.0`` default instead. This
    is also why the ~20 pre-existing judge tests that construct a bare
    ``MagicMock`` invoke result and never set ``cost_usd`` need no changes:
    they now read back ``cost_usd=0.0`` instead of a raw ``Mock`` instance,
    and none of them asserted on the field's value either way.

    A numeric-but-nonsense value (NaN, +/-Infinity, negative) is caught too
    — any of them would otherwise poison every downstream sum
    (``EvalMetrics.cost_usd = arch_cost_usd + judge_cost_usd``) and mean
    (report.py's per-config cost aggregation), and ``json.dump`` emits bare
    ``NaN``/``Infinity`` tokens that strict JSON consumers reject on the
    persisted result file. Logged, per the loud-over-silent norm — the
    caller still gets a usable ``0.0`` back, but an operator can see that an
    invocation answered with a nonsense cost rather than the figure quietly
    vanishing into a plausible-looking zero.
    """
    if not isinstance(value, (int, float)):
        return 0.0
    v = float(value)
    if not math.isfinite(v) or v < 0.0:
        logger.warning(
            f'cost_usd coercion received a non-finite or negative value '
            f'({v!r}) — degraded to 0.0 rather than poisoning a downstream '
            f'sum/mean or persisting invalid JSON.'
        )
        return 0.0
    return v


async def collect_metrics(
    workflow: TaskWorkflow,
    worktree: Path,
    task: dict,
) -> EvalMetrics:
    """Collect metrics from a completed workflow run."""
    wf_metrics = workflow.metrics

    # Plan completion
    plan = workflow.artifacts.read_plan() if workflow.artifacts else {}
    steps = plan.get('steps', [])
    done_count = sum(1 for s in steps if isinstance(s, dict) and s.get('status') == 'done')
    total_steps = len(steps) if steps else 1
    plan_completion = done_count / total_steps

    # Verification results (re-read from last run)
    from orchestrator.verify import run_verification
    verify = await run_verification(worktree, workflow.config)

    # Review stats from artifacts
    reviews = workflow.artifacts.aggregate_reviews() if workflow.artifacts else None
    blocking_issues = len(reviews.blocking_issues) if reviews else 0
    suggestions = len(reviews.suggestions) if reviews else 0

    # Git stats (diff against pre-task commit to capture all workflow changes)
    lines_changed, files_changed = await _git_diff_stats(
        worktree, task['pre_task_commit'],
    )

    # Inference speed metrics
    duration_secs = wf_metrics.total_duration_ms / 1000 if wf_metrics.total_duration_ms else 0.0
    tps = wf_metrics.total_output_tokens / duration_secs if duration_secs > 0 else 0.0
    # Via the shared predicate, NOT an inline env read: the same signal decides
    # whether build_eval_orch_config seeds the proxied price table, so the two
    # must not be able to drift apart (see is_proxied_endpoint).
    is_local = is_proxied_endpoint(workflow.config.env_overrides)

    # Cost provenance (Invariant P5): the CLI's own cost figure is wrong for a
    # proxied endpoint, so resolve cost from the config price table by the run's
    # model, tracking which source was used. collect_metrics is the IMPLEMENTER
    # path (run_eval), so the model under test is config.models.implementer.
    # ``run_architect_eval`` resolves through this SAME seam (task 3656) — with
    # ``model=EvalConfig.model`` and ``prices`` from its own eval orch config,
    # the leaves that actually produced its tokens — and then composes its
    # two-component (architect + plan-judge) label via ``compose_cost_source``.
    # So both eval paths share ONE cost-provenance seam rather than one
    # resolving and one copying the raw CLI number.
    run_model = workflow.config.models.implementer
    resolved_cost, cost_source = resolve_cost_usd(
        wf_metrics.total_input_tokens,
        wf_metrics.total_output_tokens,
        model=run_model,
        prices=workflow.config.prices,
        cli_cost_usd=wf_metrics.total_cost_usd,
        is_local_model=is_local,
    )

    m = EvalMetrics(
        tests_pass=verify.passed if verify else False,
        lint_clean=(not verify.lint_output) if verify else False,
        typecheck_clean=(not verify.type_output) if verify else False,
        plan_completion_pct=plan_completion,
        plan_steps=total_steps,
        cost_usd=resolved_cost,
        cost_source=cost_source,
        workflow_duration_ms=wf_metrics.total_duration_ms,
        turns_used=wf_metrics.total_turns,
        iterations=wf_metrics.execute_iterations,
        debug_cycles=wf_metrics.verify_attempts,
        judge_invocations=wf_metrics.judge_invocations,
        judge_cost_usd=wf_metrics.judge_cost_usd,
        judge_early_exits=wf_metrics.judge_early_exits,
        input_tokens=wf_metrics.total_input_tokens,
        output_tokens=wf_metrics.total_output_tokens,
        cache_read_tokens=wf_metrics.total_cache_read_tokens,
        cache_create_tokens=wf_metrics.total_cache_create_tokens,
        review_blocking_issues=blocking_issues,
        review_suggestions=suggestions,
        lines_changed=lines_changed,
        files_changed=files_changed,
        tokens_per_second=round(tps, 2),
        is_local_model=is_local,
    )
    # False-green guard — catches the 404-bug signature so the same class of
    # silent failure doesn't need manual quarantine in future runs.
    if _is_false_green(m, workflow.config.max_execute_iterations):
        logger.warning(
            'False-green signature for task %s: %d iters @ cap, '
            '$0 cost, 0 lines / 0 files changed, T/T/T — '
            'nulling gate fields so score is 0',
            task.get('id', '?'), m.iterations,
        )
        m.tests_pass = None
        m.lint_clean = None
        m.typecheck_clean = None
    # Null-work guard — catches the NULL-byte implementer signature where
    # real inference ran (cost > 0) but the model emitted pad tokens instead
    # of tool calls, producing zero code changes.
    elif _is_null_work(m):
        logger.warning(
            'Null-work signature for task %s: %d iters, '
            '$%.4f cost, 0 lines / 0 files changed — '
            'nulling gate fields so score is 0',
            task.get('id', '?'), m.iterations, m.cost_usd,
        )
        m.tests_pass = None
        m.lint_clean = None
        m.typecheck_clean = None

    m.composite_score = compute_composite(m)

    # Stamp the adversarial flag from the task record BEFORE recovery scoring
    # runs, so it is set regardless of whether recovery scoring succeeds, is
    # None, or raises below. The report keys its ``adversarial`` column on this
    # flag (not on ``recovery_score is not None``), so a recovery-scoring
    # FAILURE on an adversarial fixture renders as ``adversarial: true`` with a
    # null score — visibly distinct from a genuinely non-adversarial run.
    m.adversarial = bool(task.get('adversarial'))

    # Recovery-behavior scoring (eval-revival η) — a rubric DISTINCT from the
    # base composite, scored only for adversarial fixtures (returns None
    # otherwise). The local import breaks the metrics<->scoring cycle (scoring
    # imports EvalMetrics/compute_composite from this module), mirroring the
    # run_verification local import above. Guarded: a recovery-scoring bug
    # degrades to a named WARNING + null column and never nukes the run's
    # composite/cost/gate metrics.
    try:
        from orchestrator.evals.scoring import recovery_score_for_run
        m.recovery_score = await recovery_score_for_run(
            task, workflow.artifacts, worktree, task['pre_task_commit'],
        )
    except Exception:
        logger.warning(
            'recovery scoring failed for %s', task.get('id', '?'),
            exc_info=True,
        )

    return m


async def _git_diff_stats(worktree: Path, base_commit: str) -> tuple[int, int]:
    """Get lines changed and files changed vs the pre-task baseline commit.

    Returns (-1, -1) on git failure or subprocess error so callers can
    distinguish a genuine zero-change diff from a failed measurement.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            'git', 'diff', '--stat', f'{base_commit}..HEAD',
            cwd=str(worktree),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            logger.warning(
                '_git_diff_stats: git diff --stat failed (rc=%s) base=%s: %s',
                proc.returncode, base_commit,
                stderr.decode(errors='replace')[:200],
            )
            return -1, -1
        output = stdout.decode().strip()
        if not output:
            return 0, 0

        # Last line: " X files changed, Y insertions(+), Z deletions(-)"
        summary = output.split('\n')[-1]
        files_changed = 0
        lines_changed = 0

        m = re.search(r'(\d+) files? changed', summary)
        if m:
            files_changed = int(m.group(1))
        for m in re.finditer(r'(\d+) (?:insertions?|deletions?)', summary):
            lines_changed += int(m.group(1))

        return lines_changed, files_changed
    except Exception:
        logger.warning(
            '_git_diff_stats: git diff raised for base=%s',
            base_commit, exc_info=True,
        )
        return -1, -1
