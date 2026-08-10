"""Main eval orchestrator: run (task, config) pairs through the real workflow."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from shared.usage_gate import UsageGate

from orchestrator.agents.briefing import BriefingAssembler
from orchestrator.config import (
    BackendsConfig,
    BudgetsConfig,
    EffortConfig,
    ModelsConfig,
    OrchestratorConfig,
    PriceEntry,
    SandboxConfig,
)
from orchestrator.git_ops import GitOps
from orchestrator.scheduler import Scheduler, TaskAssignment
from orchestrator.workflow import (
    WorkflowOutcome,
    _inject_plan_tools_mcp,
    _meta_root_for_worktree,
    build_workflow,
)

from .configs import (
    EVAL_CONFIGS,
    JUDGE_OFAT_IMPLEMENTER_PIN,
    EvalConfig,
    claude_endpoint_price_table,
    matrix_pairs,
)
from .metrics import (
    EvalMetrics,
    coerce_cost_usd,
    collect_metrics,
    compose_cost_source,
    detect_invocation_error,
    is_proxied_endpoint,
    resolve_cost_usd,
)
from .profile import apply_eval_profile
from .snapshots import create_eval_worktree, read_python_pin

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / 'results'


@dataclass
class EvalResult:
    """Result of one (task, config) eval run."""

    task_id: str
    config_name: str
    outcome: str
    metrics: dict[str, Any]
    worktree_path: str
    wall_clock_ms: int = 0
    run_id: str = ''
    trial: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _find_repo_root(start: Path) -> Path:
    """Walk up from *start* to find the directory containing ``.git``."""
    current = start.resolve().parent if start.is_file() else start.resolve()
    while current != current.parent:
        if (current / '.git').exists():
            return current
        current = current.parent
    return start.resolve().parent


def load_task(task_path: Path) -> dict:
    """Load a task definition JSON file.

    Resolves ``project_root`` at runtime so task files are portable across
    machines.  If the path in the JSON starts with ``$REPO_ROOT`` it is
    expanded; if the hardcoded absolute path does not exist the discovered
    repository root is used instead.
    """
    with open(task_path) as f:
        task = json.load(f)

    repo_root = _find_repo_root(task_path)
    raw_root = task.get('project_root', '')

    if raw_root.startswith('$REPO_ROOT'):
        suffix = raw_root.replace('$REPO_ROOT/', '').replace('$REPO_ROOT', '')
        task['project_root'] = str(repo_root / suffix)
    elif raw_root and not Path(raw_root).exists():
        task['project_root'] = str(repo_root)

    return task


def build_eval_orch_config(
    config: EvalConfig,
    task: dict,
    base_config: OrchestratorConfig | None = None,
    memory_endpoint: str | None = None,
    architect_config: EvalConfig | None = None,
    judge_config: EvalConfig | None = None,
) -> OrchestratorConfig:
    """Build an OrchestratorConfig override for this eval run.

    Architect defaults to Claude opus-high (constant planning) — the frozen-plan
    ``run_eval`` and plan-only ``run_architect_eval`` paths both leave it there.
    Reviewer is the new 1× Opus comprehensive reviewer (matches production
    after the reviewer-panel trial replaced the 5× sonnet panel; merged
    via 594658fbe3 / 2c26a30bca).
    Only the implementer varies per eval config.

    ``architect_config`` (eval-revival μ) drives the both-live end-to-end matrix
    /confirm runs (``run_end_to_end``): when supplied, the architect's
    model/backend/effort are derived from THIS candidate instead of the hardcoded
    opus/claude/high, so an architect×implementer combo can be scored end-to-end.
    Left ``None`` (every existing caller), the architect pin is byte-identical to
    today, so the P1/B1 parity tripwire and the frozen-plan paths stay intact.

    ``judge_config`` (eval-revival ο) drives the judge OFAT axis (``run_ofat_stage``
    's judge branch via ``run_eval``): when supplied, the ζ completion judge's
    model/effort are derived from THIS candidate instead of the hardcoded
    sonnet/medium, so a cheaper judge (haiku) can be trialled and scored
    indirectly through μ's composite. Only model/effort derive — ``backends.judge``
    stays ``'claude'`` and ``budgets.judge`` stays ``0.50`` PINNED below (the judge
    is an always-Claude read-only quality call; mirrors the architect knob, which
    also never derives budget). Left ``None`` (every existing caller), the judge
    pin (sonnet/medium/claude/0.50) is byte-identical to today, so the parity
    tripwire holds.

    Two task-spec knobs override the defaults below:
      - ``max_execute_iterations``: hard ceiling on implementer iterations.
        Eval default is 20 (was 10) so workstation-tier slow models aren't
        capped before they finish — bumping was confirmed by user 2026-04-08.
      - ``max_review_cycles``: how many re-plan/re-review cycles after
        blocking issues. Eval default is 1; df_task_18 sets 2 because it
        empirically needs a second architect→implement→debug→review pass
        to clear all blockers.

    ``memory_endpoint`` (ε, D8): where eval memory writes go. Left ``None``
    (the default), the profile's non-routable null sentinel
    (``EVAL_PROFILE['fused_memory.url']``) stands — eval memory writes can
    never reach the production dark_factory store. A caller wanting to CAPTURE
    the intended writes (Boundary test B5, integration gate ι, or an operator)
    starts a ``RecordingMemorySink`` and passes its ``.url`` here; it overrides
    only ``fused_memory.url`` (preserving ``project_id`` and every other
    fused-memory leaf), so ``self.mcp.url`` — the single leaf every
    ``_write_*_to_memory`` POST funnels through — becomes the sink.

    The eval verify ``UV_PYTHON`` pin (BUG 2a) sources from the target
    ``project_root``'s current checkout (a 3.13-bearing tree) — mirroring
    production dispatch, which verifies under the target's current
    ``.python-version`` — and agrees with the setup ``uv sync`` pin
    ``snapshots._eval_setup_env`` reads from the SAME ``project_root`` (task
    2875, reverting task 2851's worktree-sourced pin). See the BUG 2a comment
    below.
    """
    if base_config is None:
        raise ValueError('build_eval_orch_config requires an explicit base_config')
    base = base_config

    # Architect pin: opus/claude/high by default (frozen-plan + plan-only paths),
    # or derived from architect_config for the both-live end-to-end run (μ). When
    # architect_config is None every value below equals the historical literal, so
    # the config is byte-identical to today and the P1/B1 parity tripwire holds.
    architect_model = architect_config.model if architect_config else 'opus'
    architect_backend = architect_config.backend if architect_config else 'claude'
    architect_effort = (architect_config.effort or 'high') if architect_config else 'high'

    # Judge pin: sonnet/medium by default (the ζ completion judge), or model/effort
    # derived from judge_config for the judge OFAT axis (ο). Backend and budget stay
    # PINNED below (always-Claude read-only judge — not derived). When judge_config
    # is None both values equal the historical literal, so the config is
    # byte-identical to today and the parity tripwire holds.
    judge_model = judge_config.model if judge_config else 'sonnet'
    judge_effort = (judge_config.effort or 'medium') if judge_config else 'medium'

    models = ModelsConfig(
        architect=architect_model,
        implementer=config.model,
        debugger=config.model,
        reviewer='opus',          # 1× opus comprehensive reviewer (production parity)
        merger='opus',
        module_tagger='sonnet',
        judge=judge_model,        # ζ completion judge — read-only, small budget (ο: derivable)
    )

    budgets = BudgetsConfig(
        architect=5.0,
        implementer=config.max_budget_usd,
        debugger=config.max_budget_usd / 2,
        reviewer=5.0,             # opus reviewer needs more headroom than sonnet
        merger=5.0,
        module_tagger=2.0,
        judge=0.50,
    )

    effort = EffortConfig(
        architect=architect_effort,
        implementer=config.effort or 'high',
        debugger=config.effort or 'high',
        reviewer='high',           # opus reviewer at high effort (matches defaults.yaml)
        merger='high',
        module_tagger='medium',
        judge=judge_effort,
    )

    backends = BackendsConfig(
        architect=architect_backend,
        implementer=config.backend,
        debugger=config.backend,
        reviewer='claude',        # reviewers always on Claude
        merger='claude',
        module_tagger='claude',
        judge='claude',           # judge always on Claude (read-only quality call)
    )

    # D5 fix: derive from the live base via model_copy instead of the
    # OrchestratorConfig(...) constructor — every field NOT named below is
    # inherited from `base` (via apply_eval_profile's own model_copy), so a
    # new production field can never silently regress to a pydantic default
    # in eval. apply_eval_profile(base) applies the documented EVAL_PROFILE
    # divergences (D3/D4/D8); the update dict below layers this run's
    # legitimate per-run overrides on top.
    profiled = apply_eval_profile(base)
    update: dict[str, Any] = {
        'models': models,
        'budgets': budgets,
        'effort': effort,
        'backends': backends,
        'max_execute_iterations': task.get('max_execute_iterations', 20),
        'max_review_cycles': task.get('max_review_cycles', 1),
        'judge_after_each_iteration': task.get('judge_after_each_iteration', True),
        'test_command': task.get('verify_commands', {}).get('test', base.test_command),
        'lint_command': task.get('verify_commands', {}).get('lint', base.lint_command),
        'type_check_command': task.get('verify_commands', {}).get('typecheck', base.type_check_command),
        'sandbox': SandboxConfig(enabled=False),
        'project_root': Path(task.get('project_root', str(base.project_root))),
        'env_overrides': config.env_overrides,
    }
    # BUG 2a: pin the eval verify interpreter to the target's own
    # .python-version (via verify_env['UV_PYTHON']) so `uv run pytest/ruff/
    # pyright` runs under the target's own 3.13 venv. verify_env is overlaid
    # LAST in verify._target_subprocess_env (wins) and survives
    # effective_verify_env, so the pin reaches every verify subprocess. Absent a
    # .python-version, inject nothing (fail-safe) — leave verify_env untouched.
    #
    # pin_source is the target project_root's CURRENT checkout — mirroring
    # production dispatch, which verifies under the target's current
    # .python-version — and the SAME tree snapshots._eval_setup_env reads for the
    # setup `uv sync`, so setup and verify agree on the interpreter. This reverts
    # task 2851's worktree-sourced pin: the eval worktree is checked out at the
    # fixture's pre_task_commit, which for older fixtures predates .python-version
    # → no pin → uv default 3.14t → aiosqlite ModuleNotFoundError at verify.
    # Sourcing from project_root (a 3.13-bearing tree) fixes that AND preserves
    # 2847's setup==verify agreement, now anchored to a 3.13-bearing tree instead
    # of an arbitrarily-old fixture baseline (task 2875).
    pin_source = Path(task.get('project_root', str(base.project_root)))
    pin = read_python_pin(pin_source)
    if pin is not None:
        update['verify_env'] = {**profiled.verify_env, 'UV_PYTHON': pin}
    # D8: an explicit recording endpoint layers over the profile's null
    # sentinel — override only fused_memory.url (preserving project_id and
    # every other fused-memory leaf from the profiled config). Left None, the
    # profile null sentinel stands and eval memory writes never reach production.
    if memory_endpoint is not None:
        update['fused_memory'] = profiled.fused_memory.model_copy(
            update={'url': memory_endpoint},
        )
    # Task 2820 (ν follow-up, escalation esc-2479-1 finding #3): auto-merge
    # claude_endpoint_price_table() into config.prices whenever THIS
    # candidate runs against a PROXIED endpoint — the SAME predicate
    # collect_metrics and run_architect_eval read before calling
    # resolve_cost_usd(..., is_local_model=...), which is why it has one home
    # (metrics.is_proxied_endpoint) instead of three inline env reads: the
    # table that gets seeded and the flag that decides whether to trust the CLI
    # figure must never disagree. Without this, an operator who forgets to seed
    # prices manually silently gets cost_source='unpriced_proxy' (there's
    # already a loud WARNING on that fallback, so this is convenience
    # hardening, not a silent-fail fix). Merge, not replace: profiled.prices
    # (the base/manually-seeded table) wins on conflict, mirroring
    # claude_endpoint_price_table()'s own default-table-then-candidate-table
    # override order.
    if is_proxied_endpoint(config.env_overrides):
        seeded_prices = {
            model: PriceEntry(**rates)
            for model, rates in claude_endpoint_price_table().items()
        }
        update['prices'] = {**seeded_prices, **profiled.prices}
    return profiled.model_copy(update=update)


async def run_eval(
    task_path: Path,
    config: EvalConfig,
    base_config: OrchestratorConfig | None = None,
    trial: int = 1,
    timeout_override: int | None = None,
    worktree_path: Path | None = None,
    memory_endpoint: str | None = None,
    judge_config: EvalConfig | None = None,
) -> EvalResult:
    """Run one (task, config) pair through PLAN→EXECUTE→VERIFY→REVIEW.

    When *worktree_path* is provided, the eval reuses an existing worktree
    instead of creating a fresh one.  The relocated
    ``.task-meta/<name>/plan.json`` (with step statuses) is used as the
    initial plan, so the workflow naturally skips already-completed steps —
    useful for resuming a blocked eval from the reviewer phase.

    *memory_endpoint* (ε, D8) is threaded straight to
    ``build_eval_orch_config``: left ``None``, the profile's null-sentinel
    isolation stands; pass a ``RecordingMemorySink().url`` to capture the
    intended memory writes instead of dropping them (see
    ``build_eval_orch_config`` for the full contract).

    *judge_config* (eval-revival ο) is the judge OFAT knob: left ``None`` (every
    existing caller) this is byte-identical to today — the result is labeled by
    ``config`` and no ``role_under_test`` stamp fires. Supplied (run_ofat_stage's
    judge branch), it (1) threads into ``build_eval_orch_config`` so ONLY the ζ
    completion judge's model/effort derive from the candidate (the implementer
    ``config`` stays pinned), (2) RELABELS the result to ``judge_config.name`` so
    the persisted JSON is keyed by the judge candidate rather than the pinned
    implementer (which would collide both judge rows), and (3) stamps
    ``metrics['role_under_test']='judge'`` so ``select_survivors`` groups the
    judge candidates as their own OFAT survivor axis.
    """
    task = load_task(task_path)
    task_id = task['id']
    project_root = Path(task['project_root'])

    # ο: when a judge candidate is under test, the persisted result must be keyed
    # by the JUDGE candidate — not the pinned implementer `config`, which would
    # collide both judge rows into one — so select_survivors ranks them as their
    # own axis. None (every existing caller) → label is config.name, unchanged.
    result_label = judge_config.name if judge_config is not None else config.name

    logger.info(f'Starting eval: {task_id} × {config.name} (trial {trial})')
    start_ms = int(time.monotonic() * 1000)

    # 1. Create or reuse worktree
    if worktree_path is not None:
        worktree = worktree_path
        run_id = worktree.name
        logger.info(f'Reusing existing worktree: {worktree}')
    else:
        worktree, run_id = await create_eval_worktree(
            project_root, task_id, task['pre_task_commit'],
            setup_commands=task.get('setup_commands'),
        )

    # 2. Build orchestrator config for this eval
    orch_config = build_eval_orch_config(
        config, task, base_config, memory_endpoint=memory_endpoint,
        judge_config=judge_config,
    )

    # 3. Build task assignment
    task_def = task.get('task_definition', {
        'title': task.get('name', task_id),
        'description': task.get('name', ''),
    })
    modules = task.get('modules', [])

    assignment = TaskAssignment(
        task_id=task_id,
        task=task_def,
        modules=list(modules),
    )

    # 4. Set up workflow dependencies
    git_ops = GitOps(orch_config.git, orch_config.project_root)
    scheduler, _ = _build_eval_scheduler(orch_config, task_id, list(modules))
    briefing = BriefingAssembler(orch_config)
    mcp = _EvalMcpStub(orch_config.fused_memory.url)

    # 5. Load plan — from existing worktree state or task JSON
    if worktree_path is not None:
        initial_plan = _resume_plan_from_worktree(worktree, task)
    else:
        initial_plan = task.get('plan')
    if not initial_plan:
        raise ValueError(
            f'Task {task_id} has no embedded plan. '
            f'Run --plan-only to generate one first.'
        )
    if not worktree_path:
        logger.info(f'Using fixed plan ({len(initial_plan.get("steps", []))} steps)')

    # 5b. Usage gate for account failover (judge hits Claude API, may cap)
    usage_gate: UsageGate | None = None
    if orch_config.usage_cap.enabled:
        try:
            usage_gate = UsageGate(orch_config.usage_cap)
        except Exception as exc:
            logger.warning(f'Failed to create UsageGate for eval: {exc} — running without failover')

    # 6. Run the real workflow
    workflow = build_workflow(
        assignment=assignment,
        config=orch_config,
        git_ops=git_ops,
        scheduler=scheduler,  # type: ignore[arg-type]
        briefing=briefing,
        mcp=mcp,  # type: ignore[arg-type]
        initial_plan=initial_plan,
        usage_gate=usage_gate,
    )

    # Override worktree since we created it ourselves
    workflow.worktree = worktree

    timeout_minutes = timeout_override or task.get('timeout_minutes', 60)
    try:
        # W9-γ: workflow.run() now returns a TerminalReport (TR-1); unwrap to
        # the WorkflowOutcome this function has always propagated.
        terminal_report = await asyncio.wait_for(
            workflow.run(), timeout=timeout_minutes * 60,
        )
        outcome = terminal_report.outcome
    except TimeoutError:
        logger.error(
            f'Eval {task_id} × {config.name} timed out after {timeout_minutes}m'
        )
        outcome = 'timeout'
    except Exception as e:
        logger.error(f'Eval {task_id} × {config.name} failed: {e}')
        outcome = WorkflowOutcome.BLOCKED

    wall_clock_ms = int(time.monotonic() * 1000) - start_ms

    # 7. Collect metrics
    try:
        metrics = await collect_metrics(workflow, worktree, task)
        metrics_dict = metrics.to_dict()
    except Exception as e:
        logger.warning(f'Metric collection failed: {e}')
        metrics_dict = {}
    # ο: stamp the judge OFAT axis so select_survivors groups judge runs as their
    # own survivor group (collect_metrics itself does not know which role/stage
    # invoked it — mirrors run_end_to_end's post-collect_metrics stamp). None →
    # no stamp fires, so every existing caller is byte-identical.
    if judge_config is not None:
        metrics_dict['role_under_test'] = 'judge'

    result = EvalResult(
        task_id=task_id,
        config_name=result_label,
        outcome=outcome.value if isinstance(outcome, WorkflowOutcome) else str(outcome),
        metrics=metrics_dict,
        worktree_path=str(worktree),
        wall_clock_ms=wall_clock_ms,
        run_id=run_id,
        trial=trial,
    )

    # 8. Persist result
    save_result(result)

    logger.info(
        f'Eval complete: {task_id} × {config.name} → {result.outcome} '
        f'(total={wall_clock_ms / 1000:.1f}s, '
        f'workflow={metrics_dict.get("workflow_duration_ms", 0) / 1000:.1f}s)'
    )
    return result


def _verdict_cost_usd(verdict: object) -> float:
    """Defensively read a plan-judge verdict's own invocation spend.

    The cost twin of the sibling defensive read
    ``getattr(verdict, 'invocation_error', None)`` in :func:`run_architect_eval`
    (task 3118's rationale, repeated here for the same reason): a
    monkeypatched, legacy, or third-party ``PlanQualityVerdict`` may not carry
    a ``cost_usd`` field at all, or may carry a non-numeric one. Either case
    must degrade exactly ONE field to ``0.0`` — never the whole eval cell, and
    never mistaken for a judge invocation that raised. An unguarded
    ``verdict.cost_usd`` read left inside the caller's ``try/except Exception``
    would swallow an ``AttributeError`` there and log "plan judge raised",
    mis-attributing an unreadable FIELD to a judge that actually ran and
    answered; left unguarded entirely, a non-numeric value (e.g. ``None``)
    would raise ``TypeError`` OUTSIDE that try/except, in the
    ``EvalMetrics(...)`` construction, crashing the whole cell.

    The ``getattr`` (missing-field / wrong-object) defense stays local to this
    function — it is specific to reading an UNTRUSTED ``verdict``, which
    ``coerce_cost_usd`` does not know how to do. Once a candidate value is in
    hand, the actual "is it a usable dollar figure" check — including the
    NaN/Infinity/negative guard (amendment, reviewer robustness: a judge
    answering a non-finite or negative cost would otherwise poison
    ``arch_cost_usd + judge_cost_usd`` and every downstream report.py mean) —
    is delegated to :func:`~orchestrator.evals.metrics.coerce_cost_usd`, the
    SAME helper the judge's own producer-side return paths use, so the two
    sides of this contract cannot drift apart.
    """
    return coerce_cost_usd(getattr(verdict, 'cost_usd', 0.0))


async def run_architect_eval(
    task_path: Path,
    config: EvalConfig,
    base_config: OrchestratorConfig | None = None,
    trial: int = 1,
    timeout_override: int | None = None,
    memory_endpoint: str | None = None,
) -> EvalResult:
    """Run ONE architect eval: invoke the architect LIVE and score its plan (θ).

    Unlike :func:`run_eval` (which FREEZES the plan and scores the implementer),
    this drives ONLY the architect — the ``_run_plan_only`` invocation sequence
    (create_eval_worktree at ``pre_task_commit`` → ``TaskArtifacts.init`` →
    ``briefing.build_architect_prompt`` → ``invoke_agent(ARCHITECT)`` →
    ``artifacts.read_plan``) — so every downstream role
    (implementer/debugger/reviewer/verify) is FROZEN (decision 8: noise
    isolation + token savings). The architect runs with THIS candidate's
    model/backend/effort/env_overrides, not the hardcoded opus-high the
    implementer path pins.

    The produced plan is scored two ways: :func:`judge_plan_quality` (the LLM
    judge, against the REAL landed reference diff
    ``pre_task_commit..reference.post_task_commit`` — the always-available
    ground truth since ζ fixtures frequently carry ``plan: null``), degrading to
    the deterministic :func:`score_plan_structure` floor on ANY judge failure.
    The judge is reached only for a plan that is
    :func:`~orchestrator.evals.judge.is_scorable_plan` (task 3302), and the
    judge REFUSES such an artifact itself (task 3303) — defense in depth around
    one predicate. Left ungated, an LLM judge returns a confident nonzero score
    for the very shape ``score_plan_structure`` short-circuits to 0.0,
    persisting a cell whose ``plan_steps=0`` contradicts its own
    ``plan_quality``; with both the call site and the instrument consulting
    ``is_scorable_plan``, no caller can write that cell.

    ``plan_quality`` is therefore a non-sentinel float whenever the architect
    was actually ASKED — with ONE deliberate exception (task 3118): when the
    architect invocation failed in a way that left NO model content to score
    AND no SCORABLE plan artifact was produced, scoring is skipped and the cell
    records ``plan_quality=None`` plus ``cap_tainted=True`` and a stage-prefixed
    ``invocation_error``. ``plan_quality is None`` on an architect run means
    exactly that case, and the plan-quality aggregates EXCLUDE such cells rather
    than averaging in a fabricated zero.

    Which failures taint, and WHY the line falls where it does:

    - **Transport refusal with no SCORABLE plan** (429 cap hit, auth failure,
      model-not-found, zero-output wedge) → TAINTED. The candidate was never
      asked; the outcome is a property of the schedule or of our configuration,
      not of the candidate. "No scorable plan" covers both an absent artifact
      and the header-only stub ``create_plan`` writes with zero steps — see
      :func:`~orchestrator.evals.judge.is_scorable_plan`.
    - **Harness error** (worktree/config/briefing/artifact-read raised) →
      TAINTED, for the same reason: charging our own crash to the candidate
      would be a fabricated score.
    - **Transport refusal that still left a plan WITH STEPS** (a cap landing
      mid-run, after the architect wrote real steps through plan-tools) → NOT
      tainted. Model content exists, so the deterministic structural floor is a
      genuine content measurement; the marker is recorded and the LLM judge is
      skipped (it would 429 in the same window), but the cell stays in the
      aggregate.
    - **Timeout** → NOT tainted, deliberately. It is marked
      (``architect:timeout: ...``) so it is never silently indistinguishable
      from a bad plan, but unlike a cap hit it is CANDIDATE-attributable: the
      model was asked and did not finish inside the operator's budget. Excluding
      it would let a pathologically slow candidate dodge the penalty its
      competitors paid, so it keeps scoring on content (an absent plan scores
      the structural floor, 0.0) and the reliability signal is carried in BOTH
      ``outcome='timeout'`` and ``invocation_error``.
    - **Ordinary content failure** (an architect that ran fine and merely
      produced a bad or absent plan) → NOT marked at all, scores 0.0. That is a
      real reliability signal, not an infra failure. When the plan is stepless
      the LLM judge is SKIPPED and the structural floor (0.0) is persisted
      directly, so the score can never be the judge's opinion of an artifact
      that carries nothing to judge (task 3302 gates here; task 3303 makes
      :func:`~orchestrator.evals.judge.judge_plan_quality` refuse it too, so
      the guarantee no longer depends on this call site alone).
    - **Refusal of the JUDGE alone** → recorded in ``invocation_error``
      (prefixed ``judge:``) but does NOT taint: the structural floor is still
      derived from a real produced plan.

    The result carries ``role_under_test='architect'`` and is persisted via
    :func:`save_result`.

    Cost accounting (eval-revival υ): the cell's ``cost_usd`` is the TOTAL
    spend of producing and scoring the plan — the architect invocation plus
    the plan judge's invocation, when the judge was actually called.
    ``judge_cost_usd`` is the judge's share of that total, a SUBSET of
    ``cost_usd`` and never a disjoint addend (metrics.py:69-71). Every
    judge-skipped branch (tainted, refused-with-a-plan, unscorable plan)
    spends nothing on the judge, so ``cost_usd`` there is architect spend
    alone.

    Cost PROVENANCE (task 3656): the ARCHITECT component of that total is
    RESOLVED per Invariant P5, through the same
    :func:`~orchestrator.evals.metrics.resolve_cost_usd` seam
    ``collect_metrics`` uses on the implementer path — so a PROXIED architect
    candidate no longer keeps the raw CLI figure P5 calls untrustworthy for a
    proxy, and ``cost_source`` is DERIVED rather than left at an unverified
    dataclass default. The PLAN-JUDGE component deliberately keeps its CLI
    figure: the judge is always a native-cloud opus call
    (:func:`~orchestrator.evals.judge.judge_plan_quality` takes neither the
    candidate's model nor its ``env_overrides``), so re-resolving it against
    the candidate's price table would price opus tokens at a vLLM rate. The
    cell's single ``cost_source`` therefore reads ``'mixed'``
    (:func:`~orchestrator.evals.metrics.compose_cost_source`) whenever those
    two components disagree AND the judge actually spent — the
    operator-visible answer to the mixed-provenance question, rather than one
    label quietly standing in for two sources. A NATIVE candidate resolves
    ``'cli'`` beside a ``'cli'`` judge, so today's cells keep both figure and
    label byte-identical. A cell that spent NOTHING (timeout, harness error,
    pre-invoke cap) skips the resolution altogether: $0.00 has no provenance to
    resolve, and resolving it would attach a loud degradation WARNING to spend
    that never happened.
    """
    from orchestrator.agents.briefing import BriefingAssembler
    from orchestrator.agents.invoke import invoke_agent
    from orchestrator.agents.roles import ARCHITECT
    from orchestrator.artifacts import TaskArtifacts
    from orchestrator.evals import snapshots
    from orchestrator.evals.judge import (
        is_scorable_plan,
        judge_plan_quality,
        score_plan_structure,
    )

    task = load_task(task_path)
    task_id = task['id']
    project_root = Path(task['project_root'])
    pre = task['pre_task_commit']

    logger.info(
        f'Starting architect eval: {task_id} × {config.name} (trial {trial})'
    )
    start_ms = int(time.monotonic() * 1000)

    # 1. Worktree at the fixture's pre_task_commit.
    worktree, run_id = await snapshots.create_eval_worktree(
        project_root, task_id, pre, setup_commands=task.get('setup_commands'),
    )

    plan: dict = {}
    # Renamed from ``cost_usd`` (eval-revival υ): this local is ONLY the
    # architect invocation's spend. The cell's persisted ``cost_usd`` below
    # additionally folds in the plan judge's spend (``judge_cost_usd``), so
    # the two must never be confused under one name.
    arch_cost_usd = 0.0
    # The architect invocation's TOKEN USAGE — the price-table inputs Invariant
    # P5 needs (see resolve_cost_usd). Pre-try for the same reason arch_cost_usd
    # is: the timeout and harness-error paths never bind ``result``, so the
    # honest zeros must already exist.
    arch_input_tokens = 0
    arch_output_tokens = 0
    # The REST of the token profile ``collect_metrics`` stamps on the implementer
    # path. Not P5 inputs — they price nothing — but on a native Claude run the
    # cache reads typically DOMINATE the profile, so a cell reporting input/output
    # beside a zeroed cache block would read as a far smaller run than it was
    # (reviewer: completeness). Pre-try for the same reason as above.
    arch_cache_read_tokens = 0
    arch_cache_create_tokens = 0
    arch_turns = 0
    # Whether this candidate runs against a PROXIED endpoint, the third P5
    # input. Read from the EvalConfig — not the built orch config — because it
    # is literally what ``invoke_agent(env_overrides=config.env_overrides)``
    # below is handed, and because reading it here means a harness crash inside
    # the try cannot leave the proxy signal unknowable. Through the SAME
    # ``is_proxied_endpoint`` predicate ``build_eval_orch_config`` keys its
    # price-table seeding on (and ``collect_metrics`` reads on the implementer
    # path), so the seeded table and the trust-the-CLI flag cannot drift apart.
    is_local = is_proxied_endpoint(config.env_overrides)
    arch_duration_ms = 0
    outcome = 'done'
    # The architect-side infra marker (task 3118): WHAT went wrong, if anything.
    # Set on the transport-refusal path (classified from the AgentResult) AND on
    # the timeout / harness-exception paths below, so no zero-content failure is
    # left byte-indistinguishable from a genuinely terrible plan.
    arch_error: str | None = None
    # Whether that failure left NO model content to score — the input to the
    # taint decision, kept SEPARATE from the marker because the two differ for a
    # timeout: a timeout is marked (so it is legible) but is candidate-
    # attributable, so it keeps scoring on content. See the scoring block below.
    arch_unmeasurable = False
    # Honor the operator's --timeout around the LIVE architect invoke, exactly
    # as run_eval bounds workflow.run(). timeout_override is in MINUTES
    # (run_eval convention — the CLI threads the same --timeout to both); without
    # this a hung architect run would block indefinitely, bounded only by
    # max_budget_usd.
    timeout_minutes = timeout_override or task.get('timeout_minutes', 60)
    # Hoisted out of the try so its PRICE TABLE survives to the cost resolution
    # after the finally. A harness crash before it is built leaves an explicit
    # ``None`` (which resolve_cost_usd tolerates) rather than an
    # UnboundLocalError that would lose the whole cell — and that path never
    # invoked the architect (0 tokens, $0.00), so no price table could have
    # changed the number anyway.
    orch_config: OrchestratorConfig | None = None
    try:
        # 2. Eval orch config (project_root / verify / profile parity).
        orch_config = build_eval_orch_config(
            config, task, base_config, memory_endpoint=memory_endpoint,
        )

        # 3. Init artifacts so the architect has a place to write plan.json.
        #    Target the RELOCATED .task-meta/<name>/ root — the SAME root the
        #    injected plan-tools server writes to (BUG 1: writer==reader), so
        #    read_plan() below picks up what the architect's plan-tools calls
        #    persisted, exactly like real dispatch.
        artifacts = TaskArtifacts(worktree, meta_root=_meta_root_for_worktree(worktree))
        task_def = task.get('task_definition', {})
        artifacts.init(
            task_id,
            task_def.get('title', ''),
            task_def.get('description', ''),
            base_commit=pre,
        )

        # 4. Build the architect prompt and invoke the architect LIVE with THIS
        #    candidate's model/backend/effort/env_overrides.
        briefing = BriefingAssembler(orch_config)
        prompt = await briefing.build_architect_prompt(task_def, worktree=worktree)
        # Wire plan-tools MCP via the SAME production seam real dispatch uses
        # (workflow._invoke): relocated meta_root + direct-interpreter launch.
        # strict_mcp_config stays default False so the ambient .mcp.json
        # escalation/fused-memory servers still merge, mirroring _invoke.
        mcp_config = _inject_plan_tools_mcp(None, worktree)
        result = await asyncio.wait_for(
            invoke_agent(
                prompt=prompt,
                system_prompt=ARCHITECT.system_prompt,
                cwd=worktree,
                model=config.model,
                max_turns=task.get('max_architect_turns', 50),
                max_budget_usd=config.max_budget_usd,
                allowed_tools=ARCHITECT.allowed_tools or None,
                disallowed_tools=ARCHITECT.disallowed_tools or None,
                effort=config.effort or 'high',
                backend=config.backend,
                env_overrides=config.env_overrides or None,
                mcp_config=mcp_config,
            ),
            timeout=timeout_minutes * 60,
        )
        arch_cost_usd = result.cost_usd
        # ``or 0``, not a bare read: AgentResult declares both ``int | None``,
        # and a provider that did not report usage must persist an honest 0
        # rather than a None that would poison the price-table arithmetic.
        arch_input_tokens = result.input_tokens or 0
        arch_output_tokens = result.output_tokens or 0
        # Same ``or 0`` contract: both cache counts are ``int | None`` too, and
        # ``turns`` is 0 when the provider does not track it.
        arch_cache_read_tokens = result.cache_read_tokens or 0
        arch_cache_create_tokens = result.cache_create_tokens or 0
        arch_turns = result.turns or 0
        arch_duration_ms = result.duration_ms
        if not result.success:
            outcome = 'blocked'
        # Was this a TRANSPORT-layer refusal (a 429 cap hit / auth failure — we
        # never got to ask the model) rather than an ordinary content failure?
        # The outcome vocabulary stays 'blocked' either way; the distinction
        # lives in metrics, which is what the report and the persisted JSON
        # read. Guarded so a classifier bug degrades to an unmarked cell rather
        # than nuking the whole run.
        try:
            arch_error = detect_invocation_error(result, backend=config.backend)
            arch_unmeasurable = arch_error is not None
        except Exception:
            logger.warning(
                f'invocation-error classification raised for {task_id} × '
                f'{config.name}; leaving the cell unmarked',
                exc_info=True,
            )
        # 5. Read the produced plan artifact (the scoring input).
        plan = artifacts.read_plan() or {}
    except TimeoutError:
        logger.error(
            f'Architect eval {task_id} × {config.name} timed out after '
            f'{timeout_minutes}m'
        )
        outcome = 'timeout'
        # MARKED but NOT unmeasurable: see the scoring block for why a timeout
        # keeps scoring on content while a cap hit does not.
        arch_error = arch_error or f'timeout: no answer within {timeout_minutes}m'
    except Exception as e:
        logger.error(f'Architect eval {task_id} × {config.name} failed: {e}')
        outcome = 'blocked'
        # A HARNESS failure (worktree/config/briefing/artifact-read raised), not
        # a candidate failure — the candidate was never even asked, so scoring it
        # 0.0 would charge our own crash to it. Marked AND unmeasurable. An
        # already-classified transport refusal is more specific, so it wins.
        # The reason is whitespace-collapsed and clipped so the marker stays a
        # single short line in the result JSON and the report tables.
        reason = ' '.join(str(e).split())[:80]
        arch_error = arch_error or f'harness_error: {type(e).__name__}: {reason}'
        arch_unmeasurable = True
    finally:
        # Plan already read above; the worktree is no longer needed (scoring
        # reads the in-memory plan + the committed reference diff).
        await snapshots.cleanup_eval_worktree(project_root, worktree)

    # 6. Materialize the landed reference diff — the always-available ground
    #    truth (ζ fixtures frequently carry plan: null).
    reference = task.get('reference') or {}
    post = reference.get('post_task_commit')
    reference_diff = ''
    if post:
        try:
            reference_diff = await snapshots.get_diff_between_commits(
                project_root, pre, post,
            )
        except Exception as e:
            logger.warning(f'reference diff failed for {task_id}: {e}')

    # 7. Score the produced plan: LLM judge vs the landed diff, degrading to the
    #    deterministic structural floor on ANY judge failure so plan_quality is a
    #    non-sentinel float — UNLESS the architect invocation failed in a way
    #    that left NOTHING to score (see the docstring's taint table for which
    #    failures qualify and why a timeout deliberately does not).
    plan_quality: float | None = None
    judge_error: str | None = None
    # The plan judge's OWN spend + invocation count (eval-revival υ). Every
    # judge-SKIPPED branch below (tainted, refused-with-a-plan, unscorable-
    # plan) leaves these at their honest zero defaults with no extra code;
    # only the healthy else-branch, where the judge is actually called, sets
    # them from the returned verdict. judge_cost_usd is a SUBSET of the
    # cell's cost_usd below, not disjoint (metrics.py:69-71) — invocations is
    # stamped on the judge having been CALLED, not on the cost being nonzero,
    # since a $0.00 refusal is still an invocation.
    judge_cost_usd = 0.0
    judge_invocations = 0
    # The taint decision consults whether the artifact is SCORABLE, not merely
    # whether one exists (reviewer: correctness). A session cap can land MID-run,
    # after the architect has already written plan.json through plan-tools MCP —
    # the common shape of a cap hit during a long campaign. When a plan WITH
    # STEPS landed, nulling it would discard a genuine content measurement
    # (exactly what the judge-only branch is careful NOT to do) while persisting
    # a self-contradictory cell: plan_steps > 0 alongside "we never got to ask
    # the model".
    #
    # Raw truthiness was INSUFFICIENT: create_plan — the architect's first
    # plan-tools call, and the only one it can reach before a 429 — persists a
    # truthy header-only dict with zero steps. A cap landing right after it left
    # tainted=False while score_plan_structure short-circuited to a fabricated
    # 0.0. is_scorable_plan is that short-circuit's own test, so the two can no
    # longer disagree.
    tainted = arch_unmeasurable and not is_scorable_plan(plan)
    if tainted:
        # We never got to ask the model AND no SCORABLE plan exists, so every
        # available number would be FABRICATED — and a fabricated 0.0 is
        # byte-indistinguishable from a genuinely terrible plan, which is the
        # defect this marker exists to remove. The judge is skipped rather than
        # invoked-and-discarded: it has nothing to judge, and inside a cap
        # window it would 429 too (the second-order failure that manufactured
        # the 0.0), burning an opus call on a doomed request.
        logger.warning(
            f'Architect eval {task_id} × {config.name}: invocation refused with '
            f'no scorable plan artifact ({arch_error}) — plan judge skipped, '
            f'plan_quality=None, cell marked cap_tainted (NOT scored 0.0)'
        )
    elif arch_unmeasurable:
        # Refused, but a plan WITH STEPS landed first: score it on the
        # deterministic structural floor and do NOT taint — symmetric with the
        # judge-only case, where a content-derived score survives an infra
        # refusal. Gating on is_scorable_plan is what makes that justification
        # true: this branch now fires ONLY when the floor can actually derive a
        # content score, never when it would short-circuit to a fabricated 0.0.
        # The LLM judge is still skipped: inside the same cap window it would
        # 429 too, and the floor is the exact degradation path a judge failure
        # already takes. The marker is still recorded so the reader knows why.
        plan_quality = score_plan_structure(plan)
        logger.warning(
            f'Architect eval {task_id} × {config.name}: invocation refused '
            f'({arch_error}) but a plan artifact exists — LLM judge skipped, '
            f'scored on the structural floor ({plan_quality}), NOT tainted'
        )
    elif not is_scorable_plan(plan):
        # The architect ran FINE and produced nothing worth scoring, so the
        # deterministic ANTI-FABRICATION floor (Graphiti e2066ec6) applies:
        # score_plan_structure short-circuits a stepless artifact to 0.0, and an
        # LLM opinion of that artifact would write a cell whose own plan_steps=0
        # CONTRADICTS its score — the shape the report-layer floor
        # (metrics.produced_a_plan, task 3302) has to defend the existing corpus
        # against.
        #
        # judge_plan_quality now refuses such an artifact ITSELF and returns the
        # same floor (task 3303), so this gate is no longer the sole correctness
        # guarantee — but it remains LOAD-BEARING, for three things the
        # instrument-level guard cannot do from where it stands:
        #   1. the taint decision below (NOT tainted: a content failure, not the
        #      3118 "we never got to ask" exclusion),
        #   2. the log line naming task_id × config.name, which the judge cannot
        #      see, and
        #   3. skipping the async call and the reference-diff-bearing prompt
        #      entirely, on the arch_unmeasurable branch's own justification:
        #      nothing to judge, a 429 inside a cap window anyway, and an opus
        #      call on an unjudgeable artifact is pure waste.
        # Both gates consult the ONE is_scorable_plan predicate, which is the
        # point: they cannot drift into disagreeing about what a plan is.
        #
        # NOT tainted: no infra failure occurred. This is a CONTENT failure and
        # must keep scoring on content — a genuine 0.0, distinct from the
        # "we never got to ask" exclusion above (task 3118).
        plan_quality = score_plan_structure(plan)
        logger.warning(
            f'Architect eval {task_id} × {config.name}: architect ran '
            f'successfully but produced no scorable plan — plan judge skipped, '
            f'scored on the structural floor ({plan_quality}), NOT tainted'
        )
    else:
        try:
            verdict = await judge_plan_quality(plan, reference_diff, task)
            plan_quality = verdict.plan_quality
            # getattr, not attribute access: a monkeypatched or legacy verdict
            # without the field must not break scoring.
            judge_error = getattr(verdict, 'invocation_error', None)
            # The judge WAS called, whatever it answered — stamp the
            # invocation before touching its cost, so a $0.00 refusal still
            # counts as one call (report.py:806-809 reads the pair together;
            # a nonzero cost beside invocations=0 would be self-contradictory).
            # _verdict_cost_usd is a DEFENSIVE read (like the getattr above):
            # an unreadable cost field degrades to 0.0 rather than raising
            # here (mis-attributing the failure to "the judge raised") or
            # later in the EvalMetrics construction (crashing the cell).
            judge_invocations = 1
            judge_cost_usd = _verdict_cost_usd(verdict)
        except Exception as e:
            # judge_invocations/judge_cost_usd stay at their 0.0/0 defaults —
            # NOT bumped to 1 here (amendment, reviewer robustness): whether
            # invoke_agent was even reached before the raise is NOT
            # determinable from this side (the raise could be pre-invoke, in
            # prompt assembly, or post-invoke, in parsing/logging inside
            # judge_plan_quality's own try/except), so crediting an
            # invocation that may never have happened would be its own
            # fabrication. But leaving BOTH fields at zero would make this
            # cell byte-indistinguishable from one where the judge was
            # SKIPPED (tainted / cap-refusal-with-a-plan / unscorable-plan) —
            # exactly the illegibility this task exists to remove, just moved
            # one level up. So record the fact on judge_error instead: it
            # rides into stage_markers → invocation_error below as
            # "judge:raised: ...", so the cell reads "judge raised, spend
            # unknown" rather than silently looking judge-free.
            reason = ' '.join(str(e).split())[:80]
            judge_error = f'raised: {type(e).__name__}: {reason}'
            logger.warning(
                f'plan judge raised for {task_id}; degrading to structural floor',
                exc_info=True,
            )
        if plan_quality is None:
            plan_quality = score_plan_structure(plan)

    wall_clock_ms = int(time.monotonic() * 1000) - start_ms

    # The marker names WHICH stage failed; the join keeps the field well-defined
    # if both ever fire (today an architect-side refusal skips the judge, so at
    # most one does). cap_tainted keys on ``tainted``, NOT on the marker: a
    # judge-only refusal, a timeout, and a refusal that still left a plan behind
    # all keep a content-derived score, so excluding those cells would discard
    # valid measurements.
    stage_markers = [
        f'{stage}:{marker}'
        for stage, marker in (('architect', arch_error), ('judge', judge_error))
        if marker
    ]

    # Cost provenance (Invariant P5) for the ARCHITECT component, through the
    # SAME seam collect_metrics uses on the implementer path — a proxied
    # endpoint's own CLI figure is untrustworthy, so it must be resolved rather
    # than copied. Two non-obvious arguments:
    #   - ``model=config.model``: this path calls
    #     ``invoke_agent(model=config.model)`` DIRECTLY, so the EvalConfig's
    #     model is literally what produced the tokens being priced.
    #     ``orch_config.models.architect`` would be the WRONG key —
    #     build_eval_orch_config pins it to the frozen 'opus' default whenever
    #     architect_config is None, which is every run_architect_eval caller,
    #     so pricing would silently look up opus for a fable or vLLM candidate.
    #   - ``prices`` from ``orch_config``, NOT ``base_config``:
    #     build_eval_orch_config auto-merges claude_endpoint_price_table() into
    #     .prices for a PROXIED candidate (task 2820), and that merge is exactly
    #     what makes 'price_table' reachable here instead of the degraded
    #     'unpriced_proxy'.
    #
    # SKIPPED entirely when NOTHING was spent — the timeout, harness-error and
    # pre-invoke cap paths never bind ``result``, so they arrive here with 0
    # tokens and $0.00. There is no provenance to resolve for a figure that is
    # not there, and resolving it anyway would fire the LOUD unpriced-proxy
    # WARNING for spend that never happened on EVERY timed-out or crashed
    # proxied cell — training operators to ignore the warning that matters, and
    # labelling a $0.00 cell 'unpriced_proxy' as if the degradation were real
    # (reviewer: log-noise). The label then stays the documented 'cli' default,
    # which is exactly what a $0.00 architect cell has always read.
    if arch_input_tokens or arch_output_tokens or arch_cost_usd:
        resolved_arch_cost, arch_cost_source = resolve_cost_usd(
            arch_input_tokens,
            arch_output_tokens,
            model=config.model,
            prices=orch_config.prices if orch_config is not None else None,
            cli_cost_usd=arch_cost_usd,
            is_local_model=is_local,
        )
    else:
        resolved_arch_cost, arch_cost_source = 0.0, 'cli'

    # Inference speed for the architect leg, the same formula collect_metrics
    # uses one path over (output tokens ÷ that leg's own duration, which is
    # what workflow_duration_ms below carries). Guarded: a timed-out or crashed
    # cell has arch_duration_ms == 0.
    arch_duration_secs = arch_duration_ms / 1000 if arch_duration_ms else 0.0
    arch_tps = (
        round(arch_output_tokens / arch_duration_secs, 2)
        if arch_duration_secs > 0 else 0.0
    )
    metrics = EvalMetrics(
        plan_quality=plan_quality,
        role_under_test='architect',
        # NO test signal exists for a plan-only cell (task 3099): this path
        # freezes implementer/debugger/reviewer/verify, so verification never
        # runs. ``None`` is the documented "unknown" sentinel; the dataclass
        # DEFAULT of ``False`` would read as "the tests failed" and hard-gate
        # ``blend_composite`` to 0.0, collapsing every architect row's composite
        # to 0.0000 and leaving ``select_survivors``' alphabetical tie-break as
        # the whole selection mechanism.
        #
        # ``True`` is NOT the fix either, on two counts:
        #   - ``build_composite_report`` draws each fixture's cost/latency FLOOR
        #     from PASSING trials, and ``ofat_candidates()`` mixes architect,
        #     implementer and judge candidates over the SAME fixtures into one
        #     result set. A ~$0.30/60s plan-only cell marked passing would
        #     become the floor for ~$5/900s full-workflow cells.
        #   - it would fabricate a 100% ``tests_pass_rate`` for a cell that
        #     never ran a test.
        tests_pass=None,
        # ``or []``, not a .get default: a plan can carry an explicit
        # ``steps: None`` (the normalizer's other empty shape), and len(None)
        # would crash the cell OUTSIDE the try above — turning a marked,
        # recoverable cap cell into a lost run.
        plan_steps=len(plan.get('steps') or []),
        # cost_usd is the cell's TOTAL spend, architect + plan judge — the
        # SUBSET invariant metrics.py:69-71 declares (judge_cost_usd is a
        # subset of cost_usd, not disjoint). judge_cost_usd below is the
        # breakdown, never an addend a report generator should sum again. A
        # judge that RAISES (the except above) leaves judge_cost_usd AND
        # judge_invocations at their 0.0/0 defaults, so that spend is
        # unknowable from here and is a documented under-report, never a
        # fabricated number — but it is NOT silently indistinguishable from
        # a judge that was SKIPPED: judge_error is set in the except block,
        # so invocation_error below reads "judge:raised: ...", telling a
        # reader "spend unknown", not "judge-free" (amendment, reviewer
        # robustness).
        # The architect component is the RESOLVED figure (Invariant P5), never
        # the raw CLI one; the judge component keeps its CLI figure because the
        # plan judge is always a native-cloud opus call. judge_cost_usd is still
        # the judge's SHARE of this total, not a separately-resolved addend.
        cost_usd=resolved_arch_cost + judge_cost_usd,
        # ONE label for a TWO-component sum: the second component is the plan
        # judge, always-native-cloud opus and therefore always CLI-sourced, so
        # this reads 'mixed' exactly when the judge actually spent AND the two
        # components' sources disagree — never letting one label quietly stand
        # in for two.
        cost_source=compose_cost_source(
            arch_cost_source, secondary_cost_usd=judge_cost_usd,
        ),
        # The architect invocation's token usage + its proxy signal — the three
        # inputs Invariant P5 resolves cost provenance from (resolve_cost_usd).
        # Stamped on the cell so the persisted JSON carries the evidence behind
        # its own cost figure, not just the figure.
        input_tokens=arch_input_tokens,
        output_tokens=arch_output_tokens,
        is_local_model=is_local,
        # The REST of that run's profile — priced by nothing, but stamped for
        # the same reason: an architect cell reporting input/output beside a
        # zeroed cache block reads as a much smaller run than it was (native
        # Claude runs are cache-read dominated). Keeps the architect cell's
        # token/turn block symmetric with collect_metrics' implementer one.
        cache_read_tokens=arch_cache_read_tokens,
        cache_create_tokens=arch_cache_create_tokens,
        turns_used=arch_turns,
        tokens_per_second=arch_tps,
        workflow_duration_ms=arch_duration_ms,
        invocation_error='; '.join(stage_markers) or None,
        cap_tainted=tainted,
        judge_cost_usd=judge_cost_usd,
        judge_invocations=judge_invocations,
    )
    result_obj = EvalResult(
        task_id=task_id,
        config_name=config.name,
        outcome=outcome,
        metrics=metrics.to_dict(),
        worktree_path=str(worktree),
        wall_clock_ms=wall_clock_ms,
        run_id=run_id,
        trial=trial,
    )
    save_result(result_obj)
    logger.info(
        f'Architect eval complete: {task_id} × {config.name} → '
        f'plan_quality={plan_quality} ({wall_clock_ms / 1000:.1f}s, '
        f'${metrics.cost_usd:.2f} incl. ${judge_cost_usd:.2f} judge)'
        + (
            f' [{"cap_tainted" if tainted else "invocation_error"}: '
            f'{metrics.invocation_error}]'
            if metrics.invocation_error else ''
        )
    )
    return result_obj


async def run_end_to_end(
    task_path: Path,
    arch_config: EvalConfig,
    impl_config: EvalConfig,
    base_config: OrchestratorConfig | None = None,
    trial: int = 1,
    timeout_override: int | None = None,
    memory_endpoint: str | None = None,
) -> EvalResult:
    """Run ONE both-live end-to-end eval: architect LIVE feeding implementer LIVE.

    Unlike :func:`run_eval` (frozen plan → implementer varies) and
    :func:`run_architect_eval` (live architect → downstream frozen), this is the
    ONLY executor where BOTH the architect and the implementer run live — the
    matrix/confirm stages (eval-revival μ). It builds the both-live orch config
    via ``build_eval_orch_config(impl_config, ..., architect_config=arch_config)``
    and constructs the workflow with ``initial_plan=None`` so the workflow runs
    its PLAN phase live (the architect plans against the fixture) and feeds that
    plan to the live implementer — the only place the plan-style/implementer
    coupling question exists (PRD decision 9).

    The result's ``config_name`` encodes the ``(architect, implementer)`` combo
    and its metrics carry ``role_under_test='end_to_end'``. Mirrors run_eval's
    timeout / exception handling and persists via :func:`save_result`.
    """
    task = load_task(task_path)
    task_id = task['id']
    project_root = Path(task['project_root'])
    config_name = f'{arch_config.name}+{impl_config.name}'

    logger.info(
        f'Starting end-to-end eval: {task_id} × {config_name} (trial {trial})'
    )
    start_ms = int(time.monotonic() * 1000)

    # 1. Fresh worktree at the fixture's pre_task_commit.
    worktree, run_id = await create_eval_worktree(
        project_root, task_id, task['pre_task_commit'],
        setup_commands=task.get('setup_commands'),
    )

    # 2. Both-live orch config: architect derived from arch_config, implementer
    #    from impl_config.
    orch_config = build_eval_orch_config(
        impl_config, task, base_config,
        memory_endpoint=memory_endpoint, architect_config=arch_config,
    )

    # 3. Task assignment.
    task_def = task.get('task_definition', {
        'title': task.get('name', task_id),
        'description': task.get('name', ''),
    })
    modules = task.get('modules', [])
    assignment = TaskAssignment(
        task_id=task_id, task=task_def, modules=list(modules),
    )

    # 4. Workflow dependencies (mirrors run_eval).
    git_ops = GitOps(orch_config.git, orch_config.project_root)
    scheduler, _ = _build_eval_scheduler(orch_config, task_id, list(modules))
    briefing = BriefingAssembler(orch_config)
    mcp = _EvalMcpStub(orch_config.fused_memory.url)

    usage_gate: UsageGate | None = None
    if orch_config.usage_cap.enabled:
        try:
            usage_gate = UsageGate(orch_config.usage_cap)
        except Exception as exc:
            logger.warning(f'Failed to create UsageGate for eval: {exc} — running without failover')

    # 5. Build the workflow with initial_plan=None → the architect plans LIVE and
    #    feeds the live implementer (the both-live path; run_eval hands a frozen
    #    plan here instead).
    workflow = build_workflow(
        assignment=assignment,
        config=orch_config,
        git_ops=git_ops,
        scheduler=scheduler,  # type: ignore[arg-type]
        briefing=briefing,
        mcp=mcp,  # type: ignore[arg-type]
        initial_plan=None,
        usage_gate=usage_gate,
    )
    workflow.worktree = worktree

    timeout_minutes = timeout_override or task.get('timeout_minutes', 60)
    try:
        terminal_report = await asyncio.wait_for(
            workflow.run(), timeout=timeout_minutes * 60,
        )
        outcome = terminal_report.outcome
    except TimeoutError:
        logger.error(
            f'End-to-end eval {task_id} × {config_name} timed out after '
            f'{timeout_minutes}m'
        )
        outcome = 'timeout'
    except Exception as e:
        logger.error(f'End-to-end eval {task_id} × {config_name} failed: {e}')
        outcome = WorkflowOutcome.BLOCKED

    wall_clock_ms = int(time.monotonic() * 1000) - start_ms

    # 6. Collect metrics, then STAMP role_under_test='end_to_end' (collect_metrics
    #    itself does not know which methodology stage invoked it).
    try:
        metrics = await collect_metrics(workflow, worktree, task)
        metrics_dict = metrics.to_dict()
    except Exception as e:
        logger.warning(f'Metric collection failed: {e}')
        metrics_dict = {}
    metrics_dict['role_under_test'] = 'end_to_end'

    result = EvalResult(
        task_id=task_id,
        config_name=config_name,
        outcome=outcome.value if isinstance(outcome, WorkflowOutcome) else str(outcome),
        metrics=metrics_dict,
        worktree_path=str(worktree),
        wall_clock_ms=wall_clock_ms,
        run_id=run_id,
        trial=trial,
    )
    save_result(result)
    logger.info(
        f'End-to-end eval complete: {task_id} × {config_name} → {result.outcome} '
        f'({wall_clock_ms / 1000:.1f}s)'
    )
    return result


def _collect_cancel_errors(done: Iterable[asyncio.Task[Any]]) -> list[asyncio.CancelledError]:
    """Return all CancelledErrors from a completed asyncio.wait done-set.

    Iterates every task in *done* and collects cancellation errors so that
    callers can log every failure before raising, rather than discarding all
    but the first (Task 586 fix).

    The primary branch — ``task.cancelled() is True`` — covers all known
    CPython 3.11+ cases, including coroutines that raise CancelledError
    internally without an explicit ``task.cancel()`` call, because the
    runtime transitions the task to the cancelled state in both scenarios.

    Belt-and-suspenders: in current CPython a coroutine raising
    CancelledError causes task.cancelled() to return True, so the
    secondary branch (task.exception() returning CancelledError while
    task.cancelled() is False) is unreachable in practice. Kept in case
    a future runtime routes coroutine-raised CancelledError via
    task.exception() instead of task.cancelled().
    """
    errors: list[asyncio.CancelledError] = []
    for task in done:
        if task.cancelled():
            errors.append(asyncio.CancelledError())
        else:
            exc = task.exception()
            if isinstance(exc, asyncio.CancelledError):
                errors.append(exc)
    return errors


async def run_eval_matrix(
    task_paths: list[Path],
    configs: list[EvalConfig] | None = None,
    base_config: OrchestratorConfig | None = None,
    max_parallel: int | None = None,
    trials: int = 1,
    force: bool = False,
    timeout_override: int | None = None,
) -> list[EvalResult]:
    """Run all (task, config, trial) combinations with bounded concurrency.

    Raises:
        asyncio.CancelledError: if any individual eval coroutine raises
            CancelledError, or if this coroutine itself is cancelled from
            outside.  In either case we log, cancel any still-running
            sibling tasks, await their cleanup, and re-raise.
    """
    configs = configs or EVAL_CONFIGS

    combos = [
        (task_path, config, t)
        for task_path in task_paths
        for config in configs
        for t in range(1, trials + 1)
    ]

    if max_parallel is None:
        max_parallel = len(combos)
    sem = asyncio.Semaphore(max_parallel)

    async def _run_one(
        task_path: Path, config: EvalConfig, trial: int,
    ) -> EvalResult | None:
        task = load_task(task_path)
        if not force and _result_exists(task['id'], config.name):
            logger.info(f'Skipping existing: {task["id"]} × {config.name}')
            return None
        async with sem:
            return await run_eval(
                task_path, config, base_config,
                trial=trial, timeout_override=timeout_override,
            )

    # Design decision: use asyncio.wait(FIRST_COMPLETED) monitor loop instead of
    # asyncio.gather(return_exceptions=True).
    #
    # asyncio.gather(return_exceptions=True) blocks until ALL tasks complete before
    # the post-gather loop can detect CancelledError and re-raise it.  For a large
    # matrix where one eval is cancelled early, N-1 siblings continue running their
    # full duration — wasting CPU proportional to matrix size × timeout_minutes.
    #
    # asyncio.wait(FIRST_COMPLETED) lets us react to each task completion
    # individually: on CancelledError we immediately cancel all remaining tasks and
    # re-raise, typically within milliseconds.  Non-cancel exceptions are still
    # logged and the loop continues — identical happy-path/error-path semantics to
    # the previous gather loop, with strictly better cancellation behaviour.
    #
    # This is the same pattern used in harness.py (lines 305, 317) for managing
    # concurrent workflow tasks.  Cleanup follows the established pattern from
    # steward.py (lines 101-104): cancel tasks explicitly then await them with
    # return_exceptions=True to ensure clean teardown before re-raising.
    active: set[asyncio.Task] = {
        asyncio.create_task(_run_one(tp, cfg, t))
        for tp, cfg, t in combos
    }
    results: list[EvalResult] = []
    # Distinguish two cancellation scenarios:
    #   Inner-task cancellation — an individual _run_one coroutine was cancelled
    #     or raised CancelledError.  asyncio.wait surfaces this via
    #     task.cancelled() or task.exception() inside the monitor loop below;
    #     we log it, cancel siblings, and re-raise to propagate.
    #   Outer-task cancellation — run_eval_matrix itself was cancelled (e.g.
    #     SIGINT / asyncio.wait_for timeout).  The CancelledError interrupts
    #     the *await asyncio.wait(...)* call directly and is caught by the
    #     outer except clause, which performs the same sibling cleanup.
    try:
        while active:
            done, active = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
            # Task 586: scan the full done batch for ALL CancelledErrors before
            # processing any results.  Multiple tasks can complete in the same
            # event-loop iteration and land in the same done set (e.g. when a
            # shutdown signal fires while two evals are parked at the same
            # await point).  The old code raised on the first cancel it saw,
            # silently discarding subsequent cancels in the batch.
            cancel_errors = _collect_cancel_errors(done)
            if cancel_errors:
                for ce in cancel_errors:
                    logger.error('Eval cancelled', exc_info=ce)
                for t in active:
                    t.cancel()
                await asyncio.gather(*active, return_exceptions=True)
                active.clear()
                raise cancel_errors[0]
            # No cancellations in this batch — handle results and non-cancel
            # exceptions.  task.cancelled() is False for all remaining tasks so
            # task.exception() / task.result() are safe to call.
            for task in done:
                exc = task.exception()
                if exc is not None:
                    logger.error('Eval failed', exc_info=exc)
                else:
                    r = task.result()
                    if r is not None:
                        results.append(r)
    except asyncio.CancelledError:
        # External cancellation (e.g. SIGINT / asyncio.wait_for timeout).
        # Cancel all remaining sibling tasks and await their cleanup before
        # re-raising so we don't leave orphaned tasks behind.
        for t in active:
            t.cancel()
        await asyncio.gather(*active, return_exceptions=True)
        raise
    return results


async def _bounded_fanout(
    thunks: list[Callable[[], Awaitable[EvalResult | None]]],
    max_parallel: int | None = None,
) -> list[EvalResult]:
    """Run each *thunk* with bounded concurrency, returning the flattened results.

    The single-sourced fan-out skeleton the μ methodology stages
    (:func:`run_ofat_stage` / :func:`run_matrix_stage` / :func:`run_confirm_stage`)
    share, mirroring :func:`run_eval_matrix`'s ``asyncio.wait(FIRST_COMPLETED)``
    monitor loop EXACTLY: a non-cancel failure in one cell is logged and the
    fan-out CONTINUES; a ``CancelledError`` (inner or external) cancels the
    still-running siblings, awaits their teardown, and re-raises. Each thunk is a
    zero-arg coroutine factory returning one ``EvalResult`` (or ``None`` to skip).
    """
    if not thunks:
        return []
    if max_parallel is None:
        max_parallel = len(thunks)
    sem = asyncio.Semaphore(max_parallel)

    async def _guarded(thunk: Callable[[], Awaitable[EvalResult | None]]) -> EvalResult | None:
        async with sem:
            return await thunk()

    active: set[asyncio.Task] = {asyncio.create_task(_guarded(t)) for t in thunks}
    results: list[EvalResult] = []
    try:
        while active:
            done, active = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
            cancel_errors = _collect_cancel_errors(done)
            if cancel_errors:
                for ce in cancel_errors:
                    logger.error('Eval cancelled', exc_info=ce)
                for t in active:
                    t.cancel()
                await asyncio.gather(*active, return_exceptions=True)
                active.clear()
                raise cancel_errors[0]
            for task in done:
                exc = task.exception()
                if exc is not None:
                    logger.error('Eval failed', exc_info=exc)
                else:
                    r = task.result()
                    if r is not None:
                        results.append(r)
    except asyncio.CancelledError:
        for t in active:
            t.cancel()
        await asyncio.gather(*active, return_exceptions=True)
        raise
    return results


async def run_ofat_stage(
    task_paths: list[Path],
    candidates: list[EvalConfig],
    base_config: OrchestratorConfig | None = None,
    max_parallel: int | None = None,
    trials: int = 1,
    timeout_override: int | None = None,
) -> list[EvalResult]:
    """OFAT screen: dispatch each candidate by role over fixtures × trials (μ/ο).

    A role-dispatching fan-out over the EXISTING frozen-input executors
    (decision 9): an implementer candidate (``role=='implementer'``) runs through
    :func:`run_eval` (frozen plan → implementer varies, arch/reviewer pinned), an
    architect candidate (``role=='architect'``) through :func:`run_architect_eval`
    (live architect → downstream frozen), and a judge candidate (``role=='judge'``,
    ο) through :func:`run_eval` with the implementer PINNED to
    ``JUDGE_OFAT_IMPLEMENTER_PIN`` and the candidate riding ``judge_config`` — so
    ONLY the ζ completion judge varies (implementer/architect/reviewer held
    constant). No new per-role machinery. Returns the flattened ``EvalResult``
    list across every ``(candidate, fixture, trial)`` cell; a failed cell is
    logged and skipped via :func:`_bounded_fanout`.
    """
    def _thunk(
        task_path: Path, candidate: EvalConfig, trial: int,
    ) -> Callable[[], Awaitable[EvalResult | None]]:
        async def _run() -> EvalResult | None:
            if candidate.role == 'architect':
                return await run_architect_eval(
                    task_path, candidate, base_config,
                    trial=trial, timeout_override=timeout_override,
                )
            if candidate.role == 'judge':
                # ο: vary ONLY the judge — pin the implementer to the fixed cloud
                # incumbent (config=JUDGE_OFAT_IMPLEMENTER_PIN) and ride the judge
                # candidate on judge_config, so run_eval derives the ζ completion
                # judge's model/effort while implementer/architect/reviewer stay
                # fixed (true OFAT). run_eval relabels + stamps role_under_test.
                return await run_eval(
                    task_path, JUDGE_OFAT_IMPLEMENTER_PIN, base_config,
                    trial=trial, timeout_override=timeout_override,
                    judge_config=candidate,
                )
            return await run_eval(
                task_path, candidate, base_config,
                trial=trial, timeout_override=timeout_override,
            )
        return _run

    thunks = [
        _thunk(tp, candidate, trial)
        for tp in task_paths
        for candidate in candidates
        for trial in range(1, trials + 1)
    ]
    return await _bounded_fanout(thunks, max_parallel)


async def run_matrix_stage(
    task_paths: list[Path],
    arch_survivors: list[EvalConfig],
    impl_survivors: list[EvalConfig],
    base_config: OrchestratorConfig | None = None,
    max_parallel: int | None = None,
    trials: int = 1,
    timeout_override: int | None = None,
) -> list[EvalResult]:
    """Matrix stage: both-live architect×implementer cross product over survivors (μ).

    Expands :func:`configs.matrix_pairs` — the FULL ``arch_survivors ×
    impl_survivors`` cross product, INCLUDING same-family diagonals (e.g.
    sonnet-arch × sonnet-impl), the pair that tests whether a plan style couples
    to its own family's implementer (PRD decision 9) — and fans
    :func:`run_end_to_end` out over ``pairs × fixtures × trials`` via the shared
    :func:`_bounded_fanout` skeleton (identical cancellation / error-continue
    semantics to :func:`run_ofat_stage`). Both roles run LIVE. Returns the
    flattened ``EvalResult`` list; a failed cell is logged and skipped.
    """
    def _thunk(
        task_path: Path, arch: EvalConfig, impl: EvalConfig, trial: int,
    ) -> Callable[[], Awaitable[EvalResult | None]]:
        async def _run() -> EvalResult | None:
            return await run_end_to_end(
                task_path, arch, impl, base_config,
                trial=trial, timeout_override=timeout_override,
            )
        return _run

    pairs = matrix_pairs(arch_survivors, impl_survivors)
    thunks = [
        _thunk(tp, arch, impl, trial)
        for tp in task_paths
        for arch, impl in pairs
        for trial in range(1, trials + 1)
    ]
    return await _bounded_fanout(thunks, max_parallel)


async def run_confirm_stage(
    task_paths: list[Path],
    arch_winner: EvalConfig,
    impl_winner: EvalConfig,
    base_config: OrchestratorConfig | None = None,
    max_parallel: int | None = None,
    trials: int = 3,
    timeout_override: int | None = None,
) -> list[EvalResult]:
    """Confirmation batch: the SINGLE winning combo × N trials, both-live (μ).

    The final methodology stage — one end-to-end confirmation of the winning
    ``(arch_winner, impl_winner)`` combo (PRD decision 10). Fans
    :func:`run_end_to_end` out over ``fixtures × trials`` for the single combo
    via the shared :func:`_bounded_fanout` skeleton (identical cancellation /
    error-continue semantics to the screen stages). ``trials`` defaults to 3 —
    decision 10's statistics floor, enough repeats for a CI95 on the winner,
    NOT the 1-trial screen default of :func:`run_ofat_stage` /
    :func:`run_matrix_stage`. Both roles run LIVE. Returns the flattened
    ``EvalResult`` list for the confirmation batch.
    """
    def _thunk(
        task_path: Path, trial: int,
    ) -> Callable[[], Awaitable[EvalResult | None]]:
        async def _run() -> EvalResult | None:
            return await run_end_to_end(
                task_path, arch_winner, impl_winner, base_config,
                trial=trial, timeout_override=timeout_override,
            )
        return _run

    thunks = [
        _thunk(tp, trial)
        for tp in task_paths
        for trial in range(1, trials + 1)
    ]
    return await _bounded_fanout(thunks, max_parallel)


def _resume_plan_from_worktree(worktree: Path, task: dict) -> dict | None:
    """Load the initial plan for a --worktree resume run.

    Reads the RELOCATED plan.json at ``_meta_root_for_worktree(worktree)/plan.json``.
    Production TaskWorkflow moved plan.json out of the legacy ``<worktree>/.task/``
    to the sibling ``.task-meta/<name>/`` root (W11 / task 2258); reading the
    stale legacy path silently missed and discarded completed-step progress on
    resume (D1 drift, task 2812). Falls back to the frozen ``task['plan']`` only
    when the relocated file is absent.
    """
    existing_plan_path = _meta_root_for_worktree(worktree) / 'plan.json'
    if existing_plan_path.exists():
        initial_plan = json.loads(existing_plan_path.read_text())
        steps = initial_plan.get('steps', [])
        done = sum(1 for s in steps if s.get('status') == 'done')
        logger.info(f'Using existing plan from worktree ({done}/{len(steps)} steps done)')
        return initial_plan
    logger.info('No existing plan in worktree — using task JSON plan')
    return task.get('plan')


def _result_exists(task_id: str, config_name: str) -> bool:
    """Check if any result already exists for this (task, config) pair."""
    if not RESULTS_DIR.exists():
        return False
    return any(RESULTS_DIR.glob(f'{task_id}__{config_name}__*.json'))


def save_result(result: EvalResult) -> Path:
    """Write eval result JSON to results directory."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f'{result.task_id}__{result.config_name}__{result.run_id}.json'
    path = RESULTS_DIR / filename
    with open(path, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)
    logger.info(f'Saved result: {path}')
    return path


def load_results() -> list[EvalResult]:
    """Load all existing eval results from the results directory."""
    results = []
    if not RESULTS_DIR.exists():
        return results
    for path in sorted(RESULTS_DIR.glob('*.json')):
        with open(path) as f:
            data = json.load(f)
        # Filter to known fields so old results with extra/missing keys load
        known = {f.name for f in EvalResult.__dataclass_fields__.values()}
        results.append(EvalResult(**{k: v for k, v in data.items() if k in known}))
    return results


# ---------------------------------------------------------------------------
# Stubs and helpers for eval mode (no real MCP HTTP connection needed)
# ---------------------------------------------------------------------------


def _build_eval_scheduler(
    orch_config: OrchestratorConfig,
    task_id: str,
    modules: list[str],
) -> tuple[Scheduler, _StubMcpSession]:
    """Build a production Scheduler wired with an in-memory MCP session stub.

    Pre-installs the module lock for ``task_id`` so that a later
    ``handle_blast_radius_expansion`` call cannot ``KeyError`` (production
    normally installs the lock in ``acquire_next``; eval mode bypasses that).

    Returns ``(scheduler, stub_session)`` so callers can inspect the stub when
    needed (e.g. tests asserting on ``_statuses``).
    """
    stub = _StubMcpSession()
    scheduler = Scheduler(orch_config, mcp_session=stub)
    # Pre-install the module lock so handle_blast_radius_expansion's
    # try_acquire_additional can find the _held[task_id] entry.
    scheduler.lock_table.try_acquire(task_id, modules)
    return scheduler, stub


class _StubMcpSession:
    """In-process MCP session stub for eval runs.

    Mirrors ``McpSession.call_tool``'s signature and JSON-RPC envelope shape
    so the production ``Scheduler`` can use it via duck-typing without any
    changes to its parsing code.

    Handles every tool the eval ``Scheduler`` dispatches via ``dispatch_tool``:
    ``set_task_status``, ``get_task``, ``get_tasks``, ``get_statuses``,
    ``update_task``, ``set_task_claimant``, ``get_external_statuses`` (plus
    ``add_dependency``, stubbed defensively though not yet dispatched).  The
    ``test_scheduler_dispatch_literal_tripwire`` guard asserts every dispatched
    literal has a branch here.  Any other tool name raises ``NotImplementedError``.
    """

    def __init__(self) -> None:
        self._statuses: dict[str, str] = {}
        self._request_id: int = 0

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _envelope(self, text: str) -> dict:
        return {
            'jsonrpc': '2.0',
            'id': self._next_id(),
            'result': {
                'content': [
                    {'type': 'text', 'text': text},
                ],
            },
        }

    async def call_tool(
        self,
        name: str,
        arguments: dict,
        timeout: float = 30,
    ) -> dict:
        """Dispatch an in-memory MCP tool call and return a JSON-RPC envelope.

        Supported tools: ``set_task_status``, ``get_task``, ``get_tasks``,
        ``get_statuses``, ``update_task``, ``set_task_claimant``,
        ``get_external_statuses``, ``add_dependency``.  Unknown tool names raise
        ``NotImplementedError``.

        .. note::
            Terminal-state enforcement is intentionally **not** simulated.
            The production fused-memory ``TaskInterceptor`` rejects transitions
            from terminal states (e.g. ``done`` → ``pending``) unless a
            ``reopen_reason`` is supplied.  This stub silently accepts any
            transition so eval flows are not blocked by status-guard logic.
            If a test needs to verify terminal-state semantics it should target
            the real fused-memory server rather than this stub.
        """
        if name == 'set_task_status':
            task_id = arguments['id']
            status = arguments['status']
            self._statuses[task_id] = status
            return self._envelope(json.dumps({'id': task_id, 'status': status}))
        if name == 'get_task':
            task_id = arguments['id']
            status = self._statuses.get(task_id)
            payload = {'id': task_id, 'status': status} if status is not None else {'id': task_id}
            return self._envelope(json.dumps(payload))
        if name == 'get_tasks':
            return self._envelope(json.dumps({'tasks': []}))
        if name == 'get_statuses':
            ids_filter = arguments.get('ids')
            if ids_filter is not None:
                mapping = {k: v for k, v in self._statuses.items() if k in ids_filter}
            else:
                mapping = dict(self._statuses)
            return self._envelope(json.dumps({'statuses': mapping}))
        if name == 'update_task':
            task_id = arguments['id']
            return self._envelope(json.dumps({'id': task_id}))
        if name == 'set_task_claimant':
            # Status-untouching heartbeat/clear path (Scheduler.set_task_claimant
            # dispatches this every ~60s in eval mode). Return a clean payload
            # with no 'error'/'success:False' key so extract_rejection → None
            # and the heartbeat logs zero warnings (B9). Status is intentionally
            # NOT recorded here, mirroring the real tool which never touches it.
            task_id = arguments['id']
            return self._envelope(json.dumps({'id': task_id}))
        if name == 'get_external_statuses':
            # Cross-project dep resolver (Scheduler.get_external_statuses parses
            # the response via parse_tool_result(result, None, dict) — the
            # whole-inner-dict contract). Return a FLAT {dep: status} dict (no
            # 'statuses' wrapper) keying EVERY requested dep so no missing-key
            # ExternalResolverError; empty deps → {}. Eval runs are single-task
            # with no cross-project deps, so this is inert in practice but
            # contract-complete (non-blocking 'done') if ever exercised.
            deps = arguments.get('deps', [])
            mapping = {d: 'done' for d in deps}
            return self._envelope(json.dumps(mapping))
        if name == 'add_dependency':
            # Not currently dispatched by scheduler.py; added defensively per
            # task spec so a future dispatch site is already stubbed. Echo the id
            # in a clean envelope.
            task_id = arguments['id']
            return self._envelope(json.dumps({'id': task_id}))
        raise NotImplementedError(
            f'_StubMcpSession: unknown tool {name!r} — add a branch if this tool is needed'
        )


class _EvalMcpStub:
    """Minimal MCP lifecycle stub for eval runs."""

    def __init__(self, url: str):
        self.url = url

    def mcp_config_json(self, escalation_url: str | None = None) -> dict:
        return {}


class _RecordingMemoryHandler(BaseHTTPRequestHandler):
    """Request handler for :class:`RecordingMemorySink`.

    Records every fused-memory ``tools/call`` write on the owning server's
    shared ``writes`` list and replies with a benign JSON-RPC success envelope
    (mirroring ``_StubMcpSession._envelope``). Any POST path is accepted — the
    workflow write path targets ``{url}/mcp/`` but ``BaseHTTPRequestHandler``
    routes every POST here regardless of path.
    """

    def do_POST(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler API name
        server: Any = self.server  # _RecordingMemoryServer (carries writes/_envelope)
        length = int(self.headers.get('Content-Length', 0) or 0)
        raw = self.rfile.read(length) if length > 0 else b''
        try:
            body = json.loads(raw) if raw else {}
        except (ValueError, TypeError):
            body = {}

        if isinstance(body, dict) and body.get('method') == 'tools/call':
            params = body.get('params') or {}
            server.writes.append((params.get('name'), params.get('arguments', {})))

        payload = json.dumps(server._envelope('ok')).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 — stdlib API name
        """Silence the default stderr access log (no per-request spam)."""


class _RecordingMemoryServer(ThreadingHTTPServer):
    """ThreadingHTTPServer carrying the shared ``writes`` list + JSON-RPC id counter."""

    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
    ) -> None:
        super().__init__(server_address, handler_class)
        self.writes: list[tuple[str | None, dict]] = []
        self._request_id = 0

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _envelope(self, text: str) -> dict:
        return {
            'jsonrpc': '2.0',
            'id': self._next_id(),
            'result': {
                'content': [
                    {'type': 'text', 'text': text},
                ],
            },
        }


class RecordingMemorySink:
    """In-process recording HTTP endpoint for eval memory writes (D8).

    A real loopback HTTP endpoint (stdlib ``http.server``) that receives the
    raw httpx POSTs the workflow's ``_write_*_to_memory`` methods send to
    ``{self.mcp.url}/mcp/``, records each fused-memory ``tools/call`` write as
    ``(tool_name, arguments)`` on :attr:`writes`, and replies with a benign
    JSON-RPC success envelope (mirroring ``_StubMcpSession._envelope`` so any
    caller that parses the reply stays happy).

    The orchestrator declares only ``httpx`` (a client) — no
    aiohttp/starlette — and the workflow write path is a raw httpx POST to a
    URL, so a real HTTP endpoint (not an MCP-session stub) is required to
    receive it. Stdlib ``http.server`` adds no dependency.

    Isolation: pass :attr:`url` as
    ``build_eval_orch_config(..., memory_endpoint=sink.url)`` (or
    ``run_eval(..., memory_endpoint=sink.url)``) so
    ``orch_config.fused_memory.url`` (hence ``self.mcp.url``) routes every eval
    memory write here instead of the real production dark_factory store. The
    caller owns the lifecycle via the context-manager protocol::

        with RecordingMemorySink() as sink:
            orch_config = build_eval_orch_config(cfg, task, base, memory_endpoint=sink.url)
            ...
            assert sink.writes  # what would have been written to production
    """

    def __init__(self) -> None:
        # Bind to an ephemeral 127.0.0.1 port immediately (HTTPServer binds in
        # __init__), so .url is valid before the serving thread starts.
        self._server = _RecordingMemoryServer(('127.0.0.1', 0), _RecordingMemoryHandler)
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        """The bound loopback base URL, e.g. ``http://127.0.0.1:54321``."""
        port = self._server.server_address[1]
        return f'http://127.0.0.1:{port}'

    @property
    def writes(self) -> list[tuple[str | None, dict]]:
        """Recorded ``(tool_name, arguments)`` for every POSTed ``tools/call``."""
        return self._server.writes

    def __enter__(self) -> RecordingMemorySink:
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name='recording-memory-sink',
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None


if __name__ == '__main__':
    from orchestrator.cli import eval_cmd
    eval_cmd()
