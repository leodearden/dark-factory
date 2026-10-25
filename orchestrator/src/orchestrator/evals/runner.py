"""Main eval orchestrator: run (task, config) pairs through the real workflow."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections.abc import Awaitable, Callable, Iterable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from shared.eval_lane import is_eval_fixture_task_id

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
    _meta_root_for_worktree,
    build_workflow,
)

from .architect_eval import (
    ArchitectEvalDeps,
    PlanScoring,
    assemble_metrics,
    cell_label,
    classify_attempt,
    invoke_architect,
    judge_plan,
    log_architect_cell,
    read_reference_diff,
)
from .configs import (
    EVAL_CONFIGS,
    JUDGE_OFAT_IMPLEMENTER_PIN,
    EvalConfig,
    claude_endpoint_price_table,
    matrix_pairs,
)
from .eval_gate import (
    GATE_UNSET,
    InjectedGate,
    campaign_usage_gate,
    owned_usage_gate,
)
from .metrics import (
    collect_metrics,
    is_proxied_endpoint,
)
from .profile import apply_eval_profile
from .replay_frame import REPLAY_FRAME_ID, ReplayFramedBriefingAssembler
from .results import CellName, EvalResult, ResultStore
from .snapshots import create_eval_worktree, read_python_pin

logger = logging.getLogger(__name__)


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

    Raises ``ValueError`` when the fixture's ``id`` is not an eval fixture id.
    """
    with open(task_path) as f:
        task = json.load(f)

    task_id = task.get('id')
    if not is_eval_fixture_task_id(task_id):
        raise ValueError(
            f'eval fixture {task_path}: id {task_id!r} is not an eval fixture id '
            '(<repo>_task_<n>[_<suffix>] or shadow_<task_id>_<cell_id>). Eval-lane '
            'escalation containment (shared/src/shared/eval_lane.py) recognises '
            'eval provenance by this grammar; an off-grammar id would let this '
            "fixture's escalations reach the production human L2 queue."
        )

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
    *,
    usage_gate: InjectedGate = GATE_UNSET,
    store: ResultStore | None = None,
) -> EvalResult:
    """Run one (task, config) pair through PLAN→EXECUTE→VERIFY→REVIEW.

    The cell persists into *store*; ``None`` persists into
    :meth:`ResultStore.packaged`.

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
    address = CellName(task_id, result_label, run_id)

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

    # 5b. Usage gate for account failover (judge hits Claude API, may cap).
    async with owned_usage_gate(
        usage_gate, orch_config, label=f'{task_id} × {config.name}',
    ) as gate:
        # 6. Run the real workflow
        workflow = build_workflow(
            assignment=assignment,
            config=orch_config,
            git_ops=git_ops,
            scheduler=scheduler,  # type: ignore[arg-type]
            briefing=briefing,
            mcp=mcp,  # type: ignore[arg-type]
            initial_plan=initial_plan,
            usage_gate=gate,
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
            task_id=address.task_id,
            config_name=address.config_name,
            outcome=outcome.value if isinstance(outcome, WorkflowOutcome) else str(outcome),
            metrics=metrics_dict,
            worktree_path=str(worktree),
            wall_clock_ms=wall_clock_ms,
            run_id=address.run_id,
            trial=trial,
        )

        # 8. Persist result
        (store or ResultStore.packaged()).save(result)

        logger.info(
            f'Eval complete: {task_id} × {config.name} → {result.outcome} '
            f'(total={wall_clock_ms / 1000:.1f}s, '
            f'workflow={metrics_dict.get("workflow_duration_ms", 0) / 1000:.1f}s)'
        )
        return result


async def run_architect_eval(
    task_path: Path,
    config: EvalConfig,
    base_config: OrchestratorConfig | None = None,
    trial: int = 1,
    timeout_override: int | None = None,
    memory_endpoint: str | None = None,
    *,
    usage_gate: InjectedGate = GATE_UNSET,
    store: ResultStore | None = None,
    deps: ArchitectEvalDeps | None = None,
) -> EvalResult:
    """Run ONE architect eval: invoke the architect live, score its plan, persist the cell.

    Unlike :func:`run_eval`, which freezes the plan and scores the implementer,
    only the architect runs, with this candidate's model, backend, effort and
    env_overrides; every downstream role is frozen. The cell is
    ``invoke_architect`` → ``read_reference_diff`` → ``classify_attempt`` →
    ``judge_plan`` (only when judged) → ``assemble_metrics``, over *deps*
    (production defaults, persisting into *store*, or
    :meth:`ResultStore.packaged` when neither is given). Rationale:
    docs/plan-scoring-and-judge.md, "The architect cell: invoke, classify,
    assemble (task 6737)".
    """
    if store is not None and deps is not None:
        raise ValueError(
            'run_architect_eval takes store= or deps=, not both: deps.store is '
            'where an injected deps persists the cell'
        )
    task = load_task(task_path)
    deps = deps or ArchitectEvalDeps(store=store or ResultStore.packaged())
    label = cell_label(task, config)
    logger.info(f'Starting architect eval: {label} (trial {trial})')
    start_ms = int(time.monotonic() * 1000)
    attempt = await invoke_architect(
        task, config, deps,
        build_orch_config=lambda: build_eval_orch_config(
            config, task, base_config, memory_endpoint=memory_endpoint,
        ),
        usage_gate=usage_gate,
        timeout_minutes=timeout_override or task.get('timeout_minutes', 60),
    )
    reference_diff = await read_reference_diff(task, deps)
    scoring = classify_attempt(attempt)
    judgement = (
        await judge_plan(attempt.plan, reference_diff, task, deps)
        if scoring is PlanScoring.JUDGE else None
    )
    metrics = assemble_metrics(attempt, scoring, judgement)
    wall_clock_ms = int(time.monotonic() * 1000) - start_ms
    result = EvalResult(
        task_id=attempt.address.task_id,
        config_name=attempt.address.config_name,
        outcome=attempt.outcome,
        metrics=metrics.to_dict(),
        worktree_path=str(attempt.worktree),
        wall_clock_ms=wall_clock_ms,
        run_id=attempt.address.run_id,
        trial=trial,
    )
    deps.store.save(result)
    log_architect_cell(label, scoring, metrics, wall_clock_ms)
    return result


async def run_end_to_end(
    task_path: Path,
    arch_config: EvalConfig,
    impl_config: EvalConfig,
    base_config: OrchestratorConfig | None = None,
    trial: int = 1,
    timeout_override: int | None = None,
    memory_endpoint: str | None = None,
    *,
    usage_gate: InjectedGate = GATE_UNSET,
    store: ResultStore | None = None,
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
    timeout / exception handling and persists into *store* (``None`` persists
    into :meth:`ResultStore.packaged`). Its
    architect is briefed in the replay frame, and the cell is stamped
    ``replay_frame`` (docs/eval-replay-frame.md).
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
    address = CellName(task_id, config_name, run_id)

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
    # TaskWorkflow's PLAN phase briefs the live architect through this; see
    # docs/eval-replay-frame.md.
    briefing = ReplayFramedBriefingAssembler(
        orch_config, base_commit=task['pre_task_commit'],
    )
    mcp = _EvalMcpStub(orch_config.fused_memory.url)

    async with owned_usage_gate(
        usage_gate, orch_config, label=f'{task_id} × {config_name}',
    ) as gate:
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
            usage_gate=gate,
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
        metrics_dict['replay_frame'] = REPLAY_FRAME_ID

        result = EvalResult(
            task_id=address.task_id,
            config_name=address.config_name,
            outcome=outcome.value if isinstance(outcome, WorkflowOutcome) else str(outcome),
            metrics=metrics_dict,
            worktree_path=str(worktree),
            wall_clock_ms=wall_clock_ms,
            run_id=address.run_id,
            trial=trial,
        )
        (store or ResultStore.packaged()).save(result)
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
    *,
    store: ResultStore | None = None,
) -> list[EvalResult]:
    """Run all (task, config, trial) combinations with bounded concurrency.

    A (task, config) pair *store* already holds is skipped unless *force*;
    every cell run persists into *store* (``None`` means
    :meth:`ResultStore.packaged`).

    Raises:
        asyncio.CancelledError: if any individual eval coroutine raises
            CancelledError, or if this coroutine itself is cancelled from
            outside.  In either case we log, cancel any still-running
            sibling tasks, await their cleanup, and re-raise.
    """
    configs = configs or EVAL_CONFIGS
    store = store or ResultStore.packaged()

    async with campaign_usage_gate(base_config) as gate:
        # ONE gate for the whole matrix (task 4427) — see campaign_usage_gate
        # for why sharing it across concurrent cells is safe, what else one gate
        # shares, and why a None ``gate`` must still be threaded AS SUCH.
        #
        # Specific to this fan-out: the `async with` spans the monitor loop AND
        # its sibling-cancellation drain, so the teardown also fires on the
        # CancelledError re-raise, where a leaked probe loop would otherwise
        # survive the aborted campaign.

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
            if not force and store.exists(task['id'], config.name):
                logger.info(f'Skipping existing: {task["id"]} × {config.name}')
                return None
            async with sem:
                return await run_eval(
                    task_path, config, base_config,
                    trial=trial, timeout_override=timeout_override,
                    usage_gate=gate, store=store,
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
    async with campaign_usage_gate(base_config) as gate:
        # ONE gate for the whole stage (task 4427) — see campaign_usage_gate for
        # why sharing it across concurrent cells is safe, what else one gate
        # shares, and why a None ``gate`` must still be threaded AS SUCH.
        def _thunk(
            task_path: Path, candidate: EvalConfig, trial: int,
        ) -> Callable[[], Awaitable[EvalResult | None]]:
            async def _run() -> EvalResult | None:
                if candidate.role == 'architect':
                    return await run_architect_eval(
                        task_path, candidate, base_config,
                        trial=trial, timeout_override=timeout_override,
                        usage_gate=gate,
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
                        judge_config=candidate, usage_gate=gate,
                    )
                return await run_eval(
                    task_path, candidate, base_config,
                    trial=trial, timeout_override=timeout_override,
                    usage_gate=gate,
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
    async with campaign_usage_gate(base_config) as gate:
        # ONE gate for the whole stage (task 4427) — see campaign_usage_gate for
        # why sharing it across concurrent cells is safe, what else one gate
        # shares, and why a None ``gate`` must still be threaded AS SUCH.
        def _thunk(
            task_path: Path, arch: EvalConfig, impl: EvalConfig, trial: int,
        ) -> Callable[[], Awaitable[EvalResult | None]]:
            async def _run() -> EvalResult | None:
                return await run_end_to_end(
                    task_path, arch, impl, base_config,
                    trial=trial, timeout_override=timeout_override,
                    usage_gate=gate,
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
    async with campaign_usage_gate(base_config) as gate:
        # ONE gate for the whole stage (task 4427) — see campaign_usage_gate for
        # why sharing it across concurrent cells is safe, what else one gate
        # shares, and why a None ``gate`` must still be threaded AS SUCH.
        def _thunk(
            task_path: Path, trial: int,
        ) -> Callable[[], Awaitable[EvalResult | None]]:
            async def _run() -> EvalResult | None:
                return await run_end_to_end(
                    task_path, arch_winner, impl_winner, base_config,
                    trial=trial, timeout_override=timeout_override,
                    usage_gate=gate,
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
    ``update_task``, ``set_task_claimant``, ``get_external_statuses``,
    ``submit_task`` (plus ``add_dependency``, stubbed defensively though not
    yet dispatched).  The ``test_scheduler_dispatch_literal_tripwire`` guard
    asserts every dispatched literal has a branch here.  Any other tool name
    raises ``NotImplementedError``.
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
        ``get_external_statuses``, ``submit_task``, ``add_dependency``.  Unknown
        tool names raise
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
        if name == 'submit_task':
            # Scheduler.submit_task files tasks for machine filers (auto-eval,
            # the flake ledger). An eval run must create none, so the answer is
            # the server's structured refusal, which parses to SubmitFailed.
            return self._envelope(json.dumps({
                'error': 'eval runs file no tasks',
                'error_type': 'EvalStubRefusal',
            }))
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
    workflow write path targets ``{url}/mcp`` but ``BaseHTTPRequestHandler``
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
    ``{self.mcp.url}/mcp``, records each fused-memory ``tools/call`` write as
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
