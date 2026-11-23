"""Drives ``runner.run_architect_eval`` hermetically for the architect-cell test files.

Also holds the canned task, prompt and agent-result inputs those files feed it.
"""

from __future__ import annotations

import contextlib
import dataclasses
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from _campaign_gate_helpers import NO_INJECTED_GATE
from shared.cli_invoke import AgentResult
from shared.config_models import UsageCapConfig

from orchestrator.agents.briefing import BriefingAssembler
from orchestrator.evals import eval_gate, runner
from orchestrator.evals.architect_eval import ArchitectEvalDeps
from orchestrator.evals.judge import PlanQualityVerdict
from orchestrator.evals.measurement import Scored
from orchestrator.evals.metrics import DECLINE_READERS
from orchestrator.evals.replay_frame import build_replay_frame_block
from orchestrator.evals.results import ResultStore

DEFAULT_JUDGE_PLAN_QUALITY = 0.77
"""The score the hermetic judge returns when a test supplies no verdict of its own."""


@dataclasses.dataclass(frozen=True)
class ArchInvokeUsage:
    """What one architect invoke reports spending, as ``AgentResult`` names it."""

    cost_usd: float
    duration_ms: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_create_tokens: int
    turns: int


DEFAULT_ARCH_INVOKE = ArchInvokeUsage(
    cost_usd=1.23, duration_ms=4567,
    input_tokens=12_000, output_tokens=3_000,
    cache_read_tokens=48_000, cache_create_tokens=6_000, turns=9,
)
"""The usage the hermetic architect invoke reports when a test supplies no result of its own."""


def _arch_task() -> dict:
    return {
        'id': 'df_task_2605',
        'name': 'implement the widget',
        'project_root': '/fake/project',
        'pre_task_commit': 'basecommit123',
        'reference': {'post_task_commit': 'postcommit456'},
        'task_definition': {'title': 'Widget', 'description': 'implement the widget'},
    }


def arch_task_no_reference() -> dict:
    """``_arch_task()`` minus the ``reference`` key.

    The shape the three back-filled fixtures had before task 3628 — a
    top-level ``post_task_commit`` with no ``reference`` block — which
    ``run_architect_eval`` reduces to an EMPTY reference diff.
    """
    task = _arch_task()
    task.pop('reference', None)
    return task


def architect_prompt_sent() -> str:
    """What the hermetic architect receives: the stubbed production briefing plus the replay frame."""
    return 'ARCH PROMPT' + build_replay_frame_block(_arch_task()['pre_task_commit'])


async def run_architect_eval_hermetic(
    cfg,
    *,
    produced_plan: dict,
    judge_return=None,
    judge_side_effect=None,
    arch_success: bool = True,
    invoke_side_effect=None,
    arch_result=None,
    orch_prices: dict | None = None,
    orch_config_side_effect=None,
    task_override=None,
    reference_diff: str = '--- a/x\n+++ b/x\n+ landed change\n',
    usage_gate=None,
    usage_gate_error=None,
    task_timeout_minutes=None,
    declines: dict[str, dict] | None = None,
    decline_reader_errors: dict[str, BaseException] | None = None,
    cleanup_side_effect=None,
    expose_mocks: dict | None = None,
    injected_gate=NO_INJECTED_GATE,
    gate_factory=None,
    cap_retry=None,
    store: ResultStore | None = None,
):
    """Drive run_architect_eval with every git/worktree/LLM boundary injected.

    The boundaries travel as one ``ArchitectEvalDeps`` passed as ``deps=``:
    worktree create/cleanup, the artifacts, the architect invoke, the plan
    judge and the reference-diff read are all doubles, and nothing is patched
    by dotted path except the public collaborators that are not deps
    (``runner.load_task``, ``runner.build_eval_orch_config``, the production
    briefing, ``run_verification`` and the cap-retry sleep).

    Returns ``(result, mocks)`` where ``mocks`` exposes the invoke/verify/save/
    judge mocks for assertions. ``invoke_side_effect`` lets a test make the live
    architect invoke raise (e.g. ``TimeoutError`` to simulate --timeout expiry).

    ``arch_result``, when supplied, is used VERBATIM as ``invoke_agent``'s
    return value instead of the default duck-typed MagicMock — the injection
    point the cap tests use to feed a REAL ``shared.cli_invoke.AgentResult``
    carrying the campaign 429 payload (``api_error_status=429`` +
    "You've hit your session limit · resets 8pm"), which the MagicMock cannot
    express because every attribute access on it returns a truthy Mock.

    ``orch_prices`` seeds the PRICE TABLE on the patched ``build_eval_orch_config``
    return, which the runner reads to resolve the architect component's cost
    provenance (Invariant P5). It defaults to an EMPTY dict rather than a bare
    ``MagicMock`` deliberately: a bare MagicMock's ``.get(model)`` returns a
    truthy Mock, which would send ``resolve_cost_usd`` down its price-table
    branch and multiply Mocks into ``cost_usd``. The default ``invoke_agent``
    return likewise carries REAL integer token/turn counts for EVERY usage leaf
    the runner reads (input/output, both cache counts, turns) for the same
    reason — a truthy Mock is one that ``or 0`` cannot rescue, and it would
    land verbatim in the persisted cell. Both make the harness model reality so
    a RED here is caused by production behaviour, never by the double.

    The FIVE DECLINE READERS on the injected artifacts double
    (``read_already_done`` / ``read_blocking_dependency`` / ``read_false_premise``
    / ``read_unactionable_task`` / ``read_ready_to_merge``) are pinned to
    ``None`` for exactly that reason (task 4760). Unpinned they are bare Mock
    attributes, so every one returns a TRUTHY Mock — and the moment the runner
    reads them, every caller of this harness would resolve to a decline it
    never asked for. ``None`` is what the real ``TaskArtifacts`` returns when
    the artifact file is absent, which is the state every pre-existing case is
    actually in, so pinning it keeps those cases meaning what they always meant.

    ``declines={'false_premise': {...}, ...}`` then opts a case IN, setting the
    named kinds' readers to return the supplied artifact dicts (the shape
    ``artifacts.write_*`` persists, including the ``reported_at`` stamp the
    terminal-kind ordering policy compares). An unnamed kind stays ``None``.
    ``decline_reader_errors={'already_done': OSError(...)}`` makes a named
    reader RAISE instead — the corrupt-artifact path, which must degrade that
    one kind rather than the cell.

    ``cleanup_side_effect`` hangs a side_effect on the injected
    ``cleanup_worktree``. It has to be a CONSTRUCTOR argument rather than
    something a caller sets on ``mocks['cleanup']`` afterwards, because the
    thing worth asserting is ORDERING *during* the run: cleanup rmtree's the
    relocated meta root, so a callback that samples state at the moment cleanup
    is awaited is the only way to prove an artifact was read while it still
    existed. ``expose_mocks`` is the dict such a callback reads the internal
    doubles from — the harness fills it BEFORE the run starts, which the
    returned ``mocks`` (available only afterwards) cannot do.

    ``orch_config_side_effect`` makes the patched ``build_eval_orch_config``
    RAISE — the harness-crash shape where the eval orch config (and therefore
    the price table) is never bound at all. It has to live here rather than in
    a caller-side ``patch``: this helper patches the same name from inside, so
    an outer patch would be shadowed.

    ``task_override`` replaces the task dict ``load_task`` returns (e.g.
    ``arch_task_no_reference()``), and ``reference_diff`` is what the injected
    ``read_reference_diff`` returns — both defaulting to today's hardcoded
    values so every existing caller keeps byte-identical behavior. Together
    they drive the ``judged_without_reference`` cases (task 3628).
    Usage-gate control (eval-revival φ, task 3630). ``build_eval_orch_config``
    is patched to return an anonymous MagicMock, so ``.usage_cap.enabled`` would
    be a TRUTHY Mock attribute — which, once ``run_architect_eval`` builds a
    gate, would have every one of this harness's pre-existing callers construct
    a REAL ``UsageGate`` out of a mock config (``_init_accounts`` +
    ``_sweep_stale_probe_dirs_once`` touch the filesystem, and whether the
    constructor raises would be undefined). So ``.usage_cap`` is pinned to a
    REAL ``UsageCapConfig(enabled=False)``: every existing test stays on the
    gate-ABSENT path, which is byte-identical to pre-φ behaviour AND is itself a
    real production configuration, so their assertions keep exactly the meaning
    they had.

    The new gate behaviours are then opt-in and explicit rather than an accident
    of MagicMock truthiness:

    - ``usage_gate=<gate>``: flips ``usage_cap.enabled`` on and patches
      ``eval_gate.UsageGate`` to a factory returning THIS gate. Exposed as
      ``mocks['UsageGate']`` (the factory) and ``mocks['gate']`` (the gate).
    - ``usage_gate_error=<exc>``: flips ``usage_cap.enabled`` on and patches
      ``eval_gate.UsageGate`` to RAISE it — the construction-failure /
      warn-and-degrade path.

    ``injected_gate=`` is a DIFFERENT knob from ``usage_gate=`` and the two must
    not be conflated (task 4427). ``usage_gate=`` patches the CONSTRUCTOR, so
    the cell still builds and OWNS its gate — that is the owned path every
    pre-4427 case exercises. ``injected_gate=`` instead forwards the value to
    ``run_architect_eval``'s own ``usage_gate=`` PARAMETER, i.e. a gate a
    campaign owner already built, which the cell must use without tearing down.
    Omitted, no ``usage_gate`` argument is passed at all, so the owned path is
    byte-identical to before; ``injected_gate=None`` passes an explicit ``None``
    (deliberately ungated), which is NOT the same as omitting it.

    ``gate_factory=`` replaces the deps' gate factory outright (the owned path
    then builds through it), and ``cap_retry=`` replaces the deps' cap-retry
    wrapper (omitted, the cell rides the production ``invoke_with_cap_retry``).

    ``shared.cli_invoke.asyncio.sleep`` is patched unconditionally (the
    ``_SLEEP_PATCH`` idiom from shared/tests/test_cap_retry.py:224) so cap-retry
    cooldowns never really sleep; exposed as ``mocks['sleep']``.

    ``store`` replaces the default ``MagicMock(spec=ResultStore)`` with a real
    store, so a cell the real producer wrote can be loaded back from disk;
    ``mocks['save']`` is then that store's bound ``save``.
    """
    if judge_return is None and judge_side_effect is None:
        judge_return = Scored(
            PlanQualityVerdict(
                plan_quality=DEFAULT_JUDGE_PLAN_QUALITY, per_criterion={}, reasoning='good',
            ),
            cost_usd=0.0,
        )

    invoke_return = arch_result if arch_result is not None else MagicMock(
        success=arch_success, output='done', **dataclasses.asdict(DEFAULT_ARCH_INVOKE),
    )
    mock_invoke = AsyncMock(return_value=invoke_return, side_effect=invoke_side_effect)

    artifacts_instance = MagicMock()
    artifacts_instance.read_plan.return_value = produced_plan
    # Absent-artifact default for every decline reader (see docstring): the
    # real TaskArtifacts returns None when the file does not exist, and a bare
    # Mock attribute would return a truthy Mock instead. The kind -> reader
    # mapping is taken from PRODUCTION rather than re-derived as
    # ``f'read_{kind}'``, which is wrong for ``unactionable``
    # (``read_unactionable_task``) and would silently stop pinning that kind.
    for _reader in DECLINE_READERS.values():
        getattr(artifacts_instance, _reader).return_value = None
    for _kind, _artifact in (declines or {}).items():
        assert _kind in DECLINE_READERS, f'unknown decline kind {_kind!r}'
        getattr(artifacts_instance, DECLINE_READERS[_kind]).return_value = _artifact
    for _kind, _exc in (decline_reader_errors or {}).items():
        assert _kind in DECLINE_READERS, f'unknown decline kind {_kind!r}'
        getattr(artifacts_instance, DECLINE_READERS[_kind]).side_effect = _exc

    # Stubs the PRODUCTION briefing only; the replay-frame subclass the runner
    # builds still appends its block for real (task 4844).
    production_architect_prompt = AsyncMock(return_value='ARCH PROMPT')

    mock_judge = AsyncMock(return_value=judge_return, side_effect=judge_side_effect)
    mock_verify = AsyncMock()
    result_store = MagicMock(spec=ResultStore) if store is None else store
    mock_sleep = AsyncMock()

    # NOT named *_config/*_cfg on purpose: this is the anonymous stand-in for
    # build_eval_orch_config's RETURN value, and a bare MagicMock() bound to a
    # config-named local is rejected by check_bare_magicmock_config.py. Only
    # ``.usage_cap`` is pinned to a real model — every other attribute stays a
    # permissive Mock, exactly as before.
    orch_stub = MagicMock()
    orch_stub.usage_cap = UsageCapConfig(enabled=False)
    orch_stub.prices = orch_prices if orch_prices is not None else {}

    # Bound to a local (rather than inlined into the patch below) so a test can
    # hang a side_effect on it and observe ORDERING against the decline reads —
    # cleanup_eval_worktree rmtree's the meta root, so anything that must read
    # an artifact has to have done so before this mock is awaited.
    mock_cleanup = AsyncMock(side_effect=cleanup_side_effect)

    # ``timeout_minutes`` rides in on the TASK dict (where it is untyped data)
    # rather than the ``timeout_override: int | None`` parameter, so a test can
    # ask for a sub-minute bound without lying to pyright about that signature.
    _task_def = _arch_task() if task_override is None else task_override
    if task_timeout_minutes is not None:
        _task_def['timeout_minutes'] = task_timeout_minutes

    gate_constructor = None
    if usage_gate is not None or usage_gate_error is not None:
        orch_stub.usage_cap = UsageCapConfig(enabled=True)
        gate_constructor = (
            MagicMock(side_effect=usage_gate_error) if usage_gate_error is not None
            else MagicMock(return_value=usage_gate)
        )

    if expose_mocks is not None:
        expose_mocks.update(artifacts=artifacts_instance, cleanup=mock_cleanup,
                            invoke=mock_invoke, judge=mock_judge)

    deps = ArchitectEvalDeps(
        store=result_store,
        create_worktree=AsyncMock(return_value=(Path('/fake/wt'), 'run-abc')),
        cleanup_worktree=mock_cleanup,
        open_artifacts=MagicMock(return_value=artifacts_instance),
        invoke=mock_invoke,
        judge=mock_judge,
        read_reference_diff=AsyncMock(return_value=reference_diff),
        gate_factory=(
            eval_gate.build_eval_usage_gate if gate_factory is None else gate_factory
        ),
        **({} if cap_retry is None else {'cap_retry': cap_retry}),
    )

    with contextlib.ExitStack() as es:
        p = es.enter_context
        if gate_constructor is not None:
            p(patch('orchestrator.evals.eval_gate.UsageGate', gate_constructor))
        p(patch('shared.cli_invoke.asyncio.sleep', mock_sleep))
        p(patch.object(BriefingAssembler, 'build_architect_prompt',
                       production_architect_prompt))
        p(patch('orchestrator.evals.runner.build_eval_orch_config',
                MagicMock(
                    return_value=orch_stub,
                    side_effect=orch_config_side_effect,
                )))
        p(patch('orchestrator.evals.runner.load_task',
                MagicMock(return_value=_task_def)))
        p(patch('orchestrator.verify.run_verification', mock_verify))
        extra: dict[str, Any] = (
            {} if injected_gate is NO_INJECTED_GATE
            else {'usage_gate': injected_gate}
        )
        result = await runner.run_architect_eval(
            Path('/fake/task.json'), cfg, base_config=MagicMock(),
            deps=deps, **extra,
        )
    return result, {
        'invoke': mock_invoke, 'verify': mock_verify,
        'save': result_store.save, 'judge': mock_judge,
        'sleep': mock_sleep, 'UsageGate': gate_constructor, 'gate': usage_gate,
        'cleanup': mock_cleanup, 'artifacts': artifacts_instance,
        'gate_factory': gate_factory,
    }


def only_record(metrics: dict) -> dict:
    """The one unmeasured record a cell persisted."""
    records = metrics['invocation_errors']
    assert isinstance(records, list) and len(records) == 1, records
    return records[0]


def token_agent_result(
    input_tokens,
    output_tokens,
    *,
    cost_usd: float = DEFAULT_ARCH_INVOKE.cost_usd,
    cache_read_tokens: int | None = None,
    cache_create_tokens: int | None = None,
):
    """A REAL ``AgentResult`` carrying explicit token counts.

    Real, not a MagicMock, because the ``int | None`` "provider did not report"
    shape is exactly what the token-usage tests pin — a MagicMock cannot express a
    genuine ``None`` on an attribute the runner reads.
    """
    return AgentResult(
        success=True,
        output='done',
        cost_usd=cost_usd,
        duration_ms=DEFAULT_ARCH_INVOKE.duration_ms,
        turns=3,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_create_tokens=cache_create_tokens,
    )


PROXY_ENV = {'ANTHROPIC_BASE_URL': 'http://localhost:8000/v1'}
