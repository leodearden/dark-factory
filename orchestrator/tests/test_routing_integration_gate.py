"""σ — Routing integration gate: the B+H boundary suite through the workflow
rig (task 2545, plans/adaptive-model-routing-prd.md §Boundary-test sketch).

All six upstream dependencies are landed on main: ε — the layered resolver
``orchestrator.routing.resolve_route`` (task 2535); ζ — typed
``model_overrides`` on task metadata (task 2536); η — the out-of-band
dispatch helper ``orchestrator.routing_dispatch.resolve_and_record_route``
(task 2537); μ — the harness-owned retry-tier bump,
``Harness._maybe_bump_routing_tier`` (task 2542); ν — simple-task
turn-cap-saturation attribution (task 2543); δ — the per-(model×role) digest
rollup, ``digest.model_role_rollup`` (task 2534). So σ adds NO production
code: every test here is a GREEN-ON-ARRIVAL integration/regression guard over
already-landed behavior, not RED→GREEN feature TDD. A scenario that comes up
RED is therefore a GENUINE integration gap, not a synthetic-input failure —
see plan.json's design_decisions for the escalation discipline this implies.

SEAM. Every unit test covering one of these six intermediates either calls
``resolve_route`` directly, or drives a single seam against a MagicMock
config / mocked scheduler / mocked report. None of them proves the pieces
INTERLOCK end to end. Production ``TaskWorkflow._invoke`` (workflow.py) calls
``invoke_with_cap_retry(invoke_fn=invoke_agent)`` against the MODULE-LEVEL
``orchestrator.workflow.invoke_agent`` — not a constructor-injected
parameter — so this suite injects one level deeper:
``monkeypatch.setattr('orchestrator.workflow.invoke_agent', fake)`` (and the
per-module twins ``orchestrator.steward.invoke_agent`` for the steward site).
This lets the REAL ``_invoke`` run — ``resolve_route``, the ceiling-spend
query, ``_record_routing_decision``, the ``routing_decision`` event emit, and
the ``metadata.routing`` merge-write all execute — and only the agent
subprocess is faked. This is the established pattern of
``test_workflow_e2e.py``, ``test_config_reload_integration_gate.py`` (task
2008), and the direct precedent ``test_verdict_servers_integration_gate.py``
(task 2488, "θ adds NO production code"). Patching ``TaskWorkflow._invoke``
wholesale (the per-role unit tests' pattern) skips route resolution
entirely; patching ``invoke_with_cap_retry`` means no runner runs at all so
nothing proves the model reached the CLI argv; patching
``resolve_and_record_route`` mocks out exactly the behavior scenario 9 is
supposed to prove.

CONFIG. A REAL ``OrchestratorConfig`` is used throughout (never a
MagicMock) — ``resolve_route`` does real membership/dict operations against
``config.routing.*`` that a bare-MagicMock child cannot satisfy. The default
``config`` fixture below (mirroring ``test_workflow_e2e.py`` /
``test_verdict_servers_integration_gate.py``) resolves through the autouse
``_isolate_orch_config`` fixture, i.e. the repo's live operational
``dark-factory-orchestrator.yaml`` layered over the bundled
``defaults.yaml`` — this is fine for scenarios that only need "some real,
internally-consistent config". Scenarios whose claim is specifically about
CODE DEFAULTS (byte-equivalence, hot-reload starting values) instead opt
into the ``code_default_config`` conftest fixture, which points
``ORCH_CONFIG_PATH`` at a guaranteed-absent file so only the bundled
``defaults.yaml`` loads — otherwise "stock" would silently mean whatever the
live operational yaml happens to contain today.

FIXTURES ARE MODULE-LOCAL. ``orchestrator/tests/conftest.py`` and
``_workflow_helpers.py`` are imported, never edited: a conftest.py edit trips
verify.py's ``has_conftest`` and widens the merge-time scoped-test selection
from a subset to the whole orchestrator package — the same rationale
documented in ``test_routing_byte_equivalence.py``:23-25 and
``test_config_reload_integration_gate.py``:16-18. ``_workflow_helpers.py``
needs no edit either: ``event_store``/``cost_store`` are plain
post-construction attributes on ``TaskWorkflow``, and the per-role route
recorder below is a module-local ``AgentStub`` subclass rather than a change
to the shared stub ~70 other test files use.

SCENARIO -> TEST CLASS MAP (PRD boundary-test numbering):
    1, 2  TestOverrideWinsAtTheSeam,                 (plan step-1)
          TestInvalidOverrideFallsThroughFailSafe
    3     TestByteEquivalenceThroughTheRig            (plan step-2)
    6     TestTierStableWithinOneDispatch             (plan step-3)
    5, 6  TestRetryTierBumpAcrossDispatches           (plan step-4;
                                                        also covers 6's
                                                        harness-boundary
                                                        DONE-no-bump half)
    7     TestCeilingFallbackDoesNotBlockDispatch      (plan step-5)
    8     TestSaturationStampRoutesNextDispatchFullPath (plan step-6)
    4     TestSimpleTaskModelReloadReachesTheSeam      (plan step-7)
    11    TestUnknownRuleKeyRejectedPriorRulesStillRoute (plan step-8)
    9     TestOutOfBandParity                          (plan step-9)
    12    TestRollupRendersRigProducedRows             (plan step-10)
    10    OUT OF SCOPE — probe-gated fable admission lives in β/ξ
          (``orchestrator.routing.probe_models``), not here; see
          ``test_routing.py`` (task beta) for that coverage. Named here so a
          reader diffing this suite against the PRD's 12-row table can see
          the gap is deliberate, not a silent omission.

Scenario 3's flat 12-role table is NOT re-parametrized here — it stays owned
by ``test_routing_byte_equivalence.ALL_DISPATCHABLE_ROLES``. This suite's
distinct, non-duplicative claim is the INTEGRATION half: the roles a real
dispatch actually invokes reach the CLI seam with byte-identical stock
values, and the plan-shape rule fires off the real ``self.plan`` — not a
hand-built ``PlanShape``.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _recording_event_store import _RecordingEventStore
from _workflow_helpers import (
    AgentStub,
    FakeMetadataBackend,
    _build_harness,
    _build_workflow,
    _derive_meta_root_like_production,  # noqa: F401  autouse fixture, see its docstring
    _init_repo,
    wire_metadata_backend,
)
from shared.cost_store import CostStore

from orchestrator.agents.invoke import AgentResult
from orchestrator.agents.roles import ARCHITECT, DEBUGGER, IMPLEMENTER
from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import GitConfig, OrchestratorConfig, SandboxConfig
from orchestrator.event_store import EventType
from orchestrator.git_ops import GitOps
from orchestrator.harness import TaskReport
from orchestrator.scheduler import TaskAssignment
from orchestrator.verify import VerifyResult
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome

# ---------------------------------------------------------------------------
# Fixtures — file-local, mirroring test_workflow_e2e.py:153-203 /
# test_verdict_servers_integration_gate.py:100-146. A REAL OrchestratorConfig
# is required (not _workflow_helpers._make's MagicMock) because the real
# _invoke calls resolve_route(route_inputs, self.config), which does real
# attribute/membership reads against config.routing.* that a MagicMock can't
# satisfy.
# ---------------------------------------------------------------------------


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A bare-minimum git repo with an initial commit."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_init_repo(repo))
    return repo


@pytest.fixture
def config(git_repo: Path) -> OrchestratorConfig:
    return OrchestratorConfig(
        project_root=git_repo,
        max_concurrent_tasks=1,
        max_execute_iterations=5,
        max_verify_attempts=3,
        max_review_cycles=2,
        git=GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
        ),
        # Real _invoke against a bare-minimum fake worktree (no linked-worktree
        # .git gitdir file) — the sandbox write-set path has nothing valid to
        # parse. These scenarios exercise routing/workflow logic, not
        # sandboxing — keep sandbox off for hermeticity (mirrors
        # test_workflow_e2e.py's config fixture).
        sandbox=SandboxConfig(enabled=False),
    )


@pytest.fixture
def git_ops(config: OrchestratorConfig) -> GitOps:
    return GitOps(config.git, config.project_root)


@pytest.fixture
def task_assignment() -> TaskAssignment:
    return TaskAssignment(
        task_id='42',
        task={
            'id': '42',
            'title': 'Routing boundary task',
            'description': 'Exercise the real _invoke routing-resolution path',
            'status': 'pending',
            'metadata': {'files': ['lib']},
            'dependencies': [],
        },
        modules=['lib'],
    )


# ---------------------------------------------------------------------------
# Shared rig helpers
# ---------------------------------------------------------------------------


class _RoutingRecorderStub(AgentStub):
    """``AgentStub`` subclass that records the resolved route per role.

    Captures ``(model, effort, max_turns, max_budget_usd, backend)`` exactly
    as they arrive at the CLI seam — i.e. downstream of ``resolve_route`` and
    ``_invoke``'s full layering — keyed by the role ``AgentStub._detect_role``
    derives from ``system_prompt``. Delegates to ``super().invoke_agent(**kwargs)``
    so every real file/git side effect ``AgentStub`` performs (writing
    plan.json, completing steps, submitting review verdicts, ...) still
    happens and ``workflow.run()`` can reach DONE.

    ``**kwargs`` (rather than mirroring ``AgentStub.invoke_agent``'s full
    explicit signature) is deliberate and robust: ``invoke_with_cap_retry``'s
    fast path calls ``invoke_fn(**invoke_kwargs, config_dir=...)`` — every
    argument arrives as a keyword (see
    ``test_verdict_servers_integration_gate.py``:249-253's identical note) —
    so this stays correct even if production adds a new keyword arg.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.route_by_role: dict[str, dict[str, object]] = {}

    async def invoke_agent(self, **kwargs):
        role = self._detect_role(kwargs['system_prompt'])
        self.route_by_role[role] = {
            'model': kwargs.get('model'),
            'effort': kwargs.get('effort'),
            'max_turns': kwargs.get('max_turns'),
            'max_budget_usd': kwargs.get('max_budget_usd'),
            'backend': kwargs.get('backend'),
        }
        return await super().invoke_agent(**kwargs)


def _routing_events(rec: _RecordingEventStore) -> list[dict]:
    """Filter *rec*'s captured events down to ``routing_decision`` entries.

    Mirrors the filter idiom in ``test_routing_dispatch._routing_entries``:69
    — each returned entry is the full ``{'task_id':..., 'data': {...}}``
    shape ``_RecordingEventStore.emit`` records, so callers index
    ``['data']`` for the payload fields.
    """
    return [entry for (etype, entry) in rec.events if etype == EventType.routing_decision]


def _artifacts_for(worktree: Path) -> TaskArtifacts:
    """Real on-disk ``TaskArtifacts`` at the derived ``.task-meta`` sibling
    root for *worktree* — the SAME root production's ``self.artifacts``
    derives via ``TaskArtifacts.meta_root_for``. Reimplemented locally
    (rather than relying on the autouse ``_derive_meta_root_like_production``
    shim, which exists for legacy bare-``TaskArtifacts(cwd)`` call sites, or
    importing across test files) per this suite's module-local-fixtures
    discipline — mirrors
    ``test_verdict_servers_integration_gate._artifacts_for`` (task 2488).
    """
    return TaskArtifacts(worktree, TaskArtifacts.meta_root_for(worktree.parent, worktree.name))


def _seed_workflow_artifacts(
    workflow: TaskWorkflow, *, tmp_path: Path, task_id: str = '42',
) -> TaskArtifacts:
    """Wire a real on-disk ``TaskArtifacts`` + worktree directory onto
    *workflow*.

    ``_build_workflow`` only constructs the ``TaskWorkflow`` — it does not
    create a worktree or set ``.worktree``/``.artifacts`` (those are normally
    set by the full ``run()`` cycle's ``_setup``). The merger byte-
    equivalence test below drives ``_resolve_and_resubmit`` directly,
    bypassing ``run()``, so this seeds what that path reads. Mirrors
    ``test_verdict_servers_integration_gate._seed_workflow_artifacts`` (task
    2488), reimplemented locally rather than imported across test files (see
    module docstring, FIXTURES ARE MODULE-LOCAL).
    """
    worktree = tmp_path / 'wt'
    worktree.mkdir(parents=True, exist_ok=True)
    artifacts = _artifacts_for(worktree)
    artifacts.init(task_id, 'Routing boundary task', 'desc', base_commit='oldbase')
    workflow.artifacts = artifacts
    workflow.worktree = worktree
    return artifacts


class _MinimalRouteRecorder:
    """Bare ``invoke_agent`` double for scenarios that drive ``_invoke``
    DIRECTLY rather than through a full ``workflow.run()`` cycle — no
    plan.json / git side effects needed, only the resolved route at the CLI
    seam. Mirrors ``test_config_reload_integration_gate.StubInvoke``: records
    every call's kwargs (in call order) and returns a canned success result.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def invoke_agent(self, **kwargs) -> AgentResult:
        self.calls.append(kwargs)
        return AgentResult(success=True, output='OK')


def _config_key(role_name: str) -> str:
    """Reviewer* collapse, reconstructed independently of
    ``orchestrator.routing._config_key`` — mirrors
    ``test_routing_byte_equivalence.py``:73-80's documented rationale:
    invariant 3 is defined as matching "the same config fields", so the
    expected side must not import the resolver's own private helper (that
    would let a bug in the helper mask itself)."""
    return 'reviewer' if role_name.startswith('reviewer') else role_name


def _expected_from_config(
    cfg: OrchestratorConfig, role_name: str,
) -> tuple[str, str, float, int]:
    """The pre-epsilon resolution: config.models/effort/budgets/max_turns
    read by the reviewer-collapsed key, with no further logic — what a stock
    dispatch (no overrides, no matching rule) must byte-equal."""
    key = _config_key(role_name)
    return (
        getattr(cfg.models, key),
        getattr(cfg.effort, key),
        getattr(cfg.budgets, key),
        getattr(cfg.max_turns, key),
    )


def _mock_verify_passes() -> AsyncMock:
    """A ``run_scoped_verification`` double that always reports success.

    Mirrors ``test_workflow_e2e.TestHappyPath.test_single_task_completes``'s
    patch — every scenario in this suite that drives a full ``workflow.run()``
    to DONE needs the VERIFY phase to pass without a real pytest/ruff/pyright
    invocation.
    """
    return AsyncMock(return_value=VerifyResult(
        passed=True, test_output='OK', lint_output='',
        type_output='', summary='All checks passed',
    ))


# ---------------------------------------------------------------------------
# Boundary scenarios 1, 2 — metadata.model_overrides at the real seam
# (plan step-1).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestOverrideWinsAtTheSeam:
    """PRD boundary scenario 1: a per-role ``metadata.model_overrides`` entry
    outranks the static per-role config layer, and the winning model actually
    reaches the CLI seam (not just ``resolve_route``'s return value) — while a
    non-overridden role in the same dispatch is untouched (per-role, not
    global)."""

    async def test_override_reaches_seam_and_is_per_role(
        self, config, git_ops, task_assignment, monkeypatch,
    ):
        # config.models.implementer is 'sonnet' at stock config (defaults.yaml)
        # -- distinct from the 'haiku' override below, so the layers genuinely
        # disagree and only a real seam proof can show which one actually won.
        assert config.models.implementer != 'haiku'
        task_assignment.task['metadata']['model_overrides'] = {'implementer': 'haiku'}

        stub = _RoutingRecorderStub()
        workflow, _scheduler = _build_workflow(config, git_ops, task_assignment, stub)
        rec = _RecordingEventStore()
        workflow.event_store = rec
        monkeypatch.setattr('orchestrator.workflow.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification', _mock_verify_passes(),
        )

        outcome = (await workflow.run()).outcome

        # (a) the override reached the CLI seam, not just the resolver.
        assert stub.route_by_role['implementer']['model'] == 'haiku'

        # (b) the implementer's routing_decision event names the override
        # layer and carries no rejection.
        impl_events = [e for e in _routing_events(rec) if e['data']['role'] == 'implementer']
        assert impl_events, f'expected an implementer routing_decision event: {rec.events!r}'
        assert impl_events[-1]['data']['source_layer'] == 'metadata_override'
        assert impl_events[-1]['data']['rejected'] == []

        # (c) a NON-overridden role in the same dispatch still resolves at the
        # config layer -- proving the override is per-role, not global.
        assert stub.route_by_role['architect']['model'] == config.models.architect
        arch_events = [e for e in _routing_events(rec) if e['data']['role'] == 'architect']
        assert arch_events, f'expected an architect routing_decision event: {rec.events!r}'
        assert arch_events[-1]['data']['source_layer'] == 'config'

        # (d) the dispatch reaches DONE.
        assert outcome == WorkflowOutcome.DONE


@pytest.mark.asyncio
class TestInvalidOverrideFallsThroughFailSafe:
    """PRD boundary scenario 2: a ``model_overrides`` entry that fails
    allowlist validation is REJECTED, resolution falls through to the next
    layer, and the dispatch is never blocked by the mis-config (invariant 2)."""

    async def test_invalid_override_falls_through_to_config_layer(
        self, config, git_ops, task_assignment, monkeypatch,
    ):
        assert 'gpt-9' not in config.routing.allowed_models
        task_assignment.task['metadata']['model_overrides'] = {'implementer': 'gpt-9'}

        stub = _RoutingRecorderStub()
        workflow, _scheduler = _build_workflow(config, git_ops, task_assignment, stub)
        rec = _RecordingEventStore()
        workflow.event_store = rec
        monkeypatch.setattr('orchestrator.workflow.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification', _mock_verify_passes(),
        )

        outcome = (await workflow.run()).outcome

        # (a) the seam saw the fallen-through config-layer model, NOT 'gpt-9'.
        assert stub.route_by_role['implementer']['model'] == config.models.implementer
        assert stub.route_by_role['implementer']['model'] != 'gpt-9'

        # (b) the implementer event records the exact rejection reason and the
        # layer resolution actually fell through to.
        impl_events = [e for e in _routing_events(rec) if e['data']['role'] == 'implementer']
        assert impl_events, f'expected an implementer routing_decision event: {rec.events!r}'
        assert impl_events[-1]['data']['rejected'] == ['metadata_override:model-not-in-allowlist']
        assert impl_events[-1]['data']['source_layer'] == 'config'

        # (c) invariant 2: a routing mis-config never blocks a dispatch.
        assert outcome == WorkflowOutcome.DONE

        # (d) the rejection is mirrored onto task metadata. `routing.latest`
        # reflects whichever role's _record_routing_decision ran LAST in this
        # dispatch -- EVERY invocation overwrites it (workflow.py:12718) -- so
        # by the time run() reaches DONE it holds a later role's (e.g.
        # reviewer/merger) clean decision, not implementer's. `history` is the
        # durable per-invocation record (bounded at 5, well above this
        # dispatch's invocation count), so scan it for implementer's own entry.
        history = workflow.task['metadata']['routing']['history']
        impl_history = [h for h in history if h['role'] == 'implementer']
        assert impl_history, f'expected an implementer entry in routing history: {history!r}'
        assert impl_history[-1]['rejected'] == ['metadata_override:model-not-in-allowlist']


# ---------------------------------------------------------------------------
# Boundary scenario 3 — byte-equivalence, integration half (plan step-2).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestByteEquivalenceThroughTheRig:
    """PRD boundary scenario 3, integration half: a full stock dispatch (no
    overrides, shipped default rules) resolves every in-band role's (model,
    effort, max_turns) exactly as pre-epsilon ``_invoke`` did — reading
    ``config.models``/``effort``/``max_turns`` by the reviewer-collapsed
    key — and the ``routing_decision`` event's ``budget_usd`` matches
    ``config.budgets`` the same way. Also proves the theta plan-shape rule
    fires off the REAL ``self.plan`` (not a hand-built ``PlanShape``).

    Uses the opt-in ``code_default_config`` conftest fixture so "stock"
    means the bundled ``defaults.yaml`` — the autouse ``_isolate_orch_config``
    otherwise pins ``ORCH_CONFIG_PATH`` at the repo's live operational yaml,
    which drifts and would make "stock" mean the wrong thing.

    The flat 12-role table (every role ``_invoke``-addressable, including
    out-of-band-only roles) stays owned by
    ``test_routing_byte_equivalence.ALL_DISPATCHABLE_ROLES`` — this class's
    distinct, non-duplicative claim is the INTEGRATION half: the roles a
    real dispatch actually invokes reach the CLI seam with byte-identical
    stock values.
    """

    #: The role set a conflict-free ``workflow.run()`` dispatch actually
    #: invokes: MERGER is deliberately excluded here -- it is only dispatched
    #: from ``_resolve_and_resubmit`` when a real merge conflict occurs
    #: (workflow.py:10859), which this trivial-repo happy path never
    #: produces (empirically confirmed: a conflict-free DONE dispatch never
    #: touches it). ``test_merger_byte_equivalent_through_conflict_resolution``
    #: below covers merger's byte-equivalence separately, through the same
    #: real seam, by driving that method directly -- the established pattern
    #: for exercising it (test_verdict_servers_integration_gate.
    #: TestMergerBoundary), since a synthetic conflict never needs a genuine
    #: git conflict to construct.
    _RUN_ROLES = frozenset({'architect', 'implementer', 'reviewer_comprehensive', 'judge'})

    @pytest.fixture
    def stock_config(self, git_repo: Path, code_default_config) -> OrchestratorConfig:
        return OrchestratorConfig(
            project_root=git_repo,
            max_concurrent_tasks=1,
            max_execute_iterations=5,
            max_verify_attempts=3,
            max_review_cycles=2,
            git=GitConfig(
                main_branch='main',
                branch_prefix='task/',
                remote='origin',
                worktree_dir='.worktrees',
            ),
            sandbox=SandboxConfig(enabled=False),
            # Opt-in (default False): brings 'judge' into the in-band role
            # set this single dispatch invokes, alongside architect/
            # implementer/reviewer*/merger. Does not change dispatch control
            # flow here — the EXECUTE loop's own pending-steps check already
            # terminates naturally once the implementer's one invocation
            # completes both PLAN steps (workflow.py:7920), independent of
            # the judge's verdict — it only adds one extra judge invocation
            # to observe.
            judge_after_each_iteration=True,
        )

    async def test_stock_dispatch_matches_config_byte_for_byte(
        self, stock_config, git_ops, task_assignment, monkeypatch,
    ):
        stub = _RoutingRecorderStub()
        workflow, _scheduler = _build_workflow(stock_config, git_ops, task_assignment, stub)
        rec = _RecordingEventStore()
        workflow.event_store = rec
        monkeypatch.setattr('orchestrator.workflow.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification', _mock_verify_passes(),
        )

        outcome = (await workflow.run()).outcome
        assert outcome == WorkflowOutcome.DONE

        # (b) the recorder covered the FULL in-band role set this real
        # dispatch invokes -- a silently-skipped role fails loudly here.
        # Pointer: the flat 12-role table (every role _invoke can address,
        # including out-of-band-only roles) lives in
        # test_routing_byte_equivalence.ALL_DISPATCHABLE_ROLES.
        assert set(stub.route_by_role) == self._RUN_ROLES

        # (a)/(c) for EVERY observed role: the seam's (model, effort,
        # max_turns) match config byte-for-byte, and the routing_decision
        # event's budget_usd matches config.budgets the same way, with every
        # event resolved at the config layer, no rule, no rejection.
        for role_name in self._RUN_ROLES:
            route = stub.route_by_role[role_name]
            expected_model, expected_effort, expected_budget, expected_max_turns = (
                _expected_from_config(stock_config, role_name)
            )
            assert route['model'] == expected_model, role_name
            assert route['effort'] == expected_effort, role_name
            assert route['max_turns'] == expected_max_turns, role_name

            role_events = [e for e in _routing_events(rec) if e['data']['role'] == role_name]
            assert role_events, f'expected a routing_decision event for {role_name}'
            data = role_events[-1]['data']
            # Assert budget on the EVENT, not the seam's max_budget_usd kwarg
            # -- data['budget_usd'] is the authoritative mirror
            # _record_routing_decision writes for cost triage (the
            # out-of-band sibling, test_routing_dispatch.
            # TestAppliedBudgetAnnotation, documents why an applied-budget
            # caller-enforced cap is annotated onto the event rather than
            # trusted from the seam kwarg alone).
            assert data['budget_usd'] == expected_budget, role_name
            assert data['source_layer'] == 'config', role_name
            assert data['rule_id'] is None, role_name
            assert data['rejected'] == [], role_name

    async def test_rust_shaped_plan_upgrades_implementer_and_debugger_not_architect(
        self, stock_config, git_ops, task_assignment, git_repo, monkeypatch,
    ):
        """(c) Rust-heuristic parity through the rig: a plan shaped like the
        historical crates/-only ``rust-large-plan-implementer`` heuristic
        (>=12 steps across >=3 distinct ``crates/``-prefixed modules) still
        upgrades implementer/debugger via the generalized ``large-plan-steps``
        rule (defaults.yaml:257) — while architect, not in that rule's role
        list, is NOT upgraded (stays source_layer='config').

        Drives ``_invoke`` directly (not a full ``workflow.run()``): only the
        resolved route matters here, so ``workflow.plan``/``workflow.modules``
        are seeded directly rather than produced by a real architect stub.
        """
        stub = _MinimalRouteRecorder()
        workflow, _scheduler = _build_workflow(stock_config, git_ops, task_assignment, stub)
        rec = _RecordingEventStore()
        workflow.event_store = rec
        monkeypatch.setattr('orchestrator.workflow.invoke_agent', stub.invoke_agent)

        workflow.plan = {'steps': [{'id': f's{i}', 'status': 'pending'} for i in range(12)]}
        workflow.modules = [
            'crates/alpha/src/lib.rs', 'crates/beta/src/lib.rs', 'crates/gamma/src/lib.rs',
        ]

        await workflow._invoke(IMPLEMENTER, 'impl prompt', git_repo)
        await workflow._invoke(DEBUGGER, 'debug prompt', git_repo)
        await workflow._invoke(ARCHITECT, 'arch prompt', git_repo)

        assert stub.calls[0]['model'] == 'opus'  # implementer, upgraded
        assert stub.calls[1]['model'] == 'opus'  # debugger, upgraded
        # architect: not in large-plan-steps' role list -- unaffected, but
        # its stock config model already happens to be 'opus' too, so the
        # discriminating assertion below is source_layer/rule_id, not model.
        assert stub.calls[2]['model'] == stock_config.models.architect

        impl_data = [e for e in _routing_events(rec) if e['data']['role'] == 'implementer'][-1]['data']
        assert impl_data['source_layer'] == 'policy_rule'
        assert impl_data['rule_id'] == 'large-plan-steps'

        dbg_data = [e for e in _routing_events(rec) if e['data']['role'] == 'debugger'][-1]['data']
        assert dbg_data['source_layer'] == 'policy_rule'
        assert dbg_data['rule_id'] == 'large-plan-steps'

        arch_data = [e for e in _routing_events(rec) if e['data']['role'] == 'architect'][-1]['data']
        assert arch_data['source_layer'] == 'config'
        assert arch_data['rule_id'] is None

    async def test_merger_byte_equivalent_through_conflict_resolution(
        self, stock_config, git_ops, task_assignment, tmp_path, monkeypatch,
    ):
        """merger is conditionally invoked -- only from
        ``TaskWorkflow._resolve_and_resubmit`` when a real merge conflict
        occurs (workflow.py:10859) -- so it never appears in the conflict-free
        ``_RUN_ROLES`` dispatch above. Drive that method directly (mirrors
        ``test_verdict_servers_integration_gate.TestMergerBoundary``), the
        same real seam as everywhere else in this suite, to prove merger's
        stock resolution is ALSO byte-equivalent.
        """
        stub = _RoutingRecorderStub()
        workflow, _scheduler = _build_workflow(stock_config, git_ops, task_assignment, stub)
        rec = _RecordingEventStore()
        workflow.event_store = rec
        _seed_workflow_artifacts(workflow, tmp_path=tmp_path)
        workflow.git_ops.rebase_onto_main = AsyncMock()  # type: ignore[method-assign]
        workflow._submit_to_merge_queue = AsyncMock(  # type: ignore[method-assign]
            return_value=WorkflowOutcome.DONE,
        )
        workflow._mark_blocked = AsyncMock(  # type: ignore[method-assign]
            return_value=WorkflowOutcome.BLOCKED,
        )
        workflow._write_merge_failure_review = MagicMock()  # type: ignore[method-assign]
        monkeypatch.setattr('orchestrator.workflow.invoke_agent', stub.invoke_agent)

        # Outcome is irrelevant here (AgentStub._merger writes no verdict, so
        # this fails safe to BLOCKED) -- only the ROUTE reaching the real
        # seam is under test.
        await workflow._resolve_and_resubmit('task/42', 'conflict in lib.py', merge_phase=True)

        expected_model, expected_effort, expected_budget, expected_max_turns = (
            _expected_from_config(stock_config, 'merger')
        )
        assert stub.route_by_role['merger']['model'] == expected_model
        assert stub.route_by_role['merger']['effort'] == expected_effort
        assert stub.route_by_role['merger']['max_turns'] == expected_max_turns

        merger_events = [e for e in _routing_events(rec) if e['data']['role'] == 'merger']
        assert merger_events, f'expected a merger routing_decision event: {rec.events!r}'
        data = merger_events[-1]['data']
        assert data['budget_usd'] == expected_budget
        assert data['source_layer'] == 'config'
        assert data['rule_id'] is None
        assert data['rejected'] == []


# ---------------------------------------------------------------------------
# Boundary scenario 6 — routing_tier is stable WITHIN one dispatch (plan
# step-3).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestTierStableWithinOneDispatch:
    """PRD boundary scenario 6: ``metadata.routing.routing_tier`` never
    changes WITHIN a single dispatch, no matter how many verify/debug
    iterations it takes to reach DONE. ``workflow.py`` only ever READS
    ``routing_tier`` (``_invoke`` at workflow.py:12227); the sole writer is
    ``Harness._maybe_bump_routing_tier`` at slot release (task mu,
    harness.py:9027), which this suite never drives here.

    Seeds ``routing_tier=1`` so the shipped ``retry-tier-up`` rule
    (defaults.yaml:320) is LIVE for the whole dispatch: if tier read/write
    ever drifted mid-dispatch, it would surface as a visible MODEL change
    (sonnet -> opus), not merely a silent counter change -- a stronger
    signal than asserting the counter alone.

    Uses ``code_default_config`` (rather than the module ``config`` fixture)
    to deliberately INSTALL the shipped default rules -- the ambient
    operational yaml happens not to override ``routing`` today, but this
    scenario's premise (retry-tier-up is live) must not depend on that
    drifting, mirroring ``test_routing_retry_tier_up.py``'s own isolation
    rationale.

    Wires a real ``FakeMetadataBackend`` onto the rig's ``FakeScheduler`` via
    ``wire_metadata_backend`` so ``_record_routing_decision``'s
    ``scheduler.update_task(..., metadata_mode='merge')`` mirror-write
    actually succeeds and is recorded -- the bare ``FakeScheduler.update_task``
    has no ``metadata_mode`` parameter and a call with it is silently
    swallowed by ``_record_routing_decision``'s best-effort try/except
    (workflow.py:12701-12717), which would make assertion (c) vacuous.
    """

    @pytest.fixture
    def stock_config(self, git_repo: Path, code_default_config) -> OrchestratorConfig:
        return OrchestratorConfig(
            project_root=git_repo,
            max_concurrent_tasks=1,
            max_execute_iterations=5,
            max_verify_attempts=3,
            max_review_cycles=2,
            git=GitConfig(
                main_branch='main',
                branch_prefix='task/',
                remote='origin',
                worktree_dir='.worktrees',
            ),
            sandbox=SandboxConfig(enabled=False),
        )

    async def test_routing_tier_stable_across_verify_debug_iterations(
        self, stock_config, git_ops, task_assignment, monkeypatch,
    ):
        assert stock_config.models.implementer == stock_config.models.debugger == 'sonnet'
        task_assignment.task['metadata']['routing'] = {'routing_tier': 1}

        stub = _RoutingRecorderStub()
        workflow, scheduler = _build_workflow(stock_config, git_ops, task_assignment, stub)
        rec = _RecordingEventStore()
        workflow.event_store = rec
        backend = FakeMetadataBackend()
        wire_metadata_backend(
            scheduler, backend, seed=task_assignment.task['metadata'], grants=True,
        )
        monkeypatch.setattr('orchestrator.workflow.invoke_agent', stub.invoke_agent)

        # False, False, then True FOREVER -- not just for 3 calls. A full
        # dispatch to DONE calls run_scoped_verification again in the merge
        # phase (workflow.py:3614, _run_merge_phase) AFTER the
        # verify/debug loop's own 3 calls; a length-3 side_effect list would
        # StopIteration on that 4th call and BLOCK the dispatch for a reason
        # unrelated to what this scenario tests.
        _verify_call_count = 0

        async def _verify_sequence(*args, **kwargs) -> VerifyResult:
            nonlocal _verify_call_count
            _verify_call_count += 1
            if _verify_call_count <= 2:
                return VerifyResult(
                    passed=False, test_output='FAILED test_farewell', lint_output='',
                    type_output='', summary='Failures: tests failed',
                )
            return VerifyResult(
                passed=True, test_output='OK', lint_output='',
                type_output='', summary='All checks passed',
            )

        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification', _verify_sequence,
        )

        outcome = (await workflow.run()).outcome

        # (d) two verify failures + debugger fixes still reach DONE.
        assert outcome == WorkflowOutcome.DONE

        # The EXECUTE->VERIFY loop re-entered implementer/debugger at least
        # three times (1 implementer from EXECUTE + 2 debugger from the two
        # verify failures).
        idx_events = [
            e['data'] for e in _routing_events(rec)
            if e['data']['role'] in ('implementer', 'debugger')
        ]
        assert len(idx_events) >= 3, (
            f'expected >=3 implementer/debugger routing_decision events '
            f'(1 implementer + 2 debugger): {idx_events!r}'
        )

        # (a) every one of those events carries the SAME routing_tier --
        # collected into a set of exactly one element.
        tiers = {e['routing_tier'] for e in idx_events}
        assert tiers == {1}, f'routing_tier drifted mid-dispatch: {idx_events!r}'

        # (b) every one of those events resolves to the SAME model -- no
        # mid-dispatch ladder drift. (Also confirms retry-tier-up genuinely
        # fired rather than being inert: both roles' stock model is
        # 'sonnet', asserted above, so a bare 'sonnet' here would mean the
        # rule never installed.)
        models = {e['model'] for e in idx_events}
        assert models == {'opus'}, f'model drifted mid-dispatch: {idx_events!r}'
        assert all(e['rule_id'] == 'retry-tier-up' for e in idx_events)
        assert all(e['source_layer'] == 'policy_rule' for e in idx_events)

        # (c) every recorded merge-mode metadata write carrying a 'routing'
        # key shows the tier VALUE unchanged at 1 -- not merely "no write
        # happened". `_invoke`'s mirror writes `routing` on every
        # invocation, so this list is expected to be non-empty.
        routing_writes = [
            call['metadata']['routing'] for call in backend.update_task_calls
            if isinstance(call.get('metadata'), dict) and 'routing' in call['metadata']
        ]
        assert routing_writes, (
            f'expected at least one merge-mode routing mirror write: '
            f'{backend.update_task_calls!r}'
        )
        assert {w['routing_tier'] for w in routing_writes} == {1}, (
            f'a routing mirror write raised routing_tier above 1 mid-dispatch: '
            f'{routing_writes!r}'
        )


# ---------------------------------------------------------------------------
# Boundary scenario 5 -- retry-tier bump on dispatch #2 after a BLOCKED
# dispatch #1 (plan step-4). Also proves the other half of scenario 6's
# guarantee at the harness boundary: a DONE outcome persists no bump.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRetryTierBumpAcrossDispatches:
    """PRD boundary scenario 5: a terminal-failed (BLOCKED) dispatch #1 bumps
    ``metadata.routing.routing_tier`` at the harness boundary
    (``Harness._maybe_bump_routing_tier``, task mu's WRITE side), and
    dispatch #2 -- seeded from that ACTUAL persisted metadata dict, never a
    hand-written ``{'routing_tier': 1}`` literal -- routes one ladder rung
    stronger via the shipped ``retry-tier-up`` rule (task mu's READ side,
    defaults.yaml:320). A hand-written literal in half 2 would re-mock
    exactly the write/read joint this scenario proves is real (plan.json's
    design_decisions).

    Half 1 drives the WRITE side against a real ``Harness``
    (``_workflow_helpers._build_harness``) whose scheduler is wired to a
    ``FakeMetadataBackend`` via ``wire_metadata_backend`` -- so the bump
    exercises genuine ``metadata_mode='merge'`` semantics (harness.py:9044-9048),
    not merely a recorded call, and ``latest``/``history``/``simple_saturated``
    -- seeded as genuinely non-default values -- are proven to ride through
    the merge untouched (harness.py:216 ``model_copy``), alongside a sibling
    ``files`` key proving the merge never clobbers unrelated metadata. Half 2
    drives the READ side through the full rig, using ``code_default_config``
    so the shipped ``retry-tier-up`` rule is guaranteed live regardless of
    whether the ambient operational yaml overrides ``routing`` (mirrors
    ``TestTierStableWithinOneDispatch``'s isolation rationale).

    ``test_done_outcome_persists_no_bump`` covers the other half of PRD
    boundary scenario 6 at the harness (WRITE) boundary --
    ``TestTierStableWithinOneDispatch`` already covers scenario 6's resolver
    (READ) half, that routing_tier is stable within one dispatch.
    """

    @pytest.fixture
    def stock_config(self, git_repo: Path, code_default_config) -> OrchestratorConfig:
        return OrchestratorConfig(
            project_root=git_repo,
            max_concurrent_tasks=1,
            max_execute_iterations=5,
            max_verify_attempts=3,
            max_review_cycles=2,
            git=GitConfig(
                main_branch='main',
                branch_prefix='task/',
                remote='origin',
                worktree_dir='.worktrees',
            ),
            sandbox=SandboxConfig(enabled=False),
        )

    @staticmethod
    def _seed_metadata() -> dict:
        """The dispatch-#1-time metadata: ``routing_tier=0`` plus a
        genuinely non-default ``latest``/``history``/``simple_saturated`` (a
        real prior decision, not ``None``/``[]``/``False``) so their survival
        through the bump is a real assertion, not vacuously true over
        already-empty/default state -- mirrors test_harness_retry_tier.
        TestBumpedRoutingDump.test_preserves_latest_history_and_saturated.
        Also carries a sibling ``files`` key, untouched by the bump's
        ``{'routing': ...}`` payload, to prove the merge write never
        clobbers sibling metadata (harness.py:9044-9046).
        """
        prior_decision = {
            'role': 'implementer', 'model': 'sonnet', 'effort': 'high',
            'budget_usd': 10.0, 'max_turns': 80, 'source_layer': 'config',
            'rule_id': None, 'rejected': [], 'routing_tier': 0, 'decided_at': None,
        }
        return {
            'files': ['lib'],
            'routing': {
                'routing_tier': 0,
                'simple_saturated': True,
                'latest': prior_decision,
                'history': [prior_decision],
            },
        }

    async def test_blocked_dispatch_bumps_tier_and_next_dispatch_routes_one_rung_up(
        self, stock_config, git_ops, task_assignment, monkeypatch,
    ):
        # -- Half 1 (WRITE side): a real Harness whose scheduler is wired to
        # a FakeMetadataBackend, so the bump below exercises genuine
        # metadata_mode='merge' semantics rather than a recorded call.
        harness = _build_harness(stock_config)
        backend = FakeMetadataBackend()
        seed_metadata = self._seed_metadata()
        wire_metadata_backend(harness.scheduler, backend, seed=seed_metadata, grants=True)

        task_assignment.task['metadata'] = self._seed_metadata()
        blocked_report = TaskReport(
            task_id=task_assignment.task_id, title='Routing boundary task',
            outcome=WorkflowOutcome.BLOCKED,
        )

        await harness._maybe_bump_routing_tier(task_assignment, blocked_report)

        assert backend.blob['routing']['routing_tier'] == 1
        # latest/history/simple_saturated ride through the bump's model_copy
        # untouched (harness.py:216) -- only the counter moved.
        assert backend.blob['routing']['simple_saturated'] is True
        assert backend.blob['routing']['latest'] == seed_metadata['routing']['latest']
        assert backend.blob['routing']['history'] == seed_metadata['routing']['history']
        # The merge write never clobbered the sibling `files` key.
        assert backend.blob['files'] == ['lib']

        # -- Half 2 (READ side): dispatch #2 is seeded from the ACTUAL
        # persisted metadata dict backend.blob just produced -- never a
        # hand-written {'routing_tier': 1} literal.
        persisted_metadata = dict(backend.blob)
        fresh_assignment = TaskAssignment(
            task_id='42',
            task={
                'id': '42',
                'title': 'Routing boundary task',
                'description': 'Exercise the real _invoke routing-resolution path',
                'status': 'pending',
                'metadata': persisted_metadata,
                'dependencies': [],
            },
            modules=['lib'],
        )
        assert stock_config.models.implementer == 'sonnet'  # so opus below is a genuine +1 rung

        stub = _RoutingRecorderStub()
        workflow, _scheduler = _build_workflow(stock_config, git_ops, fresh_assignment, stub)
        rec = _RecordingEventStore()
        workflow.event_store = rec
        monkeypatch.setattr('orchestrator.workflow.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification', _mock_verify_passes(),
        )

        outcome = (await workflow.run()).outcome
        assert outcome == WorkflowOutcome.DONE

        assert stub.route_by_role['implementer']['model'] == 'opus'

        impl_events = [e for e in _routing_events(rec) if e['data']['role'] == 'implementer']
        assert impl_events, f'expected an implementer routing_decision event: {rec.events!r}'
        data = impl_events[-1]['data']
        assert data['source_layer'] == 'policy_rule'
        assert data['rule_id'] == 'retry-tier-up'
        assert data['routing_tier'] == 1

    async def test_done_outcome_persists_no_bump(self, stock_config, task_assignment):
        """The other half of scenario 6's guarantee, at the harness
        (WRITE) boundary: a DONE (success) dispatch never bumps the tier in
        the first place -- ``_maybe_bump_routing_tier`` returns before ever
        calling ``scheduler.update_task`` (harness.py:9050)."""
        harness = _build_harness(stock_config)
        backend = FakeMetadataBackend()
        seed_metadata = self._seed_metadata()
        wire_metadata_backend(harness.scheduler, backend, seed=seed_metadata, grants=True)

        task_assignment.task['metadata'] = self._seed_metadata()
        done_report = TaskReport(
            task_id=task_assignment.task_id, title='Routing boundary task',
            outcome=WorkflowOutcome.DONE,
        )

        await harness._maybe_bump_routing_tier(task_assignment, done_report)

        assert backend.update_task_calls == []
        assert backend.blob['routing']['routing_tier'] == 0


# ---------------------------------------------------------------------------
# Boundary scenario 7 -- per-model daily ceiling exhaustion falls back one
# layer WITHOUT blocking the dispatch (plan step-5).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCeilingFallbackDoesNotBlockDispatch:
    """PRD boundary scenario 7: a per-role ``model_overrides`` pin that has
    exhausted its ``routing.per_model_daily_ceiling_usd`` ceiling falls back
    ONE layer (to ``config.models``) WITHOUT blocking the dispatch --
    invariant 6, ceilings bound spend, they never gate dispatch.

    Seeds a REAL ``shared.cost_store.CostStore`` on a tmp runs.db with an
    actual completed 'opus' invocation row via the store's own write API, so
    the ``>= ceiling`` comparison at routing.py:441-443 reads a genuine
    ``model_cost_in_window`` result (cost_store.py:265) rather than a
    stubbed number -- exercising ``TaskWorkflow._ceiling_spend_by_model``'s
    trailing-24h window query (workflow.py:12121, ``_trailing_24h_window``
    at :1140) as one wired path. ``cost_store`` is a plain post-construction
    attribute (workflow.py:1192), attached the same way ``event_store`` is
    attached elsewhere in this suite.

    Uses 'opus' rather than ``claude-fable-5``: fable is deliberately absent
    from the stock ``allowed_models`` until xi admits it, so pinning fable
    would reject on the allowlist FIRST (routing.py check order: allowlist
    -> ceiling -> capacity, routing.py:439-445) and this suite would
    silently assert the wrong rejection mechanism.
    """

    _CEILING_USD = 10.0

    @staticmethod
    async def _seed_opus_spend(cost_store: CostStore, *, total_usd: float) -> None:
        """Write one completed 'opus' invocation inside the trailing-24h
        window totalling *total_usd* -- via the store's own write API, so
        the ceiling comparison reads a real query result rather than a
        stubbed number."""
        now_iso = datetime.now(UTC).isoformat()
        await cost_store.save_invocation(
            run_id='seed-run', task_id='seed-task', project_id='dark_factory',
            account_name='seed-account', model='opus', role='implementer',
            cost_usd=total_usd, input_tokens=None, output_tokens=None,
            cache_read_tokens=None, cache_create_tokens=None, duration_ms=100,
            capped=False, started_at=now_iso, completed_at=now_iso,
        )

    async def test_ceiling_exhausted_falls_back_without_blocking(
        self, config, git_ops, task_assignment, tmp_path, monkeypatch,
    ):
        config.routing.per_model_daily_ceiling_usd = {'opus': self._CEILING_USD}
        assert 'opus' in config.routing.allowed_models
        assert config.models.implementer != 'opus'  # so a config-layer fallback is observable
        task_assignment.task['metadata']['model_overrides'] = {'implementer': 'opus'}

        cost_store = CostStore(tmp_path / 'costs.db')
        await cost_store.open()
        try:
            # >= ceiling: exactly at the boundary routing.py:442 checks.
            await self._seed_opus_spend(cost_store, total_usd=self._CEILING_USD)

            stub = _RoutingRecorderStub()
            workflow, _scheduler = _build_workflow(config, git_ops, task_assignment, stub)
            workflow.cost_store = cost_store
            rec = _RecordingEventStore()
            workflow.event_store = rec
            monkeypatch.setattr('orchestrator.workflow.invoke_agent', stub.invoke_agent)
            monkeypatch.setattr(
                'orchestrator.workflow.run_scoped_verification', _mock_verify_passes(),
            )

            outcome = (await workflow.run()).outcome

            # (a) the seam did NOT see 'opus' -- it fell back to the config layer.
            assert stub.route_by_role['implementer']['model'] == config.models.implementer
            assert stub.route_by_role['implementer']['model'] != 'opus'

            # (b) the exact rejection reason and the layer fallen through to.
            impl_events = [e for e in _routing_events(rec) if e['data']['role'] == 'implementer']
            assert impl_events, f'expected an implementer routing_decision event: {rec.events!r}'
            assert impl_events[-1]['data']['rejected'] == [
                'metadata_override:model-ceiling-exhausted',
            ]
            assert impl_events[-1]['data']['source_layer'] == 'config'

            # (c) invariant 6: a per-model ceiling never blocks the dispatch.
            assert outcome == WorkflowOutcome.DONE
        finally:
            await cost_store.close()

    async def test_below_ceiling_control_resolves_the_override(
        self, config, git_ops, task_assignment, tmp_path, monkeypatch,
    ):
        """(d) control: with the seeded spend BELOW the ceiling, the SAME
        override resolves to 'opus' with ``source_layer=='metadata_override'``
        and empty ``rejected`` -- proving the fallback above is caused by the
        ceiling specifically, not an unrelated rejection."""
        config.routing.per_model_daily_ceiling_usd = {'opus': self._CEILING_USD}
        task_assignment.task['metadata']['model_overrides'] = {'implementer': 'opus'}

        cost_store = CostStore(tmp_path / 'costs.db')
        await cost_store.open()
        try:
            await self._seed_opus_spend(cost_store, total_usd=self._CEILING_USD - 5.0)

            stub = _RoutingRecorderStub()
            workflow, _scheduler = _build_workflow(config, git_ops, task_assignment, stub)
            workflow.cost_store = cost_store
            rec = _RecordingEventStore()
            workflow.event_store = rec
            monkeypatch.setattr('orchestrator.workflow.invoke_agent', stub.invoke_agent)
            monkeypatch.setattr(
                'orchestrator.workflow.run_scoped_verification', _mock_verify_passes(),
            )

            outcome = (await workflow.run()).outcome

            assert stub.route_by_role['implementer']['model'] == 'opus'
            impl_events = [e for e in _routing_events(rec) if e['data']['role'] == 'implementer']
            assert impl_events, f'expected an implementer routing_decision event: {rec.events!r}'
            assert impl_events[-1]['data']['source_layer'] == 'metadata_override'
            assert impl_events[-1]['data']['rejected'] == []
            assert outcome == WorkflowOutcome.DONE
        finally:
            await cost_store.close()
