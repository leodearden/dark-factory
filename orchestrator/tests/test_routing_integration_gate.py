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
import copy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml
from _orch_helpers import pydantic_spec, stamp_stock_routing_config
from _recording_event_store import _RecordingEventStore
from _workflow_helpers import (
    AgentStub,
    FakeMetadataBackend,
    FakeScheduler,
    _build_harness,
    _build_workflow,
    _derive_meta_root_like_production,  # noqa: F401  autouse fixture, see its docstring
    _init_repo,
    wire_metadata_backend,
)
from escalation.models import Escalation
from pydantic import ValidationError
from shared.cost_store import CostStore

from orchestrator.agents.invoke import AgentResult
from orchestrator.agents.roles import ARCHITECT, DEBUGGER, IMPLEMENTER, SIMPLE_TASK
from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import (
    GitConfig,
    OrchestratorConfig,
    SandboxConfig,
    apply_reload,
    load_config,
)
from orchestrator.digest import (
    CostStats,
    DigestInputs,
    EscalationStats,
    model_role_rollup,
    render_digest_markdown,
)
from orchestrator.event_store import EventStore, EventType
from orchestrator.git_ops import GitOps
from orchestrator.harness import TaskReport
from orchestrator.review_checkpoint import ReviewCheckpoint
from orchestrator.run_store import RunStore
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
def stock_config(git_repo: Path, code_default_config) -> OrchestratorConfig:
    """Module-level counterpart to ``config`` above, pinned to the bundled
    ``defaults.yaml`` via the opt-in ``code_default_config`` conftest
    fixture (see module docstring's CONFIG section) rather than the autouse
    ``_isolate_orch_config`` -- shared by every scenario whose claim is
    specifically about CODE DEFAULTS (byte-equivalence, tier/saturation rule
    liveness, reload/reject starting state, digest rollup). Classes that
    need a variant override this fixture by the same name (the standard
    pytest fixture-override pattern) rather than redefining it wholesale --
    see ``TestByteEquivalenceThroughTheRig`` below.
    """
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

    def _detect_role(self, system_prompt: str) -> str:
        """Extend ``AgentStub._detect_role`` with a SIMPLE_TASK case.

        The shared helper (``_workflow_helpers.py``, imported never edited —
        see module docstring) has no branch for ``SIMPLE_TASK.system_prompt``
        ("You are a SIMPLE_TASK agent...") — every existing substring check
        (architect/implementer/debugger/reviewer/merger/judge) is false for
        it, so it falls through to ``'unknown'``. Scenario 4 (plan step-7)
        is the first in this suite to need role='simple_task' keyed
        recording, so the case is added HERE, in σ's own module-local
        subclass, rather than touching the shared stub.
        """
        if 'SIMPLE_TASK agent' in system_prompt:
            return 'simple_task'
        return super()._detect_role(system_prompt)


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


def _make_rig(
    config: OrchestratorConfig,
    git_ops: GitOps,
    assignment: TaskAssignment,
    stub: AgentStub,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fake_verify: bool = True,
) -> tuple[TaskWorkflow, FakeScheduler, _RecordingEventStore]:
    """The rig-setup preamble shared by most scenarios below: build a
    ``TaskWorkflow``/``FakeScheduler`` via ``_build_workflow``, attach a
    fresh ``_RecordingEventStore``, and inject *stub* at the real seam
    (``monkeypatch.setattr('orchestrator.workflow.invoke_agent', ...)`` --
    see module docstring's SEAM section).

    ``fake_verify=True`` (the default) additionally monkeypatches
    ``run_scoped_verification`` to :func:`_mock_verify_passes`, for
    scenarios that drive a full ``workflow.run()`` to DONE. Pass
    ``fake_verify=False`` for scenarios that drive ``_invoke`` (or another
    internal method) directly and never reach the VERIFY phase, or that
    need to install their own bespoke verification double afterward --
    callers remain free to add further ``monkeypatch.setattr`` calls, seed
    ``workflow.plan``/``.modules``/``.cost_store``, or wire a
    ``FakeMetadataBackend`` onto the returned ``scheduler`` after calling
    this.
    """
    workflow, scheduler = _build_workflow(config, git_ops, assignment, stub)
    rec = _RecordingEventStore()
    workflow.event_store = rec  # type: ignore[assignment]
    monkeypatch.setattr('orchestrator.workflow.invoke_agent', stub.invoke_agent)
    if fake_verify:
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification', _mock_verify_passes(),
        )
    return workflow, scheduler, rec


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
        workflow, _scheduler, rec = _make_rig(config, git_ops, task_assignment, stub, monkeypatch)

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
        workflow, _scheduler, rec = _make_rig(config, git_ops, task_assignment, stub, monkeypatch)

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
    def stock_config(self, stock_config: OrchestratorConfig) -> OrchestratorConfig:
        """Override (standard pytest fixture-override pattern, same name
        requesting the outer module-level fixture) rather than a wholesale
        redefinition: derive this class's variant from the shared stock
        config via ``model_copy`` so the two never drift apart on the
        fields they share.

        Opt-in (default False): brings 'judge' into the in-band role set
        this single dispatch invokes, alongside architect/implementer/
        reviewer*/merger. Does not change dispatch control flow here -- the
        EXECUTE loop's own pending-steps check already terminates naturally
        once the implementer's one invocation completes both PLAN steps
        (workflow.py:7920), independent of the judge's verdict -- it only
        adds one extra judge invocation to observe.
        """
        return stock_config.model_copy(update={'judge_after_each_iteration': True})

    async def test_stock_dispatch_matches_config_byte_for_byte(
        self, stock_config, git_ops, task_assignment, monkeypatch,
    ):
        stub = _RoutingRecorderStub()
        workflow, _scheduler, rec = _make_rig(
            stock_config, git_ops, task_assignment, stub, monkeypatch,
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
        workflow, _scheduler, rec = _make_rig(  # type: ignore[arg-type]
            stock_config, git_ops, task_assignment, stub, monkeypatch, fake_verify=False,
        )

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
        workflow, _scheduler, rec = _make_rig(
            stock_config, git_ops, task_assignment, stub, monkeypatch, fake_verify=False,
        )
        _seed_workflow_artifacts(workflow, tmp_path=tmp_path)
        workflow.git_ops.rebase_onto_main = AsyncMock()  # type: ignore[method-assign]
        workflow._submit_to_merge_queue = AsyncMock(  # type: ignore[method-assign]
            return_value=WorkflowOutcome.DONE,
        )
        workflow._mark_blocked = AsyncMock(  # type: ignore[method-assign]
            return_value=WorkflowOutcome.BLOCKED,
        )
        workflow._write_merge_failure_review = MagicMock()  # type: ignore[method-assign]

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

    # stock_config: uses the module-level fixture (declared next to `config`
    # above) unmodified -- see that fixture's docstring for why it is shared
    # rather than redefined per class.

    async def test_routing_tier_stable_across_verify_debug_iterations(
        self, stock_config, git_ops, task_assignment, monkeypatch,
    ):
        assert stock_config.models.implementer == stock_config.models.debugger == 'sonnet'
        task_assignment.task['metadata']['routing'] = {'routing_tier': 1}

        stub = _RoutingRecorderStub()
        workflow, scheduler, rec = _make_rig(
            stock_config, git_ops, task_assignment, stub, monkeypatch, fake_verify=False,
        )
        backend = FakeMetadataBackend()
        wire_metadata_backend(
            scheduler, backend, seed=task_assignment.task['metadata'], grants=True,  # type: ignore[arg-type]
        )

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

    # stock_config: uses the module-level fixture (declared next to `config`
    # above) unmodified -- see that fixture's docstring for why it is shared
    # rather than redefined per class.

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
        wire_metadata_backend(harness.scheduler, backend, seed=seed_metadata, grants=True)  # type: ignore[arg-type]

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
        workflow, _scheduler, rec = _make_rig(
            stock_config, git_ops, fresh_assignment, stub, monkeypatch,
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
        wire_metadata_backend(harness.scheduler, backend, seed=seed_metadata, grants=True)  # type: ignore[arg-type]

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
            workflow, _scheduler, rec = _make_rig(
                config, git_ops, task_assignment, stub, monkeypatch,
            )
            workflow.cost_store = cost_store

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
            workflow, _scheduler, rec = _make_rig(
                config, git_ops, task_assignment, stub, monkeypatch,
            )
            workflow.cost_store = cost_store

            outcome = (await workflow.run()).outcome

            assert stub.route_by_role['implementer']['model'] == 'opus'
            impl_events = [e for e in _routing_events(rec) if e['data']['role'] == 'implementer']
            assert impl_events, f'expected an implementer routing_decision event: {rec.events!r}'
            assert impl_events[-1]['data']['source_layer'] == 'metadata_override'
            assert impl_events[-1]['data']['rejected'] == []
            assert outcome == WorkflowOutcome.DONE
        finally:
            await cost_store.close()


# ---------------------------------------------------------------------------
# Boundary scenario 8 -- simple_task saturation stamp routes the next
# dispatch full-path (plan step-6).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSaturationStampRoutesNextDispatchFullPath:
    """PRD boundary scenario 8: once a SIMPLE_TASK dispatch demonstrably
    exhausts its turn cap, ``_stamp_simple_saturated`` (task nu,
    workflow.py:5157) retires the 'simple' label, and the NEXT dispatch --
    seeded from that ACTUAL persisted metadata, never a hand-written
    ``{'simple_saturated': True}`` literal -- takes the full architect path,
    whose ``routing_decision`` event names the attribution-only
    ``simple-saturated-full-path`` rule (defaults.yaml:294). That rule is
    byte-equivalent (the architect's config model is already opus), so the
    discriminating assertion below is ``rule_id``, not the model.

    Half 1 drives ``_run_simple_task()`` directly against a real
    ``TaskWorkflow`` -- this method's OWN internal gating (``if not
    result.success: ...``) is what is under test, not the outer
    ``_should_run_simple_task()`` dispatch gate -- with
    ``orchestrator.workflow.invoke_agent`` faked to return the exact
    ``AgentResult`` shape (``subtype='error_max_turns'``) that
    ``classify_agent_failure`` maps to ``AgentFailureKind.MAX_TURNS``,
    matching the real classifier's input contract rather than monkeypatching
    the classifier itself. The scheduler is wired to a real
    ``FakeMetadataBackend`` (``wire_metadata_backend``) so the stamp's
    ``metadata_mode='merge'`` write genuinely persists. Half 2 feeds that
    ACTUAL persisted metadata dict into a fresh dispatch.

    Uses ``code_default_config`` so the shipped ``simple-saturated-full-path``
    rule (and ``simple_task_enabled``) are guaranteed live regardless of
    whether the ambient operational yaml drifts (mirrors
    ``TestTierStableWithinOneDispatch``'s isolation rationale).
    """

    # stock_config: uses the module-level fixture (declared next to `config`
    # above) unmodified -- see that fixture's docstring for why it is shared
    # rather than redefined per class.

    @staticmethod
    async def _max_turns_invoke(**kwargs) -> AgentResult:
        """The exact ``AgentResult`` shape ``classify_agent_failure`` maps to
        ``AgentFailureKind.MAX_TURNS``: a failed, non-timed-out,
        non-backgrounded, no-api-error-status result whose ``subtype`` is
        ``'error_max_turns'`` (shared/cli_invoke.py:1167)."""
        return AgentResult(
            success=False, output='', subtype='error_max_turns',
            turns=80, output_tokens=500,
        )

    async def test_saturation_stamp_then_next_dispatch_routes_full_path(
        self, stock_config, git_ops, task_assignment, tmp_path, monkeypatch,
    ):
        assert stock_config.simple_task_enabled is True
        task_assignment.task['metadata']['complexity'] = 'simple'

        workflow, scheduler = _build_workflow(
            stock_config, git_ops, task_assignment, AgentStub(),
        )
        _seed_workflow_artifacts(workflow, tmp_path=tmp_path)
        backend = FakeMetadataBackend()
        wire_metadata_backend(
            scheduler, backend, seed=task_assignment.task['metadata'], grants=True,  # type: ignore[arg-type]
        )
        monkeypatch.setattr('orchestrator.workflow.invoke_agent', self._max_turns_invoke)

        # Premise: a clean, un-saturated, author-declared simple task takes
        # the SIMPLE_TASK path.
        assert workflow._should_run_simple_task() is True

        # -- Half 1 (WRITE side): a demonstrated turn-cap exhaustion stamps
        # simple_saturated=True through a genuine scheduler merge write AND
        # the in-memory task dict.
        outcome_1 = await workflow._run_simple_task()
        assert outcome_1 == WorkflowOutcome.REQUEUED
        assert backend.blob['routing']['simple_saturated'] is True
        assert workflow.task['metadata']['routing']['simple_saturated'] is True

        # Two writes landed in round 1: _invoke's routing-decision mirror
        # (fires first, so it still reads simple_saturated=False -- nothing
        # has stamped yet) then the stamp write itself (flips False->True;
        # _stamp_simple_saturated carries `latest`/`history` forward
        # unchanged, unlike the mirror's own with_decision -- task_metadata.
        # RoutingState). Asserted on each write's OWN identity (its
        # simple_saturated value), not a bare count: a count-only assertion
        # would also pass in the degenerate regression where the mirror
        # write silently stopped firing and only the stamp write landed --
        # asserting the FIRST call still shows False catches that.
        assert len(backend.update_task_calls) == 2, (
            f'expected exactly 2 writes in round 1 (mirror + stamp): '
            f'{backend.update_task_calls!r}'
        )
        round_1_mirror_write, round_1_stamp_write = backend.update_task_calls
        assert round_1_mirror_write['metadata']['routing']['simple_saturated'] is False, (
            f"_invoke's mirror write in round 1 fires BEFORE the stamp -- it "
            f'must still see simple_saturated=False: {round_1_mirror_write!r}'
        )
        assert round_1_stamp_write['metadata']['routing']['simple_saturated'] is True, (
            f'the stamp write itself must flip simple_saturated to True: '
            f'{round_1_stamp_write!r}'
        )

        # Idempotence: driving a second demonstrated exhaustion sees ONE
        # effective state (simple_saturated stays True, not re-toggled).
        # _stamp_simple_saturated's own early return (already True) skips
        # its write this time, so exactly one NEW write lands -- _invoke's
        # routing-decision mirror -- and it must still carry
        # simple_saturated=True forward. This is the assertion that
        # actually pins "the mirror keeps firing every round": a bare
        # count-delta (as before) would accept round 2 contributing ZERO
        # writes just as readily as one, provided round 1 had also
        # regressed to one write -- 0 == 1 - 1 is a false positive for the
        # same regression the round-1 checks above already guard against.
        outcome_2 = await workflow._run_simple_task()
        assert outcome_2 == WorkflowOutcome.REQUEUED
        assert backend.blob['routing']['simple_saturated'] is True
        assert workflow.task['metadata']['routing']['simple_saturated'] is True
        new_calls_in_round_2 = backend.update_task_calls[2:]
        assert len(new_calls_in_round_2) == 1, (
            f'expected exactly 1 NEW write in round 2 (mirror only, stamp '
            f'skipped by its own early return): {new_calls_in_round_2!r}'
        )
        round_2_mirror_write = new_calls_in_round_2[0]
        assert round_2_mirror_write['metadata']['routing']['simple_saturated'] is True, (
            f"_invoke's mirror write in round 2 must still fire and carry "
            f'simple_saturated=True forward: {round_2_mirror_write!r}'
        )

        # -- Half 2 (READ side): dispatch #2 is seeded from the ACTUAL
        # persisted metadata dict backend.blob just produced -- never a
        # hand-written {'simple_saturated': True} literal.
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

        recorder = _RoutingRecorderStub()
        workflow2, _scheduler2, rec = _make_rig(
            stock_config, git_ops, fresh_assignment, recorder, monkeypatch,
        )

        # (a) the saturated label is retired -- the full PLAN path runs, not
        # SIMPLE_TASK.
        assert workflow2._should_run_simple_task() is False

        outcome = (await workflow2.run()).outcome
        assert outcome == WorkflowOutcome.DONE
        assert 'architect' in recorder.route_by_role
        assert 'simple_task' not in recorder.route_by_role

        # (b) the architect's routing_decision event names the saturation
        # attribution rule.
        arch_events = [e for e in _routing_events(rec) if e['data']['role'] == 'architect']
        assert arch_events, f'expected an architect routing_decision event: {rec.events!r}'
        assert arch_events[-1]['data']['rule_id'] == 'simple-saturated-full-path'
        assert arch_events[-1]['data']['source_layer'] == 'policy_rule'


# ---------------------------------------------------------------------------
# Boundary scenario 4 -- models.simple_task is config-visible and flips via
# hot reload (plan step-7).
# ---------------------------------------------------------------------------

# Pinned test-only literals. The starting model is deliberately distinct from
# BOTH ModelsConfig.simple_task's own default ('sonnet') AND the reload
# target below ('haiku') -- so a field that were silently dataclass-only
# (yaml ignored) would surface as a loud assertion failure (config.models.
# simple_task would read back as 'sonnet', not 'opus'), not an accidental
# pass. The budget/max_turns pins are likewise distinct from their submodel
# defaults (2.50 / 50) so part (d)'s guard is checking a real value, not a
# default that would pass whether or not the yaml was honored.
_PINNED_SIMPLE_TASK_STARTING_MODEL = 'opus'
_PINNED_SIMPLE_TASK_RELOAD_MODEL = 'haiku'
_PINNED_SIMPLE_TASK_BUDGET = 3.75
_PINNED_SIMPLE_TASK_MAX_TURNS = 42


def _write_simple_task_orchestrator_yaml(path: Path, simple_task_model: str) -> None:
    """Write a minimal, self-contained orchestrator.yaml at *path*, pinning
    ``models.simple_task`` to *simple_task_model* plus the sibling
    ``budgets.simple_task`` / ``max_turns.simple_task`` submodel leaves.

    Deliberately leaves the DEPRECATED top-level ``simple_task_budget_usd`` /
    ``simple_task_max_turns`` scalars unset, so they load at their own
    pydantic defaults (1.50 / 30) and ``_honor_deprecated_simple_task_scalars``
    (config.py:4527) takes its no-op branch (``raw != raw_default`` is
    False) on every ``load_config()``/``apply_reload`` call below -- the
    explicit submodel pins are never at risk of a silent scalar-migration
    overwrite. ``project_root`` is left unset (defaults to ``Path('.')``,
    config.py:4338) since this scenario drives ``TaskWorkflow._invoke``
    directly (never ``.run()``), which never reads it -- mirrors
    ``test_config_reload_integration_gate._make_reload_harness``'s identical
    project-root-agnostic minimal yaml. Reimplemented locally rather than
    importing that file's ``_edit_yaml`` -- see this module's FIXTURES ARE
    MODULE-LOCAL docstring section.
    """
    path.write_text(yaml.safe_dump({
        'models': {'simple_task': simple_task_model},
        'budgets': {'simple_task': _PINNED_SIMPLE_TASK_BUDGET},
        'max_turns': {'simple_task': _PINNED_SIMPLE_TASK_MAX_TURNS},
        'max_concurrent_tasks': 1,
        'git': {
            'main_branch': 'main',
            'branch_prefix': 'task/',
            'remote': 'origin',
            'worktree_dir': '.worktrees',
        },
        'sandbox': {'enabled': False},
    }))


class TestSimpleTaskModelReloadReachesTheSeam:
    """PRD boundary scenario 4: ``models.simple_task`` is config-visible (not
    dataclass-only, invariant 4) and a hot reload of it reaches the NEXT
    invocation's real CLI seam.

    Modelled on ``test_config_reload_integration_gate.py``: a real on-disk
    yaml + real ``load_config()`` + real ``apply_reload()`` (config.py:5373),
    never a mocked loader. Part (c) drives the reloaded config through a
    DIRECT ``TaskWorkflow._invoke(...)`` call -- mirroring that file's
    ``_make_reload_workflow`` / ``workflow._invoke(IMPLEMENTER, 'do the
    task', tmp_path)`` pattern -- rather than the full ``run()`` state
    machine: this scenario's claim is narrowly "the reloaded leaf reaches
    the seam", which needs no plan/worktree machinery (``_invoke`` never
    reads ``self.artifacts``/``self.git_ops`` when ``self.artifacts is
    None`` and ``self.config.sandbox.enabled`` is False).
    """

    @pytest.mark.asyncio
    async def test_simple_task_model_reload_reaches_the_seam(
        self, tmp_path: Path, task_assignment: TaskAssignment, monkeypatch,
    ) -> None:
        config_path = tmp_path / 'orchestrator.yaml'
        _write_simple_task_orchestrator_yaml(
            config_path, _PINNED_SIMPLE_TASK_STARTING_MODEL,
        )
        monkeypatch.setenv('ORCH_CONFIG_PATH', str(config_path))

        # (a) config-visible, not dataclass-only: the pinned starting value
        # round-trips through a real load_config() off the on-disk yaml.
        live = load_config()
        assert live.models.simple_task == _PINNED_SIMPLE_TASK_STARTING_MODEL

        # (b) hot reload: rewrite the leaf on disk, load a second config
        # object off the SAME path, and apply the real diff/apply pipeline.
        _write_simple_task_orchestrator_yaml(
            config_path, _PINNED_SIMPLE_TASK_RELOAD_MODEL,
        )
        fresh = load_config()
        report = apply_reload(live, fresh)

        assert report['reloaded'] is True
        assert report['error'] is None
        assert 'models.simple_task' in report['applied']
        assert report['applied']['models.simple_task'] == {
            'old': _PINNED_SIMPLE_TASK_STARTING_MODEL,
            'new': _PINNED_SIMPLE_TASK_RELOAD_MODEL,
        }
        # apply_reload mutates `live` in place (I3 identity preservation) --
        # `live` IS the reloaded config object the rig below is built on.
        assert live.models.simple_task == _PINNED_SIMPLE_TASK_RELOAD_MODEL

        # (d) the deprecated scalars were never set (both loads used them at
        # their own pydantic defaults), so _honor_deprecated_simple_task_scalars
        # took its no-op branch on every validation pass -- the explicit
        # submodel pins from the yaml survive untouched.
        assert live.budgets.simple_task == _PINNED_SIMPLE_TASK_BUDGET
        assert live.max_turns.simple_task == _PINNED_SIMPLE_TASK_MAX_TURNS

        # (c) the reload reaches the NEXT invocation's real CLI seam.
        recorder = _RoutingRecorderStub()
        workflow, _scheduler, rec = _make_rig(
            live, GitOps(live.git, tmp_path), task_assignment, recorder, monkeypatch,
            fake_verify=False,
        )

        await workflow._invoke(SIMPLE_TASK, 'simple task prompt', tmp_path)

        assert recorder.route_by_role['simple_task']['model'] == (
            _PINNED_SIMPLE_TASK_RELOAD_MODEL
        )
        st_events = [e for e in _routing_events(rec) if e['data']['role'] == 'simple_task']
        assert st_events, f'expected a simple_task routing_decision event: {rec.events!r}'
        assert st_events[-1]['data']['model'] == _PINNED_SIMPLE_TASK_RELOAD_MODEL
        assert st_events[-1]['data']['source_layer'] == 'config'


# ---------------------------------------------------------------------------
# Boundary scenario 11 -- an unknown policy-rule key is rejected, the live
# config's prior rules survive untouched, AND still govern a real dispatch
# afterward (plan step-8).
# ---------------------------------------------------------------------------


class TestUnknownRuleKeyRejectedPriorRulesStillRoute:
    """PRD boundary scenario 11: an operator yaml edit whose policy rule
    carries an unknown ``match``/``set`` key is rejected, the live config's
    prior rules survive untouched (the rollback half), and -- the
    integration half that distinguishes this from the existing unit test
    (``test_routing_reload.py``:112-160) -- those prior rules did not
    merely survive in memory: they still GOVERN a real dispatch through the
    rig afterward.

    MECHANISM NOTE: ``RuleMatch``/``RuleSet``'s ``extra='forbid'``
    (config.py:380/406) rejects the unknown key at ``OrchestratorConfig``
    CONSTRUCTION time -- i.e. inside ``load_config()`` while building the
    candidate "fresh" config -- raising ``pydantic.ValidationError`` naming
    the key, BEFORE ``apply_reload`` is ever reached. Verified directly
    against ``test_routing_reload.py``'s existing, merged
    ``TestUnknownRuleKeyFailsClosedBeforeApply`` class, whose own docstring
    states this precisely: "an invalid fresh config...raises
    pydantic.ValidationError naming the key at OrchestratorConfig
    construction time, BEFORE apply_reload is ever reached". This class's
    mechanism mirrors that exactly -- it is NOT ``apply_reload``'s separate
    hybrid-invariant rollback path (config.py:5394-5413), which only fires
    when a fresh config that IS individually valid combines with ``live``
    into an invalid CROSS-FIELD hybrid (e.g. the steward-timeout
    invariant); an ``extra='forbid'`` violation can never produce an
    individually-valid ``fresh`` to reach that path in the first place, so
    ``apply_reload`` is never called below -- there is nothing valid for it
    to reload.
    """

    # stock_config: uses the module-level fixture (declared next to `config`
    # above) unmodified -- see that fixture's docstring for why it is shared
    # rather than redefined per class.

    @pytest.mark.asyncio
    async def test_unknown_match_key_rejected_live_unchanged_and_still_routes(
        self, stock_config, git_ops, task_assignment, git_repo, tmp_path, monkeypatch,
    ):
        # Premise: the shipped large-plan-steps rule (defaults.yaml:257) is
        # live under code_default_config -- the "known-good rule set...that
        # demonstrably fires" this scenario needs.
        assert any(r.id == 'large-plan-steps' for r in stock_config.routing.rules)

        ladder_before = list(stock_config.routing.ladder)
        allowed_models_before = list(stock_config.routing.allowed_models)
        ceilings_before = dict(stock_config.routing.per_model_daily_ceiling_usd)

        bad_yaml = tmp_path / 'bad-match.yaml'
        bad_yaml.write_text(yaml.safe_dump({
            'routing': {
                'rules': [
                    {'id': 'bad-rule', 'match': {'plan_vibes': 3}, 'set': {'model': 'opus'}},
                ],
            },
        }))

        # (a) the reload FAILS: fresh construction itself raises, naming the
        # unknown key (see class docstring's MECHANISM NOTE). Byte-identical
        # to test_routing_reload.TestUnknownRuleKeyFailsClosedBeforeApply.
        # test_unknown_match_key_raises_naming_the_key -- kept here (rather
        # than assumed) only because it is the necessary precondition for
        # part (c) below: there is nothing to prove "prior rules still
        # route" without first proving the bad reload actually failed.
        with pytest.raises(ValidationError, match='plan_vibes'):
            load_config(bad_yaml)

        # (b) live is unchanged on the fields this test doesn't hand off to
        # test_routing_reload.py: no partial mutation of ladder/
        # allowed_models/ceilings leaked in from the rejected candidate.
        # Rules-identity survival (`live.routing.rules == rules_before`) is
        # already covered by test_routing_reload.
        # TestUnknownRuleKeyFailsClosedBeforeApply.
        # test_prior_live_rules_untouched_when_fresh_construction_raises --
        # not re-asserted here.
        assert stock_config.routing.ladder == ladder_before
        assert stock_config.routing.allowed_models == allowed_models_before
        assert stock_config.routing.per_model_daily_ceiling_usd == ceilings_before

        # (c) the integration half: the prior rules did not merely survive
        # in memory -- they still GOVERN a real dispatch afterward. Mirrors
        # TestByteEquivalenceThroughTheRig's rust-heuristic-parity idiom:
        # seed workflow.plan/modules directly and drive _invoke, since only
        # the resolved route matters here.
        stub = _MinimalRouteRecorder()
        workflow, _scheduler, rec = _make_rig(  # type: ignore[arg-type]
            stock_config, git_ops, task_assignment, stub, monkeypatch, fake_verify=False,
        )

        workflow.plan = {'steps': [{'id': f's{i}', 'status': 'pending'} for i in range(12)]}
        workflow.modules = ['lib']

        await workflow._invoke(IMPLEMENTER, 'impl prompt', git_repo)

        assert stub.calls[-1]['model'] == 'opus'
        impl_data = [
            e for e in _routing_events(rec) if e['data']['role'] == 'implementer'
        ][-1]['data']
        assert impl_data['source_layer'] == 'policy_rule'
        assert impl_data['rule_id'] == 'large-plan-steps'

    # A symmetric "unknown set: key" sub-test is deliberately NOT duplicated
    # here: RuleSet's own extra='forbid' (config.py:406) is the identical
    # construction-time mechanism as RuleMatch's above, and is already
    # covered end to end by test_routing_reload.
    # TestUnknownRuleKeyFailsClosedBeforeApply.
    # test_unknown_set_key_raises_naming_the_key -- post-rejection routing
    # behavior does not depend on which closed-vocabulary class caught the
    # typo, so re-proving (a)-(c) for it here would add no new coverage.


# ---------------------------------------------------------------------------
# Boundary scenario 9 -- out-of-band site parity: steward + deep_reviewer
# each resolve through the REAL resolve_and_record_route seam and each emit
# a routing_decision event exactly once per invocation (plan step-9).
# ---------------------------------------------------------------------------


def _mk_escalation(**overrides) -> Escalation:
    """Minimal ``Escalation`` for driving ``TaskSteward._invoke_with_session``
    directly. Mirrors ``test_out_of_band_routing._mk_escalation`` -- kept
    module-local per this suite's fixtures-stay-module-local discipline (see
    module docstring)."""
    defaults: dict = dict(
        id='esc-42-1', task_id='42', agent_role='orchestrator',
        severity='blocking', category='limit_exhausted', summary='s',
    )
    defaults.update(overrides)
    return Escalation(**defaults)  # type: ignore[arg-type]


# Deliberately OUTSIDE pytest's tmp_path -- NOT a sandbox escape.
# ReviewCheckpoint._run_review (review_checkpoint.py:150-155) raises ValueError
# on any project_root containing '/tmp/pytest', ahead of every seam this suite
# patches. Mirrors test_out_of_band_routing.py's identical workaround (module-
# local per this suite's discipline, so reimplemented rather than imported).
_REVIEW_PROJECT_ROOT = Path('/tmp/dark-factory-review-nonpytest-root')


@pytest.mark.asyncio
class TestOutOfBandParity:
    """PRD boundary scenario 9: the out-of-band dispatch sites (steward,
    deep_reviewer) resolve through the REAL ``resolve_and_record_route`` seam
    -- driven end to end via ``TaskSteward._invoke_with_session`` /
    ``ReviewCheckpoint._run_review`` with ONLY ``invoke_agent`` faked, never
    by patching ``resolve_and_record_route`` itself (that shallower mock is
    exactly what ``test_out_of_band_routing.py`` uses to prove the resolver
    call SHAPE; this class proves the resolved value additionally survives
    the REAL ``invoke_with_cap_retry`` cap-retry loop down to the CLI argv --
    the same one-level-deeper seam discipline this suite uses everywhere else
    for ``TaskWorkflow._invoke``/``orchestrator.workflow.invoke_agent``).

    Steward is genuinely per-task (``task_id``/``task_metadata`` both flow
    into the resolver via ``_invoke_with_session``, steward.py:786-798), so
    its override reaches the ``metadata_override`` layer. deep_reviewer is
    STRUCTURALLY project-level -- ``_run_review`` passes NO task_id /
    task_metadata / scheduler / in_memory_task at all (review_checkpoint.py:
    184-208) -- so ``metadata_override`` is provably UNREACHABLE there,
    confirmed both by that comment and independently by
    ``TestDeepReviewerAdoptsResolveRoute``'s docstring in
    ``test_out_of_band_routing.py`` ("The metadata_override layer is
    unreachable here"). Its half therefore proves seam-parity via the one
    override channel it DOES have -- a config-layer value distinct from the
    role default -- while explicitly asserting the metadata-mirror absence
    its narrower contract implies (routing_dispatch.py:262-283).
    """

    async def test_steward_override_honored_and_emits_once(
        self, make_steward, monkeypatch,
    ) -> None:
        rec = _RecordingEventStore()
        task = {
            'id': '42', 'title': 't', 'description': 'd',
            'metadata': {'model_overrides': {'steward': 'haiku', 'triage': 'haiku'}},
        }
        # make_steward's config stamps models.steward = 'opus' (conftest.py)
        # -- distinct from 'haiku', so the layers genuinely disagree, same
        # spirit as TestOverrideWinsAtTheSeam.
        steward = make_steward(task=task, event_store=rec)
        stub = _MinimalRouteRecorder()
        monkeypatch.setattr('orchestrator.steward.invoke_agent', stub.invoke_agent)

        await steward._invoke_with_session(
            prompt='p', cwd=steward.worktree, mcp_config={'mcpServers': {}},
            per_invocation_budget=3.0, escalation=_mk_escalation(),
        )

        # The override reached the CLI seam, not just the resolver.
        assert stub.calls[-1]['model'] == 'haiku'

        entries = _routing_events(rec)
        assert len(entries) == 1, (
            f'expected exactly one routing_decision event: {rec.events!r}'
        )
        data = entries[0]['data']
        assert data['role'] == 'steward'
        assert data['source_layer'] == 'metadata_override'
        # The steward's enforced per-invocation cap is surfaced alongside the
        # resolver's advisory budget_usd, flagged advisory-only
        # (routing_dispatch.py:247-254).
        assert data['applied_budget_usd'] == pytest.approx(3.0)
        assert data['budget_usd_advisory'] is True

    @staticmethod
    def _deep_reviewer_config() -> OrchestratorConfig:
        """Mirrors ``test_out_of_band_routing._review_config``, reimplemented
        module-locally (fixtures-stay-module-local discipline -- see module
        docstring) with ``models.deep_reviewer`` set to a value DISTINCT from
        the role default ('haiku', not the RoleDefaults-supplied 'opus') so
        the seam assertion is meaningful: it proves the resolved CONFIG value
        reaches ``invoke_agent`` rather than merely coinciding with a
        default."""
        cfg = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
        cfg.project_root = _REVIEW_PROJECT_ROOT
        cfg.fused_memory.project_id = 'dark_factory'
        cfg.models.deep_reviewer = 'haiku'
        cfg.budgets.deep_reviewer = 10.0
        cfg.max_turns.deep_reviewer = 50
        cfg.effort.deep_reviewer = 'max'
        cfg.backends.deep_reviewer = 'claude'
        cfg.escalation.host = 'localhost'
        cfg.escalation.port = 8102
        stamp_stock_routing_config(cfg)
        return cfg

    async def test_deep_reviewer_resolves_and_emits_once_no_metadata_mirror(
        self, monkeypatch,
    ) -> None:
        rec = _RecordingEventStore()
        mcp = MagicMock()
        mcp.mcp_config_json.return_value = {'mcpServers': {}}
        cp = ReviewCheckpoint(self._deep_reviewer_config(), mcp, None)
        cp.event_store = rec  # type: ignore[assignment]

        stub = _MinimalRouteRecorder()
        monkeypatch.setattr('orchestrator.review_checkpoint.invoke_agent', stub.invoke_agent)
        # Verification is orthogonal to routing and is faked throughout this
        # suite (mirrors _mock_verify_passes's use for workflow's
        # run_scoped_verification); _save_report is stubbed because it writes
        # against self.config.project_root, which this fixture never creates.
        monkeypatch.setattr(
            'orchestrator.review_checkpoint.run_full_verification', _mock_verify_passes(),
        )
        monkeypatch.setattr(cp, '_save_report', lambda *a, **k: None)

        await cp._run_review('full', [])

        # The config-layer override reached the CLI seam.
        assert stub.calls[-1]['model'] == 'haiku'

        entries = _routing_events(rec)
        assert len(entries) == 1, (
            f'expected exactly one routing_decision event: {rec.events!r}'
        )
        data = entries[0]['data']
        assert data['role'] == 'deep_reviewer'
        # metadata_override is structurally unreachable here: _run_review
        # (review_checkpoint.py:190-208) passes no task_id/task_metadata, so
        # resolve_route's layer-1 override lookup is against {} and always
        # misses -- 'config' is the topmost layer this site can reach (see
        # class docstring).
        assert data['source_layer'] == 'config'
        # No applied_budget_usd annotation -- deep_reviewer never passes one
        # (byte-equivalent event, unlike steward's advisory-budget pair).
        assert 'applied_budget_usd' not in data
        assert 'budget_usd_advisory' not in data
        # No task identity flowed in, so routing_dispatch.py:271's
        # `scheduler is not None and task_id is not None` guard is
        # structurally False -- no metadata.routing mirror write is even
        # attempted (there is no scheduler/task object anywhere in this rig
        # for one to land on). The recorded event's own task_id is the
        # direct, observable witness of that honest "no task association"
        # contract.
        assert entries[0]['task_id'] is None


# ---------------------------------------------------------------------------
# Boundary scenario 12 -- the rollup renders per-(model×role) rows from rows
# a real rig run produced (plan step-10).
# ---------------------------------------------------------------------------


class _CostedInvokeStub:
    """Bare ``invoke_agent`` double (mirrors ``_MinimalRouteRecorder``) that
    returns a caller-fixed ``cost_usd``/``turns`` ``AgentResult`` instead of
    the zero-valued defaults -- this scenario's $/done and turn-cap-
    saturation assertions need real, non-accidental numbers to distinguish
    "computed correctly" from "happens to be zero"."""

    def __init__(self, *, cost_usd: float, turns: int) -> None:
        self._cost_usd = cost_usd
        self._turns = turns
        self.calls: list[dict[str, object]] = []

    async def invoke_agent(self, **kwargs) -> AgentResult:
        self.calls.append(kwargs)
        return AgentResult(
            success=True, output='OK', cost_usd=self._cost_usd, turns=self._turns,
        )


class TestRollupRendersRigProducedRows:
    """PRD boundary scenario 12: ``digest.model_role_rollup`` and
    ``render_digest_markdown`` surface real rows a real rig run produced --
    turning ``test_harness_digest_rollup.py``'s hand-seeded-fixture unit
    proof into an end-to-end one.

    Points a REAL ``CostStore``, ``EventStore`` and ``RunStore`` at ONE tmp
    ``runs.db`` (they share one file in production -- harness.py:1373-1398)
    and drives ``TaskWorkflow._invoke`` directly for TWO task dispatches
    sharing one run_id -- mirrors ``TestByteEquivalenceThroughTheRig``'s /
    ``TestUnknownRuleKeyRejectedPriorRulesStillRoute``'s direct-``_invoke``
    idiom, since only the resolved route and the telemetry rows it produces
    are under test, not a full ``run()`` state machine. Dispatch #1 resolves
    implementer at the stock config layer; dispatch #2 pins the SAME role to
    a distinct model via ``model_overrides`` -- so the rollup has >=2
    distinct (model, role) cells for one role, per the plan.

    ``task_results`` rows are HARNESS-owned (written by ``Harness._run_slot``
    from a ``TerminalReport``/``workflow.metrics`` this suite's direct-
    ``_invoke`` calls never accumulate through that path -- see module
    docstring's seam discussion). Rather than fabricate them with a raw SQL
    INSERT -- which would bypass the schema the rollup's JOIN depends on --
    the DONE/BLOCKED outcome rows below are written through the REAL
    ``RunStore.save_task_result`` API, from a ``TaskReport`` built with this
    test's own known values.
    """

    @staticmethod
    def _task_report(
        *, task_id: str, title: str, outcome: WorkflowOutcome, cost_usd: float,
    ) -> TaskReport:
        return TaskReport(
            task_id=task_id, title=title, outcome=outcome, cost_usd=cost_usd,
            completed_at=datetime.now(UTC).isoformat(),
        )

    # stock_config: uses the module-level fixture (declared next to `config`
    # above) unmodified -- see that fixture's docstring for why it is shared
    # rather than redefined per class.

    @pytest.mark.asyncio
    async def test_rollup_renders_rows_the_rig_produced(
        self, stock_config, git_ops, task_assignment, git_repo, tmp_path, monkeypatch,
    ):
        runs_db = tmp_path / 'runs.db'
        run_id = 'rig-run-1'
        event_store = EventStore(runs_db, run_id)
        cost_store = CostStore(runs_db)
        await cost_store.open()
        try:
            # RunStore's schema (task_results) isn't created by EventStore/
            # CostStore -- construct it on the shared file up front (schema
            # creation is a __init__ side effect, and the same instance is
            # reused for the writes below), mirroring
            # test_harness_digest_rollup.py's identical idiom.
            run_store = RunStore(runs_db)

            impl_max_turns = stock_config.max_turns.implementer
            assert impl_max_turns > 1, 'test needs headroom for an unsaturated call'

            # -- Dispatch #1 (task 42): stock config-layer model. One call
            # below the turn cap (unsaturated).
            stub1 = _CostedInvokeStub(cost_usd=2.50, turns=impl_max_turns - 1)
            workflow1, _s1 = _build_workflow(stock_config, git_ops, task_assignment, stub1)  # type: ignore[arg-type]
            workflow1.event_store = event_store
            workflow1.cost_store = cost_store
            workflow1.plan = {'steps': [{'id': 's1', 'status': 'pending'}]}
            workflow1.modules = ['lib']
            monkeypatch.setattr('orchestrator.workflow.invoke_agent', stub1.invoke_agent)

            await workflow1._invoke(IMPLEMENTER, 'impl prompt', git_repo)
            # Confirms the premise (no policy rule fires for this tiny plan)
            # rather than silently asserting the wrong cell below.
            assert stub1.calls[-1]['model'] == stock_config.models.implementer

            # -- Dispatch #2 (task 43): model_overrides pins the SAME role to
            # a DIFFERENT model -- the second distinct (model, role) cell the
            # plan requires. At (not over) the turn cap (saturated).
            assert stock_config.models.implementer != 'haiku'
            task_2 = {
                'id': '43',
                'title': 'Routing boundary task 2',
                'description': 'Exercise the real _invoke routing-resolution path',
                'status': 'pending',
                'metadata': {'files': ['lib'], 'model_overrides': {'implementer': 'haiku'}},
                'dependencies': [],
            }
            assignment_2 = TaskAssignment(task_id='43', task=task_2, modules=['lib'])
            stub2 = _CostedInvokeStub(cost_usd=1.25, turns=impl_max_turns)
            workflow2, _s2 = _build_workflow(stock_config, git_ops, assignment_2, stub2)  # type: ignore[arg-type]
            workflow2.event_store = event_store
            workflow2.cost_store = cost_store
            workflow2.plan = {'steps': [{'id': 's1', 'status': 'pending'}]}
            workflow2.modules = ['lib']
            monkeypatch.setattr('orchestrator.workflow.invoke_agent', stub2.invoke_agent)

            await workflow2._invoke(IMPLEMENTER, 'impl prompt', git_repo)
            assert stub2.calls[-1]['model'] == 'haiku'

            # A role that emitted invocation_end but was never routed through
            # _invoke in this window, so it carries NO routing_decision/
            # max_turns -- written via the real EventStore.emit() API
            # (never raw SQL) to prove assertion (c)'s fail-open half: a role
            # with no measurable cap reports None, not a misleading 0%/100%.
            event_store.emit(
                EventType.invocation_end, task_id='42', role='judge',
                data={'turns': 3, 'success': True},
            )

            # task_results: HARNESS-owned rows (see class docstring) --
            # written through the real RunStore API, never raw SQL.
            run_store.save_task_result(
                run_id,
                self._task_report(
                    task_id='42', title='Routing boundary task',
                    outcome=WorkflowOutcome.DONE, cost_usd=2.50,
                ),
                stock_config.fused_memory.project_id,
            )
            run_store.save_task_result(
                run_id,
                self._task_report(
                    task_id='43', title='Routing boundary task 2',
                    outcome=WorkflowOutcome.BLOCKED, cost_usd=1.25,
                ),
                stock_config.fused_memory.project_id,
            )

            window_start_iso = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
            window_end_iso = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
            rollup = model_role_rollup(runs_db, window_start_iso, window_end_iso)

            rows_by_cell = {(r.model, r.role): r for r in rollup.rows}
            stock_cell = rows_by_cell[(stock_config.models.implementer, 'implementer')]
            override_cell = rows_by_cell[('haiku', 'implementer')]

            # (a) both expected cells are present with the right invocation_count.
            assert stock_cell.invocation_count == 1
            assert override_cell.invocation_count == 1

            # (b) done/blocked counts and rates reflect the seeded outcomes;
            # cost_per_done is None (not 0.0) for the cell with zero DONE
            # tasks -- digest.py's documented None-not-zero contract.
            assert stock_cell.done_count == 1
            assert stock_cell.blocked_count == 0
            assert stock_cell.done_rate == pytest.approx(1.0)
            assert stock_cell.blocked_rate == pytest.approx(0.0)
            assert stock_cell.cost_per_done == pytest.approx(2.50)

            assert override_cell.done_count == 0
            assert override_cell.blocked_count == 1
            assert override_cell.done_rate == pytest.approx(0.0)
            assert override_cell.blocked_rate == pytest.approx(1.0)
            assert override_cell.cost_per_done is None

            # (c) turn_cap_saturation: implementer emitted a routing_decision
            # carrying max_turns for both calls (one under cap, one at cap:
            # 1 hit / 2 = 50%); judge emitted invocation_end but no
            # routing_decision in this window, so it fails open to None.
            assert rollup.turn_cap_saturation['implementer'] == pytest.approx(0.5)
            assert rollup.turn_cap_saturation['judge'] is None

            # Render: the rollup section header, both cells, and the
            # turn-cap-saturation lines all appear in the rendered digest.
            digest_inputs = DigestInputs(
                window_start_iso=window_start_iso, window_end_iso=window_end_iso,
                escalation_stats=EscalationStats(), done_count=1,
                cost_stats=CostStats(), parked_live=0, parked_window_churn=0,
                ewa_value=0.0, ewa_threshold=0.0, tripped=False,
                anomaly_flags={}, watcher_clusters=[], dry_run_proposals=[],
                model_role_rollup=rollup,
            )
            rendered = render_digest_markdown(digest_inputs)

            assert '## Per-(model×role) rollup' in rendered
            assert (
                '| model | role | invocations | done% | blocked% | cap-hit% | $/done |'
                in rendered
            )
            expected_stock_line = (
                f'| {stock_config.models.implementer} | implementer | 1 | '
                '100.0% | 0.0% | 0.0% | $2.50 |'
            )
            assert expected_stock_line in rendered
            assert '| haiku | implementer | 1 | 0.0% | 100.0% | 0.0% | — |' in rendered
            assert '### Turn-cap saturation (per role)' in rendered
            assert '- implementer: 50.0%' in rendered
            assert '- judge: n/a' in rendered
        finally:
            await cost_store.close()
