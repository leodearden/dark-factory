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
from pathlib import Path

import pytest
from _recording_event_store import _RecordingEventStore
from _workflow_helpers import (
    AgentStub,
    _build_workflow,
    _derive_meta_root_like_production,  # noqa: F401  autouse fixture, see its docstring
    _init_repo,
)

from orchestrator.config import GitConfig, OrchestratorConfig, SandboxConfig
from orchestrator.event_store import EventType
from orchestrator.git_ops import GitOps
from orchestrator.scheduler import TaskAssignment

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
