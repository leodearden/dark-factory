"""Unit + tripwire tests for the shared ``build_workflow()`` factory.

PRD eval-framework-revival §α, Contract C2, Invariant P2, Boundary test B2.

``build_workflow()`` is the SINGLE construction point for ``TaskWorkflow``
across production dispatch (``harness.py``) and eval dispatch
(``evals/runner.py``).  Routing both sites through one factory means a new
MANDATORY ``TaskWorkflow`` constructor parameter is either acquired by both
call sites automatically or breaks BOTH at once — never drifts silently.

The grep-guard tripwire (added in a later step) mechanises that invariant by
source inspection: exactly one production-source file may construct
``TaskWorkflow(`` and both dispatch sites must route through
``build_workflow(``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from _orch_helpers import pydantic_spec

import orchestrator
from orchestrator.config import OrchestratorConfig
from orchestrator.workflow import TaskWorkflow, build_workflow


def test_build_workflow_forwards_params():
    """``build_workflow(...)`` returns a ``TaskWorkflow`` forwarding every param by identity."""
    assignment = MagicMock()
    assignment.task_id = '2464'
    assignment.task = {'id': '2464', 'title': 'T', 'description': 'd'}
    assignment.modules = ['mod_a']

    config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.max_review_cycles = 2
    config.max_amendment_rounds = 1
    config.lock_depth = 2
    config.steward_completion_timeout = 300.0

    git_ops = MagicMock()
    scheduler = MagicMock()
    briefing = MagicMock()
    mcp = MagicMock()

    # Distinct opaque sentinels for the optional pass-through params — assert
    # the factory forwards the *identical* objects, not equal copies. Typed
    # ``Any`` so they satisfy build_workflow's (deliberately explicit) param
    # types while staying inert pass-through tokens.
    cost_store: Any = object()
    event_store: Any = object()
    initial_plan: Any = object()
    usage_gate: Any = object()
    escalation_queue: Any = object()

    wf = build_workflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,
        briefing=briefing,
        mcp=mcp,
        escalation_queue=escalation_queue,
        usage_gate=usage_gate,
        initial_plan=initial_plan,
        event_store=event_store,
        cost_store=cost_store,
    )

    assert isinstance(wf, TaskWorkflow)
    # Required params forwarded by identity.
    assert wf.assignment is assignment
    assert wf.config is config
    assert wf.git_ops is git_ops
    assert wf.scheduler is scheduler
    assert wf.briefing is briefing
    assert wf.mcp is mcp
    # Optional pass-through params forwarded by identity.
    assert wf.cost_store is cost_store
    assert wf.event_store is event_store
    assert wf.initial_plan is initial_plan
    assert wf.usage_gate is usage_gate
    assert wf.escalation_queue is escalation_queue


# Every ``build_workflow`` parameter, in ``TaskWorkflow.__init__`` order. The
# instance-identity test above can only cover fields the constructor stores 1:1
# on ``self``; several pass-through params are stored under a private name
# (``steward_factory`` -> ``_steward_factory``, ``cancel_event`` ->
# ``_cancel_event``, ``escalation_event`` -> ``_escalation_event``, ``run_id`` ->
# ``_process_run_id``) and ``resume_session_id`` is decomposed via ``.get(...)``
# and never stored whole — so none of those can be asserted by instance identity.
_BUILD_WORKFLOW_PARAMS = (
    'assignment', 'config', 'git_ops', 'scheduler', 'briefing', 'mcp',
    'escalation_queue', 'escalation_event', 'usage_gate', 'initial_plan',
    'steward_factory', 'merge_queue', 'merge_worker', 'merge_inflight_registry',
    'event_store', 'cost_store', 'cancel_event', 'resume_session_id', 'run_id',
)


def test_build_workflow_forwards_every_param_by_keyword():
    """Every param reaches ``TaskWorkflow(...)`` under its own keyword — no swaps.

    ``build_workflow`` forwards each field by a hand-written ``name=name`` keyword,
    so a copy/paste swap in the body (e.g. ``merge_queue=merge_worker``) would
    misroute an argument silently and the instance-identity test would still pass
    for any two interchangeable-looking params. Spying on the constructor and
    asserting each keyword carries the *identical* sentinel catches that drift
    class directly — and uniformly covers the params the identity test cannot:
    the privately-stored ones and ``resume_session_id`` (decomposed via
    ``.get(...)``, so never stored whole).
    """
    # One distinct opaque sentinel per constructor parameter. ``TaskWorkflow`` is
    # patched out, so no attribute access happens on any sentinel — plain
    # ``object()`` tokens are safe even for the required params.
    params: dict[str, Any] = {name: object() for name in _BUILD_WORKFLOW_PARAMS}

    with patch('orchestrator.workflow.TaskWorkflow') as mock_ctor:
        result = build_workflow(**params)

    mock_ctor.assert_called_once()
    call = mock_ctor.call_args
    # Every param is forwarded by keyword, never positionally.
    assert call.args == ()
    # Exactly the expected keyword set — nothing dropped, nothing extra.
    assert set(call.kwargs) == set(_BUILD_WORKFLOW_PARAMS)
    # Each keyword carries the identical sentinel; a mis-wired keyword fails here.
    for name in _BUILD_WORKFLOW_PARAMS:
        assert call.kwargs[name] is params[name], f'build_workflow misrouted {name!r}'
    # The factory returns exactly what the constructor produced.
    assert result is mock_ctor.return_value


# ── Grep-guard tripwire (PRD eval-framework-revival §α, Invariant P2 / Boundary
#    test B2) ───────────────────────────────────────────────────────────────
#
# The single-construction-point invariant is enforced by source inspection, not
# behaviour: (a) exactly one production-source file may construct ``TaskWorkflow(``
# and (b) both dispatch sites must route through ``build_workflow(``. Together
# these encode "one construction point + both sites route through it" — precisely
# what makes a new required factory arg break BOTH constructions at once rather
# than drift silently. Scoped to orchestrator/src/orchestrator/ because ~70 test
# modules legitimately construct TaskWorkflow directly and would false-fail a
# repo-wide guard.

# ``TaskWorkflow(`` not preceded by a word char or dot — a construction call,
# not ``build_workflow`` / ``class TaskWorkflow:`` / an attribute access.
_CONSTRUCT_RE = re.compile(r"(?<![\w.])TaskWorkflow\(")
# ``build_workflow(`` not preceded by a word char or dot — the factory call, not
# a ``_build_workflow`` lookalike.
_FACTORY_CALL_RE = re.compile(r"(?<![\w.])build_workflow\(")


def _orchestrator_src_root() -> Path:
    """Resolve ``orchestrator/src/orchestrator/`` from the imported package.

    ``conftest.py`` inserts this worktree's ``src`` at the front of ``sys.path``,
    so ``orchestrator.__file__`` resolves to the tree under test — the guard is
    path-independent and always scans the checkout it is run against.
    """
    return Path(orchestrator.__file__).resolve().parent


def test_single_taskworkflow_construction_point():
    """Only ``workflow.py`` may construct ``TaskWorkflow(`` in production source."""
    src_root = _orchestrator_src_root()
    offenders: set[str] = set()
    for path in src_root.rglob('*.py'):
        text = path.read_text(encoding='utf-8', errors='ignore')
        for line in text.splitlines():
            code = line.split('#', 1)[0]  # drop trailing comment
            if _CONSTRUCT_RE.search(code):
                offenders.add(path.relative_to(src_root).as_posix())
                break
    assert offenders == {'workflow.py'}, (
        'Direct TaskWorkflow(...) construction is only permitted in workflow.py '
        '(the build_workflow() factory body, the single construction point). '
        f'Offending files: {sorted(offenders)}. Route construction through '
        'build_workflow().'
    )


def test_both_dispatch_sites_route_through_factory():
    """Both dispatch sites construct via ``build_workflow(``, never ``TaskWorkflow`` directly."""
    src_root = _orchestrator_src_root()
    for rel in ('harness.py', 'evals/runner.py'):
        source = (src_root / rel).read_text(encoding='utf-8')
        assert _FACTORY_CALL_RE.search(source), (
            f'{rel} must construct its TaskWorkflow via build_workflow(); found '
            'no build_workflow( call — the dispatch site is bypassing the factory.'
        )
